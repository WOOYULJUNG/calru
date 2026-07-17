"""Shared, frozen utilities for topology analysis v1."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from repro.sagodi_protocol.artifacts import atomic_json, strict_json_load

from .artifacts import canonical_bytes, sha256_file
from .topology_models import build_topology_model
from .topology_training import TopologyBatch, hold_prediction, load_fixed_bank


ANALYSIS_CONFIG = Path(__file__).with_name("topology_analysis_v1.json")
SUPPORTED_ANALYSIS_IDS = (
    "manifold_topology_analysis_v1",
    "manifold_topology_baseline_all_analysis_v2",
    "manifold_topology_hparam_analysis_v1",
)
MODEL_ORDER = ("rnn", "gru", "lstm", "hc")
TOPOLOGY_ORDER = ("s1", "t2", "s2")


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    job_id: str
    model_id: str
    topology: str
    seed: int
    manifest: Mapping[str, Any]
    result: Mapping[str, Any]

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoint.pt"


def load_analysis_config(path: Path | str = ANALYSIS_CONFIG) -> dict[str, Any]:
    payload = strict_json_load(Path(path).expanduser().resolve(strict=True))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("analysis config must be a schema-1 object")
    if payload.get("analysis_id") not in SUPPORTED_ANALYSIS_IDS:
        raise ValueError("analysis id differs")
    expected = payload["expected_training"]
    models = tuple(expected["models"])
    if not models or len(models) != len(set(models)):
        raise ValueError("analysis model order is empty or duplicated")
    if tuple(expected["topologies"]) != TOPOLOGY_ORDER:
        raise ValueError("analysis topology order differs")
    return payload


def expected_job_id(model: str, topology: str, seed: int) -> str:
    return f"pilot__{model}__{topology}__seed{int(seed)}"


def expected_jobs(config: Mapping[str, Any]) -> list[str]:
    expected = config["expected_training"]
    if "job_ids" in expected:
        jobs = [str(value) for value in expected["job_ids"]]
        if len(jobs) != int(expected["expected_runs"]) or len(jobs) != len(set(jobs)):
            raise ValueError("explicit analysis job list is incomplete or duplicated")
        return jobs
    return [
        expected_job_id(model, topology, seed)
        for seed in expected["seeds"]
        for topology in expected["topologies"]
        for model in expected["models"]
    ]


def discover_completed_runs(
    run_root: Path | str, config: Mapping[str, Any]
) -> tuple[list[RunRecord], list[str]]:
    root = Path(run_root).expanduser().resolve(strict=True)
    records: list[RunRecord] = []
    missing: list[str] = []
    for job_id in expected_jobs(config):
        run_dir = root / job_id
        required = (
            run_dir / "COMPLETED.json",
            run_dir / "manifest.json",
            run_dir / "result.json",
            run_dir / "checkpoint.pt",
            run_dir / "trace.json",
        )
        if not all(path.is_file() for path in required):
            missing.append(job_id)
            continue
        manifest = strict_json_load(run_dir / "manifest.json")
        result = strict_json_load(run_dir / "result.json")
        records.append(
            RunRecord(
                run_dir=run_dir,
                job_id=job_id,
                model_id=str(result["model_id"]),
                topology=str(result["topology"]),
                seed=int(result["replicate_seed"]),
                manifest=manifest,
                result=result,
            )
        )
    return records, missing


def build_record_model(record: RunRecord, device: torch.device | str):
    """Reconstruct either a frozen-transfer or search-selected architecture."""

    model_seed = int(record.manifest["model_seed"])
    config = None
    if record.manifest.get("campaign_id") == "manifold_topology_hparam_v1":
        from .run_topology_hparam import load_search_config

        config = copy.deepcopy(load_search_config())
        row = config["models"][record.model_id]
        row["learning_rate"] = float(record.manifest["learning_rate"])
        if record.model_id == "hc":
            row["max_log_modulation"] = float(
                record.manifest["max_log_modulation"]
            )
            row["gate_output_bias"] = float(record.manifest["gate_output_bias"])
    model = build_topology_model(
        record.model_id,
        record.topology,
        model_seed=model_seed,
        config=config,
    ).to(device)
    model.eval()
    return model


def load_model(record: RunRecord, device: torch.device | str):
    checkpoint = torch.load(
        record.checkpoint_path, map_location=device, weights_only=False
    )
    checkpoint_manifest = checkpoint["manifest"]
    if checkpoint_manifest["job_id"] != record.job_id:
        raise ValueError(f"checkpoint job id mismatch for {record.job_id}")
    model = build_record_model(record, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def normalized_geodesic_errors(
    topology: str, prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return common [0,1] error and optional T2 worst-coordinate error."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    key = str(topology).lower()
    if key in {"s1", "t2"}:
        dimensions = 1 if key == "s1" else 2
        prediction_pair = prediction.reshape(*prediction.shape[:-1], dimensions, 2)
        target_pair = target.reshape(*target.shape[:-1], dimensions, 2)
        prediction_angle = torch.atan2(prediction_pair[..., 1], prediction_pair[..., 0])
        target_angle = torch.atan2(target_pair[..., 1], target_pair[..., 0])
        delta = torch.remainder(
            prediction_angle - target_angle + math.pi, 2.0 * math.pi
        ) - math.pi
        if key == "s1":
            return torch.abs(delta[..., 0]) / math.pi, None
        common = torch.sqrt(torch.mean(delta.square(), dim=-1)) / math.pi
        worst = torch.max(torch.abs(delta), dim=-1).values / math.pi
        return common, worst
    if key != "s2":
        raise ValueError(f"unknown topology {topology!r}")
    prediction_unit = prediction / torch.linalg.vector_norm(
        prediction, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    target_unit = target / torch.linalg.vector_norm(
        target, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    cosine = (prediction_unit * target_unit).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cosine) / math.pi, None


def output_norm_error(topology: str, prediction: torch.Tensor) -> torch.Tensor:
    key = str(topology).lower()
    if key in {"s1", "t2"}:
        dimensions = 1 if key == "s1" else 2
        pairs = prediction.reshape(*prediction.shape[:-1], dimensions, 2)
        return torch.abs(torch.linalg.vector_norm(pairs, dim=-1) - 1.0).mean(dim=-1)
    return torch.abs(torch.linalg.vector_norm(prediction, dim=-1) - 1.0)


def nmse_db(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = (prediction - target).square().mean()
    target_power = target.square().mean()
    ratio = mse / target_power.clamp_min(torch.finfo(target.dtype).eps)
    return float((10.0 * torch.log10(ratio)).cpu())


def summarize_error_tensor(
    errors: torch.Tensor, percentiles: Iterable[float] = (90, 95, 99)
) -> dict[str, float]:
    if errors.ndim != 2:
        raise ValueError("error tensor must be [T,B]")
    trials = errors.mean(dim=0)
    result = {
        "sequence_mean": float(errors.mean().cpu()),
        "terminal_mean": float(errors[-1].mean().cpu()),
        "trial_median": float(trials.median().cpu()),
    }
    for percentile in percentiles:
        result[f"trial_p{int(percentile)}"] = float(
            torch.quantile(trials, float(percentile) / 100.0).cpu()
        )
    return result


@torch.no_grad()
def forward_endpoint_states(
    model,
    inputs: torch.Tensor,
    initial_memory: torch.Tensor,
    *,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.shape[1] != initial_memory.shape[0]:
        raise ValueError("input and initial-memory batch differs")
    states: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    for start in range(0, inputs.shape[1], int(chunk_size)):
        stop = min(inputs.shape[1], start + int(chunk_size))
        output, history = model.forward_sequence(
            inputs[:, start:stop],
            initial_memory=initial_memory[start:stop],
            return_states=True,
        )
        states.append(model.primary_from_reported(history[-1]))
        predictions.append(output[-1])
    return torch.cat(states, dim=0), torch.cat(predictions, dim=0)


@torch.no_grad()
def blank_snapshots(
    model,
    state: torch.Tensor,
    horizons: Iterable[int],
) -> dict[int, torch.Tensor]:
    requested = sorted(set(int(value) for value in horizons))
    if not requested or requested[0] < 0:
        raise ValueError("blank horizons must be nonnegative")
    current = model.reported_from_primary(state)
    blank = torch.zeros(
        state.shape[0], model.input_dim, device=state.device, dtype=state.dtype
    )
    snapshots: dict[int, torch.Tensor] = {}
    if 0 in requested:
        snapshots[0] = model.primary_from_reported(current).clone()
    for step in range(1, requested[-1] + 1):
        current = model.step(blank, current)
        if step in requested:
            snapshots[step] = model.primary_from_reported(current).clone()
    return snapshots


def decode_primary(model, state: torch.Tensor) -> torch.Tensor:
    inputs = torch.zeros(
        state.shape[0], model.input_dim, device=state.device, dtype=state.dtype
    )
    reconstruct = getattr(model, "reported_from_primary_for_input", None)
    reported = (
        reconstruct(state, inputs)
        if reconstruct is not None
        else model.reported_from_primary(state)
    )
    return model.decode(reported)


def test_batch(
    topology: str,
    *,
    device: torch.device | str,
    trajectories: int,
    horizon: int = 128,
) -> TopologyBatch:
    return load_fixed_bank(
        topology,
        split="test",
        device=device,
        trajectories=int(trajectories),
        horizon=int(horizon),
    )


def validation_batch(
    topology: str,
    *,
    device: torch.device | str,
    trajectories: int,
    horizon: int = 128,
) -> TopologyBatch:
    return load_fixed_bank(
        topology,
        split="validation",
        device=device,
        trajectories=int(trajectories),
        horizon=int(horizon),
    )


def atomic_npz(path: Path | str, **arrays: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, **{name: np.asarray(value) for name, value in arrays.items()})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_csv(path: Path | str, rows: list[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", prefix=f".{destination.name}.", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def file_sha256(path: Path | str) -> str:
    return sha256_file(Path(path))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_success_map(analysis_root: Path | str) -> dict[str, bool]:
    payload = strict_json_load(Path(analysis_root) / "task" / "task_success.json")
    return {str(key): bool(value) for key, value in payload["task_success"].items()}


def load_analysis_bank(bank_root: Path | str, topology: str) -> dict[str, np.ndarray]:
    root = Path(bank_root).expanduser().resolve(strict=True)
    manifest = strict_json_load(root / "analysis_banks_manifest.json")
    row = manifest["banks"][str(topology)]
    path = root / row["path"]
    if file_sha256(path) != row["sha256"]:
        raise ValueError(f"analysis bank SHA mismatch for {topology}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


__all__ = [
    "ANALYSIS_CONFIG",
    "MODEL_ORDER",
    "RunRecord",
    "TOPOLOGY_ORDER",
    "atomic_npz",
    "blank_snapshots",
    "canonical_sha256",
    "decode_primary",
    "discover_completed_runs",
    "expected_jobs",
    "file_sha256",
    "forward_endpoint_states",
    "load_analysis_config",
    "load_analysis_bank",
    "load_model",
    "load_success_map",
    "nmse_db",
    "normalized_geodesic_errors",
    "output_norm_error",
    "summarize_error_tensor",
    "test_batch",
    "validation_batch",
    "write_csv",
]
