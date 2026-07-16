"""Targeted H-C sweep over exp(tanh) amplitude and gate initialization.

This campaign keeps the recurrent writer, hybrid RP, task streams, optimizer,
noise-free protocol, and update count fixed.  It changes only

* ``max_log_modulation`` (the ``a`` in ``exp(a * tanh(r(h)))``), and
* the output layer initialization of the state-dependent retention gate.

The launcher is resumable at completed-run granularity and distributes workers
over physical GPUs.  Each worker stores its trained checkpoint before running a
bounded checkpoint-only task/autonomous-stability screen.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from repro.sagodi_protocol import source_repaired_baselines_v6 as baseline_v6
from repro.sagodi_protocol.artifacts import (
    atomic_json,
    sha256_file,
    strict_json_load,
)
from repro.sagodi_protocol.metrics import masked_mse, task_metrics
from repro.sagodi_protocol.sagodi_primary_runner import _blank_decode_primary
from repro.sagodi_protocol.source_resolved_protocol import source_angular_integration
from repro.sagodi_protocol.state import StateAdapter
from repro.sagodi_protocol.tasks import Batch, load_fixed_bank
from repro.sagodi_protocol.train import _retention_plasticity_call

from .models import build_state_dependent_model, recurrence


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "hc_stability_sweep.json"
CAMPAIGN_ID = "hc_dynamic_retention_stability_sweep_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _native(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_config(path: Path) -> dict[str, Any]:
    config = strict_json_load(path)
    if config.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("wrong H-C stability sweep campaign id")
    if config.get("model_seeds") != [0, 1, 2]:
        raise ValueError("H-C stability sweep requires seeds 0,1,2")
    fixed = config.get("fixed_architecture", {})
    if fixed != {
        "writer_kind": "recurrent",
        "retention_mode": "hybrid_rp",
        "width": 52,
        "gate_hidden": 52,
    }:
        raise ValueError("fixed H-C architecture contract differs")
    training = config.get("training", {})
    required_training = {
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "updates": 5000,
        "batch_size": 64,
        "state_noise_std": 0.0,
        "target_noise_std": 0.0,
        "output_dropout": 0.0,
    }
    if training != required_training:
        raise ValueError("training contract differs from noise-free H-C")
    rp = config.get("retention_plasticity", {})
    required_rp = {
        "warmup_updates": 1500,
        "eta_lambda": 1000.0,
        "damage_epsilon": 3e-5,
        "intervention_interval_updates": 50,
        "probe_batch_size": 96,
        "blank_ablation_horizon": 500,
    }
    if rp != required_rp:
        raise ValueError("RP contract differs from H-C")
    cells = config.get("sweep_cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("sweep_cells must be nonempty")
    identifiers: set[str] = set()
    tuples: set[tuple[float, float, float]] = set()
    for cell in cells:
        identifier = str(cell.get("id", ""))
        if not identifier or identifier in identifiers:
            raise ValueError("sweep cell ids must be nonempty and unique")
        values = (
            float(cell["max_log_modulation"]),
            float(cell["gate_output_weight_std"]),
            float(cell["gate_output_bias"]),
        )
        if values in tuples:
            raise ValueError("duplicate H-C sweep cell")
        if not 0.0 < values[0] < math.log(2.0):
            raise ValueError("max_log_modulation lies outside the valid range")
        if values[1] < 0.0 or not all(math.isfinite(value) for value in values):
            raise ValueError("gate initialization values must be finite")
        identifiers.add(identifier)
        tuples.add(values)
    reference = config.get("reference_condition", {})
    if (
        reference.get("id") != "a0p05_zero"
        or not reference.get("reuse_existing_checkpoint")
        or reference.get("max_log_modulation") != 0.05
        or reference.get("gate_output_weight_std") != 0.0
        or reference.get("gate_output_bias") != 0.0
    ):
        raise ValueError("incumbent H-C reference contract differs")
    screen = config.get("screen", {})
    if (
        screen.get("blank_horizon") != 2048
        or screen.get("progress_interval_updates") != 50
        or screen.get("validation_interval_updates") != 500
    ):
        raise ValueError("screen contract differs")
    return config


@dataclass(frozen=True)
class Cell:
    identifier: str
    max_log_modulation: float
    gate_output_weight_std: float
    gate_output_bias: float


@dataclass(frozen=True)
class WorkerSpec:
    cell: Cell
    model_seed: int
    output_dir: str
    bank: str
    config_path: str
    updates: int
    batch_size: int
    smoke: bool

    @property
    def run_id(self) -> str:
        prefix = "smoke" if self.smoke else "train"
        return f"{prefix}__{self.cell.identifier}__seed{self.model_seed:02d}"


def _cells(config: Mapping[str, Any]) -> tuple[Cell, ...]:
    return tuple(
        Cell(
            identifier=str(item["id"]),
            max_log_modulation=float(item["max_log_modulation"]),
            gate_output_weight_std=float(item["gate_output_weight_std"]),
            gate_output_bias=float(item["gate_output_bias"]),
        )
        for item in config["sweep_cells"]
    )


def _to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        batch.inputs.to(device),
        batch.output_targets.to(device),
        batch.latent_targets.to(device),
        batch.mask.to(device),
        batch.metadata,
    )


def _training_batch(spec: WorkerSpec, update: int, device: torch.device) -> Batch:
    return source_angular_integration(
        spec.batch_size,
        0,
        stream_key=(
            baseline_v6.CAMPAIGN_ID,
            "online_train",
            spec.model_seed,
            int(update),
        ),
        device=device,
    )


def _probe_batch(
    spec: WorkerSpec, config: Mapping[str, Any], update: int, device: torch.device
) -> Batch:
    rp = config["retention_plasticity"]
    return source_angular_integration(
        int(rp["probe_batch_size"]),
        0,
        stream_key=(
            baseline_v6.CAMPAIGN_ID,
            "rp_probe",
            spec.model_seed,
            int(update),
        ),
        device=device,
    )


def _finite_model(model: torch.nn.Module, *, gradients: bool = False) -> None:
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is not None and not bool(torch.isfinite(value).all()):
            suffix = "gradient" if gradients else "parameter"
            raise FloatingPointError(f"non-finite {suffix}: {name}")


@torch.no_grad()
def _task_evaluation(model, bank: Batch) -> dict[str, Any]:
    model.eval()
    prediction = model.forward_sequence(
        bank.inputs, initial_memory=bank.output_targets[0]
    )
    if not bool(torch.isfinite(prediction).all()):
        return {"status": "nonfinite_prediction", "metrics": None}
    return {
        "status": "finite",
        "metrics": _native(
            task_metrics(
                prediction,
                bank.output_targets,
                bank.mask,
                bank.latent_targets,
            )
        ),
    }


def _state_diagnostics(rec, state: torch.Tensor) -> dict[str, Any]:
    dynamic_lambda = rec.state_dependent_lambda(state)
    gate_tanh = torch.tanh(rec.retention_gate(state))
    state64 = state.to(torch.float64)
    weights = state64.square()
    total = weights.sum().clamp_min(1.0e-24)
    coordinate_share = weights.sum(dim=0) / total
    participation_ratio = 1.0 / coordinate_share.square().sum()
    weighted_lambda = (
        weights * dynamic_lambda.to(torch.float64)
    ).sum() / total
    weighted_above = (
        weights * (dynamic_lambda > 1.0).to(torch.float64)
    ).sum() / total
    weighted_saturated = (
        weights * (gate_tanh.abs() >= 0.99).to(torch.float64)
    ).sum() / total
    norm = torch.linalg.vector_norm(state64, dim=1)
    return {
        "state_norm_median": float(norm.median()),
        "state_norm_q95": float(torch.quantile(norm, 0.95)),
        "state_norm_maximum": float(norm.max()),
        "state_energy_participation_ratio": float(participation_ratio),
        "dynamic_lambda_mean": float(dynamic_lambda.mean()),
        "dynamic_lambda_maximum": float(dynamic_lambda.max()),
        "dynamic_lambda_fraction_above_one": float(
            (dynamic_lambda > 1.0).float().mean()
        ),
        "state_energy_weighted_lambda": float(weighted_lambda),
        "state_energy_fraction_at_lambda_above_one": float(weighted_above),
        "state_energy_fraction_at_abs_gate_tanh_at_least_0p99": float(
            weighted_saturated
        ),
    }


@torch.no_grad()
def _autonomous_screen(model, bank: Batch, horizon: int) -> dict[str, Any]:
    model.eval()
    _, states = model.forward_sequence(
        bank.inputs,
        initial_memory=bank.output_targets[0],
        return_states=True,
    )
    adapter = StateAdapter(model.core)
    state = adapter.primary_from_reported(states[-1])
    rec = recurrence(model)
    endpoint = _state_diagnostics(rec, state)
    requested = {0, 1, 16, 128, 512, 1024, int(horizon)}
    trace: list[dict[str, Any]] = []
    failure_step: int | None = None
    for step in range(int(horizon) + 1):
        if not bool(torch.isfinite(state).all()):
            failure_step = step
            break
        if step in requested:
            trace.append({"step": step, **_state_diagnostics(rec, state)})
        if step < int(horizon):
            state = adapter.actual_f0(state)
    result: dict[str, Any] = {
        "blank_horizon": int(horizon),
        "stable_through_horizon": failure_step is None,
        "nonfinite_step": failure_step,
        "task_endpoint": endpoint,
        "trace": trace,
        "terminal_memory": None,
    }
    if failure_step is None:
        decoded = _blank_decode_primary(model, adapter, state)
        if bool(torch.isfinite(decoded).all()):
            target = bank.output_targets[-1]
            predicted_angle = torch.atan2(decoded[:, 1], decoded[:, 0])
            target_angle = torch.atan2(target[:, 1], target[:, 0])
            angular = torch.atan2(
                torch.sin(predicted_angle - target_angle),
                torch.cos(predicted_angle - target_angle),
            ).abs()
            result["terminal_memory"] = {
                "status": "finite",
                "mean_angular_error_radians": float(angular.mean()),
                "median_angular_error_radians": float(angular.median()),
                "maximum_angular_error_radians": float(angular.max()),
            }
        else:
            result["terminal_memory"] = {"status": "nonfinite_decode"}
    return result


def _rp_updates(spec: WorkerSpec, config: Mapping[str, Any]) -> tuple[int, ...]:
    rp = config["retention_plasticity"]
    warmup = int(rp["warmup_updates"])
    interval = int(rp["intervention_interval_updates"])
    return tuple(
        update
        for update in range(1, spec.updates + 1)
        if update > warmup and update % interval == 0
    )


def run_worker(spec: WorkerSpec, device_text: str) -> Path:
    config_path = Path(spec.config_path).resolve(strict=True)
    config = _load_config(config_path)
    output = Path(spec.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"worker output is not empty: {output}")
    device = torch.device(device_text)
    baseline_v6._configure_determinism(spec.model_seed)
    torch.set_num_threads(1)
    model = build_state_dependent_model(
        "recurrent",
        model_seed=spec.model_seed,
        retention_mode="hybrid_rp",
        gate_hidden=52,
        max_log_modulation=spec.cell.max_log_modulation,
        gate_output_weight_std=spec.cell.gate_output_weight_std,
        gate_output_bias=spec.cell.gate_output_bias,
    ).to(device)
    _finite_model(model)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    training = config["training"]
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    bank = _to_device(load_fixed_bank(spec.bank), device)
    rp_schedule = set(_rp_updates(spec, config))
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": spec.run_id,
        "cell": {
            "id": spec.cell.identifier,
            "max_log_modulation": spec.cell.max_log_modulation,
            "gate_output_weight_std": spec.cell.gate_output_weight_std,
            "gate_output_bias": spec.cell.gate_output_bias,
        },
        "model_seed": spec.model_seed,
        "writer_kind": "recurrent",
        "retention_mode": "hybrid_rp",
        "training": {**training, "effective_updates": spec.updates, "effective_batch_size": spec.batch_size},
        "retention_plasticity": config["retention_plasticity"],
        "screen": config["screen"],
        "parameters_total": total,
        "parameters_gradient_trainable": trainable,
        "paired_task_and_rp_streams_across_cells": True,
        "bank": str(Path(spec.bank).resolve()),
        "bank_sha256": sha256_file(spec.bank),
        "config_sha256": sha256_file(config_path),
        "device": device_text,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at_utc": _utc_now(),
    }
    atomic_json(output / "run_manifest.json", manifest)

    trace: list[dict[str, Any]] = []
    rp_trace: list[dict[str, Any]] = []
    started = time.time()
    progress_interval = int(config["screen"]["progress_interval_updates"])
    validation_interval = int(config["screen"]["validation_interval_updates"])
    for update in range(1, spec.updates + 1):
        model.train()
        batch = _training_batch(spec, update, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_sequence(
            batch.inputs,
            initial_memory=batch.output_targets[0],
            state_noise_std=0.0,
        )
        loss = masked_mse(prediction, batch.output_targets, batch.mask)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()
        _finite_model(model, gradients=True)
        optimizer.step()
        _finite_model(model)

        if update in rp_schedule:
            model.eval()
            probe = _probe_batch(spec, config, update, device)
            rp = config["retention_plasticity"]
            details = _retention_plasticity_call(
                model,
                probe,
                blank_horizon=int(rp["blank_ablation_horizon"]),
                eta_lambda=float(rp["eta_lambda"]),
                damage_epsilon=float(rp["damage_epsilon"]),
                initial_memory=probe.output_targets[0],
            )
            rp_trace.append({"update": update, **_native(details)})

        should_trace = update == 1 or update % progress_interval == 0 or update == spec.updates
        should_validate = update % validation_interval == 0 or update == spec.updates
        if should_trace or should_validate:
            row: dict[str, Any] = {
                "update": update,
                "train_mse": float(loss.detach().cpu()),
                "elapsed_seconds": float(time.time() - started),
                "rp_calls": len(rp_trace),
            }
            if should_validate:
                row["validation"] = _task_evaluation(model, bank)
            trace.append(row)
            atomic_json(output / "training_trace.json", trace)
            atomic_json(output / "rp_trace.json", rp_trace)
            atomic_json(
                output / "progress.json",
                {
                    "schema_version": 1,
                    "status": "running",
                    "run_id": spec.run_id,
                    "update": update,
                    "updates_total": spec.updates,
                    "latest": row,
                    "updated_at_utc": _utc_now(),
                },
            )
            print(
                f"[{spec.cell.identifier} seed={spec.model_seed}] "
                f"{update}/{spec.updates} loss={row['train_mse']:.6g} "
                f"rp={len(rp_trace)} elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )

    if len(rp_trace) != len(rp_schedule):
        raise RuntimeError(f"RP count differs: {len(rp_trace)} != {len(rp_schedule)}")
    checkpoint_path = output / "checkpoint_trained.pt"
    _atomic_torch_save(
        checkpoint_path,
        {
            "schema_version": 1,
            "checkpoint_type": CAMPAIGN_ID,
            "checkpoint_stage": "post_training_pre_screen",
            "cell": manifest["cell"],
            "model_seed": spec.model_seed,
            "updates_completed": spec.updates,
            "rp_call_count": len(rp_trace),
            "state_dict": model.state_dict(),
        },
    )
    task = _task_evaluation(model, bank)
    autonomous = _autonomous_screen(
        model,
        bank,
        horizon=(16 if spec.smoke else int(config["screen"]["blank_horizon"])),
    )
    result = {
        "schema_version": 1,
        "status": "completed",
        "campaign_id": CAMPAIGN_ID,
        "run_id": spec.run_id,
        "cell": manifest["cell"],
        "model_seed": spec.model_seed,
        "updates_completed": spec.updates,
        "rp_call_count": len(rp_trace),
        "parameters_total": total,
        "parameters_gradient_trainable": trainable,
        "task_evaluation": task,
        "autonomous_screen": autonomous,
        "elapsed_seconds": float(time.time() - started),
    }
    atomic_json(output / "training_trace.json", trace)
    atomic_json(output / "rp_trace.json", rp_trace)
    atomic_json(output / "result.json", result)
    atomic_json(
        output / "progress.json",
        {
            "schema_version": 1,
            "status": "completed",
            "run_id": spec.run_id,
            "update": spec.updates,
            "updated_at_utc": _utc_now(),
        },
    )
    atomic_json(output / "COMPLETE", {"schema_version": 1, "run_id": spec.run_id})
    return output


def _build_specs(
    root: Path, config_path: Path, bank: Path, *, smoke: bool
) -> tuple[WorkerSpec, ...]:
    config = _load_config(config_path)
    seeds = [0] if smoke else [int(value) for value in config["model_seeds"]]
    updates = 3 if smoke else int(config["training"]["updates"])
    batch_size = 4 if smoke else int(config["training"]["batch_size"])
    parent = root / ("smoke" if smoke else "runs")
    return tuple(
        WorkerSpec(
            cell=cell,
            model_seed=seed,
            output_dir=str(parent / f"{cell.identifier}__seed{seed:02d}"),
            bank=str(bank),
            config_path=str(config_path),
            updates=updates,
            batch_size=batch_size,
            smoke=smoke,
        )
        for cell in _cells(config)
        for seed in seeds
    )


def _result_complete(spec: WorkerSpec) -> bool:
    try:
        result = strict_json_load(Path(spec.output_dir) / "result.json")
        marker = strict_json_load(Path(spec.output_dir) / "COMPLETE")
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return bool(
        result.get("status") == "completed"
        and result.get("run_id") == spec.run_id
        and result.get("updates_completed") == spec.updates
        and marker.get("run_id") == spec.run_id
        and (Path(spec.output_dir) / "checkpoint_trained.pt").is_file()
    )


def _scalar_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _summarize(
    root: Path, specs: Sequence[WorkerSpec], config: Mapping[str, Any]
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for cell in _cells(config):
        cell_specs = [spec for spec in specs if spec.cell.identifier == cell.identifier]
        results = [
            strict_json_load(Path(spec.output_dir) / "result.json")
            for spec in cell_specs
            if _result_complete(spec)
        ]
        results.sort(key=lambda item: int(item["model_seed"]))
        row: dict[str, Any] = {
            "cell": {
                "id": cell.identifier,
                "max_log_modulation": cell.max_log_modulation,
                "gate_output_weight_std": cell.gate_output_weight_std,
                "gate_output_bias": cell.gate_output_bias,
            },
            "completed_seed_count": len(results),
            "per_seed": results,
        }
        finite_task = [
            item
            for item in results
            if item["task_evaluation"].get("status") == "finite"
        ]
        row["finite_task_seed_count"] = len(finite_task)
        row["blank_stable_seed_count"] = sum(
            bool(item["autonomous_screen"]["stable_through_horizon"])
            for item in results
        )
        if finite_task:
            row["task_nmse_db"] = _scalar_summary(
                [
                    float(item["task_evaluation"]["metrics"]["masked_nmse_db"])
                    for item in finite_task
                ]
            )
        if results:
            for name in (
                "state_energy_weighted_lambda",
                "state_energy_fraction_at_lambda_above_one",
                "state_energy_fraction_at_abs_gate_tanh_at_least_0p99",
                "state_energy_participation_ratio",
                "state_norm_q95",
            ):
                row[f"task_endpoint_{name}"] = _scalar_summary(
                    [
                        float(
                            item["autonomous_screen"]["task_endpoint"][name]
                        )
                        for item in results
                    ]
                )
        conditions[cell.identifier] = row
    summary = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "updated_at_utc": _utc_now(),
        "reference_condition": config["reference_condition"],
        "conditions": conditions,
    }
    atomic_json(root / "summary.json", summary)
    return summary


def _git_info(repo: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(dirty), "porcelain": dirty}


def _parse_gpus(text: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    if not values or any(not value.isdigit() for value in values):
        raise ValueError("--gpus must contain comma-separated physical ids")
    if len(values) != len(set(values)):
        raise ValueError("--gpus contains duplicate ids")
    return values


def _launch(
    *,
    root: Path,
    config_path: Path,
    bank: Path,
    gpus: tuple[str, ...],
    smoke: bool,
) -> int:
    root.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    repo = Path(__file__).resolve().parents[2]
    git = _git_info(repo)
    if not smoke and git["dirty"]:
        raise RuntimeError("main H-C sweep requires committed code and a clean worktree")
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    destination = inputs / "config.json"
    if destination.exists() and destination.read_bytes() != config_path.read_bytes():
        raise RuntimeError("campaign config snapshot differs")
    if not destination.exists():
        shutil.copy2(config_path, destination)
    identity = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": _utc_now(),
        "git": git,
        "config_sha256": sha256_file(config_path),
        "bank": str(bank),
        "bank_sha256": sha256_file(bank),
        "gpus": list(gpus),
        "smoke": smoke,
        "scientific_contract": {
            "swept_only": ["max_log_modulation", "gate_output_initialization"],
            "paired_task_and_rp_streams": True,
            "noise": "disabled",
            "reference_reused_not_retrained": config["reference_condition"],
        },
    }
    identity_path = root / ("smoke_identity.json" if smoke else "campaign_identity.json")
    if identity_path.exists():
        previous = strict_json_load(identity_path)
        for key in ("campaign_id", "config_sha256", "bank_sha256", "smoke"):
            if previous.get(key) != identity.get(key):
                raise RuntimeError(f"existing campaign identity differs at {key}")
    else:
        atomic_json(identity_path, identity)

    specs = _build_specs(root, config_path, bank, smoke=smoke)
    pending = [spec for spec in specs if not _result_complete(spec)]
    if not pending:
        _summarize(root, specs, config)
        print("all H-C sweep jobs already complete", flush=True)
        return 0
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    queue = list(pending)
    active: dict[str, tuple[subprocess.Popen, WorkerSpec, Any]] = {}
    failure = False
    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            spec = queue.pop(0)
            output = Path(spec.output_dir)
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f"refusing to overwrite partial worker: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            log_handle = (logs / f"{spec.run_id}.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            command = [
                sys.executable,
                "-m",
                "repro.state_dependent_retention.run_hc_stability_sweep",
                "--stage",
                "worker",
                "--cell",
                spec.cell.identifier,
                "--seed",
                str(spec.model_seed),
                "--output-dir",
                spec.output_dir,
                "--bank",
                spec.bank,
                "--config",
                spec.config_path,
                "--updates",
                str(spec.updates),
                "--batch-size",
                str(spec.batch_size),
                "--device",
                "cuda:0",
            ]
            if spec.smoke:
                command.append("--smoke")
            process = subprocess.Popen(
                command,
                cwd=repo,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = (process, spec, log_handle)
            print(
                f"launched {spec.run_id} on physical GPU {gpu} pid={process.pid}",
                flush=True,
            )
        time.sleep(2.0)
        for gpu, (process, spec, handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del active[gpu]
            good = code == 0 and _result_complete(spec)
            print(f"finished {spec.run_id} exit={code} verified={good}", flush=True)
            if not good:
                failure = True
                queue.clear()
        if failure and not active:
            break
    _summarize(root, specs, config)
    return 1 if failure else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "main", "worker", "status"), required=True)
    parser.add_argument("--root")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--bank")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--cell")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve(strict=True)
    config = _load_config(config_path)
    if args.stage == "worker":
        if None in (args.cell, args.seed, args.output_dir, args.bank, args.updates, args.batch_size):
            parser.error("worker requires cell, seed, output-dir, bank, updates, batch-size")
        candidates = {cell.identifier: cell for cell in _cells(config)}
        if args.cell not in candidates:
            parser.error(f"unknown sweep cell: {args.cell}")
        spec = WorkerSpec(
            cell=candidates[args.cell],
            model_seed=int(args.seed),
            output_dir=str(Path(args.output_dir).expanduser().resolve()),
            bank=str(Path(args.bank).expanduser().resolve(strict=True)),
            config_path=str(config_path),
            updates=int(args.updates),
            batch_size=int(args.batch_size),
            smoke=bool(args.smoke),
        )
        run_worker(spec, args.device)
        return 0
    if args.root is None or args.bank is None:
        parser.error("smoke/main/status require --root and --bank")
    root = Path(args.root).expanduser().resolve()
    bank = Path(args.bank).expanduser().resolve(strict=True)
    if args.stage == "status":
        specs = _build_specs(root, config_path, bank, smoke=False)
        print(json.dumps(_summarize(root, specs, config), indent=2, sort_keys=True))
        return 0
    return _launch(
        root=root,
        config_path=config_path,
        bank=bank,
        gpus=_parse_gpus(args.gpus),
        smoke=args.stage == "smoke",
    )


if __name__ == "__main__":
    raise SystemExit(main())
