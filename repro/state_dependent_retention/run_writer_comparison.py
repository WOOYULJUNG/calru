"""Train the B/B+/C state-dependent-retention writer comparison.

The launcher assigns independent workers to GPUs and is safe to leave inside a
tmux session.  All three writer variants receive the same task streams,
optimizer settings, conditional RP schedule, and outer scaffold for a given
model seed.  State, target, and output noise are disabled.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from repro.sagodi_protocol import source_repaired_baselines_v6 as baseline_v6
from repro.sagodi_protocol.artifacts import (
    atomic_json,
    canonical_hash,
    derived_seed,
    sha256_file,
    strict_json_load,
)
from repro.sagodi_protocol.metrics import masked_mse, task_metrics
from repro.sagodi_protocol.source_resolved_protocol import source_angular_integration
from repro.sagodi_protocol.tasks import Batch, load_fixed_bank
from repro.sagodi_protocol.train import _retention_plasticity_call

from .models import (
    RETENTION_MODES,
    WRITER_KINDS,
    RetentionMode,
    WriterKind,
    build_state_dependent_model,
    recurrence,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "config.json"
CAMPAIGN_ID = "state_dependent_retention_writer_v1"


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
        raise ValueError("wrong state-dependent-retention campaign config")
    if tuple(config["writers"]) != WRITER_KINDS:
        raise ValueError("writer list differs from the registered B/B+/C comparison")
    if tuple(config["retention_modes"]) != RETENTION_MODES:
        raise ValueError("retention modes differ from the registered comparison")
    if config["model_seeds"] != [0, 1, 2]:
        raise ValueError("main comparison freezes three model seeds")
    training = config["training"]
    required = {
        "learning_rate": 0.01,
        "updates": 5000,
        "batch_size": 64,
        "state_noise_std": 0.0,
        "target_noise_std": 0.0,
        "output_dropout": 0.0,
        "evaluation_state_noise_std": 0.0,
    }
    for key, expected in required.items():
        if training.get(key) != expected:
            raise ValueError(f"training contract differs at {key}")
    retention = config["retention_learning"]
    if retention["gradient_only"] != {
        "retention_plasticity_enabled": False,
        "base_retention_task_gradient": True,
        "state_dependent_gate_task_gradient": True,
        "external_damage_intervention": False,
    }:
        raise ValueError("gradient-only retention contract differs")
    hybrid = retention["hybrid_rp"]
    if (
        not hybrid["retention_plasticity_enabled"]
        or hybrid["base_retention_task_gradient"]
        or not hybrid["state_dependent_gate_task_gradient"]
        or not hybrid["external_damage_intervention"]
        or hybrid["eta_lambda"] != 1000.0
        or hybrid["damage_epsilon"] != 3e-5
        or hybrid["intervention_interval_updates"] != 50
    ):
        raise ValueError("hybrid-RP retention contract differs")
    return config


@dataclass(frozen=True)
class WorkerSpec:
    writer_kind: WriterKind
    retention_mode: RetentionMode
    model_seed: int
    output_dir: str
    evaluation_bank: str
    config_path: str
    updates: int
    batch_size: int
    smoke: bool

    @property
    def run_id(self) -> str:
        prefix = "smoke" if self.smoke else "main"
        return (
            f"{prefix}__{self.retention_mode}__{self.writer_kind}"
            f"__seed{self.model_seed:02d}"
        )


def _to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        batch.inputs.to(device),
        batch.output_targets.to(device),
        batch.latent_targets.to(device),
        batch.mask.to(device),
        batch.metadata,
    )


def _initial_memory(batch: Batch) -> torch.Tensor:
    return batch.output_targets[0]


def _training_batch(spec: WorkerSpec, update: int, device: torch.device) -> Batch:
    # Deliberately use the already-frozen baseline campaign key so every writer
    # sees exactly the online task samples used in the preceding comparison.
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
    rp = config["retention_learning"]["hybrid_rp"]
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


@torch.no_grad()
def _evaluate(model: Any, batch: Batch) -> dict[str, Any]:
    model.eval()
    prediction = model.forward_sequence(
        batch.inputs, initial_memory=_initial_memory(batch)
    )
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("non-finite held-out prediction")
    return _native(
        task_metrics(
            prediction,
            batch.output_targets,
            batch.mask,
            batch.latent_targets,
        )
    )


@torch.no_grad()
def _blank_metrics(
    model: Any, batch: Batch, *, horizon: int
) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()
    _, states = model.forward_sequence(
        batch.inputs,
        initial_memory=_initial_memory(batch),
        return_states=True,
    )
    state = states[-1]
    blank = torch.zeros(
        state.shape[0], model.input_dim, device=state.device, dtype=state.dtype
    )
    for _ in range(int(horizon)):
        state = model.step(blank, state)
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError("non-finite blank rollout")
    decoded = model.decode(state)
    target = batch.output_targets[-1]
    mse = (decoded - target).square().mean()
    predicted_angle = torch.atan2(decoded[:, 1], decoded[:, 0])
    target_angle = torch.atan2(target[:, 1], target[:, 0])
    angular = torch.atan2(
        torch.sin(predicted_angle - target_angle),
        torch.cos(predicted_angle - target_angle),
    ).abs()
    return (
        {
            "horizon": int(horizon),
            "mse": float(mse.cpu()),
            "mean_angular_error_radians": float(angular.mean().cpu()),
            "median_angular_error_radians": float(angular.median().cpu()),
        },
        state,
    )


def _finite_model(model: torch.nn.Module, *, gradients: bool = False) -> None:
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is not None and not bool(torch.isfinite(value).all()):
            suffix = "gradient" if gradients else "parameter"
            raise FloatingPointError(f"non-finite {suffix}: {name}")


def _expected_rp_updates(
    spec: WorkerSpec, config: Mapping[str, Any]
) -> tuple[int, ...]:
    if spec.retention_mode != "hybrid_rp":
        return ()
    rp = config["retention_learning"]["hybrid_rp"]
    warmup = int(rp["warmup_updates"])
    interval = int(rp["intervention_interval_updates"])
    return tuple(
        value
        for value in range(1, int(spec.updates) + 1)
        if value > warmup and value % interval == 0
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
    architecture = config["architecture"]
    model = build_state_dependent_model(
        spec.writer_kind,
        model_seed=spec.model_seed,
        retention_mode=spec.retention_mode,
        gate_hidden=int(architecture["gate_hidden"]),
        max_log_modulation=float(architecture["max_log_modulation"]),
    ).to(device)
    _finite_model(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    training = config["training"]
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    bank = _to_device(load_fixed_bank(spec.evaluation_bank), device)
    state_noise_std = float(training["state_noise_std"])
    state_noise_seed = (
        derived_seed(spec.model_seed, CAMPAIGN_ID, "paired_state_noise")
        if state_noise_std > 0.0
        else None
    )
    state_generator = (
        torch.Generator(device=device.type).manual_seed(int(state_noise_seed))
        if state_noise_seed is not None
        else None
    )
    rp_updates = set(_expected_rp_updates(spec, config))
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": spec.run_id,
        "writer_kind": spec.writer_kind,
        "retention_mode": spec.retention_mode,
        "model_seed": spec.model_seed,
        "smoke": spec.smoke,
        "parameters_total": total,
        "parameters_gradient_trainable": trainable,
        "parameter_matching": architecture["parameter_matching"],
        "state_dependent_retention": {
            "gate_hidden": int(architecture["gate_hidden"]),
            "max_log_modulation": float(architecture["max_log_modulation"]),
            "lambda_formula": architecture["lambda_formula"],
            "permits_lambda_above_one": True,
        },
        "training": {
            **training,
            "effective_updates": spec.updates,
            "effective_batch_size": spec.batch_size,
        },
        "retention_learning": config["retention_learning"][spec.retention_mode],
        "paired_streams": {
            "task_stream": "shared_across_writers_within_seed",
            "state_noise_enabled": state_noise_std > 0.0,
            "state_noise_seed": state_noise_seed,
            "state_noise_stream": (
                "shared_across_writers_within_seed"
                if state_noise_seed is not None
                else "disabled"
            ),
        },
        "evaluation_bank": str(Path(spec.evaluation_bank).resolve()),
        "evaluation_bank_sha256": sha256_file(spec.evaluation_bank),
        "config_sha256": sha256_file(config_path),
        "started_at_utc": _utc_now(),
        "device": device_text,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    atomic_json(output / "run_manifest.json", manifest)

    trace: list[dict[str, Any]] = []
    rp_trace: list[dict[str, Any]] = []
    started = time.time()
    progress_interval = int(config["evaluation"]["progress_interval_updates"])
    validation_interval = int(config["evaluation"]["validation_interval_updates"])
    for update in range(1, spec.updates + 1):
        model.train()
        batch = _training_batch(spec, update, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_sequence(
            batch.inputs,
            initial_memory=_initial_memory(batch),
            state_noise_std=state_noise_std,
            noise_generator=state_generator,
        )
        loss = masked_mse(prediction, batch.output_targets, batch.mask)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()
        _finite_model(model, gradients=True)
        optimizer.step()
        _finite_model(model)

        if update in rp_updates:
            model.eval()
            probe = _probe_batch(spec, config, update, device)
            rp = config["retention_learning"]["hybrid_rp"]
            details = _retention_plasticity_call(
                model,
                probe,
                blank_horizon=int(rp["blank_ablation_horizon"]),
                eta_lambda=float(rp["eta_lambda"]),
                damage_epsilon=float(rp["damage_epsilon"]),
                initial_memory=_initial_memory(probe),
            )
            rp_trace.append({"update": update, **_native(details)})

        should_validate = update % validation_interval == 0 or update == spec.updates
        should_trace = (
            update == 1
            or update % progress_interval == 0
            or update == spec.updates
        )
        if should_trace or should_validate:
            row: dict[str, Any] = {
                "update": update,
                "train_mse": float(loss.detach().cpu()),
                "elapsed_seconds": float(time.time() - started),
                "rp_calls": len(rp_trace),
            }
            if should_validate:
                row["validation"] = _evaluate(model, bank)
            trace.append(row)
            atomic_json(output / "training_trace.json", trace)
            atomic_json(output / "rp_trace.json", rp_trace)
            atomic_json(
                output / "progress.json",
                {
                    "schema_version": 1,
                    "status": "running",
                    "run_id": spec.run_id,
                    "writer_kind": spec.writer_kind,
                    "retention_mode": spec.retention_mode,
                    "model_seed": spec.model_seed,
                    "update": update,
                    "updates_total": spec.updates,
                    "latest": row,
                    "updated_at_utc": _utc_now(),
                },
            )
            print(
                f"[{spec.retention_mode}/{spec.writer_kind} seed={spec.model_seed}] "
                f"{update}/{spec.updates} loss={row['train_mse']:.6g} "
                f"rp={len(rp_trace)} elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )

    expected_rp_count = len(rp_updates)
    if len(rp_trace) != expected_rp_count:
        raise RuntimeError(
            f"RP call count differs: {len(rp_trace)} != {expected_rp_count}"
        )
    final_metrics = _evaluate(model, bank)
    blank_metrics, blank_terminal = _blank_metrics(
        model,
        bank,
        horizon=(8 if spec.smoke else int(config["evaluation"]["blank_horizon"])),
    )
    primary_terminal = model.primary_from_reported(blank_terminal)
    dynamic_summary = recurrence(model).dynamic_retention_summary(primary_terminal)
    result = {
        "schema_version": 1,
        "status": "completed",
        "campaign_id": CAMPAIGN_ID,
        "run_id": spec.run_id,
        "writer_kind": spec.writer_kind,
        "retention_mode": spec.retention_mode,
        "model_seed": spec.model_seed,
        "updates_completed": spec.updates,
        "rp_call_count": len(rp_trace),
        "parameters_total": total,
        "parameters_gradient_trainable": trainable,
        "final_metrics": final_metrics,
        "blank_metrics": blank_metrics,
        "dynamic_retention_on_blank_terminal": dynamic_summary,
        "elapsed_seconds": float(time.time() - started),
    }
    checkpoint = output / "checkpoint_final.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "schema_version": 1,
            "checkpoint_type": CAMPAIGN_ID,
            "writer_kind": spec.writer_kind,
            "retention_mode": spec.retention_mode,
            "model_seed": spec.model_seed,
            "architecture": config["architecture"],
            "result": result,
            "state_dict": model.state_dict(),
        },
    )
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


def _git_info(repo: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(dirty), "porcelain": dirty}


def _build_specs(
    root: Path,
    config_path: Path,
    bank: Path,
    *,
    smoke: bool,
) -> tuple[WorkerSpec, ...]:
    config = _load_config(config_path)
    writers: Sequence[WriterKind] = tuple(config["writers"])
    modes: Sequence[RetentionMode] = tuple(config["retention_modes"])
    seeds = [0] if smoke else [int(value) for value in config["model_seeds"]]
    updates = 3 if smoke else int(config["training"]["updates"])
    batch_size = 4 if smoke else int(config["training"]["batch_size"])
    parent = root / ("smoke" if smoke else "runs")
    return tuple(
        WorkerSpec(
            writer_kind=writer,
            retention_mode=mode,
            model_seed=seed,
            output_dir=str(parent / f"{mode}__{writer}__seed{seed:02d}"),
            evaluation_bank=str(bank),
            config_path=str(config_path),
            updates=updates,
            batch_size=batch_size,
            smoke=smoke,
        )
        for mode in modes
        for writer in writers
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
        and (Path(spec.output_dir) / "checkpoint_final.pt").is_file()
    )


def _summarize(root: Path, specs: Sequence[WorkerSpec]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {
        f"{mode}__{writer}": []
        for mode in RETENTION_MODES
        for writer in WRITER_KINDS
    }
    for spec in specs:
        if _result_complete(spec):
            groups[f"{spec.retention_mode}__{spec.writer_kind}"].append(
                strict_json_load(Path(spec.output_dir) / "result.json")
            )
    conditions: dict[str, Any] = {}
    for condition, results in groups.items():
        results.sort(key=lambda item: int(item["model_seed"]))
        row: dict[str, Any] = {
            "completed_seed_count": len(results),
            "per_seed": results,
        }
        if results:
            for name, getter in {
                "task_nmse_db": lambda item: item["final_metrics"]["masked_nmse_db"],
                "task_mse": lambda item: item["final_metrics"]["masked_mse"],
                "blank_mse": lambda item: item["blank_metrics"]["mse"],
                "blank_mean_angular_error_radians": lambda item: item["blank_metrics"]["mean_angular_error_radians"],
                "dynamic_lambda_maximum": lambda item: item["dynamic_retention_on_blank_terminal"]["maximum"],
                "dynamic_lambda_fraction_above_one": lambda item: item["dynamic_retention_on_blank_terminal"]["fraction_above_one"],
            }.items():
                values = np.asarray([getter(item) for item in results], dtype=np.float64)
                row[f"median_{name}"] = float(np.median(values))
                row[f"mean_{name}"] = float(values.mean())
            row["parameters_total"] = int(results[0]["parameters_total"])
            row["parameters_gradient_trainable"] = int(
                results[0]["parameters_gradient_trainable"]
            )
        conditions[condition] = row
    summary = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "updated_at_utc": _utc_now(),
        "conditions": conditions,
    }
    atomic_json(root / "summary.json", summary)
    return summary


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
        raise RuntimeError("main launch requires committed code and a clean worktree")
    snapshot = root / "inputs"
    snapshot.mkdir(exist_ok=True)
    destination = snapshot / "config.json"
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
        "evaluation_bank": str(bank),
        "evaluation_bank_sha256": sha256_file(bank),
        "gpus": list(gpus),
        "smoke": smoke,
        "scientific_contract": {
            "writers": config["writers"],
            "retention_modes": config["retention_modes"],
            "state_dependent_retention_shared": True,
            "paired_task_streams": True,
            "state_noise": "disabled",
            "parameter_matching": config["architecture"]["parameter_matching"],
        },
    }
    identity_path = root / ("smoke_identity.json" if smoke else "campaign_identity.json")
    if identity_path.exists():
        previous = strict_json_load(identity_path)
        for key in ("campaign_id", "config_sha256", "evaluation_bank_sha256", "smoke"):
            if previous.get(key) != identity.get(key):
                raise RuntimeError(f"existing campaign identity differs at {key}")
    else:
        atomic_json(identity_path, identity)

    specs = _build_specs(root, config_path, bank, smoke=smoke)
    pending = [spec for spec in specs if not _result_complete(spec)]
    if not pending:
        _summarize(root, specs)
        print("all jobs already complete", flush=True)
        return 0
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    active: dict[str, tuple[subprocess.Popen, WorkerSpec, Any]] = {}
    queue = list(pending)
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
                "repro.state_dependent_retention.run_writer_comparison",
                "--stage",
                "worker",
                "--writer",
                spec.writer_kind,
                "--retention-mode",
                spec.retention_mode,
                "--seed",
                str(spec.model_seed),
                "--output-dir",
                spec.output_dir,
                "--bank",
                spec.evaluation_bank,
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
            print(
                f"finished {spec.run_id} exit={code} verified={good}", flush=True
            )
            if not good:
                failure = True
                queue.clear()
        if failure and not active:
            break
    _summarize(root, specs)
    return 1 if failure else 0


def _parse_gpus(text: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    if not values or any(not value.isdigit() for value in values):
        raise ValueError("--gpus must contain comma-separated physical GPU ids")
    if len(values) != len(set(values)):
        raise ValueError("--gpus contains duplicate ids")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "main", "worker", "status"), required=True)
    parser.add_argument("--root")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--bank")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--writer", choices=WRITER_KINDS)
    parser.add_argument("--retention-mode", choices=RETENTION_MODES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve(strict=True)
    if args.stage == "worker":
        if None in (
            args.writer,
            args.retention_mode,
            args.seed,
            args.output_dir,
            args.bank,
            args.updates,
            args.batch_size,
        ):
            parser.error(
                "worker requires writer, retention-mode, seed, output-dir, "
                "bank, updates, batch-size"
            )
        spec = WorkerSpec(
            writer_kind=args.writer,
            retention_mode=args.retention_mode,
            model_seed=int(args.seed),
            output_dir=str(Path(args.output_dir).expanduser().resolve()),
            evaluation_bank=str(Path(args.bank).expanduser().resolve(strict=True)),
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
        summary = _summarize(root, specs)
        print(json.dumps(summary, indent=2, sort_keys=True))
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
