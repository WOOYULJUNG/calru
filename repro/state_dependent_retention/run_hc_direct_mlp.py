"""Train H-C with a direct residual MLP retention field.

This experiment changes only

    lambda(h) = base_lambda + MLP(h)

relative to the registered H-C exp(tanh) experiment.  The MLP output layer is
zero initialized, so every seed starts from exactly the same base-retention
map.  The recurrent writer, hybrid RP intervention, task streams, optimizer,
training horizon, and noise-free contract are unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from repro.sagodi_protocol import source_repaired_baselines_v6 as baseline_v6
from repro.sagodi_protocol.artifacts import atomic_json, sha256_file, strict_json_load
from repro.sagodi_protocol.metrics import masked_mse
from repro.sagodi_protocol.source_resolved_protocol import source_angular_integration
from repro.sagodi_protocol.tasks import Batch, load_fixed_bank
from repro.sagodi_protocol.train import _retention_plasticity_call

from .models import build_state_dependent_model
from .run_hc_stability_sweep import (
    _atomic_torch_save,
    _autonomous_screen,
    _finite_model,
    _native,
    _task_evaluation,
    _to_device,
    _utc_now,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "hc_direct_mlp.json"
CAMPAIGN_ID = "hc_direct_residual_mlp_v1"


@dataclass(frozen=True)
class WorkerSpec:
    model_seed: int
    output_dir: str
    bank: str
    config_path: str
    updates: int
    batch_size: int

    @property
    def run_id(self) -> str:
        return f"direct_mlp__seed{self.model_seed:02d}"


def _load_config(path: Path) -> dict[str, Any]:
    config = strict_json_load(path)
    if config.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("wrong direct-MLP campaign id")
    if config.get("model_seeds") != [0, 1, 2]:
        raise ValueError("direct-MLP campaign requires seeds 0,1,2")
    if config.get("fixed_architecture") != {
        "writer_kind": "recurrent",
        "retention_mode": "hybrid_rp",
        "retention_parameterization": "direct_residual_mlp",
        "width": 52,
        "mlp_hidden": 52,
        "mlp_output_weight_init": 0.0,
        "mlp_output_bias_init": 0.0,
    }:
        raise ValueError("direct-MLP architecture contract differs")
    if config.get("training") != {
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
    }:
        raise ValueError("direct-MLP training contract differs")
    if config.get("retention_plasticity") != {
        "warmup_updates": 1500,
        "eta_lambda": 1000.0,
        "damage_epsilon": 3e-5,
        "intervention_interval_updates": 50,
        "probe_batch_size": 96,
        "blank_ablation_horizon": 500,
    }:
        raise ValueError("direct-MLP RP contract differs")
    if config.get("screen") != {
        "blank_horizon": 2048,
        "progress_interval_updates": 50,
        "validation_interval_updates": 500,
    }:
        raise ValueError("direct-MLP screen contract differs")
    return config


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
    return source_angular_integration(
        int(config["retention_plasticity"]["probe_batch_size"]),
        0,
        stream_key=(
            baseline_v6.CAMPAIGN_ID,
            "rp_probe",
            spec.model_seed,
            int(update),
        ),
        device=device,
    )


def _rp_updates(spec: WorkerSpec, config: Mapping[str, Any]) -> set[int]:
    rp = config["retention_plasticity"]
    return {
        update
        for update in range(1, spec.updates + 1)
        if update > int(rp["warmup_updates"])
        and update % int(rp["intervention_interval_updates"]) == 0
    }


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
        gate_output_weight_std=0.0,
        gate_output_bias=0.0,
        retention_parameterization="direct_residual_mlp",
    ).to(device)
    _finite_model(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    training = config["training"]
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(training["learning_rate"]),
        betas=tuple(float(v) for v in training["betas"]),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    bank = _to_device(load_fixed_bank(spec.bank), device)
    rp_schedule = _rp_updates(spec, config)
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": spec.run_id,
        "model_seed": spec.model_seed,
        "equation": "lambda(h) = base_lambda + MLP(h)",
        "retention_parameterization": "direct_residual_mlp",
        "writer_kind": "recurrent",
        "retention_mode": "hybrid_rp",
        "training": {**training, "effective_updates": spec.updates},
        "retention_plasticity": config["retention_plasticity"],
        "parameters_total": total,
        "parameters_gradient_trainable": trainable,
        "paired_task_and_rp_streams_with_hc_sweep": True,
        "bank": str(Path(spec.bank).resolve()),
        "bank_sha256": sha256_file(spec.bank),
        "config_sha256": sha256_file(config_path),
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

        should_trace = update == 1 or update % progress_interval == 0
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
                f"[direct_mlp seed={spec.model_seed}] {update}/{spec.updates} "
                f"loss={row['train_mse']:.6g} rp={len(rp_trace)} "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )

    if len(rp_trace) != len(rp_schedule):
        raise RuntimeError("RP call count differs from schedule")
    _atomic_torch_save(
        output / "checkpoint_trained.pt",
        {
            "schema_version": 1,
            "checkpoint_type": CAMPAIGN_ID,
            "checkpoint_stage": "post_training_pre_screen",
            "model_seed": spec.model_seed,
            "updates_completed": spec.updates,
            "rp_call_count": len(rp_trace),
            "retention_parameterization": "direct_residual_mlp",
            "state_dict": model.state_dict(),
        },
    )
    task = _task_evaluation(model, bank)
    autonomous = _autonomous_screen(
        model, bank, horizon=int(config["screen"]["blank_horizon"])
    )
    result = {
        "schema_version": 1,
        "status": "completed",
        "campaign_id": CAMPAIGN_ID,
        "run_id": spec.run_id,
        "model_seed": spec.model_seed,
        "updates_completed": spec.updates,
        "rp_call_count": len(rp_trace),
        "parameters_total": total,
        "parameters_gradient_trainable": trainable,
        "task_evaluation": task,
        "autonomous_screen": autonomous,
        "elapsed_seconds": float(time.time() - started),
    }
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


def _result_complete(spec: WorkerSpec) -> bool:
    try:
        result = strict_json_load(Path(spec.output_dir) / "result.json")
        marker = strict_json_load(Path(spec.output_dir) / "COMPLETE")
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        result.get("status") == "completed"
        and result.get("run_id") == spec.run_id
        and result.get("updates_completed") == spec.updates
        and marker.get("run_id") == spec.run_id
        and (Path(spec.output_dir) / "checkpoint_trained.pt").is_file()
    )


def _specs(root: Path, config_path: Path, bank: Path) -> tuple[WorkerSpec, ...]:
    config = _load_config(config_path)
    return tuple(
        WorkerSpec(
            model_seed=int(seed),
            output_dir=str(root / "runs" / f"direct_mlp__seed{int(seed):02d}"),
            bank=str(bank),
            config_path=str(config_path),
            updates=int(config["training"]["updates"]),
            batch_size=int(config["training"]["batch_size"]),
        )
        for seed in config["model_seeds"]
    )


def _scalar(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _summarize(root: Path, specs: Sequence[WorkerSpec]) -> dict[str, Any]:
    results = [
        strict_json_load(Path(spec.output_dir) / "result.json")
        for spec in specs
        if _result_complete(spec)
    ]
    results.sort(key=lambda item: int(item["model_seed"]))
    finite = [r for r in results if r["task_evaluation"].get("status") == "finite"]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "updated_at_utc": _utc_now(),
        "completed_seed_count": len(results),
        "blank_stable_seed_count": sum(
            bool(r["autonomous_screen"]["stable_through_horizon"]) for r in results
        ),
        "per_seed": results,
    }
    if finite:
        summary["task_rmse"] = _scalar(
            [
                math.sqrt(float(r["task_evaluation"]["metrics"]["masked_mse"]))
                for r in finite
            ]
        )
    atomic_json(root / "summary.json", summary)
    return summary


def _launch(root: Path, config_path: Path, bank: Path, gpus: tuple[str, ...]) -> int:
    root.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    repo = Path(__file__).resolve().parents[2]
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("direct-MLP main run requires a clean committed worktree")
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    shutil.copy2(config_path, inputs / "config.json")
    atomic_json(
        root / "campaign_identity.json",
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "created_at_utc": _utc_now(),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip(),
            "config_sha256": sha256_file(config_path),
            "bank": str(bank),
            "bank_sha256": sha256_file(bank),
            "physical_gpus": list(gpus),
            "scientific_contract": {
                "changed_only": "retention_parameterization",
                "equation": "lambda(h) = base_lambda + MLP(h)",
                "paired_task_and_rp_streams": True,
                "noise": "disabled",
            },
        },
    )
    specs = _specs(root, config_path, bank)
    pending = [spec for spec in specs if not _result_complete(spec)]
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    active: list[tuple[subprocess.Popen, Any, WorkerSpec]] = []
    for spec, gpu in zip(pending, gpus, strict=False):
        output = Path(spec.output_dir)
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"refusing to overwrite partial worker: {output}")
        handle = (logs / f"{spec.run_id}.log").open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        command = [
            sys.executable,
            "-m",
            "repro.state_dependent_retention.run_hc_direct_mlp",
            "--stage",
            "worker",
            "--seed",
            str(spec.model_seed),
            "--root",
            str(root),
            "--bank",
            str(bank),
            "--config",
            str(config_path),
            "--device",
            "cuda:0",
        ]
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        active.append((process, handle, spec))
        print(f"launched {spec.run_id} on physical GPU {gpu} pid={process.pid}", flush=True)
    failure = False
    for process, handle, spec in active:
        code = process.wait()
        handle.close()
        good = code == 0 and _result_complete(spec)
        print(f"finished {spec.run_id} exit={code} verified={good}", flush=True)
        failure = failure or not good
    _summarize(root, specs)
    return 1 if failure else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("main", "worker", "status"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--bank", required=True)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config_path = Path(args.config).resolve(strict=True)
    bank = Path(args.bank).resolve(strict=True)
    config = _load_config(config_path)
    specs = _specs(root, config_path, bank)
    if args.stage == "status":
        print(json.dumps(_summarize(root, specs), indent=2, sort_keys=True))
        return 0
    if args.stage == "worker":
        if args.seed not in config["model_seeds"]:
            raise ValueError("worker seed is not registered")
        spec = next(spec for spec in specs if spec.model_seed == args.seed)
        run_worker(spec, args.device)
        return 0
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if len(gpus) < len([spec for spec in specs if not _result_complete(spec)]):
        raise ValueError("main stage needs one available GPU per pending seed")
    return _launch(root, config_path, bank, gpus)


if __name__ == "__main__":
    raise SystemExit(main())
