#!/usr/bin/env python3
"""Build and run isolated P0 CA-LRU confirmatory campaigns.

The launcher writes only beneath a marked experimental_v2 artifact root.  It
never passes ``--force`` to a worker, skips complete jobs, and blocks partial
artifact sets instead of overwriting them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LEGACY_DIR = REPO_ROOT / "repro" / "legacy_code"
LEGACY_RUNNER = REPO_ROOT / "repro" / "legacy_code" / "exp88_manifold_attractor_tasks.py"
AUX_RUNNER = HERE / "train_aux_blank.py"
FIXED_RUNNER = HERE / "train_fixed_retention.py"
ROOT_MARKER = ".calru_experimental_v2_root"
ROOT_MARKER_CONTENT = "calru-experimental-v2\n"
SCHEMA_VERSION = 1
COMPLETION_RECEIPT_VERSION = 1
RUNTIME_RECEIPT_VERSION = 1
THETA_CLIP = 18.0
UNIFORM_CAP_LAMBDA = (1.0 / (1.0 + math.exp(-THETA_CLIP))) ** 0.5

SOURCE_CLOSURE = (
    REPO_ROOT / "repro" / "legacy_code" / "exp88_manifold_attractor_tasks.py",
    REPO_ROOT / "repro" / "legacy_code" / "exp72_structured_attractor_tasks.py",
    REPO_ROOT / "repro" / "legacy_code" / "exp71_pan_block_pulse_hold.py",
    REPO_ROOT / "repro" / "legacy_code" / "pan_block.py",
    REPO_ROOT / "repro" / "legacy_code" / "plru_regularizers.py",
    HERE / "launch_p0.py",
    HERE / "train_aux_blank.py",
    HERE / "train_fixed_retention.py",
)

NPZ_REQUIRED_KEYS = {
    "_loss_trace.npz": ("steps", "train_total_loss", "train_task_loss"),
    "_lambda_trace.npz": (
        "steps",
        "lambdas",
        "lambda_gt_0p99",
        "pan_score_mean",
        "pan_score_max",
    ),
    "_aux_trace.npz": ("train_total_loss", "train_task_loss", "aux_steps", "aux_loss"),
}

TASKS = (
    "ring_hold",
    "ring_integrate",
    "torus_hold",
    "torus_integrate",
    "complex_curve_hold",
    "complex_curve_integrate",
    "surface_hold",
    "surface_integrate",
)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    if not slug:
        raise ValueError(f"cannot create slug from {value!r}")
    return slug


def _lambda_slug(value: float) -> str:
    text = f"{float(value):.12g}".replace("-", "m").replace(".", "p").replace("+", "")
    return _slug(text)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _atomic_write_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(encoded, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_create_json(path: Path, payload: Any) -> None:
    """Atomically publish an immutable receipt, refusing replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace immutable receipt {path}")
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to replace immutable receipt {path}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def _source_hash_closure() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in SOURCE_CLOSURE:
        resolved = path.resolve()
        if not resolved.is_file() or not _is_relative_to(resolved, REPO_ROOT.resolve()):
            raise RuntimeError(f"source closure member is missing or outside the repository: {path}")
        hashes[str(resolved.relative_to(REPO_ROOT.resolve()))] = _sha256_file(resolved)
    return dict(sorted(hashes.items()))


def _software_versions() -> dict[str, Any]:
    """Return scientific software versions without querying GPU hardware."""

    import numpy as np
    import torch

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_build": torch.backends.cudnn.version(),
    }


def _load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("campaign config must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"config schema_version must equal {SCHEMA_VERSION}")
    return payload


def _prepare_artifact_root(root: Path) -> None:
    root = root.resolve()
    marker = root / ROOT_MARKER
    if root.exists():
        if marker.exists():
            if marker.read_text(encoding="utf-8") != ROOT_MARKER_CONTENT:
                raise RuntimeError(f"invalid v2 artifact marker: {marker}")
        else:
            entries = list(root.iterdir())
            if entries:
                raise RuntimeError(
                    f"refusing non-empty unmarked artifact root {root}; this protects legacy artifacts"
                )
            marker.write_text(ROOT_MARKER_CONTENT, encoding="utf-8")
    else:
        root.mkdir(parents=True)
        marker.write_text(ROOT_MARKER_CONTENT, encoding="utf-8")


def _validate_config(config: dict) -> None:
    label = config.get("campaign_label", "")
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        raise ValueError("campaign_label must contain only letters, numbers, '.', '_' or '-'")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(not isinstance(x, int) or x < 0 for x in seeds):
        raise ValueError("seeds must be a non-empty list of non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    gpus = config.get("gpus")
    if not isinstance(gpus, list) or not gpus or any(not isinstance(x, int) or x < 0 for x in gpus):
        raise ValueError("gpus must be a non-empty list of non-negative integers")
    if len(set(gpus)) != len(gpus):
        raise ValueError("gpus must be unique")
    groups = config.get("task_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("task_groups must be a non-empty object")
    for name, tasks in groups.items():
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"task group {name!r} must be non-empty")
        unknown = sorted(set(tasks) - set(TASKS))
        if unknown:
            raise ValueError(f"task group {name!r} contains unknown tasks: {unknown}")
        if len(set(tasks)) != len(tasks):
            raise ValueError(f"task group {name!r} contains duplicate tasks")
    required_training = {
        "steps",
        "batch",
        "eval_batch",
        "analysis_batch",
        "train_min",
        "train_max",
        "id_horizon",
        "lr",
        "d_model",
        "rec_dim",
    }
    missing = sorted(required_training - set(config.get("training", {})))
    if missing:
        raise ValueError(f"training config is missing: {missing}")
    rp = config.get("rp_controls", {})
    if rp.get("enabled", False):
        if rp.get("task_group") not in groups:
            raise ValueError("rp_controls.task_group is not defined")
        for value in rp.get("uniform_lambdas", []):
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"uniform lambda must lie in (0, 1): {value}")
    names: list[str] = []
    for section in ("standard_models",):
        for spec in config.get(section, []):
            names.append(spec["name"])
            if spec["task_group"] not in groups:
                raise ValueError(f"unknown task group for {spec['name']}")
    aux = config.get("auxiliary_controls", {})
    if aux.get("enabled", False):
        if aux.get("task_group") not in groups:
            raise ValueError("auxiliary_controls.task_group is not defined")
        for spec in aux.get("models", []):
            names.append(spec["name"])
    if len(set(names)) != len(names):
        raise ValueError("model/control names must be unique across config sections")
    forbidden = config.get("compute_matched_controls", [])
    if forbidden:
        raise ValueError(
            "compute_matched_controls are deliberately unsupported: horizon information can be matched, "
            "but RP's coordinate-ablation compute has no implemented equivalent"
        )


def _implemented_model(model_name: str, task: str, d_model: int, rec_dim: int) -> tuple[bool, str]:
    """Check the current legacy normalizer and builder without changing them."""

    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))
    try:
        import exp88_manifold_attractor_tasks as exp88

        variant = exp88.normalize_model_variant(model_name)
        input_dim, output_dim = exp88.task_io_dims(task)
        exp88.build_model_variant(
            variant=variant,
            input_dim=input_dim,
            output_dim=output_dim,
            rank=exp88.model_rank_for_task(task),
            d_model=int(d_model),
            rec_dim=int(rec_dim),
            layers=1,
            dropout=0.0,
            plru_tau=0.001,
            plru_c=50.0,
            pan_lambda_min=0.90,
            pan_lambda_max=0.999,
            rank_matched_lambda_high=0.999,
            rank_matched_lambda_low=0.0,
        )
        return True, variant
    except Exception as exc:  # exact builder error belongs in the manifest
        return False, f"{type(exc).__name__}: {exc}"


def _resolve_model_training(base_training: dict, spec: dict, task: str) -> tuple[dict, dict]:
    """Apply task-specific CA-LRU parameter matching for modern full blocks."""

    resolved = dict(base_training)
    resolved.update(spec.get("training_overrides", {}))
    metadata: dict[str, Any] = {}
    if not spec.get("parameter_match_to_ca_lru", False):
        return resolved, metadata

    worker_model = spec.get("worker_model", spec["model"])
    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))
    try:
        import exp88_manifold_attractor_tasks as exp88
        from exp71_pan_block_pulse_hold import select_scaffold_matched_rec_dim

        input_dim, output_dim = exp88.task_io_dims(task)
        match = select_scaffold_matched_rec_dim(
            worker_model,
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=int(resolved.get("d_model", 96)),
            reference_rec_dim=int(spec.get("reference_rec_dim", 96)),
            layers=int(resolved.get("layers", 1)),
            dropout=float(resolved.get("dropout", 0.0)),
        )
    except Exception as exc:
        metadata["parameter_match_error"] = f"{type(exc).__name__}: {exc}"
        return resolved, metadata

    max_relative_error = float(spec.get("max_parameter_mismatch", 0.05))
    if match.relative_error > max_relative_error:
        raise ValueError(
            f"{spec['name']} parameter mismatch on {task} is {match.relative_error:.3%}, "
            f"above the configured {max_relative_error:.3%}"
        )
    resolved["rec_dim"] = int(match.candidate_rec_dim)
    metadata["parameter_match"] = {
        "reference_model": "PAN-RNW-full",
        "reference_rec_dim": int(spec.get("reference_rec_dim", 96)),
        "target_params": int(match.target_params),
        "candidate_params": int(match.candidate_params),
        "target_trainable_params": int(match.target_trainable_params),
        "candidate_trainable_params": int(match.candidate_trainable_params),
        "candidate_rec_dim": int(match.candidate_rec_dim),
        "relative_error": float(match.relative_error),
        "max_allowed_relative_error": max_relative_error,
    }
    return resolved, metadata


def _common_args(training: dict) -> list[str]:
    def many(flag: str, values: list[Any]) -> list[str]:
        return [flag, *[str(value) for value in values]]

    args = [
        "--steps",
        str(training["steps"]),
        "--batch",
        str(training["batch"]),
        "--eval-batch",
        str(training["eval_batch"]),
        "--analysis-batch",
        str(training["analysis_batch"]),
        "--train-min",
        str(training["train_min"]),
        "--train-max",
        str(training["train_max"]),
        "--id-horizon",
        str(training["id_horizon"]),
        *many("--temporal-horizons", training.get("temporal_horizons", [500, 1000, 2000])),
        *many("--post-holds", training.get("post_holds", [500, 1000])),
        *many("--recovery-steps", training.get("recovery_steps", [0, 20, 100, 500])),
        *many("--normal-radii", training.get("normal_radii", [0.25, 0.5, 1.0])),
        *many("--repeated-kicks", training.get("repeated_kicks", [1, 20, 100])),
        "--repeated-horizon",
        str(training.get("repeated_horizon", 500)),
        *many("--velocity-scales", training.get("velocity_scales", [1.0, 1.5, 2.0])),
        "--hold-min",
        str(training.get("hold_min", 5)),
        "--hold-max",
        str(training.get("hold_max", 20)),
        "--move-min",
        str(training.get("move_min", 3)),
        "--move-max",
        str(training.get("move_max", 10)),
        "--final-hold-min",
        str(training.get("final_hold_min", 20)),
        "--final-hold-max",
        str(training.get("final_hold_max", 80)),
        "--ood-hold-min",
        str(training.get("ood_hold_min", 30)),
        "--ood-hold-max",
        str(training.get("ood_hold_max", 120)),
        "--ood-final-hold-min",
        str(training.get("ood_final_hold_min", 100)),
        "--ood-final-hold-max",
        str(training.get("ood_final_hold_max", 250)),
        "--ring-velocity-deg",
        str(training.get("ring_velocity_deg", 3.0)),
        "--torus-velocity-deg",
        str(training.get("torus_velocity_deg", 2.5)),
        "--curve-velocity-deg",
        str(training.get("curve_velocity_deg", 2.5)),
        "--surface-velocity-scale",
        str(training.get("surface_velocity_scale", 0.018)),
        "--d-model",
        str(training["d_model"]),
        "--rec-dim",
        str(training["rec_dim"]),
        "--layers",
        str(training.get("layers", 1)),
        "--dropout",
        str(training.get("dropout", 0.0)),
        "--lr",
        str(training["lr"]),
        "--grad-clip",
        str(training.get("grad_clip", 1.0)),
        "--plru-tau",
        str(training.get("plru_tau", 0.001)),
        "--plru-c",
        str(training.get("plru_c", 50.0)),
        "--rank-matched-lambda-high",
        str(training.get("rank_matched_lambda_high", 0.999)),
        "--rank-matched-lambda-low",
        str(training.get("rank_matched_lambda_low", 0.0)),
        "--slow-lambda-init-mode",
        str(training.get("slow_lambda_init_mode", "linspace")),
        "--slow-lambda-min",
        str(training.get("slow_lambda_min", 0.90)),
        "--slow-lambda-max",
        str(training.get("slow_lambda_max", 0.999)),
        "--pan-lambda-min",
        str(training.get("pan_lambda_min", 0.90)),
        "--pan-lambda-max",
        str(training.get("pan_lambda_max", 0.999)),
        "--pan-warmup-frac",
        str(training.get("pan_warmup_frac", 0.3)),
        "--train-data-seed-base",
        str(training.get("train_data_seed_base", 880071)),
        "--probe-seed-base",
        str(training.get("probe_seed_base", 880072)),
        "--eval-seed-base",
        str(training.get("eval_seed_base", 880073)),
        "--device",
        "auto",
    ]
    if training.get("log_loss_trajectory", True):
        args.append("--log-loss-trajectory")
    if training.get("deterministic_training", True):
        args.append("--deterministic-training")
    if "--force" in args:
        raise AssertionError("experimental_v2 must never pass --force")
    return args


@dataclass(frozen=True)
class Job:
    job_id: str
    family: str
    condition: str
    task: str
    model: str
    worker_model: str
    seed: int
    runner: str
    command: tuple[str, ...]
    implemented: bool
    implementation_note: str
    expected: tuple[str, ...]
    log_dir: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "family": self.family,
            "condition": self.condition,
            "task": self.task,
            "model": self.model,
            "worker_model": self.worker_model,
            "seed": self.seed,
            "runner": self.runner,
            "command": list(self.command),
            "implemented": self.implemented,
            "implementation_note": self.implementation_note,
            "expected": list(self.expected),
            "log_dir": self.log_dir,
            "metadata": self.metadata,
        }


def _job(
    *,
    family: str,
    condition: str,
    task: str,
    model: str,
    worker_model: str | None = None,
    seed: int,
    runner: str,
    extra_args: list[str],
    training: dict,
    smoke: bool,
    metadata: dict[str, Any] | None = None,
) -> Job:
    condition_slug = _slug(condition)
    worker_model = worker_model or model
    tag = f"v2_{condition_slug}"
    results = f"results/{condition_slug}"
    checkpoints = f"checkpoints/{condition_slug}"
    traces = f"traces/{condition_slug}"
    logs = f"logs/{condition_slug}"
    runner_paths = {
        "legacy": "repro/legacy_code/exp88_manifold_attractor_tasks.py",
        "aux": "repro/experimental_v2/train_aux_blank.py",
        "fixed": "repro/experimental_v2/train_fixed_retention.py",
    }
    runner_rel = runner_paths[runner]
    command = [
        "<PYTHON>",
        f"<REPO_ROOT>/{runner_rel}",
        "--task",
        task,
        "--model",
        worker_model,
        "--tag",
        tag,
        "--seed",
        str(seed),
        *_common_args(training),
        *extra_args,
        "--out-dir",
        f"<CAMPAIGN_DIR>/{results}",
        "--ckpt-dir",
        f"<CAMPAIGN_DIR>/{checkpoints}",
        "--trace-dir",
        f"<CAMPAIGN_DIR>/{traces}",
    ]
    if runner in ("aux", "fixed"):
        command.extend(["--v2-artifact-root", "<ARTIFACT_ROOT>"])
    if smoke:
        command.append("--smoke")
    if "--force" in command:
        raise AssertionError("experimental_v2 must never pass --force")

    result = f"{results}/{task}_{tag}_seed{seed}.json"
    ckpt = f"{checkpoints}/exp88_{task}_{tag}_seed{seed}.pt"
    expected = [result, ckpt]
    if runner == "aux":
        expected.append(f"{traces}/{task}_{tag}_seed{seed}_aux_trace.npz")
    else:
        if training.get("log_loss_trajectory", True):
            expected.append(f"{traces}/{task}_{tag}_seed{seed}_loss_trace.npz")
        if family == "rp_control":
            expected.append(f"{traces}/{task}_{tag}_seed{seed}_lambda_trace.npz")

    d_model = int(training.get("d_model", 96))
    rec_dim = int(training.get("rec_dim", 96))
    implemented, note = _implemented_model(worker_model, task, d_model, rec_dim)
    if metadata and "parameter_match_error" in metadata:
        implemented = False
        note = f"parameter matching failed: {metadata['parameter_match_error']}"
    identity = {
        "family": family,
        "condition": condition_slug,
        "task": task,
        "model": model,
        "worker_model": worker_model,
        "seed": seed,
        "runner": runner,
        "command": command,
    }
    job_id = f"{condition_slug}-{task}-s{seed}-{hashlib.sha256(_canonical_bytes(identity)).hexdigest()[:10]}"
    return Job(
        job_id=job_id,
        family=family,
        condition=condition_slug,
        task=task,
        model=model,
        worker_model=worker_model,
        seed=seed,
        runner=runner,
        command=tuple(command),
        implemented=implemented,
        implementation_note=note,
        expected=tuple(expected),
        log_dir=logs,
        metadata=dict(metadata or {}),
    )


def _apply_enable_overrides(config: dict, enabled: set[str], disabled: set[str]) -> None:
    known: set[str] = set()
    for spec in config.get("standard_models", []):
        known.add(spec["name"])
        if spec["name"] in enabled:
            spec["enabled"] = True
        if spec["name"] in disabled:
            spec["enabled"] = False
    for spec in config.get("auxiliary_controls", {}).get("models", []):
        known.add(spec["name"])
        if spec["name"] in enabled:
            spec["enabled"] = True
        if spec["name"] in disabled:
            spec["enabled"] = False
    unknown = sorted((enabled | disabled) - known)
    if unknown:
        raise ValueError(f"unknown --enable/--disable names: {unknown}; known names: {sorted(known)}")


def build_jobs(config: dict, smoke: bool = False) -> list[Job]:
    groups = config["task_groups"]
    seeds = sorted(config["seeds"])
    training = dict(config["training"])
    jobs: list[Job] = []

    rp = config.get("rp_controls", {})
    if rp.get("enabled", False):
        rp_tasks = groups[rp["task_group"]]
        eps = float(rp.get("epsilon", 3e-5))
        probe_every = int(rp.get("probe_every", 100))
        probe_batch = int(rp.get("probe_batch", 96))
        probe_sequence_horizon = int(rp.get("probe_sequence_horizon", 260))
        probe_blank_horizon = int(rp.get("probe_blank_horizon", 500))
        controls: list[tuple[str, list[str]]] = []
        if rp.get("include_aligned", True):
            controls.append(
                (
                    "rp_aligned",
                    [
                        "--pan-score-eps",
                        str(eps),
                        "--pan-eta-lambda",
                        str(rp.get("eta_lambda", 3000)),
                        "--pan-probe-every",
                        str(probe_every),
                        "--pan-probe-batch",
                        str(probe_batch),
                        "--pan-probe-horizon",
                        str(probe_sequence_horizon),
                        "--pan-h-probe",
                        str(probe_blank_horizon),
                        "--pan-eps-mode",
                        "fixed",
                        "--pan-score-mode",
                        "damage",
                        "--log-lambda-trajectory",
                        "--lambda-log-every",
                        str(rp.get("lambda_log_every", 1000)),
                    ],
                )
            )
        if rp.get("include_off", True):
            controls.append(
                (
                    "rp_off_frozen_retention",
                    [
                        "--pan-score-eps",
                        str(eps),
                        "--pan-eta-lambda",
                        "0",
                        "--pan-probe-every",
                        str(int(training["steps"]) + 1),
                        "--log-lambda-trajectory",
                        "--lambda-log-every",
                        str(rp.get("lambda_log_every", 1000)),
                    ],
                )
            )
        uniform_values: list[tuple[str, float]] = []
        if rp.get("include_uniform_cap", True):
            uniform_values.append(("uniform_retention_theta_cap", UNIFORM_CAP_LAMBDA))
        for value in sorted(set(float(item) for item in rp.get("uniform_lambdas", []))):
            if abs(value - UNIFORM_CAP_LAMBDA) < 1e-12:
                continue
            uniform_values.append((f"uniform_retention_{_lambda_slug(value)}", value))
        for name, value in uniform_values:
            theta = math.log(float(value) ** 2 / (1.0 - float(value) ** 2))
            theta = max(-THETA_CLIP, min(THETA_CLIP, theta))
            controls.append(
                (
                    name,
                    [
                        "--pan-score-eps",
                        str(eps),
                        "--pan-eta-lambda",
                        "0",
                        "--pan-probe-every",
                        str(int(training["steps"]) + 1),
                        "--force-all-slow",
                        "--all-slow-lambda",
                        repr(float(value)),
                        "--fixed-pan-theta",
                        repr(float(theta)),
                        "--log-lambda-trajectory",
                        "--lambda-log-every",
                        str(rp.get("lambda_log_every", 1000)),
                    ],
                )
            )
        for condition, extra in controls:
            control_runner = "fixed" if condition.startswith("uniform_retention_") else "legacy"
            for task in rp_tasks:
                for seed in seeds:
                    jobs.append(
                        _job(
                            family="rp_control",
                            condition=condition,
                            task=task,
                            model="PAN-RNW-full",
                            seed=seed,
                            runner=control_runner,
                            extra_args=extra,
                            training=training,
                            smoke=smoke,
                        )
                    )

    for spec in config.get("standard_models", []):
        if not spec.get("enabled", False):
            continue
        for task in groups[spec["task_group"]]:
            model_training, match_metadata = _resolve_model_training(training, spec, task)
            for seed in seeds:
                jobs.append(
                    _job(
                        family="standard_baseline",
                        condition=spec["name"],
                        task=task,
                        model=spec["model"],
                        worker_model=spec.get("worker_model", spec["model"]),
                        seed=seed,
                        runner="legacy",
                        extra_args=list(spec.get("extra_args", [])),
                        training=model_training,
                        smoke=smoke,
                        metadata=match_metadata,
                    )
                )

    aux = config.get("auxiliary_controls", {})
    if aux.get("enabled", False):
        aux_tasks = groups[aux["task_group"]]
        aux_common = [
            "--aux-loss-weight",
            str(aux.get("loss_weight", 1.0)),
            "--aux-every",
            str(aux.get("every", 100)),
            "--aux-start-step",
            str(aux.get("start_step", -1)),
            "--aux-batch",
            str(aux.get("batch", 96)),
            "--aux-sequence-horizon",
            str(aux.get("sequence_horizon", 260)),
            "--aux-blank-horizon",
            str(aux.get("blank_horizon", 500)),
            "--pan-theta-clip",
            str(aux.get("pan_theta_clip", THETA_CLIP)),
        ]
        for spec in aux.get("models", []):
            if not spec.get("enabled", False):
                continue
            extra = [*aux_common, *list(spec.get("extra_args", []))]
            if spec.get("train_pan_theta", False):
                extra.append("--train-pan-theta")
            for task in aux_tasks:
                model_training, match_metadata = _resolve_model_training(training, spec, task)
                for seed in seeds:
                    jobs.append(
                        _job(
                            family="horizon_information_control",
                            condition=spec["name"],
                            task=task,
                            model=spec["model"],
                            worker_model=spec.get("worker_model", spec["model"]),
                            seed=seed,
                            runner="aux",
                            extra_args=extra,
                            training=model_training,
                            smoke=smoke,
                            metadata=match_metadata,
                        )
                    )

    jobs.sort(key=lambda job: (job.family, job.condition, TASKS.index(job.task), job.seed, job.job_id))
    ids = [job.job_id for job in jobs]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate deterministic job IDs")
    return jobs


def _scientific_config(config: dict, smoke: bool) -> dict:
    payload = json.loads(json.dumps(config))
    payload.pop("gpus", None)
    return {"config": payload, "smoke": bool(smoke)}


def build_manifest(config: dict, jobs: list[Job], smoke: bool) -> tuple[str, dict]:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "scientific_config": _scientific_config(config, smoke),
        "source_hash_closure": _source_hash_closure(),
        "software_versions": _software_versions(),
        "uniform_cap": {
            "theta_clip": THETA_CLIP,
            "lambda": UNIFORM_CAP_LAMBDA,
            "lambda_power_500": UNIFORM_CAP_LAMBDA**500,
            "lambda_power_1000": UNIFORM_CAP_LAMBDA**1000,
        },
        "jobs": [job.as_dict() for job in jobs],
        "runtime_contract": {
            "receipt_schema_version": RUNTIME_RECEIPT_VERSION,
            "python_placeholder": "<PYTHON>",
            "path_placeholders": ["<REPO_ROOT>", "<ARTIFACT_ROOT>", "<CAMPAIGN_DIR>"],
            "gpu_assignment_is_not_scientific_identity": True,
        },
        "fairness_notes": [
            "horizon_information_auxiliary controls are not compute-matched controls",
            "RP-off freezes the initial retention spectrum",
            "uniform_retention_theta_cap uses sqrt(sigmoid(18)), not the legacy lambda=0.999",
            "legacy lambda=0.999 retains only about 0.606 after 500 blank steps",
        ],
    }
    digest = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    campaign_id = f"{_slug(config['campaign_label'])}-{digest[:12]}"
    manifest = {**identity, "campaign_id": campaign_id, "manifest_identity_sha256": digest}
    return campaign_id, manifest


def _verified_manifest_identity(campaign_dir: Path) -> tuple[str, str]:
    path = campaign_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot parse campaign manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"campaign manifest is not a JSON object: {path}")
    campaign_id = manifest.get("campaign_id")
    claimed = manifest.get("manifest_identity_sha256")
    if not isinstance(campaign_id, str) or not isinstance(claimed, str):
        raise RuntimeError(f"campaign manifest is missing identity fields: {path}")
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"campaign_id", "manifest_identity_sha256"}
    }
    actual = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    if actual != claimed:
        raise RuntimeError(f"campaign manifest identity digest is invalid: {path}")
    if campaign_dir.name != campaign_id:
        raise RuntimeError(f"campaign directory/name mismatch: {campaign_dir.name!r} != {campaign_id!r}")
    return campaign_id, claimed


def _runtime_receipt(
    config: dict,
    *,
    max_parallel: int,
    dry_run: bool,
    selected_jobs: list[Job] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_RECEIPT_VERSION,
        "gpus": list(config["gpus"]),
        "max_parallel": int(max_parallel),
        "dry_run": bool(dry_run),
        "selected_job_ids": sorted(job.job_id for job in (selected_jobs or [])),
        "python_executable": str(Path(sys.executable).resolve()),
        "repository_root": str(REPO_ROOT.resolve()),
    }


def _write_or_check_runtime_receipt(campaign_dir: Path, receipt: dict[str, Any]) -> Path:
    """Store concrete runtime allocation outside the scientific manifest."""

    digest = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    path = campaign_dir / "runtime_receipts" / f"runtime-{digest[:16]}.json"
    encoded = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"runtime receipt digest collision or corruption: {path}")
        return path
    _atomic_create_json(path, receipt)
    return path


def _write_or_check_manifest(campaign_dir: Path, manifest: dict) -> None:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    path = campaign_dir / "manifest.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != encoded:
            raise RuntimeError(f"existing manifest differs; refusing to overwrite {path}")
        return
    _atomic_create_json(path, manifest)
    _verified_manifest_identity(campaign_dir)


def _materialize_command(command: tuple[str, ...], root: Path, campaign_dir: Path) -> list[str]:
    replacements = {
        "<PYTHON>": sys.executable,
        "<REPO_ROOT>": str(REPO_ROOT),
        "<ARTIFACT_ROOT>": str(root),
        "<CAMPAIGN_DIR>": str(campaign_dir),
    }
    result = []
    for item in command:
        for token, value in replacements.items():
            item = item.replace(token, value)
        result.append(item)
    if "--force" in result:
        raise AssertionError("refusing a command containing --force")
    return result


def _expected_paths(job: Job, campaign_dir: Path) -> list[Path]:
    campaign = campaign_dir.resolve()
    expected = [(campaign / relative).resolve() for relative in job.expected]
    if any(not _is_relative_to(path, campaign) for path in expected):
        raise RuntimeError(f"job {job.job_id} escaped its campaign directory")
    return expected


def _completion_receipt_path(job: Job, campaign_dir: Path) -> Path:
    path = (campaign_dir / "completion_receipts" / job.condition / f"{job.job_id}.json").resolve()
    if not _is_relative_to(path, campaign_dir.resolve()):
        raise RuntimeError(f"job {job.job_id} receipt escaped its campaign directory")
    return path


def _validate_result_json(job: Job, path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"JSON artifact must be a non-empty object: {path}")
    expected_tag = f"v2_{job.condition}"
    if payload.get("task") != job.task:
        raise RuntimeError(f"JSON task mismatch for {path}")
    if payload.get("seed") != job.seed:
        raise RuntimeError(f"JSON seed mismatch for {path}")
    if payload.get("tag") != expected_tag:
        raise RuntimeError(f"JSON tag mismatch for {path}")


def _validate_checkpoint(path: Path) -> None:
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"checkpoint is not loadable with torch.load(weights_only=True): {path}: {exc}") from exc
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"checkpoint must contain a non-empty state dictionary: {path}")


def _npz_required_keys(path: Path) -> tuple[str, ...]:
    for suffix, keys in NPZ_REQUIRED_KEYS.items():
        if path.name.endswith(suffix):
            return keys
    raise RuntimeError(f"unrecognized trace artifact type: {path}")


def _validate_npz(path: Path) -> None:
    import numpy as np

    required = _npz_required_keys(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(required) - set(archive.files))
            if missing:
                raise RuntimeError(f"NPZ artifact {path} is missing keys: {missing}")
            arrays = {key: np.asarray(archive[key]) for key in required}
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"invalid NPZ artifact {path}: {exc}") from exc
    empty = [key for key, value in arrays.items() if value.size == 0]
    if empty:
        raise RuntimeError(f"NPZ artifact {path} has empty required arrays: {empty}")

    if path.name.endswith("_loss_trace.npz"):
        lengths = {arrays[key].shape[0] for key in required}
        if len(lengths) != 1:
            raise RuntimeError(f"loss trace arrays have inconsistent lengths: {path}")
    elif path.name.endswith("_lambda_trace.npz"):
        length = arrays["steps"].shape[0]
        if any(arrays[key].shape[0] != length for key in required[1:]):
            raise RuntimeError(f"lambda trace arrays have inconsistent leading dimensions: {path}")
    elif path.name.endswith("_aux_trace.npz"):
        if arrays["train_total_loss"].shape[0] != arrays["train_task_loss"].shape[0]:
            raise RuntimeError(f"auxiliary training loss arrays have inconsistent lengths: {path}")
        if arrays["aux_steps"].shape[0] != arrays["aux_loss"].shape[0]:
            raise RuntimeError(f"auxiliary update arrays have inconsistent lengths: {path}")


def _validate_expected_artifacts(job: Job, campaign_dir: Path) -> dict[str, str]:
    expected = _expected_paths(job, campaign_dir)
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"job {job.job_id} is missing expected artifacts: {missing}")
    for path in expected:
        if path.suffix == ".json":
            _validate_result_json(job, path)
        elif path.suffix == ".pt":
            _validate_checkpoint(path)
        elif path.suffix == ".npz":
            _validate_npz(path)
        else:
            raise RuntimeError(f"job {job.job_id} has an unsupported expected artifact: {path}")
    return {
        str(path.relative_to(campaign_dir.resolve())): _sha256_file(path)
        for path in expected
    }


def _completion_receipt_payload(job: Job, campaign_dir: Path, hashes: dict[str, str]) -> dict[str, Any]:
    campaign_id, manifest_digest = _verified_manifest_identity(campaign_dir)
    core: dict[str, Any] = {
        "schema_version": COMPLETION_RECEIPT_VERSION,
        "campaign_id": campaign_id,
        "manifest_identity_sha256": manifest_digest,
        "job_id": job.job_id,
        "job_identity_sha256": hashlib.sha256(_canonical_bytes(job.as_dict())).hexdigest(),
        "artifact_sha256": dict(sorted(hashes.items())),
    }
    return {
        **core,
        "completion_identity_sha256": hashlib.sha256(_canonical_bytes(core)).hexdigest(),
    }


def _write_completion_receipt(job: Job, campaign_dir: Path) -> Path:
    """Validate outputs, hash them, then atomically publish the final marker."""

    hashes = _validate_expected_artifacts(job, campaign_dir)
    receipt = _completion_receipt_payload(job, campaign_dir, hashes)
    path = _completion_receipt_path(job, campaign_dir)
    _atomic_create_json(path, receipt)
    return path


def _verify_completion_receipt(job: Job, campaign_dir: Path) -> None:
    path = _completion_receipt_path(job, campaign_dir)
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid completion receipt {path}: {exc}") from exc
    if not isinstance(receipt, dict):
        raise RuntimeError(f"completion receipt must be a JSON object: {path}")
    claimed = receipt.get("completion_identity_sha256")
    core = {key: value for key, value in receipt.items() if key != "completion_identity_sha256"}
    actual = hashlib.sha256(_canonical_bytes(core)).hexdigest()
    if claimed != actual:
        raise RuntimeError(f"completion receipt identity digest is invalid: {path}")
    current_hashes = _validate_expected_artifacts(job, campaign_dir)
    expected_receipt = _completion_receipt_payload(job, campaign_dir, current_hashes)
    if receipt != expected_receipt:
        raise RuntimeError(f"completion receipt does not match current job/artifacts: {path}")


def _artifact_state(job: Job, campaign_dir: Path) -> tuple[str, list[Path]]:
    expected = _expected_paths(job, campaign_dir)
    receipt = _completion_receipt_path(job, campaign_dir)
    all_paths = [*expected, receipt]
    present = [path for path in all_paths if path.exists()]
    if not present:
        return "missing", expected
    if len(present) != len(all_paths):
        return "partial", present
    try:
        _verify_completion_receipt(job, campaign_dir)
    except Exception:
        return "partial", present
    return "complete", all_paths


def _next_log_path(job: Job, campaign_dir: Path) -> Path:
    directory = campaign_dir / job.log_dir
    directory.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 10000):
        candidate = directory / f"{job.job_id}.attempt{attempt:03d}.log"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"too many log attempts for {job.job_id}")


def _write_status(campaign_dir: Path, status: dict[str, dict]) -> None:
    _atomic_write_json(campaign_dir / "status.json", {"jobs": status})


def _terminate_process_groups(processes: list[subprocess.Popen], grace_seconds: float = 10.0) -> None:
    """Terminate complete worker process groups and reap every direct child."""

    live = [process for process in processes if process.poll() is None]
    for process in live:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + max(0.0, float(grace_seconds))
    while any(process.poll() is None for process in live) and time.monotonic() < deadline:
        time.sleep(0.05)

    for process in live:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in live:
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2.0)


def run_jobs(
    *,
    jobs: list[Job],
    root: Path,
    campaign_dir: Path,
    gpus: list[int],
    max_parallel: int,
    poll_seconds: float,
) -> int:
    unimplemented = [job for job in jobs if not job.implemented]
    if unimplemented:
        details = "\n".join(
            f"  {job.condition}: model={job.model}: {job.implementation_note}" for job in unimplemented[:10]
        )
        raise RuntimeError(
            "selected jobs use models not implemented by the current legacy builder; no job was launched:\n"
            + details
        )

    status: dict[str, dict] = {}
    pending: list[Job] = []
    blocked = 0
    for job in jobs:
        state, paths = _artifact_state(job, campaign_dir)
        if state == "complete":
            status[job.job_id] = {"state": "skipped_complete", "expected": list(job.expected)}
            print(f"[skip complete] {job.job_id}")
        elif state == "partial":
            blocked += 1
            status[job.job_id] = {
                "state": "blocked_partial",
                "present": [str(path.relative_to(campaign_dir)) for path in paths],
            }
            print(f"[block partial] {job.job_id}: {', '.join(map(str, paths))}")
        else:
            pending.append(job)
            status[job.job_id] = {"state": "pending"}
    _write_status(campaign_dir, status)

    worker_gpus = list(gpus)[: max(1, min(int(max_parallel), len(gpus)))]
    available = list(worker_gpus)
    running: dict[int, tuple[subprocess.Popen, Job, Any, Path]] = {}
    failures = 0

    def terminate_all() -> None:
        _terminate_process_groups([process for process, _, _, _ in running.values()])
        for process, job, handle, log_path in running.values():
            handle.close()
            status[job.job_id] = {
                "state": "terminated",
                "exit_code": process.returncode,
                "log": str(log_path.relative_to(campaign_dir)),
            }
        _write_status(campaign_dir, status)

    def handle_sigterm(_signum, _frame) -> None:
        raise KeyboardInterrupt

    old_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, handle_sigterm)
        while pending or running:
            while pending and available:
                gpu = available.pop(0)
                job = pending.pop(0)
                command = _materialize_command(job.command, root, campaign_dir)
                for relative in job.expected:
                    (campaign_dir / relative).parent.mkdir(parents=True, exist_ok=True)
                log_path = _next_log_path(job, campaign_dir)
                log_handle = log_path.open("x", encoding="utf-8")
                log_handle.write(
                    json.dumps(
                        {
                            "job_id": job.job_id,
                            "gpu": gpu,
                            "command": command,
                            "cwd": str(REPO_ROOT),
                            "note": "horizon auxiliary controls are not compute matched",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                log_handle.flush()
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                if "--deterministic-training" in command:
                    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                running[gpu] = (process, job, log_handle, log_path)
                status[job.job_id] = {
                    "state": "running",
                    "gpu": gpu,
                    "pid": process.pid,
                    "log": str(log_path.relative_to(campaign_dir)),
                }
                print(f"[launch gpu{gpu}] {job.job_id}")
                _write_status(campaign_dir, status)

            if not running:
                break
            time.sleep(max(0.05, float(poll_seconds)))
            for gpu in list(running):
                process, job, log_handle, log_path = running[gpu]
                code = process.poll()
                if code is None:
                    continue
                log_handle.close()
                del running[gpu]
                available.append(gpu)
                available.sort(key=worker_gpus.index)
                completion_error = ""
                receipt_path: Path | None = None
                if code == 0:
                    try:
                        receipt_path = _write_completion_receipt(job, campaign_dir)
                    except Exception as exc:
                        completion_error = f"{type(exc).__name__}: {exc}"
                artifact_state, _ = _artifact_state(job, campaign_dir)
                if code == 0 and not completion_error and artifact_state == "complete":
                    final_state = "complete"
                else:
                    failures += 1
                    final_state = "failed"
                status[job.job_id] = {
                    "state": final_state,
                    "exit_code": code,
                    "log": str(log_path.relative_to(campaign_dir)),
                    "artifact_state": artifact_state,
                }
                if receipt_path is not None:
                    status[job.job_id]["completion_receipt"] = str(
                        receipt_path.relative_to(campaign_dir)
                    )
                if completion_error:
                    status[job.job_id]["completion_error"] = completion_error
                print(f"[{final_state} gpu{gpu}] {job.job_id} exit={code}")
                _write_status(campaign_dir, status)
    except KeyboardInterrupt:
        terminate_all()
        raise
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)

    if blocked:
        print(f"blocked partial jobs: {blocked}", file=sys.stderr)
    if failures:
        print(f"failed jobs: {failures}", file=sys.stderr)
    return 1 if blocked or failures else 0


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "campaign.example.json")
    parser.add_argument("--artifact-root", type=Path, default=HERE / "artifacts")
    parser.add_argument("--gpus", default="", help="comma-separated runtime GPU override")
    parser.add_argument("--seeds", default="", help="comma-separated scientific seed override")
    parser.add_argument(
        "--uniform-lambdas",
        default="",
        help="comma-separated fixed-retention sweep; the theta-cap condition remains included",
    )
    parser.add_argument("--enable", action="append", default=[], help="enable a named configured model/control")
    parser.add_argument("--disable", action="append", default=[], help="disable a named configured model/control")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help=(
            "execute only named generated conditions (repeatable) without changing the full "
            "scientific plan or campaign identity"
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_cli()
    config = _load_config(args.config.resolve())
    config = json.loads(json.dumps(config))
    if args.gpus:
        config["gpus"] = [int(value) for value in args.gpus.split(",") if value.strip()]
    if args.seeds:
        config["seeds"] = [int(value) for value in args.seeds.split(",") if value.strip()]
    if args.uniform_lambdas:
        config.setdefault("rp_controls", {})["uniform_lambdas"] = [
            float(value) for value in args.uniform_lambdas.split(",") if value.strip()
        ]
    _apply_enable_overrides(config, set(args.enable), set(args.disable))
    _validate_config(config)
    plan_jobs = build_jobs(config, smoke=bool(args.smoke))
    jobs = plan_jobs
    if args.only:
        requested = set(args.only)
        known = {job.condition for job in plan_jobs}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown --only conditions: {unknown}; generated conditions: {sorted(known)}")
        jobs = [job for job in plan_jobs if job.condition in requested]
    if not jobs:
        raise RuntimeError("campaign contains no enabled jobs")

    # ``--only`` controls execution, not the declared scientific plan.  A
    # subset therefore resumes into the same campaign and is recorded only in
    # the separate runtime receipt.
    campaign_id, manifest = build_manifest(config, plan_jobs, smoke=bool(args.smoke))
    root = args.artifact_root.expanduser().resolve()
    _prepare_artifact_root(root)
    campaign_dir = root / campaign_id
    _write_or_check_manifest(campaign_dir, manifest)
    max_parallel = args.max_parallel or len(config["gpus"])
    if max_parallel <= 0:
        raise ValueError("--max-parallel must be positive")
    runtime_receipt = _write_or_check_runtime_receipt(
        campaign_dir,
        _runtime_receipt(
            config,
            max_parallel=max_parallel,
            dry_run=bool(args.dry_run),
            selected_jobs=jobs,
        ),
    )

    implemented = sum(job.implemented for job in jobs)
    blocked = len(jobs) - implemented
    print(f"campaign={campaign_id} jobs={len(jobs)} implemented={implemented} blocked={blocked}")
    print(f"manifest={campaign_dir / 'manifest.json'}")
    print(f"runtime_receipt={runtime_receipt}")
    if args.dry_run:
        for job in jobs:
            state = "ready" if job.implemented else f"BLOCKED ({job.implementation_note})"
            command = shlex.join(_materialize_command(job.command, root, campaign_dir))
            print(f"[{state}] {job.job_id}\n  {command}")
        return 0

    return run_jobs(
        jobs=jobs,
        root=root,
        campaign_dir=campaign_dir,
        gpus=list(config["gpus"]),
        max_parallel=max_parallel,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
