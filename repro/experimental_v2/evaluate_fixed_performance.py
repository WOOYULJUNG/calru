#!/usr/bin/env python3
"""Re-evaluate campaign checkpoints on one deterministic Exp88 test set.

Training-time metrics are deliberately ignored.  Result JSON files are used
only as strict model-construction metadata; every reported number is computed
again from the checkpoint and fixed ``.npy`` inputs created by this program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LEGACY_DIR = REPO_ROOT / "repro" / "legacy_code"
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

from exp71_pan_block_pulse_hold import (  # noqa: E402
    build_model_variant,
    is_pan_variant,
    normalize_model_variant,
)
from exp88_manifold_attractor_tasks import (  # noqa: E402
    TASKS,
    is_integrate_task,
    model_rank_for_task,
    sequence_from_qv,
    task_geometry,
    task_io_dims,
)


SCHEMA_VERSION = 1
DEFAULT_BASE_SEED = 20260712
DEFAULT_EVAL_POINTS = 128
DEFAULT_BLANK_HORIZON = 1000
SOURCE_FILES = (
    Path(__file__).resolve(),
    LEGACY_DIR / "exp88_manifold_attractor_tasks.py",
    LEGACY_DIR / "exp72_structured_attractor_tasks.py",
    LEGACY_DIR / "exp71_pan_block_pulse_hold.py",
    LEGACY_DIR / "pan_block.py",
    LEGACY_DIR / "plru_regularizers.py",
)


@dataclass(frozen=True)
class CampaignJob:
    job_id: str
    family: str
    condition: str
    task: str
    model: str
    worker_model: str
    seed: int
    result_path: Path
    checkpoint_path: Path


@dataclass(frozen=True)
class Profile:
    name: str
    horizon: int
    schedule_kind: str
    velocity_multiplier: float
    primary: bool = False


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"|")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derived_seed(base_seed: int, *parts: Any) -> int:
    value = "|".join([str(int(base_seed)), *map(str, parts)]).encode("utf-8")
    return int(hashlib.sha256(value).hexdigest()[:8], 16) & 0x7FFFFFFF


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_json(path: Path, payload: Any) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    _atomic_write(path, data)


def _safe_relative(campaign_dir: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute():
        raise ValueError(f"campaign artifact path must be relative: {relative}")
    path = (campaign_dir / raw).resolve()
    try:
        path.relative_to(campaign_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"campaign artifact escapes campaign directory: {relative}") from exc
    return path


def _one_expected(job: Mapping[str, Any], suffix: str) -> str:
    matches = [str(path) for path in job.get("expected", []) if str(path).endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"job {job.get('job_id')} must declare exactly one {suffix} artifact")
    return matches[0]


def _selected(value: Any, allowed: set[Any]) -> bool:
    return not allowed or value in allowed


def discover_jobs(
    campaign_dir: Path,
    *,
    conditions: Sequence[str] = (),
    models: Sequence[str] = (),
    tasks: Sequence[str] = (),
    seeds: Sequence[int] = (),
) -> tuple[dict[str, Any], list[CampaignJob]]:
    """Discover strict result/checkpoint pairs from a launcher manifest."""

    campaign_dir = campaign_dir.expanduser().resolve()
    manifest_path = campaign_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"campaign manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_jobs = manifest.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("campaign manifest has no jobs")
    condition_filter, model_filter = set(conditions), set(models)
    task_filter, seed_filter = set(tasks), {int(seed) for seed in seeds}
    unknown_tasks = task_filter - set(TASKS)
    if unknown_tasks:
        raise ValueError(f"unknown task filters: {sorted(unknown_tasks)}")
    selected: list[CampaignJob] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_jobs:
        required = {"job_id", "family", "condition", "task", "model", "worker_model", "seed", "expected"}
        if not isinstance(raw, dict) or not required.issubset(raw):
            raise ValueError("malformed job in campaign manifest")
        task = str(raw["task"])
        if task not in TASKS:
            raise ValueError(f"manifest job has unknown task: {task}")
        model_names = {str(raw["model"]), str(raw["worker_model"])}
        if not _selected(str(raw["condition"]), condition_filter):
            continue
        if model_filter and model_names.isdisjoint(model_filter):
            continue
        if not _selected(task, task_filter) or not _selected(int(raw["seed"]), seed_filter):
            continue
        if raw.get("implemented") is False:
            raise RuntimeError(f"selected job is marked unimplemented: {raw['job_id']}")
        result_path = _safe_relative(campaign_dir, _one_expected(raw, ".json"))
        checkpoint_path = _safe_relative(campaign_dir, _one_expected(raw, ".pt"))
        pair = (str(result_path), str(checkpoint_path))
        if pair in seen:
            raise ValueError(f"duplicate result/checkpoint pair in manifest: {pair}")
        seen.add(pair)
        selected.append(
            CampaignJob(
                job_id=str(raw["job_id"]),
                family=str(raw["family"]),
                condition=str(raw["condition"]),
                task=task,
                model=str(raw["model"]),
                worker_model=str(raw["worker_model"]),
                seed=int(raw["seed"]),
                result_path=result_path,
                checkpoint_path=checkpoint_path,
            )
        )
    if not selected:
        raise ValueError("filters selected no campaign jobs")
    selected.sort(key=lambda item: (TASKS.index(item.task), item.condition, item.model, item.seed, item.job_id))
    missing = [str(path) for item in selected for path in (item.result_path, item.checkpoint_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("selected campaign is incomplete; missing:\n" + "\n".join(missing))
    return manifest, selected


def _require_int(metadata: Mapping[str, Any], key: str) -> int:
    if key not in metadata or isinstance(metadata[key], bool):
        raise ValueError(f"result metadata is missing integer {key!r}")
    try:
        return int(metadata[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer metadata {key!r}: {metadata[key]!r}") from exc


def _optional_float(metadata: Mapping[str, Any], key: str, default: float) -> float:
    value = metadata.get(key, default)
    if value in (None, ""):
        return float(default)
    return float(value)


def read_model_metadata(job: CampaignJob) -> dict[str, Any]:
    """Read architecture identity only; never copy training-time metrics."""

    metadata = json.loads(job.result_path.read_text(encoding="utf-8"))
    required = {"task", "model", "tag", "seed", "input_dim", "output_dim", "rank_for_model", "d_model", "rec_dim", "layers"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"result metadata lacks required architecture fields {missing}: {job.result_path}")
    if str(metadata["task"]) != job.task or _require_int(metadata, "seed") != job.seed:
        raise ValueError(f"result task/seed does not match manifest job {job.job_id}")
    tag = str(metadata["tag"])
    expected_result_name = f"{job.task}_{tag}_seed{job.seed}.json"
    expected_checkpoint_name = f"exp88_{job.task}_{tag}_seed{job.seed}.pt"
    if job.result_path.name != expected_result_name or job.checkpoint_path.name != expected_checkpoint_name:
        raise ValueError(f"result tag does not match manifest artifact names for {job.job_id}")
    result_variant = normalize_model_variant(str(metadata["model"]))
    worker_variant = normalize_model_variant(job.worker_model)
    if result_variant != worker_variant:
        raise ValueError(
            f"result model {result_variant!r} does not match worker model {worker_variant!r} for {job.job_id}"
        )
    input_dim, output_dim = task_io_dims(job.task)
    expected = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "rank_for_model": model_rank_for_task(job.task),
    }
    for key, value in expected.items():
        if _require_int(metadata, key) != int(value):
            raise ValueError(f"result {key} mismatch for {job.job_id}: {metadata[key]} != {value}")
    for key in ("d_model", "rec_dim", "layers"):
        if _require_int(metadata, key) <= 0:
            raise ValueError(f"result {key} must be positive for {job.job_id}")
    return metadata


def build_and_load_model(
    job: CampaignJob, metadata: Mapping[str, Any], device: torch.device
) -> torch.nn.Module:
    """Construct from result metadata and require an exact state-dict match."""

    variant = normalize_model_variant(str(metadata["model"]))
    input_dim, output_dim = task_io_dims(job.task)
    model = build_model_variant(
        variant=variant,
        input_dim=input_dim,
        output_dim=output_dim,
        rank=model_rank_for_task(job.task),
        d_model=_require_int(metadata, "d_model"),
        rec_dim=_require_int(metadata, "rec_dim"),
        layers=_require_int(metadata, "layers"),
        dropout=_optional_float(metadata, "dropout", 0.0),
        plru_tau=_optional_float(metadata, "plru_tau", 0.001),
        plru_c=_optional_float(metadata, "plru_c", 50.0),
        pan_lambda_min=_optional_float(metadata, "pan_lambda_min", 0.90),
        pan_lambda_max=_optional_float(metadata, "pan_lambda_max", 0.999),
        rank_matched_lambda_high=_optional_float(metadata, "rank_matched_lambda_high", 0.999),
        rank_matched_lambda_low=_optional_float(metadata, "rank_matched_lambda_low", 0.0),
    ).to(device)
    try:
        state_dict = torch.load(job.checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        state_dict = torch.load(job.checkpoint_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint is not a state dict: {job.checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _van_der_corput(index: int, base: int) -> float:
    value, denominator = 0.0, 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def fixed_q0(task: str, count: int, base_seed: int) -> np.ndarray:
    geom = task_geometry(task)
    bases = (2, 3)
    shifts = [derived_seed(base_seed, task, "q0", dim) / float(0x80000000) for dim in range(geom.q_dim)]
    unit = np.empty((int(count), geom.q_dim), dtype=np.float64)
    for row in range(int(count)):
        for dim in range(geom.q_dim):
            unit[row, dim] = (_van_der_corput(row + 1, bases[dim]) + shifts[dim]) % 1.0
    if geom.angle_velocity:
        values = unit * (2.0 * math.pi)
    else:
        values = -0.8 + 1.6 * unit
    return values.astype(np.float32)


def _velocity_base_scale(geometry_name: str, training: Mapping[str, Any]) -> float:
    if geometry_name == "ring":
        return float(training.get("ring_velocity_deg", 3.0)) * math.pi / 180.0
    if geometry_name == "torus":
        return float(training.get("torus_velocity_deg", 2.5)) * math.pi / 180.0
    if geometry_name == "complex_curve":
        return float(training.get("curve_velocity_deg", 2.5)) * math.pi / 180.0
    if geometry_name == "surface":
        return float(training.get("surface_velocity_scale", 0.018))
    raise ValueError(geometry_name)


def _mode_probabilities(name: str) -> tuple[list[str], list[float]]:
    if name == "torus":
        return ["hold", "theta", "psi", "both"], [0.25] * 4
    if name == "surface":
        return ["hold", "u", "v", "both"], [0.25, 0.30, 0.30, 0.15]
    return ["hold", "both"], [0.25, 0.75]


def _bounded_range(low: Any, high: Any, horizon: int, minimum: int = 1) -> tuple[int, int]:
    upper = max(minimum, min(int(high), max(minimum, int(horizon))))
    lower = max(minimum, min(int(low), upper))
    return lower, upper


def deterministic_velocity(
    task: str,
    q0: np.ndarray,
    profile: Profile,
    training: Mapping[str, Any],
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the applied velocity and an integer segment schedule."""

    geom = task_geometry(task)
    horizon, count = int(profile.horizon), int(q0.shape[0])
    desired = np.zeros((horizon, count, geom.q_dim), dtype=np.float32)
    if is_integrate_task(task):
        if profile.schedule_kind == "temporal":
            hold_range = _bounded_range(training.get("ood_hold_min", 30), training.get("ood_hold_max", 120), horizon)
            final_range = _bounded_range(
                training.get("ood_final_hold_min", 100), training.get("ood_final_hold_max", 250), horizon
            )
            move_range = _bounded_range(training.get("move_min", 3), training.get("move_max", 10), horizon)
        elif profile.schedule_kind == "sparse":
            hold_range = _bounded_range(training.get("ood_hold_min", 30), training.get("ood_hold_max", 120), horizon)
            final_range = _bounded_range(training.get("final_hold_min", 20), training.get("final_hold_max", 80), horizon)
            move_range = (1, max(1, min(int(training.get("move_min", 3)), horizon)))
        else:
            hold_range = _bounded_range(training.get("hold_min", 5), training.get("hold_max", 20), horizon)
            final_range = _bounded_range(training.get("final_hold_min", 20), training.get("final_hold_max", 80), horizon)
            move_range = _bounded_range(training.get("move_min", 3), training.get("move_max", 10), horizon)
        # ID and velocity-scale profiles intentionally share their normalized
        # schedule and step draws; only the multiplier changes.
        schedule_identity = "id" if profile.schedule_kind in {"id", "velocity"} else profile.name
        rng = np.random.default_rng(derived_seed(base_seed, task, "velocity", schedule_identity, horizon))
        modes, probabilities = _mode_probabilities(geom.name)
        scale = _velocity_base_scale(geom.name, training) * float(profile.velocity_multiplier)
        for sample in range(count):
            final_hold = int(rng.integers(final_range[0], final_range[1] + 1))
            active = max(0, horizon - final_hold)
            sign = 1.0
            t = 0
            while t < active:
                mode = str(rng.choice(modes, p=probabilities))
                if mode == "hold":
                    length = int(rng.integers(hold_range[0], hold_range[1] + 1))
                    step = np.zeros(geom.q_dim, dtype=np.float32)
                else:
                    length = int(rng.integers(move_range[0], move_range[1] + 1))
                    step = rng.uniform(-scale, scale, size=geom.q_dim).astype(np.float32)
                    if profile.schedule_kind == "alternating":
                        step = np.abs(step) * sign
                        sign *= -1.0
                    if geom.name == "torus":
                        if mode == "theta":
                            step[1] = 0.0
                        elif mode == "psi":
                            step[0] = 0.0
                    elif geom.name == "surface":
                        if mode == "u":
                            step[1] = 0.0
                        elif mode == "v":
                            step[0] = 0.0
                end = min(active, t + length)
                desired[t:end, sample] = step
                t = end
    q_tensor = torch.from_numpy(np.ascontiguousarray(q0))
    v_tensor = torch.from_numpy(desired)
    _, _, _, auxiliary = sequence_from_qv(geom, q_tensor, v_tensor)
    applied = auxiliary["v"].cpu().numpy().astype(np.float32, copy=True)
    nonzero = np.abs(applied) > 1e-12
    schedule = np.zeros((horizon, count), dtype=np.int8)
    for dim in range(geom.q_dim):
        schedule |= (nonzero[..., dim].astype(np.int8) << dim)
    return applied, schedule


def build_profiles(
    training: Mapping[str, Any],
    *,
    id_horizon: int | None,
    temporal_horizons: Sequence[int] | None,
    velocity_scales: Sequence[float] | None,
    smoke: bool,
) -> list[Profile]:
    if smoke:
        id_value, temporal_values, scale_values = 8, [12], [1.5]
    else:
        id_value = int(id_horizon if id_horizon is not None else training.get("id_horizon", 260))
        temporal_values = list(
            temporal_horizons if temporal_horizons is not None else training.get("temporal_horizons", [500, 1000, 2000])
        )
        scale_values = list(velocity_scales if velocity_scales is not None else training.get("velocity_scales", [1.0, 1.5, 2.0]))
    if id_value <= 0 or any(int(value) <= 0 for value in temporal_values):
        raise ValueError("evaluation horizons must be positive")
    profiles = [Profile("id", id_value, "id", 1.0, primary=True)]
    for horizon in sorted(set(map(int, temporal_values)) - {id_value}):
        profiles.append(Profile(f"temporal_H{horizon}", horizon, "temporal", 1.0))
    for scale in sorted(set(map(float, scale_values))):
        if scale <= 0:
            raise ValueError("velocity scales must be positive")
        if abs(scale - 1.0) < 1e-12:
            continue
        slug = str(scale).replace(".", "p")
        profiles.append(Profile(f"velocity_{slug}x", id_value, "velocity", scale))
    if not smoke:
        profiles.extend(
            [
                Profile("velocity_sparse_1p5x", id_value, "sparse", 1.5),
                Profile("velocity_alternating_1p5x", id_value, "alternating", 1.5),
            ]
        )
    return profiles


def profiles_for_task(task: str, profiles: Sequence[Profile]) -> list[Profile]:
    if is_integrate_task(task):
        return list(profiles)
    return [profile for profile in profiles if profile.schedule_kind in {"id", "temporal"}]


def _save_npy(path: Path, array: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)


def create_assets(
    output_dir: Path,
    profiles: Sequence[Profile],
    training: Mapping[str, Any],
    eval_points: int,
    base_seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    asset_dir = output_dir / "assets"
    asset_dir.mkdir()
    arrays: dict[str, dict[str, np.ndarray]] = {}
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "base_seed": int(base_seed),
        "eval_points": int(eval_points),
        "schedule_encoding": "bit d is one when applied velocity coordinate d is nonzero",
        "tasks": {},
    }
    for task in TASKS:
        task_dir = asset_dir / task
        task_dir.mkdir()
        q0 = fixed_q0(task, int(eval_points), int(base_seed))
        _save_npy(task_dir / "q0.npy", q0)
        task_arrays: dict[str, np.ndarray] = {"q0": q0}
        task_manifest: dict[str, Any] = {
            "q0": {
                "path": f"assets/{task}/q0.npy",
                "sha256": sha256_file(task_dir / "q0.npy"),
                "shape": list(q0.shape),
                "dtype": str(q0.dtype),
            },
            "profiles": {},
        }
        for profile in profiles_for_task(task, profiles):
            velocity, schedule = deterministic_velocity(task, q0, profile, training, int(base_seed))
            velocity_name = f"velocity__{profile.name}.npy"
            schedule_name = f"schedule__{profile.name}.npy"
            _save_npy(task_dir / velocity_name, velocity)
            _save_npy(task_dir / schedule_name, schedule)
            task_arrays[f"velocity::{profile.name}"] = velocity
            task_arrays[f"schedule::{profile.name}"] = schedule
            task_manifest["profiles"][profile.name] = {
                "horizon": profile.horizon,
                "schedule_kind": profile.schedule_kind,
                "velocity_multiplier": profile.velocity_multiplier,
                "primary": profile.primary,
                "velocity": {
                    "path": f"assets/{task}/{velocity_name}",
                    "sha256": sha256_file(task_dir / velocity_name),
                    "shape": list(velocity.shape),
                    "dtype": str(velocity.dtype),
                },
                "schedule": {
                    "path": f"assets/{task}/{schedule_name}",
                    "sha256": sha256_file(task_dir / schedule_name),
                    "shape": list(schedule.shape),
                    "dtype": str(schedule.dtype),
                },
            }
        arrays[task] = task_arrays
        manifest["tasks"][task] = task_manifest
    manifest["definition_sha256"] = canonical_hash(manifest)
    atomic_json(output_dir / "asset_manifest.json", manifest)
    return arrays, manifest


def component_rmse(error: torch.Tensor) -> float:
    return float(torch.sqrt(error.square().mean()).item())


def vector_rmse(error: torch.Tensor) -> float:
    return float(torch.sqrt(error.square().sum(dim=-1).mean()).item())


@torch.inference_mode()
def evaluate_profile(
    model: torch.nn.Module,
    task: str,
    q0: np.ndarray,
    velocity: np.ndarray,
    schedule: np.ndarray,
    blank_horizon: int,
    device: torch.device,
) -> dict[str, float | int]:
    geom = task_geometry(task)
    q_tensor = torch.from_numpy(np.ascontiguousarray(q0)).to(device=device, dtype=torch.float32)
    v_tensor = torch.from_numpy(np.ascontiguousarray(velocity)).to(device=device, dtype=torch.float32)
    x, y, target, auxiliary = sequence_from_qv(geom, q_tensor, v_tensor)
    if not torch.allclose(auxiliary["v"], v_tensor, atol=2e-6, rtol=1e-6):
        raise AssertionError(f"saved velocity is not self-consistent for {task}")
    state = model.init_state(q_tensor.shape[0], device)
    sequence_sq = torch.zeros((), device=device, dtype=torch.float64)
    sequence_count = 0
    vector_sq = torch.zeros((), device=device, dtype=torch.float64)
    vector_count = 0
    moving_errors: list[torch.Tensor] = []
    hold_errors: list[torch.Tensor] = []
    schedule_tensor = torch.from_numpy(np.ascontiguousarray(schedule)).to(device=device)
    endpoint = None
    for step, x_t in enumerate(x):
        state = model.step(x_t, state)
        prediction = model.decode(state)
        error = prediction - y[step]
        sequence_sq += error.double().square().sum()
        sequence_count += error.numel()
        vector_sq += error.double().square().sum()
        vector_count += error.shape[0]
        if step > 0:
            magnitude = torch.linalg.vector_norm(error, dim=-1)
            moving = schedule_tensor[step - 1] != 0
            if bool(moving.any()):
                moving_errors.append(magnitude[moving])
            if bool((~moving).any()):
                hold_errors.append(magnitude[~moving])
        endpoint = prediction
    assert endpoint is not None
    endpoint_error = endpoint - target
    h0 = endpoint.detach().clone()
    blank = torch.zeros(q_tensor.shape[0], model.input_dim, device=device, dtype=q_tensor.dtype)
    for _ in range(int(blank_horizon)):
        state = model.step(blank, state)
    post = model.decode(state)
    post_error = post - target
    paired = post - h0
    moving_concat = torch.cat(moving_errors) if moving_errors else torch.empty(0, device=device)
    hold_concat = torch.cat(hold_errors) if hold_errors else torch.empty(0, device=device)
    return {
        "endpoint_component_rmse": component_rmse(endpoint_error),
        "endpoint_vec_rmse": vector_rmse(endpoint_error),
        "post_blank_component_rmse": component_rmse(post_error),
        "post_blank_vec_rmse": vector_rmse(post_error),
        "paired_H0_to_post_component_drift_rmse": component_rmse(paired),
        "paired_H0_to_post_vec_drift_rmse": vector_rmse(paired),
        "sequence_component_rmse": float(torch.sqrt(sequence_sq / sequence_count).item()),
        "sequence_vec_rmse": float(torch.sqrt(vector_sq / vector_count).item()),
        # These two reproduce the legacy implementation: mean per-step
        # Euclidean error, despite its historical ``vec_rmse`` label.
        "moving_step_vec_error_mean": float(moving_concat.mean().item()) if moving_concat.numel() else 0.0,
        "hold_step_vec_error_mean": float(hold_concat.mean().item()) if hold_concat.numel() else 0.0,
        "moving_step_count": int(moving_concat.numel()),
        "hold_step_count": int(hold_concat.numel()),
    }


def _retention_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    if not hasattr(model, "pan_recs_with_slices"):
        return []
    values = []
    for recurrent, _ in model.pan_recs_with_slices():
        theta = getattr(recurrent, "theta", None)
        if theta is not None:
            values.append(theta)
    return values


def _tensor_mapping_hash(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"|")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"|")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"|")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@torch.no_grad()
def apply_retention_permutation(model: torch.nn.Module, seed: int) -> dict[str, Any]:
    """Permute final PAN retention coordinates without changing their multiset."""

    parameters = _retention_parameters(model)
    if not parameters:
        raise ValueError("retention permutation requires a PAN model with theta coordinates")
    retention_ids = {id(parameter) for parameter in parameters}
    retention_names = {name for name, parameter in model.named_parameters() if id(parameter) in retention_ids}
    if len(retention_names) != len(parameters):
        raise AssertionError("could not identify every retention parameter in the state dict")
    before_non_retention = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name not in retention_names
    }
    non_retention_hash = _tensor_mapping_hash(before_non_retention)
    before_theta = torch.cat([parameter.detach().flatten().cpu() for parameter in parameters])
    before_lambda = torch.sqrt(torch.sigmoid(before_theta))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(before_theta.numel(), generator=generator)
    permuted_theta = before_theta[permutation]
    offset = 0
    for parameter in parameters:
        width = parameter.numel()
        parameter.copy_(permuted_theta[offset : offset + width].reshape(parameter.shape).to(parameter))
        offset += width
    after_theta = torch.cat([parameter.detach().flatten().cpu() for parameter in parameters])
    after_lambda = torch.sqrt(torch.sigmoid(after_theta))
    if not torch.equal(after_theta, before_theta[permutation]):
        raise AssertionError("theta permutation assignment changed values")
    if not torch.equal(after_lambda, before_lambda[permutation]):
        raise AssertionError("lambda multiset was not exactly preserved")
    after_non_retention = {
        name: value.detach()
        for name, value in model.state_dict().items()
        if name not in retention_names
    }
    after_non_retention_hash = _tensor_mapping_hash(after_non_retention)
    if after_non_retention_hash != non_retention_hash:
        raise AssertionError("a non-retention weight changed during evaluation-only permutation")
    return {
        "permutation_seed": int(seed),
        "permutation_sha256": sha256_array(permutation.numpy().astype(np.int64, copy=False)),
        "retention_coordinate_count": int(before_theta.numel()),
        "retention_order_sha256_before": sha256_array(before_lambda.numpy()),
        "retention_order_sha256_after": sha256_array(after_lambda.numpy()),
        "retention_multiset_sha256": sha256_array(torch.sort(before_lambda).values.numpy()),
        "retention_multiset_preserved": True,
        "non_retention_state_sha256_before": non_retention_hash,
        "non_retention_state_sha256_after": after_non_retention_hash,
        "non_retention_state_preserved": True,
    }


def _asset_hashes(asset_manifest: Mapping[str, Any], task: str, profile: str) -> dict[str, str]:
    task_row = asset_manifest["tasks"][task]
    profile_row = task_row["profiles"][profile]
    return {
        "q0_asset_sha256": task_row["q0"]["sha256"],
        "velocity_asset_sha256": profile_row["velocity"]["sha256"],
        "schedule_asset_sha256": profile_row["schedule"]["sha256"],
    }


def evaluate_jobs(
    jobs: Sequence[CampaignJob],
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    asset_manifest: Mapping[str, Any],
    profiles: Sequence[Profile],
    blank_horizon: int,
    device: torch.device,
    retention_permutation: bool,
    permutation_seed_base: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        metadata = read_model_metadata(job)
        result_variant = normalize_model_variant(str(metadata["model"]))
        if retention_permutation and not is_pan_variant(result_variant):
            raise ValueError(
                f"retention permutation was requested for non-PAN job {job.job_id} ({result_variant})"
            )
        model = build_and_load_model(job, metadata, device)
        conditions: list[tuple[str, dict[str, Any]]] = [("checkpoint_original", {})]
        permutation_payload: dict[str, Any] | None = None
        for evaluation_condition, condition_metadata in conditions:
            for profile in profiles_for_task(job.task, profiles):
                values = evaluate_profile(
                    model,
                    job.task,
                    arrays[job.task]["q0"],
                    arrays[job.task][f"velocity::{profile.name}"],
                    arrays[job.task][f"schedule::{profile.name}"],
                    blank_horizon,
                    device,
                )
                row: dict[str, Any] = {
                    "job_id": job.job_id,
                    "family": job.family,
                    "condition": job.condition,
                    "evaluation_condition": evaluation_condition,
                    "paper_model": job.model,
                    "worker_model": job.worker_model,
                    "result_model": result_variant,
                    "task": job.task,
                    "seed": job.seed,
                    "profile": profile.name,
                    "profile_horizon": profile.horizon,
                    "profile_schedule_kind": profile.schedule_kind,
                    "profile_velocity_multiplier": profile.velocity_multiplier,
                    "is_primary_profile": profile.primary,
                    "blank_horizon": int(blank_horizon),
                    "ambient_dimension": task_geometry(job.task).y_dim,
                    "declared_trainable_params": metadata.get("params", ""),
                    "result_sha256": sha256_file(job.result_path),
                    "checkpoint_sha256": sha256_file(job.checkpoint_path),
                    **_asset_hashes(asset_manifest, job.task, profile.name),
                    **condition_metadata,
                    **values,
                }
                if int(blank_horizon) == 1000:
                    row["postH1000_ambient_component_rmse"] = values["post_blank_component_rmse"]
                    row["postH1000_vec_rmse"] = values["post_blank_vec_rmse"]
                    row["paired_H0_to_H1000_component_drift_rmse"] = values[
                        "paired_H0_to_post_component_drift_rmse"
                    ]
                    row["paired_H0_to_H1000_vec_drift_rmse"] = values[
                        "paired_H0_to_post_vec_drift_rmse"
                    ]
                rows.append(row)
        if retention_permutation:
            permutation_seed = derived_seed(
                permutation_seed_base, job.condition, job.task, job.seed, job.job_id, "retention-permutation-v1"
            )
            permutation_payload = apply_retention_permutation(model, permutation_seed)
            for profile in profiles_for_task(job.task, profiles):
                values = evaluate_profile(
                    model,
                    job.task,
                    arrays[job.task]["q0"],
                    arrays[job.task][f"velocity::{profile.name}"],
                    arrays[job.task][f"schedule::{profile.name}"],
                    blank_horizon,
                    device,
                )
                row = {
                    "job_id": job.job_id,
                    "family": job.family,
                    "condition": job.condition,
                    "evaluation_condition": "retention_permuted",
                    "paper_model": job.model,
                    "worker_model": job.worker_model,
                    "result_model": result_variant,
                    "task": job.task,
                    "seed": job.seed,
                    "profile": profile.name,
                    "profile_horizon": profile.horizon,
                    "profile_schedule_kind": profile.schedule_kind,
                    "profile_velocity_multiplier": profile.velocity_multiplier,
                    "is_primary_profile": profile.primary,
                    "blank_horizon": int(blank_horizon),
                    "ambient_dimension": task_geometry(job.task).y_dim,
                    "declared_trainable_params": metadata.get("params", ""),
                    "result_sha256": sha256_file(job.result_path),
                    "checkpoint_sha256": sha256_file(job.checkpoint_path),
                    **_asset_hashes(asset_manifest, job.task, profile.name),
                    **permutation_payload,
                    **values,
                }
                if int(blank_horizon) == 1000:
                    row["postH1000_ambient_component_rmse"] = values["post_blank_component_rmse"]
                    row["postH1000_vec_rmse"] = values["post_blank_vec_rmse"]
                    row["paired_H0_to_H1000_component_drift_rmse"] = values[
                        "paired_H0_to_post_component_drift_rmse"
                    ]
                    row["paired_H0_to_H1000_vec_drift_rmse"] = values[
                        "paired_H0_to_post_vec_drift_rmse"
                    ]
                rows.append(row)
    return rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


METRIC_FIELDS = (
    "endpoint_component_rmse",
    "endpoint_vec_rmse",
    "post_blank_component_rmse",
    "post_blank_vec_rmse",
    "paired_H0_to_post_component_drift_rmse",
    "paired_H0_to_post_vec_drift_rmse",
    "sequence_component_rmse",
    "sequence_vec_rmse",
    "moving_step_vec_error_mean",
    "hold_step_vec_error_mean",
)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "family",
        "condition",
        "evaluation_condition",
        "paper_model",
        "worker_model",
        "result_model",
        "task",
        "profile",
        "profile_horizon",
        "profile_schedule_kind",
        "profile_velocity_multiplier",
        "is_primary_profile",
        "blank_horizon",
    )
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    output = []
    for identity in sorted(groups, key=lambda item: tuple(map(str, item))):
        group = groups[identity]
        row = dict(zip(keys, identity))
        row["n_seeds"] = len(group)
        row["seeds"] = ",".join(map(str, sorted(int(item["seed"]) for item in group)))
        for metric in METRIC_FIELDS:
            values = np.asarray([float(item[metric]) for item in group], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        if int(row["blank_horizon"]) == 1000:
            row["postH1000_ambient_component_rmse_mean"] = row["post_blank_component_rmse_mean"]
            row["postH1000_ambient_component_rmse_std"] = row["post_blank_component_rmse_std"]
        output.append(row)
    return output


def macro_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary = [row for row in rows if bool(row["is_primary_profile"])]
    keys = ("family", "condition", "evaluation_condition", "paper_model", "worker_model")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in primary:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for identity in sorted(groups, key=lambda item: tuple(map(str, item))):
        group = groups[identity]
        values = np.asarray([float(row["post_blank_component_rmse"]) for row in group], dtype=np.float64)
        output.append(
            {
                **dict(zip(keys, identity)),
                "n_runs": len(group),
                "n_tasks": len({str(row["task"]) for row in group}),
                "tasks": ",".join(sorted({str(row["task"]) for row in group}, key=TASKS.index)),
                "primary_post_blank_ambient_component_rmse_macro_mean": float(values.mean()),
                "primary_post_blank_ambient_component_rmse_macro_std_across_runs": (
                    float(values.std(ddof=1)) if values.size > 1 else 0.0
                ),
            }
        )
    return output


def capture_hashes(paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted({Path(path).resolve() for path in paths}, key=str):
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            key = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            key = str(path)
        result[key] = sha256_file(path)
    return result


def _training_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    try:
        training = manifest["scientific_config"]["config"]["training"]
    except (KeyError, TypeError) as exc:
        raise ValueError("campaign manifest lacks scientific_config.config.training") from exc
    if not isinstance(training, dict):
        raise ValueError("campaign training definition is not an object")
    return dict(training)


def _assert_fresh_output(output_dir: Path, campaign_dir: Path) -> None:
    output = output_dir.expanduser().resolve()
    campaign = campaign_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if output == campaign or campaign in output.parents:
        raise ValueError("output directory must not contain the campaign directory")
    if output in campaign.parents:
        raise ValueError("output directory must not be an ancestor of the campaign directory")


def run_evaluation(
    *,
    campaign_dir: Path,
    output_dir: Path,
    conditions: Sequence[str] = (),
    models: Sequence[str] = (),
    tasks: Sequence[str] = (),
    seeds: Sequence[int] = (),
    eval_points: int = DEFAULT_EVAL_POINTS,
    base_seed: int = DEFAULT_BASE_SEED,
    blank_horizon: int = DEFAULT_BLANK_HORIZON,
    id_horizon: int | None = None,
    temporal_horizons: Sequence[int] | None = None,
    velocity_scales: Sequence[float] | None = None,
    device_name: str = "auto",
    smoke: bool = False,
    dry_run: bool = False,
    retention_permutation: bool = False,
    permutation_seed_base: int = 20260713,
) -> dict[str, Any]:
    campaign_dir = campaign_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _assert_fresh_output(output_dir, campaign_dir)
    manifest, jobs = discover_jobs(
        campaign_dir, conditions=conditions, models=models, tasks=tasks, seeds=seeds
    )
    training = _training_from_manifest(manifest)
    if smoke:
        training.update(
            {
                "hold_min": 1,
                "hold_max": 2,
                "move_min": 1,
                "move_max": 2,
                "final_hold_min": 1,
                "final_hold_max": 2,
                "ood_hold_min": 2,
                "ood_hold_max": 3,
                "ood_final_hold_min": 2,
                "ood_final_hold_max": 3,
            }
        )
    profiles = build_profiles(
        training,
        id_horizon=id_horizon,
        temporal_horizons=temporal_horizons,
        velocity_scales=velocity_scales,
        smoke=smoke,
    )
    eval_points = min(int(eval_points), 8) if smoke else int(eval_points)
    blank_horizon = min(int(blank_horizon), 8) if smoke else int(blank_horizon)
    if eval_points <= 0 or blank_horizon < 0:
        raise ValueError("eval_points must be positive and blank_horizon non-negative")
    metadata_by_job = {job.job_id: read_model_metadata(job) for job in jobs}
    if retention_permutation:
        non_pan = [
            job.job_id
            for job in jobs
            if not is_pan_variant(normalize_model_variant(str(metadata_by_job[job.job_id]["model"])))
        ]
        if non_pan:
            raise ValueError("retention permutation requires every selected job to be PAN: " + ", ".join(non_pan))
    source_start = capture_hashes(SOURCE_FILES)
    input_paths = [campaign_dir / "manifest.json"] + [
        path for job in jobs for path in (job.result_path, job.checkpoint_path)
    ]
    input_start = capture_hashes(input_paths)
    plan = {
        "campaign_id": manifest.get("campaign_id", campaign_dir.name),
        "jobs": len(jobs),
        "task_coverage": [task for task in TASKS if any(job.task == task for job in jobs)],
        "assets_cover_all_tasks": list(TASKS),
        "profiles": [profile.__dict__ for profile in profiles],
        "eval_points": eval_points,
        "blank_horizon": blank_horizon,
        "device_requested": device_name,
        "retention_permutation": bool(retention_permutation),
        "output_dir": str(output_dir),
    }
    if dry_run:
        return plan
    if device_name == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "INCOMPLETE").write_text("evaluation in progress\n", encoding="utf-8")
    arrays, asset_manifest = create_assets(
        output_dir, profiles, training, eval_points=eval_points, base_seed=base_seed
    )
    rows = evaluate_jobs(
        jobs,
        arrays,
        asset_manifest,
        profiles,
        blank_horizon,
        device,
        retention_permutation,
        permutation_seed_base,
    )
    summaries = summarize_rows(rows)
    macros = macro_summary(rows)
    primary_rows = [row for row in rows if bool(row["is_primary_profile"])]
    write_csv(output_dir / "fixed_performance.csv", rows)
    write_csv(output_dir / "fixed_performance_summary.csv", summaries)
    write_csv(output_dir / "fixed_performance_primary.csv", primary_rows)
    write_csv(output_dir / "fixed_performance_macro.csv", macros)

    source_end = capture_hashes(SOURCE_FILES)
    input_end = capture_hashes(input_paths)
    if source_end != source_start:
        raise RuntimeError("source files changed during evaluation; leaving INCOMPLETE")
    if input_end != input_start:
        raise RuntimeError("campaign inputs changed during evaluation; leaving INCOMPLETE")
    produced = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "INCOMPLETE"
    )
    output_hashes = {path.relative_to(output_dir).as_posix(): sha256_file(path) for path in produced}
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": manifest.get("campaign_id", campaign_dir.name),
        "campaign_manifest_sha256": sha256_file(campaign_dir / "manifest.json"),
        "definition": {
            "base_seed": int(base_seed),
            "eval_points": int(eval_points),
            "blank_horizon": int(blank_horizon),
            "primary_profile": "id",
            "primary_metric": (
                "postH1000_ambient_component_rmse" if blank_horizon == 1000 else "post_blank_component_rmse"
            ),
            "temporal_and_velocity_profiles": [profile.__dict__ for profile in profiles],
            "retention_permutation": bool(retention_permutation),
            "permutation_seed_base": int(permutation_seed_base) if retention_permutation else None,
        },
        "selection": {
            "conditions": list(conditions),
            "models": list(models),
            "tasks": list(tasks),
            "seeds": list(map(int, seeds)),
            "selected_job_ids": [job.job_id for job in jobs],
            "task_coverage": plan["task_coverage"],
        },
        "asset_definition_sha256": asset_manifest["definition_sha256"],
        "source_hashes_start_and_end": source_start,
        "input_hashes_start_and_end": input_start,
        "output_hashes_before_run_manifest": output_hashes,
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "row_count": len(rows),
        "summary_row_count": len(summaries),
        "training_metrics_reused": False,
    }
    atomic_json(output_dir / "run_manifest.json", run_manifest)
    all_hashed = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "INCOMPLETE")
    checksums = "".join(
        f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n" for path in all_hashed
    ).encode("utf-8")
    _atomic_write(output_dir / "SHA256SUMS", checksums)
    # Recheck after writing the atomic provenance manifests.  COMPLETE is the
    # final filesystem mutation and therefore certifies an unchanged run.
    if capture_hashes(SOURCE_FILES) != source_start or capture_hashes(input_paths) != input_start:
        raise RuntimeError("source or input changed during finalization; leaving INCOMPLETE")
    (output_dir / "INCOMPLETE").unlink()
    complete_payload = {
        "status": "complete",
        "sha256sums_sha256": sha256_file(output_dir / "SHA256SUMS"),
        "run_manifest_sha256": sha256_file(output_dir / "run_manifest.json"),
    }
    _atomic_write(
        output_dir / "COMPLETE",
        (json.dumps(complete_payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    return {**plan, "rows": len(rows), "status": "complete", "device": str(device)}


def _csv_arg(value: str, cast: Any = str) -> list[Any]:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", default="", help="comma-separated manifest conditions")
    parser.add_argument("--models", default="", help="comma-separated paper or worker model names")
    parser.add_argument("--tasks", default="", help="comma-separated Exp88 task names")
    parser.add_argument("--seeds", default="", help="comma-separated training seeds")
    parser.add_argument("--eval-points", type=int, default=DEFAULT_EVAL_POINTS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--blank-horizon", type=int, default=DEFAULT_BLANK_HORIZON)
    parser.add_argument("--id-horizon", type=int)
    parser.add_argument("--temporal-horizons", default="", help="optional comma-separated override")
    parser.add_argument("--velocity-scales", default="", help="optional comma-separated override")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retention-permutation", action="store_true")
    parser.add_argument("--permutation-seed-base", type=int, default=20260713)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_evaluation(
        campaign_dir=args.campaign_dir,
        output_dir=args.output_dir,
        conditions=_csv_arg(args.conditions),
        models=_csv_arg(args.models),
        tasks=_csv_arg(args.tasks),
        seeds=_csv_arg(args.seeds, int),
        eval_points=args.eval_points,
        base_seed=args.base_seed,
        blank_horizon=args.blank_horizon,
        id_horizon=args.id_horizon,
        temporal_horizons=_csv_arg(args.temporal_horizons, int) or None,
        velocity_scales=_csv_arg(args.velocity_scales, float) or None,
        device_name=args.device,
        smoke=args.smoke,
        dry_run=args.dry_run,
        retention_permutation=args.retention_permutation,
        permutation_seed_base=args.permutation_seed_base,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
