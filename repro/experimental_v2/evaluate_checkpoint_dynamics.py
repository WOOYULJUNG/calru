#!/usr/bin/env python3
"""Deterministic, checkpoint-only Exp88 state-dynamics evaluation.

This module deliberately lives outside the legacy experiment tree.  It reads
legacy result JSON/checkpoints but only writes to a fresh caller-selected output
directory.  In full-block models the flat state includes a recurrent carrier
followed by a transient stream; interventions and geometric distances here use
the recurrent carrier only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LEGACY_DIR = REPO_ROOT / "repro" / "legacy_code"
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

from exp71_pan_block_pulse_hold import build_model_variant, normalize_model_variant  # noqa: E402
from exp88_manifold_attractor_tasks import (  # noqa: E402
    TASKS,
    is_integrate_task,
    model_rank_for_task,
    sequence_from_qv,
    task_geometry,
    task_io_dims,
)


SCHEMA_VERSION = 1
# The checked-out paper repository normally sits directly beneath the legacy
# artifact root.  Callers with a different layout can always pass
# ``--artifact-root`` explicitly.
DEFAULT_ARTIFACT_ROOT = REPO_ROOT.parent
DEFAULT_SPECS = HERE / "dynamics_specs.json"
DEFAULT_RECOVERY_HORIZONS = (0, 20, 100, 500, 1000)
DEFAULT_JACOBIAN_HORIZONS = (1, 20, 100, 500)
DEFAULT_RADII = (0.1, 0.3, 1.0)


@dataclass(frozen=True)
class CheckpointSpec:
    paper_model: str
    task: str
    tag: str
    result_dir: str
    checkpoint_dir: str

    def result_path(self, root: Path, seed: int) -> Path:
        return root / self.result_dir / f"{self.task}_{self.tag}_seed{seed}.json"

    def checkpoint_path(self, root: Path, seed: int) -> Path:
        return root / self.checkpoint_dir / f"exp88_{self.task}_{self.tag}_seed{seed}.pt"


@dataclass(frozen=True)
class PlanItem:
    spec: CheckpointSpec
    seed: int
    result_path: Path
    checkpoint_path: Path


@dataclass(frozen=True)
class StateLayout:
    """Validated contiguous carrier/stream layout for a legacy StepModel."""

    state_dim: int
    carrier_dim: int
    stream_dim: int

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> "StateLayout":
        state_dim = int(model.state_size)
        if hasattr(model, "recurrent_state_size"):
            carrier_dim = int(model.recurrent_state_size)
            readout_slice = getattr(model, "readout_slice", None)
            if not isinstance(readout_slice, slice):
                raise ValueError("full-block model has no readout_slice")
            if readout_slice.start != carrier_dim or readout_slice.stop != state_dim:
                raise ValueError(
                    "unsupported non-contiguous state layout: "
                    f"carrier={carrier_dim}, readout={readout_slice}, state={state_dim}"
                )
        else:
            carrier_dim = state_dim
        if not 0 < carrier_dim <= state_dim:
            raise ValueError(f"invalid carrier dimension {carrier_dim}/{state_dim}")
        return cls(state_dim=state_dim, carrier_dim=carrier_dim, stream_dim=state_dim - carrier_dim)

    def carrier(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"expected state dim {self.state_dim}, got {state.shape[-1]}")
        return state[..., : self.carrier_dim]

    def replace_carrier(self, template: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        if carrier.shape[:-1] != template.shape[:-1] or carrier.shape[-1] != self.carrier_dim:
            raise ValueError("carrier/template shape mismatch")
        if self.stream_dim == 0:
            return carrier
        return torch.cat([carrier, template[..., self.carrier_dim :]], dim=-1)

    def embed_direction(self, carrier_direction: torch.Tensor) -> torch.Tensor:
        if carrier_direction.shape[-1] != self.carrier_dim:
            raise ValueError("direction is not in carrier coordinates")
        if self.stream_dim == 0:
            return carrier_direction
        zeros = torch.zeros(
            *carrier_direction.shape[:-1],
            self.stream_dim,
            device=carrier_direction.device,
            dtype=carrier_direction.dtype,
        )
        return torch.cat([carrier_direction, zeros], dim=-1)


@dataclass(frozen=True)
class FileSnapshot:
    """Immutable digest captured before any analysis output is created."""

    path: Path
    kind: str
    sha256: str
    size: int


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_file(path: Path, kind: str) -> FileSnapshot:
    """Hash one input and reject a file that changes while it is read."""

    resolved = path.resolve(strict=True)
    before = resolved.stat()
    digest = sha256_file(resolved)
    after = resolved.stat()
    signature_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    signature_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if signature_before != signature_after:
        raise RuntimeError(f"input changed while its initial hash was captured: {resolved}")
    return FileSnapshot(path=resolved, kind=str(kind), sha256=digest, size=int(after.st_size))


def capture_source_snapshot(plan: Sequence[PlanItem], specs_path: Path) -> List[FileSnapshot]:
    """Freeze every scientific/code input before evaluation starts."""

    entries: List[Tuple[str, Path]] = [("spec", specs_path)]
    entries.extend(
        ("code", path)
        for path in (
            Path(__file__),
            LEGACY_DIR / "exp88_manifold_attractor_tasks.py",
            LEGACY_DIR / "exp72_structured_attractor_tasks.py",
            LEGACY_DIR / "exp71_pan_block_pulse_hold.py",
            LEGACY_DIR / "pan_block.py",
            LEGACY_DIR / "plru_regularizers.py",
        )
    )
    for item in plan:
        entries.append(("result", item.result_path))
        entries.append(("checkpoint", item.checkpoint_path))

    snapshots: List[FileSnapshot] = []
    seen: Dict[Path, str] = {}
    for kind, path in entries:
        resolved = path.resolve(strict=True)
        if resolved in seen:
            if seen[resolved] != kind:
                raise ValueError(
                    f"one input path is assigned conflicting roles {seen[resolved]!r}/{kind!r}: "
                    f"{resolved}"
                )
            continue
        seen[resolved] = kind
        snapshots.append(snapshot_file(resolved, kind))
    return snapshots


def verify_source_snapshot(snapshots: Sequence[FileSnapshot]) -> None:
    """Fail completion if any frozen input changed during evaluation."""

    failures = []
    for snapshot in snapshots:
        try:
            current = snapshot_file(snapshot.path, snapshot.kind)
        except (FileNotFoundError, RuntimeError) as exc:
            failures.append(str(exc))
            continue
        if current.size != snapshot.size or current.sha256 != snapshot.sha256:
            failures.append(
                f"input changed during evaluation: {snapshot.path} "
                f"({snapshot.sha256} -> {current.sha256})"
            )
    if failures:
        raise RuntimeError("source snapshot verification failed:\n" + "\n".join(failures))


def snapshot_index(snapshots: Sequence[FileSnapshot]) -> Dict[Path, FileSnapshot]:
    return {snapshot.path: snapshot for snapshot in snapshots}


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically publish a small completion/manifest file."""

    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def derived_seed(base_seed: int, *parts: Any) -> int:
    text = "|".join([str(int(base_seed))] + [str(part) for part in parts])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def parse_csv_values(value: str, cast: Any) -> List[Any]:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def safe_float(value: Any, default: float) -> float:
    if value in (None, "", "None"):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int) -> int:
    if value in (None, "", "None"):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def validate_relative_directory(value: str, field: str) -> str:
    """Accept a non-empty relative directory that cannot traverse its root."""

    path = Path(str(value))
    if path.is_absolute() or not path.parts or path == Path(".") or ".." in path.parts:
        raise ValueError(f"{field} must be a non-empty relative directory without '..': {value!r}")
    return path.as_posix()


def load_specs(path: Path) -> List[CheckpointSpec]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported spec schema: {payload.get('schema_version')!r}")
    raw_specs = payload.get("checkpoints")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("spec file must contain a non-empty checkpoints list")
    required = {"paper_model", "task", "tag", "result_dir", "checkpoint_dir"}
    specs: List[CheckpointSpec] = []
    seen = set()
    for index, item in enumerate(raw_specs):
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError(f"checkpoint spec {index} must have exactly {sorted(required)}")
        values = {key: str(item[key]) for key in required}
        values["result_dir"] = validate_relative_directory(values["result_dir"], "result_dir")
        values["checkpoint_dir"] = validate_relative_directory(
            values["checkpoint_dir"], "checkpoint_dir"
        )
        spec = CheckpointSpec(**values)
        if spec.task not in TASKS:
            raise ValueError(f"checkpoint spec {index} has unknown Exp88 task {spec.task!r}")
        key = (spec.paper_model, spec.task, spec.tag, spec.result_dir, spec.checkpoint_dir)
        if key in seen:
            raise ValueError(f"duplicate checkpoint spec: {key}")
        seen.add(key)
        specs.append(spec)
    return specs


def build_plan(
    specs: Sequence[CheckpointSpec],
    artifact_root: Path,
    tasks: Sequence[str],
    models: Sequence[str],
    seeds: Sequence[int],
) -> List[PlanItem]:
    task_filter = set(tasks)
    model_filter = set(models)
    plan: List[PlanItem] = []
    for spec in specs:
        if task_filter and spec.task not in task_filter:
            continue
        if model_filter and spec.paper_model not in model_filter:
            continue
        for seed in seeds:
            plan.append(
                PlanItem(
                    spec=spec,
                    seed=int(seed),
                    result_path=spec.result_path(artifact_root, int(seed)),
                    checkpoint_path=spec.checkpoint_path(artifact_root, int(seed)),
                )
            )
    if not plan:
        raise ValueError("filters selected no checkpoint plans")
    return plan


def validate_plan(plan: Sequence[PlanItem], artifact_root: Path, output_dir: Optional[Path]) -> None:
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"artifact root does not exist: {artifact_root}")
    root_resolved = artifact_root.resolve(strict=True)
    missing = []
    for item in plan:
        for path in (item.result_path, item.checkpoint_path):
            if not path.is_file():
                missing.append(str(path))
                continue
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(root_resolved)
            except ValueError as exc:
                raise ValueError(f"input artifact escapes artifact root: {path} -> {resolved}") from exc
    if missing:
        raise FileNotFoundError("missing input artifacts:\n" + "\n".join(missing))
    if output_dir is None:
        return
    output_resolved = output_dir.resolve()
    if output_resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_resolved}")
    protected = set()
    for item in plan:
        protected.add(item.result_path.parent.resolve())
        protected.add(item.checkpoint_path.parent.resolve())
    for path in protected:
        if output_resolved == path or path in output_resolved.parents:
            raise ValueError(f"output directory may not be inside legacy artifact directory {path}")


def load_model(item: PlanItem, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    result = json.loads(item.result_path.read_text(encoding="utf-8"))
    if result.get("task") != item.spec.task:
        raise ValueError(f"result task mismatch in {item.result_path}")
    if str(result.get("tag")) != item.spec.tag:
        raise ValueError(f"result tag mismatch in {item.result_path}")
    if int(result.get("seed")) != item.seed:
        raise ValueError(f"result seed mismatch in {item.result_path}")
    input_dim, output_dim = task_io_dims(item.spec.task)
    variant = normalize_model_variant(result.get("model") or result.get("raw_model"))
    model = build_model_variant(
        variant=variant,
        input_dim=input_dim,
        output_dim=output_dim,
        rank=model_rank_for_task(item.spec.task),
        d_model=safe_int(result.get("d_model"), 96),
        rec_dim=safe_int(result.get("rec_dim"), 96),
        layers=safe_int(result.get("layers"), 1),
        dropout=safe_float(result.get("dropout"), 0.0),
        plru_tau=safe_float(result.get("plru_tau"), 0.001),
        plru_c=safe_float(result.get("plru_c"), 50.0),
        pan_lambda_min=safe_float(result.get("pan_lambda_min"), 0.90),
        pan_lambda_max=safe_float(result.get("pan_lambda_max"), 0.999),
        rank_matched_lambda_high=safe_float(result.get("rank_matched_lambda_high"), 0.999),
        rank_matched_lambda_low=safe_float(result.get("rank_matched_lambda_low"), 0.0),
    ).to(device)
    try:
        state_dict = torch.load(item.checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(item.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, result


def geometry_grid(task: str, count: int, offset: float = 0.0) -> np.ndarray:
    geom = task_geometry(task)
    count = int(count)
    if count < 1:
        raise ValueError("point count must be positive")
    per_axis = int(math.ceil(count ** (1.0 / geom.q_dim)))
    if geom.angle_velocity:
        axes = [2.0 * math.pi * ((np.arange(per_axis) + offset) / per_axis) for _ in range(geom.q_dim)]
    else:
        # Stay away from the SurfaceGeometry clamp boundary.
        width = 1.6 / per_axis
        axes = [np.linspace(-0.8, 0.8, per_axis) + (offset - 0.5) * width for _ in range(geom.q_dim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in mesh], axis=-1)[:count]
    if geom.angle_velocity:
        points = np.remainder(points, 2.0 * math.pi)
    else:
        points = np.clip(points, -0.8, 0.8)
    return points.astype(np.float32)


def velocity_scale_for_geometry(name: str) -> float:
    if name == "ring":
        return 3.0 * math.pi / 180.0
    if name in {"torus", "complex_curve"}:
        return 2.5 * math.pi / 180.0
    if name == "surface":
        return 0.018
    raise ValueError(name)


def deterministic_velocity(task: str, count: int, horizon: int, base_seed: int) -> np.ndarray:
    geom = task_geometry(task)
    velocity = np.zeros((int(horizon), int(count), geom.q_dim), dtype=np.float32)
    if not is_integrate_task(task) or horizon <= 0:
        return velocity
    rng = np.random.default_rng(derived_seed(base_seed, task, "velocity-v1"))
    scale = velocity_scale_for_geometry(geom.name)
    final_hold = min(max(2, horizon // 5), horizon)
    active = horizon - final_hold
    for batch_index in range(count):
        t = 0
        segment = 0
        while t < active:
            length = 3 + ((batch_index + 3 * segment) % 8)
            end = min(active, t + length)
            if (batch_index + segment) % 4 != 0:
                step = rng.uniform(-scale, scale, size=geom.q_dim).astype(np.float32)
                if geom.q_dim == 2 and (batch_index + segment) % 3 == 0:
                    step[(batch_index + segment) % 2] = 0.0
                velocity[t:end, batch_index] = step
            t = end
            segment += 1
    return velocity


def make_assets(task: str, family_points: int, eval_points: int, horizon: int, base_seed: int) -> Dict[str, np.ndarray]:
    return {
        "q_family": geometry_grid(task, family_points, offset=0.0),
        "q_eval": geometry_grid(task, eval_points, offset=0.5),
        "v_eval": deterministic_velocity(task, eval_points, horizon, base_seed),
    }


def save_assets(output_dir: Path, assets_by_task: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    asset_dir = output_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=False)
    manifest: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "tasks": {}}
    for task in sorted(assets_by_task):
        task_rows = {}
        for name in sorted(assets_by_task[task]):
            array = np.asarray(assets_by_task[task][name])
            rel = Path("assets") / f"{task}__{name}.npy"
            path = output_dir / rel
            with path.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
            task_rows[name] = {
                "path": rel.as_posix(),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": sha256_file(path),
            }
        manifest["tasks"][task] = task_rows
    write_json(output_dir / "asset_manifest.json", manifest)
    return manifest


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, device=device, dtype=torch.float32)


@torch.no_grad()
def run_state(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    state = model.init_state(x.shape[1], x.device)
    for x_t in x:
        state = model.step(x_t, state)
    return state


@torch.no_grad()
def state_from_qv(model: torch.nn.Module, task: str, q0: torch.Tensor, velocity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    geom = task_geometry(task)
    x, _, target, aux = sequence_from_qv(geom, q0, velocity)
    return run_state(model, x), target, aux["q"][-1]


def wrapped_q_distance(q_a: torch.Tensor, q_b: torch.Tensor, angle_velocity: bool) -> torch.Tensor:
    diff = q_a - q_b
    if angle_velocity:
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return torch.linalg.vector_norm(diff, dim=-1)


def batch_project(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project [B,K,C] or [B,C] vectors onto [B,C,Q] orthonormal bases."""
    if vector.ndim == 2:
        coeff = torch.einsum("bc,bcq->bq", vector, basis)
        return torch.einsum("bq,bcq->bc", coeff, basis)
    if vector.ndim == 3:
        coeff = torch.einsum("bkc,bcq->bkq", vector, basis)
        return torch.einsum("bkq,bcq->bkc", coeff, basis)
    raise ValueError("projection expects rank-2 or rank-3 vectors")


def finite_difference_family(
    model: torch.nn.Module,
    task: str,
    q0: torch.Tensor,
    velocity: torch.Tensor,
    layout: StateLayout,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    geom = task_geometry(task)
    plus_states = []
    minus_states = []
    columns = []
    with torch.no_grad():
        for coordinate in range(geom.q_dim):
            delta = torch.zeros_like(q0)
            delta[:, coordinate] = float(eps)
            plus, _, _ = state_from_qv(model, task, geom.wrap_q(q0 + delta), velocity)
            minus, _, _ = state_from_qv(model, task, geom.wrap_q(q0 - delta), velocity)
            plus_states.append(plus)
            minus_states.append(minus)
            columns.append((layout.carrier(plus) - layout.carrier(minus)) / (2.0 * float(eps)))
    jacobian = torch.stack(columns, dim=-1)  # [batch, carrier, q_dim]
    basis, triangular = torch.linalg.qr(jacobian, mode="reduced")
    singular_values = torch.linalg.svdvals(jacobian)
    return (
        basis,
        singular_values,
        torch.stack(plus_states, dim=0),
        torch.stack(minus_states, dim=0),
        triangular,
    )


def roll_blank_capture(
    model: torch.nn.Module,
    state: torch.Tensor,
    horizons: Sequence[int],
    track_grad: bool = False,
) -> Dict[int, torch.Tensor]:
    requested = sorted(set(int(value) for value in horizons))
    if not requested or requested[0] < 0:
        raise ValueError("blank horizons must be nonnegative")
    blank = torch.zeros(state.shape[0], model.input_dim, device=state.device, dtype=state.dtype)
    current = state
    captured: Dict[int, torch.Tensor] = {}
    last = 0
    context = torch.enable_grad() if track_grad else torch.no_grad()
    with context:
        for horizon in requested:
            for _ in range(horizon - last):
                current = model.step(blank, current)
            captured[horizon] = current if track_grad else current.detach().clone()
            last = horizon
    return captured


def nearest_family(
    carrier: torch.Tensor,
    family_carrier: torch.Tensor,
    family_q: torch.Tensor,
    reference_q: torch.Tensor,
    angle_velocity: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distances = torch.cdist(carrier, family_carrier)
    nearest_distance, nearest_index = distances.min(dim=1)
    nearest_q = family_q[nearest_index]
    q_error = wrapped_q_distance(nearest_q, reference_q, angle_velocity)
    return nearest_distance, q_error, nearest_index


def family_local_scale(carrier: torch.Tensor) -> Tuple[float, torch.Tensor]:
    distances = torch.cdist(carrier, carrier)
    distances.fill_diagonal_(float("inf"))
    nearest, index = distances.min(dim=1)
    scale = float(nearest.median().item())
    return max(scale, 1e-8), index


def pca_metrics(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    centered = values - values.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    variance = singular.square()
    total = variance.sum()
    if float(total.item()) <= 1e-20:
        return {f"{prefix}_participation_ratio": 0.0, f"{prefix}_rank90": 0.0}
    weights = variance / total
    pr = 1.0 / weights.square().sum()
    rank90 = int((torch.cumsum(weights, dim=0) < 0.90).sum().item()) + 1
    return {f"{prefix}_participation_ratio": float(pr.item()), f"{prefix}_rank90": float(rank90)}


def direction_distance(values: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    value_norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(1e-12)
    ref_norm = torch.linalg.vector_norm(references, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.linalg.vector_norm(values / value_norm - references / ref_norm, dim=-1)


def blank_flow_analysis(
    model: torch.nn.Module,
    item: PlanItem,
    layout: StateLayout,
    q_family: torch.Tensor,
    family_state: torch.Tensor,
    family_target: torch.Tensor,
    horizons: Sequence[int],
    local_scale: float,
    neighbor_index: torch.Tensor,
) -> List[Dict[str, Any]]:
    """Measure autonomous flow of a dense, cue-defined clean state family."""
    geom = task_geometry(item.spec.task)
    carrier0 = layout.carrier(family_state)
    full0 = family_state
    with torch.no_grad():
        output0 = model.decode(family_state)
        target_mse0 = ((output0 - family_target) ** 2).mean(dim=-1)
        initial_neighbor_distance = torch.linalg.vector_norm(
            carrier0 - carrier0[neighbor_index], dim=-1
        ).clamp_min(1e-12)
        captures = roll_blank_capture(model, family_state, horizons)
        rows: List[Dict[str, Any]] = []
        for horizon in sorted(captures):
            state = captures[horizon]
            carrier = layout.carrier(state)
            output = model.decode(state)
            target_mse = ((output - family_target) ** 2).mean(dim=-1)
            nearest_distance, q_drift, _ = nearest_family(
                carrier, carrier0, q_family, q_family, geom.angle_velocity
            )
            neighbor_distance = torch.linalg.vector_norm(
                carrier - carrier[neighbor_index], dim=-1
            )
            row: Dict[str, Any] = {
                "paper_model": item.spec.paper_model,
                "task": item.spec.task,
                "tag": item.spec.tag,
                "seed": item.seed,
                "blank_horizon": int(horizon),
                "n_family": int(q_family.shape[0]),
                "carrier_dim": layout.carrier_dim,
                "stream_dim": layout.stream_dim,
                "local_family_scale": local_scale,
                "carrier_drift_norm_mean": float(
                    torch.linalg.vector_norm(carrier - carrier0, dim=-1).mean().item()
                ),
                "carrier_drift_normalized_mean": float(
                    (torch.linalg.vector_norm(carrier - carrier0, dim=-1) / local_scale).mean().item()
                ),
                "carrier_direction_drift_mean": float(direction_distance(carrier, carrier0).mean().item()),
                "carrier_norm_mean": float(torch.linalg.vector_norm(carrier, dim=-1).mean().item()),
                "carrier_norm_ratio_mean": float(
                    (
                        torch.linalg.vector_norm(carrier, dim=-1)
                        / torch.linalg.vector_norm(carrier0, dim=-1).clamp_min(1e-12)
                    ).mean().item()
                ),
                "full_state_norm_mean": float(torch.linalg.vector_norm(state, dim=-1).mean().item()),
                "full_state_norm_ratio_mean": float(
                    (
                        torch.linalg.vector_norm(state, dim=-1)
                        / torch.linalg.vector_norm(full0, dim=-1).clamp_min(1e-12)
                    ).mean().item()
                ),
                "output_pair_mse_mean": float(((output - output0) ** 2).mean(dim=-1).mean().item()),
                "target_mse_mean": float(target_mse.mean().item()),
                "clean_subtracted_target_mse_mean": float((target_mse - target_mse0).mean().item()),
                "nearest_family_distance_mean": float(nearest_distance.mean().item()),
                "nearest_family_distance_normalized_mean": float((nearest_distance / local_scale).mean().item()),
                "represented_q_drift_mean": float(q_drift.mean().item()),
                "neighbor_separation_ratio_mean": float(
                    (neighbor_distance / initial_neighbor_distance).mean().item()
                ),
            }
            row.update(pca_metrics(carrier, "carrier"))
            row.update(pca_metrics(state, "full_state"))
            rows.append(row)
    return rows


def deterministic_directions(
    basis: torch.Tensor,
    tangent_count: int,
    normal_count: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return matched unit tangent/normal carrier directions [B,K,C]."""
    batch, carrier_dim, tangent_dim = basis.shape
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    tangent_coeff = torch.randn(batch, int(tangent_count), tangent_dim, generator=generator)
    normal_raw = torch.randn(batch, int(normal_count), carrier_dim, generator=generator)
    tangent_coeff = tangent_coeff.to(device=basis.device, dtype=basis.dtype)
    normal_raw = normal_raw.to(device=basis.device, dtype=basis.dtype)
    tangent = torch.einsum("bkq,bcq->bkc", tangent_coeff, basis)
    tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-12)
    normal = normal_raw - batch_project(normal_raw, basis)
    normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    # A random vector is almost surely valid, but fail instead of silently
    # injecting a tangent or zero direction in a degenerate tiny carrier.
    if bool((normal_norm < 1e-8).any()):
        raise ValueError("could not construct carrier-normal direction")
    normal = normal / normal_norm
    return tangent, normal


def perturbation_analysis(
    model: torch.nn.Module,
    item: PlanItem,
    layout: StateLayout,
    base_state: torch.Tensor,
    target: torch.Tensor,
    target_q: torch.Tensor,
    tangent_basis: torch.Tensor,
    family_carrier: torch.Tensor,
    family_q: torch.Tensor,
    local_scale: float,
    radii: Sequence[float],
    horizons: Sequence[int],
    tangent_directions: int,
    normal_directions: int,
    direction_seed: int,
) -> List[Dict[str, Any]]:
    geom = task_geometry(item.spec.task)
    tangent, normal = deterministic_directions(
        tangent_basis, tangent_directions, normal_directions, direction_seed
    )
    directions = []
    kinds = []
    direction_ids = []
    for kind, tensor in (("tangent", tangent), ("normal", normal)):
        for direction_id in range(tensor.shape[1]):
            directions.append(tensor[:, direction_id])
            kinds.append(kind)
            direction_ids.append(direction_id)
    direction_tensor = torch.stack(directions, dim=1)  # [B,K,C]
    batch, n_directions, _ = direction_tensor.shape
    conditions = []
    for radius in radii:
        conditions.append(direction_tensor * (float(radius) * local_scale))
    delta = torch.cat(conditions, dim=1)
    # The metadata above is radius-major, matching the concatenation.
    kind_meta = kinds * len(radii)
    direction_meta = direction_ids * len(radii)
    radius_meta = []
    for radius in radii:
        radius_meta.extend([float(radius)] * n_directions)

    n_conditions = delta.shape[1]
    base_repeated = base_state[:, None, :].expand(batch, n_conditions, layout.state_dim).reshape(
        batch * n_conditions, layout.state_dim
    )
    perturbed_carrier = (
        layout.carrier(base_state)[:, None, :] + delta
    ).reshape(batch * n_conditions, layout.carrier_dim)
    perturbed_state = layout.replace_carrier(base_repeated, perturbed_carrier)
    # This is the key regression guard: stream coordinates are not kicked.
    if layout.stream_dim:
        if not torch.equal(
            perturbed_state[:, layout.carrier_dim :], base_repeated[:, layout.carrier_dim :]
        ):
            raise AssertionError("stream coordinates changed during carrier-only kick")

    all_horizons = sorted(set(int(value) for value in horizons))
    clean_captures = roll_blank_capture(model, base_state, all_horizons)
    kicked_captures = roll_blank_capture(model, perturbed_state, all_horizons)
    target_repeated = target[:, None, :].expand(batch, n_conditions, target.shape[-1]).reshape(
        batch * n_conditions, target.shape[-1]
    )
    target_q_repeated = target_q[:, None, :].expand(batch, n_conditions, target_q.shape[-1]).reshape(
        batch * n_conditions, target_q.shape[-1]
    )
    sample_ids = torch.arange(batch, device=base_state.device).repeat_interleave(n_conditions)
    initial_norm = torch.linalg.vector_norm(delta.reshape(batch * n_conditions, -1), dim=-1)

    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for horizon in all_horizons:
            clean_state = clean_captures[horizon]
            clean_repeated = clean_state[:, None, :].expand(
                batch, n_conditions, layout.state_dim
            ).reshape(batch * n_conditions, layout.state_dim)
            kicked_state = kicked_captures[horizon]
            clean_output = model.decode(clean_state)
            clean_output_repeated = clean_output[:, None, :].expand(
                batch, n_conditions, clean_output.shape[-1]
            ).reshape(batch * n_conditions, clean_output.shape[-1])
            kicked_output = model.decode(kicked_state)
            clean_target_mse = ((clean_output_repeated - target_repeated) ** 2).mean(dim=-1)
            kicked_target_mse = ((kicked_output - target_repeated) ** 2).mean(dim=-1)
            output_pair_mse = ((kicked_output - clean_output_repeated) ** 2).mean(dim=-1)
            carrier_deviation = torch.linalg.vector_norm(
                layout.carrier(kicked_state) - layout.carrier(clean_repeated), dim=-1
            )
            clean_nearest_distance, clean_q_error, _ = nearest_family(
                layout.carrier(clean_state),
                family_carrier,
                family_q,
                target_q,
                geom.angle_velocity,
            )
            clean_nearest_repeated = clean_nearest_distance[:, None].expand(
                batch, n_conditions
            ).reshape(batch * n_conditions)
            clean_q_error_repeated = clean_q_error[:, None].expand(
                batch, n_conditions
            ).reshape(batch * n_conditions)
            nearest_distance, represented_q_error, _ = nearest_family(
                layout.carrier(kicked_state),
                family_carrier,
                family_q,
                target_q_repeated,
                geom.angle_velocity,
            )
            for flat_index in range(batch * n_conditions):
                condition_index = flat_index % n_conditions
                rows.append(
                    {
                        "paper_model": item.spec.paper_model,
                        "task": item.spec.task,
                        "tag": item.spec.tag,
                        "seed": item.seed,
                        "sample": int(sample_ids[flat_index].item()),
                        "direction_kind": kind_meta[condition_index],
                        "direction_id": int(direction_meta[condition_index]),
                        "radius_local_scale": radius_meta[condition_index],
                        "blank_horizon": int(horizon),
                        "carrier_dim": layout.carrier_dim,
                        "stream_dim": layout.stream_dim,
                        "initial_kick_norm": float(initial_norm[flat_index].item()),
                        "target_mse_clean": float(clean_target_mse[flat_index].item()),
                        "target_mse_kicked": float(kicked_target_mse[flat_index].item()),
                        "clean_subtracted_target_mse": float(
                            (kicked_target_mse[flat_index] - clean_target_mse[flat_index]).item()
                        ),
                        "output_pair_mse": float(output_pair_mse[flat_index].item()),
                        "carrier_deviation_norm": float(carrier_deviation[flat_index].item()),
                        "carrier_deviation_ratio": float(
                            (carrier_deviation[flat_index] / initial_norm[flat_index].clamp_min(1e-12)).item()
                        ),
                        "nearest_family_distance_clean": float(
                            clean_nearest_repeated[flat_index].item()
                        ),
                        "nearest_family_distance": float(nearest_distance[flat_index].item()),
                        "nearest_family_distance_normalized": float(
                            (nearest_distance[flat_index] / local_scale).item()
                        ),
                        "clean_subtracted_nearest_family_distance": float(
                            (nearest_distance[flat_index] - clean_nearest_repeated[flat_index]).item()
                        ),
                        "represented_q_error_clean": float(
                            clean_q_error_repeated[flat_index].item()
                        ),
                        "represented_q_error": float(represented_q_error[flat_index].item()),
                        "clean_subtracted_represented_q_error": float(
                            (represented_q_error[flat_index] - clean_q_error_repeated[flat_index]).item()
                        ),
                    }
                )
    return rows


def retention_coordinates(model: torch.nn.Module, layout: StateLayout) -> Optional[torch.Tensor]:
    """Map exposed lambda magnitudes onto flat carrier coordinates."""
    chunks = []
    if hasattr(model, "blocks") and hasattr(model, "_rec_slices"):
        for block, rec_slice in zip(model.blocks, model._rec_slices):
            rec = block.rec
            if not hasattr(rec, "lam_mag"):
                return None
            lam = rec.lam_mag().detach().flatten()
            width = int(rec_slice.stop - rec_slice.start)
            if width == lam.numel():
                chunks.append(lam)
            elif width == 2 * lam.numel():
                chunks.append(torch.cat([lam, lam], dim=0))
            else:
                return None
    elif hasattr(model, "lam_mag"):
        lam = model.lam_mag().detach().flatten()
        if layout.carrier_dim == lam.numel():
            chunks.append(lam)
        elif layout.carrier_dim == 2 * lam.numel():
            chunks.append(torch.cat([lam, lam], dim=0))
        else:
            return None
    else:
        return None
    values = torch.cat(chunks, dim=0)
    if values.numel() != layout.carrier_dim:
        return None
    return values


def retention_alignment_metrics(
    model: torch.nn.Module,
    layout: StateLayout,
    tangent_basis: torch.Tensor,
    threshold: float,
) -> Dict[str, Any]:
    values = retention_coordinates(model, layout)
    if values is None:
        return {
            "retention_available": False,
            "retention_support_count": "",
            "retention_lambda_max": "",
            "tangent_high_retention_fraction_mean": "",
            "principal_angle_mean_deg": "",
            "principal_angle_max_deg": "",
        }
    tangent_dim = tangent_basis.shape[-1]
    support = torch.where(values >= float(threshold))[0]
    if support.numel() < tangent_dim:
        support = torch.topk(values, k=int(tangent_dim), largest=True).indices
    fractions = tangent_basis[:, support, :].square().sum(dim=(1, 2)) / float(tangent_dim)
    all_angles = []
    for basis in tangent_basis:
        singular = torch.linalg.svdvals(basis[support, :]).clamp(0.0, 1.0)
        angles = torch.arccos(singular) * (180.0 / math.pi)
        all_angles.append(angles)
    angle_values = torch.cat(all_angles)
    return {
        "retention_available": True,
        "retention_support_count": int(support.numel()),
        "retention_lambda_max": float(values.max().item()),
        "tangent_high_retention_fraction_mean": float(fractions.mean().item()),
        "principal_angle_mean_deg": float(angle_values.mean().item()),
        "principal_angle_max_deg": float(angle_values.max().item()),
    }


def transported_tangent_bases(
    model: torch.nn.Module,
    layout: StateLayout,
    plus_states: torch.Tensor,
    minus_states: torch.Tensor,
    eps: float,
    horizons: Sequence[int],
) -> Dict[int, torch.Tensor]:
    """Finite-difference tangent bases transported along the clean blank flow."""
    tangent_dim, batch, _ = plus_states.shape
    plus_capture = [roll_blank_capture(model, plus_states[j], horizons) for j in range(tangent_dim)]
    minus_capture = [roll_blank_capture(model, minus_states[j], horizons) for j in range(tangent_dim)]
    output: Dict[int, torch.Tensor] = {}
    for horizon in sorted(set(int(value) for value in horizons)):
        columns = []
        for coordinate in range(tangent_dim):
            diff = (
                layout.carrier(plus_capture[coordinate][horizon])
                - layout.carrier(minus_capture[coordinate][horizon])
            ) / (2.0 * float(eps))
            columns.append(diff)
        jacobian = torch.stack(columns, dim=-1)
        basis, _ = torch.linalg.qr(jacobian, mode="reduced")
        if basis.shape != (batch, layout.carrier_dim, tangent_dim):
            raise AssertionError("unexpected transported tangent shape")
        output[horizon] = basis
    return output


def directional_jacobian_analysis(
    model: torch.nn.Module,
    item: PlanItem,
    layout: StateLayout,
    base_state: torch.Tensor,
    tangent_basis: torch.Tensor,
    plus_states: torch.Tensor,
    minus_states: torch.Tensor,
    tangent_eps: float,
    horizons: Sequence[int],
    tangent_directions: int,
    normal_directions: int,
    direction_seed: int,
) -> List[Dict[str, Any]]:
    """Propagate zero-input JVPs without materializing a dense Jacobian."""
    requested = sorted(set(int(value) for value in horizons))
    if not requested or requested[0] < 1:
        raise ValueError("Jacobian horizons must be positive")
    tangent, normal = deterministic_directions(
        tangent_basis, tangent_directions, normal_directions, direction_seed
    )
    directions = torch.cat([tangent, normal], dim=1)
    kinds = ["tangent"] * tangent.shape[1] + ["normal"] * normal.shape[1]
    ids = list(range(tangent.shape[1])) + list(range(normal.shape[1]))
    batch, count, _ = directions.shape
    state = base_state[:, None, :].expand(batch, count, layout.state_dim).reshape(
        batch * count, layout.state_dim
    ).detach()
    carrier_direction = directions.reshape(batch * count, layout.carrier_dim)
    direction = layout.embed_direction(carrier_direction).detach()
    initial_norm = torch.linalg.vector_norm(carrier_direction, dim=-1).clamp_min(1e-12)
    sample_ids = torch.arange(batch, device=base_state.device).repeat_interleave(count)
    tangent_at_horizon = transported_tangent_bases(
        model, layout, plus_states, minus_states, tangent_eps, requested
    )
    rows: List[Dict[str, Any]] = []
    blank = torch.zeros(batch * count, model.input_dim, device=state.device, dtype=state.dtype)
    requested_set = set(requested)

    # Parameters are constants of the state map.  Sequential one-step JVPs
    # produce D(F_0^H)v while bounding graph memory at O(1) in H.
    for step in range(1, max(requested) + 1):
        state = state.detach().requires_grad_(True)
        direction = direction.detach()

        def blank_map(current: torch.Tensor) -> torch.Tensor:
            return model.step(blank, current)

        next_state, next_direction = torch.autograd.functional.jvp(
            blank_map,
            (state,),
            (direction,),
            create_graph=False,
            strict=False,
        )
        state = next_state.detach()
        direction = next_direction.detach()
        if step not in requested_set:
            continue

        carrier_jvp = layout.carrier(direction)
        output_basis = tangent_at_horizon[step]
        repeated_basis = output_basis[:, None, :, :].expand(
            batch, count, layout.carrier_dim, output_basis.shape[-1]
        ).reshape(batch * count, layout.carrier_dim, output_basis.shape[-1])
        tangent_component = batch_project(carrier_jvp, repeated_basis)
        normal_component = carrier_jvp - tangent_component
        carrier_gain = torch.linalg.vector_norm(carrier_jvp, dim=-1) / initial_norm
        full_state_gain = torch.linalg.vector_norm(direction, dim=-1) / initial_norm
        if layout.stream_dim:
            stream_gain = (
                torch.linalg.vector_norm(direction[:, layout.carrier_dim :], dim=-1)
                / initial_norm
            )
        else:
            stream_gain = torch.zeros_like(carrier_gain)
        tangent_gain = torch.linalg.vector_norm(tangent_component, dim=-1) / initial_norm
        normal_gain = torch.linalg.vector_norm(normal_component, dim=-1) / initial_norm

        decode_state = state.detach().requires_grad_(True)

        def decode_map(current: torch.Tensor) -> torch.Tensor:
            return model.decode(current)

        _, decoded_jvp = torch.autograd.functional.jvp(
            decode_map,
            (decode_state,),
            (direction,),
            create_graph=False,
            strict=False,
        )
        decoded_gain = torch.linalg.vector_norm(decoded_jvp.detach(), dim=-1) / initial_norm
        for flat_index in range(batch * count):
            condition = flat_index % count
            rows.append(
                {
                    "paper_model": item.spec.paper_model,
                    "task": item.spec.task,
                    "tag": item.spec.tag,
                    "seed": item.seed,
                    "sample": int(sample_ids[flat_index].item()),
                    "direction_kind": kinds[condition],
                    "direction_id": int(ids[condition]),
                    "blank_horizon": int(step),
                    "carrier_dim": layout.carrier_dim,
                    "stream_dim": layout.stream_dim,
                    "carrier_gain": float(carrier_gain[flat_index].item()),
                    "full_state_gain": float(full_state_gain[flat_index].item()),
                    "stream_gain": float(stream_gain[flat_index].item()),
                    "tangent_output_gain": float(tangent_gain[flat_index].item()),
                    "normal_output_gain": float(normal_gain[flat_index].item()),
                    "decoded_output_gain": float(decoded_gain[flat_index].item()),
                    "normal_gain_fraction": float(
                        (normal_gain[flat_index] / carrier_gain[flat_index].clamp_min(1e-12)).item()
                    ),
                }
            )
    return rows


def numeric_summary(
    rows: Sequence[Dict[str, Any]],
    group_keys: Sequence[str],
    excluded_numeric: Sequence[str],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)
    excluded = set(group_keys) | set(excluded_numeric)
    output = []
    for key in sorted(groups, key=lambda value: tuple(str(part) for part in value)):
        values = groups[key]
        summary: Dict[str, Any] = {name: value for name, value in zip(group_keys, key)}
        summary["n_observations"] = len(values)
        numeric_keys = []
        for name, value in values[0].items():
            if name in excluded or isinstance(value, bool):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric_keys.append(name)
        for name in sorted(numeric_keys):
            array = np.asarray([float(row[name]) for row in values], dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite metric {name} in group {key}")
            summary[f"{name}_mean"] = float(array.mean())
            summary[f"{name}_sd"] = float(array.std(ddof=1)) if array.size > 1 else 0.0
            summary[f"{name}_sem"] = (
                float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
            )
            summary[f"{name}_max"] = float(array.max())
            summary[f"{name}_fraction_gt1"] = float((array > 1.0).mean())
        output.append(summary)
    return output


def evaluate_checkpoint(
    item: PlanItem,
    assets: Dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    frozen_sources: Dict[Path, FileSnapshot],
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    model, legacy_result = load_model(item, device)
    layout = StateLayout.from_model(model)
    q_family = to_tensor(assets["q_family"], device)
    q_eval = to_tensor(assets["q_eval"], device)
    v_eval = to_tensor(assets["v_eval"], device)
    geom = task_geometry(item.spec.task)

    family_velocity = torch.zeros(
        int(args.stabilize_horizon), q_family.shape[0], geom.q_dim, device=device
    )
    family_state, family_target, _ = state_from_qv(
        model, item.spec.task, q_family, family_velocity
    )
    base_state, target, target_q = state_from_qv(model, item.spec.task, q_eval, v_eval)
    direct_velocity = torch.zeros(
        int(args.stabilize_horizon), q_eval.shape[0], geom.q_dim, device=device
    )
    direct_target_state, direct_target, _ = state_from_qv(
        model, item.spec.task, target_q, direct_velocity
    )
    tangent_basis, tangent_singular, plus_states, minus_states, _ = finite_difference_family(
        model,
        item.spec.task,
        q_eval,
        v_eval,
        layout,
        float(args.tangent_eps),
    )
    family_carrier = layout.carrier(family_state)
    local_scale, neighbor_index = family_local_scale(family_carrier)

    blank_rows = blank_flow_analysis(
        model,
        item,
        layout,
        q_family,
        family_state,
        family_target,
        args.recovery_horizons,
        local_scale,
        neighbor_index,
    )
    perturb_rows = perturbation_analysis(
        model,
        item,
        layout,
        base_state,
        target,
        target_q,
        tangent_basis,
        family_carrier,
        q_family,
        local_scale,
        args.radii,
        args.recovery_horizons,
        int(args.tangent_directions),
        int(args.normal_directions),
        derived_seed(args.asset_seed, item.spec.task, "kick-directions-v1"),
    )
    jacobian_count = min(int(args.jacobian_samples), int(base_state.shape[0]))
    jac_rows = directional_jacobian_analysis(
        model,
        item,
        layout,
        base_state[:jacobian_count],
        tangent_basis[:jacobian_count],
        plus_states[:, :jacobian_count],
        minus_states[:, :jacobian_count],
        float(args.tangent_eps),
        args.jacobian_horizons,
        int(args.jacobian_tangent_directions),
        int(args.jacobian_normal_directions),
        derived_seed(args.asset_seed, item.spec.task, "jacobian-directions-v1"),
    )

    with torch.no_grad():
        base_output = model.decode(base_state)
        checkpoint_row: Dict[str, Any] = {
            "paper_model": item.spec.paper_model,
            "task": item.spec.task,
            "tag": item.spec.tag,
            "seed": item.seed,
            "legacy_model": str(legacy_result.get("model")),
            "carrier_dim": layout.carrier_dim,
            "stream_dim": layout.stream_dim,
            "state_dim": layout.state_dim,
            "n_family": int(q_family.shape[0]),
            "n_eval": int(q_eval.shape[0]),
            "local_family_scale": local_scale,
            "endpoint_target_mse": float(((base_output - target) ** 2).mean().item()),
            "history_conditioned_carrier_gap_mean": float(
                torch.linalg.vector_norm(
                    layout.carrier(base_state) - layout.carrier(direct_target_state), dim=-1
                ).mean().item()
            ),
            "history_conditioned_carrier_gap_normalized_mean": float(
                (
                    torch.linalg.vector_norm(
                        layout.carrier(base_state) - layout.carrier(direct_target_state), dim=-1
                    )
                    / local_scale
                ).mean().item()
            ),
            "history_conditioned_output_mse": float(
                ((base_output - model.decode(direct_target_state)) ** 2).mean().item()
            ),
            "direct_cue_target_mse": float(
                ((model.decode(direct_target_state) - direct_target) ** 2).mean().item()
            ),
            "tangent_derivative_sv_min": float(tangent_singular.min().item()),
            "tangent_derivative_sv_mean": float(tangent_singular.mean().item()),
            "result_sha256": frozen_sources[item.result_path.resolve(strict=True)].sha256,
            "checkpoint_sha256": frozen_sources[
                item.checkpoint_path.resolve(strict=True)
            ].sha256,
        }
        checkpoint_row.update(
            retention_alignment_metrics(
                model, layout, tangent_basis, float(args.retention_threshold)
            )
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint_row, blank_rows, perturb_rows, jac_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], leading: Sequence[str] = ()) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    all_keys = set()
    for row in rows:
        all_keys.update(row)
    fields = [name for name in leading if name in all_keys]
    fields.extend(sorted(all_keys - set(fields)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_hash_manifest(output_dir: Path) -> str:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if (
            not path.is_file()
            or path.name in {"SHA256SUMS", "INCOMPLETE", "COMPLETE"}
            or path.name.startswith(".SHA256SUMS.tmp.")
        ):
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}")
    manifest = output_dir / "SHA256SUMS"
    atomic_write_text(manifest, "\n".join(rows) + "\n")
    return sha256_file(manifest)


def finalize_output(output_dir: Path) -> None:
    """Publish hashes first and an atomic COMPLETE marker strictly last."""

    hash_manifest_sha256 = write_hash_manifest(output_dir)
    incomplete = output_dir / "INCOMPLETE"
    if not incomplete.is_file():
        raise RuntimeError(f"missing INCOMPLETE marker before finalization: {incomplete}")
    incomplete.unlink()
    complete_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "sha256sums_sha256": hash_manifest_sha256,
    }
    atomic_write_text(
        output_dir / "COMPLETE",
        json.dumps(complete_payload, sort_keys=True, separators=(",", ":")) + "\n",
    )


def runtime_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "asset_algorithm": "exp88-fixed-grid-segment-v1",
        "state_layout_policy": "validated-recurrent-prefix-carrier-v1",
        "asset_seed": int(args.asset_seed),
        "family_points": int(args.family_points),
        "eval_points": int(args.eval_points),
        "eval_horizon": int(args.eval_horizon),
        "stabilize_horizon": int(args.stabilize_horizon),
        "tangent_eps": float(args.tangent_eps),
        "radii": list(map(float, args.radii)),
        "recovery_horizons": list(map(int, args.recovery_horizons)),
        "tangent_directions": int(args.tangent_directions),
        "normal_directions": int(args.normal_directions),
        "jacobian_samples": int(args.jacobian_samples),
        "jacobian_horizons": list(map(int, args.jacobian_horizons)),
        "jacobian_tangent_directions": int(args.jacobian_tangent_directions),
        "jacobian_normal_directions": int(args.jacobian_normal_directions),
        "retention_threshold": float(args.retention_threshold),
        "device": str(args.device),
        "smoke": bool(args.smoke),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic checkpoint-only Exp88 carrier dynamics evaluation"
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tasks", default="ring_hold,torus_integrate")
    parser.add_argument("--models", default="CA-LRU,LRU,GRU")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--asset-seed", type=int, default=880071)
    parser.add_argument("--family-points", type=int, default=64)
    parser.add_argument("--eval-points", type=int, default=64)
    parser.add_argument("--eval-horizon", type=int, default=260)
    parser.add_argument("--stabilize-horizon", type=int, default=260)
    parser.add_argument("--tangent-eps", type=float, default=1e-3)
    parser.add_argument("--radii", default=",".join(map(str, DEFAULT_RADII)))
    parser.add_argument(
        "--recovery-horizons", default=",".join(map(str, DEFAULT_RECOVERY_HORIZONS))
    )
    parser.add_argument("--tangent-directions", type=int, default=8)
    parser.add_argument("--normal-directions", type=int, default=8)
    parser.add_argument("--jacobian-samples", type=int, default=8)
    parser.add_argument(
        "--jacobian-horizons", default=",".join(map(str, DEFAULT_JACOBIAN_HORIZONS))
    )
    parser.add_argument("--jacobian-tangent-directions", type=int, default=4)
    parser.add_argument("--jacobian-normal-directions", type=int, default=8)
    parser.add_argument("--retention-threshold", type=float, default=0.99)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    args.tasks = parse_csv_values(args.tasks, str)
    args.models = parse_csv_values(args.models, str)
    args.seeds = parse_csv_values(args.seeds, int)
    args.radii = parse_csv_values(args.radii, float)
    args.recovery_horizons = parse_csv_values(args.recovery_horizons, int)
    args.jacobian_horizons = parse_csv_values(args.jacobian_horizons, int)
    if any(task not in TASKS for task in args.tasks):
        raise ValueError(f"unknown task in {args.tasks}")
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        raise ValueError("seeds must be nonnegative")
    if not args.radii or any(radius <= 0 for radius in args.radii):
        raise ValueError("radii must be positive")
    if not args.recovery_horizons or min(args.recovery_horizons) < 0:
        raise ValueError("recovery horizons must be nonnegative")
    if not args.jacobian_horizons or min(args.jacobian_horizons) < 1:
        raise ValueError("Jacobian horizons must be positive")
    positive_names = (
        "family_points",
        "eval_points",
        "eval_horizon",
        "stabilize_horizon",
        "tangent_directions",
        "normal_directions",
        "jacobian_samples",
        "jacobian_tangent_directions",
        "jacobian_normal_directions",
    )
    for name in positive_names:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    if not args.tangent_eps > 0:
        raise ValueError("tangent-eps must be positive")
    if not 0 < args.retention_threshold <= 1:
        raise ValueError("retention-threshold must be in (0,1]")
    return args


def apply_smoke_settings(args: argparse.Namespace) -> None:
    args.device = "cpu"
    args.family_points = min(int(args.family_points), 4)
    args.eval_points = min(int(args.eval_points), 2)
    args.eval_horizon = min(int(args.eval_horizon), 4)
    args.stabilize_horizon = min(int(args.stabilize_horizon), 4)
    args.radii = [float(args.radii[0])]
    args.recovery_horizons = [0, 2]
    args.tangent_directions = 1
    args.normal_directions = 1
    args.jacobian_samples = 1
    args.jacobian_horizons = [1, 2]
    args.jacobian_tangent_directions = 1
    args.jacobian_normal_directions = 1


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return HERE / "outputs" / f"dynamics-{stamp}"


def dry_run_payload(plan: Sequence[PlanItem], args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "status": "dry-run-ok",
        "artifact_root": str(args.artifact_root.resolve()),
        "specs": str(args.specs.resolve()),
        "output_dir": str(args.output_dir.resolve()) if args.output_dir else None,
        "analysis": runtime_config(args),
        "plans": [
            {
                "paper_model": item.spec.paper_model,
                "task": item.spec.task,
                "tag": item.spec.tag,
                "seed": item.seed,
                "result": str(item.result_path),
                "result_bytes": item.result_path.stat().st_size,
                "checkpoint": str(item.checkpoint_path),
                "checkpoint_bytes": item.checkpoint_path.stat().st_size,
            }
            for item in plan
        ],
    }


def run(args: argparse.Namespace) -> Path:
    if args.smoke:
        apply_smoke_settings(args)
    if args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
    artifact_root = args.artifact_root.expanduser().resolve()
    args.artifact_root = artifact_root
    args.specs = args.specs.expanduser().resolve()
    specs = load_specs(args.specs)
    plan = build_plan(specs, artifact_root, args.tasks, args.models, args.seeds)
    if args.smoke:
        plan = plan[:1]
    if args.output_dir is None and not args.dry_run:
        args.output_dir = default_output_dir()
    elif args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    validate_plan(plan, artifact_root, args.output_dir)

    if args.dry_run:
        print(json.dumps(dry_run_payload(plan, args), indent=2, sort_keys=True))
        return Path()

    # Freeze all inputs before creating the output directory.  The same bytes
    # are checked again immediately before completion is published.
    source_snapshots = capture_source_snapshot(plan, args.specs)
    frozen_sources = snapshot_index(source_snapshots)

    output_dir = args.output_dir
    assert output_dir is not None
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    (output_dir / "INCOMPLETE").write_text(
        "This marker is removed only after all outputs and hashes are written.\n",
        encoding="utf-8",
    )

    torch.manual_seed(int(args.asset_seed))
    np.random.seed(int(args.asset_seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.asset_seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    tasks = sorted({item.spec.task for item in plan})
    assets_by_task = {
        task: make_assets(
            task,
            int(args.family_points),
            int(args.eval_points),
            int(args.eval_horizon),
            int(args.asset_seed),
        )
        for task in tasks
    }
    asset_manifest = save_assets(output_dir, assets_by_task)

    checkpoint_rows: List[Dict[str, Any]] = []
    blank_rows: List[Dict[str, Any]] = []
    perturb_rows: List[Dict[str, Any]] = []
    jacobian_rows: List[Dict[str, Any]] = []
    for index, item in enumerate(plan, start=1):
        print(
            f"[{index}/{len(plan)}] {item.spec.task} {item.spec.paper_model} "
            f"tag={item.spec.tag} seed={item.seed}",
            flush=True,
        )
        checkpoint, blank, perturb, jacobian = evaluate_checkpoint(
            item, assets_by_task[item.spec.task], args, device, frozen_sources
        )
        checkpoint_rows.append(checkpoint)
        blank_rows.extend(blank)
        perturb_rows.extend(perturb)
        jacobian_rows.extend(jacobian)

    perturb_summary = numeric_summary(
        perturb_rows,
        group_keys=(
            "paper_model",
            "task",
            "tag",
            "seed",
            "direction_kind",
            "radius_local_scale",
            "blank_horizon",
        ),
        excluded_numeric=("sample", "direction_id", "carrier_dim", "stream_dim"),
    )
    jacobian_summary = numeric_summary(
        jacobian_rows,
        group_keys=(
            "paper_model",
            "task",
            "tag",
            "seed",
            "direction_kind",
            "blank_horizon",
        ),
        excluded_numeric=("sample", "direction_id", "carrier_dim", "stream_dim"),
    )

    common_leading = ("paper_model", "task", "tag", "seed")
    write_csv(output_dir / "checkpoint_metrics.csv", checkpoint_rows, common_leading)
    write_csv(output_dir / "blank_flow_metrics.csv", blank_rows, common_leading)
    write_csv(output_dir / "perturbation_metrics.csv", perturb_rows, common_leading)
    write_csv(output_dir / "perturbation_summary.csv", perturb_summary, common_leading)
    write_csv(output_dir / "jacobian_directional_metrics.csv", jacobian_rows, common_leading)
    write_csv(
        output_dir / "jacobian_directional_summary.csv", jacobian_summary, common_leading
    )

    source_artifacts = [
        {
            "paper_model": item.spec.paper_model,
            "task": item.spec.task,
            "tag": item.spec.tag,
            "seed": item.seed,
            "result_path": str(item.result_path),
            "result_sha256": frozen_sources[
                item.result_path.resolve(strict=True)
            ].sha256,
            "checkpoint_path": str(item.checkpoint_path),
            "checkpoint_sha256": frozen_sources[
                item.checkpoint_path.resolve(strict=True)
            ].sha256,
        }
        for item in plan
    ]
    spec_snapshot = frozen_sources[args.specs.resolve(strict=True)]
    code_snapshots = [snapshot for snapshot in source_snapshots if snapshot.kind == "code"]
    config = runtime_config(args)
    results = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "spec_file": str(args.specs),
        "spec_file_sha256": spec_snapshot.sha256,
        "analysis_config": config,
        "analysis_config_sha256": canonical_hash(config),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "code_sources": [
            {
                "path": str(snapshot.path),
                "sha256": snapshot.sha256,
            }
            for snapshot in code_snapshots
        ],
        "asset_manifest": asset_manifest,
        "source_artifacts": source_artifacts,
        "counts": {
            "checkpoints": len(checkpoint_rows),
            "blank_flow_rows": len(blank_rows),
            "perturbation_rows": len(perturb_rows),
            "perturbation_summary_rows": len(perturb_summary),
            "jacobian_rows": len(jacobian_rows),
            "jacobian_summary_rows": len(jacobian_summary),
        },
        "checkpoint_metrics": checkpoint_rows,
        "perturbation_summary": perturb_summary,
        "jacobian_summary": jacobian_summary,
    }
    write_json(output_dir / "results.json", results)
    verify_source_snapshot(source_snapshots)
    finalize_output(output_dir)
    print(f"complete: {output_dir}", flush=True)
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        run(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
