#!/usr/bin/env python3
"""Multi-point, carrier-only ring transport evaluation for Exp88 checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LEGACY = REPO_ROOT / "repro" / "legacy_code"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

from exp71_pan_block_pulse_hold import build_model_variant, normalize_model_variant  # noqa: E402
from exp88_manifold_attractor_tasks import (  # noqa: E402
    RingGeometry,
    model_rank_for_task,
    sequence_from_qv,
    task_io_dims,
)


@dataclass(frozen=True)
class ModelSpec:
    paper_variant: str
    result_dir: str
    checkpoint_dir: str
    tag: str


@dataclass(frozen=True)
class SourceFile:
    path: Path
    kind: str
    sha256: str
    size: int


DEFAULT_ARTIFACT_ROOT = REPO_ROOT.parent


DEFAULT_SPECS = (
    ModelSpec(
        "linear_update",
        "exp88_manifold_main_results",
        "checkpoints_exp88_manifold_main",
        "am_lru_eps3e-5",
    ),
    ModelSpec(
        "input_only_update",
        "exp88_writer_sweep_results",
        "checkpoints_exp88_writer_sweep",
        "am_lru_nw_eps3e-5",
    ),
    ModelSpec(
        "state_dependent_update",
        "exp88_writer_sweep_results",
        "checkpoints_exp88_writer_sweep",
        "am_lru_rnw_eps3e-5",
    ),
)


def parse_csv(text: str, cast):
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_file(path: Path, kind: str) -> SourceFile:
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    digest = sha256(resolved)
    after = resolved.stat()
    before_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_signature != after_signature:
        raise RuntimeError(f"input changed while its initial hash was captured: {resolved}")
    return SourceFile(resolved, str(kind), digest, int(after.st_size))


def capture_sources(records) -> list[SourceFile]:
    entries = [
        ("code", Path(__file__)),
        ("code", LEGACY / "exp88_manifold_attractor_tasks.py"),
        ("code", LEGACY / "exp72_structured_attractor_tasks.py"),
        ("code", LEGACY / "exp71_pan_block_pulse_hold.py"),
        ("code", LEGACY / "pan_block.py"),
        ("code", LEGACY / "plru_regularizers.py"),
    ]
    for _, _, result, checkpoint in records:
        entries.extend((("result", result), ("checkpoint", checkpoint)))
    snapshots = []
    seen = set()
    for kind, path in entries:
        resolved = path.resolve(strict=True)
        if resolved in seen:
            continue
        seen.add(resolved)
        snapshots.append(snapshot_file(resolved, kind))
    return snapshots


def verify_sources(snapshots: list[SourceFile]) -> None:
    failures = []
    for frozen in snapshots:
        try:
            current = snapshot_file(frozen.path, frozen.kind)
        except (FileNotFoundError, RuntimeError) as exc:
            failures.append(str(exc))
            continue
        if current.size != frozen.size or current.sha256 != frozen.sha256:
            failures.append(
                f"input changed during evaluation: {frozen.path} "
                f"({frozen.sha256} -> {current.sha256})"
            )
    if failures:
        raise RuntimeError("source snapshot verification failed:\n" + "\n".join(failures))


def atomic_write_text(path: Path, content: str) -> None:
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


def atomic_write_json(path: Path, payload) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(path, content)


def write_hash_manifest(output_dir: Path) -> str:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in {"SHA256SUMS", "INCOMPLETE", "COMPLETE"}:
            continue
        if path.name.startswith(".SHA256SUMS.tmp."):
            continue
        rows.append(f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}")
    path = output_dir / "SHA256SUMS"
    atomic_write_text(path, "\n".join(rows) + "\n")
    return sha256(path)


def finalize_output(output_dir: Path) -> None:
    hash_sha256 = write_hash_manifest(output_dir)
    incomplete = output_dir / "INCOMPLETE"
    if not incomplete.is_file():
        raise RuntimeError(f"missing INCOMPLETE marker before finalization: {incomplete}")
    incomplete.unlink()
    atomic_write_text(
        output_dir / "COMPLETE",
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "sha256sums_sha256": hash_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
    )


def circular_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def carrier_view(model, state: torch.Tensor) -> torch.Tensor:
    recurrent_size = getattr(model, "recurrent_state_size", None)
    if recurrent_size is None:
        return state
    return state[..., : int(recurrent_size)]


def condition_count(base_points: int, directions, velocity_scales, move_lengths, specs, seeds) -> int:
    return (
        int(base_points)
        * len(directions)
        * len(velocity_scales)
        * len(move_lengths)
        * len(specs)
        * len(seeds)
    )


def checkpoint_paths(artifact_root: Path, spec: ModelSpec, seed: int) -> tuple[Path, Path]:
    result = artifact_root / spec.result_dir / f"ring_integrate_{spec.tag}_seed{seed}.json"
    checkpoint = (
        artifact_root
        / spec.checkpoint_dir
        / f"exp88_ring_integrate_{spec.tag}_seed{seed}.pt"
    )
    return result, checkpoint


def validate_fresh_output(output_dir: Path, records) -> None:
    resolved = output_dir.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to reuse existing output directory: {resolved}")
    protected = {
        path.parent.resolve(strict=True)
        for _, _, result, checkpoint in records
        for path in (result, checkpoint)
    }
    for directory in protected:
        if resolved == directory or directory in resolved.parents:
            raise ValueError(f"output directory may not be inside input directory {directory}")


def load_model(
    result_path: Path,
    checkpoint_path: Path,
    spec: ModelSpec,
    seed: int,
    device: torch.device,
):
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("task") != "ring_integrate":
        raise ValueError(f"result task mismatch in {result_path}")
    if str(result.get("tag")) != spec.tag:
        raise ValueError(f"result tag mismatch in {result_path}")
    if int(result.get("seed")) != int(seed):
        raise ValueError(f"result seed mismatch in {result_path}")
    input_dim, output_dim = task_io_dims("ring_integrate")
    model = build_model_variant(
        variant=normalize_model_variant(result["model"]),
        input_dim=input_dim,
        output_dim=output_dim,
        rank=model_rank_for_task("ring_integrate"),
        d_model=int(result.get("d_model", 96)),
        rec_dim=int(result.get("rec_dim", 96)),
        layers=int(result.get("layers", 1)),
        dropout=0.0,
        plru_tau=float(result.get("plru_tau", 0.001) or 0.001),
        plru_c=float(result.get("plru_c", 50.0) or 50.0),
        pan_lambda_min=float(result.get("pan_lambda_min", 0.90) or 0.90),
        pan_lambda_max=float(result.get("pan_lambda_max", 0.999) or 0.999),
        rank_matched_lambda_high=float(
            result.get("rank_matched_lambda_high", 0.999) or 0.999
        ),
        rank_matched_lambda_low=float(
            result.get("rank_matched_lambda_low", 0.0) or 0.0
        ),
    ).to(device)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, result


@torch.no_grad()
def run_state(model, sequence: torch.Tensor) -> torch.Tensor:
    state = model.init_state(sequence.shape[1], sequence.device)
    for task_input in sequence:
        state = model.step(task_input, state)
    return state


@torch.no_grad()
def clean_state_from_q(
    model, geometry: RingGeometry, q0: torch.Tensor, horizon: int
) -> torch.Tensor:
    velocity = torch.zeros(
        int(horizon),
        q0.shape[0],
        geometry.q_dim,
        device=q0.device,
        dtype=q0.dtype,
    )
    sequence, _, _, _ = sequence_from_qv(geometry, q0, velocity)
    return run_state(model, sequence)


@torch.no_grad()
def drive_ring(model, geometry: RingGeometry, state: torch.Tensor, velocity: float, steps: int):
    current = state.clone()
    velocity_tensor = torch.full(
        (current.shape[0], 1),
        float(velocity),
        device=current.device,
        dtype=current.dtype,
    )
    velocity_features = geometry.velocity_features(velocity_tensor)
    task_input = torch.zeros(
        current.shape[0],
        model.input_dim,
        device=current.device,
        dtype=current.dtype,
    )
    task_input[:, geometry.y_dim : geometry.y_dim + velocity_features.shape[-1]] = velocity_features
    for _ in range(int(steps)):
        current = model.step(task_input, current)
    return current


@torch.no_grad()
def blank_snapshots(model, state: torch.Tensor, horizons: list[int]):
    blank = torch.zeros(
        state.shape[0], model.input_dim, device=state.device, dtype=state.dtype
    )
    current = state.clone()
    previous = 0
    snapshots = {}
    for horizon in sorted(set(int(value) for value in horizons)):
        if horizon < 0:
            raise ValueError("blank horizons must be non-negative")
        for _ in range(horizon - previous):
            current = model.step(blank, current)
        snapshots[horizon] = current.clone()
        previous = horizon
    return snapshots


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE.parents[1], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@torch.no_grad()
def evaluate_checkpoint(
    model,
    spec: ModelSpec,
    seed: int,
    args,
    device: torch.device,
):
    geometry = RingGeometry()
    two_pi = 2.0 * math.pi
    base_angles = torch.linspace(
        0.0, two_pi, int(args.base_points) + 1, device=device, dtype=torch.float32
    )[:-1, None]
    grid_angles = torch.linspace(
        0.0, two_pi, int(args.reference_grid) + 1, device=device, dtype=torch.float32
    )[:-1, None]

    base_states = clean_state_from_q(model, geometry, base_angles, int(args.clean_horizon))
    clean_states = clean_state_from_q(model, geometry, grid_angles, int(args.clean_horizon))
    clean_carrier = carrier_view(model, clean_states)
    grid_step = two_pi / int(args.reference_grid)

    rows = []
    for direction in args.directions:
        for velocity_scale in args.velocity_scales:
            velocity = (
                float(direction)
                * float(args.base_velocity_deg)
                * float(velocity_scale)
                * math.pi
                / 180.0
            )
            for move_steps in args.move_lengths:
                target_angles = torch.remainder(
                    base_angles[:, 0] + int(move_steps) * velocity, two_pi
                )
                target_indices = torch.remainder(
                    torch.round(target_angles / grid_step).long(), int(args.reference_grid)
                )
                target_carrier = clean_carrier[target_indices]
                previous_carrier = clean_carrier[
                    torch.remainder(target_indices - 1, int(args.reference_grid))
                ]
                next_carrier = clean_carrier[
                    torch.remainder(target_indices + 1, int(args.reference_grid))
                ]
                tangent_norm_per_rad = (
                    (next_carrier - previous_carrier).norm(dim=-1) / (2.0 * grid_step)
                ).clamp_min(1e-8)

                driven = drive_ring(model, geometry, base_states, velocity, int(move_steps))
                snapshots = blank_snapshots(model, driven, args.blank_horizons)
                base_prediction = model.decode(base_states)
                base_decoded_angle = torch.atan2(
                    base_prediction[:, 1], base_prediction[:, 0]
                )

                for blank_horizon, state in snapshots.items():
                    carrier = carrier_view(model, state)
                    prediction = model.decode(state)
                    decoded_angle = torch.atan2(prediction[:, 1], prediction[:, 0])
                    decoded_displacement = circular_delta(
                        decoded_angle, base_decoded_angle
                    )
                    distances = torch.cdist(carrier, clean_carrier)
                    nearest_distance, nearest_index = distances.min(dim=1)
                    nearest_angle = grid_angles[nearest_index, 0]
                    target_distance = (carrier - target_carrier).norm(dim=-1)
                    nearest_error = circular_delta(nearest_angle, target_angles).abs()
                    decoded_error = circular_delta(decoded_angle, target_angles).abs()

                    for index in range(base_angles.shape[0]):
                        scale = float(tangent_norm_per_rad[index].item())
                        rows.append(
                            {
                                "paper_variant": spec.paper_variant,
                                "legacy_tag": spec.tag,
                                "seed": int(seed),
                                "base_angle_deg": math.degrees(float(base_angles[index, 0].item())),
                                "direction": int(direction),
                                "velocity_scale": float(velocity_scale),
                                "velocity_deg": math.degrees(velocity),
                                "move_steps": int(move_steps),
                                "target_displacement_deg": math.degrees(int(move_steps) * velocity),
                                "blank_horizon": int(blank_horizon),
                                "decoded_displacement_deg": math.degrees(
                                    float(decoded_displacement[index].item())
                                ),
                                "decoded_target_error_deg": math.degrees(
                                    float(decoded_error[index].item())
                                ),
                                "nearest_clean_error_deg": math.degrees(
                                    float(nearest_error[index].item())
                                ),
                                "carrier_distance_to_target": float(
                                    target_distance[index].item()
                                ),
                                "carrier_distance_to_nearest": float(
                                    nearest_distance[index].item()
                                ),
                                "local_tangent_norm_per_rad": scale,
                                "normalized_target_distance_rad": float(
                                    target_distance[index].item()
                                )
                                / scale,
                                "normalized_nearest_distance_rad": float(
                                    nearest_distance[index].item()
                                )
                                / scale,
                            }
                        )
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("no transport rows were produced")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--variants", default="linear_update,input_only_update,state_dependent_update")
    parser.add_argument("--base-points", type=int, default=32)
    parser.add_argument("--directions", default="-1,1")
    parser.add_argument("--velocity-scales", default="1,2")
    parser.add_argument("--move-lengths", default="5,20")
    parser.add_argument("--base-velocity-deg", type=float, default=3.0)
    parser.add_argument("--clean-horizon", type=int, default=260)
    parser.add_argument("--reference-grid", type=int, default=1440)
    parser.add_argument("--blank-horizons", default="0,20,100,500,1000")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.seeds = parse_csv(args.seeds, int)
    args.directions = parse_csv(args.directions, int)
    args.velocity_scales = parse_csv(args.velocity_scales, float)
    args.move_lengths = parse_csv(args.move_lengths, int)
    args.blank_horizons = parse_csv(args.blank_horizons, int)
    requested_variants = set(parse_csv(args.variants, str))
    if (
        not args.seeds
        or any(seed < 0 for seed in args.seeds)
        or len(set(args.seeds)) != len(args.seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers")
    if not requested_variants:
        raise ValueError("at least one variant is required")
    if not args.directions or set(args.directions) - {-1, 1}:
        raise ValueError("directions must contain only -1 and/or 1")
    if len(set(args.directions)) != len(args.directions):
        raise ValueError("directions must be unique")
    if not args.velocity_scales or any(value <= 0 for value in args.velocity_scales):
        raise ValueError("velocity scales must be positive")
    if not args.move_lengths or any(value < 1 for value in args.move_lengths):
        raise ValueError("move lengths must be positive")
    if not args.blank_horizons or any(value < 0 for value in args.blank_horizons):
        raise ValueError("blank horizons must be non-negative")
    if args.base_points < 1 or args.reference_grid < 3 or args.clean_horizon < 1:
        raise ValueError("base-points, clean-horizon, and reference-grid must be positive")
    if not args.base_velocity_deg > 0:
        raise ValueError("base velocity must be positive")
    specs = [spec for spec in DEFAULT_SPECS if spec.paper_variant in requested_variants]
    if requested_variants != {spec.paper_variant for spec in specs}:
        missing = sorted(requested_variants - {spec.paper_variant for spec in specs})
        raise ValueError(f"unknown variants: {missing}")

    artifact_root = args.artifact_root.expanduser().resolve(strict=True)
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"artifact root is not a directory: {artifact_root}")
    args.artifact_root = artifact_root

    records = []
    missing_paths = []
    for spec in specs:
        for seed in args.seeds:
            result, checkpoint = checkpoint_paths(args.artifact_root, spec, seed)
            if not result.exists() or not checkpoint.exists():
                missing_paths.extend(str(path) for path in (result, checkpoint) if not path.exists())
                continue
            for path in (result, checkpoint):
                resolved = path.resolve(strict=True)
                try:
                    resolved.relative_to(artifact_root)
                except ValueError as exc:
                    raise ValueError(f"input artifact escapes artifact root: {path} -> {resolved}") from exc
            records.append((spec, seed, result, checkpoint))
    if missing_paths:
        raise FileNotFoundError("missing required artifacts:\n" + "\n".join(missing_paths))

    drive_conditions = condition_count(
        args.base_points,
        args.directions,
        args.velocity_scales,
        args.move_lengths,
        specs,
        args.seeds,
    )
    expected_rows = drive_conditions * len(set(args.blank_horizons))
    print(f"checkpoints={len(records)} drive_conditions={drive_conditions} expected_rows={expected_rows}")
    if args.dry_run:
        for spec, seed, result, checkpoint in records:
            print(spec.paper_variant, seed, result, checkpoint)
        return 0

    args.output_dir = args.output_dir.expanduser().resolve()
    validate_fresh_output(args.output_dir, records)
    frozen_sources = capture_sources(records)
    frozen_by_path = {snapshot.path: snapshot for snapshot in frozen_sources}
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=False, exist_ok=False)
    atomic_write_text(
        args.output_dir / "INCOMPLETE",
        "This marker is removed only after all outputs and hashes are written.\n",
    )
    device = torch.device(args.device)
    rows = []
    sources = []
    for spec, seed, result_path, checkpoint_path in records:
        print(f"[evaluate] {spec.paper_variant} seed={seed}", flush=True)
        model, result = load_model(result_path, checkpoint_path, spec, seed, device)
        rows.extend(evaluate_checkpoint(model, spec, seed, args, device))
        sources.append(
            {
                "spec": asdict(spec),
                "seed": seed,
                "result_path": str(result_path),
                "result_sha256": frozen_by_path[result_path.resolve(strict=True)].sha256,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": frozen_by_path[
                    checkpoint_path.resolve(strict=True)
                ].sha256,
                "legacy_model": result.get("model"),
            }
        )

    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, produced {len(rows)}")
    csv_path = args.output_dir / "ring_transport_v2.csv"
    write_csv(csv_path, rows)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "code_commit": git_commit(),
        "artifact_root": str(args.artifact_root.resolve()),
        "device": str(device),
        "base_points": args.base_points,
        "directions": args.directions,
        "velocity_scales": args.velocity_scales,
        "move_lengths": args.move_lengths,
        "base_velocity_deg": args.base_velocity_deg,
        "clean_horizon": args.clean_horizon,
        "reference_grid": args.reference_grid,
        "blank_horizons": args.blank_horizons,
        "drive_conditions": drive_conditions,
        "row_count": len(rows),
        "output_csv": csv_path.name,
        "output_sha256": sha256(csv_path),
        "code_sources": [
            {
                "path": str(snapshot.path),
                "sha256": snapshot.sha256,
            }
            for snapshot in frozen_sources
            if snapshot.kind == "code"
        ],
        "sources": sources,
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    verify_sources(frozen_sources)
    finalize_output(args.output_dir)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
