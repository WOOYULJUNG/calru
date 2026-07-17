"""Stage 5: quick tangent/normal gains and finite normal-kick recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from repro.sagodi_protocol.artifacts import atomic_json, derived_seed

from .topology_analysis_common import (
    atomic_npz,
    blank_snapshots,
    decode_primary,
    discover_completed_runs,
    forward_endpoint_states,
    load_analysis_bank,
    load_analysis_config,
    load_model,
    load_success_map,
    normalized_geodesic_errors,
    write_csv,
)


def _orthonormal_tangent(raw: torch.Tensor) -> torch.Tensor:
    # raw is [A,H,d]; reduced QR preserves the intended d-dimensional span.
    return torch.linalg.qr(raw, mode="reduced").Q


def _random_normals(
    tangent: torch.Tensor, count: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device=tangent.device)
    generator.manual_seed(int(seed))
    random = torch.randn(
        tangent.shape[0], tangent.shape[1], int(count),
        device=tangent.device,
        dtype=tangent.dtype,
        generator=generator,
    )
    projected = random - tangent @ (tangent.transpose(-2, -1) @ random)
    return torch.linalg.qr(projected, mode="reduced").Q


@torch.no_grad()
def _transition(model, primary: torch.Tensor) -> torch.Tensor:
    reported = model.reported_from_primary(primary)
    blank = torch.zeros(
        primary.shape[0], model.input_dim, device=primary.device, dtype=primary.dtype
    )
    return model.primary_from_reported(model.step(blank, reported))


@torch.no_grad()
def _central_jvp(
    model, state: torch.Tensor, vectors: torch.Tensor, epsilon: float
) -> torch.Tensor:
    anchors, hidden, directions = vectors.shape
    base = state[:, None, :].expand(anchors, directions, hidden)
    direction = vectors.transpose(1, 2)
    plus = (base + float(epsilon) * direction).reshape(anchors * directions, hidden)
    minus = (base - float(epsilon) * direction).reshape(anchors * directions, hidden)
    following_plus = _transition(model, plus).reshape(anchors, directions, hidden)
    following_minus = _transition(model, minus).reshape(anchors, directions, hidden)
    return ((following_plus - following_minus) / (2.0 * float(epsilon))).transpose(1, 2)


@torch.no_grad()
def _finite_amplification(
    model,
    state: torch.Tensor,
    vectors: torch.Tensor,
    epsilon: float,
    horizons: list[int],
) -> np.ndarray:
    anchors, hidden, directions = vectors.shape
    perturbed = (
        state[:, None, :] + float(epsilon) * vectors.transpose(1, 2)
    ).reshape(anchors * directions, hidden)
    requested = sorted(set([0, *horizons]))
    clean = blank_snapshots(model, state, requested)
    kicked = blank_snapshots(model, perturbed, requested)
    values = []
    for horizon in horizons:
        clean_repeated = clean[horizon][:, None, :].expand(anchors, directions, hidden)
        kicked_state = kicked[horizon].reshape(anchors, directions, hidden)
        gain = torch.linalg.vector_norm(kicked_state - clean_repeated, dim=-1) / float(epsilon)
        values.append(gain.cpu().numpy())
    return np.stack(values)


@torch.no_grad()
def analyze_record(record, config, bank_root: Path, analysis_root: Path, device: torch.device):
    dynamics = config["quick_dynamics"]
    bank = load_analysis_bank(bank_root, record.topology)
    geometry_path = analysis_root / "geometry" / "runs" / f"{record.job_id}.npz"
    with np.load(geometry_path, allow_pickle=False) as geometry:
        geometry_indices = np.array(geometry["anchor_index"], copy=True)
        geometry_state = np.array(geometry["endpoint_state"], copy=True)
        geometry_tangent = np.array(geometry["tangent_basis_raw"], copy=True)
    count = int(dynamics["anchors"])
    selection = np.linspace(0, len(geometry_indices) - 1, count, dtype=np.int64)
    indices = geometry_indices[selection]
    model, _ = load_model(record, device)
    state = torch.as_tensor(geometry_state[selection], device=device)
    raw_tangent = torch.as_tensor(geometry_tangent[selection], device=device)
    tangent = _orthonormal_tangent(raw_tangent)
    normal = _random_normals(
        tangent,
        int(dynamics["random_normal_directions"]),
        derived_seed(int(dynamics["seed"]), record.job_id, "normal"),
    )
    atlas_state = torch.as_tensor(geometry_state, device=device)
    centered = atlas_state - atlas_state.mean(dim=0, keepdim=True)
    state_scale = float(torch.linalg.vector_norm(centered, dim=-1).median().cpu())
    epsilon = float(dynamics["one_step_state_epsilon_relative"]) * state_scale
    clean_next = _transition(model, state)
    nearest_next = torch.cdist(clean_next, atlas_state).argmin(dim=1)
    tangent_next = _orthonormal_tangent(
        torch.as_tensor(geometry_tangent, device=device)[nearest_next]
    )
    jt = _central_jvp(model, state, tangent, epsilon)
    jn = _central_jvp(model, state, normal, epsilon)
    tangent_coordinates = tangent_next.transpose(-2, -1) @ jt
    tangent_singular = torch.linalg.svdvals(tangent_coordinates)
    tangent_residual = jt - tangent_next @ tangent_coordinates
    normal_tangent_coordinates = tangent_next.transpose(-2, -1) @ jn
    normal_residual = jn - tangent_next @ normal_tangent_coordinates
    jt_norm = torch.linalg.vector_norm(jt, dim=1).clamp_min(1e-8)
    jn_norm = torch.linalg.vector_norm(jn, dim=1).clamp_min(1e-8)
    tangent_to_normal = torch.linalg.vector_norm(tangent_residual, dim=1) / jt_norm
    normal_to_tangent = torch.linalg.vector_norm(normal_tangent_coordinates, dim=1) / jn_norm
    sampled_normal_gain = jn_norm
    sampled_normal_residual_gain = torch.linalg.vector_norm(normal_residual, dim=1)

    finite_horizons = [int(value) for value in dynamics["finite_horizons"]]
    tangent_finite = _finite_amplification(
        model, state, tangent, epsilon, finite_horizons
    )
    normal_finite = _finite_amplification(
        model, state, normal, epsilon, finite_horizons
    )

    # Full 1024-point sheet for nearest-manifold recovery distances.
    full_initial = torch.as_tensor(bank["initializer_memory"], device=device)
    full_inputs = torch.as_tensor(bank["transport_inputs"], device=device)
    full_state, _ = forward_endpoint_states(model, full_inputs, full_initial, chunk_size=128)
    recovery_horizons = [int(value) for value in dynamics["recovery_horizons"]]
    manifold_snapshots = blank_snapshots(model, full_state, recovery_horizons)
    clean_snapshots = blank_snapshots(model, state, recovery_horizons)
    radii = [float(value) for value in dynamics["kick_radii_relative"]]
    recovery_ratio = np.empty(
        (len(radii), len(recovery_horizons), count, normal.shape[-1]), dtype=np.float32
    )
    same_memory = np.empty_like(recovery_ratio)
    for radius_index, relative_radius in enumerate(radii):
        radius = relative_radius * state_scale
        kicked_initial = (
            state[:, None, :] + radius * normal.transpose(1, 2)
        ).reshape(count * normal.shape[-1], state.shape[-1])
        kicked_snapshots = blank_snapshots(model, kicked_initial, recovery_horizons)
        denominator = torch.cdist(kicked_initial, manifold_snapshots[0]).min(dim=1).values
        denominator = denominator.reshape(count, normal.shape[-1]).clamp_min(1e-8)
        for horizon_index, horizon in enumerate(recovery_horizons):
            kicked = kicked_snapshots[horizon]
            distance = torch.cdist(kicked, manifold_snapshots[horizon]).min(dim=1).values
            recovery_ratio[radius_index, horizon_index] = (
                distance.reshape(count, normal.shape[-1]) / denominator
            ).cpu().numpy()
            kicked_prediction = decode_primary(model, kicked).reshape(
                count, normal.shape[-1], -1
            )
            clean_prediction = decode_primary(model, clean_snapshots[horizon])
            clean_prediction = clean_prediction[:, None, :].expand_as(kicked_prediction)
            memory_error, _ = normalized_geodesic_errors(
                record.topology, kicked_prediction, clean_prediction
            )
            same_memory[radius_index, horizon_index] = memory_error.cpu().numpy()

    tangent_finite_mean = tangent_finite.mean(axis=(1, 2))
    normal_finite_mean = normal_finite.mean(axis=(1, 2))
    metrics = {
        "schema_version": 1,
        "job_id": record.job_id,
        "model": record.model_id,
        "topology": record.topology,
        "seed": record.seed,
        "anchors": count,
        "state_scale": state_scale,
        "one_step_tangent_gain_mean": float(tangent_singular.mean().cpu()),
        "one_step_tangent_gain_min_median": float(tangent_singular[:, -1].median().cpu()),
        "one_step_tangent_anisotropy_median": float(
            (tangent_singular[:, 0] / tangent_singular[:, -1].clamp_min(1e-8)).median().cpu()
        ),
        "one_step_tangent_to_normal_leakage_mean": float(tangent_to_normal.mean().cpu()),
        "one_step_normal_to_tangent_leakage_mean": float(normal_to_tangent.mean().cpu()),
        "one_step_sampled_normal_gain_mean": float(sampled_normal_gain.mean().cpu()),
        "one_step_sampled_normal_residual_gain_mean": float(
            sampled_normal_residual_gain.mean().cpu()
        ),
        "finite_horizons": finite_horizons,
        "finite_tangent_gain_mean": tangent_finite_mean.tolist(),
        "finite_normal_gain_mean": normal_finite_mean.tolist(),
        "finite_normal_over_tangent_ratio": (
            normal_finite_mean / np.maximum(tangent_finite_mean, 1e-8)
        ).tolist(),
        "kick_radii_relative": radii,
        "recovery_horizons": recovery_horizons,
        "recovery_ratio_median": np.median(recovery_ratio, axis=(2, 3)).tolist(),
        "same_memory_error_mean": same_memory.mean(axis=(2, 3)).tolist(),
        "all_finite": bool(
            np.isfinite(recovery_ratio).all()
            and np.isfinite(same_memory).all()
            and np.isfinite(tangent_finite).all()
            and np.isfinite(normal_finite).all()
        ),
    }
    output = analysis_root / "dynamics" / "runs"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f"{record.job_id}.json", metrics)
    atomic_npz(
        output / f"{record.job_id}.npz",
        anchor_index=indices,
        tangent_singular_values=tangent_singular.cpu().numpy().astype(np.float32),
        tangent_to_normal_leakage=tangent_to_normal.cpu().numpy().astype(np.float32),
        normal_to_tangent_leakage=normal_to_tangent.cpu().numpy().astype(np.float32),
        sampled_normal_gain=sampled_normal_gain.cpu().numpy().astype(np.float32),
        sampled_normal_residual_gain=sampled_normal_residual_gain.cpu().numpy().astype(np.float32),
        finite_horizons=np.asarray(finite_horizons, dtype=np.int64),
        finite_tangent_gain=tangent_finite.astype(np.float32),
        finite_normal_gain=normal_finite.astype(np.float32),
        kick_radii_relative=np.asarray(radii, dtype=np.float32),
        recovery_horizons=np.asarray(recovery_horizons, dtype=np.int64),
        recovery_ratio=recovery_ratio,
        same_memory_error=same_memory,
    )
    return metrics


def aggregate(analysis_root: Path) -> None:
    rows = []
    arrays = {}
    for path in sorted((analysis_root / "dynamics" / "runs").glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                key: value
                for key, value in item.items()
                if not isinstance(value, list) and key != "schema_version"
            }
        )
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
            for name in archive.files:
                arrays[f"{item['job_id']}__{name}"] = np.array(archive[name], copy=True)
    write_csv(analysis_root / "dynamics" / "dynamics_metrics.csv", rows)
    atomic_npz(analysis_root / "dynamics" / "dynamics_metrics.npz", **arrays)
    atomic_npz(analysis_root / "dynamics_metrics.npz", **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("topology_analysis_v1.json")
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_analysis_config(args.config.expanduser().resolve(strict=True))
    analysis_root = args.analysis_root.expanduser().resolve(strict=True)
    if args.aggregate_only:
        aggregate(analysis_root)
        return
    success = load_success_map(analysis_root)
    records, missing = discover_completed_runs(args.run_root.expanduser().resolve(strict=True), config)
    if missing:
        raise RuntimeError("dynamics analysis requires all completed runs")
    if not bool(config["execution"].get("analyze_structure_for_all_runs", False)):
        records = [record for record in records if success.get(record.job_id, False)]
    if args.job_id:
        records = [record for record in records if record.job_id == args.job_id]
        if not records:
            raise ValueError("job is missing or excluded by the analysis contract")
    for record in records:
        analyze_record(
            record,
            config,
            args.bank_root.expanduser().resolve(strict=True),
            analysis_root,
            torch.device(args.device),
        )
    if not args.job_id:
        aggregate(analysis_root)


if __name__ == "__main__":
    main()
