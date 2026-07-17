"""Topology-defined radial-normal kick recovery in hidden state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from repro.sagodi_protocol.artifacts import atomic_json

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
    return torch.linalg.qr(raw, mode="reduced").Q


def _output_radial_normals(topology: str, output: torch.Tensor) -> torch.Tensor:
    if topology == "s1":
        unit = output / torch.linalg.vector_norm(output, dim=-1, keepdim=True).clamp_min(1e-8)
        return unit.unsqueeze(-1)
    if topology == "t2":
        first = output[:, :2]
        second = output[:, 2:4]
        first = first / torch.linalg.vector_norm(first, dim=-1, keepdim=True).clamp_min(1e-8)
        second = second / torch.linalg.vector_norm(second, dim=-1, keepdim=True).clamp_min(1e-8)
        zeros = torch.zeros_like(first)
        return torch.stack(
            (torch.cat((first, zeros), dim=-1), torch.cat((zeros, second), dim=-1)),
            dim=-1,
        )
    if topology == "s2":
        unit = output / torch.linalg.vector_norm(output, dim=-1, keepdim=True).clamp_min(1e-8)
        return unit.unsqueeze(-1)
    raise ValueError(topology)


def _topology_radii(topology: str, output: torch.Tensor) -> torch.Tensor:
    if topology == "s1":
        return torch.linalg.vector_norm(output, dim=-1, keepdim=True)
    if topology == "t2":
        return torch.stack(
            (
                torch.linalg.vector_norm(output[..., :2], dim=-1),
                torch.linalg.vector_norm(output[..., 2:4], dim=-1),
            ),
            dim=-1,
        )
    if topology == "s2":
        return torch.linalg.vector_norm(output, dim=-1, keepdim=True)
    raise ValueError(topology)


def _decoder_vjp_radials(
    model,
    state: torch.Tensor,
    output_normals: torch.Tensor,
) -> torch.Tensor:
    vectors = []
    for index in range(output_normals.shape[-1]):
        leaf = state.detach().clone().requires_grad_(True)
        output = decode_primary(model, leaf)
        scalar = torch.sum(output * output_normals[..., index])
        gradient = torch.autograd.grad(scalar, leaf, create_graph=False)[0]
        vectors.append(gradient.detach())
    return torch.stack(vectors, dim=-1)


def _project_normal(raw: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    projected = raw - tangent @ (tangent.transpose(-2, -1) @ raw)
    norm = torch.linalg.vector_norm(projected, dim=1, keepdim=True)
    if bool(torch.any(norm < 1e-8)):
        raise RuntimeError("decoder radial direction vanished after tangent projection")
    # Reduced QR orthogonalizes the two torus radial directions.
    return torch.linalg.qr(projected, mode="reduced").Q


def analyze_record(
    record,
    config,
    bank_root: Path,
    analysis_root: Path,
    device: torch.device,
) -> dict:
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
    model.eval()
    state = torch.as_tensor(geometry_state[selection], device=device)
    tangent = _orthonormal_tangent(
        torch.as_tensor(geometry_tangent[selection], device=device)
    )
    with torch.no_grad():
        clean_output_zero = decode_primary(model, state)
        output_normals = _output_radial_normals(record.topology, clean_output_zero)
    raw_radial = _decoder_vjp_radials(model, state, output_normals)
    radial = _project_normal(raw_radial, tangent)
    # Test both outward and inward sides of the learned manifold.
    signed_radial = torch.cat((radial, -radial), dim=-1)

    atlas_state = torch.as_tensor(geometry_state, device=device)
    centered = atlas_state - atlas_state.mean(dim=0, keepdim=True)
    state_scale = float(torch.linalg.vector_norm(centered, dim=-1).median().cpu())
    radius_relative = 0.05
    radius = radius_relative * state_scale
    horizons = [int(value) for value in dynamics["recovery_horizons"]]

    with torch.no_grad():
        full_initial = torch.as_tensor(bank["initializer_memory"], device=device)
        full_inputs = torch.as_tensor(bank["transport_inputs"], device=device)
        full_state, _ = forward_endpoint_states(
            model, full_inputs, full_initial, chunk_size=128
        )
        manifold_snapshots = blank_snapshots(model, full_state, horizons)
        clean_snapshots = blank_snapshots(model, state, horizons)
        directions = signed_radial.shape[-1]
        kicked_initial = (
            state[:, None, :]
            + radius * signed_radial.transpose(1, 2)
        ).reshape(count * directions, state.shape[-1])
        kicked_snapshots = blank_snapshots(model, kicked_initial, horizons)

        hidden_denominator = torch.cdist(
            kicked_initial, manifold_snapshots[0]
        ).min(dim=1).values.reshape(count, directions).clamp_min(1e-8)
        kicked_output_zero = decode_primary(model, kicked_snapshots[0]).reshape(
            count, directions, -1
        )
        clean_radii_zero = _topology_radii(
            record.topology, decode_primary(model, clean_snapshots[0])
        )
        kicked_radii_zero = _topology_radii(record.topology, kicked_output_zero)
        output_denominator = torch.linalg.vector_norm(
            kicked_radii_zero - clean_radii_zero[:, None, :],
            dim=-1,
        ).clamp_min(1e-8)

        hidden_ratio = np.empty((len(horizons), count, directions), dtype=np.float32)
        output_ratio = np.empty_like(hidden_ratio)
        same_memory = np.empty_like(hidden_ratio)
        for horizon_index, horizon in enumerate(horizons):
            kicked = kicked_snapshots[horizon]
            hidden_distance = torch.cdist(
                kicked, manifold_snapshots[horizon]
            ).min(dim=1).values.reshape(count, directions)
            hidden_ratio[horizon_index] = (
                hidden_distance / hidden_denominator
            ).cpu().numpy()

            kicked_output = decode_primary(model, kicked).reshape(
                count, directions, -1
            )
            clean_output = decode_primary(model, clean_snapshots[horizon])
            clean_radii = _topology_radii(record.topology, clean_output)
            kicked_radii = _topology_radii(record.topology, kicked_output)
            radial_offset = torch.linalg.vector_norm(
                kicked_radii - clean_radii[:, None, :],
                dim=-1,
            )
            output_ratio[horizon_index] = (
                radial_offset / output_denominator
            ).cpu().numpy()

            repeated_clean = clean_output[:, None, :].expand_as(kicked_output)
            error, _ = normalized_geodesic_errors(
                record.topology, kicked_output, repeated_clean
            )
            same_memory[horizon_index] = error.cpu().numpy()

    index512 = horizons.index(512)
    metrics = {
        "schema_version": 1,
        "job_id": record.job_id,
        "model": record.model_id,
        "topology": record.topology,
        "seed": record.seed,
        "anchors": count,
        "topology_radial_dimensions": int(radial.shape[-1]),
        "signed_kick_directions": int(signed_radial.shape[-1]),
        "kick_radius_relative": radius_relative,
        "state_scale": state_scale,
        "recovery_horizons": horizons,
        "hidden_recovery_q_median": np.median(hidden_ratio, axis=(1, 2)).tolist(),
        "output_radial_recovery_q_median": np.median(
            output_ratio, axis=(1, 2)
        ).tolist(),
        "same_memory_error_mean": same_memory.mean(axis=(1, 2)).tolist(),
        "hidden_recovery_q512_median": float(
            np.median(hidden_ratio[index512])
        ),
        "output_radial_recovery_q512_median": float(
            np.median(output_ratio[index512])
        ),
        "same_memory_error512_mean": float(same_memory[index512].mean()),
        "all_finite": bool(
            np.isfinite(hidden_ratio).all()
            and np.isfinite(output_ratio).all()
            and np.isfinite(same_memory).all()
        ),
    }
    output_root = analysis_root / "topology_normal" / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / f"{record.job_id}.json", metrics)
    atomic_npz(
        output_root / f"{record.job_id}.npz",
        anchor_index=indices,
        radial_normal_basis=radial.cpu().numpy().astype(np.float32),
        recovery_horizons=np.asarray(horizons, dtype=np.int64),
        hidden_recovery_ratio=hidden_ratio,
        output_radial_recovery_ratio=output_ratio,
        same_memory_error=same_memory,
    )
    return metrics


def aggregate(analysis_root: Path) -> None:
    rows = []
    arrays = {}
    for path in sorted((analysis_root / "topology_normal" / "runs").glob("*.json")):
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
                arrays[f"{item['job_id']}__{name}"] = np.array(
                    archive[name], copy=True
                )
    output = analysis_root / "topology_normal"
    write_csv(output / "topology_normal_metrics.csv", rows)
    atomic_npz(output / "topology_normal_metrics.npz", **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name(
            "topology_hc_attractor_selection_v1.json"
        ),
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    config = load_analysis_config(args.config.expanduser().resolve(strict=True))
    analysis_root = args.analysis_root.expanduser().resolve(strict=True)
    if args.aggregate_only:
        aggregate(analysis_root)
        return
    success = load_success_map(analysis_root)
    records, missing = discover_completed_runs(
        args.run_root.expanduser().resolve(strict=True), config
    )
    if missing:
        raise RuntimeError("topology-normal analysis requires all completed runs")
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
