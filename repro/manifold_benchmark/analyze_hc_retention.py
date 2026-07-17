"""Stage 6 (H-C): final base/dynamic retention and normal-kick response."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from repro.sagodi_protocol.artifacts import atomic_json, derived_seed, strict_json_load

from .analyze_tangent_normal import _orthonormal_tangent, _random_normals
from .topology_analysis_common import (
    atomic_npz,
    blank_snapshots,
    build_record_model,
    discover_completed_runs,
    load_analysis_config,
    load_model,
    load_success_map,
    test_batch,
    write_csv,
)


def _base_retention(model) -> torch.Tensor:
    recurrence = model.core.blocks[0].rec
    return recurrence.lam_mag()


def _timescale(value: torch.Tensor) -> torch.Tensor:
    return -1.0 / torch.log(value.clamp(min=1e-8, max=1.0 - 1e-8))


@torch.no_grad()
def analyze_record(record, config, analysis_root: Path, device: torch.device):
    retention = config["hc_retention"]
    final_model, _ = load_model(record, device)
    initial_model = build_record_model(record, device)
    initial_model.eval()
    initial_base = _base_retention(initial_model)
    final_base = _base_retention(final_model)
    batch = test_batch(
        record.topology,
        device=device,
        trajectories=int(retention["task_trajectories"]),
    )
    _, history = final_model.forward_sequence(
        batch.inputs, initial_memory=batch.initial_memory, return_states=True
    )
    primary = final_model.primary_from_reported(history)
    task_lambda = final_model.dynamic_lambda(history)
    endpoint = primary[-1]
    blank_horizons = [int(value) for value in retention["blank_horizons"]]
    blank_states = blank_snapshots(final_model, endpoint, blank_horizons)
    blank_lambda = torch.stack(
        [
            final_model.dynamic_lambda(final_model.reported_from_primary(blank_states[horizon]))
            for horizon in blank_horizons
        ]
    )
    energy = primary.square()
    weighted_coordinate = (task_lambda * energy).sum(dim=(0, 1)) / energy.sum(
        dim=(0, 1)
    ).clamp_min(1e-8)

    geometry_path = analysis_root / "geometry" / "runs" / f"{record.job_id}.npz"
    with np.load(geometry_path, allow_pickle=False) as archive:
        states = torch.as_tensor(np.array(archive["endpoint_state"], copy=True), device=device)
        raw_tangent = torch.as_tensor(
            np.array(archive["tangent_basis_raw"], copy=True), device=device
        )
    count = int(config["quick_dynamics"]["anchors"])
    selection = np.linspace(0, states.shape[0] - 1, count, dtype=np.int64)
    anchor_state = states[selection]
    tangent = _orthonormal_tangent(raw_tangent[selection])
    normals = _random_normals(
        tangent,
        int(config["quick_dynamics"]["random_normal_directions"]),
        derived_seed(int(config["quick_dynamics"]["seed"]), record.job_id, "normal"),
    )
    state_scale = torch.linalg.vector_norm(
        states - states.mean(dim=0, keepdim=True), dim=-1
    ).median()
    radius = float(config["quick_dynamics"]["kick_radii_relative"][0]) * state_scale
    clean_lambda = final_model.dynamic_lambda(
        final_model.reported_from_primary(anchor_state)
    )
    kicked_state = (
        anchor_state[:, None, :] + radius * normals.transpose(1, 2)
    ).reshape(count * normals.shape[-1], -1)
    kicked_lambda = final_model.dynamic_lambda(
        final_model.reported_from_primary(kicked_state)
    ).reshape(count, normals.shape[-1], -1)
    lambda_response = kicked_lambda - clean_lambda[:, None, :]

    trace = strict_json_load(record.run_dir / "trace.json")
    rp_trace = trace.get("rp_trace", [])
    near = float(retention["near_one_threshold"])
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "job_id": record.job_id,
        "model": record.model_id,
        "topology": record.topology,
        "seed": record.seed,
        "initial_base_lambda_median": float(initial_base.median().cpu()),
        "initial_base_lambda_max": float(initial_base.max().cpu()),
        "final_base_lambda_median": float(final_base.median().cpu()),
        "final_base_lambda_max": float(final_base.max().cpu()),
        "final_base_near_one_count": int((final_base >= near).sum().cpu()),
        "final_base_timescale_median": float(_timescale(final_base).median().cpu()),
        "final_base_timescale_max": float(_timescale(final_base).max().cpu()),
        "base_lambda_change_mean": float((final_base - initial_base).mean().cpu()),
        "task_dynamic_lambda_min": float(task_lambda.min().cpu()),
        "task_dynamic_lambda_median": float(task_lambda.median().cpu()),
        "task_dynamic_lambda_max": float(task_lambda.max().cpu()),
        "task_dynamic_lambda_time_std_mean": float(
            task_lambda.mean(dim=1).std(dim=0).mean().cpu()
        ),
        "task_state_energy_weighted_lambda_mean": float(weighted_coordinate.mean().cpu()),
        "blank_dynamic_lambda_min": float(blank_lambda.min().cpu()),
        "blank_dynamic_lambda_median": float(blank_lambda.median().cpu()),
        "blank_dynamic_lambda_max": float(blank_lambda.max().cpu()),
        "normal_kick_lambda_response_signed_mean": float(lambda_response.mean().cpu()),
        "normal_kick_lambda_response_absolute_mean": float(
            lambda_response.abs().mean().cpu()
        ),
        "normal_kick_lambda_response_max_absolute": float(
            lambda_response.abs().max().cpu()
        ),
        "rp_call_count": len(rp_trace),
        "rp_last_summary": rp_trace[-1] if rp_trace else None,
        "rp_snapshot_limitation": retention["rp_snapshot_limitation"],
        "closed_loop_vs_frozen_schedule_in_primary": False,
        "all_finite": bool(
            torch.isfinite(task_lambda).all()
            and torch.isfinite(blank_lambda).all()
            and torch.isfinite(lambda_response).all()
        ),
    }
    output = analysis_root / "hc_retention" / "runs"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f"{record.job_id}.json", metrics)
    atomic_npz(
        output / f"{record.job_id}.npz",
        initial_base_lambda=initial_base.cpu().numpy().astype(np.float32),
        final_base_lambda=final_base.cpu().numpy().astype(np.float32),
        final_base_timescale=_timescale(final_base).cpu().numpy().astype(np.float32),
        task_dynamic_lambda=task_lambda.cpu().numpy().astype(np.float32),
        task_state_energy_weighted_lambda=weighted_coordinate.cpu().numpy().astype(np.float32),
        blank_horizons=np.asarray(blank_horizons, dtype=np.int64),
        blank_dynamic_lambda=blank_lambda.cpu().numpy().astype(np.float32),
        normal_kick_lambda_response=lambda_response.cpu().numpy().astype(np.float32),
    )
    return metrics


def aggregate(analysis_root: Path) -> None:
    rows = []
    arrays = {}
    for path in sorted((analysis_root / "hc_retention" / "runs").glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"schema_version", "rp_last_summary"}
            }
        )
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
            for name in archive.files:
                arrays[f"{item['job_id']}__{name}"] = np.array(archive[name], copy=True)
    write_csv(analysis_root / "hc_retention" / "hc_retention_metrics.csv", rows)
    atomic_npz(analysis_root / "hc_retention" / "hc_retention_metrics.npz", **arrays)
    atomic_npz(analysis_root / "hc_retention_metrics.npz", **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
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
        raise RuntimeError("H-C retention analysis requires all completed runs")
    retention_models = set(config["hc_retention"].get("models", ["hc"]))
    records = [record for record in records if record.model_id in retention_models]
    if not bool(config["execution"].get("analyze_structure_for_all_runs", False)):
        records = [record for record in records if success.get(record.job_id, False)]
    if args.job_id:
        records = [record for record in records if record.job_id == args.job_id]
        if not records:
            raise ValueError("job is missing or excluded by the retention contract")
    for record in records:
        analyze_record(record, config, analysis_root, torch.device(args.device))
    if not args.job_id:
        aggregate(analysis_root)


if __name__ == "__main__":
    main()
