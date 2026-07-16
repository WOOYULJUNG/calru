"""Stage 3: long blank-input memory curves for every completed run."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from repro.sagodi_protocol.artifacts import atomic_json

from .topology_analysis_common import (
    atomic_npz,
    blank_snapshots,
    decode_primary,
    discover_completed_runs,
    load_analysis_config,
    load_model,
    load_success_map,
    normalized_geodesic_errors,
    output_norm_error,
    test_batch,
    write_csv,
)


def _finite_scalar_or_none(value: torch.Tensor) -> float | None:
    """Return a JSON-safe scalar while preserving divergence as null."""

    scalar = float(value.detach().cpu())
    return scalar if math.isfinite(scalar) else None


@torch.no_grad()
def analyze_record(record, config, analysis_root: Path, device: torch.device) -> dict[str, Any]:
    blank = config["blank_memory"]
    model, _ = load_model(record, device)
    batch = test_batch(
        record.topology,
        device=device,
        trajectories=int(blank["test_trajectories"]),
    )
    _, history = model.forward_sequence(
        batch.inputs, initial_memory=batch.initial_memory, return_states=True
    )
    endpoint_reported = history[-1]
    endpoint = model.primary_from_reported(endpoint_reported)
    horizons = [int(value) for value in blank["horizons"]]
    snapshots = blank_snapshots(model, endpoint, horizons)
    target = batch.output_targets[-1]
    clean_prediction = model.decode(endpoint_reported)
    clean_state_norm = torch.linalg.vector_norm(snapshots[0], dim=-1)
    rows: list[dict[str, Any]] = []
    geodesic_arrays: list[np.ndarray] = []
    displacement_arrays: list[np.ndarray] = []
    state_norm_arrays: list[np.ndarray] = []
    output_norm_arrays: list[np.ndarray] = []
    for horizon in horizons:
        state = snapshots[horizon]
        prediction = clean_prediction if horizon == 0 else decode_primary(model, state)
        error, _ = normalized_geodesic_errors(record.topology, prediction, target)
        displacement, _ = normalized_geodesic_errors(
            record.topology, prediction, clean_prediction
        )
        norms = torch.linalg.vector_norm(state, dim=-1)
        norm_error = output_norm_error(record.topology, prediction)
        finite = all(
            bool(torch.isfinite(value).all())
            for value in (state, prediction, error, displacement, norms, norm_error)
        )
        rows.append(
            {
                "job_id": record.job_id,
                "model": record.model_id,
                "topology": record.topology,
                "seed": record.seed,
                "horizon": horizon,
                "memory_error_mean": _finite_scalar_or_none(error.mean()),
                "memory_error_median": _finite_scalar_or_none(error.median()),
                "clean_output_displacement_mean": _finite_scalar_or_none(
                    displacement.mean()
                ),
                "output_norm_error_mean": _finite_scalar_or_none(norm_error.mean()),
                "causal_state_norm_mean": _finite_scalar_or_none(norms.mean()),
                "causal_state_norm_ratio_to_h0": _finite_scalar_or_none(
                    (norms / clean_state_norm.clamp_min(1e-8)).mean()
                ),
                "finite": finite,
                "diverged": not finite,
            }
        )
        geodesic_arrays.append(error.cpu().numpy().astype(np.float32))
        displacement_arrays.append(displacement.cpu().numpy().astype(np.float32))
        state_norm_arrays.append(norms.cpu().numpy().astype(np.float32))
        output_norm_arrays.append(norm_error.cpu().numpy().astype(np.float32))
    run_root = analysis_root / "blank" / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        run_root / f"{record.job_id}.npz",
        horizons=np.asarray(horizons, dtype=np.int64),
        memory_error=np.stack(geodesic_arrays),
        clean_output_displacement=np.stack(displacement_arrays),
        causal_state_norm=np.stack(state_norm_arrays),
        output_norm_error=np.stack(output_norm_arrays),
    )
    payload = {
        "schema_version": 1,
        "job_id": record.job_id,
        "nonfinite_json_policy": "null_metrics_with_raw_arrays_preserved_in_npz",
        "rows": rows,
    }
    atomic_json(run_root / f"{record.job_id}.json", payload)
    return payload


def aggregate(config, analysis_root: Path) -> None:
    success = load_success_map(analysis_root)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for path in sorted((analysis_root / "blank" / "runs").glob("*.json")):
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            row["task_success"] = success.get(row["job_id"], False)
            rows.append(row)
        archive_path = path.with_suffix(".npz")
        with np.load(archive_path, allow_pickle=False) as archive:
            for name in archive.files:
                arrays[f"{payload['job_id']}__{name}"] = np.array(archive[name], copy=True)
    write_csv(analysis_root / "blank" / "blank_metrics.csv", rows)
    atomic_npz(analysis_root / "blank" / "blank_metrics.npz", **arrays)
    atomic_npz(analysis_root / "blank_metrics.npz", **arrays)


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
        aggregate(config, analysis_root)
        return
    records, missing = discover_completed_runs(
        args.run_root.expanduser().resolve(strict=True), config
    )
    if missing:
        raise RuntimeError("blank analysis requires all completed runs")
    if args.job_id:
        records = [record for record in records if record.job_id == args.job_id]
        if not records:
            raise ValueError(f"unknown completed job {args.job_id}")
    for record in records:
        analyze_record(record, config, analysis_root, torch.device(args.device))
    if not args.job_id:
        aggregate(config, analysis_root)


if __name__ == "__main__":
    main()
