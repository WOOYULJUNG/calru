"""Stage 4: transported endpoint geometry and closed-path fiber diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from repro.sagodi_protocol.artifacts import atomic_json

from .topology_analysis_common import (
    atomic_npz,
    discover_completed_runs,
    forward_endpoint_states,
    load_analysis_bank,
    load_analysis_config,
    load_model,
    load_success_map,
    normalized_geodesic_errors,
    write_csv,
)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    first = _rankdata(left)
    second = _rankdata(right)
    first -= first.mean()
    second -= second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denominator) if denominator > 0 else float("nan")


def _pairwise_latent(topology: str, latent: np.ndarray) -> np.ndarray:
    if topology in {"s1", "t2"}:
        delta = latent[:, None, :] - latent[None, :, :]
        delta = np.remainder(delta + math.pi, 2.0 * math.pi) - math.pi
        if topology == "s1":
            return np.abs(delta[..., 0])
        return np.sqrt(np.mean(delta**2, axis=-1))
    cosine = np.clip(latent @ latent.T, -1.0, 1.0)
    return np.arccos(cosine)


def _pca_metrics(states: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    centered = states - states.mean(axis=0, keepdims=True)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2 / max(1, states.shape[0] - 1)
    fraction = variance / max(np.finfo(np.float64).eps, variance.sum())
    participation = float(variance.sum() ** 2 / np.sum(variance**2))
    score = centered @ right[:3].T
    return (
        {
            "participation_ratio": participation,
            "explained_variance_top3": fraction[:3].tolist(),
            "explained_variance_top3_sum": float(fraction[:3].sum()),
        },
        score,
        fraction,
    )


@torch.no_grad()
def analyze_record(record, config, bank_root: Path, analysis_root: Path, device: torch.device):
    geometry = config["quick_geometry"]
    bank = load_analysis_bank(bank_root, record.topology)
    total = bank["initializer_memory"].shape[0]
    count = int(geometry["atlas_points"])
    indices = np.linspace(0, total - 1, count, dtype=np.int64)
    model, _ = load_model(record, device)
    initial = torch.as_tensor(bank["initializer_memory"][indices], device=device)
    inputs = torch.as_tensor(bank["transport_inputs"][:, indices], device=device)
    endpoint_target = torch.as_tensor(
        bank["transport_endpoint_target"][indices], device=device
    )
    endpoint_latent = bank["transport_endpoint_latent"][indices].astype(np.float64)
    endpoint_state, endpoint_prediction = forward_endpoint_states(
        model, inputs, initial, chunk_size=128
    )
    initializer_state = model.primary_from_reported(model.initialize(initial))
    error, _ = normalized_geodesic_errors(
        record.topology, endpoint_prediction, endpoint_target
    )

    epsilon = float(geometry["tangent_finite_difference"])
    tangent_columns: list[torch.Tensor] = []
    for dimension in range(bank["tangent_plus_initial_memory"].shape[0]):
        plus = torch.as_tensor(
            bank["tangent_plus_initial_memory"][dimension, indices], device=device
        )
        minus = torch.as_tensor(
            bank["tangent_minus_initial_memory"][dimension, indices], device=device
        )
        plus_state, _ = forward_endpoint_states(model, inputs, plus, chunk_size=128)
        minus_state, _ = forward_endpoint_states(model, inputs, minus, chunk_size=128)
        tangent_columns.append((plus_state - minus_state) / (2.0 * epsilon))
    tangent = torch.stack(tangent_columns, dim=-1).cpu().numpy().astype(np.float64)
    singular = np.linalg.svd(tangent, compute_uv=False)
    threshold = float(geometry["rank_relative_threshold"])
    effective_rank = (singular > threshold * singular[:, :1]).sum(axis=-1)
    sigma_min = singular[:, -1]
    condition = singular[:, 0] / np.maximum(sigma_min, np.finfo(np.float64).eps)

    state_np = endpoint_state.cpu().numpy().astype(np.float64)
    initializer_np = initializer_state.cpu().numpy().astype(np.float64)
    endpoint_pca, endpoint_score, endpoint_fraction = _pca_metrics(state_np)
    initializer_pca, initializer_score, initializer_fraction = _pca_metrics(initializer_np)
    latent_distance = _pairwise_latent(record.topology, endpoint_latent)
    hidden_distance = np.linalg.norm(state_np[:, None, :] - state_np[None, :, :], axis=-1)
    k = int(geometry["knn_k"])
    latent_neighbors = np.argsort(latent_distance, axis=1)[:, 1 : k + 1]
    hidden_neighbors = np.argsort(hidden_distance, axis=1)[:, 1 : k + 1]
    overlap = np.mean(
        [len(set(a.tolist()).intersection(b.tolist())) / k for a, b in zip(latent_neighbors, hidden_neighbors)]
    )
    upper = np.triu_indices(count, k=1)
    spearman = _spearman(latent_distance[upper], hidden_distance[upper])
    far = latent_distance[upper] >= np.quantile(latent_distance[upper], 0.75)
    hidden_small = hidden_distance[upper] <= np.quantile(hidden_distance[upper], 0.01)
    far_collapse_fraction = float(np.mean(hidden_small[far]))

    closed_mask = np.isin(bank["closed_anchor_index"], indices)
    closed_initial = torch.as_tensor(bank["closed_initial_memory"][closed_mask], device=device)
    closed_inputs = torch.as_tensor(bank["closed_inputs"][:, closed_mask], device=device)
    closed_state, closed_prediction = forward_endpoint_states(
        model, closed_inputs, closed_initial, chunk_size=128
    )
    paths = int(config["analysis_banks"]["closed_path_count"])
    closed_np = closed_state.cpu().numpy().reshape(count, paths, -1).astype(np.float64)
    centroid = closed_np.mean(axis=1)
    within = np.linalg.norm(closed_np - centroid[:, None], axis=-1).mean()
    centroid_distance = np.linalg.norm(
        centroid[:, None, :] - centroid[None, :, :], axis=-1
    )
    between = np.median(centroid_distance[upper])
    fiber_ratio = float(within / max(between, np.finfo(np.float64).eps))
    closed_target = torch.as_tensor(bank["closed_endpoint_target"][closed_mask], device=device)
    closed_error, _ = normalized_geodesic_errors(
        record.topology, closed_prediction, closed_target
    )
    metrics = {
        "schema_version": 1,
        "job_id": record.job_id,
        "model": record.model_id,
        "topology": record.topology,
        "seed": record.seed,
        "atlas_points": count,
        "endpoint_decoding_error_mean": float(error.mean().cpu()),
        "closed_path_decoding_error_mean": float(closed_error.mean().cpu()),
        "local_tangent_rank_mean": float(effective_rank.mean()),
        "local_tangent_rank_full_fraction": float(
            np.mean(effective_rank == tangent.shape[-1])
        ),
        "local_sigma_min_median": float(np.median(sigma_min)),
        "local_tangent_condition_median": float(np.median(condition)),
        "knn_overlap": float(overlap),
        "latent_hidden_distance_spearman": spearman,
        "far_latent_hidden_collapse_fraction": far_collapse_fraction,
        "same_memory_within_dispersion": float(within),
        "between_anchor_separation": float(between),
        "fiber_ratio": fiber_ratio,
        "endpoint_pca": endpoint_pca,
        "initializer_pca_diagnostic_only": initializer_pca,
    }
    output = analysis_root / "geometry" / "runs"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f"{record.job_id}.json", metrics)
    atomic_npz(
        output / f"{record.job_id}.npz",
        anchor_index=indices,
        endpoint_state=state_np.astype(np.float32),
        endpoint_latent=endpoint_latent.astype(np.float32),
        endpoint_pca_score=endpoint_score.astype(np.float32),
        endpoint_pca_explained_fraction=endpoint_fraction.astype(np.float32),
        initializer_pca_score=initializer_score.astype(np.float32),
        initializer_pca_explained_fraction=initializer_fraction.astype(np.float32),
        tangent_singular_values=singular.astype(np.float32),
        tangent_basis_raw=tangent.astype(np.float32),
        closed_endpoint_state=closed_np.astype(np.float32),
    )
    return metrics


def aggregate(analysis_root: Path) -> None:
    rows = []
    arrays = {}
    for path in sorted((analysis_root / "geometry" / "runs").glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"schema_version", "endpoint_pca", "initializer_pca_diagnostic_only"}
            }
        )
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
            for name in archive.files:
                arrays[f"{item['job_id']}__{name}"] = np.array(archive[name], copy=True)
    write_csv(analysis_root / "geometry" / "geometry_metrics.csv", rows)
    atomic_npz(analysis_root / "geometry" / "geometry_metrics.npz", **arrays)
    atomic_npz(analysis_root / "geometry_metrics.npz", **arrays)


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
        raise RuntimeError("geometry analysis requires all completed runs")
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
