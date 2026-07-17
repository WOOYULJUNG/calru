"""Persistent-homology comparison of learned topology under blank dynamics."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
import statistics
from typing import Any, Iterable
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from repro.sagodi_protocol.artifacts import atomic_json, strict_json_load

from .topology_analysis_common import (
    RunRecord,
    atomic_npz,
    blank_snapshots,
    discover_completed_runs,
    file_sha256,
    load_analysis_config,
    load_model,
    write_csv,
)


TOPOLOGIES = ("s1", "t2", "s2")
MODELS = ("rnn", "gru", "lstm", "calru", "hc")
MODEL_LABELS = {
    "rnn": "RNN",
    "gru": "GRU",
    "lstm": "LSTM",
    "calru": "CA-LRU",
    "hc": "H-C",
}
TOPOLOGY_LABELS = {"s1": r"$S^1$", "t2": r"$T^2$", "s2": r"$S^2$"}
COLORS = {
    "rnn": "#6C757D",
    "gru": "#E9C46A",
    "lstm": "#E76F51",
    "calru": "#2A9D8F",
    "hc": "#8E5EA2",
}


@dataclass(frozen=True)
class Target:
    record: RunRecord
    geometry_root: Path
    selection_regime: str


def _read_task_index(targets: list[Target]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for root in sorted({target.geometry_root for target in targets}):
        candidates = (
            root / "task" / "task_metrics.csv",
            root / "task_metrics.csv",
            root / "finalist_seed_metrics.csv",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"no task metrics table found under analysis root {root}"
            )
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                success = row.get("task_success", row.get("task_gate"))
                validation_error = row.get(
                    "validation_error",
                    row.get("validation_task_intrinsic_radians"),
                )
                if success is None or validation_error is None:
                    raise RuntimeError(
                        f"{path} lacks task-success or validation-error fields"
                    )
                index[row["job_id"]] = {
                    "task_success": str(success).lower() == "true",
                    "validation_error": float(validation_error),
                }
    missing = sorted(
        target.record.job_id
        for target in targets
        if target.record.job_id not in index
    )
    if missing:
        raise RuntimeError(f"missing task metrics for persistent targets: {missing}")
    return index


def _representatives(
    targets: list[Target],
    task_index: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], Target]:
    """Prefer successful seeds, then choose the best available failed seed."""

    selected: dict[tuple[str, str], Target] = {}
    for topology in TOPOLOGIES:
        for model in MODELS:
            all_candidates = [
                target
                for target in targets
                if target.record.topology == topology
                and target.record.model_id == model
            ]
            if not all_candidates:
                raise RuntimeError(f"no checkpoints for {topology}/{model}")
            successful = [
                target
                for target in all_candidates
                if bool(task_index[target.record.job_id]["task_success"])
            ]
            candidates = successful if successful else all_candidates
            selected[(model, topology)] = min(
                candidates,
                key=lambda target: (
                    float(task_index[target.record.job_id]["validation_error"]),
                    int(target.record.seed),
                ),
            )
    return selected


def _optional_ripser():
    try:
        from ripser import ripser
    except ImportError as error:
        raise RuntimeError(
            "persistent topology analysis requires "
            "`python -m pip install -e '.[topology-analysis]'`"
        ) from error
    return ripser


def _optional_bottleneck():
    try:
        from persim import bottleneck
    except ImportError as error:
        raise RuntimeError(
            "persistent topology analysis requires "
            "`python -m pip install -e '.[topology-analysis]'`"
        ) from error
    return bottleneck


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    squared = np.sum(points * points, axis=1, keepdims=True)
    distance_squared = squared + squared.T - 2.0 * points @ points.T
    np.maximum(distance_squared, 0.0, out=distance_squared)
    return np.sqrt(distance_squared)


def _knn_scale(points: np.ndarray, k: int) -> tuple[float, float]:
    distances = _pairwise_distances(points)
    neighbor = np.partition(distances, int(k), axis=1)[:, int(k)]
    positive = neighbor[neighbor > np.finfo(np.float64).eps]
    zero_fraction = float(np.mean(neighbor <= np.finfo(np.float64).eps))
    if positive.size:
        return float(np.median(positive)), zero_fraction
    upper = distances[np.triu_indices(len(points), k=1)]
    positive = upper[upper > np.finfo(np.float64).eps]
    return (float(np.median(positive)) if positive.size else 1.0), zero_fraction


def _finite_diagram(diagram: np.ndarray) -> np.ndarray:
    diagram = np.asarray(diagram, dtype=np.float64).reshape(-1, 2)
    return diagram[np.isfinite(diagram).all(axis=1)]


def _persistences(diagram: np.ndarray) -> np.ndarray:
    finite = _finite_diagram(diagram)
    if not len(finite):
        return np.empty(0, dtype=np.float64)
    return np.sort(finite[:, 1] - finite[:, 0])[::-1]


def _strong_feature_count(diagram: np.ndarray, threshold: float) -> int:
    return int(np.sum(_persistences(diagram) >= float(threshold)))


def _diagram_bottleneck(left: np.ndarray, right: np.ndarray) -> float:
    first = _finite_diagram(left)
    second = _finite_diagram(right)
    if not len(first) and not len(second):
        return 0.0
    return float(_optional_bottleneck()(first, second))


def _compute_diagrams(points: np.ndarray, *, maxdim: int, k: int) -> tuple[list[np.ndarray], float, float]:
    scale, zero_fraction = _knn_scale(points, k)
    normalized = np.asarray(points, dtype=np.float64) / max(
        scale, np.finfo(np.float64).eps
    )
    with warnings.catch_warnings():
        # A [128,128] point cloud is square when hidden width equals the fixed
        # landmark count. It is still a point cloud, not a distance matrix.
        warnings.filterwarnings(
            "ignore",
            message="The input matrix is square, but the distance_matrix flag is off.*",
            category=UserWarning,
        )
        result = _optional_ripser()(normalized, maxdim=int(maxdim), coeff=2)
    diagrams = [np.asarray(item, dtype=np.float64) for item in result["dgms"]]
    return diagrams, scale, zero_fraction


def _farthest_landmarks(points: np.ndarray, count: int) -> np.ndarray:
    result = _optional_ripser()(
        np.asarray(points, dtype=np.float64),
        maxdim=0,
        n_perm=int(count),
        coeff=2,
    )
    return np.asarray(result["idx_perm"], dtype=np.int64)


def _top_persistence(diagram: np.ndarray, rank: int) -> float:
    persistence = _persistences(diagram)
    index = int(rank) - 1
    return float(persistence[index]) if len(persistence) > index else 0.0


def _reference_for_topology(
    topology: str,
    bank_root: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(
        bank_root / f"analysis_bank_{topology}.npz", allow_pickle=False
    ) as archive:
        target = np.array(archive["transport_endpoint_target"], copy=True)
    atlas_count = int(config["geometry_atlas_points"])
    atlas_indices = np.linspace(0, len(target) - 1, atlas_count, dtype=np.int64)
    atlas = target[atlas_indices]
    landmark_local = _farthest_landmarks(
        atlas, int(config["persistent_homology_landmarks"])
    )
    points = atlas[landmark_local]
    diagrams, scale, zero_fraction = _compute_diagrams(
        points,
        maxdim=int(config["max_homology_dimension"]),
        k=int(config["knn_scale_k"]),
    )
    expected = config["expected_reduced_betti"][topology]
    expected_persistence = []
    for dimension in (1, 2):
        count = int(expected[f"h{dimension}"])
        if count:
            expected_persistence.append(
                _top_persistence(diagrams[dimension], count)
            )
    if not expected_persistence or min(expected_persistence) <= 0:
        raise RuntimeError(f"ideal {topology} reference lacks its expected topology")
    threshold = float(
        config["strong_bar_threshold_fraction_of_weakest_expected_ideal_bar"]
    ) * min(expected_persistence)
    counts = {
        f"h{dimension}": _strong_feature_count(diagrams[dimension], threshold)
        for dimension in (1, 2)
    }
    expected_counts = {name: int(value) for name, value in expected.items()}
    if counts != expected_counts:
        raise RuntimeError(
            f"ideal {topology} signature {counts} != expected {expected_counts}"
        )
    arrays = {
        "atlas_index": atlas_indices,
        "landmark_local_index": landmark_local,
        "landmark_atlas_index": atlas_indices[landmark_local],
        "landmark_points": points.astype(np.float32),
        **{
            f"diagram_h{dimension}": diagrams[dimension].astype(np.float32)
            for dimension in range(3)
        },
    }
    metrics = {
        "topology": topology,
        "expected_h1": expected_counts["h1"],
        "expected_h2": expected_counts["h2"],
        "strong_bar_threshold": threshold,
        "knn_scale": scale,
        "zero_knn_fraction": zero_fraction,
        "detected_h1": counts["h1"],
        "detected_h2": counts["h2"],
        "h1_top_persistences": _persistences(diagrams[1])[:5].tolist(),
        "h2_top_persistences": _persistences(diagrams[2])[:5].tolist(),
    }
    return metrics, arrays


def _collect_targets(args: argparse.Namespace) -> list[Target]:
    baseline_config = load_analysis_config(args.baseline_config.resolve(strict=True))
    baseline_records, missing = discover_completed_runs(
        args.baseline_run_root.resolve(strict=True), baseline_config
    )
    if missing:
        raise RuntimeError(f"missing baseline runs: {missing}")

    calru_config = load_analysis_config(args.calru_config.resolve(strict=True))
    calru_records, missing = discover_completed_runs(
        args.calru_run_root.resolve(strict=True), calru_config
    )
    if missing:
        raise RuntimeError(f"missing CA-LRU runs: {missing}")

    hc_config = load_analysis_config(args.hc_config.resolve(strict=True))
    hc_records, missing = discover_completed_runs(
        args.hc_run_root.resolve(strict=True), hc_config
    )
    if missing:
        raise RuntimeError(f"missing H-C finalist runs: {missing}")
    selection = strict_json_load(
        args.hc_analysis_root.resolve(strict=True)
        / "selected_hc_attractor_configurations.json"
    )
    if selection.get("test_bank_accessed"):
        raise RuntimeError("H-C selection artifact reports test access")
    selected_hc = {
        job_id
        for job_ids in selection["selected_jobs"].values()
        for job_id in job_ids
    }

    targets = [
        *[
            Target(
                record=record,
                geometry_root=args.baseline_analysis_root.resolve(strict=True),
                selection_regime="ring-selected zero-retuning",
            )
            for record in baseline_records
            if record.model_id in {"rnn", "gru", "lstm"}
        ],
        *[
            Target(
                record=record,
                geometry_root=args.calru_analysis_root.resolve(strict=True),
                selection_regime="topology-specific validation-selected",
            )
            for record in calru_records
            if record.model_id == "calru"
        ],
        *[
            Target(
                record=record,
                geometry_root=args.hc_analysis_root.resolve(strict=True),
                selection_regime="topology-specific CA-aware selected",
            )
            for record in hc_records
            if record.job_id in selected_hc
        ],
    ]
    expected = {
        (model, topology, seed)
        for model in MODELS
        for topology in TOPOLOGIES
        for seed in (10, 11, 12)
    }
    observed = {
        (item.record.model_id, item.record.topology, item.record.seed)
        for item in targets
    }
    if observed != expected or len(targets) != 45:
        raise RuntimeError(
            f"persistent topology target set mismatch: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return sorted(
        targets,
        key=lambda item: (
            TOPOLOGIES.index(item.record.topology),
            MODELS.index(item.record.model_id),
            item.record.seed,
        ),
    )


def _analyze_target(
    target: Target,
    *,
    config: dict[str, Any],
    reference: dict[str, dict[str, Any]],
    reference_arrays: dict[str, dict[str, np.ndarray]],
    output: Path,
    device: torch.device,
    config_sha256: str,
) -> None:
    record = target.record
    geometry_path = (
        target.geometry_root / "geometry" / "runs" / f"{record.job_id}.npz"
    )
    with np.load(geometry_path, allow_pickle=False) as archive:
        state_np = np.array(archive["endpoint_state"], copy=True)
        anchor_index = np.array(archive["anchor_index"], copy=True)
    expected_atlas = int(config["geometry_atlas_points"])
    if state_np.shape[0] != expected_atlas:
        raise RuntimeError(
            f"{record.job_id}: expected {expected_atlas} geometry points"
        )
    landmark_local = reference_arrays[record.topology]["landmark_local_index"]
    model, _ = load_model(record, device)
    state = torch.as_tensor(state_np, device=device)
    horizons = [int(value) for value in config["blank_horizons"]]
    with torch.no_grad():
        snapshots = blank_snapshots(model, state, horizons)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = []
    arrays: dict[str, np.ndarray] = {
        "anchor_index": anchor_index,
        "landmark_local_index": landmark_local,
    }
    threshold = float(reference[record.topology]["strong_bar_threshold"])
    expected_h1 = int(reference[record.topology]["expected_h1"])
    expected_h2 = int(reference[record.topology]["expected_h2"])
    for horizon in horizons:
        points = (
            snapshots[horizon][landmark_local].detach().cpu().numpy()
        )
        finite = bool(np.isfinite(points).all())
        if not finite:
            raise RuntimeError(f"{record.job_id} is nonfinite at blank {horizon}")
        diagrams, scale, zero_fraction = _compute_diagrams(
            points,
            maxdim=int(config["max_homology_dimension"]),
            k=int(config["knn_scale_k"]),
        )
        detected_h1 = _strong_feature_count(diagrams[1], threshold)
        detected_h2 = _strong_feature_count(diagrams[2], threshold)
        bottleneck_h1 = _diagram_bottleneck(
            diagrams[1], reference_arrays[record.topology]["diagram_h1"]
        )
        bottleneck_h2 = _diagram_bottleneck(
            diagrams[2], reference_arrays[record.topology]["diagram_h2"]
        )
        rows.append(
            {
                "horizon": horizon,
                "knn_scale": scale,
                "zero_knn_fraction": zero_fraction,
                "detected_h1": detected_h1,
                "detected_h2": detected_h2,
                "expected_h1": expected_h1,
                "expected_h2": expected_h2,
                "signature_distance": abs(detected_h1 - expected_h1)
                + abs(detected_h2 - expected_h2),
                "signature_match": detected_h1 == expected_h1
                and detected_h2 == expected_h2,
                "bottleneck_h1": bottleneck_h1,
                "bottleneck_h2": bottleneck_h2,
                "bottleneck_mean": 0.5 * (bottleneck_h1 + bottleneck_h2),
                "bottleneck_mean_over_threshold": 0.5
                * (bottleneck_h1 + bottleneck_h2)
                / max(threshold, np.finfo(np.float64).eps),
                "h1_persistence_1": _top_persistence(diagrams[1], 1),
                "h1_persistence_2": _top_persistence(diagrams[1], 2),
                "h1_persistence_3": _top_persistence(diagrams[1], 3),
                "h2_persistence_1": _top_persistence(diagrams[2], 1),
                "h2_persistence_2": _top_persistence(diagrams[2], 2),
                "finite": finite,
            }
        )
        arrays[f"state_h{horizon}"] = points.astype(np.float32)
        for dimension in range(3):
            arrays[f"diagram_h{horizon}_d{dimension}"] = diagrams[
                dimension
            ].astype(np.float32)
    payload = {
        "schema_version": 1,
        "job_id": record.job_id,
        "model": record.model_id,
        "topology": record.topology,
        "seed": record.seed,
        "selection_regime": target.selection_regime,
        "device": str(device),
        "analysis_config_sha256": config_sha256,
        "rows": rows,
    }
    run_output = output / "runs"
    run_output.mkdir(parents=True, exist_ok=True)
    atomic_json(run_output / f"{record.job_id}.json", payload)
    atomic_npz(run_output / f"{record.job_id}.npz", **arrays)


def _flatten_runs(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((output / "runs").glob("*.json")):
        payload = strict_json_load(path)
        for metrics in payload["rows"]:
            rows.append(
                {
                    "job_id": payload["job_id"],
                    "model": payload["model"],
                    "topology": payload["topology"],
                    "seed": payload["seed"],
                    "selection_regime": payload["selection_regime"],
                    **metrics,
                }
            )
    rows.sort(
        key=lambda row: (
            TOPOLOGIES.index(row["topology"]),
            MODELS.index(row["model"]),
            int(row["seed"]),
            int(row["horizon"]),
        )
    )
    return rows


def _median(values: Iterable[Any]) -> float:
    return float(statistics.median(float(value) for value in values))


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    metrics = (
        "knn_scale",
        "zero_knn_fraction",
        "detected_h1",
        "detected_h2",
        "signature_distance",
        "bottleneck_h1",
        "bottleneck_h2",
        "bottleneck_mean",
        "bottleneck_mean_over_threshold",
        "h1_persistence_1",
        "h1_persistence_2",
        "h1_persistence_3",
        "h2_persistence_1",
        "h2_persistence_2",
    )
    for topology in TOPOLOGIES:
        for model in MODELS:
            for horizon in sorted({int(row["horizon"]) for row in rows}):
                group = [
                    row
                    for row in rows
                    if row["topology"] == topology
                    and row["model"] == model
                    and int(row["horizon"]) == horizon
                ]
                if len(group) != 3:
                    raise RuntimeError(
                        f"expected 3 rows for {topology}/{model}/{horizon}"
                    )
                item = {
                    "topology": topology,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "horizon": horizon,
                    "seed_count": len(group),
                    "signature_match_count": sum(
                        bool(row["signature_match"]) for row in group
                    ),
                    "signature_match_fraction": sum(
                        bool(row["signature_match"]) for row in group
                    )
                    / len(group),
                    "expected_h1": int(group[0]["expected_h1"]),
                    "expected_h2": int(group[0]["expected_h2"]),
                }
                for metric in metrics:
                    values = [float(row[metric]) for row in group]
                    item[f"{metric}_median"] = _median(values)
                    item[f"{metric}_min"] = min(values)
                    item[f"{metric}_max"] = max(values)
                output.append(item)
    return output


def _threshold_sensitivity(
    targets: list[Target],
    reference: dict[str, dict[str, Any]],
    output: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_fraction = float(
        config["strong_bar_threshold_fraction_of_weakest_expected_ideal_bar"]
    )
    fractions = [float(value) for value in config["threshold_sensitivity_fractions"]]
    rows = []
    for target in targets:
        record = target.record
        weakest_expected = (
            float(reference[record.topology]["strong_bar_threshold"])
            / primary_fraction
        )
        expected_h1 = int(reference[record.topology]["expected_h1"])
        expected_h2 = int(reference[record.topology]["expected_h2"])
        with np.load(
            output / "runs" / f"{record.job_id}.npz", allow_pickle=False
        ) as archive:
            for horizon in config["blank_horizons"]:
                h1 = np.array(archive[f"diagram_h{horizon}_d1"], copy=True)
                h2 = np.array(archive[f"diagram_h{horizon}_d2"], copy=True)
                for fraction in fractions:
                    threshold = fraction * weakest_expected
                    detected_h1 = _strong_feature_count(h1, threshold)
                    detected_h2 = _strong_feature_count(h2, threshold)
                    rows.append(
                        {
                            "job_id": record.job_id,
                            "model": record.model_id,
                            "topology": record.topology,
                            "seed": record.seed,
                            "horizon": int(horizon),
                            "threshold_fraction": fraction,
                            "strong_bar_threshold": threshold,
                            "expected_h1": expected_h1,
                            "expected_h2": expected_h2,
                            "detected_h1": detected_h1,
                            "detected_h2": detected_h2,
                            "signature_match": detected_h1 == expected_h1
                            and detected_h2 == expected_h2,
                        }
                    )
    summary = []
    for topology in TOPOLOGIES:
        for model in MODELS:
            for horizon in config["blank_horizons"]:
                for fraction in fractions:
                    group = [
                        row
                        for row in rows
                        if row["topology"] == topology
                        and row["model"] == model
                        and int(row["horizon"]) == int(horizon)
                        and float(row["threshold_fraction"]) == fraction
                    ]
                    if len(group) != 3:
                        raise RuntimeError(
                            "threshold sensitivity expected three seeds for "
                            f"{topology}/{model}/{horizon}/{fraction}"
                        )
                    summary.append(
                        {
                            "topology": topology,
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "horizon": int(horizon),
                            "threshold_fraction": fraction,
                            "signature_match_count": sum(
                                bool(row["signature_match"]) for row in group
                            ),
                            "signature_match_fraction": sum(
                                bool(row["signature_match"]) for row in group
                            )
                            / len(group),
                            "detected_h1_median": _median(
                                row["detected_h1"] for row in group
                            ),
                            "detected_h2_median": _median(
                                row["detected_h2"] for row in group
                            ),
                        }
                    )
    return rows, summary


def _save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            figures / f"{stem}.{suffix}", dpi=240, bbox_inches="tight"
        )
    plt.close(figure)


def _plot_signature_heatmap(
    summary: list[dict[str, Any]], output: Path, horizons: list[int]
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for axis, topology in zip(axes, TOPOLOGIES):
        expected_row = next(row for row in summary if row["topology"] == topology)
        image = np.empty((len(MODELS), len(horizons)), dtype=float)
        labels: list[list[str]] = []
        for model_index, model in enumerate(MODELS):
            label_row = []
            for horizon_index, horizon in enumerate(horizons):
                row = next(
                    item
                    for item in summary
                    if item["topology"] == topology
                    and item["model"] == model
                    and int(item["horizon"]) == horizon
                )
                image[model_index, horizon_index] = float(
                    row["signature_match_fraction"]
                )
                label_row.append(
                    f"{row['detected_h1_median']:.0f}/"
                    f"{row['detected_h2_median']:.0f}"
                )
            labels.append(label_row)
        rendered = axis.imshow(
            image, vmin=0.0, vmax=1.0, cmap="YlGn", aspect="auto"
        )
        for row_index in range(len(MODELS)):
            for column_index in range(len(horizons)):
                axis.text(
                    column_index,
                    row_index,
                    labels[row_index][column_index],
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )
        axis.set_xticks(range(len(horizons)), [str(value) for value in horizons])
        axis.set_yticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
        axis.set_xlabel("Blank horizon")
        axis.set_title(
            f"{TOPOLOGY_LABELS[topology]} · expected persistent "
            f"$H_1/H_2$="
            f"{expected_row['expected_h1']}/{expected_row['expected_h2']}"
        )
    axes[0].set_ylabel("Model")
    figure.subplots_adjust(
        top=0.77, bottom=0.15, left=0.08, right=0.86, wspace=0.18
    )
    color_axis = figure.add_axes((0.89, 0.16, 0.016, 0.67))
    colorbar = figure.colorbar(rendered, cax=color_axis)
    colorbar.set_label("Fraction of 3 seeds matching signature")
    figure.suptitle(
        "All-seed robustness (secondary): persistent-homology signature\n"
        "Cell text: median detected persistent $H_1/H_2$ bars"
    )
    _save_figure(figure, output, "fig_topology_signature_all_seeds")


def _plot_representative_heatmap(
    rows: list[dict[str, Any]],
    representatives: dict[tuple[str, str], Target],
    output: Path,
    horizons: list[int],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 4.8))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#E5E7EB")
    rendered = None
    for axis, topology in zip(axes, TOPOLOGIES):
        expected_row = next(row for row in rows if row["topology"] == topology)
        image = np.full((len(MODELS), len(horizons)), np.nan, dtype=float)
        labels = [["N/A" for _ in horizons] for _ in MODELS]
        ylabels = []
        for model_index, model in enumerate(MODELS):
            target = representatives[(model, topology)]
            first_row = next(
                item for item in rows if item["job_id"] == target.record.job_id
            )
            successful = bool(first_row["task_success"])
            ylabels.append(
                f"{MODEL_LABELS[model]} (s{target.record.seed}"
                f"{'' if successful else '*'})"
            )
            for horizon_index, horizon in enumerate(horizons):
                row = next(
                    item
                    for item in rows
                    if item["job_id"] == target.record.job_id
                    and int(item["horizon"]) == horizon
                )
                image[model_index, horizon_index] = float(
                    bool(row["signature_match"])
                )
                labels[model_index][horizon_index] = (
                    f"{int(row['detected_h1'])}/{int(row['detected_h2'])}"
                )
        rendered = axis.imshow(
            np.ma.masked_invalid(image),
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
            aspect="auto",
        )
        for row_index in range(len(MODELS)):
            for column_index in range(len(horizons)):
                axis.text(
                    column_index,
                    row_index,
                    labels[row_index][column_index],
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )
        axis.set_xticks(range(len(horizons)), [str(value) for value in horizons])
        axis.set_yticks(range(len(MODELS)), ylabels)
        axis.set_xlabel("Blank horizon")
        axis.set_title(
            f"{TOPOLOGY_LABELS[topology]} · expected persistent "
            f"$H_1/H_2$={expected_row['expected_h1']}/{expected_row['expected_h2']}"
        )
    axes[0].set_ylabel("Model (selected seed)")
    figure.subplots_adjust(
        top=0.78, bottom=0.15, left=0.08, right=0.90, wspace=0.34
    )
    if rendered is not None:
        color_axis = figure.add_axes((0.93, 0.17, 0.012, 0.64))
        colorbar = figure.colorbar(rendered, cax=color_axis, ticks=[0, 1])
        colorbar.set_ticklabels(["mismatch", "match"])
    figure.suptitle(
        "Primary topology check: representative seed per model\n"
        "Cell text: detected persistent $H_1/H_2$ bars; "
        "* marks best-available task-failed fallback"
    )
    _save_figure(figure, output, "fig_topology_signature_representatives")


def _plot_bottleneck(
    summary: list[dict[str, Any]], output: Path, horizons: list[int]
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for axis, topology in zip(axes, TOPOLOGIES):
        for model in MODELS:
            group = [
                row
                for row in summary
                if row["topology"] == topology and row["model"] == model
            ]
            group.sort(key=lambda row: int(row["horizon"]))
            median = np.asarray(
                [row["bottleneck_mean_over_threshold_median"] for row in group]
            )
            lower = np.asarray(
                [row["bottleneck_mean_over_threshold_min"] for row in group]
            )
            upper = np.asarray(
                [row["bottleneck_mean_over_threshold_max"] for row in group]
            )
            axis.plot(
                horizons,
                median,
                marker="o",
                label=MODEL_LABELS[model],
                color=COLORS[model],
            )
            axis.fill_between(
                horizons, lower, upper, color=COLORS[model], alpha=0.12
            )
        axis.set_xscale("symlog", linthresh=128)
        axis.set_yscale("log")
        axis.set_xticks(horizons, [str(value) for value in horizons])
        axis.set_xlabel("Blank horizon")
        axis.set_title(TOPOLOGY_LABELS[topology])
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Diagram distance / strong-bar threshold ↓")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Distance from the ideal topology persistence diagram · median and seed range"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(figure, output, "fig_topology_diagram_distance")


def _plot_representative_diagrams(
    representatives: dict[tuple[str, str], Target],
    task_index: dict[str, dict[str, Any]],
    reference_arrays: dict[str, dict[str, np.ndarray]],
    output: Path,
    horizon: int,
) -> None:
    figure, axes = plt.subplots(
        len(TOPOLOGIES),
        len(MODELS) + 1,
        figsize=(17, 8.5),
        squeeze=False,
    )
    columns = ["Ideal", *[MODEL_LABELS[model] for model in MODELS]]
    for row_index, topology in enumerate(TOPOLOGIES):
        sources: list[tuple[np.ndarray, np.ndarray]] = [
            (
                reference_arrays[topology]["diagram_h1"],
                reference_arrays[topology]["diagram_h2"],
            )
        ]
        for model in MODELS:
            target = representatives[(model, topology)]
            with np.load(
                output / "runs" / f"{target.record.job_id}.npz",
                allow_pickle=False,
            ) as archive:
                sources.append(
                    (
                        np.array(archive[f"diagram_h{horizon}_d1"], copy=True),
                        np.array(archive[f"diagram_h{horizon}_d2"], copy=True),
                    )
                )
        finite_values = [
            value
            for diagrams in sources
            for diagram in diagrams
            for value in _finite_diagram(diagram).ravel()
        ]
        limit = max(finite_values, default=1.0) * 1.05
        for column_index, (h1, h2) in enumerate(sources):
            axis = axes[row_index, column_index]
            if column_index:
                model = MODELS[column_index - 1]
                target = representatives[(model, topology)]
                successful = bool(
                    task_index[target.record.job_id]["task_success"]
                )
                axis.text(
                    0.98,
                    0.03,
                    f"seed {target.record.seed}"
                    + ("" if successful else "* · failed fallback"),
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    color="#333333" if successful else "#B33A3A",
                    fontsize=7,
                )
            for diagram, label, color in (
                (h1, "$H_1$", "#4477AA"),
                (h2, "$H_2$", "#CC6677"),
            ):
                finite = _finite_diagram(diagram)
                if len(finite):
                    axis.scatter(
                        finite[:, 0],
                        finite[:, 1],
                        s=13,
                        alpha=0.75,
                        label=label,
                        color=color,
                    )
            axis.plot([0, limit], [0, limit], color="0.55", linewidth=0.8)
            axis.set_xlim(0, limit)
            axis.set_ylim(0, limit)
            if row_index == 0:
                axis.set_title(columns[column_index])
            if column_index == 0:
                axis.set_ylabel(
                    f"{TOPOLOGY_LABELS[topology]}\nDeath"
                )
            if row_index == len(TOPOLOGIES) - 1:
                axis.set_xlabel("Birth")
            if row_index == 0 and column_index == len(MODELS):
                axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"Persistent diagrams after {horizon} blank steps · "
        "model-specific representatives (* = task-failed fallback)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(
        figure,
        output,
        f"fig_representative_persistence_diagrams_h{horizon}",
    )


def analyze(args: argparse.Namespace) -> None:
    config = strict_json_load(args.persistence_config.resolve(strict=True))
    config_sha256 = file_sha256(args.persistence_config.resolve(strict=True))
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = _collect_targets(args)
    task_index = _read_task_index(targets)
    representatives = _representatives(targets, task_index)

    bank_root = args.hc_analysis_root.resolve(strict=True) / "banks"
    reference: dict[str, dict[str, Any]] = {}
    reference_arrays: dict[str, dict[str, np.ndarray]] = {}
    reference_root = output / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    for topology in TOPOLOGIES:
        metrics, arrays = _reference_for_topology(topology, bank_root, config)
        reference[topology] = metrics
        reference_arrays[topology] = arrays
        atomic_json(reference_root / f"{topology}.json", metrics)
        atomic_npz(reference_root / f"{topology}.npz", **arrays)

    devices = [
        item.strip() for item in args.devices.split(",") if item.strip()
    ]
    if not devices:
        devices = ["cpu"]
    pending = []
    for target in targets:
        artifact = output / "runs" / f"{target.record.job_id}.json"
        if not artifact.is_file():
            pending.append(target)
            continue
        payload = strict_json_load(artifact)
        if payload.get("analysis_config_sha256") != config_sha256:
            pending.append(target)
    queues = [[] for _ in devices]
    for index, target in enumerate(pending):
        queues[index % len(devices)].append(target)

    def worker(device_name: str, queue: list[Target]) -> None:
        device = torch.device(
            f"cuda:{device_name}"
            if device_name != "cpu" and torch.cuda.is_available()
            else "cpu"
        )
        for target in queue:
            _analyze_target(
                target,
                config=config,
                reference=reference,
                reference_arrays=reference_arrays,
                output=output,
                device=device,
                config_sha256=config_sha256,
            )

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(worker, device_name, queue)
            for device_name, queue in zip(devices, queues)
            if queue
        ]
        for future in futures:
            future.result()

    rows = _flatten_runs(output)
    if len(rows) != 45 * len(config["blank_horizons"]):
        raise RuntimeError(
            f"expected {45 * len(config['blank_horizons'])} "
            f"seed-horizon rows, found {len(rows)}"
        )
    for row in rows:
        task = task_index[row["job_id"]]
        row["task_success"] = bool(task["task_success"])
        row["validation_error"] = float(task["validation_error"])
    summary = _summarize(rows)
    sensitivity_rows, sensitivity_summary = _threshold_sensitivity(
        targets, reference, output, config
    )
    write_csv(output / "persistent_topology_seed_metrics.csv", rows)
    write_csv(output / "persistent_topology_summary.csv", summary)
    representative_selection = []
    representative_rows = []
    for topology in TOPOLOGIES:
        for model in MODELS:
            candidates = [
                target
                for target in targets
                if target.record.topology == topology
                and target.record.model_id == model
                and bool(task_index[target.record.job_id]["task_success"])
            ]
            target = representatives[(model, topology)]
            selected_success = bool(
                task_index[target.record.job_id]["task_success"]
            )
            representative_selection.append(
                {
                    "topology": topology,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "task_success_seed_count": len(candidates),
                    "representative_available": True,
                    "selected_job_id": target.record.job_id,
                    "selected_seed": target.record.seed,
                    "selected_task_success": selected_success,
                    "selection_kind": (
                        "best_validation_success"
                        if selected_success
                        else "best_available_failed_fallback"
                    ),
                    "selected_validation_error": task_index[
                        target.record.job_id
                    ]["validation_error"],
                    "selection_policy": (
                        "lowest validation_error among task_success=true; "
                        "if none succeed, lowest-validation-error checkpoint "
                        "is an explicit failed-task fallback"
                    ),
                }
            )
            representative_rows.extend(
                row for row in rows if row["job_id"] == target.record.job_id
            )
    write_csv(
        output / "representative_selection.csv",
        representative_selection,
    )
    write_csv(
        output / "representative_seed_metrics.csv",
        representative_rows,
    )
    write_csv(
        output / "persistent_topology_threshold_sensitivity_seed.csv",
        sensitivity_rows,
    )
    write_csv(
        output / "persistent_topology_threshold_sensitivity_summary.csv",
        sensitivity_summary,
    )
    atomic_json(
        output / "reference_topology_metrics.json",
        {
            "schema_version": 1,
            "config": config,
            "references": reference,
        },
    )
    horizons = [int(value) for value in config["blank_horizons"]]
    _plot_representative_heatmap(rows, representatives, output, horizons)
    _plot_signature_heatmap(summary, output, horizons)
    _plot_bottleneck(summary, output, horizons)
    _plot_representative_diagrams(
        representatives, task_index, reference_arrays, output, horizons[0]
    )
    _plot_representative_diagrams(
        representatives, task_index, reference_arrays, output, horizons[-1]
    )
    atomic_json(
        output / "PERSISTENT_TOPOLOGY_COMPLETED.json",
        {
            "schema_version": 1,
            "analysis_id": config["analysis_id"],
            "analysis_config_sha256": config_sha256,
            "analyzer_source_sha256": file_sha256(Path(__file__).resolve()),
            "ripser_version": package_version("ripser"),
            "source_roots": {
                "baseline_run_root": str(
                    args.baseline_run_root.expanduser().resolve()
                ),
                "baseline_analysis_root": str(
                    args.baseline_analysis_root.expanduser().resolve()
                ),
                "calru_run_root": str(args.calru_run_root.expanduser().resolve()),
                "calru_analysis_root": str(
                    args.calru_analysis_root.expanduser().resolve()
                ),
                "hc_run_root": str(args.hc_run_root.expanduser().resolve()),
                "hc_analysis_root": str(
                    args.hc_analysis_root.expanduser().resolve()
                ),
            },
            "hc_selection_sha256": file_sha256(
                args.hc_analysis_root.expanduser().resolve()
                / "selected_hc_attractor_configurations.json"
            ),
            "run_count": len(targets),
            "seed_horizon_row_count": len(rows),
            "summary_row_count": len(summary),
            "representative_row_count": len(representative_rows),
            "threshold_sensitivity_row_count": len(sensitivity_rows),
            "representative_seed_policy": (
                "lowest validation_error among task_success=true seeds; "
                "if none succeed, lowest-validation-error checkpoint "
                "is an explicit failed-task fallback"
            ),
            "representative_seeds": {
                topology: {
                    model: representatives[(model, topology)].record.seed
                    for model in MODELS
                }
                for topology in TOPOLOGIES
            },
            "figures": [
                "fig_topology_signature_representatives.png",
                "fig_topology_signature_all_seeds.png",
                "fig_topology_diagram_distance.png",
                "fig_representative_persistence_diagrams_"
                f"h{horizons[0]}.png",
                "fig_representative_persistence_diagrams_"
                f"h{horizons[-1]}.png",
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--baseline-analysis-root", type=Path, required=True)
    parser.add_argument("--calru-run-root", type=Path, required=True)
    parser.add_argument("--calru-analysis-root", type=Path, required=True)
    parser.add_argument("--hc-run-root", type=Path, required=True)
    parser.add_argument("--hc-analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5")
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path(__file__).with_name(
            "topology_baseline_all_analysis_v2.json"
        ),
    )
    parser.add_argument(
        "--calru-config",
        type=Path,
        default=Path(__file__).with_name("topology_hparam_analysis_v1.json"),
    )
    parser.add_argument(
        "--hc-config",
        type=Path,
        default=Path(__file__).with_name(
            "topology_hc_attractor_finalists_v1.json"
        ),
    )
    parser.add_argument(
        "--persistence-config",
        type=Path,
        default=Path(__file__).with_name(
            "topology_persistence_success_v2.json"
        ),
    )
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
