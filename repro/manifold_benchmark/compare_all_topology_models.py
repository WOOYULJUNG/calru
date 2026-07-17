"""Combine zero-retuning baselines and selected CA-LRU/H-C topology analyses."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


MODELS = ("rnn", "gru", "lstm", "calru", "hc")
BASELINE_MODELS = ("rnn", "gru", "lstm")
TUNED_MODELS = ("calru", "hc")
TOPOLOGIES = ("s1", "t2", "s2")
SEEDS = (10, 11, 12)
LABELS = {
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
REGIMES = {
    "rnn": "ring-selected zero-retuning",
    "gru": "ring-selected zero-retuning",
    "lstm": "ring-selected zero-retuning",
    "calru": "topology-specific validation-selected",
    "hc": "topology-specific validation-selected",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _bool(value: Any) -> bool:
    return str(value).lower() == "true"


def _median(values: Iterable[Any]) -> float:
    finite = [_number(value) for value in values]
    finite = [value for value in finite if np.isfinite(value)]
    return float(statistics.median(finite)) if finite else np.nan


def _decorate(
    rows: Iterable[dict[str, str]],
    *,
    allowed_models: tuple[str, ...],
    source: str,
) -> list[dict[str, str]]:
    decorated = []
    for original in rows:
        model = original["model"]
        if model not in allowed_models:
            continue
        row = dict(original)
        row["selection_regime"] = REGIMES[model]
        row["source_analysis"] = source
        decorated.append(row)
    return decorated


def _combined_rows(
    baseline: Path,
    selected: Path,
    relative_path: str,
) -> list[dict[str, str]]:
    return [
        *_decorate(
            _read_csv(baseline / relative_path),
            allowed_models=BASELINE_MODELS,
            source="baseline_all_v2",
        ),
        *_decorate(
            _read_csv(selected / relative_path),
            allowed_models=TUNED_MODELS,
            source="selected_hparam_v1",
        ),
    ]


def _analysis_for_row(
    row: dict[str, str],
    baseline: Path,
    selected: Path,
    hc_topology_normal: Path | None = None,
) -> Path:
    if row["source_analysis"] == "baseline_all_v2":
        return baseline
    if row["source_analysis"] == "hc_finalists_v1":
        if hc_topology_normal is None:
            raise RuntimeError("H-C topology-normal analysis root was not provided")
        return hc_topology_normal
    return selected


def _combined_topology_normal_rows(
    baseline: Path,
    selected: Path,
    task_rows: list[dict[str, str]],
    hc_topology_normal: Path | None,
) -> list[dict[str, str]]:
    rows = [
        *_decorate(
            _read_csv(
                baseline / "topology_normal" / "topology_normal_metrics.csv"
            ),
            allowed_models=BASELINE_MODELS,
            source="baseline_all_v2",
        ),
        *_decorate(
            _read_csv(
                selected / "topology_normal" / "topology_normal_metrics.csv"
            ),
            allowed_models=("calru",),
            source="selected_hparam_v1",
        ),
    ]
    if hc_topology_normal is None:
        rows.extend(
            _decorate(
                _read_csv(
                    selected / "topology_normal" / "topology_normal_metrics.csv"
                ),
                allowed_models=("hc",),
                source="selected_hparam_v1",
            )
        )
        return rows

    selected_hc_jobs = {
        row["job_id"] for row in task_rows if row["model"] == "hc"
    }
    hc_rows = _decorate(
        _read_csv(
            hc_topology_normal
            / "topology_normal"
            / "topology_normal_metrics.csv"
        ),
        allowed_models=("hc",),
        source="hc_finalists_v1",
    )
    rows.extend(row for row in hc_rows if row["job_id"] in selected_hc_jobs)
    return rows


def _geometry_extras(
    rows: list[dict[str, str]], baseline: Path, selected: Path
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        root = _analysis_for_row(row, baseline, selected)
        archive_path = root / "geometry" / "runs" / f"{row['job_id']}.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            fraction = np.asarray(archive["endpoint_pca_explained_fraction"], dtype=float)
        participation = 1.0 / max(np.finfo(float).eps, float(np.sum(fraction**2)))
        output.append(
            {
                "job_id": row["job_id"],
                "model": row["model"],
                "topology": row["topology"],
                "seed": int(row["seed"]),
                "selection_regime": row["selection_regime"],
                "endpoint_pca_participation_ratio": participation,
                "endpoint_pca_top3_fraction": float(fraction[:3].sum()),
            }
        )
    return output


def _dynamics_extras(
    rows: list[dict[str, str]], baseline: Path, selected: Path
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        root = _analysis_for_row(row, baseline, selected)
        archive_path = root / "dynamics" / "runs" / f"{row['job_id']}.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            finite_horizons = np.asarray(archive["finite_horizons"])
            finite_index = int(np.flatnonzero(finite_horizons == 128)[0])
            recovery_horizons = np.asarray(archive["recovery_horizons"])
            recovery_index = int(np.flatnonzero(recovery_horizons == 512)[0])
            radii = np.asarray(archive["kick_radii_relative"])
            radius_index = int(np.argmin(np.abs(radii - 0.05)))
            finite_tangent = np.asarray(archive["finite_tangent_gain"])[finite_index]
            finite_normal = np.asarray(archive["finite_normal_gain"])[finite_index]
            recovery = np.asarray(archive["recovery_ratio"])[
                radius_index, recovery_index
            ]
            same_memory = np.asarray(archive["same_memory_error"])[
                radius_index, recovery_index
            ]
        output.append(
            {
                "job_id": row["job_id"],
                "model": row["model"],
                "topology": row["topology"],
                "seed": int(row["seed"]),
                "selection_regime": row["selection_regime"],
                "finite128_tangent_gain_mean": float(np.mean(finite_tangent)),
                "finite128_normal_gain_mean": float(np.mean(finite_normal)),
                "recovery_q512_median": float(np.median(recovery)),
                "same_memory_error512_mean": float(np.mean(same_memory)),
            }
        )
    return output


def _seed_summary(
    task: list[dict[str, str]],
    blank: list[dict[str, str]],
    geometry: list[dict[str, str]],
    dynamics: list[dict[str, str]],
    topology_normal: list[dict[str, str]],
    geometry_extras: list[dict[str, Any]],
    dynamics_extras: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    blank_map = {
        (row["model"], row["topology"], int(row["seed"]), int(row["horizon"])): row
        for row in blank
    }
    geometry_map = {
        (row["model"], row["topology"], int(row["seed"])): row for row in geometry
    }
    dynamics_map = {
        (row["model"], row["topology"], int(row["seed"])): row for row in dynamics
    }
    topology_normal_map = {
        (row["model"], row["topology"], int(row["seed"])): row
        for row in topology_normal
    }
    geometry_extra_map = {
        (row["model"], row["topology"], int(row["seed"])): row
        for row in geometry_extras
    }
    dynamics_extra_map = {
        (row["model"], row["topology"], int(row["seed"])): row
        for row in dynamics_extras
    }
    output = []
    for task_row in task:
        key = (
            task_row["model"],
            task_row["topology"],
            int(task_row["seed"]),
        )
        blank_row = blank_map[(*key, 4096)]
        geometry_row = geometry_map[key]
        dynamics_row = dynamics_map[key]
        topology_normal_row = topology_normal_map[key]
        geometry_extra = geometry_extra_map[key]
        dynamics_extra = dynamics_extra_map[key]
        normalized_error = _number(task_row["test_mean_error"])
        output.append(
            {
                "job_id": task_row["job_id"],
                "model": key[0],
                "topology": key[1],
                "seed": key[2],
                "selection_regime": task_row["selection_regime"],
                "parameters_total": int(task_row["parameters_total"]),
                "task_success": _bool(task_row["task_success"]),
                "validation_nmse_db": _number(task_row["validation_nmse_db"]),
                "test_error_normalized": normalized_error,
                "test_error_rad": math.pi * normalized_error,
                "blank4096_error_normalized": _number(blank_row["memory_error_mean"]),
                "blank4096_diverged": _bool(blank_row["diverged"]),
                "fiber_ratio": _number(geometry_row["fiber_ratio"]),
                "local_rank_full_fraction": _number(
                    geometry_row["local_tangent_rank_full_fraction"]
                ),
                "far_collapse_fraction": _number(
                    geometry_row["far_latent_hidden_collapse_fraction"]
                ),
                "endpoint_pca_participation_ratio": geometry_extra[
                    "endpoint_pca_participation_ratio"
                ],
                "one_step_tangent_gain": _number(
                    dynamics_row["one_step_tangent_gain_mean"]
                ),
                "one_step_normal_gain": _number(
                    dynamics_row["one_step_sampled_normal_gain_mean"]
                ),
                "topology_radial_hidden_q512": _number(
                    topology_normal_row["hidden_recovery_q512_median"]
                ),
                "topology_radial_output_q512": _number(
                    topology_normal_row["output_radial_recovery_q512_median"]
                ),
                "topology_radial_same_memory_error512": _number(
                    topology_normal_row["same_memory_error512_mean"]
                ),
                **{
                    name: dynamics_extra[name]
                    for name in (
                        "finite128_tangent_gain_mean",
                        "finite128_normal_gain_mean",
                        "recovery_q512_median",
                        "same_memory_error512_mean",
                    )
                },
            }
        )
    return output


def _group_summary(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "parameters_total",
        "validation_nmse_db",
        "test_error_normalized",
        "test_error_rad",
        "blank4096_error_normalized",
        "fiber_ratio",
        "local_rank_full_fraction",
        "far_collapse_fraction",
        "endpoint_pca_participation_ratio",
        "one_step_tangent_gain",
        "one_step_normal_gain",
        "finite128_tangent_gain_mean",
        "finite128_normal_gain_mean",
        "recovery_q512_median",
        "same_memory_error512_mean",
        "topology_radial_hidden_q512",
        "topology_radial_output_q512",
        "topology_radial_same_memory_error512",
    )
    output = []
    for topology in TOPOLOGIES:
        for model in MODELS:
            group = [
                row
                for row in seed_rows
                if row["model"] == model and row["topology"] == topology
            ]
            summary: dict[str, Any] = {
                "model": model,
                "model_label": LABELS[model],
                "topology": topology,
                "selection_regime": REGIMES[model],
                "seed_count": len(group),
                "task_success_count": sum(bool(row["task_success"]) for row in group),
                "blank_diverged_count": sum(bool(row["blank4096_diverged"]) for row in group),
            }
            for metric in metrics:
                summary[f"{metric}_median"] = _median(
                    row[metric] for row in group
                )
                values = np.asarray([_number(row[metric]) for row in group])
                values = values[np.isfinite(values)]
                summary[f"{metric}_min"] = float(np.min(values)) if values.size else np.nan
                summary[f"{metric}_max"] = float(np.max(values)) if values.size else np.nan
            output.append(summary)
    return output


def _save(fig: plt.Figure, figures: Path, stem: str) -> list[str]:
    outputs = []
    for suffix in ("png", "pdf"):
        path = figures / f"{stem}.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def _seed_scatter(
    axis: plt.Axes,
    seed_rows: list[dict[str, Any]],
    *,
    topology: str,
    metric: str,
    ylabel: str,
    log: bool = False,
) -> None:
    for index, model in enumerate(MODELS):
        group = [
            row
            for row in seed_rows
            if row["topology"] == topology and row["model"] == model
        ]
        values = np.asarray([_number(row[metric]) for row in group])
        jitter = np.linspace(-0.07, 0.07, len(values))
        for offset, value, row in zip(jitter, values, group):
            axis.scatter(
                index + offset,
                value,
                facecolors=COLORS[model] if row["task_success"] else "none",
                edgecolors=COLORS[model],
                linewidth=1.0,
                s=28,
                zorder=3,
            )
        median = np.nanmedian(values)
        axis.plot(
            (index - 0.22, index + 0.22),
            (median, median),
            color="#202020",
            linewidth=1.6,
            zorder=4,
        )
    axis.set_xticks(range(len(MODELS)), [LABELS[model] for model in MODELS], rotation=24)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.22)
    if log:
        axis.set_yscale("log")


def figure_overview(
    seed_rows: list[dict[str, Any]], figures: Path
) -> list[str]:
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.0))
    for column, topology in enumerate(TOPOLOGIES):
        _seed_scatter(
            axes[0, column],
            seed_rows,
            topology=topology,
            metric="test_error_rad",
            ylabel="Test geodesic error (rad)" if column == 0 else "",
            log=True,
        )
        _seed_scatter(
            axes[1, column],
            seed_rows,
            topology=topology,
            metric="blank4096_error_normalized",
            ylabel="Blank H=4096 error (normalized)" if column == 0 else "",
            log=True,
        )
        axes[0, column].set_title(TOPOLOGY_LABELS[topology])
    fig.suptitle(
        "Task transfer and long blank memory · all three seeds (bar = median)"
    )
    fig.text(
        0.5,
        0.01,
        "RNN/GRU/LSTM: ring-selected zero-retuning · CA-LRU/H-C: topology-specific validation selection",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.99,
        0.01,
        "hollow = failed task gate",
        ha="right",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _save(fig, figures, "fig_A_all_models_task_and_blank")


def figure_blank_curves(
    blank_rows: list[dict[str, str]], figures: Path
) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1), sharey=True)
    for axis, topology in zip(axes, TOPOLOGIES):
        for model in MODELS:
            group = [
                row
                for row in blank_rows
                if row["topology"] == topology and row["model"] == model
            ]
            horizons = sorted({int(row["horizon"]) for row in group})
            curves = []
            for seed in SEEDS:
                curve = np.asarray(
                    [
                        _number(
                            next(
                                row["memory_error_mean"]
                                for row in group
                                if int(row["seed"]) == seed
                                and int(row["horizon"]) == horizon
                            )
                        )
                        for horizon in horizons
                    ]
                )
                curves.append(curve)
                axis.plot(
                    horizons,
                    curve,
                    color=COLORS[model],
                    alpha=0.16,
                    linewidth=0.7,
                )
            axis.plot(
                horizons,
                np.nanmedian(np.stack(curves), axis=0),
                color=COLORS[model],
                linewidth=2.0,
                label=LABELS[model],
            )
        axis.set_xscale("symlog", linthresh=1)
        axis.set_yscale("log")
        axis.set_title(TOPOLOGY_LABELS[topology])
        axis.set_xlabel("Blank-input horizon")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Normalized memory error")
    axes[-1].legend(frameon=False, fontsize=8, ncol=1)
    fig.suptitle("Memory drift across horizon (thin = seed, thick = median)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, figures, "fig_B_all_models_blank_curves")


def figure_geometry(
    seed_rows: list[dict[str, Any]], figures: Path
) -> list[str]:
    specifications = (
        ("fiber_ratio", "Fiber ratio ↓", None),
        ("local_rank_full_fraction", "Correct local-rank fraction ↑", 1.0),
        ("endpoint_pca_participation_ratio", "Hidden-state PCA PR", None),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    offsets = np.linspace(-0.28, 0.28, len(MODELS))
    for axis, (metric, ylabel, ideal) in zip(axes, specifications):
        for topology_index, topology in enumerate(TOPOLOGIES):
            for offset, model in zip(offsets, MODELS):
                group = [
                    row
                    for row in seed_rows
                    if row["topology"] == topology and row["model"] == model
                ]
                values = np.asarray([_number(row[metric]) for row in group])
                for value, row in zip(values, group):
                    axis.scatter(
                        topology_index + offset,
                        value,
                        facecolors=COLORS[model] if row["task_success"] else "none",
                        edgecolors=COLORS[model],
                        linewidth=1.0,
                        s=26,
                        alpha=0.85,
                    )
                axis.plot(
                    (
                        topology_index + offset - 0.045,
                        topology_index + offset + 0.045,
                    ),
                    (np.nanmedian(values), np.nanmedian(values)),
                    color="#111111",
                    linewidth=1.3,
                )
        if ideal is not None:
            axis.axhline(ideal, color="#777777", linestyle="--", linewidth=0.9)
        axis.set_xticks(range(3), [TOPOLOGY_LABELS[topology] for topology in TOPOLOGIES])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=COLORS[model], label=LABELS[model])
        for model in MODELS
    ]
    axes[-1].legend(handles=handles, frameon=False, fontsize=8)
    fig.suptitle("Learned-memory geometry · all seeds, no task-success filtering")
    fig.text(
        0.99,
        0.01,
        "hollow = failed task gate",
        ha="right",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, figures, "fig_C_all_models_geometry")


def figure_dynamics(
    seed_rows: list[dict[str, Any]], figures: Path
) -> list[str]:
    specifications = (
        ("one_step_tangent_gain", "One-step tangent gain", 1.0, False),
        ("one_step_normal_gain", "One-step normal gain ↓", 1.0, False),
        ("finite128_tangent_gain_mean", "128-step tangent gain", 1.0, True),
        ("recovery_q512_median", "Normal-kick Q at H=512 ↓", 1.0, True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    offsets = np.linspace(-0.28, 0.28, len(MODELS))
    for axis, (metric, ylabel, ideal, log) in zip(axes.flat, specifications):
        for topology_index, topology in enumerate(TOPOLOGIES):
            for offset, model in zip(offsets, MODELS):
                group = [
                    row
                    for row in seed_rows
                    if row["topology"] == topology and row["model"] == model
                ]
                values = np.asarray([_number(row[metric]) for row in group])
                for value, row in zip(values, group):
                    axis.scatter(
                        topology_index + offset,
                        value,
                        facecolors=COLORS[model] if row["task_success"] else "none",
                        edgecolors=COLORS[model],
                        linewidth=1.0,
                        s=27,
                        alpha=0.85,
                    )
                axis.plot(
                    (
                        topology_index + offset - 0.045,
                        topology_index + offset + 0.045,
                    ),
                    (np.nanmedian(values), np.nanmedian(values)),
                    color="#111111",
                    linewidth=1.3,
                )
        axis.axhline(ideal, color="#777777", linestyle="--", linewidth=0.9)
        axis.set_xticks(range(3), [TOPOLOGY_LABELS[topology] for topology in TOPOLOGIES])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
        if log:
            axis.set_yscale("log")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=COLORS[model], label=LABELS[model])
        for model in MODELS
    ]
    axes[0, 1].legend(handles=handles, frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Attractor dynamics · ideal tangent ≈ 1, normal/recovery < 1")
    fig.text(
        0.99,
        0.01,
        "hollow = failed task gate",
        ha="right",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, figures, "fig_D_all_models_dynamics")


def figure_pca(
    task_rows: list[dict[str, str]],
    baseline: Path,
    selected: Path,
    figures: Path,
) -> list[str]:
    fig, axes = plt.subplots(3, 5, figsize=(15.0, 8.8))
    for row_index, topology in enumerate(TOPOLOGIES):
        for column, model in enumerate(MODELS):
            axis = axes[row_index, column]
            task_row = next(
                row
                for row in task_rows
                if row["topology"] == topology
                and row["model"] == model
                and int(row["seed"]) == 10
            )
            root = _analysis_for_row(task_row, baseline, selected)
            with np.load(
                root / "geometry" / "runs" / f"{task_row['job_id']}.npz",
                allow_pickle=False,
            ) as archive:
                score = np.array(archive["endpoint_pca_score"], copy=True)
                latent = np.array(archive["endpoint_latent"], copy=True)
            color = (
                latent[:, 0]
                if topology in {"s1", "t2"}
                else np.arctan2(latent[:, 1], latent[:, 0])
            )
            axis.scatter(
                score[:, 0],
                score[:, 1],
                c=color,
                cmap="twilight",
                s=5.5,
                alpha=0.75,
                rasterized=True,
            )
            if row_index == 0:
                axis.set_title(LABELS[model])
            if not _bool(task_row["task_success"]):
                for spine in axis.spines.values():
                    spine.set_edgecolor("#B33A3A")
                    spine.set_linewidth(1.3)
            if column == 0:
                axis.set_ylabel(TOPOLOGY_LABELS[topology], fontsize=12)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("Transported endpoint hidden-state PCA · representative seed 10")
    fig.text(
        0.5,
        0.015,
        "Color denotes intrinsic position (first torus angle for T²). Shape is diagnostic, not a proof of topology.",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.99,
        0.015,
        "red frame = failed task gate",
        ha="right",
        fontsize=8,
        color="#B33A3A",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _save(fig, figures, "fig_E_all_models_endpoint_pca")


def figure_recovery(
    dynamics_rows: list[dict[str, str]],
    baseline: Path,
    selected: Path,
    figures: Path,
) -> list[str]:
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.2), sharex=True)
    for column, topology in enumerate(TOPOLOGIES):
        for model in MODELS:
            recovery_curves = []
            memory_curves = []
            horizons = None
            group = [
                row
                for row in dynamics_rows
                if row["topology"] == topology and row["model"] == model
            ]
            for row in group:
                root = _analysis_for_row(row, baseline, selected)
                with np.load(
                    root / "dynamics" / "runs" / f"{row['job_id']}.npz",
                    allow_pickle=False,
                ) as archive:
                    horizons = np.array(archive["recovery_horizons"], copy=True)
                    radii = np.array(archive["kick_radii_relative"], copy=True)
                    radius_index = int(np.argmin(np.abs(radii - 0.05)))
                    recovery = np.median(
                        np.array(archive["recovery_ratio"], copy=True)[radius_index],
                        axis=(1, 2),
                    )
                    same_memory = np.mean(
                        np.array(archive["same_memory_error"], copy=True)[radius_index],
                        axis=(1, 2),
                    )
                recovery_curves.append(recovery)
                memory_curves.append(same_memory)
            if horizons is None:
                continue
            axes[0, column].plot(
                horizons,
                np.nanmedian(np.stack(recovery_curves), axis=0),
                color=COLORS[model],
                linewidth=2.0,
                label=LABELS[model],
            )
            axes[1, column].plot(
                horizons,
                np.nanmedian(np.stack(memory_curves), axis=0),
                color=COLORS[model],
                linewidth=2.0,
            )
        axes[0, column].axhline(1.0, color="#777777", linestyle="--", linewidth=0.9)
        axes[0, column].axhline(0.0, color="#BBBBBB", linestyle=":", linewidth=0.8)
        axes[0, column].set_title(TOPOLOGY_LABELS[topology])
        axes[0, column].set_xscale("symlog", linthresh=1)
        axes[1, column].set_xscale("symlog", linthresh=1)
        axes[1, column].set_yscale("log")
        axes[1, column].set_xlabel("Recovery horizon")
        axes[0, column].grid(alpha=0.22)
        axes[1, column].grid(alpha=0.22)
    axes[0, 0].set_ylabel("Normal-distance ratio Q")
    axes[1, 0].set_ylabel("Same-memory error")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Finite 5%-scale normal kick · medians across three seeds")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, figures, "fig_F_all_models_normal_recovery")


def figure_topology_radial_recovery(
    topology_normal_rows: list[dict[str, str]],
    baseline: Path,
    selected: Path,
    hc_topology_normal: Path | None,
    figures: Path,
) -> list[str]:
    """Plot the topology-defined radius diagnostic separately from random normals."""

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.2), sharex=True)
    for column, topology in enumerate(TOPOLOGIES):
        for model in MODELS:
            output_curves = []
            memory_curves = []
            horizons = None
            group = [
                row
                for row in topology_normal_rows
                if row["topology"] == topology and row["model"] == model
            ]
            for row in group:
                root = _analysis_for_row(
                    row, baseline, selected, hc_topology_normal
                )
                payload = json.loads(
                    (
                        root
                        / "topology_normal"
                        / "runs"
                        / f"{row['job_id']}.json"
                    ).read_text(encoding="utf-8")
                )
                horizons = np.asarray(payload["recovery_horizons"], dtype=int)
                output_curves.append(
                    np.asarray(payload["output_radial_recovery_q_median"], dtype=float)
                )
                memory_curves.append(
                    np.asarray(payload["same_memory_error_mean"], dtype=float)
                )
            if horizons is None:
                continue
            output_median = np.nanmedian(np.stack(output_curves), axis=0)
            memory_median = np.nanmedian(np.stack(memory_curves), axis=0)
            axes[0, column].plot(
                horizons,
                np.maximum(output_median, np.finfo(float).tiny),
                color=COLORS[model],
                linewidth=2.0,
                label=LABELS[model],
            )
            axes[1, column].plot(
                horizons,
                np.maximum(memory_median, np.finfo(float).tiny),
                color=COLORS[model],
                linewidth=2.0,
            )
        axes[0, column].axhline(1.0, color="#777777", linestyle=":", linewidth=0.9)
        axes[0, column].set_title(TOPOLOGY_LABELS[topology])
        axes[0, column].set_xscale("symlog", linthresh=1)
        axes[0, column].set_yscale("log")
        axes[1, column].set_xscale("symlog", linthresh=1)
        axes[1, column].set_yscale("log")
        axes[1, column].set_xlabel("Recovery horizon")
        axes[0, column].grid(alpha=0.22)
        axes[1, column].grid(alpha=0.22)
    axes[0, 0].set_ylabel("Output-radius recovery ratio Q")
    axes[1, 0].set_ylabel("Same-memory error")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Topology-defined 5%-scale radial kick · medians across three seeds"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, figures, "fig_G_all_models_topology_radial_recovery")


def compare(args: argparse.Namespace) -> None:
    baseline = args.baseline_analysis.expanduser().resolve(strict=True)
    selected = args.selected_analysis.expanduser().resolve(strict=True)
    hc_topology_normal = (
        args.hc_topology_normal_analysis.expanduser().resolve(strict=True)
        if args.hc_topology_normal_analysis is not None
        else None
    )
    output = args.output.expanduser().resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    task = _combined_rows(baseline, selected, "task_metrics.csv")
    blank = _combined_rows(baseline, selected, "blank/blank_metrics.csv")
    geometry = _combined_rows(baseline, selected, "geometry/geometry_metrics.csv")
    dynamics = _combined_rows(baseline, selected, "dynamics/dynamics_metrics.csv")
    topology_normal = _combined_topology_normal_rows(
        baseline, selected, task, hc_topology_normal
    )
    if (
        len(task) != 45
        or len(geometry) != 45
        or len(dynamics) != 45
        or len(topology_normal) != 45
    ):
        raise RuntimeError(
            "expected 45 task/geometry/dynamics/topology-normal rows "
            f"but found {len(task)}/{len(geometry)}/{len(dynamics)}/"
            f"{len(topology_normal)}"
        )
    if len(blank) != 270:
        raise RuntimeError(f"expected 270 blank rows but found {len(blank)}")

    geometry_extra = _geometry_extras(geometry, baseline, selected)
    dynamics_extra = _dynamics_extras(dynamics, baseline, selected)
    seed_rows = _seed_summary(
        task,
        blank,
        geometry,
        dynamics,
        topology_normal,
        geometry_extra,
        dynamics_extra,
    )
    group_rows = _group_summary(seed_rows)

    _write_csv(output / "task_metrics_all_models.csv", task)
    _write_csv(output / "blank_metrics_all_models.csv", blank)
    _write_csv(output / "geometry_metrics_all_models.csv", geometry)
    _write_csv(output / "dynamics_metrics_all_models.csv", dynamics)
    _write_csv(output / "topology_radial_metrics_all_models.csv", topology_normal)
    _write_csv(output / "seed_level_comparison.csv", seed_rows)
    _write_csv(output / "model_topology_summary.csv", group_rows)

    generated = [
        *figure_overview(seed_rows, figures),
        *figure_blank_curves(blank, figures),
        *figure_geometry(seed_rows, figures),
        *figure_dynamics(seed_rows, figures),
        *figure_pca(task, baseline, selected, figures),
        *figure_recovery(dynamics, baseline, selected, figures),
        *figure_topology_radial_recovery(
            topology_normal,
            baseline,
            selected,
            hc_topology_normal,
            figures,
        ),
    ]
    manifest = {
        "schema_version": 1,
        "models": list(MODELS),
        "topologies": list(TOPOLOGIES),
        "seeds": list(SEEDS),
        "baseline_analysis": str(baseline),
        "selected_analysis": str(selected),
        "hc_topology_normal_analysis": (
            str(hc_topology_normal) if hc_topology_normal is not None else None
        ),
        "fairness_note": (
            "RNN/GRU/LSTM are ring-selected zero-retuning baselines; "
            "CA-LRU and H-C are topology-specific validation-selected models."
        ),
        "task_and_structure_failures_included": True,
        "figures": generated,
    }
    (output / "comparison_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-analysis", type=Path, required=True)
    parser.add_argument("--selected-analysis", type=Path, required=True)
    parser.add_argument(
        "--hc-topology-normal-analysis",
        type=Path,
        help=(
            "Optional H-C finalist analysis root when selected-analysis contains "
            "the chosen H-C task/geometry rows but not topology-normal rows."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
