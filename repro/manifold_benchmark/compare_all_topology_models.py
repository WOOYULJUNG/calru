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
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from repro.sagodi_protocol.artifacts import strict_json_load

from .topology_analysis_common import (
    RunRecord,
    atomic_npz,
    blank_snapshots,
    forward_endpoint_states,
    load_analysis_bank,
    load_model,
)


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


def _representative_checkpoint(
    task_rows: list[dict[str, str]],
    topology: str,
    model: str,
) -> dict[str, str]:
    """Prefer task-success seeds, then use the best available failed seed."""

    all_candidates = [
        row
        for row in task_rows
        if row["topology"] == topology
        and row["model"] == model
    ]
    if not all_candidates:
        raise RuntimeError(f"no checkpoints for {topology}/{model}")
    successful = [row for row in all_candidates if _bool(row["task_success"])]
    candidates = successful if successful else all_candidates
    return min(
        candidates,
        key=lambda row: (_number(row["validation_error"]), int(row["seed"])),
    )


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
        metadata = (
            {
                "Creator": "WOOYULJUNG/calru",
                "CreationDate": None,
                "ModDate": None,
            }
            if suffix == "pdf"
            else None
        )
        fig.savefig(path, dpi=240, bbox_inches="tight", metadata=metadata)
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


def _record_for_task(task_row: dict[str, str], analysis_root: Path) -> RunRecord:
    launcher = strict_json_load(analysis_root / "analysis_launcher_manifest.json")
    run_dir = (
        Path(launcher["run_root"]).expanduser().resolve(strict=True)
        / task_row["job_id"]
    )
    manifest = strict_json_load(run_dir / "manifest.json")
    result = strict_json_load(run_dir / "result.json")
    return RunRecord(
        run_dir=run_dir,
        job_id=task_row["job_id"],
        model_id=str(result["model_id"]),
        topology=str(result["topology"]),
        seed=int(result["replicate_seed"]),
        manifest=manifest,
        result=result,
    )


def _orient_pca(score: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Make SVD sign choices deterministic for stable paper figures."""

    oriented = np.array(score, copy=True)
    for component in range(oriented.shape[1]):
        loading = right[component]
        pivot = int(np.argmax(np.abs(loading)))
        if loading[pivot] < 0:
            oriented[:, component] *= -1.0
    return oriented


def _display_atlas_provenance(
    task_row: dict[str, str],
    analysis_root: Path,
) -> tuple[RunRecord, str, str]:
    record = _record_for_task(task_row, analysis_root)
    completion = strict_json_load(record.run_dir / "COMPLETED.json")
    checkpoint_sha256 = str(completion["artifacts"]["checkpoint.pt"])
    bank_manifest = strict_json_load(
        analysis_root / "banks" / "analysis_banks_manifest.json"
    )
    bank_sha256 = str(bank_manifest["banks"][task_row["topology"]]["sha256"])
    return record, checkpoint_sha256, bank_sha256


def _cached_display_atlas(
    task_row: dict[str, str],
    analysis_root: Path,
    output: Path,
    device: torch.device,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    record, checkpoint_sha256, bank_sha256 = _display_atlas_provenance(
        task_row, analysis_root
    )
    cache_path = (
        output
        / "display_atlas"
        / (
            f"seed{int(task_row['seed'])}__{task_row['model']}"
            f"__{task_row['topology']}.npz"
        )
    )
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            cached_horizons = tuple(
                int(value) for value in np.asarray(archive["horizons"]).tolist()
            )
            cache_matches = (
                int(np.asarray(archive["schema_version"]).item()) == 1
                and str(np.asarray(archive["job_id"]).item()) == task_row["job_id"]
                and str(np.asarray(archive["checkpoint_sha256"]).item())
                == checkpoint_sha256
                and str(np.asarray(archive["bank_sha256"]).item()) == bank_sha256
                and cached_horizons == horizons
            )
            if cache_matches:
                return {
                    "task_row": task_row,
                    "latent": np.array(archive["latent"], copy=True),
                    "states": {
                        horizon: np.array(
                            archive[f"state_h{horizon}"], copy=True
                        ).astype(np.float64)
                        for horizon in horizons
                    },
                    "cache_path": cache_path,
                    "checkpoint_sha256": checkpoint_sha256,
                    "bank_sha256": bank_sha256,
                }

    bank = load_analysis_bank(analysis_root / "banks", task_row["topology"])
    model, _ = load_model(record, device)
    initial = torch.as_tensor(bank["initializer_memory"], device=device)
    inputs = torch.as_tensor(bank["transport_inputs"], device=device)
    endpoint_state, _ = forward_endpoint_states(
        model,
        inputs,
        initial,
        chunk_size=128,
    )
    snapshots = blank_snapshots(model, endpoint_state, horizons)
    states = {
        horizon: snapshots[horizon].detach().cpu().numpy().astype(np.float64)
        for horizon in horizons
    }
    latent = bank["transport_endpoint_latent"].astype(np.float64)
    atomic_npz(
        cache_path,
        schema_version=np.asarray(1, dtype=np.int64),
        job_id=np.asarray(task_row["job_id"]),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        bank_sha256=np.asarray(bank_sha256),
        horizons=np.asarray(horizons, dtype=np.int64),
        latent=latent.astype(np.float32),
        **{
            f"state_h{horizon}": states[horizon].astype(np.float32)
            for horizon in horizons
        },
    )
    return {
        "task_row": task_row,
        "latent": latent,
        "states": states,
        "cache_path": cache_path,
        "checkpoint_sha256": checkpoint_sha256,
        "bank_sha256": bank_sha256,
    }


def _build_display_atlases(
    task_rows: list[dict[str, str]],
    baseline: Path,
    selected: Path,
    output: Path,
    device: torch.device,
    horizons: tuple[int, ...],
) -> dict[tuple[str, str], dict[str, Any]]:
    atlases: dict[tuple[str, str], dict[str, Any]] = {}
    for topology in TOPOLOGIES:
        for model in MODELS:
            task_row = _representative_checkpoint(
                task_rows, topology, model
            )
            root = _analysis_for_row(task_row, baseline, selected)
            atlases[(topology, model)] = _cached_display_atlas(
                task_row,
                root,
                output,
                device,
                horizons,
            )
    return atlases


def _single_snapshot_pca(states: np.ndarray) -> np.ndarray:
    centered = states - states.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    return _orient_pca(centered @ right[:3].T, right[:3])


def _fixed_initial_snapshot_pca(
    states: dict[int, np.ndarray],
    horizons: tuple[int, ...],
) -> dict[int, np.ndarray]:
    # Fit the origin and PC axes once at H=0. Keeping both fixed makes bulk
    # motion, contraction, expansion, folding, and convergence to discrete
    # states directly comparable across blank-input horizons.
    initial = states[horizons[0]]
    center = initial.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(initial - center, full_matrices=False)
    return {
        horizon: _orient_pca(
            (states[horizon] - center) @ right[:3].T,
            right[:3],
        )
        for horizon in horizons
    }


def _sphere_neighbor_edges(latent: np.ndarray, neighbors: int = 3) -> np.ndarray:
    cosine = np.clip(latent @ latent.T, -1.0, 1.0)
    np.fill_diagonal(cosine, -np.inf)
    nearest = np.argpartition(-cosine, kth=neighbors - 1, axis=1)[:, :neighbors]
    edges = {
        tuple(sorted((index, int(neighbor))))
        for index, row in enumerate(nearest)
        for neighbor in row
    }
    return np.asarray(sorted(edges), dtype=np.int64)


def _draw_atlas_structure(
    axis,
    topology: str,
    score: np.ndarray,
    latent: np.ndarray,
) -> None:
    line_color = "#3C4858"
    if topology == "s1":
        order = np.argsort(latent[:, 0])
        closed = np.concatenate((order, order[:1]))
        axis.plot(
            score[closed, 0],
            score[closed, 1],
            score[closed, 2],
            color=line_color,
            linewidth=0.65,
            alpha=0.55,
        )
        return
    if topology == "t2":
        side = int(round(math.sqrt(score.shape[0])))
        if side * side != score.shape[0]:
            raise ValueError("T2 display atlas must be a square grid")
        grid = score.reshape(side, side, 3)
        stride = max(1, side // 16)
        for index in range(0, side, stride):
            first = np.concatenate((grid[index], grid[index, :1]), axis=0)
            second = np.concatenate((grid[:, index], grid[:1, index]), axis=0)
            axis.plot(*first.T, color=line_color, linewidth=0.38, alpha=0.35)
            axis.plot(*second.T, color=line_color, linewidth=0.38, alpha=0.35)
        return
    edges = _sphere_neighbor_edges(latent)
    segments = score[edges]
    axis.add_collection3d(
        Line3DCollection(
            segments,
            colors=line_color,
            linewidths=0.22,
            alpha=0.20,
        )
    )


def _equalize_3d(axis, score: np.ndarray) -> None:
    lower = score.min(axis=0)
    upper = score.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.5 * float(np.max(upper - lower)), np.finfo(float).eps)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def figure_pca(
    atlases: dict[tuple[str, str], dict[str, Any]],
    figures: Path,
) -> list[str]:
    fig, axes = plt.subplots(
        3,
        5,
        figsize=(16.0, 10.0),
        subplot_kw={"projection": "3d"},
    )
    for row_index, topology in enumerate(TOPOLOGIES):
        for column, model in enumerate(MODELS):
            axis = axes[row_index, column]
            atlas = atlases.get((topology, model))
            if atlas is None:
                axis.text2D(
                    0.5,
                    0.5,
                    "No task-success seed",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#B33A3A",
                )
                if row_index == 0:
                    axis.set_title(LABELS[model])
                if column == 0:
                    axis.text2D(
                        -0.12,
                        0.5,
                        TOPOLOGY_LABELS[topology],
                        transform=axis.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=12,
                    )
                axis.set_axis_off()
                continue
            task_row = atlas["task_row"]
            latent = atlas["latent"]
            score = _single_snapshot_pca(atlas["states"][0])
            color = (
                latent[:, 0]
                if topology in {"s1", "t2"}
                else np.arctan2(latent[:, 1], latent[:, 0])
            )
            _draw_atlas_structure(axis, topology, score, latent)
            axis.scatter3D(
                score[:, 0],
                score[:, 1],
                score[:, 2],
                c=color,
                cmap="twilight",
                s=3.2,
                alpha=0.68,
                depthshade=False,
                rasterized=True,
            )
            if row_index == 0:
                axis.set_title(LABELS[model])
            representative_label = f"seed {int(task_row['seed'])}"
            representative_color = "#333333"
            if not _bool(task_row["task_success"]):
                representative_label += " · best available\nfailed task gate"
                representative_color = "#B33A3A"
            axis.text2D(
                0.5,
                0.99,
                representative_label,
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=7,
                color=representative_color,
            )
            if column == 0:
                axis.text2D(
                    -0.12,
                    0.5,
                    TOPOLOGY_LABELS[topology],
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                )
            _equalize_3d(axis, score)
            axis.view_init(elev=22, azim=-55)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_zticks([])
            axis.set_xlabel("")
            axis.set_ylabel("")
            axis.set_zlabel("")
            axis.grid(False)
            for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
                pane.set_alpha(0.0)
            axis.set_axis_off()
    fig.suptitle(
        "Transported endpoint hidden-state PCA (PC1–PC3) · 1,024-point atlas"
        " · success-first representative seed per model"
    )
    fig.text(
        0.5,
        0.015,
        "Lines show atlas adjacency (32×32 periodic grid for T²); color denotes intrinsic position. Shape is diagnostic, not a proof of topology.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _save(fig, figures, "fig_E_all_models_endpoint_pca")


def figure_pca_evolution(
    atlases: dict[tuple[str, str], dict[str, Any]],
    figures: Path,
    horizons: tuple[int, ...],
) -> list[str]:
    generated: list[str] = []
    for topology in TOPOLOGIES:
        fig, axes = plt.subplots(
            len(MODELS),
            len(horizons),
            figsize=(3.25 * len(horizons), 14.8),
            subplot_kw={"projection": "3d"},
        )
        for row_index, model in enumerate(MODELS):
            atlas = atlases.get((topology, model))
            if atlas is None:
                for column, _ in enumerate(horizons):
                    axis = axes[row_index, column]
                    axis.set_axis_off()
                    if column == 0:
                        axis.text2D(
                            0.5,
                            0.5,
                            f"{LABELS[model]}\nNo task-success seed",
                            transform=axis.transAxes,
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="#B33A3A",
                        )
                continue
            task_row = atlas["task_row"]
            latent = atlas["latent"]
            scores = _fixed_initial_snapshot_pca(atlas["states"], horizons)
            all_scores = np.concatenate(
                [scores[horizon] for horizon in horizons], axis=0
            )
            color = (
                latent[:, 0]
                if topology in {"s1", "t2"}
                else np.arctan2(latent[:, 1], latent[:, 0])
            )
            for column, horizon in enumerate(horizons):
                axis = axes[row_index, column]
                score = scores[horizon]
                _draw_atlas_structure(axis, topology, score, latent)
                axis.scatter3D(
                    score[:, 0],
                    score[:, 1],
                    score[:, 2],
                    c=color,
                    cmap="twilight",
                    s=3.0,
                    alpha=0.66,
                    depthshade=False,
                    rasterized=True,
                )
                if row_index == 0:
                    axis.set_title(f"blank H={horizon}")
                if column == 0:
                    successful = _bool(task_row["task_success"])
                    axis.text2D(
                        -0.13,
                        0.5,
                        f"{LABELS[model]} · seed {int(task_row['seed'])}"
                        + ("" if successful else " · fallback"),
                        transform=axis.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="#111111" if successful else "#B33A3A",
                    )
                _equalize_3d(axis, all_scores)
                axis.view_init(elev=22, azim=-55)
                axis.set_axis_off()
        fig.suptitle(
            f"{TOPOLOGY_LABELS[topology]} hidden-atlas evolution in fixed H=0 PC1–PC3"
            " · 1,024 points · model-specific representatives"
        )
        fig.text(
            0.5,
            0.012,
            "The H=0 PCA origin and axes stay fixed at every horizon; common row limits expose translation, contraction, and splitting. "
            "Lines track the same intrinsic neighbors; red fallback labels mark models with no task-success seed.",
            ha="center",
            fontsize=8.5,
        )
        fig.tight_layout(rect=(0.025, 0.035, 1, 0.965))
        generated.extend(
            _save(
                fig,
                figures,
                f"fig_H_all_models_{topology}_pca_evolution",
            )
        )
    return generated


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
    device = torch.device(args.device)
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

    display_horizons = (0, 128, 512, 1024, 2048, 4096)
    display_atlases = _build_display_atlases(
        task,
        baseline,
        selected,
        output,
        device,
        display_horizons,
    )
    representative_seeds = {
        topology: {
            model: (
                int(display_atlases[(topology, model)]["task_row"]["seed"])
                if (topology, model) in display_atlases
                else None
            )
            for model in MODELS
        }
        for topology in TOPOLOGIES
    }
    generated = [
        *figure_overview(seed_rows, figures),
        *figure_blank_curves(blank, figures),
        *figure_geometry(seed_rows, figures),
        *figure_dynamics(seed_rows, figures),
        *figure_pca(display_atlases, figures),
        *figure_pca_evolution(display_atlases, figures, display_horizons),
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
        "display_atlas": {
            "representative_seed_policy": (
                "lowest validation_error among task_success=true seeds; "
                "if none succeed, lowest-validation-error checkpoint "
                "is shown as an explicit failed-task fallback"
            ),
            "representative_seeds": representative_seeds,
            "points": 1024,
            "blank_horizons": list(display_horizons),
            "pca_evolution_alignment": (
                "H=0 fixed center and PC1-PC3 axes, with common axis limits "
                "within each model-topology row"
            ),
            "cache_files": [
                str(atlas["cache_path"].relative_to(output))
                for atlas in display_atlases.values()
            ],
        },
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
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device used to reconstruct the full 1,024-point endpoint atlas.",
    )
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
