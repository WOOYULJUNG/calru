"""Figures for the selected 3-seed CA-LRU versus H-C topology analysis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from repro.sagodi_protocol.artifacts import atomic_json


MODELS = ("calru", "hc")
TOPOLOGIES = ("s1", "t2", "s2")
COLORS = {"calru": "#2A9D8F", "hc": "#8E5EA2"}
LABELS = {"calru": "CA-LRU", "hc": "H-C"}
TOPOLOGY_LABELS = {"s1": r"$S^1$", "t2": r"$T^2$", "s2": r"$S^2$"}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: str | None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _save(fig, root: Path, name: str) -> list[str]:
    outputs = []
    for suffix in ("png", "pdf"):
        path = root / f"{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def figure_task(analysis: Path, figures: Path) -> list[str]:
    rows = _csv(analysis / "task_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), sharey=True)
    for axis, topology in zip(axes, TOPOLOGIES):
        for seed in (10, 11, 12):
            group = [
                next(
                    row
                    for row in rows
                    if row["topology"] == topology
                    and row["model"] == model
                    and int(row["seed"]) == seed
                )
                for model in MODELS
            ]
            values = [_number(row["test_mean_error"]) for row in group]
            axis.plot((0, 1), values, color="#B8B8B8", linewidth=0.9, zorder=1)
            for x, model, value in zip((0, 1), MODELS, values):
                axis.scatter(x, value, color=COLORS[model], s=34, zorder=2)
        axis.set_xticks((0, 1), [LABELS[model] for model in MODELS])
        axis.set_title(TOPOLOGY_LABELS[topology])
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Test normalized geodesic error")
    fig.suptitle("Selected topology-specific models with RP (paired seeds)")
    return _save(fig, figures, "fig_A_selected_test_task")


def figure_blank(analysis: Path, figures: Path) -> list[str]:
    rows = _csv(analysis / "blank" / "blank_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharey=True)
    for axis, topology in zip(axes, TOPOLOGIES):
        for model in MODELS:
            group = [
                row
                for row in rows
                if row["topology"] == topology and row["model"] == model
            ]
            horizons = sorted({int(row["horizon"]) for row in group})
            curves = []
            for seed in (10, 11, 12):
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
                axis.plot(horizons, curve, color=COLORS[model], alpha=0.25, linewidth=0.8)
            axis.plot(
                horizons,
                np.nanmedian(np.stack(curves), axis=0),
                color=COLORS[model],
                linewidth=2.2,
                label=LABELS[model],
            )
        axis.set_xscale("symlog", linthresh=1)
        axis.set_title(TOPOLOGY_LABELS[topology])
        axis.set_xlabel("Blank horizon")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Normalized memory error")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Long blank-input memory (both models use RP)")
    return _save(fig, figures, "fig_B_selected_blank_memory")


def figure_geometry(analysis: Path, figures: Path) -> list[str]:
    task = _csv(analysis / "task_metrics.csv")
    fig, axes = plt.subplots(3, 2, figsize=(7.5, 9.2))
    for row_index, topology in enumerate(TOPOLOGIES):
        for column, model in enumerate(MODELS):
            axis = axes[row_index, column]
            task_row = next(
                row
                for row in task
                if row["topology"] == topology
                and row["model"] == model
                and int(row["seed"]) == 10
            )
            job_id = task_row["job_id"]
            with np.load(
                analysis / "geometry" / "runs" / f"{job_id}.npz",
                allow_pickle=False,
            ) as archive:
                score = np.array(archive["endpoint_pca_score"], copy=True)
                latent = np.array(archive["endpoint_latent"], copy=True)
            color = (
                latent[:, 0]
                if topology in {"s1", "t2"}
                else np.arctan2(latent[:, 1], latent[:, 0])
            )
            axis.scatter(score[:, 0], score[:, 1], c=color, cmap="twilight", s=6, alpha=0.75)
            axis.set_title(f"{LABELS[model]} · {TOPOLOGY_LABELS[topology]} · seed 10")
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("Transported endpoint hidden-state PCA")
    return _save(fig, figures, "fig_C_selected_endpoint_pca")


def figure_dynamics(analysis: Path, figures: Path) -> list[str]:
    rows = _csv(analysis / "dynamics" / "dynamics_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8), sharey=True)
    for axis, topology in zip(axes, TOPOLOGIES):
        for x, model in enumerate(MODELS):
            group = [
                row
                for row in rows
                if row["topology"] == topology and row["model"] == model
            ]
            tangent = [_number(row["one_step_tangent_gain_mean"]) for row in group]
            normal = [_number(row["one_step_sampled_normal_gain_mean"]) for row in group]
            axis.scatter(
                np.full(len(tangent), x - 0.08),
                tangent,
                color=COLORS[model],
                marker="o",
                s=32,
            )
            axis.scatter(
                np.full(len(normal), x + 0.08),
                normal,
                color=COLORS[model],
                marker="x",
                s=38,
            )
        axis.axhline(1.0, color="#777777", linestyle="--", linewidth=0.9, label="neutral gain")
        axis.set_xticks((0, 1), [LABELS[model] for model in MODELS])
        axis.set_title(TOPOLOGY_LABELS[topology])
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("One-step gain (○ tangent, × normal)")
    fig.suptitle("Local tangent–normal dynamics; ideal: tangent ≈ 1, normal < 1")
    return _save(fig, figures, "fig_D_selected_tangent_normal")


def figure_recovery(analysis: Path, figures: Path) -> list[str]:
    rows = _csv(analysis / "dynamics" / "dynamics_metrics.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.0), sharex=True)
    for column, topology in enumerate(TOPOLOGIES):
        for model in MODELS:
            recovery_curves = []
            memory_curves = []
            horizons = None
            for row in rows:
                if row["topology"] != topology or row["model"] != model:
                    continue
                with np.load(
                    analysis / "dynamics" / "runs" / f"{row['job_id']}.npz",
                    allow_pickle=False,
                ) as archive:
                    horizons = np.array(archive["recovery_horizons"], copy=True)
                    recovery = np.median(
                        np.array(archive["recovery_ratio"], copy=True)[0],
                        axis=(1, 2),
                    )
                    memory = np.mean(
                        np.array(archive["same_memory_error"], copy=True)[0],
                        axis=(1, 2),
                    )
                recovery_curves.append(recovery)
                memory_curves.append(memory)
                axes[0, column].plot(horizons, recovery, color=COLORS[model], alpha=0.22)
                axes[1, column].plot(horizons, memory, color=COLORS[model], alpha=0.22)
            if horizons is not None:
                axes[0, column].plot(
                    horizons,
                    np.median(np.stack(recovery_curves), axis=0),
                    color=COLORS[model],
                    linewidth=2.1,
                    label=LABELS[model],
                )
                axes[1, column].plot(
                    horizons,
                    np.median(np.stack(memory_curves), axis=0),
                    color=COLORS[model],
                    linewidth=2.1,
                )
        axes[0, column].axhline(1.0, color="#777777", linestyle="--", linewidth=0.9)
        axes[0, column].axhline(0.0, color="#BBBBBB", linestyle=":", linewidth=0.8)
        axes[0, column].set_title(TOPOLOGY_LABELS[topology])
        axes[1, column].set_xscale("symlog", linthresh=1)
        axes[1, column].set_xlabel("Recovery horizon")
        axes[0, column].grid(alpha=0.22)
        axes[1, column].grid(alpha=0.22)
    axes[0, 0].set_ylabel("Normal-distance ratio Q")
    axes[1, 0].set_ylabel("Same-memory error")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Finite 5%-scale normal-kick recovery; ideal Q → 0")
    return _save(fig, figures, "fig_E_selected_normal_recovery")


def figure_retention(analysis: Path, figures: Path) -> list[str]:
    rows = _csv(analysis / "hc_retention" / "hc_retention_metrics.csv")
    metrics = (
        ("final_base_lambda_median", "Final base λ median"),
        ("task_dynamic_lambda_median", "Task-state λ median"),
        ("normal_kick_lambda_response_absolute_mean", "Mean |Δλ| under normal kick"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8))
    for axis, (metric, ylabel) in zip(axes, metrics):
        for topology_index, topology in enumerate(TOPOLOGIES):
            for model_index, model in enumerate(MODELS):
                values = [
                    _number(row[metric])
                    for row in rows
                    if row["topology"] == topology and row["model"] == model
                ]
                x = topology_index + (model_index - 0.5) * 0.18
                axis.scatter(np.full(len(values), x), values, color=COLORS[model], s=30)
        axis.set_xticks(range(3), [TOPOLOGY_LABELS[item] for item in TOPOLOGIES])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Retention mechanism at the final checkpoint")
    return _save(fig, figures, "fig_F_selected_retention")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.analysis_root.expanduser().resolve(strict=True)
    figures = analysis / "figures"
    figures.mkdir(exist_ok=True)
    outputs = {
        "task": figure_task(analysis, figures),
        "blank": figure_blank(analysis, figures),
        "geometry": figure_geometry(analysis, figures),
        "dynamics": figure_dynamics(analysis, figures),
        "recovery": figure_recovery(analysis, figures),
        "retention": figure_retention(analysis, figures),
    }
    atomic_json(
        figures / "figure_manifest.json",
        {
            "schema_version": 1,
            "models": list(MODELS),
            "topologies": list(TOPOLOGIES),
            "pca_is_decision_criterion": False,
            "figures": outputs,
        },
    )


if __name__ == "__main__":
    main()
