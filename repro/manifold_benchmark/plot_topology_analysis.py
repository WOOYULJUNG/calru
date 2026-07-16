"""Create the six frozen pilot figures from completed analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from repro.sagodi_protocol.artifacts import atomic_json

from .topology_analysis_common import MODEL_ORDER, TOPOLOGY_ORDER


COLORS = {"rnn": "#777777", "gru": "#4C78A8", "lstm": "#F58518", "hc": "#8E5EA2"}
LABELS = {"rnn": "RNN", "gru": "GRU", "lstm": "LSTM", "hc": "H-C"}
TOPOLOGY_LABELS = {"s1": r"$S^1$", "t2": r"$T^2$", "s2": r"$S^2$"}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save(fig, root: Path, name: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = root / f"{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def figure_task(analysis: Path, figures: Path) -> list[str]:
    rows = _csv(analysis / "task_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    for axis, topology in zip(axes, TOPOLOGY_ORDER):
        for seed in (10, 11, 12):
            points = []
            for model in MODEL_ORDER:
                row = next(
                    item
                    for item in rows
                    if item["topology"] == topology
                    and item["model"] == model
                    and int(item["seed"]) == seed
                )
                points.append(float(row["test_mean_error"]))
            axis.plot(range(4), points, color="#BBBBBB", linewidth=0.8, alpha=0.7, zorder=1)
            for x, (model, value) in enumerate(zip(MODEL_ORDER, points)):
                axis.scatter(x, value, color=COLORS[model], s=28, zorder=2)
        axis.set_title(TOPOLOGY_LABELS[topology])
        axis.set_xticks(range(4), [LABELS[item] for item in MODEL_ORDER])
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Normalized geodesic error")
    fig.suptitle("Zero-retuning topology transfer (paired trained-model seeds)")
    return _save(fig, figures, "fig_A_zero_retuning_task_transfer")


def figure_blank(analysis: Path, figures: Path) -> list[str]:
    rows = _csv(analysis / "blank" / "blank_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), sharey=True)
    for axis, topology in zip(axes, TOPOLOGY_ORDER):
        for model in MODEL_ORDER:
            model_rows = [
                row for row in rows if row["topology"] == topology and row["model"] == model
            ]
            horizons = sorted({int(row["horizon"]) for row in model_rows})
            curves = []
            for seed in (10, 11, 12):
                curve = [
                    float(
                        next(
                            row["memory_error_mean"]
                            for row in model_rows
                            if int(row["seed"]) == seed and int(row["horizon"]) == horizon
                        )
                    )
                    for horizon in horizons
                ]
                curves.append(curve)
                axis.plot(horizons, curve, color=COLORS[model], alpha=0.22, linewidth=0.8)
            axis.plot(
                horizons,
                np.mean(curves, axis=0),
                color=COLORS[model],
                linewidth=2.0,
                label=LABELS[model],
            )
        axis.set_xscale("symlog", linthresh=1.0)
        axis.set_title(TOPOLOGY_LABELS[topology])
        axis.set_xlabel("Blank horizon")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Normalized memory error")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Long blank-input memory")
    return _save(fig, figures, "fig_B_long_blank_memory")


def figure_geometry(analysis: Path, figures: Path) -> list[str]:
    task = _csv(analysis / "task_metrics.csv")
    fig, axes = plt.subplots(3, 4, figsize=(13.0, 9.5))
    for row_index, topology in enumerate(TOPOLOGY_ORDER):
        for column, model in enumerate(MODEL_ORDER):
            axis = axes[row_index, column]
            eligible = sorted(
                int(row["seed"])
                for row in task
                if row["topology"] == topology
                and row["model"] == model
                and row["task_success"] == "True"
            )
            if not eligible:
                axis.text(0.5, 0.5, "Task failure", ha="center", va="center")
                axis.set_axis_off()
                continue
            seed = eligible[0]
            job = f"pilot__{model}__{topology}__seed{seed}"
            with np.load(analysis / "geometry" / "runs" / f"{job}.npz", allow_pickle=False) as archive:
                score = np.array(archive["endpoint_pca_score"], copy=True)
                latent = np.array(archive["endpoint_latent"], copy=True)
            if topology in {"s1", "t2"}:
                color = latent[:, 0]
            else:
                color = np.arctan2(latent[:, 1], latent[:, 0])
            axis.scatter(score[:, 0], score[:, 1], c=color, cmap="twilight", s=5, alpha=0.75)
            axis.set_title(f"{LABELS[model]} (seed {seed})")
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(TOPOLOGY_LABELS[topology])
    fig.suptitle("Transported endpoint hidden-state PCA (visualization only)")
    return _save(fig, figures, "fig_C_learned_state_geometry")


def figure_dynamics(analysis: Path, figures: Path) -> list[str]:
    rows = _csv(analysis / "dynamics" / "dynamics_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), sharey=True)
    for axis, topology in zip(axes, TOPOLOGY_ORDER):
        for x, model in enumerate(MODEL_ORDER):
            group = [row for row in rows if row["topology"] == topology and row["model"] == model]
            tangent = [float(row["one_step_tangent_gain_mean"]) for row in group]
            normal = [float(row["one_step_sampled_normal_gain_mean"]) for row in group]
            axis.scatter(np.full(len(tangent), x - 0.10), tangent, color=COLORS[model], marker="o", s=25)
            axis.scatter(np.full(len(normal), x + 0.10), normal, color=COLORS[model], marker="x", s=30)
        axis.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--")
        axis.set_xticks(range(4), [LABELS[item] for item in MODEL_ORDER])
        axis.set_title(TOPOLOGY_LABELS[topology])
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("One-step gain (○ tangent, × sampled normal)")
    fig.suptitle("Local tangent–normal dynamics")
    return _save(fig, figures, "fig_D_tangent_normal_dynamics")


def figure_recovery(analysis: Path, figures: Path) -> list[str]:
    rows = _csv(analysis / "dynamics" / "dynamics_metrics.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), sharex=True)
    for column, topology in enumerate(TOPOLOGY_ORDER):
        for model in MODEL_ORDER:
            group = [row for row in rows if row["topology"] == topology and row["model"] == model]
            for row in group:
                job = row["job_id"]
                with np.load(analysis / "dynamics" / "runs" / f"{job}.npz", allow_pickle=False) as archive:
                    horizons = np.array(archive["recovery_horizons"])
                    recovery = np.array(archive["recovery_ratio"])[0].mean(axis=(1, 2))
                    memory = np.array(archive["same_memory_error"])[0].mean(axis=(1, 2))
                axes[0, column].plot(horizons, recovery, color=COLORS[model], alpha=0.25)
                axes[1, column].plot(horizons, memory, color=COLORS[model], alpha=0.25)
        axes[0, column].axhline(1.0, color="#999999", linestyle="--", linewidth=0.8)
        axes[0, column].set_title(TOPOLOGY_LABELS[topology])
        axes[1, column].set_xscale("symlog", linthresh=1.0)
        axes[1, column].set_xlabel("Recovery horizon")
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
    axes[0, 0].set_ylabel("Manifold-distance ratio Q")
    axes[1, 0].set_ylabel("Same-memory error")
    fig.suptitle("Finite normal-kick recovery (thin lines: trained-model seeds)")
    return _save(fig, figures, "fig_E_finite_perturbation_recovery")


def figure_retention(analysis: Path, figures: Path) -> list[str]:
    rows = _csv(analysis / "hc_retention" / "hc_retention_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    metrics = (
        ("final_base_lambda_median", "Final base λ median"),
        ("task_dynamic_lambda_median", "Task dynamic λ median"),
        ("normal_kick_lambda_response_absolute_mean", "|Δλ| under normal kick"),
    )
    for axis, (metric, label) in zip(axes, metrics):
        for x, topology in enumerate(TOPOLOGY_ORDER):
            values = [float(row[metric]) for row in rows if row["topology"] == topology]
            axis.scatter(np.full(len(values), x), values, color=COLORS["hc"], s=30)
        axis.set_xticks(range(3), [TOPOLOGY_LABELS[item] for item in TOPOLOGY_ORDER])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("H-C retention mechanism at the final checkpoint")
    return _save(fig, figures, "fig_F_hc_retention_mechanism")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.analysis_root.expanduser().resolve(strict=True)
    figures = analysis / "figures"
    figures.mkdir(exist_ok=True)
    outputs = {
        "figure_A": figure_task(analysis, figures),
        "figure_B": figure_blank(analysis, figures),
        "figure_C": figure_geometry(analysis, figures),
        "figure_D": figure_dynamics(analysis, figures),
        "figure_E": figure_recovery(analysis, figures),
        "figure_F": figure_retention(analysis, figures),
    }
    atomic_json(
        figures / "figure_manifest.json",
        {"schema_version": 1, "figures": outputs, "pca_is_decision_criterion": False},
    )


if __name__ == "__main__":
    main()
