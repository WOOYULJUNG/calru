"""Create comparison figures and a compact report for selected static-gate runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = {
    "untied_rnn_rp": "Shared field",
    "split_rnn_rp": "Split fields",
}
CONDITION_LABELS = {
    "no_rp": "no RP",
    "rp": "+ RP",
}
COLORS = {
    ("untied_rnn_rp", "no_rp"): "#8b95a5",
    ("untied_rnn_rp", "rp"): "#4f79a7",
    ("split_rnn_rp", "no_rp"): "#e0a24b",
    ("split_rnn_rp", "rp"): "#c84e4e",
}
ORDER = [
    ("split_rnn_rp", "no_rp"),
    ("split_rnn_rp", "rp"),
]


def _load(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/dynamics.json")):
        data = json.loads(path.read_text())
        rows.append(
            {
                "condition": path.parent.name.split("__", 1)[0],
                **data,
            }
        )
    if len(rows) != 12:
        raise ValueError(f"expected 12 dynamics results, found {len(rows)}")
    return rows


def _values(
    rows: list[dict[str, Any]],
    model: str,
    condition: str,
    topology: str,
    accessor,
) -> np.ndarray:
    group = [
        accessor(row)
        for row in rows
        if row["model"] == model
        and row["condition"] == condition
        and row["topology"] == topology
    ]
    if len(group) != 3:
        raise ValueError(
            f"expected 3 seeds for {model}/{condition}/{topology}, found {len(group)}"
        )
    return np.asarray(group, dtype=float)


def _comparison_figure(rows: list[dict[str, Any]], output: Path) -> None:
    metrics = [
        (
            "Task error",
            lambda row: row["task_intrinsic_radians"],
            "Intrinsic error (rad)",
            True,
            None,
        ),
        (
            "Blank memory, H=2048",
            lambda row: row["blank_manifold_evolution"]["2048"][
                "decoded_memory_intrinsic_radians"
            ],
            "Intrinsic error (rad)",
            True,
            None,
        ),
        (
            "Tangent neutrality",
            lambda row: abs(row["tangent_singular_mean"] - 1.0),
            r"$|\sigma_{tan}-1|$",
            True,
            None,
        ),
        (
            "Worst local normal gain",
            lambda row: row["normal_max_singular_mean"],
            r"mean $\sigma_{normal,max}$",
            False,
            1.0,
        ),
        (
            "Tangent-normal gap",
            lambda row: row["tangent_normal_gap_mean"],
            r"$\sigma_{tan}-\sigma_{normal,max}$",
            False,
            0.0,
        ),
        (
            "Finite normal recovery, H=512",
            lambda row: row["finite_local_normal_recovery"]["512"][
                "distance_ratio_median"
            ],
            "distance ratio",
            True,
            1.0,
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), constrained_layout=True)
    rng = np.random.default_rng(7)
    for axis, (title, accessor, ylabel, log_scale, reference) in zip(
        axes.flat, metrics
    ):
        positions: list[float] = []
        tick_labels: list[str] = []
        position = 0.0
        for topology in ("s1", "t2"):
            for model, condition in ORDER:
                values = _values(rows, model, condition, topology, accessor)
                color = COLORS[(model, condition)]
                axis.scatter(
                    position + rng.uniform(-0.075, 0.075, size=len(values)),
                    values,
                    color=color,
                    s=30,
                    alpha=0.8,
                    zorder=3,
                )
                median = float(np.median(values))
                q1, q3 = np.quantile(values, [0.25, 0.75])
                axis.plot(
                    [position - 0.25, position + 0.25],
                    [median, median],
                    color=color,
                    linewidth=3,
                    zorder=4,
                )
                axis.vlines(position, q1, q3, color=color, linewidth=2)
                positions.append(position)
                tick_labels.append(
                    f"{topology.upper()}\n{MODEL_LABELS[model]}\n"
                    f"{CONDITION_LABELS[condition]}"
                )
                position += 1.0
            position += 0.7
        if log_scale:
            axis.set_yscale("log")
        if reference is not None:
            axis.axhline(reference, color="#333333", linestyle="--", linewidth=1)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, tick_labels, fontsize=7)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Static-gate memory: shared vs split autonomous/writer fields",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output / "fig_selected_static_gate_dynamics.png", dpi=220)
    fig.savefig(output / "fig_selected_static_gate_dynamics.pdf")
    plt.close(fig)


def _blank_evolution_figure(rows: list[dict[str, Any]], output: Path) -> None:
    metrics = [
        ("best_global_scale", "Best global scale", False, 1.0),
        ("global_scaling_residual", "Scaling residual", True, None),
        ("pairwise_shape_distortion_std", "Pairwise shape distortion", True, None),
        ("state_direction_cosine_median", "Direction cosine", False, 1.0),
    ]
    horizons = np.asarray([128, 512, 2048])
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.2), constrained_layout=True)
    for row_index, topology in enumerate(("s1", "t2")):
        for axis, (key, title, log_scale, reference) in zip(
            axes[row_index], metrics
        ):
            for model, condition in ORDER:
                curves = np.asarray(
                    [
                        [
                            row["blank_manifold_evolution"][str(horizon)][key]
                            for horizon in horizons
                        ]
                        for row in rows
                        if row["model"] == model
                        and row["condition"] == condition
                        and row["topology"] == topology
                    ],
                    dtype=float,
                )
                median = np.median(curves, axis=0)
                q1, q3 = np.quantile(curves, [0.25, 0.75], axis=0)
                label = (
                    f"{MODEL_LABELS[model]} {CONDITION_LABELS[condition]}"
                    if row_index == 0 and axis is axes[0, 0]
                    else None
                )
                axis.plot(
                    horizons,
                    median,
                    marker="o",
                    color=COLORS[(model, condition)],
                    label=label,
                )
                axis.fill_between(
                    horizons,
                    q1,
                    q3,
                    color=COLORS[(model, condition)],
                    alpha=0.12,
                )
            axis.set_xscale("log", base=2)
            if log_scale:
                axis.set_yscale("log")
            if reference is not None:
                axis.axhline(
                    reference, color="#333333", linestyle="--", linewidth=1
                )
            axis.set_title(f"{topology.upper()} — {title}", fontsize=10)
            axis.set_xlabel("Blank horizon")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Blank-input manifold evolution (median and interquartile range over seeds)",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output / "fig_blank_manifold_evolution.png", dpi=220)
    fig.savefig(output / "fig_blank_manifold_evolution.pdf")
    plt.close(fig)


def _lambda_figure(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    bins = np.linspace(0.0, 1.0, 31)
    model = "split_rnn_rp"
    for axis, topology in zip(axes, ("s1", "t2")):
        for condition in ("no_rp", "rp"):
            values = np.concatenate(
                [
                    np.asarray(row["lambda_values"], dtype=float)
                    for row in rows
                    if row["model"] == model
                    and row["condition"] == condition
                    and row["topology"] == topology
                ]
            )
            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color=COLORS[(model, condition)],
                label=CONDITION_LABELS[condition],
            )
        axis.set_title(f"{topology.upper()} — {MODEL_LABELS[model]}")
        axis.set_xlabel(r"Retention $\lambda_j$")
        axis.set_ylabel("Density")
        axis.legend()
        axis.grid(alpha=0.2)
    fig.suptitle("Learned retention-coordinate distributions", fontweight="bold")
    fig.savefig(output / "fig_lambda_distributions.png", dpi=220)
    fig.savefig(output / "fig_lambda_distributions.pdf")
    plt.close(fig)


def _write_report(rows: list[dict[str, Any]], output: Path) -> None:
    summary_rows: list[dict[str, Any]] = []
    for topology in ("s1", "t2"):
        for model, condition in ORDER:
            group = [
                row
                for row in rows
                if row["model"] == model
                and row["condition"] == condition
                and row["topology"] == topology
            ]
            summary_rows.append(
                {
                    "topology": topology,
                    "model": model,
                    "condition": condition,
                    "task_rad_median": np.median(
                        [row["task_intrinsic_radians"] for row in group]
                    ),
                    "blank2048_rad_median": np.median(
                        [
                            row["blank_manifold_evolution"]["2048"][
                                "decoded_memory_intrinsic_radians"
                            ]
                            for row in group
                        ]
                    ),
                    "tangent_gain_median": np.median(
                        [row["tangent_singular_mean"] for row in group]
                    ),
                    "normal_gain_median": np.median(
                        [row["normal_max_singular_mean"] for row in group]
                    ),
                    "gap_median": np.median(
                        [row["tangent_normal_gap_mean"] for row in group]
                    ),
                    "normal_recovery_h512_median": np.median(
                        [
                            row["finite_local_normal_recovery"]["512"][
                                "distance_ratio_median"
                            ]
                            for row in group
                        ]
                    ),
                    "global_scale_h2048_median": np.median(
                        [
                            row["blank_manifold_evolution"]["2048"][
                                "best_global_scale"
                            ]
                            for row in group
                        ]
                    ),
                    "shape_distortion_h2048_median": np.median(
                        [
                            row["blank_manifold_evolution"]["2048"][
                                "pairwise_shape_distortion_std"
                            ]
                            for row in group
                        ]
                    ),
                }
            )
    with (output / "selected_dynamics_medians.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# Static-gate split-field experiment — selected 3-seed analysis",
        "",
        "All comparisons use 7,000 total task-gradient updates. The RP condition and "
        "its no-RP control resume from the same 5,000-update checkpoint with the "
        "same optimizer state and data stream.",
        "",
        "| Topology | Model | RP | Task rad | Blank-2048 rad | Tangent gain | "
        "Normal gain | Gap | Normal recovery | Global scale | Shape distortion |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['topology'].upper()} | {MODEL_LABELS[row['model']]} | "
            f"{CONDITION_LABELS[row['condition']]} | "
            f"{row['task_rad_median']:.4g} | "
            f"{row['blank2048_rad_median']:.4g} | "
            f"{row['tangent_gain_median']:.4g} | "
            f"{row['normal_gain_median']:.4g} | "
            f"{row['gap_median']:.4g} | "
            f"{row['normal_recovery_h512_median']:.4g} | "
            f"{row['global_scale_h2048_median']:.4g} | "
            f"{row['shape_distortion_h2048_median']:.4g} |"
        )
    lines.extend(
        [
            "",
            "Interpretation rules:",
            "",
            "- tangent gain close to 1 indicates local tangent preservation;",
            "- normal gain below 1 and a positive tangent-normal gap indicate local "
            "normal contraction;",
            "- finite normal-recovery ratio below 1 confirms contraction beyond the "
            "infinitesimal Jacobian test;",
            "- global scale near 1 with low residual supports a stationary state "
            "manifold; scale far below 1 with preserved direction instead supports "
            "directional/projective memory.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_selected_dynamics_v1"
        ),
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = _load(root)
    _comparison_figure(rows, figures)
    _blank_evolution_figure(rows, figures)
    _lambda_figure(rows, figures)
    _write_report(rows, root)
    print(root / "RESULTS.md")


if __name__ == "__main__":
    main()
