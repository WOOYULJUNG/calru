"""Plot metric-first comparison figures for the source-v6 extended pilot.

The input is the receipt-bound ``source_v6_extended_analysis`` artifact.  Each
figure uses model columns and training-noise rows so that comparisons do not
depend on matching axes across separate model-specific dashboards.  The
figures are descriptive seed-0 pilot outputs; they are not confirmatory
multi-seed summaries.  A separate right-hand ``Ideal CA`` panel is explicitly
schematic and never presented as measured data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    ("sagodi_rnn_tanh_n128", "RNN"),
    ("sagodi_gru_n128", "GRU"),
    ("sagodi_lstm_n64", "LSTM"),
    ("lru_n52", "LRU"),
)
CONDITIONS = (
    ("noise_free", "noise-free", "#2b6cb0"),
    ("positive_state_noise_training", "state-noise", "#c05621"),
)
HORIZONS = np.asarray((0, 1, 4, 16, 64, 256, 1024, 4096), dtype=float)
RADIUS_COLORS = {0.01: "#90cdf4", 0.05: "#2b6cb0", 0.1: "#1a365d"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _run_dir(root: Path, model: str, condition: str) -> Path:
    return root / "runs" / f"{model}__{condition}__seed00"


def _load_run(root: Path, model: str, condition: str) -> dict[str, Any]:
    directory = _run_dir(root, model, condition)
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return {"status": "missing", "directory": str(directory)}
    summary = _load_json(summary_path)
    result: dict[str, Any] = {
        "status": summary.get("analysis_status", "unknown"),
        "directory": str(directory),
        "summary": summary,
    }
    for stem in (
        "projected_flow_and_topology",
        "full_local_eigenspectrum",
        "finite_time_angular_memory",
        "carrier_ambient_normal_recovery",
        "asymptotic_structure",
    ):
        path = directory / f"{stem}.npz"
        if path.exists():
            result[stem] = np.load(path, allow_pickle=False)
    return result


def _is_estimable(run: Mapping[str, Any]) -> bool:
    return run.get("status") == "complete_extended_structural_analysis"


def _annotate_not_estimable(ax: plt.Axes, run: Mapping[str, Any]) -> None:
    ax.text(
        0.5,
        0.5,
        "structural analysis\nnot estimable",
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="#4a5568",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    reason = run.get("summary", {}).get("manifold_reconstruction", {}).get("reason")
    if reason:
        ax.text(
            0.5,
            0.08,
            str(reason).replace("StructuralNotEstimableError:", ""),
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            color="#718096",
            fontsize=6,
            wrap=True,
        )


def _base_grid(title: str) -> tuple[plt.Figure, np.ndarray, plt.Axes]:
    fig = plt.figure(figsize=(16.8, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 5, width_ratios=(1.0, 1.0, 1.0, 1.0, 1.08))
    axes = np.asarray(
        [[fig.add_subplot(grid[row, column]) for column in range(4)] for row in range(2)]
    )
    ideal = fig.add_subplot(grid[:, 4])
    fig.suptitle(title, fontsize=15, fontweight="bold")
    for column, (_, label) in enumerate(MODELS):
        axes[0, column].set_title(label, fontsize=11, fontweight="bold")
    for row, (_, label, _) in enumerate(CONDITIONS):
        axes[row, 0].set_ylabel(label, fontsize=10, fontweight="bold")
    ideal.set_title("Ideal CA\n(schematic)", fontsize=11, fontweight="bold")
    ideal.set_facecolor("#f0fff4")
    ideal.text(
        0.5,
        0.02,
        "conceptual reference · not measured",
        transform=ideal.transAxes,
        ha="center",
        va="bottom",
        fontsize=7,
        color="#276749",
    )
    return fig, axes, ideal


def _save(fig: plt.Figure, destination: Path, stem: str) -> None:
    fig.savefig(destination / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(destination / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _nearest_indices(angle: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if targets.size == 0:
        return np.empty((0,), dtype=int)
    distance = np.abs(np.angle(np.exp(1j * (angle[:, None] - targets[None, :]))))
    return np.argmin(distance, axis=0)


def plot_geometry_topology(root: Path, destination: Path) -> None:
    fig, axes, ideal = _base_grid("Baseline geometry and projected topology")
    runs = {(m, c): _load_run(root, m, c) for m, _ in MODELS for c, _, _ in CONDITIONS}
    points: list[np.ndarray] = []
    for run in runs.values():
        if _is_estimable(run):
            points.append(np.asarray(run["projected_flow_and_topology"]["output"]))
    if points:
        all_points = np.concatenate(points, axis=0)
        limit = float(np.max(np.abs(all_points))) * 1.08
    else:
        limit = 1.0
    for row, (condition, _, color) in enumerate(CONDITIONS):
        for column, (model, _) in enumerate(MODELS):
            ax = axes[row, column]
            run = runs[(model, condition)]
            if not _is_estimable(run):
                _annotate_not_estimable(ax, run)
                continue
            data = run["projected_flow_and_topology"]
            output = np.asarray(data["output"])
            angle = np.asarray(data["spline_angle"])
            ax.plot(output[:, 0], output[:, 1], color="#cbd5e0", linewidth=1.0)
            ax.scatter(
                output[:, 0],
                output[:, 1],
                c=angle,
                cmap="twilight",
                s=8,
                alpha=0.85,
                linewidths=0,
            )
            stable = np.asarray(data["stable_fixed_point_angle"])
            saddle = np.asarray(data["saddle_fixed_point_angle"])
            for targets, marker, marker_color, label in (
                (stable, "o", "#2f855a", "stable"),
                (saddle, "^", "#c53030", "saddle"),
            ):
                indices = _nearest_indices(angle, targets)
                if indices.size:
                    ax.scatter(
                        output[indices, 0],
                        output[indices, 1],
                        marker=marker,
                        color=marker_color,
                        s=28,
                        edgecolors="white",
                        linewidths=0.5,
                        label=label,
                        zorder=4,
                    )
            summary = run["summary"]
            topology = summary["fixed_point_topology"]
            flow = summary["projected_flow"]["uniform_norm"]
            ax.text(
                0.03,
                0.97,
                f"{topology['stable_count']}/{topology['saddle_count']}  ·  U={flow:.3g}",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.18)
            if row == 1:
                ax.set_xlabel("output x")
            if column == 0:
                ax.set_ylabel("output y")
    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor="#2f855a", markeredgecolor="white", label="stable"),
        plt.Line2D([], [], marker="^", color="none", markerfacecolor="#c53030", markeredgecolor="white", label="saddle"),
    ]
    theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    ideal.scatter(
        np.cos(theta),
        np.sin(theta),
        c=theta,
        cmap="twilight",
        s=9,
        linewidths=0,
    )
    ideal.plot(np.cos(theta), np.sin(theta), color="#68d391", linewidth=1.2)
    ideal.text(
        0.5,
        0.92,
        "continuous fixed-point ring\nno isolated stable/saddle points\nU = 0",
        transform=ideal.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#22543d",
    )
    ideal.set_xlim(-limit, limit)
    ideal.set_ylim(-limit, limit)
    ideal.set_aspect("equal", adjustable="box")
    ideal.set_xlabel("output x")
    ideal.set_ylabel("output y")
    ideal.grid(alpha=0.18)
    ideal.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    _save(fig, destination, "fig_baseline_geometry_topology")


def plot_jacobian(root: Path, destination: Path) -> None:
    fig, axes, ideal = _base_grid("Baseline local Jacobian spectrum")
    runs = {(m, c): _load_run(root, m, c) for m, _ in MODELS for c, _, _ in CONDITIONS}
    values: list[np.ndarray] = []
    for run in runs.values():
        if _is_estimable(run):
            data = run["full_local_eigenspectrum"]
            values.extend((np.asarray(data["lambda1_real"]), np.asarray(data["lambda2_real"])))
    if values:
        bound = max(0.01, float(np.max(np.abs(np.concatenate(values)))) * 1.1)
    else:
        bound = 0.01
    for row, (condition, _, color) in enumerate(CONDITIONS):
        for column, (model, _) in enumerate(MODELS):
            ax = axes[row, column]
            run = runs[(model, condition)]
            if not _is_estimable(run):
                _annotate_not_estimable(ax, run)
                continue
            data = run["full_local_eigenspectrum"]
            angle = np.asarray(run["projected_flow_and_topology"]["spline_angle"])
            lambda1 = np.asarray(data["lambda1_real"])
            lambda2 = np.asarray(data["lambda2_real"])
            gap = np.asarray(data["gap"])
            ax.plot(angle, lambda1, color=color, linewidth=1.5, label=r"$\lambda_1$")
            ax.plot(angle, lambda2, color="#4a5568", linewidth=1.2, label=r"$\lambda_2$")
            ax.axhline(0.0, color="#718096", linewidth=0.8, linestyle="--")
            ax.text(
                0.03,
                0.97,
                f"median gap={np.median(gap):.3g}",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            ax.set_xlim(0.0, 2.0 * np.pi)
            ax.set_ylim(-bound, bound)
            ax.grid(alpha=0.18)
            if row == 1:
                ax.set_xlabel(r"memory angle $\theta$")
            if column == 0:
                ax.set_ylabel(r"real part of $J_F-I$")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    schematic_normal = -0.30 * bound * np.ones_like(theta)
    ideal.plot(theta, np.zeros_like(theta), color="#2f855a", linewidth=2.0, label=r"tangent $\lambda=0$")
    ideal.plot(theta, schematic_normal, color="#276749", linewidth=1.6, label=r"normal $\lambda<0$")
    ideal.fill_between(theta, schematic_normal, 0.0, color="#68d391", alpha=0.15)
    ideal.axhline(0.0, color="#718096", linewidth=0.8, linestyle="--")
    ideal.text(
        0.5,
        0.92,
        "one neutral tangent mode\nall normal modes contracting\nnormal magnitude is schematic",
        transform=ideal.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#22543d",
    )
    ideal.set_xlim(0.0, 2.0 * np.pi)
    ideal.set_ylim(-bound, bound)
    ideal.set_xlabel(r"memory angle $\theta$")
    ideal.set_ylabel(r"real part of $J_F-I$")
    ideal.grid(alpha=0.18)
    ideal.legend(frameon=False, fontsize=7, loc="center right")
    _save(fig, destination, "fig_baseline_jacobian_spectrum")


def plot_memory(root: Path, destination: Path) -> None:
    fig, axes, ideal = _base_grid("Baseline finite-time angular memory")
    runs = {(m, c): _load_run(root, m, c) for m, _ in MODELS for c, _, _ in CONDITIONS}
    for row, (condition, _, color) in enumerate(CONDITIONS):
        for column, (model, _) in enumerate(MODELS):
            ax = axes[row, column]
            run = runs[(model, condition)]
            if not _is_estimable(run):
                _annotate_not_estimable(ax, run)
                continue
            data = run["finite_time_angular_memory"]
            time = np.asarray(data["time"], dtype=float) / 256.0
            absolute = np.asarray(data["absolute_error"])
            mean = np.asarray(data["instantaneous_mean_error"])
            lower = np.quantile(absolute, 0.05, axis=0)
            upper = np.quantile(absolute, 0.95, axis=0)
            ax.fill_between(time, lower, upper, color=color, alpha=0.16)
            ax.plot(time, mean, color=color, linewidth=1.6)
            ax.axvline(8.0, color="#718096", linestyle="--", linewidth=0.8)
            terminal = run["summary"]["finite_time_angular_memory"]["terminal_mean_error_radians"]
            ax.text(
                0.03,
                0.97,
                f"error@8T={terminal:.3f} rad",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            ax.set_xlim(0.0, 8.0)
            ax.set_ylim(0.0, np.pi)
            ax.grid(alpha=0.18)
            if row == 1:
                ax.set_xlabel(r"blank horizon $t/T$")
            if column == 0:
                ax.set_ylabel("circular error (rad)")
    ideal.plot((0.0, 8.0), (0.0, 0.0), color="#2f855a", linewidth=2.4)
    ideal.text(
        0.5,
        0.92,
        "zero angular drift\nfor every stored memory\nerror(t) = 0",
        transform=ideal.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#22543d",
    )
    ideal.set_xlim(0.0, 8.0)
    ideal.set_ylim(0.0, np.pi)
    ideal.set_xlabel(r"blank horizon $t/T$")
    ideal.set_ylabel("circular error (rad)")
    ideal.grid(alpha=0.18)
    _save(fig, destination, "fig_baseline_memory_retention")


def _normal_series(data: Mapping[str, np.ndarray], radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    family = np.asarray(data["family"]).astype(str)
    selected = (family == "ambient_normal") & np.isclose(
        np.asarray(data["radius_over_manifold_scale"], dtype=float), radius
    )
    horizon = np.asarray(data["horizon"], dtype=int)
    ratio = np.asarray(data["manifold_distance_ratio"], dtype=float)
    if ratio.ndim != 2 or ratio.shape[1] != horizon.size:
        raise ValueError("normal-recovery ratio must have shape [trial, horizon]")
    selected_horizons: list[int] = []
    medians: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for horizon_index, value in enumerate(horizon):
        mask = selected
        if not np.any(mask):
            continue
        values = ratio[mask, horizon_index]
        selected_horizons.append(int(value))
        medians.append(float(np.median(values)))
        lower.append(float(np.quantile(values, 0.1)))
        upper.append(float(np.quantile(values, 0.9)))
    return (
        np.asarray(selected_horizons, dtype=float),
        np.asarray(medians),
        np.asarray(lower),
        np.asarray(upper),
    )


def plot_normal_recovery(root: Path, destination: Path) -> None:
    fig, axes, ideal = _base_grid("Baseline finite normal-kick recovery")
    runs = {(m, c): _load_run(root, m, c) for m, _ in MODELS for c, _, _ in CONDITIONS}
    all_values: list[float] = []
    for run in runs.values():
        if _is_estimable(run):
            data = run["carrier_ambient_normal_recovery"]
            for radius in RADIUS_COLORS:
                _, median, lower, upper = _normal_series(data, radius)
                all_values.extend(np.concatenate((lower, upper)).tolist())
    ymin = max(1.0e-2, min(all_values) * 0.75) if all_values else 1.0e-2
    ymax = max(10.0, max(all_values) * 1.25) if all_values else 10.0
    for row, (condition, _, condition_color) in enumerate(CONDITIONS):
        for column, (model, _) in enumerate(MODELS):
            ax = axes[row, column]
            run = runs[(model, condition)]
            if not _is_estimable(run):
                _annotate_not_estimable(ax, run)
                continue
            data = run["carrier_ambient_normal_recovery"]
            for radius, radius_color in RADIUS_COLORS.items():
                horizon, median, lower, upper = _normal_series(data, radius)
                if not horizon.size:
                    continue
                linewidth = 2.0 if radius == 0.05 else 1.0
                alpha = 0.16 if radius == 0.05 else 0.0
                ax.plot(
                    horizon,
                    median,
                    color=radius_color,
                    linewidth=linewidth,
                    marker="o" if radius == 0.05 else None,
                    markersize=3,
                    label=f"r={radius:g}" if row == 0 and column == 0 else None,
                )
                if alpha:
                    ax.fill_between(horizon, lower, upper, color=radius_color, alpha=alpha)
            ax.axhline(1.0, color="#718096", linestyle="--", linewidth=0.8)
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_yscale("log")
            ax.set_ylim(ymin, ymax)
            ax.grid(alpha=0.18, which="both")
            if row == 1:
                ax.set_xlabel("blank horizon")
            if column == 0:
                ax.set_ylabel(r"normal distance ratio $d_t/d_0$")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    ideal_ratio = np.power(1.0 + HORIZONS, -0.5)
    ideal.plot(HORIZONS, ideal_ratio, color="#2f855a", linewidth=2.2, marker="o", markersize=3)
    ideal.axhline(1.0, color="#718096", linestyle="--", linewidth=0.8)
    ideal.text(
        0.5,
        0.92,
        "monotone normal contraction\n$d_t/d_0 < 1$ for $t>0$\nrate is schematic",
        transform=ideal.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#22543d",
    )
    ideal.set_xscale("symlog", linthresh=1.0)
    ideal.set_yscale("log")
    ideal.set_ylim(ymin, ymax)
    ideal.set_xlabel("blank horizon")
    ideal.set_ylabel(r"normal distance ratio $d_t/d_0$")
    ideal.grid(alpha=0.18, which="both")
    _save(fig, destination, "fig_baseline_normal_recovery")


def plot_basin_capacity(root: Path, destination: Path) -> None:
    fig, axes, ideal = _base_grid("Baseline asymptotic basin topology")
    runs = {(m, c): _load_run(root, m, c) for m, _ in MODELS for c, _, _ in CONDITIONS}
    max_basins = 1
    for run in runs.values():
        if _is_estimable(run):
            max_basins = max(max_basins, int(run["summary"]["asymptotic_structure"]["stable_count"]))
    for row, (condition, _, color) in enumerate(CONDITIONS):
        for column, (model, _) in enumerate(MODELS):
            ax = axes[row, column]
            run = runs[(model, condition)]
            if not _is_estimable(run):
                _annotate_not_estimable(ax, run)
                continue
            asymptotic = run["summary"]["asymptotic_structure"]
            proportions = np.asarray(asymptotic["basin_proportions"], dtype=float)
            indices = np.arange(1, proportions.size + 1)
            ax.bar(indices, proportions, color=color, alpha=0.82, width=0.72)
            ax.set_ylim(0.0, 0.7)
            ax.set_xlim(0.4, max_basins + 0.6)
            ax.set_xticks(indices)
            ax.set_xlabel("stable basin")
            if column == 0:
                ax.set_ylabel("basin proportion")
            ax.grid(axis="y", alpha=0.18)
            ax.text(
                0.03,
                0.97,
                f"H={asymptotic['shannon_entropy_nats']:.3f}\nKeff={asymptotic['effective_basin_count']:.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
    theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    ideal.scatter(
        np.cos(theta),
        np.sin(theta),
        c=theta,
        cmap="twilight",
        s=9,
        linewidths=0,
    )
    ideal.plot(np.cos(theta), np.sin(theta), color="#68d391", linewidth=1.2)
    ideal.text(
        0.5,
        0.92,
        "continuum of memory states\nno discrete basin partition\nfinite $K_{eff}$ not applicable",
        transform=ideal.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#22543d",
    )
    ideal.set_xlim(-1.35, 1.35)
    ideal.set_ylim(-1.35, 1.35)
    ideal.set_aspect("equal", adjustable="box")
    ideal.set_xticks([])
    ideal.set_yticks([])
    _save(fig, destination, "fig_baseline_basin_capacity")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.artifact_root.expanduser().resolve(strict=True)
    destination = (args.output_dir or root / "figures").expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plot_geometry_topology(root, destination)
    plot_jacobian(root, destination)
    plot_memory(root, destination)
    plot_normal_recovery(root, destination)
    plot_basin_capacity(root, destination)
    manifest = {
        "schema_version": 1,
        "artifact_root": str(root),
        "output_dir": str(destination),
        "models": [label for _, label in MODELS],
        "conditions": [label for _, label, _ in CONDITIONS],
        "seed_scope": "seed00 exploratory pilot",
        "ideal_reference": {
            "role": "conceptual_schematic_not_measured_data",
            "geometry": "continuous fixed-point ring with zero projected flow",
            "jacobian": "one neutral tangent mode and contracting normal modes",
            "memory": "zero circular drift",
            "normal_recovery": "monotone contraction toward the manifold",
            "basins": "continuous memory without discrete basin partition",
        },
        "figures": sorted(path.name for path in destination.glob("fig_baseline_*.pdf")),
        "normal_recovery": {
            "family": "ambient_normal",
            "radii_over_manifold_scale": [0.01, 0.05, 0.1],
            "summary": "median with 10-90 percentile band for radius 0.05",
        },
    }
    with (destination / "figure_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
