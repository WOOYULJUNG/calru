"""Compare the registered H-C retention-amplitude/initialization sweep.

The five data columns are H-C variants trained under the same noise-free task,
optimizer, RP schedule, and seed policy.  Structural panels use seed 0 so the
geometry remains readable; the summary figure shows every estimable seed and
does not average failed runs into successful conditions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from repro.sagodi_protocol.artifacts import atomic_json
from repro.sagodi_protocol import plot_source_v6_extended as comparison


VARIANTS = (
    ("hc_a0p01_zero", "a0p01_zero", "$a=.01$\nzero-init", 0.01),
    ("hc_a0p025_zero", "a0p025_zero", "$a=.025$\nzero-init", 0.025),
    (
        "hc_a0p025_smallnormal",
        "a0p025_smallnormal",
        "$a=.025$\nsmall-normal",
        0.025,
    ),
    (
        "hc_a0p05_negbias",
        "a0p05_negbias",
        "$a=.05$\nnegative-bias",
        0.05,
    ),
    (
        "hc_a0p05_smallnormal",
        "a0p05_smallnormal",
        "$a=.05$\nsmall-normal",
        0.05,
    ),
)
COLORS = ("#3182ce", "#2f855a", "#805ad5", "#dd6b20", "#c53030")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _analysis_seeds(root: Path) -> dict[int, dict[str, Any]]:
    payload = _load_json(root / "summary.json")
    result: dict[int, dict[str, Any]] = {}
    for seed, summary in payload.get("seeds", {}).items():
        if isinstance(summary, dict) and summary.get("analysis_status") == (
            "complete_extended_structural_analysis"
        ):
            result[int(seed)] = summary
    return result


def _training_results(root: Path, cell: str) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for path in sorted((root / "runs").glob(f"{cell}__seed*/result.json")):
        payload = _load_json(path)
        results[int(payload["model_seed"])] = payload
    return results


def _normal_ratio(summary: dict[str, Any]) -> float:
    return float(
        summary["normal_recovery"]["in_plane_radial"]["0.05"]["4096"][
            "matched_clean_state_ratio"
        ]["median"]
    )


def _largest_real_and_gap(seed_dir: Path) -> tuple[float, float]:
    data = np.load(seed_dir / "full_local_eigenspectrum.npz", allow_pickle=False)
    largest = np.asarray(data["largest_real_part"], dtype=float)
    gap = np.asarray(data["real_part_gap"], dtype=float)
    return float(np.mean(largest)), float(np.median(gap))


def _metric_payload(
    training_root: Path, analysis_roots: dict[str, Path]
) -> dict[str, Any]:
    payload: dict[str, Any] = {"schema_version": 1, "variants": {}}
    for source, cell, label, amplitude in VARIANTS:
        training = _training_results(training_root, cell)
        analysis = _analysis_seeds(analysis_roots[source])
        row: dict[str, Any] = {
            "cell": cell,
            "label": label,
            "max_log_modulation": amplitude,
            "completed_training_seeds": sorted(training),
            "blank_stable_seeds": sorted(
                seed
                for seed, result in training.items()
                if result["autonomous_screen"]["stable_through_horizon"]
            ),
            "analysis_complete_seeds": sorted(analysis),
            "seeds": {},
        }
        for seed in sorted(set(training) | set(analysis)):
            item: dict[str, Any] = {}
            if seed in training:
                result = training[seed]
                metrics = result["task_evaluation"].get("metrics")
                if metrics is not None:
                    item["task_rmse"] = math.sqrt(float(metrics["masked_mse"]))
                item["blank_stable"] = bool(
                    result["autonomous_screen"]["stable_through_horizon"]
                )
                terminal = result["autonomous_screen"].get("terminal_memory") or {}
                if terminal.get("mean_angular_error_radians") is not None:
                    item["screen_angular_error"] = float(
                        terminal["mean_angular_error_radians"]
                    )
            if seed in analysis:
                summary = analysis[seed]
                item.update(
                    {
                        "uniform_flow": float(
                            summary["projected_flow"]["uniform_norm"]
                        ),
                        "memory_error_2048": float(
                            summary["finite_time_angular_memory"][
                                "terminal_mean_error_radians"
                            ]
                        ),
                        "tangent_gain": float(
                            summary["jacobian"]["tangent_one_step_gain"]["mean"]
                        ),
                        "radial_gain": float(
                            summary["jacobian"]["in_plane_radial_one_step_gain"][
                                "mean"
                            ]
                        ),
                        "effective_basin_count": float(
                            summary["asymptotic_structure"]["effective_basin_count"]
                        ),
                        "normal_recovery_ratio": _normal_ratio(summary),
                    }
                )
                largest, gap = _largest_real_and_gap(
                    analysis_roots[source] / f"seed{seed:02d}"
                )
                item["largest_real_part"] = largest
                item["top_two_real_gap"] = gap
            row["seeds"][str(seed)] = item
        payload["variants"][source] = row
    return payload


def _scatter_metric(
    ax: plt.Axes,
    payload: dict[str, Any],
    key: str,
    title: str,
    ylabel: str,
    *,
    log: bool = False,
    reference: float | None = None,
) -> None:
    for index, ((source, _, _, _), color) in enumerate(zip(VARIANTS, COLORS)):
        seeds = payload["variants"][source]["seeds"]
        values = [
            float(item[key])
            for item in seeds.values()
            if key in item and np.isfinite(float(item[key]))
        ]
        if values:
            jitter = np.linspace(-0.11, 0.11, len(values)) if len(values) > 1 else [0.0]
            ax.scatter(
                np.asarray(jitter) + index,
                values,
                color=color,
                s=32,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )
            ax.plot(
                [index - 0.18, index + 0.18],
                [np.mean(values), np.mean(values)],
                color=color,
                linewidth=2.0,
            )
        else:
            ax.text(index, 0.5, "N/E", ha="center", va="center", transform=ax.get_xaxis_transform())
    if reference is not None:
        ax.axhline(reference, color="#718096", linestyle="--", linewidth=0.8)
    if log:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([])
    ax.grid(alpha=0.18, axis="y")


def plot_metric_summary(
    payload: dict[str, Any], destination: Path, prefix: str
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.5), constrained_layout=True)
    fig.suptitle(
        "H-C hyperparameter comparison: multi-seed dynamical metrics",
        fontsize=15,
        fontweight="bold",
    )
    specs = (
        ("task_rmse", "Task error", "RMSE", True, None),
        ("uniform_flow", "Residual manifold flow", "uniform norm", True, None),
        ("memory_error_2048", "Long-horizon memory", "mean error (rad)", True, None),
        ("tangent_gain", "Tangent preservation", "one-step gain", False, 1.0),
        ("radial_gain", "In-plane radial response", "one-step gain", False, 1.0),
        ("largest_real_part", "Largest local real part", r"mean Re$(\mu_1)$", False, 0.0),
        ("top_two_real_gap", "Top-two local gap", r"median Re$(\mu_1-\mu_2)$", True, None),
        ("effective_basin_count", "Effective asymptotic basins", "effective count", False, None),
        ("normal_recovery_ratio", "Finite radial recovery", "ratio at 4096", True, 1.0),
    )
    for ax, (key, title, ylabel, log, reference) in zip(axes.flat, specs):
        _scatter_metric(
            ax,
            payload,
            key,
            title,
            ylabel,
            log=log,
            reference=reference,
        )
    labels = []
    for source, _, label, _ in VARIANTS:
        row = payload["variants"][source]
        labels.append(
            label + f"\nstable {len(row['blank_stable_seeds'])}/3"
        )
    for ax in axes[-1]:
        ax.set_xticklabels(labels, fontsize=7)
    comparison._save(fig, destination, f"{prefix}_summary_metrics")


def plot_retention_spectrum(
    training_root: Path,
    analysis_roots: dict[str, Path],
    payload: dict[str, Any],
    destination: Path,
    prefix: str,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(VARIANTS) + 1,
        figsize=(3.25 * (len(VARIANTS) + 1), 3.8),
        constrained_layout=True,
    )
    fig.suptitle("H-C base and state-dependent retention spectra (seed 0)", fontsize=15, fontweight="bold")
    for index, ((source, cell, label, amplitude), color) in enumerate(
        zip(VARIANTS, COLORS)
    ):
        ax = axes[index]
        ax.set_title(label, fontsize=10, fontweight="bold")
        checkpoint = training_root / "runs" / f"{cell}__seed00" / "checkpoint_trained.pt"
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        theta = state_dict["core.blocks.0.rec.theta"].to(torch.float64)
        base = torch.sqrt(torch.sigmoid(theta)).numpy()
        order = np.argsort(base)
        percentile = (np.arange(base.size) + 0.5) / base.size
        ordered = base[order]
        ax.plot(percentile, ordered, color=color, linewidth=1.6, label="base")
        ax.scatter(percentile, ordered, color=color, s=8, alpha=0.7, linewidths=0)
        seed_dir = analysis_roots[source] / "seed00"
        if (seed_dir / "slow_manifold_reconstruction.npz").exists():
            state = torch.as_tensor(
                np.load(seed_dir / "slow_manifold_reconstruction.npz")["spline_state"],
                dtype=torch.float32,
            )
            key = "core.blocks.0.rec.retention_gate."
            hidden = torch.nn.functional.gelu(
                state @ state_dict[key + "0.weight"].T + state_dict[key + "0.bias"]
            )
            raw = hidden @ state_dict[key + "2.weight"].T + state_dict[key + "2.bias"]
            dynamic = torch.as_tensor(base, dtype=torch.float32)[None, :] * torch.exp(
                float(amplitude) * torch.tanh(raw)
            )
            dynamic = dynamic[:, order].numpy()
            ax.fill_between(
                percentile,
                np.quantile(dynamic, 0.05, axis=0),
                np.quantile(dynamic, 0.95, axis=0),
                color="#805ad5",
                alpha=0.17,
                label="dynamic 5–95%",
            )
            ax.plot(percentile, np.median(dynamic, axis=0), color="#805ad5", linewidth=1.1)
            dynamic_max = float(dynamic.max())
        else:
            dynamic_max = float("nan")
        ax.axhline(1.0, color="#718096", linestyle="--", linewidth=0.8)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.06)
        ax.grid(alpha=0.18)
        ax.set_xlabel("mode quantile")
        if index == 0:
            ax.set_ylabel(r"retention $\lambda$")
        stable = len(payload["variants"][source]["blank_stable_seeds"])
        ax.text(
            0.03,
            0.97,
            f"base max={base.max():.6f}\ndynamic max={dynamic_max:.6f}\nstable={stable}/3",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )
        if index == 0:
            ax.legend(frameon=False, fontsize=7, loc="lower left")
    ideal = axes[-1]
    ideal.set_title("Ideal CA\n(schematic)", fontsize=10, fontweight="bold")
    ideal.set_facecolor("#f0fff4")
    ideal.text(
        0.5,
        0.57,
        "No unique ideal coordinate spectrum",
        ha="center",
        va="center",
        transform=ideal.transAxes,
        fontsize=9,
        color="#22543d",
    )
    ideal.text(
        0.5,
        0.39,
        "CA is defined by one neutral tangent mode\nand contracting normal modes,\nnot by each diagonal retention value.",
        ha="center",
        va="center",
        transform=ideal.transAxes,
        fontsize=8,
        color="#276749",
    )
    ideal.set_xticks([])
    ideal.set_yticks([])
    comparison._save(fig, destination, f"{prefix}_retention_spectrum")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--a0p01-analysis-root", type=Path, required=True)
    parser.add_argument("--a0p025-zero-analysis-root", type=Path, required=True)
    parser.add_argument("--a0p025-smallnormal-analysis-root", type=Path, required=True)
    parser.add_argument("--a0p05-negbias-analysis-root", type=Path, required=True)
    parser.add_argument("--a0p05-smallnormal-analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="fig_hc_hparam_comparison")
    args = parser.parse_args()
    training_root = args.training_root.expanduser().resolve(strict=True)
    analysis_roots = {
        "hc_a0p01_zero": args.a0p01_analysis_root.expanduser().resolve(strict=True),
        "hc_a0p025_zero": args.a0p025_zero_analysis_root.expanduser().resolve(strict=True),
        "hc_a0p025_smallnormal": args.a0p025_smallnormal_analysis_root.expanduser().resolve(strict=True),
        "hc_a0p05_negbias": args.a0p05_negbias_analysis_root.expanduser().resolve(strict=True),
        "hc_a0p05_smallnormal": args.a0p05_smallnormal_analysis_root.expanduser().resolve(strict=True),
    }
    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    panels = tuple((source, cell, label) for source, cell, label, _ in VARIANTS)
    conditions = comparison.NO_NOISE_CONDITIONS
    payload = _metric_payload(training_root, analysis_roots)
    atomic_json(destination / "metric_summary.json", payload)
    plot_metric_summary(payload, destination, args.prefix)
    comparison.plot_geometry_topology(
        training_root, None, analysis_roots, panels, conditions, destination, args.prefix
    )
    comparison.plot_jacobian(
        training_root, None, analysis_roots, panels, conditions, destination, args.prefix
    )
    comparison.plot_memory(
        training_root, None, analysis_roots, panels, conditions, destination, args.prefix
    )
    comparison.plot_normal_recovery(
        training_root, None, analysis_roots, panels, conditions, destination, args.prefix
    )
    comparison.plot_asymptotic_memory_map(
        training_root, None, analysis_roots, panels, conditions, destination, args.prefix
    )
    plot_retention_spectrum(
        training_root, analysis_roots, payload, destination, args.prefix
    )
    figures = sorted(path.name for path in destination.glob(f"{args.prefix}_*.pdf"))
    atomic_json(
        destination / "figure_manifest.json",
        {
            "schema_version": 1,
            "training_root": str(training_root),
            "analysis_roots": {key: str(value) for key, value in analysis_roots.items()},
            "variants": [cell for _, cell, _, _ in VARIANTS],
            "figures": figures,
            "seed_policy": "multi-seed summary; seed00 descriptive structural panels",
            "failure_policy": "failed or non-estimable seeds are not imputed",
            "ideal_reference": "conceptual schematic, not measured data",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
