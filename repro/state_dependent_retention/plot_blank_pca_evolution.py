"""Visualize blank-input evolution of reconstructed ring states in fixed PCA bases.

The existing geometry/topology comparison is an output projection of one
reconstructed slow-manifold carrier.  It does not show how that carrier moves
under blank input.  This script starts from the exact reconstructed spline
states used in that figure, fits PCA once at t=0 for each model, and keeps both
the PCA origin and axes fixed throughout a deterministic blank rollout.

Two complementary outputs are produced:

* a model-by-time PCA grid; and
* full-state contraction and tangential discretization diagnostics.

The latter distinction matters: convergence to several stable fixed points can
leave the global cloud radius large even though the initially continuous ring
has become discrete.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from repro.sagodi_protocol.artifacts import atomic_json, strict_json_load
from repro.sagodi_protocol.sagodi_primary_runner import _blank_decode_primary
from repro.sagodi_protocol.source_v6_analysis_adapter import (
    SourceV6AnalysisModel,
    load_source_v6_checkpoint,
)
from repro.sagodi_protocol.state import StateAdapter

from .models import build_state_dependent_model


DEFAULT_SOURCE_ANALYSIS = Path(
    "/home/biadmin/ca_rnn/experiments/source_v6_extended_analysis-27a37b1"
)
DEFAULT_CALRU_ANALYSIS = Path(
    "/home/biadmin/ca_rnn/experiments/calru_factorial_extended_analysis-27a37b1"
)
DEFAULT_HC_ANALYSIS = Path(
    "/home/biadmin/ca_rnn/experiments/"
    "state_dependent_retention_writer_remaining_trainonly-5bf5c8a/"
    "attractor_analysis_h_c_v1"
)
DEFAULT_HC_SWEEP_ANALYSIS = Path(
    "/home/biadmin/ca_rnn/experiments/hc_a0p025_zero_ca_analysis-b219c21"
)
DEFAULT_OUTPUT = Path(
    "/home/biadmin/ca_rnn/experiments/blank_pca_evolution-57733b2/figures"
)

HORIZONS = (0, 16, 64, 256, 1024, 4096)
METRIC_HORIZONS = (0, 1, 4, 16, 64, 256, 1024, 4096)
COLORS = (
    "#2b6cb0",
    "#dd6b20",
    "#38a169",
    "#718096",
    "#805ad5",
    "#d53f8c",
    "#319795",
    "#c05621",
)


@dataclass(frozen=True)
class Panel:
    key: str
    label: str
    analysis_dir: Path
    source_plan_key: tuple[str, str] | None = None
    calru_plan_key: tuple[str, str] | None = None
    hc_checkpoint: bool = False


def _plan_checkpoint(
    root: Path, *, model_id: str, condition: str
) -> Path:
    plan = strict_json_load(root / "plan.json")
    matches = [
        Path(row["checkpoint"]).expanduser().resolve(strict=True)
        for row in plan["runs"]
        if row.get("model_id") == model_id
        and row.get("condition") == condition
        and int(row.get("model_seed", -1)) == 0
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one seed-0 checkpoint for {model_id}/{condition}, got {len(matches)}"
        )
    return matches[0]


def _panels(
    source_root: Path,
    calru_root: Path,
    hc_root: Path,
    hc_sweep_root: Path,
) -> tuple[Panel, ...]:
    return (
        Panel(
            "rnn",
            "RNN",
            source_root / "runs/sagodi_rnn_tanh_n128__noise_free__seed00",
            source_plan_key=("sagodi_rnn_tanh_n128", "noise_free"),
        ),
        Panel(
            "gru",
            "GRU",
            source_root / "runs/sagodi_gru_n128__noise_free__seed00",
            source_plan_key=("sagodi_gru_n128", "noise_free"),
        ),
        Panel(
            "lstm",
            "LSTM",
            source_root / "runs/sagodi_lstm_n64__noise_free__seed00",
            source_plan_key=("sagodi_lstm_n64", "noise_free"),
        ),
        Panel(
            "lru",
            "LRU",
            source_root / "runs/lru_n52__noise_free__seed00",
            source_plan_key=("lru_n52", "noise_free"),
        ),
        Panel(
            "no_rp",
            "CA-LRU no RP",
            calru_root / "runs/no_rp_no_noise__seed00",
            calru_plan_key=("no_rp_n52", "no_rp_no_noise"),
        ),
        Panel(
            "ca_lru",
            "CA-LRU",
            calru_root / "runs/rp_no_noise__seed00",
            calru_plan_key=("ca_lru_n52", "rp_no_noise"),
        ),
        Panel(
            "hc_a0p05",
            "H-C a=.05",
            hc_root / "seed00",
            hc_checkpoint=True,
        ),
        Panel(
            "hc_a0p025",
            "H-C a=.025",
            hc_sweep_root / "seed00",
            hc_checkpoint=True,
        ),
    )


def _load_panel_model(
    panel: Panel,
    source_root: Path,
    calru_root: Path,
    device: torch.device,
) -> tuple[SourceV6AnalysisModel, Path]:
    if panel.source_plan_key is not None:
        checkpoint = _plan_checkpoint(
            source_root,
            model_id=panel.source_plan_key[0],
            condition=panel.source_plan_key[1],
        )
        model, _ = load_source_v6_checkpoint(checkpoint, device)
        model.eval()
        return model, checkpoint
    if panel.calru_plan_key is not None:
        checkpoint = _plan_checkpoint(
            calru_root,
            model_id=panel.calru_plan_key[0],
            condition=panel.calru_plan_key[1],
        )
        model, _ = load_source_v6_checkpoint(checkpoint, device)
        model.eval()
        return model, checkpoint
    if panel.hc_checkpoint:
        summary = strict_json_load(panel.analysis_dir / "summary.json")
        checkpoint_value = summary.get("checkpoint")
        if checkpoint_value is None:
            checkpoint = (
                panel.analysis_dir.parent.parent
                / "runs/hybrid_rp__recurrent__seed00/checkpoint_trained.pt"
            ).resolve(strict=True)
        else:
            checkpoint = Path(checkpoint_value).expanduser().resolve(strict=True)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cell = payload.get("cell")
        kwargs: dict[str, float] = {}
        if isinstance(cell, Mapping):
            kwargs = {
                "max_log_modulation": float(cell["max_log_modulation"]),
                "gate_output_weight_std": float(cell["gate_output_weight_std"]),
                "gate_output_bias": float(cell["gate_output_bias"]),
            }
        inner = build_state_dependent_model(
            "recurrent",
            model_seed=0,
            retention_mode="hybrid_rp",
            **kwargs,
        ).to(device)
        inner.load_state_dict(payload["state_dict"], strict=True)
        inner.eval()
        model = SourceV6AnalysisModel(inner, panel.key).to(device)
        model.eval()
        return model, checkpoint
    raise RuntimeError(f"panel has no checkpoint binding: {panel.key}")


def _subsample_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.floor(np.arange(maximum) * count / maximum).astype(np.int64)


def _circular_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (a - b))))


def _effective_bins(angle: np.ndarray, bins: int = 128) -> float:
    counts, _ = np.histogram(
        np.remainder(angle, 2.0 * np.pi),
        bins=bins,
        range=(0.0, 2.0 * np.pi),
    )
    probability = counts[counts > 0].astype(np.float64)
    probability /= probability.sum()
    return float(np.exp(-(probability * np.log(probability)).sum()))


def _fit_fixed_pca(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = state.mean(axis=0, dtype=np.float64)
    centered = state.astype(np.float64) - center
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    variance = np.square(singular)
    explained = variance[:2] / variance.sum()
    return center, components, explained


@torch.no_grad()
def _analyze_panel(
    panel: Panel,
    source_root: Path,
    calru_root: Path,
    device: torch.device,
    maximum_points: int,
) -> dict[str, Any]:
    reconstruction_path = panel.analysis_dir / "slow_manifold_reconstruction.npz"
    if not reconstruction_path.exists():
        return {
            "key": panel.key,
            "label": panel.label,
            "status": "not_estimable",
            "reason": "slow_manifold_reconstruction.npz is absent",
        }
    with np.load(reconstruction_path, allow_pickle=False) as archive:
        initial_state = np.asarray(archive["spline_state"], dtype=np.float32)
        initial_angle = np.asarray(archive["spline_angle"], dtype=np.float64)
    order = np.argsort(np.remainder(initial_angle, 2.0 * np.pi))
    take = _subsample_indices(len(order), maximum_points)
    indices = order[take]
    initial_state = initial_state[indices]
    initial_angle = initial_angle[indices]

    asymptotic_path = panel.analysis_dir / "asymptotic_structure.npz"
    with np.load(asymptotic_path, allow_pickle=False) as archive:
        assigned = np.asarray(archive["assigned_stable_angle"], dtype=np.float64)[
            indices
        ]
        stable = np.asarray(
            archive["stable_fixed_point_angle"], dtype=np.float64
        )

    model, checkpoint = _load_panel_model(panel, source_root, calru_root, device)
    adapter = StateAdapter(model.core)
    state = torch.as_tensor(initial_state, device=device, dtype=adapter.dtype)
    if state.shape[1] != adapter.state_size:
        raise RuntimeError(
            f"{panel.key} spline state width {state.shape[1]} != primary width "
            f"{adapter.state_size}"
        )

    requested = set(METRIC_HORIZONS) | set(HORIZONS)
    maximum_horizon = max(requested)
    states: dict[int, np.ndarray] = {}
    outputs: dict[int, np.ndarray] = {}
    failure_step: int | None = None
    for step in range(maximum_horizon + 1):
        if not bool(torch.isfinite(state).all()):
            failure_step = step
            break
        if step in requested:
            decoded = _blank_decode_primary(model, adapter, state)
            states[step] = state.detach().cpu().numpy().astype(np.float32)
            outputs[step] = decoded.detach().cpu().numpy().astype(np.float32)
        if step < maximum_horizon:
            state = adapter.actual_f0(state)

    center, components, explained = _fit_fixed_pca(states[0])
    initial_centered = states[0].astype(np.float64) - states[0].mean(
        axis=0, dtype=np.float64
    )
    initial_rms = float(
        np.sqrt(np.mean(np.sum(np.square(initial_centered), axis=1)))
    )
    if not initial_rms > 0.0:
        raise RuntimeError(f"{panel.key} initial carrier cloud has zero spread")

    metrics: dict[str, Any] = {}
    projections: dict[str, np.ndarray] = {}
    for step in sorted(states):
        value = states[step].astype(np.float64)
        centered_at_time = value - value.mean(axis=0, dtype=np.float64)
        cloud_rms = float(
            np.sqrt(np.mean(np.sum(np.square(centered_at_time), axis=1)))
        )
        fixed_centered = value - center
        projection = fixed_centered @ components
        residual = fixed_centered - projection @ components.T
        total_energy = float(np.sum(np.square(fixed_centered)))
        residual_fraction = (
            float(np.sum(np.square(residual)) / total_energy)
            if total_energy > 0.0
            else 0.0
        )
        decoded = outputs[step]
        decoded_radius = np.linalg.norm(decoded, axis=1)
        decoded_angle = np.arctan2(decoded[:, 1], decoded[:, 0])
        stable_error = _circular_error(decoded_angle, assigned)
        metrics[str(step)] = {
            "cloud_rms_ratio": cloud_rms / initial_rms,
            "centroid_drift_over_initial_rms": float(
                np.linalg.norm(value.mean(axis=0) - states[0].mean(axis=0))
                / initial_rms
            ),
            "fixed_initial_pc12_residual_energy_fraction": residual_fraction,
            "effective_angular_bin_count_128": _effective_bins(decoded_angle),
            "median_decoded_output_radius": float(np.median(decoded_radius)),
            "minimum_decoded_output_radius": float(np.min(decoded_radius)),
            "median_error_to_assigned_stable_angle_radians": float(
                np.median(stable_error)
            ),
            "fraction_within_0p01_rad_of_assigned_stable": float(
                np.mean(stable_error <= 0.01)
            ),
            "fraction_within_0p05_rad_of_assigned_stable": float(
                np.mean(stable_error <= 0.05)
            ),
        }
        projections[str(step)] = projection.astype(np.float32)

    return {
        "key": panel.key,
        "label": panel.label,
        "status": "finite" if failure_step is None else "nonfinite",
        "failure_step": failure_step,
        "checkpoint": str(checkpoint),
        "analysis_dir": str(panel.analysis_dir),
        "point_count": int(initial_state.shape[0]),
        "primary_state_dimension": int(initial_state.shape[1]),
        "stable_fixed_point_count": int(stable.size),
        "initial_pc_explained_variance_fraction": explained.tolist(),
        "initial_angle": initial_angle.astype(np.float32),
        "assigned_stable_angle": assigned.astype(np.float32),
        "projections": projections,
        "metrics": metrics,
    }


def _plot_grid(results: list[dict[str, Any]], output: Path) -> None:
    rows = len(results)
    columns = len(HORIZONS)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.55 * columns, 2.25 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        "Blank-input evolution in each model's fixed initial PCA plane",
        fontsize=15,
        fontweight="bold",
    )
    scatter = None
    for row, result in enumerate(results):
        finite = result["status"] in {"finite", "nonfinite"}
        if not finite:
            for column, step in enumerate(HORIZONS):
                ax = axes[row, column]
                ax.text(
                    0.5,
                    0.5,
                    "slow manifold\nnot estimable",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#718096",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if column == 0:
                    ax.set_ylabel(result["label"], fontweight="bold")
            continue
        available = [
            np.asarray(result["projections"][str(step)])
            for step in HORIZONS
            if str(step) in result["projections"]
        ]
        limit = max(float(np.max(np.abs(value))) for value in available) * 1.08
        limit = max(limit, 1.0e-6)
        color = np.remainder(np.asarray(result["initial_angle"]), 2.0 * np.pi)
        for column, step in enumerate(HORIZONS):
            ax = axes[row, column]
            if row == 0:
                ax.set_title(f"blank t={step}", fontsize=9, fontweight="bold")
            projection = result["projections"].get(str(step))
            if projection is None:
                ax.text(0.5, 0.5, "non-finite", transform=ax.transAxes, ha="center")
                continue
            projection = np.asarray(projection)
            ax.plot(
                np.r_[projection[:, 0], projection[0, 0]],
                np.r_[projection[:, 1], projection[0, 1]],
                color="#cbd5e0",
                linewidth=0.7,
                zorder=1,
            )
            scatter = ax.scatter(
                projection[:, 0],
                projection[:, 1],
                c=color,
                cmap="twilight",
                vmin=0.0,
                vmax=2.0 * np.pi,
                s=7,
                linewidths=0,
                alpha=0.9,
                zorder=2,
            )
            metric = result["metrics"][str(step)]
            ax.text(
                0.03,
                0.97,
                f"R/R0={metric['cloud_rms_ratio']:.2f}\n"
                f"Keff={metric['effective_angular_bin_count_128']:.1f}\n"
                f"rout={metric['median_decoded_output_radius']:.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=6.5,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.15)
            if column == 0:
                explained = result["initial_pc_explained_variance_fraction"]
                ax.set_ylabel(
                    result["label"]
                    + f"\nPC1+2={sum(explained):.2f}",
                    fontweight="bold",
                    fontsize=8,
                )
            if row == rows - 1:
                ax.set_xlabel("fixed PC1 / PC2", fontsize=7)
    if scatter is not None:
        fig.colorbar(
            scatter,
            ax=axes,
            location="bottom",
            fraction=0.012,
            pad=0.015,
            label="initial ring angle",
        )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"fig_blank_pca_evolution_grid.{suffix}", dpi=220)
    plt.close(fig)


def _plot_metrics(results: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    fig.suptitle(
        "Blank-input contraction versus convergence to discrete attractors",
        fontsize=15,
        fontweight="bold",
    )
    specifications = (
        ("cloud_rms_ratio", "Full-state cloud RMS / t=0", False),
        ("effective_angular_bin_count_128", "Effective decoded angular bins", True),
        (
            "median_error_to_assigned_stable_angle_radians",
            "Median distance to eventual stable point (rad)",
            True,
        ),
        (
            "fraction_within_0p05_rad_of_assigned_stable",
            "Fraction within 0.05 rad of stable point",
            False,
        ),
    )
    for result, color in zip(results, COLORS):
        if result["status"] not in {"finite", "nonfinite"}:
            continue
        steps = np.asarray(
            [int(key) for key in result["metrics"]], dtype=np.float64
        )
        order = np.argsort(steps)
        steps = steps[order]
        for ax, (key, ylabel, log_y) in zip(axes.flat, specifications):
            values = np.asarray(
                [result["metrics"][str(int(step))][key] for step in steps]
            )
            ax.plot(
                steps,
                values,
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                color=color,
                label=result["label"],
            )
            ax.set_xscale("symlog", linthresh=1.0)
            if log_y and np.all(values > 0.0):
                ax.set_yscale("log")
            ax.set_xlabel("blank-input steps")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
    axes[0, 0].axhline(1.0, color="#718096", linestyle="--", linewidth=0.8)
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output / f"fig_blank_pca_contraction_discretization.{suffix}",
            dpi=220,
        )
    plt.close(fig)


def _native_result(result: Mapping[str, Any]) -> dict[str, Any]:
    native: dict[str, Any] = {}
    for key, value in result.items():
        if key == "projections":
            continue
        if isinstance(value, np.ndarray):
            native[key] = value.tolist()
        else:
            native[key] = value
    return native


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-analysis-root", type=Path, default=DEFAULT_SOURCE_ANALYSIS)
    parser.add_argument("--calru-analysis-root", type=Path, default=DEFAULT_CALRU_ANALYSIS)
    parser.add_argument("--hc-analysis-root", type=Path, default=DEFAULT_HC_ANALYSIS)
    parser.add_argument(
        "--hc-sweep-analysis-root", type=Path, default=DEFAULT_HC_SWEEP_ANALYSIS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-points", type=int, default=256)
    args = parser.parse_args()
    if args.maximum_points < 32:
        raise ValueError("maximum-points must be at least 32")
    roots = [
        args.source_analysis_root,
        args.calru_analysis_root,
        args.hc_analysis_root,
        args.hc_sweep_analysis_root,
    ]
    source_root, calru_root, hc_root, hc_sweep_root = [
        value.expanduser().resolve(strict=True) for value in roots
    ]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    panels = _panels(source_root, calru_root, hc_root, hc_sweep_root)
    device = torch.device(args.device)
    results = [
        _analyze_panel(
            panel,
            source_root,
            calru_root,
            device,
            int(args.maximum_points),
        )
        for panel in panels
    ]
    _plot_grid(results, output)
    _plot_metrics(results, output)
    atomic_json(
        output / "blank_pca_evolution_metrics.json",
        {
            "schema_version": 1,
            "initialization": "reconstructed_slow_manifold_spline_state",
            "pca_contract": "fit_once_at_t0_fixed_center_and_axes_per_model",
            "blank_horizons": list(HORIZONS),
            "metric_horizons": list(METRIC_HORIZONS),
            "results": [_native_result(result) for result in results],
        },
    )
    arrays: dict[str, np.ndarray] = {}
    for result in results:
        if result["status"] not in {"finite", "nonfinite"}:
            continue
        for step, projection in result["projections"].items():
            arrays[f"{result['key']}__pca_t{step}"] = np.asarray(projection)
        arrays[f"{result['key']}__initial_angle"] = np.asarray(
            result["initial_angle"]
        )
    np.savez_compressed(output / "blank_pca_evolution_projections.npz", **arrays)
    atomic_json(
        output / "figure_manifest.json",
        {
            "schema_version": 1,
            "source_analysis_root": str(source_root),
            "calru_analysis_root": str(calru_root),
            "hc_analysis_root": str(hc_root),
            "hc_sweep_analysis_root": str(hc_sweep_root),
            "figures": [
                "fig_blank_pca_evolution_grid.png",
                "fig_blank_pca_evolution_grid.pdf",
                "fig_blank_pca_contraction_discretization.png",
                "fig_blank_pca_contraction_discretization.pdf",
            ],
            "interpretation": {
                "cloud_rms_ratio": "global carrier-cloud contraction independent of centroid translation",
                "effective_angular_bin_count_128": "descriptive concentration of decoded states, not a fitted cluster count",
                "assigned_stable_angle": "stable basin assignment from the registered asymptotic analysis",
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
