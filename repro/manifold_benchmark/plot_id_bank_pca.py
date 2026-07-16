"""Visualize the fixed S1/T2/S2 ID banks with one common PCA procedure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from .artifacts import atomic_json, load_manifold_bank, sha256_file
from .generator import ManifoldBatch


TOPOLOGY_ORDER = ("s1", "t2", "s2")
TOPOLOGY_LABEL = {"s1": r"$S^1$ ring", "t2": r"$T^2$ flat torus", "s2": r"$S^2$ sphere"}


def _canonical_component_sign(components: np.ndarray) -> np.ndarray:
    result = components.copy()
    for index, component in enumerate(result):
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            result[index] *= -1.0
    return result


def fit_pca(path: np.ndarray) -> dict[str, np.ndarray | float]:
    """Fit PCA to all q0..qT embedded states and return time-major scores."""

    flat = np.asarray(path, dtype=np.float64).reshape(-1, path.shape[-1])
    mean = np.mean(flat, axis=0)
    centered = flat - mean
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = _canonical_component_sign(eigenvectors[:, order].T)
    scores = (centered @ components.T).reshape(*path.shape[:2], path.shape[-1])
    ratio = eigenvalues / np.sum(eigenvalues)
    participation_ratio = float(np.sum(eigenvalues) ** 2 / np.sum(eigenvalues**2))
    return {
        "mean": mean,
        "components": components,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": ratio,
        "scores": scores,
        "participation_ratio": participation_ratio,
    }


def embedded_path(batch: ManifoldBatch) -> np.ndarray:
    return np.concatenate((batch.initial_memory[None], batch.output_targets), axis=0)


def latent_color(batch: ManifoldBatch) -> tuple[np.ndarray, str, str, float, float]:
    if batch.metadata["topology"] == "S2":
        return batch.latent_path[..., 2], r"current $n_z$", "coolwarm", -1.0, 1.0
    angle = batch.latent_path[..., 0]
    return angle, r"current first angle", "twilight", -np.pi, np.pi


def sample_flat_indices(total: int, maximum: int) -> np.ndarray:
    if total <= maximum:
        return np.arange(total)
    return np.linspace(0, total - 1, maximum, dtype=np.int64)


def fixed_limits(score: np.ndarray, left: int, right: int) -> tuple[tuple[float, float], tuple[float, float]]:
    x = score[..., left]
    y = score[..., right]
    x_padding = max(1e-6, 0.05 * (float(np.max(x)) - float(np.min(x))))
    y_padding = max(1e-6, 0.05 * (float(np.max(y)) - float(np.min(y))))
    return (
        (float(np.min(x)) - x_padding, float(np.max(x)) + x_padding),
        (float(np.min(y)) - y_padding, float(np.max(y)) + y_padding),
    )


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_overview(
    batches: dict[str, ManifoldBatch],
    pca: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 13.2), constrained_layout=True)
    fig.suptitle(
        "Manifold benchmark v1: PCA of pooled initial states and ID targets",
        fontsize=18,
        fontweight="bold",
    )
    for row, name in enumerate(TOPOLOGY_ORDER):
        batch = batches[name]
        result = pca[name]
        score = result["scores"]
        color, color_label, cmap, vmin, vmax = latent_color(batch)
        flat_score = score.reshape(-1, score.shape[-1])
        flat_color = color.reshape(-1)
        selected = sample_flat_indices(flat_score.shape[0], 24_000)

        cloud = axes[row, 0].scatter(
            flat_score[selected, 0],
            flat_score[selected, 1],
            c=flat_color[selected],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=3,
            alpha=0.32,
            linewidths=0,
            rasterized=True,
        )
        axes[row, 0].set_title(f"{TOPOLOGY_LABEL[name]}: pooled cloud")
        axes[row, 0].set_xlabel("PC1")
        axes[row, 0].set_ylabel("PC2")
        axes[row, 0].set_aspect("equal", adjustable="box")
        fig.colorbar(cloud, ax=axes[row, 0], label=color_label, shrink=0.78)

        trajectory_ids = np.linspace(
            0, batch.trajectories - 1, min(28, batch.trajectories), dtype=np.int64
        )
        time_color = np.linspace(0.0, 1.0, score.shape[0] - 1)
        for trajectory in trajectory_ids:
            xy = score[:, trajectory, :2]
            segments = np.stack((xy[:-1], xy[1:]), axis=1)
            collection = LineCollection(
                segments,
                cmap="viridis",
                norm=plt.Normalize(0.0, 1.0),
                linewidth=0.85,
                alpha=0.48,
            )
            collection.set_array(time_color)
            axes[row, 1].add_collection(collection)
            axes[row, 1].scatter(
                xy[0, 0], xy[0, 1], s=10, color="#202020", alpha=0.6, zorder=3
            )
        xlim, ylim = fixed_limits(score, 0, 1)
        axes[row, 1].set_xlim(*xlim)
        axes[row, 1].set_ylim(*ylim)
        axes[row, 1].set_aspect("equal", adjustable="box")
        axes[row, 1].set_title("28 paired trajectories; color advances in time")
        axes[row, 1].set_xlabel("PC1")
        axes[row, 1].set_ylabel("PC2")

        ratio = result["explained_variance_ratio"]
        coordinates = np.arange(1, len(ratio) + 1)
        axes[row, 2].bar(coordinates, ratio, color="#3977a8", alpha=0.82, label="per PC")
        axes[row, 2].plot(
            coordinates,
            np.cumsum(ratio),
            marker="o",
            color="#d45539",
            label="cumulative",
        )
        axes[row, 2].set_xticks(coordinates)
        axes[row, 2].set_ylim(0.0, 1.06)
        axes[row, 2].set_xlabel("principal component")
        axes[row, 2].set_ylabel("explained variance")
        axes[row, 2].set_title(
            f"spectrum: PR={result['participation_ratio']:.2f}, "
            f"PC1+2={np.sum(ratio[:2]):.3f}"
        )
        axes[row, 2].legend(frameon=False, loc="center right")
        for axis in axes[row]:
            axis.grid(alpha=0.18)
    fig.text(
        0.5,
        -0.005,
        "T² is the canonical Clifford torus in R⁴; overlap in a 2-D PCA projection is expected and is not a topology test.",
        ha="center",
        fontsize=10,
        color="#4d5966",
    )
    save_figure(fig, output, "fig_manifold_id_pca_overview")


def plot_three_dimensional(
    batches: dict[str, ManifoldBatch],
    pca: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    fig = plt.figure(figsize=(16, 5.3), constrained_layout=True)
    fig.suptitle("Three-dimensional PCA views of the fixed ID banks", fontsize=17, fontweight="bold")
    for column, name in enumerate(TOPOLOGY_ORDER, start=1):
        axis = fig.add_subplot(1, 3, column, projection="3d")
        batch = batches[name]
        score = pca[name]["scores"]
        color, color_label, cmap, vmin, vmax = latent_color(batch)
        flat_score = score.reshape(-1, score.shape[-1])
        flat_color = color.reshape(-1)
        selected = sample_flat_indices(flat_score.shape[0], 18_000)
        z = (
            flat_score[selected, 2]
            if flat_score.shape[1] >= 3
            else np.zeros(selected.shape[0])
        )
        cloud = axis.scatter(
            flat_score[selected, 0],
            flat_score[selected, 1],
            z,
            c=flat_color[selected],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=2,
            alpha=0.26,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(TOPOLOGY_LABEL[name])
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.set_zlabel("PC3" if flat_score.shape[1] >= 3 else "zero plane")
        fig.colorbar(cloud, ax=axis, label=color_label, shrink=0.62, pad=0.08)
    save_figure(fig, output, "fig_manifold_id_pca_3d")


def plot_snapshots(
    batches: dict[str, ManifoldBatch],
    pca: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    horizon = next(iter(batches.values())).horizon
    snapshots = (0, horizon // 4, horizon // 2, horizon)
    fig, axes = plt.subplots(3, len(snapshots), figsize=(16, 11.6), constrained_layout=True)
    fig.suptitle(
        "ID state distribution at fixed times in each bank's frozen PCA plane",
        fontsize=18,
        fontweight="bold",
    )
    for row, name in enumerate(TOPOLOGY_ORDER):
        batch = batches[name]
        score = pca[name]["scores"]
        color, color_label, cmap, vmin, vmax = latent_color(batch)
        xlim, ylim = fixed_limits(score, 0, 1)
        last_cloud = None
        for column, time in enumerate(snapshots):
            axis = axes[row, column]
            last_cloud = axis.scatter(
                score[time, :, 0],
                score[time, :, 1],
                c=color[time],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=8,
                alpha=0.66,
                linewidths=0,
                rasterized=True,
            )
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"{TOPOLOGY_LABEL[name]}, t={time}")
            axis.set_xlabel("PC1")
            if column == 0:
                axis.set_ylabel("PC2")
            axis.grid(alpha=0.17)
        fig.colorbar(last_cloud, ax=axes[row, :], label=color_label, shrink=0.68)
    save_figure(fig, output, "fig_manifold_id_pca_time_snapshots")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.bank_root.expanduser().resolve(strict=True)
    output = (args.output or (root / "figures")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    batches = {name: load_manifold_bank(root / f"id_{name}.npz") for name in TOPOLOGY_ORDER}
    pca = {name: fit_pca(embedded_path(batch)) for name, batch in batches.items()}

    plot_overview(batches, pca, output)
    plot_three_dimensional(batches, pca, output)
    plot_snapshots(batches, pca, output)
    metrics = {
        "source_root": str(root),
        "source_bank_sha256": {
            name: sha256_file(root / f"id_{name}.npz") for name in TOPOLOGY_ORDER
        },
        "pca_fit_data": "initial_memory concatenated with all output_targets",
        "pca_centering": "global mean per topology",
        "topologies": {
            name: {
                "state_dimension": int(batches[name].output_targets.shape[-1]),
                "state_count": int((batches[name].horizon + 1) * batches[name].trajectories),
                "eigenvalues": pca[name]["eigenvalues"].tolist(),
                "explained_variance_ratio": pca[name]["explained_variance_ratio"].tolist(),
                "participation_ratio": pca[name]["participation_ratio"],
                "top_two_explained_variance": float(
                    np.sum(pca[name]["explained_variance_ratio"][:2])
                ),
            }
            for name in TOPOLOGY_ORDER
        },
        "interpretation_note": (
            "PCA is a linear visualization. In particular, the T2 Clifford torus "
            "lives in R4 and can self-overlap after projection to two or three PCs."
        ),
    }
    atomic_json(output / "pca_metrics.json", metrics)
    atomic_json(
        output / "figure_manifest.json",
        {
            "metrics": "pca_metrics.json",
            "figures": [
                "fig_manifold_id_pca_overview.png",
                "fig_manifold_id_pca_overview.pdf",
                "fig_manifold_id_pca_3d.png",
                "fig_manifold_id_pca_3d.pdf",
                "fig_manifold_id_pca_time_snapshots.png",
                "fig_manifold_id_pca_time_snapshots.pdf",
            ],
        },
    )


if __name__ == "__main__":
    main()
