"""Plot the staged static-gate pretraining and RP hyperparameter sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = {
    "untied_rnn_rp": "Shared field",
    "split_rnn_rp": "Split fields",
}


def _screen_rows(root: Path) -> list[dict]:
    rows = []
    for result_path in sorted(root.glob("pretrain__*/result.json")):
        result = json.loads(result_path.read_text())
        manifest = json.loads((result_path.parent / "manifest.json").read_text())
        rows.append(
            {
                "model": result["model_id"],
                "topology": result["topology"],
                "learning_rate": float(manifest["learning_rate"]),
                "initial_retention": float(
                    manifest["model"]["initial_retention"]
                ),
                "task": float(
                    result["final_validation"]["intrinsic_mean_radians"]
                ),
                "blank512": float(
                    result["blank_validation"]["512"]["intrinsic_mean_radians"]
                ),
            }
        )
    if len(rows) != 36:
        raise ValueError(f"expected 36 screen cells, found {len(rows)}")
    return rows


def _heatmaps(rows: list[dict], output: Path, metric: str, title: str) -> None:
    learning_rates = sorted({row["learning_rate"] for row in rows})
    retentions = sorted({row["initial_retention"] for row in rows})
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    images = []
    for row_index, topology in enumerate(("s1", "t2")):
        for column_index, model in enumerate(("untied_rnn_rp", "split_rnn_rp")):
            axis = axes[row_index, column_index]
            matrix = np.full((len(retentions), len(learning_rates)), np.nan)
            for row in rows:
                if row["model"] != model or row["topology"] != topology:
                    continue
                y = retentions.index(row["initial_retention"])
                x = learning_rates.index(row["learning_rate"])
                matrix[y, x] = row[metric]
            image = axis.imshow(
                np.log10(matrix),
                origin="lower",
                aspect="auto",
                cmap="viridis_r",
            )
            images.append(image)
            for y in range(len(retentions)):
                for x in range(len(learning_rates)):
                    axis.text(
                        x,
                        y,
                        f"{matrix[y, x]:.3g}",
                        ha="center",
                        va="center",
                        color=(
                            "white"
                            if np.log10(matrix[y, x])
                            > np.nanmedian(np.log10(matrix))
                            else "black"
                        ),
                        fontsize=8,
                    )
            axis.set_xticks(
                range(len(learning_rates)),
                [f"{value:g}" for value in learning_rates],
            )
            axis.set_yticks(
                range(len(retentions)),
                [f"{value:g}" for value in retentions],
            )
            axis.set_xlabel("Learning rate")
            axis.set_ylabel(r"Initial retention $\lambda_0$")
            axis.set_title(f"{topology.upper()} — {MODEL_LABELS[model]}")
    figure.colorbar(
        images[-1],
        ax=axes,
        label=f"log10 {title.lower()}",
        shrink=0.75,
    )
    figure.suptitle(f"1,500-update screen — {title}", fontweight="bold")
    stem = f"fig_screen_{metric}"
    figure.savefig(output / f"{stem}.png", dpi=220)
    figure.savefig(output / f"{stem}.pdf")
    plt.close(figure)


def _rp_scatter(root: Path, output: Path) -> None:
    branches: list[dict] = []
    controls: dict[tuple[str, str], dict] = {}
    for result_path in sorted(root.glob("rp__*/result.json")):
        result = json.loads(result_path.read_text())
        manifest = json.loads((result_path.parent / "manifest.json").read_text())
        row = {
            "model": result["model_id"],
            "topology": result["topology"],
            "task": float(result["final_validation"]["intrinsic_mean_radians"]),
            "blank2048": float(
                result["blank_validation"]["2048"]["intrinsic_mean_radians"]
            ),
            "rp": bool(result["rp_active"]),
        }
        if row["rp"]:
            config = manifest["gate_intervention_rp"]
            row.update(
                {
                    "eta": float(config["eta_lambda"]),
                    "rule": str(config["update_rule"]),
                    "cap": float(config["max_theta_step"]),
                }
            )
            branches.append(row)
        else:
            controls[(row["model"], row["topology"])] = row
    if len(branches) != 32 or len(controls) != 4:
        raise ValueError(
            f"expected 32 RP branches and 4 controls, found "
            f"{len(branches)} and {len(controls)}"
        )
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    color = {0.1: "#4f79a7", 1.0: "#c84e4e"}
    marker = {"signed": "o", "positive_only": "^"}
    for row_index, topology in enumerate(("s1", "t2")):
        for column_index, model in enumerate(("untied_rnn_rp", "split_rnn_rp")):
            axis = axes[row_index, column_index]
            group = [
                row
                for row in branches
                if row["model"] == model and row["topology"] == topology
            ]
            for row in group:
                axis.scatter(
                    row["task"],
                    row["blank2048"],
                    color=color[row["eta"]],
                    marker=marker[row["rule"]],
                    s=55 if row["cap"] == 0.02 else 30,
                    alpha=0.85,
                )
            control = controls[(model, topology)]
            axis.scatter(
                control["task"],
                control["blank2048"],
                marker="*",
                color="black",
                s=130,
                label="same-update no-RP control",
                zorder=5,
            )
            best = min(
                group,
                key=lambda row: (
                    row["blank2048"],
                    row["task"],
                ),
            )
            axis.annotate(
                f"best raw\nη={best['eta']:g}, {best['rule']}, cap={best['cap']:g}",
                (best["task"], best["blank2048"]),
                xytext=(7, -5),
                textcoords="offset points",
                fontsize=7,
            )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel("Task intrinsic error (rad)")
            axis.set_ylabel("Blank-2048 error (rad)")
            axis.set_title(f"{topology.upper()} — {MODEL_LABELS[model]}")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
    figure.suptitle(
        "Calibrated RP sweep (color: η; marker: update rule; size: step cap)",
        fontweight="bold",
    )
    figure.savefig(output / "fig_rp_task_memory_tradeoff.png", dpi=220)
    figure.savefig(output / "fig_rp_task_memory_tradeoff.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen-root",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_pretrain_screen_v1"
        ),
    )
    parser.add_argument(
        "--rp-root",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_rp_sweep_v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_sweep_figures_v1"
        ),
    )
    args = parser.parse_args()
    screen_root = args.screen_root.expanduser().resolve(strict=True)
    rp_root = args.rp_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = _screen_rows(screen_root)
    _heatmaps(rows, output, "task", "Task intrinsic error")
    _heatmaps(rows, output, "blank512", "Blank-512 intrinsic error")
    _rp_scatter(rp_root, output)
    print(output)


if __name__ == "__main__":
    main()
