"""Final 3-seed split-field CA analysis, baseline comparison, and figures."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ("rnn", "gru", "lstm", "split")
MODEL_LABELS = {
    "rnn": "RNN",
    "gru": "GRU",
    "lstm": "LSTM",
    "split": "Split-field",
}
COLORS = {
    "rnn": "#8b95a5",
    "gru": "#4f79a7",
    "lstm": "#6a9f58",
    "split": "#c84e4e",
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _records(selection: Path, final_root: Path) -> list[dict]:
    chosen = list(csv.DictReader(selection.open()))
    records = [
        {
            "topology": row["topology"],
            "seed": 10,
            "checkpoint": Path(row["checkpoint"]),
        }
        for row in chosen
    ]
    for checkpoint in sorted(final_root.glob("*/checkpoint.pt")):
        manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
        records.append(
            {
                "topology": manifest["topology"],
                "seed": int(manifest["replicate_seed"]),
                "checkpoint": checkpoint,
            }
        )
    return records


def _load_baselines(
    dynamics_path: Path, persistence_path: Path
) -> list[dict]:
    persistence_rows = list(csv.DictReader(persistence_path.open()))
    signature = {
        (row["model"], row["topology"], int(row["seed"])): (
            str(row["signature_match"]).lower() == "true"
        )
        for row in persistence_rows
        if int(row["horizon"]) == 2048
        and row["model"] in {"rnn", "gru", "lstm"}
    }
    rows = []
    for row in csv.DictReader(dynamics_path.open()):
        key = (row["model"], row["topology"], int(row["seed"]))
        rows.append(
            {
                "model": row["model"],
                "topology": row["topology"],
                "seed": int(row["seed"]),
                "task": float(row["task_intrinsic_radians"]),
                "blank2048": float(row["blank2048_memory_radians"]),
                "tangent_error": abs(
                    float(row["tangent_singular_mean"]) - 1.0
                ),
                "normal_gain": float(row["normal_max_singular_mean"]),
                "gap": float(row["tangent_normal_gap_mean"]),
                "recovery": float(row["normal_recovery_ratio_h512"]),
                "shape": float(row["shape_distortion_h2048"]),
                "topology_h2048_match": signature.get(key, False),
            }
        )
    return rows


def _figure(rows: list[dict], output: Path) -> None:
    metrics = [
        ("Task error", "task", "rad", True, None),
        ("Blank memory H=2048", "blank2048", "rad", True, None),
        ("Tangent neutrality", "tangent_error", r"$|\sigma_t-1|$", True, None),
        ("Worst normal gain", "normal_gain", r"$\sigma_{n,max}$", False, 1.0),
        ("Tangent-normal gap", "gap", r"$\sigma_t-\sigma_{n,max}$", False, 0.0),
        ("Finite normal recovery", "recovery", "distance ratio", True, 1.0),
        ("Shape distortion H=2048", "shape", "log-distance std", True, None),
        ("Topology survival H=2048", "topology_h2048_match", "fraction", False, 1.0),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(17, 8.3), constrained_layout=True)
    rng = np.random.default_rng(21)
    for axis, (title, key, ylabel, log_scale, reference) in zip(
        axes.flat, metrics
    ):
        positions = []
        labels = []
        position = 0.0
        for topology in ("s1", "t2", "s2"):
            for model in MODEL_ORDER:
                values = np.asarray(
                    [
                        float(row[key])
                        for row in rows
                        if row["model"] == model
                        and row["topology"] == topology
                    ]
                )
                if len(values) != 3:
                    raise ValueError(
                        f"expected 3 values for {model}/{topology}/{key}"
                    )
                axis.scatter(
                    position + rng.uniform(-0.07, 0.07, len(values)),
                    values,
                    color=COLORS[model],
                    s=25,
                    alpha=0.75,
                )
                median = float(np.median(values))
                axis.plot(
                    [position - 0.23, position + 0.23],
                    [median, median],
                    color=COLORS[model],
                    linewidth=3,
                )
                positions.append(position)
                labels.append(f"{topology.upper()}\n{MODEL_LABELS[model]}")
                position += 1.0
            position += 0.7
        if log_scale:
            positive = [
                float(row[key]) for row in rows if float(row[key]) > 0.0
            ]
            if positive:
                axis.set_yscale("log")
        if reference is not None:
            axis.axhline(reference, color="#222222", linestyle="--", linewidth=1)
        axis.set_xticks(positions, labels, fontsize=7, rotation=25)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="bold", fontsize=10)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Split-field approximate-CA evidence versus recurrent baselines",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output / "fig_split_vs_baselines.png", dpi=220)
    figure.savefig(output / "fig_split_vs_baselines.pdf")
    plt.close(figure)


def _report(rows: list[dict], output: Path) -> None:
    medians = []
    for topology in ("s1", "t2", "s2"):
        for model in MODEL_ORDER:
            group = [
                row
                for row in rows
                if row["model"] == model and row["topology"] == topology
            ]
            medians.append(
                {
                    "topology": topology,
                    "model": model,
                    **{
                        key: float(np.median([float(row[key]) for row in group]))
                        for key in (
                            "task",
                            "blank2048",
                            "tangent_error",
                            "normal_gain",
                            "gap",
                            "recovery",
                            "shape",
                            "topology_h2048_match",
                        )
                    },
                }
            )
    _write_csv(output / "model_topology_medians.csv", medians)
    verdicts = []
    for topology in ("s1", "t2", "s2"):
        split = next(
            row
            for row in medians
            if row["model"] == "split" and row["topology"] == topology
        )
        conventional = [
            row
            for row in medians
            if row["model"] != "split" and row["topology"] == topology
        ]
        best_task = min(row["task"] for row in conventional)
        best_blank = min(row["blank2048"] for row in conventional)
        criteria = {
            "beats_best_baseline_task": split["task"] < best_task,
            "beats_best_baseline_blank": split["blank2048"] < best_blank,
            "tangent_near_neutral": split["tangent_error"] < 0.1,
            "local_normal_contracting": split["normal_gain"] < 1.0,
            "positive_tangent_normal_gap": split["gap"] > 0.0,
            "finite_normal_recovery": split["recovery"] < 1.0,
            "topology_survives_majority": split["topology_h2048_match"] >= 2 / 3,
        }
        verdicts.append(
            {
                "topology": topology,
                "split_task": split["task"],
                "best_baseline_task": best_task,
                "split_blank2048": split["blank2048"],
                "best_baseline_blank2048": best_blank,
                **criteria,
                "strict_approximate_ca_success": all(criteria.values()),
            }
        )
    _write_csv(output / "strict_verdicts.csv", verdicts)
    lines = [
        "# Split-field approximate continuous-attractor search — final report",
        "",
        "All split-field rows contain three seeds. RNN, GRU, and LSTM use their "
        "existing three-seed topology checkpoints and the same local analysis.",
        "",
        "| Topology | Model | Task rad | Blank-2048 rad | Tangent error | "
        "Normal gain | Gap | Recovery | Shape distortion | Topology survival |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in medians:
        lines.append(
            f"| {row['topology'].upper()} | {MODEL_LABELS[row['model']]} | "
            f"{row['task']:.4g} | {row['blank2048']:.4g} | "
            f"{row['tangent_error']:.4g} | {row['normal_gain']:.4g} | "
            f"{row['gap']:.4g} | {row['recovery']:.4g} | "
            f"{row['shape']:.4g} | {row['topology_h2048_match']:.3g} |"
        )
    lines.extend(["", "## Strict verdict", ""])
    for row in verdicts:
        passed = bool(row["strict_approximate_ca_success"])
        lines.append(
            f"- {row['topology'].upper()}: "
            f"{'PASS' if passed else 'FAIL'}; "
            f"task {row['split_task']:.4g} vs best baseline "
            f"{row['best_baseline_task']:.4g}, blank-2048 "
            f"{row['split_blank2048']:.4g} vs "
            f"{row['best_baseline_blank2048']:.4g}."
        )
    lines.extend(
        [
            "",
            "A low finite-kick ratio alone is not counted as attraction because "
            "global collapse can also reduce that ratio. The strict verdict also "
            "requires tangent neutrality, local normal contraction, a positive "
            "gap, topology survival, and baseline-level task performance.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_candidate_rp_dynamics_v2/finalists.csv"
        ),
    )
    parser.add_argument(
        "--final-root",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_final_seeds_v2/final"
        ),
    )
    parser.add_argument(
        "--baseline-dynamics",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_baseline_dynamics_v1/baseline_dynamics_summary.csv"
        ),
    )
    parser.add_argument(
        "--baseline-persistence",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "persistent_topology_all_models_v1/"
            "persistent_topology_seed_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_final_analysis_v2"
        ),
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    records = _records(
        args.selection.expanduser().resolve(strict=True),
        args.final_root.expanduser().resolve(strict=True),
    )
    if len(records) != 9:
        raise ValueError(f"expected 9 final split checkpoints, found {len(records)}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]

    def run(index: int, record: dict) -> dict:
        target = output / f"split__{record['topology']}__seed{record['seed']}"
        target.mkdir(parents=True, exist_ok=True)
        device = devices[index % len(devices)]
        commands = []
        if not (target / "dynamics.json").is_file():
            commands.append(
                [
                    sys.executable,
                    "-m",
                    "repro.static_gate_pilot.analyze_checkpoint_dynamics",
                    "--checkpoint",
                    str(record["checkpoint"]),
                    "--output",
                    str(target),
                    "--device",
                    device,
                ]
            )
        if not (target / "topology.json").is_file():
            commands.append(
                [
                    sys.executable,
                    "-m",
                    "repro.static_gate_pilot.analyze_checkpoint_topology",
                    "--checkpoint",
                    str(record["checkpoint"]),
                    "--output",
                    str(target),
                    "--device",
                    device,
                ]
            )
        stderr = ""
        for command in commands:
            completed = subprocess.run(command, text=True, capture_output=True)
            stderr += completed.stderr[-8000:]
            if completed.returncode != 0:
                return {**record, "returncode": int(completed.returncode), "stderr": stderr}
        return {**record, "returncode": 0, "stderr": stderr}

    completed_records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run, index, record): record
            for index, record in enumerate(records)
        }
        for future in as_completed(futures):
            record = future.result()
            completed_records.append(record)
            print(
                f"[{len(completed_records):02d}/09] split/"
                f"{record['topology']}/seed{record['seed']} "
                f"{'ok' if record['returncode'] == 0 else 'FAILED'}",
                flush=True,
            )
            if record["returncode"] != 0:
                print(record["stderr"], flush=True)
    failures = [record for record in completed_records if record["returncode"]]
    if failures:
        raise SystemExit(f"{len(failures)} final analyses failed")

    split_rows = []
    for record in records:
        target = output / f"split__{record['topology']}__seed{record['seed']}"
        dynamics = json.loads((target / "dynamics.json").read_text())
        topology = json.loads((target / "topology.json").read_text())
        split_rows.append(
            {
                "model": "split",
                "topology": record["topology"],
                "seed": record["seed"],
                "task": dynamics["task_intrinsic_radians"],
                "blank2048": dynamics["blank_manifold_evolution"]["2048"][
                    "decoded_memory_intrinsic_radians"
                ],
                "tangent_error": abs(dynamics["tangent_singular_mean"] - 1.0),
                "normal_gain": dynamics["normal_max_singular_mean"],
                "gap": dynamics["tangent_normal_gap_mean"],
                "recovery": dynamics["finite_local_normal_recovery"]["512"][
                    "distance_ratio_median"
                ],
                "shape": dynamics["blank_manifold_evolution"]["2048"][
                    "pairwise_shape_distortion_std"
                ],
                "topology_h2048_match": topology["horizons"]["2048"][
                    "signature_match"
                ],
                "checkpoint": str(record["checkpoint"]),
            }
        )
    baseline_rows = _load_baselines(
        args.baseline_dynamics.expanduser().resolve(strict=True),
        args.baseline_persistence.expanduser().resolve(strict=True),
    )
    all_rows = baseline_rows + split_rows
    _write_csv(output / "all_seed_metrics.csv", all_rows)
    _figure(all_rows, output)
    _report(all_rows, output)
    (output / "pipeline_records.json").write_text(
        json.dumps(completed_records, indent=2, default=str) + "\n"
    )
    print(output / "RESULTS.md")


if __name__ == "__main__":
    main()
