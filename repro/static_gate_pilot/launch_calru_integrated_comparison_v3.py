"""Reanalyze selected CA-LRU checkpoints and add them to the split-field comparison."""

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


MODEL_ORDER = ("rnn", "gru", "lstm", "calru", "split")
MODEL_LABELS = {
    "rnn": "RNN",
    "gru": "GRU",
    "lstm": "LSTM",
    "calru": "CA-LRU",
    "split": "Split-field",
}
COLORS = {
    "rnn": "#8b95a5",
    "gru": "#4f79a7",
    "lstm": "#6a9f58",
    "calru": "#2a9d8f",
    "split": "#c84e4e",
}
METRIC_KEYS = (
    "task",
    "blank2048",
    "tangent_error",
    "normal_gain",
    "gap",
    "recovery",
    "shape",
    "topology_h2048_match",
)


def _metric_value(value: object) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
    return float(value)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _selected_calru_records(selection_path: Path, run_root: Path) -> list[dict]:
    selection = json.loads(selection_path.read_text())
    records = []
    for topology in ("s1", "t2", "s2"):
        selected = selection["selected"][topology]
        for row in selected["rows"]:
            run_dir = run_root / row["job_id"]
            checkpoint = run_dir / "checkpoint.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            records.append(
                {
                    "model": "calru",
                    "topology": topology,
                    "seed": int(row["seed"]),
                    "checkpoint": checkpoint,
                    "cell_id": selected["cell"]["cell_id"],
                }
            )
    if len(records) != 9:
        raise ValueError(f"expected 9 selected CA-LRU checkpoints, found {len(records)}")
    return records


def _row_from_analysis(record: dict, target: Path) -> dict:
    dynamics = json.loads((target / "dynamics.json").read_text())
    topology = json.loads((target / "topology.json").read_text())
    return {
        "model": "calru",
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
        "topology_h2048_match": topology["horizons"]["2048"]["signature_match"],
        "checkpoint": str(record["checkpoint"]),
        "cell_id": record["cell_id"],
    }


def _medians(rows: list[dict]) -> list[dict]:
    summaries = []
    for topology in ("s1", "t2", "s2"):
        for model in MODEL_ORDER:
            group = [
                row
                for row in rows
                if row["model"] == model and row["topology"] == topology
            ]
            if len(group) != 3:
                raise ValueError(
                    f"expected 3 rows for {model}/{topology}, found {len(group)}"
                )
            summaries.append(
                {
                    "topology": topology,
                    "model": model,
                    **{
                        key: float(
                            np.median([_metric_value(row[key]) for row in group])
                        )
                        for key in METRIC_KEYS
                    },
                }
            )
    return summaries


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
    figure, axes = plt.subplots(2, 4, figsize=(20, 8.5), constrained_layout=True)
    rng = np.random.default_rng(37)
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
                        _metric_value(row[key])
                        for row in rows
                        if row["model"] == model and row["topology"] == topology
                    ]
                )
                axis.scatter(
                    position + rng.uniform(-0.06, 0.06, len(values)),
                    values,
                    color=COLORS[model],
                    s=23,
                    alpha=0.75,
                )
                median = float(np.median(values))
                axis.plot(
                    [position - 0.22, position + 0.22],
                    [median, median],
                    color=COLORS[model],
                    linewidth=3,
                )
                positions.append(position)
                labels.append(f"{topology.upper()}\n{MODEL_LABELS[model]}")
                position += 1.0
            position += 0.8
        if log_scale and any(_metric_value(row[key]) > 0.0 for row in rows):
            axis.set_yscale("log")
        if reference is not None:
            axis.axhline(reference, color="#222222", linestyle="--", linewidth=1)
        axis.set_xticks(positions, labels, fontsize=6.5, rotation=27)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="bold", fontsize=10)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "CA-LRU and split-field approximate-CA evidence versus recurrent baselines",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output / "fig_calru_split_vs_baselines.png", dpi=220)
    figure.savefig(output / "fig_calru_split_vs_baselines.pdf")
    plt.close(figure)


def _report(rows: list[dict], output: Path) -> None:
    medians = _medians(rows)
    _write_csv(output / "model_topology_medians.csv", medians)
    verdicts = []
    for topology in ("s1", "t2", "s2"):
        indexed = {
            row["model"]: row
            for row in medians
            if row["topology"] == topology
        }
        conventional = [indexed[model] for model in ("rnn", "gru", "lstm")]
        for candidate in ("calru", "split"):
            row = indexed[candidate]
            criteria = {
                "beats_best_conventional_task": row["task"]
                < min(item["task"] for item in conventional),
                "beats_best_conventional_blank": row["blank2048"]
                < min(item["blank2048"] for item in conventional),
                "tangent_near_neutral": row["tangent_error"] < 0.1,
                "local_normal_contracting": row["normal_gain"] < 1.0,
                "positive_tangent_normal_gap": row["gap"] > 0.0,
                "finite_normal_recovery": row["recovery"] < 1.0,
                "topology_survives_majority": row["topology_h2048_match"] >= 2 / 3,
            }
            verdicts.append(
                {
                    "topology": topology,
                    "candidate": candidate,
                    **criteria,
                    "strict_approximate_ca_success": all(criteria.values()),
                }
            )
    _write_csv(output / "candidate_verdicts.csv", verdicts)

    lines = [
        "# CA-LRU-inclusive approximate continuous-attractor comparison",
        "",
        "All five models use three checkpoints per topology. CA-LRU uses the "
        "validation-selected 5,000-update topology-tuning-v2 cell and is "
        "reanalyzed here with exactly the same local Jacobian, finite-kick, "
        "blank-evolution, and persistent-homology implementation as split-field.",
        "",
        "| Topology | Model | Task rad | Blank-2048 rad | Tangent error | "
        "Worst normal gain | Gap | Recovery | Shape distortion | Topology survival |",
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
    lines.extend(["", "## Candidate verdicts", ""])
    for row in verdicts:
        lines.append(
            f"- {row['topology'].upper()} / {MODEL_LABELS[row['candidate']]}: "
            f"{'PASS' if row['strict_approximate_ca_success'] else 'FAIL'}."
        )
    lines.extend(
        [
            "",
            "RNN, GRU, and LSTM are the conventional baselines. CA-LRU and "
            "split-field are reported as candidate architectures, so CA-LRU is "
            "not folded into the phrase “best baseline.”",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calru-run-root",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "manifold_calru_topology_tuning_v2-e722ccf"
        ),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "manifold_calru_topology_tuning_v2-e722ccf/"
            "selection/FINAL_SELECTION.json"
        ),
    )
    parser.add_argument(
        "--existing-comparison",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_refinement_final_analysis_v3/all_seed_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_calru_integrated_comparison_v3"
        ),
    )
    parser.add_argument("--devices", default="cuda:3,cuda:4,cuda:5")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    run_root = args.calru_run_root.expanduser().resolve(strict=True)
    records = _selected_calru_records(
        args.selection.expanduser().resolve(strict=True), run_root
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]

    def run(index: int, record: dict) -> dict:
        target = output / (
            f"calru__{record['topology']}__seed{record['seed']}"
        )
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
            if completed.returncode:
                return {
                    **record,
                    "returncode": int(completed.returncode),
                    "stderr": stderr,
                }
        return {**record, "returncode": 0, "stderr": stderr}

    completed_records = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {
            executor.submit(run, index, record): record
            for index, record in enumerate(records)
        }
        for future in as_completed(futures):
            record = future.result()
            completed_records.append(record)
            print(
                f"[{len(completed_records):02d}/09] CA-LRU/"
                f"{record['topology']}/seed{record['seed']} "
                f"{'ok' if record['returncode'] == 0 else 'FAILED'}",
                flush=True,
            )
            if record["returncode"]:
                print(record["stderr"], flush=True)
    failures = [row for row in completed_records if row["returncode"]]
    if failures:
        raise SystemExit(f"{len(failures)} CA-LRU analyses failed")

    existing_rows = list(
        csv.DictReader(
            args.existing_comparison.expanduser().resolve(strict=True).open()
        )
    )
    existing_rows = [row for row in existing_rows if row["model"] != "calru"]
    calru_rows = []
    for record in records:
        target = output / (
            f"calru__{record['topology']}__seed{record['seed']}"
        )
        calru_rows.append(_row_from_analysis(record, target))
    all_rows = existing_rows + calru_rows
    _write_csv(output / "all_seed_metrics.csv", all_rows)
    _figure(all_rows, output)
    _report(all_rows, output)
    (output / "pipeline_records.json").write_text(
        json.dumps(completed_records, indent=2, default=str) + "\n"
    )
    print(output / "RESULTS.md")


if __name__ == "__main__":
    main()
