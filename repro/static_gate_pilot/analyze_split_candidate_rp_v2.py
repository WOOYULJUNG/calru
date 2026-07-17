"""Analyze RP/control candidates and select one split-field setting per topology."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _baseline_references(path: Path) -> dict[str, dict[str, float]]:
    rows = list(csv.DictReader(path.open()))
    references = {}
    for topology in ("s1", "t2", "s2"):
        model_medians = []
        for model in ("rnn", "gru", "lstm"):
            group = [
                row
                for row in rows
                if row["topology"] == topology and row["model"] == model
            ]
            if len(group) != 3:
                raise ValueError(
                    f"missing 3-seed baseline {model}/{topology}: {len(group)}"
                )
            model_medians.append(
                {
                    "model": model,
                    "task": float(
                        np.median(
                            [float(row["task_intrinsic_radians"]) for row in group]
                        )
                    ),
                    "blank2048": float(
                        np.median(
                            [float(row["blank2048_memory_radians"]) for row in group]
                        )
                    ),
                }
            )
        references[topology] = {
            "best_task": min(row["task"] for row in model_medians),
            "best_blank2048": min(row["blank2048"] for row in model_medians),
            "best_task_model": min(model_medians, key=lambda row: row["task"])[
                "model"
            ],
            "best_blank_model": min(
                model_medians, key=lambda row: row["blank2048"]
            )["model"],
        }
    return references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_candidate_rp_v2/rp_results.csv"
        ),
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_baseline_dynamics_v1/baseline_dynamics_summary.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_candidate_rp_dynamics_v2"
        ),
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    candidates = list(
        csv.DictReader(args.candidates.expanduser().resolve(strict=True).open())
    )
    references = _baseline_references(
        args.baselines.expanduser().resolve(strict=True)
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]

    def run(index: int, row: dict[str, str]) -> dict:
        target = output / f"rp_candidate_{index:02d}__{row['topology']}"
        device = devices[index % len(devices)]
        if (target / "dynamics.json").is_file():
            return {"index": index, "returncode": 0, "skipped_completed": True}
        command = [
            sys.executable,
            "-m",
            "repro.static_gate_pilot.analyze_checkpoint_dynamics",
            "--checkpoint",
            row["checkpoint"],
            "--output",
            str(target),
            "--device",
            device,
            "--trajectories",
            "128",
            "--anchors",
            "12",
            "--neighbors",
            "12",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        return {
            "index": index,
            "returncode": int(completed.returncode),
            "skipped_completed": False,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-8000:],
        }

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run, index, row): row
            for index, row in enumerate(candidates)
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"[{len(records):02d}/{len(candidates)}] RP candidate "
                f"{record['index']:02d} "
                f"{'ok' if record['returncode'] == 0 else 'FAILED'}",
                flush=True,
            )
            if record["returncode"] != 0:
                print(record["stderr"], flush=True)
    failures = [record for record in records if record["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} RP dynamics analyses failed")

    rows = []
    for index, candidate in enumerate(candidates):
        data = json.loads(
            (
                output
                / f"rp_candidate_{index:02d}__{candidate['topology']}"
                / "dynamics.json"
            ).read_text()
        )
        reference = references[candidate["topology"]]
        task = float(candidate["id_intrinsic_rad"])
        blank = float(data["blank_manifold_evolution"]["2048"][
            "decoded_memory_intrinsic_radians"
        ])
        tangent = float(data["tangent_singular_mean"])
        normal = float(data["normal_max_singular_mean"])
        recovery = float(
            data["finite_local_normal_recovery"]["512"][
                "distance_ratio_median"
            ]
        )
        shape = float(
            data["blank_manifold_evolution"]["2048"][
                "pairwise_shape_distortion_std"
            ]
        )
        ca_score = (
            abs(tangent - 1.0)
            + max(normal - 1.0, 0.0)
            + recovery
            + shape
        )
        baseline_score = (
            math.log(max(task, 1e-12) / reference["best_task"])
            + math.log(max(blank, 1e-12) / reference["best_blank2048"])
        )
        rows.append(
            {
                **candidate,
                "task_baseline_model": reference["best_task_model"],
                "task_baseline_median": reference["best_task"],
                "blank_baseline_model": reference["best_blank_model"],
                "blank_baseline_median": reference["best_blank2048"],
                "task_ratio_to_best_baseline": task / reference["best_task"],
                "blank_ratio_to_best_baseline": blank
                / reference["best_blank2048"],
                "beats_best_baseline_task": task < reference["best_task"],
                "beats_best_baseline_blank2048": blank
                < reference["best_blank2048"],
                "tangent_gain": tangent,
                "normal_gain": normal,
                "tangent_normal_gap": data["tangent_normal_gap_mean"],
                "normal_recovery_h512": recovery,
                "normal_same_memory_h512": data[
                    "finite_local_normal_recovery"
                ]["512"]["same_memory_intrinsic_radians"],
                "tangent_memory_shift_h512": data[
                    "finite_local_tangent_transport"
                ]["512"]["memory_shift_intrinsic_radians"],
                "global_scale_h2048": data["blank_manifold_evolution"][
                    "2048"
                ]["best_global_scale"],
                "scaling_residual_h2048": data["blank_manifold_evolution"][
                    "2048"
                ]["global_scaling_residual"],
                "shape_distortion_h2048": shape,
                "ca_score": ca_score,
                "baseline_performance_score": baseline_score,
                "combined_selection_score": baseline_score + ca_score,
                "dynamics_json": str(
                    output
                    / f"rp_candidate_{index:02d}__{candidate['topology']}"
                    / "dynamics.json"
                ),
            }
        )
    _write_csv(output / "rp_dynamics_summary.csv", rows)

    finalists = []
    for topology in ("s1", "t2", "s2"):
        group = [row for row in rows if row["topology"] == topology]
        strict = [
            row
            for row in group
            if str(row["beats_best_baseline_task"]) == "True"
            and str(row["beats_best_baseline_blank2048"]) == "True"
        ]
        pool = strict if strict else group
        best = min(pool, key=lambda row: float(row["combined_selection_score"]))
        finalists.append(
            {
                **best,
                "strict_baseline_gate_passed": bool(strict),
            }
        )
    _write_csv(output / "finalists.csv", finalists)
    (output / "selection.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "baseline_references": references,
                "finalists": finalists,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(output / "finalists.csv")


if __name__ == "__main__":
    main()
