"""Analyze all 5,000-update broad split-field finalists."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_broad_full_v2/full_summary.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_candidate_dynamics_v2"
        ),
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    candidates = list(
        csv.DictReader(args.candidates.expanduser().resolve(strict=True).open())
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]

    def run(index: int, row: dict[str, str]) -> dict:
        target = output / f"candidate_{index:02d}__{row['topology']}"
        device = devices[index % len(devices)]
        if (target / "dynamics.json").is_file():
            return {
                "index": index,
                "returncode": 0,
                "skipped_completed": True,
            }
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
                f"[{len(records):02d}/{len(candidates)}] candidate "
                f"{record['index']:02d} "
                f"{'ok' if record['returncode'] == 0 else 'FAILED'}",
                flush=True,
            )
            if record["returncode"] != 0:
                print(record["stderr"], flush=True)
    failures = [record for record in records if record["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} candidate analyses failed")

    rows = []
    for index, candidate in enumerate(candidates):
        data = json.loads(
            (output / f"candidate_{index:02d}__{candidate['topology']}" / "dynamics.json").read_text()
        )
        rows.append(
            {
                **candidate,
                "candidate_index": index,
                "normalized_fixedness": data["normalized_fixedness_median"],
                "tangent_gain": data["tangent_singular_mean"],
                "normal_gain": data["normal_max_singular_mean"],
                "tangent_normal_gap": data["tangent_normal_gap_mean"],
                "normal_recovery_h512": data[
                    "finite_local_normal_recovery"
                ]["512"]["distance_ratio_median"],
                "normal_same_memory_h512": data[
                    "finite_local_normal_recovery"
                ]["512"]["same_memory_intrinsic_radians"],
                "tangent_memory_shift_h512": data[
                    "finite_local_tangent_transport"
                ]["512"]["memory_shift_intrinsic_radians"],
                "blank2048_decoded_rad": data["blank_manifold_evolution"][
                    "2048"
                ]["decoded_memory_intrinsic_radians"],
                "blank2048_global_scale": data["blank_manifold_evolution"][
                    "2048"
                ]["best_global_scale"],
                "blank2048_scaling_residual": data[
                    "blank_manifold_evolution"
                ]["2048"]["global_scaling_residual"],
                "blank2048_shape_distortion": data[
                    "blank_manifold_evolution"
                ]["2048"]["pairwise_shape_distortion_std"],
                "dynamics_json": str(
                    output
                    / f"candidate_{index:02d}__{candidate['topology']}"
                    / "dynamics.json"
                ),
            }
        )
    with (output / "candidate_dynamics_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "pipeline_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    print(output / "candidate_dynamics_summary.csv")


if __name__ == "__main__":
    main()
