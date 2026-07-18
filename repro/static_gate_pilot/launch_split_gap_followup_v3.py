"""Continue CA-proxy finalists from the 2k screen to 10k updates."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


BASELINES = {
    "s1": {"task": 0.01785704493522644, "blank": 0.459777295589447},
    "t2": {"task": 0.03607459366321564, "blank": 0.7905378341674805},
    "s2": {"task": 0.04734647274017334, "blank": 1.089925765991211},
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_refinement_screen_dynamics_v3/"
            "candidate_dynamics_summary.csv"
        ),
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_side_pilot_v1/train_pool_seed10"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_gap_followup_v3"
        ),
    )
    parser.add_argument("--screen-updates", type=int, default=2000)
    parser.add_argument("--total-updates", type=int, default=10000)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    rows = list(
        csv.DictReader(args.candidates.expanduser().resolve(strict=True).open())
    )
    train_cache = args.train_cache.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = []
    for topology in ("s1", "t2", "s2"):
        group = [row for row in rows if row["topology"] == topology]
        reference = BASELINES[topology]
        for row in group:
            row["_performance_score"] = (
                float(row["id_intrinsic_rad"]) / reference["task"]
                + float(row["blank2048_decoded_rad"]) / reference["blank"]
            )
        choices = [
            (
                "best_gap",
                max(group, key=lambda row: float(row["tangent_normal_gap"])),
            ),
            (
                "best_normal",
                min(group, key=lambda row: float(row["normal_gain"])),
            ),
            (
                "best_performance",
                min(group, key=lambda row: float(row["_performance_score"])),
            ),
        ]
        seen = set()
        for reason, row in choices:
            if row["job_id"] in seen:
                continue
            seen.add(row["job_id"])
            selected.append(
                {
                    **{
                        key: value
                        for key, value in row.items()
                        if key != "_performance_score"
                    },
                    "followup_reason": reason,
                    "screen_performance_score": row["_performance_score"],
                }
            )
    _write_csv(output / "selected_2k_candidates.csv", selected)

    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    additional = int(args.total_updates) - int(args.screen_updates)
    if additional <= 0:
        raise ValueError("total updates must exceed screen updates")

    def run(index: int, row: dict) -> dict:
        source_cell = row["job_id"].split("__")[3]
        cell_id = f"10k_{row['followup_reason']}_{source_cell}"
        job_id = (
            f"pretrain__split_rnn_rp__{row['topology']}__{cell_id}__seed10"
        )
        device = devices[index % len(devices)]
        if (output / job_id / "COMPLETED.json").is_file():
            return {
                "job_id": job_id,
                "returncode": 0,
                "device": device,
                "skipped_completed": True,
            }
        command = [
            sys.executable,
            "-m",
            "repro.static_gate_pilot.run",
            "--phase",
            "pretrain",
            "--cell-id",
            cell_id,
            "--model",
            "split_rnn_rp",
            "--topology",
            row["topology"],
            "--seed",
            "10",
            "--width",
            row["width"],
            "--updates",
            str(additional),
            "--learning-rate",
            row["learning_rate"],
            "--initial-retention",
            row["initial_retention"],
            "--initial-write-gain",
            row["initial_write_gain"],
            "--recurrent-gain",
            row["recurrent_gain"],
            "--report-interval",
            "100",
            "--disable-rp",
            "--load-checkpoint",
            row["checkpoint"],
            "--load-optimizer",
            "--data-update-offset",
            str(args.screen_updates),
            "--train-cache",
            str(train_cache),
            "--device",
            device,
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        return {
            "job_id": job_id,
            "returncode": int(completed.returncode),
            "device": device,
            "skipped_completed": False,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-8000:],
        }

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run, index, row): row
            for index, row in enumerate(selected)
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"[{len(records):02d}/{len(selected)}] {record['job_id']} "
                f"{'ok' if record['returncode'] == 0 else 'FAILED'}",
                flush=True,
            )
            if record["returncode"] != 0:
                print(record["stderr"], flush=True)
    failures = [record for record in records if record["returncode"]]
    if failures:
        raise SystemExit(f"{len(failures)} gap-followup jobs failed")

    result_rows = []
    for result_path in sorted(output.glob("pretrain__*/result.json")):
        result = json.loads(result_path.read_text())
        manifest = json.loads((result_path.parent / "manifest.json").read_text())
        source = next(
            row
            for row in selected
            if row["topology"] == result["topology"]
            and row["followup_reason"] in result["job_id"]
        )
        result_rows.append(
            {
                "job_id": result["job_id"],
                "topology": result["topology"],
                "seed": result["replicate_seed"],
                "followup_reason": source["followup_reason"],
                "width": manifest["model"]["width"],
                "learning_rate": manifest["learning_rate"],
                "initial_retention": manifest["initial_retention"],
                "initial_write_gain": manifest["initial_write_gain"],
                "recurrent_gain": manifest["recurrent_gain"],
                "id_intrinsic_rad": result["final_validation"][
                    "intrinsic_mean_radians"
                ],
                "blank512_rad": result["blank_validation"]["512"][
                    "intrinsic_mean_radians"
                ],
                "blank2048_rad": result["blank_validation"]["2048"][
                    "intrinsic_mean_radians"
                ],
                "checkpoint": str(result_path.parent / "checkpoint.pt"),
            }
        )
    if len(result_rows) != len(selected):
        raise ValueError(
            f"expected {len(selected)} 10k results, found {len(result_rows)}"
        )
    _write_csv(output / "full_summary.csv", result_rows)
    (output / "pipeline_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    print(output / "full_summary.csv")


if __name__ == "__main__":
    main()
