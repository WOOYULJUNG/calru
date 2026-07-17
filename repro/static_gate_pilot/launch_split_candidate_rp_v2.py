"""Choose diverse CA-aware split finalists and run calibrated RP plus controls."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


TASK_GATES = {"s1": 0.05, "t2": 0.08, "s2": 0.08}


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
            "static_gate_split_candidate_dynamics_v2/"
            "candidate_dynamics_summary.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_candidate_rp_v2"
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
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--data-update-offset", type=int, default=5000)
    parser.add_argument("--updates", type=int, default=2000)
    args = parser.parse_args()
    rows = list(
        csv.DictReader(args.candidates.expanduser().resolve(strict=True).open())
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_cache = args.train_cache.expanduser().resolve(strict=True)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]

    parents = []
    for topology in ("s1", "t2", "s2"):
        group = [row for row in rows if row["topology"] == topology]
        passing = [
            row
            for row in group
            if float(row["id_intrinsic_rad"]) < TASK_GATES[topology]
        ]
        pool = passing if passing else group
        for row in pool:
            row["_ca_score"] = (
                abs(float(row["tangent_gain"]) - 1.0)
                + max(float(row["normal_gain"]) - 1.0, 0.0)
                + float(row["normal_recovery_h512"])
                + float(row["blank2048_shape_distortion"])
            )
        selected = {
            min(pool, key=lambda row: float(row["id_intrinsic_rad"]))["job_id"],
            min(pool, key=lambda row: float(row["blank2048_decoded_rad"]))[
                "job_id"
            ],
            min(pool, key=lambda row: float(row["_ca_score"]))["job_id"],
        }
        for row in group:
            if row["job_id"] in selected:
                parents.append(
                    {
                        **{key: value for key, value in row.items() if key != "_ca_score"},
                        "parent_label": f"{topology}_p{len([p for p in parents if p['topology'] == topology])}",
                        "task_gate_passed": row in passing,
                        "ca_selection_score": row.get("_ca_score", ""),
                    }
                )
    _write_csv(output / "rp_parents.csv", parents)

    conditions = [
        {
            "condition": "control_no_rp",
            "rp_active": False,
            "eta": 0.0,
            "rule": "signed",
        },
        {
            "condition": "eta0p1_signed",
            "rp_active": True,
            "eta": 0.1,
            "rule": "signed",
        },
        {
            "condition": "eta0p1_positive",
            "rp_active": True,
            "eta": 0.1,
            "rule": "positive_only",
        },
        {
            "condition": "eta1_signed",
            "rp_active": True,
            "eta": 1.0,
            "rule": "signed",
        },
        {
            "condition": "eta1_positive",
            "rp_active": True,
            "eta": 1.0,
            "rule": "positive_only",
        },
    ]
    cells = [
        (parent, condition) for parent in parents for condition in conditions
    ]

    def run(index: int, parent: dict, condition: dict) -> dict:
        cell_id = f"{parent['parent_label']}_{condition['condition']}"
        job_id = (
            f"rp__split_rnn_rp__{parent['topology']}__{cell_id}__seed10"
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
            "rp",
            "--cell-id",
            cell_id,
            "--model",
            "split_rnn_rp",
            "--topology",
            parent["topology"],
            "--seed",
            "10",
            "--width",
            parent["width"],
            "--updates",
            str(args.updates),
            "--learning-rate",
            parent["learning_rate"],
            "--initial-retention",
            parent["initial_retention"],
            "--initial-write-gain",
            parent["initial_write_gain"],
            "--recurrent-gain",
            parent["recurrent_gain"],
            "--report-interval",
            "100",
            "--load-checkpoint",
            parent["checkpoint"],
            "--load-optimizer",
            "--data-update-offset",
            str(args.data_update_offset),
            "--train-cache",
            str(train_cache),
            "--device",
            device,
            "--output",
            str(output),
        ]
        if condition["rp_active"]:
            command.extend(
                [
                    "--rp-warmup",
                    "0",
                    "--rp-interval",
                    "100",
                    "--rp-probe-batch-size",
                    "16",
                    "--rp-blank-horizon",
                    "512",
                    "--rp-lambda-fast",
                    "0.8",
                    "--rp-eta-lambda",
                    str(condition["eta"]),
                    "--rp-update-rule",
                    condition["rule"],
                    "--rp-max-theta-step",
                    "0.02",
                ]
            )
        else:
            command.append("--disable-rp")
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
            executor.submit(run, index, parent, condition): (parent, condition)
            for index, (parent, condition) in enumerate(cells)
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"[{len(records):02d}/{len(cells)}] {record['job_id']} "
                f"{'ok' if record['returncode'] == 0 else 'FAILED'}",
                flush=True,
            )
            if record["returncode"] != 0:
                print(record["stderr"], flush=True)
    failures = [record for record in records if record["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} candidate RP jobs failed")

    result_rows = []
    parent_lookup = {row["parent_label"]: row for row in parents}
    for result_path in sorted(output.glob("rp__*/result.json")):
        result = json.loads(result_path.read_text())
        manifest = json.loads((result_path.parent / "manifest.json").read_text())
        cell_id = result["job_id"].split("__")[3]
        parent_label = "_".join(cell_id.split("_")[:2])
        parent = parent_lookup[parent_label]
        result_rows.append(
            {
                "job_id": result["job_id"],
                "parent_label": parent_label,
                "topology": result["topology"],
                "condition": cell_id[len(parent_label) + 1 :],
                "rp_active": result["rp_active"],
                "rp_calls": result["rp_calls"],
                "id_intrinsic_rad": result["final_validation"][
                    "intrinsic_mean_radians"
                ],
                "blank128_rad": result["blank_validation"]["128"][
                    "intrinsic_mean_radians"
                ],
                "blank512_rad": result["blank_validation"]["512"][
                    "intrinsic_mean_radians"
                ],
                "blank2048_rad": result["blank_validation"]["2048"][
                    "intrinsic_mean_radians"
                ],
                "lambda_min": result["final_validation"]["lambda_minimum"],
                "lambda_mean": result["final_validation"]["lambda_mean"],
                "lambda_max": result["final_validation"]["lambda_maximum"],
                "checkpoint": str(result_path.parent / "checkpoint.pt"),
                "parent_checkpoint": parent["checkpoint"],
                "width": parent["width"],
                "learning_rate": parent["learning_rate"],
                "initial_retention": parent["initial_retention"],
                "initial_write_gain": parent["initial_write_gain"],
                "recurrent_gain": parent["recurrent_gain"],
                "rp_manifest": json.dumps(
                    manifest["gate_intervention_rp"], sort_keys=True
                ),
            }
        )
    if len(result_rows) != len(cells):
        raise ValueError(
            f"expected {len(cells)} RP/control rows, found {len(result_rows)}"
        )
    _write_csv(output / "rp_results.csv", result_rows)
    (output / "pipeline_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    print(output / "rp_results.csv")


if __name__ == "__main__":
    main()
