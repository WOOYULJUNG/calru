"""Replicate the final split-field setting for S1, T2, and S2 on seeds 11/12."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

from .cache import build_training_cache


def _run_stage(cells: list[tuple[list[str], str]], workers: int) -> list[dict]:
    def execute(command: list[str], job_id: str) -> dict:
        completed = subprocess.run(command, text=True, capture_output=True)
        return {
            "job_id": job_id,
            "returncode": int(completed.returncode),
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-8000:],
        }

    records = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(execute, command, job_id): job_id
            for command, job_id in cells
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
        raise RuntimeError(f"{len(failures)} final-seed jobs failed")
    return records


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
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_final_seeds_v2"
        ),
    )
    parser.add_argument("--seeds", default="11,12")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--pretrain-updates", type=int, default=5000)
    parser.add_argument("--final-updates", type=int, default=2000)
    args = parser.parse_args()
    selected = list(
        csv.DictReader(args.selection.expanduser().resolve(strict=True).open())
    )
    if len(selected) != 3 or {row["topology"] for row in selected} != {
        "s1",
        "t2",
        "s2",
    }:
        raise ValueError("final selection must contain exactly S1, T2, and S2")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    output = args.output.expanduser().resolve()
    pretrain_output = output / "pretrain"
    final_output = output / "final"
    pretrain_output.mkdir(parents=True, exist_ok=True)
    final_output.mkdir(parents=True, exist_ok=True)
    caches = {
        seed: build_training_cache(
            output / "data" / f"train_pool_seed{seed}",
            seed=seed,
            trajectories=4096,
            horizon=128,
        )
        for seed in seeds
    }
    pairs = [(row, seed) for row in selected for seed in seeds]
    pretrain_paths = {}
    pretrain_cells = []
    for index, (row, seed) in enumerate(pairs):
        cell_id = f"final_parent_{args.pretrain_updates}"
        job_id = f"pretrain__split_rnn_rp__{row['topology']}__{cell_id}__seed{seed}"
        checkpoint = pretrain_output / job_id / "checkpoint.pt"
        pretrain_paths[(row["topology"], seed)] = checkpoint
        if (checkpoint.parent / "COMPLETED.json").is_file():
            continue
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
            str(seed),
            "--width",
            row["width"],
            "--updates",
            str(args.pretrain_updates),
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
            "--train-cache",
            str(caches[seed]),
            "--device",
            devices[index % len(devices)],
            "--output",
            str(pretrain_output),
        ]
        pretrain_cells.append((command, job_id))
    pretrain_records = _run_stage(
        pretrain_cells, min(args.workers, max(1, len(pretrain_cells)))
    )

    final_cells = []
    for index, (row, seed) in enumerate(pairs):
        cell_id = f"final_{row['condition']}"
        job_id = f"rp__split_rnn_rp__{row['topology']}__{cell_id}__seed{seed}"
        if (final_output / job_id / "COMPLETED.json").is_file():
            continue
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
            row["topology"],
            "--seed",
            str(seed),
            "--width",
            row["width"],
            "--updates",
            str(args.final_updates),
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
            "--load-checkpoint",
            str(pretrain_paths[(row["topology"], seed)]),
            "--load-optimizer",
            "--data-update-offset",
            str(args.pretrain_updates),
            "--train-cache",
            str(caches[seed]),
            "--device",
            devices[index % len(devices)],
            "--output",
            str(final_output),
        ]
        if str(row["rp_active"]).lower() == "true":
            rp = json.loads(row["rp_manifest"])
            command.extend(
                [
                    "--rp-warmup",
                    "0",
                    "--rp-interval",
                    str(rp["interval_updates"]),
                    "--rp-probe-batch-size",
                    str(rp["probe_batch_size"]),
                    "--rp-blank-horizon",
                    str(rp["blank_horizon"]),
                    "--rp-lambda-fast",
                    str(rp["lambda_fast"]),
                    "--rp-eta-lambda",
                    str(rp["eta_lambda"]),
                    "--rp-update-rule",
                    str(rp["update_rule"]),
                    "--rp-max-theta-step",
                    str(rp["max_theta_step"]),
                ]
            )
        else:
            command.append("--disable-rp")
        final_cells.append((command, job_id))
    final_records = _run_stage(
        final_cells, min(args.workers, max(1, len(final_cells)))
    )
    (output / "pipeline_records.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection": str(args.selection.expanduser().resolve()),
                "seeds": seeds,
                "pretrain_records": pretrain_records,
                "final_records": final_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(output / "pipeline_records.json")


if __name__ == "__main__":
    main()
