"""Continue mature parent checkpoints for 2,000 updates with RP disabled."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parents",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_split_pretrain_full_v1/"
            "rp_parent_checkpoints.csv"
        ),
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_side_pilot_v1/"
            "train_pool_seed10"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_rp_sweep_v1"
        ),
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    args = parser.parse_args()
    parents = list(csv.DictReader(args.parents.expanduser().resolve(strict=True).open()))
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_cache = args.train_cache.expanduser().resolve(strict=True)

    def run(row: dict[str, str], device: str) -> tuple[str, int, str]:
        job_id = (
            f"rp__{row['model']}__{row['topology']}__control_no_rp__seed{row['seed']}"
        )
        if (output / job_id / "COMPLETED.json").is_file():
            return job_id, 0, ""
        command = [
            sys.executable,
            "-m",
            "repro.static_gate_pilot.run",
            "--phase",
            "rp",
            "--cell-id",
            "control_no_rp",
            "--model",
            row["model"],
            "--topology",
            row["topology"],
            "--seed",
            row["seed"],
            "--updates",
            "2000",
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
            "5000",
            "--train-cache",
            str(train_cache),
            "--device",
            device,
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        return job_id, int(completed.returncode), completed.stderr[-8000:]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run, row, devices[index % len(devices)]): row
            for index, row in enumerate(parents)
        }
        failures = 0
        for index, future in enumerate(as_completed(futures), 1):
            job_id, returncode, stderr = future.result()
            print(
                f"[{index}/{len(parents)}] {job_id} "
                f"{'ok' if returncode == 0 else 'FAILED'}",
                flush=True,
            )
            if returncode != 0:
                failures += 1
                print(stderr, flush=True)
    if failures:
        raise SystemExit(f"{failures} no-RP continuation controls failed")


if __name__ == "__main__":
    main()
