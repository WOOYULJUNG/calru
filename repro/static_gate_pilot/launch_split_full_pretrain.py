"""Continue selected RP-disabled checkpoints from 1,500 to 5,000 updates."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _run(
    *,
    row: dict[str, str],
    output: Path,
    train_cache: Path,
    device: str,
    additional_updates: int,
    update_offset: int,
) -> dict[str, Any]:
    model = row["model"]
    topology = row["topology"]
    seed = int(row["seed"])
    lr = float(row["learning_rate"])
    retention = float(row["initial_retention"])
    write_gain = float(row["initial_write_gain"])
    recurrent_gain = float(row["recurrent_gain"])
    source_cell = f"lr{_tag(lr)}_lam{_tag(retention)}"
    cell_id = f"full_{source_cell}_rank{row['rank']}"
    job_id = f"pretrain__{model}__{topology}__{cell_id}__seed{seed}"
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
        model,
        "--topology",
        topology,
        "--seed",
        str(seed),
        "--updates",
        str(additional_updates),
        "--learning-rate",
        str(lr),
        "--initial-retention",
        str(retention),
        "--initial-write-gain",
        str(write_gain),
        "--recurrent-gain",
        str(recurrent_gain),
        "--report-interval",
        "100",
        "--disable-rp",
        "--load-checkpoint",
        row["checkpoint"],
        "--load-optimizer",
        "--data-update-offset",
        str(update_offset),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finalists",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_split_pretrain_screen_v1/"
            "finalists.csv"
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
            "/home/biadmin/ca_rnn/experiments/static_gate_split_pretrain_full_v1"
        ),
    )
    parser.add_argument("--screen-updates", type=int, default=1500)
    parser.add_argument("--total-updates", type=int, default=5000)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.finalists.expanduser().resolve(strict=True).open()))
    if len(rows) != 8:
        raise ValueError(f"expected 8 finalists, found {len(rows)}")
    additional = int(args.total_updates) - int(args.screen_updates)
    if additional <= 0:
        raise ValueError("total updates must exceed screen updates")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_cache = args.train_cache.expanduser().resolve(strict=True)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for index, row in enumerate(rows):
            future = executor.submit(
                _run,
                row=row,
                output=output,
                train_cache=train_cache,
                device=devices[index % len(devices)],
                additional_updates=additional,
                update_offset=int(args.screen_updates),
            )
            futures[future] = row
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            status = "ok" if record["returncode"] == 0 else "FAILED"
            print(
                f"[{len(records)}/{len(rows)}] {record['job_id']} "
                f"{status} {record['device']}",
                flush=True,
            )
            if record["returncode"] != 0:
                print(record["stderr"], flush=True)
    (output / "launcher_records.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "static_gate_split_full_pretrain_v1",
                "screen_updates": int(args.screen_updates),
                "additional_updates": additional,
                "total_updates": int(args.total_updates),
                "train_cache": str(train_cache),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    failures = [record for record in records if record["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} full-pretrain jobs failed")


if __name__ == "__main__":
    main()
