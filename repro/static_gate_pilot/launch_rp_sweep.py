"""Branch calibrated RP cells from mature no-RP parent checkpoints."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ETAS = (0.1, 1.0)
UPDATE_RULES = ("signed", "positive_only")
MAX_THETA_STEPS = (0.02, 0.1)


def _tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _run(
    *,
    parent: dict[str, str],
    eta: float,
    update_rule: str,
    max_theta_step: float,
    output: Path,
    train_cache: Path,
    device: str,
    updates: int,
    update_offset: int,
) -> dict[str, Any]:
    model = parent["model"]
    topology = parent["topology"]
    seed = int(parent["seed"])
    rule_tag = "pos" if update_rule == "positive_only" else "signed"
    cell_id = (
        f"eta{_tag(eta)}_{rule_tag}_cap{_tag(max_theta_step)}"
        f"_parent_{parent['job_id'].split('__')[3]}"
    )
    job_id = f"rp__{model}__{topology}__{cell_id}__seed{seed}"
    if (output / job_id / "COMPLETED.json").is_file():
        return {
            "job_id": job_id,
            "device": device,
            "returncode": 0,
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
        model,
        "--topology",
        topology,
        "--seed",
        str(seed),
        "--updates",
        str(updates),
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
        str(update_offset),
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
        str(eta),
        "--rp-update-rule",
        update_rule,
        "--rp-max-theta-step",
        str(max_theta_step),
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
        "device": device,
        "returncode": int(completed.returncode),
        "skipped_completed": False,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-8000:],
    }


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
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--data-update-offset", type=int, default=5000)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    parents = list(csv.DictReader(args.parents.expanduser().resolve(strict=True).open()))
    if len(parents) != 4:
        raise ValueError(f"expected 4 parent checkpoints, found {len(parents)}")
    cells = [
        (parent, eta, rule, cap)
        for parent in parents
        for eta in ETAS
        for rule in UPDATE_RULES
        for cap in MAX_THETA_STEPS
    ]
    if len(cells) != 32:
        raise RuntimeError("RP sweep Cartesian product changed")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_cache = args.train_cache.expanduser().resolve(strict=True)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for index, (parent, eta, rule, cap) in enumerate(cells):
            future = executor.submit(
                _run,
                parent=parent,
                eta=eta,
                update_rule=rule,
                max_theta_step=cap,
                output=output,
                train_cache=train_cache,
                device=devices[index % len(devices)],
                updates=int(args.updates),
                update_offset=int(args.data_update_offset),
            )
            futures[future] = (parent, eta, rule, cap)
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            status = "ok" if record["returncode"] == 0 else "FAILED"
            print(
                f"[{len(records):02d}/{len(cells)}] {record['job_id']} "
                f"{status} {record['device']}",
                flush=True,
            )
            if record["returncode"] != 0:
                print(record["stderr"], flush=True)
    (output / "launcher_records.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "static_gate_rp_sweep_v1",
                "updates": int(args.updates),
                "data_update_offset": int(args.data_update_offset),
                "rp_blank_horizon": 512,
                "rp_lambda_fast": 0.8,
                "rp_interval": 100,
                "rp_calls": int(args.updates) // 100,
                "cells": 32,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    failures = [record for record in records if record["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} RP sweep jobs failed")


if __name__ == "__main__":
    main()
