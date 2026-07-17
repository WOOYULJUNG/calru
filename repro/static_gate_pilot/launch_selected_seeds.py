"""Replicate selected split/shared and RP settings on seeds 11 and 12."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .cache import build_training_cache


def _tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _execute(command: list[str], job_id: str, device: str) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True)
    return {
        "job_id": job_id,
        "device": device,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-8000:],
    }


def _run_stage(
    cells: list[tuple[list[str], str, str]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_execute, command, job_id, device): job_id
            for command, job_id, device in cells
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            status = "ok" if record["returncode"] == 0 else "FAILED"
            print(
                f"[{len(records)}/{len(cells)}] {record['job_id']} "
                f"{status} {record['device']}",
                flush=True,
            )
            if record["returncode"] != 0:
                print(record["stderr"], flush=True)
    failures = [record for record in records if record["returncode"] != 0]
    if failures:
        raise RuntimeError(f"{len(failures)} selected-seed jobs failed")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_rp_sweep_v1/"
            "rp_finalists.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_selected_seeds_v1"
        ),
    )
    parser.add_argument("--seeds", default="11,12")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    selected = list(
        csv.DictReader(args.selection.expanduser().resolve(strict=True).open())
    )
    if len(selected) != 4:
        raise ValueError(f"expected 4 selected RP configurations, found {len(selected)}")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output = args.output.expanduser().resolve()
    pretrain_output = output / "pretrain"
    control_output = output / "control"
    rp_output = output / "rp"
    pretrain_output.mkdir(parents=True, exist_ok=True)
    control_output.mkdir(parents=True, exist_ok=True)
    rp_output.mkdir(parents=True, exist_ok=True)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    caches = {
        seed: build_training_cache(
            output / "data" / f"train_pool_seed{seed}",
            seed=seed,
            trajectories=4096,
            horizon=128,
        )
        for seed in seeds
    }

    pretrain_cells: list[tuple[list[str], str, str]] = []
    pretrain_paths: dict[tuple[str, str, int], Path] = {}
    selected_seed_pairs = [
        (row, seed) for row in selected for seed in seeds
    ]
    for index, (row, seed) in enumerate(selected_seed_pairs):
        model, topology = row["model"], row["topology"]
        cell_id = "selected_no_rp"
        job_id = f"pretrain__{model}__{topology}__{cell_id}__seed{seed}"
        checkpoint = pretrain_output / job_id / "checkpoint.pt"
        pretrain_paths[(model, topology, seed)] = checkpoint
        if (checkpoint.parent / "COMPLETED.json").is_file():
            continue
        device = devices[index % len(devices)]
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
            "5000",
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
            device,
            "--output",
            str(pretrain_output),
        ]
        pretrain_cells.append((command, job_id, device))
    pretrain_records = _run_stage(
        pretrain_cells, workers=min(args.workers, max(1, len(pretrain_cells)))
    )

    control_cells: list[tuple[list[str], str, str]] = []
    for index, (row, seed) in enumerate(selected_seed_pairs):
        model, topology = row["model"], row["topology"]
        cell_id = "selected_control_no_rp"
        job_id = f"rp__{model}__{topology}__{cell_id}__seed{seed}"
        if (control_output / job_id / "COMPLETED.json").is_file():
            continue
        device = devices[index % len(devices)]
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
            str(pretrain_paths[(model, topology, seed)]),
            "--load-optimizer",
            "--data-update-offset",
            "5000",
            "--train-cache",
            str(caches[seed]),
            "--device",
            device,
            "--output",
            str(control_output),
        ]
        control_cells.append((command, job_id, device))
    control_records = _run_stage(
        control_cells, workers=min(args.workers, max(1, len(control_cells)))
    )

    rp_cells: list[tuple[list[str], str, str]] = []
    for index, (row, seed) in enumerate(selected_seed_pairs):
        model, topology = row["model"], row["topology"]
        rule_tag = "pos" if row["update_rule"] == "positive_only" else "signed"
        cell_id = (
            f"selected_eta{_tag(float(row['eta']))}_{rule_tag}"
            f"_cap{_tag(float(row['max_theta_step']))}"
        )
        job_id = f"rp__{model}__{topology}__{cell_id}__seed{seed}"
        if (rp_output / job_id / "COMPLETED.json").is_file():
            continue
        device = devices[index % len(devices)]
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
            "--load-checkpoint",
            str(pretrain_paths[(model, topology, seed)]),
            "--load-optimizer",
            "--data-update-offset",
            "5000",
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
            row["eta"],
            "--rp-update-rule",
            row["update_rule"],
            "--rp-max-theta-step",
            row["max_theta_step"],
            "--train-cache",
            str(caches[seed]),
            "--device",
            device,
            "--output",
            str(rp_output),
        ]
        rp_cells.append((command, job_id, device))
    rp_records = _run_stage(
        rp_cells, workers=min(args.workers, max(1, len(rp_cells)))
    )
    (output / "pipeline_records.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "static_gate_selected_seeds_v1",
                "seeds": seeds,
                "selection": str(args.selection.expanduser().resolve()),
                "pretrain_records": pretrain_records,
                "control_records": control_records,
                "rp_records": rp_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(output / "pipeline_records.json")


if __name__ == "__main__":
    main()
