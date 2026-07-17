"""Launch the RP-disabled shared-vs-split field pretraining screen."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from repro.sagodi_protocol.artifacts import strict_json_load

from .cache import build_training_cache


CONFIG_PATH = Path(__file__).with_name("split_field_sweep_v1.json")


def _tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _run_cell(
    *,
    output: Path,
    train_cache: Path,
    model: str,
    topology: str,
    seed: int,
    width: int,
    updates: int,
    learning_rate: float,
    initial_retention: float,
    initial_write_gain: float,
    recurrent_gain: float,
    device: str,
) -> dict[str, Any]:
    cell_id = f"lr{_tag(learning_rate)}_lam{_tag(initial_retention)}"
    job_id = f"pretrain__{model}__{topology}__{cell_id}__seed{seed}"
    run_dir = output / job_id
    if (run_dir / "COMPLETED.json").is_file():
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
        "pretrain",
        "--cell-id",
        cell_id,
        "--model",
        model,
        "--topology",
        topology,
        "--seed",
        str(seed),
        "--width",
        str(width),
        "--updates",
        str(updates),
        "--learning-rate",
        str(learning_rate),
        "--initial-retention",
        str(initial_retention),
        "--initial-write-gain",
        str(initial_write_gain),
        "--recurrent-gain",
        str(recurrent_gain),
        "--report-interval",
        "100",
        "--disable-rp",
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
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
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
            "static_gate_split_pretrain_screen_v1"
        ),
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--devices")
    args = parser.parse_args()
    config = strict_json_load(args.config.expanduser().resolve(strict=True))
    if config.get("campaign_id") != "static_gate_split_field_sweep_v1":
        raise ValueError("unexpected split-field campaign")
    screen = config["screen"]
    expected = (
        len(config["models"])
        * len(config["topologies"])
        * len(screen["learning_rates"])
        * len(screen["initial_retentions"])
    )
    if expected != int(screen["run_count"]):
        raise ValueError("configured run count differs from Cartesian product")
    workers = (
        int(config["execution"]["workers"])
        if args.workers is None
        else int(args.workers)
    )
    devices = (
        [str(value) for value in config["execution"]["devices"]]
        if args.devices is None
        else [value.strip() for value in args.devices.split(",") if value.strip()]
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_cache = build_training_cache(
        args.train_cache,
        seed=int(config["seed"]),
        trajectories=int(config["training"]["train_pool_trajectories"]),
        horizon=int(config["training"]["horizon"]),
    )
    cells = [
        (model, topology, float(lr), float(retention))
        for model in config["models"]
        for topology in config["topologies"]
        for lr in screen["learning_rates"]
        for retention in screen["initial_retentions"]
    ]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for index, (model, topology, lr, retention) in enumerate(cells):
            future = executor.submit(
                _run_cell,
                output=output,
                train_cache=train_cache,
                model=model,
                topology=topology,
                seed=int(config["seed"]),
                width=int(config["width"]),
                updates=int(screen["updates"]),
                learning_rate=lr,
                initial_retention=retention,
                initial_write_gain=float(screen["initial_write_gain"]),
                recurrent_gain=float(screen["recurrent_gain"]),
                device=devices[index % len(devices)],
            )
            futures[future] = (model, topology, lr, retention)
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
                "campaign_id": config["campaign_id"],
                "config_path": str(args.config.expanduser().resolve()),
                "train_cache": str(train_cache),
                "workers": workers,
                "devices": devices,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    failures = [record for record in records if record["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} split-field screen jobs failed")


if __name__ == "__main__":
    main()
