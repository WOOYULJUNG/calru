"""Launch the split-field-only task and retention hyperparameter screen."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import subprocess
import sys

from repro.sagodi_protocol.artifacts import strict_json_load

from .cache import build_training_cache


CONFIG_PATH = Path(__file__).with_name("split_field_broad_v2.json")


def _tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


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
            "static_gate_split_broad_v2"
        ),
    )
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    config = strict_json_load(args.config.expanduser().resolve(strict=True))
    if config["campaign_id"] != "static_gate_split_field_broad_v2":
        raise ValueError("unexpected campaign")
    screen = config["screen"]
    cells = list(
        itertools.product(
            config["topologies"],
            screen["widths"],
            screen["learning_rates"],
            screen["initial_retentions"],
            screen["initial_write_gains"],
            screen["recurrent_gains"],
        )
    )
    if len(cells) != int(screen["run_count"]):
        raise ValueError(
            f"configured run count {screen['run_count']} != {len(cells)}"
        )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_cache = build_training_cache(
        args.train_cache,
        seed=int(config["seed"]),
        trajectories=int(config["training"]["train_pool_trajectories"]),
        horizon=int(config["training"]["horizon"]),
    )
    devices = [str(value) for value in config["execution"]["devices"]]
    workers = (
        int(config["execution"]["workers"])
        if args.workers is None
        else int(args.workers)
    )

    def run(index: int, cell: tuple) -> dict:
        topology, width, learning_rate, retention, gamma, recurrent_gain = cell
        cell_id = (
            f"w{width}_lr{_tag(learning_rate)}_lam{_tag(retention)}"
            f"_gam{_tag(gamma)}_rg{_tag(recurrent_gain)}"
        )
        job_id = (
            f"pretrain__{config['model']}__{topology}__{cell_id}"
            f"__seed{config['seed']}"
        )
        run_dir = output / job_id
        device = devices[index % len(devices)]
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
            str(config["model"]),
            "--topology",
            str(topology),
            "--seed",
            str(config["seed"]),
            "--width",
            str(width),
            "--updates",
            str(screen["updates"]),
            "--learning-rate",
            str(learning_rate),
            "--initial-retention",
            str(retention),
            "--initial-write-gain",
            str(gamma),
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

    records = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run, index, cell): cell
            for index, cell in enumerate(cells)
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"[{len(records):03d}/{len(cells)}] {record['job_id']} "
                f"{'ok' if record['returncode'] == 0 else 'FAILED'} "
                f"{record['device']}",
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
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    failures = [record for record in records if record["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} broad split-field jobs failed")


if __name__ == "__main__":
    main()
