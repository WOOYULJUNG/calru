"""Launch no-update gate-damage diagnostics over full-pretrain checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _run(
    checkpoint: Path,
    *,
    output: Path,
    train_cache: Path,
    device: str,
) -> dict[str, Any]:
    target = output / checkpoint.parent.name
    if (target / "summary.json").is_file():
        return {
            "job_id": checkpoint.parent.name,
            "device": device,
            "returncode": 0,
            "skipped_completed": True,
        }
    command = [
        sys.executable,
        "-m",
        "repro.static_gate_pilot.diagnose_gate_damage",
        "--checkpoint",
        str(checkpoint),
        "--train-cache",
        str(train_cache),
        "--output",
        str(target),
        "--device",
        device,
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    return {
        "job_id": checkpoint.parent.name,
        "device": device,
        "returncode": int(completed.returncode),
        "skipped_completed": False,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-8000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_split_pretrain_full_v1"
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
            "/home/biadmin/ca_rnn/experiments/static_gate_gate_damage_v1"
        ),
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    args = parser.parse_args()
    checkpoints = sorted(
        args.checkpoints_root.expanduser().resolve(strict=True).glob(
            "pretrain__*/checkpoint.pt"
        )
    )
    if len(checkpoints) != 8:
        raise ValueError(f"expected 8 full-pretrain checkpoints, found {len(checkpoints)}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_cache = args.train_cache.expanduser().resolve(strict=True)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(checkpoints)) as executor:
        futures = {
            executor.submit(
                _run,
                checkpoint,
                output=output,
                train_cache=train_cache,
                device=devices[index % len(devices)],
            ): checkpoint
            for index, checkpoint in enumerate(checkpoints)
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            status = "ok" if record["returncode"] == 0 else "FAILED"
            print(
                f"[{len(records)}/{len(checkpoints)}] {record['job_id']} "
                f"{status} {record['device']}",
                flush=True,
            )
            if record["returncode"] != 0:
                print(record["stderr"], flush=True)
    (output / "launcher_records.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "diagnostic": "gate_intervention_damage_no_parameter_updates",
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    failures = [record for record in records if record["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} gate-damage diagnostics failed")


if __name__ == "__main__":
    main()
