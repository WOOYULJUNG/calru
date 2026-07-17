"""Run local tangent/normal analysis over selected no-RP and RP checkpoints."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _records(selection: Path, selected_root: Path) -> list[dict[str, Any]]:
    chosen = list(csv.DictReader(selection.open()))
    records: list[dict[str, Any]] = []
    for row in chosen:
        records.append(
            {
                "condition": "no_rp",
                "model": row["model"],
                "topology": row["topology"],
                "seed": 10,
                "checkpoint": Path(row["control_checkpoint"]),
            }
        )
        records.append(
            {
                "condition": "rp",
                "model": row["model"],
                "topology": row["topology"],
                "seed": 10,
                "checkpoint": Path(row["checkpoint"]),
            }
        )
    for condition, subdir in (("no_rp", "control"), ("rp", "rp")):
        for checkpoint in sorted((selected_root / subdir).glob("*/checkpoint.pt")):
            payload = json.loads((checkpoint.parent / "manifest.json").read_text())
            records.append(
                {
                    "condition": condition,
                    "model": payload["model"]["model_id"],
                    "topology": payload["topology"],
                    "seed": int(payload["replicate_seed"]),
                    "checkpoint": checkpoint,
                }
            )
    return records


def _run(record: dict[str, Any], *, output: Path, device: str) -> dict[str, Any]:
    target = (
        output
        / f"{record['condition']}__{record['model']}__{record['topology']}"
        f"__seed{record['seed']}"
    )
    if (target / "dynamics.json").is_file():
        return {
            **record,
            "device": device,
            "returncode": 0,
            "skipped_completed": True,
        }
    command = [
        sys.executable,
        "-m",
        "repro.static_gate_pilot.analyze_checkpoint_dynamics",
        "--checkpoint",
        str(record["checkpoint"]),
        "--output",
        str(target),
        "--device",
        device,
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    return {
        **record,
        "device": device,
        "returncode": int(completed.returncode),
        "skipped_completed": False,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-8000:],
    }


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
        "--selected-root",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_selected_seeds_v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_selected_dynamics_v1"
        ),
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    records = _records(
        args.selection.expanduser().resolve(strict=True),
        args.selected_root.expanduser().resolve(strict=True),
    )
    if len(records) != 24:
        raise ValueError(f"expected 24 selected checkpoints, found {len(records)}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    completed_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run,
                record,
                output=output,
                device=devices[index % len(devices)],
            ): record
            for index, record in enumerate(records)
        }
        for future in as_completed(futures):
            record = future.result()
            completed_records.append(record)
            status = "ok" if record["returncode"] == 0 else "FAILED"
            print(
                f"[{len(completed_records):02d}/{len(records)}] "
                f"{record['condition']}/{record['model']}/{record['topology']}/"
                f"seed{record['seed']} {status}",
                flush=True,
            )
            if record["returncode"] != 0:
                print(record["stderr"], flush=True)
    failures = [
        record for record in completed_records if record["returncode"] != 0
    ]
    if failures:
        raise SystemExit(f"{len(failures)} dynamics analyses failed")

    rows = []
    for path in sorted(output.glob("*/dynamics.json")):
        data = json.loads(path.read_text())
        condition = path.parent.name.split("__", 1)[0]
        rows.append(
            {
                "condition": condition,
                **data,
                "normal_recovery_ratio_h128": data[
                    "finite_local_normal_recovery"
                ]["128"]["distance_ratio_median"],
                "normal_recovery_ratio_h512": data[
                    "finite_local_normal_recovery"
                ]["512"]["distance_ratio_median"],
                "normal_same_memory_h512": data[
                    "finite_local_normal_recovery"
                ]["512"]["same_memory_intrinsic_radians"],
                "tangent_memory_shift_h512": data[
                    "finite_local_tangent_transport"
                ]["512"]["memory_shift_intrinsic_radians"],
                "blank2048_memory_radians": data["blank_manifold_evolution"][
                    "2048"
                ]["decoded_memory_intrinsic_radians"],
                "blank2048_global_scale": data["blank_manifold_evolution"][
                    "2048"
                ]["best_global_scale"],
                "blank2048_scaling_residual": data[
                    "blank_manifold_evolution"
                ]["2048"]["global_scaling_residual"],
                "blank2048_shape_distortion": data[
                    "blank_manifold_evolution"
                ]["2048"]["pairwise_shape_distortion_std"],
            }
        )
    with (output / "dynamics_summary.csv").open("w", newline="") as handle:
        fields = [
            "condition",
            "model",
            "topology",
            "seed",
            "task_intrinsic_radians",
            "endpoint_participation_ratio",
            "normalized_fixedness_median",
            "tangent_singular_mean",
            "normal_max_singular_mean",
            "normal_max_singular_worst",
            "tangent_normal_gap_mean",
            "tangent_to_normal_coupling_mean",
            "blank_field_tangent_fraction_mean",
            "lambda_min",
            "lambda_mean",
            "lambda_max",
            "lambda_std",
            "normal_recovery_ratio_h128",
            "normal_recovery_ratio_h512",
            "normal_same_memory_h512",
            "tangent_memory_shift_h512",
            "blank2048_memory_radians",
            "blank2048_global_scale",
            "blank2048_scaling_residual",
            "blank2048_shape_distortion",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(output / "dynamics_summary.csv")


if __name__ == "__main__":
    main()
