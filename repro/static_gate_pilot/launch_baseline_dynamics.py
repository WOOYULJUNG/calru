"""Run the same local CA diagnostics on RNN/GRU/LSTM topology baselines."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "manifold_topology_transfer_v1-pilot-1f4a80d"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_baseline_dynamics_v1"
        ),
    )
    parser.add_argument("--models", default="rnn,gru,lstm")
    parser.add_argument("--topologies", default="s1,t2,s2")
    parser.add_argument("--seeds", default="10,11,12")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    root = args.checkpoints.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    topologies = [
        value.strip() for value in args.topologies.split(",") if value.strip()
    ]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    cells = [
        (model, topology, seed)
        for model in models
        for topology in topologies
        for seed in seeds
    ]

    def run(
        model: str,
        topology: str,
        seed: int,
        device: str,
    ) -> dict:
        checkpoint = (
            root / f"pilot__{model}__{topology}__seed{seed}" / "checkpoint.pt"
        )
        target = output / f"baseline__{model}__{topology}__seed{seed}"
        if (target / "dynamics.json").is_file():
            return {
                "model": model,
                "topology": topology,
                "seed": seed,
                "returncode": 0,
                "skipped_completed": True,
            }
        command = [
            sys.executable,
            "-m",
            "repro.static_gate_pilot.analyze_checkpoint_dynamics",
            "--checkpoint",
            str(checkpoint.resolve(strict=True)),
            "--output",
            str(target),
            "--device",
            device,
            "--trajectories",
            "128",
            "--anchors",
            "8",
            "--neighbors",
            "12",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        return {
            "model": model,
            "topology": topology,
            "seed": seed,
            "device": device,
            "returncode": int(completed.returncode),
            "skipped_completed": False,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-8000:],
        }

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run,
                model,
                topology,
                seed,
                devices[index % len(devices)],
            ): (model, topology, seed)
            for index, (model, topology, seed) in enumerate(cells)
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"[{len(records):02d}/{len(cells)}] {record['model']}/"
                f"{record['topology']}/seed{record['seed']} "
                f"{'ok' if record['returncode'] == 0 else 'FAILED'}",
                flush=True,
            )
            if record["returncode"] != 0:
                print(record["stderr"], flush=True)
    failures = [record for record in records if record["returncode"] != 0]
    (output / "launcher_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    if failures:
        raise SystemExit(f"{len(failures)} baseline dynamics jobs failed")

    rows = []
    for path in sorted(output.glob("*/dynamics.json")):
        data = json.loads(path.read_text())
        rows.append(
            {
                "model": data["model"],
                "topology": data["topology"],
                "seed": data["seed"],
                "task_intrinsic_radians": data["task_intrinsic_radians"],
                "blank2048_memory_radians": data["blank_manifold_evolution"][
                    "2048"
                ]["decoded_memory_intrinsic_radians"],
                "tangent_singular_mean": data["tangent_singular_mean"],
                "normal_max_singular_mean": data["normal_max_singular_mean"],
                "tangent_normal_gap_mean": data["tangent_normal_gap_mean"],
                "normal_recovery_ratio_h512": data[
                    "finite_local_normal_recovery"
                ]["512"]["distance_ratio_median"],
                "shape_distortion_h2048": data["blank_manifold_evolution"][
                    "2048"
                ]["pairwise_shape_distortion_std"],
            }
        )
    with (output / "baseline_dynamics_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output / "baseline_dynamics_summary.csv")


if __name__ == "__main__":
    main()
