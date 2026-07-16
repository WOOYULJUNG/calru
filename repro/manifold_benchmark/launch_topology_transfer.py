"""Launch the complete paired topology grid with one restartable queue per GPU."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from repro.sagodi_protocol.artifacts import atomic_json

from .run_topology_transfer import STAGES
from .topology_models import MODEL_IDS, load_transfer_config
from .topology_training import TOPOLOGIES


def _jobs(stage: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    if stage == "fixed_overfit":
        seeds = [int(config["debug"]["online_smoke_seed"])]
    elif stage == "online_smoke":
        seeds = [int(config["debug"]["online_smoke_seed"])]
    else:
        seeds = [int(value) for value in config["pilot"]["seeds"]]
    # Interleave models so each device receives a mix of computational costs.
    return [
        {"stage": stage, "model": model, "topology": topology, "seed": seed}
        for seed in seeds
        for topology in TOPOLOGIES
        for model in MODEL_IDS
    ]


def _job_id(job: dict[str, Any]) -> str:
    return (
        f"{job['stage']}__{job['model']}__{job['topology']}__seed{job['seed']}"
    )


def _estimated_cost(job: dict[str, Any]) -> float:
    """Scheduling-only cost; it never changes a scientific job setting."""

    return {"rnn": 1.0, "gru": 1.4, "lstm": 1.4, "hc": 3.0}[str(job["model"])]


def _balanced_queues(
    jobs: list[dict[str, Any]], worker_count: int
) -> tuple[list[list[dict[str, Any]]], list[float]]:
    queues: list[list[dict[str, Any]]] = [[] for _ in range(int(worker_count))]
    loads = [0.0 for _ in range(int(worker_count))]
    ordered = sorted(
        enumerate(jobs), key=lambda item: (-_estimated_cost(item[1]), item[0])
    )
    for _, job in ordered:
        worker = min(range(int(worker_count)), key=lambda index: (loads[index], index))
        queues[worker].append(job)
        loads[worker] += _estimated_cost(job)
    return queues, loads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("topology_transfer_v1.json"),
    )
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--validation-trajectories", type=int, default=64)
    args = parser.parse_args()
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("at least one CUDA device is required")
    config = load_transfer_config(args.config)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = _jobs(args.stage, config)
    queues, estimated_loads = _balanced_queues(jobs, len(devices))
    atomic_json(
        output / "launcher_manifest.json",
        {
            "schema_version": 1,
            "stage": args.stage,
            "devices": devices,
            "jobs": jobs,
            "queue_assignment": {
                device: [_job_id(job) for job in queue]
                for device, queue in zip(devices, queues)
            },
            "queue_estimated_relative_load": {
                device: load for device, load in zip(devices, estimated_loads)
            },
            "restart_policy": "skip_only_jobs_with_COMPLETED_json",
        },
    )
    log_root = output / "launcher_logs"
    log_root.mkdir(exist_ok=True)

    def worker(device: str, queue: list[dict[str, Any]]) -> None:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = device
        environment["PYTHONPATH"] = os.getcwd()
        for job in queue:
            job_id = _job_id(job)
            if (output / job_id / "COMPLETED.json").is_file():
                continue
            command = [
                sys.executable,
                "-m",
                "repro.manifold_benchmark.run_topology_transfer",
                "--stage",
                str(job["stage"]),
                "--model",
                str(job["model"]),
                "--topology",
                str(job["topology"]),
                "--seed",
                str(job["seed"]),
                "--device",
                "cuda:0",
                "--config",
                str(args.config),
                "--output",
                str(output),
                "--report-interval",
                str(args.report_interval),
                "--validation-trajectories",
                str(args.validation_trajectories),
            ]
            with (log_root / f"{job_id}.log").open("a", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=os.getcwd(),
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{job_id} failed on physical GPU {device}; see launcher log"
                )

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(worker, device, queue)
            for device, queue in zip(devices, queues)
            if queue
        ]
        for future in futures:
            future.result()
    atomic_json(
        output / "LAUNCHER_COMPLETED.json",
        {"schema_version": 1, "stage": args.stage, "completed_jobs": len(jobs)},
    )


if __name__ == "__main__":
    main()
