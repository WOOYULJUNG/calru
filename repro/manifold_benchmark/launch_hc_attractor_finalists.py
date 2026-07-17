"""Train missing seeds for CA-aware H-C finalists, reusing exact prior runs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

from repro.sagodi_protocol.artifacts import atomic_json, strict_json_load


def _slug(value: float) -> str:
    return (
        format(float(value), ".6g")
        .replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
    )


def _key(row: dict[str, Any], seed: int) -> tuple[Any, ...]:
    return (
        row["topology"],
        int(seed),
        float(row["learning_rate"]),
        float(row["max_log_modulation"]),
        float(row["gate_output_bias"]),
        float(row["rp_eta_lambda"]),
        int(row["rp_interval"]),
        int(row["rp_warmup"]),
    )


def _manifest_key(manifest: dict[str, Any]) -> tuple[Any, ...]:
    return (
        manifest["topology"],
        int(manifest["replicate_seed"]),
        float(manifest["learning_rate"]),
        float(manifest["max_log_modulation"]),
        float(manifest["gate_output_bias"]),
        float(manifest["rp_eta_lambda"]),
        int(manifest["rp_interval"]),
        int(manifest["rp_warmup"]),
    )


def _prior_runs(source_root: Path) -> dict[tuple[Any, ...], Path]:
    matches: dict[tuple[Any, ...], Path] = {}
    for manifest_path in source_root.glob("*/manifest.json"):
        run_dir = manifest_path.parent
        if not (run_dir / "COMPLETED.json").is_file():
            continue
        manifest = strict_json_load(manifest_path)
        if manifest.get("model", {}).get("model_id") != "hc":
            continue
        key = _manifest_key(manifest)
        # Prefer robust runs over seed-10 screening duplicates.
        priority = {"robust": 3, "refine": 2, "broad": 1}.get(
            str(manifest.get("phase")), 0
        )
        current = matches.get(key)
        if current is None:
            matches[key] = run_dir
            continue
        current_manifest = strict_json_load(current / "manifest.json")
        current_priority = {"robust": 3, "refine": 2, "broad": 1}.get(
            str(current_manifest.get("phase")), 0
        )
        if priority > current_priority:
            matches[key] = run_dir
    return matches


def _new_job(
    topology: str,
    finalist_index: int,
    row: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    cell_id = (
        f"f{finalist_index}_lr{_slug(row['learning_rate'])}"
        f"_a{_slug(row['max_log_modulation'])}"
        f"_b{_slug(row['gate_output_bias'])}"
        f"_eta{_slug(row['rp_eta_lambda'])}_i{int(row['rp_interval'])}"
    )
    return {
        "phase": "attractor_robust",
        "cell_id": cell_id,
        "job_id": f"attractor_robust__hc__{topology}__{cell_id}__seed{int(seed)}",
        "model": "hc",
        "topology": topology,
        "seed": int(seed),
        "learning_rate": float(row["learning_rate"]),
        "max_log_modulation": float(row["max_log_modulation"]),
        "gate_output_bias": float(row["gate_output_bias"]),
        "rp_eta_lambda": float(row["rp_eta_lambda"]),
        "rp_interval": int(row["rp_interval"]),
        "rp_warmup": int(row["rp_warmup"]),
        "updates": 5000,
    }


def launch(args: argparse.Namespace) -> None:
    selection_path = args.selection.expanduser().resolve(strict=True)
    selection = strict_json_load(selection_path)["selected"]
    source_root = args.source_root.expanduser().resolve(strict=True)
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prior = _prior_runs(source_root)

    entries: list[dict[str, Any]] = []
    new_jobs: list[dict[str, Any]] = []
    for topology in ("s1", "t2", "s2"):
        for finalist_index, row in enumerate(selection[topology], 1):
            for seed in (10, 11, 12):
                existing = prior.get(_key({**row, "topology": topology}, seed))
                if existing is not None:
                    destination = output_root / existing.name
                    if not destination.exists():
                        destination.symlink_to(existing, target_is_directory=True)
                    entries.append(
                        {
                            "topology": topology,
                            "finalist_index": finalist_index,
                            "seed": seed,
                            "job_id": existing.name,
                            "source": "reused_prior_run",
                            "run_dir": str(destination),
                            "hyperparameters": {
                                name: row[name]
                                for name in (
                                    "learning_rate",
                                    "max_log_modulation",
                                    "gate_output_bias",
                                    "rp_eta_lambda",
                                    "rp_interval",
                                    "rp_warmup",
                                )
                            },
                        }
                    )
                    continue
                job = _new_job(topology, finalist_index, row, seed)
                new_jobs.append(job)
                entries.append(
                    {
                        "topology": topology,
                        "finalist_index": finalist_index,
                        "seed": seed,
                        "job_id": job["job_id"],
                        "source": "new_training",
                        "run_dir": str(output_root / job["job_id"]),
                        "hyperparameters": {
                            name: job[name]
                            for name in (
                                "learning_rate",
                                "max_log_modulation",
                                "gate_output_bias",
                                "rp_eta_lambda",
                                "rp_interval",
                                "rp_warmup",
                            )
                        },
                    }
                )

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("at least one device is required")
    atomic_json(
        output_root / "finalist_training_manifest.json",
        {
            "schema_version": 1,
            "selection": str(selection_path),
            "selection_test_bank_accessed": False,
            "source_root": str(source_root),
            "output_root": str(output_root),
            "devices": devices,
            "entries": entries,
            "new_jobs": new_jobs,
            "reused_runs": sum(row["source"] == "reused_prior_run" for row in entries),
            "new_training_runs": len(new_jobs),
        },
    )

    queues = [new_jobs[index:: len(devices)] for index in range(len(devices))]
    lock = threading.Lock()
    status = {"completed": 0, "failed": 0, "skipped_existing": 0}
    started = time.time()
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    def write_status() -> None:
        atomic_json(
            output_root / "FINALIST_TRAINING_STATUS.json",
            {
                "schema_version": 1,
                **status,
                "new_training_runs": len(new_jobs),
                "elapsed_seconds": time.time() - started,
            },
        )

    def worker(device: str, queue: list[dict[str, Any]]) -> None:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = device
        environment["PYTHONPATH"] = str(Path.cwd().resolve())
        for job in queue:
            run_dir = output_root / job["job_id"]
            if (run_dir / "COMPLETED.json").is_file():
                with lock:
                    status["skipped_existing"] += 1
                    write_status()
                continue
            command = [
                sys.executable,
                "-m",
                "repro.manifold_benchmark.run_topology_hparam",
                "--phase",
                job["phase"],
                "--cell-id",
                job["cell_id"],
                "--job-id",
                job["job_id"],
                "--model",
                job["model"],
                "--topology",
                job["topology"],
                "--seed",
                str(job["seed"]),
                "--learning-rate",
                str(job["learning_rate"]),
                "--max-log-modulation",
                str(job["max_log_modulation"]),
                "--gate-output-bias",
                str(job["gate_output_bias"]),
                "--rp-eta-lambda",
                str(job["rp_eta_lambda"]),
                "--rp-interval",
                str(job["rp_interval"]),
                "--rp-warmup",
                str(job["rp_warmup"]),
                "--updates",
                str(job["updates"]),
                "--report-interval",
                "100",
                "--device",
                "cuda:0",
                "--config",
                str(args.training_config.expanduser().resolve(strict=True)),
                "--output",
                str(output_root),
            ]
            success = False
            return_codes = []
            for attempt in (1, 2):
                with (log_root / f"{job['job_id']}.log").open(
                    "a", encoding="utf-8"
                ) as log:
                    log.write(
                        f"\n=== attempt {attempt}/2 on physical GPU {device} ===\n"
                    )
                    log.flush()
                    completed = subprocess.run(
                        command,
                        cwd=Path.cwd(),
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                return_codes.append(int(completed.returncode))
                success = (
                    completed.returncode == 0
                    and (run_dir / "COMPLETED.json").is_file()
                )
                if success:
                    break
            if not success:
                atomic_json(
                    run_dir / "FAILED.json",
                    {
                        "schema_version": 1,
                        "job_id": job["job_id"],
                        "return_codes": return_codes,
                        "log": str(log_root / f"{job['job_id']}.log"),
                    },
                )
            with lock:
                status["completed" if success else "failed"] += 1
                write_status()

    write_status()
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(worker, device, queue)
            for device, queue in zip(devices, queues)
            if queue
        ]
        for future in futures:
            future.result()
    all_complete = all(
        (output_root / entry["job_id"] / "COMPLETED.json").is_file()
        for entry in entries
    )
    atomic_json(
        output_root / "FINALIST_TRAINING_COMPLETED.json",
        {
            "schema_version": 1,
            **status,
            "entry_count": len(entries),
            "new_training_runs": len(new_jobs),
            "reused_runs": sum(row["source"] == "reused_prior_run" for row in entries),
            "all_27_finalist_runs_complete": all_complete,
            "elapsed_seconds": time.time() - started,
        },
    )
    if not all_complete:
        raise RuntimeError("one or more finalist runs failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5")
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path(__file__).with_name("topology_hparam_v1.json"),
    )
    launch(parser.parse_args())


if __name__ == "__main__":
    main()
