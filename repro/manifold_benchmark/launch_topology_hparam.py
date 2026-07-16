"""Restartable six-GPU broad-to-narrow CA-LRU/H-C topology search."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

from repro.sagodi_protocol.artifacts import atomic_json, strict_json_load

from .run_topology_hparam import CONFIG_PATH, load_search_config


def _number_slug(value: float) -> str:
    return format(float(value), ".6g").replace("-", "m").replace("+", "").replace(".", "p")


def _job_id(job: dict[str, Any]) -> str:
    return (
        f"{job['phase']}__{job['model']}__{job['topology']}__"
        f"{job['cell_id']}__seed{int(job['seed'])}"
    )


def _base_job(
    *,
    phase: str,
    model: str,
    topology: str,
    seed: int,
    learning_rate: float,
    max_log_modulation: float = 0.05,
    gate_output_bias: float = -0.1,
    rp_eta_lambda: float = 1000.0,
    rp_interval: int = 50,
    rp_warmup: int = 1500,
    updates: int = 5000,
    cell_id: str | None = None,
) -> dict[str, Any]:
    if cell_id is None:
        pieces = [f"lr{_number_slug(learning_rate)}"]
        if model == "hc":
            pieces.extend(
                [
                    f"a{_number_slug(max_log_modulation)}",
                    f"b{_number_slug(gate_output_bias)}",
                ]
            )
        pieces.extend([f"eta{_number_slug(rp_eta_lambda)}", f"i{int(rp_interval)}"])
        cell_id = "_".join(pieces)
    job = {
        "phase": phase,
        "cell_id": cell_id,
        "model": model,
        "topology": topology,
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "max_log_modulation": float(max_log_modulation),
        "gate_output_bias": float(gate_output_bias),
        "rp_eta_lambda": float(rp_eta_lambda),
        "rp_interval": int(rp_interval),
        "rp_warmup": int(rp_warmup),
        "updates": int(updates),
    }
    job["job_id"] = _job_id(job)
    return job


def smoke_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    row = config["search"]["smoke"]
    return [
        _base_job(
            phase="smoke",
            model=model,
            topology=topology,
            seed=int(row["seed"]),
            learning_rate=float(config["models"][model]["learning_rate"]),
            rp_warmup=200,
            updates=int(row["updates"]),
            cell_id="registered_debug",
        )
        for topology in config["search"]["topologies"]
        for model in ("calru", "hc")
    ]


def broad_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    row = config["search"]["broad"]
    jobs: list[dict[str, Any]] = []
    for topology in config["search"]["topologies"]:
        for learning_rate in row["hc_learning_rates"]:
            for modulation in row["hc_max_log_modulations"]:
                jobs.append(
                    _base_job(
                        phase="broad",
                        model="hc",
                        topology=topology,
                        seed=int(row["seed"]),
                        learning_rate=float(learning_rate),
                        max_log_modulation=float(modulation),
                    )
                )
        for cell in row["hc_extra_cells"]:
            jobs.append(
                _base_job(
                    phase="broad",
                    model="hc",
                    topology=topology,
                    seed=int(row["seed"]),
                    learning_rate=float(cell["learning_rate"]),
                    max_log_modulation=float(cell["max_log_modulation"]),
                )
            )
        for learning_rate in row["calru_learning_rates"]:
            jobs.append(
                _base_job(
                    phase="broad",
                    model="calru",
                    topology=topology,
                    seed=int(row["seed"]),
                    learning_rate=float(learning_rate),
                )
            )
    return jobs


def _completed_result(root: Path, job: dict[str, Any]) -> dict[str, Any] | None:
    run_dir = root / job["job_id"]
    if not (run_dir / "COMPLETED.json").is_file():
        return None
    result = strict_json_load(run_dir / "result.json")
    manifest = strict_json_load(run_dir / "manifest.json")
    return {"job": job, "result": result, "manifest": manifest}


def _selection_key(candidate: dict[str, Any], topology: str) -> tuple[Any, ...]:
    result = candidate["result"]
    final = result["final_validation"]
    blank = result["blank_validation"]
    h4096 = blank["horizons"].get("4096", {})
    blank_finite = bool(blank["all_finite"])
    task_gate = bool(result["beats_hold_baseline_intrinsic"]) and float(
        result["validation_nmse_db"]
    ) < -20.0
    task_error = float(final["intrinsic_mean_radians"])
    blank_error = (
        float(h4096["intrinsic_mean_radians"])
        if h4096.get("intrinsic_mean_radians") is not None
        else math.inf
    )
    # S1 prioritizes survival through the long blank because the prior H-C
    # winner collapsed after late training.  T2/S2 first require task fit.
    if topology == "s1":
        return (not blank_finite, not task_gate, task_error, blank_error)
    return (not task_gate, task_error, not blank_finite, blank_error)


def _json_selection_key(candidate: dict[str, Any], topology: str) -> list[Any]:
    return [
        value if not isinstance(value, float) or math.isfinite(value) else None
        for value in _selection_key(candidate, topology)
    ]


def select_broad(root: Path, jobs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    path = root / "selection" / "broad_selection.json"
    if path.is_file():
        return strict_json_load(path)
    selected: dict[str, Any] = {}
    candidates_payload: dict[str, Any] = {}
    for topology in config["search"]["topologies"]:
        selected[topology] = {}
        candidates_payload[topology] = {}
        for model in ("hc", "calru"):
            candidates = [
                item
                for job in jobs
                if job["topology"] == topology and job["model"] == model
                if (item := _completed_result(root, job)) is not None
            ]
            if not candidates:
                raise RuntimeError(f"no completed broad candidate for {model}/{topology}")
            candidates.sort(key=lambda item: _selection_key(item, topology))
            selected[topology][model] = candidates[0]["job"]
            candidates_payload[topology][model] = [
                {
                    "job_id": item["job"]["job_id"],
                    "selection_key": _json_selection_key(item, topology),
                    "validation_nmse_db": item["result"]["validation_nmse_db"],
                    "validation_intrinsic_mean_radians": item["result"]["final_validation"][
                        "intrinsic_mean_radians"
                    ],
                    "blank_all_finite": item["result"]["blank_validation"]["all_finite"],
                }
                for item in candidates
            ]
    payload = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_bank_accessed": False,
        "final_checkpoint_only": True,
        "selected": selected,
        "ranked_candidates": candidates_payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload)
    return payload


def refine_jobs(config: dict[str, Any], broad_selection: dict[str, Any]) -> list[dict[str, Any]]:
    row = config["search"]["refine"]
    jobs: list[dict[str, Any]] = []
    for topology in config["search"]["topologies"]:
        hc = broad_selection["selected"][topology]["hc"]
        for bias in row["hc_gate_output_biases"]:
            for dose in row["rp_doses"]:
                jobs.append(
                    _base_job(
                        phase="refine",
                        model="hc",
                        topology=topology,
                        seed=int(row["seed"]),
                        learning_rate=float(hc["learning_rate"]),
                        max_log_modulation=float(hc["max_log_modulation"]),
                        gate_output_bias=float(bias),
                        rp_eta_lambda=float(dose["eta_lambda"]),
                        rp_interval=int(dose["intervention_interval_updates"]),
                    )
                )
        calru = broad_selection["selected"][topology]["calru"]
        for seed in row["calru_additional_seeds"]:
            jobs.append(
                _base_job(
                    phase="refine",
                    model="calru",
                    topology=topology,
                    seed=int(seed),
                    learning_rate=float(calru["learning_rate"]),
                )
            )
    return jobs


def select_refine(root: Path, jobs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    path = root / "selection" / "refine_selection.json"
    if path.is_file():
        return strict_json_load(path)
    count = int(config["search"]["robust"]["hc_finalists_per_topology"])
    selected: dict[str, Any] = {}
    ranked: dict[str, Any] = {}
    for topology in config["search"]["topologies"]:
        candidates = [
            item
            for job in jobs
            if job["topology"] == topology and job["model"] == "hc"
            if (item := _completed_result(root, job)) is not None
        ]
        if len(candidates) < count:
            raise RuntimeError(f"too few completed H-C refine candidates for {topology}")
        candidates.sort(key=lambda item: _selection_key(item, topology))
        selected[topology] = [item["job"] for item in candidates[:count]]
        ranked[topology] = [
            {
                "job_id": item["job"]["job_id"],
                "selection_key": _json_selection_key(item, topology),
                "validation_nmse_db": item["result"]["validation_nmse_db"],
                "validation_intrinsic_mean_radians": item["result"]["final_validation"][
                    "intrinsic_mean_radians"
                ],
                "blank_all_finite": item["result"]["blank_validation"]["all_finite"],
            }
            for item in candidates
        ]
    payload = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_bank_accessed": False,
        "final_checkpoint_only": True,
        "selected_hc_finalists": selected,
        "ranked_candidates": ranked,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload)
    return payload


def robust_jobs(config: dict[str, Any], refine_selection: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    seeds = config["search"]["robust"]["additional_seeds"]
    for topology in config["search"]["topologies"]:
        for index, finalist in enumerate(refine_selection["selected_hc_finalists"][topology], 1):
            for seed in seeds:
                job = _base_job(
                    phase="robust",
                    model="hc",
                    topology=topology,
                    seed=int(seed),
                    learning_rate=float(finalist["learning_rate"]),
                    max_log_modulation=float(finalist["max_log_modulation"]),
                    gate_output_bias=float(finalist["gate_output_bias"]),
                    rp_eta_lambda=float(finalist["rp_eta_lambda"]),
                    rp_interval=int(finalist["rp_interval"]),
                    cell_id=f"finalist{index}_{finalist['cell_id']}",
                )
                jobs.append(job)
    return jobs


def _balanced_queues(jobs: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    queues: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    loads = [0 for _ in range(workers)]
    for job in sorted(jobs, key=lambda item: item["job_id"]):
        index = min(range(workers), key=lambda value: (loads[value], value))
        queues[index].append(job)
        loads[index] += int(job["updates"])
    return queues


def run_phase(
    *,
    phase: str,
    jobs: list[dict[str, Any]],
    root: Path,
    devices: list[str],
    config_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    queues = _balanced_queues(jobs, len(devices))
    phase_root = root / "phases" / phase
    log_root = phase_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        phase_root / "launcher_manifest.json",
        {
            "schema_version": 1,
            "phase": phase,
            "jobs": jobs,
            "devices": devices,
            "queue_assignment": {
                device: [job["job_id"] for job in queue]
                for device, queue in zip(devices, queues)
            },
            "restart_policy": "resume_progress_or_skip_completed",
            "individual_failure_policy": "retry_then_record_and_continue",
        },
    )
    attempts = int(config["execution"]["attempts_per_job"])
    lock = threading.Lock()
    state = {"completed": 0, "failed": 0, "skipped_existing": 0}
    started = time.time()

    def write_status() -> None:
        atomic_json(
            root / "CAMPAIGN_STATUS.json",
            {
                "schema_version": 1,
                "active_phase": phase,
                "phase_jobs": len(jobs),
                **state,
                "elapsed_seconds": time.time() - started,
                "updated_unix_seconds": time.time(),
            },
        )

    def worker(device: str, queue: list[dict[str, Any]]) -> None:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = device
        environment["PYTHONPATH"] = os.getcwd()
        for job in queue:
            run_dir = root / job["job_id"]
            if (run_dir / "COMPLETED.json").is_file():
                with lock:
                    state["skipped_existing"] += 1
                    write_status()
                continue
            command = [
                sys.executable,
                "-m",
                "repro.manifold_benchmark.run_topology_hparam",
                "--phase",
                str(job["phase"]),
                "--cell-id",
                str(job["cell_id"]),
                "--job-id",
                str(job["job_id"]),
                "--model",
                str(job["model"]),
                "--topology",
                str(job["topology"]),
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
                str(config["training"]["report_interval"]),
                "--device",
                "cuda:0",
                "--config",
                str(config_path),
                "--output",
                str(root),
            ]
            return_codes: list[int] = []
            for attempt in range(1, attempts + 1):
                with (log_root / f"{job['job_id']}.log").open("a", encoding="utf-8") as log:
                    log.write(f"\n=== attempt {attempt}/{attempts} on physical GPU {device} ===\n")
                    log.flush()
                    completed = subprocess.run(
                        command,
                        cwd=os.getcwd(),
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        text=True,
                    )
                return_codes.append(int(completed.returncode))
                if completed.returncode == 0 and (run_dir / "COMPLETED.json").is_file():
                    break
            success = (run_dir / "COMPLETED.json").is_file()
            if not success:
                run_dir.mkdir(parents=True, exist_ok=True)
                atomic_json(
                    run_dir / "FAILED.json",
                    {
                        "schema_version": 1,
                        "job_id": job["job_id"],
                        "phase": phase,
                        "return_codes": return_codes,
                        "individual_failure_did_not_stop_other_jobs": True,
                        "log": str(log_root / f"{job['job_id']}.log"),
                    },
                )
            with lock:
                state["completed" if success else "failed"] += 1
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
    payload = {
        "schema_version": 1,
        "phase": phase,
        "jobs": len(jobs),
        **state,
        "all_jobs_accounted_for": sum(state.values()) == len(jobs),
        "completed_or_preexisting": state["completed"] + state["skipped_existing"],
    }
    atomic_json(phase_root / "PHASE_FINISHED.json", payload)
    return payload


def _run_sequence(args: argparse.Namespace, config: dict[str, Any]) -> None:
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = args.config.expanduser().resolve(strict=True)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("at least one CUDA device is required")
    atomic_json(
        root / "campaign_manifest.json",
        {
            "schema_version": 1,
            "campaign_id": config["campaign_id"],
            "config": config,
            "devices": devices,
            "selection_split": "validation",
            "test_bank_accessed": False,
            "resume_command": (
                f"PYTHONPATH=. {sys.executable} -m "
                "repro.manifold_benchmark.launch_topology_hparam "
                f"--phase all --devices {','.join(devices)} --config {config_path} "
                f"--output {root}"
            ),
        },
    )

    smoke = smoke_jobs(config)
    broad = broad_jobs(config)
    if args.phase in {"smoke", "all"}:
        status = run_phase(
            phase="smoke", jobs=smoke, root=root, devices=devices,
            config_path=config_path, config=config
        )
        if status["completed_or_preexisting"] != len(smoke):
            raise RuntimeError("smoke phase did not complete all six adapter checks")
        if args.phase == "smoke":
            return
    if args.phase in {"broad", "all"}:
        run_phase(
            phase="broad", jobs=broad, root=root, devices=devices,
            config_path=config_path, config=config
        )
        if args.phase == "broad":
            select_broad(root, broad, config)
            return
    broad_selection = select_broad(root, broad, config)
    refine = refine_jobs(config, broad_selection)
    if args.phase in {"refine", "all"}:
        run_phase(
            phase="refine", jobs=refine, root=root, devices=devices,
            config_path=config_path, config=config
        )
        if args.phase == "refine":
            select_refine(root, refine, config)
            return
    refine_selection = select_refine(root, refine, config)
    robust = robust_jobs(config, refine_selection)
    if args.phase in {"robust", "all"}:
        run_phase(
            phase="robust", jobs=robust, root=root, devices=devices,
            config_path=config_path, config=config
        )
    expected = smoke + broad + refine + robust
    completed = sum((root / job["job_id"] / "COMPLETED.json").is_file() for job in expected)
    failed = [job["job_id"] for job in expected if (root / job["job_id"] / "FAILED.json").is_file()]
    atomic_json(
        root / "CAMPAIGN_TRAINING_FINISHED.json",
        {
            "schema_version": 1,
            "campaign_id": config["campaign_id"],
            "expected_jobs_including_smoke": len(expected),
            "completed_jobs": completed,
            "failed_jobs": failed,
            "selection_split": "validation",
            "test_bank_accessed": False,
            "broad_selection": broad_selection["selected"],
            "refine_selection": refine_selection["selected_hc_finalists"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "broad", "refine", "robust", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    _run_sequence(args, load_search_config(args.config))


if __name__ == "__main__":
    main()
