"""Wait for the 36-run pilot, execute frozen analysis v1, and aggregate seeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable

from repro.sagodi_protocol.artifacts import atomic_json, strict_json_load

from .topology_analysis_common import (
    discover_completed_runs,
    expected_jobs,
    load_analysis_config,
    write_csv,
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"analysis command failed; see {log_path}")


def _module_command(module: str, *arguments: str) -> list[str]:
    return [sys.executable, "-m", f"repro.manifold_benchmark.{module}", *arguments]


def _wait_for_training(run_root: Path, expected_count: int, poll_seconds: int) -> None:
    while True:
        completed = len(list(run_root.glob("pilot__*/COMPLETED.json")))
        launcher_completed = (run_root / "LAUNCHER_COMPLETED.json").is_file()
        if completed == int(expected_count) and launcher_completed:
            return
        launcher_logs = run_root / "launcher_logs"
        failures = []
        if launcher_logs.is_dir():
            for path in launcher_logs.glob("*.log"):
                text = path.read_text(encoding="utf-8", errors="replace")
                if "Traceback (most recent call last)" in text:
                    failures.append(path.name)
        if failures:
            raise RuntimeError(f"training launcher logs contain failures: {failures}")
        time.sleep(min(30, max(1, int(poll_seconds))))


def _parallel_per_run(
    *,
    module: str,
    job_ids: list[str],
    devices: list[str],
    common_arguments: list[str],
    cwd: Path,
    log_root: Path,
) -> None:
    queues = [job_ids[index:: len(devices)] for index in range(len(devices))]

    def worker(device: str, queue: list[str]) -> None:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = device
        environment["PYTHONPATH"] = str(cwd)
        for job_id in queue:
            command = _module_command(
                module,
                *common_arguments,
                "--job-id",
                job_id,
                "--device",
                "cuda:0",
            )
            _run(
                command,
                cwd=cwd,
                log_path=log_root / module / f"{job_id}.log",
                environment=environment,
            )

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(worker, device, queue)
            for device, queue in zip(devices, queues)
            if queue
        ]
        for future in futures:
            future.result()


def _aggregate_seed_tables(analysis_root: Path, config: dict[str, Any]) -> None:
    task = _read_csv(analysis_root / "task_metrics.csv")
    blank = _read_csv(analysis_root / "blank" / "blank_metrics.csv")
    geometry_path = analysis_root / "geometry" / "geometry_metrics.csv"
    dynamics_path = analysis_root / "dynamics" / "dynamics_metrics.csv"
    retention_path = analysis_root / "hc_retention" / "hc_retention_metrics.csv"
    geometry = _read_csv(geometry_path) if geometry_path.stat().st_size else []
    dynamics = _read_csv(dynamics_path) if dynamics_path.stat().st_size else []
    retention = _read_csv(retention_path) if retention_path.stat().st_size else []
    blank_max = {
        row["job_id"]: row
        for row in blank
        if int(row["horizon"]) == max(config["blank_memory"]["horizons"])
    }
    geometry_map = {row["job_id"]: row for row in geometry}
    dynamics_map = {row["job_id"]: row for row in dynamics}
    retention_map = {row["job_id"]: row for row in retention}
    seed_rows: list[dict[str, Any]] = []
    for row in task:
        job_id = row["job_id"]
        merged = dict(row)
        for prefix, source, names in (
            (
                "blank4096",
                blank_max.get(job_id, {}),
                ("memory_error_mean", "clean_output_displacement_mean", "causal_state_norm_ratio_to_h0"),
            ),
            (
                "geometry",
                geometry_map.get(job_id, {}),
                ("local_tangent_rank_full_fraction", "knn_overlap", "latent_hidden_distance_spearman", "fiber_ratio"),
            ),
            (
                "dynamics",
                dynamics_map.get(job_id, {}),
                ("one_step_tangent_gain_mean", "one_step_sampled_normal_gain_mean", "one_step_tangent_to_normal_leakage_mean"),
            ),
            (
                "retention",
                retention_map.get(job_id, {}),
                ("final_base_lambda_median", "task_dynamic_lambda_median", "rp_call_count"),
            ),
        ):
            for name in names:
                merged[f"{prefix}_{name}"] = source.get(name)
        seed_rows.append(merged)
    write_csv(analysis_root / "seed_level_summary.csv", seed_rows)

    numeric_metrics = (
        "test_mean_error",
        "test_terminal_error",
        "test_trial_p95",
        "test_output_norm_error",
        "blank4096_memory_error_mean",
        "geometry_fiber_ratio",
        "dynamics_one_step_tangent_gain_mean",
        "dynamics_one_step_sampled_normal_gain_mean",
    )
    group_rows: list[dict[str, Any]] = []
    for topology in config["expected_training"]["topologies"]:
        for model in config["expected_training"]["models"]:
            group = [
                row for row in seed_rows if row["topology"] == topology and row["model"] == model
            ]
            summary: dict[str, Any] = {
                "model": model,
                "topology": topology,
                "seed_count": len(group),
                "task_success_count": sum(row["task_success"] == "True" for row in group),
            }
            for metric in numeric_metrics:
                values = [value for row in group if (value := _number(row.get(metric))) is not None]
                summary[f"{metric}_n"] = len(values)
                summary[f"{metric}_mean"] = statistics.mean(values) if values else None
                summary[f"{metric}_median"] = statistics.median(values) if values else None
                summary[f"{metric}_min"] = min(values) if values else None
                summary[f"{metric}_max"] = max(values) if values else None
            group_rows.append(summary)
    write_csv(analysis_root / "model_topology_summary.csv", group_rows)

    by_key = {(row["model"], row["topology"], int(row["seed"])): row for row in seed_rows}
    paired: list[dict[str, Any]] = []
    models = tuple(config["expected_training"]["models"])
    comparison_models = ("calru",) if "calru" in models else ("rnn", "gru", "lstm")
    for topology in config["expected_training"]["topologies"]:
        for baseline in comparison_models:
            for seed in config["expected_training"]["seeds"]:
                hc = by_key[("hc", topology, int(seed))]
                other = by_key[(baseline, topology, int(seed))]
                hc_test = _number(hc["test_mean_error"])
                other_test = _number(other["test_mean_error"])
                paired.append(
                    {
                        "topology": topology,
                        "baseline": baseline,
                        "seed": seed,
                        "hc_minus_baseline_test_error": (
                            None
                            if hc_test is None or other_test is None
                            else hc_test - other_test
                        ),
                        "hc_minus_baseline_blank4096_error": (
                            None
                            if _number(hc.get("blank4096_memory_error_mean")) is None
                            or _number(other.get("blank4096_memory_error_mean")) is None
                            else _number(hc["blank4096_memory_error_mean"])
                            - _number(other["blank4096_memory_error_mean"])
                        ),
                    }
                )
    write_csv(analysis_root / "paired_seed_differences.csv", paired)

    advance: dict[str, Any] = {}
    for topology in config["expected_training"]["topologies"]:
        group = [row for row in seed_rows if row["model"] == "hc" and row["topology"] == topology]
        finite = sum(row["finite"] == "True" for row in group)
        beats_hold = sum(float(row["test_hold_relative_improvement"]) > 0 for row in group)
        eligible = sum(row["task_success"] == "True" for row in group)
        advance[topology] = {
            "finite_seed_count": finite,
            "beats_hold_seed_count": beats_hold,
            "validation_eligible_seed_count": eligible,
            "task_advance": finite == 3 and beats_hold >= 2 and eligible >= 2,
        }
    atomic_json(
        analysis_root / "pilot_advance_rule.json",
        {"schema_version": 1, "topology": advance, "p_values_reported": False},
    )


def orchestrate(args: argparse.Namespace) -> None:
    cwd = Path.cwd().resolve()
    config_path = args.config.expanduser().resolve(strict=True)
    config = load_analysis_config(config_path)
    run_root = args.run_root.expanduser().resolve(strict=True)
    analysis_root = args.analysis_root.expanduser().resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("at least one device is required")
    atomic_json(
        analysis_root / "analysis_launcher_manifest.json",
        {
            "schema_version": 1,
            "analysis_id": config["analysis_id"],
            "run_root": str(run_root),
            "analysis_root": str(analysis_root),
            "devices": devices,
            "analysis_config": str(config_path),
            "wait_for_all_selected_checkpoints_before_test_access": True,
        },
    )
    if args.wait:
        _wait_for_training(
            run_root,
            int(config["expected_training"]["expected_runs"]),
            int(args.poll_seconds),
        )
    log_root = analysis_root / "logs"
    bank_root = analysis_root / "banks"
    if not (bank_root / "analysis_banks_manifest.json").is_file():
        _run(
            _module_command(
                "make_analysis_banks", "--output", str(bank_root), "--config", str(config_path)
            ),
            cwd=cwd,
            log_path=log_root / "make_analysis_banks.log",
        )
    _run(
        _module_command(
            "analyze_run_integrity",
            "--run-root",
            str(run_root),
            "--output",
            str(analysis_root),
            "--config",
            str(config_path),
            "--require-complete",
        ),
        cwd=cwd,
        log_path=log_root / "run_integrity.log",
    )
    task_environment = dict(os.environ)
    task_environment["CUDA_VISIBLE_DEVICES"] = devices[0]
    task_environment["PYTHONPATH"] = str(cwd)
    _run(
        _module_command(
            "analyze_transfer_task",
            "--run-root",
            str(run_root),
            "--analysis-root",
            str(analysis_root),
            "--config",
            str(config_path),
            "--device",
            "cuda:0",
        ),
        cwd=cwd,
        log_path=log_root / "analyze_transfer_task.log",
        environment=task_environment,
    )
    records, missing = discover_completed_runs(run_root, config)
    if missing:
        raise RuntimeError("training set became incomplete after integrity stage")
    success = strict_json_load(analysis_root / "task" / "task_success.json")["task_success"]
    all_jobs = [record.job_id for record in records]
    eligible_jobs = [record.job_id for record in records if success.get(record.job_id, False)]
    structure_jobs = (
        all_jobs
        if bool(config["execution"].get("analyze_structure_for_all_runs", False))
        else eligible_jobs
    )
    retention_models = set(config["hc_retention"].get("models", ["hc"]))
    retention_jobs = [
        record.job_id
        for record in records
        if record.model_id in retention_models
        and (
            bool(config["execution"].get("analyze_structure_for_all_runs", False))
            or success.get(record.job_id, False)
        )
    ]
    common = [
        "--run-root",
        str(run_root),
        "--analysis-root",
        str(analysis_root),
        "--config",
        str(config_path),
    ]
    _parallel_per_run(
        module="analyze_blank_memory",
        job_ids=all_jobs,
        devices=devices,
        common_arguments=common,
        cwd=cwd,
        log_root=log_root,
    )
    _run(
        _module_command("analyze_blank_memory", *common, "--aggregate-only"),
        cwd=cwd,
        log_path=log_root / "aggregate_blank.log",
    )
    bank_common = [*common, "--bank-root", str(bank_root)]
    _parallel_per_run(
        module="analyze_manifold_geometry",
        job_ids=structure_jobs,
        devices=devices,
        common_arguments=bank_common,
        cwd=cwd,
        log_root=log_root,
    )
    _run(
        _module_command("analyze_manifold_geometry", *bank_common, "--aggregate-only"),
        cwd=cwd,
        log_path=log_root / "aggregate_geometry.log",
    )
    _parallel_per_run(
        module="analyze_tangent_normal",
        job_ids=structure_jobs,
        devices=devices,
        common_arguments=bank_common,
        cwd=cwd,
        log_root=log_root,
    )
    _run(
        _module_command("analyze_tangent_normal", *bank_common, "--aggregate-only"),
        cwd=cwd,
        log_path=log_root / "aggregate_dynamics.log",
    )
    _parallel_per_run(
        module="analyze_hc_retention",
        job_ids=retention_jobs,
        devices=devices,
        common_arguments=common,
        cwd=cwd,
        log_root=log_root,
    )
    _run(
        _module_command("analyze_hc_retention", *common, "--aggregate-only"),
        cwd=cwd,
        log_path=log_root / "aggregate_hc_retention.log",
    )
    _aggregate_seed_tables(analysis_root, config)
    plot_module = (
        "plot_topology_hparam_analysis"
        if config["analysis_id"] == "manifold_topology_hparam_analysis_v1"
        else "plot_topology_analysis"
    )
    _run(
        _module_command(plot_module, "--analysis-root", str(analysis_root)),
        cwd=cwd,
        log_path=log_root / "plot_topology_analysis.log",
    )
    atomic_json(
        analysis_root / "ANALYSIS_COMPLETED.json",
        {
            "schema_version": 1,
            "analysis_id": config["analysis_id"],
            "completed_training_runs": len(records),
            "task_success_runs": len(eligible_jobs),
            "structure_analysis_runs": len(structure_jobs),
            "retention_analysis_runs": len(retention_jobs),
            "failed_runs_remain_in_denominator": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("topology_analysis_v1.json")
    )
    args = parser.parse_args()
    orchestrate(args)


if __name__ == "__main__":
    main()
