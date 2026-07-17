"""CA-aware, validation-only reanalysis and finalist selection for H-C."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np

from repro.sagodi_protocol.artifacts import atomic_json

from .aggregate_topology_pilot import _module_command, _parallel_per_run, _run
from .topology_analysis_common import (
    discover_completed_runs,
    load_analysis_config,
    write_csv,
)


TOPOLOGIES = ("s1", "t2", "s2")
OBJECTIVES = (
    "validation_blank_h4096_intrinsic_radians",
    "absolute_log_finite128_tangent_gain",
    "topology_radial_output_q512",
    "topology_radial_same_memory_error512",
    "fiber_ratio",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _task_gate(result: dict[str, Any], config: dict[str, Any]) -> bool:
    threshold = float(config["eligibility"]["validation_nmse_db_strictly_below"])
    return bool(
        result["finite"]
        and result["final_validation"]["finite"]
        and result["beats_hold_baseline_intrinsic"]
        and float(result["validation_nmse_db"]) < threshold
        and result["blank_validation"]["all_finite"]
    )


def _rank_fraction(values: list[float]) -> list[float]:
    count = len(values)
    if count <= 1:
        return [0.0] * count
    order = sorted(range(count), key=lambda index: values[index])
    ranks = [0.0] * count
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and values[order[stop]] == values[order[start]]:
            stop += 1
        average = 0.5 * (start + stop - 1)
        for position in range(start, stop):
            ranks[order[position]] = average / (count - 1)
        start = stop
    return ranks


def _pareto_flags(rows: list[dict[str, Any]]) -> list[bool]:
    values = np.asarray(
        [[float(row[name]) for name in OBJECTIVES] for row in rows],
        dtype=np.float64,
    )
    flags = []
    for index in range(len(rows)):
        weakly_better = np.all(values <= values[index], axis=1)
        strictly_better = np.any(values < values[index], axis=1)
        flags.append(not bool(np.any(weakly_better & strictly_better)))
    return flags


def _config_key(manifest: dict[str, Any]) -> tuple[Any, ...]:
    return (
        manifest["topology"],
        float(manifest["learning_rate"]),
        float(manifest["max_log_modulation"]),
        float(manifest["gate_output_bias"]),
        float(manifest["rp_eta_lambda"]),
        int(manifest["rp_interval"]),
        int(manifest["rp_warmup"]),
    )


def _prefer_duplicate(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    phase_priority = {"refine": 2, "broad": 1}
    return max(
        (left, right),
        key=lambda row: (phase_priority.get(row["phase"], 0), row["job_id"]),
    )


def _metric_rows(
    run_root: Path,
    analysis_root: Path,
    records,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    geometry = {
        row["job_id"]: row
        for row in _read_csv(analysis_root / "geometry" / "geometry_metrics.csv")
    }
    dynamics = {
        row["job_id"]: row
        for row in _read_csv(analysis_root / "dynamics" / "dynamics_metrics.csv")
    }
    topology_normal = {
        row["job_id"]: row
        for row in _read_csv(
            analysis_root / "topology_normal" / "topology_normal_metrics.csv"
        )
    }
    rows: list[dict[str, Any]] = []
    for record in records:
        result = dict(record.result)
        manifest = dict(record.manifest)
        eligible = _task_gate(result, config)
        if not eligible:
            rows.append(
                {
                    "job_id": record.job_id,
                    "topology": record.topology,
                    "seed": record.seed,
                    "phase": manifest["phase"],
                    "learning_rate": manifest["learning_rate"],
                    "max_log_modulation": manifest["max_log_modulation"],
                    "gate_output_bias": manifest["gate_output_bias"],
                    "rp_eta_lambda": manifest["rp_eta_lambda"],
                    "rp_interval": manifest["rp_interval"],
                    "rp_warmup": manifest["rp_warmup"],
                    "validation_task_gate": False,
                    "validation_nmse_db": result["validation_nmse_db"],
                    "validation_task_intrinsic_radians": result["final_validation"][
                        "intrinsic_mean_radians"
                    ],
                    "validation_blank_h4096_intrinsic_radians": result[
                        "blank_validation"
                    ]["horizons"]["4096"].get("intrinsic_mean_radians"),
                    "duplicate_configuration": False,
                    "pareto_optimal": False,
                    "balanced_ca_rank": None,
                }
            )
            continue
        geometry_row = geometry[record.job_id]
        dynamics_row = dynamics[record.job_id]
        topology_normal_row = topology_normal[record.job_id]
        with np.load(
            analysis_root / "dynamics" / "runs" / f"{record.job_id}.npz",
            allow_pickle=False,
        ) as archive:
            finite_horizons = np.asarray(archive["finite_horizons"])
            finite_index = int(np.flatnonzero(finite_horizons == 128)[0])
            tangent_gain = float(
                np.mean(np.asarray(archive["finite_tangent_gain"])[finite_index])
            )
            normal_gain = float(
                np.mean(np.asarray(archive["finite_normal_gain"])[finite_index])
            )
            recovery_horizons = np.asarray(archive["recovery_horizons"])
            recovery_index = int(np.flatnonzero(recovery_horizons == 512)[0])
            radii = np.asarray(archive["kick_radii_relative"])
            radius_index = int(np.argmin(np.abs(radii - 0.05)))
            q512 = float(
                np.median(
                    np.asarray(archive["recovery_ratio"])[
                        radius_index, recovery_index
                    ]
                )
            )
            same_memory = float(
                np.mean(
                    np.asarray(archive["same_memory_error"])[
                        radius_index, recovery_index
                    ]
                )
            )
        rows.append(
            {
                "job_id": record.job_id,
                "topology": record.topology,
                "seed": record.seed,
                "phase": manifest["phase"],
                "learning_rate": manifest["learning_rate"],
                "max_log_modulation": manifest["max_log_modulation"],
                "gate_output_bias": manifest["gate_output_bias"],
                "rp_eta_lambda": manifest["rp_eta_lambda"],
                "rp_interval": manifest["rp_interval"],
                "rp_warmup": manifest["rp_warmup"],
                "validation_task_gate": True,
                "validation_nmse_db": result["validation_nmse_db"],
                "validation_task_intrinsic_radians": result["final_validation"][
                    "intrinsic_mean_radians"
                ],
                "validation_blank_h4096_intrinsic_radians": result[
                    "blank_validation"
                ]["horizons"]["4096"]["intrinsic_mean_radians"],
                "finite128_tangent_gain": tangent_gain,
                "absolute_log_finite128_tangent_gain": abs(math.log(max(1e-12, tangent_gain))),
                "finite128_normal_gain": normal_gain,
                "finite_normal_kick_q512": q512,
                "same_memory_error512": same_memory,
                "topology_radial_hidden_q512": float(
                    topology_normal_row["hidden_recovery_q512_median"]
                ),
                "topology_radial_output_q512": float(
                    topology_normal_row["output_radial_recovery_q512_median"]
                ),
                "topology_radial_same_memory_error512": float(
                    topology_normal_row["same_memory_error512_mean"]
                ),
                "fiber_ratio": float(geometry_row["fiber_ratio"]),
                "local_tangent_rank_full_fraction": float(
                    geometry_row["local_tangent_rank_full_fraction"]
                ),
                "far_latent_hidden_collapse_fraction": float(
                    geometry_row["far_latent_hidden_collapse_fraction"]
                ),
                "one_step_tangent_gain": float(
                    dynamics_row["one_step_tangent_gain_mean"]
                ),
                "one_step_normal_gain": float(
                    dynamics_row["one_step_sampled_normal_gain_mean"]
                ),
                "duplicate_configuration": False,
                "pareto_optimal": False,
                "balanced_ca_rank": None,
            }
        )

    record_map = {record.job_id: record for record in records}
    preferred: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if not row["validation_task_gate"]:
            continue
        key = _config_key(dict(record_map[row["job_id"]].manifest))
        if key in preferred:
            winner = _prefer_duplicate(preferred[key], row)
            loser = row if winner is preferred[key] else preferred[key]
            loser["duplicate_configuration"] = True
            preferred[key] = winner
        else:
            preferred[key] = row

    for topology in TOPOLOGIES:
        candidates = [
            row
            for row in rows
            if row["topology"] == topology
            and row["validation_task_gate"]
            and not row["duplicate_configuration"]
        ]
        for objective in OBJECTIVES:
            ranks = _rank_fraction([float(row[objective]) for row in candidates])
            for row, rank in zip(candidates, ranks):
                row[f"rank_{objective}"] = rank
        flags = _pareto_flags(candidates)
        for row, flag in zip(candidates, flags):
            row["pareto_optimal"] = flag
            row["balanced_ca_rank"] = statistics.mean(
                float(row[f"rank_{objective}"]) for objective in OBJECTIVES
            )
    return rows


def _select_finalists(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    count = int(config["selection_protocol"]["finalists_per_topology"])
    selected: dict[str, Any] = {}
    for topology in TOPOLOGIES:
        eligible = [
            row
            for row in rows
            if row["topology"] == topology
            and row["validation_task_gate"]
            and not row["duplicate_configuration"]
        ]
        pareto = [row for row in eligible if row["pareto_optimal"]]
        ordered_pareto = sorted(
            pareto,
            key=lambda row: (
                float(row["balanced_ca_rank"]),
                float(row["validation_task_intrinsic_radians"]),
                row["job_id"],
            ),
        )
        ordered_all = sorted(
            eligible,
            key=lambda row: (
                float(row["balanced_ca_rank"]),
                float(row["validation_task_intrinsic_radians"]),
                row["job_id"],
            ),
        )
        finalists = ordered_pareto[:count]
        for row in ordered_all:
            if len(finalists) >= count:
                break
            if row not in finalists:
                finalists.append(row)
        selected[topology] = [
            {
                key: row[key]
                for key in (
                    "job_id",
                    "phase",
                    "learning_rate",
                    "max_log_modulation",
                    "gate_output_bias",
                    "rp_eta_lambda",
                    "rp_interval",
                    "rp_warmup",
                    "validation_task_intrinsic_radians",
                    "validation_blank_h4096_intrinsic_radians",
                    "finite128_tangent_gain",
                    "finite128_normal_gain",
                    "finite_normal_kick_q512",
                    "same_memory_error512",
                    "topology_radial_hidden_q512",
                    "topology_radial_output_q512",
                    "topology_radial_same_memory_error512",
                    "fiber_ratio",
                    "balanced_ca_rank",
                    "pareto_optimal",
                )
            }
            for row in finalists
        ]
    return selected


def analyze(args: argparse.Namespace) -> None:
    cwd = Path.cwd().resolve()
    run_root = args.run_root.expanduser().resolve(strict=True)
    analysis_root = args.analysis_root.expanduser().resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)
    config_path = args.config.expanduser().resolve(strict=True)
    config = load_analysis_config(config_path)
    records, missing = discover_completed_runs(run_root, config)
    if missing:
        raise RuntimeError(f"missing completed candidates: {missing}")
    if any(record.manifest.get("test_bank_accessed") for record in records):
        raise RuntimeError("candidate training manifest reports test-bank access")
    if any(record.result.get("test_bank_accessed") for record in records):
        raise RuntimeError("candidate result reports test-bank access")

    success = {
        record.job_id: _task_gate(dict(record.result), config)
        for record in records
    }
    task_root = analysis_root / "task"
    task_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        task_root / "task_success.json",
        {
            "schema_version": 1,
            "selection_split": "validation",
            "test_bank_accessed": False,
            "task_success": success,
        },
    )
    atomic_json(
        analysis_root / "selection_analysis_manifest.json",
        {
            "schema_version": 1,
            "analysis_id": config["analysis_id"],
            "run_root": str(run_root),
            "analysis_root": str(analysis_root),
            "candidate_count": len(records),
            "eligible_candidate_runs": sum(success.values()),
            "selection_protocol": config["selection_protocol"],
            "test_bank_accessed": False,
        },
    )

    log_root = analysis_root / "logs"
    bank_root = analysis_root / "banks"
    if not (bank_root / "analysis_banks_manifest.json").is_file():
        _run(
            _module_command(
                "make_analysis_banks",
                "--output",
                str(bank_root),
                "--config",
                str(config_path),
            ),
            cwd=cwd,
            log_path=log_root / "make_analysis_banks.log",
        )
    job_ids = [record.job_id for record in records if success[record.job_id]]
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    common = [
        "--run-root",
        str(run_root),
        "--analysis-root",
        str(analysis_root),
        "--config",
        str(config_path),
        "--bank-root",
        str(bank_root),
    ]
    for module in (
        "analyze_manifold_geometry",
        "analyze_tangent_normal",
        "analyze_topology_normal",
    ):
        artifact_directory = {
            "analyze_manifold_geometry": "geometry",
            "analyze_tangent_normal": "dynamics",
            "analyze_topology_normal": "topology_normal",
        }[module]
        pending = [
            job_id
            for job_id in job_ids
            if not (
                analysis_root / artifact_directory / "runs" / f"{job_id}.json"
            ).is_file()
        ]
        if pending:
            _parallel_per_run(
                module=module,
                job_ids=pending,
                devices=devices,
                common_arguments=common,
                cwd=cwd,
                log_root=log_root,
            )
        _run(
            _module_command(module, *common, "--aggregate-only"),
            cwd=cwd,
            log_path=log_root / f"aggregate_{module}.log",
            environment={**os.environ, "PYTHONPATH": str(cwd)},
        )

    rows = _metric_rows(run_root, analysis_root, records, config)
    write_csv(analysis_root / "hc_attractor_candidate_metrics.csv", rows)
    selected = _select_finalists(rows, config)
    atomic_json(
        analysis_root / "hc_attractor_finalists.json",
        {
            "schema_version": 1,
            "selection_split": "validation_plus_independent_dynamics_bank",
            "test_bank_accessed": False,
            "candidate_runs": len(records),
            "eligible_candidate_runs": sum(success.values()),
            "deduplicated_eligible_configurations": sum(
                row["validation_task_gate"] and not row["duplicate_configuration"]
                for row in rows
            ),
            "objectives": list(OBJECTIVES),
            "selection_rule": (
                "task/finite gates, Pareto front, then equal-weight mean fractional "
                "rank over five CA objectives; task error is tie-break only"
            ),
            "selected": selected,
        },
    )
    atomic_json(
        analysis_root / "SELECTION_ANALYSIS_COMPLETED.json",
        {
            "schema_version": 1,
            "candidate_runs": len(records),
            "eligible_candidate_runs": sum(success.values()),
            "geometry_runs": len(job_ids),
            "dynamics_runs": len(job_ids),
            "topology_normal_runs": len(job_ids),
            "test_bank_accessed": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("topology_hc_attractor_selection_v1.json"),
    )
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
