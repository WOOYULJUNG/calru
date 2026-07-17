"""Three-seed CA-aware analysis and final H-C configuration selection."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from repro.sagodi_protocol.artifacts import atomic_json, strict_json_load

from .aggregate_topology_pilot import _module_command, _parallel_per_run, _run
from .analyze_hc_attractor_sweep import _rank_fraction
from .topology_analysis_common import (
    discover_completed_runs,
    load_analysis_config,
    write_csv,
)


TOPOLOGIES = ("s1", "t2", "s2")
GROUP_OBJECTIVES = (
    "median_validation_blank_h4096_intrinsic_radians",
    "median_absolute_log_finite128_tangent_gain",
    "median_topology_radial_output_q512",
    "median_topology_radial_same_memory_error512",
    "median_fiber_ratio",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _task_gate(result: dict[str, Any], config: dict[str, Any]) -> bool:
    return bool(
        result["finite"]
        and result["final_validation"]["finite"]
        and result["blank_validation"]["all_finite"]
        and result["beats_hold_baseline_intrinsic"]
        and float(result["validation_nmse_db"])
        < float(config["eligibility"]["validation_nmse_db_strictly_below"])
    )


def _median(rows: list[dict[str, Any]], name: str) -> float:
    return float(statistics.median(float(row[name]) for row in rows))


def _per_seed_rows(
    records,
    entry_map: dict[str, dict[str, Any]],
    analysis_root: Path,
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
    output = []
    for record in records:
        entry = entry_map[record.job_id]
        result = dict(record.result)
        geometry_row = geometry[record.job_id]
        dynamics_row = dynamics[record.job_id]
        topology_row = topology_normal[record.job_id]
        with np.load(
            analysis_root / "dynamics" / "runs" / f"{record.job_id}.npz",
            allow_pickle=False,
        ) as archive:
            finite_horizons = np.asarray(archive["finite_horizons"])
            index128 = int(np.flatnonzero(finite_horizons == 128)[0])
            tangent = float(
                np.mean(np.asarray(archive["finite_tangent_gain"])[index128])
            )
            normal = float(
                np.mean(np.asarray(archive["finite_normal_gain"])[index128])
            )
            recovery_horizons = np.asarray(archive["recovery_horizons"])
            index512 = int(np.flatnonzero(recovery_horizons == 512)[0])
            radii = np.asarray(archive["kick_radii_relative"])
            radius_index = int(np.argmin(np.abs(radii - 0.05)))
            random_q = float(
                np.median(
                    np.asarray(archive["recovery_ratio"])[radius_index, index512]
                )
            )
        output.append(
            {
                "job_id": record.job_id,
                "topology": record.topology,
                "finalist_index": int(entry["finalist_index"]),
                "seed": record.seed,
                "source": entry["source"],
                **entry["hyperparameters"],
                "task_gate": _task_gate(result, config),
                "validation_nmse_db": float(result["validation_nmse_db"]),
                "validation_task_intrinsic_radians": float(
                    result["final_validation"]["intrinsic_mean_radians"]
                ),
                "validation_blank_h4096_intrinsic_radians": float(
                    result["blank_validation"]["horizons"]["4096"][
                        "intrinsic_mean_radians"
                    ]
                ),
                "finite128_tangent_gain": tangent,
                "absolute_log_finite128_tangent_gain": abs(
                    math.log(max(1e-12, tangent))
                ),
                "finite128_normal_gain": normal,
                "random_normal_q512": random_q,
                "topology_radial_hidden_q512": float(
                    topology_row["hidden_recovery_q512_median"]
                ),
                "topology_radial_output_q512": float(
                    topology_row["output_radial_recovery_q512_median"]
                ),
                "topology_radial_same_memory_error512": float(
                    topology_row["same_memory_error512_mean"]
                ),
                "fiber_ratio": float(geometry_row["fiber_ratio"]),
                "local_tangent_rank_full_fraction": float(
                    geometry_row["local_tangent_rank_full_fraction"]
                ),
                "one_step_tangent_gain": float(
                    dynamics_row["one_step_tangent_gain_mean"]
                ),
                "one_step_normal_gain": float(
                    dynamics_row["one_step_sampled_normal_gain_mean"]
                ),
            }
        )
    return output


def _group_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for topology in TOPOLOGIES:
        for finalist_index in (1, 2, 3):
            group = [
                row
                for row in seed_rows
                if row["topology"] == topology
                and int(row["finalist_index"]) == finalist_index
            ]
            if len(group) != 3:
                raise RuntimeError(
                    f"expected three seeds for {topology}/f{finalist_index}"
                )
            first = group[0]
            output.append(
                {
                    "topology": topology,
                    "finalist_index": finalist_index,
                    **{
                        name: first[name]
                        for name in (
                            "learning_rate",
                            "max_log_modulation",
                            "gate_output_bias",
                            "rp_eta_lambda",
                            "rp_interval",
                            "rp_warmup",
                        )
                    },
                    "task_gate_count": sum(bool(row["task_gate"]) for row in group),
                    "median_validation_task_intrinsic_radians": _median(
                        group, "validation_task_intrinsic_radians"
                    ),
                    "median_validation_blank_h4096_intrinsic_radians": _median(
                        group, "validation_blank_h4096_intrinsic_radians"
                    ),
                    "median_finite128_tangent_gain": _median(
                        group, "finite128_tangent_gain"
                    ),
                    "median_absolute_log_finite128_tangent_gain": _median(
                        group, "absolute_log_finite128_tangent_gain"
                    ),
                    "median_finite128_normal_gain": _median(
                        group, "finite128_normal_gain"
                    ),
                    "median_random_normal_q512": _median(
                        group, "random_normal_q512"
                    ),
                    "median_topology_radial_hidden_q512": _median(
                        group, "topology_radial_hidden_q512"
                    ),
                    "median_topology_radial_output_q512": _median(
                        group, "topology_radial_output_q512"
                    ),
                    "median_topology_radial_same_memory_error512": _median(
                        group, "topology_radial_same_memory_error512"
                    ),
                    "median_fiber_ratio": _median(group, "fiber_ratio"),
                    "median_local_tangent_rank_full_fraction": _median(
                        group, "local_tangent_rank_full_fraction"
                    ),
                    "configuration_gate": sum(
                        bool(row["task_gate"]) for row in group
                    )
                    >= 2,
                    "pareto_optimal": False,
                    "balanced_ca_rank": None,
                    "selected": False,
                }
            )
    for topology in TOPOLOGIES:
        candidates = [
            row
            for row in output
            if row["topology"] == topology and row["configuration_gate"]
        ]
        if not candidates:
            continue
        for objective in GROUP_OBJECTIVES:
            ranks = _rank_fraction(
                [float(row[objective]) for row in candidates]
            )
            for row, rank in zip(candidates, ranks):
                row[f"rank_{objective}"] = rank
        values = np.asarray(
            [[float(row[name]) for name in GROUP_OBJECTIVES] for row in candidates]
        )
        flags = []
        for index in range(len(candidates)):
            weak = np.all(values <= values[index], axis=1)
            strict = np.any(values < values[index], axis=1)
            flags.append(not bool(np.any(weak & strict)))
        for row, flag in zip(candidates, flags):
            row["pareto_optimal"] = flag
            row["balanced_ca_rank"] = statistics.mean(
                float(row[f"rank_{objective}"])
                for objective in GROUP_OBJECTIVES
            )
        ordered = sorted(
            candidates,
            key=lambda row: (
                -int(row["task_gate_count"]),
                not bool(row["pareto_optimal"]),
                float(row["balanced_ca_rank"]),
                float(row["median_validation_task_intrinsic_radians"]),
                int(row["finalist_index"]),
            ),
        )
        ordered[0]["selected"] = True
    return output


def analyze(args: argparse.Namespace) -> None:
    cwd = Path.cwd().resolve()
    run_root = args.run_root.expanduser().resolve(strict=True)
    completed = strict_json_load(run_root / "FINALIST_TRAINING_COMPLETED.json")
    if not completed["all_27_finalist_runs_complete"]:
        raise RuntimeError("finalist training is incomplete")
    training_manifest = strict_json_load(
        run_root / "finalist_training_manifest.json"
    )
    entry_map = {
        row["job_id"]: row for row in training_manifest["entries"]
    }
    config_path = args.config.expanduser().resolve(strict=True)
    config = load_analysis_config(config_path)
    records, missing = discover_completed_runs(run_root, config)
    if missing:
        raise RuntimeError(f"missing finalist runs: {missing}")
    if set(entry_map) != {record.job_id for record in records}:
        raise RuntimeError("analysis config and finalist manifest differ")
    if any(record.manifest.get("test_bank_accessed") for record in records):
        raise RuntimeError("training manifest reports test access")
    if any(record.result.get("test_bank_accessed") for record in records):
        raise RuntimeError("training result reports test access")

    analysis_root = args.analysis_root.expanduser().resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)
    gates = {
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
            "task_success": gates,
        },
    )
    bank_root = analysis_root / "banks"
    log_root = analysis_root / "logs"
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
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    job_ids = [record.job_id for record in records]
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
    for module, artifact_directory in (
        ("analyze_manifold_geometry", "geometry"),
        ("analyze_tangent_normal", "dynamics"),
        ("analyze_topology_normal", "topology_normal"),
    ):
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

    seed_rows = _per_seed_rows(
        records, entry_map, analysis_root, config
    )
    group_rows = _group_rows(seed_rows)
    write_csv(analysis_root / "finalist_seed_metrics.csv", seed_rows)
    write_csv(analysis_root / "finalist_configuration_summary.csv", group_rows)
    selected = {
        row["topology"]: row for row in group_rows if row["selected"]
    }
    if set(selected) != set(TOPOLOGIES):
        raise RuntimeError("a topology has no selected H-C configuration")
    selected_jobs = {
        topology: [
            row["job_id"]
            for row in seed_rows
            if row["topology"] == topology
            and int(row["finalist_index"])
            == int(selected[topology]["finalist_index"])
        ]
        for topology in TOPOLOGIES
    }
    atomic_json(
        analysis_root / "selected_hc_attractor_configurations.json",
        {
            "schema_version": 1,
            "selection_split": "validation_plus_independent_dynamics_bank",
            "test_bank_accessed": False,
            "selection_rule": config["selection_protocol"],
            "selected": selected,
            "selected_jobs": selected_jobs,
        },
    )
    atomic_json(
        analysis_root / "FINALIST_ANALYSIS_COMPLETED.json",
        {
            "schema_version": 1,
            "run_count": len(records),
            "configuration_count": len(group_rows),
            "selected_configuration_count": len(selected),
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
        default=Path(__file__).with_name(
            "topology_hc_attractor_finalists_v1.json"
        ),
    )
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
