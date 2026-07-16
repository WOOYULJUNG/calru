"""Run and summarize full CA analyses after an H-C local grid finishes.

The training sweep is intentionally kept separate from this script.  This
postprocessor waits for every registered screen result, excludes cells that do
not pass the all-seed blank-stability gate, runs the existing checkpoint-only
attractor analysis on the remaining cells, and emits one auditable comparison
table.  A previously analyzed reference condition can be included without
retraining or recomputing it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from repro.sagodi_protocol.artifacts import atomic_json


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result.append(float(value))
    return result


def _median(values: Iterable[Any]) -> float | None:
    finite = _finite(values)
    return float(statistics.median(finite)) if finite else None


def _mean(values: Iterable[Any]) -> float | None:
    finite = _finite(values)
    return float(statistics.fmean(finite)) if finite else None


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _wait_for_results(
    training_root: Path,
    *,
    expected: int,
    poll_seconds: float,
) -> None:
    while True:
        count = sum(1 for _ in (training_root / "runs").glob("*/result.json"))
        print(f"screen results: {count}/{expected}", flush=True)
        if count >= expected:
            return
        time.sleep(poll_seconds)


def _screen_rows(
    training_root: Path,
    cells: list[dict[str, Any]],
    seeds: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for cell in cells:
        cell_id = str(cell["id"])
        results: list[dict[str, Any]] = []
        missing: list[int] = []
        for seed in seeds:
            path = training_root / "runs" / f"{cell_id}__seed{seed:02d}" / "result.json"
            if path.is_file():
                results.append(_read_json(path))
            else:
                missing.append(seed)
        stable = [
            result
            for result in results
            if bool(_nested(result, "autonomous_screen", "stable_through_horizon"))
        ]
        rows[cell_id] = {
            "cell": cell,
            "seed_count": len(results),
            "missing_seeds": missing,
            "blank_stable_seed_count": len(stable),
            "eligible_for_full_analysis": len(stable) == len(seeds) and not missing,
            "task_nmse_per_seed": [
                _nested(result, "task_evaluation", "metrics", "masked_nmse")
                for result in results
            ],
            "task_rmse_per_seed": [
                math.sqrt(float(value)) if value is not None and float(value) >= 0 else None
                for value in (
                    _nested(result, "task_evaluation", "metrics", "masked_mse")
                    for result in results
                )
            ],
            "blank_error_per_seed": [
                _nested(
                    result,
                    "autonomous_screen",
                    "terminal_memory",
                    "mean_angular_error_radians",
                )
                for result in results
            ],
        }
    return rows


def _reference_screen_row(
    summary_path: Path,
    *,
    summary_condition: str,
    cell: dict[str, Any],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    summary = _read_json(summary_path)
    condition = summary["conditions"][summary_condition]
    results = condition["per_seed"]
    stable = [
        result
        for result in results
        if bool(_nested(result, "autonomous_screen", "stable_through_horizon"))
    ]
    return {
        "cell": cell,
        "seed_count": len(results),
        "missing_seeds": [],
        "blank_stable_seed_count": len(stable),
        "eligible_for_full_analysis": len(stable) == len(seeds),
        "task_nmse_per_seed": [
            _nested(result, "task_evaluation", "metrics", "masked_nmse")
            for result in results
        ],
        "task_rmse_per_seed": [
            math.sqrt(float(_nested(result, "task_evaluation", "metrics", "masked_mse")))
            for result in results
        ],
        "blank_error_per_seed": [
            _nested(
                result,
                "autonomous_screen",
                "terminal_memory",
                "mean_angular_error_radians",
            )
            for result in results
        ],
        "reference": True,
    }


def _run_analysis(
    *,
    training_root: Path,
    bank: Path,
    output: Path,
    cell_id: str,
    seeds: tuple[int, ...],
    physical_gpu: str,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "analysis.log"
    command = [
        sys.executable,
        "-m",
        "repro.state_dependent_retention.analyze_attractor",
        "--root",
        str(training_root),
        "--bank",
        str(bank),
        "--output",
        str(output),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--retention-mode",
        "hybrid_rp",
        "--sweep-cell",
        cell_id,
        "--device",
        "cuda:0",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = physical_gpu
    print(f"launching {cell_id} on physical GPU {physical_gpu}", flush=True)
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    result = {
        "cell_id": cell_id,
        "physical_gpu": physical_gpu,
        "returncode": int(process.returncode),
        "elapsed_seconds": float(time.monotonic() - started),
        "log": str(log_path),
    }
    print(f"finished {cell_id}: {result}", flush=True)
    return result


def _analysis_metrics(summary_path: Path) -> dict[str, Any]:
    if not summary_path.is_file():
        return {"analysis_complete_seed_count": 0, "analysis_missing": True}
    summary = _read_json(summary_path)
    seeds = [
        value
        for value in summary.get("seeds", {}).values()
        if value.get("analysis_status") == "complete_extended_structural_analysis"
    ]
    recovery = [
        _nested(
            seed,
            "normal_recovery",
            "in_plane_radial",
            "0.05",
            "4096",
            "matched_clean_state_ratio",
            "median",
        )
        for seed in seeds
    ]
    return {
        "analysis_complete_seed_count": len(seeds),
        "memory_error_2048_per_seed": [
            _nested(seed, "finite_time_angular_memory", "terminal_mean_error_radians")
            for seed in seeds
        ],
        "tangent_gain_per_seed": [
            _nested(seed, "jacobian", "tangent_one_step_gain", "mean")
            for seed in seeds
        ],
        "radial_gain_per_seed": [
            _nested(seed, "jacobian", "in_plane_radial_one_step_gain", "mean")
            for seed in seeds
        ],
        "tangent_minus_radial_per_seed": [
            _nested(seed, "jacobian", "tangent_minus_radial_gain", "mean")
            for seed in seeds
        ],
        "normal_recovery_4096_per_seed": recovery,
        "uniform_flow_per_seed": [
            _nested(seed, "projected_flow", "uniform_norm") for seed in seeds
        ],
        "effective_basin_count_per_seed": [
            _nested(seed, "asymptotic_structure", "effective_basin_count")
            for seed in seeds
        ],
    }


def _aggregate(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for source, destination in (
        ("task_nmse_per_seed", "task_nmse_median"),
        ("task_rmse_per_seed", "task_rmse_median"),
        ("blank_error_per_seed", "blank_error_median"),
        ("memory_error_2048_per_seed", "memory_error_2048_median"),
        ("tangent_gain_per_seed", "tangent_gain_median"),
        ("radial_gain_per_seed", "radial_gain_median"),
        ("tangent_minus_radial_per_seed", "tangent_minus_radial_median"),
        ("normal_recovery_4096_per_seed", "normal_recovery_4096_median"),
        ("uniform_flow_per_seed", "uniform_flow_median"),
        ("effective_basin_count_per_seed", "effective_basin_count_median"),
    ):
        result[destination] = _median(result.get(source, []))
        result[destination.replace("_median", "_mean")] = _mean(
            result.get(source, [])
        )
    tangent = result.get("tangent_gain_median")
    result["tangent_neutrality_error"] = (
        abs(float(tangent) - 1.0) if tangent is not None else None
    )
    return result


def _rank_rows(rows: dict[str, dict[str, Any]], seeds: tuple[int, ...]) -> list[str]:
    eligible = [
        cell_id
        for cell_id, row in rows.items()
        if row.get("eligible_for_full_analysis")
        and row.get("analysis_complete_seed_count") == len(seeds)
    ]
    metrics = (
        "task_nmse_median",
        "blank_error_median",
        "memory_error_2048_median",
        "tangent_neutrality_error",
        "normal_recovery_4096_median",
        "uniform_flow_median",
    )
    scores = {cell_id: 0 for cell_id in eligible}
    for metric in metrics:
        def metric_key(cell_id: str) -> tuple[bool, float]:
            value = rows[cell_id].get(metric)
            return value is None, math.inf if value is None else float(value)

        ordered = sorted(
            eligible,
            key=metric_key,
        )
        for rank, cell_id in enumerate(ordered, start=1):
            scores[cell_id] += rank
            rows[cell_id].setdefault("diagnostic_ranks", {})[metric] = rank
    for cell_id, score in scores.items():
        rows[cell_id]["diagnostic_rank_sum"] = score
    return sorted(eligible, key=lambda cell_id: (scores[cell_id], cell_id))


def _format(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_markdown(
    path: Path,
    *,
    rows: dict[str, dict[str, Any]],
    order: list[str],
) -> None:
    ordered_ids = order + sorted(set(rows) - set(order))
    lines = [
        "# H-C local-grid CA comparison",
        "",
        "The rank sum is a diagnostic ordering, not an automatic scientific decision. "
        "Only cells with all model seeds blank-stable and all full analyses complete are ranked.",
        "",
        "| Cell | Stable | Task NMSE | Blank error | H=2048 memory | Tangent gain | Radial gain | H=4096 recovery | Uniform flow | Rank sum |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell_id in ordered_ids:
        row = rows[cell_id]
        lines.append(
            "| "
            + " | ".join(
                (
                    cell_id,
                    f"{row.get('blank_stable_seed_count', 0)}/{row.get('seed_count', 0)}",
                    _format(row.get("task_nmse_median")),
                    _format(row.get("blank_error_median")),
                    _format(row.get("memory_error_2048_median")),
                    _format(row.get("tangent_gain_median")),
                    _format(row.get("radial_gain_median")),
                    _format(row.get("normal_recovery_4096_median")),
                    _format(row.get("uniform_flow_median")),
                    _format(row.get("diagnostic_rank_sum")),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "Selection should prioritize: all-seed stability, near-neutral tangent dynamics, "
            "finite normal recovery, bounded finite-time memory drift, and acceptable task error. "
            "Effective basin count is reported descriptively and is not treated as a monotone score.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--reference-screen-summary", type=Path)
    parser.add_argument("--reference-summary-condition", default="a0p05_negbias")
    parser.add_argument("--reference-analysis-root", type=Path)
    args = parser.parse_args()

    training_root = args.training_root.expanduser().resolve(strict=True)
    config_path = args.config.expanduser().resolve(strict=True)
    bank = args.bank.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = _read_json(config_path)
    cells = list(config["sweep_cells"])
    seeds = tuple(int(seed) for seed in config["model_seeds"])
    expected = len(cells) * len(seeds)
    _wait_for_results(
        training_root,
        expected=expected,
        poll_seconds=args.poll_seconds,
    )

    rows = _screen_rows(training_root, cells, seeds)
    reference_cell = dict(config["reference_condition"])
    reference_id = str(reference_cell["id"])
    if args.reference_screen_summary:
        rows[reference_id] = _reference_screen_row(
            args.reference_screen_summary.expanduser().resolve(strict=True),
            summary_condition=args.reference_summary_condition,
            cell=reference_cell,
            seeds=seeds,
        )

    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not gpus:
        raise ValueError("at least one GPU must be provided")
    eligible = [
        cell_id
        for cell_id, row in rows.items()
        if row["eligible_for_full_analysis"] and cell_id != reference_id
    ]
    launch_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(
                _run_analysis,
                training_root=training_root,
                bank=bank,
                output=output_root / cell_id,
                cell_id=cell_id,
                seeds=seeds,
                physical_gpu=gpus[index % len(gpus)],
            ): cell_id
            for index, cell_id in enumerate(eligible)
        }
        for future in as_completed(futures):
            launch_results.append(future.result())

    for cell_id, row in rows.items():
        if cell_id == reference_id and args.reference_analysis_root:
            summary_path = (
                args.reference_analysis_root.expanduser().resolve(strict=True)
                / "summary.json"
            )
        else:
            summary_path = output_root / cell_id / "summary.json"
        row.update(_analysis_metrics(summary_path))
        rows[cell_id] = _aggregate(row)

    order = _rank_rows(rows, seeds)
    payload = {
        "schema_version": 1,
        "training_root": str(training_root),
        "config": str(config_path),
        "bank": str(bank),
        "selection_policy": {
            "eligibility": "all registered model seeds blank-stable and full CA analysis complete",
            "ordering": "equal-weight rank sum over six diagnostic metrics; manual review required",
            "metrics": [
                "task_nmse_median",
                "blank_error_median",
                "memory_error_2048_median",
                "tangent_neutrality_error",
                "normal_recovery_4096_median",
                "uniform_flow_median",
            ],
        },
        "diagnostic_order": order,
        "analysis_launches": launch_results,
        "cells": rows,
    }
    atomic_json(output_root / "comparison.json", payload)
    _write_markdown(output_root / "comparison.md", rows=rows, order=order)
    print(f"diagnostic order: {order}", flush=True)
    print(f"comparison: {output_root / 'comparison.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
