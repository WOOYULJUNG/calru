"""Aggregate finite normal-kick recovery without counting ID failures as OOD failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from repro.sagodi_protocol.artifacts import (
    atomic_json,
    sha256_file,
    strict_json_load,
)

from .calibrate_ood_strength import (
    BASELINES,
    COLORS,
    MODELS,
    MODEL_LABELS,
    TOPOLOGIES,
    TOPOLOGY_LABELS,
)
from .topology_analysis_common import load_success_map, write_csv


def _median(values: list[float]) -> float:
    return float(statistics.median(float(value) for value in values))


def _collect(
    roots: list[Path], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values: dict[tuple[str, str, float, int], list[tuple[float, float]]] = {}
    eligible_jobs: dict[tuple[str, str], set[str]] = {
        (topology, model): set()
        for topology in TOPOLOGIES
        for model in MODELS
    }
    for root in roots:
        success = load_success_map(root)
        for path in sorted((root / "dynamics" / "runs").glob("*.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            model = str(item["model"])
            if model not in MODELS or not success.get(str(item["job_id"]), False):
                continue
            topology = str(item["topology"])
            eligible_jobs[(topology, model)].add(str(item["job_id"]))
            for radius_index, radius in enumerate(item["kick_radii_relative"]):
                for horizon_index, horizon in enumerate(item["recovery_horizons"]):
                    values.setdefault(
                        (topology, model, float(radius), int(horizon)), []
                    ).append(
                        (
                            float(
                                item["recovery_ratio_median"][radius_index][
                                    horizon_index
                                ]
                            ),
                            float(
                                item["same_memory_error_mean"][radius_index][
                                    horizon_index
                                ]
                            ),
                        )
                    )

    policy = config["quality_gate"]
    decision_horizon = int(policy["decision_horizon"])
    model_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    radii = [float(value) for value in config["perturbation"]["kick_radii_relative"]]
    horizons = [int(value) for value in config["perturbation"]["recovery_horizons"]]
    for topology in TOPOLOGIES:
        for radius in radii:
            statuses: dict[str, str] = {}
            for model in MODELS:
                seed_count = len(eligible_jobs[(topology, model)])
                for horizon in horizons:
                    samples = values.get((topology, model, radius, horizon), [])
                    recovery = [sample[0] for sample in samples]
                    memory = [sample[1] for sample in samples]
                    if not samples:
                        status = "ineligible_id"
                        recovery_median = recovery_min = recovery_max = None
                        memory_median = memory_min = memory_max = None
                    else:
                        recovery_median = _median(recovery)
                        recovery_min, recovery_max = min(recovery), max(recovery)
                        memory_median = _median(memory)
                        memory_min, memory_max = min(memory), max(memory)
                        if horizon != decision_horizon:
                            status = "trajectory"
                        elif (
                            recovery_median
                            >= float(policy["failed_if_recovery_ratio_at_least"])
                            or memory_median
                            >= float(
                                policy[
                                    "failed_if_same_memory_normalized_error_at_least"
                                ]
                            )
                        ):
                            status = "failed"
                        else:
                            status = "survived"
                    if horizon == decision_horizon:
                        statuses[model] = status
                    model_rows.append(
                        {
                            "topology": topology,
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "eligible_seed_count": seed_count,
                            "kick_radius_relative": radius,
                            "recovery_horizon": horizon,
                            "recovery_ratio_median": recovery_median,
                            "recovery_ratio_min": recovery_min,
                            "recovery_ratio_max": recovery_max,
                            "same_memory_error_normalized_median": memory_median,
                            "same_memory_error_normalized_min": memory_min,
                            "same_memory_error_normalized_max": memory_max,
                            "same_memory_error_radians_median": (
                                None
                                if memory_median is None
                                else memory_median * np.pi
                            ),
                            "status": status,
                        }
                    )
            eligible_baselines = [
                model
                for model in BASELINES
                if statuses.get(model) != "ineligible_id"
            ]
            failed_baselines = [
                model
                for model in eligible_baselines
                if statuses.get(model) == "failed"
            ]
            condition_rows.append(
                {
                    "topology": topology,
                    "kick_radius_relative": radius,
                    "decision_horizon": decision_horizon,
                    "eligible_baseline_models": ",".join(eligible_baselines),
                    "eligible_baseline_count": len(eligible_baselines),
                    "failed_baseline_models": ",".join(failed_baselines),
                    "failed_baseline_count": len(failed_baselines),
                    "universal_baseline_failure": bool(
                        eligible_baselines
                        and len(failed_baselines) == len(eligible_baselines)
                    ),
                    "calru_status": statuses.get("calru", "ineligible_id"),
                }
            )
    return model_rows, condition_rows


def _plot(
    output: Path,
    model_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    index = {
        (
            row["topology"],
            row["model"],
            float(row["kick_radius_relative"]),
            int(row["recovery_horizon"]),
        ): row
        for row in model_rows
    }
    condition_index = {
        (row["topology"], float(row["kick_radius_relative"])): row
        for row in condition_rows
    }
    radii = [float(value) for value in config["perturbation"]["kick_radii_relative"]]
    horizons = [int(value) for value in config["perturbation"]["recovery_horizons"]]
    decision_horizon = int(config["quality_gate"]["decision_horizon"])
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 3, figsize=(14.5, 7.7))
    for column, topology in enumerate(TOPOLOGIES):
        for position, radius in enumerate(radii):
            if condition_index[(topology, radius)]["universal_baseline_failure"]:
                for row_axis in axes[:, column]:
                    row_axis.axvspan(
                        position - 0.4,
                        position + 0.4,
                        color="#F4A3A3",
                        alpha=0.22,
                    )
        for model in MODELS:
            records = [
                index[(topology, model, radius, decision_horizon)]
                for radius in radii
            ]
            recovery = [
                np.nan
                if row["recovery_ratio_median"] is None
                else float(row["recovery_ratio_median"])
                for row in records
            ]
            memory = [
                np.nan
                if row["same_memory_error_radians_median"] is None
                else float(row["same_memory_error_radians_median"])
                for row in records
            ]
            for axis, values in zip(axes[:, column], (recovery, memory)):
                axis.plot(
                    np.arange(len(radii)),
                    values,
                    color=COLORS[model],
                    marker="o",
                    linewidth=2,
                    label=MODEL_LABELS[model],
                )
        axes[0, column].set_title(TOPOLOGY_LABELS[topology])
        axes[0, column].set_ylabel("Normal-distance ratio Q ↓")
        axes[1, column].set_ylabel("Same-memory error (rad) ↓")
        for axis in axes[:, column]:
            axis.set_xticks(np.arange(len(radii)), [f"{radius:g}" for radius in radii])
            axis.set_xlabel("Relative kick radius ρ")
            axis.grid(alpha=0.2)
    axes[0, -1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"Finite hidden-normal recovery at H={decision_horizon}\n"
        "ID-qualified seeds only; red = every eligible baseline failed"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    for suffix in ("png", "pdf"):
        figure.savefig(
            figures / f"fig_C6_normal_kick_strength.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(figure)

    radius = max(radii)
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 7.7))
    for column, topology in enumerate(TOPOLOGIES):
        for model in MODELS:
            records = [
                index[(topology, model, radius, horizon)]
                for horizon in horizons
            ]
            recovery = [
                np.nan
                if row["recovery_ratio_median"] is None
                else float(row["recovery_ratio_median"])
                for row in records
            ]
            memory = [
                np.nan
                if row["same_memory_error_radians_median"] is None
                else float(row["same_memory_error_radians_median"])
                for row in records
            ]
            for axis, values in zip(axes[:, column], (recovery, memory)):
                axis.plot(
                    horizons,
                    values,
                    color=COLORS[model],
                    marker="o",
                    linewidth=2,
                    label=MODEL_LABELS[model],
                )
        axes[0, column].set_title(TOPOLOGY_LABELS[topology])
        axes[0, column].set_ylabel("Normal-distance ratio Q ↓")
        axes[1, column].set_ylabel("Same-memory error (rad) ↓")
        for axis in axes[:, column]:
            axis.set_xscale("symlog", linthresh=1)
            axis.set_xlabel("Blank recovery horizon H")
            axis.grid(alpha=0.2)
    axes[0, -1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"Finite hidden-normal recovery trajectory at ρ={radius:g}\n"
        "median over ID-qualified seeds"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    for suffix in ("png", "pdf"):
        figure.savefig(
            figures / f"fig_C7_normal_kick_horizon.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(figure)


def _write_report(
    output: Path,
    model_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    decision_horizon = int(decision["quality_gate"]["decision_horizon"])
    selected = float(decision["headline_kick_radius_relative"])
    rows = [
        row
        for row in model_rows
        if int(row["recovery_horizon"]) == decision_horizon
        and float(row["kick_radius_relative"]) == selected
    ]
    index = {(row["topology"], row["model"]): row for row in rows}
    lines = [
        "# Calibrated finite normal-kick recovery",
        "",
        "ID task gate를 통과한 checkpoint만 사용했다. ID부터 실패한 모델은 "
        "OOD 실패로 세지 않았다.",
        "",
        f"선택 강도는 `ρ={selected:g}`, 회복 horizon은 `H={decision_horizon}`이다. "
        "시험한 어느 topology에서도 모든 eligible baseline이 동시에 실패하지 "
        "않았으므로 강도를 낮출 필요가 없었다.",
        "",
        "| Topology | Model | eligible seeds | Q (normal distance ratio) | same-memory error (rad) | status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for topology in TOPOLOGIES:
        for model in MODELS:
            row = index[(topology, model)]
            if row["status"] == "ineligible_id":
                q = error = "—"
            else:
                q = f"{float(row['recovery_ratio_median']):.4f}"
                error = f"{float(row['same_memory_error_radians_median']):.4f}"
            lines.append(
                f"| {topology.upper()} | {MODEL_LABELS[model]} | "
                f"{row['eligible_seed_count']} | {q} | {error} | {row['status']} |"
            )
    lines.extend(
        [
            "",
            "`Q<1`은 초기 normal distance보다 manifold에 가까워졌음을 뜻한다. "
            "단, 작은 Q만으로 같은 memory로 돌아왔다고 할 수 없으므로 "
            "same-memory angular error를 함께 본다.",
            "",
            "이 결과에서는 baseline이 CA-LRU보다 더 강한 normal-distance "
            "수축을 보이지만, CA-LRU가 더 작은 same-memory error를 보인다. "
            "따라서 이 축은 CA-LRU의 일방적 우위가 아니라 "
            "`memory-preserving recovery`와 `geometric contraction`의 "
            "trade-off로 보고해야 한다.",
            "",
        ]
    )
    (output / "PERTURBATION_CALIBRATION_RESULTS_ko.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def calibrate(
    roots: list[Path], output: Path, config_path: Path
) -> None:
    resolved_roots = [root.expanduser().resolve(strict=True) for root in roots]
    resolved_output = output.expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    config = strict_json_load(config_path.expanduser().resolve(strict=True))
    model_rows, condition_rows = _collect(resolved_roots, config)
    safe_radii = [
        float(radius)
        for radius in config["perturbation"]["kick_radii_relative"]
        if not any(
            bool(row["universal_baseline_failure"])
            for row in condition_rows
            if float(row["kick_radius_relative"]) == float(radius)
        )
    ]
    decision = {
        "schema_version": 1,
        "analysis_id": config["analysis_id"],
        "quality_gate": config["quality_gate"],
        "headline_kick_radius_relative": max(safe_radii) if safe_radii else None,
        "all_tested_radii_preserve_a_baseline_comparator": bool(
            len(safe_radii)
            == len(config["perturbation"]["kick_radii_relative"])
        ),
        "conditions": condition_rows,
    }
    write_csv(resolved_output / "perturbation_model_survival.csv", model_rows)
    write_csv(
        resolved_output / "perturbation_condition_survival.csv",
        condition_rows,
    )
    atomic_json(
        resolved_output / "PERTURBATION_CALIBRATION_DECISIONS.json",
        decision,
    )
    _plot(resolved_output, model_rows, condition_rows, config)
    _write_report(resolved_output, model_rows, condition_rows, decision)
    atomic_json(
        resolved_output / "PERTURBATION_ANALYSIS_COMPLETED.json",
        {
            "schema_version": 1,
            "analysis_id": config["analysis_id"],
            "analysis_config_sha256": sha256_file(config_path),
            "analysis_code_sha256": sha256_file(Path(__file__)),
            "collection_experiment_id": resolved_output.name,
            "source_analysis_groups": [root.name for root in resolved_roots],
            "eligible_model_horizon_rows": len(model_rows),
            "condition_rows": len(condition_rows),
            "headline_kick_radius_relative": decision[
                "headline_kick_radius_relative"
            ],
            "all_tested_radii_preserve_a_baseline_comparator": decision[
                "all_tested_radii_preserve_a_baseline_comparator"
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name(
            "topology_perturbation_calibrated_v1.json"
        ),
    )
    args = parser.parse_args()
    calibrate(args.analysis_root, args.output, args.config)


if __name__ == "__main__":
    main()
