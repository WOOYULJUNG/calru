"""Calibrate OOD strength using only checkpoints that pass the frozen ID task."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from repro.sagodi_protocol.artifacts import atomic_json, strict_json_load

from .topology_analysis_common import write_csv


MODELS = ("rnn", "gru", "lstm", "calru")
BASELINES = ("rnn", "gru", "lstm")
TOPOLOGIES = ("s1", "t2", "s2")
MODEL_LABELS = {
    "rnn": "RNN",
    "gru": "GRU",
    "lstm": "LSTM",
    "calru": "CA-LRU",
}
TOPOLOGY_LABELS = {"s1": r"$S^1$", "t2": r"$T^2$", "s2": r"$S^2$"}
COLORS = {
    "rnn": "#6C757D",
    "gru": "#E9C46A",
    "lstm": "#E76F51",
    "calru": "#2A9D8F",
}


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _median(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot take median of an empty sequence")
    return float(statistics.median(materialized))


def _conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    result = [dict(row) for row in config["conditions"]]
    for horizon in config["post_blank_horizons_by_condition"]["id_h128"]:
        result.append(
            {
                "id": f"blank_h{int(horizon)}",
                "axis": "blank",
                "value": int(horizon),
                "label": f"H={int(horizon)}",
                "horizon": int(horizon),
            }
        )
    return result


def _metric_for_condition(
    condition: dict[str, Any],
) -> str:
    if str(condition["axis"]) == "blank":
        return (
            f"post_blank_{int(condition['horizon'])}_"
            "mean_error_radians"
        )
    return "terminal_mean_error_radians"


def _classify(
    *,
    finite_fraction: float,
    chance_ratio: float,
    id_error_ratio: float,
    policy: dict[str, Any],
) -> str:
    if finite_fraction < 0.5:
        return "chance_collapsed"
    if (
        chance_ratio
        >= float(policy["collapsed_if_chance_ratio_at_least"])
    ):
        return "chance_collapsed"
    if id_error_ratio > float(policy["collapsed_if_id_error_ratio_above"]):
        return "severely_degraded"
    if (
        chance_ratio
        >= float(policy["transition_if_chance_ratio_at_least"])
        or id_error_ratio
        > float(policy["transition_if_id_error_ratio_above"])
    ):
        return "transition"
    return "alive"


def _condition_statistics(
    rows: list[dict[str, str]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy = config["calibration_policy"]
    by_job_condition = {
        (row["job_id"], row["condition_id"]): row for row in rows
    }
    qualified_jobs: dict[tuple[str, str], list[str]] = {}
    for topology in TOPOLOGIES:
        for model in MODELS:
            qualified_jobs[(topology, model)] = sorted(
                {
                    row["job_id"]
                    for row in rows
                    if row["topology"] == topology
                    and row["model"] == model
                    and _bool(row["task_success"])
                }
            )

    model_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    for condition in _conditions(config):
        condition_id = str(condition["id"])
        source_condition = (
            "id_h128"
            if str(condition["axis"]) == "blank"
            else condition_id
        )
        metric = _metric_for_condition(condition)
        for topology in TOPOLOGIES:
            statuses: dict[str, str] = {}
            for model in MODELS:
                jobs = qualified_jobs[(topology, model)]
                if not jobs:
                    statuses[model] = "ineligible_id"
                    model_rows.append(
                        {
                            "topology": topology,
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "condition_id": condition_id,
                            "condition_axis": condition["axis"],
                            "condition_value": condition["value"],
                            "condition_label": condition["label"],
                            "eligible_seed_count": 0,
                            "finite_seed_count": 0,
                            "error_median_radians": None,
                            "error_min_radians": None,
                            "error_max_radians": None,
                            "chance_ratio_median": None,
                            "id_error_ratio_median": None,
                            "status": "ineligible_id",
                        }
                    )
                    continue

                errors = []
                chance_ratios = []
                id_ratios = []
                for job_id in jobs:
                    row = by_job_condition[(job_id, source_condition)]
                    id_row = by_job_condition[(job_id, "id_h128")]
                    error = _number(row.get(metric))
                    id_error = _number(
                        id_row.get("terminal_mean_error_radians")
                    )
                    if error is None or id_error is None:
                        continue
                    if str(condition["axis"]) == "blank":
                        chance_reference = np.pi / 2.0
                    else:
                        chance_reference = _number(
                            row.get("hold_baseline_mean_error_radians")
                        )
                    if chance_reference is None or chance_reference <= 0.0:
                        continue
                    errors.append(error)
                    chance_ratios.append(error / chance_reference)
                    id_ratios.append(error / max(id_error, 1e-8))

                finite_fraction = len(errors) / len(jobs)
                if errors:
                    chance_ratio = _median(chance_ratios)
                    id_ratio = _median(id_ratios)
                    status = _classify(
                        finite_fraction=finite_fraction,
                        chance_ratio=chance_ratio,
                        id_error_ratio=id_ratio,
                        policy=policy,
                    )
                    error_median = _median(errors)
                    error_min = min(errors)
                    error_max = max(errors)
                else:
                    chance_ratio = float("inf")
                    id_ratio = float("inf")
                    status = "chance_collapsed"
                    error_median = None
                    error_min = None
                    error_max = None
                statuses[model] = status
                model_rows.append(
                    {
                        "topology": topology,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "condition_id": condition_id,
                        "condition_axis": condition["axis"],
                        "condition_value": condition["value"],
                        "condition_label": condition["label"],
                        "eligible_seed_count": len(jobs),
                        "finite_seed_count": len(errors),
                        "error_median_radians": error_median,
                        "error_min_radians": error_min,
                        "error_max_radians": error_max,
                        "chance_ratio_median": chance_ratio,
                        "id_error_ratio_median": id_ratio,
                        "status": status,
                    }
                )

            eligible_baselines = [
                model
                for model in BASELINES
                if statuses[model] != "ineligible_id"
            ]
            collapsed_baselines = [
                model
                for model in eligible_baselines
                if statuses[model]
                in {"chance_collapsed", "severely_degraded"}
            ]
            chance_collapsed_baselines = [
                model
                for model in eligible_baselines
                if statuses[model] == "chance_collapsed"
            ]
            severely_degraded_baselines = [
                model
                for model in eligible_baselines
                if statuses[model] == "severely_degraded"
            ]
            condition_rows.append(
                {
                    "topology": topology,
                    "condition_id": condition_id,
                    "condition_axis": condition["axis"],
                    "condition_value": condition["value"],
                    "condition_label": condition["label"],
                    "eligible_baseline_models": ",".join(eligible_baselines),
                    "eligible_baseline_count": len(eligible_baselines),
                    "alive_baseline_count": sum(
                        statuses[model] == "alive"
                        for model in eligible_baselines
                    ),
                    "transition_baseline_count": sum(
                        statuses[model] == "transition"
                        for model in eligible_baselines
                    ),
                    "collapsed_baseline_count": len(collapsed_baselines),
                    "collapsed_baseline_models": ",".join(
                        collapsed_baselines
                    ),
                    "chance_collapsed_baseline_count": len(
                        chance_collapsed_baselines
                    ),
                    "chance_collapsed_baseline_models": ",".join(
                        chance_collapsed_baselines
                    ),
                    "severely_degraded_baseline_count": len(
                        severely_degraded_baselines
                    ),
                    "severely_degraded_baseline_models": ",".join(
                        severely_degraded_baselines
                    ),
                    "universal_baseline_failure": bool(
                        eligible_baselines
                        and len(collapsed_baselines)
                        == len(eligible_baselines)
                    ),
                    "universal_baseline_chance_collapse": bool(
                        eligible_baselines
                        and len(chance_collapsed_baselines)
                        == len(eligible_baselines)
                    ),
                    "calru_status": statuses["calru"],
                }
            )
    return model_rows, condition_rows


def _branch_conditions(
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    conditions = _conditions(config)
    by_axis: dict[str, list[dict[str, Any]]] = {}
    for condition in conditions:
        by_axis.setdefault(str(condition["axis"]), []).append(condition)
    return {
        "length": sorted(
            by_axis["length"], key=lambda row: float(row["value"])
        ),
        "velocity_low": sorted(
            [
                row
                for row in by_axis["velocity"]
                if float(row["value"]) < 1.0
            ],
            key=lambda row: float(row["value"]),
            reverse=True,
        ),
        "velocity_high": sorted(
            [
                row
                for row in by_axis["velocity"]
                if float(row["value"]) > 1.0
            ],
            key=lambda row: float(row["value"]),
        ),
        "correlation_low": sorted(
            [
                row
                for row in by_axis["smoothness"]
                if float(row["value"]) < 1.0
            ],
            key=lambda row: float(row["value"]),
            reverse=True,
        ),
        "correlation_high": sorted(
            [
                row
                for row in by_axis["smoothness"]
                if float(row["value"]) > 1.0
            ],
            key=lambda row: float(row["value"]),
        ),
        "dwell": sorted(
            by_axis["dwell"], key=lambda row: float(row["value"])
        ),
        "blank": sorted(
            [
                row
                for row in by_axis["blank"]
                if int(row["value"]) > 0
            ],
            key=lambda row: int(row["value"]),
        ),
    }


def _decisions(
    config: dict[str, Any],
    condition_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    index = {
        (row["topology"], row["condition_id"]): row
        for row in condition_rows
    }
    branches: dict[str, Any] = {}
    for branch, conditions in _branch_conditions(config).items():
        selected: dict[str, Any] | None = None
        stress: list[str] = []
        boundary_seen = False
        details = []
        for condition in conditions:
            failed_topologies = [
                topology
                for topology in TOPOLOGIES
                if bool(
                    index[
                        (topology, str(condition["id"]))
                    ]["universal_baseline_failure"]
                )
            ]
            shared_safe = not failed_topologies
            if not boundary_seen and shared_safe:
                selected = condition
            else:
                boundary_seen = True
                stress.append(str(condition["id"]))
            details.append(
                {
                    "condition_id": str(condition["id"]),
                    "shared_baseline_survival": shared_safe,
                    "universal_failure_topologies": failed_topologies,
                }
            )
        branches[branch] = {
            "headline_condition_id": (
                None if selected is None else str(selected["id"])
            ),
            "headline_label": (
                None if selected is None else str(selected["label"])
            ),
            "stress_condition_ids": stress,
            "conditions": details,
        }
    return {
        "schema_version": 1,
        "analysis_id": config["analysis_id"],
        "policy": config["calibration_policy"],
        "branches": branches,
    }


def _plot_conditions(
    model_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    *,
    output: Path,
    stem: str,
    title: str,
) -> None:
    model_index = {
        (row["topology"], row["model"], row["condition_id"]): row
        for row in model_rows
    }
    condition_index = {
        (row["topology"], row["condition_id"]): row
        for row in condition_rows
    }
    x = np.arange(len(conditions))
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=True)
    for axis, topology in zip(axes, TOPOLOGIES):
        for position, condition in enumerate(conditions):
            if bool(
                condition_index[
                    (topology, str(condition["id"]))
                ]["universal_baseline_failure"]
            ):
                axis.axvspan(
                    position - 0.45,
                    position + 0.45,
                    color="#F4A3A3",
                    alpha=0.22,
                    zorder=0,
                )
        for model in MODELS:
            records = [
                model_index[(topology, model, str(condition["id"]))]
                for condition in conditions
            ]
            values = [
                np.nan
                if row["error_median_radians"] is None
                else float(row["error_median_radians"])
                for row in records
            ]
            lower = [
                np.nan
                if row["error_min_radians"] is None
                else float(row["error_min_radians"])
                for row in records
            ]
            upper = [
                np.nan
                if row["error_max_radians"] is None
                else float(row["error_max_radians"])
                for row in records
            ]
            axis.plot(
                x,
                np.maximum(values, 1e-5),
                color=COLORS[model],
                marker="o",
                linewidth=2,
                label=MODEL_LABELS[model],
            )
            if np.isfinite(lower).any():
                axis.fill_between(
                    x,
                    np.maximum(lower, 1e-5),
                    np.maximum(upper, 1e-5),
                    color=COLORS[model],
                    alpha=0.10,
                )
        axis.set_yscale("log")
        axis.set_xticks(
            x,
            [str(condition["label"]) for condition in conditions],
            rotation=30,
        )
        eligible = [
            MODEL_LABELS[model]
            for model in BASELINES
            if model_index[
                (topology, model, str(conditions[0]["id"]))
            ]["status"]
            != "ineligible_id"
        ]
        axis.set_title(
            f"{TOPOLOGY_LABELS[topology]}\n"
            f"ID-qualified baselines: {', '.join(eligible)}"
        )
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Intrinsic error (rad) ↓")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        title
        + "\nmedian and range over ID-qualified seeds; red = all eligible baselines failed the OOD quality gate"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            figures / f"{stem}.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(figure)


def _write_report(
    output: Path,
    config: dict[str, Any],
    model_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    decisions: dict[str, Any],
) -> None:
    eligible = {
        topology: {
            model: max(
                int(row["eligible_seed_count"])
                for row in model_rows
                if row["topology"] == topology and row["model"] == model
            )
            for model in MODELS
        }
        for topology in TOPOLOGIES
    }
    failures = [
        row
        for row in condition_rows
        if bool(row["universal_baseline_failure"])
    ]
    lines = [
        "# Calibrated topology OOD — exploratory strength selection",
        "",
        "이 보고서는 OOD 강도를 고르는 calibration 결과다. ID task gate를 "
        "통과한 checkpoint만 모델별 집계에 포함하며, confirmatory seed를 "
        "추가하기 전의 탐색 결과로 취급한다.",
        "",
        "## ID-qualified seed 수",
        "",
        "| Topology | RNN | GRU | LSTM | CA-LRU |",
        "|---|---:|---:|---:|---:|",
    ]
    for topology in TOPOLOGIES:
        lines.append(
            f"| {topology.upper()} | "
            + " | ".join(
                str(eligible[topology][model]) for model in MODELS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 강도 판정",
            "",
            "`chance_collapsed`는 ID-qualified seed median이 "
            "no-update/chance 기준의 75% 이상인 경우다. "
            "`severely_degraded`는 chance보다 낫지만 자기 ID error의 "
            "8배를 초과한 경우다. `transition`은 각각 50% 또는 4배를 "
            "넘지만 두 failure 기준에는 도달하지 않은 경우다.",
            "",
            "| Branch | Headline maximum | Stress-only conditions |",
            "|---|---:|---|",
        ]
    )
    for branch, row in decisions["branches"].items():
        lines.append(
            f"| {branch} | {row['headline_label'] or 'none'} | "
            f"{', '.join(row['stress_condition_ids']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## 모든 eligible baseline이 무너진 조건",
            "",
        ]
    )
    if failures:
        for row in failures:
            reasons = []
            if row["chance_collapsed_baseline_models"]:
                reasons.append(
                    "chance="
                    + str(row["chance_collapsed_baseline_models"])
                )
            if row["severely_degraded_baseline_models"]:
                reasons.append(
                    ">8x ID="
                    + str(row["severely_degraded_baseline_models"])
                )
            lines.append(
                f"- {row['topology'].upper()} / {row['condition_label']}: "
                f"{'; '.join(reasons)}"
            )
    else:
        lines.append("- 없음")
    lines.extend(
        [
            "",
            "## 해석 규칙",
            "",
            "- headline figure의 최대 강도는 세 topology 모두에서 적어도 "
            "하나의 ID-qualified baseline이 살아 있는 공통 구간으로 제한한다.",
            "- 그보다 강한 조건은 삭제하지 않고 failure-limit stress로 분리한다.",
            "- S2의 RNN·GRU처럼 ID부터 실패한 모델은 OOD failure 분모에서 "
            "제외한다.",
            "- trajectory는 paired evaluation sample이며 통계적 replicate는 "
            "trained-model seed다.",
            "",
        ]
    )
    (output / "CALIBRATION_RESULTS_ko.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def calibrate(
    analysis_root: Path,
    config_path: Path,
) -> None:
    output = analysis_root.expanduser().resolve(strict=True)
    config = strict_json_load(config_path.expanduser().resolve(strict=True))
    with (output / "ood_seed_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    model_rows, condition_rows = _condition_statistics(rows, config)
    decisions = _decisions(config, condition_rows)
    write_csv(output / "ood_model_survival.csv", model_rows)
    write_csv(output / "ood_condition_survival.csv", condition_rows)
    atomic_json(output / "OOD_CALIBRATION_DECISIONS.json", decisions)

    condition_by_id = {
        str(row["id"]): row for row in _conditions(config)
    }
    id_condition = condition_by_id["id_h128"]
    axes = [
        (
            "length",
            [id_condition]
            + sorted(
                [
                    row
                    for row in config["conditions"]
                    if row["axis"] == "length"
                ],
                key=lambda row: float(row["value"]),
            ),
            "fig_C1_length_calibration",
            "Nested-prefix length OOD calibration",
        ),
        (
            "velocity",
            sorted(
                [id_condition]
                + [
                    row
                    for row in config["conditions"]
                    if row["axis"] == "velocity"
                ],
                key=lambda row: (
                    1.0
                    if row["id"] == "id_h128"
                    else float(row["value"])
                ),
            ),
            "fig_C2_velocity_calibration",
            "Paired velocity-scale OOD calibration",
        ),
        (
            "smoothness",
            sorted(
                [id_condition]
                + [
                    row
                    for row in config["conditions"]
                    if row["axis"] == "smoothness"
                ],
                key=lambda row: (
                    1.0
                    if row["id"] == "id_h128"
                    else float(row["value"])
                ),
            ),
            "fig_C3_correlation_calibration",
            "GP temporal-correlation OOD calibration",
        ),
        (
            "dwell",
            [id_condition]
            + sorted(
                [
                    row
                    for row in config["conditions"]
                    if row["axis"] == "dwell"
                ],
                key=lambda row: float(row["value"]),
            ),
            "fig_C4_dwell_calibration",
            "Contiguous zero-velocity dwell calibration",
        ),
        (
            "blank",
            sorted(
                [
                    row
                    for row in _conditions(config)
                    if row["axis"] == "blank"
                ],
                key=lambda row: int(row["value"]),
            ),
            "fig_C5_blank_calibration",
            "Post-trajectory blank-memory calibration",
        ),
    ]
    for _, conditions, stem, title in axes:
        _plot_conditions(
            model_rows,
            condition_rows,
            conditions,
            output=output,
            stem=stem,
            title=title,
        )
    _write_report(
        output,
        config,
        model_rows,
        condition_rows,
        decisions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name(
            "topology_ood_calibrated_v3.json"
        ),
    )
    args = parser.parse_args()
    calibrate(args.analysis_root, args.config)


if __name__ == "__main__":
    main()
