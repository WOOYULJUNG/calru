"""Evaluate paired topology OOD banks on frozen recurrent checkpoints."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import statistics
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from repro.sagodi_protocol.artifacts import atomic_json, strict_json_load

from .artifacts import load_manifold_bank, sha256_file
from .topology_analysis_common import (
    RunRecord,
    atomic_npz,
    discover_completed_runs,
    load_analysis_config,
    load_model,
    normalized_geodesic_errors,
    output_norm_error,
    write_csv,
)


MODELS = ("rnn", "gru", "lstm", "calru")
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


@dataclass(frozen=True)
class Target:
    record: RunRecord
    analysis_root: Path
    selection_regime: str
    task_success: bool
    validation_error: float


def _read_task_rows(root: Path) -> dict[str, dict[str, str]]:
    path = root / "task" / "task_metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["job_id"]: row for row in csv.DictReader(handle)}


def _collect_targets(
    args: argparse.Namespace, config: dict[str, Any]
) -> list[Target]:
    baseline_root = args.baseline_analysis_root.expanduser().resolve(strict=True)
    calru_root = args.calru_analysis_root.expanduser().resolve(strict=True)
    baseline_config = load_analysis_config(args.baseline_config.resolve(strict=True))
    baseline_records, missing = discover_completed_runs(
        args.baseline_run_root.expanduser().resolve(strict=True), baseline_config
    )
    if missing:
        raise RuntimeError(f"missing baseline runs: {missing}")
    calru_config = load_analysis_config(args.calru_config.resolve(strict=True))
    calru_records, missing = discover_completed_runs(
        args.calru_run_root.expanduser().resolve(strict=True), calru_config
    )
    if missing:
        raise RuntimeError(f"missing CA-LRU runs: {missing}")
    baseline_task = _read_task_rows(baseline_root)
    calru_task = _read_task_rows(calru_root)

    targets = []
    for record in baseline_records:
        if record.model_id not in {"rnn", "gru", "lstm"}:
            continue
        row = baseline_task[record.job_id]
        targets.append(
            Target(
                record=record,
                analysis_root=baseline_root,
                selection_regime="ring-selected zero-retuning",
                task_success=str(row["task_success"]).lower() == "true",
                validation_error=float(row["validation_error"]),
            )
        )
    for record in calru_records:
        if record.model_id != "calru":
            continue
        row = calru_task[record.job_id]
        targets.append(
            Target(
                record=record,
                analysis_root=calru_root,
                selection_regime="topology-specific validation-selected",
                task_success=str(row["task_success"]).lower() == "true",
                validation_error=float(row["validation_error"]),
            )
        )
    expected = {
        (model, topology, seed)
        for model in config["models"]
        for topology in config["topologies"]
        for seed in config["seeds"]
    }
    observed = {
        (target.record.model_id, target.record.topology, target.record.seed)
        for target in targets
    }
    if observed != expected or len(targets) != 36:
        raise RuntimeError(
            f"OOD target mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    return sorted(
        targets,
        key=lambda target: (
            TOPOLOGIES.index(target.record.topology),
            MODELS.index(target.record.model_id),
            target.record.seed,
        ),
    )


def _representatives(targets: list[Target]) -> dict[tuple[str, str], Target]:
    selected = {}
    for topology in TOPOLOGIES:
        for model in MODELS:
            all_candidates = [
                target
                for target in targets
                if target.record.topology == topology
                and target.record.model_id == model
            ]
            successful = [
                target for target in all_candidates if target.task_success
            ]
            candidates = successful if successful else all_candidates
            selected[(model, topology)] = min(
                candidates,
                key=lambda target: (
                    target.validation_error,
                    target.record.seed,
                ),
            )
    return selected


def _forward_condition(
    model,
    topology: str,
    bank,
    post_blank_horizons: list[int],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    # Loaded bank arrays are deliberately read-only.  ``torch.tensor`` makes
    # the ownership boundary explicit and avoids writable-view ambiguity.
    inputs = torch.tensor(bank.inputs, device=device)
    initial = torch.tensor(bank.initial_memory, device=device)
    target = torch.tensor(bank.output_targets, device=device)
    state = model.initialize(initial)
    predictions = torch.empty_like(target)
    finite = True
    divergence_step: int | None = None
    with torch.no_grad():
        for step in range(inputs.shape[0]):
            state = model.step(inputs[step], state)
            prediction = model.decode(state)
            if not torch.isfinite(state).all() or not torch.isfinite(prediction).all():
                finite = False
                divergence_step = step
                break
            predictions[step].copy_(prediction)

    batch_size = int(initial.shape[0])
    if not finite:
        penalty = np.full(batch_size, np.pi, dtype=np.float32)
        metrics: dict[str, Any] = {
            "finite": False,
            "divergence_step": divergence_step,
            "sequence_mean_error_radians": float(np.pi),
            "terminal_mean_error_radians": float(np.pi),
            "last_quarter_mean_error_radians": float(np.pi),
            "trial_median_error_radians": float(np.pi),
            "trial_p90_error_radians": float(np.pi),
            "moving_step_mean_error_radians": float(np.pi),
            "dwell_step_mean_error_radians": float(np.pi),
            "component_mse": None,
            "output_norm_absolute_error": None,
            "hold_baseline_mean_error_radians": None,
        }
        arrays = {"terminal_error_radians": penalty}
        for horizon in post_blank_horizons:
            metrics[f"post_blank_{horizon}_mean_error_radians"] = float(np.pi)
            metrics[f"post_blank_{horizon}_median_error_radians"] = float(np.pi)
            arrays[f"post_blank_{horizon}_error_radians"] = penalty.copy()
        return metrics, arrays

    normalized, _ = normalized_geodesic_errors(topology, predictions, target)
    errors = normalized * np.pi
    trial_errors = errors.mean(dim=0)
    quarter = max(1, int(errors.shape[0]) // 4)
    dwell = torch.tensor(
        bank.dwell_mask[..., 0] > 0.5, device=device, dtype=torch.bool
    )
    moving_mean = (
        float(errors[dwell].mean().cpu()) if bool(dwell.any()) else None
    )
    blank_mask = ~dwell
    dwell_mean = (
        float(errors[blank_mask].mean().cpu())
        if bool(blank_mask.any())
        else None
    )
    hold = initial.unsqueeze(0).expand_as(target)
    hold_error, _ = normalized_geodesic_errors(topology, hold, target)
    terminal = errors[-1]
    metrics = {
        "finite": True,
        "divergence_step": None,
        "sequence_mean_error_radians": float(errors.mean().cpu()),
        "terminal_mean_error_radians": float(terminal.mean().cpu()),
        "last_quarter_mean_error_radians": float(errors[-quarter:].mean().cpu()),
        "trial_median_error_radians": float(trial_errors.median().cpu()),
        "trial_p90_error_radians": float(
            torch.quantile(trial_errors, 0.9).cpu()
        ),
        "moving_step_mean_error_radians": moving_mean,
        "dwell_step_mean_error_radians": dwell_mean,
        "component_mse": float((predictions - target).square().mean().cpu()),
        "output_norm_absolute_error": float(
            output_norm_error(topology, predictions).mean().cpu()
        ),
        "hold_baseline_mean_error_radians": float(
            (hold_error * np.pi).mean().cpu()
        ),
    }
    arrays = {
        "terminal_error_radians": terminal.detach().cpu().numpy().astype(np.float32)
    }

    requested = sorted(set(int(value) for value in post_blank_horizons))
    endpoint_target = target[-1]
    blank = torch.zeros(
        batch_size,
        model.input_dim,
        device=device,
        dtype=inputs.dtype,
    )
    current = state
    snapshots: dict[int, torch.Tensor] = {0: predictions[-1]}
    with torch.no_grad():
        for step in range(1, requested[-1] + 1):
            current = model.step(blank, current)
            if step in requested:
                snapshots[step] = model.decode(current)
    for horizon in requested:
        normalized_blank, _ = normalized_geodesic_errors(
            topology, snapshots[horizon], endpoint_target
        )
        blank_error = normalized_blank * np.pi
        metrics[f"post_blank_{horizon}_mean_error_radians"] = float(
            blank_error.mean().cpu()
        )
        metrics[f"post_blank_{horizon}_median_error_radians"] = float(
            blank_error.median().cpu()
        )
        arrays[f"post_blank_{horizon}_error_radians"] = (
            blank_error.detach().cpu().numpy().astype(np.float32)
        )
    return metrics, arrays


def _analyze_target(
    target: Target,
    *,
    config: dict[str, Any],
    config_sha256: str,
    analysis_code_sha256: str,
    bank_root: Path,
    bank_manifest_sha256: str,
    output: Path,
    device: torch.device,
) -> None:
    model, _ = load_model(target.record, device)
    rows = []
    arrays: dict[str, np.ndarray] = {}
    for condition in config["conditions"]:
        condition_id = str(condition["id"])
        path = bank_root / "banks" / (
            f"{condition_id}__{target.record.topology}.npz"
        )
        bank = load_manifold_bank(path)
        metrics, condition_arrays = _forward_condition(
            model,
            target.record.topology,
            bank,
            [int(value) for value in config["post_blank_horizons"]],
            device,
        )
        rows.append(
            {
                "condition_id": condition_id,
                "condition_axis": condition["axis"],
                "condition_value": condition["value"],
                "condition_label": condition["label"],
                "horizon": int(condition["horizon"]),
                "velocity_scale": float(condition.get("velocity_scale", 1.0)),
                "gp_length_scale": float(condition.get("gp_length_scale", 1.0)),
                "dwell_profile": str(
                    condition.get("dwell_profile", "variable_sparsity")
                ),
                **metrics,
            }
        )
        for name, value in condition_arrays.items():
            arrays[f"{condition_id}__{name}"] = value
    payload = {
        "schema_version": 1,
        "analysis_id": config["analysis_id"],
        "analysis_config_sha256": config_sha256,
        "analysis_code_sha256": analysis_code_sha256,
        "bank_manifest_sha256": bank_manifest_sha256,
        "checkpoint_sha256": sha256_file(target.record.checkpoint_path),
        "job_id": target.record.job_id,
        "model": target.record.model_id,
        "topology": target.record.topology,
        "seed": target.record.seed,
        "task_success": target.task_success,
        "validation_error": target.validation_error,
        "selection_regime": target.selection_regime,
        "device": str(device),
        "rows": rows,
    }
    runs = output / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    atomic_json(runs / f"{target.record.job_id}.json", payload)
    atomic_npz(runs / f"{target.record.job_id}.npz", **arrays)


def _flatten(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((output / "runs").glob("*.json")):
        payload = strict_json_load(path)
        for row in payload["rows"]:
            rows.append(
                {
                    "job_id": payload["job_id"],
                    "model": payload["model"],
                    "model_label": MODEL_LABELS[payload["model"]],
                    "topology": payload["topology"],
                    "seed": payload["seed"],
                    "task_success": payload["task_success"],
                    "validation_error": payload["validation_error"],
                    "selection_regime": payload["selection_regime"],
                    **row,
                }
            )
    rows.sort(
        key=lambda row: (
            TOPOLOGIES.index(row["topology"]),
            MODELS.index(row["model"]),
            int(row["seed"]),
            row["condition_id"],
        )
    )
    return rows


def _finite(values: Iterable[Any]) -> list[float]:
    result = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if np.isfinite(number):
            result.append(number)
    return result


def _summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = [
        "sequence_mean_error_radians",
        "terminal_mean_error_radians",
        "last_quarter_mean_error_radians",
        "trial_median_error_radians",
        "trial_p90_error_radians",
        "moving_step_mean_error_radians",
        "dwell_step_mean_error_radians",
        "component_mse",
        "output_norm_absolute_error",
        "hold_baseline_mean_error_radians",
        *[
            f"post_blank_{horizon}_{stat}_error_radians"
            for horizon in config["post_blank_horizons"]
            for stat in ("mean", "median")
        ],
    ]
    output = []
    for topology in TOPOLOGIES:
        for model in MODELS:
            for condition in config["conditions"]:
                group = [
                    row
                    for row in rows
                    if row["topology"] == topology
                    and row["model"] == model
                    and row["condition_id"] == condition["id"]
                ]
                if len(group) != 3:
                    raise RuntimeError(
                        f"expected three OOD seeds for {topology}/{model}/"
                        f"{condition['id']}"
                    )
                item = {
                    "topology": topology,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "condition_id": condition["id"],
                    "condition_axis": condition["axis"],
                    "condition_value": condition["value"],
                    "condition_label": condition["label"],
                    "seed_count": len(group),
                    "task_success_count": sum(
                        bool(row["task_success"]) for row in group
                    ),
                    "finite_count": sum(bool(row["finite"]) for row in group),
                }
                for metric in metrics:
                    values = _finite(row.get(metric) for row in group)
                    item[f"{metric}_median"] = (
                        float(statistics.median(values)) if values else None
                    )
                    item[f"{metric}_min"] = min(values) if values else None
                    item[f"{metric}_max"] = max(values) if values else None
                output.append(item)
    return output


def _save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    root = output / "figures"
    root.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(root / f"{stem}.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(figure)


def _representative_row(
    rows: list[dict[str, Any]],
    representatives: dict[tuple[str, str], Target],
    topology: str,
    model: str,
    condition_id: str,
) -> dict[str, Any]:
    target = representatives[(model, topology)]
    return next(
        row
        for row in rows
        if row["job_id"] == target.record.job_id
        and row["condition_id"] == condition_id
    )


def _plot_axis(
    rows: list[dict[str, Any]],
    representatives: dict[tuple[str, str], Target],
    config: dict[str, Any],
    output: Path,
    *,
    axis_name: str,
    stem: str,
    title: str,
    metric: str = "terminal_mean_error_radians",
) -> None:
    conditions = [config["conditions"][0]] + [
        row for row in config["conditions"] if row["axis"] == axis_name
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), sharey=True)
    x = np.arange(len(conditions))
    for axis, topology in zip(axes, TOPOLOGIES):
        for model in MODELS:
            target = representatives[(model, topology)]
            values = [
                float(
                    _representative_row(
                        rows, representatives, topology, model, condition["id"]
                    )[metric]
                )
                for condition in conditions
            ]
            axis.plot(
                x,
                np.maximum(values, 1e-5),
                marker="o",
                linewidth=2,
                color=COLORS[model],
                linestyle="-" if target.task_success else "--",
                label=MODEL_LABELS[model]
                + ("" if target.task_success else "*"),
            )
        axis.set_yscale("log")
        axis.set_xticks(x, [str(row["label"]) for row in conditions], rotation=30)
        axis.set_title(TOPOLOGY_LABELS[topology])
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Terminal intrinsic error (rad) ↓")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle(title + "\n* best-available task-failed fallback")
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    _save_figure(figure, output, stem)


def _plot_combined(
    rows: list[dict[str, Any]],
    representatives: dict[tuple[str, str], Target],
    config: dict[str, Any],
    output: Path,
) -> None:
    conditions = [config["conditions"][0]] + [
        row for row in config["conditions"] if row["axis"] == "combined"
    ]
    metrics = (
        ("terminal_mean_error_radians", "Terminal error (rad) ↓"),
        ("post_blank_512_mean_error_radians", "After 512 blank steps (rad) ↓"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 7.4), sharex=True)
    x = np.arange(len(conditions))
    for row_index, (metric, ylabel) in enumerate(metrics):
        for column, topology in enumerate(TOPOLOGIES):
            axis = axes[row_index, column]
            for model in MODELS:
                target = representatives[(model, topology)]
                values = [
                    float(
                        _representative_row(
                            rows,
                            representatives,
                            topology,
                            model,
                            condition["id"],
                        )[metric]
                    )
                    for condition in conditions
                ]
                axis.plot(
                    x,
                    np.maximum(values, 1e-5),
                    marker="o",
                    linewidth=2,
                    color=COLORS[model],
                    linestyle="-" if target.task_success else "--",
                    label=MODEL_LABELS[model]
                    + ("" if target.task_success else "*"),
                )
            axis.set_yscale("log")
            axis.grid(alpha=0.22)
            if row_index == 0:
                axis.set_title(TOPOLOGY_LABELS[topology])
            if row_index == 1:
                axis.set_xticks(
                    x,
                    [str(row["label"]) for row in conditions],
                    rotation=25,
                )
            if column == 0:
                axis.set_ylabel(ylabel)
    axes[0, -1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Combined length × velocity stress and post-OOD retention\n"
        "* best-available task-failed fallback"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(figure, output, "fig_O4_combined_and_postblank")


def _write_results_report(
    output: Path,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    representatives: dict[tuple[str, str], Target],
) -> None:
    representative_index = {
        (row["topology"], row["model"], row["condition_id"]): row
        for row in rows
        if row["job_id"]
        == representatives[(row["model"], row["topology"])].record.job_id
    }
    summary_index = {
        (row["topology"], row["model"], row["condition_id"]): row
        for row in summary
    }

    lines = [
        "# Topology OOD generalization v1 — 결과",
        "",
        "이 분석은 학습된 checkpoint를 다시 최적화하지 않고, 동일한 frozen "
        "parent에서 파생한 paired OOD bank로 평가한다. 대표선은 task-success "
        "seed 중 validation error가 가장 작은 seed를 사용하며, 성공 seed가 "
        "없으면 명시적인 failed-task fallback을 사용한다. 전체 3-seed 결과는 "
        "`ood_all_seed_summary.csv`에 별도로 보존한다.",
        "",
        "## 대표 seed 선택",
        "",
        "| Topology | Model | Seed | Task gate | Selection |",
        "|---|---:|---:|---:|---|",
    ]
    for topology in TOPOLOGIES:
        for model in MODELS:
            target = representatives[(model, topology)]
            kind = (
                "best success"
                if target.task_success
                else "best available failed fallback"
            )
            lines.append(
                f"| {topology.upper()} | {MODEL_LABELS[model]} | "
                f"{target.record.seed} | {str(target.task_success).lower()} | "
                f"{kind} |"
            )

    lines.extend(
        [
            "",
            "## 핵심 수치",
            "",
            "아래 값은 대표 seed의 intrinsic terminal error이며 단위는 radian이다.",
            "",
            "| Topology | Model | ID T=128 | T=2048 | T=1024, 2× | "
            "ID 후 blank 512 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for topology in TOPOLOGIES:
        for model in MODELS:
            id_row = representative_index[(topology, model, "id_h128")]
            length_row = representative_index[(topology, model, "length_h2048")]
            combined_row = representative_index[
                (topology, model, "combined_h1024_x2")
            ]
            lines.append(
                f"| {topology.upper()} | {MODEL_LABELS[model]} | "
                f"{float(id_row['terminal_mean_error_radians']):.4f} | "
                f"{float(length_row['terminal_mean_error_radians']):.4f} | "
                f"{float(combined_row['terminal_mean_error_radians']):.4f} | "
                f"{float(id_row['post_blank_512_mean_error_radians']):.4f} |"
            )

    lines.extend(["", "## 3-seed robustness", ""])
    for topology in TOPOLOGIES:
        ca_row = summary_index[(topology, "calru", "id_h128")]
        ca_blank = float(
            ca_row["post_blank_512_mean_error_radians_median"]
        )
        baseline_blank = min(
            float(
                summary_index[
                    (topology, model, "id_h128")
                ]["post_blank_512_mean_error_radians_median"]
            )
            for model in ("rnn", "gru", "lstm")
        )
        lines.append(
            f"- {topology.upper()}: ID 뒤 blank 512에서 CA-LRU의 3-seed "
            f"median은 {ca_blank:.4f} rad이고 가장 좋은 baseline median "
            f"{baseline_blank:.4f} rad보다 {baseline_blank / ca_blank:.2f}배 "
            "낮다."
        )

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- CA-LRU의 가장 일관된 이점은 입력 종료 뒤 기억 유지다. 이 결과는 "
            "세 topology의 3-seed median에서 모두 유지된다.",
            "- 계속 입력을 적분하는 transport 일반화는 topology 의존적이다. "
            "CA-LRU는 T2 길이 OOD에서 강하지만, S1의 매우 긴 horizon과 S2의 "
            "transport에서는 GRU/LSTM보다 빠르게 악화된다.",
            "- dwell 분포와 control smoothness 변화에는 대체로 안정적이지만, "
            "길이와 속도를 동시에 키운 combined stress에서는 LSTM이 더 강하다.",
            "- 따라서 현재 결과는 'CA-LRU가 모든 OOD에서 우월하다'가 아니라, "
            "'retention plasticity가 blank-memory 안정성을 크게 높이지만 "
            "input-driven transport 정확도는 별도의 병목이다'를 지지한다.",
            "",
            "## 해석 제한",
            "",
            "- baseline은 ring-selected 설정을 topology에 이전했고, CA-LRU는 "
            "topology별 validation-selected 설정이다. 이 차이는 표와 manifest에 "
            "그대로 기록한다.",
            "- RNN은 T2/S2에서, GRU는 S2에서 task-success seed가 없어 대표선이 "
            "failed-task fallback이다. 해당 선을 성공 모델과 동등한 증거로 "
            "해석하면 안 된다.",
            "- 대표 seed figure는 구조를 읽기 위한 primary visualization이고, "
            "재현성 판단은 반드시 전체 3-seed CSV와 함께 해야 한다.",
            "",
        ]
    )
    (output / "RESULTS_ko.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> None:
    config_path = args.config.expanduser().resolve(strict=True)
    config = strict_json_load(config_path)
    if tuple(config["models"]) != MODELS:
        raise ValueError("OOD model order differs")
    if tuple(config["topologies"]) != TOPOLOGIES:
        raise ValueError("OOD topology order differs")
    config_sha256 = sha256_file(config_path)
    analysis_code_sha256 = sha256_file(Path(__file__))
    bank_root = args.bank_root.expanduser().resolve(strict=True)
    bank_manifest_path = bank_root / "ood_banks_manifest.json"
    bank_manifest = strict_json_load(bank_manifest_path)
    if not bank_manifest.get("pass"):
        raise RuntimeError("OOD bank manifest did not pass")
    bank_manifest_sha256 = sha256_file(bank_manifest_path)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = _collect_targets(args, config)
    representatives = _representatives(targets)

    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        devices = ["cpu"]
    pending = []
    for target in targets:
        path = output / "runs" / f"{target.record.job_id}.json"
        if not path.is_file():
            pending.append(target)
            continue
        payload = strict_json_load(path)
        if (
            payload.get("analysis_config_sha256") != config_sha256
            or payload.get("analysis_code_sha256") != analysis_code_sha256
            or payload.get("bank_manifest_sha256") != bank_manifest_sha256
        ):
            pending.append(target)
    queues = [[] for _ in devices]
    for index, target in enumerate(pending):
        queues[index % len(devices)].append(target)

    def worker(device_name: str, queue: list[Target]) -> None:
        device = torch.device(
            f"cuda:{device_name}"
            if device_name != "cpu" and torch.cuda.is_available()
            else "cpu"
        )
        for target in queue:
            _analyze_target(
                target,
                config=config,
                config_sha256=config_sha256,
                analysis_code_sha256=analysis_code_sha256,
                bank_root=bank_root,
                bank_manifest_sha256=bank_manifest_sha256,
                output=output,
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(worker, device_name, queue)
            for device_name, queue in zip(devices, queues)
            if queue
        ]
        for future in futures:
            future.result()

    rows = _flatten(output)
    expected_rows = len(targets) * len(config["conditions"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} OOD rows, found {len(rows)}")
    representative_rows = [
        row
        for row in rows
        if row["job_id"]
        == representatives[(row["model"], row["topology"])].record.job_id
    ]
    selection_rows = []
    for topology in TOPOLOGIES:
        for model in MODELS:
            target = representatives[(model, topology)]
            selection_rows.append(
                {
                    "topology": topology,
                    "model": model,
                    "selected_job_id": target.record.job_id,
                    "selected_seed": target.record.seed,
                    "selected_task_success": target.task_success,
                    "selected_validation_error": target.validation_error,
                    "selection_kind": (
                        "best_validation_success"
                        if target.task_success
                        else "best_available_failed_fallback"
                    ),
                }
            )
    summary = _summary(rows, config)
    write_csv(output / "ood_seed_metrics.csv", rows)
    write_csv(output / "ood_representative_metrics.csv", representative_rows)
    write_csv(output / "ood_representative_selection.csv", selection_rows)
    write_csv(output / "ood_all_seed_summary.csv", summary)
    provenance_rows = [
        {
            "job_id": target.record.job_id,
            "model": target.record.model_id,
            "topology": target.record.topology,
            "seed": target.record.seed,
            "source_experiment_id": target.record.run_dir.parent.name,
            "checkpoint_sha256": sha256_file(target.record.checkpoint_path),
            "training_manifest_sha256": sha256_file(
                target.record.run_dir / "manifest.json"
            ),
            "training_result_sha256": sha256_file(
                target.record.run_dir / "result.json"
            ),
        }
        for target in targets
    ]
    write_csv(output / "ood_checkpoint_provenance.csv", provenance_rows)
    _write_results_report(output, rows, summary, representatives)

    _plot_axis(
        rows,
        representatives,
        config,
        output,
        axis_name="length",
        stem="fig_O1_length_generalization",
        title="Zero-retraining sequence-length generalization",
    )
    _plot_axis(
        rows,
        representatives,
        config,
        output,
        axis_name="velocity",
        stem="fig_O2_velocity_generalization",
        title="Paired velocity-scale generalization",
    )
    _plot_axis(
        rows,
        representatives,
        config,
        output,
        axis_name="dwell",
        stem="fig_O3_dwell_generalization",
        title="Dwell-distribution generalization",
    )
    _plot_axis(
        rows,
        representatives,
        config,
        output,
        axis_name="smoothness",
        stem="fig_O3b_smoothness_generalization",
        title="Control smoothness generalization",
    )
    _plot_combined(rows, representatives, config, output)

    atomic_json(
        output / "OOD_ANALYSIS_COMPLETED.json",
        {
            "schema_version": 1,
            "analysis_id": config["analysis_id"],
            "analysis_config_sha256": config_sha256,
            "analysis_code_sha256": analysis_code_sha256,
            "bank_builder_code_sha256": sha256_file(
                Path(__file__).with_name("build_ood_banks.py")
            ),
            "bank_manifest_sha256": bank_manifest_sha256,
            "bank_source_experiment_id": bank_root.name,
            "run_count": len(targets),
            "condition_count": len(config["conditions"]),
            "seed_condition_row_count": len(rows),
            "representative_row_count": len(representative_rows),
            "representative_seeds": {
                topology: {
                    model: representatives[(model, topology)].record.seed
                    for model in MODELS
                }
                for topology in TOPOLOGIES
            },
            "fallbacks": [
                {
                    "topology": topology,
                    "model": model,
                    "seed": representatives[(model, topology)].record.seed,
                }
                for topology in TOPOLOGIES
                for model in MODELS
                if not representatives[(model, topology)].task_success
            ],
            "figures": [
                "fig_O1_length_generalization.png",
                "fig_O2_velocity_generalization.png",
                "fig_O3_dwell_generalization.png",
                "fig_O3b_smoothness_generalization.png",
                "fig_O4_combined_and_postblank.png",
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--baseline-analysis-root", type=Path, required=True)
    parser.add_argument("--calru-run-root", type=Path, required=True)
    parser.add_argument("--calru-analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("topology_ood_v1.json"),
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path(__file__).with_name(
            "topology_baseline_all_analysis_v2.json"
        ),
    )
    parser.add_argument(
        "--calru-config",
        type=Path,
        default=Path(__file__).with_name("topology_hparam_analysis_v1.json"),
    )
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
