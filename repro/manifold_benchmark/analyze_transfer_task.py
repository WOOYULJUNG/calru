"""Stage 2: zero-retuning validation eligibility and final test metrics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from repro.sagodi_protocol.artifacts import atomic_json, strict_json_load

from .topology_analysis_common import (
    atomic_npz,
    discover_completed_runs,
    load_analysis_config,
    load_model,
    nmse_db,
    normalized_geodesic_errors,
    output_norm_error,
    summarize_error_tensor,
    test_batch,
    validation_batch,
    write_csv,
)
from .topology_training import hold_prediction


@torch.no_grad()
def _evaluate(model, batch, percentiles: list[int]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    prediction = model.forward_sequence(
        batch.inputs, initial_memory=batch.initial_memory
    )
    errors, worst = normalized_geodesic_errors(
        batch.topology, prediction, batch.output_targets
    )
    hold_errors, hold_worst = normalized_geodesic_errors(
        batch.topology, hold_prediction(batch), batch.output_targets
    )
    metrics: dict[str, Any] = {
        **summarize_error_tensor(errors, percentiles),
        "nmse_db": nmse_db(prediction, batch.output_targets),
        "component_mse": float((prediction - batch.output_targets).square().mean().cpu()),
        "output_norm_error_mean": float(output_norm_error(batch.topology, prediction).mean().cpu()),
        "hold_sequence_mean": float(hold_errors.mean().cpu()),
        "hold_relative_improvement": float(
            (hold_errors.mean() - errors.mean()).div(hold_errors.mean().clamp_min(1e-8)).cpu()
        ),
        "finite": bool(torch.isfinite(prediction).all().item()),
    }
    if worst is not None:
        metrics["worst_coordinate_sequence_mean"] = float(worst.mean().cpu())
        metrics["hold_worst_coordinate_sequence_mean"] = float(hold_worst.mean().cpu())
    else:
        metrics["worst_coordinate_sequence_mean"] = None
        metrics["hold_worst_coordinate_sequence_mean"] = None
    arrays = {
        "normalized_geodesic_error": errors.cpu().numpy().astype(np.float32),
        "hold_normalized_geodesic_error": hold_errors.cpu().numpy().astype(np.float32),
        "output_norm_error": output_norm_error(batch.topology, prediction)
        .cpu()
        .numpy()
        .astype(np.float32),
    }
    if worst is not None:
        arrays["worst_coordinate_error"] = worst.cpu().numpy().astype(np.float32)
    return metrics, arrays


def analyze(
    run_root: Path,
    analysis_root: Path,
    config_path: Path,
    device: torch.device,
) -> None:
    config = load_analysis_config(config_path)
    integrity = strict_json_load(analysis_root / "run_integrity.json")
    if not integrity.get("ready_for_test_analysis", False):
        raise RuntimeError("run integrity has not authorized test-bank access")
    records, missing = discover_completed_runs(run_root, config)
    if missing:
        raise RuntimeError("task analysis refuses a partial run set")
    task_config = config["task_transfer"]
    eligibility = config["eligibility"]
    output = analysis_root / "task"
    run_output = output / "runs"
    run_output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    success: dict[str, bool] = {}
    cached_batches: dict[tuple[str, str], Any] = {}
    for record in records:
        model, _ = load_model(record, device)
        key_validation = (record.topology, "validation")
        key_test = (record.topology, "test")
        if key_validation not in cached_batches:
            cached_batches[key_validation] = validation_batch(
                record.topology,
                device=device,
                trajectories=int(task_config["test_trajectories"]),
            )
            cached_batches[key_test] = test_batch(
                record.topology,
                device=device,
                trajectories=int(task_config["test_trajectories"]),
            )
        validation_metrics, _ = _evaluate(
            model,
            cached_batches[key_validation],
            list(task_config["trial_percentiles"]),
        )
        test_metrics, arrays = _evaluate(
            model,
            cached_batches[key_test],
            list(task_config["trial_percentiles"]),
        )
        eligible = bool(
            validation_metrics["finite"]
            and validation_metrics["nmse_db"]
            < float(eligibility["validation_nmse_db_strictly_below"])
            and validation_metrics["sequence_mean"]
            < validation_metrics["hold_sequence_mean"]
        )
        success[record.job_id] = eligible
        row = {
            "job_id": record.job_id,
            "model": record.model_id,
            "topology": record.topology,
            "seed": record.seed,
            "completed": True,
            "task_success": eligible,
            "parameters_total": record.manifest["model"]["parameters_total"],
            "validation_nmse_db": validation_metrics["nmse_db"],
            "validation_error": validation_metrics["sequence_mean"],
            "validation_hold_error": validation_metrics["hold_sequence_mean"],
            "test_mean_error": test_metrics["sequence_mean"],
            "test_terminal_error": test_metrics["terminal_mean"],
            "test_trial_median": test_metrics["trial_median"],
            "test_trial_p90": test_metrics["trial_p90"],
            "test_trial_p95": test_metrics["trial_p95"],
            "test_trial_p99": test_metrics["trial_p99"],
            "test_output_norm_error": test_metrics["output_norm_error_mean"],
            "test_hold_error": test_metrics["hold_sequence_mean"],
            "test_hold_relative_improvement": test_metrics["hold_relative_improvement"],
            "test_worst_coordinate_error": test_metrics[
                "worst_coordinate_sequence_mean"
            ],
            "finite": test_metrics["finite"],
        }
        rows.append(row)
        atomic_npz(run_output / f"{record.job_id}.npz", **arrays)
        atomic_json(
            run_output / f"{record.job_id}.json",
            {
                "schema_version": 1,
                "job_id": record.job_id,
                "validation": validation_metrics,
                "test": test_metrics,
                "task_success": eligible,
            },
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rows.sort(key=lambda item: (item["topology"], item["model"], item["seed"]))
    write_csv(output / "task_metrics.csv", rows)
    write_csv(analysis_root / "task_metrics.csv", rows)
    counts = {
        f"{model}_{topology}": sum(
            success.get(f"pilot__{model}__{topology}__seed{seed}", False)
            for seed in config["expected_training"]["seeds"]
        )
        for topology in config["expected_training"]["topologies"]
        for model in config["expected_training"]["models"]
    }
    success_payload = {
            "schema_version": 1,
            "eligibility": eligibility,
            "task_success": success,
            "success_seed_count_out_of_3": counts,
            "test_access_authorized_by": "analysis/run_integrity.json",
        }
    atomic_json(output / "task_success.json", success_payload)
    atomic_json(analysis_root / "task_success.json", success_payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("topology_analysis_v1.json")
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    analyze(
        args.run_root.expanduser().resolve(strict=True),
        args.analysis_root.expanduser().resolve(strict=True),
        args.config.expanduser().resolve(strict=True),
        torch.device(args.device),
    )


if __name__ == "__main__":
    main()
