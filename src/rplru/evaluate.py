"""Evaluate a checkpoint on the fixed integration ID/OOD grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .artifacts import atomic_json, atomic_npz, derived_seed
from .checkpoint import (
    load_checkpoint_model,
    load_checkpoint_protocol,
    validate_evaluation_compatibility,
)
from .config import DEFAULT_PROTOCOL, load_protocol
from .metrics import evaluate_batch, regression_metrics
from .task import TaskSpec, generate_batch


def _integers(value: str | None, default: Iterable[int]) -> tuple[int, ...]:
    if value is None:
        return tuple(int(item) for item in default)
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("condition list must not be empty")
    return result


def _weighted(rows: list[dict[str, Any]], key: str, weight: str) -> float | None:
    usable = [
        row
        for row in rows
        if row[key] is not None and int(row[weight]) > 0
    ]
    if not usable:
        return None
    denominator = sum(int(row[weight]) for row in usable)
    return sum(float(row[key]) * int(row[weight]) for row in usable) / denominator


@torch.no_grad()
def evaluate_condition(
    model,
    protocol,
    *,
    dimension: int,
    update_count: int,
    segment_hold: int,
    trajectories: int,
    evaluation_batch_size: int,
    device: torch.device | str,
    trial_data: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    predictions = []
    targets = []
    chunks = []
    condition_seed = derived_seed(
        protocol.evaluation.bank_seed,
        "ood",
        dimension,
        update_count,
        segment_hold,
    )
    for start in range(0, int(trajectories), int(evaluation_batch_size)):
        count = min(int(evaluation_batch_size), int(trajectories) - start)
        batch = generate_batch(
            TaskSpec.controlled(
                dimension, count, update_count, segment_hold
            ),
            protocol.task,
            seed=condition_seed,
            sample_offset=start,
            device=device,
            audit=False,
        )
        output = evaluate_batch(model, batch)
        predictions.append(output.final_prediction.detach().cpu())
        targets.append(batch.final_targets.detach().cpu())
        chunks.append(output.metrics)
    final_prediction = torch.cat(predictions)
    final_target = torch.cat(targets)
    final = regression_metrics(final_prediction, final_target)
    if trial_data is not None:
        trial_data["squared_error_per_coordinate"] = (
            (final_prediction - final_target).square().numpy().astype(np.float32)
        )
    paired_weight = "paired_hold_segments"
    row = {
        **final,
        "write_post_mse_per_coordinate": _weighted(
            chunks, "write_post_mse_per_coordinate", "write_events"
        ),
        "add_post_mse_per_coordinate": _weighted(
            chunks, "add_post_mse_per_coordinate", "add_events"
        ),
        "boundary_post_mse_per_coordinate": _weighted(
            chunks, "boundary_post_mse_per_coordinate", paired_weight
        ),
        "hold_end_mse_per_coordinate": _weighted(
            chunks, "hold_end_mse_per_coordinate", paired_weight
        ),
        "retention_degradation_per_coordinate": _weighted(
            chunks, "retention_degradation_per_coordinate", paired_weight
        ),
        "boundary_state_energy_per_coordinate": _weighted(
            chunks, "boundary_state_energy_per_coordinate", paired_weight
        ),
        "hold_end_state_energy_per_coordinate": _weighted(
            chunks, "hold_end_state_energy_per_coordinate", paired_weight
        ),
        "dimension": int(dimension),
        "update_count": int(update_count),
        "segment_hold": int(segment_hold),
        "total_blank_steps": int((update_count + 1) * segment_hold),
        "sequence_steps": int(
            1 + update_count + (update_count + 1) * segment_hold
        ),
        "trajectories": int(trajectories),
        "regime": protocol.evaluation.regime(update_count, segment_hold),
    }
    return row


def evaluate_grid(
    *,
    protocol_path: str | Path,
    run_dir: str | Path,
    output_dir: str | Path,
    device: str,
    evaluation_batch_size: int,
    trajectories: int | None,
    update_counts: str | None,
    segment_holds: str | None,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    model, checkpoint = load_checkpoint_model(
        run_dir,
        protocol,
        device=device,
        allow_protocol_mismatch=True,
    )
    spec = checkpoint["train_spec"]
    dimension = int(spec["dimension"])
    training_protocol = load_checkpoint_protocol(run_dir, checkpoint)
    validate_evaluation_compatibility(
        training_protocol, protocol, dimension=dimension
    )
    updates = _integers(update_counts, protocol.evaluation.update_counts)
    holds = _integers(
        segment_holds, protocol.evaluation.segment_hold_lengths
    )
    count = (
        protocol.evaluation.trajectories
        if trajectories is None
        else int(trajectories)
    )
    rows = []
    trial_squared_errors = []
    for update_count in updates:
        for hold in holds:
            trial_data: dict[str, np.ndarray] = {}
            row = evaluate_condition(
                model,
                protocol,
                dimension=dimension,
                update_count=update_count,
                segment_hold=hold,
                trajectories=count,
                evaluation_batch_size=evaluation_batch_size,
                device=device,
                trial_data=trial_data,
            )
            squared_error = trial_data["squared_error_per_coordinate"]
            if squared_error.shape != (count, dimension):
                raise AssertionError("trial-level OOD error has the wrong shape")
            reconstructed_mse = float(squared_error.mean(dtype=np.float64))
            tolerance = max(1e-8, abs(float(row["mse_per_coordinate"])) * 1e-5)
            if abs(reconstructed_mse - float(row["mse_per_coordinate"])) > tolerance:
                raise AssertionError("trial errors do not reconstruct aggregate MSE")
            rows.append(row)
            trial_squared_errors.append(squared_error)
            print(
                f"M={update_count:2d} H={hold:4d} "
                f"{rows[-1]['regime']:11s} NMSE={rows[-1]['nmse']:.6g}",
                flush=True,
            )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trial_error_path = output / "ood_trial_errors.npz"
    atomic_npz(
        trial_error_path,
        squared_error_per_coordinate=np.stack(trial_squared_errors, axis=0),
        condition_update_count=np.asarray(
            [row["update_count"] for row in rows], dtype=np.int64
        ),
        condition_segment_hold=np.asarray(
            [row["segment_hold"] for row in rows], dtype=np.int64
        ),
        sample_index=np.arange(count, dtype=np.int64),
        dimension=np.asarray(dimension, dtype=np.int64),
    )
    with (output / "ood_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema_version": 1,
        "protocol": str(protocol.source),
        "protocol_sha256": protocol.sha256,
        "training_protocol": str(training_protocol.source),
        "training_protocol_sha256": training_protocol.sha256,
        "run_dir": str(Path(run_dir).resolve()),
        "checkpoint_source": checkpoint["source"],
        "train_spec": spec,
        "evaluation_batch_size": int(evaluation_batch_size),
        "trajectories_per_condition": count,
        "update_counts": list(updates),
        "segment_holds": list(holds),
        "condition_count": len(rows),
        "trial_error_artifact": {
            "path": str(trial_error_path.resolve()),
            "array": "squared_error_per_coordinate",
            "axes": ["condition", "trial", "coordinate"],
            "shape": [len(rows), count, dimension],
            "dtype": "float32",
            "condition_order": "same as rows",
        },
        "rows": rows,
    }
    atomic_json(output / "ood_evaluation.json", result)
    atomic_json(
        output / "COMPLETED.json",
        {
            "schema_version": 1,
            "status": "complete",
            "condition_count": len(rows),
            "trial_error_shape": [len(rows), count, dimension],
            "protocol_sha256": protocol.sha256,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--trajectories", type=int)
    parser.add_argument("--update-counts")
    parser.add_argument("--segment-holds")
    args = parser.parse_args()
    result = evaluate_grid(
        protocol_path=args.protocol,
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        device=args.device,
        evaluation_batch_size=args.evaluation_batch_size,
        trajectories=args.trajectories,
        update_counts=args.update_counts,
        segment_holds=args.segment_holds,
    )
    print(f"wrote {result['condition_count']} conditions to {args.output_dir}")


if __name__ == "__main__":
    main()
