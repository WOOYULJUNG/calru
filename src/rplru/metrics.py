"""Task and decomposition metrics for intermittent vector integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .models import SequenceModel
from .task import ADD, WRITE, IntermittentBatch


def regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must be matching sample x dimension tensors")
    residual = prediction - target
    numerator = residual.square().sum()
    centered = target - target.mean(dim=0, keepdim=True)
    denominator = centered.square().sum()
    per_coordinate_mse = residual.square().mean()
    return {
        "nmse": float((numerator / (denominator + float(epsilon))).cpu()),
        "mse_per_coordinate": float(per_coordinate_mse.cpu()),
        "rmse_per_coordinate": float(torch.sqrt(per_coordinate_mse).cpu()),
        "target_variance_per_coordinate": float(target.var(dim=0, unbiased=False).mean().cpu()),
    }


@dataclass
class EvaluationOutput:
    metrics: dict[str, Any]
    final_prediction: torch.Tensor
    boundary_states: torch.Tensor | None
    boundary_targets: torch.Tensor | None


@torch.no_grad()
def evaluate_batch(
    model: SequenceModel,
    batch: IntermittentBatch,
    *,
    collect_boundary_states: bool = False,
) -> EvaluationOutput:
    """Stream through a possibly very long OOD batch without storing all states."""

    inputs = batch.inputs
    device = inputs.device
    state = model.initial_state(
        batch.batch_size, device=device, dtype=inputs.dtype
    )
    final_prediction = torch.empty_like(batch.final_targets)
    max_segments = int(batch.segment_holds.shape[1])
    boundary_error = torch.full(
        (batch.batch_size, max_segments), torch.nan, device=device
    )
    hold_error = torch.full_like(boundary_error, torch.nan)
    boundary_norm = torch.full_like(boundary_error, torch.nan)
    hold_norm = torch.full_like(boundary_error, torch.nan)
    segment_index = torch.full(
        (batch.batch_size,), -1, dtype=torch.long, device=device
    )
    write_errors = []
    add_errors = []
    selected_states = []
    selected_targets = []
    sample_index = torch.arange(batch.batch_size, device=device)

    for time, x_t in enumerate(inputs):
        state = model.step(x_t, state)
        boundary = batch.boundary_mask[time]
        if torch.any(boundary):
            segment_index[boundary] += 1
            prediction = model.decode(state[boundary])
            target = batch.memory_targets[time, boundary]
            error = (prediction - target).square().mean(dim=-1)
            selected_sample = sample_index[boundary]
            selected_segment = segment_index[boundary]
            boundary_error[selected_sample, selected_segment] = error
            boundary_norm[selected_sample, selected_segment] = state[
                boundary
            ].square().mean(dim=-1)
            operation = batch.operation[time, boundary]
            if torch.any(operation == WRITE):
                write_errors.append(error[operation == WRITE])
            if torch.any(operation == ADD):
                add_errors.append(error[operation == ADD])
            if collect_boundary_states:
                selected_states.append(state[boundary].detach().cpu())
                selected_targets.append(target.detach().cpu())

        hold_end = batch.hold_end_mask[time]
        if torch.any(hold_end):
            prediction = model.decode(state[hold_end])
            target = batch.memory_targets[time, hold_end]
            error = (prediction - target).square().mean(dim=-1)
            selected_sample = sample_index[hold_end]
            selected_segment = segment_index[hold_end]
            hold_error[selected_sample, selected_segment] = error
            hold_norm[selected_sample, selected_segment] = state[
                hold_end
            ].square().mean(dim=-1)

        final = batch.lengths.to(device) == time + 1
        if torch.any(final):
            final_prediction[final] = model.decode(state[final])

    final_metrics = regression_metrics(final_prediction, batch.final_targets)
    paired = torch.isfinite(boundary_error) & torch.isfinite(hold_error)
    if not torch.any(paired):
        raise ValueError("evaluation batch contains no paired hold segments")

    def concatenated_mean(values: list[torch.Tensor]) -> float | None:
        if not values:
            return None
        return float(torch.cat(values).mean().cpu())

    metrics: dict[str, Any] = {
        **final_metrics,
        "write_post_mse_per_coordinate": concatenated_mean(write_errors),
        "add_post_mse_per_coordinate": concatenated_mean(add_errors),
        "boundary_post_mse_per_coordinate": float(
            boundary_error[paired].mean().cpu()
        ),
        "hold_end_mse_per_coordinate": float(hold_error[paired].mean().cpu()),
        "retention_degradation_per_coordinate": float(
            (hold_error[paired] - boundary_error[paired]).mean().cpu()
        ),
        "boundary_state_energy_per_coordinate": float(
            boundary_norm[paired].mean().cpu()
        ),
        "hold_end_state_energy_per_coordinate": float(
            hold_norm[paired].mean().cpu()
        ),
        "samples": batch.batch_size,
        "write_events": int(sum(value.numel() for value in write_errors)),
        "add_events": int(sum(value.numel() for value in add_errors)),
        "paired_hold_segments": int(paired.sum().cpu()),
        "dimension": batch.dimension,
        "mean_update_count": float(batch.update_counts.float().mean().cpu()),
        "mean_total_blank_steps": float(
            batch.total_blank_steps.float().mean().cpu()
        ),
        "maximum_sequence_steps": batch.time_steps,
    }
    return EvaluationOutput(
        metrics=metrics,
        final_prediction=final_prediction,
        boundary_states=(
            torch.cat(selected_states, dim=0) if selected_states else None
        ),
        boundary_targets=(
            torch.cat(selected_targets, dim=0) if selected_targets else None
        ),
    )
