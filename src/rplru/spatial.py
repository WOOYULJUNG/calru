"""Spatial localization for intermittent vector integration.

The temporal task is unchanged.  Only the final target is discretized into
axis-aligned bins on [-1, 1], and the readout emits independent logits for
each coordinate.  Retention Plasticity uses the same cross-entropy readout
loss as task training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .artifacts import atomic_json, atomic_npz, atomic_torch_save, derived_seed
from .config import DEFAULT_PROTOCOL, load_protocol
from .extended_baselines import ChronoLSTMModel, OrthogonalRNNModel, S4DLegSModel
from .models import (
    GRUModel,
    LRUModel,
    LSTMModel,
    RNNModel,
    RPLRUModel,
    SequenceModel,
    count_parameters,
)
from .task import TaskSpec, generate_batch


SPATIAL_MODELS = (
    "rnn",
    "gru",
    "lstm",
    "chrono_lstm",
    "lru",
    "s4d_legs",
    "orthogonal_rnn",
    "rp_lru",
)
REPRESENTATIVE_CELLS = {
    "id": (4, 50),
    "delay_ood": (4, 1000),
    "update_ood": (16, 50),
    "joint_ood": (16, 1000),
}


def grid_labels(values: torch.Tensor, bins: int) -> torch.Tensor:
    """Return coordinate-wise labels with boundaries uniformly in [-1,1]."""

    if int(bins) < 2:
        raise ValueError("grid classification requires at least two bins")
    if values.ndim != 2:
        raise ValueError("grid targets must be sample x dimension")
    if torch.any(values < -1.000001) or torch.any(values > 1.000001):
        raise ValueError("grid targets escaped [-1,1]")
    scaled = (values + 1.0) * (0.5 * int(bins))
    return torch.floor(scaled).to(torch.long).clamp_(0, int(bins) - 1)


def coordinate_ce(logits: torch.Tensor, labels: torch.Tensor, bins: int) -> torch.Tensor:
    if logits.shape[:-1] != labels.shape[:-1] or logits.shape[-1] != labels.shape[-1] * int(bins):
        raise ValueError("classification logits and labels disagree")
    shaped = logits.reshape(*labels.shape, int(bins))
    return F.cross_entropy(
        shaped.reshape(-1, int(bins)), labels.reshape(-1), reduction="mean"
    )


def per_sample_ce(logits: torch.Tensor, labels: torch.Tensor, bins: int) -> torch.Tensor:
    """Coordinate-summed CE with arbitrary leading dimensions."""

    if logits.shape[-1] != labels.shape[-1] * int(bins):
        raise ValueError("classification logits and labels disagree")
    shaped = logits.reshape(*logits.shape[:-1], labels.shape[-1], int(bins))
    expanded_labels = labels
    while expanded_labels.ndim < shaped.ndim - 1:
        expanded_labels = expanded_labels.unsqueeze(0)
    expanded_labels = expanded_labels.expand(*shaped.shape[:-1])
    log_probability = F.log_softmax(shaped, dim=-1)
    selected = torch.gather(
        log_probability, -1, expanded_labels.unsqueeze(-1)
    ).squeeze(-1)
    return -selected.sum(dim=-1)


def classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor, bins: int
) -> dict[str, Any]:
    shaped = logits.reshape(labels.shape[0], labels.shape[1], int(bins))
    prediction = shaped.argmax(dim=-1)
    correct = prediction == labels
    confusion = torch.zeros(
        labels.shape[1], int(bins), int(bins), dtype=torch.int64, device=labels.device
    )
    for coordinate in range(labels.shape[1]):
        index = labels[:, coordinate] * int(bins) + prediction[:, coordinate]
        confusion[coordinate] = torch.bincount(
            index, minlength=int(bins) ** 2
        ).reshape(int(bins), int(bins))
    return {
        "cross_entropy": float(coordinate_ce(logits, labels, bins).detach().cpu()),
        "coordinate_accuracy": float(correct.float().mean().cpu()),
        "exact_accuracy": float(correct.all(dim=-1).float().mean().cpu()),
        "per_coordinate_accuracy": [
            float(correct[:, coordinate].float().mean().cpu())
            for coordinate in range(labels.shape[1])
        ],
        "confusion": confusion.detach().cpu().tolist(),
        "samples": int(labels.shape[0]),
    }


def class_balance(labels: torch.Tensor, bins: int) -> dict[str, Any]:
    counts = []
    for coordinate in range(labels.shape[1]):
        counts.append(
            torch.bincount(labels[:, coordinate], minlength=int(bins)).cpu().tolist()
        )
    proportions = np.asarray(counts, dtype=np.float64) / float(labels.shape[0])
    return {
        "counts": counts,
        "proportions": proportions.tolist(),
        "maximum_absolute_deviation_from_uniform": float(
            np.abs(proportions - 1.0 / int(bins)).max()
        ),
    }


def build_classifier(
    name: str,
    *,
    dimension: int,
    bins: int,
    width: int,
    lru_lambda_min: float = 0.90,
    chrono_t_max: int = 255,
    retention_mode: str = "rp",
    tau_sat: float = 16.64,
    fixed_unit_count: int = 0,
    fixed_fast_lambda: float = 0.0,
    fixed_subset_seed: int = 0,
) -> SequenceModel:
    input_dim = int(dimension) + 2
    output_dim = int(dimension) * int(bins)
    if name == "rnn":
        return RNNModel(input_dim, output_dim, int(width))
    if name == "gru":
        return GRUModel(input_dim, output_dim, int(width))
    if name == "lstm":
        return LSTMModel(input_dim, output_dim, int(width))
    if name == "chrono_lstm":
        return ChronoLSTMModel(
            input_dim, output_dim, int(width), chrono_t_max=int(chrono_t_max)
        )
    if name == "lru":
        return LRUModel(
            input_dim,
            output_dim,
            int(width),
            lambda_min=float(lru_lambda_min),
            lambda_max=0.999,
        )
    if name == "s4d_legs":
        return S4DLegSModel(input_dim, output_dim, int(width))
    if name == "orthogonal_rnn":
        return OrthogonalRNNModel(input_dim, output_dim, int(width))
    if name == "rp_lru":
        return RPLRUModel(
            input_dim,
            output_dim,
            int(width),
            retention_mode=retention_mode,
            initial_lambda_low=0.98,
            initial_lambda_high=0.999,
            tau_sat=tau_sat,
            fixed_unit_count=fixed_unit_count,
            fixed_fast_lambda=fixed_fast_lambda,
            fixed_subset_seed=fixed_subset_seed,
        )
    raise ValueError(f"unsupported spatial model: {name}")


def matched_width(
    name: str,
    *,
    dimension: int,
    bins: int,
    target: int,
    lru_lambda_min: float = 0.90,
    chrono_t_max: int = 255,
) -> tuple[int, int]:
    if name == "rp_lru":
        model = build_classifier(
            name,
            dimension=dimension,
            bins=bins,
            width=91,
            lru_lambda_min=lru_lambda_min,
            chrono_t_max=chrono_t_max,
        )
        return 91, count_parameters(model)
    low, high = 4, 1024
    best: tuple[int, int] | None = None
    while low <= high:
        middle = (low + high) // 2
        count = count_parameters(
            build_classifier(
                name,
                dimension=dimension,
                bins=bins,
                width=middle,
                lru_lambda_min=lru_lambda_min,
                chrono_t_max=chrono_t_max,
            )
        )
        if best is None or abs(count - target) < abs(best[1] - target):
            best = (middle, count)
        if count < target:
            low = middle + 1
        else:
            high = middle - 1
    assert best is not None
    candidates = range(max(4, best[0] - 2), best[0] + 3)
    return min(
        (
            (
                width,
                count_parameters(
                    build_classifier(
                        name,
                        dimension=dimension,
                        bins=bins,
                        width=width,
                        lru_lambda_min=lru_lambda_min,
                        chrono_t_max=chrono_t_max,
                    )
                ),
            )
            for width in candidates
        ),
        key=lambda item: (abs(item[1] - target), item[0]),
    )


@torch.no_grad()
def classification_rp_update(
    model: RPLRUModel,
    states: torch.Tensor,
    continuous_targets: torch.Tensor,
    *,
    dimension: int,
    bins: int,
    horizon: int,
    eta_lambda: float,
    threshold: float,
    normalization_epsilon: float = 1e-8,
) -> dict[str, Any]:
    labels = grid_labels(continuous_targets, bins)
    decay = model.retention().pow(int(horizon))
    clean_logits = model.decode(states * decay)
    clean_loss = per_sample_ce(clean_logits, labels, bins)
    probe_count = states.shape[0]
    ablated = states.unsqueeze(0).expand(model.width, probe_count, model.width).clone()
    coordinate = torch.arange(model.width, device=states.device)
    ablated[coordinate, :, coordinate] = 0.0
    ablated_logits = model.decode(ablated * decay)
    ablated_loss = per_sample_ce(ablated_logits, labels, bins)
    damage = (ablated_loss - clean_loss.unsqueeze(0)).mean(dim=1)
    energy = states.square().mean(dim=0)
    score = damage / (energy + float(normalization_epsilon))
    allocation = score - float(threshold)
    model.theta.add_(float(eta_lambda) * allocation)
    retention = model.retention()
    return {
        "clean_coordinate_ce": float(clean_loss.mean().cpu() / int(dimension)),
        "damage_mean": float(damage.mean().cpu()),
        "normalized_score_min": float(score.min().cpu()),
        "normalized_score_mean": float(score.mean().cpu()),
        "normalized_score_max": float(score.max().cpu()),
        "positive_allocation_fraction": float((allocation > 0).float().mean().cpu()),
        "lambda_min": float(retention.min().cpu()),
        "lambda_mean": float(retention.mean().cpu()),
        "lambda_max": float(retention.max().cpu()),
        "exact_unit_count": int((retention == 1.0).sum().cpu()),
    }


def classification_recall_grad_update(
    model: RPLRUModel,
    states: torch.Tensor,
    continuous_targets: torch.Tensor,
    *,
    bins: int,
    horizon: int,
    eta_gradient: float,
) -> dict[str, Any]:
    """Schedule-matched CE RecallGrad with the real sigmoid derivative."""

    states = states.detach()
    labels = grid_labels(continuous_targets.detach(), bins)
    with torch.enable_grad():
        theta = model.theta.detach().clone().requires_grad_(True)
        retention = model._map_theta(theta)
        logits = model.decode(states * retention.pow(int(horizon)))
        loss = per_sample_ce(logits, labels, bins).mean()
        gradient = torch.autograd.grad(loss, theta)[0]
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("classification RecallGrad is non-finite")
    with torch.no_grad():
        model.theta.add_(-float(eta_gradient) * gradient)
        if not torch.isfinite(model.theta).all():
            raise FloatingPointError("classification RecallGrad logits are non-finite")
        after = model.retention()
    return {
        "recall_cross_entropy": float(loss.detach().cpu()),
        "gradient_l2": float(torch.linalg.vector_norm(gradient).detach().cpu()),
        "gradient_max_abs": float(gradient.abs().max().detach().cpu()),
        "lambda_min": float(after.min().cpu()),
        "lambda_mean": float(after.mean().cpu()),
        "lambda_max": float(after.max().cpu()),
        "exact_unit_count": int((after == 1.0).sum().cpu()),
    }


@torch.no_grad()
def final_logits(model: SequenceModel, batch) -> torch.Tensor:
    state = model.initial_state(
        batch.batch_size, device=batch.inputs.device, dtype=batch.inputs.dtype
    )
    output = torch.empty(
        batch.batch_size, model.output_dim, device=batch.inputs.device, dtype=batch.inputs.dtype
    )
    sample = torch.arange(batch.batch_size, device=batch.inputs.device)
    for time, x_t in enumerate(batch.inputs):
        state = model.step(x_t, state)
        final = batch.lengths == time + 1
        if torch.any(final):
            output[sample[final]] = model.decode(state[final])
    return output


@dataclass(frozen=True)
class SpatialSpec:
    model: str
    bins: int
    seed: int
    learning_rate: float
    updates: int
    batch_size: int
    evaluation_trajectories: int
    dimension: int = 2
    rp_horizon: int = 50
    rp_threshold: float = 0.20
    rp_eta: float = 1.0
    rp_warmup: int = 3000
    rp_interval: int = 100
    lru_lambda_min: float = 0.90
    chrono_t_max: int = 255
    retention_mode: str = "rp"
    retention_learning_rate: float | None = None
    tau_sat: float = 16.64
    fixed_unit_count: int = 0
    fixed_fast_lambda: float = 0.0
    fixed_subset_seed: int | None = None


def run(spec: SpatialSpec, *, protocol_path: Path, output: Path, device: str) -> dict[str, Any]:
    if spec.model not in SPATIAL_MODELS or spec.bins not in (3, 5):
        raise ValueError("unsupported spatial condition")
    if spec.rp_horizon <= 0 or spec.rp_threshold < 0.0 or spec.rp_eta <= 0.0:
        raise ValueError("invalid RP horizon, threshold, or step size")
    if spec.rp_warmup < 0 or spec.rp_interval <= 0:
        raise ValueError("invalid RP warmup or interval")
    protocol = load_protocol(protocol_path)
    task = replace(
        protocol.task,
        initial_value_low=-1.0,
        initial_value_high=1.0,
    )
    task.validate()
    torch.manual_seed(int(spec.seed))
    target_model = build_classifier(
        "rp_lru", dimension=spec.dimension, bins=spec.bins, width=91
    )
    target_count = count_parameters(target_model)
    width, parameter_count = matched_width(
        spec.model,
        dimension=spec.dimension,
        bins=spec.bins,
        target=target_count,
        lru_lambda_min=spec.lru_lambda_min,
        chrono_t_max=spec.chrono_t_max,
    )
    torch.manual_seed(int(spec.seed))
    model = build_classifier(
        spec.model,
        dimension=spec.dimension,
        bins=spec.bins,
        width=width,
        lru_lambda_min=spec.lru_lambda_min,
        chrono_t_max=spec.chrono_t_max,
        retention_mode=spec.retention_mode,
        tau_sat=spec.tau_sat,
        fixed_unit_count=spec.fixed_unit_count,
        fixed_fast_lambda=spec.fixed_fast_lambda,
        fixed_subset_seed=(
            spec.seed if spec.fixed_subset_seed is None else spec.fixed_subset_seed
        ),
    ).to(device)
    parameter_count = count_parameters(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_groups = model.optimizer_parameter_groups(
        learning_rate=spec.learning_rate, weight_decay=0.0
    )
    scheduled_parameter = (
        model.schedule_task_gradient_parameter()
        if isinstance(model, RPLRUModel)
        else None
    )
    if scheduled_parameter is not None and spec.retention_learning_rate is not None:
        scheduled_id = id(scheduled_parameter)
        adjusted = []
        for group in parameter_groups:
            kept = [parameter for parameter in group["params"] if id(parameter) != scheduled_id]
            if kept:
                adjusted.append({**group, "params": kept})
        adjusted.append(
            {
                "params": [scheduled_parameter],
                "lr": float(spec.retention_learning_rate),
                "weight_decay": 0.0,
                "group_name": "scheduled_retention_parameter",
            }
        )
        parameter_groups = adjusted
    optimizer = torch.optim.AdamW(parameter_groups)
    train_spec = TaskSpec.training(spec.dimension, spec.batch_size, task)
    validation_spec = TaskSpec.training(spec.dimension, 1024, task)
    validation = generate_batch(
        validation_spec,
        task,
        seed=derived_seed(
            protocol.evaluation.bank_seed,
            "grid_classification_validation",
            spec.bins,
        ),
        device=device,
    )
    validation_labels = grid_labels(validation.final_targets, spec.bins)
    balance = class_balance(validation_labels, spec.bins)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "experiment": "spatial_localization",
        "spec": asdict(spec),
        "device": str(device),
        "protocol": "protocol.json",
        "task": asdict(task),
        "grid": {
            "range": [-1.0, 1.0],
            "bins_per_coordinate": spec.bins,
            "boundaries": np.linspace(-1.0, 1.0, spec.bins + 1)[1:-1].tolist(),
            "chance_coordinate": 1.0 / spec.bins,
            "chance_exact": 1.0 / (spec.bins ** spec.dimension),
        },
        "model": {
            "width": width,
            "parameter_count": parameter_count,
            "target_parameter_count": target_count,
            "relative_gap": abs(parameter_count - target_count) / target_count,
        },
        "validation_class_balance": balance,
    }
    atomic_json(output / "protocol.json", protocol.raw)
    atomic_json(output / "manifest.json", manifest)
    loss_trace = []
    validation_trace = []
    rp_trace = []
    started = time.time()
    for update in range(1, spec.updates + 1):
        batch = generate_batch(
            train_spec,
            task,
            seed=derived_seed(spec.seed, "grid_classification_train", spec.bins, update),
            device=device,
            audit=False,
        )
        logits = model(batch.inputs, batch.lengths)
        labels = grid_labels(batch.final_targets, spec.bins)
        loss = coordinate_ce(logits, labels, spec.bins)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if scheduled_parameter is not None and update <= spec.rp_warmup:
            scheduled_parameter.grad = None
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        loss_trace.append(float(loss.detach().cpu()))

        if (
            isinstance(model, RPLRUModel)
            and model.probe_update_enabled
            and update > spec.rp_warmup
            and (update - spec.rp_warmup) % spec.rp_interval == 0
        ):
            model.eval()
            with torch.no_grad():
                states = model.forward_states(batch.inputs)
                selected_states = states[batch.boundary_mask]
                selected_targets = batch.memory_targets[batch.boundary_mask]
                if selected_states.shape[0] > 96:
                    generator = torch.Generator(device=device).manual_seed(
                        derived_seed(spec.seed, "grid_rp_probe", update)
                    )
                    selected = torch.randperm(
                        selected_states.shape[0], generator=generator, device=device
                    )[:96]
                    selected_states = selected_states[selected]
                    selected_targets = selected_targets[selected]
                if model.rp_enabled:
                    rp_row = classification_rp_update(
                        model,
                        selected_states.detach(),
                        selected_targets.detach(),
                        dimension=spec.dimension,
                        bins=spec.bins,
                        horizon=spec.rp_horizon,
                        eta_lambda=spec.rp_eta,
                        threshold=spec.rp_threshold,
                    )
                elif model.rp_grad_enabled:
                    rp_row = classification_recall_grad_update(
                        model,
                        selected_states.detach(),
                        selected_targets.detach(),
                        bins=spec.bins,
                        horizon=spec.rp_horizon,
                        eta_gradient=spec.rp_eta,
                    )
                else:
                    raise AssertionError("unsupported classification probe update")
            rp_trace.append({"update": update, **rp_row})
            model.train()

        if update == 1 or update == spec.updates or update % 500 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = final_logits(model, validation)
                row = {
                    "update": update,
                    "training_loss": loss_trace[-1],
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    **classification_metrics(
                        val_logits, validation_labels, spec.bins
                    ),
                }
            validation_trace.append(row)
            atomic_json(
                output / "progress.json",
                {
                    "status": "running" if update < spec.updates else "training_complete",
                    "update": update,
                    "elapsed_seconds": time.time() - started,
                    "validation": row,
                    "rp_calls": len(rp_trace),
                },
            )
            model.train()

    model.eval()
    evaluation = {}
    for regime, (updates, hold) in REPRESENTATIVE_CELLS.items():
        controlled = generate_batch(
            TaskSpec.controlled(
                spec.dimension, spec.evaluation_trajectories, updates, hold
            ),
            task,
            seed=derived_seed(
                protocol.evaluation.bank_seed,
                "grid_classification_eval",
                spec.bins,
                updates,
                hold,
            ),
            device=device,
        )
        labels = grid_labels(controlled.final_targets, spec.bins)
        logits = final_logits(model, controlled)
        evaluation[regime] = {
            "update_count": updates,
            "segment_hold": hold,
            **classification_metrics(logits, labels, spec.bins),
            "class_balance": class_balance(labels, spec.bins),
        }

    retention = None
    if isinstance(model, RPLRUModel):
        values = model.retention().detach().cpu().numpy()
        retention = {
            "exact_unit_count": int((values == 1.0).sum()),
            "effective_count_h17017": int(
                (np.power(values.astype(np.float64), 17017) >= 0.99).sum()
            ),
            "below_half_count": int((values < 0.5).sum()),
            "mean_lambda": float(values.mean()),
        }
        atomic_npz(
            output / "retention.npz",
            theta=model.retention_logits().detach().cpu().numpy(),
            retention=values,
        )
    result = {
        "schema_version": 1,
        "status": "complete",
        "spec": asdict(spec),
        "elapsed_seconds": time.time() - started,
        "final_validation": validation_trace[-1],
        "evaluation": evaluation,
        "retention": retention,
        "rp_calls": len(rp_trace),
    }
    atomic_torch_save(
        output / "checkpoint.pt",
        {
            "model_state_dict": model.state_dict(),
            "manifest": manifest,
            "result": result,
        },
    )
    atomic_json(output / "training_trace.json", {"validation": validation_trace, "rp": rp_trace})
    np.save(output / "loss.npy", np.asarray(loss_trace, dtype=np.float32))
    atomic_json(output / "result.json", result)
    atomic_json(output / "COMPLETED.json", {"status": "complete", "result": result})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=SPATIAL_MODELS, required=True)
    parser.add_argument("--bins", type=int, choices=(3, 5), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--updates", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-trajectories", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rp-threshold", type=float, default=0.20)
    parser.add_argument("--rp-eta", type=float, default=1.0)
    parser.add_argument("--rp-warmup", type=int, default=3000)
    parser.add_argument("--rp-interval", type=int, default=100)
    parser.add_argument("--lru-lambda-min", type=float, default=0.90)
    parser.add_argument("--chrono-t-max", type=int, default=255)
    parser.add_argument("--retention-mode", default="rp")
    parser.add_argument("--retention-learning-rate", type=float)
    parser.add_argument("--tau-sat", type=float, default=16.64)
    parser.add_argument("--fixed-unit-count", type=int, default=0)
    parser.add_argument("--fixed-fast-lambda", type=float, default=0.0)
    parser.add_argument("--fixed-subset-seed", type=int)
    args = parser.parse_args()
    spec = SpatialSpec(
        model=args.model,
        bins=args.bins,
        seed=args.seed,
        learning_rate=args.learning_rate,
        updates=args.updates,
        batch_size=args.batch_size,
        evaluation_trajectories=args.evaluation_trajectories,
        rp_threshold=args.rp_threshold,
        rp_eta=args.rp_eta,
        rp_warmup=args.rp_warmup,
        rp_interval=args.rp_interval,
        lru_lambda_min=args.lru_lambda_min,
        chrono_t_max=args.chrono_t_max,
        retention_mode=args.retention_mode,
        retention_learning_rate=args.retention_learning_rate,
        tau_sat=args.tau_sat,
        fixed_unit_count=args.fixed_unit_count,
        fixed_fast_lambda=args.fixed_fast_lambda,
        fixed_subset_seed=args.fixed_subset_seed,
    )
    result = run(
        spec,
        protocol_path=Path(args.protocol),
        output=Path(args.output_dir),
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
