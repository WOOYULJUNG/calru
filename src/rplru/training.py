"""Final-only task training and boundary-only Retention Plasticity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time
from typing import Any, Callable

import torch

from .artifacts import derived_seed
from .config import Protocol
from .metrics import evaluate_batch
from .models import RPLRUModel, SequenceModel
from .task import TaskSpec, generate_batch


@dataclass(frozen=True)
class TrainSpec:
    model: str
    dimension: int
    width: int
    learning_rate: float
    seed: int
    optimizer_updates: int
    batch_size: int
    retention_mode: str = "rp"
    initial_lambda: float = 0.9
    initial_lambda_low: float | None = None
    initial_lambda_high: float | None = None
    all_slow_lambda: float | None = None
    rp_eta_lambda: float | None = None
    rp_eta_gradient: float | None = None
    rp_retention_threshold: float | None = None
    rp_probe_horizon: int | None = None
    retention_learning_rate: float | None = None
    tau_sat: float = 16.64
    fixed_unit_count: int = 0
    fixed_fast_lambda: float = 0.0
    fixed_subset_seed: int = 0
    chrono_t_max: int = 255

    def validate(self) -> None:
        if self.dimension <= 0 or self.width <= 0:
            raise ValueError("dimension and width must be positive")
        if self.learning_rate <= 0 or self.optimizer_updates <= 0:
            raise ValueError("learning rate and optimizer updates must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if self.retention_learning_rate is not None and self.retention_learning_rate <= 0:
            raise ValueError("retention learning rate must be positive")
        if self.tau_sat <= 0:
            raise ValueError("saturation threshold must be positive")
        if not 0 <= self.fixed_unit_count <= self.width:
            raise ValueError("fixed-unit count must be in [0, width]")
        if not 0.0 <= self.fixed_fast_lambda <= 1.0:
            raise ValueError("fixed fast lambda must be in [0, 1]")
        if self.model == "chrono_lstm" and self.chrono_t_max <= 2:
            raise ValueError("chrono LSTM requires T_max > 2")
        if (self.initial_lambda_low is None) != (
            self.initial_lambda_high is None
        ):
            raise ValueError("initial lambda range requires both endpoints")
        if self.initial_lambda_low is not None and not (
            0
            < self.initial_lambda_low
            < self.initial_lambda_high
            < 1
        ):
            raise ValueError("initial lambda range must satisfy 0 < low < high < 1")
        if self.model == "rp_lru" and self.retention_mode == "rp":
            if (
                self.rp_eta_lambda is None
                or self.rp_eta_lambda <= 0
                or self.rp_retention_threshold is None
                or self.rp_retention_threshold < 0
                or self.rp_probe_horizon is None
                or self.rp_probe_horizon <= 0
            ):
                raise ValueError("RP model requires complete RP hyperparameters")
        if self.model == "rp_lru" and self.retention_mode in {
            "rp_grad",
            "direct_lambda_grad",
            "ste_recall_grad",
        }:
            if (
                self.rp_eta_gradient is None
                or self.rp_eta_gradient <= 0
                or self.rp_probe_horizon is None
                or self.rp_probe_horizon <= 0
            ):
                raise ValueError(
                    "recall-gradient model requires eta_gradient and probe horizon"
                )
            if self.rp_eta_lambda is not None or self.rp_retention_threshold is not None:
                raise ValueError(
                    "recall-gradient modes do not use eta_lambda or a threshold"
                )
        if self.retention_mode == "fixed_binary" and self.retention_learning_rate is not None:
            raise ValueError("FixedBinary has no trainable retention parameter")


@dataclass
class TrainingOutput:
    model: SequenceModel
    result: dict[str, Any]
    loss_trace: list[float]
    validation_trace: list[dict[str, Any]]
    rp_trace: list[dict[str, Any]]
    rp_score_trace: list[dict[str, Any]]
    rp_coordinate_trace: list[dict[str, Any]]
    retention_steps: list[int]
    retention_trace: list[list[float]]
    theta_trace: list[list[float]]


def _boundary_probe(
    states: torch.Tensor,
    batch,
    *,
    maximum_pairs: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    selected_states = states[batch.boundary_mask]
    selected_targets = batch.memory_targets[batch.boundary_mask]
    if selected_states.shape[0] != selected_targets.shape[0]:
        raise AssertionError("boundary states and targets differ")
    if selected_states.shape[0] > int(maximum_pairs):
        generator = torch.Generator(device=states.device).manual_seed(int(seed))
        selected_indices = torch.randperm(
            selected_states.shape[0], generator=generator, device=states.device
        )[: int(maximum_pairs)]
        selected_states = selected_states[selected_indices]
        selected_targets = selected_targets[selected_indices]
    else:
        selected_indices = torch.arange(
            selected_states.shape[0], device=states.device
        )
    index_bytes = (
        selected_indices.detach().cpu().to(torch.int64).numpy().tobytes()
    )
    metadata = {
        "available_probe_pairs": int(batch.boundary_mask.sum().item()),
        "selected_probe_pairs": int(selected_states.shape[0]),
        "probe_subsample_seed": int(seed),
        "probe_indices_sha256": hashlib.sha256(index_bytes).hexdigest(),
    }
    return selected_states.detach(), selected_targets.detach(), metadata


def train_model(
    model: SequenceModel,
    protocol: Protocol,
    spec: TrainSpec,
    *,
    device: torch.device | str,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingOutput:
    spec.validate()
    torch.manual_seed(int(spec.seed))
    device_object = torch.device(device)
    model = model.to(device_object)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_groups = model.optimizer_parameter_groups(
        learning_rate=float(spec.learning_rate),
        weight_decay=float(protocol.training.weight_decay),
    )
    scheduled_parameter = (
        model.schedule_task_gradient_parameter()
        if isinstance(model, RPLRUModel)
        else None
    )
    if scheduled_parameter is not None and spec.retention_learning_rate is not None:
        scheduled_id = id(scheduled_parameter)
        adjusted_groups = []
        for group in parameter_groups:
            kept = [parameter for parameter in group["params"] if id(parameter) != scheduled_id]
            if kept:
                adjusted_groups.append({**group, "params": kept})
        adjusted_groups.append(
            {
                "params": [scheduled_parameter],
                "lr": float(spec.retention_learning_rate),
                "weight_decay": float(protocol.training.weight_decay),
                "group_name": "scheduled_retention_parameter",
            }
        )
        parameter_groups = adjusted_groups
    grouped_ids = [
        id(parameter)
        for group in parameter_groups
        for parameter in group["params"]
    ]
    expected_ids = [id(parameter) for parameter in parameters]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != set(
        expected_ids
    ):
        raise AssertionError("optimizer groups must partition trainable parameters")
    optimizer = torch.optim.AdamW(parameter_groups)
    train_task = TaskSpec.training(spec.dimension, spec.batch_size, protocol.task)
    validation_task = TaskSpec.training(
        spec.dimension, protocol.training.validation_trajectories, protocol.task
    )
    validation = generate_batch(
        validation_task,
        protocol.task,
        seed=derived_seed(
            protocol.evaluation.bank_seed, "validation", spec.dimension
        ),
        device=device_object,
    )
    loss_trace: list[float] = []
    validation_trace: list[dict[str, Any]] = []
    rp_trace: list[dict[str, Any]] = []
    rp_score_trace: list[dict[str, Any]] = []
    rp_coordinate_trace: list[dict[str, Any]] = []
    retention_steps: list[int] = []
    retention_trace: list[list[float]] = []
    theta_trace: list[list[float]] = []
    if isinstance(model, RPLRUModel):
        retention_steps.append(0)
        retention_trace.append(
            [float(value) for value in model.retention().detach().cpu()]
        )
        theta_trace.append(
            [float(value) for value in model.retention_logits().detach().cpu()]
        )
    started = time.time()
    model.train()

    for update in range(1, spec.optimizer_updates + 1):
        batch = generate_batch(
            train_task,
            protocol.task,
            seed=derived_seed(spec.seed, "online_train", update),
            device=device_object,
            audit=False,
        )
        prediction = model(batch.inputs, batch.lengths)
        loss = (prediction - batch.final_targets).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite task loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if scheduled_parameter is not None and update <= protocol.rp.warmup_updates:
            scheduled_parameter.grad = None
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            float(protocol.training.gradient_clip_norm),
            error_if_nonfinite=True,
        )
        optimizer.step()
        loss_trace.append(float(loss.detach().cpu()))

        rp_due = (
            isinstance(model, RPLRUModel)
            and model.probe_update_enabled
            and update > protocol.rp.warmup_updates
            and (update - protocol.rp.warmup_updates)
            % protocol.rp.interval_updates
            == 0
        )
        if rp_due:
            model.eval()
            with torch.no_grad():
                states = model.forward_states(batch.inputs)
                probe_states, probe_targets, probe_metadata = _boundary_probe(
                    states,
                    batch,
                    maximum_pairs=protocol.rp.probe_pairs,
                    seed=derived_seed(spec.seed, "rp_probe", update),
                )
            if model.rp_enabled:
                rp_result, score_diagnostics = model.rp_update(
                    probe_states,
                    probe_targets,
                    horizon=int(spec.rp_probe_horizon),
                    eta_lambda=float(spec.rp_eta_lambda),
                    retention_threshold=float(spec.rp_retention_threshold),
                    normalization_epsilon=protocol.rp.normalization_epsilon,
                    return_diagnostics=True,
                )
                rp_score_trace.append(
                    {
                        "update": int(update),
                        "retention_threshold": float(
                            spec.rp_retention_threshold
                        ),
                        "eta_lambda": float(spec.rp_eta_lambda),
                        **{
                            name: value.detach().cpu()
                            for name, value in score_diagnostics.items()
                        },
                    }
                )
            elif model.rp_grad_enabled:
                rp_result, coordinate_diagnostics = model.rp_grad_update(
                    probe_states,
                    probe_targets,
                    horizon=int(spec.rp_probe_horizon),
                    eta_gradient=float(spec.rp_eta_gradient),
                )
                rp_coordinate_trace.append(
                    {
                        "update": int(update),
                        **{
                            name: value.detach().cpu()
                            for name, value in coordinate_diagnostics.items()
                        },
                    }
                )
            else:
                raise AssertionError("unknown probe-update mode")
            rp_trace.append(
                {
                    "update": update,
                    "update_rule": model.retention_mode,
                    **probe_metadata,
                    **rp_result.as_dict(),
                }
            )
            retention_steps.append(update)
            retention_trace.append(
                [float(value) for value in model.retention().detach().cpu()]
            )
            theta_trace.append(
                [float(value) for value in model.retention_logits().detach().cpu()]
            )
            model.train()

        retention_log_due = (
            isinstance(model, RPLRUModel)
            and update % protocol.rp.interval_updates == 0
        )
        if retention_log_due and retention_steps[-1] != update:
            retention_steps.append(update)
            retention_trace.append(
                [float(value) for value in model.retention().detach().cpu()]
            )
            theta_trace.append(
                [float(value) for value in model.retention_logits().detach().cpu()]
            )

        validate = (
            update == 1
            or update == spec.optimizer_updates
            or update % protocol.training.validation_interval == 0
        )
        if validate:
            model.eval()
            evaluation = evaluate_batch(model, validation)
            row = {
                "update": update,
                "task_loss": loss_trace[-1],
                "gradient_norm": float(gradient_norm.detach().cpu()),
                **evaluation.metrics,
            }
            validation_trace.append(row)
            if progress is not None:
                progress(
                    {
                        "status": (
                            "training_complete"
                            if update == spec.optimizer_updates
                            else "running"
                        ),
                        "spec": asdict(spec),
                        "update": update,
                        "validation_nmse": row["nmse"],
                        "validation_mse_per_coordinate": row[
                            "mse_per_coordinate"
                        ],
                        "rp_calls": len(rp_trace),
                        "elapsed_seconds": time.time() - started,
                    }
                )
            model.train()

    if isinstance(model, RPLRUModel) and retention_steps[-1] != spec.optimizer_updates:
        retention_steps.append(spec.optimizer_updates)
        retention_trace.append(
            [float(value) for value in model.retention().detach().cpu()]
        )
        theta_trace.append(
            [float(value) for value in model.retention_logits().detach().cpu()]
        )
    final_validation = validation_trace[-1]
    result = {
        "schema_version": 1,
        "status": "complete",
        "train_spec": asdict(spec),
        "final_validation": final_validation,
        "minimum_validation_nmse": min(
            row["nmse"] for row in validation_trace
        ),
        "rp_calls": len(rp_trace),
        "elapsed_seconds": time.time() - started,
    }
    return TrainingOutput(
        model=model,
        result=result,
        loss_trace=loss_trace,
        validation_trace=validation_trace,
        rp_trace=rp_trace,
        rp_score_trace=rp_score_trace,
        rp_coordinate_trace=rp_coordinate_trace,
        retention_steps=retention_steps,
        retention_trace=retention_trace,
        theta_trace=theta_trace,
    )
