"""Evaluate the project-level engineering benefits of recurrent memory.

This module is intentionally separate from the Ságodi-matched primary
analysis.  It measures temporal and velocity-scale OOD integration and
state-perturbation retention, has no binary CA gate, and never interprets a
perturbation trajectory as manifold recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .artifacts import (
    atomic_bytes,
    atomic_json,
    canonical_hash,
    derived_seed,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from .models import ProtocolModel, load_checkpoint
from .state import StateAdapter
from .tasks import angular_integration


SCHEMA_VERSION = 1
ANALYSIS_NAME = "calru_engineering_benefit_single_checkpoint_v1"
ANALYSIS_ROLE = "project_engineering_utility_not_sagodi_matched_analysis"
MODEL_IDS = (
    "rnn_param206",
    "gru_sagodi_param135",
    "lstm_param109",
    "lru_param96",
    "no_rp",
    "ca_lru",
)
MODEL_SEEDS = tuple(range(10))
FULL_TRIALS = 256
FULL_T = 256
TEMPORAL_MULTIPLIERS = (1, 2, 4, 8, 16)
VELOCITY_SCALES = (1.0, 2.0, 4.0)
PERTURBATION_MAGNITUDES = (0.0, 0.01, 0.1, 1.0)
PERTURBATION_HORIZON_MULTIPLIERS = (1, 4, 16)


class EngineeringFreezeError(ValueError):
    """The declarative engineering-benefit freeze differs from v1."""


@dataclass(frozen=True)
class EngineeringBenefitSpec:
    trial_count: int
    task_horizon: int
    task_base_seed: int
    temporal_multipliers: tuple[int, ...]
    velocity_scales: tuple[float, ...]
    perturbation_magnitudes: tuple[float, ...]
    perturbation_horizon_multipliers: tuple[int, ...]
    perturbation_direction_seed: int
    dt: float
    gp_length_scale: float
    gp_std: float
    gp_jitter: float
    radius_epsilon: float
    smoke: bool

    @property
    def maximum_horizon(self) -> int:
        return self.task_horizon * max(self.temporal_multipliers)


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise EngineeringFreezeError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def load_engineering_freeze(
    path: Path | str, *, smoke: bool = False
) -> tuple[dict[str, Any], EngineeringBenefitSpec]:
    """Load and fail-closed validate the exact v1 declarative freeze."""

    payload = strict_json_load(path)
    if not isinstance(payload, Mapping):
        raise EngineeringFreezeError("freeze must be a JSON object")
    _require_keys(
        payload,
        {
            "schema_version",
            "freeze_id",
            "analysis_role",
            "claim_policy",
            "parent_contract",
            "evaluation",
            "smoke_override",
        },
        "freeze",
    )
    if payload.get("schema_version") != 1:
        raise EngineeringFreezeError("schema_version must be 1")
    if payload.get("freeze_id") != "calru_engineering_benefit_v1":
        raise EngineeringFreezeError("freeze_id changed")
    if payload.get("analysis_role") != ANALYSIS_ROLE:
        raise EngineeringFreezeError("analysis role changed")
    expected_claim = {
        "binary_continuous_attractor_gates": False,
        "expected_direction_pass_thresholds": False,
        "model_or_seed_exclusion": "forbidden",
        "primary_outputs": "individual_errors_and_descriptive_summaries",
        "sagodi_metric_association": "deferred_join_on_model_id_and_model_seed",
    }
    if payload.get("claim_policy") != expected_claim:
        raise EngineeringFreezeError("claim policy changed")
    expected_parent = {
        "campaign_type": "sagodi_primary_main_v3",
        "required_completed_training_runs": 60,
        "model_order": list(MODEL_IDS),
        "model_seeds": list(MODEL_SEEDS),
        "checkpoint_selection": "final_update_5000",
        "failed_seed_replacement_policy": "forbidden",
    }
    if payload.get("parent_contract") != expected_parent:
        raise EngineeringFreezeError("parent campaign contract changed")

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise EngineeringFreezeError("evaluation must be an object")
    _require_keys(
        evaluation,
        {
            "trial_count",
            "task_horizon",
            "task_base_seed",
            "task",
            "temporal_ood",
            "velocity_scale_ood",
            "state_perturbation_retention",
            "numerics",
        },
        "evaluation",
    )
    task = evaluation.get("task")
    temporal = evaluation.get("temporal_ood")
    velocity = evaluation.get("velocity_scale_ood")
    perturbation = evaluation.get("state_perturbation_retention")
    numerics = evaluation.get("numerics")
    if not all(
        isinstance(item, Mapping)
        for item in (task, temporal, velocity, perturbation, numerics)
    ):
        raise EngineeringFreezeError("evaluation sub-blocks must be objects")
    if task != {
        "name": "angular_integration",
        "dimensions": 1,
        "init_mode": "hidden-init",
        "delta_t": 0.1,
        "gp_length_scale": 1.0,
        "gp_standard_deviation": 1.0,
        "gp_cholesky_jitter": 1e-6,
        "block_construction": (
            "sixteen_independent_deterministic_in_distribution_256_step_"
            "GP_velocity_blocks"
        ),
        "angle_construction": (
            "one_continuous_angle_integrated_across_concatenated_blocks"
        ),
        "prefix_pairing": True,
    }:
        raise EngineeringFreezeError("task construction changed")
    if temporal != {
        "length_multipliers": list(TEMPORAL_MULTIPLIERS),
        "execution": "one_causal_16T_rollout_with_nested_prefix_readouts",
        "individual_metrics": [
            "final_absolute_circular_error_radians",
            "prefix_mean_absolute_circular_error_radians",
        ],
    }:
        raise EngineeringFreezeError("temporal OOD contract changed")
    if velocity != {
        "scales": list(VELOCITY_SCALES),
        "base_bank": "temporal_ood_first_block",
        "scale_one_execution": "reuse_temporal_ood_first_T_prefix",
        "target_rule": "integrate_scaled_velocity_from_the_same_initial_angle",
        "individual_metrics": [
            "final_absolute_circular_error_radians",
            "sequence_mean_absolute_circular_error_radians",
        ],
    }:
        raise EngineeringFreezeError("velocity-scale OOD contract changed")
    if perturbation != {
        "endpoint_source": "noise_free_first_task_block_endpoint",
        "state_space": "full_primary_Markov_state",
        "relative_rms_magnitudes": list(PERTURBATION_MAGNITUDES),
        "blank_horizon_multipliers": list(PERTURBATION_HORIZON_MULTIPLIERS),
        "direction_distribution": (
            "isotropic_standard_normal_normalized_to_unit_coordinate_RMS_per_trial"
        ),
        "direction_seed": 918273,
        "magnitude_rule": (
            "perturbation_coordinate_RMS_equals_relative_magnitude_times_"
            "endpoint_state_coordinate_RMS_per_trial"
        ),
        "references": [
            "pre_perturbation_decoded_memory",
            "clean_paired_blank_trajectory",
        ],
        "interpretation": "engineering_robustness_not_manifold_recovery",
    }:
        raise EngineeringFreezeError("state-perturbation contract changed")
    if numerics != {
        "model_evaluation_noise": False,
        "torch_gradients": False,
        "stored_float_dtype": "float32",
        "angle_decoder": "atan2_sin_cos",
        "circular_error_range": "zero_to_pi",
        "utility_scalar_nonfinite_policy": (
            "null_unless_all_registered_trial_values_are_finite"
        ),
        "angle_utility_radius_policy": (
            "null_unless_all_corresponding_estimate_and_reference_radii_are_"
            "finite_and_at_least_epsilon"
        ),
        "undefined_angle_radius_reporting_epsilon": 1e-12,
    }:
        raise EngineeringFreezeError("numeric contract changed")
    if int(evaluation.get("trial_count", -1)) != FULL_TRIALS:
        raise EngineeringFreezeError("full trial_count must be 256")
    if int(evaluation.get("task_horizon", -1)) != FULL_T:
        raise EngineeringFreezeError("full task_horizon must be 256")
    smoke_block = payload.get("smoke_override")
    expected_smoke = {
        "trial_count": 8,
        "task_horizon": 8,
        "temporal_length_multipliers": list(TEMPORAL_MULTIPLIERS),
        "velocity_scales": list(VELOCITY_SCALES),
        "perturbation_relative_rms_magnitudes": list(PERTURBATION_MAGNITUDES),
        "perturbation_blank_horizon_multipliers": list(
            PERTURBATION_HORIZON_MULTIPLIERS
        ),
    }
    if smoke_block != expected_smoke:
        raise EngineeringFreezeError("smoke override changed")
    trial_count = int(smoke_block["trial_count"] if smoke else evaluation["trial_count"])
    task_horizon = int(
        smoke_block["task_horizon"] if smoke else evaluation["task_horizon"]
    )
    spec = EngineeringBenefitSpec(
        trial_count=trial_count,
        task_horizon=task_horizon,
        task_base_seed=int(evaluation["task_base_seed"]),
        temporal_multipliers=TEMPORAL_MULTIPLIERS,
        velocity_scales=VELOCITY_SCALES,
        perturbation_magnitudes=PERTURBATION_MAGNITUDES,
        perturbation_horizon_multipliers=PERTURBATION_HORIZON_MULTIPLIERS,
        perturbation_direction_seed=int(perturbation["direction_seed"]),
        dt=float(task["delta_t"]),
        gp_length_scale=float(task["gp_length_scale"]),
        gp_std=float(task["gp_standard_deviation"]),
        gp_jitter=float(task["gp_cholesky_jitter"]),
        radius_epsilon=float(numerics["undefined_angle_radius_reporting_epsilon"]),
        smoke=bool(smoke),
    )
    return dict(payload), spec


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _circular_absolute_error(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = torch.atan2(torch.sin(estimate - target), torch.cos(estimate - target))
    return difference.abs()


def _decode_angle_and_radius(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if output.ndim != 2 or output.shape[-1] != 2:
        raise ValueError("engineering evaluation requires output [batch,2] as (cos,sin)")
    return torch.atan2(output[:, 1], output[:, 0]), torch.linalg.vector_norm(output, dim=-1)


def _finite_summary(value: np.ndarray) -> dict[str, int | float | None]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, int | float | None] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(array.size - finite.size),
        "finite_fraction": float(finite.size / array.size) if array.size else 0.0,
    }
    if not finite.size:
        result.update(
            {
                "mean": None,
                "std": None,
                "median": None,
                "q05": None,
                "q95": None,
                "min": None,
                "max": None,
            }
        )
        return result
    result.update(
        {
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=0)),
            "median": float(np.median(finite)),
            "q05": float(np.quantile(finite, 0.05)),
            "q95": float(np.quantile(finite, 0.95)),
            "min": float(finite.min()),
            "max": float(finite.max()),
        }
    )
    return result


def _registered_utility_mean(
    summary: Mapping[str, int | float | None],
    *,
    corresponding_radii: Sequence[np.ndarray] = (),
    radius_epsilon: float | None = None,
) -> float | None:
    """Return a scalar only when the registered denominator is fully finite.

    The descriptive summary intentionally preserves a finite-only mean for
    diagnosis.  It must never become the cross-seed utility scalar when even
    one registered trial failed numerically.
    """

    if int(summary["count"]) <= 0 or int(summary["nonfinite_count"]) != 0:
        return None
    if corresponding_radii and radius_epsilon is None:
        raise ValueError("radius_epsilon is required with corresponding radii")
    for radius in corresponding_radii:
        values = np.asarray(radius, dtype=np.float64)
        if not bool(np.isfinite(values).all()) or bool(
            np.any(values < float(radius_epsilon))
        ):
            return None
    value = summary["mean"]
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _generate_concatenated_gp_bank(
    spec: EngineeringBenefitSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Generate independent T-step GP blocks and one continuously integrated angle."""

    # Generate and integrate on CPU float64.  Besides avoiding a 4096x4096 GP
    # factorization, this keeps the target construction deterministic on CUDA,
    # where cumsum does not provide a deterministic kernel.
    blocks: list[torch.Tensor] = []
    initial_angle: torch.Tensor | None = None
    block_seeds: list[int] = []
    for index in range(max(spec.temporal_multipliers)):
        batch = angular_integration(
            spec.trial_count,
            spec.task_base_seed,
            dimensions=1,
            init_mode="hidden-init",
            horizon=spec.task_horizon,
            dt=spec.dt,
            gp_length_scale=spec.gp_length_scale,
            gp_std=spec.gp_std,
            gp_jitter=spec.gp_jitter,
            stream_key=("engineering_benefit_v1", "temporal_block", index),
            dtype=torch.float64,
            device="cpu",
        )
        blocks.append(batch.inputs)
        block_seeds.append(int(batch.metadata["derived_seed"]))
        if initial_angle is None:
            initial_angle = torch.as_tensor(
                batch.metadata["initial_latents"],
                device="cpu",
                dtype=torch.float64,
            )[:, 0]
    if initial_angle is None:  # pragma: no cover - frozen block count is positive
        raise RuntimeError("no GP blocks were generated")
    velocity_cpu = torch.cat(blocks, dim=0)
    target_angle_cpu = initial_angle[None, :] + spec.dt * torch.cumsum(
        velocity_cpu[:, :, 0], dim=0
    )
    initial_memory_cpu = torch.stack(
        (torch.cos(initial_angle), torch.sin(initial_angle)), dim=-1
    )
    velocity = velocity_cpu.to(device=device, dtype=dtype)
    target_angle = target_angle_cpu.to(device=device, dtype=dtype)
    initial_memory = initial_memory_cpu.to(device=device, dtype=dtype)
    bank = {
        "construction": "independent_T_step_GP_blocks_continuous_angle",
        "block_derived_seeds": block_seeds,
        "velocity_sha256": _array_sha256(
            velocity_cpu.numpy().astype(np.float32, copy=False)
        ),
        "initial_angle_sha256": _array_sha256(
            initial_angle.numpy().astype(np.float32, copy=False)
        ),
    }
    return velocity, target_angle, initial_memory, bank


@torch.no_grad()
def _rollout_angles(
    model: ProtocolModel,
    inputs: torch.Tensor,
    initial_memory: torch.Tensor,
    target_angle: torch.Tensor,
    *,
    endpoint_step: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    reported = model.initial_state(
        inputs.shape[1], inputs.device, initial_memory=initial_memory
    )
    errors = torch.empty(
        inputs.shape[0], inputs.shape[1], device=inputs.device, dtype=inputs.dtype
    )
    radii = torch.empty_like(errors)
    endpoint: torch.Tensor | None = None
    for index, token in enumerate(inputs):
        reported = model.step(token, reported)
        angle, radius = _decode_angle_and_radius(model.decode(reported))
        errors[index] = _circular_absolute_error(angle, target_angle[index])
        radii[index] = radius
        if endpoint_step is not None and index + 1 == int(endpoint_step):
            endpoint = reported.detach().clone()
    return errors, radii, endpoint


def _relative_rms_perturbations(
    primary: torch.Tensor,
    magnitudes: Sequence[float],
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [magnitude,batch,state] perturbations with exact relative RMS."""

    primary_cpu = primary.detach().cpu().numpy().astype(np.float64, copy=False)
    rng = np.random.default_rng(int(seed))
    direction = rng.standard_normal(size=primary_cpu.shape)
    direction_rms = np.sqrt(np.mean(np.square(direction), axis=1, keepdims=True))
    direction /= np.maximum(direction_rms, np.finfo(np.float64).tiny)
    state_rms = np.sqrt(np.mean(np.square(primary_cpu), axis=1, keepdims=True))
    perturbations = np.stack(
        [float(magnitude) * state_rms * direction for magnitude in magnitudes], axis=0
    )
    denominator = np.linalg.norm(primary_cpu, axis=1)
    realized = np.empty((len(magnitudes), primary_cpu.shape[0]), dtype=np.float64)
    for index, delta in enumerate(perturbations):
        delta_norm = np.linalg.norm(delta, axis=1)
        realized[index] = np.divide(
            delta_norm,
            denominator,
            out=np.zeros_like(delta_norm),
            where=denominator > 0,
        )
    return (
        torch.as_tensor(perturbations, device=primary.device, dtype=primary.dtype),
        torch.as_tensor(realized, device=primary.device, dtype=primary.dtype),
    )


@torch.no_grad()
def _perturbation_retention(
    model: ProtocolModel,
    adapter: StateAdapter,
    endpoint_reported: torch.Tensor,
    spec: EngineeringBenefitSpec,
    *,
    model_seed: int,
) -> dict[str, np.ndarray]:
    batch = endpoint_reported.shape[0]
    primary = adapter.primary_from_reported(endpoint_reported)
    pre_angle, pre_radius = _decode_angle_and_radius(model.decode(endpoint_reported))
    direction_seed = derived_seed(
        spec.perturbation_direction_seed,
        "engineering_benefit_v1",
        "state_perturbation",
        int(model_seed),
    )
    perturbation, realized = _relative_rms_perturbations(
        primary, spec.perturbation_magnitudes, seed=direction_seed
    )
    # Group zero is the clean paired blank trajectory.  Groups 1..M are the
    # registered perturbation magnitudes, including a separately executed 0.
    grouped_primary = torch.cat(
        [primary[None, :, :], primary[None, :, :] + perturbation], dim=0
    )
    group_count = grouped_primary.shape[0]
    reported = adapter.reported_from_primary(
        grouped_primary.reshape(group_count * batch, primary.shape[-1])
    )
    horizon_steps = tuple(
        spec.task_horizon * value for value in spec.perturbation_horizon_multipliers
    )
    memory_error = torch.empty(
        len(spec.perturbation_magnitudes),
        len(horizon_steps),
        batch,
        device=primary.device,
        dtype=primary.dtype,
    )
    paired_error = torch.empty_like(memory_error)
    output_radius = torch.empty_like(memory_error)
    clean_output_radius = torch.empty(
        len(horizon_steps),
        batch,
        device=primary.device,
        dtype=primary.dtype,
    )
    blank = adapter.zero_input(reported)
    horizon_index = {step: index for index, step in enumerate(horizon_steps)}
    for step in range(1, max(horizon_steps) + 1):
        reported = model.step(blank, reported)
        if step not in horizon_index:
            continue
        output = model.decode(reported).reshape(group_count, batch, 2)
        angles = torch.atan2(output[:, :, 1], output[:, :, 0])
        radii = torch.linalg.vector_norm(output, dim=-1)
        target_index = horizon_index[step]
        clean_output_radius[target_index] = radii[0]
        for magnitude_index in range(len(spec.perturbation_magnitudes)):
            perturbed_angle = angles[magnitude_index + 1]
            memory_error[magnitude_index, target_index] = _circular_absolute_error(
                perturbed_angle, pre_angle
            )
            paired_error[magnitude_index, target_index] = _circular_absolute_error(
                perturbed_angle, angles[0]
            )
            output_radius[magnitude_index, target_index] = radii[magnitude_index + 1]
    to_numpy = lambda tensor: tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    return {
        "perturbation_relative_rms_magnitudes": np.asarray(
            spec.perturbation_magnitudes, dtype=np.float32
        ),
        "perturbation_blank_horizon_steps": np.asarray(horizon_steps, dtype=np.int64),
        "perturbation_memory_absolute_error": to_numpy(memory_error),
        "perturbation_clean_paired_absolute_error": to_numpy(paired_error),
        "perturbation_output_radius": to_numpy(output_radius),
        "perturbation_clean_output_radius": to_numpy(clean_output_radius),
        "perturbation_realized_relative_l2": to_numpy(realized),
        "perturbation_pre_memory_angle": to_numpy(pre_angle),
        "perturbation_pre_memory_output_radius": to_numpy(pre_radius),
        "perturbation_direction_seed": np.asarray(direction_seed, dtype=np.int64),
    }


def _summarize_results(
    arrays: Mapping[str, np.ndarray], spec: EngineeringBenefitSpec
) -> tuple[dict[str, Any], dict[str, float | None]]:
    temporal: dict[str, Any] = {}
    velocity: dict[str, Any] = {}
    perturbation: dict[str, Any] = {}
    utility: dict[str, float | None] = {}
    for index, multiplier in enumerate(spec.temporal_multipliers):
        label = f"{multiplier}T"
        horizon_steps = spec.task_horizon * multiplier
        final = arrays["temporal_final_absolute_error"][index]
        prefix = arrays["temporal_prefix_mean_absolute_error"][index]
        final_radius = arrays["temporal_output_radius"][horizon_steps - 1]
        prefix_radius = arrays["temporal_output_radius"][:horizon_steps]
        temporal[label] = {
            "horizon_steps": int(horizon_steps),
            "final_absolute_circular_error_radians": _finite_summary(final),
            "prefix_mean_absolute_circular_error_radians": _finite_summary(prefix),
            "final_output_radius": _finite_summary(final_radius),
            "prefix_output_radius": _finite_summary(prefix_radius),
        }
        utility[f"temporal/{label}/final_mean_error_radians"] = (
            _registered_utility_mean(
                temporal[label]["final_absolute_circular_error_radians"],
                corresponding_radii=(final_radius,),
                radius_epsilon=spec.radius_epsilon,
            )
        )
        utility[f"temporal/{label}/prefix_mean_error_radians"] = (
            _registered_utility_mean(
                temporal[label]["prefix_mean_absolute_circular_error_radians"],
                corresponding_radii=(prefix_radius,),
                radius_epsilon=spec.radius_epsilon,
            )
        )
    for index, scale in enumerate(spec.velocity_scales):
        label = f"{scale:g}x"
        final = arrays["velocity_final_absolute_error"][index]
        sequence = arrays["velocity_sequence_mean_absolute_error"][index]
        final_radius = arrays["velocity_output_radius"][index, -1]
        sequence_radius = arrays["velocity_output_radius"][index]
        velocity[label] = {
            "scale": float(scale),
            "final_absolute_circular_error_radians": _finite_summary(final),
            "sequence_mean_absolute_circular_error_radians": _finite_summary(sequence),
            "final_output_radius": _finite_summary(final_radius),
            "sequence_output_radius": _finite_summary(sequence_radius),
        }
        utility[f"velocity/{label}/final_mean_error_radians"] = (
            _registered_utility_mean(
                velocity[label]["final_absolute_circular_error_radians"],
                corresponding_radii=(final_radius,),
                radius_epsilon=spec.radius_epsilon,
            )
        )
        utility[f"velocity/{label}/sequence_mean_error_radians"] = (
            _registered_utility_mean(
                velocity[label]["sequence_mean_absolute_circular_error_radians"],
                corresponding_radii=(sequence_radius,),
                radius_epsilon=spec.radius_epsilon,
            )
        )
    for magnitude_index, magnitude in enumerate(spec.perturbation_magnitudes):
        magnitude_label = f"relative_rms_{magnitude:g}"
        perturbation[magnitude_label] = {}
        for horizon_index, multiplier in enumerate(
            spec.perturbation_horizon_multipliers
        ):
            horizon_label = f"{multiplier}T"
            memory = arrays["perturbation_memory_absolute_error"][
                magnitude_index, horizon_index
            ]
            paired = arrays["perturbation_clean_paired_absolute_error"][
                magnitude_index, horizon_index
            ]
            perturbed_radius = arrays["perturbation_output_radius"][
                magnitude_index, horizon_index
            ]
            clean_radius = arrays["perturbation_clean_output_radius"][horizon_index]
            pre_radius = arrays["perturbation_pre_memory_output_radius"]
            perturbation[magnitude_label][horizon_label] = {
                "horizon_steps": int(spec.task_horizon * multiplier),
                "memory_absolute_circular_error_radians": _finite_summary(memory),
                "clean_paired_absolute_circular_error_radians": _finite_summary(paired),
                "perturbed_output_radius": _finite_summary(perturbed_radius),
                "clean_paired_output_radius": _finite_summary(clean_radius),
                "pre_memory_output_radius": _finite_summary(pre_radius),
            }
            stem = f"perturbation/{magnitude_label}/{horizon_label}"
            utility[f"{stem}/memory_mean_error_radians"] = (
                _registered_utility_mean(
                    perturbation[magnitude_label][horizon_label][
                        "memory_absolute_circular_error_radians"
                    ],
                    corresponding_radii=(perturbed_radius, pre_radius),
                    radius_epsilon=spec.radius_epsilon,
                )
            )
            utility[f"{stem}/clean_paired_mean_error_radians"] = (
                _registered_utility_mean(
                    perturbation[magnitude_label][horizon_label][
                        "clean_paired_absolute_circular_error_radians"
                    ],
                    corresponding_radii=(perturbed_radius, clean_radius),
                    radius_epsilon=spec.radius_epsilon,
                )
            )
    return (
        {
            "temporal_ood": temporal,
            "velocity_scale_ood": velocity,
            "state_perturbation_retention": perturbation,
        },
        utility,
    )


def _initialize_output(destination: Path, identity_payload: Mapping[str, Any]) -> str:
    identity = canonical_hash(identity_payload)
    normalized_identity_payload = json.loads(
        json.dumps(
            dict(identity_payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    marker_payload = {
        "schema_version": 1,
        "analysis": ANALYSIS_NAME,
        "analysis_identity": identity,
        "identity_payload": normalized_identity_payload,
    }
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".engineering_benefit_root.json"
    if marker.exists():
        if strict_json_load(marker) != marker_payload:
            raise RuntimeError("engineering output marker differs; choose a new directory")
    else:
        if any(destination.iterdir()):
            raise RuntimeError("engineering output directory is nonempty without a marker")
        atomic_json(marker, marker_payload)
    return identity


def run_engineering_benefit(
    *,
    checkpoint_path: Path | str,
    freeze_path: Path | str,
    output_dir: Path | str,
    model_id: str,
    model_seed: int,
    device: str = "cuda:0",
    smoke: bool = False,
) -> Path:
    """Run or resume the complete engineering evaluation for one checkpoint."""

    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    freeze_source = Path(freeze_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    if model_id not in MODEL_IDS:
        raise ValueError(f"model_id must be one of {MODEL_IDS}")
    if int(model_seed) not in MODEL_SEEDS and not smoke:
        raise ValueError("full engineering evaluation model_seed must be 0..9")
    freeze, spec = load_engineering_freeze(freeze_source, smoke=smoke)
    identity_payload = {
        "analysis": ANALYSIS_NAME,
        "schema_version": SCHEMA_VERSION,
        "analysis_role": ANALYSIS_ROLE,
        "checkpoint_sha256": sha256_file(checkpoint),
        "freeze_file_sha256": sha256_file(freeze_source),
        "freeze_canonical_fingerprint": canonical_hash(freeze),
        "model_id": model_id,
        "model_seed": int(model_seed),
        "spec": asdict(spec),
    }
    identity = _initialize_output(destination, identity_payload)
    job_id = f"engineering-benefit-{model_id}-seed{int(model_seed)}-{identity[:12]}"
    completion = destination / "completion_receipt.json"
    if completion.is_file():
        valid, reason = verify_completion_receipt(
            completion,
            expected_job_id=job_id,
            expected_metadata={
                "analysis_identity": identity,
                "analysis_role": ANALYSIS_ROLE,
                "model_id": model_id,
                "model_seed": int(model_seed),
            },
        )
        if not valid:
            raise RuntimeError(f"existing engineering completion is invalid: {reason}")
        receipt = strict_json_load(completion)
        required = {
            "engineering_benefit_freeze.json",
            "individual_errors.npz",
            "manifest.json",
            "summary.json",
            "COMPLETE",
        }
        if not isinstance(receipt.get("artifacts"), Mapping) or not required.issubset(
            receipt["artifacts"]
        ):
            raise RuntimeError("existing engineering receipt lacks required artifacts")
        return destination / "summary.json"

    target_device = torch.device(device)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model, checkpoint_payload = load_checkpoint(checkpoint, target_device)
    model.eval()
    if model.config.name != model_id:
        raise ValueError("checkpoint model id differs from requested join key")
    extra = checkpoint_payload.get("extra")
    if not isinstance(extra, Mapping) or int(extra.get("model_seed", -1)) != int(
        model_seed
    ):
        raise ValueError("checkpoint model seed differs from requested join key")
    if model.input_dim != 1 or model.output_dim != 2 or model.config.init_mode != "hidden_init":
        raise ValueError("checkpoint is not a hidden-init one-angle integration model")
    adapter = StateAdapter(model.core)
    dtype = next(model.parameters()).dtype
    freeze_copy = destination / "engineering_benefit_freeze.json"
    if freeze_copy.exists() and freeze_copy.read_bytes() != freeze_source.read_bytes():
        raise RuntimeError("copied engineering freeze differs")
    if not freeze_copy.exists():
        atomic_bytes(freeze_copy, freeze_source.read_bytes())
    atomic_json(
        destination / "progress.json",
        {
            "schema_version": 1,
            "analysis_identity": identity,
            "stage": "deterministic_evaluation",
            "updated_unix_seconds": time.time(),
        },
    )

    velocity, temporal_target, initial_memory, bank = _generate_concatenated_gp_bank(
        spec, device=target_device, dtype=dtype
    )
    temporal_error, temporal_radius, endpoint = _rollout_angles(
        model,
        velocity,
        initial_memory,
        temporal_target,
        endpoint_step=spec.task_horizon,
    )
    if endpoint is None:
        raise RuntimeError("first task-block endpoint was not captured")
    temporal_steps = np.asarray(
        [spec.task_horizon * value for value in spec.temporal_multipliers],
        dtype=np.int64,
    )
    temporal_error_np = temporal_error.detach().cpu().numpy().astype(np.float32, copy=False)
    temporal_radius_np = temporal_radius.detach().cpu().numpy().astype(np.float32, copy=False)
    temporal_final = np.stack(
        [temporal_error_np[step - 1] for step in temporal_steps], axis=0
    )
    temporal_prefix_mean = np.stack(
        [np.mean(temporal_error_np[:step], axis=0) for step in temporal_steps], axis=0
    ).astype(np.float32, copy=False)

    velocity_error = np.empty(
        (len(spec.velocity_scales), spec.task_horizon, spec.trial_count),
        dtype=np.float32,
    )
    velocity_radius = np.empty_like(velocity_error)
    velocity_error[0] = temporal_error_np[: spec.task_horizon]
    velocity_radius[0] = temporal_radius_np[: spec.task_horizon]
    base_velocity = velocity[: spec.task_horizon]
    initial_angle = torch.atan2(initial_memory[:, 1], initial_memory[:, 0])
    for scale_index, scale in enumerate(spec.velocity_scales[1:], start=1):
        scaled_input = base_velocity * float(scale)
        scaled_target_cpu = initial_angle.detach().cpu().to(torch.float64)[
            None, :
        ] + spec.dt * torch.cumsum(
            scaled_input[:, :, 0].detach().cpu().to(torch.float64), dim=0
        )
        scaled_target = scaled_target_cpu.to(device=target_device, dtype=dtype)
        error, radius, _ = _rollout_angles(
            model, scaled_input, initial_memory, scaled_target
        )
        velocity_error[scale_index] = error.detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        velocity_radius[scale_index] = radius.detach().cpu().numpy().astype(
            np.float32, copy=False
        )
    velocity_final = velocity_error[:, -1, :]
    velocity_sequence_mean = np.mean(velocity_error, axis=1).astype(
        np.float32, copy=False
    )
    perturbation_arrays = _perturbation_retention(
        model, adapter, endpoint, spec, model_seed=int(model_seed)
    )
    arrays: dict[str, np.ndarray] = {
        "temporal_horizon_steps": temporal_steps,
        "temporal_absolute_error": temporal_error_np,
        "temporal_output_radius": temporal_radius_np,
        "temporal_final_absolute_error": temporal_final,
        "temporal_prefix_mean_absolute_error": temporal_prefix_mean,
        "velocity_scales": np.asarray(spec.velocity_scales, dtype=np.float32),
        "velocity_absolute_error": velocity_error,
        "velocity_output_radius": velocity_radius,
        "velocity_final_absolute_error": velocity_final,
        "velocity_sequence_mean_absolute_error": velocity_sequence_mean,
        "initial_angle": initial_angle.detach().cpu().numpy().astype(np.float32, copy=False),
        **perturbation_arrays,
    }
    summaries, utility_metrics = _summarize_results(arrays, spec)
    arrays_path = destination / "individual_errors.npz"
    _atomic_npz(arrays_path, **arrays)
    join_key = f"{model_id}::seed{int(model_seed):02d}"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_NAME,
        "analysis_role": ANALYSIS_ROLE,
        "analysis_identity": identity,
        "smoke": bool(smoke),
        "model_id": model_id,
        "model_seed": int(model_seed),
        "join_key": join_key,
        "main_run_id": f"primary_main__{model_id}__seed{int(model_seed):02d}",
        "checkpoint_sha256": sha256_file(checkpoint),
        "freeze_file_sha256": sha256_file(freeze_copy),
        "freeze_canonical_fingerprint": canonical_hash(freeze),
        "spec": asdict(spec),
        "task_bank": bank,
        "primary_state_dimension": int(adapter.primary_dim),
        "trial_count": int(spec.trial_count),
        "binary_ca_gates": False,
        "expected_direction_pass_thresholds": False,
        "utility_scalar_nonfinite_policy": (
            "null_unless_all_registered_trial_values_are_finite"
        ),
        "angle_utility_radius_policy": (
            "null_unless_all_corresponding_estimate_and_reference_radii_are_"
            "finite_and_at_least_epsilon"
        ),
        "interpretation": "engineering_robustness_not_manifold_recovery",
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_NAME,
        "analysis_role": ANALYSIS_ROLE,
        "analysis_identity": identity,
        "smoke": bool(smoke),
        "model_id": model_id,
        "model_seed": int(model_seed),
        "join_key": join_key,
        "main_run_id": manifest["main_run_id"],
        "trial_count": int(spec.trial_count),
        "descriptive_results": summaries,
        "utility_metrics": utility_metrics,
        "utility_scalar_nonfinite_policy": (
            "null_unless_all_registered_trial_values_are_finite"
        ),
        "angle_utility_radius_policy": (
            "null_unless_all_corresponding_estimate_and_reference_radii_are_"
            "finite_and_at_least_epsilon"
        ),
        "nonfinite_diagnostics": {
            "temporal_error_nonfinite_count": int(
                np.sum(~np.isfinite(temporal_error_np))
            ),
            "velocity_error_nonfinite_count": int(
                np.sum(~np.isfinite(velocity_error))
            ),
            "perturbation_memory_error_nonfinite_count": int(
                np.sum(
                    ~np.isfinite(arrays["perturbation_memory_absolute_error"])
                )
            ),
            "perturbation_clean_paired_error_nonfinite_count": int(
                np.sum(
                    ~np.isfinite(
                        arrays["perturbation_clean_paired_absolute_error"]
                    )
                )
            ),
            "role": "diagnostic_and_utility_nulling_not_seed_exclusion",
        },
        "output_radius_diagnostics": {
            "temporal_radius_below_epsilon_count": int(
                np.sum(temporal_radius_np < spec.radius_epsilon)
            ),
            "temporal_nonfinite_radius_count": int(
                np.sum(~np.isfinite(temporal_radius_np))
            ),
            "velocity_radius_below_epsilon_count": int(
                np.sum(velocity_radius < spec.radius_epsilon)
            ),
            "velocity_nonfinite_radius_count": int(
                np.sum(~np.isfinite(velocity_radius))
            ),
            "perturbation_radius_below_epsilon_count": int(
                np.sum(
                    arrays["perturbation_output_radius"] < spec.radius_epsilon
                )
            ),
            "perturbation_nonfinite_radius_count": int(
                np.sum(~np.isfinite(arrays["perturbation_output_radius"]))
            ),
            "epsilon": float(spec.radius_epsilon),
            "role": "diagnostic_not_exclusion_threshold",
        },
        "claim_thresholds": None,
        "seed_excluded": False,
    }
    manifest_path = destination / "manifest.json"
    summary_path = destination / "summary.json"
    complete_path = destination / "COMPLETE"
    atomic_json(manifest_path, manifest)
    atomic_json(summary_path, summary)
    atomic_json(
        complete_path,
        {
            "schema_version": 1,
            "status": "complete",
            "analysis_identity": identity,
            "model_id": model_id,
            "model_seed": int(model_seed),
            "join_key": join_key,
            "summary_sha256": sha256_file(summary_path),
            "individual_errors_sha256": sha256_file(arrays_path),
        },
    )
    atomic_json(
        destination / "progress.json",
        {
            "schema_version": 1,
            "analysis_identity": identity,
            "stage": "complete",
            "updated_unix_seconds": time.time(),
        },
    )
    write_completion_receipt(
        completion,
        job_id=job_id,
        artifacts=[
            freeze_copy,
            arrays_path,
            manifest_path,
            summary_path,
            complete_path,
        ],
        metadata={
            "analysis_identity": identity,
            "analysis_role": ANALYSIS_ROLE,
            "model_id": model_id,
            "model_seed": int(model_seed),
            "join_key": join_key,
            "claim_thresholds": None,
        },
    )
    valid, reason = verify_completion_receipt(
        completion,
        expected_job_id=job_id,
        expected_metadata={
            "analysis_identity": identity,
            "analysis_role": ANALYSIS_ROLE,
            "model_id": model_id,
            "model_seed": int(model_seed),
        },
    )
    if not valid:
        raise RuntimeError(f"engineering completion failed verification: {reason}")
    return summary_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", choices=MODEL_IDS, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_engineering_benefit(
        checkpoint_path=args.checkpoint,
        freeze_path=args.freeze,
        output_dir=args.output_dir,
        model_id=args.model_id,
        model_seed=args.model_seed,
        device=args.device,
        smoke=args.smoke,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "summary": str(summary),
                "analysis_role": ANALYSIS_ROLE,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
