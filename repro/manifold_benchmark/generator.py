"""Manifold benchmark generator v1.

This module is intentionally separate from both the Ságodi source-reproduction
task and the legacy Exp88 generators.  It implements one common stochastic
parent and topology-specific deterministic integrators for ``S1``, ``Td`` and
``S2``.  All public arrays are time-major unless their name explicitly denotes
an initial condition or trajectory identifier.

The indexing contract is frozen as::

    initial_memory = Phi(q_0)
    inputs[t]       = control applied from q_t to q_{t+1}
    output_targets[t] = Phi(q_{t+1})

For ``S2`` the model input is a state-independent angular-velocity command
``omega_t`` after the global dwell mask.  The state follows the corresponding
Rodrigues rotation, and ``omega_t x n_t`` is stored separately as the
instantaneous effective tangent velocity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np


GENERATOR_VERSION = "manifold-benchmark-generator-v1"
BANK_SCHEMA_VERSION = 2
TRAINING_HORIZON = 128
DELTA_T = 0.1
GP_LENGTH_SCALE = 1.0
GP_STD = 1.0
GP_CHOLESKY_JITTER = 1e-6
GP_GRID_SPACING = 2.0 / (TRAINING_HORIZON - 1)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def keyed_seed(base_seed: int, *keys: Any) -> int:
    """Return a stable 63-bit seed without relying on Python's hash order."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)):
        raise TypeError("base_seed must be an integer")
    if int(base_seed) < 0:
        raise ValueError("base_seed must be non-negative")
    payload = {
        "namespace": GENERATOR_VERSION,
        "base_seed": int(base_seed),
        "keys": list(keys),
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _positive_int(name: str, value: int) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_float(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _storage_dtype(value: np.dtype | str | type[np.floating[Any]]) -> np.dtype:
    dtype = np.dtype(value)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("storage dtype must be float32 or float64")
    return dtype


def _readonly(value: np.ndarray, *, dtype: np.dtype | None = None) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ParentSpec:
    """Stochastic parent shared by all topology and OOD conditions."""

    trajectories: int = 1024
    max_horizon: int = 2048
    max_torus_dimension: int = 8
    training_horizon: int = TRAINING_HORIZON
    delta_t: float = DELTA_T
    gp_grid_spacing: float = GP_GRID_SPACING
    gp_length_scale: float = GP_LENGTH_SCALE
    gp_std: float = GP_STD
    gp_cholesky_jitter: float = GP_CHOLESKY_JITTER
    task_seed: int = 20260716
    sample_seed: int = 0
    split: str = "validation_id"

    def __post_init__(self) -> None:
        _positive_int("trajectories", self.trajectories)
        _positive_int("max_horizon", self.max_horizon)
        _positive_int("max_torus_dimension", self.max_torus_dimension)
        _positive_int("training_horizon", self.training_horizon)
        if int(self.max_horizon) < int(self.training_horizon):
            raise ValueError("max_horizon must cover training_horizon")
        if int(self.max_torus_dimension) < 3:
            raise ValueError("max_torus_dimension must be at least 3 for S2 pairing")
        _positive_float("delta_t", self.delta_t)
        _positive_float("gp_grid_spacing", self.gp_grid_spacing)
        _positive_float("gp_length_scale", self.gp_length_scale)
        _positive_float("gp_std", self.gp_std)
        _positive_float("gp_cholesky_jitter", self.gp_cholesky_jitter)
        if int(self.task_seed) < 0 or int(self.sample_seed) < 0:
            raise ValueError("task_seed and sample_seed must be non-negative")
        if not str(self.split).strip():
            raise ValueError("split must be nonempty")


@dataclass(frozen=True)
class ConditionSpec:
    """One deterministic view of a stochastic parent bank."""

    horizon: int = TRAINING_HORIZON
    velocity_scale: float = 1.0
    gp_length_scale: float = GP_LENGTH_SCALE
    gp_std: float = GP_STD
    dwell_profile: str = "variable_sparsity"
    dwell_active_probability: float | None = None
    dwell_block_start: int | None = None
    dwell_block_length: int = 0
    condition_axis: str = "id"
    condition_value: str = "id"

    def __post_init__(self) -> None:
        _positive_int("horizon", self.horizon)
        _positive_float("velocity_scale", self.velocity_scale)
        _positive_float("gp_length_scale", self.gp_length_scale)
        _positive_float("gp_std", self.gp_std)
        allowed = {"variable_sparsity", "dense", "all_blank", "fixed_probability"}
        if str(self.dwell_profile) not in allowed:
            raise ValueError(f"dwell_profile must be one of {sorted(allowed)}")
        if self.dwell_profile == "fixed_probability":
            if self.dwell_active_probability is None:
                raise ValueError("fixed_probability requires dwell_active_probability")
            probability = float(self.dwell_active_probability)
            if not 0.0 <= probability <= 1.0:
                raise ValueError("dwell_active_probability must be in [0, 1]")
        elif self.dwell_active_probability is not None:
            raise ValueError(
                "dwell_active_probability is only valid for fixed_probability"
            )
        if int(self.dwell_block_length) < 0:
            raise ValueError("dwell_block_length must be non-negative")
        if int(self.dwell_block_length) > 0:
            if self.dwell_block_start is None or int(self.dwell_block_start) < 0:
                raise ValueError("a nonempty dwell block requires a non-negative start")
            if int(self.dwell_block_start) + int(self.dwell_block_length) > int(
                self.horizon
            ):
                raise ValueError("dwell block exceeds condition horizon")
        elif self.dwell_block_start is not None:
            raise ValueError("dwell_block_start requires a nonempty dwell block")


@dataclass(frozen=True)
class ParentBank:
    """Random primitives from which paired condition banks are derived."""

    white_noise: np.ndarray
    id_base_drive: np.ndarray
    q0_angles: np.ndarray
    q0_sphere_gaussian: np.ndarray
    sparsity_parameters: np.ndarray
    mask_uniform_randoms: np.ndarray
    trajectory_id: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        arrays = {
            "white_noise": self.white_noise,
            "id_base_drive": self.id_base_drive,
            "q0_angles": self.q0_angles,
            "q0_sphere_gaussian": self.q0_sphere_gaussian,
            "sparsity_parameters": self.sparsity_parameters,
            "mask_uniform_randoms": self.mask_uniform_randoms,
            "trajectory_id": self.trajectory_id,
        }
        if any(not isinstance(value, np.ndarray) for value in arrays.values()):
            raise TypeError("ParentBank fields must be NumPy arrays")
        time, batch, dimension = self.white_noise.shape
        if self.white_noise.ndim != 3 or self.id_base_drive.shape != (time, batch, dimension):
            raise ValueError("white_noise and id_base_drive must share [T,B,D]")
        if self.q0_angles.shape != (batch, dimension):
            raise ValueError("q0_angles must have shape [B,D]")
        if self.q0_sphere_gaussian.shape != (batch, 3):
            raise ValueError("q0_sphere_gaussian must have shape [B,3]")
        if self.sparsity_parameters.shape != (batch,):
            raise ValueError("sparsity_parameters must have shape [B]")
        if self.mask_uniform_randoms.shape != (time, batch, 1):
            raise ValueError("mask_uniform_randoms must have shape [T,B,1]")
        if self.trajectory_id.shape != (batch,):
            raise ValueError("trajectory_id must have shape [B]")
        if len(np.unique(self.trajectory_id)) != batch:
            raise ValueError("trajectory identifiers must be unique")
        for name, value in arrays.items():
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def max_horizon(self) -> int:
        return int(self.white_noise.shape[0])

    @property
    def trajectories(self) -> int:
        return int(self.white_noise.shape[1])

    @property
    def max_dimension(self) -> int:
        return int(self.white_noise.shape[2])


@dataclass(frozen=True)
class ManifoldBatch:
    """Schema-v2 model-ready derived bank."""

    initial_memory: np.ndarray
    inputs: np.ndarray
    output_targets: np.ndarray
    latent_targets: np.ndarray
    latent_path: np.ndarray
    base_drive: np.ndarray
    effective_velocity: np.ndarray
    dwell_mask: np.ndarray
    trajectory_id: np.ndarray
    mask: np.ndarray
    metadata: Mapping[str, Any]
    latent_unwrapped: np.ndarray | None = None

    def __post_init__(self) -> None:
        required = {
            "initial_memory": self.initial_memory,
            "inputs": self.inputs,
            "output_targets": self.output_targets,
            "latent_targets": self.latent_targets,
            "latent_path": self.latent_path,
            "base_drive": self.base_drive,
            "effective_velocity": self.effective_velocity,
            "dwell_mask": self.dwell_mask,
            "trajectory_id": self.trajectory_id,
            "mask": self.mask,
        }
        if any(not isinstance(value, np.ndarray) for value in required.values()):
            raise TypeError("ManifoldBatch fields must be NumPy arrays")
        if self.inputs.ndim != 3:
            raise ValueError("inputs must have shape [T,B,F]")
        time, batch = self.inputs.shape[:2]
        for name in (
            "output_targets",
            "latent_targets",
            "base_drive",
            "effective_velocity",
            "dwell_mask",
            "mask",
        ):
            value = required[name]
            if value.ndim != 3 or value.shape[:2] != (time, batch):
                raise ValueError(f"{name} must share inputs [T,B]")
        if self.initial_memory.ndim != 2 or self.initial_memory.shape[0] != batch:
            raise ValueError("initial_memory must have shape [B,F]")
        if self.latent_path.ndim != 3 or self.latent_path.shape[:2] != (
            time + 1,
            batch,
        ):
            raise ValueError("latent_path must have shape [T+1,B,F]")
        if self.latent_targets.shape != self.latent_path[1:].shape:
            raise ValueError("latent_targets must match latent_path[1:] shape")
        if self.dwell_mask.shape != (time, batch, 1):
            raise ValueError("dwell_mask must have shape [T,B,1]")
        if self.mask.shape != self.output_targets.shape:
            raise ValueError("mask must exactly match output_targets")
        if self.trajectory_id.shape != (batch,):
            raise ValueError("trajectory_id must have shape [B]")
        if self.latent_unwrapped is not None:
            if not isinstance(self.latent_unwrapped, np.ndarray):
                raise TypeError("latent_unwrapped must be a NumPy array")
            if self.latent_unwrapped.shape[:2] != (time + 1, batch):
                raise ValueError("latent_unwrapped must have shape [T+1,B,F]")
            object.__setattr__(
                self, "latent_unwrapped", _readonly(self.latent_unwrapped)
            )
        for name, value in required.items():
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def horizon(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def trajectories(self) -> int:
        return int(self.inputs.shape[1])


@lru_cache(maxsize=16)
def _rbf_cholesky(
    horizon: int,
    grid_spacing: float,
    length_scale: float,
    gp_std: float,
    jitter: float,
) -> np.ndarray:
    steps = _positive_int("horizon", horizon)
    spacing = _positive_float("grid_spacing", grid_spacing)
    scale = _positive_float("length_scale", length_scale)
    std = _positive_float("gp_std", gp_std)
    diagonal_jitter = _positive_float("jitter", jitter)
    offsets = np.arange(steps, dtype=np.float64) * spacing
    lag = offsets[:, None] - offsets[None, :]
    covariance = std**2 * np.exp(-(lag**2) / (2.0 * scale**2))
    covariance.flat[:: steps + 1] += diagonal_jitter
    factor = np.linalg.cholesky(covariance)
    factor.setflags(write=False)
    return factor


def _apply_gp_factor(
    white_noise: np.ndarray,
    *,
    grid_spacing: float,
    length_scale: float,
    gp_std: float,
    jitter: float,
) -> np.ndarray:
    time, batch, dimension = white_noise.shape
    factor = _rbf_cholesky(time, grid_spacing, length_scale, gp_std, jitter)
    flattened = np.ascontiguousarray(white_noise).reshape(time, batch * dimension)
    return (factor @ flattened).reshape(time, batch, dimension)


def make_parent_bank(spec: ParentSpec) -> ParentBank:
    """Generate all random primitives with independent role-specific streams."""

    if not isinstance(spec, ParentSpec):
        raise TypeError("spec must be ParentSpec")
    root = keyed_seed(spec.task_seed, "parent", spec.sample_seed, spec.split)
    role_seeds = {
        role: keyed_seed(root, role)
        for role in (
            "task_map_seed",
            "q0_angles",
            "q0_sphere_gaussian",
            "control_white",
            "dwell",
            "trajectory",
            "split",
        )
    }
    shape = (
        int(spec.max_horizon),
        int(spec.trajectories),
        int(spec.max_torus_dimension),
    )
    white_noise = np.random.default_rng(role_seeds["control_white"]).standard_normal(
        size=shape
    )
    id_base_drive = _apply_gp_factor(
        white_noise,
        grid_spacing=spec.gp_grid_spacing,
        length_scale=spec.gp_length_scale,
        gp_std=spec.gp_std,
        jitter=spec.gp_cholesky_jitter,
    )
    q0_angles = np.random.default_rng(role_seeds["q0_angles"]).uniform(
        -math.pi,
        math.pi,
        size=(spec.trajectories, spec.max_torus_dimension),
    )
    q0_sphere_gaussian = np.random.default_rng(
        role_seeds["q0_sphere_gaussian"]
    ).standard_normal(size=(spec.trajectories, 3))
    dwell_rng = np.random.default_rng(role_seeds["dwell"])
    sparsity_parameters = dwell_rng.uniform(0.0, 2.0, size=spec.trajectories)
    mask_uniform_randoms = dwell_rng.uniform(
        0.0, 1.0, size=(spec.max_horizon, spec.trajectories, 1)
    )
    trajectory_id = np.fromiter(
        (
            keyed_seed(role_seeds["trajectory"], "trajectory", index)
            for index in range(spec.trajectories)
        ),
        dtype=np.int64,
        count=spec.trajectories,
    )
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "bank_schema_version": BANK_SCHEMA_VERSION,
        "artifact_kind": "parent_stochastic_bank",
        "parent_spec": asdict(spec),
        "role_seeds": role_seeds,
        "gp_grid": "s_t=-1+t*2/(T_train-1)",
        "gp_grid_start": -1.0,
        "target_indexing": "initial_memory_q0_target_t_is_q_t_plus_1",
        "dwell_semantics": "global_true_zero_velocity",
    }
    return ParentBank(
        white_noise=white_noise,
        id_base_drive=id_base_drive,
        q0_angles=q0_angles,
        q0_sphere_gaussian=q0_sphere_gaussian,
        sparsity_parameters=sparsity_parameters,
        mask_uniform_randoms=mask_uniform_randoms,
        trajectory_id=trajectory_id,
        metadata=metadata,
    )


def _parent_spec(parent: ParentBank) -> ParentSpec:
    payload = parent.metadata.get("parent_spec")
    if not isinstance(payload, Mapping):
        raise ValueError("parent metadata has no parent_spec")
    return ParentSpec(**dict(payload))


def _condition_drive(parent: ParentBank, condition: ConditionSpec) -> np.ndarray:
    if condition.horizon > parent.max_horizon:
        raise ValueError("condition horizon exceeds parent max_horizon")
    parent_spec = _parent_spec(parent)
    if (
        float(condition.gp_length_scale) == float(parent_spec.gp_length_scale)
        and float(condition.gp_std) == float(parent_spec.gp_std)
    ):
        drive = parent.id_base_drive
    else:
        drive = _apply_gp_factor(
            parent.white_noise,
            grid_spacing=parent_spec.gp_grid_spacing,
            length_scale=condition.gp_length_scale,
            gp_std=condition.gp_std,
            jitter=parent_spec.gp_cholesky_jitter,
        )
    return float(condition.velocity_scale) * drive[: condition.horizon]


def _condition_dwell(parent: ParentBank, condition: ConditionSpec) -> np.ndarray:
    time = int(condition.horizon)
    profile = str(condition.dwell_profile)
    if profile == "variable_sparsity":
        active_probability = np.minimum(1.0, parent.sparsity_parameters)
        dwell = parent.mask_uniform_randoms[:time] < active_probability[None, :, None]
    elif profile == "dense":
        dwell = np.ones((time, parent.trajectories, 1), dtype=bool)
    elif profile == "all_blank":
        dwell = np.zeros((time, parent.trajectories, 1), dtype=bool)
    else:
        probability = float(condition.dwell_active_probability)
        dwell = parent.mask_uniform_randoms[:time] < probability
    dwell = np.array(dwell, dtype=np.float64, copy=True)
    if condition.dwell_block_length:
        start = int(condition.dwell_block_start)
        dwell[start : start + int(condition.dwell_block_length)] = 0.0
    return dwell


def _wrap_angles(value: np.ndarray) -> np.ndarray:
    return np.remainder(value + math.pi, 2.0 * math.pi) - math.pi


def _torus_embedding(value: np.ndarray) -> np.ndarray:
    output = np.empty((*value.shape[:-1], 2 * value.shape[-1]), dtype=np.float64)
    output[..., 0::2] = np.cos(value)
    output[..., 1::2] = np.sin(value)
    return output


def _base_metadata(
    parent: ParentBank,
    condition: ConditionSpec,
    *,
    topology: str,
    intrinsic_dimension: int,
    latent_representation_dimension: int,
    initial_memory_dimension: int,
    input_dimension: int,
    output_dimension: int,
    input_semantics: str,
) -> dict[str, Any]:
    parent_spec = _parent_spec(parent)
    condition_payload = asdict(condition)
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "bank_schema_version": BANK_SCHEMA_VERSION,
        "artifact_kind": "derived_condition_bank",
        "topology": topology,
        "intrinsic_dimension": int(intrinsic_dimension),
        "latent_representation_dimension": int(latent_representation_dimension),
        "initial_memory_dimension": int(initial_memory_dimension),
        "input_dimension": int(input_dimension),
        "output_dimension": int(output_dimension),
        "training_horizon": int(parent_spec.training_horizon),
        "max_parent_horizon": int(parent_spec.max_horizon),
        "delta_t": float(parent_spec.delta_t),
        "gp_grid_spacing": float(parent_spec.gp_grid_spacing),
        "gp_length_scale": float(condition.gp_length_scale),
        "gp_std": float(condition.gp_std),
        "dwell_distribution": str(condition.dwell_profile),
        "target_indexing": "initial_memory_q0_target_t_is_q_t_plus_1",
        "input_semantics": input_semantics,
        "task_seed": int(parent_spec.task_seed),
        "sample_seed": int(parent_spec.sample_seed),
        "split": str(parent_spec.split),
        "condition_spec": condition_payload,
        "condition_spec_sha256": hashlib.sha256(
            _canonical_bytes(condition_payload)
        ).hexdigest(),
        "parent_bank_sha256": parent.metadata.get("archive_sha256"),
        "parent_white_noise_sha256": parent.metadata.get("white_noise_sha256"),
    }
    return metadata


def derive_torus(
    parent: ParentBank,
    *,
    dimensions: int,
    condition: ConditionSpec = ConditionSpec(),
    energy_mode: str = "coordinate_matched",
    storage_dtype: np.dtype | str | type[np.floating[Any]] = np.float32,
) -> ManifoldBatch:
    """Derive a flat-torus ``T^d`` bank using nested master coordinates."""

    dimension = _positive_int("dimensions", dimensions)
    if dimension > parent.max_dimension:
        raise ValueError("requested torus dimension exceeds parent master dimension")
    if energy_mode not in ("coordinate_matched", "energy_matched"):
        raise ValueError("energy_mode must be coordinate_matched or energy_matched")
    dtype = _storage_dtype(storage_dtype)
    parent_spec = _parent_spec(parent)
    base_drive = _condition_drive(parent, condition)[..., :dimension].copy()
    if energy_mode == "energy_matched":
        base_drive /= math.sqrt(dimension)
    dwell = _condition_dwell(parent, condition)
    inputs = dwell * base_drive
    q0 = parent.q0_angles[:, :dimension]
    latent_unwrapped = np.empty(
        (condition.horizon + 1, parent.trajectories, dimension), dtype=np.float64
    )
    latent_unwrapped[0] = q0
    for step in range(condition.horizon):
        latent_unwrapped[step + 1] = (
            latent_unwrapped[step] + float(parent_spec.delta_t) * inputs[step]
        )
    latent_path = _wrap_angles(latent_unwrapped)
    embedded = _torus_embedding(latent_path)
    topology = "S1" if dimension == 1 else f"T{dimension}"
    metadata = _base_metadata(
        parent,
        condition,
        topology=topology,
        intrinsic_dimension=dimension,
        latent_representation_dimension=dimension,
        initial_memory_dimension=2 * dimension,
        input_dimension=dimension,
        output_dimension=2 * dimension,
        input_semantics="global-dwell-masked intrinsic velocity",
    )
    metadata.update(
        {
            "torus_embedding": "flat_product_cos_sin",
            "dimension_pairing": "prefix_of_parent_max_dimension",
            "energy_mode": energy_mode,
        }
    )
    return ManifoldBatch(
        initial_memory=_torus_embedding(q0).astype(dtype),
        inputs=inputs.astype(dtype),
        output_targets=embedded[1:].astype(dtype),
        latent_targets=latent_path[1:].astype(dtype),
        latent_path=latent_path.astype(dtype),
        base_drive=base_drive.astype(dtype),
        effective_velocity=inputs.astype(dtype),
        dwell_mask=dwell.astype(dtype),
        trajectory_id=parent.trajectory_id.copy(),
        mask=np.ones_like(embedded[1:], dtype=dtype),
        latent_unwrapped=latent_unwrapped.astype(dtype),
        metadata=metadata,
    )


def derive_s1(
    parent: ParentBank,
    *,
    condition: ConditionSpec = ConditionSpec(),
    storage_dtype: np.dtype | str | type[np.floating[Any]] = np.float32,
) -> ManifoldBatch:
    """Derive the ring anchor task from the first master coordinate."""

    return derive_torus(
        parent,
        dimensions=1,
        condition=condition,
        energy_mode="coordinate_matched",
        storage_dtype=storage_dtype,
    )


def derive_s2(
    parent: ParentBank,
    *,
    condition: ConditionSpec = ConditionSpec(),
    storage_dtype: np.dtype | str | type[np.floating[Any]] = np.float32,
) -> ManifoldBatch:
    """Derive a sphere task from independent 3-D rotation commands."""

    dtype = _storage_dtype(storage_dtype)
    parent_spec = _parent_spec(parent)
    base_drive = _condition_drive(parent, condition)[..., :3].copy()
    dwell = _condition_dwell(parent, condition)
    inputs = dwell * base_drive
    gaussian = parent.q0_sphere_gaussian
    gaussian_norm = np.linalg.norm(gaussian, axis=1, keepdims=True)
    if np.any(gaussian_norm == 0.0):
        raise RuntimeError("zero Gaussian vector cannot define a sphere initial state")
    q0 = gaussian / gaussian_norm
    latent_path = np.empty(
        (condition.horizon + 1, parent.trajectories, 3), dtype=np.float64
    )
    effective_velocity = np.empty_like(inputs)
    latent_path[0] = q0
    max_rotation_angle = 0.0
    for step in range(condition.horizon):
        current = latent_path[step]
        omega = inputs[step]
        tangent = np.cross(omega, current)
        effective_velocity[step] = tangent
        angular_speed = np.linalg.norm(omega, axis=1, keepdims=True)
        alpha = float(parent_spec.delta_t) * angular_speed
        max_rotation_angle = max(max_rotation_angle, float(np.max(alpha)))
        moving = alpha[:, 0] > 1e-14
        following = current.copy()
        if np.any(moving):
            moving_alpha = alpha[moving]
            axis = omega[moving] / angular_speed[moving]
            axis_cross_state = np.cross(axis, current[moving])
            axis_projection = np.sum(axis * current[moving], axis=1, keepdims=True)
            following[moving] = (
                np.cos(moving_alpha) * current[moving]
                + np.sin(moving_alpha) * axis_cross_state
                + (1.0 - np.cos(moving_alpha)) * axis_projection * axis
            )
            following[moving] /= np.linalg.norm(
                following[moving], axis=1, keepdims=True
            )
        latent_path[step + 1] = following
    metadata = _base_metadata(
        parent,
        condition,
        topology="S2",
        intrinsic_dimension=2,
        latent_representation_dimension=3,
        initial_memory_dimension=3,
        input_dimension=3,
        output_dimension=3,
        input_semantics=(
            "global-dwell-masked state-independent 3D angular velocity omega_t"
        ),
    )
    metadata.update(
        {
            "effective_velocity_semantics": "inputs[t] cross n_t",
            "sphere_update": "Rodrigues_rotation_R(delta_t_times_omega_t)",
            "q0_distribution": "normalized_isotropic_gaussian",
            "maximum_rotation_command_radians": max_rotation_angle,
        }
    )
    return ManifoldBatch(
        initial_memory=q0.astype(dtype),
        inputs=inputs.astype(dtype),
        output_targets=latent_path[1:].astype(dtype),
        latent_targets=latent_path[1:].astype(dtype),
        latent_path=latent_path.astype(dtype),
        base_drive=base_drive.astype(dtype),
        effective_velocity=effective_velocity.astype(dtype),
        dwell_mask=dwell.astype(dtype),
        trajectory_id=parent.trajectory_id.copy(),
        mask=np.ones_like(latent_path[1:], dtype=dtype),
        metadata=metadata,
        latent_unwrapped=None,
    )


__all__ = [
    "BANK_SCHEMA_VERSION",
    "DELTA_T",
    "GENERATOR_VERSION",
    "GP_CHOLESKY_JITTER",
    "GP_GRID_SPACING",
    "GP_LENGTH_SCALE",
    "GP_STD",
    "TRAINING_HORIZON",
    "ConditionSpec",
    "ManifoldBatch",
    "ParentBank",
    "ParentSpec",
    "derive_s1",
    "derive_s2",
    "derive_torus",
    "keyed_seed",
    "make_parent_bank",
]
