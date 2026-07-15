"""Checkpoint runner for the Ságodi-based, explicitly adapted ring analysis.

This module deliberately keeps the primary analysis separate from
``phase1_analysis.py``.  The latter contains legacy project-specific settling,
isotropic recovery, projected-JVP, and C1--C4 diagnostics; none of those
diagnostics is called here.  This runner instead adds a separately labelled,
threshold-free carrier tangent-complement recovery diagnostic to the primary
evaluation.

Two project resolutions are made explicit in every result:

* a state-noise-disabled held-out GP task rollout of length ``T`` is followed
  by an exact ``16T`` deterministic zero-input rollout; and
* for a possibly nonlinear decoder, projected flow is the common discrete
  finite-step displacement ``D(F0(s)) - D(s)``.

Full paper-task runs fail closed on 1,024 trajectories/spline points, ``T=256``,
and ``16T=4096``.  The explicit ``--source-v6`` adaptation keeps the same counts
but freezes the public-code task at ``T=128`` and ``16T=2048``. ``--smoke``
permits smaller values for focused integration tests and labels every artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .artifacts import (
    atomic_json,
    canonical_hash,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from .config import (
    SAGODI_PRIMARY_MAIN_TRACK,
    AngularTaskSpec,
    load_protocol,
    protocol_fingerprint,
)
from .metrics import task_metrics
from .models import ProtocolModel, load_checkpoint
from .sagodi_primary_analysis import (
    StructuralNotEstimableError,
    angle_from_output,
    asymptotic_memory_metrics,
    carrier_ambient_normal_recovery,
    circular_absolute_error,
    cyclic_flow_reversal_topology,
    dense_full_jacobian_eigenspectrum,
    finite_time_angular_memory,
    output_projected_flow,
    signed_angular_flow,
    stable_basin_capacity,
)
from .state import StateAdapter
from .tasks import angular_integration
from .source_resolved_protocol import source_angular_integration
from .source_v6_analysis_adapter import (
    SOURCE_MODEL_IDS as SOURCE_V6_MODEL_IDS,
    load_source_v6_checkpoint,
)


FULL_TRAJECTORY_COUNT = 1024
FULL_SPLINE_COUNT = 1024
FULL_TASK_HORIZON = 256
FULL_BLANK_HORIZON = 16 * FULL_TASK_HORIZON
INCLUSION_NMSE_DB = -20.0
SLOW_RELATIVE_SPEED = 1.0e-3
DEFAULT_SPECTRUM_CHUNK_SIZE = 32
NORMAL_RECOVERY_SEED = 314159
NORMAL_RECOVERY_ANCHOR_COUNT = 32
NORMAL_RECOVERY_AMBIENT_DIRECTIONS = 4
NORMAL_RECOVERY_RADII_OVER_MANIFOLD_SCALE = (0.01, 0.05, 0.1)
NORMAL_RECOVERY_HORIZONS = (0, 1, 4, 16, 64, 256, 1024, 4096)
SCHEMA_VERSION = 1


def primary_analysis_spec_payload(spec: PrimaryAnalysisSpec) -> dict[str, Any]:
    """Return the frozen spec using only JSON-native container types."""

    return json.loads(json.dumps(asdict(spec)))


def primary_analysis_identity_payload(
    *,
    checkpoint_sha256: str,
    protocol_sha256: str,
    protocol_fingerprint_value: str,
    model_name: str,
    spec: "PrimaryAnalysisSpec",
) -> dict[str, Any]:
    """Build the one canonical child identity used by runner and campaign."""

    package = Path(__file__).resolve().parent
    return {
        "analysis": "sagodi_primary_single_checkpoint",
        "schema_version": SCHEMA_VERSION,
        "analysis_code_sha256": {
            "sagodi_primary_runner.py": sha256_file(Path(__file__).resolve()),
            "sagodi_primary_analysis.py": sha256_file(
                package / "sagodi_primary_analysis.py"
            ),
            "state.py": sha256_file(package / "state.py"),
            "models.py": sha256_file(package / "models.py"),
            "tasks.py": sha256_file(package / "tasks.py"),
            "source_v6_analysis_adapter.py": sha256_file(
                package / "source_v6_analysis_adapter.py"
            ),
            "source_resolved_protocol.py": sha256_file(
                package / "source_resolved_protocol.py"
            ),
        },
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": protocol_sha256,
        "protocol_fingerprint": protocol_fingerprint_value,
        "model_name": model_name,
        # Dataclass tuples must cross the JSON artifact boundary as lists so
        # the campaign can compare the persisted identity payload exactly.
        "spec": primary_analysis_spec_payload(spec),
    }


@dataclass(frozen=True)
class PrimaryAnalysisSpec:
    trajectory_count: int = FULL_TRAJECTORY_COUNT
    spline_count: int = FULL_SPLINE_COUNT
    task_horizon: int = FULL_TASK_HORIZON
    blank_horizon: int = FULL_BLANK_HORIZON
    spectrum_chunk_size: int = DEFAULT_SPECTRUM_CHUNK_SIZE
    candidate_distance_chunk_size: int = 16384
    slow_relative_speed: float = SLOW_RELATIVE_SPEED
    inclusion_nmse_db: float = INCLUSION_NMSE_DB
    flow_zero_tolerance: float = 0.0
    normal_recovery_seed: int = NORMAL_RECOVERY_SEED
    normal_recovery_anchor_count: int = NORMAL_RECOVERY_ANCHOR_COUNT
    normal_recovery_ambient_directions: int = NORMAL_RECOVERY_AMBIENT_DIRECTIONS
    normal_recovery_radii_over_manifold_scale: tuple[float, ...] = (
        NORMAL_RECOVERY_RADII_OVER_MANIFOLD_SCALE
    )
    normal_recovery_horizons: tuple[int, ...] = NORMAL_RECOVERY_HORIZONS
    source_v6: bool = False
    smoke: bool = False

    def validate(self) -> None:
        integer_fields = (
            "trajectory_count",
            "spline_count",
            "task_horizon",
            "blank_horizon",
            "spectrum_chunk_size",
            "candidate_distance_chunk_size",
        )
        for name in integer_fields:
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.trajectory_count < 4 or self.spline_count < 4:
            raise ValueError("periodic reconstruction requires at least four points")
        if not math.isfinite(self.slow_relative_speed) or self.slow_relative_speed <= 0:
            raise ValueError("slow_relative_speed must be finite and positive")
        if not math.isfinite(self.inclusion_nmse_db):
            raise ValueError("inclusion_nmse_db must be finite")
        if not math.isfinite(self.flow_zero_tolerance) or self.flow_zero_tolerance < 0:
            raise ValueError("flow_zero_tolerance must be finite and non-negative")
        if self.normal_recovery_seed < 0:
            raise ValueError("normal_recovery_seed must be non-negative")
        if self.normal_recovery_anchor_count <= 0:
            raise ValueError("normal_recovery_anchor_count must be positive")
        if self.normal_recovery_ambient_directions <= 0:
            raise ValueError("normal_recovery_ambient_directions must be positive")
        if tuple(self.normal_recovery_radii_over_manifold_scale) != tuple(
            sorted(set(self.normal_recovery_radii_over_manifold_scale))
        ) or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.normal_recovery_radii_over_manifold_scale
        ):
            raise ValueError("normal recovery radius fractions must be positive and increasing")
        if (
            not self.normal_recovery_horizons
            or self.normal_recovery_horizons[0] != 0
            or tuple(self.normal_recovery_horizons)
            != tuple(sorted(set(self.normal_recovery_horizons)))
        ):
            raise ValueError("normal recovery horizons must be increasing and start at zero")
        if not self.smoke:
            expected = {
                "trajectory_count": FULL_TRAJECTORY_COUNT,
                "spline_count": FULL_SPLINE_COUNT,
                "task_horizon": 128 if self.source_v6 else FULL_TASK_HORIZON,
                "blank_horizon": 2048 if self.source_v6 else FULL_BLANK_HORIZON,
                "slow_relative_speed": SLOW_RELATIVE_SPEED,
                "inclusion_nmse_db": INCLUSION_NMSE_DB,
                "normal_recovery_seed": NORMAL_RECOVERY_SEED,
                "normal_recovery_anchor_count": NORMAL_RECOVERY_ANCHOR_COUNT,
                "normal_recovery_ambient_directions": NORMAL_RECOVERY_AMBIENT_DIRECTIONS,
                "normal_recovery_radii_over_manifold_scale": NORMAL_RECOVERY_RADII_OVER_MANIFOLD_SCALE,
                "normal_recovery_horizons": NORMAL_RECOVERY_HORIZONS,
            }
            for name, frozen in expected.items():
                if getattr(self, name) != frozen:
                    raise ValueError(
                        f"full Ságodi-primary analysis freezes {name}={frozen}"
                    )


@dataclass(frozen=True)
class ReconstructionResult:
    spline_angle: torch.Tensor
    spline_state: torch.Tensor
    selected_state: torch.Tensor
    selected_candidate_time: np.ndarray
    selected_candidate_trajectory: np.ndarray
    selected_decoded_output: np.ndarray
    selected_output_distance: np.ndarray
    knot_angle: torch.Tensor
    knot_state: torch.Tensor
    maximum_speed: np.ndarray
    candidate_count_per_trajectory: np.ndarray
    qa: dict[str, Any]


def _atomic_npz(path: Path, **arrays: Any) -> None:
    """Atomically write a compressed NumPy archive."""

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


def _as_numpy(value: torch.Tensor, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    array = value.detach().cpu().numpy()
    return array.astype(dtype, copy=False) if dtype is not None else array


def _finite_summary(values: np.ndarray | torch.Tensor) -> dict[str, float] | None:
    array = _as_numpy(values) if isinstance(values, torch.Tensor) else np.asarray(values)
    finite = np.asarray(array, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "median": float(np.median(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q95": float(np.quantile(finite, 0.95)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _registered_numeric_summary(
    values: np.ndarray | torch.Tensor,
) -> dict[str, float | int]:
    """Summarize a registered array without silently dropping non-finite cells."""

    array = _as_numpy(values) if isinstance(values, torch.Tensor) else np.asarray(values)
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(flat)
    registered = int(flat.size)
    finite_count = int(finite_mask.sum())
    missing = registered - finite_count
    if not registered:
        raise ValueError("registered numeric summary requires at least one value")
    if missing:
        raise StructuralNotEstimableError(
            f"registered recovery metric has {missing}/{registered} non-finite values"
        )
    return {
        "registered_count": registered,
        "finite_count": finite_count,
        "missing_or_nonfinite_count": missing,
        "mean": float(flat.mean()),
        "population_std": float(flat.std(ddof=0)),
        "median": float(np.median(flat)),
        "q05": float(np.quantile(flat, 0.05)),
        "q95": float(np.quantile(flat, 0.95)),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


def _canonical_radius_key(value: float) -> str:
    return format(float(value), ".12g")


def _persisted_direction_arrays_and_qa(
    recovery: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recompute QA from the exact direction arrays persisted in the NPZ.

    Torch float32 norm/dot reductions can differ from a verifier's float64
    recomputation of the persisted float32 components.  The persisted QA must
    describe the artifact itself, not a device-dependent intermediate value.
    """

    direction = _as_numpy(recovery.direction)
    tangent = _as_numpy(recovery.tangent)
    direction64 = np.asarray(direction, dtype=np.float64)
    tangent64 = np.asarray(tangent, dtype=np.float64)
    direction_norm_error = np.abs(np.linalg.norm(direction64, axis=1) - 1.0)
    absolute_tangent_dot_direction = np.abs(
        np.sum(direction64 * tangent64, axis=1)
    )
    return (
        direction,
        tangent,
        direction_norm_error,
        absolute_tangent_dot_direction,
    )


def _carrier_normal_recovery_summary(
    recovery: Any,
    *,
    seed: int,
    anchor_count: int,
    ambient_directions_per_anchor: int,
    radii_over_manifold_scale: tuple[float, ...],
    horizons: tuple[int, ...],
    smoke: bool,
) -> dict[str, Any]:
    """Build denominator-preserving summaries, keeping direction families apart."""

    family = np.asarray(recovery.family, dtype="U32")
    radius = _as_numpy(recovery.radius_over_manifold_scale, dtype=np.float64)
    horizon = _as_numpy(recovery.horizon, dtype=np.int64)
    metric_arrays = {
        "manifold_distance": _as_numpy(recovery.manifold_distance),
        "manifold_distance_ratio": _as_numpy(recovery.manifold_distance_ratio),
        "same_memory_error_radians": _as_numpy(
            recovery.same_memory_error_radians
        ),
        "clean_manifold_distance": _as_numpy(recovery.clean_manifold_distance),
        "clean_same_memory_error_radians": _as_numpy(
            recovery.clean_same_memory_error_radians
        ),
        "excess_same_memory_error_radians": _as_numpy(
            recovery.excess_same_memory_error_radians
        ),
        "manifold_distance_minus_clean": _as_numpy(
            recovery.manifold_distance_minus_clean
        ),
        "distance_to_matched_clean_state": _as_numpy(
            recovery.distance_to_matched_clean_state
        ),
        "distance_to_matched_clean_state_ratio": _as_numpy(
            recovery.distance_to_matched_clean_state_ratio
        ),
    }
    if any(value.shape != (family.size, horizon.size) for value in metric_arrays.values()):
        raise ValueError("carrier recovery metric arrays do not share [trial,horizon]")

    metrics_by_family: dict[str, Any] = {}
    base_count_by_family: dict[str, int] = {}
    horizon_count_by_family: dict[str, int] = {}
    for family_name in ("ambient_normal", "in_plane_radial"):
        family_mask = family == family_name
        family_count = int(family_mask.sum())
        if family_count <= 0:
            raise StructuralNotEstimableError(
                f"registered carrier direction family {family_name} is empty"
            )
        base_count_by_family[family_name] = family_count
        horizon_count_by_family[family_name] = family_count * int(horizon.size)
        by_radius: dict[str, Any] = {}
        for radius_value in radii_over_manifold_scale:
            radius_mask = family_mask & np.isclose(
                radius, float(radius_value), rtol=0.0, atol=1.0e-12
            )
            if not bool(radius_mask.any()):
                raise StructuralNotEstimableError(
                    f"registered radius {radius_value} is absent for {family_name}"
                )
            by_horizon: dict[str, Any] = {}
            for column, horizon_value in enumerate(horizon.tolist()):
                by_horizon[str(int(horizon_value))] = {
                    name: _registered_numeric_summary(values[radius_mask, column])
                    for name, values in metric_arrays.items()
                }
            by_radius[_canonical_radius_key(radius_value)] = {
                "registered_base_perturbation_count": int(radius_mask.sum()),
                "registered_horizon_record_count": int(radius_mask.sum())
                * int(horizon.size),
                "by_horizon": by_horizon,
            }
        metrics_by_family[family_name] = {
            "registered_base_perturbation_count": family_count,
            "registered_horizon_record_count": family_count * int(horizon.size),
            "by_radius": by_radius,
        }

    ambient_mask = family == "ambient_normal"
    ambient_aggregate = {
        name: {
            str(int(horizon_value)): _registered_numeric_summary(
                values[ambient_mask, column]
            )
            for column, horizon_value in enumerate(horizon.tolist())
        }
        for name, values in metric_arrays.items()
    }
    observed_anchor_index = _as_numpy(recovery.anchor_index, dtype=np.int64)
    unique_anchor_index = np.unique(observed_anchor_index)
    if unique_anchor_index.size != int(anchor_count):
        raise StructuralNotEstimableError(
            "carrier recovery did not retain every registered unique anchor"
        )
    (
        _,
        _,
        direction_norm_error,
        tangent_dot_direction,
    ) = _persisted_direction_arrays_and_qa(recovery)
    initial_distance = metric_arrays["manifold_distance"][:, 0]
    return {
        "role": (
            "project_defined_primary_descriptive_carrier_normal_recovery_"
            "no_threshold_or_binary_gate"
        ),
        "state_space": "minimum_causal_primary_carrier_state",
        "manifold_source": "reconstructed_primary_carrier_periodic_cubic_spline",
        "distance_definition": (
            "Euclidean nearest distance to the registered sampled carrier spline"
        ),
        "tangent_definition": (
            "normalized central periodic finite difference of carrier spline state"
        ),
        "direction_families": {
            "ambient_normal": (
                "seeded Gaussian full-state direction projected by I-tt^T"
            ),
            "in_plane_radial": (
                "two signed tangent-orthogonal directions in global carrier two-PC plane"
            ),
        },
        "matched_clean_control": (
            "same carrier anchor rolled under identical deterministic blank dynamics; "
            "excess metrics subtract ordinary clean on-manifold drift"
        ),
        "scope": (
            "sampled Euclidean tangent-complement recovery on the reconstructed "
            "carrier spline; this is not a proof of an invariant stable normal "
            "bundle over the entire continuous manifold"
        ),
        "deterministic_design": {
            "seed": int(seed),
            "anchor_count": int(anchor_count),
            "anchor_indices": unique_anchor_index.tolist(),
            "ambient_directions_per_anchor": int(ambient_directions_per_anchor),
            "in_plane_radial_directions_per_anchor": 2,
            "radii_over_manifold_scale": [
                float(value) for value in radii_over_manifold_scale
            ],
            "horizons": [int(value) for value in horizons],
            "smoke_reduction": bool(smoke),
            "registered_base_perturbation_count_by_family": base_count_by_family,
            "registered_base_perturbation_count": int(family.size),
            "registered_horizon_record_count_by_family": horizon_count_by_family,
            "registered_horizon_record_count": int(family.size * horizon.size),
        },
        "manifold_scale": float(recovery.manifold_scale.detach().cpu()),
        "numerical_qa": {
            "unique_anchor_count": int(unique_anchor_index.size),
            "expected_anchor_count": int(anchor_count),
            "maximum_direction_norm_error": float(direction_norm_error.max()),
            "maximum_absolute_tangent_dot_direction": float(
                tangent_dot_direction.max()
            ),
            "initial_manifold_distance": _registered_numeric_summary(
                initial_distance
            ),
            "initial_manifold_distance_all_finite": bool(
                np.isfinite(initial_distance).all()
            ),
            "ambient_normal_base_perturbation_count": int(
                base_count_by_family["ambient_normal"]
            ),
            "in_plane_radial_base_perturbation_count": int(
                base_count_by_family["in_plane_radial"]
            ),
            "manifold_scale_finite_positive": bool(
                math.isfinite(float(recovery.manifold_scale.detach().cpu()))
                and float(recovery.manifold_scale.detach().cpu()) > 0.0
            ),
        },
        "metrics_by_family": metrics_by_family,
        # These ambient-only aliases are the primary cross-seed campaign fields.
        **ambient_aggregate,
        "claim_gate": False,
    }


@torch.no_grad()
def _retention_finite_precision_diagnostics(model: ProtocolModel) -> dict[str, Any]:
    """Record when learned retentions round to one in the runtime dtype.

    RP clips ``theta`` at 18 and the recurrence evaluates
    ``sqrt(sigmoid(theta))`` in the checkpoint dtype.  In float32 this may
    round to exactly one even though a float64 recomputation remains below
    one.  This is numerical provenance only; it is never a claim gate.
    """

    runtime = model.retention_values().detach().reshape(-1)
    result: dict[str, Any] = {
        "role": "descriptive_numerical_provenance_not_a_claim_gate",
        "runtime_retention_available": bool(runtime.numel()),
        "runtime_dtype": str(runtime.dtype),
        "runtime_retention_count": int(runtime.numel()),
        "runtime_exact_one_count": int((runtime == 1).sum().item()),
        "runtime_nonfinite_count": int((~torch.isfinite(runtime)).sum().item()),
        "runtime_min": float(runtime.min().item()) if runtime.numel() else None,
        "runtime_max": float(runtime.max().item()) if runtime.numel() else None,
        "interpretation": (
            "An exact runtime value of one can be float32 sigmoid/sqrt "
            "saturation and must not by itself be interpreted as an exact "
            "neutral direction or exact continuous attractor."
        ),
    }
    theta_parts = [
        recurrence.theta.detach().reshape(-1)
        for recurrence, _ in model.pan_recs_with_slices()
        if hasattr(recurrence, "theta")
    ]
    if not theta_parts:
        result["theta_parameterization_available"] = False
        return result
    theta = torch.cat(theta_parts)
    theta64 = theta.to(dtype=torch.float64)
    q64 = torch.sigmoid(theta64).clamp(1.0e-8, 1.0 - 1.0e-8)
    retention64 = torch.sqrt(q64)
    distance64 = 1.0 - retention64
    result.update(
        {
            "theta_parameterization_available": True,
            "theta_count": int(theta.numel()),
            "theta_dtype": str(theta.dtype),
            "theta_at_positive_clip_18_count": int((theta >= 18.0).sum().item()),
            "float64_recomputed_exact_one_count": int(
                (retention64 == 1.0).sum().item()
            ),
            "float64_recomputed_retention_min": float(retention64.min().item()),
            "float64_recomputed_retention_max": float(retention64.max().item()),
            "float64_min_distance_below_one": float(distance64.min().item()),
        }
    )
    return result


def _uniform_angles(count: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(count, device=device, dtype=dtype) * (
        2.0 * math.pi / float(count)
    )


def _blank_decode_primary(
    model: ProtocolModel,
    adapter: StateAdapter,
    primary_state: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the actual decoder as a function of blank-map primary state.

    A stream-decoded full block stores the latest generated stream outside the
    minimal Markov state.  Under blank input that stream is a deterministic
    function of the supplied recurrent carrier.  Reconstructing it with the
    protocol model's own block computation provides ``D(s)`` without advancing
    ``s``.  StepBaseline-style models decode the primary state directly.
    """

    state_spec = adapter.state_spec()
    if bool(state_spec["decode_requires_reported_stream"]):
        blank = adapter.zero_input(primary_state)
        reported = model._reported_from_noisy_full_block(blank, primary_state)
        output = model.decode(reported)
    else:
        output = adapter.decode(primary_state)
    if output.shape != (primary_state.shape[0], 2):
        raise ValueError("primary decoder must return [batch,2]")
    if not bool(torch.isfinite(output).all()):
        raise StructuralNotEstimableError(
            "primary decoder returned non-finite output"
        )
    return output


def _reported_with_blank_decode_stream(
    model: ProtocolModel,
    adapter: StateAdapter,
    primary_state: torch.Tensor,
) -> torch.Tensor:
    if bool(adapter.state_spec()["decode_requires_reported_stream"]):
        return model._reported_from_noisy_full_block(
            adapter.zero_input(primary_state), primary_state
        )
    return adapter.reported_from_primary(primary_state)


def periodic_cubic_resample(
    knot_angle: torch.Tensor,
    knot_state: torch.Tensor,
    query_angle: torch.Tensor,
) -> torch.Tensor:
    """Coordinate-wise periodic cubic spline with an explicit cyclic solve."""

    if knot_angle.ndim != 1 or knot_state.ndim != 2 or query_angle.ndim != 1:
        raise ValueError("periodic spline expects angle [K], state [K,D], query [N]")
    if knot_angle.numel() != knot_state.shape[0] or knot_angle.numel() < 4:
        raise ValueError("periodic cubic spline requires at least four paired knots")
    if (
        knot_angle.device != knot_state.device
        or query_angle.device != knot_state.device
        or knot_angle.dtype != knot_state.dtype
        or query_angle.dtype != knot_state.dtype
    ):
        raise ValueError("periodic spline tensors must share dtype and device")
    period = 2.0 * math.pi
    wrapped = torch.remainder(knot_angle, period)
    order = torch.argsort(wrapped)
    x = wrapped[order]
    y = knot_state[order]
    widths = torch.cat((x[1:] - x[:-1], x[:1] + period - x[-1:]))
    spacing_floor = 100.0 * torch.finfo(x.dtype).eps
    if bool((widths <= spacing_floor).any()):
        raise ValueError("periodic spline knots are duplicated or numerically coincident")

    count = int(x.numel())
    matrix = torch.zeros(count, count, device=x.device, dtype=x.dtype)
    index = torch.arange(count, device=x.device)
    previous = torch.remainder(index - 1, count)
    following = torch.remainder(index + 1, count)
    h_previous = widths[previous]
    h_following = widths[index]
    matrix[index, previous] = h_previous
    matrix[index, index] = 2.0 * (h_previous + h_following)
    matrix[index, following] = h_following
    right = 6.0 * (
        (y[following] - y) / h_following[:, None]
        - (y - y[previous]) / h_previous[:, None]
    )
    second = torch.linalg.solve(matrix, right)

    query = torch.remainder(query_angle, period)
    interval = torch.remainder(torch.searchsorted(x, query, right=True) - 1, count)
    next_interval = torch.remainder(interval + 1, count)
    displacement = torch.remainder(query - x[interval], period)
    width = widths[interval]
    b = displacement / width
    a = 1.0 - b
    result = (
        a[:, None] * y[interval]
        + b[:, None] * y[next_interval]
        + (
            (a.pow(3) - a)[:, None] * second[interval]
            + (b.pow(3) - b)[:, None] * second[next_interval]
        )
        * width.square()[:, None]
        / 6.0
    )
    if not bool(torch.isfinite(result).all()):
        raise StructuralNotEstimableError(
            "periodic spline produced non-finite states"
        )
    return result


def _deterministic_knot_indices(
    decoded_angle: np.ndarray,
    candidate_time: np.ndarray,
    candidate_trajectory: np.ndarray,
    *,
    trajectory_count: int,
    tolerance: float,
) -> np.ndarray:
    """Merge cyclic near-duplicate knots by retaining the earliest candidate."""

    angle = np.remainder(np.asarray(decoded_angle, dtype=np.float64), 2.0 * math.pi)
    time_index = np.asarray(candidate_time, dtype=np.int64)
    trajectory_index = np.asarray(candidate_trajectory, dtype=np.int64)
    if not (angle.shape == time_index.shape == trajectory_index.shape):
        raise ValueError("candidate knot arrays must have equal shape")
    flat = time_index * int(trajectory_count) + trajectory_index
    # Exact candidate duplicates are merged before angular near-duplicates.
    _, exact = np.unique(flat, return_index=True)
    exact.sort()
    angle = angle[exact]
    flat = flat[exact]
    order = np.argsort(angle, kind="mergesort")
    angle = angle[order]
    flat = flat[order]
    source = exact[order]
    count = int(angle.size)
    if not count:
        return np.empty(0, dtype=np.int64)

    parent = np.arange(count, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(count - 1):
        if angle[left + 1] - angle[left] <= float(tolerance):
            union(left, left + 1)
    if count > 1 and angle[0] + 2.0 * math.pi - angle[-1] <= float(tolerance):
        union(count - 1, 0)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    representatives = [
        min(members, key=lambda item: (int(flat[item]), int(source[item])))
        for members in groups.values()
    ]
    representatives.sort(key=lambda item: float(angle[item]))
    return source[np.asarray(representatives, dtype=np.int64)]


def _nearest_output_candidates(
    candidate_output: np.ndarray,
    target_angle: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact nearest candidates in decoded 2-D output space, chunked in RAM."""

    candidates = np.asarray(candidate_output, dtype=np.float64)
    target = np.stack((np.cos(target_angle), np.sin(target_angle)), axis=1)
    if candidates.ndim != 2 or candidates.shape[1] != 2 or not candidates.shape[0]:
        raise ValueError("candidate_output must be nonempty [candidate,2]")
    best_distance = np.full(target.shape[0], np.inf, dtype=np.float64)
    best_index = np.full(target.shape[0], -1, dtype=np.int64)
    for start in range(0, candidates.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), candidates.shape[0])
        values = candidates[start:stop]
        distance_squared = (
            np.square(target).sum(axis=1, keepdims=True)
            + np.square(values).sum(axis=1)[None, :]
            - 2.0 * target @ values.T
        )
        distance_squared = np.maximum(distance_squared, 0.0)
        local = np.argmin(distance_squared, axis=1)
        observed = distance_squared[np.arange(target.shape[0]), local]
        # Strict inequality preserves the earlier time-major candidate on ties.
        update = observed < best_distance
        best_distance[update] = observed[update]
        best_index[update] = start + local[update]
    if bool((best_index < 0).any()):
        raise RuntimeError("nearest candidate search left an unassigned target")
    return best_index, np.sqrt(best_distance)


def _nearest_output_candidates_torch(
    candidate_output: torch.Tensor,
    target_angle: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact device-side equivalent of :func:`_nearest_output_candidates`.

    Full reconstruction can expose hundreds of thousands of slow candidates.
    Keeping the 2-D search on the model device avoids a multi-billion-operation
    NumPy loop while preserving the same Euclidean objective and deterministic
    earliest-candidate tie policy.
    """

    if candidate_output.ndim != 2 or candidate_output.shape[1] != 2:
        raise ValueError("candidate_output must have shape [candidate,2]")
    if not candidate_output.shape[0] or target_angle.ndim != 1:
        raise ValueError("candidate_output and target_angle must be nonempty")
    if candidate_output.device != target_angle.device or candidate_output.dtype != target_angle.dtype:
        raise ValueError("candidate_output and target_angle must share dtype and device")
    target = torch.stack((torch.cos(target_angle), torch.sin(target_angle)), dim=1)
    best_distance = torch.full(
        (target.shape[0],),
        torch.inf,
        device=target.device,
        dtype=target.dtype,
    )
    best_index = torch.full(
        (target.shape[0],), -1, device=target.device, dtype=torch.int64
    )
    target_squared = target.square().sum(dim=1, keepdim=True)
    for start in range(0, candidate_output.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), candidate_output.shape[0])
        values = candidate_output[start:stop]
        distance_squared = (
            target_squared
            + values.square().sum(dim=1).unsqueeze(0)
            - 2.0 * target @ values.transpose(0, 1)
        ).clamp_min_(0.0)
        observed, local = distance_squared.min(dim=1)
        update = observed < best_distance
        best_distance[update] = observed[update]
        best_index[update] = int(start) + local[update]
    if bool((best_index < 0).any()):
        raise RuntimeError("device nearest-candidate search left an unassigned target")
    return best_index, torch.sqrt(best_distance)


@torch.no_grad()
def reconstruct_slow_manifold(
    model: ProtocolModel,
    adapter: StateAdapter,
    endpoint_reported: torch.Tensor,
    target_angle: torch.Tensor,
    spec: PrimaryAnalysisSpec,
) -> ReconstructionResult:
    """Reconstruct the slow ring from exactly one task endpoint per trajectory."""

    spec.validate()
    trajectories = int(spec.trajectory_count)
    if endpoint_reported.shape != (trajectories, adapter.reported_dim):
        raise ValueError("endpoint_reported does not match the trajectory freeze")
    if target_angle.shape != (spec.spline_count,):
        raise ValueError("target_angle does not match the spline freeze")
    horizon = int(spec.blank_horizon)
    storage_dtype = (
        np.float64 if endpoint_reported.dtype == torch.float64 else np.float32
    )
    # A full float32 discovery trace is ~48 MiB.  Keeping it on the model
    # device avoids 8,192 tiny synchronizing device-to-host copies; one bulk
    # copy is both faster and still modest relative to the trained models.
    speeds_device = torch.empty(
        horizon,
        trajectories,
        device=endpoint_reported.device,
        dtype=endpoint_reported.dtype,
    )
    decoded_output_device = torch.empty(
        horizon,
        trajectories,
        2,
        device=endpoint_reported.device,
        dtype=endpoint_reported.dtype,
    )
    reported = endpoint_reported
    primary = adapter.primary_from_reported(reported)
    for step in range(horizon):
        reported = adapter.reported_step(reported, adapter.zero_input(reported))
        next_primary = adapter.primary_from_reported(reported)
        output = adapter.decode(reported)
        speed = torch.linalg.vector_norm(next_primary - primary, dim=-1)
        if not bool(torch.isfinite(output).all() and torch.isfinite(speed).all()):
            raise StructuralNotEstimableError(
                f"non-finite blank rollout at step {step + 1}"
            )
        speeds_device[step] = speed
        decoded_output_device[step] = output
        primary = next_primary

    speeds = _as_numpy(speeds_device, dtype=storage_dtype)
    decoded_output = _as_numpy(decoded_output_device, dtype=storage_dtype)

    maximum_speed = speeds.max(axis=0)
    positive = maximum_speed > 0.0
    candidate_mask = positive[None, :] & (
        speeds <= float(spec.slow_relative_speed) * maximum_speed[None, :]
    )
    # Exactly stationary trajectories contribute one representative, not H copies.
    candidate_mask[0, ~positive] = True
    candidate_flat = np.flatnonzero(candidate_mask.reshape(-1))
    if not candidate_flat.size:
        raise StructuralNotEstimableError(
            "no state satisfied the per-trajectory slow criterion"
        )
    flattened_output = decoded_output.reshape(-1, 2)
    target_numpy = _as_numpy(target_angle, dtype=np.float64)
    candidate_flat_device = torch.as_tensor(
        candidate_flat, device=endpoint_reported.device, dtype=torch.int64
    )
    selected_in_candidates_device, selected_distance_device = (
        _nearest_output_candidates_torch(
            decoded_output_device.reshape(-1, 2)[candidate_flat_device],
            target_angle,
            chunk_size=spec.candidate_distance_chunk_size,
        )
    )
    selected_in_candidates = _as_numpy(
        selected_in_candidates_device, dtype=np.int64
    )
    selected_distance = _as_numpy(
        selected_distance_device, dtype=np.float64
    )
    selected_flat = candidate_flat[selected_in_candidates]
    selected_time = selected_flat // trajectories
    selected_trajectory = selected_flat % trajectories
    selected_output = flattened_output[selected_flat]

    selected_state = torch.empty(
        spec.spline_count,
        adapter.primary_dim,
        device=endpoint_reported.device,
        dtype=endpoint_reported.dtype,
    )
    reported = endpoint_reported
    for step in range(horizon):
        reported = adapter.reported_step(reported, adapter.zero_input(reported))
        wanted = np.flatnonzero(selected_time == step)
        if wanted.size:
            destination = torch.as_tensor(wanted, device=endpoint_reported.device)
            source = torch.as_tensor(
                selected_trajectory[wanted], device=endpoint_reported.device
            )
            selected_state[destination] = adapter.primary_from_reported(reported)[source]

    selected_angle = np.arctan2(selected_output[:, 1], selected_output[:, 0])
    knot_tolerance = max(
        100.0 * torch.finfo(endpoint_reported.dtype).eps,
        1.0e-12,
    )
    knot_index = _deterministic_knot_indices(
        selected_angle,
        selected_time,
        selected_trajectory,
        trajectory_count=trajectories,
        tolerance=knot_tolerance,
    )
    if knot_index.size < 4:
        raise StructuralNotEstimableError(
            f"periodic spline is not estimable from {knot_index.size} unique knots"
        )
    knot_angle = torch.as_tensor(
        selected_angle[knot_index],
        device=endpoint_reported.device,
        dtype=endpoint_reported.dtype,
    )
    knot_state = selected_state[
        torch.as_tensor(knot_index, device=endpoint_reported.device)
    ]
    spline_state = periodic_cubic_resample(knot_angle, knot_state, target_angle)

    selected_angle_error = np.abs(
        np.arctan2(
            np.sin(selected_angle - target_numpy),
            np.cos(selected_angle - target_numpy),
        )
    )
    sorted_knots = np.sort(np.remainder(_as_numpy(knot_angle), 2.0 * math.pi))
    maximum_gap = float(
        np.diff(np.concatenate((sorted_knots, sorted_knots[:1] + 2.0 * math.pi))).max()
    )
    seam_epsilon = max(1.0e-4, 10.0 * math.sqrt(float(knot_tolerance)))
    seam_query = torch.tensor(
        [0.0, 2.0 * math.pi, -seam_epsilon, seam_epsilon],
        device=target_angle.device,
        dtype=target_angle.dtype,
    )
    seam_state = periodic_cubic_resample(knot_angle, knot_state, seam_query)
    seam_c0 = torch.linalg.vector_norm(seam_state[0] - seam_state[1])
    seam_slope_jump = torch.linalg.vector_norm(
        (seam_state[0] - seam_state[2]) / seam_epsilon
        - (seam_state[3] - seam_state[0]) / seam_epsilon
    )
    qa = {
        "role": "descriptive_numerical_QA_not_CA_gate",
        "candidate_definition": (
            "speed_t <= 1e-3 * per_trajectory_max_speed; exactly stationary "
            "trajectory contributes one representative"
        ),
        "nearest_candidate_metric": "euclidean_distance_in_actual_decoded_output_R2",
        "candidate_count": int(candidate_flat.size),
        "candidate_trajectory_coverage": int(
            np.count_nonzero(candidate_mask.sum(axis=0))
        ),
        "candidate_count_per_trajectory": _finite_summary(
            candidate_mask.sum(axis=0)
        ),
        "stationary_trajectory_count": int((~positive).sum()),
        "selected_unique_candidate_count": int(np.unique(selected_flat).size),
        "selected_output_distance": _finite_summary(selected_distance),
        "selected_angle_error_radians": _finite_summary(selected_angle_error),
        "knot_count_after_deterministic_merge": int(knot_index.size),
        "knot_merge_tolerance_radians": float(knot_tolerance),
        "maximum_circular_knot_gap_radians": maximum_gap,
        "periodic_seam_C0_l2": float(seam_c0.detach().cpu()),
        "periodic_seam_finite_difference_slope_jump_l2": float(
            seam_slope_jump.detach().cpu()
        ),
        "spline_method": "coordinatewise_periodic_cubic_cyclic_linear_solve",
    }
    return ReconstructionResult(
        spline_angle=target_angle,
        spline_state=spline_state,
        selected_state=selected_state,
        selected_candidate_time=selected_time.astype(np.int64, copy=False),
        selected_candidate_trajectory=selected_trajectory.astype(np.int64, copy=False),
        selected_decoded_output=selected_output.astype(storage_dtype, copy=False),
        selected_output_distance=selected_distance,
        knot_angle=knot_angle,
        knot_state=knot_state,
        maximum_speed=maximum_speed,
        candidate_count_per_trajectory=candidate_mask.sum(axis=0).astype(
            np.int32, copy=False
        ),
        qa=qa,
    )


@torch.no_grad()
def _task_rollout(
    model: ProtocolModel,
    adapter: StateAdapter,
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial_memory = batch.initial_memory
    if initial_memory is None:
        raise ValueError("hidden-init angular batch has no initial memory")
    reported = model.initial_state(
        batch.batch_size, batch.inputs.device, initial_memory=initial_memory
    )
    predictions: list[torch.Tensor] = []
    for token in batch.inputs:
        reported = model.step(token, reported)
        predictions.append(model.decode(reported))
    prediction = torch.stack(predictions, dim=0)
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("task rollout returned non-finite predictions")
    if adapter.primary_from_reported(reported).shape[-1] != adapter.primary_dim:
        raise RuntimeError("task endpoint did not preserve full primary state")
    return prediction, reported


def _source_v6_task_batch(
    spec: PrimaryAnalysisSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    """Generate the registered T128 public-code task with q1 initialization."""

    batch = source_angular_integration(
        int(spec.trajectory_count),
        0,
        stream_key=("sagodi_source_repaired_baselines_v6", "primary_analysis_bank"),
        device=device,
    )
    if batch.inputs.shape[0] != int(spec.task_horizon):
        raise RuntimeError("source-v6 analysis task horizon differs")
    return SimpleNamespace(
        inputs=batch.inputs.to(dtype=dtype),
        output_targets=batch.output_targets.to(dtype=dtype),
        latent_targets=batch.latent_targets.to(dtype=dtype),
        mask=batch.mask.to(dtype=dtype),
        initial_memory=batch.output_targets[0].to(dtype=dtype),
        batch_size=batch.batch_size,
        metadata=batch.metadata,
    )


def _spectrum_chunk_path(directory: Path, start: int, stop: int) -> Path:
    return directory / f"spectrum_{start:04d}_{stop:04d}.npz"


def _spectrum_chunk_payload_is_complete(
    payload: Mapping[str, Any], *, row_count: int, state_dimension: int
) -> bool:
    """Reject partial/non-finite spectra, including corrupted resumed chunks."""

    try:
        eigen_shape = (int(row_count), int(state_dimension))
        vector_shape = (int(row_count),)
        map_eigenvalues = np.asarray(payload["map_eigenvalues"])
        vector_eigenvalues = np.asarray(payload["vector_field_eigenvalues"])
        largest = np.asarray(payload["largest_real_part"])
        second = np.asarray(payload["second_largest_real_part"])
        gap = np.asarray(payload["real_part_gap"])
        map_radius = np.asarray(payload["map_spectral_radius"])
        tau_largest = np.asarray(payload["tau_largest"])
        tau_second = np.asarray(payload["tau_second"])
        if map_eigenvalues.shape != eigen_shape or vector_eigenvalues.shape != eigen_shape:
            return False
        if any(
            value.shape != vector_shape
            for value in (
                largest,
                second,
                gap,
                map_radius,
                tau_largest,
                tau_second,
            )
        ):
            return False
        if not all(
            np.isfinite(value).all()
            for value in (
                map_eigenvalues,
                vector_eigenvalues,
                largest,
                second,
                gap,
                map_radius,
            )
        ):
            return False
        # The reference path diagonalizes J_F and J_F-I independently, and
        # LAPACK does not promise the same eigenvalue ordering.  Compare each
        # row as a canonically sorted complex multiset rather than elementwise.
        for map_row, vector_row in zip(map_eigenvalues, vector_eigenvalues):
            if not np.allclose(
                np.sort_complex(vector_row),
                np.sort_complex(map_row - 1.0),
                rtol=1e-5,
                atol=1e-7,
            ):
                return False
        if not np.allclose(gap, largest - second, rtol=1e-6, atol=1e-8):
            return False
        if not np.allclose(
            map_radius,
            np.abs(map_eigenvalues).max(axis=1),
            rtol=1e-6,
            atol=1e-8,
        ):
            return False
        for eigen_real, tau in ((largest, tau_largest), (second, tau_second)):
            negative = eigen_real < 0.0
            if not np.isfinite(tau[negative]).all():
                return False
            if not np.isnan(tau[~negative]).all():
                return False
            if negative.any() and not np.allclose(
                tau[negative], -1.0 / eigen_real[negative], rtol=1e-6, atol=1e-8
            ):
                return False
        method = np.asarray(payload["computation_method"])
        audit = np.asarray(payload["fast_path_audit_status"])
        fallback = np.asarray(payload["fallback_reason"])
        if method.shape != () or audit.shape != () or fallback.shape != ():
            return False
        if str(method.item()) not in {
            "torch.func.vmap_jacrev_dense_JF_then_exact_spectral_shift",
            "reference_autograd_functional_dense_full_jacobian",
        }:
            return False
        if str(audit.item()) not in {
            "not_requested_after_first_chunk",
            "passed_against_slow_core_first_point",
            "fallback",
        }:
            return False
        for key in ("fast_path_audit_max_abs", "fast_path_audit_tolerance"):
            if np.asarray(payload[key]).shape != ():
                return False
        return True
    except (KeyError, TypeError, ValueError, FloatingPointError):
        return False


def _valid_spectrum_chunk(
    path: Path,
    *,
    identity: str,
    start: int,
    stop: int,
    state_dimension: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            expected = {
                "identity": identity,
                "start": start,
                "stop": stop,
                "state_dimension": state_dimension,
            }
            for key, value in expected.items():
                observed = payload[key].item()
                if observed != value:
                    return False
            return _spectrum_chunk_payload_is_complete(
                payload,
                row_count=stop - start,
                state_dimension=state_dimension,
            )
    except (KeyError, OSError, ValueError):
        return False


def _fast_dense_spectrum_chunk(
    state: torch.Tensor,
    adapter: StateAdapter,
    *,
    audit_against_reference: bool,
    jacrev_output_chunk_size: int = 16,
) -> dict[str, Any]:
    """Vectorize dense Jacobians across points and immediately discard them.

    ``torch.func.vmap(jacrev(...))`` still computes the exact dense full-state
    Jacobian; it only batches independent manifold points and output-basis
    reverse passes.  The spectrum of ``J_F-I`` is obtained by the exact
    spectral-shift identity ``eig(J_F-I) = eig(J_F)-1``.  The first chunk is
    audited against the slower model-independent core before this fast path is
    trusted for a checkpoint.
    """

    def single_map(value: torch.Tensor) -> torch.Tensor:
        result = adapter.actual_f0(value.unsqueeze(0)).squeeze(0)
        if result.shape != value.shape:
            raise ValueError("blank map changed the flattened full-state dimension")
        return result

    jacobian_function = torch.func.jacrev(
        single_map, chunk_size=int(jacrev_output_chunk_size)
    )
    map_jacobian = torch.func.vmap(jacobian_function)(state)
    if map_jacobian.shape != (
        state.shape[0],
        state.shape[1],
        state.shape[1],
    ):
        raise RuntimeError("torch.func returned an invalid batched full Jacobian")
    if not bool(torch.isfinite(map_jacobian).all()):
        raise StructuralNotEstimableError(
            "torch.func full Jacobian contains non-finite values"
        )

    audit_status = "not_requested_after_first_chunk"
    audit_max_abs = np.nan
    audit_tolerance = np.nan
    if audit_against_reference:
        reference = dense_full_jacobian_eigenspectrum(
            state[:1], adapter.actual_f0
        ).map_jacobian
        delta = (map_jacobian[:1] - reference).abs()
        audit_max_abs = float(delta.max().detach().cpu())
        scale = float(reference.abs().max().detach().cpu())
        epsilon = torch.finfo(state.dtype).eps
        audit_tolerance = max(1.0e-10, 64.0 * float(epsilon) * max(1.0, scale))
        if audit_max_abs > audit_tolerance:
            raise RuntimeError(
                "torch.func Jacobian audit disagreed with reference core: "
                f"{audit_max_abs} > {audit_tolerance}"
            )
        audit_status = "passed_against_slow_core_first_point"

    map_eigenvalues = torch.linalg.eigvals(map_jacobian)
    if not bool(torch.isfinite(map_eigenvalues).all()):
        raise StructuralNotEstimableError(
            "dense map-Jacobian eigenspectrum contains non-finite values"
        )
    # Spectral shift is exact for every square matrix and avoids a second
    # O(D^3) eigendecomposition at each of 1,024 points.
    vector_eigenvalues = map_eigenvalues - 1.0
    sorted_real = torch.sort(vector_eigenvalues.real, dim=1, descending=True).values
    if not bool(torch.isfinite(vector_eigenvalues).all() and torch.isfinite(sorted_real).all()):
        raise StructuralNotEstimableError(
            "dense vector-field eigenspectrum contains non-finite values"
        )
    return {
        "map_eigenvalues": _as_numpy(map_eigenvalues),
        "vector_field_eigenvalues": _as_numpy(vector_eigenvalues),
        "largest_real_part": _as_numpy(sorted_real[:, 0]),
        "second_largest_real_part": _as_numpy(sorted_real[:, 1]),
        "real_part_gap": _as_numpy(sorted_real[:, 0] - sorted_real[:, 1]),
        "map_spectral_radius": _as_numpy(map_eigenvalues.abs().max(dim=1).values),
        "computation_method": "torch.func.vmap_jacrev_dense_JF_then_exact_spectral_shift",
        "fast_path_audit_status": audit_status,
        "fast_path_audit_max_abs": audit_max_abs,
        "fast_path_audit_tolerance": audit_tolerance,
    }


def _reference_dense_spectrum_chunk(
    state: torch.Tensor,
    adapter: StateAdapter,
    *,
    fallback_reason: str,
) -> dict[str, Any]:
    spectrum = dense_full_jacobian_eigenspectrum(state, adapter.actual_f0)
    map_eigenvalues = _as_numpy(spectrum.map_eigenvalues)
    vector_eigenvalues = _as_numpy(spectrum.vector_field_eigenvalues)
    return {
        "map_eigenvalues": map_eigenvalues,
        "vector_field_eigenvalues": vector_eigenvalues,
        "largest_real_part": _as_numpy(spectrum.largest_real_part),
        "second_largest_real_part": _as_numpy(spectrum.second_largest_real_part),
        "real_part_gap": _as_numpy(spectrum.real_part_gap),
        "map_spectral_radius": np.abs(map_eigenvalues).max(axis=1),
        "computation_method": "reference_autograd_functional_dense_full_jacobian",
        "fast_path_audit_status": "fallback",
        "fast_path_audit_max_abs": np.nan,
        "fast_path_audit_tolerance": np.nan,
        "fallback_reason": fallback_reason,
    }


def compute_resumable_full_spectrum(
    state: torch.Tensor,
    adapter: StateAdapter,
    *,
    output_dir: Path,
    identity: str,
    chunk_size: int,
    progress_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Compute all dense spectra in atomic chunks and retain only eigenvalues."""

    chunk_dir = output_dir / "spectrum_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    point_count, state_dimension = map(int, state.shape)
    paths: list[Path] = []
    force_reference_fallback = False
    for start in range(0, point_count, int(chunk_size)):
        stop = min(start + int(chunk_size), point_count)
        path = _spectrum_chunk_path(chunk_dir, start, stop)
        if not _valid_spectrum_chunk(
            path,
            identity=identity,
            start=start,
            stop=stop,
            state_dimension=state_dimension,
        ):
            chunk_state = state[start:stop]
            try:
                if force_reference_fallback:
                    raise RuntimeError("fast path disabled after first-chunk fallback")
                result = _fast_dense_spectrum_chunk(
                    chunk_state,
                    adapter,
                    audit_against_reference=start == 0,
                )
            except torch.cuda.OutOfMemoryError:
                raise
            except (NotImplementedError, RuntimeError) as error:
                if start == 0:
                    force_reference_fallback = True
                result = _reference_dense_spectrum_chunk(
                    chunk_state,
                    adapter,
                    fallback_reason=f"{type(error).__name__}:{error}",
                )
            map_eigenvalues = result["map_eigenvalues"]
            vector_eigenvalues = result["vector_field_eigenvalues"]
            map_radius = result["map_spectral_radius"]
            largest = result["largest_real_part"]
            second = result["second_largest_real_part"]
            gap = result["real_part_gap"]
            if not all(
                np.isfinite(np.asarray(value)).all()
                for value in (
                    map_eigenvalues,
                    vector_eigenvalues,
                    map_radius,
                    largest,
                    second,
                    gap,
                )
            ):
                raise StructuralNotEstimableError(
                    "full local eigenspectrum contains non-finite values"
                )
            tau_largest = np.full(largest.shape, np.nan, dtype=np.float64)
            tau_second = np.full(second.shape, np.nan, dtype=np.float64)
            tau_largest[largest < 0] = -1.0 / largest[largest < 0]
            tau_second[second < 0] = -1.0 / second[second < 0]
            if not (
                np.isfinite(tau_largest[largest < 0]).all()
                and np.isfinite(tau_second[second < 0]).all()
            ):
                raise StructuralNotEstimableError(
                    "negative real-part timescale contains non-finite values"
                )
            _atomic_npz(
                path,
                identity=np.asarray(identity),
                start=np.asarray(start, dtype=np.int64),
                stop=np.asarray(stop, dtype=np.int64),
                state_dimension=np.asarray(state_dimension, dtype=np.int64),
                map_eigenvalues=map_eigenvalues,
                vector_field_eigenvalues=vector_eigenvalues,
                largest_real_part=largest,
                second_largest_real_part=second,
                real_part_gap=gap,
                map_spectral_radius=map_radius,
                tau_largest=tau_largest,
                tau_second=tau_second,
                computation_method=np.asarray(result["computation_method"]),
                fast_path_audit_status=np.asarray(
                    result["fast_path_audit_status"]
                ),
                fast_path_audit_max_abs=np.asarray(
                    result["fast_path_audit_max_abs"], dtype=np.float64
                ),
                fast_path_audit_tolerance=np.asarray(
                    result["fast_path_audit_tolerance"], dtype=np.float64
                ),
                fallback_reason=np.asarray(result.get("fallback_reason", "")),
            )
        elif start == 0:
            with np.load(path, allow_pickle=False) as payload:
                force_reference_fallback = (
                    payload["computation_method"].item()
                    == "reference_autograd_functional_dense_full_jacobian"
                )
        paths.append(path)
        atomic_json(
            progress_path,
            {
                "schema_version": SCHEMA_VERSION,
                "analysis_identity": identity,
                "stage": "full_local_jacobian_eigenspectrum",
                "completed_points": stop,
                "total_points": point_count,
                "chunk_size": int(chunk_size),
                "updated_unix_seconds": time.time(),
            },
        )

    arrays: dict[str, list[np.ndarray]] = {
        "map_eigenvalues": [],
        "vector_field_eigenvalues": [],
        "largest_real_part": [],
        "second_largest_real_part": [],
        "real_part_gap": [],
        "map_spectral_radius": [],
        "tau_largest": [],
        "tau_second": [],
    }
    computation_methods: list[str] = []
    fast_path_audits: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            for key in arrays:
                arrays[key].append(np.asarray(payload[key]))
            method = str(payload["computation_method"].item())
            computation_methods.append(method)
            audit_status = str(payload["fast_path_audit_status"].item())
            if audit_status != "not_requested_after_first_chunk":
                fast_path_audits.append(
                    {
                        "status": audit_status,
                        "max_abs_difference": (
                            None
                            if not np.isfinite(payload["fast_path_audit_max_abs"].item())
                            else float(payload["fast_path_audit_max_abs"].item())
                        ),
                        "tolerance": (
                            None
                            if not np.isfinite(payload["fast_path_audit_tolerance"].item())
                            else float(payload["fast_path_audit_tolerance"].item())
                        ),
                        "fallback_reason": str(payload["fallback_reason"].item()),
                    }
                )
    combined = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    destination = output_dir / "full_local_eigenspectrum.npz"
    _atomic_npz(
        destination,
        analysis_identity=np.asarray(identity),
        spline_index=np.arange(point_count, dtype=np.int64),
        lambda1_real=combined["largest_real_part"],
        lambda2_real=combined["second_largest_real_part"],
        gap=combined["real_part_gap"],
        **combined,
    )
    summary = {
        "method": "dense_full_state_autograd_JF_and_JF_minus_I_at_every_spline_point",
        "state_dimension": state_dimension,
        "point_count": point_count,
        "complex_eigenvalues_retained": True,
        "computation_methods": sorted(set(computation_methods)),
        "fast_path_audit": fast_path_audits,
        "vector_field_spectrum_method": (
            "exact spectral-shift eig(J_F-I)=eig(J_F)-1 on torch.func path; "
            "direct eigendecomposition on reference fallback"
        ),
        "primary_sort_convention": "descending_real_part_of_JF_minus_I",
        "largest_real_part": _finite_summary(combined["largest_real_part"]),
        "second_largest_real_part": _finite_summary(
            combined["second_largest_real_part"]
        ),
        "top_two_real_part_gap": _finite_summary(combined["real_part_gap"]),
        "map_spectral_radius": _finite_summary(combined["map_spectral_radius"]),
        "map_spectral_radius_below_one_fraction": float(
            np.mean(combined["map_spectral_radius"] < 1.0)
        ),
        "tau_largest_when_negative": _finite_summary(combined["tau_largest"]),
        "tau_second_when_negative": _finite_summary(combined["tau_second"]),
        "claim_threshold": None,
    }
    return destination, summary


@torch.no_grad()
def finite_blank_memory(
    model: ProtocolModel,
    adapter: StateAdapter,
    spline_state: torch.Tensor,
    target_angle: torch.Tensor,
    *,
    horizon: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Roll the 1,024 spline memories through all times ``0..16T``."""

    memories = int(spline_state.shape[0])
    predicted_device = torch.empty(
        memories,
        int(horizon) + 1,
        device=spline_state.device,
        dtype=spline_state.dtype,
    )
    reported = _reported_with_blank_decode_stream(model, adapter, spline_state)
    output = adapter.decode(reported)
    predicted_device[:, 0] = angle_from_output(output)
    for step in range(1, int(horizon) + 1):
        reported = adapter.reported_step(reported, adapter.zero_input(reported))
        output = adapter.decode(reported)
        predicted_device[:, step] = angle_from_output(output)
    predicted = _as_numpy(predicted_device, dtype=np.float32)
    predicted_tensor = torch.as_tensor(predicted, dtype=torch.float32)
    target_cpu = target_angle.detach().cpu().to(dtype=torch.float32)
    metrics = finite_time_angular_memory(predicted_tensor, target_cpu)
    arrays = {
        "time": np.arange(int(horizon) + 1, dtype=np.int64),
        "target_angle": _as_numpy(target_cpu),
        "predicted_angle": predicted,
        "signed_error": _as_numpy(metrics.signed_error, dtype=np.float32),
        "absolute_error": _as_numpy(metrics.absolute_error, dtype=np.float32),
        "instantaneous_minimum_error": _as_numpy(metrics.minimum_error),
        "instantaneous_mean_error": _as_numpy(metrics.mean_error),
        "instantaneous_maximum_error": _as_numpy(metrics.maximum_error),
        "cumulative_minimum_error": _as_numpy(metrics.cumulative_minimum_error),
        "cumulative_mean_error": _as_numpy(metrics.cumulative_mean_error),
        "cumulative_maximum_error": _as_numpy(metrics.cumulative_maximum_error),
    }
    named: dict[str, Any] = {}
    for multiplier in (1, 3, 5, 7, 9):
        index = multiplier * FULL_TASK_HORIZON
        if index <= horizon:
            named[f"{multiplier}T"] = {
                "step": index,
                "instantaneous_mean_error_radians": float(metrics.mean_error[index]),
                "instantaneous_maximum_error_radians": float(metrics.maximum_error[index]),
                "cumulative_mean_error_radians": float(
                    metrics.cumulative_mean_error[index]
                ),
            }
    summary = {
        "initial_memory_count": memories,
        "time_indexing": "inclusive_0_through_blank_horizon",
        "blank_horizon": int(horizon),
        "named_horizons": named,
        "terminal_mean_error_radians": float(metrics.mean_error[-1]),
        "terminal_maximum_error_radians": float(metrics.maximum_error[-1]),
    }
    return arrays, summary


def _cyclic_spacing(angle: Sequence[float]) -> list[float]:
    values = np.sort(np.remainder(np.asarray(tuple(angle), dtype=np.float64), 2.0 * math.pi))
    if not values.size:
        return []
    return np.diff(np.concatenate((values, values[:1] + 2.0 * math.pi))).tolist()


def _geometric_basin_assignments(
    initial_angle: torch.Tensor,
    capacity: Any,
) -> torch.Tensor:
    """Assign angles by the saddle-to-saddle basin containing each angle."""

    angle = torch.remainder(initial_angle, 2.0 * math.pi)
    assignment = torch.full(
        angle.shape, -1, dtype=torch.int64, device=angle.device
    )
    for index, (left, width) in enumerate(
        zip(capacity.left_saddle_angle, capacity.basin_width_radians)
    ):
        relative = torch.remainder(angle - left, 2.0 * math.pi)
        inside = relative < width
        # The exact right boundary belongs deterministically to the next basin.
        assignment[inside & (assignment < 0)] = int(index)
    if bool((assignment < 0).any()):
        # Floating seam roundoff can leave an exact boundary unassigned.
        stable = capacity.stable_fixed_point_angle
        distance = circular_absolute_error(angle[:, None], stable[None, :])
        missing = assignment < 0
        assignment[missing] = distance[missing].argmin(dim=1)
    return assignment


def _asymptotic_summary(
    topology: Any,
    initial_angle: torch.Tensor,
    terminal_angle: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    stable_angle = torch.tensor(
        [item.angle for item in topology.stable],
        dtype=initial_angle.dtype,
        device=initial_angle.device,
    )
    saddle_angle = torch.tensor(
        [item.angle for item in topology.saddles],
        dtype=initial_angle.dtype,
        device=initial_angle.device,
    )
    arrays: dict[str, np.ndarray] = {
        "initial_angle": _as_numpy(initial_angle),
        "observed_terminal_angle": _as_numpy(terminal_angle),
        "stable_fixed_point_angle": _as_numpy(stable_angle),
        "saddle_fixed_point_angle": _as_numpy(saddle_angle),
    }
    if topology.kind == "limit_cycle":
        observed = asymptotic_memory_metrics(
            initial_angle,
            terminal_angle,
            stable_angle,
            topology="limit_cycle",
        )
        arrays["observed_terminal_absolute_error"] = _as_numpy(
            observed.observed_terminal_absolute_error
        )
        return (
            {
                "topology": "limit_cycle",
                "fixed_point_basin_capacity": None,
                "capacity_status": "N/A_for_limit_cycle",
                "asymptotic_mean_error_radians": None,
                "asymptotic_maximum_error_radians": math.pi,
                "asymptotic_maximum_error_basis": (
                    "unidirectional_cycle_eventually_reaches_antipodal_phase"
                ),
                "observed_terminal_mean_error_radians": float(
                    observed.observed_terminal_mean_error
                ),
                "observed_terminal_maximum_error_radians": float(
                    observed.observed_terminal_maximum_error
                ),
            },
            arrays,
        )
    if topology.kind != "fixed_points" or not stable_angle.numel():
        return (
            {
                "topology": topology.kind,
                "fixed_point_basin_capacity": None,
                "capacity_status": "not_estimable_without_isolated_alternating_fixed_points",
            },
            arrays,
        )
    try:
        capacity = stable_basin_capacity(stable_angle, saddle_angle)
    except StructuralNotEstimableError as error:
        return (
            {
                "topology": topology.kind,
                "fixed_point_basin_capacity": None,
                "capacity_status": f"not_estimable:{type(error).__name__}:{error}",
            },
            arrays,
        )
    assignment = _geometric_basin_assignments(initial_angle, capacity)
    assigned = capacity.stable_fixed_point_angle[assignment]
    asymptotic_error = circular_absolute_error(assigned, initial_angle)
    empirical_counts = torch.bincount(
        assignment, minlength=capacity.stable_fixed_point_angle.numel()
    )
    arrays.update(
        {
            "geometric_left_saddle_angle": _as_numpy(capacity.left_saddle_angle),
            "geometric_right_saddle_angle": _as_numpy(capacity.right_saddle_angle),
            "geometric_basin_width_radians": _as_numpy(capacity.basin_width_radians),
            "geometric_basin_proportions": _as_numpy(capacity.basin_proportions),
            "uniform_grid_basin_assignment": _as_numpy(assignment),
            "uniform_grid_basin_counts": _as_numpy(empirical_counts),
            "assigned_stable_angle": _as_numpy(assigned),
            "asymptotic_absolute_error": _as_numpy(asymptotic_error),
        }
    )
    return (
        {
            "topology": "fixed_points",
            "capacity_status": "estimated_from_flow_reversal_saddle_boundaries",
            "stable_count": int(stable_angle.numel()),
            "saddle_count": int(saddle_angle.numel()),
            "stable_ordered_spacing_radians": _cyclic_spacing(_as_numpy(stable_angle)),
            "saddle_ordered_spacing_radians": _cyclic_spacing(_as_numpy(saddle_angle)),
            "basin_width_radians": _as_numpy(
                capacity.basin_width_radians
            ).tolist(),
            "basin_proportions": _as_numpy(capacity.basin_proportions).tolist(),
            "shannon_entropy_nats": float(capacity.shannon_entropy_nats),
            "effective_basin_count": float(capacity.effective_basin_count),
            "asymptotic_mean_error_radians": float(asymptotic_error.mean()),
            "asymptotic_maximum_error_radians": float(asymptotic_error.max()),
        },
        arrays,
    )


def _protocol_task_batch(
    protocol: Mapping[str, Any],
    task_spec: AngularTaskSpec,
    spec: PrimaryAnalysisSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    kwargs = task_spec.generator_kwargs()
    if spec.smoke:
        kwargs["horizon"] = int(spec.task_horizon)
        # The generator's registered mask string names 256 steps even in a
        # smoke reduction; this is metadata-only in smoke artifacts.
    task_seed = int(protocol["seed_policy"]["task_seed"])
    return angular_integration(
        spec.trajectory_count,
        task_seed,
        **kwargs,
        stream_key=("sagodi_primary_direct_1024_task_trajectories",),
        dtype=dtype,
        device=device,
    )


def _initialize_output(
    output_dir: Path,
    identity_payload: Mapping[str, Any],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = canonical_hash(identity_payload)
    identity_path = output_dir / "analysis_identity.json"
    if identity_path.exists():
        observed = json.loads(identity_path.read_text(encoding="utf-8"))
        if observed.get("analysis_identity") != identity:
            raise FileExistsError(
                "output directory belongs to a different analysis identity"
            )
    else:
        unexpected = [path for path in output_dir.iterdir() if path.name != "spectrum_chunks"]
        if unexpected:
            raise FileExistsError("refusing to reuse an unidentified non-empty output directory")
        atomic_json(
            identity_path,
            {
                "schema_version": SCHEMA_VERSION,
                "analysis_identity": identity,
                "identity_payload": dict(identity_payload),
            },
        )
    return identity


def _publish_structural_not_estimable(
    *,
    destination: Path,
    base_summary: dict[str, Any],
    progress: Path,
    completion: Path,
    identity: str,
    model_name: str,
    task_trajectory_path: Path,
    failed_stage: str,
    error: StructuralNotEstimableError,
    completed_artifacts: Sequence[Path] = (),
) -> Path:
    """Publish a terminal, denominator-preserving numerical non-estimability.

    This path is reserved for the dedicated structural/domain exception.
    Ordinary ``ValueError`` programmer-contract failures, CUDA OOMs, killed
    processes, I/O failures, source drift, and other infrastructure errors are
    intentionally not classified here; the campaign leaves retryable failures
    under the same seed.
    """

    if not isinstance(error, StructuralNotEstimableError):
        raise TypeError(
            "only StructuralNotEstimableError can publish scientific non-estimability"
        )

    reason = f"{type(error).__name__}:{error}"
    base_summary["analysis_status"] = "structural_analysis_not_estimable"
    base_summary["structural_numerical_failure"] = {
        "status": "not_estimable",
        "failed_stage": failed_stage,
        "reason": reason,
        "claim_gate": False,
        "seed_replacement": False,
    }
    summary_path = destination / "summary.json"
    atomic_json(summary_path, base_summary)
    atomic_json(
        progress,
        {
            "schema_version": SCHEMA_VERSION,
            "analysis_identity": identity,
            "stage": "complete_structural_analysis_not_estimable",
            "failed_stage": failed_stage,
            "updated_unix_seconds": time.time(),
        },
    )
    artifacts: list[Path] = [
        destination / "analysis_identity.json",
        progress,
        task_trajectory_path,
    ]
    artifacts.extend(path for path in completed_artifacts if path.is_file())
    artifacts.append(summary_path)
    # Preserve order while avoiding duplicate receipt entries.
    unique_artifacts = list(dict.fromkeys(artifacts))
    write_completion_receipt(
        completion,
        job_id=f"sagodi-primary-{model_name}-{identity[:12]}",
        artifacts=unique_artifacts,
        metadata={
            "analysis_identity": identity,
            "analysis_status": "structural_analysis_not_estimable",
            "failed_stage": failed_stage,
        },
    )
    return summary_path


def _bound_main_validation_metrics(
    *,
    checkpoint: Path,
    checkpoint_payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_source: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the frozen 2,048-trial ID metric without touching discovery data.

    The main trainer embeds its final evaluation metrics and immutable
    evaluation-bank SHA-256 in ``checkpoint.extra``.  That is already bound by
    the checkpoint bytes.  When the checkpoint remains in its training run
    directory, the sibling metrics/manifest/receipt are verified as a second
    binding.  A relocated checkpoint is accepted only when the complete
    checkpoint-extra binding is present; a loose sibling JSON is never
    trusted by itself.
    """

    phase = protocol.get("phase1_ring_pilot")
    evaluation = protocol.get("evaluation")
    if not isinstance(phase, Mapping) or phase.get("protocol_track") != SAGODI_PRIMARY_MAIN_TRACK:
        raise ValueError("full primary analysis requires a primary-main protocol")
    if not isinstance(evaluation, Mapping):
        raise ValueError("primary-main protocol has no evaluation freeze")
    registered_counts = {
        key: int(evaluation.get(key, -1))
        for key in ("validation_trials", "id_test_trials")
    }
    if 2048 not in registered_counts.values():
        raise ValueError("primary-main protocol does not bind a 2,048-trial ID evaluation")

    extra = checkpoint_payload.get("extra")
    if not isinstance(extra, Mapping):
        raise ValueError("full main checkpoint has no identity/validation extra block")
    expected_identity = {
        "protocol_track": SAGODI_PRIMARY_MAIN_TRACK,
        "protocol_file_sha256": sha256_file(protocol_source),
        "protocol_canonical_fingerprint": protocol_fingerprint(protocol),
    }
    for key, expected in expected_identity.items():
        if extra.get(key) != expected:
            raise ValueError(f"main checkpoint validation binding mismatch for {key}")
    evaluation_bank_sha256 = extra.get("evaluation_bank_sha256")
    if (
        not isinstance(evaluation_bank_sha256, str)
        or len(evaluation_bank_sha256) != 64
        or any(character not in "0123456789abcdef" for character in evaluation_bank_sha256)
    ):
        raise ValueError("full main checkpoint lacks a valid evaluation-bank SHA-256")

    embedded = extra.get("task_metrics")
    metrics: dict[str, Any] | None = dict(embedded) if isinstance(embedded, Mapping) else None
    sibling_metrics_path = checkpoint.parent / "task_metrics.json"
    sibling_manifest_path = checkpoint.parent / "manifest.json"
    sibling_receipt_path = checkpoint.parent / "completion_receipt.json"
    sibling_binding: dict[str, Any] | None = None
    sibling_presence = [
        sibling_metrics_path.exists(),
        sibling_manifest_path.exists(),
        sibling_receipt_path.exists(),
    ]
    if any(sibling_presence):
        if not all(sibling_presence):
            raise ValueError("main checkpoint has an incomplete sibling validation binding")
        valid, reason = verify_completion_receipt(sibling_receipt_path)
        if not valid:
            raise ValueError(f"main training receipt is invalid: {reason}")
        sibling_metrics = strict_json_load(sibling_metrics_path)
        sibling_manifest = strict_json_load(sibling_manifest_path)
        if not isinstance(sibling_metrics, Mapping) or not isinstance(sibling_manifest, Mapping):
            raise ValueError("main sibling validation artifacts are malformed")
        if metrics is not None and dict(sibling_metrics) != metrics:
            raise ValueError("checkpoint-extra and sibling task metrics differ")
        metrics = dict(sibling_metrics)
        expected_manifest = {
            "checkpoint_sha256": sha256_file(checkpoint),
            "evaluation_bank_sha256": evaluation_bank_sha256,
            "protocol_canonical_fingerprint": expected_identity[
                "protocol_canonical_fingerprint"
            ],
            "protocol_track": SAGODI_PRIMARY_MAIN_TRACK,
            "task_metrics_path": "task_metrics.json",
        }
        for key, expected in expected_manifest.items():
            if sibling_manifest.get(key) != expected:
                raise ValueError(f"main sibling manifest mismatch for {key}")
        sibling_binding = {
            "task_metrics_sha256": sha256_file(sibling_metrics_path),
            "manifest_sha256": sha256_file(sibling_manifest_path),
            "completion_receipt_sha256": sha256_file(sibling_receipt_path),
        }

    if metrics is None:
        raise ValueError(
            "full main checkpoint lacks frozen task_metrics in checkpoint extra or a verified sibling"
        )
    try:
        inclusion_value = float(metrics["masked_nmse_db"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("bound main task metrics lack finite masked_nmse_db") from error
    if not math.isfinite(inclusion_value):
        raise ValueError("bound main masked_nmse_db is non-finite")
    return metrics, {
        "source": (
            "checkpoint_extra_plus_verified_training_siblings"
            if sibling_binding is not None
            else "checkpoint_extra_bound_by_checkpoint_bytes"
        ),
        "checkpoint_sha256": sha256_file(checkpoint),
        "evaluation_bank_sha256": evaluation_bank_sha256,
        "registered_evaluation_trial_counts": registered_counts,
        "metric_key": "masked_nmse_db",
        "sibling_binding": sibling_binding,
    }


def run_primary_analysis(
    *,
    checkpoint_path: Path | str,
    protocol_path: Path | str,
    output_dir: Path | str,
    device: str = "cuda:0",
    spec: PrimaryAnalysisSpec | None = None,
) -> Path:
    """Run or resume one checkpoint's primary analysis and return its summary."""

    active_spec = spec or PrimaryAnalysisSpec()
    active_spec.validate()
    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    protocol_source = Path(protocol_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    protocol = load_protocol(protocol_source)
    task_spec = AngularTaskSpec.from_protocol(protocol)
    if (
        not active_spec.source_v6
        and not active_spec.smoke
        and task_spec.sequence_steps != active_spec.task_horizon
    ):
        raise ValueError("protocol task horizon differs from the primary analysis freeze")
    target_device = torch.device(device)
    if active_spec.source_v6:
        model, checkpoint_payload = load_source_v6_checkpoint(checkpoint, target_device)
    else:
        model, checkpoint_payload = load_checkpoint(checkpoint, target_device)
    model.eval()
    if model.input_dim != 1 or model.output_dim != 2 or model.config.init_mode != "hidden_init":
        raise ValueError("checkpoint is not a hidden-init one-angle integration model")
    valid_models = (
        set(SOURCE_V6_MODEL_IDS)
        if active_spec.source_v6
        else {str(entry["id"]) for entry in protocol["phase1_ring_pilot"]["models"]}
    )
    if model.config.name not in valid_models:
        raise ValueError("checkpoint model is outside the supplied protocol")
    adapter = StateAdapter(model.core)
    dtype = next(model.parameters()).dtype
    identity_payload = primary_analysis_identity_payload(
        checkpoint_sha256=sha256_file(checkpoint),
        protocol_sha256=sha256_file(protocol_source),
        protocol_fingerprint_value=protocol_fingerprint(protocol),
        model_name=model.config.name,
        spec=active_spec,
    )
    bound_validation_metrics: dict[str, Any] | None = None
    bound_validation_binding: dict[str, Any] | None = None
    if not active_spec.smoke:
        if active_spec.source_v6:
            result = checkpoint_payload.get("result")
            metrics = result.get("final_metrics") if isinstance(result, Mapping) else None
            if not isinstance(metrics, Mapping):
                raise ValueError("source-v6 checkpoint lacks final clean task metrics")
            bound_validation_metrics = dict(metrics)
            bound_validation_binding = {
                "source": "source_v6_checkpoint_bound_result",
                "checkpoint_sha256": sha256_file(checkpoint),
                "metric_key": "masked_nmse_db",
            }
        else:
            bound_validation_metrics, bound_validation_binding = (
                _bound_main_validation_metrics(
                    checkpoint=checkpoint,
                    checkpoint_payload=checkpoint_payload,
                    protocol=protocol,
                    protocol_source=protocol_source,
                )
            )
    identity = _initialize_output(destination, identity_payload)
    completion = destination / "completion_receipt.json"
    if completion.is_file():
        valid, reason = verify_completion_receipt(
            completion,
            expected_job_id=f"sagodi-primary-{model.config.name}-{identity[:12]}",
            expected_metadata={"analysis_identity": identity},
        )
        if not valid:
            raise RuntimeError(f"existing completion receipt is invalid: {reason}")
        return destination / "summary.json"
    progress = destination / "progress.json"
    atomic_json(
        progress,
        {
            "schema_version": SCHEMA_VERSION,
            "analysis_identity": identity,
            "stage": "direct_task_trajectory_generation",
            "updated_unix_seconds": time.time(),
        },
    )

    batch = (
        _source_v6_task_batch(active_spec, device=target_device, dtype=dtype)
        if active_spec.source_v6
        else _protocol_task_batch(
            protocol, task_spec, active_spec, device=target_device, dtype=dtype
        )
    )
    prediction, endpoint_reported = _task_rollout(model, adapter, batch)
    discovery_task_metrics = task_metrics(
        prediction, batch.output_targets, batch.mask, batch.latent_targets
    )
    task_trajectory_path = destination / "direct_task_trajectories.npz"
    _atomic_npz(
        task_trajectory_path,
        inputs=_as_numpy(batch.inputs),
        output_targets=_as_numpy(batch.output_targets),
        latent_targets=_as_numpy(batch.latent_targets),
        mask=_as_numpy(batch.mask),
        initial_memory=_as_numpy(batch.initial_memory),
        task_prediction=_as_numpy(prediction),
        endpoint_reported_state=_as_numpy(endpoint_reported),
        endpoint_primary_state=_as_numpy(
            adapter.primary_from_reported(endpoint_reported)
        ),
        analysis_recurrent_state_noise_enabled=np.asarray(False, dtype=np.bool_),
        analysis_recurrent_state_noise_std=np.asarray(0.0, dtype=np.float64),
    )
    eligible = bool(
        bound_validation_metrics is not None
        and float(bound_validation_metrics["masked_nmse_db"])
        < float(active_spec.inclusion_nmse_db)
    )
    base_summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "sagodi_primary_single_checkpoint",
        "analysis_identity": identity,
        "analysis_role": "Ságodi_evaluation_tool_not_CA_LRU_method",
        "smoke": bool(active_spec.smoke),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "extra": checkpoint_payload.get("extra", {}),
        },
        "protocol": {
            "path": str(protocol_source),
            "sha256": sha256_file(protocol_source),
            "fingerprint": protocol_fingerprint(protocol),
            "freeze_id": protocol["freeze_id"],
        },
        "model": model.metadata(),
        "retention_finite_precision_diagnostics": (
            _retention_finite_precision_diagnostics(model)
        ),
        "state_spec": adapter.state_spec(),
        "analysis_spec": primary_analysis_spec_payload(active_spec),
        "project_resolutions": {
            "task_to_blank_staging": (
                "state-noise-disabled held-out GP task rollout for T steps, followed "
                "by exact 16T deterministic zero-input dynamics; the sampled GP "
                "velocity is task signal, not additive recurrent-state noise"
            ),
            "source_v6_public_code_task": bool(active_spec.source_v6),
            "nonlinear_decoder_projected_flow": (
                "D(F0(s))-D(s), common one-step discrete adaptation of the paper's "
                "linear output-projected vector field"
            ),
            "candidate_selection": (
                "nearest in Euclidean actual decoded output space to 1024 uniform "
                "unit-ring targets"
            ),
            "analysis_state_noise": {
                "enabled": False,
                "task_rollout_call": "ProtocolModel.step(..., state_noise=None)",
                "blank_rollout_call": "deterministic StateAdapter blank-input map",
                "model_mode": "eval",
                "training_noise_is_not_replayed": True,
            },
        },
        "frozen_main_id_validation": {
            "metrics": bound_validation_metrics,
            "binding": bound_validation_binding,
            "used_for_structural_summary_eligibility": not active_spec.smoke,
        },
        "discovery_task_metrics": {
            "metrics": discovery_task_metrics,
            "role": "descriptive_only_not_used_for_minus20dB_inclusion",
            "trajectory_count": int(active_spec.trajectory_count),
        },
        "structural_summary_eligibility": {
            "criterion": (
                "source_v6_checkpoint_clean_fixed_bank_masked_NMSE_dB < -20"
                if active_spec.source_v6
                else "frozen_2048_trial_main_ID_masked_NMSE_dB < -20"
            ),
            "threshold_nmse_db": float(active_spec.inclusion_nmse_db),
            "eligible": bool(eligible),
            "smoke_bypass": bool(active_spec.smoke),
        },
    }
    if not eligible and not active_spec.smoke:
        base_summary["analysis_status"] = "ineligible_for_structural_summary"
        summary_path = destination / "summary.json"
        atomic_json(summary_path, base_summary)
        atomic_json(
            progress,
            {
                "schema_version": SCHEMA_VERSION,
                "analysis_identity": identity,
                "stage": "complete_ineligible_for_structural_summary",
                "updated_unix_seconds": time.time(),
            },
        )
        write_completion_receipt(
            completion,
            job_id=f"sagodi-primary-{model.config.name}-{identity[:12]}",
            artifacts=[
                destination / "analysis_identity.json",
                progress,
                task_trajectory_path,
                summary_path,
            ],
            metadata={
                "analysis_identity": identity,
                "analysis_status": "ineligible_for_structural_summary",
            },
        )
        return summary_path

    atomic_json(
        progress,
        {
            "schema_version": SCHEMA_VERSION,
            "analysis_identity": identity,
            "stage": "slow_manifold_reconstruction",
            "updated_unix_seconds": time.time(),
        },
    )
    spline_angle = _uniform_angles(
        active_spec.spline_count, device=target_device, dtype=dtype
    )
    try:
        reconstruction = reconstruct_slow_manifold(
            model, adapter, endpoint_reported, spline_angle, active_spec
        )
    except torch.cuda.OutOfMemoryError:
        # Resource failures are retryable infrastructure outcomes, never a
        # scientific non-estimability label for this seed.
        raise
    except StructuralNotEstimableError as error:
        base_summary["manifold_reconstruction"] = {
            "status": "not_estimable",
            "reason": f"{type(error).__name__}:{error}",
            "claim_gate": False,
        }
        return _publish_structural_not_estimable(
            destination=destination,
            base_summary=base_summary,
            progress=progress,
            completion=completion,
            identity=identity,
            model_name=model.config.name,
            task_trajectory_path=task_trajectory_path,
            failed_stage="slow_manifold_reconstruction",
            error=error,
        )

    reconstruction_path = destination / "slow_manifold_reconstruction.npz"
    _atomic_npz(
        reconstruction_path,
        spline_angle=_as_numpy(reconstruction.spline_angle),
        spline_state=_as_numpy(reconstruction.spline_state),
        selected_state=_as_numpy(reconstruction.selected_state),
        selected_candidate_time=reconstruction.selected_candidate_time,
        selected_candidate_trajectory=reconstruction.selected_candidate_trajectory,
        selected_decoded_output=reconstruction.selected_decoded_output,
        selected_output_distance=reconstruction.selected_output_distance,
        knot_angle=_as_numpy(reconstruction.knot_angle),
        knot_state=_as_numpy(reconstruction.knot_state),
        maximum_speed=reconstruction.maximum_speed,
        candidate_count_per_trajectory=reconstruction.candidate_count_per_trajectory,
    )
    base_summary["manifold_reconstruction"] = {
        "status": "estimated",
        "trajectory_source": (
            "direct_exact_count_state_noise_disabled_held_out_GP_task_trajectories"
        ),
        "task_horizon": active_spec.task_horizon,
        "autonomous_blank_horizon": active_spec.blank_horizon,
        "slow_relative_speed": active_spec.slow_relative_speed,
        "spline_point_count": active_spec.spline_count,
        "qa": reconstruction.qa,
        "claim_gate": False,
    }

    normal_anchor_count = int(active_spec.normal_recovery_anchor_count)
    normal_horizons = tuple(active_spec.normal_recovery_horizons)
    if active_spec.smoke:
        normal_anchor_count = min(normal_anchor_count, active_spec.spline_count)
        normal_horizons = tuple(
            value for value in normal_horizons if value <= active_spec.blank_horizon
        )
        if active_spec.blank_horizon not in normal_horizons:
            normal_horizons = (*normal_horizons, int(active_spec.blank_horizon))
    atomic_json(
        progress,
        {
            "schema_version": SCHEMA_VERSION,
            "analysis_identity": identity,
            "stage": "carrier_ambient_normal_recovery",
            "updated_unix_seconds": time.time(),
        },
    )
    try:
        recovery = carrier_ambient_normal_recovery(
            reconstruction.spline_state,
            reconstruction.spline_angle,
            adapter.actual_f0,
            lambda state: _blank_decode_primary(model, adapter, state),
            anchor_count=normal_anchor_count,
            ambient_directions_per_anchor=(
                active_spec.normal_recovery_ambient_directions
            ),
            radii_over_manifold_scale=tuple(
                active_spec.normal_recovery_radii_over_manifold_scale
            ),
            horizons=normal_horizons,
            seed=active_spec.normal_recovery_seed,
            distance_chunk_size=active_spec.candidate_distance_chunk_size,
        )
        recovery_summary = _carrier_normal_recovery_summary(
            recovery,
            seed=active_spec.normal_recovery_seed,
            anchor_count=normal_anchor_count,
            ambient_directions_per_anchor=(
                active_spec.normal_recovery_ambient_directions
            ),
            radii_over_manifold_scale=tuple(
                active_spec.normal_recovery_radii_over_manifold_scale
            ),
            horizons=normal_horizons,
            smoke=active_spec.smoke,
        )
    except torch.cuda.OutOfMemoryError:
        raise
    except StructuralNotEstimableError as error:
        return _publish_structural_not_estimable(
            destination=destination,
            base_summary=base_summary,
            progress=progress,
            completion=completion,
            identity=identity,
            model_name=model.config.name,
            task_trajectory_path=task_trajectory_path,
            failed_stage="carrier_ambient_normal_recovery",
            error=error,
            completed_artifacts=(reconstruction_path,),
        )
    recovery_path = destination / "carrier_ambient_normal_recovery.npz"
    (
        persisted_direction,
        persisted_tangent,
        persisted_direction_norm_error,
        persisted_absolute_tangent_dot_direction,
    ) = _persisted_direction_arrays_and_qa(recovery)
    _atomic_npz(
        recovery_path,
        family=np.asarray(recovery.family, dtype="U32"),
        anchor_index=_as_numpy(recovery.anchor_index, dtype=np.int64),
        anchor_angle=_as_numpy(recovery.anchor_angle),
        direction_index=_as_numpy(recovery.direction_index, dtype=np.int64),
        radius_over_manifold_scale=_as_numpy(
            recovery.radius_over_manifold_scale
        ),
        radius_absolute=_as_numpy(recovery.radius_absolute),
        direction=persisted_direction,
        tangent=persisted_tangent,
        direction_norm_error=persisted_direction_norm_error,
        absolute_tangent_dot_direction=(
            persisted_absolute_tangent_dot_direction
        ),
        horizon=_as_numpy(recovery.horizon, dtype=np.int64),
        nearest_manifold_index=_as_numpy(
            recovery.nearest_manifold_index, dtype=np.int64
        ),
        manifold_distance=_as_numpy(recovery.manifold_distance),
        manifold_distance_ratio=_as_numpy(recovery.manifold_distance_ratio),
        decoded_angle=_as_numpy(recovery.decoded_angle),
        same_memory_error_radians=_as_numpy(
            recovery.same_memory_error_radians
        ),
        clean_manifold_distance=_as_numpy(recovery.clean_manifold_distance),
        clean_decoded_angle=_as_numpy(recovery.clean_decoded_angle),
        clean_same_memory_error_radians=_as_numpy(
            recovery.clean_same_memory_error_radians
        ),
        excess_same_memory_error_radians=_as_numpy(
            recovery.excess_same_memory_error_radians
        ),
        manifold_distance_minus_clean=_as_numpy(
            recovery.manifold_distance_minus_clean
        ),
        distance_to_matched_clean_state=_as_numpy(
            recovery.distance_to_matched_clean_state
        ),
        distance_to_matched_clean_state_ratio=_as_numpy(
            recovery.distance_to_matched_clean_state_ratio
        ),
        manifold_scale=np.asarray(
            float(recovery.manifold_scale.detach().cpu()), dtype=np.float64
        ),
        analysis_recurrent_state_noise_enabled=np.asarray(False, dtype=np.bool_),
        analysis_recurrent_state_noise_std=np.asarray(0.0, dtype=np.float64),
    )
    base_summary["carrier_ambient_normal_recovery"] = recovery_summary

    try:
        with torch.no_grad():
            projected = output_projected_flow(
                reconstruction.spline_state,
                adapter.actual_f0,
                lambda state: _blank_decode_primary(model, adapter, state),
            )
            angular_flow = signed_angular_flow(
                projected.output, projected.projected_vector_field
            )
        topology = cyclic_flow_reversal_topology(
            reconstruction.spline_angle,
            angular_flow,
            zero_tolerance=active_spec.flow_zero_tolerance,
            angle_tolerance=max(100.0 * torch.finfo(dtype).eps, 1.0e-12),
        )
    except StructuralNotEstimableError as error:
        return _publish_structural_not_estimable(
            destination=destination,
            base_summary=base_summary,
            progress=progress,
            completion=completion,
            identity=identity,
            model_name=model.config.name,
            task_trajectory_path=task_trajectory_path,
            failed_stage="projected_flow_and_fixed_point_topology",
            error=error,
            completed_artifacts=(reconstruction_path, recovery_path),
        )
    projected_path = destination / "projected_flow_and_topology.npz"
    _atomic_npz(
        projected_path,
        spline_angle=_as_numpy(reconstruction.spline_angle),
        output=_as_numpy(projected.output),
        next_output=_as_numpy(projected.next_output),
        projected_vector_field=_as_numpy(projected.projected_vector_field),
        pointwise_euclidean_norm=_as_numpy(projected.pointwise_norm),
        signed_angular_flow=_as_numpy(angular_flow),
        stable_fixed_point_angle=np.asarray(
            [item.angle for item in topology.stable], dtype=np.float64
        ),
        saddle_fixed_point_angle=np.asarray(
            [item.angle for item in topology.saddles], dtype=np.float64
        ),
    )
    topology_summary = {
        "kind": topology.kind,
        "orientation": topology.orientation,
        "zero_tolerance": topology.zero_tolerance,
        "angle_tolerance": topology.angle_tolerance,
        "stable_count": len(topology.stable),
        "saddle_count": len(topology.saddles),
        "stable_angles": [item.angle for item in topology.stable],
        "saddle_angles": [item.angle for item in topology.saddles],
        "classification_scope": (
            "stable_or_repelling_along_manifold_from_flow_reversal; full spectra "
            "reported separately without a binary normal-stability gate"
        ),
        "root_policy": "cyclic_adjacent_sign_reversal_with_linear_interpolation",
        "limit_cycle_is_analysis_failure": False,
    }
    flow_summary = {
        "definition": "D(F0(m(theta)))-D(m(theta))",
        "adaptation": "finite_step_discrete_nonlinear_decoder_project_resolution",
        "uniform_norm": float(projected.uniform_norm.detach().cpu()),
        "uniform_norm_definition": "max_theta Euclidean norm of two-dimensional projected vector",
        "pointwise_norm": _finite_summary(projected.pointwise_norm),
        "signed_angular_flow": _finite_summary(angular_flow),
    }

    try:
        spectrum_path, spectrum_summary = compute_resumable_full_spectrum(
            reconstruction.spline_state,
            adapter,
            output_dir=destination,
            identity=identity,
            chunk_size=active_spec.spectrum_chunk_size,
            progress_path=progress,
        )
    except StructuralNotEstimableError as error:
        return _publish_structural_not_estimable(
            destination=destination,
            base_summary=base_summary,
            progress=progress,
            completion=completion,
            identity=identity,
            model_name=model.config.name,
            task_trajectory_path=task_trajectory_path,
            failed_stage="full_local_jacobian_eigenspectrum",
            error=error,
            completed_artifacts=(
                reconstruction_path,
                recovery_path,
                projected_path,
            ),
        )
    atomic_json(
        progress,
        {
            "schema_version": SCHEMA_VERSION,
            "analysis_identity": identity,
            "stage": "finite_time_blank_memory",
            "updated_unix_seconds": time.time(),
        },
    )
    try:
        finite_arrays, finite_summary = finite_blank_memory(
            model,
            adapter,
            reconstruction.spline_state,
            reconstruction.spline_angle,
            horizon=active_spec.blank_horizon,
        )
    except StructuralNotEstimableError as error:
        return _publish_structural_not_estimable(
            destination=destination,
            base_summary=base_summary,
            progress=progress,
            completion=completion,
            identity=identity,
            model_name=model.config.name,
            task_trajectory_path=task_trajectory_path,
            failed_stage="finite_time_blank_memory",
            error=error,
            completed_artifacts=(
                reconstruction_path,
                recovery_path,
                projected_path,
                spectrum_path,
            ),
        )
    finite_path = destination / "finite_time_angular_memory.npz"
    _atomic_npz(finite_path, **finite_arrays)
    terminal_angle = torch.as_tensor(
        finite_arrays["predicted_angle"][:, -1], dtype=torch.float32
    )
    initial_angle = reconstruction.spline_angle.detach().cpu().to(dtype=torch.float32)
    try:
        asymptotic_summary, asymptotic_arrays = _asymptotic_summary(
            topology, initial_angle, terminal_angle
        )
    except StructuralNotEstimableError as error:
        return _publish_structural_not_estimable(
            destination=destination,
            base_summary=base_summary,
            progress=progress,
            completion=completion,
            identity=identity,
            model_name=model.config.name,
            task_trajectory_path=task_trajectory_path,
            failed_stage="asymptotic_memory_structure",
            error=error,
            completed_artifacts=(
                reconstruction_path,
                recovery_path,
                projected_path,
                spectrum_path,
                finite_path,
            ),
        )
    asymptotic_path = destination / "asymptotic_structure.npz"
    _atomic_npz(asymptotic_path, **asymptotic_arrays)

    base_summary.update(
        {
            "analysis_status": "complete_structural_summary_eligible",
            "projected_flow": flow_summary,
            "fixed_point_topology": topology_summary,
            "full_local_eigenspectrum": spectrum_summary,
            "finite_time_angular_memory": finite_summary,
            "asymptotic_structure": asymptotic_summary,
        }
    )
    summary_path = destination / "summary.json"
    atomic_json(summary_path, base_summary)
    atomic_json(
        progress,
        {
            "schema_version": SCHEMA_VERSION,
            "analysis_identity": identity,
            "stage": "complete",
            "updated_unix_seconds": time.time(),
        },
    )
    write_completion_receipt(
        completion,
        job_id=f"sagodi-primary-{model.config.name}-{identity[:12]}",
        artifacts=[
            destination / "analysis_identity.json",
            progress,
            task_trajectory_path,
            reconstruction_path,
            recovery_path,
            projected_path,
            spectrum_path,
            finite_path,
            asymptotic_path,
            summary_path,
        ],
        metadata={
            "analysis_identity": identity,
            "analysis_status": "complete_structural_summary_eligible",
            "model_id": model.config.name,
            "checkpoint_sha256": sha256_file(checkpoint),
        },
    )
    return summary_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spectrum-chunk-size", type=int, default=DEFAULT_SPECTRUM_CHUNK_SIZE)
    parser.add_argument("--source-v6", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trajectory-count", type=int)
    parser.add_argument("--spline-count", type=int)
    parser.add_argument("--task-horizon", type=int)
    parser.add_argument("--blank-horizon", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    overrides = {
        "trajectory_count": args.trajectory_count,
        "spline_count": args.spline_count,
        "task_horizon": args.task_horizon,
        "blank_horizon": args.blank_horizon,
    }
    if not args.smoke and any(value is not None for value in overrides.values()):
        raise ValueError("analysis-size overrides require --smoke")
    defaults = PrimaryAnalysisSpec(
        task_horizon=128 if args.source_v6 else FULL_TASK_HORIZON,
        blank_horizon=2048 if args.source_v6 else FULL_BLANK_HORIZON,
        source_v6=bool(args.source_v6),
    )
    spec = PrimaryAnalysisSpec(
        trajectory_count=(
            defaults.trajectory_count
            if args.trajectory_count is None
            else args.trajectory_count
        ),
        spline_count=(
            defaults.spline_count if args.spline_count is None else args.spline_count
        ),
        task_horizon=(
            defaults.task_horizon if args.task_horizon is None else args.task_horizon
        ),
        blank_horizon=(
            defaults.blank_horizon if args.blank_horizon is None else args.blank_horizon
        ),
        spectrum_chunk_size=args.spectrum_chunk_size,
        source_v6=bool(args.source_v6),
        smoke=bool(args.smoke),
    )
    summary = run_primary_analysis(
        checkpoint_path=args.checkpoint,
        protocol_path=args.protocol,
        output_dir=args.output,
        device=args.device,
        spec=spec,
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
