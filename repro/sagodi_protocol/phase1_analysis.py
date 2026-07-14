"""CA-LRU-specific supplementary ring diagnostics for one checkpoint.

This historical module is **not** the Ságodi-primary analysis used by the v3
paper comparison.  Its settling, recovery, projected-JVP, and numerical
C1--C4/L0--L3 gates are project-specific extensions.  The v3 primary path is
implemented in :mod:`sagodi_primary_runner` and intentionally never imports
this module.

The module constructs two independently labelled state banks: Ságodi Track A
(task-driven endpoints followed by a 16T blank rollout, slow-state selection,
and periodic resampling) and the task-conditioned canonical/eight-path bank.
The Track-A spline is the primary analysis manifold.  The eight-path bank is
kept independent and is used only for correspondence, projection, and fiber
diagnostics.
Failures are scientific outcomes and produce normal failure artifacts; only a
state-transition preflight failure, non-finite computation, or artifact error
raises.

The full (non-smoke) analysis uses the frozen primary settings:

* 1,024 uniformly spaced ring anchors;
* exact 1,024-step (4T) blank drift;
* radial and eight sampled ambient-normal kicks at ``rho = 0.1 * R_s`` with
  fixed-manifold distance recovery at ``H = 1,5,20,100,500,1024``;
* a 0.01-radian tangent-equivariance shift;
* sampled tangent/radial/ambient JVP gains at H=50.

The former clean-paired endpoint deviation is retained only as a diagnostic;
it cannot satisfy C3. ``--smoke`` reduces counts and horizons solely to validate the pipeline.  All
effective values are recorded and smoke output is ineligible for claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .artifacts import (
    atomic_json,
    derived_seed,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from .analysis_freeze import (
    analysis_freeze_fingerprint,
    load_analysis_freeze,
)
from .audit import AuditConfig, run_phase0_audit
from .config import (
    DEFAULT_PROTOCOL_PATH,
    AngularTaskSpec,
    load_protocol,
    protocol_fingerprint,
)
from .manifold_diagnostics import (
    manifold_recovery_diagnostics,
    settling_quality_diagnostics,
)
from .metrics import distribution_summary, task_metrics, wrap_angle
from .models import ProtocolModel, load_checkpoint
from .state import StateAdapter
from .tasks import Batch, load_fixed_bank, sample_angular_integration
from .train import _validate_evaluation_batch


ANALYSIS_SCHEMA_VERSION = 4
SETTLE_HORIZONS = (0, 5, 20, 100)
RECOVERY_HORIZONS = (1, 5, 20, 100, 500, 1024)
PATH_COUNT = 8
SLOW_RELATIVE_SPEED = 1.0e-3
TRACK_A_COVERAGE_Q95_CELLS = 2.0
TRACK_A_COVERAGE_MAX_CELLS = 4.0
TRACK_A_COVERAGE_MAX_GAP_CELLS = 8.0
DEFAULT_CLEAN_ADHERENCE_Q95_MAX = 0.01
DEFAULT_RECOVERY_MEDIAN_MAX = 0.5
DEFAULT_RECOVERY_Q95_STRICT_MAX = 1.0
DEFAULT_SETTLING_Q95_MAX = 0.01
DEFAULT_TRACK_A_COVERAGE_BINS = 32
DEFAULT_TRACK_A_COVERAGE_MIN_FRACTION = 1.0
DEFAULT_TRACK_A_NORMALIZED_TANGENT_SPEED_MIN = 1.0e-3
DEFAULT_TRACK_A_SEAM_PROBE_RADIANS = 1.0e-3
DEFAULT_TRACK_A_SEAM_C0_OVER_RS_MAX = 1.0e-5
DEFAULT_TRACK_A_SEAM_C1_RELATIVE_MAX = 1.0e-2
LEGACY_V1_ANALYSIS_FREEZE_IDS = {
    "sagodi_phase01_ring_pilot_v1",
    "calru_native_sagodi_ring_pilot_v1",
}


@dataclass(frozen=True)
class AnalysisPlan:
    atlas_count: int
    task_trials: int
    task_horizon: int
    kick_anchor_count: int
    jacobian_anchor_count: int
    ambient_directions: int
    drift_horizon: int
    recovery_horizon: int
    recovery_horizons: tuple[int, ...]
    jacobian_horizon: int
    tangent_shift: float
    tangent_fd_epsilon: float
    kick_relative_radius: float
    slow_rollout_horizon: int
    path_steps: int
    path_count: int
    settle_horizons: tuple[int, ...]
    clean_adherence_q95_max: float
    recovery_median_max: float
    recovery_q95_strict_max: float
    settle_absolute_change_q95_max: float
    settle_positive_expansion_q95_max: float
    settle_mean_sheet_distance_q95_max: float
    track_a_coverage_bin_count: int
    track_a_coverage_min_fraction: float
    track_a_normalized_tangent_speed_min: float
    track_a_seam_probe_radians: float
    track_a_seam_c0_over_rs_max: float
    track_a_seam_c1_relative_max: float
    projection_density: int
    smoke: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "atlas_count": self.atlas_count,
            "task_trials": self.task_trials,
            "task_horizon": self.task_horizon,
            "kick_anchor_count": self.kick_anchor_count,
            "jacobian_anchor_count": self.jacobian_anchor_count,
            "ambient_directions": self.ambient_directions,
            "drift_horizon": self.drift_horizon,
            "recovery_horizon": self.recovery_horizon,
            "recovery_horizons": list(self.recovery_horizons),
            "jacobian_horizon": self.jacobian_horizon,
            "tangent_shift_radians": self.tangent_shift,
            "tangent_fd_epsilon_radians": self.tangent_fd_epsilon,
            "kick_relative_radius": self.kick_relative_radius,
            "slow_rollout_horizon": self.slow_rollout_horizon,
            "slow_rollout_multiple_of_task_T": self.slow_rollout_horizon / self.task_horizon,
            "slow_relative_speed_threshold": SLOW_RELATIVE_SPEED,
            "path_steps": self.path_steps,
            "path_count": self.path_count,
            "settle_horizons": list(self.settle_horizons),
            "settling_quality_controls": {
                "absolute_symmetric_relative_variance_change_q95_max": self.settle_absolute_change_q95_max,
                "positive_relative_variance_expansion_q95_max": self.settle_positive_expansion_q95_max,
                "F0_power_5_mean_sheet_manifold_distance_over_Rs_q95_max": self.settle_mean_sheet_distance_q95_max,
                "status": "pilot_QA_not_claim_threshold",
            },
            "track_a_geometry_quality_controls": {
                "coarse_coverage_bin_count": self.track_a_coverage_bin_count,
                "minimum_coarse_bin_occupancy_fraction": self.track_a_coverage_min_fraction,
                "minimum_normalized_tangent_speed": self.track_a_normalized_tangent_speed_min,
                "seam_probe_radians": self.track_a_seam_probe_radians,
                "seam_C0_distance_over_Rs_max": self.track_a_seam_c0_over_rs_max,
                "seam_C1_relative_difference_max": self.track_a_seam_c1_relative_max,
            },
            "c3_manifold_distance_controls": {
                "clean_adherence_q95_max_at_every_registered_horizon": self.clean_adherence_q95_max,
                "primary_recovery_median_max": self.recovery_median_max,
                "primary_recovery_q95_strict_max": self.recovery_q95_strict_max,
                "legacy_clean_paired_metric_role": "diagnostic_only",
            },
            "projection_dense_spline_factor": self.projection_density,
            "smoke": self.smoke,
        }


def _analysis_thresholds(
    analysis_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if analysis_contract is None:
        return {
            "recovery_horizons": RECOVERY_HORIZONS,
            "primary_horizon": 500,
            "clean_adherence_q95_max": DEFAULT_CLEAN_ADHERENCE_Q95_MAX,
            "recovery_median_max": DEFAULT_RECOVERY_MEDIAN_MAX,
            "recovery_q95_strict_max": DEFAULT_RECOVERY_Q95_STRICT_MAX,
            "settle_absolute_change_q95_max": DEFAULT_SETTLING_Q95_MAX,
            "settle_positive_expansion_q95_max": DEFAULT_SETTLING_Q95_MAX,
            "settle_mean_sheet_distance_q95_max": DEFAULT_SETTLING_Q95_MAX,
            "track_a_coverage_bin_count": DEFAULT_TRACK_A_COVERAGE_BINS,
            "track_a_coverage_min_fraction": DEFAULT_TRACK_A_COVERAGE_MIN_FRACTION,
            "track_a_normalized_tangent_speed_min": DEFAULT_TRACK_A_NORMALIZED_TANGENT_SPEED_MIN,
            "track_a_seam_probe_radians": DEFAULT_TRACK_A_SEAM_PROBE_RADIANS,
            "track_a_seam_c0_over_rs_max": DEFAULT_TRACK_A_SEAM_C0_OVER_RS_MAX,
            "track_a_seam_c1_relative_max": DEFAULT_TRACK_A_SEAM_C1_RELATIVE_MAX,
        }
    c3 = analysis_contract["c3_normal_recovery"]
    settling = analysis_contract["settling"]
    geometry = analysis_contract["manifold"]["primary"]["quality_gates"]
    primary = c3["thresholds"]["primary_horizon_recovery"]
    return {
        "recovery_horizons": tuple(int(value) for value in c3["registered_horizons"]),
        "primary_horizon": int(c3["primary_horizon"]),
        "clean_adherence_q95_max": float(
            c3["thresholds"]["clean_adherence_all_horizons"]["value"]
        ),
        "recovery_median_max": float(primary["median"]["value"]),
        "recovery_q95_strict_max": float(primary["q95"]["value"]),
        "settle_absolute_change_q95_max": float(
            settling["absolute_symmetric_relative_change"]["threshold"]["value"]
        ),
        "settle_positive_expansion_q95_max": float(
            settling["positive_expansion"]["threshold"]["value"]
        ),
        "settle_mean_sheet_distance_q95_max": float(
            settling["mean_sheet_rollout"]["threshold"]["value"]
        ),
        "track_a_coverage_bin_count": int(geometry["coarse_coverage_bin_count"]),
        "track_a_coverage_min_fraction": float(
            geometry["minimum_coarse_bin_occupancy_fraction"]
        ),
        "track_a_normalized_tangent_speed_min": float(
            geometry["minimum_normalized_tangent_speed"]
        ),
        "track_a_seam_probe_radians": float(geometry["seam_probe_radians"]),
        "track_a_seam_c0_over_rs_max": float(
            geometry["seam_C0_distance_over_Rs_max"]
        ),
        "track_a_seam_c1_relative_max": float(
            geometry["seam_C1_relative_difference_max"]
        ),
    }


def _build_plan(
    protocol: Mapping[str, Any],
    smoke: bool,
    analysis_contract: Mapping[str, Any] | None = None,
) -> AnalysisPlan:
    evaluation = protocol["evaluation"]
    task = protocol["phase1_ring_pilot"]["task"]
    qa = protocol["qa_thresholds"]
    thresholds = _analysis_thresholds(analysis_contract)
    if smoke:
        return AnalysisPlan(
            atlas_count=32,
            task_trials=8,
            task_horizon=8,
            kick_anchor_count=8,
            jacobian_anchor_count=8,
            ambient_directions=2,
            drift_horizon=8,
            recovery_horizon=5,
            recovery_horizons=(1, 5),
            jacobian_horizon=3,
            tangent_shift=float(evaluation["tangent_shift_radians"]["primary"]),
            tangent_fd_epsilon=float(qa["tangent_finite_difference_radians"][1]),
            kick_relative_radius=float(evaluation["primary_kick_relative_radius"]),
            slow_rollout_horizon=128,
            path_steps=8,
            path_count=PATH_COUNT,
            settle_horizons=SETTLE_HORIZONS,
            clean_adherence_q95_max=float(thresholds["clean_adherence_q95_max"]),
            recovery_median_max=float(thresholds["recovery_median_max"]),
            recovery_q95_strict_max=float(thresholds["recovery_q95_strict_max"]),
            settle_absolute_change_q95_max=float(thresholds["settle_absolute_change_q95_max"]),
            settle_positive_expansion_q95_max=float(thresholds["settle_positive_expansion_q95_max"]),
            settle_mean_sheet_distance_q95_max=float(thresholds["settle_mean_sheet_distance_q95_max"]),
            track_a_coverage_bin_count=min(
                int(thresholds["track_a_coverage_bin_count"]), 32
            ),
            track_a_coverage_min_fraction=float(
                thresholds["track_a_coverage_min_fraction"]
            ),
            track_a_normalized_tangent_speed_min=float(
                thresholds["track_a_normalized_tangent_speed_min"]
            ),
            track_a_seam_probe_radians=float(
                thresholds["track_a_seam_probe_radians"]
            ),
            track_a_seam_c0_over_rs_max=float(
                thresholds["track_a_seam_c0_over_rs_max"]
            ),
            track_a_seam_c1_relative_max=float(
                thresholds["track_a_seam_c1_relative_max"]
            ),
            projection_density=int(qa["projection"]["ring_dense_spline_factor"]),
            smoke=True,
        )
    return AnalysisPlan(
        atlas_count=int(evaluation["manifold_anchor_count"]),
        task_trials=int(evaluation["id_test_trials"]),
        task_horizon=int(task["sequence_steps"]),
        kick_anchor_count=int(evaluation["finite_kick_anchor_count"]),
        jacobian_anchor_count=int(evaluation["jacobian_anchor_count"]),
        ambient_directions=int(evaluation["ambient_random_directions_per_anchor"]),
        drift_horizon=int(evaluation["exact_four_T_horizon"]),
        recovery_horizon=int(thresholds["primary_horizon"]),
        recovery_horizons=tuple(int(value) for value in thresholds["recovery_horizons"]),
        jacobian_horizon=50,
        tangent_shift=float(evaluation["tangent_shift_radians"]["primary"]),
        tangent_fd_epsilon=float(qa["tangent_finite_difference_radians"][1]),
        kick_relative_radius=float(evaluation["primary_kick_relative_radius"]),
        slow_rollout_horizon=16 * int(task["sequence_steps"]),
        path_steps=int(task["sequence_steps"]),
        path_count=PATH_COUNT,
        settle_horizons=SETTLE_HORIZONS,
        clean_adherence_q95_max=float(thresholds["clean_adherence_q95_max"]),
        recovery_median_max=float(thresholds["recovery_median_max"]),
        recovery_q95_strict_max=float(thresholds["recovery_q95_strict_max"]),
        settle_absolute_change_q95_max=float(thresholds["settle_absolute_change_q95_max"]),
        settle_positive_expansion_q95_max=float(thresholds["settle_positive_expansion_q95_max"]),
        settle_mean_sheet_distance_q95_max=float(thresholds["settle_mean_sheet_distance_q95_max"]),
        track_a_coverage_bin_count=int(thresholds["track_a_coverage_bin_count"]),
        track_a_coverage_min_fraction=float(
            thresholds["track_a_coverage_min_fraction"]
        ),
        track_a_normalized_tangent_speed_min=float(
            thresholds["track_a_normalized_tangent_speed_min"]
        ),
        track_a_seam_probe_radians=float(
            thresholds["track_a_seam_probe_radians"]
        ),
        track_a_seam_c0_over_rs_max=float(
            thresholds["track_a_seam_c0_over_rs_max"]
        ),
        track_a_seam_c1_relative_max=float(
            thresholds["track_a_seam_c1_relative_max"]
        ),
        projection_density=int(qa["projection"]["ring_dense_spline_factor"]),
        smoke=False,
    )


def _model_dtype(model: torch.nn.Module) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def _angle_embedding(angles: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)


@torch.no_grad()
def _atlas_states(
    model: ProtocolModel,
    adapter: StateAdapter,
    angles: torch.Tensor,
) -> torch.Tensor:
    memory = _angle_embedding(angles)
    reported = model.initial_state(
        angles.numel(), angles.device, initial_memory=memory
    )
    return adapter.primary_from_reported(reported)


@torch.no_grad()
def _canonical_reported_states(
    model: ProtocolModel,
    angles: torch.Tensor,
) -> torch.Tensor:
    return model.initial_state(
        angles.numel(), angles.device, initial_memory=_angle_embedding(angles)
    )


def _require_finite(label: str, *values: torch.Tensor | np.ndarray) -> None:
    for value in values:
        finite = (
            bool(torch.isfinite(value).all())
            if isinstance(value, torch.Tensor)
            else bool(np.isfinite(value).all())
        )
        if not finite:
            raise RuntimeError(f"non-finite values in {label}")


def _circular_distance_numpy(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(left - right), np.cos(left - right)))


def _periodic_cubic_resample(
    knot_angles: torch.Tensor,
    knot_states: torch.Tensor,
    query_angles: torch.Tensor,
) -> torch.Tensor:
    """Periodic cubic-spline interpolation with an explicit cyclic solve.

    Knots may be irregular, but must be unique modulo 2pi.  The cyclic linear
    system is intentionally explicit so this protocol has no SciPy runtime
    dependency and the interpolation method remains inspectable.
    """

    if knot_angles.ndim != 1 or knot_states.ndim != 2:
        raise ValueError("periodic spline expects angles [N] and states [N,D]")
    if knot_angles.numel() != knot_states.shape[0] or knot_angles.numel() < 4:
        raise ValueError("periodic cubic spline requires at least four paired knots")
    period = 2.0 * math.pi
    angles = torch.remainder(knot_angles, period)
    order = torch.argsort(angles)
    x = angles[order]
    y = knot_states[order]
    h = torch.cat((x[1:] - x[:-1], (x[:1] + period) - x[-1:]))
    spacing_floor = 100.0 * torch.finfo(x.dtype).eps
    if bool((h <= spacing_floor).any()):
        raise ValueError("periodic spline knots are duplicated or numerically coincident")

    n = int(x.numel())
    matrix = torch.zeros(n, n, device=x.device, dtype=x.dtype)
    index = torch.arange(n, device=x.device)
    previous = torch.remainder(index - 1, n)
    following = torch.remainder(index + 1, n)
    h_previous = h[previous]
    h_following = h[index]
    matrix[index, previous] = h_previous
    matrix[index, index] = 2.0 * (h_previous + h_following)
    matrix[index, following] = h_following
    y_previous = y[previous]
    y_following = y[following]
    right = 6.0 * (
        (y_following - y) / h_following[:, None]
        - (y - y_previous) / h_previous[:, None]
    )
    second = torch.linalg.solve(matrix, right)

    query = torch.remainder(query_angles, period)
    interval = torch.searchsorted(x, query, right=True) - 1
    interval = torch.remainder(interval, n)
    next_interval = torch.remainder(interval + 1, n)
    x0 = x[interval]
    width = h[interval]
    displacement = torch.remainder(query - x0, period)
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
    _require_finite("periodic cubic spline", result)
    return result


@torch.no_grad()
def _build_ring_projector(
    atlas_angles: torch.Tensor,
    atlas_state: torch.Tensor,
    *,
    density: int = 8,
) -> dict[str, torch.Tensor | int | str]:
    count = int(atlas_angles.numel()) * int(density)
    dense_angles = -math.pi + 2.0 * math.pi * torch.arange(
        count, device=atlas_angles.device, dtype=atlas_angles.dtype
    ) / float(count)
    dense_state = _periodic_cubic_resample(atlas_angles, atlas_state, dense_angles)
    step = 2.0 * math.pi / float(count)
    derivative = (
        torch.roll(dense_state, shifts=-1, dims=0)
        - torch.roll(dense_state, shifts=1, dims=0)
    ) / (2.0 * step)
    return {
        "atlas_angles": atlas_angles,
        "atlas_state": atlas_state,
        "dense_angles": dense_angles,
        "dense_state": dense_state,
        "dense_derivative": derivative,
        "density": int(density),
        "method": f"{int(density)}x_periodic_cubic_spline_exact_dense_NN_plus_local_tangent_projection",
    }


@torch.no_grad()
def _project_ring(
    query: torch.Tensor,
    projector: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    dense_state = projector["dense_state"]
    dense_angles = projector["dense_angles"]
    derivative = projector["dense_derivative"]
    distance_to_dense, dense_index = _nearest_atlas(query, dense_state)
    base = dense_state[dense_index]
    tangent_derivative = derivative[dense_index]
    denominator = tangent_derivative.square().sum(dim=-1)
    step = 2.0 * math.pi / float(dense_angles.numel())
    local_offset = (
        ((query - base) * tangent_derivative).sum(dim=-1)
        / denominator.clamp_min(100.0 * torch.finfo(query.dtype).eps)
    ).clamp(min=-0.5 * step, max=0.5 * step)
    projected_state = base + local_offset[:, None] * tangent_derivative
    projected_angle = wrap_angle(dense_angles[dense_index] + local_offset)
    distance = torch.linalg.vector_norm(query - projected_state, dim=-1)
    tangent_norm = torch.linalg.vector_norm(tangent_derivative, dim=-1)
    frame_floor = 100.0 * torch.finfo(query.dtype).eps
    tangent_frame = tangent_derivative / tangent_norm.clamp_min(frame_floor)[:, None]
    tangent_frame_valid = tangent_norm > frame_floor
    anchor_count = int(projector["atlas_angles"].numel())
    original_index = torch.remainder(
        torch.round((projected_angle + math.pi) * anchor_count / (2.0 * math.pi)).long(),
        anchor_count,
    )
    _require_finite(
        "ring dense-spline projection", distance_to_dense, projected_state,
        projected_angle, distance,
    )
    return {
        "distance": distance,
        "angle": projected_angle,
        "state": projected_state,
        "dense_index": dense_index,
        "original_index": original_index,
        "local_offset": local_offset,
        # This is the same local frame used by the projector refinement.  A
        # numerically collapsed derivative deliberately yields the zero
        # vector and a false validity flag instead of inventing a direction.
        "tangent_frame": tangent_frame,
        "tangent_frame_norm": tangent_norm,
        "tangent_frame_valid": tangent_frame_valid,
    }


@torch.no_grad()
def _projection_quality_audit(
    projector: Mapping[str, Any],
    atlas_state: torch.Tensor,
    atlas_tangent: torch.Tensor,
    center: torch.Tensor,
    rs: torch.Tensor,
    *,
    heldout_angles: torch.Tensor,
    heldout_task_mean_state: torch.Tensor | None,
    selected_settling_horizon: int | None,
    q95_max: float,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    count = int(atlas_state.shape[0])
    density = int(projector["density"])

    # The known-q family must be independent of the spline being audited.
    # ``heldout_task_mean_state`` is produced by eight fresh task paths at
    # mid-cell angles and blank-settled for the *primary atlas's* selected
    # horizon.  In particular, no state in this family is sampled from
    # ``projector["dense_state"]``.
    known_available = (
        heldout_task_mean_state is not None
        and selected_settling_horizon is not None
        and int(heldout_task_mean_state.shape[0]) == int(heldout_angles.numel())
    )
    if known_available:
        assert heldout_task_mean_state is not None
        _require_finite(
            "held-out task-conditioned projection QA",
            heldout_angles,
            heldout_task_mean_state,
        )
        known_projection = _project_ring(heldout_task_mean_state, projector)
        known_error = (
            wrap_angle(known_projection["angle"] - heldout_angles).abs() / math.pi
        )
        known_summary: dict[str, Any] = {
            **_summary(known_error),
            "status": "evaluated",
        }
        known_projected_angles = known_projection["angle"]
    else:
        known_error = torch.empty(
            0, device=atlas_state.device, dtype=atlas_state.dtype
        )
        known_projected_angles = known_error
        known_summary = {
            "status": "not_evaluated",
            "reason": "diagnostic task sheet has no corrected settling horizon",
        }

    sample_index = _stratified_indices(count, min(256, count), atlas_state.device)
    base = atlas_state[sample_index]
    tangent = atlas_tangent[sample_index]
    radial = base - center
    radial = radial - (radial * tangent).sum(dim=-1, keepdim=True) * tangent
    norm = torch.linalg.vector_norm(radial, dim=-1, keepdim=True)
    bad = norm.squeeze(-1) <= 100.0 * torch.finfo(base.dtype).eps
    if bool(bad.any()):
        basis = torch.zeros_like(radial)
        basis[:, 0] = 1.0
        basis = basis - (basis * tangent).sum(dim=-1, keepdim=True) * tangent
        radial = torch.where(bad[:, None], basis, radial)
        norm = torch.linalg.vector_norm(radial, dim=-1, keepdim=True)
    radial = radial / norm.clamp_min(100.0 * torch.finfo(base.dtype).eps)
    synthetic = base + 0.05 * rs * radial
    synthetic_projection = _project_ring(synthetic, projector)
    synthetic_error = (
        wrap_angle(
            synthetic_projection["angle"] - projector["atlas_angles"][sample_index]
        ).abs()
        / math.pi
    )
    combined = torch.cat((known_error, synthetic_error))
    synthetic_summary = _summary(synthetic_error)
    summary = {
        "passed": bool(
            known_available
            and float(known_summary["q95"]) <= float(q95_max)
            and float(synthetic_summary["q95"]) <= float(q95_max)
        ),
        "q95_max": float(q95_max),
        "method": projector["method"],
        "dense_factor": density,
        "families_must_pass_separately": True,
        "known_q_midcell": known_summary,
        "synthetic_off_manifold_0p05_Rs_radial": synthetic_summary,
        "combined": _summary(combined),
        "known_q_definition": (
            "mid-cell angles absent from the primary atlas; states are the mean "
            "of eight independent task-conditioned paths blank-settled at the "
            "diagnostic task sheet's selected settling horizon"
        ),
        "known_q_state_source": "independent_task_conditioned_eight_path_mean_not_spline_generated",
        "known_q_angle_rule": "stratified primary-atlas cells plus exactly one-half cell",
        "known_q_count": int(heldout_angles.numel()),
        "known_q_selected_settling_horizon": selected_settling_horizon,
        "synthetic_definition": "original anchors plus 0.05_Rs local radial offsets",
    }
    arrays = {
        "known_q_angles": heldout_angles,
        "known_q_task_mean_primary_carrier": (
            heldout_task_mean_state
            if heldout_task_mean_state is not None
            else torch.empty(
                0, atlas_state.shape[1],
                device=atlas_state.device, dtype=atlas_state.dtype,
            )
        ),
        "known_q_projection_error": known_error,
        "known_q_projected_angles": known_projected_angles,
        "synthetic_anchor_indices": sample_index,
        "synthetic_projection_error": synthetic_error,
        "synthetic_projected_angles": synthetic_projection["angle"],
    }
    return summary, arrays


def _ring_path_velocity_bank(
    target_angles: torch.Tensor,
    *,
    steps: int,
    dt: float = 0.1,
) -> torch.Tensor:
    """Eight deterministic velocity paths from q0=0 to every target angle.

    The profiles vary pulse position, order, and shape.  Seven paths also add
    a zero-net excursion, so paths remain distinct even for the zero-angle
    target.  The last token is corrected in floating point so every path has
    exactly the requested integrated displacement under the task convention.
    """

    if int(steps) < 8:
        raise ValueError("eight-path bank requires at least eight velocity steps")
    count = int(steps)
    device, dtype = target_angles.device, target_angles.dtype
    t = (torch.arange(count, device=device, dtype=dtype) + 0.5) / float(count)
    profiles = torch.zeros(PATH_COUNT, count, device=device, dtype=dtype)
    profiles[0].fill_(1.0)
    profiles[1, : max(1, count // 4)] = 1.0
    profiles[2, -max(1, count // 4) :] = 1.0
    middle = max(1, count // 4)
    start = (count - middle) // 2
    profiles[3, start : start + middle] = 1.0
    profiles[4] = t
    profiles[5] = 1.0 - t
    profiles[6] = 0.2 + torch.sin(math.pi * t).square()
    profiles[7] = 0.2 + torch.cos(2.0 * math.pi * t).square()
    profiles = profiles / profiles.sum(dim=1, keepdim=True)

    loops = torch.zeros_like(profiles)
    for path in range(1, PATH_COUNT):
        harmonic = (path + 1) // 2
        wave = (
            torch.sin(2.0 * math.pi * harmonic * t)
            if path % 2
            else torch.cos(2.0 * math.pi * harmonic * t)
        )
        wave = wave - wave.mean()
        loops[path] = 0.35 * wave / wave.abs().sum().clamp_min(torch.finfo(dtype).eps)

    displacement = wrap_angle(target_angles)
    angular_increment = displacement[:, None, None] * profiles[None, :, :]
    angular_increment = angular_increment + loops[None, :, :]
    angular_increment[:, :, -1] += displacement[:, None] - angular_increment.sum(dim=-1)
    velocity = angular_increment / float(dt)
    _require_finite("ring path velocity bank", velocity)
    return velocity


@torch.no_grad()
def _run_velocity_path_endpoints(
    model: ProtocolModel,
    adapter: StateAdapter,
    velocity: torch.Tensor,
) -> torch.Tensor:
    """Run [anchor,path,time] velocity paths and return [anchor,path,state]."""

    anchors, paths, steps = velocity.shape
    flat = anchors * paths
    q0_memory = torch.tensor(
        [1.0, 0.0], device=velocity.device, dtype=velocity.dtype
    ).expand(flat, -1)
    reported = model.initial_state(flat, velocity.device, initial_memory=q0_memory)
    for step in range(steps):
        token = velocity[:, :, step].reshape(flat, 1)
        reported = model.step(token, reported)
    endpoint = adapter.primary_from_reported(reported)
    return endpoint.reshape(anchors, paths, adapter.primary_dim)


@torch.no_grad()
def _heldout_task_conditioned_states(
    model: ProtocolModel,
    adapter: StateAdapter,
    angles: torch.Tensor,
    *,
    path_steps: int,
    settling_horizon: int,
) -> dict[str, torch.Tensor | int | str]:
    """Create an independent known-q bank for projection QA.

    The path construction is shared with the diagnostic task sheet, but the
    target angles are disjoint mid-cell points and all model trajectories are
    freshly executed.  The caller supplies the primary atlas's already chosen
    settling horizon; this function never selects a more favorable horizon.
    """

    velocity = _ring_path_velocity_bank(angles, steps=int(path_steps))
    paths = _run_velocity_path_endpoints(model, adapter, velocity)
    current = paths.reshape(-1, adapter.primary_dim)
    for _ in range(int(settling_horizon)):
        current = adapter.actual_f0(current)
    settled_paths = current.reshape_as(paths)
    mean_state = settled_paths.mean(dim=1)
    _require_finite(
        "held-out task-conditioned known-q bank", velocity, settled_paths, mean_state
    )
    return {
        "angles": angles,
        "velocity": velocity,
        "settled_path_state": settled_paths,
        "mean_state": mean_state,
        "settling_horizon": int(settling_horizon),
        "source": "fresh_eight_path_task_rollouts_at_disjoint_mid_cell_angles",
    }


def _stratified_circular_source_indices(
    source_angles: torch.Tensor,
    target_angles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign one distinct source trial to each uniformly spaced target angle.

    The assignment depends only on ground-truth task latents, never on a model
    output or hidden state.  Stable sorting plus the fixed target order makes
    ties deterministic.  The full protocol has 4,096 sources for 1,024
    targets; smoke generation also guarantees at least one source per target.
    """

    if source_angles.ndim != 1 or target_angles.ndim != 1:
        raise ValueError("Track-A source and target angles must be rank-1")
    sources = int(source_angles.numel())
    targets = int(target_angles.numel())
    if sources < targets:
        raise ValueError(
            f"Track-A discovery needs at least {targets} source trials, got {sources}"
        )
    source = source_angles.detach().cpu().double().numpy()
    target = target_angles.detach().cpu().double().numpy()
    distance = _circular_distance_numpy(target[:, None], source[None, :])
    ranked = np.argsort(distance, axis=1, kind="mergesort")
    used = np.zeros(sources, dtype=bool)
    selected = np.empty(targets, dtype=np.int64)
    error = np.empty(targets, dtype=np.float64)
    for target_index in range(targets):
        available = ranked[target_index][~used[ranked[target_index]]]
        if available.size == 0:  # pragma: no cover - guarded by sources >= targets
            raise RuntimeError("Track-A stratified source assignment exhausted candidates")
        source_index = int(available[0])
        selected[target_index] = source_index
        error[target_index] = distance[target_index, source_index]
        used[source_index] = True
    return (
        torch.as_tensor(selected, device=target_angles.device, dtype=torch.long),
        torch.as_tensor(error, device=target_angles.device, dtype=target_angles.dtype),
    )


@torch.no_grad()
def _task_driven_track_a_endpoints(
    model: ProtocolModel,
    evaluation: Mapping[str, torch.Tensor],
    target_angles: torch.Tensor,
) -> dict[str, Any]:
    """Create model-independent stratified task endpoints for Track A.

    Source trials are chosen by their ground-truth final latent.  Only after
    that frozen selection are the task inputs run through the model, without
    state noise, to obtain the reported recurrent endpoint states.
    """

    inputs = evaluation.get("discovery_inputs", evaluation["inputs"])
    latents = evaluation.get("discovery_latents", evaluation["latents"])
    initial_memory = evaluation.get(
        "discovery_initial_memory", evaluation["initial_memory"]
    )
    final_angles = latents[-1, :, 0]
    source_indices, selection_error = _stratified_circular_source_indices(
        final_angles, target_angles
    )
    selected_inputs = inputs[:, source_indices]
    selected_memory = initial_memory[source_indices]
    reported = model.initial_state(
        int(target_angles.numel()), target_angles.device, initial_memory=selected_memory
    )
    for token in selected_inputs:
        reported = model.step(token, reported)
    selected_final_angles = final_angles[source_indices]
    _require_finite(
        "Track-A task-driven endpoints",
        reported,
        selected_final_angles,
        selection_error,
    )
    return {
        "reported_state": reported,
        "source_indices": source_indices,
        "source_final_angles": selected_final_angles,
        "source_selection_error": selection_error,
        "source_trial_count": int(final_angles.numel()),
        "selected_trial_count": int(source_indices.numel()),
        "selection_rule": (
            "distinct_fixed_evaluation_trials_greedily_nearest_to_uniform_targets_"
            "by_ground_truth_final_latent_stable_ties"
        ),
        "task_rollout_horizon": int(selected_inputs.shape[0]),
        "state_noise_standard_deviation": 0.0,
    }


def _track_a_coverage_quality(
    selected_decoded: np.ndarray,
    selected_pair: np.ndarray,
    target_angles: torch.Tensor,
    *,
    coarse_bin_count: int = DEFAULT_TRACK_A_COVERAGE_BINS,
    minimum_coarse_occupancy_fraction: float = DEFAULT_TRACK_A_COVERAGE_MIN_FRACTION,
) -> tuple[dict[str, Any], np.ndarray]:
    """Audit that selected spline knots cover the full decoded output ring."""

    targets = target_angles.detach().cpu().double().numpy()
    selected = selected_decoded.astype(np.float64, copy=False)
    error = _circular_distance_numpy(selected, targets)
    _, unique_index = np.unique(selected_pair, axis=0, return_index=True)
    unique_index.sort()
    knot_angles = np.remainder(selected[unique_index], 2.0 * math.pi)
    knot_angles = np.unique(np.round(knot_angles, decimals=10))
    sorted_knots = np.sort(knot_angles)
    if sorted_knots.size:
        gaps = np.diff(np.concatenate((sorted_knots, sorted_knots[:1] + 2.0 * math.pi)))
        maximum_gap = float(gaps.max())
    else:
        maximum_gap = float(2.0 * math.pi)
    cell_width = 2.0 * math.pi / float(target_angles.numel())
    bins = min(int(coarse_bin_count), int(target_angles.numel()))
    if bins <= 0:
        raise ValueError("Track-A coverage bin count must be positive")
    if not 0.0 <= float(minimum_coarse_occupancy_fraction) <= 1.0:
        raise ValueError("Track-A minimum coarse-bin occupancy must be in [0,1]")
    nearest_target_index = np.remainder(
        np.rint(
            np.remainder(selected + math.pi, 2.0 * math.pi)
            * float(target_angles.numel())
            / (2.0 * math.pi)
        ).astype(np.int64),
        int(target_angles.numel()),
    )
    coarse_bin_index = np.minimum(
        bins - 1,
        nearest_target_index * bins // int(target_angles.numel()),
    )
    occupied_coarse_bins = int(np.unique(coarse_bin_index).size)
    coarse_occupancy_fraction = occupied_coarse_bins / float(bins)
    q95 = float(np.quantile(error, 0.95))
    maximum = float(error.max())
    thresholds = {
        "selection_error_q95_max_radians": TRACK_A_COVERAGE_Q95_CELLS * cell_width,
        "selection_error_max_radians": TRACK_A_COVERAGE_MAX_CELLS * cell_width,
        "maximum_circular_knot_gap_radians": (
            TRACK_A_COVERAGE_MAX_GAP_CELLS * cell_width
        ),
        "minimum_unique_knots": 4,
        "coarse_coverage_bin_count": bins,
        "minimum_coarse_bin_occupancy_fraction": float(
            minimum_coarse_occupancy_fraction
        ),
    }
    passed = bool(
        sorted_knots.size >= int(thresholds["minimum_unique_knots"])
        and q95 <= float(thresholds["selection_error_q95_max_radians"])
        and maximum <= float(thresholds["selection_error_max_radians"])
        and maximum_gap <= float(thresholds["maximum_circular_knot_gap_radians"])
        and coarse_occupancy_fraction
        >= float(thresholds["minimum_coarse_bin_occupancy_fraction"])
    )
    return {
        "passed": passed,
        "target_count": int(target_angles.numel()),
        "unique_selected_candidate_pairs": int(unique_index.size),
        "unique_decoded_knots": int(sorted_knots.size),
        "cell_width_radians": cell_width,
        "selection_error_radians": {
            "mean": float(error.mean()),
            "median": float(np.median(error)),
            "q95": q95,
            "max": maximum,
        },
        "maximum_circular_knot_gap_radians": maximum_gap,
        "occupied_coarse_bins": occupied_coarse_bins,
        "coarse_bin_occupancy_fraction": coarse_occupancy_fraction,
        "thresholds": thresholds,
        "scope": "full_ring_decoded_angle_coverage_QA_not_CA_claim_threshold",
    }, error


@torch.no_grad()
def _slow_state_reconstruction(
    model: ProtocolModel,
    adapter: StateAdapter,
    target_angles: torch.Tensor,
    *,
    horizon: int,
    initial_reported_states: torch.Tensor | None = None,
    start_state_rule: str | None = None,
    coarse_coverage_bin_count: int = DEFAULT_TRACK_A_COVERAGE_BINS,
    minimum_coarse_occupancy_fraction: float = DEFAULT_TRACK_A_COVERAGE_MIN_FRACTION,
) -> dict[str, Any]:
    """Ságodi Track A with exact global circular candidate selection.

    ``initial_reported_states`` makes task-driven endpoints the primary path.
    Omitting it preserves the historical canonical-initialization API for
    focused unit tests and sensitivity analyses.
    """

    anchors = int(target_angles.numel())
    if initial_reported_states is None:
        initial_reported = _canonical_reported_states(model, target_angles)
        effective_start_rule = (
            start_state_rule or "uniform_task_defined_hidden_initialization"
        )
    else:
        initial_reported = initial_reported_states
        effective_start_rule = (
            start_state_rule
            or "stratified_fixed_evaluation_task_driven_endpoints"
        )
    if initial_reported.ndim != 2 or initial_reported.shape != (
        anchors,
        adapter.reported_dim,
    ):
        raise ValueError(
            "Track-A initial reported states have shape "
            f"{tuple(initial_reported.shape)}, expected "
            f"({anchors}, {adapter.reported_dim})"
        )
    if initial_reported.device != target_angles.device:
        raise ValueError("Track-A initial states and target angles must share a device")
    if initial_reported.dtype != target_angles.dtype:
        raise ValueError("Track-A initial states and target angles must share a dtype")
    _require_finite("Track-A initial reported states", initial_reported)
    reported = initial_reported
    primary = adapter.primary_from_reported(reported)
    speeds = np.empty((int(horizon), anchors), dtype=np.float32)
    decoded = np.empty((int(horizon), anchors), dtype=np.float32)
    for step in range(int(horizon)):
        blank = adapter.zero_input(reported)
        next_reported = adapter.reported_step(reported, blank)
        next_primary = adapter.primary_from_reported(next_reported)
        output = adapter.decode(next_reported)
        speed = torch.linalg.vector_norm(next_primary - primary, dim=-1)
        angle = torch.atan2(output[:, 1], output[:, 0])
        _require_finite("Track-A blank rollout", next_reported, speed, angle)
        speeds[step] = speed.detach().cpu().float().numpy()
        decoded[step] = angle.detach().cpu().float().numpy()
        reported, primary = next_reported, next_primary

    maximum = speeds.max(axis=0)
    threshold = SLOW_RELATIVE_SPEED * maximum
    positive_maximum = maximum > 0.0
    candidate_mask = positive_maximum[None, :] & (
        speeds <= threshold[None, :]
    )
    # An exactly stationary trajectory is the limiting slow case, not a
    # reconstruction failure.  Keep one representative rather than H copies.
    all_zero = ~positive_maximum
    candidate_mask[0, all_zero] = True
    candidate_time, candidate_trajectory = np.nonzero(candidate_mask)
    candidate_angle = decoded[candidate_time, candidate_trajectory]
    reasons: list[str] = []
    selected_time = np.full(anchors, -1, dtype=np.int64)
    selected_trajectory = np.full(anchors, -1, dtype=np.int64)
    selected_decoded = np.full(anchors, np.nan, dtype=np.float32)
    selected_state = torch.empty(
        0, adapter.primary_dim, device=target_angles.device, dtype=target_angles.dtype
    )
    resampled = selected_state
    coverage: dict[str, Any] = {
        "passed": False,
        "reason": "no_candidates",
        "scope": "full_ring_decoded_angle_coverage_QA_not_CA_claim_threshold",
    }
    selection_error = np.empty(0, dtype=np.float64)

    if candidate_angle.size == 0:
        reasons.append("no_state_satisfied_relative_speed_criterion")
    else:
        wrapped_candidate = np.remainder(candidate_angle, 2.0 * math.pi)
        order = np.argsort(wrapped_candidate, kind="mergesort")
        sorted_angle = wrapped_candidate[order]
        targets = np.remainder(target_angles.detach().cpu().double().numpy(), 2.0 * math.pi)
        position = np.searchsorted(sorted_angle, targets, side="left")
        right_position = np.remainder(position, sorted_angle.size)
        left_position = np.remainder(position - 1, sorted_angle.size)
        right_index = order[right_position]
        left_index = order[left_position]
        right_distance = _circular_distance_numpy(candidate_angle[right_index], targets)
        left_distance = _circular_distance_numpy(candidate_angle[left_index], targets)
        chosen = np.where(left_distance <= right_distance, left_index, right_index)
        selected_time = candidate_time[chosen].astype(np.int64, copy=False)
        selected_trajectory = candidate_trajectory[chosen].astype(np.int64, copy=False)
        selected_decoded = candidate_angle[chosen].astype(np.float32, copy=False)

        selected_state = torch.empty(
            anchors, adapter.primary_dim,
            device=target_angles.device, dtype=target_angles.dtype,
        )
        # Replay from the same task-driven endpoints used to discover the
        # candidate times.  Reverting to canonical initialization here would
        # silently pair endpoint-derived indices with unrelated states.
        reported = initial_reported
        for step in range(int(horizon)):
            reported = adapter.reported_step(reported, adapter.zero_input(reported))
            wanted = np.nonzero(selected_time == step)[0]
            if wanted.size:
                destination = torch.as_tensor(wanted, device=target_angles.device)
                source = torch.as_tensor(
                    selected_trajectory[wanted], device=target_angles.device
                )
                selected_state[destination] = adapter.primary_from_reported(reported)[source]

        pair = np.stack((selected_time, selected_trajectory), axis=1)
        unique_pair = np.unique(pair, axis=0).shape[0]
        selected_wrapped = np.remainder(selected_decoded.astype(np.float64), 2.0 * math.pi)
        unique_angles = np.unique(np.round(selected_wrapped, decimals=10))
        minimum_unique = 4
        if unique_pair < minimum_unique:
            reasons.append(
                f"selected_candidate_coverage_{unique_pair}_below_{minimum_unique}"
            )
        if unique_angles.size < minimum_unique:
            reasons.append(
                f"decoded_knot_coverage_{unique_angles.size}_below_{minimum_unique}"
            )
        mathematical_reasons = list(reasons)
        if not mathematical_reasons:
            try:
                # Multiple targets may choose the same candidate.  Use each
                # selected candidate once; this preserves exact Track-A
                # selection while avoiding duplicate spline knots.
                _, unique_index = np.unique(pair, axis=0, return_index=True)
                unique_index.sort()
                knot_angles = torch.as_tensor(
                    selected_decoded[unique_index],
                    device=target_angles.device,
                    dtype=target_angles.dtype,
                )
                knot_states = selected_state[
                    torch.as_tensor(unique_index, device=target_angles.device)
                ]
                resampled = _periodic_cubic_resample(
                    knot_angles, knot_states, target_angles
                )
            except (RuntimeError, ValueError) as exc:
                reasons.append(f"periodic_cubic_resampling_failed:{type(exc).__name__}:{exc}")
        coverage, selection_error = _track_a_coverage_quality(
            selected_decoded,
            pair,
            target_angles,
            coarse_bin_count=coarse_coverage_bin_count,
            minimum_coarse_occupancy_fraction=minimum_coarse_occupancy_fraction,
        )
        if not bool(coverage["passed"]):
            reasons.append("full_ring_candidate_coverage_failed")

    candidate_per_trajectory = candidate_mask.sum(axis=0).astype(np.int64)
    return {
        "success": not reasons,
        "failure_reasons": reasons,
        "start_state_rule": effective_start_rule,
        "rollout_horizon": int(horizon),
        "trajectory_count": anchors,
        "candidate_threshold": (
            "speed_t <= 1e-3 * max_t speed_t per nonstationary trajectory; "
            "one representative retained when max_t speed_t == 0"
        ),
        "candidate_count": int(candidate_angle.size),
        "candidate_trajectory_coverage": int(np.count_nonzero(candidate_per_trajectory)),
        "max_speed": maximum,
        "candidate_count_per_trajectory": candidate_per_trajectory,
        "selected_time": selected_time,
        "selected_trajectory": selected_trajectory,
        "selected_decoded_angle": selected_decoded,
        "selected_target_error_radians": selection_error,
        "selected_state": selected_state,
        "resampled_state": resampled,
        "coverage": coverage,
        "geometry_available": bool(
            resampled.ndim == 2
            and tuple(resampled.shape) == (anchors, adapter.primary_dim)
        ),
        "interpolation": (
            "periodic_cubic_spline_cyclic_linear_system"
            if resampled.ndim == 2
            and tuple(resampled.shape) == (anchors, adapter.primary_dim)
            else None
        ),
    }


@torch.no_grad()
def _task_conditioned_atlas(
    model: ProtocolModel,
    adapter: StateAdapter,
    angles: torch.Tensor,
    plan: AnalysisPlan,
    *,
    primary_projector: Mapping[str, Any] | None = None,
    primary_state_scale: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build the independent eight-path task sheet and test its settling.

    Settling is meaningful only relative to the already reconstructed Track-A
    manifold.  When that primary geometry is unavailable, the task states are
    still returned for diagnostics but no settling horizon is selected.
    """

    canonical = _atlas_states(model, adapter, angles)
    velocity = _ring_path_velocity_bank(angles, steps=plan.path_steps)
    endpoint = _run_velocity_path_endpoints(model, adapter, velocity)
    needed = sorted(set(plan.settle_horizons) | {value + 5 for value in plan.settle_horizons})
    states: dict[int, torch.Tensor] = {0: endpoint}
    current = endpoint.reshape(-1, adapter.primary_dim)
    for step in range(1, max(needed) + 1):
        current = adapter.actual_f0(current)
        if step in needed:
            states[step] = current.reshape_as(endpoint).clone()

    primary_geometry_available = (
        primary_projector is not None and primary_state_scale is not None
    )
    if (primary_projector is None) != (primary_state_scale is None):
        raise ValueError(
            "task-sheet settling requires both primary_projector and primary_state_scale"
        )

    absolute_q95: list[float | None] = []
    expansion_q95: list[float | None] = []
    sheet_distance_q95: list[float | None] = []
    systematic_fraction: list[float | None] = []
    criteria_pass: list[bool] = []
    quality_by_horizon: list[dict[str, Any] | None] = []
    for horizon in plan.settle_horizons:
        state = states[horizon]
        if not primary_geometry_available:
            quality_by_horizon.append(None)
            absolute_q95.append(None)
            expansion_q95.append(None)
            sheet_distance_q95.append(None)
            systematic_fraction.append(None)
            criteria_pass.append(False)
            continue
        assert primary_projector is not None
        assert primary_state_scale is not None
        quality = settling_quality_diagnostics(
            state,
            state_scale=primary_state_scale,
            actual_f0=adapter.actual_f0,
            project_mean_sheet=lambda value: _project_ring(
                value, primary_projector
            ),
            blank_steps=5,
            absolute_relative_change_q95_max=(
                plan.settle_absolute_change_q95_max
            ),
            positive_expansion_q95_max=(
                plan.settle_positive_expansion_q95_max
            ),
            mean_sheet_distance_q95_max=(
                plan.settle_mean_sheet_distance_q95_max
            ),
        )
        quality_by_horizon.append(quality)
        absolute_q95.append(float(quality["absolute_relative_change_q95"].cpu()))
        expansion_q95.append(float(quality["positive_expansion_q95"].cpu()))
        sheet_distance_q95.append(float(quality["mean_sheet_distance_q95"].cpu()))
        systematic_fraction.append(float(quality["systematic_decrease_fraction"]))
        criteria_pass.append(bool(quality["passed"]))

    selected_index = next((i for i, passed in enumerate(criteria_pass) if passed), None)
    selected_horizon = (
        int(plan.settle_horizons[selected_index]) if selected_index is not None else None
    )
    fallback_horizon = max(plan.settle_horizons)
    selected_paths = (
        states[selected_horizon]
        if selected_horizon is not None
        else states[fallback_horizon]
    )
    selected_mean = selected_paths.mean(dim=1)
    try:
        rs, _ = _state_scale(selected_mean)
        within = (
            (selected_paths - selected_mean[:, None, :]).square().sum(dim=-1).mean(dim=1)
            / rs.square()
        )
        distance = torch.cdist(selected_mean, selected_mean).square() / rs.square()
        distance.fill_diagonal_(float("inf"))
        between = distance.min(dim=1).values
        chi = float(
            within.median().cpu()
            / (between.median().cpu() + torch.finfo(selected_mean.dtype).eps)
        )
        scale_valid = math.isfinite(float(rs)) and float(rs) > 0.0
    except ValueError:
        rs = torch.tensor(float("nan"), device=angles.device, dtype=angles.dtype)
        within = torch.full_like(angles, float("nan"))
        between = torch.full_like(angles, float("nan"))
        chi = float("nan")
        scale_valid = False
    valid = bool(selected_horizon is not None and scale_valid)
    return {
        "valid": valid,
        "classification": "settled_task_sheet" if valid else "transient_task_sheet",
        "canonical_state": canonical,
        "path_velocity": velocity,
        "path_endpoint_state": endpoint,
        "settled_state_by_horizon": torch.stack(
            [states[horizon] for horizon in plan.settle_horizons], dim=0
        ),
        "settle_horizons": np.asarray(plan.settle_horizons, dtype=np.int64),
        "selected_horizon": selected_horizon,
        "selected_state_scale_valid": scale_valid,
        "selected_path_state": selected_paths,
        "selected_mean_state": selected_mean,
        "criteria": {
            "primary_track_a_geometry_available": primary_geometry_available,
            "absolute_symmetric_relative_variance_change_q95": absolute_q95,
            "positive_relative_variance_expansion_q95": expansion_q95,
            "F0_power_5_mean_sheet_manifold_distance_over_Rs_q95": sheet_distance_q95,
            "systematic_transverse_decrease_fraction": systematic_fraction,
            "passed": criteria_pass,
            "rules": {
                "absolute_symmetric_relative_variance_change_q95_max": plan.settle_absolute_change_q95_max,
                "positive_relative_variance_expansion_q95_max": plan.settle_positive_expansion_q95_max,
                "F0_power_5_mean_sheet_manifold_distance_over_Rs_q95_max": plan.settle_mean_sheet_distance_q95_max,
                "source_status": "corrected_v2_settling_QA_not_CA_claim_threshold",
            },
        },
        "quality_by_horizon": quality_by_horizon,
        "within_variance_normalized": within,
        "between_nearest_separation_normalized": between,
        "chi_fiber": chi,
        "near_single_sheet": bool(valid and math.isfinite(chi) and chi <= 0.1),
    }


@torch.no_grad()
def _task_track_a_correspondence(
    task_atlas: Mapping[str, Any],
    *,
    track_a_state: torch.Tensor,
    track_a_angles: torch.Tensor,
    projector: Mapping[str, Any],
    state_scale: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Compare the independent task sheet with the primary Track-A curve."""

    path_state = task_atlas["selected_path_state"]
    mean_state = task_atlas["selected_mean_state"]
    anchors, paths, dimension = path_state.shape
    flat_projection = _project_ring(path_state.reshape(-1, dimension), projector)
    path_distance = flat_projection["distance"].reshape(anchors, paths) / state_scale
    path_angle = flat_projection["angle"].reshape(anchors, paths)
    path_angle_error = (
        wrap_angle(path_angle - track_a_angles[:, None]).abs() / math.pi
    )
    mean_projection = _project_ring(mean_state, projector)
    mean_distance = mean_projection["distance"] / state_scale
    mean_angle_error = (
        wrap_angle(mean_projection["angle"] - track_a_angles).abs() / math.pi
    )

    pairwise = torch.cdist(track_a_state, mean_state) / state_scale
    track_a_to_task_distance, track_a_to_task_index = pairwise.min(dim=1)
    task_to_track_a_distance, task_to_track_a_index = pairwise.min(dim=0)
    symmetric_hausdorff = torch.maximum(
        track_a_to_task_distance.max(), task_to_track_a_distance.max()
    )
    path_per_anchor_max = path_distance.max(dim=1).values
    path_angle_per_anchor_max = path_angle_error.max(dim=1).values
    valid = bool(task_atlas["valid"])
    summary = {
        "role": "independent_correspondence_and_fiber_diagnostic_only",
        "primary_manifold": "Sagodi_Track_A_periodic_spline",
        "task_sheet_settling_valid": valid,
        "selected_task_sheet_horizon": task_atlas["selected_horizon"],
        "mean_state_distance_to_track_A_over_Rs": _summary(mean_distance),
        "path_state_distance_to_track_A_over_Rs_per_anchor_max": _summary(
            path_per_anchor_max
        ),
        "mean_projected_angle_error_pi_normalized": _summary(mean_angle_error),
        "path_projected_angle_error_pi_normalized_per_anchor_max": _summary(
            path_angle_per_anchor_max
        ),
        "fiber_spread_chi": (
            float(task_atlas["chi_fiber"])
            if math.isfinite(float(task_atlas["chi_fiber"]))
            else None
        ),
        "directed_hausdorff_track_A_to_task_mean_over_Rs": float(
            track_a_to_task_distance.max().cpu()
        ),
        "directed_hausdorff_task_mean_to_track_A_over_Rs": float(
            task_to_track_a_distance.max().cpu()
        ),
        "symmetric_hausdorff_over_Rs": float(symmetric_hausdorff.cpu()),
        "claim_gate": False,
    }
    arrays = {
        "task_path_distance_to_track_A_over_Rs": path_distance,
        "task_path_projected_angle": path_angle,
        "task_path_projected_angle_error_pi_normalized": path_angle_error,
        "task_mean_distance_to_track_A_over_Rs": mean_distance,
        "task_mean_projected_angle": mean_projection["angle"],
        "task_mean_projected_angle_error_pi_normalized": mean_angle_error,
        "track_A_to_task_mean_distance_over_Rs": track_a_to_task_distance,
        "track_A_to_task_mean_nearest_index": track_a_to_task_index,
        "task_mean_to_track_A_distance_over_Rs": task_to_track_a_distance,
        "task_mean_to_track_A_nearest_index": task_to_track_a_index,
    }
    return summary, arrays


@torch.no_grad()
def _roll_f0(adapter: StateAdapter, state: torch.Tensor, horizon: int) -> torch.Tensor:
    current = state
    for _ in range(int(horizon)):
        current = adapter.actual_f0(current)
    return current


@torch.no_grad()
def _nearest_atlas(
    query: torch.Tensor,
    atlas: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for start in range(0, query.shape[0], int(chunk_size)):
        values = torch.cdist(query[start : start + int(chunk_size)], atlas)
        distance, index = values.min(dim=1)
        distances.append(distance)
        indices.append(index)
    return torch.cat(distances), torch.cat(indices)


def _fit_heldout_linear_decoder(
    carrier: torch.Tensor,
    angles: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Fit on alternating anchors and evaluate only the held-out anchors."""

    state = carrier.detach().cpu().double().numpy()
    target = _angle_embedding(angles).detach().cpu().double().numpy()
    index = np.arange(state.shape[0])
    train = index % 2 == 0
    heldout = ~train
    design = np.concatenate((state[train], np.ones((int(train.sum()), 1))), axis=1)
    coefficient, _, _, _ = np.linalg.lstsq(design, target[train], rcond=None)
    weight = coefficient[:-1]
    bias = coefficient[-1]
    prediction = state[heldout] @ weight + bias
    decoded = np.arctan2(prediction[:, 1], prediction[:, 0])
    true = angles.detach().cpu().double().numpy()[heldout]
    delta = np.arctan2(np.sin(decoded - true), np.cos(decoded - true))
    error = np.abs(delta) / math.pi
    summary = distribution_summary(error)
    return weight, bias, index[train], index[heldout], summary


def _linear_decode_angles(
    state: torch.Tensor,
    weight: np.ndarray,
    bias: np.ndarray,
) -> torch.Tensor:
    w = torch.as_tensor(weight, device=state.device, dtype=state.dtype)
    b = torch.as_tensor(bias, device=state.device, dtype=state.dtype)
    embedded = state @ w + b
    return torch.atan2(embedded[..., 1], embedded[..., 0])


@torch.no_grad()
def _tangent_geometry(
    angles: torch.Tensor,
    base_state: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    plus = _periodic_cubic_resample(angles, base_state, angles + float(epsilon))
    minus = _periodic_cubic_resample(angles, base_state, angles - float(epsilon))
    derivative = (plus - minus) / (2.0 * float(epsilon))
    speed = torch.linalg.vector_norm(derivative, dim=-1)
    floor = 100.0 * torch.finfo(derivative.dtype).eps
    tangent = derivative / speed.clamp_min(floor)[:, None]
    curvature = (plus - 2.0 * base_state + minus) / float(epsilon) ** 2
    return derivative, tangent, speed, curvature


def _tangent_finite_difference_sensitivity(
    angles: torch.Tensor,
    atlas_state: torch.Tensor,
    epsilons: Sequence[float],
) -> dict[str, Any]:
    geometries: dict[float, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for epsilon in epsilons:
        geometries[float(epsilon)] = _tangent_geometry(
            angles, atlas_state, float(epsilon)
        )
    comparisons: list[dict[str, Any]] = []
    values = tuple(float(value) for value in epsilons)
    for left, right in zip(values[:-1], values[1:]):
        left_derivative, left_tangent, _, _ = geometries[left]
        right_derivative, right_tangent, _, _ = geometries[right]
        denominator = 0.5 * (
            torch.linalg.vector_norm(left_derivative, dim=-1)
            + torch.linalg.vector_norm(right_derivative, dim=-1)
        )
        relative = torch.linalg.vector_norm(
            left_derivative - right_derivative, dim=-1
        ) / denominator.clamp_min(100.0 * torch.finfo(atlas_state.dtype).eps)
        principal_cosine = (left_tangent * right_tangent).sum(dim=-1).abs()
        comparisons.append(
            {
                "epsilon_left": left,
                "epsilon_right": right,
                "derivative_relative_difference": _summary(relative),
                "absolute_tangent_principal_cosine": _summary(principal_cosine),
            }
        )
    return {
        "epsilons": values,
        "geometries": geometries,
        "comparisons": comparisons,
        "status": "diagnostic_only_no_preregistered_cutoff",
    }


@torch.no_grad()
def _track_a_geometry_quality(
    angles: torch.Tensor,
    atlas_state: torch.Tensor,
    tangent_speed: torch.Tensor,
    state_scale: torch.Tensor,
    *,
    minimum_normalized_tangent_speed: float,
    seam_probe_radians: float,
    seam_c0_over_rs_max: float,
    seam_c1_relative_max: float,
) -> dict[str, Any]:
    """Fail closed on a collapsed chart or a non-periodic spline seam."""

    epsilon = float(seam_probe_radians)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("Track-A seam probe must be finite and positive")
    seam = torch.tensor([-math.pi], device=angles.device, dtype=angles.dtype)
    seam_state = _periodic_cubic_resample(angles, atlas_state, seam)
    period_state = _periodic_cubic_resample(
        angles, atlas_state, seam + 2.0 * math.pi
    )
    left_state = _periodic_cubic_resample(angles, atlas_state, seam - epsilon)
    right_state = _periodic_cubic_resample(angles, atlas_state, seam + epsilon)
    left_derivative = (seam_state - left_state) / epsilon
    right_derivative = (right_state - seam_state) / epsilon
    seam_c0 = torch.linalg.vector_norm(period_state - seam_state) / state_scale
    derivative_scale = 0.5 * (
        torch.linalg.vector_norm(left_derivative)
        + torch.linalg.vector_norm(right_derivative)
    )
    seam_c1 = torch.linalg.vector_norm(left_derivative - right_derivative) / (
        derivative_scale.clamp_min(100.0 * torch.finfo(atlas_state.dtype).eps)
    )
    normalized_speed = tangent_speed / state_scale
    minimum_speed = float(normalized_speed.min().cpu())
    c0_value = float(seam_c0.cpu())
    c1_value = float(seam_c1.cpu())
    speed_pass = minimum_speed >= float(minimum_normalized_tangent_speed)
    c0_pass = c0_value <= float(seam_c0_over_rs_max)
    c1_pass = c1_value <= float(seam_c1_relative_max)
    return {
        "passed": bool(speed_pass and c0_pass and c1_pass),
        "minimum_normalized_tangent_speed": minimum_speed,
        "seam_C0_distance_over_Rs": c0_value,
        "seam_C1_relative_difference": c1_value,
        "criteria": {
            "minimum_normalized_tangent_speed": float(
                minimum_normalized_tangent_speed
            ),
            "seam_probe_radians": epsilon,
            "seam_C0_distance_over_Rs_max": float(seam_c0_over_rs_max),
            "seam_C1_relative_difference_max": float(seam_c1_relative_max),
        },
        "component_passed": {
            "tangent_speed_floor": speed_pass,
            "periodic_seam_C0": c0_pass,
            "periodic_seam_C1": c1_pass,
        },
    }


def _ring_local_rank_metrics(
    derivative: torch.Tensor,
    state_scale: torch.Tensor,
    *,
    minimum_ratio: float,
    required_fraction: float,
) -> dict[str, Any]:
    """Evaluate the frozen local-rank gate for a one-dimensional ring.

    At each anchor the chart Jacobian is ``[carrier_dim, 1]``.  Its sole
    singular value is therefore both sigma_min and sigma_max.  The ratio is
    one for a resolved local direction, but is explicitly set to zero when
    the normalized singular value is numerically indistinguishable from zero;
    this prevents a collapsed 0/0 chart from passing the otherwise-trivial
    d=1 ratio.
    """

    if derivative.ndim != 2:
        raise ValueError("ring tangent derivative must have shape [anchor, carrier]")
    if not 0.0 < float(minimum_ratio) <= 1.0:
        raise ValueError("minimum local singular-value ratio must be in (0, 1]")
    if not 0.0 < float(required_fraction) <= 1.0:
        raise ValueError("required local-rank atlas fraction must be in (0, 1]")
    jacobian = derivative.unsqueeze(-1)
    singular = torch.linalg.svdvals(jacobian)
    sigma_max = singular[..., 0] / state_scale
    sigma_min = singular[..., -1] / state_scale
    numerical_zero_tolerance = 100.0 * torch.finfo(derivative.dtype).eps
    nonzero = sigma_max > numerical_zero_tolerance
    ratio = torch.where(
        nonzero,
        sigma_min / sigma_max.clamp_min(numerical_zero_tolerance),
        torch.zeros_like(sigma_max),
    )
    indicator = nonzero & (ratio >= float(minimum_ratio))
    fraction = float(indicator.double().mean().cpu())
    return {
        "normalized_sigma_min": sigma_min,
        "normalized_sigma_max": sigma_max,
        "normalized_sigma_d_over_sigma_1": ratio,
        "numerically_nonzero": nonzero,
        "qualifying_indicator": indicator,
        "summary": {
            "intrinsic_dimension_d": 1,
            "normalized_sigma_min": _summary(sigma_min),
            "normalized_sigma_max": _summary(sigma_max),
            "normalized_sigma_d_over_sigma_1": _summary(ratio),
            "minimum_normalized_sigma_d_over_sigma_1": float(minimum_ratio),
            "required_atlas_fraction": float(required_fraction),
            "qualifying_atlas_fraction": fraction,
            "numerically_nonzero_atlas_fraction": float(
                nonzero.double().mean().cpu()
            ),
            "numerical_zero_tolerance_normalized_sigma_max": float(
                numerical_zero_tolerance
            ),
            "numerical_zero_policy": (
                "set sigma_d/sigma_1 to zero when normalized sigma_1 is at "
                "or below 100 machine eps"
            ),
            "passed": bool(fraction >= float(required_fraction)),
        },
    }


def _state_scale(carrier: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center = carrier.mean(dim=0)
    radius = torch.linalg.vector_norm(carrier - center, dim=-1)
    scale = radius.median()
    floor = 100.0 * torch.finfo(carrier.dtype).eps
    if not bool(torch.isfinite(scale)) or float(scale) <= floor:
        raise ValueError("task-conditioned atlas collapsed: R_s is numerically zero")
    return scale, center


def _stratified_indices(total: int, count: int, device: torch.device) -> torch.Tensor:
    count = min(int(count), int(total))
    values = torch.floor(torch.arange(count, device=device) * (float(total) / count)).long()
    return values.clamp_max(int(total) - 1)


def _normal_directions(
    state: torch.Tensor,
    tangent: torch.Tensor,
    curvature: torch.Tensor,
    center: torch.Tensor,
    ambient_count: int,
    seed: int,
    raw_ambient: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct centered radial and sampled ambient carrier normals."""

    centered = state - center
    radial = centered - (centered * tangent).sum(dim=-1, keepdim=True) * tangent
    radial_norm = torch.linalg.vector_norm(radial, dim=-1, keepdim=True)
    bad = radial_norm.squeeze(-1) <= 100.0 * torch.finfo(state.dtype).eps
    if bool(bad.any()):
        fallback = curvature - (curvature * tangent).sum(dim=-1, keepdim=True) * tangent
        radial = torch.where(bad[:, None], fallback, radial)
        radial_norm = torch.linalg.vector_norm(radial, dim=-1, keepdim=True)
    if bool((radial_norm <= 100.0 * torch.finfo(state.dtype).eps).any()):
        raise ValueError("could not define a radial carrier normal")
    radial = radial / radial_norm

    if raw_ambient is None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        raw = torch.randn(
            state.shape[0], int(ambient_count), state.shape[1],
            generator=generator,
            dtype=torch.float64,
        ).to(device=state.device, dtype=state.dtype)
    else:
        expected = (state.shape[0], int(ambient_count), state.shape[1])
        if tuple(raw_ambient.shape) != expected:
            raise ValueError(
                f"raw perturbation bank has shape {tuple(raw_ambient.shape)}, expected {expected}"
            )
        raw = raw_ambient.to(device=state.device, dtype=state.dtype)
    raw = raw - (raw * tangent[:, None, :]).sum(dim=-1, keepdim=True) * tangent[:, None, :]
    # Removing radial makes the reported ambient family distinct from the
    # explicitly tested task-active radial direction.  It remains a sampled
    # carrier normal, not a strict worst normal direction.
    raw = raw - (raw * radial[:, None, :]).sum(dim=-1, keepdim=True) * radial[:, None, :]
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    if bool((norm <= 100.0 * torch.finfo(state.dtype).eps).any()):
        raise ValueError("carrier dimension is too small for sampled ambient normals")
    ambient = raw / norm
    return radial, ambient


def _normal_direction_orthogonality(
    tangent: torch.Tensor,
    radial: torch.Tensor,
    ambient: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Absolute tangent dot products for every sampled normal direction."""

    radial_error = (radial * tangent).sum(dim=-1).abs()
    ambient_error = (
        ambient * tangent[:, None, :]
    ).sum(dim=-1).abs()
    return {
        "radial": radial_error,
        "ambient": ambient_error,
    }


def _normal_direction_qa_summary(
    kick: Mapping[str, torch.Tensor],
    jacobian: Mapping[str, torch.Tensor],
    *,
    maximum: float,
) -> dict[str, Any]:
    values = torch.cat(
        (
            kick["radial"].reshape(-1),
            kick["ambient"].reshape(-1),
            jacobian["radial"].reshape(-1),
            jacobian["ambient"].reshape(-1),
        )
    )
    observed_max = float(values.max().cpu()) if values.numel() else float("inf")
    return {
        "metric": "absolute_inner_product_of_unit_tangent_and_sampled_unit_normal",
        "threshold_max": float(maximum),
        "passed": bool(observed_max <= float(maximum)),
        "overall": _summary(values),
        "kick_radial": _summary(kick["radial"]),
        "kick_ambient": _summary(kick["ambient"]),
        "jacobian_radial": _summary(jacobian["radial"]),
        "jacobian_ambient": _summary(jacobian["ambient"]),
        "families": ["finite_kick_radial", "finite_kick_ambient", "jacobian_radial", "jacobian_ambient"],
    }


@torch.no_grad()
def _finite_kicks(
    adapter: StateAdapter,
    atlas_state: torch.Tensor,
    atlas_angles: torch.Tensor,
    atlas_tangent: torch.Tensor,
    anchor_indices: torch.Tensor,
    radial: torch.Tensor,
    ambient: torch.Tensor,
    radius: torch.Tensor,
    horizon: int,
    projector: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Retain the former clean-paired endpoint decomposition as diagnostics.

    These values are not distances to the manifold and may not satisfy C3.
    Primary recovery is computed by :func:`manifold_recovery_diagnostics`.
    """

    base = atlas_state[anchor_indices]
    directions = torch.cat((radial[:, None, :], ambient), dim=1)
    anchors, conditions, dimension = directions.shape
    delta0 = float(radius) * directions
    perturbed0 = (base[:, None, :] + delta0).reshape(anchors * conditions, dimension)
    clean_h = _roll_f0(adapter, base, horizon)
    perturbed_h = _roll_f0(adapter, perturbed0, horizon).reshape(anchors, conditions, dimension)
    difference = perturbed_h - clean_h[:, None, :]

    clean_projection = _project_ring(clean_h, projector)
    clean_index = clean_projection["original_index"]
    endpoint_tangent = atlas_tangent[clean_index]
    tangent_component = (difference * endpoint_tangent[:, None, :]).sum(dim=-1)
    normal_component = difference - tangent_component[..., None] * endpoint_tangent[:, None, :]
    r_n = torch.linalg.vector_norm(normal_component, dim=-1) / radius
    l_t = tangent_component.abs() / radius
    r_same = torch.linalg.vector_norm(difference, dim=-1) / radius

    flat_perturbed = perturbed_h.reshape(anchors * conditions, dimension)
    perturbed_projection = _project_ring(flat_perturbed, projector)
    perturbed_index = perturbed_projection["original_index"]
    perturbed_angle = perturbed_projection["angle"].reshape(anchors, conditions)
    clean_angle = clean_projection["angle"]
    excess = wrap_angle(perturbed_angle - clean_angle[:, None]).abs() / math.pi
    return {
        "directions": directions,
        "r_n": r_n,
        "l_t": l_t,
        "r_same": r_same,
        "e_excess": excess,
        "clean_endpoint_indices": clean_index,
        "perturbed_endpoint_indices": perturbed_index.reshape(anchors, conditions),
    }


@torch.no_grad()
def _tangent_equivariance(
    adapter: StateAdapter,
    atlas_state: torch.Tensor,
    atlas_angles: torch.Tensor,
    anchor_indices: torch.Tensor,
    derivative: torch.Tensor,
    shift: float,
    horizon: int,
    projector: Mapping[str, Any],
) -> torch.Tensor:
    angle = atlas_angles[anchor_indices]
    base = atlas_state[anchor_indices]
    additive = base + derivative[anchor_indices] * float(shift)
    true_shifted = _periodic_cubic_resample(
        atlas_angles, atlas_state, angle + float(shift)
    )
    additive_h = _roll_f0(adapter, additive, horizon)
    shifted_h = _roll_f0(adapter, true_shifted, horizon)
    additive_projection = _project_ring(additive_h, projector)
    shifted_projection = _project_ring(shifted_h, projector)
    return (
        wrap_angle(additive_projection["angle"] - shifted_projection["angle"]).abs()
        / math.pi
    )


def _sampled_jvp_gains(
    adapter: StateAdapter,
    atlas_state: torch.Tensor,
    atlas_tangent: torch.Tensor,
    anchor_indices: torch.Tensor,
    radial: torch.Tensor,
    ambient: torch.Tensor,
    horizon: int,
    projector: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Evaluate the sampled projected tangent/normal cocycles.

    The primary normal quantity is

    ``P_N(q_H) J_{H-1} P_N(q_{H-1}) ... P_N(q_1) J_0 v_N``.

    Every frame ``q_t`` is obtained by projecting the *actual clean F0
    rollout*, never a perturbed rollout.  For auditability, the former
    endpoint-only projection of the unprojected Jacobian product is retained
    under explicitly non-primary keys.
    """

    if int(horizon) <= 0:
        raise ValueError("sampled cocycle horizon must be positive")
    base = atlas_state[anchor_indices]
    anchors, dimension = base.shape
    normal_conditions = 1 + int(ambient.shape[1])
    all_conditions = 1 + normal_conditions  # one tangent, then radial+ambient

    initial_projection = _project_ring(base, projector)
    frame = initial_projection["tangent_frame"]
    supplied_normals = torch.cat((radial[:, None, :], ambient), dim=1)
    initial_normal = supplied_normals - (
        supplied_normals * frame[:, None, :]
    ).sum(dim=-1, keepdim=True) * frame[:, None, :]
    initial_normal_norm = torch.linalg.vector_norm(initial_normal, dim=-1, keepdim=True)
    frame_floor = 100.0 * torch.finfo(base.dtype).eps
    initial_normal = initial_normal / initial_normal_norm.clamp_min(frame_floor)

    primary_vector = torch.cat((frame[:, None, :], initial_normal), dim=1)
    endpoint_proxy_vector = primary_vector.clone()
    clean_state = base

    projected_angle_trace = [initial_projection["angle"].detach()]
    projected_frame_trace = [frame.detach()]
    frame_valid_trace = [initial_projection["tangent_frame_valid"].detach()]
    primary_tangent_gain_trace = [
        torch.linalg.vector_norm(primary_vector[:, 0, :], dim=-1).detach()
    ]
    primary_normal_gain_trace = [
        torch.linalg.vector_norm(primary_vector[:, 1:, :], dim=-1).detach()
    ]

    for _ in range(int(horizon)):
        # Both the primary projected cocycle and endpoint-only diagnostic are
        # differentiated along identical copies of the actual clean state.
        combined_vector = torch.cat(
            (primary_vector, endpoint_proxy_vector), dim=1
        )
        copies = int(combined_vector.shape[1])
        repeated_state = clean_state[:, None, :].expand(
            anchors, copies, dimension
        ).reshape(anchors * copies, dimension)
        flat_vector = combined_vector.reshape(anchors * copies, dimension)
        repeated_state = repeated_state.detach().requires_grad_(True)
        next_repeated_state, next_flat_vector = torch.autograd.functional.jvp(
            adapter.actual_f0,
            (repeated_state,),
            (flat_vector.detach(),),
            create_graph=False,
            strict=False,
        )
        next_state_by_copy = next_repeated_state.detach().reshape(
            anchors, copies, dimension
        )
        clean_state = next_state_by_copy[:, 0, :]
        next_vector = next_flat_vector.detach().reshape(anchors, copies, dimension)
        primary_raw = next_vector[:, :all_conditions, :]
        endpoint_proxy_vector = next_vector[:, all_conditions:, :]

        projection = _project_ring(clean_state, projector)
        frame = projection["tangent_frame"]
        tangent_raw = primary_raw[:, 0, :]
        tangent_coefficient = (tangent_raw * frame).sum(dim=-1, keepdim=True)
        projected_tangent = tangent_coefficient * frame
        normal_raw = primary_raw[:, 1:, :]
        projected_normal = normal_raw - (
            normal_raw * frame[:, None, :]
        ).sum(dim=-1, keepdim=True) * frame[:, None, :]
        primary_vector = torch.cat(
            (projected_tangent[:, None, :], projected_normal), dim=1
        )

        projected_angle_trace.append(projection["angle"].detach())
        projected_frame_trace.append(frame.detach())
        frame_valid_trace.append(projection["tangent_frame_valid"].detach())
        primary_tangent_gain_trace.append(
            torch.linalg.vector_norm(projected_tangent, dim=-1).detach()
        )
        primary_normal_gain_trace.append(
            torch.linalg.vector_norm(projected_normal, dim=-1).detach()
        )

    per_step_projected_tangent_gain = torch.linalg.vector_norm(
        primary_vector[:, 0, :], dim=-1
    )
    primary_radial_gain = torch.linalg.vector_norm(
        primary_vector[:, 1, :], dim=-1
    )
    primary_ambient_gain = torch.linalg.vector_norm(
        primary_vector[:, 2:, :], dim=-1
    )
    primary_normal_max = torch.cat(
        (primary_radial_gain[:, None], primary_ambient_gain), dim=1
    ).max(dim=1).values

    # Non-primary historical proxy: J_H...J_1 is left unprojected until the
    # endpoint, where its tangent and normal components are split once.
    endpoint_tangent_component = (
        endpoint_proxy_vector * frame[:, None, :]
    ).sum(dim=-1)
    endpoint_normal_component = endpoint_proxy_vector - (
        endpoint_tangent_component[..., None] * frame[:, None, :]
    )
    proxy_tangent_gain = endpoint_tangent_component[:, 0].abs()
    proxy_radial_gain = torch.linalg.vector_norm(
        endpoint_normal_component[:, 1, :], dim=-1
    )
    proxy_ambient_gain = torch.linalg.vector_norm(
        endpoint_normal_component[:, 2:, :], dim=-1
    )
    proxy_normal_max = torch.cat(
        (proxy_radial_gain[:, None], proxy_ambient_gain), dim=1
    ).max(dim=1).values

    tiny = torch.finfo(base.dtype).tiny

    def exponents(
        tangent_gain: torch.Tensor, normal_gain: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ell_t = torch.log(tangent_gain.clamp_min(tiny)) / float(horizon)
        ell_n = torch.log(normal_gain.clamp_min(tiny)) / float(horizon)
        return ell_t, ell_n, ell_t - ell_n

    # The protocol's tangent block is endpoint-projected only:
    # T(q_H)^T J_{0:H} T(q_0).  Intermediate P_T factors would define a
    # different cocycle.  In contrast, the primary normal block above applies
    # P_N at every step exactly as frozen.
    primary_tangent_gain = proxy_tangent_gain
    primary_ell_t, primary_ell_n, primary_gamma = exponents(
        primary_tangent_gain, primary_normal_max
    )
    proxy_ell_t, proxy_ell_n, proxy_gamma = exponents(
        proxy_tangent_gain, proxy_normal_max
    )
    return {
        "primary_tangent_gain": primary_tangent_gain,
        "primary_radial_normal_gain": primary_radial_gain,
        "primary_ambient_normal_gain": primary_ambient_gain,
        "primary_sampled_normal_gain_max": primary_normal_max,
        "primary_ell_t": primary_ell_t,
        "primary_ell_n_sampled_max": primary_ell_n,
        "primary_gamma_sampled": primary_gamma,
        "diagnostic_per_step_projected_tangent_gain": per_step_projected_tangent_gain,
        "endpoint_proxy_tangent_gain": proxy_tangent_gain,
        "endpoint_proxy_radial_normal_gain": proxy_radial_gain,
        "endpoint_proxy_ambient_normal_gain": proxy_ambient_gain,
        "endpoint_proxy_sampled_normal_gain_max": proxy_normal_max,
        "endpoint_proxy_ell_t": proxy_ell_t,
        "endpoint_proxy_ell_n_sampled_max": proxy_ell_n,
        "endpoint_proxy_gamma_sampled": proxy_gamma,
        "clean_projected_angle_trace": torch.stack(projected_angle_trace),
        "clean_projected_tangent_frame_trace": torch.stack(projected_frame_trace),
        "clean_projected_frame_valid_trace": torch.stack(frame_valid_trace),
        "primary_tangent_gain_trace": torch.stack(primary_tangent_gain_trace),
        "primary_normal_gain_trace": torch.stack(primary_normal_gain_trace),
    }


def _neighborhood_metrics(
    atlas_state: torch.Tensor,
    atlas_angles: torch.Tensor,
    *,
    neighbors: int = 10,
) -> dict[str, Any]:
    """Exact standard trustworthiness/continuity on all ring anchors."""

    n = int(atlas_state.shape[0])
    k = min(int(neighbors), max(1, (n - 1) // 2))
    if n < 4 or 2 * n - 3 * k - 1 <= 0:
        raise ValueError("too few anchors for trustworthiness/continuity")
    hidden_distance = torch.cdist(atlas_state, atlas_state)
    latent_distance = wrap_angle(
        atlas_angles[:, None] - atlas_angles[None, :]
    ).abs()
    hidden_order = torch.argsort(hidden_distance, dim=1, stable=True)
    latent_order = torch.argsort(latent_distance, dim=1, stable=True)
    hidden_rank = torch.empty_like(hidden_order)
    latent_rank = torch.empty_like(latent_order)
    ranks = torch.arange(n, device=atlas_state.device).expand(n, -1)
    hidden_rank.scatter_(1, hidden_order, ranks)
    latent_rank.scatter_(1, latent_order, ranks)
    hidden_knn = hidden_order[:, 1 : k + 1]
    latent_knn = latent_order[:, 1 : k + 1]
    hidden_membership = torch.zeros(n, n, device=atlas_state.device, dtype=torch.bool)
    latent_membership = torch.zeros_like(hidden_membership)
    hidden_membership.scatter_(1, hidden_knn, True)
    latent_membership.scatter_(1, latent_knn, True)
    unexpected_hidden = hidden_membership & ~latent_membership
    missing_hidden = latent_membership & ~hidden_membership
    penalty_trust = (latent_rank - k).clamp_min(0)[unexpected_hidden].double().sum()
    penalty_continuity = (hidden_rank - k).clamp_min(0)[missing_hidden].double().sum()
    normalization = 2.0 / (n * k * (2 * n - 3 * k - 1))
    trust = float(1.0 - normalization * float(penalty_trust.cpu()))
    continuity = float(1.0 - normalization * float(penalty_continuity.cpu()))

    upper = torch.triu_indices(n, n, offset=1, device=atlas_state.device)
    hidden_pair = hidden_distance[upper[0], upper[1]]
    latent_pair = latent_distance[upper[0], upper[1]]
    hidden_pair_rank = torch.argsort(torch.argsort(hidden_pair, stable=True), stable=True).double()
    latent_pair_rank = torch.argsort(torch.argsort(latent_pair, stable=True), stable=True).double()
    hidden_pair_rank -= hidden_pair_rank.mean()
    latent_pair_rank -= latent_pair_rank.mean()
    spearman = float(
        (hidden_pair_rank * latent_pair_rank).sum().cpu()
        / (
            torch.linalg.vector_norm(hidden_pair_rank)
            * torch.linalg.vector_norm(latent_pair_rank)
        ).clamp_min(torch.finfo(torch.float64).eps).cpu()
    )
    return {
        "trustworthiness": trust,
        "continuity": continuity,
        "neighbors_k": k,
        "anchor_count": n,
        "scope": "all_primary_atlas_anchors",
        "latent_distance": "absolute_wrapped_ring_angle",
        "hidden_distance": "euclidean_primary_carrier",
        "tie_break": "stable_anchor_index",
        "pairwise_spearman": spearman,
    }


def _load_perturbation_bank(
    path: Path | str | None,
    *,
    plan: AnalysisPlan,
    state_dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    if path is None:
        if not plan.smoke:
            raise ValueError("--perturbation-bank is required for non-smoke Phase-1 analysis")
        return None, None, {
            "source": "deterministic_smoke_fallback",
            "sha256": None,
            "common_preprojection_bank": False,
        }
    source = Path(path).expanduser().resolve(strict=True)
    sidecar = Path(f"{source}.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"perturbation-bank checksum sidecar missing: {sidecar}")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    digest = sha256_file(source)
    if len(fields) != 2 or fields[0].lower() != digest or fields[1] != source.name:
        raise ValueError("perturbation-bank SHA-256 sidecar mismatch")
    with np.load(source, allow_pickle=False) as archive:
        required = {"finite_raw", "jacobian_raw", "metadata_json"}
        if set(archive.files) != required:
            raise ValueError(
                f"perturbation bank keys must be {sorted(required)}, got {sorted(archive.files)}"
            )
        finite = np.asarray(archive["finite_raw"])
        jacobian = np.asarray(archive["jacobian_raw"])
        metadata_bytes = np.asarray(archive["metadata_json"], dtype=np.uint8).tobytes()
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("unsupported perturbation-bank metadata schema")
    if metadata.get("generator") != "numpy.random.Generator(numpy.random.PCG64)":
        raise ValueError("unexpected perturbation-bank random generator")
    if metadata.get("dtype") != "float32" or finite.dtype != np.float32 or jacobian.dtype != np.float32:
        raise ValueError("perturbation bank must be float32")
    if metadata.get("finite_raw_shape") != list(finite.shape):
        raise ValueError("finite_raw metadata shape mismatch")
    if metadata.get("jacobian_raw_shape") != list(jacobian.shape):
        raise ValueError("jacobian_raw metadata shape mismatch")
    if metadata.get("smoke_rule") != "use deterministic prefix slices":
        raise ValueError("perturbation bank does not freeze the smoke prefix rule")
    required_shape = (
        max(plan.kick_anchor_count, plan.jacobian_anchor_count),
        plan.ambient_directions,
        int(state_dimension),
    )
    for label, array, anchors in (
        ("finite_raw", finite, plan.kick_anchor_count),
        ("jacobian_raw", jacobian, plan.jacobian_anchor_count),
    ):
        if array.ndim != 3:
            raise ValueError(f"{label} must be rank 3")
        if array.shape[0] < anchors or array.shape[1] < plan.ambient_directions:
            raise ValueError(f"{label} is smaller than the effective analysis plan")
        if array.shape[2] != state_dimension:
            raise ValueError(
                f"{label} state dimension {array.shape[2]} != checkpoint {state_dimension}"
            )
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError(f"{label} must contain finite floating values")
    finite_tensor = torch.as_tensor(
        finite[: plan.kick_anchor_count, : plan.ambient_directions],
        device=device, dtype=dtype,
    )
    jacobian_tensor = torch.as_tensor(
        jacobian[: plan.jacobian_anchor_count, : plan.ambient_directions],
        device=device, dtype=dtype,
    )
    return finite_tensor, jacobian_tensor, {
        "source": str(source),
        "sha256": digest,
        "metadata": metadata,
        "common_preprojection_bank": True,
        "full_shape": list(finite.shape),
        "effective_finite_shape": list(finite_tensor.shape),
        "effective_jacobian_shape": list(jacobian_tensor.shape),
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().float().numpy())
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _evaluation_batch(
    *,
    path: Path | str | None,
    plan: AnalysisPlan,
    protocol: Mapping[str, Any],
    task_spec: AngularTaskSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    seed_policy = protocol["seed_policy"]
    if path is None:
        if not plan.smoke:
            raise ValueError("--evaluation-bank is required for non-smoke Phase-1 analysis")
        generated_count = max(int(plan.task_trials), int(plan.atlas_count))
        batch = sample_angular_integration(
            generated_count,
            plan.task_horizon,
            int(seed_policy["task_seed"]),
            int(seed_policy["evaluation_bank_seed"]),
            "hidden-init",
            device=device,
            dtype=dtype,
            task_spec=task_spec,
            allow_horizon_override=plan.smoke,
        )
        source = {
            "source": "deterministic_generator_fallback",
            "sha256": None,
            "fixed_bank": False,
        }
    else:
        bank_path = Path(path).expanduser().resolve(strict=True)
        batch = load_fixed_bank(bank_path, device=device)
        _validate_evaluation_batch(
            batch,
            protocol,
            require_full_protocol_shape=not plan.smoke,
        )
        source = {
            "source": str(bank_path),
            "sha256": sha256_file(bank_path),
            "fixed_bank": True,
        }
    bound_task_spec = batch.metadata.get("resolved_task_spec_sha256")
    if bound_task_spec is not None and bound_task_spec != task_spec.fingerprint():
        raise ValueError("evaluation bank resolved-task fingerprint differs from protocol")
    if batch.inputs.shape[-1] != 1 or batch.output_targets.shape[-1] != 2:
        raise ValueError("evaluation bank is not a single-ring angular-integration bank")
    if batch.metadata.get("task_name") != "angular_integration":
        raise ValueError("evaluation bank task_name is not angular_integration")
    if batch.metadata.get("init_mode") != "hidden-init":
        raise ValueError("evaluation bank initialization mode is not hidden-init")
    if int(batch.metadata.get("task_seed", -1)) != int(seed_policy["task_seed"]):
        raise ValueError("evaluation bank task seed differs from protocol")
    if int(batch.metadata.get("sample_seed", -1)) != int(seed_policy["evaluation_bank_seed"]):
        raise ValueError("evaluation bank sample seed differs from protocol")
    if batch.time_steps < plan.task_horizon or batch.batch_size < plan.task_trials:
        raise ValueError("evaluation bank is smaller than the effective analysis plan")
    if not plan.smoke and (
        batch.time_steps != plan.task_horizon or batch.batch_size != plan.task_trials
    ):
        raise ValueError("non-smoke evaluation bank must exactly match the frozen plan")
    memory = batch.initial_memory
    if memory is None:
        raise ValueError("hidden-init evaluation bank has no initial memory")
    time = plan.task_horizon
    count = plan.task_trials
    discovery_count = int(batch.batch_size)
    view = {
        "inputs": batch.inputs[:time, :count].to(device=device, dtype=dtype),
        "targets": batch.output_targets[:time, :count].to(device=device, dtype=dtype),
        "latents": batch.latent_targets[:time, :count].to(device=device, dtype=dtype),
        "mask": batch.mask[:time, :count].to(device=device, dtype=dtype),
        "initial_memory": memory[:count].to(device=device, dtype=dtype),
        # The clean task metric keeps its frozen effective count above.  Track A
        # independently sees the complete fixed evaluation bank so 1,024
        # model-independent, ground-truth-stratified discovery trials can be
        # selected from all 4,096 full-protocol trials.
        "discovery_inputs": batch.inputs[:time, :discovery_count].to(
            device=device, dtype=dtype
        ),
        "discovery_latents": batch.latent_targets[:time, :discovery_count].to(
            device=device, dtype=dtype
        ),
        "discovery_initial_memory": memory[:discovery_count].to(
            device=device, dtype=dtype
        ),
    }
    _require_finite("evaluation bank", *view.values())
    source["effective_shape"] = {
        "time": time,
        "batch": count,
        "input": 1,
        "output": 2,
        "track_a_discovery_batch": discovery_count,
    }
    source["resolved_task_spec_sha256"] = task_spec.fingerprint()
    source["bank_metadata_task_spec_sha256"] = bound_task_spec
    source["legacy_v1_metadata_binding"] = bound_task_spec is None
    return view, source


def _verify_analysis_freeze_parent(
    *,
    analysis_contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_path: Path,
    run_dir: Path,
    campaign_identity: str | None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Bind reanalysis to the immutable parent campaign and manifest."""

    parent = analysis_contract["parent_training"]
    expected_protocol = {
        "freeze_id": str(protocol["freeze_id"]),
        "protocol_canonical_fingerprint": protocol_fingerprint(protocol),
        "protocol_file_sha256": sha256_file(protocol_path),
    }
    for key, observed in expected_protocol.items():
        if observed != parent[key]:
            raise ValueError(
                f"analysis freeze parent mismatch for {key}: "
                f"{observed!r} != {parent[key]!r}"
            )
    if campaign_identity != parent["scientific_identity"]:
        raise ValueError("analysis freeze scientific identity differs from parent run")
    if run_dir.parent.name != "runs":
        raise ValueError("reanalysis parent run must be below the campaign runs directory")
    campaign_root = run_dir.parent.parent.resolve(strict=True)
    if campaign_root.name != parent["campaign_id"]:
        raise ValueError("analysis freeze campaign_id differs from parent artifact root")
    manifest_path = campaign_root / "manifest.json"
    if sha256_file(manifest_path) != parent["manifest_sha256"]:
        raise ValueError("parent campaign manifest SHA-256 differs from analysis freeze")
    manifest = strict_json_load(manifest_path)
    manifest_expected = {
        "campaign_id": parent["campaign_id"],
        "scientific_identity": parent["scientific_identity"],
        "protocol_canonical_fingerprint": parent[
            "protocol_canonical_fingerprint"
        ],
        "protocol_file_sha256": parent["protocol_file_sha256"],
    }
    for key, expected in manifest_expected.items():
        if manifest.get(key) != expected:
            raise ValueError(f"parent campaign manifest mismatch for {key}")

    run_matrix = manifest.get("run_matrix")
    if not isinstance(run_matrix, list):
        raise ValueError("parent campaign manifest has no frozen run matrix")
    matching_runs = [
        item
        for item in run_matrix
        if isinstance(item, Mapping) and item.get("run_id") == run_dir.name
    ]
    if len(matching_runs) != 1:
        raise ValueError(
            "reanalysis run directory name is not exactly one frozen parent run id"
        )
    frozen_run = matching_runs[0]
    expectations = manifest.get("receipt_expectations")
    if not isinstance(expectations, Mapping):
        raise ValueError("parent campaign manifest has no receipt expectations")
    expectation = expectations.get(f"training:{run_dir.name}")
    if not isinstance(expectation, Mapping):
        raise ValueError("parent campaign has no training receipt expectation for run")
    expected_output = (campaign_root / str(expectation.get("output"))).resolve(
        strict=True
    )
    if expected_output != run_dir:
        raise ValueError("parent training output path differs from frozen run id")
    expected_receipt_metadata = {
        "campaign_scientific_identity": parent["scientific_identity"],
        "protocol_fingerprint": parent["protocol_canonical_fingerprint"],
        "run_id": run_dir.name,
        "stage": "training",
    }
    valid, reason = verify_completion_receipt(
        run_dir / "completion_receipt.json",
        expected_job_id=str(expectation.get("job_id")),
        expected_metadata=expected_receipt_metadata,
    )
    if not valid:
        raise ValueError(f"parent training receipt is invalid: {reason}")
    receipt = strict_json_load(run_dir / "completion_receipt.json")
    training_manifest = strict_json_load(run_dir / "manifest.json")
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_digest = sha256_file(checkpoint_path)
    if int(receipt.get("schema_version", -1)) != 2:
        raise ValueError("parent training receipt must use relocatable schema 2")
    receipt_artifacts = receipt.get("artifacts")
    if not isinstance(receipt_artifacts, Mapping):
        raise ValueError("parent training receipt artifacts are malformed")
    if receipt_artifacts.get("checkpoint.pt") != checkpoint_digest:
        raise ValueError("parent training receipt does not bind checkpoint bytes")
    if receipt_artifacts.get("manifest.json") != sha256_file(
        run_dir / "manifest.json"
    ):
        raise ValueError("parent training receipt does not bind run manifest")
    if training_manifest.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError("parent run manifest checkpoint hash mismatch")
    if training_manifest.get("campaign_identity") != parent["scientific_identity"]:
        raise ValueError("parent run manifest campaign identity mismatch")
    if training_manifest.get("protocol_canonical_fingerprint") != parent[
        "protocol_canonical_fingerprint"
    ]:
        raise ValueError("parent run manifest protocol fingerprint mismatch")
    frozen_model = frozen_run.get("model")
    if not isinstance(frozen_model, Mapping):
        raise ValueError("frozen parent run model entry is malformed")
    if training_manifest.get("model_id") != frozen_model.get("id"):
        raise ValueError("parent run manifest model id differs from run matrix")
    if int(training_manifest.get("model_seed", -1)) != int(
        frozen_run.get("model_seed", -2)
    ):
        raise ValueError("parent run manifest seed differs from run matrix")
    return campaign_root, dict(parent), dict(manifest)


def _verify_parent_bound_bank(
    *,
    parent_root: Path,
    parent_manifest: Mapping[str, Any],
    bank_key: str,
    supplied_path: Path | str | None,
) -> None:
    """Require the exact file and sidecar frozen by the immutable parent."""

    if supplied_path is None:
        raise ValueError(f"parent-bound reanalysis requires --{bank_key.replace('_', '-')}")
    expected = parent_manifest.get(bank_key)
    if not isinstance(expected, Mapping):
        raise ValueError(f"parent manifest has no valid {bank_key} binding")
    expected_path = (parent_root / str(expected.get("path"))).resolve(strict=True)
    observed_path = Path(supplied_path).expanduser().resolve(strict=True)
    if observed_path != expected_path:
        raise ValueError(f"{bank_key} path differs from immutable parent manifest")
    if sha256_file(observed_path) != expected.get("sha256"):
        raise ValueError(f"{bank_key} bytes differ from immutable parent manifest")
    sidecar = Path(f"{observed_path}.sha256")
    if not sidecar.is_file() or sha256_file(sidecar) != expected.get(
        "sidecar_sha256"
    ):
        raise ValueError(f"{bank_key} checksum sidecar differs from parent manifest")


def _verify_checkpoint_identity(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    run_dir: Path,
    checkpoint: Path,
    checkpoint_payload: Mapping[str, Any],
    model: ProtocolModel,
    evaluation_source: Mapping[str, Any],
    campaign_identity: str | None,
    smoke: bool,
) -> None:
    if campaign_identity is not None and (
        len(campaign_identity) != 64
        or any(character not in "0123456789abcdef" for character in campaign_identity.lower())
    ):
        raise ValueError("campaign identity must be a 64-character hexadecimal digest")
    if smoke:
        return
    if campaign_identity is None:
        raise ValueError("--campaign-identity is required for non-smoke Phase-1 analysis")
    extra = checkpoint_payload.get("extra")
    if not isinstance(extra, Mapping):
        raise ValueError("checkpoint has no protocol identity metadata")
    expected = {
        "protocol_freeze_id": protocol["freeze_id"],
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_canonical_fingerprint": protocol_fingerprint(protocol),
        "campaign_identity": campaign_identity,
        "evaluation_bank_sha256": evaluation_source.get("sha256"),
    }
    for key, value in expected.items():
        if extra.get(key) != value:
            raise ValueError(
                f"checkpoint identity mismatch for {key}: {extra.get(key)!r} != {value!r}"
            )
    model_name = model.config.name
    valid_models = {entry["id"] for entry in protocol["phase1_ring_pilot"]["models"]}
    if model_name not in valid_models:
        raise ValueError("checkpoint model is outside the frozen run matrix")
    if int(extra.get("model_seed", -1)) not in protocol["seed_policy"]["pilot_model_seeds"]:
        raise ValueError("checkpoint model seed is outside the frozen pilot set")
    if float(extra.get("learning_rate", float("nan"))) not in protocol["phase1_ring_pilot"]["training"]["learning_rate"]["active_launch_values"]:
        raise ValueError("checkpoint learning rate is outside the active frozen values")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("non-smoke checkpoint run manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_expected = {
        "protocol_freeze_id": protocol["freeze_id"],
        "protocol_file_sha256": expected["protocol_file_sha256"],
        "protocol_canonical_fingerprint": expected["protocol_canonical_fingerprint"],
        "campaign_identity": campaign_identity,
        "evaluation_bank_sha256": evaluation_source.get("sha256"),
        "model_id": model_name,
        "model_seed": int(extra["model_seed"]),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    for key, value in manifest_expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"run manifest identity mismatch for {key}")
    learning_rate = float(extra["learning_rate"])
    expected_job_id = f"{model_name}-seed{int(extra['model_seed'])}-lr{learning_rate:g}"
    valid, reason = verify_completion_receipt(
        run_dir / "completion_receipt.json",
        expected_job_id=expected_job_id,
        expected_metadata={
            "campaign_identity": campaign_identity,
            "protocol_canonical_fingerprint": expected["protocol_canonical_fingerprint"],
            "model_id": model_name,
            "model_seed": int(extra["model_seed"]),
            "evaluation_bank_sha256": evaluation_source.get("sha256"),
        },
    )
    if not valid:
        raise ValueError(f"training completion receipt validation failed: {reason}")


@torch.no_grad()
def _task_evoked_primary_state(
    model: ProtocolModel,
    adapter: StateAdapter,
    evaluation: Mapping[str, torch.Tensor],
    *,
    count: int = 2,
) -> torch.Tensor:
    batch = min(int(count), int(evaluation["inputs"].shape[1]))
    reported = model.initial_state(
        batch,
        evaluation["inputs"].device,
        initial_memory=evaluation["initial_memory"][:batch],
    )
    for token in evaluation["inputs"][:, :batch]:
        reported = model.step(token, reported)
    primary = adapter.primary_from_reported(reported)
    _require_finite("task-evoked checkpoint state", primary)
    return primary


@torch.no_grad()
def _autonomous_radial_trace(
    adapter: StateAdapter,
    *,
    atlas_state: torch.Tensor,
    atlas_tangent: torch.Tensor,
    center: torch.Tensor,
    radius: torch.Tensor,
    projector: Mapping[str, Any],
    steps: int = 20,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    anchor = atlas_state[:1]
    tangent = atlas_tangent[:1]
    centered = anchor - center
    radial = centered - (centered * tangent).sum(dim=-1, keepdim=True) * tangent
    radial_norm = torch.linalg.vector_norm(radial, dim=-1, keepdim=True)
    if float(radial_norm.min()) <= 100.0 * torch.finfo(anchor.dtype).eps:
        raise RuntimeError("radial trace could not define a nonzero radial perturbation")
    radial = radial / radial_norm
    primary = anchor + radius * radial
    reported = adapter.reported_from_primary(primary)
    pre_carrier: list[np.ndarray] = []
    post_carrier: list[np.ndarray] = []
    pre_stream: list[np.ndarray] = []
    post_stream: list[np.ndarray] = []
    inputs: list[np.ndarray] = []
    decoder_inputs: list[np.ndarray] = []
    decoder_outputs: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    nearest_distance: list[float] = []
    nearest_index: list[int] = []
    overwrite_mask = np.zeros(adapter.reported_dim, dtype=np.uint8)
    reset_mask = np.zeros(adapter.reported_dim, dtype=np.uint8)
    if adapter.is_full_block and not adapter.carry_stream:
        overwrite_mask[adapter.carrier_dim :] = 1
    for _ in range(int(steps)):
        pre = adapter.unpack(reported)
        blank = adapter.zero_input(reported)
        next_reported = adapter.reported_step(reported, blank)
        post = adapter.unpack(next_reported)
        next_primary = adapter.primary_from_reported(next_reported)
        decoded = adapter.decode(next_reported)
        projection = _project_ring(next_primary, projector)
        distance = projection["distance"]
        index = projection["original_index"]
        _require_finite("20-step autonomous trace", blank, next_reported, decoded, distance)
        pre_carrier.append(pre.carrier.detach().cpu().numpy())
        post_carrier.append(post.carrier.detach().cpu().numpy())
        pre_stream.append(
            np.empty((1, 0), dtype=np.float32)
            if pre.stream is None else pre.stream.detach().cpu().numpy()
        )
        post_stream.append(
            np.empty((1, 0), dtype=np.float32)
            if post.stream is None else post.stream.detach().cpu().numpy()
        )
        inputs.append(blank.detach().cpu().numpy())
        if adapter.is_full_block and str(getattr(adapter.model, "decode_mode", "")) == "stream":
            assert post.stream is not None
            decoder_inputs.append(post.stream.detach().cpu().numpy())
        else:
            decoder_inputs.append(next_reported.detach().cpu().numpy())
        decoder_outputs.append(decoded.detach().cpu().numpy())
        residuals.append((next_primary - primary).detach().cpu().numpy())
        nearest_distance.append(float(distance.cpu()))
        nearest_index.append(int(index.cpu()))
        primary, reported = next_primary, next_reported
    arrays = {
        "step": np.arange(1, int(steps) + 1, dtype=np.int64),
        "actual_input": np.concatenate(inputs, axis=0),
        "pre_carrier": np.concatenate(pre_carrier, axis=0),
        "post_carrier": np.concatenate(post_carrier, axis=0),
        "pre_stream": np.concatenate(pre_stream, axis=0),
        "post_stream": np.concatenate(post_stream, axis=0),
        "overwrite_mask": np.broadcast_to(overwrite_mask, (int(steps), adapter.reported_dim)).copy(),
        "reset_mask": np.broadcast_to(reset_mask, (int(steps), adapter.reported_dim)).copy(),
        "post_reported_state": np.concatenate(
            [np.concatenate((carrier, stream), axis=1) for carrier, stream in zip(post_carrier, post_stream)],
            axis=0,
        ),
        "decoder_input": np.concatenate(decoder_inputs, axis=0),
        "decoder_output": np.concatenate(decoder_outputs, axis=0),
        "F0_residual": np.concatenate(residuals, axis=0),
        "nearest_manifold_distance": np.asarray(nearest_distance, dtype=np.float64),
        "nearest_manifold_distance_over_Rs": np.asarray(nearest_distance, dtype=np.float64) / float(radius / 0.1),
        "nearest_manifold_index": np.asarray(nearest_index, dtype=np.int64),
    }
    summary = {
        "schema_version": 1,
        "start": "primary_atlas_anchor_0_plus_0.1_Rs_radial",
        "steps": int(steps),
        "actual_external_input": "literal_zero_velocity_tensor",
        "all_inputs_exact_zero": bool(np.count_nonzero(arrays["actual_input"]) == 0),
        "state_spec": adapter.state_spec(),
        "overwrite_mask_definition": "one marks reported coordinates overwritten before next-step feedback",
        "reset_mask_definition": "one would mark externally reset coordinates; all false in this autonomous trace",
        "decoder_input": (
            "post-transition stream/readout slice"
            if adapter.is_full_block and str(getattr(adapter.model, "decode_mode", "")) == "stream"
            else "complete post-transition reported recurrent state"
        ),
        "F0_residual": "post_primary_minus_pre_primary",
        "nearest_manifold": projector["method"],
    }
    return summary, arrays


def _summary(value: torch.Tensor | np.ndarray) -> dict[str, float]:
    return distribution_summary(value)


def _gate(
    gate_id: str,
    *,
    metric: str,
    threshold: Any,
    value: Any,
    passed: bool | None,
    status: str,
    source_artifact: str,
    reason: str | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "gate_id": gate_id,
        "metric": metric,
        "threshold": threshold,
        "value": value,
        "passed": passed,
        "status": status,
        "source_artifact": source_artifact,
    }
    if reason is not None:
        payload["reason"] = reason
    if scope is not None:
        payload["scope"] = scope
    return payload


def _exact_axis_gate(preflight_audit: Mapping[str, Any]) -> dict[str, Any]:
    screen = preflight_audit["checks"]["architecture_exactness_screen"]
    ruled_out = bool(screen.get("exact_continuum_ruled_out", False))
    return _gate(
        "exact_axis",
        metric="all_point_fixedness_tangent_neutrality_and_stable_normal_bundle",
        threshold="exact_CA_definition",
        value={
            "architecture_exactness_screen": {
                "completed": screen.get("completed"),
                "structural_classification": screen.get("structural_classification"),
                "unique_affine_fixed_point": screen.get("unique_affine_fixed_point"),
                "exact_continuum_ruled_out": ruled_out,
            }
        },
        passed=False if ruled_out else None,
        status="ruled_out" if ruled_out else "not_evaluated",
        source_artifact="checkpoint_state_transition_audit.json",
        reason=(
            "actual-map affine screen found a unique fixed point, ruling out a nonzero exact continuum"
            if ruled_out
            else "finite sampled Phase-1 analysis cannot prove exact all-point conditions"
        ),
    )


def _claim_gate(
    protocol: Mapping[str, Any],
    plan: AnalysisPlan,
    task_result: Mapping[str, Any],
    paper_noise_result: Mapping[str, Any],
    decoder_summary: Mapping[str, float],
    rank_summary: Mapping[str, Any],
    neighborhood: Mapping[str, Any],
    chi_fiber: float,
    track_a_success: bool,
    settling_valid: bool,
    settling_summary: Mapping[str, Any],
    primary_atlas_valid: bool,
    projection_qa: Mapping[str, Any],
    invariance_summary: Mapping[str, float],
    drift_summary: Mapping[str, float],
    clean_adherence: Mapping[str, Any],
    radial_recovery: Mapping[str, float],
    ambient_recovery: Mapping[str, float],
    recovery_input_valid: bool,
    same_memory: Mapping[str, float],
    tangent_equivariance: Mapping[str, float],
    jvp: Mapping[str, Any],
    normal_direction_qa: Mapping[str, Any],
    preflight_audit: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = protocol["claim_gates"]
    smoke_reason = "smoke settings do not evaluate the frozen primary horizon/count" if plan.smoke else None

    def evaluated_status(
        passed: bool,
        *,
        eligible: bool = True,
        ineligible_reason: str = "primary atlas reconstruction is inconclusive",
    ) -> tuple[str, bool | None, str | None]:
        if plan.smoke:
            return "not_evaluated", None, smoke_reason
        if not eligible:
            return "inconclusive", None, ineligible_reason
        return ("passed" if passed else "failed"), passed, None

    task_pass = float(task_result["masked_nmse_db"]) < float(thresholds["task"]["threshold"])
    decode_pass = (
        float(decoder_summary["mean"]) <= float(thresholds["c1_decoding"]["mean_max"])
        and float(decoder_summary["q95"]) <= float(thresholds["c1_decoding"]["q95_max"])
    )
    rank_pass = bool(rank_summary["passed"])
    neighborhood_pass = (
        float(neighborhood["trustworthiness"])
        >= float(thresholds["c1_neighborhood"]["minimum_each"])
        and float(neighborhood["continuity"])
        >= float(thresholds["c1_neighborhood"]["minimum_each"])
    )
    fiber_pass = math.isfinite(float(chi_fiber)) and float(chi_fiber) <= float(
        thresholds["c1_sheet_fiber"]["maximum"]
    )
    invariance_pass = float(invariance_summary["q95"]) <= float(thresholds["invariance"]["q95_max"])
    drift_pass = (
        float(drift_summary["mean"]) <= float(thresholds["c2_drift"]["mean_max"])
        and float(drift_summary["q95"]) <= float(thresholds["c2_drift"]["q95_max"])
    )
    clean_adherence_pass = bool(
        clean_adherence["all_registered_horizons_passed"]
    )
    radial_pass = (
        float(radial_recovery["median"]) <= plan.recovery_median_max
        and float(radial_recovery["q95"]) < plan.recovery_q95_strict_max
    )
    ambient_pass = (
        float(ambient_recovery["median"]) <= plan.recovery_median_max
        and float(ambient_recovery["q95"]) < plan.recovery_q95_strict_max
    )
    same_pass = (
        float(same_memory["mean"]) <= float(thresholds["c3_same_memory"]["mean_max"])
        and float(same_memory["q95"]) <= float(thresholds["c3_same_memory"]["q95_max"])
    )
    tangent_eq_pass = (
        float(tangent_equivariance["mean"]) <= float(thresholds["tangent_equivariance"]["mean_max"])
        and float(tangent_equivariance["q95"]) <= float(thresholds["tangent_equivariance"]["q95_max"])
    )
    tangent_gain_q95 = float(jvp["tangent_gain"]["q95"])
    tangent_pass = tangent_gain_q95 <= float(thresholds["tangent_non_expansion"]["q95_max"])
    paper_noise_pass = float(paper_noise_result["masked_nmse_db"]) < float(
        thresholds["c3_paper_noise"]["threshold"]
    )
    sampled_gap_pass = float(jvp["sampled_joint_fraction"]) >= float(
        thresholds["sampled_normal_gap"]["required_joint_atlas_fraction"]
    )
    projection_valid = bool(projection_qa["passed"])
    normal_direction_valid = bool(normal_direction_qa["passed"])
    cocycle_frames_valid = bool(jvp["all_clean_projector_frames_valid"])
    projected_metrics_eligible = bool(primary_atlas_valid and projection_valid)
    projected_normal_metrics_eligible = bool(
        projected_metrics_eligible and normal_direction_valid
    )
    recovery_metrics_eligible = bool(
        projected_normal_metrics_eligible and recovery_input_valid
    )
    projected_cocycle_eligible = bool(
        projected_normal_metrics_eligible and cocycle_frames_valid
    )

    computed = {
        "task": task_pass,
        "c1_decoding": decode_pass,
        "c1_rank": rank_pass,
        "c1_neighborhood": neighborhood_pass,
        "c1_sheet_fiber": fiber_pass,
        "invariance": invariance_pass,
        "c2_drift": drift_pass,
        "c3_clean_adherence": clean_adherence_pass,
        "c3_normal_recovery": (
            clean_adherence_pass and radial_pass and ambient_pass
        ),
        "c3_same_memory": same_pass,
        "c3_paper_noise": paper_noise_pass,
        "tangent_equivariance": tangent_eq_pass,
        "tangent_non_expansion": tangent_pass,
        "sampled_normal_gap": sampled_gap_pass,
    }
    gates: dict[str, dict[str, Any]] = {}
    status, passed, reason = evaluated_status(task_pass)
    gates["task"] = _gate(
        "task", metric="masked_target_power_nmse_db",
        threshold={"operator": "less_than", "value": thresholds["task"]["threshold"]},
        value=float(task_result["masked_nmse_db"]), passed=passed, status=status,
        source_artifact="analysis.json", reason=reason,
    )
    status, passed, reason = evaluated_status(
        decode_pass, eligible=primary_atlas_valid,
        ineligible_reason="primary Track-A reconstruction failed",
    )
    gates["c1_decoding"] = _gate(
        "c1_decoding", metric="held_out_linear_decoder_pi_normalized_geodesic",
        threshold={"mean_max": thresholds["c1_decoding"]["mean_max"], "q95_max": thresholds["c1_decoding"]["q95_max"]},
        value=dict(decoder_summary), passed=passed, status=status,
        source_artifact="analysis.json", reason=reason,
    )
    status, passed, reason = evaluated_status(
        rank_pass, eligible=primary_atlas_valid,
        ineligible_reason="primary Track-A reconstruction failed",
    )
    gates["c1_rank"] = _gate(
        "c1_rank", metric="normalized_sigma_d_over_sigma_1_at_each_anchor",
        threshold={
            "minimum_normalized_sigma_d_over_sigma_1": thresholds["c1_rank"]["minimum"],
            "minimum_qualifying_atlas_fraction": thresholds["c1_rank"]["required_atlas_fraction"],
        },
        value=dict(rank_summary), passed=passed, status=status,
        source_artifact="analysis_arrays.npz", reason=reason,
    )
    status, passed, reason = evaluated_status(track_a_success)
    gates["atlas_reconstruction"] = _gate(
        "atlas_reconstruction",
        metric="Sagodi_Track_A_slow_state_periodic_reconstruction",
        threshold={"relative_speed": SLOW_RELATIVE_SPEED, "rollout": "16T"},
        value={"track_a_success": track_a_success},
        passed=passed, status=status, source_artifact="analysis_arrays.npz", reason=reason,
    )
    status, passed, reason = evaluated_status(
        settling_valid,
        eligible=primary_atlas_valid,
        ineligible_reason="task-sheet settling requires a valid primary Track-A manifold",
    )
    gates["task_sheet_settling"] = _gate(
        "task_sheet_settling",
        metric="two_sided_variance_plateau_expansion_and_F0_power5_manifold_adherence",
        threshold={
            "absolute_symmetric_relative_change_q95_max": plan.settle_absolute_change_q95_max,
            "positive_expansion_q95_max": plan.settle_positive_expansion_q95_max,
            "mean_sheet_distance_over_Rs_q95_max": plan.settle_mean_sheet_distance_q95_max,
        },
        value=dict(settling_summary),
        passed=passed,
        status=status,
        source_artifact="analysis.json",
        reason=reason,
        scope="correspondence_and_fiber_QA_not_primary_manifold_construction",
    )
    status, passed, reason = evaluated_status(
        neighborhood_pass, eligible=primary_atlas_valid,
        ineligible_reason="neighborhood is diagnostic only because primary atlas is invalid",
    )
    gates["c1_neighborhood"] = _gate(
        "c1_neighborhood", metric="trustworthiness_and_continuity", threshold={"minimum_each": 0.95},
        value=dict(neighborhood), passed=passed, status=status,
        source_artifact="analysis.json", reason=reason,
    )
    status, passed, reason = evaluated_status(
        fiber_pass,
        eligible=bool(primary_atlas_valid and settling_valid),
        ineligible_reason="eight-path task sheet did not pass corrected settling QA",
    )
    gates["c1_sheet_fiber"] = _gate(
        "c1_sheet_fiber", metric="chi_fiber", threshold={"maximum": 0.1},
        value=float(chi_fiber) if math.isfinite(float(chi_fiber)) else None,
        passed=passed, status=status, source_artifact="analysis_arrays.npz", reason=reason,
    )
    projection_status, projection_passed, projection_reason = evaluated_status(
        projection_valid,
        eligible=primary_atlas_valid,
        ineligible_reason="projection QA is inapplicable because primary atlas is invalid",
    )
    gates["projection_quality"] = _gate(
        "projection_quality", metric="known_q_and_synthetic_projection_error_q95",
        threshold={"q95_max": projection_qa["q95_max"]}, value=dict(projection_qa),
        passed=projection_passed, status=projection_status,
        source_artifact="analysis_arrays.npz", reason=projection_reason,
    )
    normal_status, normal_passed, normal_reason = evaluated_status(
        normal_direction_valid,
        eligible=primary_atlas_valid,
        ineligible_reason="normal-direction QA is inapplicable because primary atlas is invalid",
    )
    gates["normal_direction_quality"] = _gate(
        "normal_direction_quality",
        metric="sampled_tangent_normal_absolute_inner_product_max",
        threshold={"maximum": normal_direction_qa["threshold_max"]},
        value=dict(normal_direction_qa),
        passed=normal_passed,
        status=normal_status,
        source_artifact="analysis_arrays.npz",
        reason=normal_reason,
    )
    status, passed, reason = evaluated_status(
        invariance_pass, eligible=projected_metrics_eligible,
        ineligible_reason="primary atlas or projection QA is invalid",
    )
    gates["invariance"] = _gate(
        "invariance", metric="nearest_dense_atlas_r_inv", threshold={"q95_max": thresholds["invariance"]["q95_max"]},
        value=dict(invariance_summary), passed=passed, status=status,
        source_artifact="analysis_arrays.npz", reason=reason,
    )
    status, passed, reason = evaluated_status(
        drift_pass, eligible=projected_metrics_eligible,
        ineligible_reason="primary atlas or projection QA is invalid",
    )
    gates["c2_drift"] = _gate(
        "c2_drift", metric="pi_normalized_nearest_atlas_drift",
        threshold={"horizon": 1024, "mean_max": thresholds["c2_drift"]["mean_max"], "q95_max": thresholds["c2_drift"]["q95_max"]},
        value={"effective_horizon": plan.drift_horizon, **dict(drift_summary)}, passed=passed, status=status,
        source_artifact="analysis_arrays.npz", reason=reason,
    )
    status, passed, reason = evaluated_status(
        clean_adherence_pass,
        eligible=projected_metrics_eligible,
        ineligible_reason="primary Track-A manifold or projection QA is invalid",
    )
    gates["c3_clean_adherence"] = _gate(
        "c3_clean_adherence",
        metric="D_clean_H_distance_to_fixed_Track_A_over_Rs",
        threshold={
            "registered_horizons": list(plan.recovery_horizons),
            "q95_max_at_every_horizon": plan.clean_adherence_q95_max,
        },
        value=dict(clean_adherence),
        passed=passed,
        status=status,
        source_artifact="analysis_arrays.npz",
        reason=reason,
    )
    status, passed, reason = evaluated_status(
        clean_adherence_pass and radial_pass and ambient_pass,
        eligible=recovery_metrics_eligible,
        ineligible_reason=(
            "primary Track-A, projection/normal QA, initial manifold-distance "
            "denominator, or clean projector frame is invalid"
        ),
    )
    gates["c3_normal_recovery"] = _gate(
        "c3_normal_recovery", metric="Q_recovery_distance_to_fixed_Track_A_by_direction_family",
        threshold={
            "rho_over_Rs": plan.kick_relative_radius,
            "horizon": plan.recovery_horizon,
            "median_max": plan.recovery_median_max,
            "q95_less_than": plan.recovery_q95_strict_max,
            "clean_adherence_required": True,
            "families_must_pass_separately": True,
        },
        value={
            "effective_horizon": plan.recovery_horizon,
            "clean_adherence_passed": clean_adherence_pass,
            "denominator_and_clean_frame_valid": recovery_input_valid,
            "radial": dict(radial_recovery),
            "ambient_per_anchor_sampled_max": dict(ambient_recovery),
        },
        passed=passed, status=status, source_artifact="analysis_arrays.npz", reason=reason,
        scope="radial_plus_eight_sampled_ambient_normals" if not plan.smoke else "smoke_sample",
    )
    status, passed, reason = evaluated_status(
        same_pass, eligible=recovery_metrics_eligible,
        ineligible_reason="recovery projection or initial-distance denominator is invalid",
    )
    gates["c3_same_memory"] = _gate(
        "c3_same_memory", metric="projected_memory_displacement_after_manifold_recovery",
        threshold={"horizon": 500, "mean_max": 0.05, "q95_max": 0.1},
        value={"effective_horizon": plan.recovery_horizon, **dict(same_memory)}, passed=passed, status=status,
        source_artifact="analysis_arrays.npz", reason=reason,
    )
    status, passed, reason = evaluated_status(paper_noise_pass)
    gates["c3_paper_noise"] = _gate(
        "c3_paper_noise", metric="masked_nmse_db_under_coordinate_std_0p1", threshold={"less_than": -20.0},
        value=dict(paper_noise_result), passed=passed, status=status,
        source_artifact="analysis.json", reason=reason,
    )
    status, passed, reason = evaluated_status(
        tangent_eq_pass, eligible=projected_metrics_eligible,
        ineligible_reason="primary atlas or projection QA is invalid",
    )
    gates["tangent_equivariance"] = _gate(
        "tangent_equivariance", metric="additive_tangent_vs_true_shift_E_T",
        threshold={"shift_radians": 0.01, "mean_max": 0.05, "q95_max": 0.1},
        value={"effective_horizon": plan.recovery_horizon, **dict(tangent_equivariance)}, passed=passed, status=status,
        source_artifact="analysis_arrays.npz", reason=reason,
    )
    status, passed, reason = evaluated_status(
        tangent_pass,
        eligible=bool(projected_metrics_eligible and cocycle_frames_valid),
        ineligible_reason="primary atlas, projection QA, or clean cocycle frame is invalid",
    )
    gates["tangent_non_expansion"] = _gate(
        "tangent_non_expansion", metric="endpoint_projected_ring_tangent_block_H50_gain_q95",
        threshold={"horizon": 50, "q95_max": 1.05},
        value={"effective_horizon": plan.jacobian_horizon, **dict(jvp["tangent_gain"])}, passed=passed, status=status,
        source_artifact="analysis_arrays.npz", reason=reason,
        scope="T(q_H)^T_J_0_to_H_T(q_0)_without_intermediate_P_T",
    )
    status, passed, reason = evaluated_status(
        sampled_gap_pass, eligible=projected_cocycle_eligible,
        ineligible_reason="primary atlas, projection QA, normal-direction QA, or clean cocycle frame is invalid",
    )
    gates["sampled_normal_gap"] = _gate(
        "sampled_normal_gap", metric="sampled_per_step_projected_normal_cocycle_gamma_H50", threshold=thresholds["sampled_normal_gap"],
        value={
            "effective_horizon": plan.jacobian_horizon,
            "sampled_normal_gain_max": dict(jvp["sampled_normal_gain_max"]),
            "sampled_joint_fraction": float(jvp["sampled_joint_fraction"]),
        },
        passed=passed, status=status, source_artifact="analysis_arrays.npz",
        reason=reason,
        scope="frozen_sampled_per_step_P_N_normal_cocycle_gate_not_exact_worst_normal",
    )
    gates["strict_worst_normal"] = _gate(
        "strict_worst_normal",
        metric="one_step_and_H50_worst_normal_singular_gain",
        threshold="strict_projected_normal_operator_norm",
        value=None,
        passed=None,
        status="not_evaluated",
        source_artifact="",
        reason=(
            "this module samples radial and ambient directions but does not "
            "optimize or materialize the strict worst normal singular vector"
        ),
        scope="unavailable_not_inferred_from_random_samples",
    )
    gates["c4"] = _gate(
        "c4", metric=thresholds["c4"]["metric"], threshold={"minimum": thresholds["c4"]["minimum"]},
        value=None, passed=None, status="not_evaluated", source_artifact="",
        reason=thresholds["c4"]["reason_not_evaluated"],
    )
    gates["model_seeds"] = _gate(
        "model_seeds", metric="main_model_seed_L3_pass_count", threshold={"minimum": 8, "total": 10},
        value=None, passed=None, status="not_evaluated", source_artifact="",
        reason=thresholds["model_seeds"]["reason_not_evaluated"],
    )
    gates["exact_axis"] = _exact_axis_gate(preflight_audit)

    l0 = bool(not plan.smoke and computed["task"])
    l1 = bool(
        l0 and primary_atlas_valid and settling_valid and projection_valid
        and computed["c1_decoding"]
        and computed["c1_rank"] and computed["c1_neighborhood"]
        and computed["c1_sheet_fiber"] and computed["c2_drift"]
    )
    l2 = bool(
        l1 and normal_direction_valid and recovery_input_valid and cocycle_frames_valid
        and computed["invariance"] and computed["c3_normal_recovery"]
        and computed["c3_same_memory"] and computed["c3_paper_noise"]
        and computed["tangent_equivariance"] and computed["tangent_non_expansion"]
        and computed["sampled_normal_gap"]
    )
    l3 = False
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "phase": "phase1_ring_pilot",
        "pilot_only": True,
        "smoke": plan.smoke,
        "gates": gates,
        "levels": {
            "L0": {"passed": l0, "reason": "task gate" if not plan.smoke else "smoke is non-claim"},
            "L1": {"passed": l1, "reason": "L0 plus primary-atlas C1 and C2 gates"},
            "L2": {"passed": l2, "reason": "L1 plus invariance, C3, tangent, and sampled normal-gap gates"},
            "L3": {"passed": l3, "reason": "C4 and 10-main-seed gates are not evaluated in the pilot"},
            "exact": {
                "passed": False,
                "reason": (
                    "ruled out by checkpoint architecture exactness screen"
                    if gates["exact_axis"]["status"] == "ruled_out"
                    else "exact axis not evaluated"
                ),
            },
        },
        "approximate_ca_claim_allowed": False,
        "interpretation": "pilot_evidence_only",
    }


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _task_failure_claim(
    protocol: Mapping[str, Any],
    plan: AnalysisPlan,
    task_result: Mapping[str, Any],
    preflight_audit: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = protocol["claim_gates"]
    task_value = float(task_result["masked_nmse_db"])
    gates: dict[str, dict[str, Any]] = {
        "task": _gate(
            "task",
            metric="masked_target_power_nmse_db",
            threshold={"operator": "less_than", "value": thresholds["task"]["threshold"]},
            value=task_value,
            passed=False,
            status="failed",
            source_artifact="analysis.json",
            reason="seed remains in the all-started-seed denominator",
        )
    }
    manifold_gate_ids = (
        "atlas_reconstruction", "task_sheet_settling", "c1_decoding", "c1_rank", "c1_neighborhood",
        "c1_sheet_fiber", "projection_quality", "normal_direction_quality",
        "invariance", "c2_drift", "c3_clean_adherence", "c3_normal_recovery",
        "c3_same_memory", "c3_paper_noise", "tangent_equivariance",
        "tangent_non_expansion", "sampled_normal_gap", "strict_worst_normal",
    )
    for gate_id in manifold_gate_ids:
        gates[gate_id] = _gate(
            gate_id,
            metric="not_computed_after_task_inclusion_failure",
            threshold=thresholds.get(gate_id),
            value=None,
            passed=False,
            status="not_applicable_task_failure",
            source_artifact="analysis.json",
            reason="Section 3.2 excludes task-failure seeds from manifold fitting without deleting them",
        )
    gates["c4"] = _gate(
        "c4", metric=thresholds["c4"]["metric"],
        threshold={"minimum": thresholds["c4"]["minimum"]}, value=None,
        passed=None, status="not_evaluated", source_artifact="",
        reason=thresholds["c4"]["reason_not_evaluated"],
    )
    gates["model_seeds"] = _gate(
        "model_seeds", metric="main_model_seed_L3_pass_count",
        threshold={"minimum": 8, "total": 10}, value=None,
        passed=None, status="not_evaluated", source_artifact="",
        reason=thresholds["model_seeds"]["reason_not_evaluated"],
    )
    gates["exact_axis"] = _exact_axis_gate(preflight_audit)
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "phase": "phase1_ring_pilot",
        "pilot_only": True,
        "smoke": plan.smoke,
        "inclusion": "task_failure_manifold_fitting_not_applicable",
        "gates": gates,
        "levels": {
            "L0": {"passed": False, "reason": "task gate failed"},
            "L1": {"passed": False, "reason": "not applicable after task gate failure"},
            "L2": {"passed": False, "reason": "not applicable after task gate failure"},
            "L3": {"passed": False, "reason": "not applicable after task gate failure"},
            "exact": {
                "passed": False,
                "reason": "ruled out by architecture screen" if gates["exact_axis"]["status"] == "ruled_out" else "not evaluated",
            },
        },
        "approximate_ca_claim_allowed": False,
        "interpretation": "task_failure_retained_in_all_seed_denominator",
    }


def analyze_checkpoint(
    *,
    protocol_path: Path | str,
    run_dir: Path | str,
    output_dir: Path | str,
    device: torch.device | str = "cuda:0",
    smoke: bool = False,
    evaluation_bank: Path | str | None = None,
    perturbation_bank: Path | str | None = None,
    campaign_identity: str | None = None,
    analysis_freeze_path: Path | str | None = None,
) -> Path:
    """Analyze ``run_dir/checkpoint.pt`` and atomically publish Phase-1 output."""

    protocol_path = Path(protocol_path).expanduser().resolve(strict=True)
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    checkpoint = (run_dir / "checkpoint.pt").resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    analysis_contract: dict[str, Any] | None = None
    analysis_freeze_source: Path | None = None
    if analysis_freeze_path is not None:
        if bool(smoke):
            raise ValueError("parent-bound v2 reanalysis freeze cannot be used for smoke")
        analysis_freeze_source = (
            Path(analysis_freeze_path).expanduser().resolve(strict=True)
        )
        analysis_contract = load_analysis_freeze(analysis_freeze_source)

    protocol = load_protocol(protocol_path)
    if (
        not bool(smoke)
        and analysis_contract is None
        and protocol["freeze_id"] in LEGACY_V1_ANALYSIS_FREEZE_IDS
    ):
        raise ValueError(
            "legacy v1 full analysis requires an explicit corrected v2 analysis freeze"
        )
    parent_root: Path | None = None
    parent_binding: dict[str, Any] | None = None
    parent_manifest: dict[str, Any] | None = None
    if analysis_contract is not None:
        parent_root, parent_binding, parent_manifest = _verify_analysis_freeze_parent(
            analysis_contract=analysis_contract,
            protocol=protocol,
            protocol_path=protocol_path,
            run_dir=run_dir,
            campaign_identity=campaign_identity,
        )
        _verify_parent_bound_bank(
            parent_root=parent_root,
            parent_manifest=parent_manifest,
            bank_key="evaluation_bank",
            supplied_path=evaluation_bank,
        )
        _verify_parent_bound_bank(
            parent_root=parent_root,
            parent_manifest=parent_manifest,
            bank_key="perturbation_bank",
            supplied_path=perturbation_bank,
        )
        try:
            destination.relative_to(parent_root)
        except ValueError:
            pass
        else:
            raise ValueError(
                "v2 reanalysis output must be outside the immutable parent campaign root"
            )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty analysis directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    analysis_binding_path: Path | None = None
    analysis_binding: dict[str, Any] | None = None
    if analysis_contract is not None:
        assert analysis_freeze_source is not None
        analysis_binding = {
            "schema_version": 1,
            "analysis_freeze_id": analysis_contract["freeze_id"],
            "analysis_freeze_path": str(analysis_freeze_source),
            "analysis_freeze_file_sha256": sha256_file(analysis_freeze_source),
            "analysis_freeze_canonical_fingerprint": analysis_freeze_fingerprint(
                analysis_contract
            ),
            "parent_training": parent_binding,
            "claim_scope": analysis_contract["claim_scope"],
        }
        analysis_binding_path = destination / "analysis_freeze_binding.json"
        atomic_json(analysis_binding_path, analysis_binding)

    plan = _build_plan(protocol, bool(smoke), analysis_contract)
    task_spec = AngularTaskSpec.from_protocol(protocol)
    target_device = torch.device(device)
    model, checkpoint_payload = load_checkpoint(checkpoint, target_device)
    if model.config.init_mode != "hidden_init":
        raise ValueError("Phase-1 ring atlas requires a hidden_init checkpoint")
    if model.input_dim != 1 or model.output_dim != 2 or model.config.initial_memory_dim != 2:
        raise ValueError("Phase-1 checkpoint dimensions do not match the frozen ring task")
    model.eval()
    adapter = StateAdapter(model.core)
    dtype = _model_dtype(model)
    seed_policy = protocol["seed_policy"]

    evaluation, evaluation_source = _evaluation_batch(
        path=evaluation_bank,
        plan=plan,
        protocol=protocol,
        task_spec=task_spec,
        device=target_device, dtype=dtype,
    )
    _verify_checkpoint_identity(
        protocol=protocol,
        protocol_path=protocol_path,
        run_dir=run_dir,
        checkpoint=checkpoint,
        checkpoint_payload=checkpoint_payload,
        model=model,
        evaluation_source=evaluation_source,
        campaign_identity=campaign_identity,
        smoke=plan.smoke,
    )
    finite_raw, jacobian_raw, perturbation_source = _load_perturbation_bank(
        perturbation_bank,
        plan=plan,
        state_dimension=adapter.primary_dim,
        device=target_device,
        dtype=dtype,
    )
    if parent_root is not None:
        assert parent_manifest is not None
        # Recheck after both banks have been parsed.  This closes replacement
        # races during loading while the analysis itself uses the tensors
        # already held in memory.
        _verify_parent_bound_bank(
            parent_root=parent_root,
            parent_manifest=parent_manifest,
            bank_key="evaluation_bank",
            supplied_path=evaluation_bank,
        )
        _verify_parent_bound_bank(
            parent_root=parent_root,
            parent_manifest=parent_manifest,
            bank_key="perturbation_bank",
            supplied_path=perturbation_bank,
        )
    with torch.no_grad():
        prediction = model.forward_sequence(
            evaluation["inputs"], initial_memory=evaluation["initial_memory"]
        )
    _require_finite("clean task prediction", prediction)
    task_result = task_metrics(
        prediction, evaluation["targets"], evaluation["mask"], evaluation["latents"]
    )
    if not all(math.isfinite(float(value)) for value in task_result.values() if isinstance(value, (int, float))):
        raise RuntimeError("non-finite clean task metric")

    task_evoked = _task_evoked_primary_state(model, adapter, evaluation)
    preflight_audit = run_phase0_audit(
        adapter,
        state=task_evoked,
        config=AuditConfig(
            batch_size=int(task_evoked.shape[0]),
            seed=derived_seed(20260713, "checkpoint_preflight", checkpoint.name),
            random_directions=2 if plan.smoke else 16,
            architecture_max_dimension=512,
        ),
    )
    preflight_path = destination / "checkpoint_state_transition_audit.json"
    atomic_json(preflight_path, preflight_audit)
    if not bool(preflight_audit["passed"]):
        raise RuntimeError(
            "checkpoint task-evoked Phase-0 state-transition audit failed; claim artifacts were not written"
        )

    task_pass = float(task_result["masked_nmse_db"]) < float(
        protocol["claim_gates"]["task"]["threshold"]
    )
    base_analysis: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis": "sagodi_phase1_ring_single_checkpoint",
        "pilot_only": True,
        "smoke": bool(smoke),
        "campaign_identity": campaign_identity,
        "protocol_freeze_id": protocol["freeze_id"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "analysis_freeze_binding": analysis_binding,
        "resolved_task_spec": task_spec.resolved_payload(),
        "resolved_task_spec_sha256": task_spec.fingerprint(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_extra": checkpoint_payload.get("extra", {}),
        "model": model.metadata(),
        "state_spec": adapter.state_spec(),
        "primary_analysis_state": "carrier_only_minimal_markov_state",
        "plan": plan.as_dict(),
        "evaluation_bank": evaluation_source,
        "perturbation_bank": perturbation_source,
        "task_metrics": task_result,
        "task_inclusion": {
            "threshold_nmse_db_less_than": float(protocol["claim_gates"]["task"]["threshold"]),
            "passed": task_pass,
            "manifold_fitting_eligible": bool(task_pass or plan.smoke),
            "smoke_bypass": bool(plan.smoke and not task_pass),
        },
        "checkpoint_preflight": {
            "passed": True,
            "artifact": preflight_path.name,
            "probe_source": "trained_checkpoint_task_evoked_primary_state",
        },
    }

    def receipt_artifacts(*paths: Path) -> list[Path]:
        artifacts = list(paths)
        if analysis_binding_path is not None:
            artifacts.append(analysis_binding_path)
        return artifacts

    def receipt_analysis_metadata() -> dict[str, Any]:
        if analysis_binding is None:
            return {
                "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
                "analysis_freeze_id": None,
                "analysis_freeze_canonical_fingerprint": None,
            }
        return {
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "analysis_freeze_id": analysis_binding["analysis_freeze_id"],
            "analysis_freeze_canonical_fingerprint": analysis_binding[
                "analysis_freeze_canonical_fingerprint"
            ],
        }

    if not task_pass and not plan.smoke:
        analysis = {
            **base_analysis,
            "manifold_analysis": "not_applicable_task_failure",
            "limitations": [
                "Section 3.2 task-success inclusion failed; manifold fitting was not run",
                "the started seed remains in the all-seed success-rate denominator",
            ],
        }
        claim = _task_failure_claim(protocol, plan, task_result, preflight_audit)
        analysis_path = destination / "analysis.json"
        arrays_path = destination / "analysis_arrays.npz"
        claim_path = destination / "claim_gate.json"
        atomic_json(analysis_path, analysis)
        _atomic_npz(
            arrays_path,
            {
                "task_prediction": prediction.detach().cpu().numpy(),
                "task_targets": evaluation["targets"].detach().cpu().numpy(),
                "task_mask": evaluation["mask"].detach().cpu().numpy(),
            },
        )
        atomic_json(claim_path, claim)
        write_completion_receipt(
            destination / "completion_receipt.json",
            job_id=f"phase1-analysis-{checkpoint.parent.name}",
            artifacts=receipt_artifacts(
                preflight_path, analysis_path, arrays_path, claim_path
            ),
            metadata={
                "pilot_only": True, "smoke": False,
                "campaign_identity": campaign_identity,
                "checkpoint_sha256": base_analysis["checkpoint_sha256"],
                "task_gate_passed": False, "manifold_fitting": "not_applicable",
                "evaluation_bank_sha256": evaluation_source["sha256"],
                "perturbation_bank_sha256": perturbation_source["sha256"],
                "resolved_task_spec_sha256": task_spec.fingerprint(),
                **receipt_analysis_metadata(),
                "L3": False,
            },
        )
        return destination

    noise_seed = derived_seed(
        int(seed_policy["perturbation_bank_seed"]), "paper_coordinate_noise_std_0p1"
    )
    noise_generator = torch.Generator(device=target_device).manual_seed(noise_seed)
    with torch.no_grad():
        paper_noise_prediction = model.forward_sequence(
            evaluation["inputs"],
            initial_memory=evaluation["initial_memory"],
            state_noise_std=0.1,
            noise_generator=noise_generator,
        )
    _require_finite("paper coordinate-noise task prediction", paper_noise_prediction)
    paper_noise_result = task_metrics(
        paper_noise_prediction,
        evaluation["targets"], evaluation["mask"], evaluation["latents"],
    )

    atlas_angles = -math.pi + 2.0 * math.pi * torch.arange(
        plan.atlas_count, device=target_device, dtype=dtype
    ) / float(plan.atlas_count)
    discovery = _task_driven_track_a_endpoints(model, evaluation, atlas_angles)
    slow = _slow_state_reconstruction(
        model,
        adapter,
        atlas_angles,
        horizon=plan.slow_rollout_horizon,
        initial_reported_states=discovery["reported_state"],
        start_state_rule="stratified_fixed_evaluation_task_driven_endpoints",
        coarse_coverage_bin_count=plan.track_a_coverage_bin_count,
        minimum_coarse_occupancy_fraction=plan.track_a_coverage_min_fraction,
    )
    primary_atlas_valid = bool(slow["success"])
    atlas_state = slow["resampled_state"]
    if not bool(slow["geometry_available"]):
        task_atlas = _task_conditioned_atlas(
            model, adapter, atlas_angles, plan
        )
        zero_summary = {
            "mean": 0.0,
            "median": 0.0,
            "q95": 0.0,
            "q99": 0.0,
            "max": 0.0,
        }
        projection_qa = {
            "passed": False,
            "q95_max": float(
                protocol["qa_thresholds"]["projection"]["implied_q95_error_max"]
            ),
            "status": "not_evaluated_track_a_reconstruction_failed",
        }
        claim = _claim_gate(
            protocol=protocol,
            plan=plan,
            task_result=task_result,
            paper_noise_result=paper_noise_result,
            decoder_summary=zero_summary,
            rank_summary={
                "passed": False,
                "qualifying_atlas_fraction": 0.0,
                "minimum_normalized_sigma_d_over_sigma_1": float(
                    protocol["claim_gates"]["c1_rank"]["minimum"]
                ),
                "required_atlas_fraction": float(
                    protocol["claim_gates"]["c1_rank"]["required_atlas_fraction"]
                ),
            },
            neighborhood={"trustworthiness": 0.0, "continuity": 0.0},
            chi_fiber=float(task_atlas["chi_fiber"]),
            track_a_success=False,
            settling_valid=False,
            settling_summary=task_atlas["criteria"],
            primary_atlas_valid=False,
            projection_qa=projection_qa,
            invariance_summary=zero_summary,
            drift_summary=zero_summary,
            clean_adherence={
                "all_registered_horizons_passed": False,
                "registered_horizons": list(plan.recovery_horizons),
                "by_horizon": {},
            },
            radial_recovery=zero_summary,
            ambient_recovery=zero_summary,
            recovery_input_valid=False,
            same_memory=zero_summary,
            tangent_equivariance=zero_summary,
            jvp={
                "tangent_gain": {"q95": 0.0},
                "sampled_normal_gain_max": zero_summary,
                "sampled_joint_fraction": 0.0,
                "all_clean_projector_frames_valid": False,
            },
            normal_direction_qa={"passed": False, "threshold_max": 0.0},
            preflight_audit=preflight_audit,
        )
        analysis = {
            **base_analysis,
            "paper_coordinate_noise_std_0p1_task_metrics": paper_noise_result,
            "paper_coordinate_noise_seed": noise_seed,
            "perturbation_bank": perturbation_source,
            "manifold_analysis": "inconclusive_track_a_reconstruction_failed",
            "atlas": {
                "construction": "none_no_primary_fallback",
                "anchor_count": plan.atlas_count,
                "primary_eligible": False,
                "primary_source": "track_a_periodic_spline",
                "Sagodi_Track_A": {
                    "success": False,
                    "failure_reasons": slow["failure_reasons"],
                    "start_state_rule": slow["start_state_rule"],
                    "rollout_horizon": slow["rollout_horizon"],
                    "trajectory_count": slow["trajectory_count"],
                    "candidate_threshold": slow["candidate_threshold"],
                    "candidate_count": slow["candidate_count"],
                    "candidate_trajectory_coverage": slow[
                        "candidate_trajectory_coverage"
                    ],
                    "coverage": slow["coverage"],
                    "interpolation": slow["interpolation"],
                },
                "task_conditioned": {
                    "role": "correspondence_and_fiber_diagnostic_only",
                    "valid": task_atlas["valid"],
                    "selected_horizon": task_atlas["selected_horizon"],
                    "chi_fiber": (
                        task_atlas["chi_fiber"]
                        if math.isfinite(float(task_atlas["chi_fiber"]))
                        else None
                    ),
                },
            },
            "limitations": [
                "Track-A periodic reconstruction failed; no task-atlas fallback was used",
                "all projection-dependent gates are inconclusive",
            ],
        }

        def failed_array(value: torch.Tensor) -> np.ndarray:
            return value.detach().cpu().numpy()

        analysis_path = destination / "analysis.json"
        arrays_path = destination / "analysis_arrays.npz"
        claim_path = destination / "claim_gate.json"
        atomic_json(analysis_path, analysis)
        _atomic_npz(
            arrays_path,
            {
                "atlas_angles": failed_array(atlas_angles),
                "atlas_primary_carrier": failed_array(slow["resampled_state"]),
                "atlas_primary_eligible": np.asarray(False, dtype=np.uint8),
                "track_a_resampled_primary_carrier": failed_array(
                    slow["resampled_state"]
                ),
                "track_a_selected_primary_carrier": failed_array(
                    slow["selected_state"]
                ),
                "track_a_selected_time": np.asarray(
                    slow["selected_time"], dtype=np.int64
                ),
                "track_a_selected_trajectory": np.asarray(
                    slow["selected_trajectory"], dtype=np.int64
                ),
                "track_a_selected_decoded_angle": np.asarray(
                    slow["selected_decoded_angle"]
                ),
                "track_a_selected_target_error_radians": np.asarray(
                    slow["selected_target_error_radians"]
                ),
                "track_a_discovery_source_indices": failed_array(
                    discovery["source_indices"]
                ),
                "track_a_discovery_source_final_angles": failed_array(
                    discovery["source_final_angles"]
                ),
                "track_a_discovery_source_selection_error": failed_array(
                    discovery["source_selection_error"]
                ),
                "track_a_discovery_endpoint_reported_state": failed_array(
                    discovery["reported_state"]
                ),
                "task_path_endpoint_primary_carrier": failed_array(
                    task_atlas["path_endpoint_state"]
                ),
                "task_selected_path_primary_carrier": failed_array(
                    task_atlas["selected_path_state"]
                ),
            },
        )
        atomic_json(claim_path, claim)
        write_completion_receipt(
            destination / "completion_receipt.json",
            job_id=f"phase1-analysis-{checkpoint.parent.name}",
            artifacts=receipt_artifacts(
                preflight_path, analysis_path, arrays_path, claim_path
            ),
            metadata={
                "pilot_only": True,
                "smoke": bool(smoke),
                "checkpoint_sha256": base_analysis["checkpoint_sha256"],
                "campaign_identity": campaign_identity,
                "evaluation_bank_sha256": evaluation_source["sha256"],
                "perturbation_bank_sha256": perturbation_source["sha256"],
                "resolved_task_spec_sha256": task_spec.fingerprint(),
                **receipt_analysis_metadata(),
                "task_gate_passed": task_pass,
                "primary_atlas_eligible": False,
                "projection_quality_passed": False,
                "strict_worst_normal_evaluated": False,
                "L3": False,
            },
        )
        return destination
    rs, center = _state_scale(atlas_state)
    tangent_sensitivity = _tangent_finite_difference_sensitivity(
        atlas_angles,
        atlas_state,
        protocol["qa_thresholds"]["tangent_finite_difference_radians"],
    )
    derivative, tangent, tangent_speed, curvature = tangent_sensitivity["geometries"][
        float(plan.tangent_fd_epsilon)
    ]
    track_a_geometry_quality = _track_a_geometry_quality(
        atlas_angles,
        atlas_state,
        tangent_speed,
        rs,
        minimum_normalized_tangent_speed=(
            plan.track_a_normalized_tangent_speed_min
        ),
        seam_probe_radians=plan.track_a_seam_probe_radians,
        seam_c0_over_rs_max=plan.track_a_seam_c0_over_rs_max,
        seam_c1_relative_max=plan.track_a_seam_c1_relative_max,
    )
    primary_atlas_valid = bool(
        slow["success"] and track_a_geometry_quality["passed"]
    )
    track_a_failure_reasons = list(slow["failure_reasons"])
    if not bool(track_a_geometry_quality["passed"]):
        track_a_failure_reasons.append("track_a_geometry_quality_failed")
    rank_result = _ring_local_rank_metrics(
        derivative,
        rs,
        minimum_ratio=float(protocol["claim_gates"]["c1_rank"]["minimum"]),
        required_fraction=float(
            protocol["claim_gates"]["c1_rank"]["required_atlas_fraction"]
        ),
    )
    rank_summary = rank_result["summary"]
    decoder_weight, decoder_bias, decoder_train, decoder_heldout, decoder_summary = (
        _fit_heldout_linear_decoder(atlas_state, atlas_angles)
    )
    neighborhood = _neighborhood_metrics(atlas_state, atlas_angles)
    projector = _build_ring_projector(
        atlas_angles, atlas_state, density=plan.projection_density
    )
    task_atlas = _task_conditioned_atlas(
        model,
        adapter,
        atlas_angles,
        plan,
        primary_projector=projector,
        primary_state_scale=rs,
    )
    task_correspondence, task_correspondence_arrays = (
        _task_track_a_correspondence(
            task_atlas,
            track_a_state=atlas_state,
            track_a_angles=atlas_angles,
            projector=projector,
            state_scale=rs,
        )
    )
    heldout_count = min(256, plan.atlas_count)
    heldout_cell_indices = _stratified_indices(
        plan.atlas_count, heldout_count, target_device
    )
    heldout_angles = wrap_angle(
        atlas_angles[heldout_cell_indices] + math.pi / float(plan.atlas_count)
    )
    selected_settling_horizon = task_atlas["selected_horizon"]
    heldout_task_bank: dict[str, Any] | None = None
    if selected_settling_horizon is not None:
        heldout_task_bank = _heldout_task_conditioned_states(
            model,
            adapter,
            heldout_angles,
            path_steps=plan.path_steps,
            settling_horizon=int(selected_settling_horizon),
        )
    projection_qa, projection_arrays = _projection_quality_audit(
        projector, atlas_state, tangent, center, rs,
        heldout_angles=heldout_angles,
        heldout_task_mean_state=(
            None if heldout_task_bank is None else heldout_task_bank["mean_state"]
        ),
        selected_settling_horizon=selected_settling_horizon,
        q95_max=float(protocol["qa_thresholds"]["projection"]["implied_q95_error_max"]),
    )

    with torch.no_grad():
        f0_state = adapter.actual_f0(atlas_state)
        fixedness = torch.linalg.vector_norm(f0_state - atlas_state, dim=-1) / rs
        f0_projection = _project_ring(f0_state, projector)
        invariance_index = f0_projection["original_index"]
        invariance = f0_projection["distance"] / rs
        drift_state = _roll_f0(adapter, atlas_state, plan.drift_horizon)
        drift_projection = _project_ring(drift_state, projector)
        drift_index = drift_projection["original_index"]
        drift_error = wrap_angle(drift_projection["angle"] - atlas_angles).abs() / math.pi

    kick_indices = _stratified_indices(
        plan.atlas_count, plan.kick_anchor_count, target_device
    )
    perturb_seed = derived_seed(
        int(seed_policy["perturbation_bank_seed"]),
        "phase1_ring_radial_ambient", plan.kick_anchor_count, plan.ambient_directions,
    )
    radial, ambient = _normal_directions(
        atlas_state[kick_indices], tangent[kick_indices], curvature[kick_indices],
        center, plan.ambient_directions, perturb_seed, raw_ambient=finite_raw,
    )
    kick_normal_orthogonality = _normal_direction_orthogonality(
        tangent[kick_indices], radial, ambient
    )
    kick_directions = torch.cat((radial[:, None, :], ambient), dim=1)
    with torch.no_grad():
        manifold_recovery = manifold_recovery_diagnostics(
            atlas_state[kick_indices],
            kick_directions,
            radius=plan.kick_relative_radius * rs,
            state_scale=rs,
            horizons=plan.recovery_horizons,
            actual_f0=adapter.actual_f0,
            project=lambda value: _project_ring(value, projector),
        )
        kick = _finite_kicks(
            adapter, atlas_state, atlas_angles, tangent, kick_indices,
            radial, ambient, plan.kick_relative_radius * rs, plan.recovery_horizon,
            projector,
        )
        tangent_error = _tangent_equivariance(
            adapter, atlas_state, atlas_angles, kick_indices, derivative,
            plan.tangent_shift, plan.recovery_horizon, projector,
        )

    jacobian_indices = _stratified_indices(
        plan.atlas_count, plan.jacobian_anchor_count, target_device
    )
    jacobian_radial, jacobian_ambient = _normal_directions(
        atlas_state[jacobian_indices], tangent[jacobian_indices], curvature[jacobian_indices],
        center, plan.ambient_directions,
        derived_seed(int(seed_policy["perturbation_bank_seed"]), "phase1_sampled_jvp"),
        raw_ambient=jacobian_raw,
    )
    jacobian_normal_orthogonality = _normal_direction_orthogonality(
        tangent[jacobian_indices], jacobian_radial, jacobian_ambient
    )
    normal_direction_qa = _normal_direction_qa_summary(
        kick_normal_orthogonality,
        jacobian_normal_orthogonality,
        maximum=float(
            protocol["qa_thresholds"]["tangent_normal_orthogonality_error_max"]
        ),
    )
    perturbation_source["realized_projected_direction_sha256"] = hashlib.sha256(
        (
            _tensor_sha256(torch.cat((radial[:, None], ambient), dim=1))
            + _tensor_sha256(torch.cat((jacobian_radial[:, None], jacobian_ambient), dim=1))
        ).encode("ascii")
    ).hexdigest()
    sampled_jvp = _sampled_jvp_gains(
        adapter, atlas_state, tangent, jacobian_indices,
        jacobian_radial, jacobian_ambient, plan.jacobian_horizon, projector,
    )

    trace_summary, trace_arrays = _autonomous_radial_trace(
        adapter, atlas_state=atlas_state, atlas_tangent=tangent,
        center=center, radius=plan.kick_relative_radius * rs,
        projector=projector, steps=20,
    )

    registered_horizons = [
        int(value) for value in manifold_recovery["horizons"].cpu().tolist()
    ]
    primary_recovery_index = registered_horizons.index(plan.recovery_horizon)
    recovery_q_primary = manifold_recovery["manifold_recovery_Q"][
        primary_recovery_index
    ]
    radial_recovery = _summary(recovery_q_primary[:, 0])
    ambient_per_anchor_max = recovery_q_primary[:, 1:].max(dim=1).values
    ambient_recovery = _summary(ambient_per_anchor_max)
    same_memory_per_anchor_max = manifold_recovery["same_memory_E_excess"][
        primary_recovery_index
    ].max(dim=1).values
    same_memory = _summary(same_memory_per_anchor_max)
    clean_adherence_by_horizon = {
        str(horizon): _summary(
            manifold_recovery["clean_manifold_distance_over_Rs"][index]
        )
        for index, horizon in enumerate(registered_horizons)
    }
    clean_adherence_q95_max_observed = max(
        float(summary["q95"])
        for summary in clean_adherence_by_horizon.values()
    )
    clean_adherence = {
        "metric": "D_clean(H)=d(F0^H(m),M)/Rs",
        "registered_horizons": registered_horizons,
        "by_horizon": clean_adherence_by_horizon,
        "q95_max_over_registered_horizons": clean_adherence_q95_max_observed,
        "q95_threshold": plan.clean_adherence_q95_max,
        "all_registered_horizons_passed": bool(
            clean_adherence_q95_max_observed <= plan.clean_adherence_q95_max
        ),
    }
    recovery_input_valid = bool(
        manifold_recovery["denominator_valid"].all().cpu()
        and manifold_recovery[
            "clean_projection_tangent_frame_valid"
        ].all().cpu()
    )
    initial_distance_summary = {
        "radial": _summary(
            manifold_recovery["initial_manifold_distance_over_Rs"][:, 0]
        ),
        "ambient_per_anchor_sampled_max": _summary(
            manifold_recovery["initial_manifold_distance_over_Rs"][:, 1:]
            .max(dim=1)
            .values
        ),
    }
    perturbed_distance_primary = manifold_recovery[
        "perturbed_manifold_distance_over_Rs"
    ][primary_recovery_index]
    perturbed_distance_summary = {
        "radial": _summary(perturbed_distance_primary[:, 0]),
        "ambient_per_anchor_sampled_max": _summary(
            perturbed_distance_primary[:, 1:].max(dim=1).values
        ),
    }
    legacy_paired_primary = manifold_recovery[
        "paired_endpoint_normal_deviation_over_radius"
    ][primary_recovery_index]
    legacy_paired_summary = {
        "radial": _summary(legacy_paired_primary[:, 0]),
        "ambient_per_anchor_sampled_max": _summary(
            legacy_paired_primary[:, 1:].max(dim=1).values
        ),
    }
    tangent_equivariance = _summary(tangent_error)
    invariance_summary = _summary(invariance)
    fixedness_summary = _summary(fixedness)
    drift_summary = _summary(drift_error)
    sampled_joint = (
        (sampled_jvp["primary_ell_n_sampled_max"] < 0)
        & (sampled_jvp["primary_gamma_sampled"] > 0)
    )
    jvp_summary = {
        "label": "primary_sampled_per_step_projected_normal_cocycle",
        "primary": True,
        "definition": "P_N(q_t) is applied after every JVP step along the actual clean F0 rollout",
        "tangent_definition": "T(q_H)^T J_{0:H} T(q_0), with no intermediate P_T factors",
        "clean_frame_source": "ring_projector_local_tangent_frame_at_each_actual_clean_state",
        "normal_projection_schedule": "every_step_including_initial_normalization",
        "horizon": plan.jacobian_horizon,
        "tangent_gain": _summary(sampled_jvp["primary_tangent_gain"]),
        "radial_normal_gain": _summary(sampled_jvp["primary_radial_normal_gain"]),
        "ambient_normal_gain": _summary(sampled_jvp["primary_ambient_normal_gain"]),
        "sampled_normal_gain_max": _summary(sampled_jvp["primary_sampled_normal_gain_max"]),
        "ell_t": _summary(sampled_jvp["primary_ell_t"]),
        "ell_n_sampled_max": _summary(sampled_jvp["primary_ell_n_sampled_max"]),
        "gamma_sampled": _summary(sampled_jvp["primary_gamma_sampled"]),
        "diagnostic_per_step_projected_tangent_gain": _summary(
            sampled_jvp["diagnostic_per_step_projected_tangent_gain"]
        ),
        "sampled_joint_fraction": float(sampled_joint.float().mean().cpu()),
        "all_clean_projector_frames_valid": bool(
            sampled_jvp["clean_projected_frame_valid_trace"].all().cpu()
        ),
        "endpoint_only_J_product_proxy": {
            "label": "non_primary_endpoint_only_projected_J_product_proxy",
            "primary": False,
            "definition": "unprojected J_H...J_1 directions split by P_T/P_N only at the endpoint",
            "tangent_gain": _summary(sampled_jvp["endpoint_proxy_tangent_gain"]),
            "radial_normal_gain": _summary(sampled_jvp["endpoint_proxy_radial_normal_gain"]),
            "ambient_normal_gain": _summary(sampled_jvp["endpoint_proxy_ambient_normal_gain"]),
            "sampled_normal_gain_max": _summary(sampled_jvp["endpoint_proxy_sampled_normal_gain_max"]),
            "ell_t": _summary(sampled_jvp["endpoint_proxy_ell_t"]),
            "ell_n_sampled_max": _summary(sampled_jvp["endpoint_proxy_ell_n_sampled_max"]),
            "gamma_sampled": _summary(sampled_jvp["endpoint_proxy_gamma_sampled"]),
        },
        "strict_worst_normal_evaluated": False,
    }

    analysis = {
        **base_analysis,
        "paper_coordinate_noise_std_0p1_task_metrics": paper_noise_result,
        "paper_coordinate_noise_seed": noise_seed,
        "perturbation_bank": perturbation_source,
        "atlas": {
            "construction": "Sagodi_Track_A_task_endpoint_slow_state_periodic_spline",
            "primary_source": "track_a_resampled_state",
            "anchor_count": plan.atlas_count,
            "state_scale_Rs": float(rs.cpu()),
            "primary_eligible": primary_atlas_valid,
            "eligibility_requires": [
                "Sagodi_Track_A_reconstruction_success",
                "full_ring_candidate_coverage_QA",
                "all_registered_coarse_coverage_bins_occupied",
                "normalized_tangent_speed_floor",
                "periodic_spline_seam_C0_C1_continuity",
            ],
            "heldout_linear_decoder": dict(decoder_summary),
            "decoder_train_anchors": int(decoder_train.size),
            "decoder_heldout_anchors": int(decoder_heldout.size),
            "tangent_speed": _summary(tangent_speed),
            "tangent_finite_difference_sensitivity": {
                "epsilons_radians": list(tangent_sensitivity["epsilons"]),
                "comparisons": tangent_sensitivity["comparisons"],
                "status": tangent_sensitivity["status"],
            },
            "local_rank": rank_summary,
            "rank_fraction": rank_summary["qualifying_atlas_fraction"],
            "neighborhood": neighborhood,
            "projection_quality": projection_qa,
            "normal_direction_quality": normal_direction_qa,
            "chi_fiber": task_atlas["chi_fiber"] if math.isfinite(task_atlas["chi_fiber"]) else None,
            "Sagodi_Track_A": {
                "success": primary_atlas_valid,
                "slow_state_reconstruction_success": slow["success"],
                "failure_reasons": track_a_failure_reasons,
                "start_state_rule": slow["start_state_rule"],
                "rollout_horizon": slow["rollout_horizon"],
                "trajectory_count": slow["trajectory_count"],
                "candidate_threshold": slow["candidate_threshold"],
                "candidate_count": slow["candidate_count"],
                "candidate_trajectory_coverage": slow["candidate_trajectory_coverage"],
                "coverage": slow["coverage"],
                "geometry_quality": track_a_geometry_quality,
                "interpolation": slow["interpolation"],
                "task_endpoint_discovery": {
                    "selection_rule": discovery["selection_rule"],
                    "source_trial_count": discovery["source_trial_count"],
                    "selected_trial_count": discovery["selected_trial_count"],
                    "task_rollout_horizon": discovery["task_rollout_horizon"],
                    "state_noise_standard_deviation": discovery[
                        "state_noise_standard_deviation"
                    ],
                    "source_selection_error_radians": _summary(
                        discovery["source_selection_error"]
                    ),
                },
            },
            "task_conditioned": {
                "role": "independent_correspondence_projection_and_fiber_diagnostic_only",
                "canonical": "state immediately after hidden initialization for q",
                "path_start_angle": 0.0,
                "path_count": plan.path_count,
                "path_variants": "distributed, early, late, middle, shaped, and zero-net excursion profiles",
                "settle_horizons": list(plan.settle_horizons),
                "selected_horizon": task_atlas["selected_horizon"],
                "valid": task_atlas["valid"],
                "classification": task_atlas["classification"],
                "selected_state_scale_valid": task_atlas["selected_state_scale_valid"],
                "settling_criteria": task_atlas["criteria"],
                "chi_fiber": task_atlas["chi_fiber"] if math.isfinite(task_atlas["chi_fiber"]) else None,
                "near_single_sheet": task_atlas["near_single_sheet"],
                "correspondence_to_primary_track_A": task_correspondence,
            },
        },
        "blank_flow": {
            "fixedness_r_fp": fixedness_summary,
            "invariance_r_inv_nearest_dense_atlas": invariance_summary,
            "drift": {"horizon": plan.drift_horizon, **drift_summary},
        },
        "finite_kicks": {
            "primary_metric": "distance_to_fixed_Track_A_manifold",
            "clean_paired": True,
            "carrier_only": True,
            "rho_over_Rs": plan.kick_relative_radius,
            "rho": float((plan.kick_relative_radius * rs).cpu()),
            "registered_horizons": registered_horizons,
            "primary_horizon": plan.recovery_horizon,
            "clean_adherence": clean_adherence,
            "initial_manifold_distance_over_Rs": initial_distance_summary,
            "denominator_and_clean_frame_valid": recovery_input_valid,
            "radial_Q_recovery": radial_recovery,
            "ambient_Q_recovery_per_anchor_sampled_max": ambient_recovery,
            "perturbed_manifold_distance_over_Rs_at_primary_horizon": perturbed_distance_summary,
            "same_memory_E_excess_per_anchor_sampled_max": same_memory,
            "legacy_paired_endpoint_normal_deviation_over_rho": {
                "role": "diagnostic_only_may_not_satisfy_C3",
                **legacy_paired_summary,
            },
            "ambient_direction_label": "sampled_tangent_and_radial_orthogonal_carrier_normal",
            "ambient_directions_per_anchor": plan.ambient_directions,
        },
        "tangent_equivariance": {
            "shift_radians": plan.tangent_shift,
            "horizon": plan.recovery_horizon,
            "error": tangent_equivariance,
        },
        "sampled_jacobian": jvp_summary,
        "autonomous_radial_trace": trace_summary,
        "limitations": [
            "single pilot checkpoint is not an independent seed-level claim",
            "sampled ambient directions are not the strict worst normal singular direction",
            "C4 parameter perturbations are not evaluated",
            "exact all-point fixedness and stable-normal-bundle conditions are not evaluated",
            "the primary normal gap uses sampled per-step projected directions, not an optimized strict-worst operator direction",
            "task-sheet settling is a correspondence/fiber QA and does not replace Track A",
            "all CA conclusions are finite-horizon approximate-CA diagnostics only",
        ],
    }

    claim = _claim_gate(
        protocol=protocol,
        plan=plan,
        task_result=task_result,
        paper_noise_result=paper_noise_result,
        decoder_summary=decoder_summary,
        rank_summary=rank_summary,
        neighborhood=neighborhood,
        chi_fiber=float(task_atlas["chi_fiber"]),
        track_a_success=primary_atlas_valid,
        settling_valid=bool(task_atlas["valid"]),
        settling_summary=task_atlas["criteria"],
        primary_atlas_valid=primary_atlas_valid,
        projection_qa=projection_qa,
        invariance_summary=invariance_summary,
        drift_summary=drift_summary,
        clean_adherence=clean_adherence,
        radial_recovery=radial_recovery,
        ambient_recovery=ambient_recovery,
        recovery_input_valid=recovery_input_valid,
        same_memory=same_memory,
        tangent_equivariance=tangent_equivariance,
        jvp=jvp_summary,
        normal_direction_qa=normal_direction_qa,
        preflight_audit=preflight_audit,
    )

    def array(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()

    kind = np.asarray(
        ["radial"] + ["ambient_sampled"] * plan.ambient_directions,
        dtype="U32",
    )
    heldout_velocity = (
        torch.empty(
            0, PATH_COUNT, plan.path_steps,
            device=target_device, dtype=dtype,
        )
        if heldout_task_bank is None
        else heldout_task_bank["velocity"]
    )
    heldout_path_state = (
        torch.empty(
            0, PATH_COUNT, adapter.primary_dim,
            device=target_device, dtype=dtype,
        )
        if heldout_task_bank is None
        else heldout_task_bank["settled_path_state"]
    )
    settling_quality = task_atlas["quality_by_horizon"]
    if any(value is None for value in settling_quality):
        raise RuntimeError(
            "primary Track-A geometry exists but task-sheet settling was not evaluated"
        )
    settling_tensor_fields = (
        "within_path_variance",
        "next_within_path_variance",
        "absolute_symmetric_relative_variance_change",
        "positive_relative_variance_expansion",
        "within_path_residual_over_Rs",
        "next_within_path_residual_over_Rs",
        "systematic_decrease_indicator",
        "rolled_mean_state",
        "mean_of_rolled_path_states",
        "rolled_mean_manifold_distance_over_Rs",
        "rolled_mean_projected_angle",
    )
    settling_arrays = {
        f"task_settling_{field}": array(
            torch.stack([value[field] for value in settling_quality])
        )
        for field in settling_tensor_fields
    }
    arrays = {
        "atlas_angles": array(atlas_angles),
        "atlas_primary_carrier": array(atlas_state),
        "atlas_primary_eligible": np.asarray(primary_atlas_valid, dtype=np.uint8),
        "projection_dense_angles": array(projector["dense_angles"]),
        "projection_dense_primary_carrier": array(projector["dense_state"]),
        **{f"projection_qa_{key}": array(value) for key, value in projection_arrays.items()},
        "projection_qa_known_q_source_cell_indices": array(heldout_cell_indices),
        "projection_qa_known_q_task_path_velocity": array(heldout_velocity),
        "projection_qa_known_q_task_settled_path_primary_carrier": array(heldout_path_state),
        "track_a_resampled_primary_carrier": array(slow["resampled_state"]),
        "track_a_selected_primary_carrier": array(slow["selected_state"]),
        "track_a_selected_time": np.asarray(slow["selected_time"], dtype=np.int64),
        "track_a_selected_trajectory": np.asarray(slow["selected_trajectory"], dtype=np.int64),
        "track_a_selected_decoded_angle": np.asarray(slow["selected_decoded_angle"]),
        "track_a_selected_target_error_radians": np.asarray(
            slow["selected_target_error_radians"]
        ),
        "track_a_max_speed": np.asarray(slow["max_speed"]),
        "track_a_candidate_count_per_trajectory": np.asarray(slow["candidate_count_per_trajectory"]),
        "track_a_discovery_source_indices": array(discovery["source_indices"]),
        "track_a_discovery_source_final_angles": array(
            discovery["source_final_angles"]
        ),
        "track_a_discovery_source_selection_error": array(
            discovery["source_selection_error"]
        ),
        "track_a_discovery_endpoint_reported_state": array(
            discovery["reported_state"]
        ),
        "task_canonical_primary_carrier": array(task_atlas["canonical_state"]),
        "task_path_velocity": array(task_atlas["path_velocity"]),
        "task_path_endpoint_primary_carrier": array(task_atlas["path_endpoint_state"]),
        "task_settle_horizons": np.asarray(task_atlas["settle_horizons"], dtype=np.int64),
        "task_path_settled_primary_carrier": array(task_atlas["settled_state_by_horizon"]),
        "task_selected_path_primary_carrier": array(task_atlas["selected_path_state"]),
        "task_fiber_within_variance_normalized": array(task_atlas["within_variance_normalized"]),
        "task_between_nearest_separation_normalized": array(task_atlas["between_nearest_separation_normalized"]),
        **settling_arrays,
        **{
            f"task_track_A_correspondence_{key}": array(value)
            for key, value in task_correspondence_arrays.items()
        },
        "atlas_tangent": array(tangent),
        "atlas_tangent_derivative": array(derivative),
        "atlas_tangent_speed": array(tangent_speed),
        "atlas_rank_normalized_sigma_min": array(rank_result["normalized_sigma_min"]),
        "atlas_rank_normalized_sigma_max": array(rank_result["normalized_sigma_max"]),
        "atlas_rank_normalized_sigma_d_over_sigma_1": array(
            rank_result["normalized_sigma_d_over_sigma_1"]
        ),
        "atlas_rank_numerically_nonzero": array(rank_result["numerically_nonzero"]),
        "atlas_rank_qualifying_indicator": array(rank_result["qualifying_indicator"]),
        **{
            f"atlas_tangent_derivative_eps_{str(epsilon).replace('.', 'p')}": array(geometry[0])
            for epsilon, geometry in tangent_sensitivity["geometries"].items()
        },
        **{
            f"atlas_tangent_speed_eps_{str(epsilon).replace('.', 'p')}": array(geometry[2])
            for epsilon, geometry in tangent_sensitivity["geometries"].items()
        },
        "linear_decoder_weight": np.asarray(decoder_weight),
        "linear_decoder_bias": np.asarray(decoder_bias),
        "decoder_train_indices": np.asarray(decoder_train, dtype=np.int64),
        "decoder_heldout_indices": np.asarray(decoder_heldout, dtype=np.int64),
        "fixedness_r_fp": array(fixedness),
        "invariance_r_inv": array(invariance),
        "invariance_nearest_indices": array(invariance_index),
        "drift_error": array(drift_error),
        "drift_nearest_indices": array(drift_index),
        "kick_anchor_indices": array(kick_indices),
        "kick_direction_kind": kind,
        "kick_directions": array(kick["directions"]),
        "kick_radial_tangent_orthogonality_error": array(
            kick_normal_orthogonality["radial"]
        ),
        "kick_ambient_tangent_orthogonality_error": array(
            kick_normal_orthogonality["ambient"]
        ),
        **{
            f"manifold_recovery_{key}": array(value)
            for key, value in manifold_recovery.items()
        },
        "diagnostic_paired_endpoint_normal_deviation_over_rho": array(
            manifold_recovery[
                "paired_endpoint_normal_deviation_over_radius"
            ]
        ),
        "diagnostic_legacy_paired_endpoint_R_N": array(kick["r_n"]),
        "diagnostic_legacy_paired_endpoint_L_T": array(kick["l_t"]),
        "diagnostic_legacy_paired_endpoint_R_same": array(kick["r_same"]),
        "diagnostic_legacy_paired_endpoint_E_excess": array(kick["e_excess"]),
        "tangent_equivariance_E_T": array(tangent_error),
        "jacobian_anchor_indices": array(jacobian_indices),
        "jacobian_radial_tangent_orthogonality_error": array(
            jacobian_normal_orthogonality["radial"]
        ),
        "jacobian_ambient_tangent_orthogonality_error": array(
            jacobian_normal_orthogonality["ambient"]
        ),
        "sampled_primary_endpoint_tangent_gain": array(sampled_jvp["primary_tangent_gain"]),
        "sampled_projected_cocycle_radial_normal_gain": array(sampled_jvp["primary_radial_normal_gain"]),
        "sampled_projected_cocycle_ambient_normal_gain": array(sampled_jvp["primary_ambient_normal_gain"]),
        "sampled_projected_cocycle_normal_gain_max": array(sampled_jvp["primary_sampled_normal_gain_max"]),
        "sampled_projected_cocycle_ell_T": array(sampled_jvp["primary_ell_t"]),
        "sampled_projected_cocycle_ell_N_max": array(sampled_jvp["primary_ell_n_sampled_max"]),
        "sampled_projected_cocycle_gamma": array(sampled_jvp["primary_gamma_sampled"]),
        "sampled_projected_cocycle_clean_angle_trace": array(sampled_jvp["clean_projected_angle_trace"]),
        "sampled_projected_cocycle_clean_tangent_frame_trace": array(sampled_jvp["clean_projected_tangent_frame_trace"]),
        "sampled_projected_cocycle_clean_frame_valid_trace": array(sampled_jvp["clean_projected_frame_valid_trace"]),
        "diagnostic_per_step_projected_tangent_gain": array(sampled_jvp["diagnostic_per_step_projected_tangent_gain"]),
        "diagnostic_per_step_projected_tangent_gain_trace": array(sampled_jvp["primary_tangent_gain_trace"]),
        "sampled_projected_cocycle_normal_gain_trace": array(sampled_jvp["primary_normal_gain_trace"]),
        "endpoint_only_J_product_proxy_tangent_gain": array(sampled_jvp["endpoint_proxy_tangent_gain"]),
        "endpoint_only_J_product_proxy_radial_normal_gain": array(sampled_jvp["endpoint_proxy_radial_normal_gain"]),
        "endpoint_only_J_product_proxy_ambient_normal_gain": array(sampled_jvp["endpoint_proxy_ambient_normal_gain"]),
        "endpoint_only_J_product_proxy_normal_gain_max": array(sampled_jvp["endpoint_proxy_sampled_normal_gain_max"]),
        "endpoint_only_J_product_proxy_ell_T": array(sampled_jvp["endpoint_proxy_ell_t"]),
        "endpoint_only_J_product_proxy_ell_N_max": array(sampled_jvp["endpoint_proxy_ell_n_sampled_max"]),
        "endpoint_only_J_product_proxy_gamma": array(sampled_jvp["endpoint_proxy_gamma_sampled"]),
    }
    analysis_path = destination / "analysis.json"
    arrays_path = destination / "analysis_arrays.npz"
    claim_path = destination / "claim_gate.json"
    trace_path = destination / "autonomous_radial_trace.npz"
    trace_json_path = destination / "autonomous_radial_trace.json"
    atomic_json(analysis_path, analysis)
    _atomic_npz(arrays_path, arrays)
    _atomic_npz(trace_path, trace_arrays)
    atomic_json(trace_json_path, trace_summary)
    atomic_json(claim_path, claim)
    write_completion_receipt(
        destination / "completion_receipt.json",
        job_id=f"phase1-analysis-{checkpoint.parent.name}",
        artifacts=receipt_artifacts(
            preflight_path,
            analysis_path,
            arrays_path,
            trace_path,
            trace_json_path,
            claim_path,
        ),
        metadata={
            "pilot_only": True,
            "smoke": bool(smoke),
            "checkpoint_sha256": analysis["checkpoint_sha256"],
            "campaign_identity": campaign_identity,
            "evaluation_bank_sha256": evaluation_source["sha256"],
            "perturbation_bank_sha256": perturbation_source["sha256"],
            "realized_projected_direction_sha256": perturbation_source["realized_projected_direction_sha256"],
            "resolved_task_spec_sha256": task_spec.fingerprint(),
            **receipt_analysis_metadata(),
            "task_gate_passed": task_pass,
            "primary_atlas_eligible": primary_atlas_valid,
            "projection_quality_passed": bool(projection_qa["passed"]),
            "strict_worst_normal_evaluated": False,
            "L3": False,
        },
    )
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-bank", type=Path)
    parser.add_argument("--perturbation-bank", type=Path)
    parser.add_argument("--campaign-identity")
    parser.add_argument("--analysis-freeze", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = analyze_checkpoint(
        protocol_path=args.protocol,
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        device=args.device,
        smoke=bool(args.smoke),
        evaluation_bank=args.evaluation_bank,
        perturbation_bank=args.perturbation_bank,
        campaign_identity=args.campaign_identity,
        analysis_freeze_path=args.analysis_freeze,
    )
    print(json.dumps({"status": "complete", "output_dir": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AnalysisPlan", "analyze_checkpoint", "main"]
