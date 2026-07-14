"""Pure manifold-distance diagnostics for autonomous recurrent dynamics.

The helpers in this module deliberately know nothing about a concrete model,
atlas implementation, or artifact schema.  Callers supply the autonomous map
and a projection callback.  This keeps the scientific definitions testable in
isolation and prevents clean-paired trajectory contraction from being
mistaken for attraction to a manifold.

The projection callback must return ``distance``, ``angle``, and
``tangent_frame`` tensors for a batch of states.  It may additionally return a
boolean ``tangent_frame_valid`` tensor.  These keys match the ring projector
used by :mod:`repro.sagodi_protocol.phase1_analysis` without importing that
module.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch


AutonomousMap = Callable[[torch.Tensor], torch.Tensor]
Projection = Callable[[torch.Tensor], Mapping[str, torch.Tensor]]


def _require_floating_matrix(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor with shape [N,D]")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _positive_scalar(
    name: str,
    value: torch.Tensor | float,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if result.ndim != 0 or not bool(torch.isfinite(result)) or not bool(result > 0):
        raise ValueError(f"{name} must be a finite positive scalar")
    return result


def _normalized_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in horizons)
    if not values:
        raise ValueError("horizons must be non-empty")
    if any(value < 0 for value in values):
        raise ValueError("horizons must be non-negative")
    if tuple(sorted(set(values))) != values:
        raise ValueError("horizons must be strictly increasing and unique")
    return values


def _apply_f0(
    actual_f0: AutonomousMap,
    state: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    following = actual_f0(state)
    if not isinstance(following, torch.Tensor) or following.shape != state.shape:
        raise ValueError(f"actual_f0 returned an invalid shape for {label}")
    if following.device != state.device or following.dtype != state.dtype:
        raise ValueError(f"actual_f0 changed dtype or device for {label}")
    if not bool(torch.isfinite(following).all()):
        raise ValueError(f"actual_f0 returned non-finite values for {label}")
    return following


def _projection_fields(
    project: Projection,
    state: torch.Tensor,
    *,
    require_tangent: bool,
) -> dict[str, torch.Tensor]:
    raw = project(state)
    if not isinstance(raw, Mapping):
        raise ValueError("projection callback must return a mapping")
    required = {"distance", "angle"}
    if require_tangent:
        required.add("tangent_frame")
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"projection callback is missing keys: {sorted(missing)}")

    count, dimension = state.shape
    distance = raw["distance"]
    angle = raw["angle"]
    if distance.shape != (count,):
        raise ValueError("projection distance must have shape [N]")
    if angle.shape != (count,):
        raise ValueError("ring projection angle must have shape [N]")
    values: dict[str, torch.Tensor] = {"distance": distance, "angle": angle}

    if require_tangent:
        tangent = raw["tangent_frame"]
        if tangent.shape != (count, dimension):
            raise ValueError("projection tangent_frame must have shape [N,D]")
        values["tangent_frame"] = tangent
        valid = raw.get(
            "tangent_frame_valid",
            torch.ones(count, device=state.device, dtype=torch.bool),
        )
        if valid.shape != (count,):
            raise ValueError("projection tangent_frame_valid must have shape [N]")
        values["tangent_frame_valid"] = valid.to(dtype=torch.bool)

    for name, value in values.items():
        if value.device != state.device:
            raise ValueError(f"projection field {name!r} changed device")
        if name != "tangent_frame_valid":
            if not value.is_floating_point() or value.dtype != state.dtype:
                raise ValueError(
                    f"projection field {name!r} must share the state dtype"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f"projection field {name!r} contains non-finite values"
                )
    if bool((distance < 0).any()):
        raise ValueError("projection distances must be non-negative")
    if require_tangent:
        tangent_norm = torch.linalg.vector_norm(values["tangent_frame"], dim=-1)
        valid = values["tangent_frame_valid"]
        tolerance = 1000.0 * torch.finfo(state.dtype).eps
        if bool(valid.any()) and not bool(
            torch.allclose(
                tangent_norm[valid],
                torch.ones_like(tangent_norm[valid]),
                atol=tolerance,
                rtol=tolerance,
            )
        ):
            raise ValueError("valid projection tangent frames must be unit norm")
    return values


def _ring_geodesic(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    delta = torch.atan2(torch.sin(left - right), torch.cos(left - right))
    return delta.abs() / math.pi


@torch.no_grad()
def manifold_recovery_diagnostics(
    base_state: torch.Tensor,
    directions: torch.Tensor,
    *,
    radius: torch.Tensor | float,
    state_scale: torch.Tensor | float,
    horizons: Sequence[int],
    actual_f0: AutonomousMap,
    project: Projection,
    denominator_floor_epsilon_multiplier: float = 100.0,
) -> dict[str, torch.Tensor]:
    """Evaluate clean adherence and finite-kick recovery at many horizons.

    Args:
        base_state: States on the candidate manifold, shape ``[A,D]``.
        directions: Unit kick directions, shape ``[A,C,D]``.  Direction
            families and their aggregation remain the caller's responsibility.
        radius: Nominal finite-kick radius in state-space units.
        state_scale: Frozen manifold scale :math:`R_s`.
        horizons: Strictly increasing autonomous horizons, optionally including
            zero.
        actual_f0: The actual blank-input state transition.
        project: Projection onto the *fixed candidate manifold*.

    Returns:
        Tensor-only fields ready for lossless NPZ storage.  In particular,
        ``manifold_recovery_Q`` uses the actual initial manifold distance as
        its denominator, while ``paired_endpoint_normal_deviation_over_radius``
        preserves the former clean-paired metric under an explicitly
        diagnostic name.
    """

    _require_floating_matrix("base_state", base_state)
    if directions.ndim != 3 or directions.shape[0] != base_state.shape[0]:
        raise ValueError("directions must have shape [A,C,D] aligned with base_state")
    if directions.shape[2] != base_state.shape[1] or directions.shape[1] < 1:
        raise ValueError("directions must contain at least one D-dimensional direction")
    if directions.device != base_state.device or directions.dtype != base_state.dtype:
        raise ValueError("directions must share base_state dtype and device")
    if not bool(torch.isfinite(directions).all()):
        raise ValueError("directions contain non-finite values")
    direction_norm = torch.linalg.vector_norm(directions, dim=-1)
    tolerance = 1000.0 * torch.finfo(base_state.dtype).eps
    if not bool(
        torch.allclose(
            direction_norm,
            torch.ones_like(direction_norm),
            atol=tolerance,
            rtol=tolerance,
        )
    ):
        raise ValueError("directions must be unit norm")

    ordered_horizons = _normalized_horizons(horizons)
    rho = _positive_scalar("radius", radius, reference=base_state)
    rs = _positive_scalar("state_scale", state_scale, reference=base_state)
    multiplier = float(denominator_floor_epsilon_multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("denominator_floor_epsilon_multiplier must be positive")

    anchors, conditions, dimension = directions.shape
    perturbed = base_state[:, None, :] + rho * directions
    perturbed = perturbed.reshape(anchors * conditions, dimension)
    initial_projection = _projection_fields(
        project, perturbed, require_tangent=False
    )
    initial_distance = initial_projection["distance"].reshape(anchors, conditions)
    absolute_floor = rs * (multiplier * torch.finfo(base_state.dtype).eps)
    denominator_valid = initial_distance > absolute_floor
    denominator = initial_distance.clamp_min(absolute_floor)

    clean = base_state
    clean_distance: list[torch.Tensor] = []
    perturbed_distance: list[torch.Tensor] = []
    recovery_q: list[torch.Tensor] = []
    clean_angle: list[torch.Tensor] = []
    perturbed_angle: list[torch.Tensor] = []
    same_memory: list[torch.Tensor] = []
    paired_normal: list[torch.Tensor] = []
    tangent_valid: list[torch.Tensor] = []
    horizon_set = set(ordered_horizons)

    for step in range(ordered_horizons[-1] + 1):
        if step in horizon_set:
            clean_projection = _projection_fields(
                project, clean, require_tangent=True
            )
            perturbed_projection = _projection_fields(
                project, perturbed, require_tangent=False
            )
            distance_h = perturbed_projection["distance"].reshape(
                anchors, conditions
            )
            angle_h = perturbed_projection["angle"].reshape(anchors, conditions)
            clean_angle_h = clean_projection["angle"]

            difference = perturbed.reshape(anchors, conditions, dimension) - clean[:, None, :]
            tangent = clean_projection["tangent_frame"]
            tangent_component = (difference * tangent[:, None, :]).sum(dim=-1)
            normal_component = difference - tangent_component[..., None] * tangent[:, None, :]

            clean_distance.append(clean_projection["distance"] / rs)
            perturbed_distance.append(distance_h / rs)
            recovery_q.append(distance_h / denominator)
            clean_angle.append(clean_angle_h)
            perturbed_angle.append(angle_h)
            same_memory.append(_ring_geodesic(angle_h, clean_angle_h[:, None]))
            paired_normal.append(
                torch.linalg.vector_norm(normal_component, dim=-1) / rho
            )
            tangent_valid.append(clean_projection["tangent_frame_valid"])

        if step != ordered_horizons[-1]:
            clean = _apply_f0(actual_f0, clean, label="clean trajectory")
            perturbed = _apply_f0(
                actual_f0, perturbed, label="perturbed trajectory"
            )

    return {
        "horizons": torch.tensor(
            ordered_horizons, device=base_state.device, dtype=torch.int64
        ),
        "initial_manifold_distance_over_Rs": initial_distance / rs,
        "initial_manifold_distance_over_radius": initial_distance / rho,
        "denominator_floor_over_Rs": absolute_floor / rs,
        "denominator_valid": denominator_valid,
        "clean_manifold_distance_over_Rs": torch.stack(clean_distance),
        "perturbed_manifold_distance_over_Rs": torch.stack(perturbed_distance),
        "manifold_recovery_Q": torch.stack(recovery_q),
        "clean_projected_angle": torch.stack(clean_angle),
        "perturbed_projected_angle": torch.stack(perturbed_angle),
        "same_memory_E_excess": torch.stack(same_memory),
        "paired_endpoint_normal_deviation_over_radius": torch.stack(paired_normal),
        "clean_projection_tangent_frame_valid": torch.stack(tangent_valid),
    }


@torch.no_grad()
def settling_quality_diagnostics(
    path_state: torch.Tensor,
    *,
    state_scale: torch.Tensor | float,
    actual_f0: AutonomousMap,
    project_mean_sheet: Projection,
    blank_steps: int = 5,
    absolute_relative_change_q95_max: float = 0.01,
    positive_expansion_q95_max: float = 0.01,
    mean_sheet_distance_q95_max: float = 0.01,
    denominator_floor_epsilon_multiplier: float = 100.0,
) -> dict[str, Any]:
    """Test whether an eight-path sheet has genuinely settled.

    The variance plateau check is two-sided.  A separate positive-expansion
    statistic makes the failure mode explicit, and the mean state is rolled
    independently before measuring its distance to the current mean sheet.
    This last step is intentionally *not* replaced by the mean of individually
    rolled path states because the autonomous map may be nonlinear.
    """

    if path_state.ndim != 3 or not path_state.is_floating_point():
        raise ValueError("path_state must be a floating tensor with shape [A,P,D]")
    if path_state.shape[0] < 1 or path_state.shape[1] < 2 or path_state.shape[2] < 1:
        raise ValueError(
            "path_state must contain anchors, at least two paths, and state dimensions"
        )
    if not bool(torch.isfinite(path_state).all()):
        raise ValueError("path_state contains non-finite values")
    if int(blank_steps) != blank_steps or int(blank_steps) <= 0:
        raise ValueError("blank_steps must be a positive integer")
    steps = int(blank_steps)
    rs = _positive_scalar("state_scale", state_scale, reference=path_state)

    thresholds = (
        float(absolute_relative_change_q95_max),
        float(positive_expansion_q95_max),
        float(mean_sheet_distance_q95_max),
    )
    if any(not math.isfinite(value) or value < 0 for value in thresholds):
        raise ValueError("settling thresholds must be finite and non-negative")
    multiplier = float(denominator_floor_epsilon_multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("denominator_floor_epsilon_multiplier must be positive")

    anchors, paths, dimension = path_state.shape
    mean_state = path_state.mean(dim=1)
    following_paths = path_state.reshape(anchors * paths, dimension)
    rolled_mean = mean_state
    for _ in range(steps):
        following_paths = _apply_f0(
            actual_f0, following_paths, label="settling path rollout"
        )
        rolled_mean = _apply_f0(
            actual_f0, rolled_mean, label="settling mean-sheet rollout"
        )
    following_paths = following_paths.reshape_as(path_state)
    following_mean = following_paths.mean(dim=1)

    variance = (path_state - mean_state[:, None, :]).square().sum(dim=-1).mean(dim=1)
    next_variance = (
        (following_paths - following_mean[:, None, :]).square().sum(dim=-1).mean(dim=1)
    )
    variance_floor = (rs * multiplier * torch.finfo(path_state.dtype).eps).square()
    symmetric_denominator = torch.maximum(
        torch.maximum(variance, next_variance), variance_floor
    )
    current_denominator = variance.clamp_min(variance_floor)
    absolute_change = (next_variance - variance).abs() / symmetric_denominator
    positive_expansion = (next_variance - variance).clamp_min(0.0) / current_denominator

    mean_projection = _projection_fields(
        project_mean_sheet, rolled_mean, require_tangent=False
    )
    mean_sheet_distance = mean_projection["distance"] / rs
    residual = torch.sqrt(variance.clamp_min(0.0)) / rs
    next_residual = torch.sqrt(next_variance.clamp_min(0.0)) / rs
    systematic_decrease = next_residual < 0.99 * residual

    absolute_q95 = torch.quantile(absolute_change, 0.95)
    expansion_q95 = torch.quantile(positive_expansion, 0.95)
    sheet_q95 = torch.quantile(mean_sheet_distance, 0.95)
    absolute_pass = bool(absolute_q95 <= thresholds[0])
    expansion_pass = bool(expansion_q95 <= thresholds[1])
    sheet_pass = bool(sheet_q95 <= thresholds[2])

    return {
        "blank_steps": steps,
        "within_path_variance": variance,
        "next_within_path_variance": next_variance,
        "absolute_symmetric_relative_variance_change": absolute_change,
        "positive_relative_variance_expansion": positive_expansion,
        "within_path_residual_over_Rs": residual,
        "next_within_path_residual_over_Rs": next_residual,
        "systematic_decrease_indicator": systematic_decrease,
        "systematic_decrease_fraction": float(
            systematic_decrease.float().mean().cpu()
        ),
        "rolled_mean_state": rolled_mean,
        "mean_of_rolled_path_states": following_mean,
        "rolled_mean_manifold_distance_over_Rs": mean_sheet_distance,
        "rolled_mean_projected_angle": mean_projection["angle"],
        "absolute_relative_change_q95": absolute_q95,
        "positive_expansion_q95": expansion_q95,
        "mean_sheet_distance_q95": sheet_q95,
        "absolute_relative_change_passed": absolute_pass,
        "positive_expansion_passed": expansion_pass,
        "mean_sheet_adherence_passed": sheet_pass,
        "passed": bool(absolute_pass and expansion_pass and sheet_pass),
    }


__all__ = [
    "AutonomousMap",
    "Projection",
    "manifold_recovery_diagnostics",
    "settling_quality_diagnostics",
]
