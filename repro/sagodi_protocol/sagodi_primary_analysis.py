r"""Numerical core for the Ságodi-based, explicitly adapted ring analysis.

This module contains only model-independent definitions used by the primary
continuous-attractor analysis.  It intentionally does not contain claim
thresholds, binary gates, settling/recovery experiments, or projected-JVP
proxies.  A campaign runner supplies a flattened full Markov state, its
blank-input transition ``F0``, and the task-output decoder.

The recurrent models in this repository are discrete-time systems.  We
therefore retain both conventions needed for an auditable comparison:

* the map Jacobian, :math:`J_F = \partial F_0 / \partial s`; and
* the vector-field Jacobian, :math:`J_v = J_F-I`, for
  :math:`v(s)=F_0(s)-s`.

The output-projected field is the common one-step displacement
``decode(F0(s)) - decode(s)``.  This equals projection of the state vector
field for a linear decoder and remains well-defined for the nonlinear
decoders used by CA-LRU.  It is a finite-step discrete analogue, not an
infinitesimal decoder push-forward.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch


TensorMap = Callable[[torch.Tensor], torch.Tensor]
Decoder = Callable[[torch.Tensor], torch.Tensor]
TopologyKind = Literal[
    "fixed_points",
    "limit_cycle",
    "stationary_continuum",
    "unidirectional_with_stationary_samples",
]
FlowOrientation = Literal["positive", "negative"]
FixedPointKind = Literal["stable", "saddle"]


class StructuralNotEstimableError(ValueError):
    """A registered numerical/domain condition prevents estimation.

    Ordinary ``ValueError`` remains a caller/programmer contract violation and
    must not be converted into a scientific outcome for a trained seed.
    """


@dataclass(frozen=True)
class OutputProjectedFlow:
    """A blank-input state displacement and its decoded displacement."""

    state: torch.Tensor
    next_state: torch.Tensor
    state_vector_field: torch.Tensor
    output: torch.Tensor
    next_output: torch.Tensor
    projected_vector_field: torch.Tensor
    pointwise_norm: torch.Tensor
    uniform_norm: torch.Tensor


@dataclass(frozen=True)
class FlowReversal:
    """One linearly interpolated zero of the signed angular flow.

    ``left_index`` and ``right_index`` refer to the caller's original sample
    order.  A positive-to-negative reversal is stable along the ring and a
    negative-to-positive reversal is a saddle along the ring.
    """

    angle: float
    kind: FixedPointKind
    left_index: int
    right_index: int
    left_flow: float
    right_flow: float


@dataclass(frozen=True)
class FlowTopology:
    """Cyclic flow-reversal result for a sampled one-dimensional manifold."""

    kind: TopologyKind
    reversals: tuple[FlowReversal, ...]
    orientation: FlowOrientation | None
    zero_tolerance: float
    angle_tolerance: float

    @property
    def stable(self) -> tuple[FlowReversal, ...]:
        return tuple(item for item in self.reversals if item.kind == "stable")

    @property
    def saddles(self) -> tuple[FlowReversal, ...]:
        return tuple(item for item in self.reversals if item.kind == "saddle")

    @property
    def is_limit_cycle(self) -> bool:
        return self.kind == "limit_cycle"


@dataclass(frozen=True)
class DenseJacobianEigenspectrum:
    """Dense full-state Jacobians and their complete complex spectra.

    Eigenvalues in the ``*_sorted`` fields are ordered by descending real
    part.  The reported top-two values and gap use the vector-field spectrum
    ``J_F-I``: ``real_part_gap = largest - second_largest``.
    """

    map_jacobian: torch.Tensor
    vector_field_jacobian: torch.Tensor
    map_eigenvalues: torch.Tensor
    vector_field_eigenvalues: torch.Tensor
    map_eigenvalues_sorted: torch.Tensor
    vector_field_eigenvalues_sorted: torch.Tensor
    largest_real_part: torch.Tensor
    second_largest_real_part: torch.Tensor
    real_part_gap: torch.Tensor


@dataclass(frozen=True)
class AngularMemoryMetrics:
    """Per-memory and across-memory finite-time circular errors."""

    predicted_angle: torch.Tensor
    target_angle: torch.Tensor
    signed_error: torch.Tensor
    absolute_error: torch.Tensor
    minimum_error: torch.Tensor
    mean_error: torch.Tensor
    maximum_error: torch.Tensor
    cumulative_minimum_error: torch.Tensor
    cumulative_mean_error: torch.Tensor
    cumulative_maximum_error: torch.Tensor


@dataclass(frozen=True)
class StableBasinCapacity:
    """Geometric stable-basin widths and entropy on a one-dimensional ring."""

    stable_fixed_point_angle: torch.Tensor
    left_saddle_angle: torch.Tensor
    right_saddle_angle: torch.Tensor
    basin_width_radians: torch.Tensor
    basin_proportions: torch.Tensor
    shannon_entropy_nats: torch.Tensor
    effective_basin_count: torch.Tensor


@dataclass(frozen=True)
class AsymptoticMemoryMetrics:
    """Stable-basin capacity and asymptotic angular error.

    Capacity is defined only when isolated stable fixed points exist.  For a
    limit cycle, basin assignments, fixed-point errors, Shannon entropy, and
    effective basin count are ``None``; observed endpoint errors are retained
    but must not be presented as an infinite-time fixed-point capacity.
    """

    topology: Literal["fixed_points", "limit_cycle"]
    initial_angle: torch.Tensor
    observed_terminal_angle: torch.Tensor
    observed_terminal_absolute_error: torch.Tensor
    observed_terminal_mean_error: torch.Tensor
    observed_terminal_maximum_error: torch.Tensor
    stable_fixed_point_angle: torch.Tensor
    basin_assignment: torch.Tensor | None
    basin_counts: torch.Tensor | None
    basin_proportions: torch.Tensor | None
    shannon_entropy_nats: torch.Tensor | None
    effective_basin_count: torch.Tensor | None
    assigned_stable_angle: torch.Tensor | None
    asymptotic_absolute_error: torch.Tensor | None
    asymptotic_mean_error: torch.Tensor | None
    asymptotic_maximum_error: torch.Tensor | None


def _require_floating_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating torch tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _require_state_matrix(name: str, value: torch.Tensor) -> None:
    _require_floating_tensor(name, value)
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [N,D]")


def _apply_matrix_callback(
    callback: TensorMap,
    state: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    result = callback(state)
    if not isinstance(result, torch.Tensor) or result.shape != state.shape:
        raise ValueError(f"{label} must return a tensor with shape {tuple(state.shape)}")
    if result.device != state.device or result.dtype != state.dtype:
        raise ValueError(f"{label} must preserve state dtype and device")
    if not bool(torch.isfinite(result).all()):
        raise StructuralNotEstimableError(f"{label} returned non-finite values")
    return result


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle to the principal interval ``[-pi, pi]``."""

    _require_floating_tensor("angle", angle)
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def circular_difference(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Signed shortest displacement ``estimate - target`` on the circle."""

    _require_floating_tensor("estimate", estimate)
    _require_floating_tensor("target", target)
    try:
        estimate_b, target_b = torch.broadcast_tensors(estimate, target)
    except RuntimeError as error:
        raise ValueError("estimate and target angles are not broadcast-compatible") from error
    if estimate_b.device != target_b.device or estimate_b.dtype != target_b.dtype:
        raise ValueError("estimate and target must share dtype and device")
    return wrap_angle(estimate_b - target_b)


def circular_absolute_error(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Absolute shortest circular error in radians, in ``[0, pi]``."""

    return circular_difference(estimate, target).abs()


def angle_from_output(output: torch.Tensor) -> torch.Tensor:
    """Decode ``[..., (cos, sin)]`` outputs to angles with ``atan2(sin, cos)``."""

    try:
        _require_floating_tensor("output", output)
    except ValueError as error:
        if (
            isinstance(output, torch.Tensor)
            and output.is_floating_point()
            and output.numel() > 0
            and not bool(torch.isfinite(output).all())
        ):
            raise StructuralNotEstimableError(
                "output contains non-finite values"
            ) from error
        raise
    if output.ndim < 1 or output.shape[-1] != 2:
        raise ValueError("output must have final shape [...,2] ordered as (cos,sin)")
    radius = torch.linalg.vector_norm(output, dim=-1)
    if bool((radius == 0).any()):
        raise StructuralNotEstimableError(
            "angle is undefined for a zero output vector"
        )
    return torch.atan2(output[..., 1], output[..., 0])


def discrete_vector_field(state: torch.Tensor, autonomous_map: TensorMap) -> torch.Tensor:
    """Return the common discrete vector field ``F0(state) - state``."""

    _require_state_matrix("state", state)
    return _apply_matrix_callback(
        autonomous_map, state, label="autonomous_map"
    ) - state


def output_projected_flow(
    state: torch.Tensor,
    autonomous_map: TensorMap,
    decoder: Decoder,
) -> OutputProjectedFlow:
    """Evaluate one-step output displacement and its uniform Euclidean norm.

    The uniform norm is the maximum of the pointwise Euclidean norms over the
    supplied manifold sample.  The decoder may be nonlinear, but must return
    one output vector per state.
    """

    _require_state_matrix("state", state)
    next_state = _apply_matrix_callback(
        autonomous_map, state, label="autonomous_map"
    )
    output = decoder(state)
    next_output = decoder(next_state)
    if (
        not isinstance(output, torch.Tensor)
        or not isinstance(next_output, torch.Tensor)
        or output.ndim != 2
        or next_output.ndim != 2
    ):
        raise ValueError("decoder must return a floating tensor with shape [N,2]")
    if (
        next_output.shape != output.shape
        or output.shape != (state.shape[0], 2)
    ):
        raise ValueError("decoder outputs must align with the state batch as [N,2]")
    for name, value in (("output", output), ("next_output", next_output)):
        if not value.is_floating_point() or value.dtype != state.dtype:
            raise ValueError(f"decoder {name} must preserve state dtype")
        if value.device != state.device:
            raise ValueError(f"decoder {name} changed device")
        if not bool(torch.isfinite(value).all()):
            raise StructuralNotEstimableError(f"decoder {name} is non-finite")
    projected = next_output - output
    norms = torch.linalg.vector_norm(projected, dim=-1)
    return OutputProjectedFlow(
        state=state,
        next_state=next_state,
        state_vector_field=next_state - state,
        output=output,
        next_output=next_output,
        projected_vector_field=projected,
        pointwise_norm=norms,
        uniform_norm=norms.max(),
    )


def signed_angular_flow(
    output: torch.Tensor,
    projected_vector_field: torch.Tensor,
    *,
    minimum_radius: float = 1.0e-12,
) -> torch.Tensor:
    """Return signed angular flow from a two-dimensional projected field.

    For ``output=(x,y)`` and projected field ``(u,v)``, this evaluates
    ``(x*v - y*u)/(x*x + y*y)``.  It is the angular-rate convention whose
    sign determines flow reversal.  ``minimum_radius`` is an explicit
    numerical exclusion radius, not a scientific claim threshold.
    """

    _require_floating_tensor("output", output)
    _require_floating_tensor("projected_vector_field", projected_vector_field)
    if output.shape != projected_vector_field.shape or output.ndim != 2:
        raise ValueError("output and projected_vector_field must share shape [N,2]")
    if output.shape[-1] != 2:
        raise ValueError("signed angular flow requires two-dimensional outputs")
    if output.dtype != projected_vector_field.dtype or output.device != projected_vector_field.device:
        raise ValueError("output and projected_vector_field must share dtype and device")
    threshold = float(minimum_radius)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("minimum_radius must be finite and non-negative")
    radius_squared = output.square().sum(dim=-1)
    if bool((radius_squared <= threshold * threshold).any()):
        raise StructuralNotEstimableError(
            "signed angular flow is undefined inside minimum_radius"
        )
    cross = output[:, 0] * projected_vector_field[:, 1]
    cross = cross - output[:, 1] * projected_vector_field[:, 0]
    return cross / radius_squared


def cyclic_flow_reversal_topology(
    angle: torch.Tensor,
    angular_flow: torch.Tensor,
    *,
    zero_tolerance: float = 0.0,
    angle_tolerance: float = 1.0e-12,
) -> FlowTopology:
    """Find stable/saddle flow reversals on a cyclic one-dimensional sample.

    Samples are wrapped to ``[0,2*pi)`` and sorted.  Values satisfying
    ``abs(flow) <= zero_tolerance`` are treated as numerical zero.  The zero
    samples are compressed, then each neighboring pair of *nonzero* samples
    with opposite signs brackets one root.  Its position is obtained by
    linear interpolation of the two endpoint flow values over the forward
    circular arc, including across the seam.  This gives one root for a
    tolerated zero plateau instead of one root per zero-valued grid point.

    A ``+ -> -`` reversal is stable along the ring; ``- -> +`` is a saddle.
    Strictly nonzero flow of one sign is reported as a limit cycle.  All-zero
    flow is a stationary continuum.  One-signed flow containing numerical
    zeros is kept distinct because flow-reversal sampling alone cannot decide
    whether the zeros are isolated fixed points or numerical contact.
    """

    _require_floating_tensor("angle", angle)
    _require_floating_tensor("angular_flow", angular_flow)
    if angle.ndim != 1 or angular_flow.ndim != 1 or angle.shape != angular_flow.shape:
        raise ValueError("angle and angular_flow must share one-dimensional shape [N]")
    if angle.numel() < 3:
        raise ValueError("at least three cyclic samples are required")
    if angle.dtype != angular_flow.dtype or angle.device != angular_flow.device:
        raise ValueError("angle and angular_flow must share dtype and device")
    zero_tol = float(zero_tolerance)
    angle_tol = float(angle_tolerance)
    if not math.isfinite(zero_tol) or zero_tol < 0:
        raise ValueError("zero_tolerance must be finite and non-negative")
    if not math.isfinite(angle_tol) or angle_tol < 0:
        raise ValueError("angle_tolerance must be finite and non-negative")

    two_pi = 2.0 * math.pi
    wrapped = torch.remainder(angle.detach(), two_pi)
    order = torch.argsort(wrapped)
    sorted_angle = wrapped[order].to(dtype=torch.float64, device="cpu")
    sorted_flow = angular_flow.detach()[order].to(dtype=torch.float64, device="cpu")
    original_index = order.to(device="cpu")

    forward_gaps = torch.diff(
        torch.cat((sorted_angle, sorted_angle[:1] + two_pi))
    )
    if bool((forward_gaps <= angle_tol).any()):
        raise ValueError("cyclic angle samples must be unique within angle_tolerance")

    nonzero = torch.nonzero(sorted_flow.abs() > zero_tol, as_tuple=False).flatten()
    if nonzero.numel() == 0:
        return FlowTopology(
            kind="stationary_continuum",
            reversals=(),
            orientation=None,
            zero_tolerance=zero_tol,
            angle_tolerance=angle_tol,
        )

    signs = torch.sign(sorted_flow[nonzero]).to(dtype=torch.int64)
    reversals: list[FlowReversal] = []
    for position in range(nonzero.numel()):
        left_sorted = int(nonzero[position])
        right_sorted = int(nonzero[(position + 1) % nonzero.numel()])
        left_sign = int(signs[position])
        right_sign = int(signs[(position + 1) % nonzero.numel()])
        if left_sign == right_sign:
            continue

        left_angle = float(sorted_angle[left_sorted])
        right_angle = float(sorted_angle[right_sorted])
        span = (right_angle - left_angle) % two_pi
        left_flow = float(sorted_flow[left_sorted])
        right_flow = float(sorted_flow[right_sorted])
        fraction = abs(left_flow) / (abs(left_flow) + abs(right_flow))
        root = (left_angle + fraction * span) % two_pi
        reversals.append(
            FlowReversal(
                angle=root,
                kind="stable" if left_sign > right_sign else "saddle",
                left_index=int(original_index[left_sorted]),
                right_index=int(original_index[right_sorted]),
                left_flow=left_flow,
                right_flow=right_flow,
            )
        )

    reversals.sort(key=lambda item: item.angle)
    if reversals:
        return FlowTopology(
            kind="fixed_points",
            reversals=tuple(reversals),
            orientation=None,
            zero_tolerance=zero_tol,
            angle_tolerance=angle_tol,
        )

    orientation: FlowOrientation = "positive" if int(signs[0]) > 0 else "negative"
    has_stationary_samples = nonzero.numel() != angle.numel()
    return FlowTopology(
        kind=(
            "unidirectional_with_stationary_samples"
            if has_stationary_samples
            else "limit_cycle"
        ),
        reversals=(),
        orientation=orientation,
        zero_tolerance=zero_tol,
        angle_tolerance=angle_tol,
    )


def _sort_eigenvalues_by_real_part(eigenvalues: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(eigenvalues.real, dim=-1, descending=True)
    return torch.gather(eigenvalues, dim=-1, index=order)


def dense_full_jacobian_eigenspectrum(
    state: torch.Tensor,
    autonomous_map: TensorMap,
    *,
    create_graph: bool = False,
) -> DenseJacobianEigenspectrum:
    """Compute dense full-state ``J_F`` and ``J_F-I`` spectra at every state.

    ``autonomous_map`` accepts and returns a matrix ``[N,D]``.  For composite
    recurrent states (for example LSTM), callers must flatten the *entire*
    Markov state before invoking this function.  No projection, JVP, or
    tangent alignment substitutes for the dense Jacobian here.
    """

    _require_state_matrix("state", state)
    if state.shape[1] < 2:
        raise ValueError("state dimension must be at least two to define a top-two gap")

    jacobians: list[torch.Tensor] = []
    for sample in state:
        point = sample.detach().clone().requires_grad_(True)

        def single_state_map(value: torch.Tensor) -> torch.Tensor:
            result = autonomous_map(value.unsqueeze(0))
            if not isinstance(result, torch.Tensor) or result.shape != (1, state.shape[1]):
                raise ValueError("autonomous_map must preserve flattened state shape [N,D]")
            if result.device != value.device or result.dtype != value.dtype:
                raise ValueError("autonomous_map must preserve state dtype and device")
            return result.squeeze(0)

        jacobian = torch.autograd.functional.jacobian(
            single_state_map,
            point,
            create_graph=create_graph,
            strict=False,
            vectorize=False,
        )
        if jacobian.shape != (state.shape[1], state.shape[1]):
            raise RuntimeError("autograd returned an invalid full-state Jacobian")
        if not bool(torch.isfinite(jacobian).all()):
            raise StructuralNotEstimableError(
                "autonomous_map Jacobian contains non-finite values"
            )
        jacobians.append(jacobian)

    map_jacobian = torch.stack(jacobians, dim=0)
    identity = torch.eye(
        state.shape[1], dtype=state.dtype, device=state.device
    ).expand(state.shape[0], -1, -1)
    vector_field_jacobian = map_jacobian - identity
    map_eigenvalues = torch.linalg.eigvals(map_jacobian)
    vector_field_eigenvalues = torch.linalg.eigvals(vector_field_jacobian)
    if not bool(torch.isfinite(map_eigenvalues).all()):
        raise StructuralNotEstimableError(
            "map-Jacobian eigenspectrum contains non-finite values"
        )
    if not bool(torch.isfinite(vector_field_eigenvalues).all()):
        raise StructuralNotEstimableError(
            "vector-field eigenspectrum contains non-finite values"
        )
    map_sorted = _sort_eigenvalues_by_real_part(map_eigenvalues)
    vector_sorted = _sort_eigenvalues_by_real_part(vector_field_eigenvalues)
    largest = vector_sorted[:, 0].real
    second = vector_sorted[:, 1].real
    return DenseJacobianEigenspectrum(
        map_jacobian=map_jacobian,
        vector_field_jacobian=vector_field_jacobian,
        map_eigenvalues=map_eigenvalues,
        vector_field_eigenvalues=vector_field_eigenvalues,
        map_eigenvalues_sorted=map_sorted,
        vector_field_eigenvalues_sorted=vector_sorted,
        largest_real_part=largest,
        second_largest_real_part=second,
        real_part_gap=largest - second,
    )


def finite_time_angular_memory(
    predicted_angle: torch.Tensor,
    target_angle: torch.Tensor,
) -> AngularMemoryMetrics:
    """Compute Ságodi-style finite-time angular error summaries.

    ``predicted_angle`` has shape ``[memory,time]``.  ``target_angle`` may be
    either one angle per memory (shape ``[memory]``) or a full aligned matrix.
    Cumulative fields are time-prefix averages of the corresponding per-time
    minimum, mean, and maximum errors.
    """

    _require_floating_tensor("predicted_angle", predicted_angle)
    _require_floating_tensor("target_angle", target_angle)
    if predicted_angle.ndim != 2:
        raise ValueError("predicted_angle must have shape [memory,time]")
    if target_angle.dtype != predicted_angle.dtype or target_angle.device != predicted_angle.device:
        raise ValueError("target_angle must share predicted_angle dtype and device")
    if target_angle.shape == (predicted_angle.shape[0],):
        aligned_target = target_angle[:, None].expand_as(predicted_angle)
    elif target_angle.shape == predicted_angle.shape:
        aligned_target = target_angle
    else:
        raise ValueError("target_angle must have shape [memory] or [memory,time]")

    signed = circular_difference(predicted_angle, aligned_target)
    absolute = signed.abs()
    minimum = absolute.min(dim=0).values
    mean = absolute.mean(dim=0)
    maximum = absolute.max(dim=0).values
    denominator = torch.arange(
        1,
        predicted_angle.shape[1] + 1,
        dtype=predicted_angle.dtype,
        device=predicted_angle.device,
    )
    return AngularMemoryMetrics(
        predicted_angle=predicted_angle,
        target_angle=aligned_target,
        signed_error=signed,
        absolute_error=absolute,
        minimum_error=minimum,
        mean_error=mean,
        maximum_error=maximum,
        cumulative_minimum_error=minimum.cumsum(dim=0) / denominator,
        cumulative_mean_error=mean.cumsum(dim=0) / denominator,
        cumulative_maximum_error=maximum.cumsum(dim=0) / denominator,
    )


def finite_time_angular_memory_from_output(
    predicted_output: torch.Tensor,
    target_angle: torch.Tensor,
) -> AngularMemoryMetrics:
    """Decode ``[memory,time,2]`` output vectors and compute memory metrics."""

    return finite_time_angular_memory(angle_from_output(predicted_output), target_angle)


def stable_basin_capacity(
    stable_fixed_point_angle: torch.Tensor,
    saddle_fixed_point_angle: torch.Tensor,
    *,
    angle_tolerance: float = 1.0e-12,
) -> StableBasinCapacity:
    """Compute basin arc fractions and Shannon capacity from flow reversals.

    On a one-dimensional circular flow, the basin of each stable fixed point
    is bounded by the immediately preceding and following saddles.  Stable
    and saddle angles are wrapped to ``[0,2*pi)``, jointly sorted, and required
    to alternate around the complete cycle.  The basin proportions are those
    saddle-to-saddle arc lengths divided by ``2*pi``.  Entropy uses natural
    logarithms.  A limit cycle has no such inputs and must be represented as
    undefined capacity by the caller, rather than by passing empty tensors.
    """

    _require_floating_tensor("stable_fixed_point_angle", stable_fixed_point_angle)
    _require_floating_tensor("saddle_fixed_point_angle", saddle_fixed_point_angle)
    if stable_fixed_point_angle.ndim != 1 or saddle_fixed_point_angle.ndim != 1:
        raise ValueError("stable and saddle fixed-point angles must be one-dimensional")
    if stable_fixed_point_angle.shape != saddle_fixed_point_angle.shape:
        raise ValueError("a cyclic alternating flow must have equal stable and saddle counts")
    if (
        stable_fixed_point_angle.dtype != saddle_fixed_point_angle.dtype
        or stable_fixed_point_angle.device != saddle_fixed_point_angle.device
    ):
        raise ValueError("stable and saddle angles must share dtype and device")
    tolerance = float(angle_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("angle_tolerance must be finite and non-negative")

    two_pi = 2.0 * math.pi
    stable = torch.remainder(stable_fixed_point_angle, two_pi)
    saddle = torch.remainder(saddle_fixed_point_angle, two_pi)
    combined = torch.cat((stable, saddle))
    # zero denotes stable and one denotes saddle in the joint cyclic ordering.
    kinds = torch.cat(
        (
            torch.zeros(stable.numel(), dtype=torch.int64, device=stable.device),
            torch.ones(saddle.numel(), dtype=torch.int64, device=stable.device),
        )
    )
    order = torch.argsort(combined)
    ordered_angle = combined[order]
    ordered_kind = kinds[order]
    cyclic_gap = torch.diff(torch.cat((ordered_angle, ordered_angle[:1] + two_pi)))
    if bool((cyclic_gap <= tolerance).any()):
        raise StructuralNotEstimableError(
            "fixed-point angles are not numerically distinct within angle_tolerance"
        )
    if bool((ordered_kind == torch.roll(ordered_kind, shifts=-1)).any()):
        raise StructuralNotEstimableError(
            "stable and saddle fixed points do not alternate around the ring"
        )

    stable_order = torch.argsort(stable)
    stable_sorted = stable[stable_order]
    left_saddles: list[torch.Tensor] = []
    right_saddles: list[torch.Tensor] = []
    widths: list[torch.Tensor] = []
    for value in stable_sorted:
        forward = torch.remainder(saddle - value, two_pi)
        backward = torch.remainder(value - saddle, two_pi)
        right = saddle[torch.argmin(forward)]
        left = saddle[torch.argmin(backward)]
        left_saddles.append(left)
        right_saddles.append(right)
        # With one stable point and one saddle, that same saddle is both basin
        # boundaries.  The basin excludes one measure-zero point but occupies
        # the full circle; remainder(right-left, 2*pi) would report zero.
        if stable_sorted.numel() == 1:
            widths.append(value.new_tensor(two_pi))
        else:
            widths.append(torch.remainder(right - left, two_pi))

    left_tensor = torch.stack(left_saddles)
    right_tensor = torch.stack(right_saddles)
    width_tensor = torch.stack(widths)
    proportions = width_tensor / two_pi
    scale = max(100.0 * torch.finfo(proportions.dtype).eps, tolerance)
    if not bool(
        torch.isclose(
            proportions.sum(),
            torch.ones((), dtype=proportions.dtype, device=proportions.device),
            atol=scale,
            rtol=scale,
        )
    ):
        raise StructuralNotEstimableError(
            "stable basin arcs do not form one complete ring"
        )
    entropy = -(proportions * torch.log(proportions)).sum()
    return StableBasinCapacity(
        stable_fixed_point_angle=stable_sorted,
        left_saddle_angle=left_tensor,
        right_saddle_angle=right_tensor,
        basin_width_radians=width_tensor,
        basin_proportions=proportions,
        shannon_entropy_nats=entropy,
        effective_basin_count=torch.exp(entropy),
    )


def asymptotic_memory_metrics(
    initial_angle: torch.Tensor,
    observed_terminal_angle: torch.Tensor,
    stable_fixed_point_angle: torch.Tensor,
    *,
    topology: Literal["fixed_points", "limit_cycle"],
) -> AsymptoticMemoryMetrics:
    """Estimate stable basin fractions and infinite-time angular error.

    For fixed-point topology, each observed terminal angle is assigned to the
    nearest supplied stable fixed point by circular distance.  Basin
    proportions are empirical fractions of the supplied initial-memory bank.
    Shannon entropy uses the natural logarithm, and
    ``effective_basin_count = exp(entropy)`` is reported without thresholding.
    Infinite-time errors compare each initial memory with its assigned stable
    fixed-point angle, not with the merely finite terminal sample.

    For limit-cycle topology, ``stable_fixed_point_angle`` must be empty and
    stable-basin capacity is explicitly undefined.
    """

    _require_floating_tensor("initial_angle", initial_angle)
    _require_floating_tensor("observed_terminal_angle", observed_terminal_angle)
    if not isinstance(stable_fixed_point_angle, torch.Tensor) or not stable_fixed_point_angle.is_floating_point():
        raise ValueError("stable_fixed_point_angle must be a floating torch tensor")
    if initial_angle.ndim != 1 or observed_terminal_angle.shape != initial_angle.shape:
        raise ValueError("initial and observed terminal angles must share shape [N]")
    if stable_fixed_point_angle.ndim != 1:
        raise ValueError("stable_fixed_point_angle must have shape [K]")
    for name, value in (
        ("observed_terminal_angle", observed_terminal_angle),
        ("stable_fixed_point_angle", stable_fixed_point_angle),
    ):
        if value.dtype != initial_angle.dtype or value.device != initial_angle.device:
            raise ValueError(f"{name} must share initial_angle dtype and device")
        if value.numel() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")
    if topology not in {"fixed_points", "limit_cycle"}:
        raise ValueError("topology must be 'fixed_points' or 'limit_cycle'")

    terminal_error = circular_absolute_error(observed_terminal_angle, initial_angle)
    stable_sorted = torch.sort(
        torch.remainder(stable_fixed_point_angle, 2.0 * math.pi)
    ).values

    if topology == "limit_cycle":
        if stable_sorted.numel() != 0:
            raise ValueError("limit-cycle topology cannot have stable fixed points")
        return AsymptoticMemoryMetrics(
            topology=topology,
            initial_angle=initial_angle,
            observed_terminal_angle=observed_terminal_angle,
            observed_terminal_absolute_error=terminal_error,
            observed_terminal_mean_error=terminal_error.mean(),
            observed_terminal_maximum_error=terminal_error.max(),
            stable_fixed_point_angle=stable_sorted,
            basin_assignment=None,
            basin_counts=None,
            basin_proportions=None,
            shannon_entropy_nats=None,
            effective_basin_count=None,
            assigned_stable_angle=None,
            asymptotic_absolute_error=None,
            asymptotic_mean_error=None,
            asymptotic_maximum_error=None,
        )

    if stable_sorted.numel() == 0:
        raise ValueError("fixed-point topology requires at least one stable fixed point")
    pairwise_error = circular_absolute_error(
        observed_terminal_angle[:, None], stable_sorted[None, :]
    )
    assignment = pairwise_error.argmin(dim=1)
    counts = torch.bincount(assignment, minlength=stable_sorted.numel())
    proportions = counts.to(dtype=initial_angle.dtype) / float(initial_angle.numel())
    positive = proportions > 0
    entropy = -(proportions[positive] * torch.log(proportions[positive])).sum()
    assigned = stable_sorted[assignment]
    asymptotic_error = circular_absolute_error(assigned, initial_angle)
    return AsymptoticMemoryMetrics(
        topology=topology,
        initial_angle=initial_angle,
        observed_terminal_angle=observed_terminal_angle,
        observed_terminal_absolute_error=terminal_error,
        observed_terminal_mean_error=terminal_error.mean(),
        observed_terminal_maximum_error=terminal_error.max(),
        stable_fixed_point_angle=stable_sorted,
        basin_assignment=assignment,
        basin_counts=counts,
        basin_proportions=proportions,
        shannon_entropy_nats=entropy,
        effective_basin_count=torch.exp(entropy),
        assigned_stable_angle=assigned,
        asymptotic_absolute_error=asymptotic_error,
        asymptotic_mean_error=asymptotic_error.mean(),
        asymptotic_maximum_error=asymptotic_error.max(),
    )
