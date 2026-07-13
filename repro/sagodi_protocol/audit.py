"""Phase-0 state and blank-map audit for the Ságodi protocol.

No continuous-attractor metric should be generated before this audit passes.
The checks here operate on the wrapped model's literal ``step`` method and do
not substitute a retention-only recurrence for the actual blank-input map.
All returned data are JSON serializable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from .state import StateAdapter


@dataclass(frozen=True)
class AuditConfig:
    """Numerical settings for :func:`run_phase0_audit`."""

    batch_size: int = 2
    seed: int = 20260713
    random_directions: int = 16
    state_scale: float = 0.25
    finite_difference_epsilon: Optional[float] = None
    jvp_atol: Optional[float] = None
    jvp_rtol: Optional[float] = None
    architecture_max_dimension: int = 512

    def validate(self) -> None:
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.random_directions) <= 0:
            raise ValueError("random_directions must be positive")
        if not math.isfinite(float(self.state_scale)) or float(self.state_scale) <= 0.0:
            raise ValueError("state_scale must be finite and positive")
        if self.finite_difference_epsilon is not None and (
            not math.isfinite(float(self.finite_difference_epsilon))
            or float(self.finite_difference_epsilon) <= 0.0
        ):
            raise ValueError("finite_difference_epsilon must be finite and positive")
        if int(self.architecture_max_dimension) <= 0:
            raise ValueError("architecture_max_dimension must be positive")


def _cpu_random(
    shape: tuple[int, ...],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.randn(shape, generator=generator, dtype=torch.float64, device="cpu")
    return value.to(device=device, dtype=dtype)


def _max_abs(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().abs().max().cpu().item())


def _l2(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach()).cpu().item())


def _mean_row_l2(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach(), dim=-1).mean().cpu().item())


def _finite_or_none(value: float) -> Optional[float]:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _dtype_tolerances(
    dtype: torch.dtype,
    epsilon: Optional[float],
    atol: Optional[float],
    rtol: Optional[float],
) -> tuple[float, float, float]:
    if dtype in (torch.float64, torch.complex128):
        defaults = (1.0e-6, 2.0e-7, 2.0e-5)
    elif dtype in (torch.float16, torch.bfloat16):
        defaults = (2.0e-2, 3.0e-2, 1.5e-1)
    else:
        defaults = (1.0e-3, 7.5e-4, 3.0e-2)
    return (
        float(defaults[0] if epsilon is None else epsilon),
        float(defaults[1] if atol is None else atol),
        float(defaults[2] if rtol is None else rtol),
    )


def _prepare_primary_state(
    adapter: StateAdapter,
    state: Optional[torch.Tensor],
    config: AuditConfig,
) -> torch.Tensor:
    if state is not None:
        if not isinstance(state, torch.Tensor) or state.ndim != 2:
            raise ValueError("state must be a rank-2 torch.Tensor")
        if state.shape[-1] == adapter.reported_dim and adapter.reported_dim != adapter.primary_dim:
            state = adapter.primary_from_reported(state)
        adapter._validate(state, adapter.primary_dim, "audit primary state")
        return state.detach().clone()
    random_state = _cpu_random(
        (int(config.batch_size), adapter.primary_dim),
        seed=int(config.seed),
        device=adapter.device,
        dtype=adapter.dtype,
    )
    return random_state * float(config.state_scale)


def check_zero_input(adapter: StateAdapter, state: torch.Tensor) -> Dict[str, Any]:
    """Verify the externally supplied blank and record encoder bias activity."""

    zero = adapter.zero_input(state)
    external_exact = bool(torch.count_nonzero(zero).item() == 0)
    report: Dict[str, Any] = {
        "passed": external_exact,
        "shape": list(zero.shape),
        "dtype": str(zero.dtype).replace("torch.", ""),
        "device": str(zero.device),
        "nonzero_count": int(torch.count_nonzero(zero).item()),
        "max_abs": _max_abs(zero),
        "definition": "literal tensor passed as x_t to model.step",
    }
    encoder = getattr(adapter.model, "encoder", None)
    if callable(encoder):
        try:
            with torch.no_grad():
                encoded = encoder(zero)
            report["encoder_blank_drive"] = {
                "available": True,
                "shape": list(encoded.shape),
                "l2": _l2(encoded),
                "max_abs": _max_abs(encoded),
                "exactly_zero": bool(torch.count_nonzero(encoded).item() == 0),
                "note": "nonzero encoder bias remains part of the actual F0 map",
            }
        except Exception as exc:  # pragma: no cover - defensive for foreign wrappers
            report["encoder_blank_drive"] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        report["encoder_blank_drive"] = {"available": False, "reason": "no_encoder"}
    return report


def check_pack_unpack_round_trip(
    adapter: StateAdapter, state: torch.Tensor
) -> Dict[str, Any]:
    """Verify lossless embedding/projection of the declared primary state."""

    reported = adapter.reported_from_primary(state)
    recovered = adapter.primary_from_reported(reported)
    exact = bool(torch.equal(state, recovered))
    parts = adapter.unpack(reported)
    repacked = adapter.pack(parts)
    reported_exact = bool(torch.equal(reported, repacked))
    return {
        "passed": bool(exact and reported_exact),
        "primary_round_trip_bitwise_equal": exact,
        "reported_component_round_trip_bitwise_equal": reported_exact,
        "primary_max_abs_difference": _max_abs(recovered - state),
        "primary_dimension": adapter.primary_dim,
        "reported_dimension": adapter.reported_dim,
    }


def check_analysis_mode(adapter: StateAdapter) -> Dict[str, Any]:
    """Record that stochastic training modules are disabled during analysis."""

    dropout_modules = [
        module for module in adapter.model.modules() if isinstance(module, torch.nn.Dropout)
    ]
    dropout_disabled = all(not module.training for module in dropout_modules)
    model_eval = not adapter.model.training
    return {
        "passed": bool(model_eval and dropout_disabled),
        "model_eval_mode": model_eval,
        "dropout_module_count": len(dropout_modules),
        "all_dropout_modules_in_eval_mode": dropout_disabled,
        "training_state_noise": "not_part_of_model.step_and_not_supplied",
        "data_augmentation": "not_part_of_blank_map_audit",
    }


def check_map_inventory(adapter: StateAdapter, state: torch.Tensor) -> Dict[str, Any]:
    """Materialize the three blank-map definitions required by the protocol."""

    reported = adapter.reported_from_primary(state)
    zero = adapter.zero_input(reported)
    with torch.no_grad():
        next_reported = adapter.reported_step(reported, zero)
        next_primary = adapter.primary_from_reported(next_reported)
        next_carrier = adapter.unpack(next_reported).carrier
    primary_matches_carrier = bool(
        adapter.primary_dim == adapter.carrier_dim and torch.equal(next_primary, next_carrier)
    )
    return {
        "passed": True,
        "F0_primary": {
            "definition": "primary_from_reported(model.step(literal_zero, reported_from_primary(s)))",
            "input_dimension": adapter.primary_dim,
            "output_dimension": adapter.primary_dim,
        },
        "F0_carrier_if_applicable": {
            "definition": "unpack(model.step(literal_zero, reported_state)).carrier",
            "input_layout": "reported_state",
            "output_dimension": adapter.carrier_dim,
            "identical_to_primary_for_this_model": primary_matches_carrier,
        },
        "F0_reported_full": {
            "definition": "model.step(literal_zero, reported_state)",
            "input_dimension": adapter.reported_dim,
            "output_dimension": adapter.reported_dim,
        },
        "reported_stream_is_overwritten": bool(
            adapter.is_full_block and not adapter.carry_stream
        ),
    }


def check_determinism(adapter: StateAdapter, state: torch.Tensor) -> Dict[str, Any]:
    """Call F0 twice on identical values with the model in evaluation mode."""

    with torch.no_grad():
        first = adapter.actual_f0(state.clone())
        second = adapter.actual_f0(state.clone())
    difference = second - first
    finite = bool(torch.isfinite(first).all() and torch.isfinite(second).all())
    bitwise = bool(torch.equal(first, second))
    tolerance_equal = bool(torch.allclose(first, second, rtol=0.0, atol=0.0, equal_nan=False))
    return {
        "passed": bool(finite and tolerance_equal),
        "model_eval_mode": not adapter.model.training,
        "finite": finite,
        "bitwise_equal": bitwise,
        "zero_tolerance_equal": tolerance_equal,
        "max_abs_difference": _max_abs(difference),
    }


def check_no_external_hidden_cache(
    adapter: StateAdapter, state: torch.Tensor, *, seed: int
) -> Dict[str, Any]:
    """Detect transition state that is retained outside the packed Markov state.

    The same literal ``F0(state)`` is evaluated before and after an unrelated
    five-step trajectory.  A module-side step counter or hidden cache that
    affects the transition makes these two values differ even though the
    explicit input and packed recurrent state are identical.
    """

    distraction = _cpu_random(
        tuple(state.shape),
        seed=int(seed),
        device=state.device,
        dtype=state.dtype,
    )
    with torch.no_grad():
        before = adapter.actual_f0(state.clone())
        current = distraction
        for _ in range(5):
            current = adapter.actual_f0(current)
        after = adapter.actual_f0(state.clone())
    delta = after - before
    finite = bool(torch.isfinite(before).all() and torch.isfinite(after).all())
    bitwise = bool(torch.equal(before, after))
    return {
        "passed": bool(finite and bitwise),
        "intervening_steps": 5,
        "finite": finite,
        "bitwise_equal_after_unrelated_trajectory": bitwise,
        "max_abs_difference": _max_abs(delta),
        "interpretation": (
            "no transition-relevant step index or hidden cache was observed "
            "outside the packed state"
            if finite and bitwise
            else "the transition depends on module-side state not represented in the packed state"
        ),
    }


def check_float64_subset(
    adapter: StateAdapter,
    state: torch.Tensor,
    *,
    seed: int,
    directions: int = 4,
) -> Dict[str, Any]:
    """Re-run a deterministic subset in float64 and compare with float32.

    Parameters are copied before conversion, so this check cannot mutate the
    model used by the remaining audit or by a caller.
    """

    try:
        model64 = copy.deepcopy(adapter.model).to(device=state.device, dtype=torch.float64)
        adapter64 = StateAdapter(model64)
        state64 = state.detach().to(dtype=torch.float64)
        with torch.no_grad():
            output_reference = adapter.actual_f0(state).detach().to(dtype=torch.float64)
            output64 = adapter64.actual_f0(state64)
        delta = output64 - output_reference
        scale = max(_l2(output64) + _l2(output_reference), torch.finfo(torch.float64).eps)
        relative_l2 = _l2(delta) / scale
        close = bool(
            torch.isfinite(output64).all()
            and torch.allclose(output64, output_reference, atol=2.0e-6, rtol=2.0e-5)
        )
        determinism = check_determinism(adapter64, state64)
        jvp = check_autograd_jvp(
            adapter64,
            state64,
            directions=int(directions),
            seed=int(seed),
        )
        passed = bool(close and determinism["passed"] and jvp["passed"])
        return {
            "passed": passed,
            "available": True,
            "model_was_deepcopied": True,
            "reference_dtype": str(state.dtype).replace("torch.", ""),
            "analysis_dtype": "float64",
            "output_close": close,
            "output_atol": 2.0e-6,
            "output_rtol": 2.0e-5,
            "output_max_abs_difference": _max_abs(delta),
            "output_symmetric_relative_l2_error": relative_l2,
            "determinism": determinism,
            "autograd_vs_central_finite_difference": jvp,
        }
    except Exception as exc:
        return {
            "passed": False,
            "available": False,
            "model_was_deepcopied": True,
            "analysis_dtype": "float64",
            "error": f"{type(exc).__name__}: {exc}",
        }


def check_actual_f0(adapter: StateAdapter, state: torch.Tensor) -> Dict[str, Any]:
    """Check that the adapter map is the carrier projection of actual step(0)."""

    zero = adapter.zero_input(state)
    reported = adapter.reported_from_primary(state)
    if adapter.is_full_block and not adapter.carry_stream:
        # Use a deliberately nonzero old stream.  Equality then checks both the
        # projection and the claimed overwrite semantics of actual model.step.
        parts = adapter.unpack(reported)
        assert parts.stream is not None
        stream = torch.linspace(
            -0.75,
            0.75,
            adapter.stream_dim,
            device=state.device,
            dtype=state.dtype,
        ).unsqueeze(0).expand(state.shape[0], -1)
        reported_for_direct = adapter.pack(parts.carrier, stream)
    else:
        reported_for_direct = reported

    with torch.no_grad():
        adapted = adapter.actual_f0(state)
        direct_reported = adapter.model.step(zero, reported_for_direct)
        direct = adapter.primary_from_reported(direct_reported)
    difference = adapted - direct
    finite = bool(torch.isfinite(adapted).all() and torch.isfinite(direct).all())
    exact = bool(torch.equal(adapted, direct))
    close = bool(torch.allclose(adapted, direct, rtol=0.0, atol=0.0, equal_nan=False))
    residual = adapted - state
    return {
        "passed": bool(finite and close),
        "uses_literal_model_step": True,
        "external_input_l2": _l2(zero),
        "matches_direct_step_bitwise": exact,
        "matches_direct_step_zero_tolerance": close,
        "projection_max_abs_difference": _max_abs(difference),
        "primary_output_mean_l2": _mean_row_l2(adapted),
        "fixedness_residual_mean_l2": _mean_row_l2(residual),
        "finite": finite,
        "direct_step_used_nonzero_old_stream": bool(
            adapter.is_full_block and not adapter.carry_stream
        ),
    }


def check_autograd_jvp(
    adapter: StateAdapter,
    state: torch.Tensor,
    *,
    directions: int,
    seed: int,
    epsilon: Optional[float] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare F0 JVPs with central finite differences on random directions."""

    fd_epsilon, abs_tolerance, rel_tolerance = _dtype_tolerances(
        state.dtype, epsilon, atol, rtol
    )
    rows = []
    passed_count = 0
    max_abs_error = 0.0
    max_relative_error = 0.0
    blank = adapter.zero_input(state)

    def map_function(current: torch.Tensor) -> torch.Tensor:
        return adapter.primary_step(current, blank)

    for index in range(int(directions)):
        direction = _cpu_random(
            tuple(state.shape),
            seed=int(seed) + 104729 * (index + 1),
            device=state.device,
            dtype=state.dtype,
        )
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1.0e-12)
        base = state.detach()
        _, autograd_value = torch.autograd.functional.jvp(
            map_function,
            (base,),
            (direction,),
            create_graph=False,
            strict=False,
        )
        with torch.no_grad():
            plus = map_function(base + fd_epsilon * direction)
            minus = map_function(base - fd_epsilon * direction)
            finite_difference = (plus - minus) / (2.0 * fd_epsilon)
        delta = autograd_value - finite_difference
        absolute_error = _max_abs(delta)
        denominator = max(
            _l2(autograd_value) + _l2(finite_difference),
            float(torch.finfo(state.dtype).eps) if state.dtype.is_floating_point else 1.0e-12,
        )
        relative_error = _l2(delta) / denominator
        finite = bool(
            torch.isfinite(autograd_value).all()
            and torch.isfinite(finite_difference).all()
        )
        close = bool(
            finite
            and torch.allclose(
                autograd_value,
                finite_difference,
                atol=abs_tolerance,
                rtol=rel_tolerance,
                equal_nan=False,
            )
        )
        if close:
            passed_count += 1
        max_abs_error = max(max_abs_error, absolute_error)
        max_relative_error = max(max_relative_error, relative_error)
        rows.append(
            {
                "direction": index,
                "passed": close,
                "finite": finite,
                "autograd_l2": _l2(autograd_value),
                "finite_difference_l2": _l2(finite_difference),
                "max_abs_error": absolute_error,
                "symmetric_relative_l2_error": relative_error,
            }
        )
    return {
        "passed": passed_count == int(directions),
        "directions": int(directions),
        "passed_directions": passed_count,
        "finite_difference": "central",
        "epsilon": fd_epsilon,
        "atol": abs_tolerance,
        "rtol": rel_tolerance,
        "max_abs_error": max_abs_error,
        "max_symmetric_relative_l2_error": max_relative_error,
        "per_direction": rows,
    }


def _known_structural_case(adapter: StateAdapter) -> Dict[str, Any]:
    """Conservative source-level classification of legacy recurrence forms."""

    model_name = type(adapter.model).__name__
    if not adapter.is_full_block:
        if model_name in {
            "DiagSSMBaseline",
            "ComplexLRUUnitBaseline",
            "RealDiagLRUUnitBaseline",
            "PLRUUnitBaseline",
            "PANUnitBaseline",
        }:
            return {
                "recognized": True,
                "form": "affine_linear_blank_map",
                "reason": f"recognized legacy {model_name}.step formula",
            }
        return {
            "recognized": False,
            "form": "model_defined",
            "reason": "StepBaseline subclass is not on the conservative linear-form allowlist",
        }

    blocks = getattr(adapter.model, "blocks", None)
    if (
        blocks is not None
        and len(blocks) == 1
        and not adapter.carry_stream
        and hasattr(blocks[0], "rec")
    ):
        rec_name = type(blocks[0].rec).__name__
        if rec_name in {
            "ComplexLRURec",
            "RealDiagLRURec",
            "PLRURec",
            "PANRec",
            "RGLRURec",
        }:
            return {
                "recognized": True,
                "form": "affine_linear_primary_blank_map",
                "reason": (
                    "single recurrent block; overwritten stream cannot feed back; "
                    f"{rec_name} is affine in state for a fixed blank drive"
                ),
            }
        if rec_name == "PANNonlinearWriterRec" and str(
            getattr(blocks[0].rec, "writer_mode", "")
        ) == "input":
            return {
                "recognized": True,
                "form": "affine_linear_primary_blank_map",
                "reason": (
                    "single recurrent block with input-only writer; blank drive is "
                    "state-independent and old stream is overwritten"
                ),
            }
        if rec_name == "PANNonlinearWriterRec" and str(
            getattr(blocks[0].rec, "writer_mode", "")
        ) == "recurrent":
            # This writer is g(h, u)-g(h, 0), so it vanishes identically in h
            # only when the *actual drive received by the recurrence* is zero.
            # External x=0 is insufficient when an encoder bias is present.
            try:
                zero_external = torch.zeros(
                    1,
                    adapter.input_dim,
                    device=adapter.device,
                    dtype=adapter.dtype,
                )
                with torch.no_grad():
                    stream = adapter.model.encoder(zero_external)
                    recurrent_drive = blocks[0].norm_in(stream)
                drive_exactly_zero = bool(
                    torch.count_nonzero(recurrent_drive).item() == 0
                )
                drive_max_abs = _max_abs(recurrent_drive)
            except Exception as exc:  # pragma: no cover - foreign full block
                return {
                    "recognized": False,
                    "form": "conditional_recurrent_writer",
                    "reason": (
                        "could not verify the recurrent writer's blank drive: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            if drive_exactly_zero:
                return {
                    "recognized": True,
                    "form": "diagonal_linear_primary_blank_map",
                    "blank_recurrent_drive_exactly_zero": True,
                    "blank_recurrent_drive_max_abs": drive_max_abs,
                    "reason": (
                        "single recurrent block with u=0 exactly; recurrent writer "
                        "g(h,u)-g(h,0) vanishes identically and F0(h)=Lambda h"
                    ),
                }
            return {
                "recognized": False,
                "form": "state_dependent_writer_under_nonzero_blank_drive",
                "blank_recurrent_drive_exactly_zero": False,
                "blank_recurrent_drive_max_abs": drive_max_abs,
                "reason": (
                    "external input is zero but encoder/norm produces u!=0, so the "
                    "recurrent writer need not vanish"
                ),
            }
    return {
        "recognized": False,
        "form": "coupled_or_nonlinear_full_map",
        "reason": "architecture is not covered by a conservative affine source-level rule",
    }


def architecture_exactness_screen(
    adapter: StateAdapter,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    seed: int = 20260713,
    max_dimension: int = 512,
) -> Dict[str, Any]:
    """Screen whether actual F0 is an affine map with a unique fixed point.

    The batched basis test reconstructs the candidate affine linear part from
    the *actual primary map*, then validates it on independent random states.
    If ``I-A`` is nonsingular, that affine map has only one fixed point and an
    exact fixed-point continuum is ruled out for that screened architecture.
    This is an architecture/numerical screen, not a proof about an arbitrary
    untested nonlinear state domain; the report records that limitation.
    """

    dimension = int(adapter.primary_dim)
    structural = _known_structural_case(adapter)
    if dimension > int(max_dimension):
        return {
            "passed": True,
            "completed": False,
            "reason": "primary_dimension_exceeds_dense_screen_limit",
            "primary_dimension": dimension,
            "max_dimension": int(max_dimension),
            "structural_classification": structural,
            "exact_continuum_ruled_out": False,
            "recommendation": "continue exactness audit with architecture-specific JVP/operator tests",
        }

    target_device = torch.device(device) if device is not None else adapter.device
    target_dtype = dtype if dtype is not None else adapter.dtype
    if target_dtype in (torch.float16, torch.bfloat16):
        # The model parameters still determine the executable dtype.  Record
        # the lower-precision screen with appropriately loose tolerances.
        affine_atol, affine_rtol = 2.0e-2, 5.0e-2
    elif target_dtype == torch.float64:
        affine_atol, affine_rtol = 2.0e-9, 2.0e-8
    else:
        affine_atol, affine_rtol = 5.0e-5, 2.0e-4

    zero = torch.zeros(1, dimension, device=target_device, dtype=target_dtype)
    basis_scale = 0.5
    basis = torch.eye(dimension, device=target_device, dtype=target_dtype) * basis_scale
    with torch.no_grad():
        offset = adapter.actual_f0(zero)[0]
        responses = adapter.actual_f0(basis)
    # Row i is F(scale e_i)-F(0), hence rows are the transpose of the
    # conventional column-action matrix A.
    row_action = (responses - offset.unsqueeze(0)) / basis_scale
    matrix = row_action.transpose(0, 1).contiguous()

    probes = _cpu_random(
        (5, dimension),
        seed=int(seed) + 7919,
        device=target_device,
        dtype=target_dtype,
    ) * 0.73
    with torch.no_grad():
        observed = adapter.actual_f0(probes)
        predicted = offset.unsqueeze(0) + probes @ matrix.transpose(0, 1)
    affine_delta = observed - predicted
    affine_consistent = bool(
        torch.isfinite(observed).all()
        and torch.allclose(
            observed,
            predicted,
            atol=affine_atol,
            rtol=affine_rtol,
            equal_nan=False,
        )
    )

    matrix_cpu = matrix.detach().to(device="cpu", dtype=torch.float64)
    offset_cpu = offset.detach().to(device="cpu", dtype=torch.float64)
    identity = torch.eye(dimension, dtype=torch.float64)
    singular = torch.linalg.svdvals(matrix_cpu)
    gap_singular = torch.linalg.svdvals(identity - matrix_cpu)
    eigenvalues = torch.linalg.eigvals(matrix_cpu)
    operator_norm = float(singular.max().item()) if singular.numel() else 0.0
    spectral_radius = (
        float(eigenvalues.abs().max().item()) if eigenvalues.numel() else 0.0
    )
    minimum_fixed_gap = (
        float(gap_singular.min().item()) if gap_singular.numel() else 0.0
    )
    off_diagonal = matrix_cpu - torch.diag(torch.diagonal(matrix_cpu))
    off_diagonal_max = _max_abs(off_diagonal)
    matrix_scale = max(_max_abs(matrix_cpu), 1.0)
    diagonal_consistent = off_diagonal_max <= affine_atol + affine_rtol * matrix_scale
    strict_contraction = bool(operator_norm < 1.0 - max(affine_atol, 1.0e-8))
    unique_fixed_point = bool(
        affine_consistent and minimum_fixed_gap > 10.0 * max(affine_atol, 1.0e-10)
    )

    fixed_point_residual: Optional[float] = None
    fixed_point_norm: Optional[float] = None
    if unique_fixed_point:
        fixed_point = torch.linalg.solve(identity - matrix_cpu, offset_cpu)
        residual = matrix_cpu @ fixed_point + offset_cpu - fixed_point
        fixed_point_residual = _l2(residual)
        fixed_point_norm = _l2(fixed_point)

    matrix_array = np.ascontiguousarray(matrix_cpu.numpy().astype("<f8", copy=False))
    matrix_digest = hashlib.sha256(matrix_array.tobytes(order="C")).hexdigest()
    ruled_out = bool(unique_fixed_point)
    recommendation = (
        "stop exact-continuum claim; evaluate approximate-CA C1-C4"
        if ruled_out
        else "exact continuum not ruled out by Phase-0 screen; continue fixedness audit"
    )
    return {
        "passed": True,
        "completed": True,
        "method": "actual_F0_batched_basis_affine_screen",
        "primary_dimension": dimension,
        "structural_classification": structural,
        "affine_consistent": affine_consistent,
        "affine_validation_max_abs_error": _max_abs(affine_delta),
        "affine_validation_mean_l2_error": _mean_row_l2(affine_delta),
        "affine_atol": affine_atol,
        "affine_rtol": affine_rtol,
        "homogeneous_zero_fixed": bool(
            _max_abs(offset_cpu) <= affine_atol
        ),
        "offset_l2": _l2(offset_cpu),
        "diagonal_in_reported_coordinates": bool(diagonal_consistent),
        "off_diagonal_max_abs": off_diagonal_max,
        "operator_norm": _finite_or_none(operator_norm),
        "spectral_radius": _finite_or_none(spectral_radius),
        "strict_euclidean_contraction": strict_contraction,
        "minimum_singular_value_I_minus_A": _finite_or_none(minimum_fixed_gap),
        "unique_affine_fixed_point": unique_fixed_point,
        "fixed_point_norm": _finite_or_none(fixed_point_norm) if fixed_point_norm is not None else None,
        "fixed_point_equation_residual_l2": (
            _finite_or_none(fixed_point_residual)
            if fixed_point_residual is not None
            else None
        ),
        "linear_part_shape": [dimension, dimension],
        "linear_part_float64_sha256": matrix_digest,
        "exact_continuum_ruled_out": ruled_out,
        "evidence_scope": (
            "source-level recognized form plus actual-map numerical validation"
            if structural["recognized"]
            else "actual-map numerical screen on basis and random probes"
        ),
        "limitation": (
            "A numerical affine screen is not a global proof for an unrecognized "
            "nonlinear architecture."
        ),
        "recommendation": recommendation,
    }


def run_phase0_audit(
    model_or_adapter: torch.nn.Module | StateAdapter,
    *,
    state: Optional[torch.Tensor] = None,
    config: Optional[AuditConfig] = None,
    **config_overrides: Any,
) -> Dict[str, Any]:
    """Run the complete Phase-0 gate and return a JSON-serializable report.

    ``state`` may be either a primary state or, when dimensions differ, a
    legacy reported state.  Supplying a task-evoked state is recommended for
    checkpoint audits; otherwise a deterministic nonzero probe state is used.
    """

    if config is not None and config_overrides:
        raise ValueError("pass either config or keyword overrides, not both")
    effective = config if config is not None else AuditConfig(**config_overrides)
    effective.validate()
    adapter = (
        model_or_adapter
        if isinstance(model_or_adapter, StateAdapter)
        else StateAdapter(model_or_adapter)
    )
    primary = _prepare_primary_state(adapter, state, effective)

    was_training = bool(adapter.model.training)
    cuda_devices = []
    if primary.device.type == "cuda":
        cuda_devices = [
            primary.device.index
            if primary.device.index is not None
            else torch.cuda.current_device()
        ]
    adapter.model.eval()
    try:
        # Isolate any accidental model-side random draw while still allowing
        # the determinism check to observe two successive calls.
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(int(effective.seed))
            if cuda_devices:
                torch.cuda.manual_seed_all(int(effective.seed))
            pack_unpack = check_pack_unpack_round_trip(adapter, primary)
            analysis_mode = check_analysis_mode(adapter)
            map_inventory = check_map_inventory(adapter, primary)
            zero_check = check_zero_input(adapter, primary)
            determinism = check_determinism(adapter, primary)
            no_hidden_cache = check_no_external_hidden_cache(
                adapter, primary, seed=int(effective.seed) + 17
            )
            actual_f0 = check_actual_f0(adapter, primary)
            reported = adapter.reported_from_primary(primary)
            stream = adapter.stream_intervention(reported)
            jvp = check_autograd_jvp(
                adapter,
                primary,
                directions=int(effective.random_directions),
                seed=int(effective.seed) + 1,
                epsilon=effective.finite_difference_epsilon,
                atol=effective.jvp_atol,
                rtol=effective.jvp_rtol,
            )
            exactness = architecture_exactness_screen(
                adapter,
                dtype=primary.dtype,
                device=primary.device,
                seed=int(effective.seed) + 2,
                max_dimension=int(effective.architecture_max_dimension),
            )
            float64_subset = check_float64_subset(
                adapter,
                primary,
                seed=int(effective.seed) + 3,
                directions=min(4, int(effective.random_directions)),
            )
    finally:
        adapter.model.train(was_training)

    checks = {
        "state_transition_inventory": {
            "passed": True,
            "state_spec": adapter.state_spec(),
        },
        "pack_unpack_round_trip": pack_unpack,
        "required_blank_maps": map_inventory,
        "analysis_mode": analysis_mode,
        "zero_input": zero_check,
        "determinism": determinism,
        "no_external_hidden_cache": no_hidden_cache,
        "stream_intervention": stream,
        "actual_f0": actual_f0,
        "autograd_vs_central_finite_difference": jvp,
        "float64_subset": float64_subset,
        "architecture_exactness_screen": exactness,
    }
    passed = all(bool(check.get("passed", False)) for check in checks.values())
    report: Dict[str, Any] = {
        "schema_version": 1,
        "audit": "sagodi_phase0_state_audit",
        "passed": passed,
        "state_spec": adapter.state_spec(),
        "configuration": asdict(effective),
        "probe_state": {
            "shape": list(primary.shape),
            "dtype": str(primary.dtype).replace("torch.", ""),
            "device": str(primary.device),
            "l2": _l2(primary),
            "source": "caller" if state is not None else "deterministic_random_nonzero",
        },
        "checks": checks,
        "model_training_mode_before_audit": was_training,
        "model_training_mode_restored": bool(adapter.model.training == was_training),
    }
    # Fail here during development if a Tensor or non-finite JSON number leaked
    # into the artifact contract.  ``allow_nan=False`` is stricter than the
    # default JSON encoder used by many experiment scripts.
    json.dumps(report, sort_keys=True, allow_nan=False)
    return report


def phase0_audit(
    model_or_adapter: torch.nn.Module | StateAdapter,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Concise alias for :func:`run_phase0_audit`."""

    return run_phase0_audit(model_or_adapter, **kwargs)


__all__ = [
    "AuditConfig",
    "architecture_exactness_screen",
    "check_analysis_mode",
    "check_actual_f0",
    "check_autograd_jvp",
    "check_determinism",
    "check_float64_subset",
    "check_no_external_hidden_cache",
    "check_map_inventory",
    "check_pack_unpack_round_trip",
    "check_zero_input",
    "phase0_audit",
    "run_phase0_audit",
]
