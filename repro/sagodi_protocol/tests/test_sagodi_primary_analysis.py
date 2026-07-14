from __future__ import annotations

import math

import pytest
import torch

from repro.sagodi_protocol.sagodi_primary_analysis import (
    angle_from_output,
    asymptotic_memory_metrics,
    circular_absolute_error,
    circular_difference,
    cyclic_flow_reversal_topology,
    dense_full_jacobian_eigenspectrum,
    discrete_vector_field,
    finite_time_angular_memory,
    finite_time_angular_memory_from_output,
    output_projected_flow,
    signed_angular_flow,
    stable_basin_capacity,
)


DTYPE = torch.float64


def test_circular_angle_helpers_use_cos_sin_order_and_shortest_arc():
    target = torch.tensor([math.pi, -math.pi + 0.2], dtype=DTYPE)
    estimate = torch.tensor([-math.pi + 0.1, math.pi - 0.1], dtype=DTYPE)

    signed = circular_difference(estimate, target)
    absolute = circular_absolute_error(estimate, target)

    torch.testing.assert_close(
        signed, torch.tensor([0.1, -0.3], dtype=DTYPE), atol=1.0e-14, rtol=0.0
    )
    torch.testing.assert_close(absolute, signed.abs())

    output = torch.stack((torch.cos(estimate), torch.sin(estimate)), dim=-1)
    torch.testing.assert_close(angle_from_output(output), estimate, atol=1.0e-14, rtol=0.0)


def test_output_projected_flow_is_finite_step_difference_and_uniform_norm():
    angle = torch.arange(4, dtype=DTYPE) * (math.pi / 2.0)
    state = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    rate = 0.125

    def autonomous_map(value: torch.Tensor) -> torch.Tensor:
        tangent = torch.stack((-value[:, 1], value[:, 0]), dim=-1)
        return value + rate * tangent

    expected = autonomous_map(state) - state
    torch.testing.assert_close(discrete_vector_field(state, autonomous_map), expected)

    result = output_projected_flow(state, autonomous_map, lambda value: value)
    torch.testing.assert_close(result.state_vector_field, expected)
    torch.testing.assert_close(result.projected_vector_field, expected)
    torch.testing.assert_close(
        result.pointwise_norm, torch.full((4,), rate, dtype=DTYPE)
    )
    torch.testing.assert_close(result.uniform_norm, torch.tensor(rate, dtype=DTYPE))
    torch.testing.assert_close(
        signed_angular_flow(result.output, result.projected_vector_field),
        torch.full((4,), rate, dtype=DTYPE),
    )


def test_output_projected_flow_uses_decode_of_both_endpoints_for_nonlinear_decoder():
    state = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE)
    result = output_projected_flow(
        state,
        lambda value: 2.0 * value,
        lambda value: value.square(),
    )

    # decode(2s)-decode(s) = 3s^2.  This deliberately differs from applying
    # the decoder directly to F0(s)-s, which would produce only s^2.
    torch.testing.assert_close(result.projected_vector_field, 3.0 * state.square())


def test_signed_angular_flow_rejects_undefined_origin():
    output = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=DTYPE)
    field = torch.ones_like(output)
    with pytest.raises(ValueError, match="undefined inside minimum_radius"):
        signed_angular_flow(output, field)


def test_cyclic_flow_reversal_finds_stable_saddle_and_seam_roots():
    # Deliberately shuffled to verify that roots are found in cyclic angle order.
    angle = torch.tensor([math.pi, 0.0, 1.5 * math.pi, 0.5 * math.pi], dtype=DTYPE)
    flow = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=DTYPE)

    result = cyclic_flow_reversal_topology(angle, flow)

    assert result.kind == "fixed_points"
    assert not result.is_limit_cycle
    assert [item.kind for item in result.reversals] == [
        "stable",
        "saddle",
        "stable",
        "saddle",
    ]
    assert [item.angle for item in result.stable] == pytest.approx(
        [math.pi / 4.0, 5.0 * math.pi / 4.0]
    )
    assert [item.angle for item in result.saddles] == pytest.approx(
        [3.0 * math.pi / 4.0, 7.0 * math.pi / 4.0]
    )
    # The last saddle is the cyclic 3pi/2 -> 0 seam bracket.
    assert result.saddles[-1].left_index == 2
    assert result.saddles[-1].right_index == 1


def test_cyclic_flow_topology_distinguishes_limit_cycle_and_zero_cases():
    angle = torch.arange(5, dtype=DTYPE) * (2.0 * math.pi / 5.0)

    positive = cyclic_flow_reversal_topology(angle, torch.ones(5, dtype=DTYPE))
    assert positive.kind == "limit_cycle"
    assert positive.is_limit_cycle
    assert positive.orientation == "positive"

    stationary = cyclic_flow_reversal_topology(angle, torch.zeros(5, dtype=DTYPE))
    assert stationary.kind == "stationary_continuum"
    assert stationary.orientation is None

    unresolved = cyclic_flow_reversal_topology(
        angle,
        torch.tensor([1.0, 1.0e-8, 1.0, 1.0, 1.0], dtype=DTYPE),
        zero_tolerance=1.0e-7,
    )
    assert unresolved.kind == "unidirectional_with_stationary_samples"
    assert unresolved.orientation == "positive"


def test_cyclic_flow_reversal_compresses_tolerated_zero_plateau():
    angle = torch.arange(6, dtype=DTYPE) * (math.pi / 3.0)
    flow = torch.tensor([1.0, 1.0e-9, -1.0, -1.0, -1.0e-9, 1.0], dtype=DTYPE)

    result = cyclic_flow_reversal_topology(angle, flow, zero_tolerance=1.0e-8)

    assert len(result.reversals) == 2
    assert [item.kind for item in result.reversals] == ["stable", "saddle"]
    assert result.reversals[0].angle == pytest.approx(math.pi / 3.0)
    assert result.reversals[1].angle == pytest.approx(4.0 * math.pi / 3.0)


def test_cyclic_flow_reversal_rejects_duplicate_seam_angles():
    angle = torch.tensor([0.0, math.pi, 2.0 * math.pi], dtype=DTYPE)
    flow = torch.tensor([1.0, -1.0, 1.0], dtype=DTYPE)
    with pytest.raises(ValueError, match="unique"):
        cyclic_flow_reversal_topology(angle, flow)


def test_dense_jacobian_keeps_map_and_vector_field_complex_spectra():
    matrix = torch.tensor(
        [[0.9, -0.2, 0.0], [0.2, 0.9, 0.0], [0.0, 0.0, 0.5]], dtype=DTYPE
    )
    state = torch.tensor([[1.0, 2.0, 3.0], [-0.5, 0.25, 1.5]], dtype=DTYPE)

    result = dense_full_jacobian_eigenspectrum(
        state, lambda value: value @ matrix.T
    )

    torch.testing.assert_close(
        result.map_jacobian, matrix.expand(state.shape[0], -1, -1)
    )
    identity = torch.eye(3, dtype=DTYPE)
    torch.testing.assert_close(
        result.vector_field_jacobian,
        (matrix - identity).expand(state.shape[0], -1, -1),
    )
    assert torch.is_complex(result.map_eigenvalues)
    assert torch.is_complex(result.vector_field_eigenvalues)
    assert float(result.map_eigenvalues.imag.abs().max()) == pytest.approx(0.2)
    torch.testing.assert_close(
        result.largest_real_part, torch.full((2,), -0.1, dtype=DTYPE)
    )
    torch.testing.assert_close(
        result.second_largest_real_part, torch.full((2,), -0.1, dtype=DTYPE)
    )
    torch.testing.assert_close(result.real_part_gap, torch.zeros(2, dtype=DTYPE))


def test_dense_jacobian_top_two_gap_uses_jf_minus_identity():
    diagonal = torch.tensor([1.0, 0.8, 0.5], dtype=DTYPE)
    matrix = torch.diag(diagonal)
    state = torch.tensor([[0.1, 0.2, 0.3]], dtype=DTYPE)

    result = dense_full_jacobian_eigenspectrum(
        state, lambda value: value @ matrix.T
    )

    torch.testing.assert_close(
        result.vector_field_eigenvalues_sorted.real,
        torch.tensor([[0.0, -0.2, -0.5]], dtype=DTYPE),
        atol=1.0e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.real_part_gap, torch.tensor([0.2], dtype=DTYPE), atol=1.0e-14, rtol=0.0
    )


def test_finite_time_angular_memory_reports_prefix_averages():
    predicted = torch.tensor(
        [[0.0, 0.1, 0.2], [math.pi - 0.1, -math.pi + 0.05, -math.pi + 0.15]],
        dtype=DTYPE,
    )
    target = torch.tensor([0.0, math.pi], dtype=DTYPE)

    result = finite_time_angular_memory(predicted, target)

    torch.testing.assert_close(
        result.absolute_error,
        torch.tensor([[0.0, 0.1, 0.2], [0.1, 0.05, 0.15]], dtype=DTYPE),
        atol=1.0e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.mean_error,
        torch.tensor([0.05, 0.075, 0.175], dtype=DTYPE),
        atol=1.0e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.cumulative_mean_error,
        torch.tensor([0.05, 0.0625, 0.1], dtype=DTYPE),
        atol=1.0e-14,
        rtol=0.0,
    )

    output = torch.stack((torch.cos(predicted), torch.sin(predicted)), dim=-1)
    from_output = finite_time_angular_memory_from_output(output, target)
    torch.testing.assert_close(from_output.absolute_error, result.absolute_error)


def test_asymptotic_fixed_point_metrics_use_empirical_basins_and_natural_log():
    initial = torch.tensor([0.0, 0.1, math.pi, math.pi + 0.1], dtype=DTYPE)
    terminal = torch.tensor([0.01, -0.01, math.pi + 0.01, math.pi - 0.01], dtype=DTYPE)
    # Input order is intentionally reversed; output order is canonicalized.
    stable = torch.tensor([math.pi, 0.0], dtype=DTYPE)

    result = asymptotic_memory_metrics(
        initial, terminal, stable, topology="fixed_points"
    )

    torch.testing.assert_close(
        result.stable_fixed_point_angle, torch.tensor([0.0, math.pi], dtype=DTYPE)
    )
    torch.testing.assert_close(result.basin_counts, torch.tensor([2, 2]))
    torch.testing.assert_close(
        result.basin_proportions, torch.tensor([0.5, 0.5], dtype=DTYPE)
    )
    assert float(result.shannon_entropy_nats) == pytest.approx(math.log(2.0))
    assert float(result.effective_basin_count) == pytest.approx(2.0)
    torch.testing.assert_close(
        result.asymptotic_absolute_error,
        torch.tensor([0.0, 0.1, 0.0, 0.1], dtype=DTYPE),
        atol=1.0e-14,
        rtol=0.0,
    )
    assert float(result.asymptotic_mean_error) == pytest.approx(0.05)
    assert float(result.asymptotic_maximum_error) == pytest.approx(0.1)


def test_stable_basin_capacity_uses_neighboring_saddle_arc_widths():
    stable = torch.tensor([math.pi, 0.0], dtype=DTYPE)
    saddle = torch.tensor([1.5 * math.pi, 0.25 * math.pi], dtype=DTYPE)

    result = stable_basin_capacity(stable, saddle)

    torch.testing.assert_close(
        result.stable_fixed_point_angle, torch.tensor([0.0, math.pi], dtype=DTYPE)
    )
    torch.testing.assert_close(
        result.basin_width_radians,
        torch.tensor([0.75 * math.pi, 1.25 * math.pi], dtype=DTYPE),
    )
    torch.testing.assert_close(
        result.basin_proportions, torch.tensor([0.375, 0.625], dtype=DTYPE)
    )
    expected_entropy = -(0.375 * math.log(0.375) + 0.625 * math.log(0.625))
    assert float(result.shannon_entropy_nats) == pytest.approx(expected_entropy)
    assert float(result.effective_basin_count) == pytest.approx(math.exp(expected_entropy))


def test_single_stable_single_saddle_basin_is_the_complete_circle():
    stable = torch.tensor([0.0], dtype=DTYPE)
    saddle = torch.tensor([3.0], dtype=DTYPE)

    result = stable_basin_capacity(stable, saddle)

    torch.testing.assert_close(
        result.basin_width_radians, torch.tensor([2.0 * math.pi], dtype=DTYPE)
    )
    torch.testing.assert_close(
        result.basin_proportions, torch.tensor([1.0], dtype=DTYPE)
    )
    assert float(result.shannon_entropy_nats) == pytest.approx(0.0)
    assert float(result.effective_basin_count) == pytest.approx(1.0)


def test_stable_basin_capacity_requires_cyclic_alternation():
    stable = torch.tensor([0.0, 0.5], dtype=DTYPE)
    saddle = torch.tensor([math.pi, 1.5 * math.pi], dtype=DTYPE)
    with pytest.raises(ValueError, match="alternate"):
        stable_basin_capacity(stable, saddle)


def test_asymptotic_limit_cycle_marks_basin_capacity_undefined():
    initial = torch.tensor([0.0, 1.0], dtype=DTYPE)
    terminal = torch.tensor([0.1, 1.1], dtype=DTYPE)
    result = asymptotic_memory_metrics(
        initial,
        terminal,
        torch.empty(0, dtype=DTYPE),
        topology="limit_cycle",
    )

    assert result.basin_assignment is None
    assert result.basin_proportions is None
    assert result.shannon_entropy_nats is None
    assert result.effective_basin_count is None
    assert result.asymptotic_absolute_error is None
    assert float(result.observed_terminal_mean_error) == pytest.approx(0.1)
