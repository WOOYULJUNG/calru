from __future__ import annotations

import math

import torch

from repro.sagodi_protocol.manifold_diagnostics import (
    manifold_recovery_diagnostics,
    settling_quality_diagnostics,
)


def _ring_states(count: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    angles = -torch.pi + 2.0 * torch.pi * torch.arange(
        count, dtype=torch.float64
    ) / float(count)
    states = torch.stack(
        (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=-1
    )
    return angles, states


def _analytic_ring_projection(state: torch.Tensor) -> dict[str, torch.Tensor]:
    xy = state[:, :2]
    radius = torch.linalg.vector_norm(xy, dim=-1)
    safe_radius = radius.clamp_min(100.0 * torch.finfo(state.dtype).eps)
    unit = xy / safe_radius[:, None]
    angle = torch.atan2(unit[:, 1], unit[:, 0])
    projected = torch.cat((unit, torch.zeros_like(state[:, 2:])), dim=-1)
    tangent = torch.stack(
        (-torch.sin(angle), torch.cos(angle), torch.zeros_like(angle)), dim=-1
    )
    return {
        "distance": torch.linalg.vector_norm(state - projected, dim=-1),
        "angle": angle,
        "tangent_frame": tangent,
        "tangent_frame_valid": radius > 100.0 * torch.finfo(state.dtype).eps,
    }


def _radial_and_ambient(states: torch.Tensor) -> torch.Tensor:
    radial = states.clone()
    ambient = torch.zeros_like(states)
    ambient[:, 2] = 1.0
    return torch.stack((radial, ambient), dim=1)


def test_global_origin_contraction_no_longer_counts_as_manifold_recovery():
    _, ring = _ring_states()
    contraction = 0.9985

    result = manifold_recovery_diagnostics(
        ring,
        _radial_and_ambient(ring),
        radius=0.1,
        state_scale=1.0,
        horizons=(0, 1, 5, 20, 100, 500),
        actual_f0=lambda state: contraction * state,
        project=_analytic_ring_projection,
    )

    endpoint = -1
    legacy = result["paired_endpoint_normal_deviation_over_radius"][endpoint]
    clean_distance = result["clean_manifold_distance_over_Rs"][endpoint]
    recovery_q = result["manifold_recovery_Q"][endpoint]

    # The former clean-paired metric meets the frozen median<=.5/q95<1 gate.
    assert float(torch.quantile(legacy, 0.95)) < 0.5
    # Both trajectories have actually collapsed far inside the candidate ring.
    assert float(torch.quantile(clean_distance, 0.95)) > 0.5
    assert float(torch.quantile(recovery_q, 0.05)) > 4.0
    assert bool(result["denominator_valid"].all())


def test_exact_ring_with_normal_contraction_has_clean_adherence_and_recovery():
    _, ring = _ring_states()
    contraction = 0.5

    def contract_only_normals(state: torch.Tensor) -> torch.Tensor:
        xy = state[:, :2]
        radius = torch.linalg.vector_norm(xy, dim=-1)
        unit = xy / radius.clamp_min(1.0e-12)[:, None]
        following_radius = 1.0 + contraction * (radius - 1.0)
        following_xy = unit * following_radius[:, None]
        return torch.cat((following_xy, contraction * state[:, 2:]), dim=-1)

    result = manifold_recovery_diagnostics(
        ring,
        _radial_and_ambient(ring),
        radius=0.1,
        state_scale=1.0,
        horizons=(0, 1, 5, 20),
        actual_f0=contract_only_normals,
        project=_analytic_ring_projection,
    )

    torch.testing.assert_close(
        result["clean_manifold_distance_over_Rs"],
        torch.zeros_like(result["clean_manifold_distance_over_Rs"]),
        atol=2.0e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result["manifold_recovery_Q"][0],
        torch.ones_like(result["manifold_recovery_Q"][0]),
        atol=2.0e-14,
        rtol=0.0,
    )
    expected_h5 = torch.full_like(result["manifold_recovery_Q"][2], contraction**5)
    torch.testing.assert_close(
        result["manifold_recovery_Q"][2], expected_h5, atol=2.0e-13, rtol=0.0
    )
    assert float(result["manifold_recovery_Q"][-1].max()) < 1.0e-5
    assert float(result["same_memory_E_excess"].max()) < 1.0e-14
    assert bool(result["denominator_valid"].all())
    assert result["clean_projected_angle"].shape == (4, 32)
    assert result["perturbed_projected_angle"].shape == (4, 32, 2)


def test_settling_rejects_expanding_path_variance_even_when_sheet_mean_is_fixed():
    _, ring = _ring_states(count=16)
    offsets = torch.tensor([-0.1, 0.1], dtype=ring.dtype)
    paths = ring[:, None, :].repeat(1, 2, 1)
    paths[:, :, 2] = offsets

    def expand_fiber(state: torch.Tensor) -> torch.Tensor:
        following = state.clone()
        following[:, 2] = 1.1 * following[:, 2]
        return following

    result = settling_quality_diagnostics(
        paths,
        state_scale=1.0,
        actual_f0=expand_fiber,
        project_mean_sheet=_analytic_ring_projection,
        blank_steps=1,
    )

    assert float(result["absolute_relative_change_q95"]) > 0.17
    assert float(result["positive_expansion_q95"]) > 0.20
    assert result["absolute_relative_change_passed"] is False
    assert result["positive_expansion_passed"] is False
    assert result["mean_sheet_adherence_passed"] is True
    assert result["passed"] is False


def test_settling_requires_mean_sheet_adherence_in_addition_to_variance_plateau():
    _, ring = _ring_states(count=16)
    offsets = torch.tensor([-0.1, 0.1], dtype=ring.dtype)
    paths = ring[:, None, :].repeat(1, 2, 1)
    paths[:, :, 2] = offsets

    def move_whole_sheet_inward(state: torch.Tensor) -> torch.Tensor:
        following = state.clone()
        following[:, :2] = 0.98 * following[:, :2]
        return following

    result = settling_quality_diagnostics(
        paths,
        state_scale=1.0,
        actual_f0=move_whole_sheet_inward,
        project_mean_sheet=_analytic_ring_projection,
        blank_steps=1,
    )

    assert float(result["absolute_relative_change_q95"]) < 1.0e-12
    assert float(result["positive_expansion_q95"]) < 1.0e-12
    assert float(result["mean_sheet_distance_q95"]) > 0.019
    assert result["absolute_relative_change_passed"] is True
    assert result["positive_expansion_passed"] is True
    assert result["mean_sheet_adherence_passed"] is False
    assert result["passed"] is False


def test_settling_accepts_a_two_sided_plateau_on_an_invariant_sheet():
    _, ring = _ring_states(count=16)
    offsets = torch.tensor([-0.1, 0.1], dtype=ring.dtype)
    paths = ring[:, None, :].repeat(1, 2, 1)
    paths[:, :, 2] = offsets

    result = settling_quality_diagnostics(
        paths,
        state_scale=1.0,
        actual_f0=lambda state: state.clone(),
        project_mean_sheet=_analytic_ring_projection,
        blank_steps=5,
    )

    assert result["absolute_relative_change_passed"] is True
    assert result["positive_expansion_passed"] is True
    assert result["mean_sheet_adherence_passed"] is True
    assert result["passed"] is True
    assert math.isclose(result["systematic_decrease_fraction"], 0.0)
