from __future__ import annotations

import numpy as np
import torch

from repro.manifold_benchmark.analyze_subspace_attraction import (
    empirical_slow_subspace,
    intersection_normal_basis,
    outside_normal_basis,
    stationarity_metrics,
)


def test_intersection_and_outside_bases_have_expected_geometry() -> None:
    dtype = torch.float64
    subspace = torch.eye(6, dtype=dtype)[:, :3]
    tangent = torch.stack(
        (
            torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype),
            torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype),
        ),
        dim=1,
    )
    tangent = torch.linalg.qr(tangent, mode="reduced").Q
    inside = intersection_normal_basis(subspace, tangent)
    outside = outside_normal_basis(
        subspace, tangent, count=2, seed=123
    )
    assert inside.shape == (6, 1)
    assert outside.shape == (6, 2)
    assert torch.allclose(
        tangent.T @ inside, torch.zeros(2, 1, dtype=dtype), atol=1e-10
    )
    assert torch.allclose(
        subspace.T @ outside, torch.zeros(3, 2, dtype=dtype), atol=1e-10
    )
    assert torch.allclose(
        tangent.T @ outside, torch.zeros(2, 2, dtype=dtype), atol=1e-10
    )


class _DiagonalBlankModel:
    input_dim = 1

    def __init__(self, diagonal: torch.Tensor) -> None:
        self.diagonal = diagonal

    def reported_from_primary(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def primary_from_reported(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def step(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del inputs
        return state * self.diagonal


def test_empirical_slow_subspace_recovers_slowest_diagonal_axes() -> None:
    diagonal = torch.tensor([0.99, 0.95, 0.5, 0.1], dtype=torch.float64)
    model = _DiagonalBlankModel(diagonal)
    states = torch.randn(5, 4, dtype=torch.float64)
    basis, eigenvalues = empirical_slow_subspace(
        model, states, horizon=4, minimum_gain=0.8, chunk_size=2
    )
    projector = basis @ basis.T
    expected = torch.diag(
        torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    )
    assert torch.allclose(projector, expected, atol=1e-10)
    assert torch.all(eigenvalues[:-1] >= eigenvalues[1:])


class _DecodeIdentityModel(_DiagonalBlankModel):
    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, :2]


def test_stationarity_detects_exact_global_scaling(monkeypatch) -> None:
    model = _DecodeIdentityModel(torch.ones(3))
    theta = torch.linspace(0, 2 * torch.pi, 17, dtype=torch.float64)[:-1]
    initial = torch.stack(
        (torch.cos(theta), torch.sin(theta), torch.ones_like(theta) * 0.1),
        dim=1,
    )
    atlas = {0: initial, 128: 0.4 * initial}
    subspace = torch.eye(3, dtype=torch.float64)[:, :2]

    from repro.manifold_benchmark import analyze_subspace_attraction as module

    monkeypatch.setattr(module, "decode_primary", lambda _model, state: state[:, :2])
    rows, _ = stationarity_metrics(model, "s1", atlas, subspace)
    final = rows[-1]
    assert np.isclose(final["best_global_scale"], 0.4, atol=1e-10)
    assert final["global_scaling_residual_root"] < 1e-10
    assert final["pairwise_log_distortion_std"] < 1e-10
    assert final["carrier_direction_angle_median_rad"] < 1e-7
    assert final["decoded_same_memory_error_mean"] < 1e-7
