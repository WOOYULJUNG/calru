from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest
import torch

from repro.manifold_benchmark.analyze_blank_memory import _finite_scalar_or_none
from repro.manifold_benchmark.analyze_manifold_geometry import (
    _pairwise_latent,
    _spearman,
)
from repro.manifold_benchmark.analyze_tangent_normal import (
    _orthonormal_tangent,
    _random_normals,
)
from repro.manifold_benchmark.make_analysis_banks import make_bank
from repro.manifold_benchmark.topology_analysis_common import (
    expected_jobs,
    load_analysis_config,
    nmse_db,
    normalized_geodesic_errors,
    summarize_error_tensor,
)
from repro.manifold_benchmark.topology_models import build_topology_model


def test_blank_json_scalar_maps_nonfinite_values_to_null():
    assert _finite_scalar_or_none(torch.tensor(float("nan"))) is None
    assert _finite_scalar_or_none(torch.tensor(float("inf"))) is None
    assert _finite_scalar_or_none(torch.tensor(float("-inf"))) is None
    assert _finite_scalar_or_none(torch.tensor(1.25)) == pytest.approx(1.25)


def test_hc_reported_state_reconstruction_matches_full_block_step():
    model = build_topology_model("hc", "s1", model_seed=123)
    memory = torch.randn(5, 2)
    inputs = torch.randn(5, 1)
    initial = model.initialize(memory)
    reported = model.step(inputs, initial)
    primary = model.primary_from_reported(reported)
    reconstructed = model.reported_from_primary_for_input(primary, inputs)
    torch.testing.assert_close(reconstructed, reported)
    torch.testing.assert_close(model.decode(reconstructed), model.decode(reported))


def test_analysis_contract_has_exact_36_run_denominator():
    config = load_analysis_config()
    jobs = expected_jobs(config)
    assert len(jobs) == 36
    assert len(set(jobs)) == 36
    assert config["expected_training"]["git_commit"].startswith("1f4a80d")
    assert config["execution"][
        "analysis_test_bank_access_only_after_all_training_completion"
    ]


def test_normalized_geodesic_definitions_have_common_range():
    s1_target = torch.tensor([[[1.0, 0.0]]])
    s1_prediction = torch.tensor([[[-1.0, 0.0]]])
    s1, _ = normalized_geodesic_errors("s1", s1_prediction, s1_target)
    torch.testing.assert_close(s1, torch.ones_like(s1))

    t2_target = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    t2_prediction = torch.tensor([[[-1.0, 0.0, 1.0, 0.0]]])
    t2, worst = normalized_geodesic_errors("t2", t2_prediction, t2_target)
    torch.testing.assert_close(t2, torch.full_like(t2, 1.0 / math.sqrt(2.0)))
    torch.testing.assert_close(worst, torch.ones_like(worst))

    s2_target = torch.tensor([[[1.0, 0.0, 0.0]]])
    s2_prediction = -s2_target
    s2, _ = normalized_geodesic_errors("s2", s2_prediction, s2_target)
    torch.testing.assert_close(s2, torch.ones_like(s2))


def test_nmse_and_trial_summary_are_component_normalized():
    target = torch.ones(3, 2, 2)
    prediction = torch.zeros_like(target)
    assert nmse_db(prediction, target) == pytest.approx(0.0)
    summary = summarize_error_tensor(torch.tensor([[0.0, 0.5], [0.5, 1.0]]))
    assert summary["sequence_mean"] == pytest.approx(0.5)
    assert summary["terminal_mean"] == pytest.approx(0.75)


@pytest.mark.parametrize("topology", ("s1", "t2", "s2"))
def test_small_analysis_banks_are_deterministic_and_exactly_closed(topology):
    config = deepcopy(load_analysis_config())
    config["analysis_banks"].update(
        atlas_points=64,
        transport_horizon=8,
        closed_path_count=2,
        closed_half_horizon=4,
    )
    left, left_metadata = make_bank(topology, config)
    right, right_metadata = make_bank(topology, config)
    assert left.keys() == right.keys()
    for name in left:
        np.testing.assert_array_equal(left[name], right[name])
    assert left_metadata == right_metadata
    assert left_metadata["oracle_max_closed_path_error_radians"] < 1e-6
    assert left["initializer_memory"].shape[0] == 64
    assert left["transport_inputs"].shape[:2] == (8, 64)
    np.testing.assert_array_equal(
        left["transport_inputs"],
        np.repeat(left["transport_inputs"][:, :1], 64, axis=1),
    )
    assert left["closed_inputs"].shape[:2] == (8, 128)
    assert left_metadata["initializer_atlas_is_diagnostic_only"]
    assert left_metadata["transported_endpoint_atlas_is_primary"]
    assert "common_nonzero_schedule" in left_metadata["transport_control_pairing"]


def test_projected_random_normals_are_orthogonal_to_tangent():
    raw = torch.randn(7, 12, 2)
    tangent = _orthonormal_tangent(raw)
    normal = _random_normals(tangent, 5, seed=123)
    tangent_gram = tangent.transpose(-2, -1) @ tangent
    normal_gram = normal.transpose(-2, -1) @ normal
    cross = tangent.transpose(-2, -1) @ normal
    torch.testing.assert_close(
        tangent_gram, torch.eye(2).expand_as(tangent_gram), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        normal_gram, torch.eye(5).expand_as(normal_gram), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(cross, torch.zeros_like(cross), atol=1e-5, rtol=1e-5)


def test_latent_distance_and_spearman_are_topology_aware():
    ring = np.array([[-math.pi + 0.01], [math.pi - 0.01], [0.0]])
    distance = _pairwise_latent("s1", ring)
    assert distance[0, 1] == pytest.approx(0.02)
    assert _spearman(np.arange(10.0), np.arange(10.0) ** 3) == pytest.approx(1.0)
