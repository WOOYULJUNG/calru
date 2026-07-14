from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import repro.sagodi_protocol.phase1_analysis as phase1_module
from repro.sagodi_protocol.artifacts import (
    atomic_json,
    sha256_file,
    verify_completion_receipt,
    write_completion_receipt,
)
from repro.sagodi_protocol.config import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
    protocol_fingerprint,
)
from repro.sagodi_protocol.models import (
    ModelConfig,
    build_protocol_model,
    checkpoint_payload,
    load_checkpoint,
)
from repro.sagodi_protocol.phase1_analysis import (
    _build_plan,
    _build_ring_projector,
    _claim_gate,
    _neighborhood_metrics,
    _periodic_cubic_resample,
    _project_ring,
    _projection_quality_audit,
    _ring_local_rank_metrics,
    _ring_path_velocity_bank,
    _sampled_jvp_gains,
    _slow_state_reconstruction,
    _track_a_coverage_quality,
    _track_a_geometry_quality,
    _verify_parent_bound_bank,
    _verify_checkpoint_identity,
    analyze_checkpoint,
    main,
)
from repro.sagodi_protocol.state import StateAdapter


def _tiny_run(path: Path) -> Path:
    path.mkdir()
    torch.manual_seed(123)
    model = build_protocol_model(
        ModelConfig(
            name="gru",
            input_dim=1,
            output_dim=2,
            initial_memory_dim=2,
            init_mode="hidden_init",
            width=4,
        )
    )
    torch.save(
        checkpoint_payload(
            model,
            {
                "model_seed": 100,
                "protocol_freeze_id": "sagodi_phase01_ring_pilot_v1",
                "synthetic_smoke_checkpoint": True,
            },
        ),
        path / "checkpoint.pt",
    )
    return path


def test_primary_plan_freezes_requested_ring_counts_and_horizons():
    plan = _build_plan(load_protocol(DEFAULT_PROTOCOL_PATH), smoke=False)
    assert plan.atlas_count == 1024
    assert plan.drift_horizon == 1024
    assert plan.recovery_horizon == 500
    assert plan.jacobian_horizon == 50
    assert plan.ambient_directions == 8
    assert plan.kick_relative_radius == 0.1
    assert plan.tangent_shift == 0.01


def test_smoke_analysis_emits_primary_carrier_arrays_and_conservative_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def successful_track_a(model, adapter, target_angles, *, horizon, **kwargs):
        state = torch.stack(
            (
                torch.cos(target_angles),
                torch.sin(target_angles),
                0.2 * torch.cos(2.0 * target_angles),
                0.2 * torch.sin(2.0 * target_angles),
            ),
            dim=1,
        )
        count = int(target_angles.numel())
        return {
            "success": True,
            "failure_reasons": [],
            "start_state_rule": "test_task_driven_endpoints",
            "rollout_horizon": int(horizon),
            "trajectory_count": count,
            "candidate_threshold": "test_fixture",
            "candidate_count": count,
            "candidate_trajectory_coverage": count,
            "max_speed": np.zeros(count, dtype=np.float32),
            "candidate_count_per_trajectory": np.ones(count, dtype=np.int64),
            "selected_time": np.zeros(count, dtype=np.int64),
            "selected_trajectory": np.arange(count, dtype=np.int64),
            "selected_decoded_angle": target_angles.detach().cpu().numpy(),
            "selected_target_error_radians": np.zeros(count),
            "selected_state": state,
            "resampled_state": state,
            "coverage": {"passed": True, "scope": "test_fixture"},
            "geometry_available": True,
            "interpolation": "test_periodic_ring",
        }

    monkeypatch.setattr(
        phase1_module, "_slow_state_reconstruction", successful_track_a
    )
    run_dir = _tiny_run(tmp_path / "run")
    output = analyze_checkpoint(
        protocol_path=DEFAULT_PROTOCOL_PATH,
        run_dir=run_dir,
        output_dir=tmp_path / "analysis",
        device="cpu",
        smoke=True,
    )
    assert output == (tmp_path / "analysis").resolve()
    assert (output / "analysis.json").is_file()
    assert (output / "analysis_arrays.npz").is_file()
    assert (output / "claim_gate.json").is_file()
    valid, reason = verify_completion_receipt(output / "completion_receipt.json")
    assert valid, reason

    analysis = json.loads((output / "analysis.json").read_text())
    assert analysis["primary_analysis_state"] == "carrier_only_minimal_markov_state"
    assert analysis["plan"]["atlas_count"] == 32
    assert analysis["plan"]["drift_horizon"] == 8
    assert analysis["plan"]["recovery_horizon"] == 5
    assert analysis["sampled_jacobian"]["strict_worst_normal_evaluated"] is False
    assert analysis["sampled_jacobian"]["primary"] is True
    assert analysis["sampled_jacobian"]["normal_projection_schedule"] == "every_step_including_initial_normalization"
    assert analysis["sampled_jacobian"]["endpoint_only_J_product_proxy"]["primary"] is False
    assert analysis["finite_kicks"]["clean_paired"] is True
    assert analysis["finite_kicks"]["ambient_directions_per_anchor"] == 2
    assert analysis["atlas"]["primary_source"] == "track_a_resampled_state"
    assert analysis["atlas"]["construction"].startswith("Sagodi_Track_A")
    assert analysis["atlas"]["Sagodi_Track_A"]["rollout_horizon"] == 128
    assert analysis["atlas"]["task_conditioned"]["path_count"] == 8
    assert analysis["atlas"]["task_conditioned"]["settle_horizons"] == [0, 5, 20, 100]
    assert analysis["atlas"]["projection_quality"]["known_q_state_source"].endswith(
        "not_spline_generated"
    )
    assert analysis["checkpoint_preflight"]["passed"] is True
    assert (output / "checkpoint_state_transition_audit.json").is_file()
    assert (output / "autonomous_radial_trace.json").is_file()
    assert (output / "autonomous_radial_trace.npz").is_file()

    with np.load(output / "analysis_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["atlas_angles"].shape == (32,)
        assert arrays["atlas_primary_carrier"].shape == (32, 4)
        np.testing.assert_array_equal(
            arrays["atlas_primary_carrier"],
            arrays["track_a_resampled_primary_carrier"],
        )
        assert arrays["atlas_tangent"].shape == (32, 4)
        assert arrays["diagnostic_legacy_paired_endpoint_R_N"].shape == (
            8,
            3,
        )  # radial + two sampled ambient
        assert arrays["kick_direction_kind"].tolist() == [
            "radial",
            "ambient_sampled",
            "ambient_sampled",
        ]
        assert arrays["sampled_primary_endpoint_tangent_gain"].shape == (8,)
        assert arrays["sampled_projected_cocycle_clean_tangent_frame_trace"].shape == (4, 8, 4)
        assert arrays["endpoint_only_J_product_proxy_tangent_gain"].shape == (8,)
        assert arrays["atlas_rank_normalized_sigma_d_over_sigma_1"].shape == (32,)
        assert arrays["atlas_rank_normalized_sigma_min"].shape == (32,)
        assert arrays["atlas_rank_normalized_sigma_max"].shape == (32,)
        assert arrays["task_path_velocity"].shape == (32, 8, 8)
        assert arrays["task_path_endpoint_primary_carrier"].shape == (32, 8, 4)
        assert arrays["task_path_settled_primary_carrier"].shape == (4, 32, 8, 4)
        assert arrays["track_a_discovery_source_indices"].shape == (32,)
        assert np.unique(arrays["track_a_discovery_source_indices"]).size == 32
        assert arrays["track_a_discovery_endpoint_reported_state"].shape == (32, 4)
        assert arrays["track_a_discovery_source_selection_error"].shape == (32,)
        # The synthetic Track-A fixture is deliberately unrelated to the tiny
        # model's task sheet, so corrected settling rejects the correspondence
        # bank instead of silently using it as a primary/fallback manifold.
        assert arrays["projection_qa_known_q_task_mean_primary_carrier"].shape == (0, 4)
        assert arrays["projection_qa_known_q_task_settled_path_primary_carrier"].shape == (0, 8, 4)
        assert arrays["manifold_recovery_horizons"].tolist() == [1, 5]
        assert arrays["manifold_recovery_clean_manifold_distance_over_Rs"].shape == (2, 8)
        assert arrays["manifold_recovery_manifold_recovery_Q"].shape == (2, 8, 3)
        assert arrays["task_settling_positive_relative_variance_expansion"].shape == (4, 32)
        assert np.isfinite(
            arrays["diagnostic_legacy_paired_endpoint_R_N"]
        ).all()
        assert np.isfinite(arrays["sampled_projected_cocycle_gamma"]).all()
        assert arrays["kick_radial_tangent_orthogonality_error"].max() <= 1e-6
        assert arrays["jacobian_ambient_tangent_orthogonality_error"].max() <= 1e-6

    claim = json.loads((output / "claim_gate.json").read_text())
    assert claim["pilot_only"] is True
    assert claim["smoke"] is True
    assert claim["gates"]["sampled_normal_gap"]["status"] == "not_evaluated"
    assert "not_exact_worst" in claim["gates"]["sampled_normal_gap"]["scope"]
    assert claim["gates"]["strict_worst_normal"]["status"] == "not_evaluated"
    assert claim["gates"]["strict_worst_normal"]["passed"] is None
    assert claim["gates"]["task_sheet_settling"]["status"] == "not_evaluated"
    assert claim["gates"]["c3_clean_adherence"]["status"] == "not_evaluated"
    assert claim["gates"]["c4"]["status"] == "not_evaluated"
    assert claim["gates"]["model_seeds"]["status"] == "not_evaluated"
    assert claim["gates"]["exact_axis"]["status"] == "not_evaluated"
    assert claim["gates"]["c1_rank"]["threshold"] == {
        "minimum_normalized_sigma_d_over_sigma_1": 0.001,
        "minimum_qualifying_atlas_fraction": 0.99,
    }
    assert "normalized_sigma_d_over_sigma_1" in claim["gates"]["c1_rank"]["value"]
    assert claim["levels"]["L3"]["passed"] is False
    assert claim["approximate_ca_claim_allowed"] is False
    assert all(
        gate["status"] == "not_evaluated" and gate["passed"] is None
        for gate in claim["gates"].values()
        if gate["gate_id"] in {"task", "c1_decoding", "c1_rank", "invariance", "c2_drift"}
    )

    with np.load(output / "autonomous_radial_trace.npz", allow_pickle=False) as trace:
        assert trace["actual_input"].shape == (20, 1)
        assert np.count_nonzero(trace["actual_input"]) == 0
        assert trace["overwrite_mask"].shape == (20, 4)
        assert trace["reset_mask"].shape == (20, 4)
        assert np.count_nonzero(trace["reset_mask"]) == 0


def test_track_a_failure_has_no_task_atlas_fallback_and_projected_gates_are_inconclusive(
    tmp_path: Path,
):
    run_dir = _tiny_run(tmp_path / "run")
    output = analyze_checkpoint(
        protocol_path=DEFAULT_PROTOCOL_PATH,
        run_dir=run_dir,
        output_dir=tmp_path / "analysis",
        device="cpu",
        smoke=True,
    )
    analysis = json.loads((output / "analysis.json").read_text())
    assert analysis["manifold_analysis"] == "inconclusive_track_a_reconstruction_failed"
    assert analysis["atlas"]["construction"] == "none_no_primary_fallback"
    assert analysis["atlas"]["primary_eligible"] is False
    with np.load(output / "analysis_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["atlas_primary_carrier"].shape[0] == 0
        np.testing.assert_array_equal(
            arrays["atlas_primary_carrier"],
            arrays["track_a_resampled_primary_carrier"],
        )
    claim = json.loads((output / "claim_gate.json").read_text())
    for gate_id in (
        "c1_decoding",
        "invariance",
        "c2_drift",
        "c3_normal_recovery",
        "c3_same_memory",
        "tangent_equivariance",
        "sampled_normal_gap",
    ):
        # Smoke uses the stricter not_evaluated label; a full run with the same
        # reconstruction failure is inconclusive through primary_atlas_valid.
        assert claim["gates"][gate_id]["passed"] is None
        assert claim["gates"][gate_id]["status"] in {"not_evaluated", "inconclusive"}


def test_legacy_v1_full_analysis_requires_explicit_v2_analysis_freeze(
    tmp_path: Path,
):
    run_dir = _tiny_run(tmp_path / "run")
    with pytest.raises(ValueError, match="explicit corrected v2 analysis freeze"):
        analyze_checkpoint(
            protocol_path=DEFAULT_PROTOCOL_PATH,
            run_dir=run_dir,
            output_dir=tmp_path / "analysis",
            device="cpu",
            smoke=False,
        )
    assert not (tmp_path / "analysis").exists()


def test_periodic_cubic_spline_is_continuous_across_ring_seam():
    dtype = torch.float64
    knots = 0.13 + 2.0 * torch.pi * torch.arange(64, dtype=dtype) / 64.0
    state = torch.stack((torch.cos(knots), torch.sin(knots)), dim=1)
    query = torch.tensor(
        [-1.0e-5, 0.0, 1.0e-5, 2.0 * torch.pi - 1.0e-5], dtype=dtype
    )
    observed = _periodic_cubic_resample(knots, state, query)
    expected = torch.stack((torch.cos(query), torch.sin(query)), dim=1)
    torch.testing.assert_close(observed, expected, atol=2.0e-6, rtol=2.0e-6)
    torch.testing.assert_close(observed[0], observed[-1], atol=3.0e-5, rtol=0.0)


def test_track_a_geometry_quality_gates_tangent_floor_and_periodic_seam() -> None:
    angles = -torch.pi + 2.0 * torch.pi * torch.arange(
        128, dtype=torch.float64
    ) / 128.0
    state = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    speed = torch.ones_like(angles)
    quality = _track_a_geometry_quality(
        angles,
        state,
        speed,
        torch.tensor(1.0, dtype=torch.float64),
        minimum_normalized_tangent_speed=1e-3,
        seam_probe_radians=1e-3,
        seam_c0_over_rs_max=1e-5,
        seam_c1_relative_max=1e-2,
    )
    assert quality["passed"] is True
    collapsed = _track_a_geometry_quality(
        angles,
        state,
        torch.zeros_like(speed),
        torch.tensor(1.0, dtype=torch.float64),
        minimum_normalized_tangent_speed=1e-3,
        seam_probe_radians=1e-3,
        seam_c0_over_rs_max=1e-5,
        seam_c1_relative_max=1e-2,
    )
    assert collapsed["passed"] is False
    assert collapsed["component_passed"]["tangent_speed_floor"] is False


def test_track_a_coverage_reports_and_requires_all_coarse_bins() -> None:
    target = -torch.pi + 2.0 * torch.pi * torch.arange(
        64, dtype=torch.float64
    ) / 64.0
    selected = target.numpy().copy()
    pair = np.stack((np.zeros(64, dtype=np.int64), np.arange(64)), axis=1)
    quality, _ = _track_a_coverage_quality(
        selected,
        pair,
        target,
        coarse_bin_count=32,
        minimum_coarse_occupancy_fraction=1.0,
    )
    assert quality["passed"] is True
    assert quality["occupied_coarse_bins"] == 32

    missing_sector = selected.copy()
    missing_sector[:8] = selected[8]
    quality, _ = _track_a_coverage_quality(
        missing_sector,
        pair,
        target,
        coarse_bin_count=32,
        minimum_coarse_occupancy_fraction=1.0,
    )
    assert quality["passed"] is False
    assert quality["coarse_bin_occupancy_fraction"] < 1.0


def test_parent_bound_bank_requires_exact_path_bytes_and_sidecar(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    bank_dir = parent / "evaluation_bank"
    bank_dir.mkdir(parents=True)
    bank = bank_dir / "bank.npz"
    bank.write_bytes(b"frozen-bank")
    sidecar = Path(f"{bank}.sha256")
    sidecar.write_text(f"{sha256_file(bank)}  {bank.name}\n")
    manifest = {
        "evaluation_bank": {
            "path": str(bank.relative_to(parent)),
            "sha256": sha256_file(bank),
            "sidecar_sha256": sha256_file(sidecar),
        }
    }
    _verify_parent_bound_bank(
        parent_root=parent,
        parent_manifest=manifest,
        bank_key="evaluation_bank",
        supplied_path=bank,
    )
    other = tmp_path / "other.npz"
    other.write_bytes(bank.read_bytes())
    with pytest.raises(ValueError, match="path differs"):
        _verify_parent_bound_bank(
            parent_root=parent,
            parent_manifest=manifest,
            bank_key="evaluation_bank",
            supplied_path=other,
        )
    bank.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="bytes differ"):
        _verify_parent_bound_bank(
            parent_root=parent,
            parent_manifest=manifest,
            bank_key="evaluation_bank",
            supplied_path=bank,
        )


def test_dense_spline_projection_refines_radial_points_across_seam():
    angles = -torch.pi + 2.0 * torch.pi * torch.arange(64, dtype=torch.float64) / 64.0
    state = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    projector = _build_ring_projector(angles, state, density=8)
    query_angles = torch.tensor([-torch.pi + 1e-4, torch.pi - 1e-4, 0.31], dtype=torch.float64)
    query = 1.07 * torch.stack((torch.cos(query_angles), torch.sin(query_angles)), dim=1)
    projection = _project_ring(query, projector)
    error = torch.atan2(
        torch.sin(projection["angle"] - query_angles),
        torch.cos(projection["angle"] - query_angles),
    ).abs()
    assert float(error.max()) < 3e-4


def test_projection_quality_known_q_uses_supplied_heldout_task_states_and_rejects_bias():
    angles = -torch.pi + 2.0 * torch.pi * torch.arange(64, dtype=torch.float64) / 64.0
    state = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    tangent = torch.stack((-torch.sin(angles), torch.cos(angles)), dim=1)
    projector = _build_ring_projector(angles, state, density=8)
    heldout = angles + torch.pi / 64.0
    # These states deliberately encode q+0.2 while their known labels remain q.
    # If QA regenerated points from its own spline, this bias would disappear.
    biased = torch.stack(
        (torch.cos(heldout + 0.2), torch.sin(heldout + 0.2)), dim=1
    )
    summary, arrays = _projection_quality_audit(
        projector,
        state,
        tangent,
        torch.zeros(2, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        heldout_angles=heldout,
        heldout_task_mean_state=biased,
        selected_settling_horizon=20,
        q95_max=0.002,
    )
    assert summary["passed"] is False
    assert summary["known_q_midcell"]["q95"] > 0.05
    assert summary["known_q_state_source"].endswith("not_spline_generated")
    torch.testing.assert_close(arrays["known_q_task_mean_primary_carrier"], biased)


def test_ring_local_rank_gate_reports_sigma_ratio_and_zero_chart_failure():
    derivative = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float64)
    result = _ring_local_rank_metrics(
        derivative,
        torch.tensor(2.0, dtype=torch.float64),
        minimum_ratio=1e-3,
        required_fraction=0.99,
    )
    torch.testing.assert_close(
        result["normalized_sigma_min"], torch.tensor([2.5, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result["normalized_sigma_max"], torch.tensor([2.5, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result["normalized_sigma_d_over_sigma_1"],
        torch.tensor([1.0, 0.0], dtype=torch.float64),
    )
    assert result["summary"]["qualifying_atlas_fraction"] == 0.5
    assert result["summary"]["passed"] is False


def test_sampled_normal_primary_projects_every_clean_step_and_keeps_endpoint_proxy():
    angles = -torch.pi + 2.0 * torch.pi * torch.arange(64, dtype=torch.float64) / 64.0
    state = torch.stack(
        (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=1
    )
    tangent = torch.stack(
        (-torch.sin(angles), torch.cos(angles), torch.zeros_like(angles)), dim=1
    )
    radial = state.clone()
    indices = torch.arange(0, 64, 8)
    ambient = torch.zeros(indices.numel(), 1, 3, dtype=torch.float64)
    ambient[:, 0, 2] = 1.0
    theta = 2.0 * torch.pi / 64.0
    matrix = torch.tensor(
        [
            [1.08 * np.cos(theta), -0.92 * np.sin(theta), 0.35],
            [1.08 * np.sin(theta), 0.92 * np.cos(theta), -0.20],
            [0.0, 0.0, 0.8],
        ],
        dtype=torch.float64,
    )

    class Adapter:
        @staticmethod
        def actual_f0(value):
            return value @ matrix.T

    result = _sampled_jvp_gains(
        Adapter(),
        state,
        tangent,
        indices,
        radial[indices],
        ambient,
        horizon=3,
        projector=_build_ring_projector(angles, state, density=8),
    )
    assert result["clean_projected_tangent_frame_trace"].shape == (4, 8, 3)
    assert bool(result["clean_projected_frame_valid_trace"].all())
    # Normal-to-tangent mixing is removed after every step only in the
    # primary normal cocycle, so it must differ from endpoint-only splitting.
    assert not torch.allclose(
        result["primary_ambient_normal_gain"],
        result["endpoint_proxy_ambient_normal_gain"],
    )
    # The primary tangent block is endpoint-projected without intermediate P_T.
    torch.testing.assert_close(
        result["primary_tangent_gain"], result["endpoint_proxy_tangent_gain"]
    )


def test_normal_qa_failure_makes_dependent_gates_inconclusive_with_null_passed():
    protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
    summary = {"mean": 0.0, "median": 0.0, "q95": 0.0, "q99": 0.0, "max": 0.0}
    claim = _claim_gate(
        protocol=protocol,
        plan=_build_plan(protocol, smoke=False),
        task_result={"masked_nmse_db": -30.0},
        paper_noise_result={"masked_nmse_db": -30.0},
        decoder_summary=summary,
        rank_summary={
            "passed": True,
            "qualifying_atlas_fraction": 1.0,
            "minimum_normalized_sigma_d_over_sigma_1": 1e-3,
            "required_atlas_fraction": 0.99,
        },
        neighborhood={"trustworthiness": 1.0, "continuity": 1.0},
        chi_fiber=0.0,
        track_a_success=True,
        settling_valid=True,
        settling_summary={"passed": [True]},
        primary_atlas_valid=True,
        projection_qa={"passed": True, "q95_max": 0.002},
        invariance_summary=summary,
        drift_summary=summary,
        clean_adherence={
            "all_registered_horizons_passed": True,
            "registered_horizons": [1, 5, 20, 100, 500, 1024],
            "by_horizon": {},
        },
        radial_recovery=summary,
        ambient_recovery=summary,
        recovery_input_valid=True,
        same_memory=summary,
        tangent_equivariance=summary,
        jvp={
            "tangent_gain": {"q95": 1.0},
            "sampled_normal_gain_max": summary,
            "sampled_joint_fraction": 1.0,
            "all_clean_projector_frames_valid": True,
        },
        normal_direction_qa={"passed": False, "threshold_max": 1e-6},
        preflight_audit={"checks": {"architecture_exactness_screen": {}}},
    )
    assert claim["gates"]["normal_direction_quality"]["status"] == "failed"
    for gate_id in ("c3_normal_recovery", "c3_same_memory", "sampled_normal_gap"):
        assert claim["gates"][gate_id]["status"] == "inconclusive"
        assert claim["gates"][gate_id]["passed"] is None
    assert claim["levels"]["L2"]["passed"] is False


def test_eight_velocity_paths_have_exact_target_displacement_and_are_distinct():
    angles = torch.tensor([-2.1, 0.0, 1.7], dtype=torch.float64)
    velocity = _ring_path_velocity_bank(angles, steps=32, dt=0.1)
    assert velocity.shape == (3, 8, 32)
    integrated = 0.1 * velocity.sum(dim=-1)
    torch.testing.assert_close(integrated, angles[:, None].expand(-1, 8), atol=1e-12, rtol=0.0)
    # Zero-net excursions ensure q=0 still has eight distinct input histories.
    assert torch.unique(velocity[1], dim=0).shape[0] == 8


def test_neighborhood_metrics_are_exact_for_canonical_circle():
    angles = -torch.pi + 2.0 * torch.pi * torch.arange(64, dtype=torch.float64) / 64.0
    state = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    result = _neighborhood_metrics(state, angles, neighbors=6)
    assert result["trustworthiness"] > 0.999
    assert result["continuity"] > 0.999
    assert result["pairwise_spearman"] > 0.999


class _ContractingRing(torch.nn.Module):
    state_size = 2
    input_dim = 1

    def init_state(self, batch: int, device):
        return torch.zeros(batch, 2, device=device, dtype=torch.float64)

    def initial_state(self, batch: int, device, initial_memory):
        return initial_memory.to(device=device, dtype=torch.float64)

    def step(self, input_tensor, state):
        return 0.9 * state

    def decode(self, state):
        return state


class _IdentityRing(_ContractingRing):
    def step(self, input_tensor, state):
        return state


def test_slow_state_reconstruction_accepts_exactly_stationary_ring():
    model = _IdentityRing().double()
    adapter = StateAdapter(model)
    angles = -torch.pi + 2.0 * torch.pi * torch.arange(32, dtype=torch.float64) / 32.0
    result = _slow_state_reconstruction(model, adapter, angles, horizon=16)
    assert result["success"] is True
    assert result["coverage"]["passed"] is True
    assert result["candidate_trajectory_coverage"] == 32
    assert np.count_nonzero(result["max_speed"]) == 0
    assert np.all(result["candidate_count_per_trajectory"] == 1)
    torch.testing.assert_close(
        result["resampled_state"],
        torch.stack((torch.cos(angles), torch.sin(angles)), dim=1),
        atol=2e-5,
        rtol=2e-5,
    )


def test_slow_state_reconstruction_replays_the_supplied_task_endpoints():
    model = _IdentityRing().double()
    adapter = StateAdapter(model)
    angles = -torch.pi + 2.0 * torch.pi * torch.arange(32, dtype=torch.float64) / 32.0
    radius = torch.linspace(1.0, 1.31, 32, dtype=torch.float64)
    endpoint = radius[:, None] * torch.stack(
        (torch.cos(angles), torch.sin(angles)), dim=1
    )
    result = _slow_state_reconstruction(
        model,
        adapter,
        angles,
        horizon=16,
        initial_reported_states=endpoint,
        start_state_rule="test_supplied_task_endpoints",
    )
    assert result["success"] is True
    assert result["start_state_rule"] == "test_supplied_task_endpoints"
    assert np.all(result["selected_time"] == 0)
    assert np.array_equal(result["selected_trajectory"], np.arange(32))
    # This exact equality guards the second-pass replay: the selected state
    # must come from the supplied endpoint bank, not a canonical re-init.
    torch.testing.assert_close(result["selected_state"], endpoint, rtol=0.0, atol=0.0)


def test_slow_state_reconstruction_rejects_clustered_arc_coverage():
    model = _IdentityRing().double()
    adapter = StateAdapter(model)
    target = -torch.pi + 2.0 * torch.pi * torch.arange(32, dtype=torch.float64) / 32.0
    clustered_angles = torch.linspace(-0.6, 0.6, 32, dtype=torch.float64)
    clustered_state = torch.stack(
        (torch.cos(clustered_angles), torch.sin(clustered_angles)), dim=1
    )
    result = _slow_state_reconstruction(
        model,
        adapter,
        target,
        horizon=8,
        initial_reported_states=clustered_state,
        start_state_rule="test_clustered_task_endpoints",
    )
    assert result["geometry_available"] is True
    assert result["coverage"]["passed"] is False
    assert result["success"] is False
    assert "full_ring_candidate_coverage_failed" in result["failure_reasons"]
    assert (
        result["coverage"]["maximum_circular_knot_gap_radians"]
        > result["coverage"]["thresholds"]["maximum_circular_knot_gap_radians"]
    )


def test_slow_state_reconstruction_uses_relative_speed_and_periodic_resampling():
    model = _ContractingRing().double()
    adapter = StateAdapter(model)
    angles = -torch.pi + 2.0 * torch.pi * torch.arange(32, dtype=torch.float64) / 32.0
    result = _slow_state_reconstruction(model, adapter, angles, horizon=128)
    assert result["success"] is True
    assert result["coverage"]["passed"] is True
    assert result["candidate_trajectory_coverage"] == 32
    assert result["resampled_state"].shape == (32, 2)
    assert np.all(result["candidate_count_per_trajectory"] > 0)


def test_full_checkpoint_identity_requires_matching_manifest_and_receipt(tmp_path: Path):
    run_dir = _tiny_run(tmp_path / "run")
    checkpoint = run_dir / "checkpoint.pt"
    protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
    fingerprint = protocol_fingerprint(protocol)
    campaign = "a" * 64
    bank_hash = "b" * 64
    model, payload = load_checkpoint(checkpoint, "cpu")
    payload["extra"].update(
        {
            "protocol_file_sha256": sha256_file(DEFAULT_PROTOCOL_PATH),
            "protocol_canonical_fingerprint": fingerprint,
            "campaign_identity": campaign,
            "evaluation_bank_sha256": bank_hash,
            "learning_rate": 0.01,
        }
    )
    # Re-save the identity-bearing payload, then freeze its hash in the run manifest.
    torch.save(payload, checkpoint)
    manifest = {
        "protocol_freeze_id": protocol["freeze_id"],
        "protocol_file_sha256": sha256_file(DEFAULT_PROTOCOL_PATH),
        "protocol_canonical_fingerprint": fingerprint,
        "campaign_identity": campaign,
        "evaluation_bank_sha256": bank_hash,
        "model_id": "gru",
        "model_seed": 100,
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    atomic_json(run_dir / "manifest.json", manifest)
    write_completion_receipt(
        run_dir / "completion_receipt.json",
        job_id="gru-seed100-lr0.01",
        artifacts=[checkpoint, run_dir / "manifest.json"],
        metadata={
            "campaign_identity": campaign,
            "protocol_canonical_fingerprint": fingerprint,
            "model_id": "gru",
            "model_seed": 100,
            "evaluation_bank_sha256": bank_hash,
        },
    )
    _verify_checkpoint_identity(
        protocol=protocol,
        protocol_path=DEFAULT_PROTOCOL_PATH.resolve(),
        run_dir=run_dir.resolve(),
        checkpoint=checkpoint.resolve(),
        checkpoint_payload=payload,
        model=model,
        evaluation_source={"sha256": bank_hash},
        campaign_identity=campaign,
        smoke=False,
    )
    with pytest.raises(ValueError, match="campaign_identity"):
        _verify_checkpoint_identity(
            protocol=protocol,
            protocol_path=DEFAULT_PROTOCOL_PATH.resolve(),
            run_dir=run_dir.resolve(),
            checkpoint=checkpoint.resolve(),
            checkpoint_payload=payload,
            model=model,
            evaluation_source={"sha256": bank_hash},
            campaign_identity="c" * 64,
            smoke=False,
        )


def test_cli_accepts_protocol_run_output_device_and_smoke(tmp_path: Path, capsys):
    run_dir = _tiny_run(tmp_path / "run")
    output = tmp_path / "cli-analysis"
    code = main(
        [
            "--protocol",
            str(DEFAULT_PROTOCOL_PATH),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--smoke",
        ]
    )
    assert code == 0
    message = json.loads(capsys.readouterr().out)
    assert message["status"] == "complete"
    assert Path(message["output_dir"]) == output.resolve()
    assert (output / "completion_receipt.json").is_file()
