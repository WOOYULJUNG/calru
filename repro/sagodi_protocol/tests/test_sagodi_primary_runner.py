from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import repro.sagodi_protocol.sagodi_primary_runner as primary_runner
from repro.sagodi_protocol.sagodi_primary_analysis import (
    cyclic_flow_reversal_topology,
)
from repro.sagodi_protocol.sagodi_primary_runner import (
    PrimaryAnalysisSpec,
    _asymptotic_summary,
    _bound_main_validation_metrics,
    _deterministic_knot_indices,
    _nearest_output_candidates,
    _nearest_output_candidates_torch,
    _publish_structural_not_estimable,
    _spectrum_chunk_payload_is_complete,
    compute_resumable_full_spectrum,
    finite_blank_memory,
    periodic_cubic_resample,
    reconstruct_slow_manifold,
)
from repro.sagodi_protocol.artifacts import sha256_file, strict_json_load, verify_completion_receipt
from repro.sagodi_protocol.state import StateAdapter


class IdentityRingCore(nn.Module):
    input_dim = 1
    state_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()), requires_grad=False)

    def init_state(self, batch: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(int(batch), 2, device=device)

    def step(self, input_tensor: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.scale * state

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return state


class DiagonalCore(IdentityRingCore):
    def __init__(self) -> None:
        super().__init__()
        self.diagonal = nn.Parameter(
            torch.tensor([1.0, 0.5], dtype=torch.float64), requires_grad=False
        )

    def step(self, input_tensor: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return state * self.diagonal


def _smoke_spec(*, count: int = 8, blank_horizon: int = 3) -> PrimaryAnalysisSpec:
    return PrimaryAnalysisSpec(
        trajectory_count=count,
        spline_count=count,
        task_horizon=4,
        blank_horizon=blank_horizon,
        spectrum_chunk_size=2,
        candidate_distance_chunk_size=3,
        smoke=True,
    )


def test_full_spec_rejects_scientific_count_override() -> None:
    with pytest.raises(ValueError, match="trajectory_count=1024"):
        PrimaryAnalysisSpec(trajectory_count=8).validate()


def test_numerical_failure_publishes_terminal_not_estimable_receipt(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "analysis"
    destination.mkdir()
    identity_file = destination / "analysis_identity.json"
    progress = destination / "progress.json"
    task = destination / "direct_task_trajectories.npz"
    identity_file.write_text("{}\n", encoding="utf-8")
    progress.write_text("{}\n", encoding="utf-8")
    task.write_bytes(b"task")

    summary_path = _publish_structural_not_estimable(
        destination=destination,
        base_summary={"schema_version": 1},
        progress=progress,
        completion=destination / "completion_receipt.json",
        identity="a" * 64,
        model_name="ca_lru",
        task_trajectory_path=task,
        failed_stage="projected_flow_and_fixed_point_topology",
        error=ValueError("undefined output angle"),
    )

    summary = strict_json_load(summary_path)
    assert summary["analysis_status"] == "structural_analysis_not_estimable"
    assert summary["structural_numerical_failure"]["seed_replacement"] is False
    valid, reason = verify_completion_receipt(
        destination / "completion_receipt.json",
        expected_metadata={
            "analysis_identity": "a" * 64,
            "analysis_status": "structural_analysis_not_estimable",
            "failed_stage": "projected_flow_and_fixed_point_topology",
        },
    )
    assert valid, reason


def _minimal_main_validation_binding(tmp_path: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-bound-metrics")
    protocol_source = tmp_path / "protocol.json"
    protocol_source.write_text("{}\n", encoding="utf-8")
    protocol: dict[str, object] = {
        "phase1_ring_pilot": {"protocol_track": "sagodi_primary_v3_main"},
        "evaluation": {"validation_trials": 2048, "id_test_trials": 2048},
    }
    extra: dict[str, object] = {
        "protocol_track": "sagodi_primary_v3_main",
        "protocol_file_sha256": sha256_file(protocol_source),
        "protocol_canonical_fingerprint": "b" * 64,
        "evaluation_bank_sha256": "a" * 64,
        "task_metrics": {"masked_nmse_db": -21.25, "masked_nmse": 0.0075},
    }
    return checkpoint, protocol_source, protocol, {"extra": extra}


def test_full_inclusion_uses_checkpoint_bound_2048_id_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(primary_runner, "protocol_fingerprint", lambda _: "b" * 64)
    checkpoint, protocol_source, protocol, payload = _minimal_main_validation_binding(
        tmp_path
    )
    metrics, binding = _bound_main_validation_metrics(
        checkpoint=checkpoint,
        checkpoint_payload=payload,
        protocol=protocol,
        protocol_source=protocol_source,
    )
    assert metrics["masked_nmse_db"] == -21.25
    assert binding["evaluation_bank_sha256"] == "a" * 64
    assert binding["source"] == "checkpoint_extra_bound_by_checkpoint_bytes"


def test_full_inclusion_fails_closed_without_bound_metric_or_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(primary_runner, "protocol_fingerprint", lambda _: "b" * 64)
    checkpoint, protocol_source, protocol, payload = _minimal_main_validation_binding(
        tmp_path
    )
    payload["extra"].pop("task_metrics")
    with pytest.raises(ValueError, match="lacks frozen task_metrics"):
        _bound_main_validation_metrics(
            checkpoint=checkpoint,
            checkpoint_payload=payload,
            protocol=protocol,
            protocol_source=protocol_source,
        )
    payload["extra"]["task_metrics"] = {"masked_nmse_db": -30.0}
    payload["extra"].pop("evaluation_bank_sha256")
    with pytest.raises(ValueError, match="evaluation-bank SHA-256"):
        _bound_main_validation_metrics(
            checkpoint=checkpoint,
            checkpoint_payload=payload,
            protocol=protocol,
            protocol_source=protocol_source,
        )


def test_periodic_cubic_resample_is_continuous_and_exact_at_knots() -> None:
    angle = torch.arange(8, dtype=torch.float64) * (2.0 * math.pi / 8.0)
    state = torch.stack((torch.cos(angle), torch.sin(angle), torch.cos(2 * angle)), dim=1)
    observed = periodic_cubic_resample(
        angle,
        state,
        torch.cat((angle, torch.tensor([0.0, 2.0 * math.pi], dtype=torch.float64))),
    )
    torch.testing.assert_close(observed[:8], state, atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(observed[-2], observed[-1], atol=1.0e-12, rtol=0.0)


def test_nearest_output_candidates_uses_decoded_radius_not_only_angle() -> None:
    candidates = np.asarray([[0.1, 0.0], [1.1, 0.2], [-1.0, 0.0]])
    index, distance = _nearest_output_candidates(
        candidates, np.asarray([0.0, math.pi]), chunk_size=2
    )
    assert index.tolist() == [1, 2]
    np.testing.assert_allclose(distance, [math.sqrt(0.05), 0.0], atol=1.0e-12)

    torch_index, torch_distance = _nearest_output_candidates_torch(
        torch.as_tensor(candidates, dtype=torch.float64),
        torch.tensor([0.0, math.pi], dtype=torch.float64),
        chunk_size=2,
    )
    assert torch_index.tolist() == index.tolist()
    np.testing.assert_allclose(torch_distance.numpy(), distance, atol=1.0e-12)


def test_knot_merge_is_cyclic_and_deterministic() -> None:
    selected = _deterministic_knot_indices(
        np.asarray([1.0e-10, 2.0 * math.pi - 1.0e-10, 1.0, 2.0, 3.0]),
        np.asarray([2, 1, 0, 0, 0]),
        np.asarray([0, 0, 1, 2, 3]),
        trajectory_count=8,
        tolerance=1.0e-6,
    )
    # The seam pair keeps time=1 rather than time=2; output indices are sorted
    # by the retained angle after merging.
    assert set(selected.tolist()) == {1, 2, 3, 4}


def test_stationary_trajectories_each_contribute_one_slow_candidate() -> None:
    count = 8
    angle = torch.arange(count, dtype=torch.float64) * (2.0 * math.pi / count)
    endpoint = torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)
    adapter = StateAdapter(IdentityRingCore().double())
    reconstruction = reconstruct_slow_manifold(
        None,  # model is not needed for a StepBaseline-style adapter
        adapter,
        endpoint,
        angle,
        _smoke_spec(count=count),
    )
    assert reconstruction.qa["candidate_count"] == count
    assert reconstruction.qa["stationary_trajectory_count"] == count
    assert reconstruction.selected_candidate_time.tolist() == [0] * count
    assert float(reconstruction.selected_output_distance.max()) < 1.0e-7
    torch.testing.assert_close(reconstruction.spline_state, endpoint, atol=1.0e-9, rtol=1.0e-9)


def test_full_spectrum_chunks_resume_and_preserve_complex_dtype(tmp_path: Path) -> None:
    core = DiagonalCore()
    adapter = StateAdapter(core)
    state = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=torch.float64,
    )
    progress = tmp_path / "progress.json"
    artifact, summary = compute_resumable_full_spectrum(
        state,
        adapter,
        output_dir=tmp_path,
        identity="a" * 64,
        chunk_size=2,
        progress_path=progress,
    )
    mtimes = {path: path.stat().st_mtime_ns for path in (tmp_path / "spectrum_chunks").glob("*.npz")}
    second, _ = compute_resumable_full_spectrum(
        state,
        adapter,
        output_dir=tmp_path,
        identity="a" * 64,
        chunk_size=2,
        progress_path=progress,
    )
    assert second == artifact
    assert mtimes == {
        path: path.stat().st_mtime_ns
        for path in (tmp_path / "spectrum_chunks").glob("*.npz")
    }
    with np.load(artifact, allow_pickle=False) as payload:
        assert np.iscomplexobj(payload["map_eigenvalues"])
        np.testing.assert_allclose(
            np.sort(payload["map_eigenvalues"].real, axis=1),
            np.asarray([[0.5, 1.0]] * 4),
            atol=1.0e-12,
        )
    assert summary["point_count"] == 4
    assert summary["state_dimension"] == 2
    assert summary["computation_methods"] == [
        "torch.func.vmap_jacrev_dense_JF_then_exact_spectral_shift"
    ]
    assert summary["fast_path_audit"][0]["status"] == (
        "passed_against_slow_core_first_point"
    )


def test_resumed_spectrum_recomputes_a_nonfinite_corrupted_chunk(
    tmp_path: Path,
) -> None:
    adapter = StateAdapter(DiagonalCore())
    state = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=torch.float64,
    )
    progress = tmp_path / "progress.json"
    compute_resumable_full_spectrum(
        state,
        adapter,
        output_dir=tmp_path,
        identity="b" * 64,
        chunk_size=2,
        progress_path=progress,
    )
    chunk = sorted((tmp_path / "spectrum_chunks").glob("*.npz"))[0]
    with np.load(chunk, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    corrupted = payload["map_eigenvalues"].copy()
    corrupted[0, 0] = np.nan + 0.0j
    payload["map_eigenvalues"] = corrupted
    with chunk.open("wb") as handle:
        np.savez_compressed(handle, **payload)

    artifact, summary = compute_resumable_full_spectrum(
        state,
        adapter,
        output_dir=tmp_path,
        identity="b" * 64,
        chunk_size=2,
        progress_path=progress,
    )
    with np.load(artifact, allow_pickle=False) as result:
        assert np.isfinite(result["map_eigenvalues"]).all()
        assert np.isfinite(result["vector_field_eigenvalues"]).all()
    assert summary["largest_real_part"]["count"] == state.shape[0]


def test_spectrum_chunk_validator_accepts_permuted_equivalent_complex_spectra() -> None:
    map_eigenvalues = np.asarray([[1.0 + 1.0j, 1.0 - 1.0j]])
    vector_eigenvalues = np.asarray([[0.0 - 1.0j, 0.0 + 1.0j]])
    payload = {
        "map_eigenvalues": map_eigenvalues,
        "vector_field_eigenvalues": vector_eigenvalues,
        "largest_real_part": np.asarray([0.0]),
        "second_largest_real_part": np.asarray([0.0]),
        "real_part_gap": np.asarray([0.0]),
        "map_spectral_radius": np.asarray([math.sqrt(2.0)]),
        "tau_largest": np.asarray([np.nan]),
        "tau_second": np.asarray([np.nan]),
        "computation_method": np.asarray(
            "reference_autograd_functional_dense_full_jacobian"
        ),
        "fast_path_audit_status": np.asarray("fallback"),
        "fast_path_audit_max_abs": np.asarray(np.nan),
        "fast_path_audit_tolerance": np.asarray(np.nan),
        "fallback_reason": np.asarray("test"),
    }
    assert _spectrum_chunk_payload_is_complete(
        payload, row_count=1, state_dimension=2
    )


def test_finite_blank_memory_includes_time_zero() -> None:
    count = 8
    angle = torch.arange(count, dtype=torch.float32) * (2.0 * math.pi / count)
    state = torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)
    adapter = StateAdapter(IdentityRingCore())
    arrays, summary = finite_blank_memory(
        None,
        adapter,
        state,
        angle,
        horizon=4,
    )
    assert arrays["predicted_angle"].shape == (count, 5)
    np.testing.assert_allclose(arrays["absolute_error"], 0.0, atol=4.0e-7)
    assert summary["time_indexing"] == "inclusive_0_through_blank_horizon"


def test_fixed_point_asymptotic_summary_uses_saddle_basin_widths() -> None:
    angle = torch.arange(8, dtype=torch.float64) * (2.0 * math.pi / 8.0)
    flow = torch.sin(2.0 * angle)
    topology = cyclic_flow_reversal_topology(angle, flow, zero_tolerance=1.0e-12)
    terminal = angle.clone()
    summary, arrays = _asymptotic_summary(topology, angle, terminal)
    assert summary["topology"] == "fixed_points"
    assert summary["stable_count"] == 2
    assert math.isclose(summary["shannon_entropy_nats"], math.log(2.0), abs_tol=1.0e-12)
    np.testing.assert_allclose(arrays["geometric_basin_proportions"], [0.5, 0.5])


def test_single_basin_asymptotic_summary_assigns_every_memory_to_it() -> None:
    angle = torch.arange(1024, dtype=torch.float64) * (2.0 * math.pi / 1024.0)
    topology = cyclic_flow_reversal_topology(
        angle, torch.sin(angle), zero_tolerance=1.0e-12
    )

    summary, arrays = _asymptotic_summary(topology, angle, angle)

    assert len(topology.stable) == 1
    assert len(topology.saddles) == 1
    assert summary["effective_basin_count"] == pytest.approx(1.0)
    assert summary["shannon_entropy_nats"] == pytest.approx(0.0)
    assert arrays["uniform_grid_basin_assignment"].tolist() == [0] * 1024


def test_limit_cycle_capacity_is_na_and_asymptotic_worst_error_is_pi() -> None:
    angle = torch.arange(8, dtype=torch.float64) * (2.0 * math.pi / 8.0)
    topology = cyclic_flow_reversal_topology(angle, torch.ones_like(angle))
    summary, _ = _asymptotic_summary(topology, angle, angle)
    assert summary["topology"] == "limit_cycle"
    assert summary["fixed_point_basin_capacity"] is None
    assert summary["asymptotic_mean_error_radians"] is None
    assert summary["asymptotic_maximum_error_radians"] == math.pi
