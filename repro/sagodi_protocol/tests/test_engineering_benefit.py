from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from repro.sagodi_protocol.artifacts import (
    canonical_hash,
    sha256_file,
    verify_completion_receipt,
)
from repro.sagodi_protocol.engineering_benefit_campaign import (
    ANALYSIS_ROLE,
    CAMPAIGN_SCOPE,
    CAMPAIGN_TYPE,
    EXPECTED_UTILITY_METRIC_KEYS,
    EngineeringRun,
    _metric_summary,
    _validate_launch_code_identity,
    verify_engineering_output,
)
from repro.sagodi_protocol.engineering_benefit_runner import (
    EngineeringFreezeError,
    _finite_summary,
    _generate_concatenated_gp_bank,
    _registered_utility_mean,
    _relative_rms_perturbations,
    load_engineering_freeze,
    run_engineering_benefit,
)
from repro.sagodi_protocol.models import (
    ModelConfig,
    build_protocol_model,
    checkpoint_payload,
)


PACKAGE = Path(__file__).resolve().parents[1]
FREEZE = PACKAGE / "engineering_benefit_freeze_v1.json"


def test_freeze_encodes_exact_no_gate_engineering_contract() -> None:
    payload, spec = load_engineering_freeze(FREEZE)
    assert payload["analysis_role"] == ANALYSIS_ROLE
    assert payload["claim_policy"]["binary_continuous_attractor_gates"] is False
    assert payload["claim_policy"]["expected_direction_pass_thresholds"] is False
    assert spec.trial_count == 256
    assert spec.task_horizon == 256
    assert spec.temporal_multipliers == (1, 2, 4, 8, 16)
    assert spec.velocity_scales == (1.0, 2.0, 4.0)
    assert spec.perturbation_magnitudes == (0.0, 0.01, 0.1, 1.0)
    assert spec.perturbation_horizon_multipliers == (1, 4, 16)
    _, smoke = load_engineering_freeze(FREEZE, smoke=True)
    assert smoke.trial_count == 8
    assert smoke.task_horizon == 8
    assert smoke.smoke is True


def test_freeze_validation_fails_closed_on_claim_gate_change(tmp_path: Path) -> None:
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    payload["claim_policy"]["binary_continuous_attractor_gates"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EngineeringFreezeError, match="claim policy"):
        load_engineering_freeze(changed)


def test_smoke_bank_uses_deterministic_independent_prefix_blocks() -> None:
    _, spec = load_engineering_freeze(FREEZE, smoke=True)
    first = _generate_concatenated_gp_bank(
        spec, device=torch.device("cpu"), dtype=torch.float32
    )
    second = _generate_concatenated_gp_bank(
        spec, device=torch.device("cpu"), dtype=torch.float32
    )
    velocity, angle, memory, metadata = first
    torch.testing.assert_close(velocity, second[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(angle, second[1], rtol=0.0, atol=0.0)
    assert velocity.shape == (128, 8, 1)
    assert angle.shape == (128, 8)
    assert memory.shape == (8, 2)
    assert len(set(metadata["block_derived_seeds"])) == 16
    expected = torch.atan2(memory[:, 1], memory[:, 0])[None, :] + 0.1 * torch.cumsum(
        velocity[:, :, 0], dim=0
    )
    torch.testing.assert_close(angle, expected)


def test_relative_rms_perturbation_has_exact_registered_magnitude() -> None:
    state = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 0.5, 3.0]],
        dtype=torch.float64,
    )
    magnitudes = (0.0, 0.01, 0.1, 1.0)
    perturbation, realized = _relative_rms_perturbations(
        state, magnitudes, seed=123
    )
    assert perturbation.shape == (4, 2, 4)
    expected = torch.tensor(magnitudes, dtype=torch.float64)[:, None].expand(4, 2)
    torch.testing.assert_close(realized, expected, rtol=1e-12, atol=1e-12)
    repeated, _ = _relative_rms_perturbations(state, magnitudes, seed=123)
    torch.testing.assert_close(perturbation, repeated, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_utility_mean_is_null_if_any_registered_trial_is_nonfinite(
    bad_value: float,
) -> None:
    summary = _finite_summary(np.asarray([0.25, bad_value], dtype=np.float64))
    assert summary["finite_count"] == 1
    assert summary["mean"] == 0.25  # retained for diagnosis only
    assert _registered_utility_mean(summary) is None
    finite = _finite_summary(np.asarray([0.25, 0.75], dtype=np.float64))
    assert _registered_utility_mean(finite) == 0.5


def test_angle_utility_is_null_for_undefined_output_radius() -> None:
    summary = _finite_summary(np.asarray([0.0, 0.0], dtype=np.float64))
    assert (
        _registered_utility_mean(
            summary,
            corresponding_radii=(np.asarray([1.0, 0.0]),),
            radius_epsilon=1e-12,
        )
        is None
    )
    assert _registered_utility_mean(
        summary,
        corresponding_radii=(np.asarray([1.0, 1.0]),),
        radius_epsilon=1e-12,
    ) == pytest.approx(0.0)


def test_campaign_schema_has_exact_stable_join_metrics() -> None:
    assert CAMPAIGN_TYPE == "calru_engineering_benefit_v1"
    assert CAMPAIGN_SCOPE.endswith("no_binary_ca_gates")
    assert len(EXPECTED_UTILITY_METRIC_KEYS) == 40
    assert len(set(EXPECTED_UTILITY_METRIC_KEYS)) == 40
    assert "temporal/16T/prefix_mean_error_radians" in EXPECTED_UTILITY_METRIC_KEYS
    assert (
        "perturbation/relative_rms_0/16T/memory_mean_error_radians"
        in EXPECTED_UTILITY_METRIC_KEYS
    )
    summary = _metric_summary([1.0, 2.0, None])
    assert summary["seed_count"] == 3
    assert summary["finite_seed_count"] == 2
    assert summary["mean"] == 1.5


def test_full_campaign_requires_exact_clean_main_commit() -> None:
    parent = SimpleNamespace(binding={"main_code_commit": "a" * 40})
    _validate_launch_code_identity(
        {"code_commit": "a" * 40, "worktree_dirty": False},
        parent,
        smoke=False,
    )
    with pytest.raises(RuntimeError, match="exact main-training code commit"):
        _validate_launch_code_identity(
            {"code_commit": "b" * 40, "worktree_dirty": False},
            parent,
            smoke=False,
        )
    with pytest.raises(RuntimeError, match="clean committed worktree"):
        _validate_launch_code_identity(
            {"code_commit": "a" * 40, "worktree_dirty": True},
            parent,
            smoke=False,
        )
    _validate_launch_code_identity(
        {"code_commit": "b" * 40, "worktree_dirty": True},
        parent,
        smoke=True,
    )


def test_single_checkpoint_smoke_is_receipted_resumable_and_stores_individual_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(7)
    model = build_protocol_model(
        ModelConfig("ca_lru", input_dim=1, output_dim=2, width=8)
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(checkpoint_payload(model, {"model_seed": 0}), checkpoint)
    freeze_payload, smoke_spec = load_engineering_freeze(FREEZE, smoke=True)
    campaign_identity = "c" * 64
    freeze_fingerprint = canonical_hash(freeze_payload)
    monkeypatch.setenv("CALRU_CAMPAIGN_SCIENTIFIC_IDENTITY", campaign_identity)
    monkeypatch.setenv("CALRU_PROTOCOL_FINGERPRINT", freeze_fingerprint)
    monkeypatch.setenv("CALRU_RUN_ID", "engineering_benefit__ca_lru__seed00")
    monkeypatch.setenv("CALRU_STAGE", "engineering_benefit_evaluation")
    output = tmp_path / "engineering"
    summary_path = run_engineering_benefit(
        checkpoint_path=checkpoint,
        freeze_path=FREEZE,
        output_dir=output,
        model_id="ca_lru",
        model_seed=0,
        device="cpu",
        smoke=True,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["analysis_role"] == ANALYSIS_ROLE
    assert summary["claim_thresholds"] is None
    assert summary["seed_excluded"] is False
    assert set(summary["utility_metrics"]) == set(EXPECTED_UTILITY_METRIC_KEYS)
    valid, reason = verify_completion_receipt(output / "completion_receipt.json")
    assert valid, reason
    campaign_run = EngineeringRun(
        model_id="ca_lru",
        model_seed=0,
        checkpoint=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        training_receipt=tmp_path / "unused-training-receipt.json",
        training_receipt_sha256="d" * 64,
    )
    campaign_manifest = {
        "scientific_identity": campaign_identity,
        "freeze_canonical_fingerprint": freeze_fingerprint,
        "freeze_file_sha256": sha256_file(FREEZE),
        "spec": asdict(smoke_spec),
    }
    campaign_valid, campaign_reason, bindings = verify_engineering_output(
        output, campaign_run, campaign_manifest
    )
    assert campaign_valid, campaign_reason
    assert bindings is not None
    assert bindings["summary_sha256"] == sha256_file(summary_path)
    with np.load(output / "individual_errors.npz", allow_pickle=False) as arrays:
        assert arrays["temporal_absolute_error"].shape == (128, 8)
        assert arrays["velocity_absolute_error"].shape == (3, 8, 8)
        assert arrays["perturbation_memory_absolute_error"].shape == (4, 3, 8)
        assert arrays["perturbation_clean_paired_absolute_error"].shape == (4, 3, 8)
        np.testing.assert_allclose(
            arrays["perturbation_realized_relative_l2"],
            np.broadcast_to(
                np.asarray([0.0, 0.01, 0.1, 1.0], dtype=np.float32)[:, None],
                (4, 8),
            ),
            atol=2e-6,
        )
        np.testing.assert_allclose(
            arrays["perturbation_clean_paired_absolute_error"][0],
            0.0,
            atol=0.0,
        )
    before = (output / "completion_receipt.json").stat().st_mtime_ns
    resumed = run_engineering_benefit(
        checkpoint_path=checkpoint,
        freeze_path=FREEZE,
        output_dir=output,
        model_id="ca_lru",
        model_seed=0,
        device="cpu",
        smoke=True,
    )
    assert resumed == summary_path
    assert (output / "completion_receipt.json").stat().st_mtime_ns == before


def test_nonfinite_checkpoint_completes_with_null_utilities_and_campaign_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_protocol_model(
        ModelConfig("ca_lru", input_dim=1, output_dim=2, width=8)
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    checkpoint = tmp_path / "nonfinite.pt"
    torch.save(checkpoint_payload(model, {"model_seed": 1}), checkpoint)
    freeze_payload, smoke_spec = load_engineering_freeze(FREEZE, smoke=True)
    campaign_identity = "e" * 64
    freeze_fingerprint = canonical_hash(freeze_payload)
    monkeypatch.setenv("CALRU_CAMPAIGN_SCIENTIFIC_IDENTITY", campaign_identity)
    monkeypatch.setenv("CALRU_PROTOCOL_FINGERPRINT", freeze_fingerprint)
    monkeypatch.setenv("CALRU_RUN_ID", "engineering_benefit__ca_lru__seed01")
    monkeypatch.setenv("CALRU_STAGE", "engineering_benefit_evaluation")
    output = tmp_path / "engineering-nonfinite"
    summary_path = run_engineering_benefit(
        checkpoint_path=checkpoint,
        freeze_path=FREEZE,
        output_dir=output,
        model_id="ca_lru",
        model_seed=1,
        device="cpu",
        smoke=True,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert all(value is None for value in summary["utility_metrics"].values())
    assert summary["nonfinite_diagnostics"]["temporal_error_nonfinite_count"] > 0
    assert summary["output_radius_diagnostics"]["temporal_nonfinite_radius_count"] > 0
    campaign_run = EngineeringRun(
        model_id="ca_lru",
        model_seed=1,
        checkpoint=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        training_receipt=tmp_path / "unused-training-receipt.json",
        training_receipt_sha256="f" * 64,
    )
    valid, reason, _ = verify_engineering_output(
        output,
        campaign_run,
        {
            "scientific_identity": campaign_identity,
            "freeze_canonical_fingerprint": freeze_fingerprint,
            "freeze_file_sha256": sha256_file(FREEZE),
            "spec": asdict(smoke_spec),
        },
    )
    assert valid, reason
