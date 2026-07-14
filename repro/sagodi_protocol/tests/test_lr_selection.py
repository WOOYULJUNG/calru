from __future__ import annotations

import copy
import json
import signal
import sys
from pathlib import Path

import numpy as np
import pytest

from repro.sagodi_protocol import lr_selection as lr_selection_module
from repro.sagodi_protocol.artifacts import (
    atomic_bytes,
    atomic_json,
    canonical_hash,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from repro.sagodi_protocol.config import (
    SAGODI_LR_SELECTION_PROTOCOL_PATH,
    load_protocol,
    protocol_fingerprint,
)
from repro.sagodi_protocol.lr_selection import (
    COMPLETE_MARKER_NAME,
    COMPLETION_RECEIPT_NAME,
    EXPECTED_LEARNING_RATES,
    EXPECTED_MODELS,
    EXPECTED_SELECTION_SEEDS,
    IncompleteSelectionError,
    MANIFEST_NAME,
    MATRIX_NAME,
    SELECTION_RECEIPT_NAME,
    SUMMARY_NAME,
    SelectionProtocolError,
    _build_manifest,
    _child_environment,
    _configure_linux_parent_death_signal,
    _final_receipt_metadata,
    _phase0_artifact_names,
    _preserve_finalization,
    _receipt_metadata,
    _recorded_live_processes,
    _recover_training_attempt,
    _selection_rule_payload,
    _selection_runs_from_manifest,
    _verify_complete_marker,
    _verify_manifest_identity,
    _verify_phase0_output,
    _verify_training_output,
    aggregate_lr_selection,
    build_selection_plan,
    strict_aggregate_lr_selection,
    verify_selection_completion,
    write_selection_artifacts,
)


def _rows(loss_by_model_lr: dict[tuple[str, float], float] | None = None):
    loss_by_model_lr = loss_by_model_lr or {}
    rows = []
    for model_index, model in enumerate(EXPECTED_MODELS):
        for lr_index, lr in enumerate(EXPECTED_LEARNING_RATES):
            base = loss_by_model_lr.get((model, lr), 1.0 + model_index + lr_index)
            for seed_index, seed in enumerate(EXPECTED_SELECTION_SEEDS):
                rows.append(
                    {
                        "model_id": model,
                        "model_seed": seed,
                        "learning_rate": lr,
                        "status": "complete",
                        "completed_updates": 100,
                        "loss_at_required_update": base + seed_index * 0.01,
                    }
                )
    return rows


def test_campaign_plan_is_exact_validated_80_run_cross_product():
    protocol = load_protocol(SAGODI_LR_SELECTION_PROTOCOL_PATH)
    plan = build_selection_plan(protocol)

    assert len(plan) == 80
    assert len({run.run_id for run in plan}) == 80
    assert {run.model_id for run in plan} == set(EXPECTED_MODELS)
    assert {run.model_seed for run in plan} == set(EXPECTED_SELECTION_SEEDS)
    assert {run.learning_rate for run in plan} == set(EXPECTED_LEARNING_RATES)
    assert {
        run.model_id: run.hidden_width for run in plan
    } == {
        "ca_lru": 96,
        "no_rp": 96,
        "gru_sagodi_width96": 96,
        "gru_sagodi_param135": 135,
    }
    assert all(run.required_update == 100 for run in plan)


def test_campaign_plan_rejects_rule_or_seed_tamper():
    protocol = load_protocol(SAGODI_LR_SELECTION_PROTOCOL_PATH)
    tampered_rule = copy.deepcopy(protocol)
    tampered_rule["phase1_ring_pilot"]["training"]["learning_rate"][
        "selection_rule"
    ]["tie_policy"] = "largest_lr"
    with pytest.raises((SelectionProtocolError, ValueError)):
        build_selection_plan(tampered_rule)

    tampered_seed = copy.deepcopy(protocol)
    tampered_seed["seed_policy"]["selection_model_seeds"][-1] = 9999
    with pytest.raises((SelectionProtocolError, ValueError)):
        build_selection_plan(tampered_seed)


def test_aggregation_uses_arithmetic_mean_at_update_100_only():
    losses = {}
    for model in EXPECTED_MODELS:
        losses[(model, 1e-2)] = 4.0
        losses[(model, 1e-3)] = 1.0
        losses[(model, 1e-4)] = 2.0
        losses[(model, 1e-5)] = 3.0
    result = strict_aggregate_lr_selection(_rows(losses))

    assert result["complete"] is True
    for winner in result["winners"].values():
        assert winner["learning_rate"] == pytest.approx(1e-3)
        # Five values 1.00, 1.01, ..., 1.04 have arithmetic mean 1.02.
        assert winner["mean_online_training_loss_at_required_update"] == pytest.approx(
            1.02
        )


def test_exact_mean_tie_is_broken_by_smaller_numeric_learning_rate():
    losses = {}
    for model in EXPECTED_MODELS:
        losses[(model, 1e-2)] = 5.0
        losses[(model, 1e-3)] = 1.0
        losses[(model, 1e-4)] = 1.0
        losses[(model, 1e-5)] = 4.0
    result = strict_aggregate_lr_selection(_rows(losses))

    for winner in result["winners"].values():
        assert winner["learning_rate"] == pytest.approx(1e-4)


@pytest.mark.parametrize("failure", ["missing", "failed", "nonfinite"])
def test_one_bad_seed_makes_entire_model_lr_ineligible(failure: str):
    rows = _rows()
    target = next(
        row
        for row in rows
        if row["model_id"] == "ca_lru"
        and row["learning_rate"] == 1e-2
        and row["model_seed"] == 1102
    )
    if failure == "missing":
        rows.remove(target)
    elif failure == "failed":
        target["status"] = "failed"
    else:
        target["loss_at_required_update"] = float("nan")

    result = aggregate_lr_selection(rows)
    candidate = next(
        item
        for item in result["candidates"]["ca_lru"]
        if item["learning_rate"] == 1e-2
    )
    assert candidate["eligible"] is False
    assert candidate["mean_online_training_loss_at_required_update"] is None
    assert any("seed1102" in reason for reason in candidate["ineligibility_reasons"])


def test_strict_aggregation_fails_when_a_model_has_no_eligible_lr():
    rows = _rows()
    for row in rows:
        if row["model_id"] == "gru_sagodi_param135":
            row["status"] = "failed"
    with pytest.raises(IncompleteSelectionError, match="gru_sagodi_param135"):
        strict_aggregate_lr_selection(rows)


def test_strict_aggregation_rejects_79_rows_even_if_all_models_have_winners():
    rows = _rows()
    rows.remove(
        next(
            row
            for row in rows
            if row["model_id"] == "ca_lru"
            and row["learning_rate"] == 1e-2
            and row["model_seed"] == 1100
        )
    )
    # The diagnostic aggregator may still expose winners from other eligible
    # LRs, but the strict publication path must never do so.
    assert aggregate_lr_selection(rows)["complete"] is True
    with pytest.raises(IncompleteSelectionError, match="exactly 80"):
        strict_aggregate_lr_selection(rows)


def _minimal_campaign_manifest(*, smoke: bool = True) -> dict:
    return {
        "scientific_identity": "a" * 64,
        "protocol_canonical_fingerprint": "b" * 64,
        "smoke": smoke,
        "code": {"code_commit": "test-commit"},
        "resolved_task_spec_sha256": "c" * 64,
        "evaluation_bank": {"sha256": "d" * 64},
        "gpus": [0],
    }


def _write_fake_training_output(
    output: Path, run, manifest: dict
) -> None:
    atomic_bytes(output / "checkpoint.pt", b"checkpoint")
    np.savez_compressed(
        output / "training_trace.npz",
        step=np.asarray([1, 2], dtype=np.int64),
        masked_mse=np.asarray([0.5, 0.25], dtype=np.float32),
    )
    atomic_json(output / "task_metrics.json", {"train_loss_last": 0.25})
    atomic_json(output / "rp_trace.json", [])
    model_metadata = {
        "parameters_total": run.parameter_count,
        "model_config": {"width": run.hidden_width},
    }
    atomic_json(
        output / "config.json",
        {
            "physical_gpu_id": "0",
            "phase0_state_spec_sha256": None,
            "train_spec": {
                "model_name": run.model_id,
                "model_seed": run.model_seed,
                "learning_rate": run.learning_rate,
            },
            "model": model_metadata,
            "training": {
                "steps": 2,
                "batch_size": 4,
                "rp_enabled_by_protocol": False,
                "expected_rp_steps": [],
            },
        },
    )
    atomic_json(
        output / "manifest.json",
        {
            "checkpoint_sha256": sha256_file(output / "checkpoint.pt"),
            "campaign_identity": manifest["scientific_identity"],
            "code_commit": manifest["code"]["code_commit"],
            "protocol_canonical_fingerprint": manifest[
                "protocol_canonical_fingerprint"
            ],
            "model_id": run.model_id,
            "model_seed": run.model_seed,
            "parameter_count": run.parameter_count,
            "architecture_metadata": model_metadata,
            "resolved_task_spec_sha256": manifest["resolved_task_spec_sha256"],
            "evaluation_bank_sha256": manifest["evaluation_bank"]["sha256"],
            "state_spec_sha256": None,
            "physical_gpu_id": "0",
            "rp_schedule": {"expected_steps": [], "actual_steps": [], "calls": 0},
        },
    )
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=run.receipt_job_id,
        artifacts=[
            output / name
            for name in (
                "config.json",
                "checkpoint.pt",
                "training_trace.npz",
                "task_metrics.json",
                "rp_trace.json",
                "manifest.json",
            )
        ],
        metadata={
            **_receipt_metadata(manifest, "training", run.run_id),
            "physical_gpu_id": "0",
            "rp_calls": 0,
        },
    )


def test_training_verifier_recursively_detects_rp_trace_tamper(tmp_path: Path):
    run = build_selection_plan(load_protocol(SAGODI_LR_SELECTION_PROTOCOL_PATH))[0]
    manifest = _minimal_campaign_manifest()
    _write_fake_training_output(tmp_path, run, manifest)
    valid, reason, _ = _verify_training_output(
        tmp_path, run, manifest, root=tmp_path.parent
    )
    assert valid, reason

    atomic_json(tmp_path / "rp_trace.json", [{"forged": True}])
    # Model an attacker who also regenerates the child receipt.  The verifier
    # must inspect RP semantics, not merely notice the stale hash.
    write_completion_receipt(
        tmp_path / "completion_receipt.json",
        job_id=run.receipt_job_id,
        artifacts=[
            tmp_path / name
            for name in (
                "config.json",
                "checkpoint.pt",
                "training_trace.npz",
                "task_metrics.json",
                "rp_trace.json",
                "manifest.json",
            )
        ],
        metadata={
            **_receipt_metadata(manifest, "training", run.run_id),
            "physical_gpu_id": "0",
            "rp_calls": 0,
        },
    )
    valid, reason, _ = _verify_training_output(
        tmp_path, run, manifest, root=tmp_path.parent
    )
    assert not valid
    assert "exact empty JSON array" in reason


def test_phase0_verifier_detects_state_spec_tamper_and_exact_set(tmp_path: Path):
    manifest = _minimal_campaign_manifest()
    for name in sorted(_phase0_artifact_names()):
        path = tmp_path / name
        if name == "manifest.json":
            atomic_json(
                path,
                {
                    "protocol_canonical_fingerprint": manifest[
                        "protocol_canonical_fingerprint"
                    ]
                },
            )
        elif name == "phase0_gate.json":
            atomic_json(path, {"passed": True})
        else:
            atomic_bytes(path, b"phase0-bound")
    write_completion_receipt(
        tmp_path / "completion_receipt.json",
        job_id="phase0_state_audit",
        artifacts=[tmp_path / name for name in sorted(_phase0_artifact_names())],
        metadata=_receipt_metadata(manifest, "phase0", "phase0"),
    )
    valid, reason = _verify_phase0_output(tmp_path, manifest)
    assert valid, reason

    atomic_bytes(tmp_path / "model=ca_lru" / "state_spec.json", b"tampered")
    valid, reason = _verify_phase0_output(tmp_path, manifest)
    assert not valid
    assert "hash mismatch" in reason


def test_manifest_identity_reloads_exact_on_disk_manifest(tmp_path: Path):
    signed = {
        "smoke": True,
        "run_matrix": [],
        "freeze_eligible": False,
        "freeze_status": "pending_all_80_verified_success_receipts",
    }
    manifest = {
        **signed,
        "scientific_identity_payload": signed,
        "scientific_identity": canonical_hash(signed),
    }
    atomic_json(tmp_path / "lr_selection_manifest.json", manifest)
    valid, reason = _verify_manifest_identity(tmp_path, manifest)
    assert valid, reason

    tampered = dict(manifest)
    tampered["unsigned_injection"] = True
    atomic_json(tmp_path / "lr_selection_manifest.json", tampered)
    valid, reason = _verify_manifest_identity(tmp_path, manifest)
    assert not valid
    assert "on-disk" in reason


def test_recorded_live_process_and_unverifiable_running_record_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    identity = {"pid": 321, "start_time_ticks": 99}
    monkeypatch.setattr(
        "repro.sagodi_protocol.lr_selection._process_identity",
        lambda pid: identity if pid == 321 else None,
    )
    assert _recorded_live_processes(
        {
            "phase0": {
                "state": "running",
                "pid": 321,
                "process_identity": identity,
            }
        }
    ) == ["phase0:pid=321"]
    assert _recorded_live_processes(
        {"phase0": {"state": "running", "pid": 321}}
    ) == ["phase0:pid=321:unverifiable-identity"]


def test_child_environment_forces_determinism_and_physical_gpu_mapping(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "wrong")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,4")
    manifest = {
        "scientific_identity": "a" * 64,
        "protocol_canonical_fingerprint": "b" * 64,
    }
    env = _child_environment(manifest, stage="training", run_id="run", gpu=2)
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["CALRU_PHYSICAL_GPU_ID"] == "2"


def test_manifest_starts_pending_and_never_claims_freeze_eligibility():
    protocol = load_protocol(SAGODI_LR_SELECTION_PROTOCOL_PATH)
    plan = build_selection_plan(protocol)
    manifest = _build_manifest(
        protocol=protocol,
        protocol_path=SAGODI_LR_SELECTION_PROTOCOL_PATH,
        fingerprint=protocol_fingerprint(protocol),
        evaluation_bank={"path": "bank.npz", "sha256": "d" * 64},
        plan=plan,
        source_hashes={},
        git_state={"code_commit": "test", "worktree_dirty": False},
        environment={},
        python=sys.executable,
        gpus=(0,),
        smoke=False,
    )
    assert manifest["freeze_eligible"] is False
    assert manifest["freeze_status"] == (
        "pending_all_80_verified_success_receipts"
    )
    assert manifest["scientific_identity_payload"]["freeze_eligible"] is False


def test_linux_parent_death_signal_installs_sigkill_and_checks_ppid_race():
    calls = []

    def fake_prctl(*args):
        calls.append(args)
        return 0

    assert _configure_linux_parent_death_signal(
        1234, prctl_call=fake_prctl, getppid=lambda: 1234
    ) is True
    assert calls == [(1, signal.SIGKILL, 0, 0, 0)]
    assert _configure_linux_parent_death_signal(
        1234, prctl_call=fake_prctl, getppid=lambda: 4321
    ) is False

    with pytest.raises(OSError, match="PR_SET_PDEATHSIG"):
        _configure_linux_parent_death_signal(
            1234, prctl_call=lambda *_args: -1, getppid=lambda: 1234
        )


def test_resume_recovers_verified_completed_attempt_before_relaunch(tmp_path: Path):
    run = build_selection_plan(load_protocol(SAGODI_LR_SELECTION_PROTOCOL_PATH))[0]
    manifest = _minimal_campaign_manifest()
    attempt = (
        tmp_path
        / "attempts"
        / "lr_selection_training"
        / run.run_id
        / "attempt-1"
    )
    _write_fake_training_output(attempt, run, manifest)
    recovered, reason = _recover_training_attempt(
        root=tmp_path,
        run=run,
        manifest=manifest,
        status={
            "jobs": {
                run.run_id: {
                    "state": "running",
                    "attempt_dir": str(attempt),
                }
            }
        },
    )
    assert recovered, reason
    assert not attempt.exists()
    assert (tmp_path / "runs" / run.run_id / "completion_receipt.json").is_file()


def test_manifest_plan_reader_requires_exact_frozen_eighty_runs():
    plan = build_selection_plan(load_protocol(SAGODI_LR_SELECTION_PROTOCOL_PATH))
    assert _selection_runs_from_manifest(
        {"run_matrix": [run.payload() for run in plan]}
    ) == plan
    tampered = [run.payload() for run in plan[:-1]]
    with pytest.raises(ValueError, match="80-run"):
        _selection_runs_from_manifest({"run_matrix": tampered})


def _minimal_pending_final_manifest() -> dict:
    signed = {
        "protocol_canonical_fingerprint": "b" * 64,
        "smoke": True,
        "selection_rule": {"rule_id": "test"},
        "run_matrix": [],
        "freeze_eligible": False,
        "freeze_status": "pending_all_80_verified_success_receipts",
    }
    return {
        **signed,
        "scientific_identity_payload": signed,
        "scientific_identity": canonical_hash(signed),
    }


def _minimal_final_summary(manifest: dict) -> dict:
    return {
        "campaign_scientific_identity": manifest["scientific_identity"],
        "selection_rule": manifest["selection_rule"],
        "selection_performed": False,
        "winners": {},
        "run_count": 80,
        "successful_receipt_count": 80,
    }


def test_complete_marker_is_receipt_bound_and_partial_finalization_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = _minimal_pending_final_manifest()
    summary = _minimal_final_summary(manifest)
    atomic_json(tmp_path / MANIFEST_NAME, manifest)

    original_atomic_json = lr_selection_module.atomic_json

    def crash_before_complete(path, payload):
        if Path(path).name == COMPLETE_MARKER_NAME:
            raise RuntimeError("simulated crash before COMPLETE")
        return original_atomic_json(path, payload)

    monkeypatch.setattr(lr_selection_module, "atomic_json", crash_before_complete)
    with pytest.raises(RuntimeError, match="before COMPLETE"):
        write_selection_artifacts(tmp_path, manifest, summary, b"run_id\n")
    assert (tmp_path / SELECTION_RECEIPT_NAME).is_file()
    assert not (tmp_path / COMPLETE_MARKER_NAME).exists()
    assert not (tmp_path / COMPLETION_RECEIPT_NAME).exists()

    # Even a legacy-style completion receipt re-signed without COMPLETE is
    # rejected explicitly by the final verifier before child traversal.
    write_completion_receipt(
        tmp_path / COMPLETION_RECEIPT_NAME,
        job_id="lr_selection_campaign_complete",
        artifacts=[
            tmp_path / MANIFEST_NAME,
            tmp_path / SUMMARY_NAME,
            tmp_path / MATRIX_NAME,
            tmp_path / SELECTION_RECEIPT_NAME,
        ],
        metadata=_final_receipt_metadata(manifest, summary),
    )
    valid, reason = verify_selection_completion(tmp_path, manifest)
    assert not valid
    assert "COMPLETE" in reason

    monkeypatch.setattr(lr_selection_module, "atomic_json", original_atomic_json)
    _preserve_finalization(tmp_path)
    for name in (
        SUMMARY_NAME,
        MATRIX_NAME,
        SELECTION_RECEIPT_NAME,
        COMPLETION_RECEIPT_NAME,
        COMPLETE_MARKER_NAME,
    ):
        assert not (tmp_path / name).exists()

    write_selection_artifacts(tmp_path, manifest, summary, b"run_id\n")
    valid, reason = _verify_complete_marker(tmp_path, manifest, summary)
    assert valid, reason
    valid, reason = verify_completion_receipt(
        tmp_path / COMPLETION_RECEIPT_NAME,
        expected_job_id="lr_selection_campaign_complete",
        expected_metadata=_final_receipt_metadata(manifest, summary),
    )
    assert valid, reason
    completion = strict_json_load(tmp_path / COMPLETION_RECEIPT_NAME)
    assert set(completion["artifacts"]) == {
        MANIFEST_NAME,
        SUMMARY_NAME,
        MATRIX_NAME,
        SELECTION_RECEIPT_NAME,
        COMPLETE_MARKER_NAME,
    }

    marker = strict_json_load(tmp_path / COMPLETE_MARKER_NAME)
    marker["verified_success_receipt_count"] = 79
    atomic_json(tmp_path / COMPLETE_MARKER_NAME, marker)
    valid, reason = _verify_complete_marker(tmp_path, manifest, summary)
    assert not valid
    assert "differs" in reason
    valid, reason = verify_completion_receipt(tmp_path / COMPLETION_RECEIPT_NAME)
    assert not valid
    assert "hash mismatch" in reason
