from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from repro.sagodi_protocol.artifacts import (
    atomic_json,
    canonical_hash,
    sha256_file,
    verify_completion_receipt,
    write_completion_receipt,
)
from repro.sagodi_protocol.aggregation import (
    aggregation_specification,
    build_pilot_aggregation,
)
from repro.sagodi_protocol.orchestrate import (
    ActiveJob,
    Job,
    _materialize_evaluation_bank,
    _materialize_perturbation_bank,
    _acquire_campaign_lock,
    _prepare_root,
    _preserve_invalid_output,
    _receipt_metadata,
    _training_jobs,
    _analysis_jobs,
    _terminate_active,
    _validated_gpu_ids,
    _write_pilot_aggregation,
    compute_campaign_completion,
    run_campaign,
    verify_campaign_output_receipt,
)
from repro.sagodi_protocol.config import DEFAULT_PROTOCOL_PATH, load_protocol
from repro.sagodi_protocol.status import main as status_main
from repro.sagodi_protocol.status import summarize


def _manifest() -> dict:
    expectations = {
        "phase0": {
            "job_id": "phase0_state_audit",
            "stage": "phase0",
            "run_id": "phase0",
            "output": "phase0",
        },
        "training:r0": {
            "job_id": "gru-seed100-lr0.01",
            "stage": "training",
            "run_id": "r0",
            "output": "runs/r0",
        },
        "analysis:r0": {
            "job_id": "phase1-analysis-r0",
            "stage": "analysis",
            "run_id": "r0",
            "output": "analysis/r0",
        },
    }
    payload = {
        "campaign_id": "test-campaign",
        "protocol_canonical_fingerprint": "f" * 64,
        "receipt_expectations": expectations,
    }
    return {
        "schema_version": 2,
        **payload,
        "scientific_identity_payload": payload,
        "scientific_identity": canonical_hash(payload),
    }


def _write_identity_receipt(
    root: Path,
    manifest: dict,
    key: str,
    *,
    metadata_override: dict | None = None,
) -> Path:
    expectation = manifest["receipt_expectations"][key]
    output = root / expectation["output"]
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "artifact.json"
    atomic_json(artifact, {"key": key})
    metadata = _receipt_metadata(
        manifest, expectation["stage"], expectation["run_id"]
    )
    metadata.update(metadata_override or {})
    artifacts = [artifact]
    if expectation["stage"] == "training":
        checkpoint = output / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint-r0")
        training_manifest = output / "manifest.json"
        atomic_json(
            training_manifest,
            {
                "checkpoint_sha256": sha256_file(checkpoint),
                "campaign_identity": manifest["scientific_identity"],
                "protocol_canonical_fingerprint": manifest[
                    "protocol_canonical_fingerprint"
                ],
            },
        )
        artifacts.extend([checkpoint, training_manifest])
    elif expectation["stage"] == "analysis":
        checkpoint = root / "runs" / expectation["run_id"] / "checkpoint.pt"
        metadata.setdefault("checkpoint_sha256", sha256_file(checkpoint))
        claim = output / "claim_gate.json"
        if claim.is_file():
            artifacts.append(claim)
    receipt = output / "completion_receipt.json"
    write_completion_receipt(
        receipt,
        job_id=expectation["job_id"],
        artifacts=artifacts,
        metadata=metadata,
    )
    return receipt


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_atomic_json_rejects_nonfinite_without_creating_file(tmp_path: Path, nonfinite: float):
    destination = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        atomic_json(destination, {"value": nonfinite})
    assert not destination.exists()


def test_receipt_rejects_nonfinite_metadata_without_creating_receipt(tmp_path: Path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("ok")
    receipt = tmp_path / "completion_receipt.json"
    with pytest.raises(ValueError):
        write_completion_receipt(
            receipt,
            job_id="job",
            artifacts=[artifact],
            metadata={"bad": float("nan")},
        )
    assert not receipt.exists()


def test_status_rejects_nonstandard_nonfinite_json(tmp_path: Path):
    (tmp_path / "manifest.json").write_text('{"schema_version":2,"bad":NaN}\n')
    assert status_main([str(tmp_path)]) == 2


def test_schema2_receipt_survives_atomic_directory_move_and_checks_identity(tmp_path: Path):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    artifact = attempt / "data.bin"
    artifact.write_bytes(b"payload")
    identity = {
        "campaign_scientific_identity": "abc",
        "protocol_fingerprint": "def",
        "run_id": "r0",
        "stage": "training",
    }
    receipt = attempt / "completion_receipt.json"
    write_completion_receipt(receipt, job_id="expected", artifacts=[artifact], metadata=identity)
    payload = json.loads(receipt.read_text())
    assert payload["schema_version"] == 2
    assert payload["artifacts"] == {"data.bin": sha256_file(artifact)}

    final = tmp_path / "final"
    os.replace(attempt, final)
    valid, reason = verify_completion_receipt(
        final / "completion_receipt.json",
        expected_job_id="expected",
        expected_metadata=identity,
    )
    assert valid, reason
    valid, reason = verify_completion_receipt(
        final / "completion_receipt.json", expected_job_id="wrong"
    )
    assert not valid and "job_id mismatch" in reason


def test_partial_final_is_preserved_and_no_longer_blocks_retry(tmp_path: Path):
    root = tmp_path / "campaign"
    partial = root / "runs" / "r0"
    partial.mkdir(parents=True)
    (partial / "partial.txt").write_text("keep me")
    preserved = _preserve_invalid_output(root, "training", "r0", partial)
    assert not partial.exists()
    assert (preserved / "partial.txt").read_text() == "keep me"
    assert "attempts/training/r0/recovered-" in preserved.as_posix()


def test_campaign_evaluation_bank_is_single_immutable_artifact(tmp_path: Path):
    protocol = {
        "phase1_ring_pilot": {"task": {"sequence_steps": 8}},
        "seed_policy": {"task_seed": 3, "evaluation_bank_seed": 7},
        "evaluation": {"id_test_trials": 4},
    }
    first = _materialize_evaluation_bank(tmp_path, protocol)
    second = _materialize_evaluation_bank(tmp_path, protocol)
    assert first == second
    assert first["trials"] == 4 and first["horizon"] == 8
    assert (tmp_path / first["path"]).is_file()
    assert Path(f"{tmp_path / first['path']}.sha256").is_file()


def test_campaign_perturbation_bank_freezes_common_raw_directions(tmp_path: Path):
    protocol = {
        "seed_policy": {"perturbation_bank_seed": 0},
        "phase1_ring_pilot": {"training": {"width": 6}},
        "evaluation": {
            "finite_kick_anchor_count": 4,
            "jacobian_anchor_count": 3,
            "ambient_random_directions_per_anchor": 2,
        },
    }
    first = _materialize_perturbation_bank(tmp_path, protocol)
    second = _materialize_perturbation_bank(tmp_path, protocol)
    assert first == second
    assert first["specification_sha256"] == canonical_hash(first["specification"])
    with np.load(tmp_path / first["path"], allow_pickle=False) as archive:
        assert archive["finite_raw"].shape == (4, 2, 6)
        assert archive["jacobian_raw"].shape == (3, 2, 6)


def test_child_commands_receive_frozen_banks_and_full_only_state_spec(tmp_path: Path):
    protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
    common = dict(
        protocol=protocol,
        root=tmp_path,
        python="/python",
        protocol_path=DEFAULT_PROTOCOL_PATH,
        evaluation_bank=tmp_path / "evaluation.npz",
        campaign_identity="identity",
    )
    smoke_jobs = _training_jobs(**common, smoke=True)
    full_jobs = _training_jobs(**common, smoke=False)
    assert "--evaluation-bank" in smoke_jobs[0].command
    assert "--campaign-identity" in smoke_jobs[0].command
    assert "--state-spec" not in smoke_jobs[0].command
    assert "--state-spec" in full_jobs[0].command
    analyses = _analysis_jobs(
        smoke_jobs,
        root=tmp_path,
        python="/python",
        protocol_path=DEFAULT_PROTOCOL_PATH,
        evaluation_bank=tmp_path / "evaluation.npz",
        perturbation_bank=tmp_path / "perturbation.npz",
        campaign_identity="identity",
        smoke=True,
    )
    assert "--evaluation-bank" in analyses[0].command
    assert "--perturbation-bank" in analyses[0].command
    assert "--campaign-identity" in analyses[0].command


def test_status_marks_dead_recorded_pid_stale_and_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = _manifest()
    atomic_json(tmp_path / "manifest.json", manifest)
    atomic_json(
        tmp_path / "status.json",
        {
            "stage": "training",
            "jobs": {"training:r0": {"state": "running", "pid": 999_999_999}},
        },
    )
    monkeypatch.setattr(
        "repro.sagodi_protocol.status._verify_campaign_inputs", lambda *args: None
    )
    report = summarize(tmp_path)
    training = next(row for row in report["jobs"] if row["stage"] == "training")
    assert training["state"] == "stale"
    assert training["pid_alive"] is False
    assert report["campaign_complete"] is False
    assert status_main([str(tmp_path)]) == 1


def test_status_rejects_live_but_reused_pid_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = _manifest()
    atomic_json(tmp_path / "manifest.json", manifest)
    atomic_json(
        tmp_path / "status.json",
        {
            "stage": "training",
            "jobs": {
                "training:r0": {
                    "state": "running",
                    "pid": os.getpid(),
                    "process_identity": {
                        "pid": os.getpid(),
                        "boot_id": "wrong",
                        "proc_start_ticks": -1,
                        "cmdline_sha256": "0" * 64,
                    },
                }
            },
        },
    )
    monkeypatch.setattr(
        "repro.sagodi_protocol.status._verify_campaign_inputs", lambda *args: None
    )
    report = summarize(tmp_path)
    training = next(row for row in report["jobs"] if row["stage"] == "training")
    assert training["pid_alive"] is True
    assert training["pid_identity_matches"] is False
    assert training["state"] == "stale"


def test_complete_marker_is_valid_only_for_current_gate_and_all_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = _manifest()
    atomic_json(tmp_path / "manifest.json", manifest)
    for key in manifest["receipt_expectations"]:
        _write_identity_receipt(tmp_path, manifest, key)
    atomic_json(tmp_path / "phase0" / "phase0_gate.json", {"passed": True})
    complete, reasons, completion_hashes = compute_campaign_completion(tmp_path, manifest)
    assert complete, reasons
    atomic_json(
        tmp_path / "COMPLETE",
        {
            "schema_version": 3,
            "campaign_id": manifest["campaign_id"],
            "scientific_identity": manifest["scientific_identity"],
            "protocol_fingerprint": manifest["protocol_canonical_fingerprint"],
            "receipt_sha256": completion_hashes["receipts"],
            "pilot_aggregation_sha256": completion_hashes["pilot_aggregation"],
            "completed_at": 1.0,
        },
    )
    monkeypatch.setattr(
        "repro.sagodi_protocol.status._verify_campaign_inputs", lambda *args: None
    )
    report = summarize(tmp_path)
    assert report["computed_complete"] is True
    assert report["complete_marker"]["valid"] is True
    assert report["campaign_complete"] is True
    assert status_main([str(tmp_path), "--json"]) == 0

    atomic_json(tmp_path / "phase0" / "phase0_gate.json", {"passed": False})
    report = summarize(tmp_path)
    assert report["campaign_complete"] is False
    assert report["complete_marker"]["valid"] is False
    assert status_main([str(tmp_path)]) == 1


def test_analysis_receipt_is_invalidated_when_training_checkpoint_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = _manifest()
    atomic_json(tmp_path / "manifest.json", manifest)
    for key in manifest["receipt_expectations"]:
        _write_identity_receipt(tmp_path, manifest, key)
    atomic_json(tmp_path / "phase0" / "phase0_gate.json", {"passed": True})
    valid, reason = verify_campaign_output_receipt(
        tmp_path, manifest, "analysis:r0"
    )
    assert valid, reason

    (tmp_path / "runs" / "r0" / "checkpoint.pt").write_bytes(b"tampered")
    valid, reason = verify_campaign_output_receipt(
        tmp_path, manifest, "analysis:r0"
    )
    assert not valid
    assert "training" in reason
    complete, reasons, _ = compute_campaign_completion(tmp_path, manifest)
    assert not complete
    assert reasons["analysis:r0"] != "ok"

    monkeypatch.setattr(
        "repro.sagodi_protocol.status._verify_campaign_inputs", lambda *args: None
    )
    report = summarize(tmp_path)
    analysis = next(row for row in report["jobs"] if row["stage"] == "analysis")
    assert analysis["receipt_valid"] is False
    assert report["campaign_complete"] is False


def test_campaign_lock_refuses_live_owner_and_releases(tmp_path: Path):
    _prepare_root(tmp_path)
    lock = _acquire_campaign_lock(tmp_path)
    assert lock.path.is_file()
    try:
        with pytest.raises(RuntimeError, match="live orchestrator"):
            _acquire_campaign_lock(tmp_path)
    finally:
        lock.release()
    assert not (tmp_path / ".orchestrator.lock").exists()


def test_campaign_lock_preserves_stale_owner_before_reacquiring(tmp_path: Path):
    _prepare_root(tmp_path)
    atomic_json(
        tmp_path / ".orchestrator.lock",
        {
            "schema_version": 1,
            "token": "stale",
            "owner_process_identity": {
                "pid": 999_999_999,
                "boot_id": "dead",
                "proc_start_ticks": 1,
                "cmdline_sha256": "0" * 64,
            },
            "acquired_at": 0.0,
        },
    )
    lock = _acquire_campaign_lock(tmp_path)
    try:
        preserved = list(
            (tmp_path / "attempts" / "campaign_setup" / "orchestrator_lock").glob(
                "recovered-*"
            )
        )
        assert len(preserved) == 1
        assert json.loads(preserved[0].read_text())["token"] == "stale"
    finally:
        lock.release()


def test_campaign_lock_guard_blocks_during_delayed_payload_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A not-yet-visible payload must never be mistaken for a stale lock."""

    _prepare_root(tmp_path)
    import repro.sagodi_protocol.orchestrate as orchestrate

    real_atomic_json = orchestrate.atomic_json
    entered = threading.Event()
    continue_write = threading.Event()
    acquired: list = []
    failures: list[BaseException] = []

    def delayed_atomic_json(path, payload):
        if Path(path).name == ".orchestrator.lock":
            entered.set()
            if not continue_write.wait(timeout=5.0):
                raise TimeoutError("unit test did not release delayed payload write")
        return real_atomic_json(path, payload)

    def acquire_in_thread():
        try:
            acquired.append(_acquire_campaign_lock(tmp_path))
        except BaseException as exc:  # pragma: no cover - reported below
            failures.append(exc)

    monkeypatch.setattr(orchestrate, "atomic_json", delayed_atomic_json)
    thread = threading.Thread(target=acquire_in_thread, daemon=True)
    thread.start()
    assert entered.wait(timeout=5.0)
    assert not (tmp_path / ".orchestrator.lock").exists()
    with pytest.raises(RuntimeError, match="live orchestrator"):
        _acquire_campaign_lock(tmp_path)
    assert not (
        tmp_path / "attempts" / "campaign_setup" / "orchestrator_lock"
    ).exists()
    continue_write.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert failures == []
    assert len(acquired) == 1
    acquired[0].release()


def _pilot_aggregation_manifest(tmp_path: Path, *, models=("ca_lru", "no_rp", "gru")):
    runs = []
    expectations = {
        "phase0": {
            "job_id": "phase0_state_audit",
            "stage": "phase0",
            "run_id": "phase0",
            "output": "phase0",
        }
    }
    for model in models:
        for seed in range(100, 105):
            run_id = f"{model}-seed{seed}"
            runs.append(
                {
                    "run_id": run_id,
                    "model": {"id": model},
                    "model_seed": seed,
                    "learning_rate": 0.01,
                }
            )
            expectations[f"training:{run_id}"] = {
                "job_id": f"{model}-seed{seed}-lr0.01",
                "stage": "training",
                "run_id": run_id,
                "output": f"runs/{run_id}",
            }
            expectations[f"analysis:{run_id}"] = {
                "job_id": f"phase1-analysis-{run_id}",
                "stage": "analysis",
                "run_id": run_id,
                "output": f"analysis/{run_id}",
            }
    payload = {
        "campaign_id": "aggregation-test",
        "protocol_canonical_fingerprint": "a" * 64,
        "run_matrix": runs,
        "receipt_expectations": expectations,
        "pilot_aggregation": aggregation_specification(),
    }
    manifest = {
        "schema_version": 2,
        **payload,
        "scientific_identity_payload": payload,
        "scientific_identity": canonical_hash(payload),
    }
    for run in runs:
        seed = int(run["model_seed"])
        task_success = seed != 104
        output = tmp_path / f"analysis/{run['run_id']}"
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            output / "claim_gate.json",
            {
                "pilot_only": True,
                "gates": {
                    "task": {
                        "gate_id": "task",
                        "metric": "masked_target_power_nmse_db",
                        "status": "passed" if task_success else "failed",
                        "passed": task_success,
                        "value": float(seed - 125),
                    },
                    "c2_drift": {
                        "gate_id": "c2_drift",
                        "metric": "pi_normalized_nearest_atlas_drift",
                        "status": "passed" if task_success else "not_applicable_task_failure",
                        "passed": task_success,
                        "value": (
                            {"mean": 0.001 * (seed - 99), "q95": 0.002 * (seed - 99)}
                            if task_success
                            else None
                        ),
                    },
                    "c4": {
                        "gate_id": "c4",
                        "metric": "not_evaluated",
                        "status": "not_evaluated",
                        "passed": None,
                        "value": None,
                    },
                },
            },
        )
    return manifest


def test_pilot_aggregation_uses_all_15_denominators_and_no_inference(tmp_path: Path):
    manifest = _pilot_aggregation_manifest(tmp_path)
    summary, matrix = build_pilot_aggregation(tmp_path, manifest)
    assert summary["started_run_count"] == 15
    assert summary["task_success_count"] == 12
    assert summary["pilot_seeds"] == [100, 101, 102, 103, 104]
    assert summary["inference"]["confirmatory"] is False
    assert summary["inference"]["p_values_computed"] is False
    assert len(summary["runs"]) == 15
    assert len(matrix.decode().splitlines()) == 16
    for model in ("ca_lru", "no_rp", "gru"):
        aggregate = summary["per_model"][model]
        assert aggregate["all_started"]["denominator"] == 5
        assert aggregate["all_started"]["task_success_count"] == 4
        assert aggregate["task_success_conditional"]["denominator"] == 4
        task_numeric = aggregate["all_started"]["gate_summaries"]["task"][
            "numeric_value_summaries"
        ]["value"]
        assert task_numeric["n"] == 5
        assert task_numeric["sample_std"] is not None
        assert task_numeric["iqr"] == pytest.approx(2.0)


def test_aggregation_corruption_invalidates_campaign_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = _pilot_aggregation_manifest(tmp_path, models=("gru",))
    atomic_json(tmp_path / "manifest.json", manifest)
    atomic_json(tmp_path / "phase0" / "phase0_gate.json", {"passed": True})
    for key in manifest["receipt_expectations"]:
        _write_identity_receipt(tmp_path, manifest, key)
    hashes = _write_pilot_aggregation(tmp_path, manifest)
    assert set(hashes) == {
        "pilot_aggregation/pilot_summary.json",
        "pilot_aggregation/pilot_run_matrix.csv",
    }
    complete, reasons, completion_hashes = compute_campaign_completion(tmp_path, manifest)
    assert complete, reasons
    assert completion_hashes["pilot_aggregation"] == hashes
    atomic_json(
        tmp_path / "COMPLETE",
        {
            "schema_version": 3,
            "campaign_id": manifest["campaign_id"],
            "scientific_identity": manifest["scientific_identity"],
            "protocol_fingerprint": manifest["protocol_canonical_fingerprint"],
            "receipt_sha256": completion_hashes["receipts"],
            "pilot_aggregation_sha256": hashes,
            "completed_at": 1.0,
        },
    )
    monkeypatch.setattr(
        "repro.sagodi_protocol.status._verify_campaign_inputs", lambda *args: None
    )
    report = summarize(tmp_path)
    assert report["pilot_aggregation"]["valid"] is True
    assert report["complete_marker"]["valid"] is True
    assert report["campaign_complete"] is True

    matrix = tmp_path / "pilot_aggregation" / "pilot_run_matrix.csv"
    matrix.write_bytes(matrix.read_bytes() + b"corruption\n")
    complete, reasons, completion_hashes = compute_campaign_completion(tmp_path, manifest)
    assert not complete
    assert "differ" in reasons["pilot_aggregation"]
    assert completion_hashes["pilot_aggregation"] == {}
    report = summarize(tmp_path)
    assert report["pilot_aggregation"]["valid"] is False
    assert report["complete_marker"]["valid"] is False
    assert report["campaign_complete"] is False


def test_duplicate_gpu_ids_are_rejected_before_campaign_root_creation(tmp_path: Path):
    with pytest.raises(ValueError, match="duplicate GPU ids"):
        _validated_gpu_ids((0, 0))
    root = tmp_path / "duplicate-root"
    with pytest.raises(ValueError, match="duplicate GPU ids"):
        run_campaign(
            protocol_path=DEFAULT_PROTOCOL_PATH,
            artifact_root=root,
            python=sys.executable,
            gpus=(0, 0),
            smoke=True,
            dry_run=True,
        )
    assert not root.exists()


def test_peer_termination_records_exit_and_closes_all_handles(tmp_path: Path):
    root = tmp_path / "campaign"
    root.mkdir()
    status = {"jobs": {}}
    active = {}
    for gpu in (0, 1):
        attempt = root / f"attempt-{gpu}"
        attempt.mkdir()
        log = root / f"gpu{gpu}.log"
        handle = log.open("ab", buffering=0)
        process = subprocess.Popen(
            ["bash", "-c", "sleep 60"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        job = Job(f"r{gpu}", "training", root / "runs" / f"r{gpu}", (), "native")
        active[gpu] = ActiveJob(job, process, handle, gpu, attempt, log)
    _terminate_active(
        active,
        status=status,
        root=root,
        reason="unit-test peer failure",
        grace=0.2,
    )
    assert active == {}
    assert all(entry["state"] == "terminated" for entry in status["jobs"].values())
    assert all(entry["exit_code"] is not None for entry in status["jobs"].values())
