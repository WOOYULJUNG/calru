from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from repro.sagodi_protocol import reanalyze
from repro.sagodi_protocol.artifacts import (
    atomic_json,
    canonical_hash,
    sha256_file,
    write_completion_receipt,
)
from repro.sagodi_protocol.config import (
    expand_phase1_runs,
    load_protocol,
    protocol_fingerprint,
)


PROTOCOL = Path(__file__).resolve().parents[1] / "calru_native_sagodi_ring_pilot_v1.yaml"


@pytest.fixture(autouse=True)
def _clear_campaign_receipt_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CALRU_CAMPAIGN_SCIENTIFIC_IDENTITY",
        "CALRU_PROTOCOL_FINGERPRINT",
        "CALRU_RUN_ID",
        "CALRU_STAGE",
    ):
        monkeypatch.delenv(name, raising=False)


def _bank(root: Path, relative: str, payload: bytes) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return {
        "path": relative,
        "sha256": digest,
        "sidecar_sha256": sha256_file(sidecar),
    }


def _synthetic_parent(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    protocol = load_protocol(PROTOCOL)
    matrix = list(expand_phase1_runs(protocol))
    campaign_id = "synthetic-parent-campaign"
    root = tmp_path / campaign_id
    root.mkdir()
    evaluation = _bank(root, "evaluation_bank/angular_integration_id.npz", b"evaluation")
    perturbation = _bank(root, "perturbation_bank/phase1_ring_seed0.npz", b"perturbation")
    expectations: dict[str, Any] = {}
    for run in matrix:
        run_id = run["run_id"]
        model = run["model"]["id"]
        seed = int(run["model_seed"])
        lr = float(run["learning_rate"])
        expectations[f"training:{run_id}"] = {
            "job_id": f"{model}-seed{seed}-lr{lr:g}",
            "output": f"runs/{run_id}",
            "run_id": run_id,
            "stage": "training",
        }
    signed = {
        "campaign_id": campaign_id,
        "protocol_file_sha256": sha256_file(PROTOCOL),
        "protocol_canonical_fingerprint": protocol_fingerprint(protocol),
        "run_matrix": matrix,
        "evaluation_bank": evaluation,
        "perturbation_bank": perturbation,
        "receipt_expectations": expectations,
    }
    identity = canonical_hash(signed)
    manifest = {
        "schema_version": 2,
        **signed,
        "scientific_identity_payload": signed,
        "scientific_identity": identity,
    }
    atomic_json(root / "manifest.json", manifest)
    contract = {
        "freeze_id": "synthetic-analysis-freeze",
        "freeze_status": "pilot_reanalysis_only",
        "claim_scope": {
            "claim": "finite_time_approximate_continuous_attractor_only",
            "confirmatory": False,
            "may_support_l3_claim": False,
            "may_support_c4_claim": False,
            "may_support_main_seed_claim": False,
            "seed_scope": "parent_pilot_seeds_only",
            "causal_interpretation": "none",
        },
        "parent_training": {
            "freeze_id": protocol["freeze_id"],
            "protocol_canonical_fingerprint": protocol_fingerprint(protocol),
            "protocol_file_sha256": sha256_file(PROTOCOL),
            "campaign_id": campaign_id,
            "scientific_identity": identity,
            "manifest_sha256": sha256_file(root / "manifest.json"),
        },
    }
    return root, manifest, contract


def _publish_training_run(root: Path, manifest: dict[str, Any], index: int = 0) -> str:
    run = manifest["run_matrix"][index]
    run_id = str(run["run_id"])
    output = root / "runs" / run_id
    output.mkdir(parents=True)
    checkpoint = output / "checkpoint.pt"
    checkpoint.write_bytes(f"checkpoint:{run_id}".encode())
    run_manifest = {
        "schema_version": 1,
        "protocol_freeze_id": run["freeze_id"],
        "protocol_file_sha256": manifest["protocol_file_sha256"],
        "protocol_canonical_fingerprint": manifest["protocol_canonical_fingerprint"],
        "campaign_identity": manifest["scientific_identity"],
        "evaluation_bank_sha256": manifest["evaluation_bank"]["sha256"],
        "model_id": run["model"]["id"],
        "model_seed": int(run["model_seed"]),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    atomic_json(output / "manifest.json", run_manifest)
    expectation = manifest["receipt_expectations"][f"training:{run_id}"]
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=expectation["job_id"],
        artifacts=[checkpoint, output / "manifest.json"],
        metadata={
            "campaign_scientific_identity": manifest["scientific_identity"],
            "protocol_fingerprint": manifest["protocol_canonical_fingerprint"],
            "run_id": run_id,
            "stage": "training",
            "freeze_id": run["freeze_id"],
            "protocol_canonical_fingerprint": manifest["protocol_canonical_fingerprint"],
            "campaign_identity": manifest["scientific_identity"],
            "model_id": run["model"]["id"],
            "model_seed": int(run["model_seed"]),
            "evaluation_bank_sha256": manifest["evaluation_bank"]["sha256"],
        },
    )
    return run_id


def _minimal_launch_manifest(
    contract: dict[str, Any],
    freeze_file: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    freeze_fp = reanalyze.analysis_freeze_fingerprint(contract)
    return {
        "analysis_schema_version": reanalyze.EXPECTED_ANALYSIS_SCHEMA_VERSION,
        "claim_scope": contract["claim_scope"],
        "analysis_freeze": {
            "freeze_id": contract["freeze_id"],
            "file_sha256": sha256_file(freeze_file),
            "canonical_fingerprint": freeze_fp,
        },
        "parent": {
            **contract["parent_training"],
            "evaluation_bank": {"sha256": "e" * 64},
            "perturbation_bank": {"sha256": "p" * 64},
        },
        "scientific_identity_payload": {},
        "_checkpoint": checkpoint_sha256,
    }


def _analysis_output(
    output: Path,
    binding: reanalyze.RunBinding,
    launch_manifest: dict[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    freeze_binding = {
        "schema_version": 1,
        "analysis_freeze_id": launch_manifest["analysis_freeze"]["freeze_id"],
        "analysis_freeze_path": "/immutable/freeze.yaml",
        "analysis_freeze_file_sha256": launch_manifest["analysis_freeze"]["file_sha256"],
        "analysis_freeze_canonical_fingerprint": launch_manifest["analysis_freeze"]["canonical_fingerprint"],
        "parent_training": {
            key: launch_manifest["parent"][key]
            for key in (
                "freeze_id",
                "protocol_canonical_fingerprint",
                "protocol_file_sha256",
                "campaign_id",
                "scientific_identity",
                "manifest_sha256",
            )
        },
        "claim_scope": launch_manifest["claim_scope"],
    }
    atomic_json(output / "analysis_freeze_binding.json", freeze_binding)
    atomic_json(
        output / "analysis.json",
        {
            "schema_version": launch_manifest["analysis_schema_version"],
            "pilot_only": True,
            "smoke": False,
            "checkpoint_sha256": binding.checkpoint_sha256,
            "analysis_freeze_binding": freeze_binding,
            "scope": "test",
        },
    )
    atomic_json(
        output / "claim_gate.json",
        {
            "schema_version": launch_manifest["analysis_schema_version"],
            "pilot_only": True,
            "smoke": False,
            "approximate_ca_claim_allowed": False,
            "levels": {"L3": {"passed": False}},
            "gates": {
                "c4": {"status": "not_evaluated", "passed": None},
                "model_seeds": {"status": "not_evaluated", "passed": None},
            },
        },
    )
    _rewrite_analysis_receipt(output, binding, launch_manifest)


def _rewrite_analysis_receipt(
    output: Path,
    binding: reanalyze.RunBinding,
    launch_manifest: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=f"phase1-analysis-{binding.run_id}",
        artifacts=[
            output / "analysis_freeze_binding.json",
            output / "analysis.json",
            output / "claim_gate.json",
        ],
        metadata=(
            reanalyze._analysis_receipt_metadata(launch_manifest, binding)
            if metadata is None
            else metadata
        ),
    )


def test_parent_manifest_tamper_is_rejected(tmp_path: Path) -> None:
    root, _, contract = _synthetic_parent(tmp_path)
    with (root / "manifest.json").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(reanalyze.ParentVerificationError, match="manifest SHA-256"):
        reanalyze.inspect_parent_campaign(
            parent_root=root,
            protocol_path=PROTOCOL,
            analysis_contract=contract,
        )


def test_output_inside_parent_is_rejected_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, contract = _synthetic_parent(tmp_path)
    freeze_file = tmp_path / "analysis-freeze.json"
    freeze_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(reanalyze, "load_analysis_freeze", lambda _: contract)
    artifact = root / "forbidden-analysis"
    with pytest.raises(reanalyze.ReanalysisError, match="must not overlap"):
        reanalyze.run_reanalysis(
            parent_root=root,
            artifact_root=artifact,
            protocol_path=PROTOCOL,
            analysis_freeze_path=freeze_file,
            python=sys.executable,
            gpus=(0,),
            available_only=True,
        )
    assert not artifact.exists()


def test_default_mode_rejects_incomplete_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, contract = _synthetic_parent(tmp_path)
    freeze_file = tmp_path / "analysis-freeze.json"
    freeze_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(reanalyze, "load_analysis_freeze", lambda _: contract)
    with pytest.raises(reanalyze.IncompleteParentError, match="requires all 15"):
        reanalyze.run_reanalysis(
            parent_root=root,
            artifact_root=tmp_path / "analysis",
            protocol_path=PROTOCOL,
            analysis_freeze_path=freeze_file,
            python=sys.executable,
            gpus=(0,),
        )


def test_available_only_freezes_verified_subset_and_never_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, contract = _synthetic_parent(tmp_path)
    selected_id = _publish_training_run(root, manifest, 0)
    freeze_file = tmp_path / "analysis-freeze.json"
    freeze_file.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(reanalyze, "load_analysis_freeze", lambda _: contract)
    monkeypatch.setattr(reanalyze, "_environment_fingerprint", lambda *_: {"env": "test"})
    monkeypatch.setattr(reanalyze, "_git_state", lambda *_: {"git": "test"})
    monkeypatch.setattr(reanalyze, "_source_hashes", lambda *_: {"source.py": "s" * 64})

    def fake_queue(**kwargs: Any) -> None:
        job = kwargs["jobs"][0]
        _analysis_output(job.output_dir, job.binding, kwargs["manifest"])
        kwargs["status"]["jobs"][job.binding.run_id].update(
            state="completed", reason="fake verified child"
        )

    monkeypatch.setattr(reanalyze, "_run_queue", fake_queue)
    artifact = tmp_path / "analysis"
    reanalyze.run_reanalysis(
        parent_root=root,
        artifact_root=artifact,
        protocol_path=PROTOCOL,
        analysis_freeze_path=freeze_file,
        python=sys.executable,
        gpus=(0,),
        available_only=True,
    )
    output_manifest = json.loads((artifact / "manifest.json").read_text())
    status = json.loads((artifact / "status.json").read_text())
    matrix = json.loads((artifact / "run_matrix.json").read_text())
    assert output_manifest["scope"] == "pilot_reanalysis_only"
    assert output_manifest["selection"]["selected_run_count"] == 1
    assert output_manifest["run_matrix"][0]["run_id"] == selected_id
    assert status["stage"] == "partial_complete"
    assert status["campaign_complete"] is False
    assert status["jobs"][selected_id]["state"] == "completed"
    assert sum(item["state"] == "pending" for item in matrix["runs"]) == 14
    assert not (artifact / "COMPLETE").exists()


def test_root_identity_is_initialized_and_compared_under_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, contract = _synthetic_parent(tmp_path)
    freeze_file = tmp_path / "analysis-freeze.json"
    freeze_file.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(reanalyze, "load_analysis_freeze", lambda _: contract)
    monkeypatch.setattr(reanalyze, "_environment_fingerprint", lambda *_: {"env": "test"})
    monkeypatch.setattr(reanalyze, "_git_state", lambda *_: {"git": "test"})
    monkeypatch.setattr(reanalyze, "_source_hashes", lambda *_: {"source.py": "s" * 64})

    original_lock = reanalyze._exclusive_lock
    original_prepare = reanalyze._prepare_root
    lock_held = False

    @contextmanager
    def observing_lock(path: Path):
        nonlocal lock_held
        with original_lock(path):
            lock_held = True
            try:
                yield
            finally:
                lock_held = False

    def checked_prepare(path: Path, manifest: dict[str, Any]) -> None:
        assert lock_held, "root marker/manifest initialization escaped the exclusive lock"
        original_prepare(path, manifest)

    monkeypatch.setattr(reanalyze, "_exclusive_lock", observing_lock)
    monkeypatch.setattr(reanalyze, "_prepare_root", checked_prepare)
    artifact = tmp_path / "analysis"
    reanalyze.run_reanalysis(
        parent_root=root,
        artifact_root=artifact,
        protocol_path=PROTOCOL,
        analysis_freeze_path=freeze_file,
        python=sys.executable,
        gpus=(0,),
        available_only=True,
    )
    assert (artifact / reanalyze.ROOT_MARKER).is_file()

    tampered = json.loads((artifact / "manifest.json").read_text())
    tampered["scope"] = "tampered"
    atomic_json(artifact / "manifest.json", tampered)
    with pytest.raises(reanalyze.ReanalysisError, match="manifest differs"):
        reanalyze.run_reanalysis(
            parent_root=root,
            artifact_root=artifact,
            protocol_path=PROTOCOL,
            analysis_freeze_path=freeze_file,
            python=sys.executable,
            gpus=(0,),
            available_only=True,
        )


def test_campaign_id_is_unique_to_mode_and_selected_checkpoint_snapshot(
    tmp_path: Path,
) -> None:
    root, manifest, contract = _synthetic_parent(tmp_path)
    run_id = _publish_training_run(root, manifest, 0)
    snapshot = reanalyze.inspect_parent_campaign(
        parent_root=root,
        protocol_path=PROTOCOL,
        analysis_contract=contract,
    )
    binding = snapshot.available[run_id]
    freeze_file = tmp_path / "analysis-freeze.json"
    freeze_file.write_text(json.dumps(contract), encoding="utf-8")
    common = {
        "snapshot": snapshot,
        "freeze_path": freeze_file,
        "freeze": contract,
        "python": sys.executable,
        "gpus": (0,),
        "code": {"git": "test"},
        "source_hashes": {"source.py": "s" * 64},
        "environment": {"env": "test"},
    }
    partial = reanalyze._manifest_payload(
        **common, selected=(binding,), available_only=True
    )
    full_mode = reanalyze._manifest_payload(
        **common, selected=(binding,), available_only=False
    )
    changed_checkpoint = reanalyze._manifest_payload(
        **common,
        selected=(replace(binding, checkpoint_sha256="0" * 64),),
        available_only=True,
    )
    assert len({partial["campaign_id"], full_mode["campaign_id"], changed_checkpoint["campaign_id"]}) == 3
    assert partial["campaign_id"].endswith(
        partial["selection"]["snapshot_binding_sha256"][:12]
    )


def test_checkpoint_tamper_breaks_training_receipt_binding(tmp_path: Path) -> None:
    root, manifest, contract = _synthetic_parent(tmp_path)
    run_id = _publish_training_run(root, manifest, 0)
    with (root / "runs" / run_id / "checkpoint.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(reanalyze.ParentVerificationError, match="receipt invalid"):
        reanalyze.inspect_parent_campaign(
            parent_root=root,
            protocol_path=PROTOCOL,
            analysis_contract=contract,
        )


def test_resume_requires_receipt_bound_to_freeze_and_checkpoint(tmp_path: Path) -> None:
    root, manifest, contract = _synthetic_parent(tmp_path)
    run_id = _publish_training_run(root, manifest, 0)
    snapshot = reanalyze.inspect_parent_campaign(
        parent_root=root,
        protocol_path=PROTOCOL,
        analysis_contract=contract,
    )
    binding = snapshot.available[run_id]
    freeze_file = tmp_path / "analysis-freeze.json"
    freeze_file.write_text(json.dumps(contract), encoding="utf-8")
    launch_manifest = _minimal_launch_manifest(
        contract, freeze_file, binding.checkpoint_sha256
    )
    output = tmp_path / "reanalysis-output"
    _analysis_output(output, binding, launch_manifest)
    valid, reason = reanalyze.verify_analysis_output(output, binding, launch_manifest)
    assert valid, reason
    job = reanalyze.AnalysisJob(binding=binding, output_dir=output)
    assert reanalyze._job_state(job, launch_manifest)[0] == "completed"

    wrong_checkpoint = replace(binding, checkpoint_sha256="0" * 64)
    valid, reason = reanalyze.verify_analysis_output(
        output, wrong_checkpoint, launch_manifest
    )
    assert not valid
    assert "checkpoint_sha256" in reason


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    (
        ("pilot_only", "non-smoke pilot"),
        ("approximate_claim", "improperly allows"),
        ("l3", "L3 must be false"),
        ("c4", "c4 must remain not_evaluated"),
        ("model_seeds", "model_seeds must remain not_evaluated"),
    ),
)
def test_self_consistent_tampered_receipt_cannot_cross_claim_boundary(
    tmp_path: Path,
    mutation: str,
    expected_reason: str,
) -> None:
    root, manifest, contract = _synthetic_parent(tmp_path)
    run_id = _publish_training_run(root, manifest, 0)
    snapshot = reanalyze.inspect_parent_campaign(
        parent_root=root,
        protocol_path=PROTOCOL,
        analysis_contract=contract,
    )
    binding = snapshot.available[run_id]
    freeze_file = tmp_path / "analysis-freeze.json"
    freeze_file.write_text(json.dumps(contract), encoding="utf-8")
    launch_manifest = _minimal_launch_manifest(contract, freeze_file, binding.checkpoint_sha256)
    output = tmp_path / "reanalysis-output"
    _analysis_output(output, binding, launch_manifest)
    claim = json.loads((output / "claim_gate.json").read_text())
    if mutation == "pilot_only":
        claim["pilot_only"] = False
    elif mutation == "approximate_claim":
        claim["approximate_ca_claim_allowed"] = True
    elif mutation == "l3":
        claim["levels"]["L3"]["passed"] = True
    else:
        claim["gates"][mutation] = {"status": "passed", "passed": True}
    atomic_json(output / "claim_gate.json", claim)
    # Reissue a fully self-consistent receipt over the modified claim artifact.
    # Integrity hashes alone must not make forbidden pilot claim semantics valid.
    _rewrite_analysis_receipt(output, binding, launch_manifest)
    valid, reason = reanalyze.verify_analysis_output(output, binding, launch_manifest)
    assert not valid
    assert expected_reason in reason


@pytest.mark.parametrize(
    ("metadata_key", "tampered_value"),
    (("analysis_schema_version", 3), ("L3", True)),
)
def test_resume_receipt_requires_analysis_schema_and_false_l3(
    tmp_path: Path,
    metadata_key: str,
    tampered_value: Any,
) -> None:
    root, manifest, contract = _synthetic_parent(tmp_path)
    run_id = _publish_training_run(root, manifest, 0)
    snapshot = reanalyze.inspect_parent_campaign(
        parent_root=root,
        protocol_path=PROTOCOL,
        analysis_contract=contract,
    )
    binding = snapshot.available[run_id]
    freeze_file = tmp_path / "analysis-freeze.json"
    freeze_file.write_text(json.dumps(contract), encoding="utf-8")
    launch_manifest = _minimal_launch_manifest(contract, freeze_file, binding.checkpoint_sha256)
    output = tmp_path / "reanalysis-output"
    _analysis_output(output, binding, launch_manifest)
    metadata = reanalyze._analysis_receipt_metadata(launch_manifest, binding)
    metadata[metadata_key] = tampered_value
    _rewrite_analysis_receipt(output, binding, launch_manifest, metadata=metadata)
    valid, reason = reanalyze.verify_analysis_output(output, binding, launch_manifest)
    assert not valid
    assert metadata_key in reason


@pytest.mark.parametrize("binding_mutation", ("claim_scope", "parent_freeze"))
def test_resume_requires_exact_claim_scope_and_parent_freeze_binding(
    tmp_path: Path,
    binding_mutation: str,
) -> None:
    root, manifest, contract = _synthetic_parent(tmp_path)
    run_id = _publish_training_run(root, manifest, 0)
    snapshot = reanalyze.inspect_parent_campaign(
        parent_root=root,
        protocol_path=PROTOCOL,
        analysis_contract=contract,
    )
    binding = snapshot.available[run_id]
    freeze_file = tmp_path / "analysis-freeze.json"
    freeze_file.write_text(json.dumps(contract), encoding="utf-8")
    launch_manifest = _minimal_launch_manifest(contract, freeze_file, binding.checkpoint_sha256)
    output = tmp_path / "reanalysis-output"
    _analysis_output(output, binding, launch_manifest)
    freeze_binding = json.loads((output / "analysis_freeze_binding.json").read_text())
    if binding_mutation == "claim_scope":
        freeze_binding["claim_scope"]["confirmatory"] = True
        expected_reason = "claim_scope"
    else:
        freeze_binding["parent_training"]["freeze_id"] = "wrong-parent-freeze"
        expected_reason = "freeze_id"
    atomic_json(output / "analysis_freeze_binding.json", freeze_binding)
    analysis = json.loads((output / "analysis.json").read_text())
    analysis["analysis_freeze_binding"] = freeze_binding
    atomic_json(output / "analysis.json", analysis)
    _rewrite_analysis_receipt(output, binding, launch_manifest)
    valid, reason = reanalyze.verify_analysis_output(output, binding, launch_manifest)
    assert not valid
    assert expected_reason in reason
