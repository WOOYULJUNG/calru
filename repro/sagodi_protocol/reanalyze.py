"""Strict analysis-only queue for the immutable v1 pilot checkpoints.

This launcher deliberately does not reuse the v1 campaign's ``analysis/``
directory or its aggregation semantics.  It verifies the frozen parent, takes
an immutable snapshot of verified training checkpoints, and invokes
``phase1_analysis`` with the corrected v2 analysis freeze in a separate root.

``--available-only`` is a convenience for an in-progress parent campaign.  It
freezes only the checkpoints that are complete at planning time and can never
produce a campaign ``COMPLETE`` marker.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, IO, Mapping, Sequence

from .analysis_freeze import (
    analysis_freeze_fingerprint,
    load_analysis_freeze,
)
from .artifacts import (
    RECEIPT_IDENTITY_ENV,
    atomic_json,
    canonical_hash,
    sha256_file,
    strict_json_load,
    strict_json_loads,
    verify_completion_receipt,
)
from .config import expand_phase1_runs, load_protocol, protocol_fingerprint


ROOT_MARKER = ".calru_analysis_reanalysis_root_v1.json"
LOCK_FILE = ".reanalyze.lock"
EXPECTED_PARENT_RUNS = 15
EXPECTED_ANALYSIS_SCHEMA_VERSION = 4
SCOPE = "pilot_reanalysis_only"


class ReanalysisError(RuntimeError):
    """Base class for fail-closed reanalysis errors."""


class ParentVerificationError(ReanalysisError):
    """The immutable parent or one of its published runs is inconsistent."""


class IncompleteParentError(ReanalysisError):
    """The default all-run mode was requested before all training completed."""


class ReanalysisRunFailed(ReanalysisError):
    """At least one child analysis exited or verified unsuccessfully."""


@dataclass(frozen=True)
class RunBinding:
    run_id: str
    model_id: str
    model_seed: int
    learning_rate: float
    run_dir: Path
    checkpoint_sha256: str
    run_manifest_sha256: str
    training_receipt_sha256: str

    def as_manifest_entry(self, parent_root: Path) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "model_seed": self.model_seed,
            "learning_rate": self.learning_rate,
            "parent_run_path": self.run_dir.relative_to(parent_root).as_posix(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "run_manifest_sha256": self.run_manifest_sha256,
            "training_receipt_sha256": self.training_receipt_sha256,
        }


@dataclass(frozen=True)
class ParentSnapshot:
    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    protocol_path: Path
    protocol_sha256: str
    protocol_fingerprint: str
    evaluation_bank: Path
    perturbation_bank: Path
    available: Mapping[str, RunBinding]
    missing: Mapping[str, str]
    run_matrix: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class AnalysisJob:
    binding: RunBinding
    output_dir: Path


@dataclass
class ActiveJob:
    job: AnalysisJob
    gpu: int
    attempt_dir: Path
    log_path: Path
    process: subprocess.Popen[Any]
    log_handle: IO[str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ParentVerificationError(message)


def _safe_child(root: Path, raw_relative: Any, label: str) -> Path:
    if not isinstance(raw_relative, str) or not raw_relative:
        raise ParentVerificationError(f"{label} path is not a nonempty string")
    candidate = (root / raw_relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ParentVerificationError(f"{label} path escapes parent root") from exc
    return candidate


def _verify_bank(root: Path, specification: Any, label: str) -> Path:
    _require(isinstance(specification, Mapping), f"{label} specification is not an object")
    bank = _safe_child(root, specification.get("path"), label)
    _require(bank.is_file(), f"{label} is missing: {bank}")
    expected = specification.get("sha256")
    _require(sha256_file(bank) == expected, f"{label} SHA-256 mismatch")
    sidecar = bank.with_name(bank.name + ".sha256")
    _require(sidecar.is_file(), f"{label} SHA-256 sidecar is missing")
    _require(
        sha256_file(sidecar) == specification.get("sidecar_sha256"),
        f"{label} sidecar SHA-256 mismatch",
    )
    expected_sidecar = f"{expected}  {bank.name}\n".encode("utf-8")
    _require(sidecar.read_bytes() == expected_sidecar, f"{label} sidecar content mismatch")
    return bank


def _parent_receipt_metadata(manifest: Mapping[str, Any], run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "campaign_scientific_identity": manifest["scientific_identity"],
        "protocol_fingerprint": manifest["protocol_canonical_fingerprint"],
        "run_id": run["run_id"],
        "stage": "training",
        "freeze_id": manifest["run_matrix"][0]["freeze_id"],
        "protocol_canonical_fingerprint": manifest["protocol_canonical_fingerprint"],
        "campaign_identity": manifest["scientific_identity"],
        "model_id": run["model"]["id"],
        "model_seed": int(run["model_seed"]),
        "evaluation_bank_sha256": manifest["evaluation_bank"]["sha256"],
    }


def _verify_published_training_run(
    root: Path,
    manifest: Mapping[str, Any],
    run: Mapping[str, Any],
) -> RunBinding | None:
    run_id = str(run["run_id"])
    expectations = manifest.get("receipt_expectations")
    _require(isinstance(expectations, Mapping), "parent receipt expectations are missing")
    expectation = expectations.get(f"training:{run_id}")
    _require(isinstance(expectation, Mapping), f"missing training expectation for {run_id}")
    _require(expectation.get("run_id") == run_id, f"training expectation run_id mismatch: {run_id}")
    _require(expectation.get("stage") == "training", f"training expectation stage mismatch: {run_id}")
    expected_relative = f"runs/{run_id}"
    _require(
        expectation.get("output") == expected_relative,
        f"training expectation output mismatch: {run_id}",
    )
    run_dir = _safe_child(root, expectation["output"], f"training run {run_id}")
    if not run_dir.exists():
        return None
    _require(run_dir.is_dir(), f"published training output is not a directory: {run_id}")

    receipt_path = run_dir / "completion_receipt.json"
    valid, reason = verify_completion_receipt(
        receipt_path,
        expected_job_id=str(expectation.get("job_id")),
        expected_metadata=_parent_receipt_metadata(manifest, run),
    )
    _require(valid, f"published training receipt invalid for {run_id}: {reason}")
    checkpoint = run_dir / "checkpoint.pt"
    run_manifest_path = run_dir / "manifest.json"
    _require(checkpoint.is_file(), f"published checkpoint is missing: {run_id}")
    _require(run_manifest_path.is_file(), f"published run manifest is missing: {run_id}")
    checkpoint_sha = sha256_file(checkpoint)
    run_manifest_sha = sha256_file(run_manifest_path)
    try:
        run_manifest = strict_json_load(run_manifest_path)
        receipt = strict_json_load(receipt_path)
    except (OSError, ValueError, TypeError) as exc:
        raise ParentVerificationError(f"cannot parse published run {run_id}: {exc}") from exc
    _require(isinstance(run_manifest, Mapping), f"run manifest is not an object: {run_id}")
    expected_manifest = {
        "protocol_freeze_id": manifest["run_matrix"][0]["freeze_id"],
        "protocol_file_sha256": manifest["protocol_file_sha256"],
        "protocol_canonical_fingerprint": manifest["protocol_canonical_fingerprint"],
        "campaign_identity": manifest["scientific_identity"],
        "evaluation_bank_sha256": manifest["evaluation_bank"]["sha256"],
        "model_id": run["model"]["id"],
        "model_seed": int(run["model_seed"]),
        "checkpoint_sha256": checkpoint_sha,
    }
    for key, expected in expected_manifest.items():
        _require(
            run_manifest.get(key) == expected,
            f"run manifest {key} mismatch for {run_id}",
        )
    _require(receipt.get("schema_version") == 2, f"training receipt schema is not 2: {run_id}")
    artifacts = receipt.get("artifacts")
    _require(isinstance(artifacts, Mapping), f"training receipt artifacts invalid: {run_id}")
    _require(
        artifacts.get("checkpoint.pt") == checkpoint_sha,
        f"training receipt does not bind checkpoint.pt: {run_id}",
    )
    _require(
        artifacts.get("manifest.json") == run_manifest_sha,
        f"training receipt does not bind manifest.json: {run_id}",
    )
    return RunBinding(
        run_id=run_id,
        model_id=str(run["model"]["id"]),
        model_seed=int(run["model_seed"]),
        learning_rate=float(run["learning_rate"]),
        run_dir=run_dir,
        checkpoint_sha256=checkpoint_sha,
        run_manifest_sha256=run_manifest_sha,
        training_receipt_sha256=sha256_file(receipt_path),
    )


def inspect_parent_campaign(
    *,
    parent_root: Path | str,
    protocol_path: Path | str,
    analysis_contract: Mapping[str, Any],
) -> ParentSnapshot:
    """Verify immutable parent inputs and classify all 15 published runs."""

    root = Path(parent_root).expanduser().resolve(strict=True)
    _require(root.is_dir(), "parent root is not a directory")
    parent = analysis_contract.get("parent_training")
    _require(isinstance(parent, Mapping), "analysis freeze parent_training is missing")
    _require(root.name == parent.get("campaign_id"), "parent campaign directory name mismatch")
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), "parent manifest is missing")
    manifest_sha = sha256_file(manifest_path)
    _require(manifest_sha == parent.get("manifest_sha256"), "parent manifest SHA-256 mismatch")
    try:
        manifest = strict_json_load(manifest_path)
    except (OSError, ValueError, TypeError) as exc:
        raise ParentVerificationError(f"cannot parse parent manifest: {exc}") from exc
    _require(isinstance(manifest, Mapping), "parent manifest is not an object")
    _require(manifest.get("schema_version") == 2, "unsupported parent manifest schema")
    exact_parent_fields = {
        "campaign_id": parent.get("campaign_id"),
        "scientific_identity": parent.get("scientific_identity"),
        "protocol_canonical_fingerprint": parent.get("protocol_canonical_fingerprint"),
        "protocol_file_sha256": parent.get("protocol_file_sha256"),
    }
    for key, expected in exact_parent_fields.items():
        _require(manifest.get(key) == expected, f"parent manifest {key} mismatch")
    payload = manifest.get("scientific_identity_payload")
    _require(isinstance(payload, Mapping), "parent scientific identity payload is missing")
    _require(
        canonical_hash(payload) == manifest.get("scientific_identity"),
        "parent scientific identity payload hash mismatch",
    )
    for key in ("campaign_id", "protocol_file_sha256", "protocol_canonical_fingerprint"):
        _require(manifest.get(key) == payload.get(key), f"parent signed payload {key} mismatch")

    protocol_source = Path(protocol_path).expanduser().resolve(strict=True)
    protocol_sha = sha256_file(protocol_source)
    _require(protocol_sha == parent.get("protocol_file_sha256"), "parent protocol file SHA-256 mismatch")
    protocol = load_protocol(protocol_source)
    fingerprint = protocol_fingerprint(protocol)
    _require(
        fingerprint == parent.get("protocol_canonical_fingerprint"),
        "parent protocol canonical fingerprint mismatch",
    )
    _require(protocol.get("freeze_id") == parent.get("freeze_id"), "parent protocol freeze_id mismatch")
    expanded = tuple(expand_phase1_runs(protocol))
    _require(len(expanded) == EXPECTED_PARENT_RUNS, "protocol does not expand to exactly 15 parent runs")
    parent_matrix = manifest.get("run_matrix")
    _require(isinstance(parent_matrix, list), "parent run matrix is not an array")
    _require(parent_matrix == list(expanded), "parent manifest run matrix differs from protocol")

    evaluation_bank = _verify_bank(root, manifest.get("evaluation_bank"), "evaluation bank")
    perturbation_bank = _verify_bank(root, manifest.get("perturbation_bank"), "perturbation bank")
    available: dict[str, RunBinding] = {}
    missing: dict[str, str] = {}
    for run in expanded:
        binding = _verify_published_training_run(root, manifest, run)
        run_id = str(run["run_id"])
        if binding is None:
            missing[run_id] = "published verified training output is not yet available"
        else:
            available[run_id] = binding
    return ParentSnapshot(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        protocol_path=protocol_source,
        protocol_sha256=protocol_sha,
        protocol_fingerprint=fingerprint,
        evaluation_bank=evaluation_bank,
        perturbation_bank=perturbation_bank,
        available=available,
        missing=missing,
        run_matrix=expanded,
    )


def _resolve_python(python: str) -> str:
    candidate = Path(python).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.resolve()
    else:
        found = shutil.which(str(candidate))
        if found is None:
            raise FileNotFoundError(f"Python executable not found: {python}")
        resolved = Path(found).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Python executable not found: {resolved}")
    return str(resolved)


def _validated_gpus(gpus: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in gpus)
    if not values or any(value < 0 for value in values) or len(set(values)) != len(values):
        raise ValueError("GPU ids must be a nonempty sequence of unique nonnegative integers")
    return values


def _environment_fingerprint(python: str, gpus: Sequence[int]) -> dict[str, Any]:
    script = r'''
import json, platform, sys
import numpy as np
import torch
devices = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": p.name, "total_memory": int(p.total_memory),
                        "compute_capability": [int(p.major), int(p.minor)]})
print(json.dumps({"python": platform.python_version(), "executable": sys.executable,
                  "platform": platform.platform(), "numpy": np.__version__,
                  "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                  "cuda_available": torch.cuda.is_available(), "cuda_devices": devices},
                 sort_keys=True, allow_nan=False))
'''
    result = subprocess.run(
        [python, "-c", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    payload = strict_json_loads(result.stdout)
    available = {int(item["index"]) for item in payload.get("cuda_devices", [])}
    missing = sorted(set(map(int, gpus)).difference(available))
    if missing:
        raise ValueError(f"selected GPU ids are unavailable: {missing}")
    payload["python_executable_sha256"] = sha256_file(python)
    payload["selected_gpu_indices"] = list(map(int, gpus))
    return payload


def _git_state(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
    ).stdout.encode("utf-8")
    return {
        "code_commit": commit,
        "worktree_dirty": bool(porcelain.strip()),
        "porcelain_sha256": hashlib.sha256(porcelain).hexdigest(),
    }


def _source_hashes(repo_root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parent
    paths = sorted(package.glob("*.py"))
    paths.extend(
        repo_root / relative
        for relative in (
            "repro/legacy_code/exp71_pan_block_pulse_hold.py",
            "repro/legacy_code/exp72_structured_attractor_tasks.py",
            "repro/legacy_code/pan_block.py",
            "repro/legacy_code/plru_regularizers.py",
        )
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ReanalysisError(f"analysis source files are missing: {missing}")
    return {
        path.resolve().relative_to(repo_root).as_posix(): sha256_file(path)
        for path in sorted(paths)
    }


def _paths_overlap(parent: Path, artifact: Path) -> bool:
    try:
        artifact.relative_to(parent)
        return True
    except ValueError:
        pass
    try:
        parent.relative_to(artifact)
        return True
    except ValueError:
        return False


def _manifest_payload(
    *,
    snapshot: ParentSnapshot,
    freeze_path: Path,
    freeze: Mapping[str, Any],
    selected: Sequence[RunBinding],
    available_only: bool,
    python: str,
    gpus: Sequence[int],
    code: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    freeze_fp = analysis_freeze_fingerprint(freeze)
    run_entries = [item.as_manifest_entry(snapshot.root) for item in selected]
    selection_mode = "available_only_snapshot" if available_only else "all_15_parent_runs"
    snapshot_binding_sha256 = canonical_hash(
        {
            "mode": selection_mode,
            "selected_checkpoint_bindings": run_entries,
        }
    )
    payload = {
        "campaign_type": "analysis_only_parent_checkpoint_reanalysis",
        "scope": SCOPE,
        "confirmatory": False,
        "analysis_schema_version": EXPECTED_ANALYSIS_SCHEMA_VERSION,
        "claim_scope": dict(freeze["claim_scope"]),
        "analysis_freeze": {
            "freeze_id": freeze["freeze_id"],
            "file_sha256": sha256_file(freeze_path),
            "canonical_fingerprint": freeze_fp,
        },
        "parent": {
            "root": str(snapshot.root),
            "freeze_id": freeze["parent_training"]["freeze_id"],
            "manifest_sha256": snapshot.manifest_sha256,
            "campaign_id": snapshot.manifest["campaign_id"],
            "scientific_identity": snapshot.manifest["scientific_identity"],
            "protocol_file_sha256": snapshot.protocol_sha256,
            "protocol_canonical_fingerprint": snapshot.protocol_fingerprint,
            "evaluation_bank": {
                "path": snapshot.evaluation_bank.relative_to(snapshot.root).as_posix(),
                "sha256": snapshot.manifest["evaluation_bank"]["sha256"],
                "sidecar_sha256": snapshot.manifest["evaluation_bank"]["sidecar_sha256"],
            },
            "perturbation_bank": {
                "path": snapshot.perturbation_bank.relative_to(snapshot.root).as_posix(),
                "sha256": snapshot.manifest["perturbation_bank"]["sha256"],
                "sidecar_sha256": snapshot.manifest["perturbation_bank"]["sidecar_sha256"],
            },
        },
        "selection": {
            "mode": selection_mode,
            "snapshot_binding_sha256": snapshot_binding_sha256,
            "campaign_completion_allowed": not available_only,
            "expected_parent_run_count": EXPECTED_PARENT_RUNS,
            "selected_run_count": len(run_entries),
            "missing_at_plan_time": sorted(snapshot.missing),
        },
        "run_matrix": run_entries,
        "code": dict(code),
        "source_hashes": dict(source_hashes),
        "environment": dict(environment),
        "python": python,
        "gpus": list(map(int, gpus)),
    }
    identity = canonical_hash(payload)
    campaign_id = (
        f"{freeze['freeze_id']}-{freeze_fp[:12]}-"
        f"{snapshot.manifest_sha256[:12]}-{snapshot_binding_sha256[:12]}"
    )
    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "scientific_identity_payload": payload,
        "scientific_identity": identity,
        **payload,
        "analysis_freeze_path": str(freeze_path),
        "protocol_path": str(snapshot.protocol_path),
    }


def _prepare_root(root: Path, manifest: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    marker_path = root / ROOT_MARKER
    marker = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "scientific_identity": manifest["scientific_identity"],
        "scope": SCOPE,
        "parent_manifest_sha256": manifest["parent"]["manifest_sha256"],
        "analysis_freeze_canonical_fingerprint": manifest["analysis_freeze"]["canonical_fingerprint"],
    }
    if marker_path.exists():
        if strict_json_load(marker_path) != marker:
            raise ReanalysisError("analysis-only root marker differs; choose a new artifact root")
    else:
        unexpected_before_marker = [
            path.name for path in root.iterdir() if path.name != LOCK_FILE
        ]
        if unexpected_before_marker:
            raise ReanalysisError("artifact root is nonempty and lacks the analysis-only root marker")
        atomic_json(marker_path, marker)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if strict_json_load(manifest_path) != manifest:
            raise ReanalysisError("reanalysis manifest differs; choose a new artifact root")
    else:
        allowed = {ROOT_MARKER, LOCK_FILE}
        unexpected = [path.name for path in root.iterdir() if path.name not in allowed]
        if unexpected:
            raise ReanalysisError(f"unexpected files before manifest creation: {sorted(unexpected)}")
        atomic_json(manifest_path, manifest)


@contextmanager
def _exclusive_lock(root: Path):
    path = root / LOCK_FILE
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReanalysisError(f"another reanalysis orchestrator holds {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}) + "\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _analysis_receipt_metadata(manifest: Mapping[str, Any], binding: RunBinding) -> dict[str, Any]:
    return {
        "pilot_only": True,
        "smoke": False,
        "campaign_identity": manifest["parent"]["scientific_identity"],
        "checkpoint_sha256": binding.checkpoint_sha256,
        "evaluation_bank_sha256": manifest["parent"]["evaluation_bank"]["sha256"],
        "perturbation_bank_sha256": manifest["parent"]["perturbation_bank"]["sha256"],
        "analysis_freeze_id": manifest["analysis_freeze"]["freeze_id"],
        "analysis_freeze_canonical_fingerprint": manifest["analysis_freeze"]["canonical_fingerprint"],
        "analysis_schema_version": manifest["analysis_schema_version"],
        "L3": False,
    }


def verify_analysis_output(
    output_dir: Path,
    binding: RunBinding,
    manifest: Mapping[str, Any],
) -> tuple[bool, str]:
    receipt = output_dir / "completion_receipt.json"
    valid, reason = verify_completion_receipt(
        receipt,
        expected_job_id=f"phase1-analysis-{binding.run_id}",
        expected_metadata=_analysis_receipt_metadata(manifest, binding),
    )
    if not valid:
        return False, reason
    try:
        freeze_binding = strict_json_load(output_dir / "analysis_freeze_binding.json")
        analysis = strict_json_load(output_dir / "analysis.json")
        claim = strict_json_load(output_dir / "claim_gate.json")
        receipt_payload = strict_json_load(receipt)
    except (OSError, ValueError, TypeError) as exc:
        return False, f"cannot parse analysis binding: {exc}"
    receipt_artifacts = receipt_payload.get("artifacts")
    if not isinstance(receipt_artifacts, Mapping):
        return False, "analysis receipt artifacts are not an object"
    for name in ("analysis_freeze_binding.json", "analysis.json", "claim_gate.json"):
        path = output_dir / name
        if not path.is_file():
            return False, f"analysis output is missing {name}"
        if receipt_artifacts.get(name) != sha256_file(path):
            return False, f"analysis receipt does not bind {name}"
    expected_binding = {
        "analysis_freeze_id": manifest["analysis_freeze"]["freeze_id"],
        "analysis_freeze_file_sha256": manifest["analysis_freeze"]["file_sha256"],
        "analysis_freeze_canonical_fingerprint": manifest["analysis_freeze"]["canonical_fingerprint"],
    }
    for key, expected in expected_binding.items():
        if freeze_binding.get(key) != expected:
            return False, f"analysis freeze binding mismatch for {key}"
    parent_binding = freeze_binding.get("parent_training")
    if not isinstance(parent_binding, Mapping):
        return False, "analysis parent binding is missing"
    parent_expected = {
        "freeze_id": manifest["parent"]["freeze_id"],
        "campaign_id": manifest["parent"]["campaign_id"],
        "scientific_identity": manifest["parent"]["scientific_identity"],
        "manifest_sha256": manifest["parent"]["manifest_sha256"],
        "protocol_file_sha256": manifest["parent"]["protocol_file_sha256"],
        "protocol_canonical_fingerprint": manifest["parent"]["protocol_canonical_fingerprint"],
    }
    for key, expected in parent_expected.items():
        if parent_binding.get(key) != expected:
            return False, f"analysis parent binding mismatch for {key}"
    if freeze_binding.get("claim_scope") != manifest["claim_scope"]:
        return False, "analysis freeze claim_scope mismatch"
    if analysis.get("checkpoint_sha256") != binding.checkpoint_sha256:
        return False, "analysis.json checkpoint SHA-256 mismatch"
    if analysis.get("schema_version") != manifest["analysis_schema_version"]:
        return False, "analysis.json schema version mismatch"
    if analysis.get("pilot_only") is not True or analysis.get("smoke") is not False:
        return False, "analysis.json is not a non-smoke pilot analysis"
    embedded = analysis.get("analysis_freeze_binding")
    if not isinstance(embedded, Mapping):
        return False, "analysis.json analysis freeze binding is missing"
    if embedded != freeze_binding:
        return False, "analysis.json freeze binding differs from the receipted binding artifact"
    if not isinstance(claim, Mapping):
        return False, "claim_gate.json is not an object"
    if claim.get("schema_version") != manifest["analysis_schema_version"]:
        return False, "claim gate schema version mismatch"
    if claim.get("pilot_only") is not True or claim.get("smoke") is not False:
        return False, "claim gate is not a non-smoke pilot result"
    if claim.get("approximate_ca_claim_allowed") is not False:
        return False, "claim gate improperly allows an approximate-CA claim"
    levels = claim.get("levels")
    if not isinstance(levels, Mapping) or not isinstance(levels.get("L3"), Mapping):
        return False, "claim gate L3 level is missing"
    if levels["L3"].get("passed") is not False:
        return False, "claim gate L3 must be false for pilot reanalysis"
    gates = claim.get("gates")
    if not isinstance(gates, Mapping):
        return False, "claim gate gates object is missing"
    for gate_name in ("c4", "model_seeds"):
        gate = gates.get(gate_name)
        if not isinstance(gate, Mapping):
            return False, f"claim gate {gate_name} is missing"
        if gate.get("status") != "not_evaluated" or gate.get("passed") is not None:
            return False, f"claim gate {gate_name} must remain not_evaluated"
    return True, "verified analysis receipt/freeze/checkpoint binding"


def _unique_attempt(root: Path, run_id: str, label: str = "attempt") -> Path:
    parent = root / "attempts" / "analysis" / run_id
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        token = f"{time.time_ns()}-pid{os.getpid()}-{uuid.uuid4().hex[:8]}"
        candidate = parent / f"{label}-{token}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise ReanalysisError(f"could not allocate attempt directory for {run_id}")


def _preserve_invalid_output(root: Path, job: AnalysisJob) -> Path:
    destination = _unique_attempt(root, job.binding.run_id, label="recovered")
    destination.rmdir()
    os.replace(job.output_dir, destination)
    return destination


def _job_state(job: AnalysisJob, manifest: Mapping[str, Any]) -> tuple[str, str]:
    valid, reason = verify_analysis_output(job.output_dir, job.binding, manifest)
    if valid:
        return "completed", reason
    if job.output_dir.exists():
        return "invalid", reason
    return "pending", reason


def _status_payload(
    snapshot: ParentSnapshot,
    selected: Sequence[RunBinding],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    chosen = {item.run_id: item for item in selected}
    jobs: dict[str, Any] = {}
    for run in snapshot.run_matrix:
        run_id = str(run["run_id"])
        binding = chosen.get(run_id)
        jobs[run_id] = {
            "model_id": run["model"]["id"],
            "model_seed": int(run["model_seed"]),
            "planned": binding is not None,
            "parent_training": "verified" if run_id in snapshot.available else "pending",
            "state": "pending",
            "reason": (
                "awaiting analysis"
                if binding is not None
                else snapshot.missing.get(run_id, "not selected by immutable plan")
            ),
            "checkpoint_sha256": binding.checkpoint_sha256 if binding else None,
        }
    return {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "scientific_identity": manifest["scientific_identity"],
        "scope": SCOPE,
        "stage": "analysis",
        "available_only": manifest["selection"]["mode"] == "available_only_snapshot",
        "campaign_complete": False,
        "completion_marker_written": False,
        "expected_parent_runs": EXPECTED_PARENT_RUNS,
        "planned_runs": len(selected),
        "missing_parent_runs": sorted(snapshot.missing),
        "jobs": jobs,
        "started_at": time.time(),
        "updated_at": time.time(),
    }


def _write_status_and_matrix(root: Path, status: Mapping[str, Any]) -> None:
    atomic_json(root / "status.json", status)
    matrix = {
        "schema_version": 1,
        "scope": SCOPE,
        "campaign_id": status["campaign_id"],
        "campaign_complete": status["campaign_complete"],
        "runs": [
            {"run_id": run_id, **entry}
            for run_id, entry in sorted(status["jobs"].items())
        ],
    }
    atomic_json(root / "run_matrix.json", matrix)


def _verify_bound_inputs(
    *,
    snapshot: ParentSnapshot,
    selected: Sequence[RunBinding],
    freeze_path: Path,
    freeze: Mapping[str, Any],
    manifest: Mapping[str, Any],
    repo_root: Path,
) -> None:
    if sha256_file(snapshot.manifest_path) != manifest["parent"]["manifest_sha256"]:
        raise ParentVerificationError("parent manifest changed after planning")
    if sha256_file(snapshot.protocol_path) != manifest["parent"]["protocol_file_sha256"]:
        raise ParentVerificationError("parent protocol changed after planning")
    if sha256_file(freeze_path) != manifest["analysis_freeze"]["file_sha256"]:
        raise ReanalysisError("analysis freeze file changed after planning")
    if analysis_freeze_fingerprint(freeze) != manifest["analysis_freeze"]["canonical_fingerprint"]:
        raise ReanalysisError("analysis freeze canonical fingerprint changed")
    if _git_state(repo_root) != manifest["code"]:
        raise ReanalysisError("git state changed after reanalysis planning")
    if _source_hashes(repo_root) != manifest["source_hashes"]:
        raise ReanalysisError("analysis source hashes changed after planning")
    for label, key in (
        ("evaluation bank", "evaluation_bank"),
        ("perturbation bank", "perturbation_bank"),
    ):
        verified_path = _verify_bank(snapshot.root, snapshot.manifest[key], label)
        if verified_path != getattr(snapshot, key):
            raise ParentVerificationError(f"{label} path changed after planning")
    for binding in selected:
        if sha256_file(binding.run_dir / "checkpoint.pt") != binding.checkpoint_sha256:
            raise ParentVerificationError(f"checkpoint changed after planning: {binding.run_id}")
        if sha256_file(binding.run_dir / "manifest.json") != binding.run_manifest_sha256:
            raise ParentVerificationError(f"run manifest changed after planning: {binding.run_id}")
        if sha256_file(binding.run_dir / "completion_receipt.json") != binding.training_receipt_sha256:
            raise ParentVerificationError(f"training receipt changed after planning: {binding.run_id}")


def _command(
    *,
    job: AnalysisJob,
    attempt: Path,
    snapshot: ParentSnapshot,
    freeze_path: Path,
    python: str,
) -> list[str]:
    return [
        python,
        "-m",
        "repro.sagodi_protocol.phase1_analysis",
        "--protocol",
        str(snapshot.protocol_path),
        "--run-dir",
        str(job.binding.run_dir),
        "--output-dir",
        str(attempt),
        "--evaluation-bank",
        str(snapshot.evaluation_bank),
        "--perturbation-bank",
        str(snapshot.perturbation_bank),
        "--campaign-identity",
        str(snapshot.manifest["scientific_identity"]),
        "--analysis-freeze",
        str(freeze_path),
        "--device",
        "cuda:0",
    ]


def _terminate(active: Mapping[int, ActiveJob]) -> None:
    for item in active.values():
        if item.process.poll() is None:
            item.process.terminate()
    deadline = time.time() + 10.0
    for item in active.values():
        remaining = max(0.0, deadline - time.time())
        try:
            item.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            item.process.kill()
            item.process.wait()
        item.log_handle.close()


def _run_queue(
    *,
    root: Path,
    jobs: Sequence[AnalysisJob],
    gpus: Sequence[int],
    snapshot: ParentSnapshot,
    freeze_path: Path,
    freeze: Mapping[str, Any],
    manifest: Mapping[str, Any],
    repo_root: Path,
    python: str,
    status: dict[str, Any],
) -> None:
    pending = list(jobs)
    active: dict[int, ActiveJob] = {}
    try:
        while pending or active:
            while pending and len(active) < len(gpus):
                gpu = next(value for value in gpus if value not in active)
                job = pending.pop(0)
                state, reason = _job_state(job, manifest)
                if state == "completed":
                    status["jobs"][job.binding.run_id].update(state="completed", reason=reason)
                    status["updated_at"] = time.time()
                    _write_status_and_matrix(root, status)
                    continue
                if state == "invalid":
                    preserved = _preserve_invalid_output(root, job)
                    status["jobs"][job.binding.run_id]["preserved_invalid_output"] = str(preserved)
                _verify_bound_inputs(
                    snapshot=snapshot, selected=[item.binding for item in jobs],
                    freeze_path=freeze_path, freeze=freeze, manifest=manifest,
                    repo_root=repo_root,
                )
                attempt = _unique_attempt(root, job.binding.run_id)
                log_dir = root / "logs" / job.binding.run_id
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{attempt.name}.log"
                log_handle = log_path.open("w", encoding="utf-8")
                env = os.environ.copy()
                for identity_variable in RECEIPT_IDENTITY_ENV.values():
                    env.pop(identity_variable, None)
                env.update(
                    {
                        "PYTHONNOUSERSITE": "1",
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "CUBLAS_WORKSPACE_CONFIG": env.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
                    }
                )
                process = subprocess.Popen(
                    _command(job=job, attempt=attempt, snapshot=snapshot,
                             freeze_path=freeze_path, python=python),
                    cwd=repo_root,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active[gpu] = ActiveJob(job, gpu, attempt, log_path, process, log_handle)
                status["jobs"][job.binding.run_id].update(
                    state="running", reason=f"pid={process.pid} gpu={gpu}",
                    attempt=str(attempt), log=str(log_path),
                )
                status["updated_at"] = time.time()
                _write_status_and_matrix(root, status)

            finished = [gpu for gpu, item in active.items() if item.process.poll() is not None]
            if not finished:
                time.sleep(0.2)
                continue
            for gpu in finished:
                item = active.pop(gpu)
                returncode = int(item.process.returncode or 0)
                item.log_handle.close()
                entry = status["jobs"][item.job.binding.run_id]
                if returncode != 0:
                    entry.update(state="failed", reason=f"child exit code {returncode}")
                else:
                    _verify_bound_inputs(
                        snapshot=snapshot, selected=[job.binding for job in jobs],
                        freeze_path=freeze_path, freeze=freeze, manifest=manifest,
                        repo_root=repo_root,
                    )
                    valid, reason = verify_analysis_output(
                        item.attempt_dir, item.job.binding, manifest
                    )
                    if not valid:
                        entry.update(state="failed", reason=f"invalid child receipt: {reason}")
                    else:
                        if item.job.output_dir.exists():
                            existing, _ = _job_state(item.job, manifest)
                            if existing == "completed":
                                entry.update(state="completed", reason="concurrent verified publication")
                                status["updated_at"] = time.time()
                                _write_status_and_matrix(root, status)
                                continue
                            _preserve_invalid_output(root, item.job)
                        item.job.output_dir.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(item.attempt_dir, item.job.output_dir)
                        published, published_reason = _job_state(item.job, manifest)
                        if published != "completed":
                            entry.update(state="failed", reason=f"published output invalid: {published_reason}")
                        else:
                            entry.update(state="completed", reason=published_reason)
                status["updated_at"] = time.time()
                _write_status_and_matrix(root, status)
    except BaseException:
        _terminate(active)
        raise


def _write_complete(root: Path, manifest: Mapping[str, Any], jobs: Sequence[AnalysisJob]) -> None:
    receipts: dict[str, str] = {}
    for job in jobs:
        valid, reason = verify_analysis_output(job.output_dir, job.binding, manifest)
        if not valid:
            raise ReanalysisError(f"cannot complete campaign; {job.binding.run_id}: {reason}")
        receipts[job.binding.run_id] = sha256_file(job.output_dir / "completion_receipt.json")
    payload = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "scientific_identity": manifest["scientific_identity"],
        "scope": SCOPE,
        "confirmatory": False,
        "parent_manifest_sha256": manifest["parent"]["manifest_sha256"],
        "analysis_freeze_canonical_fingerprint": manifest["analysis_freeze"]["canonical_fingerprint"],
        "analysis_receipt_sha256": receipts,
    }
    path = root / "COMPLETE"
    if path.exists():
        if strict_json_load(path) != payload:
            raise ReanalysisError("existing COMPLETE marker is inconsistent")
    else:
        atomic_json(path, payload)


def run_reanalysis(
    *,
    parent_root: Path | str,
    artifact_root: Path | str,
    protocol_path: Path | str,
    analysis_freeze_path: Path | str,
    python: str,
    gpus: Sequence[int],
    available_only: bool = False,
) -> Path:
    """Plan, resume, and execute the corrected analysis-only GPU queue."""

    freeze_path = Path(analysis_freeze_path).expanduser().resolve(strict=True)
    freeze = load_analysis_freeze(freeze_path)
    snapshot = inspect_parent_campaign(
        parent_root=parent_root,
        protocol_path=protocol_path,
        analysis_contract=freeze,
    )
    if snapshot.missing and not available_only:
        raise IncompleteParentError(
            f"default reanalysis requires all {EXPECTED_PARENT_RUNS} verified training runs; "
            f"missing {len(snapshot.missing)}: {sorted(snapshot.missing)}"
        )
    selected = [
        snapshot.available[str(run["run_id"])]
        for run in snapshot.run_matrix
        if str(run["run_id"]) in snapshot.available
    ]
    if not available_only and len(selected) != EXPECTED_PARENT_RUNS:
        raise IncompleteParentError("full reanalysis plan does not contain exactly 15 runs")

    root = Path(artifact_root).expanduser().resolve()
    if _paths_overlap(snapshot.root, root):
        raise ReanalysisError("analysis artifact root must not overlap the immutable parent root")
    repo_root = Path(__file__).resolve().parents[2]
    resolved_python = _resolve_python(python)
    selected_gpus = _validated_gpus(gpus)
    code = _git_state(repo_root)
    sources = _source_hashes(repo_root)
    environment = _environment_fingerprint(resolved_python, selected_gpus)
    manifest = _manifest_payload(
        snapshot=snapshot,
        freeze_path=freeze_path,
        freeze=freeze,
        selected=selected,
        available_only=available_only,
        python=resolved_python,
        gpus=selected_gpus,
        code=code,
        source_hashes=sources,
        environment=environment,
    )
    root.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(root):
        # Root identity and its immutable manifest are initialized and compared
        # only while the exclusive guard is held.  This prevents two distinct
        # first-launch configurations from cross-publishing marker/manifest
        # bytes into the same previously empty directory.
        _prepare_root(root, manifest)
        _verify_bound_inputs(
            snapshot=snapshot, selected=selected, freeze_path=freeze_path,
            freeze=freeze, manifest=manifest, repo_root=repo_root,
        )
        status = _status_payload(snapshot, selected, manifest)
        jobs = [
            AnalysisJob(binding=item, output_dir=root / "analysis" / item.run_id)
            for item in selected
        ]
        for job in jobs:
            state, reason = _job_state(job, manifest)
            if state == "completed":
                status["jobs"][job.binding.run_id].update(state="completed", reason=reason)
        _write_status_and_matrix(root, status)
        _run_queue(
            root=root, jobs=jobs, gpus=selected_gpus, snapshot=snapshot,
            freeze_path=freeze_path, freeze=freeze, manifest=manifest,
            repo_root=repo_root, python=resolved_python, status=status,
        )
        planned_not_completed = [
            run_id
            for run_id, item in status["jobs"].items()
            if item["planned"] and item["state"] != "completed"
        ]
        failed = [run_id for run_id, item in status["jobs"].items() if item["state"] == "failed"]
        if planned_not_completed and not failed:
            for run_id in planned_not_completed:
                status["jobs"][run_id].update(
                    state="failed",
                    reason="analysis queue ended without a verified published output",
                )
            failed = list(planned_not_completed)
        if failed:
            status.update(stage="failed", campaign_complete=False, completion_marker_written=False)
            if available_only:
                status["completion_prohibited_reason"] = (
                    "--available-only cannot mark a partial pilot snapshot complete"
                )
        elif available_only:
            status.update(
                stage="partial_complete",
                campaign_complete=False,
                completion_marker_written=False,
                completion_prohibited_reason=(
                    "--available-only freezes a partial pilot snapshot and cannot mark the campaign complete"
                ),
            )
        else:
            if len(jobs) != EXPECTED_PARENT_RUNS or any(
                status["jobs"][job.binding.run_id]["state"] != "completed" for job in jobs
            ):
                raise ReanalysisError("full reanalysis ended without 15 verified outputs")
            _write_complete(root, manifest, jobs)
            status.update(stage="complete", campaign_complete=True, completion_marker_written=True)
        status["updated_at"] = time.time()
        _write_status_and_matrix(root, status)
        if failed:
            raise ReanalysisRunFailed(f"reanalysis failed for {failed}")
    return root


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--analysis-freeze", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--available-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    gpus = tuple(int(item.strip()) for item in args.gpus.split(",") if item.strip())
    try:
        output = run_reanalysis(
            parent_root=args.parent_root,
            artifact_root=args.artifact_root,
            protocol_path=args.protocol,
            analysis_freeze_path=args.analysis_freeze,
            python=args.python,
            gpus=gpus,
            available_only=bool(args.available_only),
        )
    except ReanalysisRunFailed as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), flush=True)
        return 1
    print(json.dumps({"status": "finished", "artifact_root": str(output)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IncompleteParentError",
    "ParentVerificationError",
    "ReanalysisError",
    "ReanalysisRunFailed",
    "inspect_parent_campaign",
    "run_reanalysis",
    "verify_analysis_output",
]
