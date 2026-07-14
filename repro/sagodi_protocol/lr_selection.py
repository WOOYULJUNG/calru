"""Strict, resumable training-only learning-rate selection campaign.

This module intentionally stops after the 100-update selector runs.  It does
not run manifold diagnostics and none of its artifacts are evidence for a
continuous-attractor claim.  A full (non-smoke) result chooses one learning
rate per model by the arithmetic mean of the *online training loss at update
100* across all five frozen selection seeds.  Evaluation-bank metrics are
recorded for audit only and never enter the selector.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import io
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import (
    RECEIPT_IDENTITY_ENV,
    atomic_bytes,
    atomic_json,
    canonical_hash,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from .config import (
    AngularTaskSpec,
    expand_phase1_runs,
    load_protocol,
    protocol_fingerprint,
)
from .orchestrate import (
    OUTPUT_DIR_TOKEN,
    _acquire_campaign_lock,
    _environment_fingerprint,
    _git_state,
    _materialize_evaluation_bank,
    _preserve_invalid_output,
    _process_identity,
    _resolve_python,
    _source_hashes,
    _unique_attempt_path,
    _validated_gpu_ids,
)
from .tasks import load_fixed_bank
from .train import _validate_evaluation_batch


ROOT_MARKER = ".calru_lr_selection_root_v1.json"
MANIFEST_NAME = "lr_selection_manifest.json"
SUMMARY_NAME = "lr_selection_summary.json"
MATRIX_NAME = "lr_selection_run_matrix.csv"
SELECTION_RECEIPT_NAME = "lr_selection_receipt.json"
COMPLETION_RECEIPT_NAME = "completion_receipt.json"
COMPLETE_MARKER_NAME = "COMPLETE"
FAILURE_DIR_NAME = "failures"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

PHASE0_MODEL_ARTIFACT_NAMES = (
    "blank_map_trace.npz",
    "blank_map_trace_metadata.json",
    "determinism_check.json",
    "exactness_screen.json",
    "float64_subset_check.json",
    "hidden_cache_check.json",
    "jacobian_check.json",
    "state_spec.json",
    "state_transition_audit.json",
)

EXPECTED_MODELS = (
    "ca_lru",
    "no_rp",
    "gru_sagodi_width96",
    "gru_sagodi_param135",
)
EXPECTED_SELECTION_SEEDS = (1100, 1101, 1102, 1103, 1104)
EXPECTED_LEARNING_RATES = (1e-2, 1e-3, 1e-4, 1e-5)
EXPECTED_MODEL_WIDTHS = {
    "ca_lru": 96,
    "no_rp": 96,
    "gru_sagodi_width96": 96,
    "gru_sagodi_param135": 135,
}
EXPECTED_PARAMETER_COUNTS = {
    "ca_lru": 56834,
    "no_rp": 56834,
    "gru_sagodi_width96": 28898,
    "gru_sagodi_param135": 56432,
}
EXPECTED_BATCH_SIZE = 64
REQUIRED_UPDATE = 100
SELECTION_RULE_ID = "mean_online_training_loss_at_update_100"
SELECTION_SCOPE = "training_only_lr_selection_no_manifold_analysis_no_ca_evidence"


class SelectionProtocolError(ValueError):
    """The selector freeze does not encode the required 4 x 5 x 4 design."""


class IncompleteSelectionError(RuntimeError):
    """At least one model has no eligible learning rate."""


@dataclass(frozen=True)
class SelectionRun:
    run_id: str
    model_id: str
    hidden_width: int
    parameter_count: int
    batch_size: int
    model_seed: int
    learning_rate: float
    required_update: int = REQUIRED_UPDATE

    @property
    def receipt_job_id(self) -> str:
        return (
            f"{self.model_id}-seed{int(self.model_seed)}-"
            f"lr{float(self.learning_rate):g}"
        )

    def payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "hidden_width": int(self.hidden_width),
            "parameter_count": int(self.parameter_count),
            "batch_size": int(self.batch_size),
            "model_seed": int(self.model_seed),
            "learning_rate": float(self.learning_rate),
            "required_update": int(self.required_update),
            "selection_metric": "online_training_masked_mse",
        }


def _as_sequence(value: Any, label: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SelectionProtocolError(f"{label} must be an array")
    return tuple(value)


def _selection_rule_payload(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return the executable selector rule, independent of prose labels."""

    training = protocol["phase1_ring_pilot"]["training"]
    declared = training["learning_rate"].get("selection_rule")
    expected_declared = {
        "primary_metric": "mean_online_training_loss_at_update_100",
        "seed_aggregation": "arithmetic_mean_across_five_selection_model_seeds",
        "selection_scope": "separately_per_model",
        "winner": "minimum_primary_metric",
        "validation_metrics_role": "secondary_non_selecting",
        "tie_policy": "smaller_numeric_learning_rate",
        "failed_run_policy": (
            "no_scientific_failure_inference_from_nonzero_exit_oom_kill_or_invalid_receipt"
        ),
        "failed_run_retry_policy": (
            "infrastructure_or_unknown_failure_aborts_campaign_and_is_resume_eligible"
        ),
        "campaign_completion": "all_80_runs_must_have_verified_success_receipts",
    }
    if declared != expected_declared:
        raise SelectionProtocolError(
            "declared LR-selection rule differs from the executable rule"
        )
    return {
        "rule_id": SELECTION_RULE_ID,
        "objective": "minimize",
        "statistic": "arithmetic_mean",
        "source_metric": "recorded_online_training_masked_mse",
        "source_update": REQUIRED_UPDATE,
        "aggregation_unit": "model_x_learning_rate_across_selection_model_seeds",
        "all_five_seeds_required": True,
        "nonfinite_missing_or_failed_lr_is_ineligible": True,
        "tie_break": "smaller_numeric_learning_rate",
        "validation_and_task_metrics_affect_selection": False,
        "declared_protocol_rule": expected_declared,
    }


def build_selection_plan(protocol: Mapping[str, Any]) -> tuple[SelectionRun, ...]:
    """Expand and independently audit the frozen 80-run selector matrix."""

    phase = protocol.get("phase1_ring_pilot")
    if not isinstance(phase, Mapping):
        raise SelectionProtocolError("phase1_ring_pilot must be an object")
    if phase.get("protocol_track") != "sagodi_paper_aligned_lr_selection":
        raise SelectionProtocolError("protocol is not the Ságodi LR-selection track")
    training = phase.get("training")
    seeds = protocol.get("seed_policy")
    if not isinstance(training, Mapping) or not isinstance(seeds, Mapping):
        raise SelectionProtocolError("training and seed_policy must be objects")

    raw_models = _as_sequence(phase.get("models"), "phase1 models")
    model_ids: list[str] = []
    widths: dict[str, int] = {}
    parameter_counts: dict[str, int] = {}
    for raw in raw_models:
        if not isinstance(raw, Mapping):
            raise SelectionProtocolError("each model entry must be an object")
        model_id = str(raw.get("id"))
        if model_id in widths:
            raise SelectionProtocolError(f"duplicate model id: {model_id}")
        try:
            width = int(raw["hidden_width"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SelectionProtocolError(
                f"model {model_id} must declare an integer hidden_width"
            ) from exc
        if width <= 0 or isinstance(raw.get("hidden_width"), bool):
            raise SelectionProtocolError(f"model {model_id} has invalid hidden_width")
        model_ids.append(model_id)
        widths[model_id] = width
        try:
            parameter_count = int(raw["parameter_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SelectionProtocolError(
                f"model {model_id} must declare an integer parameter_count"
            ) from exc
        if parameter_count <= 0 or isinstance(raw.get("parameter_count"), bool):
            raise SelectionProtocolError(f"model {model_id} has invalid parameter_count")
        parameter_counts[model_id] = parameter_count
    if tuple(model_ids) != EXPECTED_MODELS:
        raise SelectionProtocolError(
            f"selection models must be exactly {EXPECTED_MODELS!r}, got {tuple(model_ids)!r}"
        )

    raw_selection_seeds = _as_sequence(
        seeds.get("selection_model_seeds"), "selection_model_seeds"
    )
    if any(isinstance(item, bool) for item in raw_selection_seeds):
        raise SelectionProtocolError("selection seeds must be integers, not booleans")
    selection_seeds = tuple(int(item) for item in raw_selection_seeds)
    if selection_seeds != EXPECTED_SELECTION_SEEDS:
        raise SelectionProtocolError(
            f"selection seeds must be exactly {EXPECTED_SELECTION_SEEDS!r}"
        )

    learning_rate = training.get("learning_rate")
    if not isinstance(learning_rate, Mapping):
        raise SelectionProtocolError("training.learning_rate must be an object")
    raw_grid = learning_rate.get("grid")
    if raw_grid is None:
        # This fallback keeps the plan reader useful while rejecting legacy
        # pilot freezes, whose seed/model/track checks above cannot pass.
        raw_grid = learning_rate.get("pilot_grid")
    grid = tuple(float(item) for item in _as_sequence(raw_grid, "learning-rate grid"))
    if grid != EXPECTED_LEARNING_RATES:
        raise SelectionProtocolError(
            f"learning-rate grid must be exactly {EXPECTED_LEARNING_RATES!r}"
        )
    active = tuple(
        float(item)
        for item in _as_sequence(
            learning_rate.get("active_launch_values"), "active learning rates"
        )
    )
    if active != EXPECTED_LEARNING_RATES:
        raise SelectionProtocolError("all four grid values must be active")
    if int(training.get("optimizer_updates", -1)) != REQUIRED_UPDATE:
        raise SelectionProtocolError("LR selector must run exactly 100 optimizer updates")

    # Reuse the validated configuration expander so the selector, train CLI,
    # and any future main-training freeze share exactly the same run IDs and
    # per-model widths.  The independent checks above prevent a permissive
    # expander change from silently altering the selector design.
    expanded = expand_phase1_runs(protocol)
    plan: list[SelectionRun] = []
    for item in expanded:
        model_id = str(item["model"]["id"])
        model_seed = int(item["model_seed"])
        lr = float(item["learning_rate"])
        width = int(item["width"])
        if model_id not in widths or width != widths[model_id]:
            raise SelectionProtocolError("expanded run has the wrong model width")
        if model_seed not in selection_seeds or lr not in grid:
            raise SelectionProtocolError("expanded run lies outside the selector cross-product")
        plan.append(
            SelectionRun(
                run_id=str(item["run_id"]),
                model_id=model_id,
                hidden_width=width,
                parameter_count=parameter_counts[model_id],
                batch_size=int(item["batch_size"]),
                model_seed=model_seed,
                learning_rate=lr,
            )
        )
    expected = phase.get("run_matrix", {}).get("expected_training_runs")
    if expected != 80 or len(plan) != 80:
        raise SelectionProtocolError("selector run matrix must contain exactly 80 runs")
    if len({run.run_id for run in plan}) != len(plan):
        raise SelectionProtocolError("selector run ids are not unique")
    return tuple(plan)


def aggregate_lr_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str] = EXPECTED_MODELS,
    seeds: Sequence[int] = EXPECTED_SELECTION_SEEDS,
    learning_rates: Sequence[float] = EXPECTED_LEARNING_RATES,
    required_update: int = REQUIRED_UPDATE,
) -> dict[str, Any]:
    """Pure selector implementation with strict completeness and tie handling."""

    models = tuple(str(value) for value in models)
    seeds = tuple(int(value) for value in seeds)
    learning_rates = tuple(float(value) for value in learning_rates)
    expected_keys = {
        (model, seed, lr)
        for model in models
        for seed in seeds
        for lr in learning_rates
    }
    indexed: dict[tuple[str, int, float], Mapping[str, Any]] = {}
    unexpected: list[tuple[str, int, float]] = []
    for row in rows:
        key = (
            str(row.get("model_id")),
            int(row.get("model_seed", -1)),
            float(row.get("learning_rate", float("nan"))),
        )
        if key in indexed:
            raise ValueError(f"duplicate LR-selection row: {key}")
        indexed[key] = row
        if key not in expected_keys:
            unexpected.append(key)
    if unexpected:
        raise ValueError(f"unexpected LR-selection rows: {unexpected}")

    candidates: dict[str, list[dict[str, Any]]] = {model: [] for model in models}
    for model in models:
        for lr in learning_rates:
            failures: list[str] = []
            values: list[float] = []
            per_seed: list[dict[str, Any]] = []
            for seed in seeds:
                key = (model, seed, lr)
                row = indexed.get(key)
                if row is None:
                    failures.append(f"seed{seed}:missing")
                    per_seed.append({"model_seed": seed, "eligible": False, "reason": "missing"})
                    continue
                status = str(row.get("status", ""))
                try:
                    completed = int(row.get("completed_updates", -1))
                    loss = float(row.get("loss_at_required_update", float("nan")))
                except (TypeError, ValueError):
                    completed = -1
                    loss = float("nan")
                reason: str | None = None
                if status != "complete":
                    reason = f"status={status or 'missing'}"
                elif completed < int(required_update):
                    reason = f"completed_updates={completed}"
                elif not math.isfinite(loss):
                    reason = "loss_nonfinite_or_missing"
                if reason is not None:
                    failures.append(f"seed{seed}:{reason}")
                    per_seed.append(
                        {"model_seed": seed, "eligible": False, "reason": reason}
                    )
                else:
                    values.append(loss)
                    per_seed.append(
                        {
                            "model_seed": seed,
                            "eligible": True,
                            "loss_at_required_update": loss,
                        }
                    )
            eligible = not failures and len(values) == len(seeds)
            mean_loss = math.fsum(values) / len(values) if eligible else None
            candidates[model].append(
                {
                    "learning_rate": lr,
                    "eligible": eligible,
                    "mean_online_training_loss_at_required_update": mean_loss,
                    "required_update": int(required_update),
                    "required_seed_count": len(seeds),
                    "ineligibility_reasons": failures,
                    "per_seed": per_seed,
                }
            )

    winners: dict[str, Any] = {}
    incomplete_models: list[str] = []
    for model in models:
        eligible = [item for item in candidates[model] if item["eligible"]]
        if not eligible:
            incomplete_models.append(model)
            continue
        winner = min(
            eligible,
            key=lambda item: (
                float(item["mean_online_training_loss_at_required_update"]),
                float(item["learning_rate"]),
            ),
        )
        winners[model] = {
            "learning_rate": float(winner["learning_rate"]),
            "mean_online_training_loss_at_required_update": float(
                winner["mean_online_training_loss_at_required_update"]
            ),
            "required_update": int(required_update),
            "selection_seed_count": len(seeds),
        }
    return {
        "schema_version": 1,
        "selection_rule": {
            "rule_id": SELECTION_RULE_ID,
            "statistic": "arithmetic_mean",
            "source_metric": "recorded_online_training_masked_mse",
            "source_update": int(required_update),
            "all_expected_seeds_required": True,
            "tie_break": "smaller_numeric_learning_rate",
            "validation_and_task_metrics_affect_selection": False,
        },
        "complete": not incomplete_models,
        "expected_rows": len(expected_keys),
        "observed_rows": len(indexed),
        "incomplete_models": incomplete_models,
        "candidates": candidates,
        "winners": winners,
    }


def strict_aggregate_lr_selection(
    rows: Sequence[Mapping[str, Any]], **kwargs: Any
) -> dict[str, Any]:
    result = aggregate_lr_selection(rows, **kwargs)
    all_candidates = [
        candidate
        for candidates in result["candidates"].values()
        for candidate in candidates
    ]
    exact_success_matrix = (
        result["expected_rows"] == 80
        and result["observed_rows"] == 80
        and len(all_candidates) == 16
        and all(candidate["eligible"] for candidate in all_candidates)
    )
    if not exact_success_matrix:
        ineligible = [
            f"{model}@{candidate['learning_rate']:g}"
            for model, candidates in result["candidates"].items()
            for candidate in candidates
            if not candidate["eligible"]
        ]
        raise IncompleteSelectionError(
            "strict LR selection requires exactly 80 unique successful rows "
            "(4 models x 5 exact seeds x 4 exact LRs); "
            f"observed={result['observed_rows']}, ineligible={ineligible}"
        )
    return result


def _receipt_metadata(manifest: Mapping[str, Any], stage: str, run_id: str) -> dict[str, str]:
    return {
        "campaign_scientific_identity": str(manifest["scientific_identity"]),
        "protocol_fingerprint": str(manifest["protocol_canonical_fingerprint"]),
        "run_id": str(run_id),
        "stage": str(stage),
    }


def _child_environment(
    manifest: Mapping[str, Any], *, stage: str, run_id: str, gpu: int | None = None
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
        env["CALRU_PHYSICAL_GPU_ID"] = str(int(gpu))
    metadata = _receipt_metadata(manifest, stage, run_id)
    for key, variable in RECEIPT_IDENTITY_ENV.items():
        env[variable] = metadata[key]
    return env


def _require_unmasked_parent_cuda_environment() -> None:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError(
            "preexisting CUDA_VISIBLE_DEVICES is forbidden because --gpus uses physical ids"
        )


def _configure_linux_parent_death_signal(
    expected_parent_pid: int,
    *,
    prctl_call: Any | None = None,
    getppid: Any | None = None,
) -> bool:
    """Install PDEATHSIG and report whether the original parent still exists.

    ``Popen`` does not return until the child has crossed its pre-exec setup,
    so installing this before ``exec`` closes the launch-to-status-write orphan
    window.  The PPID comparison handles the race where the parent died before
    the child managed to install PDEATHSIG.
    """

    if not sys.platform.startswith("linux"):
        raise RuntimeError("LR selection requires Linux PR_SET_PDEATHSIG support")
    expected_parent_pid = int(expected_parent_pid)
    if expected_parent_pid <= 1:
        raise ValueError("expected parent pid must identify a live campaign process")
    if prctl_call is None:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl_call = libc.prctl
    result = int(prctl_call(1, signal.SIGKILL, 0, 0, 0))  # PR_SET_PDEATHSIG
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, "prctl(PR_SET_PDEATHSIG, SIGKILL) failed")
    observed_parent = int((getppid or os.getppid)())
    return observed_parent == expected_parent_pid


def _parent_death_sigkill_preexec(expected_parent_pid: int) -> Any:
    """Return the child hook that makes an unrecorded orphan impossible."""

    def install() -> None:
        if not _configure_linux_parent_death_signal(expected_parent_pid):
            # The parent died before PDEATHSIG was installed.  SIGKILL cannot
            # be caught; the fallback exit only documents the intended state.
            os.kill(os.getpid(), signal.SIGKILL)
            os._exit(128 + signal.SIGKILL)

    return install


def _root_marker_payload(
    protocol: Mapping[str, Any], fingerprint: str, smoke: bool
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_type": "strict_training_only_lr_selection",
        "freeze_id": str(protocol["freeze_id"]),
        "protocol_canonical_fingerprint": str(fingerprint),
        "smoke": bool(smoke),
        "cross_campaign_resume_allowed": False,
        "scope": SELECTION_SCOPE,
    }


def _prepare_selection_root(
    root: Path, protocol: Mapping[str, Any], fingerprint: str, smoke: bool
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ROOT_MARKER
    expected = _root_marker_payload(protocol, fingerprint, smoke)
    if marker.exists():
        if strict_json_load(marker) != expected:
            raise RuntimeError("LR-selection root marker differs; choose a new artifact root")
        return
    if any(root.iterdir()):
        raise RuntimeError("artifact root is nonempty and lacks the LR-selection marker")
    atomic_json(marker, expected)


def _write_or_check_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    path = root / MANIFEST_NAME
    if path.exists():
        if strict_json_load(path) != dict(manifest):
            raise RuntimeError("LR-selection campaign manifest differs")
    else:
        atomic_json(path, dict(manifest))


def _build_manifest(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    fingerprint: str,
    evaluation_bank: Mapping[str, Any],
    plan: Sequence[SelectionRun],
    source_hashes: Mapping[str, str],
    git_state: Mapping[str, Any],
    environment: Mapping[str, Any],
    python: str,
    gpus: Sequence[int],
    smoke: bool,
) -> dict[str, Any]:
    task_spec = AngularTaskSpec.from_protocol(protocol)
    run_matrix = [run.payload() for run in plan]
    scientific_payload = {
        "campaign_id": f"{protocol['freeze_id']}-{fingerprint[:12]}",
        "campaign_type": "strict_training_only_lr_selection",
        "scope": SELECTION_SCOPE,
        "pilot_only": True,
        "confirmatory": False,
        "ca_evidence": False,
        "manifold_analysis_performed": False,
        "freeze_eligible": False,
        "freeze_status": "pending_all_80_verified_success_receipts",
        "smoke": bool(smoke),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_canonical_fingerprint": fingerprint,
        "source_protocol_sha256": protocol["source_protocol"]["sha256"],
        "resolved_task_spec": task_spec.resolved_payload(),
        "resolved_task_spec_sha256": task_spec.fingerprint(),
        "source_hashes": dict(source_hashes),
        "code": dict(git_state),
        "environment": dict(environment),
        "determinism_environment": {
            "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
            "parent_CUDA_VISIBLE_DEVICES": "required_unset",
            "child_GPU_mapping": "physical_id_isolated_as_cuda_0",
        },
        "python": str(Path(python).resolve()),
        "gpus": [int(gpu) for gpu in gpus],
        "evaluation_bank": dict(evaluation_bank),
        "selection_rule": _selection_rule_payload(protocol),
        "run_matrix": run_matrix,
        "expected_training_runs": 80,
        "phase0_required": True,
    }
    return {
        "schema_version": 1,
        **scientific_payload,
        "protocol_file": str(protocol_path),
        "scientific_identity_payload": scientific_payload,
        "scientific_identity": canonical_hash(scientific_payload),
    }


def _verify_manifest_identity(
    root: Path, manifest: Mapping[str, Any], *, require_disk: bool = True
) -> tuple[bool, str]:
    signed = manifest.get("scientific_identity_payload")
    if not isinstance(signed, Mapping):
        return False, "scientific_identity_payload is not an object"
    if canonical_hash(signed) != manifest.get("scientific_identity"):
        return False, "scientific identity does not match its signed payload"
    for key, value in signed.items():
        if manifest.get(key) != value:
            return False, f"manifest field {key!r} differs from signed payload"
    if manifest.get("freeze_eligible") is not False or manifest.get(
        "freeze_status"
    ) != "pending_all_80_verified_success_receipts":
        return False, "campaign manifest must remain freeze-ineligible and pending"
    if require_disk:
        try:
            observed = strict_json_load(Path(root) / MANIFEST_NAME)
        except (OSError, ValueError, TypeError) as exc:
            return False, f"on-disk LR-selection manifest cannot be read: {exc}"
        if observed != dict(manifest):
            return False, "on-disk LR-selection manifest differs from active manifest"
    return True, "verified scientific identity and on-disk manifest"


def _verify_campaign_inputs(
    root: Path, manifest: Mapping[str, Any], repo_root: Path
) -> None:
    valid, reason = _verify_manifest_identity(root, manifest)
    if not valid:
        raise RuntimeError(reason)
    protocol_path = Path(str(manifest["protocol_file"])).resolve()
    if sha256_file(protocol_path) != manifest["protocol_file_sha256"]:
        raise RuntimeError("LR-selection protocol file changed")
    protocol = load_protocol(protocol_path)
    if protocol_fingerprint(protocol) != manifest["protocol_canonical_fingerprint"]:
        raise RuntimeError("LR-selection canonical protocol fingerprint changed")
    if [run.payload() for run in build_selection_plan(protocol)] != manifest["run_matrix"]:
        raise RuntimeError("LR-selection run matrix changed")
    package = Path(__file__).resolve().parent
    if _source_hashes(repo_root, package) != manifest["source_hashes"]:
        raise RuntimeError("LR-selection source code changed")
    if _git_state(repo_root) != manifest["code"]:
        raise RuntimeError("LR-selection git state changed")
    source_note = repo_root / str(protocol["source_protocol"]["path"])
    if not source_note.is_file() or sha256_file(source_note) != manifest["source_protocol_sha256"]:
        raise RuntimeError("normative source protocol note is missing or changed")
    marker = strict_json_load(root / ROOT_MARKER)
    if marker != _root_marker_payload(
        protocol,
        str(manifest["protocol_canonical_fingerprint"]),
        bool(manifest["smoke"]),
    ):
        raise RuntimeError("LR-selection root marker changed")
    bank = manifest["evaluation_bank"]
    bank_path = root / str(bank["path"])
    sidecar = Path(f"{bank_path}.sha256")
    if sha256_file(bank_path) != bank["sha256"]:
        raise RuntimeError("LR-selection evaluation bank changed")
    if sha256_file(sidecar) != bank["sidecar_sha256"]:
        raise RuntimeError("LR-selection evaluation-bank sidecar changed")
    task_spec = AngularTaskSpec.from_protocol(protocol)
    if bank.get("resolved_task_spec_sha256") != task_spec.fingerprint():
        raise RuntimeError("evaluation bank task-spec binding changed")
    batch = load_fixed_bank(bank_path)
    _validate_evaluation_batch(batch, protocol, require_full_protocol_shape=True)


def _phase0_artifact_names() -> set[str]:
    names = {"manifest.json", "phase0_gate.json"}
    names.update(
        f"model={model_id}/{name}"
        for model_id in EXPECTED_MODELS
        for name in PHASE0_MODEL_ARTIFACT_NAMES
    )
    return names


def _verify_phase0_output(
    output: Path, manifest: Mapping[str, Any]
) -> tuple[bool, str]:
    valid, reason = verify_completion_receipt(
        output / "completion_receipt.json",
        expected_job_id="phase0_state_audit",
        expected_metadata=_receipt_metadata(manifest, "phase0", "phase0"),
    )
    if not valid:
        return False, reason
    try:
        receipt = strict_json_load(output / "completion_receipt.json")
        gate = strict_json_load(output / "phase0_gate.json")
    except (OSError, ValueError, TypeError) as exc:
        return False, f"invalid Phase-0 gate: {exc}"
    if receipt.get("schema_version") != 2:
        return False, "Phase-0 receipt must use relocatable schema 2"
    if set(receipt.get("artifacts", {})) != _phase0_artifact_names():
        return False, "Phase-0 receipt artifact set differs from the frozen exact set"
    try:
        phase0_manifest = strict_json_load(output / "manifest.json")
    except (OSError, ValueError, TypeError) as exc:
        return False, f"invalid Phase-0 manifest: {exc}"
    if phase0_manifest.get("protocol_canonical_fingerprint") != manifest.get(
        "protocol_canonical_fingerprint"
    ):
        return False, "Phase-0 protocol fingerprint mismatch"
    if gate.get("passed") is not True:
        return False, "Phase-0 gate did not pass"
    return True, "verified Phase-0 receipt and gate"


def _recorded_live_processes(status: Mapping[str, Any]) -> list[str]:
    records: list[tuple[str, Any]] = [("phase0", status.get("phase0"))]
    jobs = status.get("jobs")
    if isinstance(jobs, Mapping):
        records.extend((str(run_id), value) for run_id, value in jobs.items())
    live: list[str] = []
    for label, raw in records:
        if not isinstance(raw, Mapping) or raw.get("state") != "running":
            continue
        expected = raw.get("process_identity")
        try:
            pid = int(raw.get("pid", expected.get("pid") if isinstance(expected, Mapping) else None))
        except (KeyError, TypeError, ValueError):
            live.append(f"{label}:unverifiable-running-record")
            continue
        observed = _process_identity(pid)
        if observed is None:
            continue
        if not isinstance(expected, Mapping):
            live.append(f"{label}:pid={pid}:unverifiable-identity")
        elif observed == dict(expected):
            live.append(f"{label}:pid={pid}")
    return live


def _fail_if_recorded_process_is_live(status: Mapping[str, Any]) -> None:
    live = _recorded_live_processes(status)
    if live:
        raise RuntimeError(
            "refusing to resume while recorded child processes are still live: "
            + ", ".join(live)
        )


def _attempt_candidates(
    root: Path,
    *,
    stage: str,
    job_id: str,
    recorded_attempt: Any = None,
) -> tuple[Path, ...]:
    candidates: set[Path] = set()
    if isinstance(recorded_attempt, str):
        candidate = Path(recorded_attempt).resolve()
        try:
            candidate.relative_to(Path(root).resolve())
        except ValueError:
            pass
        else:
            if candidate.is_dir():
                candidates.add(candidate)
    parent = Path(root) / "attempts" / stage / job_id
    if parent.is_dir():
        candidates.update(
            path.resolve()
            for path in parent.iterdir()
            if path.is_dir() and path.name.startswith("attempt-")
        )
    return tuple(sorted(candidates, key=lambda value: value.as_posix()))


def _recover_phase0_attempt(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
) -> tuple[bool, str]:
    raw = status.get("phase0")
    recorded_attempt = raw.get("attempt_dir") if isinstance(raw, Mapping) else None
    valid_attempts = []
    for candidate in _attempt_candidates(
        root,
        stage="phase0",
        job_id="phase0",
        recorded_attempt=recorded_attempt,
    ):
        valid, _ = _verify_phase0_output(candidate, manifest)
        if valid:
            valid_attempts.append(candidate)
    if len(valid_attempts) > 1:
        raise RuntimeError("multiple valid unpublished Phase-0 attempts require manual audit")
    if not valid_attempts:
        return False, "no valid unpublished Phase-0 attempt"
    output = Path(root) / "phase0"
    if output.exists():
        raise RuntimeError("cannot recover Phase-0 attempt over an existing output")
    os.replace(valid_attempts[0], output)
    valid, reason = _verify_phase0_output(output, manifest)
    if not valid:
        raise RuntimeError(f"recovered Phase-0 attempt failed verification: {reason}")
    return True, f"recovered verified attempt {valid_attempts[0]}"


def _run_phase0(
    *,
    root: Path,
    repo_root: Path,
    manifest: Mapping[str, Any],
    protocol_path: Path,
    python: str,
    smoke: bool,
    status: dict[str, Any],
) -> None:
    output = root / "phase0"
    valid, reason = _verify_phase0_output(output, manifest)
    if valid:
        status["phase0"] = {"state": "complete", "reason": reason}
        atomic_json(root / "status.json", status)
        return
    if output.exists():
        preserved = _preserve_invalid_output(root, "phase0", "phase0", output)
        status["phase0"] = {
            "state": "pending",
            "reason": f"invalid output preserved at {preserved}: {reason}",
        }
        atomic_json(root / "status.json", status)

    recovered, recovery_reason = _recover_phase0_attempt(
        root=root, manifest=manifest, status=status
    )
    if recovered:
        status["phase0"] = {"state": "complete", "reason": recovery_reason}
        atomic_json(root / "status.json", status)
        return

    _verify_campaign_inputs(root, manifest, repo_root)
    attempt = _unique_attempt_path(root, "phase0", "phase0")
    log_dir = root / "logs" / "phase0"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{attempt.name}.log"
    command = [
        python,
        "-m",
        "repro.sagodi_protocol.phase0",
        "--protocol",
        str(protocol_path),
        "--output-root",
        str(attempt),
        "--device",
        "cpu",
    ]
    if smoke:
        command.append("--smoke")
    with log.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=_child_environment(manifest, stage="phase0", run_id="phase0"),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=_parent_death_sigkill_preexec(os.getpid()),
        )
        status["phase0"] = {
            "state": "running",
            "pid": process.pid,
            "process_identity": _process_identity(process.pid),
            "attempt_dir": str(attempt),
            "log": str(log),
        }
        atomic_json(root / "status.json", status)
        try:
            exit_code = process.wait()
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    _verify_campaign_inputs(root, manifest, repo_root)
    valid, reason = _verify_phase0_output(attempt, manifest)
    if exit_code != 0 or not valid:
        status["phase0"] = {
            "state": "failed",
            "exit_code": exit_code,
            "reason": reason,
            "attempt_dir": str(attempt),
            "log": str(log),
        }
        atomic_json(root / "status.json", status)
        raise RuntimeError(f"Phase-0 failed: exit={exit_code}; {reason}")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(attempt, output)
    published, reason = _verify_phase0_output(output, manifest)
    if not published:
        raise RuntimeError(f"published Phase-0 output failed verification: {reason}")
    status["phase0"] = {
        "state": "complete",
        "exit_code": exit_code,
        "reason": reason,
        "log": str(log),
    }
    atomic_json(root / "status.json", status)


def _training_output_dir(root: Path, run: SelectionRun) -> Path:
    return root / "runs" / run.run_id


def _failure_output_dir(root: Path, run: SelectionRun) -> Path:
    return root / FAILURE_DIR_NAME / run.run_id


def _verify_training_output(
    output: Path,
    run: SelectionRun,
    manifest: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> tuple[bool, str, dict[str, str] | None]:
    receipt_path = output / "completion_receipt.json"
    valid, reason = verify_completion_receipt(
        receipt_path,
        expected_job_id=run.receipt_job_id,
        expected_metadata=_receipt_metadata(manifest, "training", run.run_id),
    )
    if not valid:
        return False, f"invalid training receipt: {reason}", None
    required = (
        "config.json",
        "checkpoint.pt",
        "training_trace.npz",
        "task_metrics.json",
        "rp_trace.json",
        "manifest.json",
    )
    if any(not (output / name).is_file() for name in required):
        return False, "required training artifacts are missing", None
    try:
        child_manifest = strict_json_load(output / "manifest.json")
        child_receipt = strict_json_load(receipt_path)
        child_config = strict_json_load(output / "config.json")
        rp_trace = strict_json_load(output / "rp_trace.json")
    except (OSError, ValueError, TypeError) as exc:
        return False, f"training provenance cannot be read: {exc}", None
    if child_receipt.get("schema_version") != 2:
        return False, "training receipt must use relocatable schema 2", None
    if set(child_receipt.get("artifacts", {})) != set(required):
        return False, "training receipt artifact set differs from the frozen exact set", None
    checkpoint_hash = sha256_file(output / "checkpoint.pt")
    if child_manifest.get("checkpoint_sha256") != checkpoint_hash:
        return False, "training manifest checkpoint hash mismatch", None
    if child_manifest.get("campaign_identity") != manifest["scientific_identity"]:
        return False, "training manifest campaign identity mismatch", None
    if child_manifest.get("code_commit") != manifest["code"]["code_commit"]:
        return False, "training manifest code commit mismatch", None
    if child_manifest.get("protocol_canonical_fingerprint") != manifest[
        "protocol_canonical_fingerprint"
    ]:
        return False, "training manifest protocol fingerprint mismatch", None
    if child_manifest.get("model_id") != run.model_id:
        return False, "training manifest model id mismatch", None
    if int(child_manifest.get("model_seed", -1)) != run.model_seed:
        return False, "training manifest model seed mismatch", None
    if int(child_manifest.get("parameter_count", -1)) != run.parameter_count:
        return False, "training manifest parameter count mismatch", None
    architecture = child_manifest.get("architecture_metadata")
    if not isinstance(architecture, Mapping):
        return False, "training architecture metadata is malformed", None
    model_config = architecture.get("model_config")
    if not isinstance(model_config, Mapping) or int(model_config.get("width", -1)) != run.hidden_width:
        return False, "training manifest hidden width mismatch", None
    if child_manifest.get("resolved_task_spec_sha256") != manifest[
        "resolved_task_spec_sha256"
    ]:
        return False, "training manifest task-spec binding mismatch", None
    if child_manifest.get("evaluation_bank_sha256") != manifest["evaluation_bank"][
        "sha256"
    ]:
        return False, "training manifest evaluation-bank binding mismatch", None
    train_spec = child_config.get("train_spec")
    training_config = child_config.get("training")
    if not isinstance(train_spec, Mapping) or not isinstance(training_config, Mapping):
        return False, "training config is malformed", None
    configured_model = child_config.get("model")
    if not isinstance(configured_model, Mapping):
        return False, "training config model metadata is malformed", None
    configured_model_spec = configured_model.get("model_config")
    if (
        not isinstance(configured_model_spec, Mapping)
        or int(configured_model_spec.get("width", -1)) != run.hidden_width
        or int(configured_model.get("parameters_total", -1)) != run.parameter_count
    ):
        return False, "training config width/parameter count mismatch", None
    if train_spec.get("model_name") != run.model_id:
        return False, "training config model id mismatch", None
    if int(train_spec.get("model_seed", -1)) != run.model_seed:
        return False, "training config model seed mismatch", None
    try:
        configured_lr = float(train_spec.get("learning_rate"))
    except (TypeError, ValueError):
        return False, "training config learning rate is malformed", None
    if configured_lr != run.learning_rate:
        return False, "training config learning rate mismatch", None
    expected_steps = 2 if manifest["smoke"] else run.required_update
    if int(training_config.get("steps", -1)) != expected_steps:
        return False, "training config update count mismatch", None
    expected_batch_size = min(run.batch_size, 4) if manifest["smoke"] else run.batch_size
    if int(training_config.get("batch_size", -1)) != expected_batch_size:
        return False, "training config batch size mismatch", None
    if training_config.get("rp_enabled_by_protocol") is not False:
        return False, "selector training config did not disable RP", None
    if training_config.get("expected_rp_steps") != []:
        return False, "selector training config contains forbidden RP steps", None
    rp_schedule = child_manifest.get("rp_schedule")
    if not isinstance(rp_schedule, Mapping) or rp_schedule.get("expected_steps") != []:
        return False, "selector training manifest contains forbidden RP steps", None
    if rp_schedule.get("actual_steps") != [] or int(rp_schedule.get("calls", -1)) != 0:
        return False, "selector training manifest records forbidden RP calls", None
    if int(child_receipt.get("metadata", {}).get("rp_calls", -1)) != 0:
        return False, "selector training receipt records forbidden RP calls", None
    if not isinstance(rp_trace, list) or rp_trace != []:
        return False, (
            "selector rp_trace.json must be the schema-defined zero-call value: "
            "an exact empty JSON array"
        ), None

    state_spec_in_config = child_config.get("phase0_state_spec_sha256")
    state_spec_in_manifest = child_manifest.get("state_spec_sha256")
    if manifest["smoke"]:
        if state_spec_in_config is not None or state_spec_in_manifest is not None:
            return False, "smoke training unexpectedly binds a Phase-0 state spec", None
    else:
        if root is None:
            return False, "campaign root is required to verify the Phase-0 state spec", None
        state_spec_path = Path(root) / "phase0" / f"model={run.model_id}" / "state_spec.json"
        if not state_spec_path.is_file():
            return False, "Phase-0 state spec is missing", None
        state_spec_digest = sha256_file(state_spec_path)
        if state_spec_in_config != state_spec_digest or state_spec_in_manifest != state_spec_digest:
            return False, "training output/Phase-0 state-spec binding mismatch", None

    physical_gpu_id = child_manifest.get("physical_gpu_id")
    if child_config.get("physical_gpu_id") != physical_gpu_id:
        return False, "training GPU provenance differs between config and manifest", None
    if child_receipt.get("metadata", {}).get("physical_gpu_id") != physical_gpu_id:
        return False, "training receipt physical GPU provenance mismatch", None
    try:
        physical_gpu = int(physical_gpu_id)
    except (TypeError, ValueError):
        return False, "training physical GPU id is missing or malformed", None
    if physical_gpu not in tuple(int(value) for value in manifest["gpus"]):
        return False, "training physical GPU id is outside the campaign GPU set", None

    try:
        with np.load(output / "training_trace.npz", allow_pickle=False) as archive:
            observed_steps = np.asarray(archive["step"], dtype=np.int64)
            observed_losses = np.asarray(archive["masked_mse"], dtype=np.float64)
        task_metrics = strict_json_load(output / "task_metrics.json")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"training trace cannot be audited: {exc}", None
    expected_step_array = np.arange(1, expected_steps + 1, dtype=np.int64)
    if observed_steps.ndim != 1 or not np.array_equal(observed_steps, expected_step_array):
        return False, "training trace steps are not the exact frozen sequence", None
    if observed_losses.shape != observed_steps.shape or not np.isfinite(observed_losses).all():
        return False, "training trace losses are malformed or non-finite", None
    if expected_steps == run.required_update:
        try:
            recorded_last = float(task_metrics["train_loss_last"])
        except (KeyError, TypeError, ValueError):
            return False, "task metrics lack the update-100 online training loss", None
        if float(np.float32(recorded_last)) != float(observed_losses[-1]):
            return False, "update-100 loss differs between trace and task metrics", None
    artifacts = child_receipt.get("artifacts", {})
    expected_artifact_hashes = {
        name: sha256_file(output / name)
        for name in required
    }
    for name, digest in expected_artifact_hashes.items():
        if artifacts.get(name) != digest:
            return False, f"training receipt does not bind {name}", None
    bindings = {
        "checkpoint_sha256": checkpoint_hash,
        "completion_receipt_sha256": sha256_file(receipt_path),
        "training_manifest_sha256": sha256_file(output / "manifest.json"),
        "training_trace_sha256": sha256_file(output / "training_trace.npz"),
        "task_metrics_sha256": sha256_file(output / "task_metrics.json"),
        "rp_trace_sha256": sha256_file(output / "rp_trace.json"),
        "config_sha256": sha256_file(output / "config.json"),
        "physical_gpu_id": str(physical_gpu),
    }
    return True, "verified training receipt/checkpoint binding", bindings


def _training_command(
    *,
    run: SelectionRun,
    root: Path,
    protocol_path: Path,
    python: str,
    evaluation_bank: Path,
    manifest: Mapping[str, Any],
    smoke: bool,
) -> tuple[str, ...]:
    command = [
        python,
        "-m",
        "repro.sagodi_protocol.train",
        "--protocol",
        str(protocol_path),
        "--model",
        run.model_id,
        "--model-seed",
        str(run.model_seed),
        "--learning-rate",
        str(run.learning_rate),
        "--output-dir",
        OUTPUT_DIR_TOKEN,
        "--evaluation-bank",
        str(evaluation_bank),
        "--campaign-identity",
        str(manifest["scientific_identity"]),
        "--device",
        "cuda:0",
    ]
    if smoke:
        command.append("--smoke")
    else:
        command.extend(
            [
                "--state-spec",
                str(root / "phase0" / f"model={run.model_id}" / "state_spec.json"),
            ]
        )
    return tuple(command)


def _recover_training_attempt(
    *,
    root: Path,
    run: SelectionRun,
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
) -> tuple[bool, str]:
    jobs = status.get("jobs")
    raw = jobs.get(run.run_id) if isinstance(jobs, Mapping) else None
    recorded_attempt = raw.get("attempt_dir") if isinstance(raw, Mapping) else None
    valid_attempts: list[Path] = []
    for candidate in _attempt_candidates(
        root,
        stage="lr_selection_training",
        job_id=run.run_id,
        recorded_attempt=recorded_attempt,
    ):
        valid, _, _ = _verify_training_output(
            candidate, run, manifest, root=root
        )
        if valid:
            valid_attempts.append(candidate)
    if len(valid_attempts) > 1:
        raise RuntimeError(
            f"multiple valid unpublished attempts exist for {run.run_id}; manual audit required"
        )
    if not valid_attempts:
        return False, "no valid unpublished training attempt"
    output = _training_output_dir(root, run)
    failure = _failure_output_dir(root, run)
    if output.exists() or failure.exists():
        raise RuntimeError(
            f"cannot recover {run.run_id} over an existing terminal outcome"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(valid_attempts[0], output)
    valid, reason, _ = _verify_training_output(
        output, run, manifest, root=root
    )
    if not valid:
        raise RuntimeError(f"recovered training attempt failed verification: {reason}")
    return True, f"recovered verified attempt {valid_attempts[0]}"


def _terminate_processes(active: Mapping[int, dict[str, Any]]) -> None:
    for item in active.values():
        process = item["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 10.0
    while time.time() < deadline and any(
        item["process"].poll() is None for item in active.values()
    ):
        time.sleep(0.1)
    for item in active.values():
        process = item["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
        item["handle"].close()


def _run_training_jobs(
    *,
    root: Path,
    repo_root: Path,
    manifest: Mapping[str, Any],
    plan: Sequence[SelectionRun],
    protocol_path: Path,
    python: str,
    evaluation_bank: Path,
    gpus: Sequence[int],
    smoke: bool,
    status: dict[str, Any],
) -> None:
    pending: list[SelectionRun] = []
    for run in plan:
        output = _training_output_dir(root, run)
        failure = _failure_output_dir(root, run)
        if failure.exists():
            raise RuntimeError(
                f"legacy terminal-failure output exists for {run.run_id}; "
                "the all-80-valid selector requires manual audit"
            )
        valid, reason, _ = _verify_training_output(output, run, manifest, root=root)
        if valid:
            status["jobs"][run.run_id] = {"state": "complete", "reason": reason}
            continue
        if output.exists():
            preserved = _preserve_invalid_output(root, "lr_selection_training", run.run_id, output)
            reason = f"invalid output preserved at {preserved}: {reason}"
        recovered, recovery_reason = _recover_training_attempt(
            root=root,
            run=run,
            manifest=manifest,
            status=status,
        )
        if recovered:
            status["jobs"][run.run_id] = {
                "state": "complete",
                "reason": recovery_reason,
            }
            continue
        status["jobs"][run.run_id] = {"state": "pending", "reason": reason}
        pending.append(run)
    atomic_json(root / "status.json", status)

    available = sorted(int(gpu) for gpu in gpus)
    active: dict[int, dict[str, Any]] = {}
    try:
        while pending or active:
            while pending and available:
                _verify_campaign_inputs(root, manifest, repo_root)
                gpu = available.pop(0)
                run = pending.pop(0)
                attempt = _unique_attempt_path(
                    root, "lr_selection_training", run.run_id
                )
                log_dir = root / "logs" / "lr_selection_training" / run.run_id
                log_dir.mkdir(parents=True, exist_ok=True)
                log = log_dir / f"{attempt.name}.log"
                handle = log.open("ab", buffering=0)
                command = [
                    str(attempt) if token == OUTPUT_DIR_TOKEN else token
                    for token in _training_command(
                        run=run,
                        root=root,
                        protocol_path=protocol_path,
                        python=python,
                        evaluation_bank=evaluation_bank,
                        manifest=manifest,
                        smoke=smoke,
                    )
                ]
                process = subprocess.Popen(
                    command,
                    cwd=repo_root,
                    env=_child_environment(
                        manifest, stage="training", run_id=run.run_id, gpu=gpu
                    ),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    preexec_fn=_parent_death_sigkill_preexec(os.getpid()),
                )
                active[gpu] = {
                    "run": run,
                    "attempt": attempt,
                    "log": log,
                    "handle": handle,
                    "process": process,
                    "process_identity": _process_identity(process.pid),
                }
                status["jobs"][run.run_id] = {
                    "state": "running",
                    "gpu": gpu,
                    "pid": process.pid,
                    "process_identity": active[gpu]["process_identity"],
                    "attempt_dir": str(attempt),
                    "log": str(log),
                    "started_at": time.time(),
                }
                atomic_json(root / "status.json", status)
                print(f"[launch lr-selection gpu{gpu}] {run.run_id}", flush=True)
            finished = [
                gpu
                for gpu, item in active.items()
                if item["process"].poll() is not None
            ]
            if not finished:
                time.sleep(0.5)
                continue
            for gpu in finished:
                item = active.pop(gpu)
                run = item["run"]
                process = item["process"]
                exit_code = process.poll()
                item["handle"].close()
                _verify_campaign_inputs(root, manifest, repo_root)
                valid, reason, _ = _verify_training_output(
                    item["attempt"], run, manifest, root=root
                )
                if exit_code != 0 or not valid:
                    status["jobs"][run.run_id] = {
                        "state": "failed_retryable_infrastructure_or_unknown",
                        "gpu": gpu,
                        "pid": process.pid,
                        "exit_code": exit_code,
                        "reason": (
                            "nonzero exit or invalid receipt is not scientific "
                            f"failure evidence: {reason}"
                        ),
                        "attempt_dir": str(item["attempt"]),
                        "log": str(item["log"]),
                    }
                    atomic_json(root / "status.json", status)
                    raise RuntimeError(
                        f"LR-selection child {run.run_id} failed without "
                        "receipt-bound scientific numerical-failure evidence; "
                        f"campaign aborted for audited resume (exit={exit_code}; {reason})"
                    )
                output = _training_output_dir(root, run)
                output.parent.mkdir(parents=True, exist_ok=True)
                os.replace(item["attempt"], output)
                published, reason, _ = _verify_training_output(
                    output, run, manifest, root=root
                )
                if not published:
                    raise RuntimeError(
                        f"published LR-selection run failed verification: {reason}"
                    )
                status["jobs"][run.run_id] = {
                    "state": "complete",
                    "gpu": gpu,
                    "pid": process.pid,
                    "exit_code": exit_code,
                    "reason": reason,
                    "log": str(item["log"]),
                }
                atomic_json(root / "status.json", status)
                available.append(gpu)
                available.sort()
                print(f"[complete lr-selection gpu{gpu}] {run.run_id}", flush=True)
    except BaseException:
        _terminate_processes(active)
        raise


def _read_training_row(
    root: Path, run: SelectionRun, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    output = _training_output_dir(root, run)
    valid, reason, bindings = _verify_training_output(
        output, run, manifest, root=root
    )
    if not valid or bindings is None:
        raise IncompleteSelectionError(f"{run.run_id}: {reason}")
    with np.load(output / "training_trace.npz", allow_pickle=False) as archive:
        steps = np.asarray(archive["step"], dtype=np.int64)
        losses = np.asarray(archive["masked_mse"], dtype=np.float64)
    if steps.ndim != 1 or losses.ndim != 1 or steps.shape != losses.shape:
        raise ValueError(f"{run.run_id}: malformed training trace")
    matches = np.flatnonzero(steps == run.required_update)
    if matches.size == 1:
        loss = float(losses[int(matches[0])])
        completed = int(steps.max()) if steps.size else 0
        row_status = "complete" if math.isfinite(loss) else "nonfinite"
    else:
        loss = float("nan")
        completed = int(steps.max()) if steps.size else 0
        row_status = "smoke_incomplete" if manifest["smoke"] else "missing_update"
    metrics = strict_json_load(output / "task_metrics.json")
    return {
        "run_id": run.run_id,
        "model_id": run.model_id,
        "hidden_width": run.hidden_width,
        "model_seed": run.model_seed,
        "learning_rate": run.learning_rate,
        "required_update": run.required_update,
        "completed_updates": completed,
        "loss_at_required_update": loss,
        "status": row_status,
        "selection_metric_source": "training_trace.npz:masked_mse",
        "task_metrics_recorded_not_used_for_selection": metrics,
        "artifact_bindings": bindings,
    }


def _read_outcome_row(
    root: Path, run: SelectionRun, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    success = _training_output_dir(root, run)
    failure = _failure_output_dir(root, run)
    if failure.exists():
        raise IncompleteSelectionError(
            f"{run.run_id}: terminal scientific-failure records are not authorized"
        )
    if success.exists():
        return _read_training_row(root, run, manifest)
    raise IncompleteSelectionError(f"{run.run_id}: no verified successful outcome")


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = (
        "run_id",
        "model_id",
        "hidden_width",
        "model_seed",
        "learning_rate",
        "required_update",
        "completed_updates",
        "loss_at_required_update",
        "status",
        "selection_metric_source",
        "task_metrics_recorded_not_used_for_selection_json",
        "checkpoint_sha256",
        "completion_receipt_sha256",
        "training_manifest_sha256",
        "training_trace_sha256",
        "task_metrics_sha256",
        "rp_trace_sha256",
        "config_sha256",
        "physical_gpu_id",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        bindings = row["artifact_bindings"]
        writer.writerow(
            {
                "run_id": row["run_id"],
                "model_id": row["model_id"],
                "hidden_width": row["hidden_width"],
                "model_seed": row["model_seed"],
                "learning_rate": format(float(row["learning_rate"]), ".17g"),
                "required_update": row["required_update"],
                "completed_updates": row["completed_updates"],
                "loss_at_required_update": (
                    format(float(row["loss_at_required_update"]), ".17g")
                    if math.isfinite(float(row["loss_at_required_update"]))
                    else ""
                ),
                "status": row["status"],
                "selection_metric_source": row["selection_metric_source"],
                "task_metrics_recorded_not_used_for_selection_json": json.dumps(
                    row["task_metrics_recorded_not_used_for_selection"],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                **bindings,
            }
        )
    return stream.getvalue().encode("utf-8")


def _selection_runs_from_manifest(
    manifest: Mapping[str, Any]
) -> tuple[SelectionRun, ...]:
    raw = manifest.get("run_matrix")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("manifest run_matrix must be an array")
    runs: list[SelectionRun] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("manifest run_matrix entries must be objects")
        runs.append(
            SelectionRun(
                run_id=str(item["run_id"]),
                model_id=str(item["model_id"]),
                hidden_width=int(item["hidden_width"]),
                parameter_count=int(item["parameter_count"]),
                batch_size=int(item["batch_size"]),
                model_seed=int(item["model_seed"]),
                learning_rate=float(item["learning_rate"]),
                required_update=int(item["required_update"]),
            )
        )
    if len({run.run_id for run in runs}) != len(runs):
        raise ValueError("manifest run_matrix contains duplicate run ids")
    expected_keys = {
        (model_id, seed, learning_rate)
        for model_id in EXPECTED_MODELS
        for seed in EXPECTED_SELECTION_SEEDS
        for learning_rate in EXPECTED_LEARNING_RATES
    }
    observed_keys = {
        (run.model_id, run.model_seed, run.learning_rate) for run in runs
    }
    if len(runs) != 80 or observed_keys != expected_keys:
        raise ValueError("manifest run_matrix is not the exact frozen 80-run cross-product")
    for run in runs:
        if (
            run.hidden_width != EXPECTED_MODEL_WIDTHS[run.model_id]
            or run.parameter_count != EXPECTED_PARAMETER_COUNTS[run.model_id]
            or run.batch_size != EXPECTED_BATCH_SIZE
            or run.required_update != REQUIRED_UPDATE
        ):
            raise ValueError(f"manifest run dimensions changed for {run.run_id}")
    return tuple(runs)


def _selection_artifact_paths(
    root: Path,
    manifest: Mapping[str, Any],
    plan: Sequence[SelectionRun],
) -> list[Path]:
    root = Path(root)
    bank_path = root / str(manifest["evaluation_bank"]["path"])
    paths = [
        root / MANIFEST_NAME,
        root / SUMMARY_NAME,
        root / MATRIX_NAME,
        bank_path,
        Path(f"{bank_path}.sha256"),
        root / "phase0" / "completion_receipt.json",
    ]
    paths.extend(root / "phase0" / name for name in sorted(_phase0_artifact_names()))
    for run in plan:
        success = _training_output_dir(root, run)
        failure = _failure_output_dir(root, run)
        if failure.exists() or not success.exists():
            raise IncompleteSelectionError(
                f"{run.run_id}: all-80-valid policy requires a successful outcome"
            )
        paths.extend(
            success / name
            for name in (
                "checkpoint.pt",
                "completion_receipt.json",
                "config.json",
                "manifest.json",
                "training_trace.npz",
                "task_metrics.json",
                "rp_trace.json",
            )
        )
    return paths


def _relative_artifact_keys(root: Path, paths: Iterable[Path]) -> set[str]:
    resolved_root = Path(root).resolve()
    result: set[str] = set()
    for raw in paths:
        path = Path(raw).resolve()
        try:
            relative = path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"selection artifact escapes campaign root: {path}") from exc
        result.add(relative.as_posix())
    return result


def _final_receipt_metadata(
    manifest: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **_receipt_metadata(manifest, "lr_selection", "lr_selection"),
        "scope": SELECTION_SCOPE,
        "pilot_only": True,
        "confirmatory": False,
        "ca_evidence": False,
        "manifold_analysis_performed": False,
        "smoke": bool(manifest["smoke"]),
        "freeze_eligible": bool(not manifest["smoke"] and summary["selection_performed"]),
        "selection_rule": manifest["selection_rule"],
        "winners": summary["winners"],
    }


def _complete_marker_payload(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    selection_receipt_sha256: str,
) -> dict[str, Any]:
    if int(summary.get("run_count", -1)) != 80 or int(
        summary.get("successful_receipt_count", -1)
    ) != 80:
        raise ValueError("COMPLETE requires exactly 80 verified successful receipts")
    return {
        "schema_version": 1,
        "status": "complete",
        "campaign_scientific_identity": manifest["scientific_identity"],
        "scope": SELECTION_SCOPE,
        "smoke": bool(manifest["smoke"]),
        "freeze_eligible": bool(
            not manifest["smoke"] and summary["selection_performed"]
        ),
        "selection_performed": bool(summary["selection_performed"]),
        "run_count": 80,
        "verified_success_receipt_count": 80,
        "selection_receipt_sha256": str(selection_receipt_sha256),
        "winners": summary["winners"],
    }


def _verify_complete_marker(
    root: Path, manifest: Mapping[str, Any], summary: Mapping[str, Any]
) -> tuple[bool, str]:
    root = Path(root)
    selection_receipt = root / SELECTION_RECEIPT_NAME
    marker = root / COMPLETE_MARKER_NAME
    if not selection_receipt.is_file():
        return False, "selection receipt is missing before COMPLETE verification"
    try:
        observed = strict_json_load(marker)
        expected = _complete_marker_payload(
            manifest,
            summary,
            selection_receipt_sha256=sha256_file(selection_receipt),
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, f"COMPLETE marker cannot be verified: {exc}"
    if observed != expected:
        return False, "COMPLETE marker content differs from the verified final state"
    return True, "verified COMPLETE marker"


def write_selection_artifacts(
    root: Path,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    matrix_bytes: bytes,
    *,
    child_artifacts: Iterable[Path] = (),
) -> None:
    """Atomically write both nested final receipts for a completed campaign."""

    root = Path(root)
    summary_path = root / SUMMARY_NAME
    matrix_path = root / MATRIX_NAME
    atomic_json(summary_path, dict(summary))
    atomic_bytes(matrix_path, matrix_bytes)
    selection_artifacts = [
        root / MANIFEST_NAME,
        summary_path,
        matrix_path,
        *[Path(path) for path in child_artifacts],
    ]
    metadata = _final_receipt_metadata(manifest, summary)
    selection_receipt = root / SELECTION_RECEIPT_NAME
    write_completion_receipt(
        selection_receipt,
        job_id="lr_selection_aggregate",
        artifacts=selection_artifacts,
        metadata=metadata,
    )
    complete_marker = root / COMPLETE_MARKER_NAME
    atomic_json(
        complete_marker,
        _complete_marker_payload(
            manifest,
            summary,
            selection_receipt_sha256=sha256_file(selection_receipt),
        ),
    )
    write_completion_receipt(
        root / COMPLETION_RECEIPT_NAME,
        job_id="lr_selection_campaign_complete",
        artifacts=[
            root / MANIFEST_NAME,
            summary_path,
            matrix_path,
            selection_receipt,
            complete_marker,
        ],
        metadata=metadata,
    )


def verify_selection_completion(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[bool, str]:
    root = Path(root)
    valid, reason = _verify_manifest_identity(root, manifest)
    if not valid:
        return False, reason
    try:
        summary = strict_json_load(root / SUMMARY_NAME)
        expected_metadata = _final_receipt_metadata(manifest, summary)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, f"cannot read final LR-selection summary: {exc}"
    valid, reason = verify_completion_receipt(
        root / SELECTION_RECEIPT_NAME,
        expected_job_id="lr_selection_aggregate",
        expected_metadata=expected_metadata,
    )
    if not valid:
        return False, f"invalid LR-selection receipt: {reason}"
    valid, reason = verify_completion_receipt(
        root / COMPLETION_RECEIPT_NAME,
        expected_job_id="lr_selection_campaign_complete",
        expected_metadata=expected_metadata,
    )
    if not valid:
        return False, f"invalid campaign completion receipt: {reason}"
    valid, reason = _verify_complete_marker(root, manifest, summary)
    if not valid:
        return False, reason
    try:
        plan = _selection_runs_from_manifest(manifest)
        phase0_valid, phase0_reason = _verify_phase0_output(root / "phase0", manifest)
        if not phase0_valid:
            return False, f"invalid nested Phase-0 outcome: {phase0_reason}"
        rows = [_read_outcome_row(root, run, manifest) for run in plan]
        expected_artifacts = _relative_artifact_keys(
            root, _selection_artifact_paths(root, manifest, plan)
        )
        selection_receipt = strict_json_load(root / SELECTION_RECEIPT_NAME)
        completion_receipt = strict_json_load(root / COMPLETION_RECEIPT_NAME)
    except (OSError, ValueError, TypeError, KeyError, IncompleteSelectionError) as exc:
        return False, f"recursive LR-selection verification failed: {exc}"
    if selection_receipt.get("schema_version") != 2:
        return False, "LR-selection receipt must use schema 2"
    if set(selection_receipt.get("artifacts", {})) != expected_artifacts:
        return False, "LR-selection receipt artifact set differs from exact terminal outcomes"
    expected_completion_artifacts = {
        MANIFEST_NAME,
        SUMMARY_NAME,
        MATRIX_NAME,
        SELECTION_RECEIPT_NAME,
        COMPLETE_MARKER_NAME,
    }
    if completion_receipt.get("schema_version") != 2 or set(
        completion_receipt.get("artifacts", {})
    ) != expected_completion_artifacts:
        return False, "campaign completion receipt artifact set differs"
    if summary.get("campaign_scientific_identity") != manifest["scientific_identity"]:
        return False, "summary campaign identity mismatch"
    if summary.get("selection_rule") != manifest["selection_rule"]:
        return False, "summary selection rule mismatch"
    if bool(summary.get("selection_performed")) == bool(manifest["smoke"]):
        return False, "summary smoke/selection status is inconsistent"
    if summary.get("smoke") is not bool(manifest["smoke"]):
        return False, "summary smoke label mismatch"
    if summary.get("freeze_eligible") is not bool(not manifest["smoke"]):
        return False, "summary freeze-eligibility label mismatch"
    if summary.get("scope") != SELECTION_SCOPE or int(summary.get("run_count", -1)) != len(plan):
        return False, "summary scope or terminal-outcome count mismatch"
    try:
        recomputed = (
            aggregate_lr_selection(rows)
            if manifest["smoke"]
            else strict_aggregate_lr_selection(rows)
        )
    except (ValueError, IncompleteSelectionError) as exc:
        return False, f"strict final aggregation failed: {exc}"
    expected_winners = {} if manifest["smoke"] else recomputed["winners"]
    if summary.get("aggregation") != recomputed or summary.get("winners") != expected_winners:
        return False, "summary aggregation/winners do not match terminal outcomes"
    successful_bindings = {
        str(row["run_id"]): dict(row["artifact_bindings"])
        for row in rows
    }
    if summary.get("training_artifact_bindings") != successful_bindings:
        return False, "summary successful-training bindings mismatch"
    expected_terminal = {
        str(row["run_id"]): str(row["status"]) for row in rows
    }
    if summary.get("terminal_outcomes") != expected_terminal:
        return False, "summary terminal-outcome map mismatch"
    return True, "verified nested LR-selection completion receipts"


def _preserve_finalization(root: Path) -> None:
    names = (
        SUMMARY_NAME,
        MATRIX_NAME,
        SELECTION_RECEIPT_NAME,
        COMPLETION_RECEIPT_NAME,
        COMPLETE_MARKER_NAME,
    )
    present = [root / name for name in names if (root / name).exists()]
    if not present:
        return
    destination = _unique_attempt_path(root, "finalization", "lr_selection", label="recovered")
    for path in present:
        os.replace(path, destination / path.name)


def _finalize(
    root: Path,
    manifest: Mapping[str, Any],
    plan: Sequence[SelectionRun],
) -> dict[str, Any]:
    valid, _ = verify_selection_completion(root, manifest)
    if valid:
        return strict_json_load(root / SUMMARY_NAME)
    _preserve_finalization(root)
    rows = [_read_outcome_row(root, run, manifest) for run in plan]
    if manifest["smoke"]:
        aggregation = aggregate_lr_selection(rows)
        winners: dict[str, Any] = {}
        selection_performed = False
    else:
        aggregation = strict_aggregate_lr_selection(rows)
        winners = aggregation["winners"]
        selection_performed = True
    summary = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "campaign_scientific_identity": manifest["scientific_identity"],
        "protocol_file_sha256": manifest["protocol_file_sha256"],
        "protocol_canonical_fingerprint": manifest["protocol_canonical_fingerprint"],
        "scope": SELECTION_SCOPE,
        "pilot_only": True,
        "confirmatory": False,
        "ca_evidence": False,
        "manifold_analysis_performed": False,
        "smoke": bool(manifest["smoke"]),
        "freeze_eligible": bool(not manifest["smoke"]),
        "selection_performed": selection_performed,
        "selection_rule": manifest["selection_rule"],
        "winners": winners,
        "aggregation": aggregation,
        "run_count": len(rows),
        "training_artifact_bindings": {
            str(row["run_id"]): dict(row["artifact_bindings"])
            for row in rows
        },
        "terminal_outcomes": {
            str(row["run_id"]): str(row["status"]) for row in rows
        },
        "successful_receipt_count": len(rows),
        "validation_and_task_metrics_recorded_but_not_used": True,
    }
    child_artifacts = _selection_artifact_paths(root, manifest, plan)
    child_artifacts = [
        path
        for path in child_artifacts
        if path.name not in {MANIFEST_NAME, SUMMARY_NAME, MATRIX_NAME}
        or path.parent != Path(root)
    ]
    write_selection_artifacts(
        root,
        manifest,
        summary,
        _csv_bytes(rows),
        child_artifacts=child_artifacts,
    )
    valid, reason = verify_selection_completion(root, manifest)
    if not valid:
        raise RuntimeError(f"final LR-selection receipt verification failed: {reason}")
    return summary


def run_lr_selection_campaign(
    *,
    protocol_path: Path,
    artifact_root: Path,
    python: str,
    gpus: Sequence[int],
    smoke: bool = False,
) -> Path:
    protocol_path = Path(protocol_path).resolve()
    artifact_root = Path(artifact_root).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    python = _resolve_python(python)
    gpus = _validated_gpu_ids(gpus)
    protocol = load_protocol(protocol_path)
    fingerprint = protocol_fingerprint(protocol)
    plan = build_selection_plan(protocol)
    git_state = _git_state(repo_root)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError(
            "preexisting CUDA_VISIBLE_DEVICES is forbidden because --gpus uses physical ids"
        )
    if not smoke and git_state["worktree_dirty"]:
        raise RuntimeError(
            "full LR selection requires a clean committed git worktree"
        )
    _prepare_selection_root(artifact_root, protocol, fingerprint, smoke)
    lock = _acquire_campaign_lock(artifact_root)
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"LR-selection campaign interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        environment = _environment_fingerprint(python, gpus)
        evaluation_bank = _materialize_evaluation_bank(artifact_root, protocol)
        source_hashes = _source_hashes(repo_root, Path(__file__).resolve().parent)
        manifest = _build_manifest(
            protocol=protocol,
            protocol_path=protocol_path,
            fingerprint=fingerprint,
            evaluation_bank=evaluation_bank,
            plan=plan,
            source_hashes=source_hashes,
            git_state=git_state,
            environment=environment,
            python=python,
            gpus=gpus,
            smoke=smoke,
        )
        _write_or_check_manifest(artifact_root, manifest)
        _verify_campaign_inputs(artifact_root, manifest, repo_root)
        completed, _ = verify_selection_completion(artifact_root, manifest)
        if completed:
            return artifact_root
        status: dict[str, Any]
        status_path = artifact_root / "status.json"
        if status_path.exists():
            existing = strict_json_load(status_path)
            status = dict(existing) if isinstance(existing, Mapping) else {}
        else:
            status = {}
        _fail_if_recorded_process_is_live(status)
        status.update(
            {
                "schema_version": 1,
                "campaign_id": manifest["campaign_id"],
                "scientific_identity": manifest["scientific_identity"],
                "scope": SELECTION_SCOPE,
                "smoke": bool(smoke),
                "stage": "phase0",
                "jobs": dict(status.get("jobs", {})),
            }
        )
        atomic_json(status_path, status)
        _run_phase0(
            root=artifact_root,
            repo_root=repo_root,
            manifest=manifest,
            protocol_path=protocol_path,
            python=python,
            smoke=smoke,
            status=status,
        )
        status["stage"] = "training_only_lr_selection"
        atomic_json(status_path, status)
        evaluation_path = artifact_root / str(evaluation_bank["path"])
        _run_training_jobs(
            root=artifact_root,
            repo_root=repo_root,
            manifest=manifest,
            plan=plan,
            protocol_path=protocol_path,
            python=python,
            evaluation_bank=evaluation_path,
            gpus=gpus,
            smoke=smoke,
            status=status,
        )
        _verify_campaign_inputs(artifact_root, manifest, repo_root)
        status["stage"] = "finalizing_selection"
        atomic_json(status_path, status)
        try:
            summary = _finalize(artifact_root, manifest, plan)
        except IncompleteSelectionError as exc:
            status["stage"] = "incomplete_no_eligible_learning_rate"
            status["selection_performed"] = False
            status["freeze_eligible"] = False
            status["reason"] = str(exc)
            status["completed_at"] = time.time()
            atomic_json(status_path, status)
            raise RuntimeError(
                f"LR-selection campaign has no eligible LR for at least one model: {exc}"
            ) from exc
        status["stage"] = "complete"
        status["freeze_eligible"] = bool(not smoke)
        status["selection_performed"] = bool(summary["selection_performed"])
        status["winners"] = summary["winners"]
        status["completed_at"] = time.time()
        atomic_json(status_path, status)
        return artifact_root
    finally:
        lock.release()
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    gpus = tuple(int(item) for item in args.gpus.split(",") if item.strip())
    output = run_lr_selection_campaign(
        protocol_path=args.protocol,
        artifact_root=args.artifact_root,
        python=args.python,
        gpus=gpus,
        smoke=args.smoke,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "artifact_root": str(output),
                "scope": SELECTION_SCOPE,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
