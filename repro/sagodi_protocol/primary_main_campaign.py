"""Run the selector-bound six-model Ságodi primary training campaign.

This module deliberately stops at trained checkpoints.  It does not import or
invoke a manifold-analysis module, and its campaign/child receipts explicitly
state that the resulting training artifacts are not themselves continuous-
attractor evidence.

A full launch is possible only after the parent 120-run LR selector verifies
recursively.  The committed template and the six selected learning rates are
then resolved into an immutable executable protocol *before* the first child
is launched.  Failed children are retained as attempts; a main seed is never
silently replaced by a different seed.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from .config import load_protocol, protocol_fingerprint
from .lr_selection_v3 import (
    EXPECTED_MODEL_IDS,
    PROTOCOL_COPY as SELECTOR_PROTOCOL_COPY,
    SELECTOR_COPY as SELECTOR_SPEC_COPY,
    build_selection_plan,
    load_selector_spec,
    validate_protocol_binding,
    verify_selection_completion,
)
from .orchestrate import (
    _acquire_campaign_lock,
    _environment_fingerprint,
    _git_state,
    _materialize_evaluation_bank,
    _preserve_invalid_output,
    _resolve_python,
    _source_hashes,
    _unique_attempt_path,
    _validated_gpu_ids,
)


CAMPAIGN_MODE = "sagodi_primary_main_v3"
SCOPE = "primary_main_training_only_no_manifold_analysis"
PROTOCOL_TRACK = "sagodi_primary_v3_main"
EXPECTED_MAIN_SEEDS = tuple(range(10))
EXPECTED_UPDATES = 5000
EXPECTED_BATCH_SIZE = 64
EXPECTED_RUN_COUNT = 60
EXPECTED_RP_STEPS = tuple(range(1550, 5001, 50))

ROOT_MARKER = ".calru_primary_main_v3_root.json"
TEMPLATE_COPY = "primary_main_template.json"
RESOLVED_PROTOCOL = "resolved_primary_main_protocol.yaml"
MANIFEST = "primary_main_manifest.json"
STATUS = "status.json"
SUMMARY = "primary_main_summary.json"
COMPLETION_RECEIPT = "completion_receipt.json"
COMPLETE = "COMPLETE"


class MainTemplateError(ValueError):
    """The committed main template differs from the preregistered contract."""


class ParentSelectorError(RuntimeError):
    """The supplied parent selector is incomplete or no longer verifies."""


@dataclass(frozen=True)
class MainModel:
    model_id: str
    hidden_width: int
    parameter_count: int
    learning_rate: float


@dataclass(frozen=True)
class MainRun:
    model: MainModel
    model_seed: int

    @property
    def run_id(self) -> str:
        return f"primary_main__{self.model.model_id}__seed{self.model_seed:02d}"

    @property
    def receipt_job_id(self) -> str:
        return (
            f"{self.model.model_id}-seed{self.model_seed}-"
            f"lr{self.model.learning_rate:g}"
        )

    def payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_id": self.model.model_id,
            "hidden_width": self.model.hidden_width,
            "parameter_count": self.model.parameter_count,
            "model_seed": self.model_seed,
            "learning_rate": self.model.learning_rate,
            "optimizer_updates": EXPECTED_UPDATES,
            "batch_size": EXPECTED_BATCH_SIZE,
        }


@dataclass(frozen=True)
class ParentSelector:
    root: Path
    protocol: dict[str, Any]
    manifest: dict[str, Any]
    models: tuple[MainModel, ...]
    binding: dict[str, Any]


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise MainTemplateError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def load_main_template(path: Path) -> dict[str, Any]:
    payload = strict_json_load(path)
    if not isinstance(payload, Mapping):
        raise MainTemplateError("main template must be a JSON object")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "template_id",
            "campaign_mode",
            "scope",
            "selector_contract",
            "main_model_seeds",
            "training",
            "retention_plasticity",
            "run_matrix",
        },
        "template",
    )
    if payload.get("schema_version") != 1:
        raise MainTemplateError("template schema_version must be 1")
    if payload.get("template_id") != "sagodi_primary_main_v3_template":
        raise MainTemplateError("template_id changed")
    if payload.get("campaign_mode") != CAMPAIGN_MODE or payload.get("scope") != SCOPE:
        raise MainTemplateError("main campaign mode/scope changed")

    selector = payload.get("selector_contract")
    if not isinstance(selector, Mapping):
        raise MainTemplateError("selector_contract must be an object")
    if selector != {
        "campaign_type": "six_model_lr_selection_v3",
        "scope": "training_only_lr_selection_no_manifold_analysis_no_ca_evidence",
        "model_order": list(EXPECTED_MODEL_IDS),
        "selection_seed_count": 5,
        "optimizer_updates": 100,
        "expected_training_runs": 120,
    }:
        raise MainTemplateError("selector_contract changed")
    if tuple(payload.get("main_model_seeds", ())) != EXPECTED_MAIN_SEEDS:
        raise MainTemplateError("main_model_seeds must be exactly 0..9")

    training = payload.get("training")
    if not isinstance(training, Mapping) or training != {
        "optimizer_updates": EXPECTED_UPDATES,
        "batch_size": EXPECTED_BATCH_SIZE,
        "optimizer": {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
        },
        "state_noise_coordinate_standard_deviation": 0.1,
        "state_noise_coordinate_variance": 0.01,
        "gradient_clipping_policy": "none",
        "gradient_clipping_numeric_value": None,
    }:
        raise MainTemplateError("main training contract changed")
    rp = payload.get("retention_plasticity")
    if not isinstance(rp, Mapping) or rp != {
        "enabled_model": "ca_lru",
        "warmup_updates": 1500,
        "interval_updates": 50,
        "calls_after_warmup": 70,
        "probe_batch_size": 256,
        "probe_horizon": 256,
        "blank_ablation_horizon": 256,
        "probe_noise_enabled": False,
        "eta_lambda": 3000.0,
        "damage_epsilon": 3e-5,
    }:
        raise MainTemplateError("Retention Plasticity main contract changed")
    if tuple(range(1550, 5001, 50)) != EXPECTED_RP_STEPS:
        raise AssertionError("internal RP schedule constant changed")
    matrix = payload.get("run_matrix")
    if not isinstance(matrix, Mapping) or matrix != {
        "expected_training_runs": EXPECTED_RUN_COUNT,
        "task_seed_fixed": 0,
        "data_stream_seed_fixed": 0,
    }:
        raise MainTemplateError("main run matrix changed")
    return dict(payload)


def verify_parent_selector(selector_root: Path) -> ParentSelector:
    """Recursively verify all 120 selector children and return its six winners."""

    root = Path(selector_root).resolve(strict=True)
    try:
        spec = load_selector_spec(root / SELECTOR_SPEC_COPY)
        protocol = load_protocol(root / SELECTOR_PROTOCOL_COPY)
        validate_protocol_binding(protocol, spec)
        plan = build_selection_plan(spec)
        manifest = strict_json_load(root / "lr_selection_manifest.json")
        summary = strict_json_load(root / "lr_selection_summary.json")
        complete = strict_json_load(root / "COMPLETE")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ParentSelectorError(f"parent selector is unreadable: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        raise ParentSelectorError("parent selector manifest/summary is malformed")
    valid, reason = verify_selection_completion(root, manifest, spec, plan)
    if not valid:
        raise ParentSelectorError(f"parent selector recursive verification failed: {reason}")
    if manifest.get("campaign_id") != "sagodi_six_model_lr_selection_v3":
        raise ParentSelectorError("parent selector campaign id differs")
    if manifest.get("campaign_type") != "six_model_lr_selection_v3":
        raise ParentSelectorError("parent selector campaign type differs")
    if manifest.get("scope") != (
        "training_only_lr_selection_no_manifold_analysis_no_ca_evidence"
    ):
        raise ParentSelectorError("parent selector scope differs")
    if manifest.get("smoke") is not False or summary.get("selection_performed") is not True:
        raise ParentSelectorError("main training requires a completed full selector")
    selector_code = manifest.get("code")
    if (
        not isinstance(selector_code, Mapping)
        or selector_code.get("worktree_dirty") is not False
        or not isinstance(selector_code.get("code_commit"), str)
        or len(selector_code["code_commit"]) != 40
    ):
        raise ParentSelectorError("parent selector was not produced from a clean commit")
    if complete.get("verified_success_receipt_count") != 120:
        raise ParentSelectorError("parent selector does not bind 120 successful receipts")
    winners = summary.get("winners")
    if not isinstance(winners, Mapping) or set(winners) != set(EXPECTED_MODEL_IDS):
        raise ParentSelectorError("parent selector winners differ from six-model order")

    models: list[MainModel] = []
    for declared in spec.models:
        winner = winners.get(declared.model_id)
        if not isinstance(winner, Mapping):
            raise ParentSelectorError(f"winner missing for {declared.model_id}")
        try:
            rate = float(winner["learning_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ParentSelectorError("winner learning rate is malformed") from exc
        if not math.isfinite(rate) or rate not in spec.learning_rates:
            raise ParentSelectorError(f"winner rate is outside selector grid: {rate}")
        if int(winner.get("selection_seed_count", -1)) != 5:
            raise ParentSelectorError("winner is not based on all five selection seeds")
        models.append(
            MainModel(
                model_id=declared.model_id,
                hidden_width=declared.hidden_width,
                parameter_count=declared.parameter_count,
                learning_rate=rate,
            )
        )
    binding = {
        "schema_version": 1,
        "campaign_id": "sagodi_six_model_lr_selection_v3",
        "scientific_identity": manifest["scientific_identity"],
        "selector_protocol_canonical_fingerprint": manifest[
            "protocol_canonical_fingerprint"
        ],
        "manifest_sha256": sha256_file(root / "lr_selection_manifest.json"),
        "summary_sha256": sha256_file(root / "lr_selection_summary.json"),
        "selection_receipt_sha256": sha256_file(
            root / "lr_selection_receipt.json"
        ),
        "completion_receipt_sha256": sha256_file(
            root / "completion_receipt.json"
        ),
        "selector_complete_sha256": sha256_file(root / "COMPLETE"),
        "selector_code_commit": selector_code["code_commit"],
        "verified_nested_training_receipts": 120,
        "selected_learning_rates": {
            model.model_id: model.learning_rate for model in models
        },
    }
    return ParentSelector(root, protocol, manifest, tuple(models), binding)


def materialize_resolved_protocol(
    template: Mapping[str, Any], parent: ParentSelector
) -> dict[str, Any]:
    """Resolve a full executable main protocol without mutating the selector copy."""

    protocol = copy.deepcopy(parent.protocol)
    protocol["freeze_id"] = "sagodi_primary_main_v3_resolved"
    protocol["freeze_status"] = "resolved_before_training"
    protocol["campaign_mode"] = CAMPAIGN_MODE
    protocol["parent_selector"] = copy.deepcopy(parent.binding)

    reporting = protocol["reporting"]
    reporting["training_track"] = PROTOCOL_TRACK
    reporting["display_label"] = "six-model selector-bound Ságodi primary training"
    reporting["analysis_role"] = "primary_comparative_training"
    reporting["bit_exact_official_implementation"] = False
    reporting["protocol_A_eligible"] = False
    reporting["protocol_B_confirmatory_eligible"] = False

    scope = protocol["scope"]
    scope["selection_results_are_approximate_ca_evidence"] = False
    scope["old_campaign_results_may_be_pooled"] = False
    scope["later_phases"] = [
        {
            "id": "sagodi_primary_analysis_v3",
            "enabled": False,
            "activation_gate": "verified_60_run_primary_main_training_receipt",
        }
    ]
    seeds = protocol["seed_policy"]
    seeds["main_model_seeds"] = list(EXPECTED_MAIN_SEEDS)

    phase = protocol["phase1_ring_pilot"]
    phase["protocol_track"] = PROTOCOL_TRACK
    phase["purpose"] = "six_model_primary_main_training"
    phase["confirmatory"] = True
    training = phase["training"]
    training["optimizer_updates"] = EXPECTED_UPDATES
    training["batch_size"] = EXPECTED_BATCH_SIZE
    training["optimizer"] = copy.deepcopy(template["training"]["optimizer"])
    learning_rates = {model.model_id: model.learning_rate for model in parent.models}
    lr_block = training["learning_rate"]
    lr_block["active_launch_values"] = sorted(set(learning_rates.values()))
    lr_block["selection_status"] = "verified_parent_selector_bound"
    lr_block["selected_by_model"] = learning_rates
    lr_block["source"] = "selector_bound"
    lr_block["selector_binding"] = {
        "scientific_identity": parent.binding["scientific_identity"],
        "summary_sha256": parent.binding["summary_sha256"],
        "selection_rule": "minimum_mean_online_training_loss_at_update_100",
    }
    state_noise = training["state_noise"]
    state_noise["enabled"] = True
    state_noise["coordinate_standard_deviation"] = 0.1
    state_noise["coordinate_variance"] = 0.01
    state_noise["analysis_noise_enabled"] = False
    training["gradient_clipping"] = {
        "policy": "none",
        "frozen_numeric_value": None,
    }
    training["checkpoint_selection"] = "final_update_5000"
    rp = template["retention_plasticity"]
    training["rp_schedule_for_ca_lru"] = {
        "enabled_during_training": True,
        "warmup_updates": int(rp["warmup_updates"]),
        "interval_updates": int(rp["interval_updates"]),
        "calls_after_warmup": int(rp["calls_after_warmup"]),
        "probe_batch_size": int(rp["probe_batch_size"]),
        "probe_horizon": int(rp["probe_horizon"]),
        "blank_ablation_horizon": int(rp["blank_ablation_horizon"]),
        "probe_noise_enabled": bool(rp["probe_noise_enabled"]),
        "eta_lambda": float(rp["eta_lambda"]),
        "damage_epsilon": float(rp["damage_epsilon"]),
    }
    phase["run_matrix"] = {
        "cross_product": ["models", "main_model_seeds"],
        "learning_rate_assignment": "selected_by_model",
        "expected_training_runs": EXPECTED_RUN_COUNT,
        "task_seed_fixed": 0,
        "data_stream_seed_fixed": 0,
    }
    evaluation = protocol["evaluation"]
    evaluation["analysis_enabled"] = False
    evaluation["selection_results_are_approximate_ca_evidence"] = False
    evaluation["disabled_reason"] = "primary_training_campaign_only"
    claim_gates = protocol["claim_gates"]
    claim_gates["status"] = "disabled_until_separate_primary_analysis_receipt"
    claim_gates["all_approximate_ca_claims_enabled"] = False
    claim_gates["selection_artifacts_may_be_reused_as_ca_evidence"] = False
    return protocol


def build_main_plan(parent: ParentSelector) -> tuple[MainRun, ...]:
    plan = tuple(
        MainRun(model=model, model_seed=seed)
        for model in parent.models
        for seed in EXPECTED_MAIN_SEEDS
    )
    if len(plan) != EXPECTED_RUN_COUNT or len({run.run_id for run in plan}) != len(plan):
        raise RuntimeError("main plan is not exactly 60 unique model/seed runs")
    return plan


def _receipt_metadata(
    manifest: Mapping[str, Any], stage: str, run_id: str
) -> dict[str, str]:
    return {
        "campaign_scientific_identity": str(manifest["scientific_identity"]),
        "protocol_fingerprint": str(manifest["protocol_canonical_fingerprint"]),
        "run_id": str(run_id),
        "stage": str(stage),
    }


def _child_environment(
    manifest: Mapping[str, Any], run: MainRun, gpu: int
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    env["CALRU_PHYSICAL_GPU_ID"] = str(int(gpu))
    metadata = _receipt_metadata(manifest, "primary_main_training", run.run_id)
    for key, variable in RECEIPT_IDENTITY_ENV.items():
        env[variable] = metadata[key]
    return env


def _child_preexec(expected_parent: int) -> Any:
    def configure() -> None:
        os.setsid()
        if not sys.platform.startswith("linux"):
            os._exit(125)
        libc = ctypes.CDLL(None, use_errno=True)
        if int(libc.prctl(1, signal.SIGKILL, 0, 0, 0)) != 0:
            os._exit(125)
        if os.getppid() != expected_parent:
            os.kill(os.getpid(), signal.SIGKILL)

    return configure


def _prepare_root(
    root: Path,
    template_path: Path,
    template: Mapping[str, Any],
    parent: ParentSelector,
    resolved: Mapping[str, Any],
    *,
    smoke: bool,
) -> None:
    marker_payload = {
        "schema_version": 1,
        "campaign_type": CAMPAIGN_MODE,
        "scope": SCOPE,
        "template_source_sha256": sha256_file(template_path),
        "template_canonical_fingerprint": canonical_hash(template),
        "parent_selector": parent.binding,
        "resolved_protocol_canonical_fingerprint": canonical_hash(resolved),
        "smoke": bool(smoke),
    }
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ROOT_MARKER
    if marker.exists():
        if strict_json_load(marker) != marker_payload:
            raise RuntimeError("main artifact-root marker differs; choose a new root")
    else:
        if any(root.iterdir()):
            raise RuntimeError("main artifact root is nonempty and has no v3 marker")
        atomic_json(marker, marker_payload)
    copies = (
        (template_path.read_bytes(), root / TEMPLATE_COPY),
        (json.dumps(resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n", root / RESOLVED_PROTOCOL),
    )
    for payload, destination in copies:
        if destination.exists():
            if destination.read_bytes() != payload:
                raise RuntimeError(f"immutable main campaign file differs: {destination.name}")
        else:
            atomic_bytes(destination, payload)


def _build_manifest(
    *,
    root: Path,
    parent: ParentSelector,
    plan: Sequence[MainRun],
    evaluation_bank: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    git_state: Mapping[str, Any],
    environment: Mapping[str, Any],
    python: str,
    gpus: Sequence[int],
    smoke: bool,
) -> dict[str, Any]:
    protocol = load_protocol(root / RESOLVED_PROTOCOL)
    payload = {
        "campaign_type": CAMPAIGN_MODE,
        "scope": SCOPE,
        "ca_evidence": False,
        "manifold_analysis_performed": False,
        "confirmatory_primary_training": True,
        "smoke": bool(smoke),
        "parent_selector": parent.binding,
        "template_file": TEMPLATE_COPY,
        "template_file_sha256": sha256_file(root / TEMPLATE_COPY),
        "resolved_protocol_file": RESOLVED_PROTOCOL,
        "resolved_protocol_file_sha256": sha256_file(root / RESOLVED_PROTOCOL),
        "protocol_canonical_fingerprint": protocol_fingerprint(protocol),
        "source_protocol": dict(protocol["source_protocol"]),
        "source_hashes": dict(source_hashes),
        "code": dict(git_state),
        "environment": dict(environment),
        "python": str(Path(python).resolve()),
        "gpus": [int(gpu) for gpu in gpus],
        "evaluation_bank": dict(evaluation_bank),
        "run_matrix": [run.payload() for run in plan],
        "expected_training_runs": EXPECTED_RUN_COUNT,
        "failed_seed_replacement_policy": "forbidden",
        "eligibility_policy": "record_later_never_replace_training_seed",
    }
    return {
        "schema_version": 1,
        **payload,
        "scientific_identity_payload": payload,
        "scientific_identity": canonical_hash(payload),
    }


def _write_or_check(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if strict_json_load(path) != dict(payload):
            raise RuntimeError(f"immutable main campaign file differs: {path.name}")
    else:
        atomic_json(path, dict(payload))


def _verify_parent_binding_fast(parent: ParentSelector) -> None:
    paths = {
        "manifest_sha256": parent.root / "lr_selection_manifest.json",
        "summary_sha256": parent.root / "lr_selection_summary.json",
        "selection_receipt_sha256": parent.root / "lr_selection_receipt.json",
        "completion_receipt_sha256": parent.root / "completion_receipt.json",
        "selector_complete_sha256": parent.root / "COMPLETE",
    }
    for key, path in paths.items():
        if not path.is_file() or sha256_file(path) != parent.binding[key]:
            raise RuntimeError(f"parent selector binding changed: {key}")


def _verify_campaign_inputs(
    root: Path,
    manifest: Mapping[str, Any],
    parent: ParentSelector,
    repo_root: Path,
) -> None:
    signed = manifest.get("scientific_identity_payload")
    if not isinstance(signed, Mapping) or canonical_hash(signed) != manifest.get(
        "scientific_identity"
    ):
        raise RuntimeError("main manifest scientific identity is invalid")
    if strict_json_load(root / MANIFEST) != dict(manifest):
        raise RuntimeError("on-disk main manifest differs")
    if _git_state(repo_root) != manifest["code"]:
        raise RuntimeError("git state changed after main manifest creation")
    if _source_hashes(repo_root, Path(__file__).resolve().parent) != manifest["source_hashes"]:
        raise RuntimeError("campaign Python source hashes changed")
    for key in ("template", "resolved_protocol"):
        name = manifest[f"{key}_file"]
        if sha256_file(root / name) != manifest[f"{key}_file_sha256"]:
            raise RuntimeError(f"frozen {key} changed")
    protocol = load_protocol(root / RESOLVED_PROTOCOL)
    if protocol_fingerprint(protocol) != manifest["protocol_canonical_fingerprint"]:
        raise RuntimeError("resolved protocol semantics changed")
    source = manifest.get("source_protocol")
    if not isinstance(source, Mapping):
        raise RuntimeError("main source-protocol binding is malformed")
    source_path = repo_root / str(source.get("path", ""))
    if not source_path.is_file() or sha256_file(source_path) != source.get("sha256"):
        raise RuntimeError("normative main source protocol is missing or changed")
    _verify_parent_binding_fast(parent)
    bank = manifest["evaluation_bank"]
    bank_path = root / str(bank["path"])
    sidecar = Path(f"{bank_path}.sha256")
    if sha256_file(bank_path) != bank["sha256"] or sha256_file(sidecar) != bank[
        "sidecar_sha256"
    ]:
        raise RuntimeError("fixed main evaluation bank changed")


def _training_output(root: Path, run: MainRun) -> Path:
    return root / "training" / f"model={run.model.model_id}" / f"seed={run.model_seed}"


def _expected_child_rp(run: MainRun, *, smoke: bool) -> tuple[int, ...]:
    if run.model.model_id != "ca_lru":
        return ()
    return (1, 2) if smoke else EXPECTED_RP_STEPS


def _verify_training_output(
    output: Path,
    run: MainRun,
    manifest: Mapping[str, Any],
    *,
    parent: ParentSelector,
) -> tuple[bool, str, dict[str, str] | None]:
    valid, reason = verify_completion_receipt(
        output / COMPLETION_RECEIPT,
        expected_job_id=run.receipt_job_id,
        expected_metadata=_receipt_metadata(
            manifest, "primary_main_training", run.run_id
        ),
    )
    if not valid:
        return False, reason, None
    required = {
        "config.json",
        "checkpoint.pt",
        "training_trace.npz",
        "task_metrics.json",
        "rp_trace.json",
        "manifest.json",
    }
    try:
        receipt = strict_json_load(output / COMPLETION_RECEIPT)
        config = strict_json_load(output / "config.json")
        child = strict_json_load(output / "manifest.json")
        metrics = strict_json_load(output / "task_metrics.json")
        rp_trace = strict_json_load(output / "rp_trace.json")
        with np.load(output / "training_trace.npz", allow_pickle=False) as archive:
            steps = np.asarray(archive["step"], dtype=np.int64)
            losses = np.asarray(archive["masked_mse"], dtype=np.float64)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, f"main training output is unreadable: {exc}", None
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or not required.issubset(artifacts):
        return False, "main training receipt lacks required artifacts", None
    spec = config.get("train_spec")
    training = config.get("training")
    metadata = config.get("model")
    if not all(isinstance(item, Mapping) for item in (spec, training, metadata)):
        return False, "main training config is malformed", None
    expected_steps = 2 if manifest["smoke"] else EXPECTED_UPDATES
    expected_batch = 4 if manifest["smoke"] else EXPECTED_BATCH_SIZE
    try:
        checks = (
            (spec.get("model_name") == run.model.model_id, "model id"),
            (int(spec.get("model_seed", -1)) == run.model_seed, "model seed"),
            (
                float(spec.get("learning_rate", float("nan")))
                == run.model.learning_rate,
                "learning rate",
            ),
            (int(training.get("steps", -1)) == expected_steps, "update count"),
            (int(training.get("batch_size", -1)) == expected_batch, "batch size"),
            (
                int(metadata.get("parameters_total", -1))
                == run.model.parameter_count,
                "parameter count",
            ),
        )
    except (TypeError, ValueError):
        return False, "main training numeric config is malformed", None
    for passed, label in checks:
        if not passed:
            return False, f"main training {label} mismatch", None
    model_config = metadata.get("model_config")
    try:
        observed_width = (
            int(model_config.get("width", -1))
            if isinstance(model_config, Mapping)
            else -1
        )
    except (TypeError, ValueError):
        observed_width = -1
    if observed_width != run.model.hidden_width:
        return False, "main training hidden width mismatch", None
    expected_rp = _expected_child_rp(run, smoke=bool(manifest["smoke"]))
    if not isinstance(rp_trace, list) or not all(
        isinstance(item, Mapping) for item in rp_trace
    ):
        return False, "main training RP trace is malformed", None
    try:
        actual_rp = tuple(int(item.get("step", -1)) for item in rp_trace)
    except (TypeError, ValueError):
        return False, "main training RP trace steps are malformed", None
    if actual_rp != expected_rp or training.get("expected_rp_steps") != list(expected_rp):
        return False, "main training RP step schedule mismatch", None
    expected_enabled = run.model.model_id == "ca_lru"
    if training.get("rp_enabled_by_protocol") is not expected_enabled:
        return False, "main training RP enablement mismatch", None
    try:
        observed_noise_std = float(
            training.get("state_noise_coordinate_std", float("nan"))
        )
    except (TypeError, ValueError):
        observed_noise_std = float("nan")
    if training.get("state_noise_enabled") is not True or observed_noise_std != 0.1:
        return False, "main training state-noise contract mismatch", None
    if training.get("gradient_clipping") != {
        "policy": "none",
        "frozen_numeric_value": None,
    }:
        return False, "main training unexpectedly clipped gradients", None
    if child.get("campaign_identity") != manifest["scientific_identity"]:
        return False, "main child campaign identity mismatch", None
    if child.get("protocol_canonical_fingerprint") != manifest[
        "protocol_canonical_fingerprint"
    ]:
        return False, "main child protocol fingerprint mismatch", None
    if child.get("protocol_track") != PROTOCOL_TRACK:
        return False, "main child protocol track mismatch", None
    if child.get("campaign_type") != CAMPAIGN_MODE or child.get(
        "artifact_role"
    ) != "primary_main_training":
        return False, "main child artifact role/campaign type mismatch", None
    if child.get("parent_selector") != parent.binding:
        return False, "main child parent-selector binding mismatch", None
    if child.get("pilot_only") is not False:
        return False, "main child is incorrectly labelled pilot-only", None
    if child.get("ca_evidence") is not False or child.get(
        "manifold_analysis_performed"
    ) is not False:
        return False, "main child must not claim CA evidence", None
    try:
        child_seed = int(child.get("model_seed", -1))
    except (TypeError, ValueError):
        child_seed = -1
    if child.get("model_id") != run.model.model_id or child_seed != run.model_seed:
        return False, "main child model identity mismatch", None
    receipt_metadata = receipt.get("metadata")
    if not isinstance(receipt_metadata, Mapping):
        return False, "main child receipt metadata is malformed", None
    try:
        receipt_rp_calls = int(receipt_metadata.get("rp_calls", -1))
    except (TypeError, ValueError):
        receipt_rp_calls = -1
    if (
        receipt_metadata.get("pilot_only") is not False
        or receipt_metadata.get("campaign_type") != CAMPAIGN_MODE
        or receipt_metadata.get("artifact_role") != "primary_main_training"
        or receipt_metadata.get("parent_selector") != parent.binding
        or receipt_rp_calls != len(expected_rp)
    ):
        return False, "main child receipt scientific labels mismatch", None
    if child.get("evaluation_bank_sha256") != manifest["evaluation_bank"]["sha256"]:
        return False, "main child evaluation-bank binding mismatch", None
    state_path = parent.root / "phase0" / f"model={run.model.model_id}" / "state_spec.json"
    expected_state_hash = None if manifest["smoke"] else sha256_file(state_path)
    if child.get("state_spec_sha256") != expected_state_hash:
        return False, "main child parent Phase-0 state binding mismatch", None
    if steps.ndim != 1 or not np.array_equal(
        steps, np.arange(1, expected_steps + 1, dtype=np.int64)
    ):
        return False, "main training steps are not exact", None
    if losses.shape != steps.shape or not np.isfinite(losses).all():
        return False, "main training losses are malformed/non-finite", None
    try:
        recorded = float(metrics["train_loss_last"])
    except (KeyError, TypeError, ValueError):
        return False, "main task metrics lack train_loss_last", None
    if float(np.float32(recorded)) != float(losses[-1]):
        return False, "main trace/final online loss mismatch", None
    return True, "verified primary-main training receipt", {
        "completion_receipt_sha256": sha256_file(output / COMPLETION_RECEIPT),
        "checkpoint_sha256": sha256_file(output / "checkpoint.pt"),
        "training_trace_sha256": sha256_file(output / "training_trace.npz"),
        "task_metrics_sha256": sha256_file(output / "task_metrics.json"),
        "manifest_sha256": sha256_file(output / "manifest.json"),
    }


def _recover_training_attempt(
    root: Path,
    run: MainRun,
    manifest: Mapping[str, Any],
    parent: ParentSelector,
) -> tuple[bool, str]:
    attempt_parent = root / "attempts" / "primary_main_training" / run.run_id
    if not attempt_parent.is_dir():
        return False, "no unpublished attempt"
    valid_attempts = []
    for candidate in sorted(attempt_parent.iterdir()):
        if candidate.is_dir() and _verify_training_output(
            candidate, run, manifest, parent=parent
        )[0]:
            valid_attempts.append(candidate)
    if len(valid_attempts) > 1:
        raise RuntimeError(f"multiple valid attempts for {run.run_id}; audit required")
    if not valid_attempts:
        return False, "no verified unpublished attempt"
    output = _training_output(root, run)
    if output.exists():
        raise RuntimeError(f"cannot recover {run.run_id} over existing output")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(valid_attempts[0], output)
    return True, f"recovered verified attempt {valid_attempts[0]}"


def _training_command(
    root: Path,
    attempt: Path,
    run: MainRun,
    manifest: Mapping[str, Any],
    parent: ParentSelector,
    python: str,
) -> list[str]:
    command = [
        python,
        "-m",
        "repro.sagodi_protocol.train",
        "--protocol",
        str(root / RESOLVED_PROTOCOL),
        "--model",
        run.model.model_id,
        "--model-seed",
        str(run.model_seed),
        "--learning-rate",
        str(run.model.learning_rate),
        "--output-dir",
        str(attempt),
        "--evaluation-bank",
        str(root / manifest["evaluation_bank"]["path"]),
        "--campaign-identity",
        str(manifest["scientific_identity"]),
        "--device",
        "cuda:0",
    ]
    if manifest["smoke"]:
        command.append("--smoke")
    else:
        command.extend(
            [
                "--state-spec",
                str(
                    parent.root
                    / "phase0"
                    / f"model={run.model.model_id}"
                    / "state_spec.json"
                ),
            ]
        )
    return command


def _terminate_active(active: Mapping[int, Mapping[str, Any]]) -> None:
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
    root: Path,
    manifest: Mapping[str, Any],
    parent: ParentSelector,
    plan: Sequence[MainRun],
    python: str,
    gpus: Sequence[int],
    status: dict[str, Any],
    repo_root: Path,
) -> None:
    jobs = status.setdefault("jobs", {})
    pending: list[MainRun] = []
    for run in plan:
        output = _training_output(root, run)
        valid, reason, _ = _verify_training_output(
            output, run, manifest, parent=parent
        )
        if valid:
            jobs[run.run_id] = {"state": "complete", "reason": reason}
            continue
        if output.exists():
            preserved = _preserve_invalid_output(
                root, "primary_main_training", run.run_id, output
            )
            reason = f"invalid output preserved at {preserved}: {reason}"
        recovered, recovery_reason = _recover_training_attempt(
            root, run, manifest, parent
        )
        if recovered:
            jobs[run.run_id] = {"state": "complete", "reason": recovery_reason}
            continue
        jobs[run.run_id] = {"state": "pending", "reason": reason}
        pending.append(run)
    atomic_json(root / STATUS, status)

    available = sorted(int(gpu) for gpu in gpus)
    active: dict[int, dict[str, Any]] = {}
    try:
        while pending or active:
            while pending and available:
                _verify_campaign_inputs(root, manifest, parent, repo_root)
                gpu = available.pop(0)
                run = pending.pop(0)
                attempt = _unique_attempt_path(
                    root, "primary_main_training", run.run_id
                )
                log_dir = root / "logs" / "primary_main_training" / run.run_id
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{attempt.name}.log"
                handle = log_path.open("ab", buffering=0)
                process = subprocess.Popen(
                    _training_command(
                        root, attempt, run, manifest, parent, python
                    ),
                    cwd=Path(__file__).resolve().parents[2],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=_child_environment(manifest, run, gpu),
                    preexec_fn=_child_preexec(os.getpid()),
                )
                jobs[run.run_id] = {
                    "state": "running",
                    "pid": process.pid,
                    "gpu": gpu,
                    "attempt_dir": str(attempt),
                    "log": str(log_path),
                }
                active[gpu] = {
                    "run": run,
                    "process": process,
                    "attempt": attempt,
                    "handle": handle,
                    "log": log_path,
                }
                atomic_json(root / STATUS, status)
            progressed = False
            for gpu, item in list(active.items()):
                return_code = item["process"].poll()
                if return_code is None:
                    continue
                progressed = True
                item["handle"].close()
                run = item["run"]
                attempt = item["attempt"]
                del active[gpu]
                available.append(gpu)
                available.sort()
                if return_code != 0:
                    jobs[run.run_id] = {
                        "state": "retryable_failure_same_seed_only",
                        "return_code": return_code,
                        "attempt_dir": str(attempt),
                        "log": str(item["log"]),
                    }
                    atomic_json(root / STATUS, status)
                    raise RuntimeError(
                        f"main training {run.run_id} exited {return_code}; "
                        f"resume same seed after inspection: {item['log']}"
                    )
                valid, reason, _ = _verify_training_output(
                    attempt, run, manifest, parent=parent
                )
                if not valid:
                    jobs[run.run_id] = {
                        "state": "retryable_invalid_output_same_seed_only",
                        "reason": reason,
                        "attempt_dir": str(attempt),
                    }
                    atomic_json(root / STATUS, status)
                    raise RuntimeError(
                        f"main training {run.run_id} output invalid: {reason}"
                    )
                output = _training_output(root, run)
                output.parent.mkdir(parents=True, exist_ok=True)
                os.replace(attempt, output)
                valid, reason, _ = _verify_training_output(
                    output, run, manifest, parent=parent
                )
                if not valid:
                    raise RuntimeError(f"published main output became invalid: {reason}")
                jobs[run.run_id] = {"state": "complete", "reason": reason}
                atomic_json(root / STATUS, status)
            if not progressed and active:
                time.sleep(0.2)
    except BaseException:
        _terminate_active(active)
        raise


def _finalize(
    root: Path,
    manifest: Mapping[str, Any],
    parent: ParentSelector,
    plan: Sequence[MainRun],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    child_receipts: list[Path] = []
    for run in plan:
        output = _training_output(root, run)
        valid, reason, bindings = _verify_training_output(
            output, run, manifest, parent=parent
        )
        if not valid or bindings is None:
            raise RuntimeError(f"cannot finalize {run.run_id}: {reason}")
        metrics = strict_json_load(output / "task_metrics.json")
        rows.append(
            {
                **run.payload(),
                "status": "complete",
                "validation_masked_nmse_db": metrics.get("masked_nmse_db"),
                "structural_summary_eligible_nmse_lt_minus20db": (
                    float(metrics["masked_nmse_db"]) < -20.0
                    if "masked_nmse_db" in metrics
                    else None
                ),
                "train_loss_last": metrics.get("train_loss_last"),
                "rp_calls": metrics.get("rp_calls"),
                "artifact_bindings": bindings,
                "eligibility": "not_assigned_by_training_campaign",
            }
        )
        child_receipts.append(output / COMPLETION_RECEIPT)
    summary = {
        "schema_version": 1,
        "campaign_type": CAMPAIGN_MODE,
        "campaign_scientific_identity": manifest["scientific_identity"],
        "scope": SCOPE,
        "ca_evidence": False,
        "manifold_analysis_performed": False,
        "confirmatory_primary_training": True,
        "smoke": bool(manifest["smoke"]),
        "expected_training_runs": EXPECTED_RUN_COUNT,
        "verified_training_receipt_count": len(rows),
        "failed_seed_replacements": 0,
        "parent_selector": parent.binding,
        "runs": rows,
    }
    atomic_json(root / SUMMARY, summary)
    metadata = {
        **_receipt_metadata(manifest, "finalization", "primary_main_v3"),
        "run_count": len(rows),
        "ca_evidence": False,
        "manifold_analysis_performed": False,
    }
    atomic_json(
        root / COMPLETE,
        {
            "schema_version": 1,
            "status": "complete",
            "campaign_scientific_identity": manifest["scientific_identity"],
            "scope": SCOPE,
            "verified_training_receipt_count": len(rows),
            "failed_seed_replacements": 0,
            "summary_sha256": sha256_file(root / SUMMARY),
        },
    )
    write_completion_receipt(
        root / COMPLETION_RECEIPT,
        job_id="sagodi_primary_main_v3_campaign_complete",
        artifacts=[
            root / MANIFEST,
            root / TEMPLATE_COPY,
            root / RESOLVED_PROTOCOL,
            root / SUMMARY,
            root / COMPLETE,
            *child_receipts,
        ],
        metadata=metadata,
    )
    valid, reason = verify_main_completion(root, manifest, parent, plan)
    if not valid:
        raise RuntimeError(f"final main recursive receipt verification failed: {reason}")
    return summary


def verify_main_completion(
    root: Path,
    manifest: Mapping[str, Any],
    parent: ParentSelector,
    plan: Sequence[MainRun],
) -> tuple[bool, str]:
    try:
        complete = strict_json_load(root / COMPLETE)
        summary = strict_json_load(root / SUMMARY)
        for run in plan:
            valid, reason, _ = _verify_training_output(
                _training_output(root, run), run, manifest, parent=parent
            )
            if not valid:
                return False, f"invalid nested run {run.run_id}: {reason}"
        metadata = {
            **_receipt_metadata(manifest, "finalization", "primary_main_v3"),
            "run_count": EXPECTED_RUN_COUNT,
            "ca_evidence": False,
            "manifold_analysis_performed": False,
        }
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, f"main completion is unreadable: {exc}"
    valid, reason = verify_completion_receipt(
        root / COMPLETION_RECEIPT,
        expected_job_id="sagodi_primary_main_v3_campaign_complete",
        expected_metadata=metadata,
    )
    if not valid:
        return False, reason
    if summary.get("verified_training_receipt_count") != EXPECTED_RUN_COUNT:
        return False, "main summary run count mismatch"
    if complete.get("summary_sha256") != sha256_file(root / SUMMARY):
        return False, "main COMPLETE does not bind summary"
    if complete.get("failed_seed_replacements") != 0:
        return False, "main completion contains seed replacement"
    return True, "verified all 60 nested primary-main training receipts"


def run_primary_main_campaign(
    *,
    selector_root: Path,
    template_path: Path,
    artifact_root: Path,
    python: str,
    gpus: Sequence[int],
    smoke: bool = False,
) -> Path:
    selector_root = Path(selector_root).resolve(strict=True)
    template_path = Path(template_path).resolve(strict=True)
    root = Path(artifact_root).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    python = _resolve_python(python)
    gpus = _validated_gpu_ids(gpus)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES; --gpus uses physical GPU ids")
    template = load_main_template(template_path)
    parent = verify_parent_selector(selector_root)
    resolved = materialize_resolved_protocol(template, parent)
    plan = build_main_plan(parent)
    git_state = _git_state(repo_root)
    if not smoke and git_state["worktree_dirty"]:
        raise RuntimeError("full primary-main training requires a clean committed worktree")
    if not smoke and git_state["code_commit"] != parent.binding[
        "selector_code_commit"
    ]:
        raise RuntimeError(
            "full primary-main training must use the same code commit as its selector"
        )
    _prepare_root(
        root, template_path, template, parent, resolved, smoke=bool(smoke)
    )
    # Fail before acquiring GPUs if the resolved bytes do not pass the active
    # fail-closed configuration validator.
    load_protocol(root / RESOLVED_PROTOCOL)
    lock = _acquire_campaign_lock(root)
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"primary-main campaign interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        protocol = load_protocol(root / RESOLVED_PROTOCOL)
        evaluation_bank = _materialize_evaluation_bank(root, protocol)
        manifest = _build_manifest(
            root=root,
            parent=parent,
            plan=plan,
            evaluation_bank=evaluation_bank,
            source_hashes=_source_hashes(repo_root, Path(__file__).resolve().parent),
            git_state=git_state,
            environment=_environment_fingerprint(python, gpus),
            python=python,
            gpus=gpus,
            smoke=smoke,
        )
        _write_or_check(root / MANIFEST, manifest)
        _verify_campaign_inputs(root, manifest, parent, repo_root)
        complete, _ = verify_main_completion(root, manifest, parent, plan)
        if complete:
            return root
        status: dict[str, Any] = {
            "schema_version": 1,
            "campaign_type": CAMPAIGN_MODE,
            "scientific_identity": manifest["scientific_identity"],
            "scope": SCOPE,
            "ca_evidence": False,
            "manifold_analysis_performed": False,
            "smoke": bool(smoke),
            "stage": "primary_main_training",
            "jobs": {},
        }
        if (root / STATUS).exists():
            existing = strict_json_load(root / STATUS)
            if isinstance(existing, Mapping):
                status["jobs"] = dict(existing.get("jobs", {}))
        atomic_json(root / STATUS, status)
        _run_training_jobs(
            root, manifest, parent, plan, python, gpus, status, repo_root
        )
        # Re-run the recursive parent verification at the final scientific
        # boundary; fast hash checks above protect the launch loop itself.
        verified_parent = verify_parent_selector(selector_root)
        if verified_parent.binding != parent.binding:
            raise RuntimeError("parent selector identity changed during main training")
        _verify_campaign_inputs(root, manifest, parent, repo_root)
        status["stage"] = "finalizing"
        atomic_json(root / STATUS, status)
        summary = _finalize(root, manifest, parent, plan)
        status["stage"] = "complete"
        status["verified_training_receipt_count"] = summary[
            "verified_training_receipt_count"
        ]
        status["completed_at"] = time.time()
        atomic_json(root / STATUS, status)
        return root
    finally:
        lock.release()
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = run_primary_main_campaign(
        selector_root=args.selector_root,
        template_path=args.template,
        artifact_root=args.artifact_root,
        python=args.python,
        gpus=tuple(int(value) for value in args.gpus.split(",") if value.strip()),
        smoke=args.smoke,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "artifact_root": str(output),
                "scope": SCOPE,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
