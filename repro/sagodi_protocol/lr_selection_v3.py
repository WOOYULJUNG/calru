"""Declarative six-model, training-only learning-rate selection campaign.

This runner is intentionally separate from :mod:`lr_selection`, whose public
contract is the historical four-model/80-run v2 freeze.  The v3 runner takes a
small JSON specification that declares the exact models, widths, parameter
counts, seeds and learning rates.  It launches only ``phase0`` and ``train``;
it never imports or invokes a manifold-analysis entry point.

For a full campaign, every cell in the 6 x 5 x 4 matrix must have a recursively
verified training receipt.  The winner for each model is the learning rate
with the smallest arithmetic mean of the online training loss recorded at
update 100 across all five selection seeds.  An exact tie is broken in favour
of the smaller numeric learning rate.  Evaluation metrics are retained for
audit but do not participate in selection.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
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
from .config import load_protocol, protocol_fingerprint
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


EXPECTED_MODEL_IDS = (
    "rnn_param206",
    "gru_sagodi_param135",
    "lstm_param109",
    "lru_param96",
    "no_rp",
    "ca_lru",
)
EXPECTED_LEARNING_RATES = (1e-2, 1e-3, 1e-4, 1e-5)
EXPECTED_SELECTION_SEED_COUNT = 5
EXPECTED_UPDATE = 100
EXPECTED_BATCH_SIZE = 64
EXPECTED_RUN_COUNT = 120
SCOPE = "training_only_lr_selection_no_manifold_analysis_no_ca_evidence"
ROOT_MARKER = ".calru_lr_selection_v3_root.json"
SELECTOR_COPY = "selector_spec.json"
PROTOCOL_COPY = "protocol.yaml"
MANIFEST = "lr_selection_manifest.json"
STATUS = "status.json"
SUMMARY = "lr_selection_summary.json"
MATRIX = "lr_selection_run_matrix.csv"
SELECTION_RECEIPT = "lr_selection_receipt.json"
COMPLETION_RECEIPT = "completion_receipt.json"
COMPLETE = "COMPLETE"

EXPECTED_SELECTION_RULE = {
    "primary_metric": "online_training_masked_mse_at_update_100",
    "seed_aggregation": "arithmetic_mean_across_all_five_selection_seeds",
    "winner": "minimum_primary_metric",
    "tie_break": "smaller_numeric_learning_rate",
    "validation_metrics_role": "recorded_but_non_selecting",
    "all_matrix_cells_required": True,
}


class SelectorSpecError(ValueError):
    """The declarative selector does not encode the frozen 120-run design."""


class IncompleteSelectionError(RuntimeError):
    """At least one required selector cell is absent, invalid or failed."""


@dataclass(frozen=True)
class SelectorModel:
    model_id: str
    hidden_width: int
    parameter_count: int

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "hidden_width": int(self.hidden_width),
            "parameter_count": int(self.parameter_count),
        }


@dataclass(frozen=True)
class SelectorSpec:
    campaign_id: str
    models: tuple[SelectorModel, ...]
    selection_model_seeds: tuple[int, ...]
    learning_rates: tuple[float, ...]
    optimizer_updates: int
    batch_size: int
    expected_runs: int
    selection_rule: Mapping[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "campaign_id": self.campaign_id,
            "scope": SCOPE,
            "models": [model.payload() for model in self.models],
            "selection_model_seeds": list(self.selection_model_seeds),
            "learning_rates": list(self.learning_rates),
            "optimizer_updates": int(self.optimizer_updates),
            "batch_size": int(self.batch_size),
            "expected_runs": int(self.expected_runs),
            "selection_rule": dict(self.selection_rule),
        }


@dataclass(frozen=True)
class SelectionRun:
    run_id: str
    model_id: str
    hidden_width: int
    parameter_count: int
    model_seed: int
    learning_rate: float
    required_update: int
    batch_size: int

    @property
    def receipt_job_id(self) -> str:
        return (
            f"{self.model_id}-seed{self.model_seed}-"
            f"lr{self.learning_rate:g}"
        )

    def payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "hidden_width": int(self.hidden_width),
            "parameter_count": int(self.parameter_count),
            "model_seed": int(self.model_seed),
            "learning_rate": float(self.learning_rate),
            "required_update": int(self.required_update),
            "batch_size": int(self.batch_size),
        }


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise SelectorSpecError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _array(value: Any, label: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise SelectorSpecError(f"{label} must be a JSON array")
    return tuple(value)


def parse_selector_spec(payload: Mapping[str, Any]) -> SelectorSpec:
    """Validate and normalize a JSON selector declaration."""

    if not isinstance(payload, Mapping):
        raise SelectorSpecError("selector specification must be a JSON object")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "campaign_id",
            "scope",
            "models",
            "selection_model_seeds",
            "learning_rates",
            "optimizer_updates",
            "batch_size",
            "expected_runs",
            "selection_rule",
        },
        "selector",
    )
    if payload.get("schema_version") != 1:
        raise SelectorSpecError("selector schema_version must be 1")
    campaign_id = payload.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise SelectorSpecError("campaign_id must be a nonempty string")
    if payload.get("scope") != SCOPE:
        raise SelectorSpecError("selector scope permits training-only LR selection")

    raw_models = _array(payload.get("models"), "models")
    models: list[SelectorModel] = []
    for index, raw in enumerate(raw_models):
        if not isinstance(raw, Mapping):
            raise SelectorSpecError(f"models[{index}] must be an object")
        _require_exact_keys(
            raw, {"id", "hidden_width", "parameter_count"}, f"models[{index}]"
        )
        model_id = raw.get("id")
        if not isinstance(model_id, str):
            raise SelectorSpecError(f"models[{index}].id must be a string")
        width = raw.get("hidden_width")
        count = raw.get("parameter_count")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
        ):
            raise SelectorSpecError(f"models[{index}].hidden_width must be positive")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise SelectorSpecError(
                f"models[{index}].parameter_count must be positive"
            )
        models.append(SelectorModel(model_id, width, count))
    if tuple(model.model_id for model in models) != EXPECTED_MODEL_IDS:
        raise SelectorSpecError(
            f"model ids/order must be exactly {EXPECTED_MODEL_IDS!r}"
        )

    raw_seeds = _array(
        payload.get("selection_model_seeds"), "selection_model_seeds"
    )
    if len(raw_seeds) != EXPECTED_SELECTION_SEED_COUNT:
        raise SelectorSpecError("exactly five selection model seeds are required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds):
        raise SelectorSpecError("selection model seeds must be integers")
    seeds = tuple(int(seed) for seed in raw_seeds)
    if len(set(seeds)) != len(seeds):
        raise SelectorSpecError("selection model seeds must be unique")

    raw_rates = _array(payload.get("learning_rates"), "learning_rates")
    if any(isinstance(rate, bool) for rate in raw_rates):
        raise SelectorSpecError("learning rates must be numeric, not boolean")
    try:
        rates = tuple(float(rate) for rate in raw_rates)
    except (TypeError, ValueError) as exc:
        raise SelectorSpecError("learning rates must be finite numbers") from exc
    if rates != EXPECTED_LEARNING_RATES:
        raise SelectorSpecError(
            f"learning rates/order must be exactly {EXPECTED_LEARNING_RATES!r}"
        )

    updates = payload.get("optimizer_updates")
    batch_size = payload.get("batch_size")
    run_count = payload.get("expected_runs")
    if updates != EXPECTED_UPDATE:
        raise SelectorSpecError("optimizer_updates must be exactly 100")
    if batch_size != EXPECTED_BATCH_SIZE:
        raise SelectorSpecError("batch_size must be exactly 64")
    if run_count != EXPECTED_RUN_COUNT:
        raise SelectorSpecError("expected_runs must be exactly 120")
    rule = payload.get("selection_rule")
    if rule != EXPECTED_SELECTION_RULE:
        raise SelectorSpecError("selection_rule differs from the executable rule")
    return SelectorSpec(
        campaign_id=campaign_id,
        models=tuple(models),
        selection_model_seeds=seeds,
        learning_rates=rates,
        optimizer_updates=updates,
        batch_size=batch_size,
        expected_runs=run_count,
        selection_rule=dict(rule),
    )


def load_selector_spec(path: Path) -> SelectorSpec:
    payload = strict_json_load(Path(path))
    return parse_selector_spec(payload)


def _float_token(value: float) -> str:
    return format(float(value), ".10g").replace("-", "m").replace(".", "p")


def build_selection_plan(spec: SelectorSpec) -> tuple[SelectionRun, ...]:
    plan = tuple(
        SelectionRun(
            run_id=(
                f"lrsel_v3__{model.model_id}__w{model.hidden_width:03d}__"
                f"seed{seed}__lr{_float_token(rate)}"
            ),
            model_id=model.model_id,
            hidden_width=model.hidden_width,
            parameter_count=model.parameter_count,
            model_seed=seed,
            learning_rate=rate,
            required_update=spec.optimizer_updates,
            batch_size=spec.batch_size,
        )
        for model in spec.models
        for seed in spec.selection_model_seeds
        for rate in spec.learning_rates
    )
    if len(plan) != spec.expected_runs or len({run.run_id for run in plan}) != len(plan):
        raise SelectorSpecError("expanded selector matrix is not exactly 120 unique runs")
    return plan


def validate_protocol_binding(protocol: Mapping[str, Any], spec: SelectorSpec) -> None:
    """Require the executable training protocol to encode the same matrix."""

    phase = protocol.get("phase1_ring_pilot")
    seeds = protocol.get("seed_policy")
    phase0 = protocol.get("phase0_state_audit")
    if not isinstance(phase, Mapping) or not isinstance(seeds, Mapping):
        raise SelectorSpecError("protocol lacks phase1_ring_pilot/seed_policy")
    if not isinstance(phase0, Mapping):
        raise SelectorSpecError("protocol lacks phase0_state_audit")
    raw_models = phase.get("models")
    if not isinstance(raw_models, list):
        raise SelectorSpecError("protocol phase1 models must be an array")
    observed_models: list[tuple[str, int, int | None]] = []
    for raw in raw_models:
        if not isinstance(raw, Mapping):
            raise SelectorSpecError("protocol phase1 model entry is malformed")
        count = raw.get("parameter_count")
        observed_models.append(
            (
                str(raw.get("id")),
                int(raw.get("hidden_width", -1)),
                None if count is None else int(count),
            )
        )
    expected_models = [
        (model.model_id, model.hidden_width, model.parameter_count)
        for model in spec.models
    ]
    if observed_models != expected_models:
        raise SelectorSpecError(
            "protocol model ids/widths/parameter counts differ from selector JSON"
        )
    if tuple(phase0.get("models", ())) != EXPECTED_MODEL_IDS:
        raise SelectorSpecError("protocol Phase-0 models differ from selector models")
    if tuple(seeds.get("selection_model_seeds", ())) != spec.selection_model_seeds:
        raise SelectorSpecError("protocol selection seeds differ from selector JSON")
    training = phase.get("training")
    if not isinstance(training, Mapping):
        raise SelectorSpecError("protocol training block is malformed")
    learning_rate = training.get("learning_rate")
    if not isinstance(learning_rate, Mapping):
        raise SelectorSpecError("protocol learning-rate block is malformed")
    if tuple(float(value) for value in learning_rate.get("active_launch_values", ())) != spec.learning_rates:
        raise SelectorSpecError("protocol active learning rates differ from selector JSON")
    if int(training.get("optimizer_updates", -1)) != spec.optimizer_updates:
        raise SelectorSpecError("protocol optimizer updates differ from selector JSON")
    if int(training.get("batch_size", -1)) != spec.batch_size:
        raise SelectorSpecError("protocol batch size differs from selector JSON")
    run_matrix = phase.get("run_matrix")
    if not isinstance(run_matrix, Mapping) or int(
        run_matrix.get("expected_training_runs", -1)
    ) != spec.expected_runs:
        raise SelectorSpecError("protocol expected run count differs from selector JSON")
    rp = training.get("rp_schedule_for_ca_lru")
    if not isinstance(rp, Mapping) or rp.get("enabled_during_selector") is not False:
        raise SelectorSpecError("Retention Plasticity must be disabled during LR selection")


def aggregate_selection(
    rows: Sequence[Mapping[str, Any]], spec: SelectorSpec, *, strict: bool = False
) -> dict[str, Any]:
    """Aggregate declared rows without looking at validation/task metrics."""

    expected = {
        (model.model_id, seed, rate)
        for model in spec.models
        for seed in spec.selection_model_seeds
        for rate in spec.learning_rates
    }
    indexed: dict[tuple[str, int, float], Mapping[str, Any]] = {}
    for row in rows:
        try:
            key = (
                str(row["model_id"]),
                int(row["model_seed"]),
                float(row["learning_rate"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed LR-selection row") from exc
        if key not in expected:
            raise ValueError(f"unexpected LR-selection row: {key}")
        if key in indexed:
            raise ValueError(f"duplicate LR-selection row: {key}")
        indexed[key] = row

    candidates: dict[str, list[dict[str, Any]]] = {}
    winners: dict[str, dict[str, Any]] = {}
    ineligible: list[str] = []
    for model in spec.models:
        model_candidates: list[dict[str, Any]] = []
        for rate in spec.learning_rates:
            values: list[float] = []
            reasons: list[str] = []
            per_seed: list[dict[str, Any]] = []
            for seed in spec.selection_model_seeds:
                row = indexed.get((model.model_id, seed, rate))
                reason: str | None = None
                loss = float("nan")
                if row is None:
                    reason = "missing"
                else:
                    try:
                        loss = float(row.get("loss_at_required_update", float("nan")))
                        updates = int(row.get("completed_updates", -1))
                    except (TypeError, ValueError):
                        updates = -1
                    if row.get("status") != "complete":
                        reason = f"status={row.get('status', 'missing')}"
                    elif updates != spec.optimizer_updates:
                        reason = f"completed_updates={updates}"
                    elif not math.isfinite(loss):
                        reason = "loss_nonfinite_or_missing"
                if reason is None:
                    values.append(loss)
                    per_seed.append(
                        {"model_seed": seed, "eligible": True, "loss": loss}
                    )
                else:
                    reasons.append(f"seed{seed}:{reason}")
                    per_seed.append(
                        {"model_seed": seed, "eligible": False, "reason": reason}
                    )
            eligible = not reasons and len(values) == len(spec.selection_model_seeds)
            candidate = {
                "learning_rate": rate,
                "eligible": eligible,
                "mean_online_training_loss_at_update_100": (
                    math.fsum(values) / len(values) if eligible else None
                ),
                "ineligibility_reasons": reasons,
                "per_seed": per_seed,
            }
            model_candidates.append(candidate)
            if not eligible:
                ineligible.append(f"{model.model_id}@{rate:g}")
        candidates[model.model_id] = model_candidates
        eligible_candidates = [item for item in model_candidates if item["eligible"]]
        if eligible_candidates:
            winner = min(
                eligible_candidates,
                key=lambda item: (
                    float(item["mean_online_training_loss_at_update_100"]),
                    float(item["learning_rate"]),
                ),
            )
            winners[model.model_id] = {
                "learning_rate": float(winner["learning_rate"]),
                "mean_online_training_loss_at_update_100": float(
                    winner["mean_online_training_loss_at_update_100"]
                ),
                "selection_seed_count": len(spec.selection_model_seeds),
            }

    exact = (
        len(indexed) == spec.expected_runs
        and len(ineligible) == 0
        and len(winners) == len(spec.models)
    )
    result = {
        "schema_version": 1,
        "selection_rule": dict(spec.selection_rule),
        "expected_rows": spec.expected_runs,
        "observed_rows": len(indexed),
        "complete": exact,
        "ineligible_candidates": ineligible,
        "candidates": candidates,
        "winners": winners,
    }
    if strict and not exact:
        raise IncompleteSelectionError(
            "strict selection requires 120 unique successful update-100 rows; "
            f"observed={len(indexed)}, ineligible={ineligible}"
        )
    return result


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
    manifest: Mapping[str, Any], stage: str, run_id: str, gpu: int | None = None
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
        env["CALRU_PHYSICAL_GPU_ID"] = str(int(gpu))
    metadata = _receipt_metadata(manifest, stage, run_id)
    for key, variable in RECEIPT_IDENTITY_ENV.items():
        env[variable] = metadata[key]
    return env


def _child_preexec(expected_parent: int) -> Any:
    def configure() -> None:
        os.setsid()
        if not sys.platform.startswith("linux"):
            os._exit(125)
        libc = ctypes.CDLL(None, use_errno=True)
        if int(libc.prctl(1, signal.SIGKILL, 0, 0, 0)) != 0:  # PR_SET_PDEATHSIG
            os._exit(125)
        if os.getppid() != expected_parent:
            os.kill(os.getpid(), signal.SIGKILL)

    return configure


def _prepare_root(
    root: Path,
    selector_path: Path,
    protocol_path: Path,
    spec: SelectorSpec,
    protocol_fp: str,
    smoke: bool,
) -> None:
    marker_payload = {
        "schema_version": 1,
        "campaign_type": "six_model_lr_selection_v3",
        "campaign_id": spec.campaign_id,
        "scope": SCOPE,
        "selector_source_sha256": sha256_file(selector_path),
        "selector_canonical_fingerprint": canonical_hash(spec.payload()),
        "protocol_source_sha256": sha256_file(protocol_path),
        "protocol_canonical_fingerprint": protocol_fp,
        "smoke": bool(smoke),
    }
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ROOT_MARKER
    if marker.exists():
        if strict_json_load(marker) != marker_payload:
            raise RuntimeError("artifact-root marker differs; choose a new root")
    else:
        if any(root.iterdir()):
            raise RuntimeError("artifact root is nonempty and has no v3 marker")
        atomic_json(marker, marker_payload)
    for source, name in ((selector_path, SELECTOR_COPY), (protocol_path, PROTOCOL_COPY)):
        destination = root / name
        payload = source.read_bytes()
        if destination.exists():
            if destination.read_bytes() != payload:
                raise RuntimeError(f"frozen campaign copy changed: {name}")
        else:
            atomic_bytes(destination, payload)


def _build_manifest(
    *,
    root: Path,
    spec: SelectorSpec,
    protocol: Mapping[str, Any],
    plan: Sequence[SelectionRun],
    evaluation_bank: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    git_state: Mapping[str, Any],
    environment: Mapping[str, Any],
    python: str,
    gpus: Sequence[int],
    smoke: bool,
) -> dict[str, Any]:
    protocol_fp = protocol_fingerprint(dict(protocol))
    scientific_payload = {
        "campaign_id": spec.campaign_id,
        "campaign_type": "six_model_lr_selection_v3",
        "scope": SCOPE,
        "pilot_only": True,
        "confirmatory": False,
        "ca_evidence": False,
        "manifold_analysis_performed": False,
        "smoke": bool(smoke),
        "selector_file": SELECTOR_COPY,
        "selector_file_sha256": sha256_file(root / SELECTOR_COPY),
        "selector_canonical_fingerprint": canonical_hash(spec.payload()),
        "protocol_file": PROTOCOL_COPY,
        "protocol_file_sha256": sha256_file(root / PROTOCOL_COPY),
        "protocol_canonical_fingerprint": protocol_fp,
        "source_protocol": dict(protocol["source_protocol"]),
        "source_hashes": dict(source_hashes),
        "code": dict(git_state),
        "environment": dict(environment),
        "python": str(Path(python).resolve()),
        "gpus": [int(gpu) for gpu in gpus],
        "evaluation_bank": dict(evaluation_bank),
        "selection_rule": dict(spec.selection_rule),
        "run_matrix": [run.payload() for run in plan],
        "expected_training_runs": spec.expected_runs,
        "phase0_required": True,
    }
    return {
        "schema_version": 1,
        **scientific_payload,
        "scientific_identity_payload": scientific_payload,
        "scientific_identity": canonical_hash(scientific_payload),
    }


def _write_or_check(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if strict_json_load(path) != dict(payload):
            raise RuntimeError(f"immutable campaign file differs: {path.name}")
    else:
        atomic_json(path, dict(payload))


def _verify_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    signed = manifest.get("scientific_identity_payload")
    if not isinstance(signed, Mapping):
        raise RuntimeError("manifest scientific identity payload is malformed")
    if canonical_hash(signed) != manifest.get("scientific_identity"):
        raise RuntimeError("manifest scientific identity is invalid")
    for key, value in signed.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"manifest field {key!r} differs from signed payload")
    if strict_json_load(root / MANIFEST) != dict(manifest):
        raise RuntimeError("on-disk campaign manifest differs")
    if sha256_file(root / SELECTOR_COPY) != manifest["selector_file_sha256"]:
        raise RuntimeError("frozen selector copy changed")
    if sha256_file(root / PROTOCOL_COPY) != manifest["protocol_file_sha256"]:
        raise RuntimeError("frozen protocol copy changed")
    if protocol_fingerprint(load_protocol(root / PROTOCOL_COPY)) != manifest[
        "protocol_canonical_fingerprint"
    ]:
        raise RuntimeError("frozen protocol semantics changed")


def _verify_campaign_inputs(
    root: Path, manifest: Mapping[str, Any], repo_root: Path
) -> None:
    """Fail before every launch if any signed code or data input changed."""

    _verify_manifest(root, manifest)
    if _git_state(repo_root) != manifest["code"]:
        raise RuntimeError("git state changed after LR-selection manifest creation")
    observed_sources = _source_hashes(repo_root, Path(__file__).resolve().parent)
    if observed_sources != manifest["source_hashes"]:
        raise RuntimeError("campaign Python source hashes changed")
    source = manifest.get("source_protocol")
    if not isinstance(source, Mapping):
        raise RuntimeError("manifest source-protocol binding is malformed")
    source_path = repo_root / str(source.get("path", ""))
    if not source_path.is_file() or sha256_file(source_path) != source.get("sha256"):
        raise RuntimeError("normative source protocol note is missing or changed")
    bank = manifest.get("evaluation_bank")
    if not isinstance(bank, Mapping):
        raise RuntimeError("manifest evaluation-bank binding is malformed")
    bank_path = root / str(bank.get("path", ""))
    sidecar = Path(f"{bank_path}.sha256")
    if not bank_path.is_file() or sha256_file(bank_path) != bank.get("sha256"):
        raise RuntimeError("fixed evaluation bank is missing or changed")
    if not sidecar.is_file() or sha256_file(sidecar) != bank.get("sidecar_sha256"):
        raise RuntimeError("fixed evaluation-bank sidecar is missing or changed")


def _phase0_output_valid(
    root: Path, manifest: Mapping[str, Any], spec: SelectorSpec
) -> tuple[bool, str]:
    output = root / "phase0"
    valid, reason = verify_completion_receipt(
        output / COMPLETION_RECEIPT,
        expected_job_id="phase0_state_audit",
        expected_metadata=_receipt_metadata(manifest, "phase0", "phase0"),
    )
    if not valid:
        return False, reason
    try:
        gate = strict_json_load(output / "phase0_gate.json")
        phase_manifest = strict_json_load(output / "manifest.json")
    except (OSError, ValueError, TypeError) as exc:
        return False, f"Phase-0 output is unreadable: {exc}"
    if gate.get("passed") is not True:
        return False, "Phase-0 gate did not pass"
    models = gate.get("models")
    if not isinstance(models, Mapping) or set(models) != {
        model.model_id for model in spec.models
    }:
        # JSON objects are written canonically, so key order is deliberately
        # not used as scientific evidence here.  The selector order remains
        # frozen in its array and in the expanded run matrix.
        return False, "Phase-0 model set differs from selector"
    if any(not isinstance(item, Mapping) or item.get("passed") is not True for item in models.values()):
        return False, "at least one Phase-0 model did not pass"
    if phase_manifest.get("protocol_canonical_fingerprint") != manifest[
        "protocol_canonical_fingerprint"
    ]:
        return False, "Phase-0 protocol fingerprint mismatch"
    for model in spec.models:
        if not (output / f"model={model.model_id}" / "state_spec.json").is_file():
            return False, f"Phase-0 state spec missing for {model.model_id}"
    return True, "verified Phase-0 gate and receipt"


def _run_phase0(
    root: Path,
    manifest: Mapping[str, Any],
    spec: SelectorSpec,
    python: str,
    smoke: bool,
) -> None:
    valid, _ = _phase0_output_valid(root, manifest, spec)
    if valid:
        return
    output = root / "phase0"
    if output.exists():
        _preserve_invalid_output(root, "phase0", "phase0", output)
    attempt_parent = root / "attempts" / "phase0" / "phase0"
    valid_attempts: list[Path] = []
    if attempt_parent.is_dir():
        for candidate in sorted(attempt_parent.iterdir()):
            if not candidate.is_dir():
                continue
            # Verify in place by temporarily addressing the candidate as the
            # Phase-0 root.  The verifier otherwise has no root-relative state.
            original = root / "phase0"
            if original.exists():
                raise RuntimeError("unexpected Phase-0 output during recovery")
            os.replace(candidate, original)
            candidate_valid, _ = _phase0_output_valid(root, manifest, spec)
            os.replace(original, candidate)
            if candidate_valid:
                valid_attempts.append(candidate)
    if len(valid_attempts) > 1:
        raise RuntimeError("multiple valid unpublished Phase-0 attempts require audit")
    if valid_attempts:
        os.replace(valid_attempts[0], output)
        return
    attempt = _unique_attempt_path(root, "phase0", "phase0")
    log_dir = root / "logs" / "phase0"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{attempt.name}.log"
    command = [
        python,
        "-m",
        "repro.sagodi_protocol.phase0",
        "--protocol",
        str(root / PROTOCOL_COPY),
        "--output-root",
        str(attempt),
        "--device",
        "cpu",
    ]
    if smoke:
        command.append("--smoke")
    with log_path.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=_child_environment(manifest, "phase0", "phase0"),
            preexec_fn=_child_preexec(os.getpid()),
        )
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Phase-0 exited {return_code}; see {log_path}")
    os.replace(attempt, output)
    valid, reason = _phase0_output_valid(root, manifest, spec)
    if not valid:
        preserved = _preserve_invalid_output(root, "phase0", "phase0", output)
        raise RuntimeError(f"invalid Phase-0 output preserved at {preserved}: {reason}")


def _training_output(root: Path, run: SelectionRun) -> Path:
    return (
        root
        / "training"
        / f"model={run.model_id}"
        / f"seed={run.model_seed}"
        / f"lr={_float_token(run.learning_rate)}"
    )


def _recover_training_attempt(
    root: Path,
    run: SelectionRun,
    manifest: Mapping[str, Any],
) -> tuple[bool, str]:
    """Publish one verified orphaned attempt, if present, without retraining."""

    parent = root / "attempts" / "lr_selection_training" / run.run_id
    if not parent.is_dir():
        return False, "no unpublished attempt"
    valid_attempts: list[Path] = []
    for candidate in sorted(parent.iterdir()):
        if not candidate.is_dir():
            continue
        valid, _, _ = _verify_training_output(
            candidate, run, manifest, root=root
        )
        if valid:
            valid_attempts.append(candidate)
    if len(valid_attempts) > 1:
        raise RuntimeError(
            f"multiple valid unpublished attempts for {run.run_id}; manual audit required"
        )
    if not valid_attempts:
        return False, "no verified unpublished attempt"
    output = _training_output(root, run)
    if output.exists():
        raise RuntimeError(f"cannot recover {run.run_id} over an existing output")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(valid_attempts[0], output)
    valid, reason, _ = _verify_training_output(output, run, manifest, root=root)
    if not valid:
        raise RuntimeError(f"recovered output failed verification: {reason}")
    return True, f"recovered verified attempt {valid_attempts[0]}"


def _verify_training_output(
    output: Path,
    run: SelectionRun,
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[bool, str, dict[str, str] | None]:
    valid, reason = verify_completion_receipt(
        output / COMPLETION_RECEIPT,
        expected_job_id=run.receipt_job_id,
        expected_metadata=_receipt_metadata(manifest, "training", run.run_id),
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
        child_manifest = strict_json_load(output / "manifest.json")
        metrics = strict_json_load(output / "task_metrics.json")
        rp_trace = strict_json_load(output / "rp_trace.json")
        with np.load(output / "training_trace.npz", allow_pickle=False) as archive:
            steps = np.asarray(archive["step"], dtype=np.int64)
            losses = np.asarray(archive["masked_mse"], dtype=np.float64)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, f"training output is unreadable: {exc}", None
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or not required.issubset(artifacts):
        return False, "training receipt lacks required artifacts", None
    train_spec = config.get("train_spec")
    training = config.get("training")
    model_metadata = config.get("model")
    if not all(isinstance(item, Mapping) for item in (train_spec, training, model_metadata)):
        return False, "training config is malformed", None
    if train_spec.get("model_name") != run.model_id:
        return False, "training model id mismatch", None
    if int(train_spec.get("model_seed", -1)) != run.model_seed:
        return False, "training model seed mismatch", None
    try:
        configured_lr = float(train_spec.get("learning_rate"))
    except (TypeError, ValueError):
        return False, "training learning rate is malformed", None
    if configured_lr != run.learning_rate:
        return False, "training learning rate mismatch", None
    expected_steps = 2 if manifest["smoke"] else run.required_update
    expected_batch = min(4, run.batch_size) if manifest["smoke"] else run.batch_size
    if int(training.get("steps", -1)) != expected_steps:
        return False, "training update count mismatch", None
    if int(training.get("batch_size", -1)) != expected_batch:
        return False, "training batch size mismatch", None
    if training.get("rp_enabled_by_protocol") is not False:
        return False, "Retention Plasticity was enabled during LR selection", None
    if training.get("expected_rp_steps") != [] or rp_trace != []:
        return False, "selector contains forbidden RP calls", None
    model_config = model_metadata.get("model_config")
    if not isinstance(model_config, Mapping):
        return False, "training model metadata lacks model_config", None
    if int(model_config.get("width", -1)) != run.hidden_width:
        return False, "training hidden width mismatch", None
    if int(model_metadata.get("parameters_total", -1)) != run.parameter_count:
        return False, "training parameter count differs from selector declaration", None
    if child_manifest.get("campaign_identity") != manifest["scientific_identity"]:
        return False, "training campaign identity mismatch", None
    if child_manifest.get("protocol_canonical_fingerprint") != manifest[
        "protocol_canonical_fingerprint"
    ]:
        return False, "training protocol fingerprint mismatch", None
    if child_manifest.get("model_id") != run.model_id or int(
        child_manifest.get("model_seed", -1)
    ) != run.model_seed:
        return False, "training manifest model identity mismatch", None
    if int(child_manifest.get("parameter_count", -1)) != run.parameter_count:
        return False, "training manifest parameter count mismatch", None
    if child_manifest.get("evaluation_bank_sha256") != manifest["evaluation_bank"][
        "sha256"
    ]:
        return False, "training evaluation-bank binding mismatch", None
    if child_manifest.get("rp_schedule") != {
        "expected_steps": [],
        "actual_steps": [],
        "calls": 0,
    }:
        return False, "training manifest contains forbidden RP schedule", None
    expected_state = None
    if not manifest["smoke"]:
        state_path = root / "phase0" / f"model={run.model_id}" / "state_spec.json"
        expected_state = sha256_file(state_path)
    if child_manifest.get("state_spec_sha256") != expected_state:
        return False, "training Phase-0 state binding mismatch", None
    if steps.ndim != 1 or not np.array_equal(
        steps, np.arange(1, expected_steps + 1, dtype=np.int64)
    ):
        return False, "training steps are not the exact sequence", None
    if losses.shape != steps.shape or not np.isfinite(losses).all():
        return False, "training losses are malformed or non-finite", None
    try:
        recorded_loss = float(metrics["train_loss_last"])
    except (KeyError, TypeError, ValueError):
        return False, "task metrics lack the final online training loss", None
    if float(np.float32(recorded_loss)) != float(losses[-1]):
        return False, "training trace and task-metric loss differ", None
    bindings = {
        "completion_receipt_sha256": sha256_file(output / COMPLETION_RECEIPT),
        "checkpoint_sha256": sha256_file(output / "checkpoint.pt"),
        "training_trace_sha256": sha256_file(output / "training_trace.npz"),
        "task_metrics_sha256": sha256_file(output / "task_metrics.json"),
        "manifest_sha256": sha256_file(output / "manifest.json"),
    }
    return True, "verified training output", bindings


def _training_command(
    root: Path,
    attempt: Path,
    run: SelectionRun,
    manifest: Mapping[str, Any],
    python: str,
) -> list[str]:
    command = [
        python,
        "-m",
        "repro.sagodi_protocol.train",
        "--protocol",
        str(root / PROTOCOL_COPY),
        "--model",
        run.model_id,
        "--model-seed",
        str(run.model_seed),
        "--learning-rate",
        str(run.learning_rate),
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
                str(root / "phase0" / f"model={run.model_id}" / "state_spec.json"),
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
    plan: Sequence[SelectionRun],
    python: str,
    gpus: Sequence[int],
    status: dict[str, Any],
    repo_root: Path,
) -> None:
    pending: list[SelectionRun] = []
    jobs = status.setdefault("jobs", {})
    for run in plan:
        output = _training_output(root, run)
        valid, reason, _ = _verify_training_output(
            output, run, manifest, root=root
        )
        if valid:
            jobs[run.run_id] = {"state": "complete", "reason": reason}
            continue
        if output.exists():
            preserved = _preserve_invalid_output(
                root, "lr_selection_training", run.run_id, output
            )
            reason = f"invalid output preserved at {preserved}: {reason}"
        recovered, recovery_reason = _recover_training_attempt(
            root, run, manifest
        )
        if recovered:
            jobs[run.run_id] = {
                "state": "complete",
                "reason": recovery_reason,
            }
            continue
        jobs[run.run_id] = {"state": "pending", "reason": reason}
        pending.append(run)
    atomic_json(root / STATUS, status)

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
                log_path = log_dir / f"{attempt.name}.log"
                handle = log_path.open("ab", buffering=0)
                process = subprocess.Popen(
                    _training_command(root, attempt, run, manifest, python),
                    cwd=Path(__file__).resolve().parents[2],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=_child_environment(manifest, "training", run.run_id, gpu),
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
                process = item["process"]
                return_code = process.poll()
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
                        "state": "retryable_failure",
                        "return_code": return_code,
                        "attempt_dir": str(attempt),
                        "log": str(item["log"]),
                    }
                    atomic_json(root / STATUS, status)
                    raise RuntimeError(
                        f"training {run.run_id} exited {return_code}; resume after "
                        f"inspection, log={item['log']}"
                    )
                valid, reason, _ = _verify_training_output(
                    attempt, run, manifest, root=root
                )
                if not valid:
                    jobs[run.run_id] = {
                        "state": "retryable_invalid_output",
                        "reason": reason,
                        "attempt_dir": str(attempt),
                    }
                    atomic_json(root / STATUS, status)
                    raise RuntimeError(
                        f"training {run.run_id} produced invalid output: {reason}"
                    )
                output = _training_output(root, run)
                output.parent.mkdir(parents=True, exist_ok=True)
                os.replace(attempt, output)
                valid, reason, _ = _verify_training_output(
                    output, run, manifest, root=root
                )
                if not valid:
                    raise RuntimeError(f"published training output became invalid: {reason}")
                jobs[run.run_id] = {"state": "complete", "reason": reason}
                atomic_json(root / STATUS, status)
            if not progressed and active:
                time.sleep(0.2)
    except BaseException:
        _terminate_active(active)
        raise


def _row_from_output(
    root: Path, run: SelectionRun, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    output = _training_output(root, run)
    valid, reason, bindings = _verify_training_output(
        output, run, manifest, root=root
    )
    if not valid or bindings is None:
        raise IncompleteSelectionError(f"{run.run_id}: {reason}")
    with np.load(output / "training_trace.npz", allow_pickle=False) as archive:
        steps = np.asarray(archive["step"], dtype=np.int64)
        losses = np.asarray(archive["masked_mse"], dtype=np.float64)
    completed = int(steps[-1])
    loss = (
        float(losses[run.required_update - 1])
        if completed >= run.required_update
        else None
    )
    return {
        **run.payload(),
        "status": "complete",
        "completed_updates": completed,
        "loss_at_required_update": loss,
        "artifact_bindings": bindings,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = (
        "run_id",
        "model_id",
        "hidden_width",
        "parameter_count",
        "model_seed",
        "learning_rate",
        "required_update",
        "completed_updates",
        "loss_at_required_update",
        "status",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return stream.getvalue().encode("utf-8")


def _nested_receipts(root: Path, plan: Sequence[SelectionRun]) -> list[Path]:
    return [
        root / "phase0" / COMPLETION_RECEIPT,
        *[_training_output(root, run) / COMPLETION_RECEIPT for run in plan],
    ]


def _preserve_finalization(root: Path) -> None:
    paths = [
        root / name
        for name in (SUMMARY, MATRIX, SELECTION_RECEIPT, COMPLETE, COMPLETION_RECEIPT)
        if (root / name).exists()
    ]
    if not paths:
        return
    destination = _unique_attempt_path(
        root, "finalization", "lr_selection_v3", label="recovered"
    )
    for path in paths:
        os.replace(path, destination / path.name)


def _finalize(
    root: Path,
    manifest: Mapping[str, Any],
    spec: SelectorSpec,
    plan: Sequence[SelectionRun],
) -> dict[str, Any]:
    _preserve_finalization(root)
    rows = [_row_from_output(root, run, manifest) for run in plan]
    if manifest["smoke"]:
        aggregation = aggregate_selection(rows, spec, strict=False)
        winners: dict[str, Any] = {}
        selection_performed = False
    else:
        aggregation = aggregate_selection(rows, spec, strict=True)
        winners = aggregation["winners"]
        selection_performed = True
    summary = {
        "schema_version": 1,
        "campaign_id": spec.campaign_id,
        "campaign_scientific_identity": manifest["scientific_identity"],
        "scope": SCOPE,
        "pilot_only": True,
        "confirmatory": False,
        "ca_evidence": False,
        "manifold_analysis_performed": False,
        "smoke": bool(manifest["smoke"]),
        "selection_performed": selection_performed,
        "selection_rule": dict(spec.selection_rule),
        "run_count": len(rows),
        "verified_success_receipt_count": len(rows),
        "aggregation": aggregation,
        "winners": winners,
        "training_artifact_bindings": {
            row["run_id"]: row["artifact_bindings"] for row in rows
        },
        "validation_and_task_metrics_recorded_but_not_used": True,
    }
    atomic_json(root / SUMMARY, summary)
    atomic_bytes(root / MATRIX, _csv_bytes(rows))
    metadata = {
        **_receipt_metadata(manifest, "finalization", "lr_selection_v3"),
        "run_count": len(rows),
        "selection_performed": selection_performed,
    }
    write_completion_receipt(
        root / SELECTION_RECEIPT,
        job_id="lr_selection_v3_aggregate",
        artifacts=[
            root / MANIFEST,
            root / SELECTOR_COPY,
            root / PROTOCOL_COPY,
            root / SUMMARY,
            root / MATRIX,
            *_nested_receipts(root, plan),
        ],
        metadata=metadata,
    )
    atomic_json(
        root / COMPLETE,
        {
            "schema_version": 1,
            "status": "complete",
            "campaign_scientific_identity": manifest["scientific_identity"],
            "scope": SCOPE,
            "run_count": len(rows),
            "verified_success_receipt_count": len(rows),
            "selection_performed": selection_performed,
            "selection_receipt_sha256": sha256_file(root / SELECTION_RECEIPT),
            "winners": winners,
        },
    )
    write_completion_receipt(
        root / COMPLETION_RECEIPT,
        job_id="lr_selection_v3_campaign_complete",
        artifacts=[
            root / MANIFEST,
            root / SUMMARY,
            root / MATRIX,
            root / SELECTION_RECEIPT,
            root / COMPLETE,
        ],
        metadata=metadata,
    )
    valid, reason = verify_selection_completion(root, manifest, spec, plan)
    if not valid:
        raise RuntimeError(f"final selector receipt verification failed: {reason}")
    return summary


def verify_selection_completion(
    root: Path,
    manifest: Mapping[str, Any],
    spec: SelectorSpec,
    plan: Sequence[SelectionRun],
) -> tuple[bool, str]:
    try:
        _verify_manifest(root, manifest)
        summary = strict_json_load(root / SUMMARY)
        complete = strict_json_load(root / COMPLETE)
        rows = [_row_from_output(root, run, manifest) for run in plan]
        aggregation = (
            aggregate_selection(rows, spec, strict=False)
            if manifest["smoke"]
            else aggregate_selection(rows, spec, strict=True)
        )
        metadata = {
            **_receipt_metadata(manifest, "finalization", "lr_selection_v3"),
            "run_count": len(rows),
            "selection_performed": not bool(manifest["smoke"]),
        }
    except (OSError, ValueError, TypeError, KeyError, IncompleteSelectionError) as exc:
        return False, f"recursive completion verification failed: {exc}"
    valid, reason = verify_completion_receipt(
        root / SELECTION_RECEIPT,
        expected_job_id="lr_selection_v3_aggregate",
        expected_metadata=metadata,
    )
    if not valid:
        return False, f"invalid selection receipt: {reason}"
    valid, reason = verify_completion_receipt(
        root / COMPLETION_RECEIPT,
        expected_job_id="lr_selection_v3_campaign_complete",
        expected_metadata=metadata,
    )
    if not valid:
        return False, f"invalid campaign receipt: {reason}"
    expected_winners = {} if manifest["smoke"] else aggregation["winners"]
    if summary.get("run_count") != spec.expected_runs:
        return False, "summary run count mismatch"
    if summary.get("aggregation") != aggregation:
        return False, "summary aggregation mismatch"
    if summary.get("winners") != expected_winners:
        return False, "summary winners mismatch"
    if complete.get("selection_receipt_sha256") != sha256_file(
        root / SELECTION_RECEIPT
    ):
        return False, "COMPLETE does not bind the selection receipt"
    if complete.get("winners") != expected_winners:
        return False, "COMPLETE winners mismatch"
    return True, "verified all 120 nested selector receipts"


def run_lr_selection_campaign(
    *,
    selector_path: Path,
    protocol_path: Path,
    artifact_root: Path,
    python: str,
    gpus: Sequence[int],
    smoke: bool = False,
) -> Path:
    selector_path = Path(selector_path).resolve(strict=True)
    protocol_path = Path(protocol_path).resolve(strict=True)
    root = Path(artifact_root).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    python = _resolve_python(python)
    gpus = _validated_gpu_ids(gpus)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES; --gpus uses physical GPU ids")
    spec = load_selector_spec(selector_path)
    protocol = load_protocol(protocol_path)
    validate_protocol_binding(protocol, spec)
    plan = build_selection_plan(spec)
    git_state = _git_state(repo_root)
    if not smoke and git_state["worktree_dirty"]:
        raise RuntimeError("full LR selection requires a clean committed worktree")
    protocol_fp = protocol_fingerprint(protocol)
    _prepare_root(
        root, selector_path, protocol_path, spec, protocol_fp, bool(smoke)
    )
    lock = _acquire_campaign_lock(root)
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"LR-selection campaign interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        # All executable stages use the immutable protocol copy below the
        # campaign root, never an ambient file that can change mid-run.
        protocol = load_protocol(root / PROTOCOL_COPY)
        evaluation_bank = _materialize_evaluation_bank(root, protocol)
        manifest = _build_manifest(
            root=root,
            spec=spec,
            protocol=protocol,
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
        _verify_campaign_inputs(root, manifest, repo_root)
        complete, _ = verify_selection_completion(root, manifest, spec, plan)
        if complete:
            return root
        status: dict[str, Any] = {
            "schema_version": 1,
            "campaign_id": spec.campaign_id,
            "scientific_identity": manifest["scientific_identity"],
            "scope": SCOPE,
            "smoke": bool(smoke),
            "stage": "phase0",
            "jobs": {},
        }
        if (root / STATUS).exists():
            existing = strict_json_load(root / STATUS)
            if isinstance(existing, Mapping):
                status["jobs"] = dict(existing.get("jobs", {}))
        atomic_json(root / STATUS, status)
        _run_phase0(root, manifest, spec, python, smoke)
        status["stage"] = "training_only_lr_selection"
        atomic_json(root / STATUS, status)
        _run_training_jobs(
            root, manifest, plan, python, gpus, status, repo_root
        )
        _verify_campaign_inputs(root, manifest, repo_root)
        status["stage"] = "finalizing"
        atomic_json(root / STATUS, status)
        summary = _finalize(root, manifest, spec, plan)
        status["stage"] = "complete"
        status["selection_performed"] = summary["selection_performed"]
        status["winners"] = summary["winners"]
        status["completed_at"] = time.time()
        atomic_json(root / STATUS, status)
        return root
    finally:
        lock.release()
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = run_lr_selection_campaign(
        selector_path=args.selector,
        protocol_path=args.protocol,
        artifact_root=args.artifact_root,
        python=args.python,
        gpus=tuple(int(value) for value in args.gpus.split(",") if value.strip()),
        smoke=args.smoke,
    )
    print(
        json.dumps(
            {"status": "complete", "artifact_root": str(output), "scope": SCOPE},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
