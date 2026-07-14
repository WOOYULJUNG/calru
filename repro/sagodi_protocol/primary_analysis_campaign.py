"""Run the receipt-bound Ságodi-based primary analysis over all 60 checkpoints.

The campaign is intentionally descriptive.  A trained seed may finish in one
of three terminal states: eligible and structurally estimable, ineligible by
the registered task-NMSE rule, or structurally not estimable.  The latter two
states are retained in the denominator and never replaced by another seed.

Only the Ságodi-based primary runner is launched here.  Its v3.1 scope includes
one explicitly project-defined extension: finite carrier ambient-normal
recovery without a threshold or binary gate.  Legacy C1--C4 recovery,
settling, and projected-JVP diagnostics remain in the separate supplementary
track and are neither scheduled nor aggregated by this module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import (
    RECEIPT_IDENTITY_ENV,
    atomic_json,
    canonical_hash,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from .config import load_protocol, protocol_fingerprint
from .lr_selection_v3 import EXPECTED_MODEL_IDS
from .orchestrate import (
    _acquire_campaign_lock,
    _environment_fingerprint,
    _git_state,
    _preserve_invalid_output,
    _resolve_python,
    _source_hashes,
    _unique_attempt_path,
    _validated_gpu_ids,
)
from .primary_main_campaign import (
    COMPLETE as MAIN_COMPLETE,
    COMPLETION_RECEIPT as MAIN_COMPLETION_RECEIPT,
    MANIFEST as MAIN_MANIFEST,
    RESOLVED_PROTOCOL as MAIN_RESOLVED_PROTOCOL,
    SUMMARY as MAIN_SUMMARY,
    _child_preexec,
)
from .sagodi_primary_runner import (
    PrimaryAnalysisSpec,
    primary_analysis_identity_payload,
    primary_analysis_spec_payload,
)


CAMPAIGN_MODE = "sagodi_primary_analysis_v3"
PROTOCOL_REVISION = "sagodi_primary_v3_1"
SCOPE = (
    "sagodi_based_primary_plus_project_defined_carrier_ambient_normal_"
    "recovery_no_binary_ca_lru_gates"
)
EXPECTED_RUN_COUNT = 60
EXPECTED_MODEL_SEEDS = tuple(range(10))
TERMINAL_ANALYSIS_STATUSES = (
    "complete_structural_summary_eligible",
    "ineligible_for_structural_summary",
    "structural_analysis_not_estimable",
)

ROOT_MARKER = ".calru_sagodi_primary_analysis_v3_root.json"
MANIFEST = "primary_analysis_manifest.json"
STATUS = "status.json"
SUMMARY = "primary_analysis_summary.json"
COMPLETE = "COMPLETE"
COMPLETION_RECEIPT = "completion_receipt.json"

NORMAL_RECOVERY_ARTIFACT = "carrier_ambient_normal_recovery.npz"
NORMAL_RECOVERY_SEED = 314159
NORMAL_RECOVERY_FULL_ANCHOR_COUNT = 32
NORMAL_RECOVERY_DIRECTIONS = {"ambient_normal": 4, "in_plane_radial": 2}
NORMAL_RECOVERY_RADII = (0.01, 0.05, 0.1)
NORMAL_RECOVERY_FULL_HORIZONS = (0, 1, 4, 16, 64, 256, 1024, 4096)
NORMAL_RECOVERY_DIRECTION_QA_MAX = 1.0e-4
NORMAL_RECOVERY_METRICS = (
    "manifold_distance",
    "manifold_distance_ratio",
    "same_memory_error_radians",
    "clean_manifold_distance",
    "clean_same_memory_error_radians",
    "excess_same_memory_error_radians",
    "distance_to_matched_clean_state",
    "distance_to_matched_clean_state_ratio",
    "manifold_distance_minus_clean",
)
NORMAL_RECOVERY_STATISTICS = (
    "mean",
    "population_std",
    "median",
    "q05",
    "q95",
    "min",
    "max",
)


@dataclass(frozen=True)
class AnalysisRun:
    model_id: str
    model_seed: int
    learning_rate: float
    hidden_width: int
    parameter_count: int
    training_run_id: str
    training_output: Path
    checkpoint_sha256: str
    training_receipt_sha256: str
    training_outcome: dict[str, Any]

    @property
    def run_id(self) -> str:
        return f"sagodi_primary__{self.model_id}__seed{self.model_seed:02d}"

    def payload(self, main_root: Path) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "model_seed": self.model_seed,
            "learning_rate": self.learning_rate,
            "hidden_width": self.hidden_width,
            "parameter_count": self.parameter_count,
            "training_run_id": self.training_run_id,
            "training_output_relative_to_main_root": self.training_output.relative_to(
                main_root
            ).as_posix(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "training_receipt_sha256": self.training_receipt_sha256,
        }


@dataclass(frozen=True)
class VerifiedMain:
    root: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    protocol: dict[str, Any]
    runs: tuple[AnalysisRun, ...]
    binding: dict[str, Any]


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _main_training_output(root: Path, model_id: str, model_seed: int) -> Path:
    return root / "training" / f"model={model_id}" / f"seed={model_seed}"


def _expected_main_receipt_metadata(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "campaign_scientific_identity": manifest["scientific_identity"],
        "protocol_fingerprint": manifest["protocol_canonical_fingerprint"],
        "run_id": "primary_main_v3",
        "stage": "finalization",
        "run_count": EXPECTED_RUN_COUNT,
        "ca_evidence": False,
        "manifold_analysis_performed": False,
    }


def _expected_training_metadata(
    manifest: Mapping[str, Any], run_id: str
) -> dict[str, Any]:
    return {
        "campaign_scientific_identity": manifest["scientific_identity"],
        "protocol_fingerprint": manifest["protocol_canonical_fingerprint"],
        "run_id": run_id,
        "stage": "primary_main_training",
    }


def verify_main_artifacts(main_root: Path | str) -> VerifiedMain:
    """Recursively verify the completed 60-run parent training campaign."""

    root = Path(main_root).expanduser().resolve(strict=True)
    try:
        manifest = dict(_require_mapping(strict_json_load(root / MAIN_MANIFEST), "main manifest"))
        summary = dict(_require_mapping(strict_json_load(root / MAIN_SUMMARY), "main summary"))
        complete = _require_mapping(strict_json_load(root / MAIN_COMPLETE), "main COMPLETE")
        signed = _require_mapping(
            manifest.get("scientific_identity_payload"),
            "main scientific_identity_payload",
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError(f"main campaign is unreadable: {error}") from error
    if canonical_hash(signed) != manifest.get("scientific_identity"):
        raise RuntimeError("main scientific identity does not verify")
    if manifest.get("campaign_type") != "sagodi_primary_main_v3":
        raise RuntimeError("main campaign type differs from sagodi_primary_main_v3")
    if manifest.get("expected_training_runs") != EXPECTED_RUN_COUNT:
        raise RuntimeError("main manifest does not declare exactly 60 training runs")
    code = _require_mapping(manifest.get("code"), "main code binding")
    main_code_commit = code.get("code_commit")
    if (
        not isinstance(main_code_commit, str)
        or len(main_code_commit) != 40
        or any(character not in "0123456789abcdef" for character in main_code_commit)
    ):
        raise RuntimeError("main manifest does not bind a valid clean Git commit")
    if manifest.get("smoke") is False and code.get("worktree_dirty") is not False:
        raise RuntimeError("full main campaign was not produced from a clean worktree")
    if summary.get("verified_training_receipt_count") != EXPECTED_RUN_COUNT:
        raise RuntimeError("main summary does not bind exactly 60 training receipts")
    if complete.get("verified_training_receipt_count") != EXPECTED_RUN_COUNT:
        raise RuntimeError("main COMPLETE does not bind exactly 60 training receipts")
    if complete.get("failed_seed_replacements") != 0:
        raise RuntimeError("main campaign contains forbidden seed replacement")
    if complete.get("summary_sha256") != sha256_file(root / MAIN_SUMMARY):
        raise RuntimeError("main COMPLETE summary binding changed")

    valid, reason = verify_completion_receipt(
        root / MAIN_COMPLETION_RECEIPT,
        expected_job_id="sagodi_primary_main_v3_campaign_complete",
        expected_metadata=_expected_main_receipt_metadata(manifest),
    )
    if not valid:
        raise RuntimeError(f"main recursive completion receipt failed: {reason}")

    protocol_name = manifest.get("resolved_protocol_file")
    if protocol_name != MAIN_RESOLVED_PROTOCOL:
        raise RuntimeError("main resolved protocol filename differs")
    protocol_path = root / protocol_name
    if sha256_file(protocol_path) != manifest.get("resolved_protocol_file_sha256"):
        raise RuntimeError("main resolved protocol bytes changed")
    protocol = load_protocol(protocol_path)
    if protocol_fingerprint(protocol) != manifest.get("protocol_canonical_fingerprint"):
        raise RuntimeError("main resolved protocol semantics changed")

    raw_plan = manifest.get("run_matrix")
    raw_rows = summary.get("runs")
    if not isinstance(raw_plan, list) or not isinstance(raw_rows, list):
        raise RuntimeError("main run matrix/summary rows are malformed")
    if len(raw_plan) != EXPECTED_RUN_COUNT or len(raw_rows) != EXPECTED_RUN_COUNT:
        raise RuntimeError("main run matrix is not exactly 60 rows")
    summary_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in raw_rows:
        item = _require_mapping(row, "main summary run")
        key = (str(item.get("model_id")), int(item.get("model_seed", -1)))
        if key in summary_by_key:
            raise RuntimeError(f"duplicate main summary row: {key}")
        summary_by_key[key] = item

    observed_order: list[str] = []
    runs: list[AnalysisRun] = []
    for item in raw_plan:
        planned = _require_mapping(item, "main planned run")
        model_id = str(planned.get("model_id"))
        model_seed = int(planned.get("model_seed", -1))
        if model_id not in observed_order:
            observed_order.append(model_id)
        key = (model_id, model_seed)
        row = summary_by_key.get(key)
        if row is None:
            raise RuntimeError(f"main summary lacks planned run {key}")
        if row.get("status") != "complete":
            raise RuntimeError(f"main training outcome is not complete: {key}")
        for field in ("learning_rate", "hidden_width", "parameter_count", "run_id"):
            if row.get(field) != planned.get(field):
                raise RuntimeError(f"main plan/summary mismatch for {key}: {field}")
        output = _main_training_output(root, model_id, model_seed)
        learning_rate = float(planned["learning_rate"])
        run_id = str(planned["run_id"])
        expected_job_id = f"{model_id}-seed{model_seed}-lr{learning_rate:g}"
        valid, reason = verify_completion_receipt(
            output / "completion_receipt.json",
            expected_job_id=expected_job_id,
            expected_metadata=_expected_training_metadata(manifest, run_id),
        )
        if not valid:
            raise RuntimeError(f"invalid nested main run {key}: {reason}")
        required = {
            "config.json",
            "checkpoint.pt",
            "training_trace.npz",
            "task_metrics.json",
            "rp_trace.json",
            "manifest.json",
        }
        receipt = _require_mapping(
            strict_json_load(output / "completion_receipt.json"),
            f"training receipt {key}",
        )
        artifacts = _require_mapping(receipt.get("artifacts"), f"training artifacts {key}")
        if not required.issubset(artifacts):
            raise RuntimeError(f"nested training receipt lacks required artifacts: {key}")
        child = _require_mapping(
            strict_json_load(output / "manifest.json"), f"training manifest {key}"
        )
        config = _require_mapping(
            strict_json_load(output / "config.json"), f"training config {key}"
        )
        task_metrics = _require_mapping(
            strict_json_load(output / "task_metrics.json"), f"training metrics {key}"
        )
        train_spec = _require_mapping(config.get("train_spec"), f"train spec {key}")
        if child.get("campaign_identity") != manifest["scientific_identity"]:
            raise RuntimeError(f"nested training campaign identity changed: {key}")
        if child.get("protocol_canonical_fingerprint") != manifest[
            "protocol_canonical_fingerprint"
        ]:
            raise RuntimeError(f"nested training protocol binding changed: {key}")
        if child.get("model_id") != model_id or int(child.get("model_seed", -1)) != model_seed:
            raise RuntimeError(f"nested training model identity changed: {key}")
        if train_spec.get("model_name") != model_id or int(
            train_spec.get("model_seed", -1)
        ) != model_seed:
            raise RuntimeError(f"nested training config identity changed: {key}")
        validation_nmse = _validate_finite_number(
            task_metrics.get("masked_nmse_db"), f"training validation NMSE {key}"
        )
        if row.get("validation_masked_nmse_db") != validation_nmse:
            raise RuntimeError(f"main summary validation NMSE binding changed: {key}")
        if row.get("structural_summary_eligible_nmse_lt_minus20db") is not (
            validation_nmse < -20.0
        ):
            raise RuntimeError(f"main summary eligibility binding changed: {key}")
        checkpoint_hash = sha256_file(output / "checkpoint.pt")
        bindings = _require_mapping(row.get("artifact_bindings"), f"artifact bindings {key}")
        if bindings.get("checkpoint_sha256") != checkpoint_hash:
            raise RuntimeError(f"main summary checkpoint binding changed: {key}")
        training_receipt_hash = sha256_file(output / "completion_receipt.json")
        if bindings.get("completion_receipt_sha256") != training_receipt_hash:
            raise RuntimeError(f"main summary training-receipt binding changed: {key}")
        runs.append(
            AnalysisRun(
                model_id=model_id,
                model_seed=model_seed,
                learning_rate=learning_rate,
                hidden_width=int(planned["hidden_width"]),
                parameter_count=int(planned["parameter_count"]),
                training_run_id=run_id,
                training_output=output,
                checkpoint_sha256=checkpoint_hash,
                training_receipt_sha256=training_receipt_hash,
                training_outcome=dict(row),
            )
        )
    if tuple(observed_order) != EXPECTED_MODEL_IDS:
        raise RuntimeError("main model order differs from the frozen six-model comparison")
    expected_keys = {
        (model_id, seed)
        for model_id in EXPECTED_MODEL_IDS
        for seed in EXPECTED_MODEL_SEEDS
    }
    if {(run.model_id, run.model_seed) for run in runs} != expected_keys:
        raise RuntimeError("main run matrix is not the exact six-model by ten-seed product")

    binding = {
        "schema_version": 1,
        "campaign_scientific_identity": manifest["scientific_identity"],
        "protocol_canonical_fingerprint": manifest[
            "protocol_canonical_fingerprint"
        ],
        "manifest_sha256": sha256_file(root / MAIN_MANIFEST),
        "summary_sha256": sha256_file(root / MAIN_SUMMARY),
        "complete_sha256": sha256_file(root / MAIN_COMPLETE),
        "completion_receipt_sha256": sha256_file(root / MAIN_COMPLETION_RECEIPT),
        "resolved_protocol_sha256": sha256_file(protocol_path),
        "main_code_commit": main_code_commit,
        "verified_nested_training_receipts": EXPECTED_RUN_COUNT,
        "training_artifacts": [
            {
                "run_id": run.run_id,
                "checkpoint_sha256": run.checkpoint_sha256,
                "training_receipt_sha256": run.training_receipt_sha256,
            }
            for run in runs
        ],
    }
    return VerifiedMain(root, manifest, summary, protocol, tuple(runs), binding)


def _analysis_spec(*, smoke: bool) -> PrimaryAnalysisSpec:
    if not smoke:
        return PrimaryAnalysisSpec()
    return PrimaryAnalysisSpec(
        trajectory_count=8,
        spline_count=8,
        task_horizon=8,
        blank_horizon=16,
        spectrum_chunk_size=4,
        smoke=True,
    )


def _require_exact_main_commit(
    git_state: Mapping[str, Any], main: VerifiedMain, *, smoke: bool
) -> None:
    """Keep every full downstream stage on the selector/main source commit."""

    if smoke:
        return
    if git_state.get("worktree_dirty") is not False:
        raise RuntimeError("full primary analysis requires a clean committed worktree")
    if git_state.get("code_commit") != main.binding.get("main_code_commit"):
        raise RuntimeError(
            "full primary analysis must use the exact selector/main code commit"
        )


def _analysis_identity(
    run: AnalysisRun, main: VerifiedMain, spec: PrimaryAnalysisSpec
) -> tuple[str, dict[str, Any]]:
    payload = primary_analysis_identity_payload(
        checkpoint_sha256=run.checkpoint_sha256,
        protocol_sha256=main.binding["resolved_protocol_sha256"],
        protocol_fingerprint_value=main.binding[
            "protocol_canonical_fingerprint"
        ],
        model_name=run.model_id,
        spec=spec,
    )
    return canonical_hash(payload), payload


def _analysis_output(root: Path, run: AnalysisRun) -> Path:
    return root / "analysis" / f"model={run.model_id}" / f"seed={run.model_seed}"


def _campaign_receipt_metadata(
    manifest: Mapping[str, Any], run: AnalysisRun
) -> dict[str, str]:
    return {
        "campaign_scientific_identity": str(manifest["scientific_identity"]),
        "protocol_fingerprint": str(manifest["protocol_canonical_fingerprint"]),
        "run_id": run.run_id,
        "stage": "sagodi_primary_analysis",
    }


def _validate_finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not numeric") from error
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is non-finite")
    return number


def _normal_recovery_design(spec: PrimaryAnalysisSpec) -> dict[str, Any]:
    anchor_count = (
        min(NORMAL_RECOVERY_FULL_ANCHOR_COUNT, int(spec.spline_count))
        if spec.smoke
        else NORMAL_RECOVERY_FULL_ANCHOR_COUNT
    )
    if spec.smoke:
        horizons = tuple(
            value
            for value in NORMAL_RECOVERY_FULL_HORIZONS
            if value <= int(spec.blank_horizon)
        )
        if int(spec.blank_horizon) not in horizons:
            horizons = (*horizons, int(spec.blank_horizon))
    else:
        horizons = NORMAL_RECOVERY_FULL_HORIZONS
    base_by_family = {
        family: anchor_count * direction_count * len(NORMAL_RECOVERY_RADII)
        for family, direction_count in NORMAL_RECOVERY_DIRECTIONS.items()
    }
    records_by_family = {
        family: count * len(horizons) for family, count in base_by_family.items()
    }
    return {
        "seed": NORMAL_RECOVERY_SEED,
        "anchor_count": anchor_count,
        "anchor_indices": [
            (index * int(spec.spline_count)) // anchor_count
            for index in range(anchor_count)
        ],
        "ambient_directions_per_anchor": NORMAL_RECOVERY_DIRECTIONS[
            "ambient_normal"
        ],
        "in_plane_radial_directions_per_anchor": NORMAL_RECOVERY_DIRECTIONS[
            "in_plane_radial"
        ],
        "radii_over_manifold_scale": list(NORMAL_RECOVERY_RADII),
        "horizons": list(horizons),
        "smoke_reduction": bool(spec.smoke),
        "registered_base_perturbation_count_by_family": base_by_family,
        "registered_base_perturbation_count": sum(base_by_family.values()),
        "registered_horizon_record_count_by_family": records_by_family,
        "registered_horizon_record_count": sum(records_by_family.values()),
    }


def _validate_normal_recovery_statistics(
    value: Any, *, expected_count: int, label: str
) -> Mapping[str, Any]:
    stats = _require_mapping(value, label)
    if int(stats.get("registered_count", -1)) != int(expected_count):
        raise RuntimeError(f"{label} registered count differs from the freeze")
    if int(stats.get("finite_count", -1)) != int(expected_count):
        raise RuntimeError(f"{label} omits a finite registered value")
    if int(stats.get("missing_or_nonfinite_count", -1)) != 0:
        raise RuntimeError(f"{label} contains a missing or non-finite value")
    numeric = {
        name: _validate_finite_number(stats.get(name), f"{label} {name}")
        for name in NORMAL_RECOVERY_STATISTICS
    }
    if numeric["population_std"] < 0.0:
        raise RuntimeError(f"{label} population_std is negative")
    if not (
        numeric["min"]
        <= numeric["q05"]
        <= numeric["median"]
        <= numeric["q95"]
        <= numeric["max"]
    ):
        raise RuntimeError(f"{label} order statistics are inconsistent")
    return stats


def _validate_carrier_normal_recovery_summary(
    value: Any, spec: PrimaryAnalysisSpec
) -> Mapping[str, Any]:
    recovery = _require_mapping(value, "carrier ambient-normal recovery")
    if recovery.get("state_space") != "minimum_causal_primary_carrier_state":
        raise RuntimeError("normal recovery does not use the minimum causal carrier state")
    role = str(recovery.get("role", ""))
    if "project_defined" not in role or "descriptive" not in role:
        raise RuntimeError(
            "normal recovery must identify itself as a project-defined descriptive extension"
        )
    design = _require_mapping(
        recovery.get("deterministic_design"), "normal recovery deterministic design"
    )
    expected = _normal_recovery_design(spec)
    for key in (
        "seed",
        "anchor_count",
        "anchor_indices",
        "ambient_directions_per_anchor",
        "in_plane_radial_directions_per_anchor",
        "radii_over_manifold_scale",
        "horizons",
        "smoke_reduction",
        "registered_base_perturbation_count_by_family",
        "registered_base_perturbation_count",
        "registered_horizon_record_count_by_family",
        "registered_horizon_record_count",
    ):
        if design.get(key) != expected[key]:
            raise RuntimeError(f"normal recovery frozen design differs: {key}")
    manifold_scale = _validate_finite_number(
        recovery.get("manifold_scale"), "normal recovery manifold scale"
    )
    if manifold_scale <= 0.0:
        raise RuntimeError("normal recovery manifold scale must be positive")
    qa = _require_mapping(recovery.get("numerical_qa"), "normal recovery numerical QA")
    if int(qa.get("unique_anchor_count", -1)) != int(expected["anchor_count"]):
        raise RuntimeError("normal recovery QA unique-anchor count differs")
    if int(qa.get("expected_anchor_count", -1)) != int(expected["anchor_count"]):
        raise RuntimeError("normal recovery QA expected-anchor count differs")
    if qa.get("initial_manifold_distance_all_finite") is not True:
        raise RuntimeError("normal recovery initial manifold distance is not all finite")
    if qa.get("manifold_scale_finite_positive") is not True:
        raise RuntimeError("normal recovery QA rejects its manifold scale")
    for name in (
        "maximum_direction_norm_error",
        "maximum_absolute_tangent_dot_direction",
    ):
        qa_value = _validate_finite_number(qa.get(name), f"normal recovery QA {name}")
        if qa_value < 0.0:
            raise RuntimeError(f"normal recovery QA {name} is negative")
        if qa_value > NORMAL_RECOVERY_DIRECTION_QA_MAX:
            raise RuntimeError(f"normal recovery QA {name} exceeds construction tolerance")
    _validate_normal_recovery_statistics(
        qa.get("initial_manifold_distance"),
        expected_count=int(expected["registered_base_perturbation_count"]),
        label="normal recovery QA initial manifold distance",
    )
    for family in NORMAL_RECOVERY_DIRECTIONS:
        qa_key = f"{family}_base_perturbation_count"
        if int(qa.get(qa_key, -1)) != int(
            expected["registered_base_perturbation_count_by_family"][family]
        ):
            raise RuntimeError(f"normal recovery QA count differs: {family}")
    metrics_by_family = _require_mapping(
        recovery.get("metrics_by_family"), "normal recovery family metrics"
    )
    if set(metrics_by_family) != set(NORMAL_RECOVERY_DIRECTIONS):
        raise RuntimeError("normal recovery direction families differ from the freeze")
    for family, direction_count in NORMAL_RECOVERY_DIRECTIONS.items():
        family_payload = _require_mapping(
            metrics_by_family.get(family), f"normal recovery {family}"
        )
        by_radius = _require_mapping(
            family_payload.get("by_radius"), f"normal recovery {family} by radius"
        )
        expected_radius_keys = {format(radius, "g") for radius in NORMAL_RECOVERY_RADII}
        if set(by_radius) != expected_radius_keys:
            raise RuntimeError(f"normal recovery {family} radii differ from the freeze")
        expected_count = int(expected["anchor_count"]) * direction_count
        for radius_key in sorted(expected_radius_keys, key=float):
            radius_payload = _require_mapping(
                by_radius.get(radius_key),
                f"normal recovery {family} radius {radius_key}",
            )
            by_horizon = _require_mapping(
                radius_payload.get("by_horizon"),
                f"normal recovery {family} radius {radius_key} by horizon",
            )
            if set(by_horizon) != {str(value) for value in expected["horizons"]}:
                raise RuntimeError(
                    f"normal recovery {family} radius {radius_key} horizons differ"
                )
            for horizon in expected["horizons"]:
                horizon_payload = _require_mapping(
                    by_horizon.get(str(horizon)),
                    f"normal recovery {family} radius {radius_key} horizon {horizon}",
                )
                for metric in NORMAL_RECOVERY_METRICS:
                    _validate_normal_recovery_statistics(
                        horizon_payload.get(metric),
                        expected_count=expected_count,
                        label=(
                            f"normal recovery {family} radius {radius_key} "
                            f"horizon {horizon} {metric}"
                        ),
                    )
    if recovery.get("claim_gate") is not False:
        raise RuntimeError("normal recovery must explicitly disable its claim gate")
    for forbidden in ("threshold", "pass_threshold", "passed", "expected_direction_gate"):
        if forbidden in recovery:
            raise RuntimeError(f"normal recovery may not define {forbidden}")
    return recovery


def _validate_carrier_normal_recovery_artifact(
    path: Path,
    spec: PrimaryAnalysisSpec,
    recovery_summary: Mapping[str, Any],
) -> None:
    expected = _normal_recovery_design(spec)
    base_arrays = {
        "family",
        "anchor_index",
        "anchor_angle",
        "direction_index",
        "radius_over_manifold_scale",
        "radius_absolute",
        "direction",
        "tangent",
        "direction_norm_error",
        "absolute_tangent_dot_direction",
    }
    matrix_arrays = {
        "nearest_manifold_index",
        "manifold_distance",
        "manifold_distance_ratio",
        "decoded_angle",
        "same_memory_error_radians",
        "clean_manifold_distance",
        "clean_decoded_angle",
        "clean_same_memory_error_radians",
        "excess_same_memory_error_radians",
        "distance_to_matched_clean_state",
        "distance_to_matched_clean_state_ratio",
        "manifold_distance_minus_clean",
    }
    required = base_arrays | matrix_arrays | {
        "horizon",
        "manifold_scale",
    }
    try:
        with np.load(path, allow_pickle=False) as arrays:
            missing = sorted(required - set(arrays.files))
            if missing:
                raise RuntimeError(
                    f"normal recovery artifact omits registered arrays: {missing}"
                )
            base_count = int(expected["registered_base_perturbation_count"])
            horizon_count = len(expected["horizons"])
            for name in base_arrays:
                array = np.asarray(arrays[name])
                if array.ndim < 1 or int(array.shape[0]) != base_count:
                    raise RuntimeError(
                        f"normal recovery artifact {name} has the wrong base-trial count"
                    )
                if name != "family" and not np.isfinite(array).all():
                    raise RuntimeError(
                        f"normal recovery artifact {name} contains NaN or Inf"
                    )
            for name in matrix_arrays:
                array = np.asarray(arrays[name])
                if array.shape != (base_count, horizon_count):
                    raise RuntimeError(
                        f"normal recovery artifact {name} must have [trial,horizon] shape"
                    )
                if not np.isfinite(array).all():
                    raise RuntimeError(
                        f"normal recovery artifact {name} contains NaN or Inf"
                    )
            families = np.asarray(arrays["family"]).astype(str)
            if set(families.tolist()) != set(NORMAL_RECOVERY_DIRECTIONS):
                raise RuntimeError("normal recovery artifact families differ from freeze")
            horizons = np.asarray(arrays["horizon"], dtype=np.int64)
            if horizons.shape != (horizon_count,) or horizons.tolist() != expected[
                "horizons"
            ]:
                raise RuntimeError("normal recovery artifact horizons differ from freeze")
            radii = np.asarray(arrays["radius_over_manifold_scale"], dtype=np.float64)
            anchors = np.asarray(arrays["anchor_index"], dtype=np.int64)
            direction_indices = np.asarray(arrays["direction_index"], dtype=np.int64)
            directions = np.asarray(arrays["direction"], dtype=np.float64)
            tangents = np.asarray(arrays["tangent"], dtype=np.float64)
            if directions.ndim != 2 or tangents.shape != directions.shape:
                raise RuntimeError(
                    "normal recovery direction/tangent arrays must share [trial,state]"
                )
            computed_norm_error = np.abs(np.linalg.norm(directions, axis=1) - 1.0)
            computed_abs_dot = np.abs(np.sum(directions * tangents, axis=1))
            stored_norm_error = np.asarray(
                arrays["direction_norm_error"], dtype=np.float64
            )
            stored_abs_dot = np.asarray(
                arrays["absolute_tangent_dot_direction"], dtype=np.float64
            )
            if not np.allclose(
                stored_norm_error, computed_norm_error, rtol=1e-6, atol=1e-10
            ):
                raise RuntimeError("normal recovery direction-norm QA is inconsistent")
            if not np.allclose(
                stored_abs_dot, computed_abs_dot, rtol=1e-6, atol=1e-10
            ):
                raise RuntimeError("normal recovery tangent-orthogonality QA is inconsistent")
            if (
                float(computed_norm_error.max())
                > NORMAL_RECOVERY_DIRECTION_QA_MAX
            ):
                raise RuntimeError(
                    "normal recovery directions exceed the normalization tolerance"
                )
            if (
                float(computed_abs_dot.max())
                > NORMAL_RECOVERY_DIRECTION_QA_MAX
            ):
                raise RuntimeError(
                    "normal recovery directions exceed the tangent-orthogonality tolerance"
                )
            scale = np.asarray(arrays["manifold_scale"], dtype=np.float64)
            if scale.shape != () or not np.isfinite(scale).all() or float(scale) <= 0.0:
                raise RuntimeError("normal recovery artifact manifold scale is invalid")
            if not math.isclose(
                float(scale),
                _validate_finite_number(
                    recovery_summary.get("manifold_scale"),
                    "normal recovery summary manifold scale",
                ),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise RuntimeError("normal recovery summary manifold scale differs from NPZ")
            summary_families = _require_mapping(
                recovery_summary.get("metrics_by_family"),
                "normal recovery summary family metrics",
            )
            for family, direction_count in NORMAL_RECOVERY_DIRECTIONS.items():
                for radius in NORMAL_RECOVERY_RADII:
                    mask = (families == family) & np.isclose(
                        radii, radius, rtol=0.0, atol=1e-12
                    )
                    expected_cell = int(expected["anchor_count"]) * direction_count
                    if int(mask.sum()) != expected_cell:
                        raise RuntimeError(
                            "normal recovery artifact has an incomplete "
                            f"family/radius cell: {family}/{radius:g}"
                        )
                    if set(anchors[mask].tolist()) != set(expected["anchor_indices"]):
                        raise RuntimeError(
                            "normal recovery artifact omits a registered anchor: "
                            f"{family}/{radius:g}"
                        )
                    for anchor in expected["anchor_indices"]:
                        at_anchor = mask & (anchors == int(anchor))
                        if set(direction_indices[at_anchor].tolist()) != set(
                            range(direction_count)
                        ):
                            raise RuntimeError(
                                "normal recovery artifact direction indices differ: "
                                f"{family}/{radius:g}/anchor={anchor}"
                            )
                    radius_key = format(radius, "g")
                    family_summary = _require_mapping(
                        summary_families.get(family),
                        f"normal recovery summary {family}",
                    )
                    radius_summary = _require_mapping(
                        _require_mapping(
                            family_summary.get("by_radius"),
                            f"normal recovery summary {family} radii",
                        ).get(radius_key),
                        f"normal recovery summary {family}/{radius_key}",
                    )
                    horizon_summaries = _require_mapping(
                        radius_summary.get("by_horizon"),
                        f"normal recovery summary {family}/{radius_key} horizons",
                    )
                    for column, horizon in enumerate(expected["horizons"]):
                        horizon_summary = _require_mapping(
                            horizon_summaries.get(str(horizon)),
                            f"normal recovery summary {family}/{radius_key}/{horizon}",
                        )
                        for metric in NORMAL_RECOVERY_METRICS:
                            values = np.asarray(arrays[metric], dtype=np.float64)[
                                mask, column
                            ]
                            recomputed = {
                                "registered_count": int(values.size),
                                "finite_count": int(np.isfinite(values).sum()),
                                "missing_or_nonfinite_count": int(
                                    values.size - np.isfinite(values).sum()
                                ),
                                "mean": float(values.mean()),
                                "population_std": float(values.std(ddof=0)),
                                "median": float(np.median(values)),
                                "q05": float(np.quantile(values, 0.05)),
                                "q95": float(np.quantile(values, 0.95)),
                                "min": float(values.min()),
                                "max": float(values.max()),
                            }
                            observed = _require_mapping(
                                horizon_summary.get(metric),
                                (
                                    "normal recovery summary "
                                    f"{family}/{radius_key}/{horizon}/{metric}"
                                ),
                            )
                            for count_name in (
                                "registered_count",
                                "finite_count",
                                "missing_or_nonfinite_count",
                            ):
                                if int(observed.get(count_name, -1)) != recomputed[
                                    count_name
                                ]:
                                    raise RuntimeError(
                                        "normal recovery summary count differs from NPZ: "
                                        f"{family}/{radius_key}/{horizon}/{metric}/{count_name}"
                                    )
                            for statistic in NORMAL_RECOVERY_STATISTICS:
                                if not math.isclose(
                                    _validate_finite_number(
                                        observed.get(statistic),
                                        (
                                            "normal recovery summary "
                                            f"{family}/{radius_key}/{horizon}/{metric}/{statistic}"
                                        ),
                                    ),
                                    recomputed[statistic],
                                    rel_tol=1e-9,
                                    abs_tol=1e-12,
                                ):
                                    raise RuntimeError(
                                        "normal recovery summary statistic differs from NPZ: "
                                        f"{family}/{radius_key}/{horizon}/{metric}/{statistic}"
                                    )
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"normal recovery artifact is unreadable: {error}") from error


def verify_analysis_output(
    output: Path,
    run: AnalysisRun,
    main: VerifiedMain,
    manifest: Mapping[str, Any],
    spec: PrimaryAnalysisSpec,
) -> tuple[bool, str, dict[str, Any] | None]:
    identity, identity_payload = _analysis_identity(run, main, spec)
    try:
        identity_file = _require_mapping(
            strict_json_load(output / "analysis_identity.json"), "analysis identity"
        )
        summary = dict(
            _require_mapping(strict_json_load(output / "summary.json"), "analysis summary")
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        return False, f"analysis output unreadable: {error}", None
    observed_identity_payload = identity_file.get("identity_payload")
    if (
        identity_file.get("analysis_identity") != identity
        or not isinstance(observed_identity_payload, Mapping)
        or canonical_hash(observed_identity_payload) != canonical_hash(identity_payload)
    ):
        return False, "analysis identity differs from checkpoint/protocol/spec binding", None
    status = summary.get("analysis_status")
    if status not in TERMINAL_ANALYSIS_STATUSES:
        return False, f"analysis has non-terminal status: {status!r}", None
    expected_metadata: dict[str, Any] = {
        **_campaign_receipt_metadata(manifest, run),
        "analysis_identity": identity,
        "analysis_status": status,
    }
    valid, reason = verify_completion_receipt(
        output / "completion_receipt.json",
        expected_job_id=f"sagodi-primary-{run.model_id}-{identity[:12]}",
        expected_metadata=expected_metadata,
    )
    if not valid:
        return False, reason, None
    try:
        receipt = _require_mapping(
            strict_json_load(output / "completion_receipt.json"),
            "analysis completion receipt",
        )
        receipt_artifacts = _require_mapping(
            receipt.get("artifacts"), "analysis receipt artifacts"
        )
        artifact_names = {Path(str(name)).as_posix() for name in receipt_artifacts}
        base_artifacts = {
            "analysis_identity.json",
            "progress.json",
            "direct_task_trajectories.npz",
            "summary.json",
        }
        prefix_by_failed_stage = {
            "slow_manifold_reconstruction": set(),
            "carrier_ambient_normal_recovery": {
                "slow_manifold_reconstruction.npz"
            },
            "projected_flow_and_fixed_point_topology": {
                "slow_manifold_reconstruction.npz",
                NORMAL_RECOVERY_ARTIFACT,
            },
            "full_local_jacobian_eigenspectrum": {
                "slow_manifold_reconstruction.npz",
                NORMAL_RECOVERY_ARTIFACT,
                "projected_flow_and_topology.npz",
            },
            "finite_time_blank_memory": {
                "slow_manifold_reconstruction.npz",
                NORMAL_RECOVERY_ARTIFACT,
                "projected_flow_and_topology.npz",
                "full_local_eigenspectrum.npz",
            },
            "asymptotic_memory_structure": {
                "slow_manifold_reconstruction.npz",
                NORMAL_RECOVERY_ARTIFACT,
                "projected_flow_and_topology.npz",
                "full_local_eigenspectrum.npz",
                "finite_time_angular_memory.npz",
            },
        }
        if status == "complete_structural_summary_eligible":
            required_artifacts = base_artifacts | {
                "slow_manifold_reconstruction.npz",
                NORMAL_RECOVERY_ARTIFACT,
                "projected_flow_and_topology.npz",
                "full_local_eigenspectrum.npz",
                "finite_time_angular_memory.npz",
                "asymptotic_structure.npz",
            }
        elif status == "structural_analysis_not_estimable":
            failure = _require_mapping(
                summary.get("structural_numerical_failure"),
                "structural numerical failure",
            )
            failed_stage = str(failure.get("failed_stage", ""))
            if failed_stage not in prefix_by_failed_stage:
                raise RuntimeError("not-estimable analysis has an unknown failed stage")
            required_artifacts = base_artifacts | prefix_by_failed_stage[failed_stage]
        else:
            required_artifacts = base_artifacts
        missing_artifacts = sorted(required_artifacts - artifact_names)
        if missing_artifacts:
            raise RuntimeError(
                f"analysis receipt omits required artifacts: {missing_artifacts}"
            )
        if NORMAL_RECOVERY_ARTIFACT in required_artifacts:
            recovery_summary = _validate_carrier_normal_recovery_summary(
                summary.get("carrier_ambient_normal_recovery"), spec
            )
            _validate_carrier_normal_recovery_artifact(
                output / NORMAL_RECOVERY_ARTIFACT, spec, recovery_summary
            )
        checkpoint = _require_mapping(summary.get("checkpoint"), "analysis checkpoint")
        protocol = _require_mapping(summary.get("protocol"), "analysis protocol")
        eligibility = _require_mapping(
            summary.get("structural_summary_eligibility"), "analysis eligibility"
        )
        if checkpoint.get("sha256") != run.checkpoint_sha256:
            raise RuntimeError("analysis checkpoint hash differs")
        if protocol.get("sha256") != main.binding[
            "resolved_protocol_sha256"
        ] or protocol.get("fingerprint") != main.binding[
            "protocol_canonical_fingerprint"
        ]:
            raise RuntimeError("analysis resolved-protocol binding differs")
        if summary.get("smoke") is not bool(spec.smoke):
            raise RuntimeError("analysis smoke label differs")
        eligible = eligibility.get("eligible")
        if not isinstance(eligible, bool):
            raise RuntimeError("analysis eligibility must be boolean")
        if status == "ineligible_for_structural_summary" and eligible:
            raise RuntimeError("ineligible terminal status contradicts eligibility")
        if not spec.smoke:
            if status == "ineligible_for_structural_summary" and eligible is not False:
                raise RuntimeError("full ineligible status requires eligible=false")
            if status in {
                "complete_structural_summary_eligible",
                "structural_analysis_not_estimable",
            } and eligible is not True:
                raise RuntimeError(
                    "full structural terminal status requires frozen-NMSE eligibility"
                )
        estimable = status == "complete_structural_summary_eligible"
        if estimable:
            _validate_finite_number(
                _require_mapping(summary.get("projected_flow"), "projected flow").get(
                    "uniform_norm"
                ),
                "uniform flow norm",
            )
            topology = _require_mapping(
                summary.get("fixed_point_topology"), "fixed-point topology"
            )
            if topology.get("kind") not in {
                "fixed_points",
                "limit_cycle",
                "stationary_continuum",
                "unidirectional_with_stationary_samples",
            }:
                raise RuntimeError("unknown topology kind")
            spectrum = _require_mapping(
                summary.get("full_local_eigenspectrum"), "full eigenspectrum"
            )
            if int(spectrum.get("point_count", -1)) != int(spec.spline_count):
                raise RuntimeError("full eigenspectrum does not cover every spline point")
            for field, label in (
                ("largest_real_part", "largest real part"),
                ("second_largest_real_part", "second-largest real part"),
                ("top_two_real_part_gap", "top-two real-part gap"),
                ("map_spectral_radius", "map spectral radius"),
            ):
                stats = _require_mapping(spectrum.get(field), label)
                if int(stats.get("count", -1)) != int(spec.spline_count):
                    raise RuntimeError(
                        f"{label} omits one or more registered spline points"
                    )
                for statistic in ("mean", "min", "max"):
                    _validate_finite_number(
                        stats.get(statistic), f"{label} {statistic}"
                    )
            _validate_finite_number(
                spectrum.get("map_spectral_radius_below_one_fraction"),
                "map spectral-radius-below-one fraction",
            )
            _require_mapping(
                summary.get("finite_time_angular_memory"), "finite-time memory"
            )
            _require_mapping(summary.get("asymptotic_structure"), "asymptotic structure")
    except (RuntimeError, TypeError, ValueError, KeyError) as error:
        return False, str(error), None
    return True, "verified terminal Ságodi-primary analysis receipt", summary


def _prepare_root(
    root: Path,
    main: VerifiedMain,
    spec: PrimaryAnalysisSpec,
    *,
    smoke: bool,
) -> None:
    marker_payload = {
        "schema_version": 1,
        "campaign_mode": CAMPAIGN_MODE,
        "scope": SCOPE,
        "main_binding": main.binding,
        "analysis_spec": primary_analysis_spec_payload(spec),
        "smoke": bool(smoke),
    }
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ROOT_MARKER
    if marker.exists():
        if strict_json_load(marker) != marker_payload:
            raise RuntimeError("analysis artifact-root marker differs; choose a new root")
    else:
        if any(root.iterdir()):
            raise RuntimeError("analysis artifact root is nonempty and unmarked")
        atomic_json(marker, marker_payload)


def _build_manifest(
    root: Path,
    main: VerifiedMain,
    spec: PrimaryAnalysisSpec,
    *,
    python: str,
    gpus: Sequence[int],
    smoke: bool,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    package_dir = Path(__file__).resolve().parent
    payload = {
        "campaign_mode": CAMPAIGN_MODE,
        "scope": SCOPE,
        "analysis_role": "Ságodi_evaluation_tool_not_CA_LRU_method",
        "failed_seed_replacement_policy": "forbidden",
        "smoke": bool(smoke),
        "main_binding": main.binding,
        "protocol_canonical_fingerprint": main.binding[
            "protocol_canonical_fingerprint"
        ],
        "analysis_spec": primary_analysis_spec_payload(spec),
        "run_matrix": [run.payload(main.root) for run in main.runs],
        "expected_analysis_runs": EXPECTED_RUN_COUNT,
        "code": _git_state(repo_root),
        "source_hashes": _source_hashes(repo_root, package_dir),
        "environment": _environment_fingerprint(python, gpus),
        "python": str(Path(python).resolve()),
        "gpus": [int(gpu) for gpu in gpus],
    }
    return {
        "schema_version": 1,
        **payload,
        "scientific_identity_payload": payload,
        "scientific_identity": canonical_hash(payload),
    }


def _write_or_check_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    if path.exists():
        if strict_json_load(path) != dict(manifest):
            raise RuntimeError("immutable primary-analysis manifest differs")
    else:
        atomic_json(path, dict(manifest))


def _verify_campaign_inputs(
    root: Path,
    manifest: Mapping[str, Any],
    main: VerifiedMain,
) -> None:
    signed = _require_mapping(
        manifest.get("scientific_identity_payload"), "analysis scientific identity"
    )
    if canonical_hash(signed) != manifest.get("scientific_identity"):
        raise RuntimeError("analysis manifest scientific identity is invalid")
    if strict_json_load(root / MANIFEST) != dict(manifest):
        raise RuntimeError("on-disk analysis manifest differs")
    repo_root = Path(__file__).resolve().parents[2]
    if _git_state(repo_root) != manifest["code"]:
        raise RuntimeError("git state changed after analysis manifest creation")
    if _source_hashes(repo_root, Path(__file__).resolve().parent) != manifest[
        "source_hashes"
    ]:
        raise RuntimeError("campaign Python sources changed after manifest creation")
    current = verify_main_artifacts(main.root)
    if current.binding != main.binding:
        raise RuntimeError("parent main-training artifact binding changed")


def _analysis_command(
    run: AnalysisRun,
    main: VerifiedMain,
    attempt: Path,
    python: str,
    spec: PrimaryAnalysisSpec,
) -> list[str]:
    command = [
        python,
        "-m",
        "repro.sagodi_protocol.sagodi_primary_runner",
        "--checkpoint",
        str(run.training_output / "checkpoint.pt"),
        "--protocol",
        str(main.root / MAIN_RESOLVED_PROTOCOL),
        "--output",
        str(attempt),
        "--device",
        "cuda:0",
        "--spectrum-chunk-size",
        str(spec.spectrum_chunk_size),
    ]
    if spec.smoke:
        command.extend(
            [
                "--smoke",
                "--trajectory-count",
                str(spec.trajectory_count),
                "--spline-count",
                str(spec.spline_count),
                "--task-horizon",
                str(spec.task_horizon),
                "--blank-horizon",
                str(spec.blank_horizon),
            ]
        )
    return command


def _child_environment(
    manifest: Mapping[str, Any], run: AnalysisRun, gpu: int
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    environment["CALRU_PHYSICAL_GPU_ID"] = str(int(gpu))
    metadata = _campaign_receipt_metadata(manifest, run)
    for key, variable in RECEIPT_IDENTITY_ENV.items():
        environment[variable] = metadata[key]
    return environment


def _attempt_parent(root: Path, run: AnalysisRun) -> Path:
    return root / "attempts" / "sagodi_primary_analysis" / run.run_id


def _recover_or_resume_attempt(
    root: Path,
    run: AnalysisRun,
    main: VerifiedMain,
    manifest: Mapping[str, Any],
    spec: PrimaryAnalysisSpec,
) -> tuple[Path | None, str]:
    parent = _attempt_parent(root, run)
    if not parent.is_dir():
        return None, "no prior attempt"
    attempts = [
        path
        for path in sorted(parent.iterdir())
        if path.is_dir() and path.name.startswith("attempt-")
    ]
    valid = [
        path
        for path in attempts
        if verify_analysis_output(path, run, main, manifest, spec)[0]
    ]
    if len(valid) > 1:
        raise RuntimeError(f"multiple valid unpublished attempts for {run.run_id}")
    output = _analysis_output(root, run)
    if valid:
        if output.exists():
            raise RuntimeError(f"cannot recover {run.run_id} over existing output")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(valid[0], output)
        return None, f"recovered verified attempt {valid[0]}"
    identity, _ = _analysis_identity(run, main, spec)
    resumable: list[Path] = []
    for path in attempts:
        identity_path = path / "analysis_identity.json"
        if not identity_path.is_file():
            continue
        try:
            payload = strict_json_load(identity_path)
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(payload, Mapping) and payload.get("analysis_identity") == identity:
            resumable.append(path)
    if len(resumable) > 1:
        raise RuntimeError(f"multiple resumable attempts for {run.run_id}; audit required")
    if resumable:
        return resumable[0], f"resume existing attempt {resumable[0]}"
    return None, "no resumable attempt"


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


def _run_jobs(
    root: Path,
    main: VerifiedMain,
    manifest: Mapping[str, Any],
    spec: PrimaryAnalysisSpec,
    python: str,
    gpus: Sequence[int],
    status: dict[str, Any],
) -> None:
    jobs = status.setdefault("jobs", {})
    pending: list[tuple[AnalysisRun, Path | None]] = []
    for run in main.runs:
        output = _analysis_output(root, run)
        valid, reason, _ = verify_analysis_output(
            output, run, main, manifest, spec
        )
        if valid:
            jobs[run.run_id] = {"state": "complete", "reason": reason}
            continue
        if output.exists():
            preserved = _preserve_invalid_output(
                root, "sagodi_primary_analysis", run.run_id, output
            )
            reason = f"invalid final output preserved at {preserved}: {reason}"
        attempt, recovery_reason = _recover_or_resume_attempt(
            root, run, main, manifest, spec
        )
        published = _analysis_output(root, run)
        if published.exists():
            jobs[run.run_id] = {"state": "complete", "reason": recovery_reason}
            continue
        jobs[run.run_id] = {
            "state": "pending",
            "reason": reason,
            "resume": recovery_reason,
        }
        pending.append((run, attempt))
    atomic_json(root / STATUS, status)

    available = sorted(int(gpu) for gpu in gpus)
    active: dict[int, dict[str, Any]] = {}
    try:
        while pending or active:
            while pending and available:
                _verify_campaign_inputs(root, manifest, main)
                gpu = available.pop(0)
                run, attempt = pending.pop(0)
                if attempt is None:
                    attempt = _unique_attempt_path(
                        root, "sagodi_primary_analysis", run.run_id
                    )
                log_dir = root / "logs" / "sagodi_primary_analysis" / run.run_id
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{attempt.name}.log"
                handle = log_path.open("ab", buffering=0)
                process = subprocess.Popen(
                    _analysis_command(run, main, attempt, python, spec),
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
                        "state": "retryable_failure_same_seed_and_attempt",
                        "return_code": return_code,
                        "attempt_dir": str(attempt),
                        "log": str(item["log"]),
                    }
                    atomic_json(root / STATUS, status)
                    raise RuntimeError(
                        f"analysis {run.run_id} exited {return_code}; resume the same "
                        f"attempt after inspection: {item['log']}"
                    )
                valid, reason, _ = verify_analysis_output(
                    attempt, run, main, manifest, spec
                )
                if not valid:
                    jobs[run.run_id] = {
                        "state": "retryable_invalid_output_same_seed_and_attempt",
                        "reason": reason,
                        "attempt_dir": str(attempt),
                    }
                    atomic_json(root / STATUS, status)
                    raise RuntimeError(
                        f"analysis {run.run_id} returned an invalid terminal output: {reason}"
                    )
                output = _analysis_output(root, run)
                output.parent.mkdir(parents=True, exist_ok=True)
                os.replace(attempt, output)
                valid, reason, _ = verify_analysis_output(
                    output, run, main, manifest, spec
                )
                if not valid:
                    raise RuntimeError(f"published analysis became invalid: {reason}")
                jobs[run.run_id] = {"state": "complete", "reason": reason}
                atomic_json(root / STATUS, status)
            if not progressed and active:
                time.sleep(0.2)
    except BaseException:
        _terminate_active(active)
        raise


def _estimable_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    flow = _require_mapping(summary["projected_flow"], "projected flow")
    topology = _require_mapping(summary["fixed_point_topology"], "topology")
    spectrum = _require_mapping(summary["full_local_eigenspectrum"], "spectrum")
    finite = _require_mapping(summary["finite_time_angular_memory"], "finite memory")
    asymptotic = _require_mapping(summary["asymptotic_structure"], "asymptotic")
    normal_recovery = _require_mapping(
        summary["carrier_ambient_normal_recovery"],
        "carrier ambient-normal recovery",
    )
    return {
        "uniform_flow_norm": _validate_finite_number(
            flow["uniform_norm"], "uniform flow norm"
        ),
        "topology": {
            "kind": topology.get("kind"),
            "stable_count": int(topology.get("stable_count", 0)),
            "saddle_count": int(topology.get("saddle_count", 0)),
        },
        "top_two_real_part_gap": dict(
            _require_mapping(
                spectrum["top_two_real_part_gap"], "top-two real-part gap"
            )
        ),
        "largest_vector_field_real_part": dict(
            _require_mapping(
                spectrum["largest_real_part"], "largest vector-field real part"
            )
        ),
        "second_largest_vector_field_real_part": dict(
            _require_mapping(
                spectrum["second_largest_real_part"],
                "second-largest vector-field real part",
            )
        ),
        "map_spectral_radius": dict(
            _require_mapping(spectrum["map_spectral_radius"], "map spectral radius")
        ),
        "map_spectral_radius_below_one_fraction": _validate_finite_number(
            spectrum["map_spectral_radius_below_one_fraction"],
            "map spectral-radius-below-one fraction",
        ),
        "finite_time_angular_error": {
            "named_horizons": finite.get("named_horizons", {}),
            "terminal_mean_error_radians": finite.get(
                "terminal_mean_error_radians"
            ),
            "terminal_maximum_error_radians": finite.get(
                "terminal_maximum_error_radians"
            ),
        },
        "asymptotic_capacity": {
            key: asymptotic.get(key)
            for key in (
                "topology",
                "capacity_status",
                "stable_count",
                "saddle_count",
                "shannon_entropy_nats",
                "effective_basin_count",
                "asymptotic_mean_error_radians",
                "asymptotic_maximum_error_radians",
                "fixed_point_basin_capacity",
            )
        },
        "carrier_ambient_normal_recovery": dict(normal_recovery),
    }


def _nested_value(value: Any, path: Sequence[str]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _descriptive_values(
    values: Sequence[Any], *, registered_denominator: int, conditional_denominator: int
) -> dict[str, Any]:
    finite: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    missing = int(conditional_denominator) - len(finite)
    result: dict[str, Any] = {
        "registered_seed_denominator": int(registered_denominator),
        "eligible_and_estimable_conditional_denominator": int(
            conditional_denominator
        ),
        "finite_value_count": len(finite),
        "missing_or_nonfinite_within_conditional_denominator": missing,
        "mean": None,
        "population_std": None,
        "median": None,
        "q05": None,
        "q95": None,
        "min": None,
        "max": None,
    }
    if finite:
        ordered = sorted(finite)

        def quantile(probability: float) -> float:
            if len(ordered) == 1:
                return float(ordered[0])
            position = probability * (len(ordered) - 1)
            lower = int(math.floor(position))
            upper = int(math.ceil(position))
            weight = position - lower
            return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

        result.update(
            {
                "mean": float(math.fsum(finite) / len(finite)),
                "population_std": float(statistics.pstdev(finite)),
                "median": float(statistics.median(finite)),
                "q05": quantile(0.05),
                "q95": quantile(0.95),
                "min": float(min(finite)),
                "max": float(max(finite)),
            }
        )
    return result


def _model_numeric_summaries(
    group: Sequence[Mapping[str, Any]], spec: PrimaryAnalysisSpec
) -> dict[str, Any]:
    """Summarize registered metrics without hiding any seed denominator."""

    included = [
        row for row in group if bool(row["included_in_primary_structural_summary"])
    ]
    metric_paths: dict[str, tuple[str, ...]] = {
        "uniform_projected_flow_norm": ("uniform_flow_norm",),
        "lambda1_vector_field_real_part_manifold_mean": (
            "largest_vector_field_real_part",
            "mean",
        ),
        "lambda2_vector_field_real_part_manifold_mean": (
            "second_largest_vector_field_real_part",
            "mean",
        ),
        "top_two_vector_field_real_part_gap_manifold_mean": (
            "top_two_real_part_gap",
            "mean",
        ),
        "map_spectral_radius_manifold_mean": ("map_spectral_radius", "mean"),
        "map_spectral_radius_below_one_fraction": (
            "map_spectral_radius_below_one_fraction",
        ),
        "stable_fixed_point_count": ("topology", "stable_count"),
        "saddle_fixed_point_count": ("topology", "saddle_count"),
        "finite_terminal_mean_error_radians": (
            "finite_time_angular_error",
            "terminal_mean_error_radians",
        ),
        "finite_terminal_maximum_error_radians": (
            "finite_time_angular_error",
            "terminal_maximum_error_radians",
        ),
        "asymptotic_shannon_entropy_nats_when_defined": (
            "asymptotic_capacity",
            "shannon_entropy_nats",
        ),
        "asymptotic_effective_basin_count_when_defined": (
            "asymptotic_capacity",
            "effective_basin_count",
        ),
        "asymptotic_mean_error_radians_when_defined": (
            "asymptotic_capacity",
            "asymptotic_mean_error_radians",
        ),
        "asymptotic_maximum_error_radians_when_defined": (
            "asymptotic_capacity",
            "asymptotic_maximum_error_radians",
        ),
    }
    for horizon in ("1T", "3T", "5T", "7T", "9T"):
        for field in (
            "instantaneous_mean_error_radians",
            "instantaneous_maximum_error_radians",
            "cumulative_mean_error_radians",
        ):
            metric_paths[f"finite_{horizon}_{field}"] = (
                "finite_time_angular_error",
                "named_horizons",
                horizon,
                field,
            )
    design = _normal_recovery_design(spec)
    for family in NORMAL_RECOVERY_DIRECTIONS:
        for radius in NORMAL_RECOVERY_RADII:
            radius_key = format(radius, "g")
            for horizon in design["horizons"]:
                for metric in NORMAL_RECOVERY_METRICS:
                    metric_paths[
                        f"carrier_recovery_{family}_r{radius_key}_h{horizon}_{metric}_trial_mean"
                    ] = (
                        "carrier_ambient_normal_recovery",
                        "metrics_by_family",
                        family,
                        "by_radius",
                        radius_key,
                        "by_horizon",
                        str(horizon),
                        metric,
                        "mean",
                    )

    conditional: dict[str, Any] = {}
    for label, path in metric_paths.items():
        conditional[label] = _descriptive_values(
            [_nested_value(row.get("sagodi_metrics"), path) for row in included],
            registered_denominator=len(group),
            conditional_denominator=len(included),
        )
    validation = _descriptive_values(
        [
            _nested_value(row, ("training_outcome", "validation_masked_nmse_db"))
            for row in group
        ],
        registered_denominator=len(group),
        conditional_denominator=len(group),
    )
    return {
        "interpretation": (
            "Ságodi-based metrics and the project-defined descriptive carrier normal-"
            "recovery extension are conditional on frozen-NMSE eligibility and structural "
            "estimability; registered ten-seed and finite-value denominators are explicit"
        ),
        "validation_masked_nmse_db_all_registered_seeds": validation,
        "sagodi_metrics_conditional_on_eligible_and_estimable": conditional,
    }


def aggregate_results(
    root: Path,
    main: VerifiedMain,
    manifest: Mapping[str, Any],
    spec: PrimaryAnalysisSpec,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    child_receipts: list[Path] = []
    for run in main.runs:
        output = _analysis_output(root, run)
        valid, reason, analysis = verify_analysis_output(
            output, run, main, manifest, spec
        )
        if not valid or analysis is None:
            raise RuntimeError(f"cannot aggregate {run.run_id}: {reason}")
        status = str(analysis["analysis_status"])
        eligibility = _require_mapping(
            analysis["structural_summary_eligibility"], "eligibility"
        )
        eligible = bool(eligibility["eligible"])
        estimable = status == "complete_structural_summary_eligible"
        included = eligible and estimable
        rows.append(
            {
                "run_id": run.run_id,
                "model_id": run.model_id,
                "model_seed": run.model_seed,
                "training_outcome": {
                    "status": run.training_outcome.get("status"),
                    "validation_masked_nmse_db": run.training_outcome.get(
                        "validation_masked_nmse_db"
                    ),
                    "structural_summary_eligible_nmse_lt_minus20db": (
                        run.training_outcome.get(
                            "structural_summary_eligible_nmse_lt_minus20db"
                        )
                    ),
                    "train_loss_last": run.training_outcome.get("train_loss_last"),
                    "rp_calls": run.training_outcome.get("rp_calls"),
                    "checkpoint_sha256": run.checkpoint_sha256,
                    "training_receipt_sha256": run.training_receipt_sha256,
                },
                "analysis_status": status,
                "eligible_by_nmse_rule": eligible,
                "structurally_estimable": estimable,
                "included_in_primary_structural_summary": included,
                "sagodi_metrics": _estimable_metrics(analysis) if included else None,
                "analysis_receipt_sha256": sha256_file(
                    output / "completion_receipt.json"
                ),
            }
        )
        child_receipts.append(output / "completion_receipt.json")

    model_summaries: dict[str, Any] = {}
    for model_id in EXPECTED_MODEL_IDS:
        group = [row for row in rows if row["model_id"] == model_id]
        statuses = Counter(row["analysis_status"] for row in group)
        eligible_count = sum(bool(row["eligible_by_nmse_rule"]) for row in group)
        included_count = sum(
            bool(row["included_in_primary_structural_summary"]) for row in group
        )
        model_summaries[model_id] = {
            "registered_training_seed_count": len(group),
            "complete_training_outcome_count": sum(
                row["training_outcome"]["status"] == "complete" for row in group
            ),
            "eligibility_count": eligible_count,
            "eligibility_rate_all_10_registered_seeds": eligible_count / 10.0,
            "eligible_and_estimable_count": included_count,
            "eligible_and_estimable_rate_all_10_registered_seeds": included_count
            / 10.0,
            "analysis_status_counts": dict(sorted(statuses.items())),
            "failed_seed_replacements": 0,
            "numeric_descriptive_summaries": _model_numeric_summaries(group, spec),
        }
    total_eligible = sum(bool(row["eligible_by_nmse_rule"]) for row in rows)
    total_included = sum(
        bool(row["included_in_primary_structural_summary"]) for row in rows
    )
    return {
        "schema_version": 1,
        "campaign_mode": CAMPAIGN_MODE,
        "campaign_scientific_identity": manifest["scientific_identity"],
        "scope": SCOPE,
        "protocol_revision": PROTOCOL_REVISION,
        "analysis_role": (
            "Ságodi_based_evaluation_with_project_defined_descriptive_"
            "carrier_normal_recovery_not_CA_LRU_method"
        ),
        "smoke": bool(spec.smoke),
        "registered_training_outcome_count": len(rows),
        "verified_analysis_receipt_count": len(child_receipts),
        "eligibility_count": total_eligible,
        "eligibility_rate_all_60_registered_seeds": total_eligible / 60.0,
        "eligible_and_estimable_count": total_included,
        "eligible_and_estimable_rate_all_60_registered_seeds": total_included
        / 60.0,
        "failed_seed_replacements": 0,
        "main_binding": main.binding,
        "model_summaries": model_summaries,
        "runs": rows,
    }


def _finalize(
    root: Path,
    main: VerifiedMain,
    manifest: Mapping[str, Any],
    spec: PrimaryAnalysisSpec,
) -> dict[str, Any]:
    summary = aggregate_results(root, main, manifest, spec)
    atomic_json(root / SUMMARY, summary)
    atomic_json(
        root / COMPLETE,
        {
            "schema_version": 1,
            "status": "complete",
            "campaign_scientific_identity": manifest["scientific_identity"],
            "registered_training_outcome_count": EXPECTED_RUN_COUNT,
            "verified_analysis_receipt_count": EXPECTED_RUN_COUNT,
            "failed_seed_replacements": 0,
            "summary_sha256": sha256_file(root / SUMMARY),
        },
    )
    child_receipts = [
        _analysis_output(root, run) / "completion_receipt.json" for run in main.runs
    ]
    write_completion_receipt(
        root / COMPLETION_RECEIPT,
        job_id="sagodi_primary_analysis_v3_campaign_complete",
        artifacts=[root / MANIFEST, root / SUMMARY, root / COMPLETE, *child_receipts],
        metadata={
            "campaign_scientific_identity": manifest["scientific_identity"],
            "protocol_fingerprint": manifest["protocol_canonical_fingerprint"],
            "run_id": "sagodi_primary_analysis_v3",
            "stage": "finalization",
            "run_count": EXPECTED_RUN_COUNT,
            "failed_seed_replacements": 0,
        },
    )
    valid, reason = verify_campaign_completion(root, main, manifest, spec)
    if not valid:
        raise RuntimeError(f"final recursive analysis receipt failed: {reason}")
    return summary


def verify_campaign_completion(
    root: Path,
    main: VerifiedMain,
    manifest: Mapping[str, Any],
    spec: PrimaryAnalysisSpec,
) -> tuple[bool, str]:
    try:
        complete = _require_mapping(strict_json_load(root / COMPLETE), "COMPLETE")
        summary = _require_mapping(strict_json_load(root / SUMMARY), "summary")
        for run in main.runs:
            valid, reason, _ = verify_analysis_output(
                _analysis_output(root, run), run, main, manifest, spec
            )
            if not valid:
                return False, f"invalid nested analysis {run.run_id}: {reason}"
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        return False, f"analysis completion is unreadable: {error}"
    valid, reason = verify_completion_receipt(
        root / COMPLETION_RECEIPT,
        expected_job_id="sagodi_primary_analysis_v3_campaign_complete",
        expected_metadata={
            "campaign_scientific_identity": manifest["scientific_identity"],
            "protocol_fingerprint": manifest["protocol_canonical_fingerprint"],
            "run_id": "sagodi_primary_analysis_v3",
            "stage": "finalization",
            "run_count": EXPECTED_RUN_COUNT,
            "failed_seed_replacements": 0,
        },
    )
    if not valid:
        return False, reason
    if complete.get("summary_sha256") != sha256_file(root / SUMMARY):
        return False, "COMPLETE summary binding changed"
    if complete.get("verified_analysis_receipt_count") != EXPECTED_RUN_COUNT:
        return False, "COMPLETE analysis count differs"
    if summary.get("registered_training_outcome_count") != EXPECTED_RUN_COUNT:
        return False, "summary denominator differs"
    if summary.get("failed_seed_replacements") != 0:
        return False, "summary contains seed replacement"
    return True, "verified all 60 nested Ságodi-primary analysis receipts"


def run_primary_analysis_campaign(
    *,
    main_root: Path,
    artifact_root: Path,
    python: str,
    gpus: Sequence[int],
    smoke: bool = False,
) -> Path:
    main = verify_main_artifacts(main_root)
    if bool(main.manifest.get("smoke")) is not bool(smoke):
        raise RuntimeError(
            "--smoke must be used with a smoke main-training fixture; full and smoke "
            "artifacts cannot be mixed"
        )
    root = Path(artifact_root).expanduser().resolve()
    python = _resolve_python(python)
    gpus = _validated_gpu_ids(gpus)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES; --gpus uses physical GPU ids")
    repo_root = Path(__file__).resolve().parents[2]
    git_state = _git_state(repo_root)
    _require_exact_main_commit(git_state, main, smoke=smoke)
    spec = _analysis_spec(smoke=smoke)
    spec.validate()
    _prepare_root(root, main, spec, smoke=smoke)
    lock = _acquire_campaign_lock(root)
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"primary analysis interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        manifest = _build_manifest(
            root, main, spec, python=python, gpus=gpus, smoke=smoke
        )
        _write_or_check_manifest(root / MANIFEST, manifest)
        _verify_campaign_inputs(root, manifest, main)
        complete, _ = verify_campaign_completion(root, main, manifest, spec)
        if complete:
            return root
        status: dict[str, Any] = {
            "schema_version": 1,
            "campaign_mode": CAMPAIGN_MODE,
            "scientific_identity": manifest["scientific_identity"],
            "scope": SCOPE,
            "smoke": bool(smoke),
            "stage": "sagodi_primary_analysis",
            "jobs": {},
        }
        if (root / STATUS).exists():
            existing = strict_json_load(root / STATUS)
            if isinstance(existing, Mapping):
                status["jobs"] = dict(existing.get("jobs", {}))
        atomic_json(root / STATUS, status)
        _run_jobs(root, main, manifest, spec, python, gpus, status)
        _verify_campaign_inputs(root, manifest, main)
        status["stage"] = "finalizing"
        atomic_json(root / STATUS, status)
        summary = _finalize(root, main, manifest, spec)
        status["stage"] = "complete"
        status["verified_analysis_receipt_count"] = summary[
            "verified_analysis_receipt_count"
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
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = run_primary_analysis_campaign(
        main_root=args.main_root,
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
