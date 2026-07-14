"""Join frozen Ságodi dynamics and engineering-utility results descriptively.

This stage is deliberately downstream of both completed 60-run campaigns.  It
does not train, select, gate, rank, or exclude a model.  Every registered
``(model_id, model_seed)`` pair remains visible, including pairs for which a
Ságodi quantity is ineligible or structurally not estimable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import (
    atomic_bytes,
    atomic_json,
    canonical_hash,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from .engineering_benefit_campaign import (
    CAMPAIGN_TYPE as ENGINEERING_CAMPAIGN_TYPE,
    COMPLETE as ENGINEERING_COMPLETE,
    COMPLETION_RECEIPT as ENGINEERING_COMPLETION_RECEIPT,
    EXPECTED_RUNS as ENGINEERING_EXPECTED_RUNS,
    EXPECTED_UTILITY_METRIC_KEYS,
    MANIFEST as ENGINEERING_MANIFEST,
    SUMMARY as ENGINEERING_SUMMARY,
    verify_engineering_campaign_completion,
    verify_main_parent as verify_engineering_main_parent,
)
from .engineering_benefit_runner import MODEL_IDS, MODEL_SEEDS
from .orchestrate import _acquire_campaign_lock, _git_state, _source_hashes
from .primary_analysis_campaign import (
    CAMPAIGN_MODE as PRIMARY_CAMPAIGN_MODE,
    COMPLETE as PRIMARY_COMPLETE,
    COMPLETION_RECEIPT as PRIMARY_COMPLETION_RECEIPT,
    EXPECTED_RUN_COUNT as PRIMARY_EXPECTED_RUNS,
    MANIFEST as PRIMARY_MANIFEST,
    SUMMARY as PRIMARY_SUMMARY,
    verify_campaign_completion as verify_primary_campaign_completion,
    verify_main_artifacts as verify_primary_main_artifacts,
)
from .sagodi_primary_runner import PrimaryAnalysisSpec


SCHEMA_VERSION = 1
CAMPAIGN_TYPE = "calru_dynamics_utility_association_v1"
ANALYSIS_ROLE = "strictly_descriptive_cross_campaign_association_not_causal_evidence"
EXPECTED_PAIRS = 60
FREEZE_CANONICAL_FINGERPRINT = (
    "dbaa5b7d76af9c22768ec07b090686ccccba2248c0a9064b85433a50d8ac1662"
)

ROOT_MARKER = ".calru_dynamics_utility_association_v1_root.json"
FREEZE_COPY = "dynamics_utility_association_freeze.json"
MANIFEST = "association_manifest.json"
SUMMARY = "dynamics_utility_association_summary.json"
COMPLETE = "COMPLETE"
COMPLETION_RECEIPT = "completion_receipt.json"


@dataclass(frozen=True)
class FrozenAssociation:
    association_id: str
    x_source: str
    x_label: str
    x_beneficial_direction: str
    y_source: str
    y_label: str
    y_beneficial_direction: str
    x_interpretation: str | None = None


@dataclass(frozen=True)
class VerifiedParent:
    root: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    binding: dict[str, Any]


_MISSING = object()
_PRIMARY_PATHS: dict[str, tuple[str, ...]] = {
    "timescale_gap_vs_temporal_16T_error": (
        "sagodi_metrics",
        "top_two_real_part_gap",
        "mean",
    ),
    "uniform_flow_vs_clean_16T_blank_retention_error": (
        "sagodi_metrics",
        "uniform_flow_norm",
    ),
    "effective_capacity_vs_temporal_16T_error": (
        "sagodi_metrics",
        "asymptotic_capacity",
        "effective_basin_count",
    ),
    "basin_entropy_vs_temporal_16T_error": (
        "sagodi_metrics",
        "asymptotic_capacity",
        "shannon_entropy_nats",
    ),
}


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _exact_keys() -> tuple[tuple[str, int], ...]:
    return tuple((model_id, seed) for model_id in MODEL_IDS for seed in MODEL_SEEDS)


def load_association_freeze(
    path: Path | str,
) -> tuple[dict[str, Any], tuple[FrozenAssociation, ...]]:
    """Load the byte-independent canonical v1 freeze, failing closed on edits."""

    payload = strict_json_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError("association freeze must be a JSON object")
    if canonical_hash(payload) != FREEZE_CANONICAL_FINGERPRINT:
        raise ValueError("association freeze differs from preregistered v1 semantics")
    if payload.get("schema_version") != 1 or payload.get("freeze_id") != CAMPAIGN_TYPE:
        raise ValueError("association freeze identity differs")
    if payload.get("analysis_role") != ANALYSIS_ROLE:
        raise ValueError("association analysis role differs")
    parent = _require_mapping(payload.get("parent_contract"), "parent contract")
    checks = {
        "shared_main_campaign_type": "sagodi_primary_main_v3",
        "primary_campaign_mode": PRIMARY_CAMPAIGN_MODE,
        "engineering_campaign_type": ENGINEERING_CAMPAIGN_TYPE,
        "engineering_freeze_id": "calru_engineering_benefit_v1",
        "required_completed_parent_runs_each": EXPECTED_PAIRS,
        "required_model_order": list(MODEL_IDS),
        "required_model_seeds": list(MODEL_SEEDS),
        "join_key": ["model_id", "model_seed"],
    }
    for key, expected in checks.items():
        if parent.get(key) != expected:
            raise ValueError(f"association parent contract changed: {key}")
    raw_associations = payload.get("associations")
    if not isinstance(raw_associations, list) or len(raw_associations) != 4:
        raise ValueError("association freeze must contain exactly four associations")
    associations: list[FrozenAssociation] = []
    for raw in raw_associations:
        item = _require_mapping(raw, "association")
        association = FrozenAssociation(**dict(item))
        if association.association_id not in _PRIMARY_PATHS:
            raise ValueError(f"unknown frozen association: {association.association_id}")
        if association.x_beneficial_direction not in {"higher", "lower"}:
            raise ValueError("x beneficial direction must be higher or lower")
        if association.y_beneficial_direction not in {"higher", "lower"}:
            raise ValueError("y beneficial direction must be higher or lower")
        metric_prefix = "engineering.runs[].utility_metrics."
        if not association.y_source.startswith(metric_prefix):
            raise ValueError("engineering association source is malformed")
        if association.y_source[len(metric_prefix) :] not in EXPECTED_UTILITY_METRIC_KEYS:
            raise ValueError("association requests an unregistered engineering metric")
        associations.append(association)
    if tuple(item.association_id for item in associations) != tuple(_PRIMARY_PATHS):
        raise ValueError("frozen association order differs")
    return dict(payload), tuple(associations)


def _row_index(
    rows: Any, *, label: str
) -> dict[tuple[str, int], Mapping[str, Any]]:
    if not isinstance(rows, list) or len(rows) != EXPECTED_PAIRS:
        raise RuntimeError(f"{label} must contain exactly {EXPECTED_PAIRS} rows")
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    observed_order: list[tuple[str, int]] = []
    for raw in rows:
        item = _require_mapping(raw, f"{label} row")
        model_id = str(item.get("model_id"))
        try:
            model_seed = int(item.get("model_seed"))
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{label} model_seed is malformed") from error
        key = (model_id, model_seed)
        if key in result:
            raise RuntimeError(f"{label} contains duplicate pair {key}")
        result[key] = item
        observed_order.append(key)
    expected = _exact_keys()
    if set(result) != set(expected):
        raise RuntimeError(f"{label} is not the exact frozen six-by-ten product")
    if tuple(observed_order) != expected:
        raise RuntimeError(f"{label} pair order differs from the frozen order")
    return result


def verify_primary_parent(
    primary_root: Path | str, main_root: Path | str
) -> VerifiedParent:
    """Recursively verify the primary campaign and all 60 nested analyses."""

    root = Path(primary_root).expanduser().resolve(strict=True)
    main = verify_primary_main_artifacts(main_root)
    manifest = dict(
        _require_mapping(strict_json_load(root / PRIMARY_MANIFEST), "primary manifest")
    )
    summary = dict(
        _require_mapping(strict_json_load(root / PRIMARY_SUMMARY), "primary summary")
    )
    if manifest.get("campaign_mode") != PRIMARY_CAMPAIGN_MODE:
        raise RuntimeError("primary parent campaign mode differs")
    if manifest.get("smoke") is not False:
        raise RuntimeError("association stage requires the full primary campaign")
    signed = _require_mapping(
        manifest.get("scientific_identity_payload"), "primary scientific identity"
    )
    if canonical_hash(signed) != manifest.get("scientific_identity"):
        raise RuntimeError("primary campaign scientific identity does not verify")
    if manifest.get("main_binding") != main.binding:
        raise RuntimeError("primary campaign no longer binds the verified main campaign")
    code = _require_mapping(manifest.get("code"), "primary campaign code binding")
    if (
        code.get("worktree_dirty") is not False
        or code.get("code_commit") != main.binding.get("main_code_commit")
    ):
        raise RuntimeError("primary campaign was not run at the exact clean main commit")
    spec = PrimaryAnalysisSpec()
    if manifest.get("analysis_spec") != asdict(spec):
        raise RuntimeError("primary analysis spec differs from the full frozen spec")
    valid, reason = verify_primary_campaign_completion(root, main, manifest, spec)
    if not valid:
        raise RuntimeError(f"primary campaign recursive verification failed: {reason}")
    if summary.get("campaign_scientific_identity") != manifest["scientific_identity"]:
        raise RuntimeError("primary summary scientific identity differs")
    if summary.get("verified_analysis_receipt_count") != PRIMARY_EXPECTED_RUNS:
        raise RuntimeError("primary summary does not bind 60 analysis receipts")
    if summary.get("failed_seed_replacements") != 0:
        raise RuntimeError("primary summary contains forbidden seed replacement")
    _row_index(summary.get("runs"), label="primary summary")
    binding = {
        "schema_version": 1,
        "campaign_mode": PRIMARY_CAMPAIGN_MODE,
        "scientific_identity": manifest["scientific_identity"],
        "manifest_sha256": sha256_file(root / PRIMARY_MANIFEST),
        "summary_sha256": sha256_file(root / PRIMARY_SUMMARY),
        "complete_sha256": sha256_file(root / PRIMARY_COMPLETE),
        "completion_receipt_sha256": sha256_file(root / PRIMARY_COMPLETION_RECEIPT),
        "verified_run_count": EXPECTED_PAIRS,
        "main_binding": main.binding,
    }
    return VerifiedParent(root, manifest, summary, binding)


def verify_engineering_parent(
    engineering_root: Path | str,
    main_root: Path | str,
    selector_root: Path | str,
) -> VerifiedParent:
    """Recursively verify the engineering campaign and all 60 child arrays."""

    root = Path(engineering_root).expanduser().resolve(strict=True)
    main = verify_engineering_main_parent(main_root, selector_root)
    manifest = dict(
        _require_mapping(
            strict_json_load(root / ENGINEERING_MANIFEST), "engineering manifest"
        )
    )
    summary = dict(
        _require_mapping(
            strict_json_load(root / ENGINEERING_SUMMARY), "engineering summary"
        )
    )
    if manifest.get("campaign_type") != ENGINEERING_CAMPAIGN_TYPE:
        raise RuntimeError("engineering parent campaign type differs")
    if manifest.get("smoke") is not False:
        raise RuntimeError("association stage requires the full engineering campaign")
    signed = _require_mapping(
        manifest.get("scientific_identity_payload"), "engineering scientific identity"
    )
    if canonical_hash(signed) != manifest.get("scientific_identity"):
        raise RuntimeError("engineering campaign scientific identity does not verify")
    if manifest.get("parent_main") != main.binding:
        raise RuntimeError("engineering campaign no longer binds the verified main campaign")
    code = _require_mapping(manifest.get("code"), "engineering campaign code binding")
    if (
        code.get("worktree_dirty") is not False
        or code.get("code_commit") != main.binding.get("main_code_commit")
    ):
        raise RuntimeError("engineering campaign was not run at the exact clean main commit")
    valid, reason = verify_engineering_campaign_completion(root, manifest, main.runs)
    if not valid:
        raise RuntimeError(f"engineering recursive verification failed: {reason}")
    if summary.get("campaign_scientific_identity") != manifest["scientific_identity"]:
        raise RuntimeError("engineering summary scientific identity differs")
    if summary.get("verified_run_count") != ENGINEERING_EXPECTED_RUNS:
        raise RuntimeError("engineering summary does not bind 60 child receipts")
    if summary.get("failed_seed_replacements") != 0:
        raise RuntimeError("engineering summary contains forbidden seed replacement")
    if summary.get("excluded_seed_count") != 0:
        raise RuntimeError("engineering summary contains forbidden seed exclusion")
    if summary.get("utility_metric_keys") != list(EXPECTED_UTILITY_METRIC_KEYS):
        raise RuntimeError("engineering utility metric schema differs")
    _row_index(summary.get("runs"), label="engineering summary")
    binding = {
        "schema_version": 1,
        "campaign_type": ENGINEERING_CAMPAIGN_TYPE,
        "scientific_identity": manifest["scientific_identity"],
        "manifest_sha256": sha256_file(root / ENGINEERING_MANIFEST),
        "summary_sha256": sha256_file(root / ENGINEERING_SUMMARY),
        "complete_sha256": sha256_file(root / ENGINEERING_COMPLETE),
        "completion_receipt_sha256": sha256_file(
            root / ENGINEERING_COMPLETION_RECEIPT
        ),
        "freeze_canonical_fingerprint": manifest["freeze_canonical_fingerprint"],
        "verified_run_count": EXPECTED_PAIRS,
        "parent_main": main.binding,
    }
    return VerifiedParent(root, manifest, summary, binding)


def _shared_main_binding(
    primary: VerifiedParent, engineering: VerifiedParent
) -> dict[str, Any]:
    left = _require_mapping(primary.binding["main_binding"], "primary main binding")
    right = _require_mapping(
        engineering.binding["parent_main"], "engineering main binding"
    )
    root_pairs = {
        "scientific_identity": (
            left.get("campaign_scientific_identity"),
            right.get("scientific_identity"),
        ),
        "manifest_sha256": (left.get("manifest_sha256"), right.get("manifest_sha256")),
        "summary_sha256": (left.get("summary_sha256"), right.get("summary_sha256")),
        "complete_sha256": (left.get("complete_sha256"), right.get("complete_sha256")),
        "completion_receipt_sha256": (
            left.get("completion_receipt_sha256"),
            right.get("completion_receipt_sha256"),
        ),
        "main_code_commit": (
            left.get("main_code_commit"),
            right.get("main_code_commit"),
        ),
    }
    for label, (first, second) in root_pairs.items():
        if first is None or first != second:
            raise RuntimeError(f"primary and engineering main bindings differ: {label}")

    primary_rows = _row_index(primary.summary.get("runs"), label="primary summary")
    engineering_nested = right.get("nested_training_artifacts")
    if not isinstance(engineering_nested, list) or len(engineering_nested) != EXPECTED_PAIRS:
        raise RuntimeError("engineering main binding lacks 60 nested training artifacts")
    nested_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for raw in engineering_nested:
        item = _require_mapping(raw, "engineering nested training artifact")
        key = (str(item.get("model_id")), int(item.get("model_seed", -1)))
        if key in nested_by_key:
            raise RuntimeError(f"duplicate engineering nested main binding: {key}")
        nested_by_key[key] = item
    if set(nested_by_key) != set(_exact_keys()):
        raise RuntimeError("engineering nested main binding is not the frozen 60 pairs")
    for key in _exact_keys():
        training = _require_mapping(
            primary_rows[key].get("training_outcome"), "primary training outcome"
        )
        nested = nested_by_key[key]
        for field in ("checkpoint_sha256", "training_receipt_sha256"):
            if training.get(field) != nested.get(field):
                raise RuntimeError(f"cross-parent nested binding differs for {key}: {field}")
    return {
        "schema_version": 1,
        "campaign_type": "sagodi_primary_main_v3",
        "scientific_identity": root_pairs["scientific_identity"][0],
        "protocol_canonical_fingerprint": left.get(
            "protocol_canonical_fingerprint"
        ),
        "manifest_sha256": root_pairs["manifest_sha256"][0],
        "summary_sha256": root_pairs["summary_sha256"][0],
        "complete_sha256": root_pairs["complete_sha256"][0],
        "completion_receipt_sha256": root_pairs[
            "completion_receipt_sha256"
        ][0],
        "main_code_commit": root_pairs["main_code_commit"][0],
        "verified_nested_training_receipt_count": EXPECTED_PAIRS,
        "exact_join_key_product_verified": True,
    }


def _validate_launch_code_identity(
    git_state: Mapping[str, Any], shared_main: Mapping[str, Any]
) -> None:
    """Require the clean exact commit shared by training and both parents."""

    if git_state.get("worktree_dirty") is not False:
        raise RuntimeError("full association stage requires a clean committed worktree")
    if git_state.get("code_commit") != shared_main.get("main_code_commit"):
        raise RuntimeError(
            "full association stage must use the exact selector/main/analysis commit"
        )


def _lookup(value: Any, path: Sequence[str]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _finite_or_reason(value: Any, *, prefix: str) -> tuple[float | None, str | None]:
    if value is _MISSING or value is None:
        return None, f"{prefix}_metric_missing"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{prefix}_metric_nonfinite"
    number = float(value)
    if not math.isfinite(number):
        return None, f"{prefix}_metric_nonfinite"
    return number, None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size < 2:
        return None
    if method == "spearman_average_rank":
        left = _average_ranks(left)
        right = _average_ranks(right)
    elif method != "pearson_product_moment":
        raise ValueError(f"unknown association method: {method}")
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    if denominator == 0.0 or not math.isfinite(denominator):
        return None
    result = float(np.dot(left, right) / denominator)
    if not math.isfinite(result):
        return None
    return max(-1.0, min(1.0, result))


def _bootstrap_seed(
    base_seed: int, association_id: str, scope: str, method: str
) -> int:
    material = f"{int(base_seed)}|{association_id}|{scope}|{method}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def _bootstrap_interval(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    association_id: str,
    scope: str,
    base_seed: int,
    resamples: int,
) -> dict[str, Any]:
    seed = _bootstrap_seed(base_seed, association_id, scope, method)
    rng = np.random.default_rng(seed)
    coefficients: list[float] = []
    for _ in range(int(resamples)):
        index = rng.integers(0, x.size, size=x.size)
        coefficient = _correlation(x[index], y[index], method)
        if coefficient is not None:
            coefficients.append(coefficient)
    if coefficients:
        lower, upper = np.quantile(
            np.asarray(coefficients, dtype=np.float64), (0.025, 0.975), method="linear"
        )
        interval: list[float] | None = [float(lower), float(upper)]
    else:
        interval = None
    return {
        "paired_resamples_requested": int(resamples),
        "valid_bootstrap_count": len(coefficients),
        "omitted_nonfinite_bootstrap_count": int(resamples) - len(coefficients),
        "seed": seed,
        "raw_percentile_95_interval": interval,
        "interpretation": "descriptive_resampling_interval_not_a_confirmatory_test",
    }


def _association_statistic(
    rows: Sequence[Mapping[str, Any]],
    association: FrozenAssociation,
    *,
    scope: str,
    method: str,
    base_seed: int,
    resamples: int,
    minimum_pairs: int,
) -> dict[str, Any]:
    included = [item for item in rows if item["included"]]
    result: dict[str, Any] = {
        "method": method,
        "registered_pair_count": len(rows),
        "complete_pair_count": len(included),
        "raw_correlation": None,
        "benefit_aligned_correlation": None,
        "bootstrap": None,
    }
    if len(included) < minimum_pairs:
        result["status"] = "insufficient_complete_pairs"
        return result
    x = np.asarray([item["x_value_or_null"] for item in included], dtype=np.float64)
    y = np.asarray([item["y_value_or_null"] for item in included], dtype=np.float64)
    x_constant = bool(np.all(x == x[0]))
    y_constant = bool(np.all(y == y[0]))
    if x_constant or y_constant:
        if x_constant and y_constant:
            result["status"] = "constant_x_and_y"
        elif x_constant:
            result["status"] = "constant_x"
        else:
            result["status"] = "constant_y"
        return result
    raw = _correlation(x, y, method)
    if raw is None:
        result["status"] = "non_estimable_numeric_correlation"
        return result
    x_sign = 1.0 if association.x_beneficial_direction == "higher" else -1.0
    y_sign = 1.0 if association.y_beneficial_direction == "higher" else -1.0
    alignment = x_sign * y_sign
    bootstrap = _bootstrap_interval(
        x,
        y,
        method=method,
        association_id=association.association_id,
        scope=scope,
        base_seed=base_seed,
        resamples=resamples,
    )
    raw_interval = bootstrap["raw_percentile_95_interval"]
    if raw_interval is None:
        aligned_interval = None
    elif alignment > 0:
        aligned_interval = list(raw_interval)
    else:
        aligned_interval = [-raw_interval[1], -raw_interval[0]]
    bootstrap["benefit_aligned_percentile_95_interval"] = aligned_interval
    result.update(
        {
            "status": "estimated",
            "raw_correlation": raw,
            "benefit_alignment_multiplier": alignment,
            "benefit_aligned_correlation": alignment * raw,
            "benefit_aligned_sign_interpretation": (
                "positive_means_better_dynamics_co_occurs_with_better_utility"
            ),
            "bootstrap": bootstrap,
        }
    )
    return result


def _association_rows(
    primary_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
    engineering_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
    association: FrozenAssociation,
) -> list[dict[str, Any]]:
    y_key = association.y_source.split("utility_metrics.", 1)[1]
    rows: list[dict[str, Any]] = []
    for model_id, seed in _exact_keys():
        primary = primary_by_key[(model_id, seed)]
        engineering = engineering_by_key[(model_id, seed)]
        eligible = primary.get("eligible_by_nmse_rule") is True
        estimable = primary.get("structurally_estimable") is True
        primary_included = primary.get("included_in_primary_structural_summary") is True
        if primary_included != (eligible and estimable):
            raise RuntimeError(f"primary inclusion flags contradict for {(model_id, seed)}")
        if engineering.get("seed_excluded") is not False:
            raise RuntimeError(f"engineering seed exclusion changed for {(model_id, seed)}")
        expected_join = f"{model_id}::seed{seed:02d}"
        if engineering.get("join_key") != expected_join:
            raise RuntimeError(f"engineering join key differs for {(model_id, seed)}")
        reasons: list[str] = []
        if not eligible:
            reasons.append("primary_ineligible")
        if primary.get("analysis_status") == "structural_analysis_not_estimable":
            reasons.append("primary_structurally_not_estimable")
        raw_x = _lookup(primary, _PRIMARY_PATHS[association.association_id])
        x_value, x_reason = _finite_or_reason(raw_x, prefix="primary")
        if x_reason is not None:
            reasons.append(x_reason)
        utility = _require_mapping(
            engineering.get("utility_metrics"), "engineering utility metrics"
        )
        if set(utility) != set(EXPECTED_UTILITY_METRIC_KEYS):
            raise RuntimeError(f"engineering metric keys differ for {(model_id, seed)}")
        utility_status = _require_mapping(
            engineering.get("utility_metric_status"),
            "engineering utility metric status",
        )
        if set(utility_status) != set(EXPECTED_UTILITY_METRIC_KEYS):
            raise RuntimeError(
                f"engineering metric-status keys differ for {(model_id, seed)}"
            )
        raw_y = utility[y_key] if y_key in utility else _MISSING
        registered_status = utility_status.get(y_key)
        if raw_y is None and registered_status == "engineering_metric_nonfinite":
            y_value, y_reason = None, "engineering_metric_nonfinite"
        else:
            if registered_status != "complete":
                raise RuntimeError(
                    f"engineering utility status contradicts value for "
                    f"{(model_id, seed, y_key)}"
                )
            y_value, y_reason = _finite_or_reason(raw_y, prefix="engineering")
        if y_reason is not None:
            reasons.append(y_reason)
        included = primary_included and x_value is not None and y_value is not None
        rows.append(
            {
                "model_id": model_id,
                "model_seed": seed,
                "join_key": expected_join,
                "primary_analysis_status": primary.get("analysis_status"),
                "primary_eligible": eligible,
                "primary_estimable": estimable,
                "engineering_status": "complete_verified",
                "x_value_or_null": x_value,
                "y_value_or_null": y_value,
                "included": included,
                "missing_reasons": reasons,
            }
        )
    return rows


def build_association_summary(
    primary_summary: Mapping[str, Any],
    engineering_summary: Mapping[str, Any],
    freeze: Mapping[str, Any],
    associations: Sequence[FrozenAssociation],
    *,
    primary_binding: Mapping[str, Any] | None = None,
    engineering_binding: Mapping[str, Any] | None = None,
    shared_main_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build all frozen associations while preserving the all-60 denominator."""

    primary_by_key = _row_index(primary_summary.get("runs"), label="primary summary")
    engineering_by_key = _row_index(
        engineering_summary.get("runs"), label="engineering summary"
    )
    statistics = _require_mapping(freeze.get("statistics"), "statistics freeze")
    bootstrap = _require_mapping(statistics.get("bootstrap"), "bootstrap freeze")
    minimum_pairs = int(statistics["minimum_complete_pairs"])
    base_seed = int(bootstrap["base_seed"])
    resamples = int(bootstrap["paired_resamples_with_replacement"])
    methods = tuple(str(value) for value in statistics["methods"])

    association_outputs: list[dict[str, Any]] = []
    for association in associations:
        rows = _association_rows(
            primary_by_key, engineering_by_key, association
        )
        included_count = sum(bool(item["included"]) for item in rows)
        reason_counts = Counter(
            reason for item in rows for reason in item["missing_reasons"]
        )
        scopes: dict[str, Any] = {}
        scope_rows = [("pooled_all_six_models", rows)] + [
            (
                f"model={model_id}",
                [item for item in rows if item["model_id"] == model_id],
            )
            for model_id in MODEL_IDS
        ]
        for scope, selected in scope_rows:
            scopes[scope] = {
                "registered_pair_count": len(selected),
                "complete_pair_count": sum(
                    bool(item["included"]) for item in selected
                ),
                "statistics": {
                    method: _association_statistic(
                        selected,
                        association,
                        scope=scope,
                        method=method,
                        base_seed=base_seed,
                        resamples=resamples,
                        minimum_pairs=minimum_pairs,
                    )
                    for method in methods
                },
            }
        association_outputs.append(
            {
                "association_id": association.association_id,
                "x_source": association.x_source,
                "x_label": association.x_label,
                "x_interpretation": association.x_interpretation,
                "x_beneficial_direction": association.x_beneficial_direction,
                "y_source": association.y_source,
                "y_label": association.y_label,
                "y_beneficial_direction": association.y_beneficial_direction,
                "registered_pair_count": EXPECTED_PAIRS,
                "included_pair_count": included_count,
                "missing_pair_count": EXPECTED_PAIRS - included_count,
                "missing_reason_counts": {
                    reason: int(reason_counts.get(reason, 0))
                    for reason in freeze["missingness_accounting"]["reason_taxonomy"]
                },
                "rows": rows,
                "scopes": scopes,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_type": CAMPAIGN_TYPE,
        "analysis_role": ANALYSIS_ROLE,
        "interpretation": (
            "strictly_descriptive_associations; no_causal_or_binary_gate_claim"
        ),
        "registered_join_pair_count": EXPECTED_PAIRS,
        "model_order": list(MODEL_IDS),
        "model_seeds": list(MODEL_SEEDS),
        "failed_seed_replacements": 0,
        "excluded_registered_pairs": 0,
        "freeze_canonical_fingerprint": canonical_hash(freeze),
        "primary_parent": dict(primary_binding or {}),
        "engineering_parent": dict(engineering_binding or {}),
        "shared_main_binding": dict(shared_main_binding or {}),
        "pooled_interpretation_warning": (
            "pooled_associations_mix_between_model_and_within_model_variation"
        ),
        "within_model_interpretation_warning": (
            "ten_seed_descriptive_association_with_no_inferential_claim"
        ),
        "associations": association_outputs,
    }


def _prepare_root(
    root: Path,
    freeze_source: Path,
    freeze: Mapping[str, Any],
    primary: VerifiedParent,
    engineering: VerifiedParent,
    shared_main: Mapping[str, Any],
) -> None:
    marker_payload = {
        "schema_version": 1,
        "campaign_type": CAMPAIGN_TYPE,
        "freeze_source_sha256": sha256_file(freeze_source),
        "freeze_canonical_fingerprint": canonical_hash(freeze),
        "primary_parent": primary.binding,
        "engineering_parent": engineering.binding,
        "shared_main_binding": dict(shared_main),
    }
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ROOT_MARKER
    if marker.exists():
        if strict_json_load(marker) != marker_payload:
            raise RuntimeError("association root marker differs")
    else:
        if any(root.iterdir()):
            raise RuntimeError("association artifact root is nonempty and unmarked")
        atomic_json(marker, marker_payload)
    copy = root / FREEZE_COPY
    if copy.exists():
        if copy.read_bytes() != freeze_source.read_bytes():
            raise RuntimeError("immutable association freeze copy differs")
    else:
        atomic_bytes(copy, freeze_source.read_bytes())


def _build_manifest(
    *,
    freeze: Mapping[str, Any],
    freeze_copy: Path,
    associations: Sequence[FrozenAssociation],
    primary: VerifiedParent,
    engineering: VerifiedParent,
    shared_main: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    payload = {
        "campaign_type": CAMPAIGN_TYPE,
        "analysis_role": ANALYSIS_ROLE,
        "causal_claims": False,
        "binary_gates": False,
        "expected_direction_pass_thresholds": False,
        "registered_pair_count": EXPECTED_PAIRS,
        "association_ids": [item.association_id for item in associations],
        "freeze_file": FREEZE_COPY,
        "freeze_file_sha256": sha256_file(freeze_copy),
        "freeze_canonical_fingerprint": canonical_hash(freeze),
        "primary_parent": primary.binding,
        "engineering_parent": engineering.binding,
        "shared_main_binding": dict(shared_main),
        "code": _git_state(repo_root),
        "source_hashes": _source_hashes(repo_root, Path(__file__).resolve().parent),
    }
    return {
        "schema_version": 1,
        **payload,
        "scientific_identity_payload": payload,
        "scientific_identity": canonical_hash(payload),
    }


def _receipt_metadata(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "campaign_scientific_identity": manifest["scientific_identity"],
        "protocol_fingerprint": manifest["freeze_canonical_fingerprint"],
        "run_id": "dynamics_utility_association_v1",
        "stage": "finalization",
        "analysis_role": ANALYSIS_ROLE,
        "run_count": EXPECTED_PAIRS,
        "causal_claims": False,
        "binary_gates": False,
    }


def verify_association_completion(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[bool, str]:
    try:
        summary = _require_mapping(strict_json_load(root / SUMMARY), "summary")
        complete = _require_mapping(strict_json_load(root / COMPLETE), "COMPLETE")
        freeze = strict_json_load(root / FREEZE_COPY)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return False, f"association completion is unreadable: {error}"
    if canonical_hash(freeze) != manifest.get("freeze_canonical_fingerprint"):
        return False, "association freeze semantics changed"
    if summary.get("registered_join_pair_count") != EXPECTED_PAIRS:
        return False, "association summary denominator differs"
    if summary.get("excluded_registered_pairs") != 0:
        return False, "association summary excludes a registered pair"
    if summary.get("primary_parent") != manifest.get("primary_parent"):
        return False, "association primary-parent binding differs"
    if summary.get("engineering_parent") != manifest.get("engineering_parent"):
        return False, "association engineering-parent binding differs"
    associations = summary.get("associations")
    if not isinstance(associations, list) or len(associations) != 4:
        return False, "association summary does not contain four estimands"
    for item in associations:
        if not isinstance(item, Mapping) or item.get("registered_pair_count") != 60:
            return False, "association estimand denominator differs"
        if not isinstance(item.get("rows"), list) or len(item["rows"]) != 60:
            return False, "association estimand does not preserve 60 rows"
    if complete.get("summary_sha256") != sha256_file(root / SUMMARY):
        return False, "association COMPLETE summary hash differs"
    valid, reason = verify_completion_receipt(
        root / COMPLETION_RECEIPT,
        expected_job_id="calru_dynamics_utility_association_v1_complete",
        expected_metadata=_receipt_metadata(manifest),
    )
    if not valid:
        return False, reason
    return True, "verified frozen descriptive association output"


def run_dynamics_utility_association(
    *,
    primary_root: Path,
    engineering_root: Path,
    main_root: Path,
    selector_root: Path,
    freeze_path: Path,
    artifact_root: Path,
) -> Path:
    """Verify both parents, join exact keys, and atomically publish the summary."""

    repo_root = Path(__file__).resolve().parents[2]
    git_state = _git_state(repo_root)
    freeze_source = Path(freeze_path).expanduser().resolve(strict=True)
    freeze, associations = load_association_freeze(freeze_source)
    primary = verify_primary_parent(primary_root, main_root)
    engineering = verify_engineering_parent(
        engineering_root, main_root, selector_root
    )
    shared_main = _shared_main_binding(primary, engineering)
    _validate_launch_code_identity(git_state, shared_main)
    root = Path(artifact_root).expanduser().resolve()
    _prepare_root(
        root,
        freeze_source,
        freeze,
        primary,
        engineering,
        shared_main,
    )
    lock = _acquire_campaign_lock(root)
    try:
        manifest = _build_manifest(
            freeze=freeze,
            freeze_copy=root / FREEZE_COPY,
            associations=associations,
            primary=primary,
            engineering=engineering,
            shared_main=shared_main,
            repo_root=repo_root,
        )
        manifest_path = root / MANIFEST
        if manifest_path.exists():
            if strict_json_load(manifest_path) != manifest:
                raise RuntimeError("immutable association manifest differs")
        else:
            atomic_json(manifest_path, manifest)
        complete, _ = verify_association_completion(root, manifest)
        if complete:
            return root
        summary = build_association_summary(
            primary.summary,
            engineering.summary,
            freeze,
            associations,
            primary_binding=primary.binding,
            engineering_binding=engineering.binding,
            shared_main_binding=shared_main,
        )
        summary["campaign_scientific_identity"] = manifest["scientific_identity"]
        atomic_json(root / SUMMARY, summary)
        atomic_json(
            root / COMPLETE,
            {
                "schema_version": 1,
                "status": "complete",
                "campaign_scientific_identity": manifest["scientific_identity"],
                "registered_join_pair_count": EXPECTED_PAIRS,
                "excluded_registered_pairs": 0,
                "summary_sha256": sha256_file(root / SUMMARY),
            },
        )
        write_completion_receipt(
            root / COMPLETION_RECEIPT,
            job_id="calru_dynamics_utility_association_v1_complete",
            artifacts=[
                root / MANIFEST,
                root / FREEZE_COPY,
                root / SUMMARY,
                root / COMPLETE,
            ],
            metadata=_receipt_metadata(manifest),
        )
        valid, reason = verify_association_completion(root, manifest)
        if not valid:
            raise RuntimeError(f"association final verification failed: {reason}")
        return root
    finally:
        lock.release()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--engineering-root", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--selector-root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = run_dynamics_utility_association(
        primary_root=args.primary_root,
        engineering_root=args.engineering_root,
        main_root=args.main_root,
        selector_root=args.selector_root,
        freeze_path=args.freeze,
        artifact_root=args.artifact_root,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "artifact_root": str(root),
                "analysis_role": ANALYSIS_ROLE,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
