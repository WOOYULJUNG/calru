"""Strict freeze loader for the corrected v2 analysis-only overlay.

The ``calru_native_sagodi_ring_analysis_v2.yaml`` file intentionally contains
JSON.  JSON is a subset of YAML and lets us reject duplicate keys and
non-finite constants without adding a YAML dependency.

This module does *not* launch training or analysis.  It only validates and
fingerprints the immutable contract under which already-created v1 pilot
checkpoints may be reanalysed in a separate artifact root.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import strict_json_load


DEFAULT_ANALYSIS_FREEZE_PATH = Path(__file__).with_name(
    "calru_native_sagodi_ring_analysis_v2.yaml"
)

PARENT_TRAINING_BINDING = {
    "freeze_id": "calru_native_sagodi_ring_pilot_v1",
    "protocol_canonical_fingerprint": (
        "668867dfa36a4d6b4eb57fb63b61236f91c29756334f885f182bc9ed83ce1e1a"
    ),
    "protocol_file_sha256": (
        "49c000690fb0a2be14188fe4281264181e5195f194e17ced8530af9fc8005891"
    ),
    "campaign_id": "calru_native_sagodi_ring_pilot_v1-668867dfa36a",
    "scientific_identity": (
        "93f3466a7b978dece3e0b09d743f0ef2fbe421d9fd3ea48fdfb897099a6543c9"
    ),
    "manifest_sha256": (
        "204ec47c9b40105955a5697bade32974ac6c7ef0ebd401d9caa83365eced77fb"
    ),
}

REGISTERED_HORIZONS = (1, 5, 20, 100, 500, 1024)
PRIMARY_HORIZON = 500
PERTURBATION_FAMILIES = ("radial", "ambient_normal")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AnalysisFreezeError(ValueError):
    """Raised when the v2 reanalysis contract is malformed or contradictory."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisFreezeError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be an array",
    )
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    _require(not missing, f"{label} is missing keys: {missing}")
    _require(not extra, f"{label} has extra keys: {extra}")


def _finite_tree(value: Any, label: str = "analysis freeze") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label} contains a non-finite number")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{label}[{index}]")


def _exact(value: Any, required: Any, label: str) -> None:
    _require(value == required, f"{label} must be {required!r}, got {value!r}")


def _validate_threshold(
    value: Any,
    *,
    label: str,
    statistic: str,
    operator: str,
    threshold: float,
) -> None:
    item = _mapping(value, label)
    _exact_keys(item, {"statistic", "operator", "value"}, label)
    _exact(item["statistic"], statistic, f"{label}.statistic")
    _exact(item["operator"], operator, f"{label}.operator")
    _require(
        type(item["value"]) in (int, float),  # bool is deliberately excluded
        f"{label}.value must be numeric",
    )
    _exact(float(item["value"]), threshold, f"{label}.value")


def load_analysis_freeze(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the JSON-compatible YAML v2 analysis freeze."""

    freeze_path = Path(path) if path is not None else DEFAULT_ANALYSIS_FREEZE_PATH
    try:
        data = strict_json_load(freeze_path)
    except FileNotFoundError as exc:
        raise AnalysisFreezeError(f"analysis freeze not found: {freeze_path}") from exc
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise AnalysisFreezeError(
            f"invalid JSON-compatible analysis freeze {freeze_path}: {exc}"
        ) from exc
    _require(isinstance(data, dict), "analysis freeze root must be an object")
    validate_analysis_freeze(data)
    return data


def validate_analysis_freeze(freeze: Mapping[str, Any]) -> None:
    """Reject incomplete, ambiguous, or scientifically contradictory freezes."""

    root = _mapping(freeze, "analysis freeze")
    _finite_tree(root)
    _exact_keys(
        root,
        {
            "schema_version",
            "freeze_id",
            "freeze_status",
            "purpose",
            "parent_training",
            "claim_scope",
            "manifold",
            "c3_normal_recovery",
            "settling",
        },
        "analysis freeze",
    )
    _exact(root["schema_version"], "1.0.0", "schema_version")
    _exact(
        root["freeze_id"],
        "calru_native_sagodi_ring_analysis_v2",
        "freeze_id",
    )
    _exact(root["freeze_status"], "pilot_reanalysis_only", "freeze_status")

    purpose = _mapping(root["purpose"], "purpose")
    _exact_keys(
        purpose,
        {
            "operation",
            "creates_training_runs",
            "artifact_root_policy",
            "may_mutate_parent_artifacts",
        },
        "purpose",
    )
    _exact(
        purpose["operation"],
        "reanalyse_existing_immutable_checkpoints",
        "purpose.operation",
    )
    _exact(purpose["creates_training_runs"], False, "purpose.creates_training_runs")
    _exact(
        purpose["artifact_root_policy"],
        "separate_from_parent_campaign",
        "purpose.artifact_root_policy",
    )
    _exact(
        purpose["may_mutate_parent_artifacts"],
        False,
        "purpose.may_mutate_parent_artifacts",
    )

    parent = _mapping(root["parent_training"], "parent_training")
    _exact_keys(parent, set(PARENT_TRAINING_BINDING), "parent_training")
    for key, required in PARENT_TRAINING_BINDING.items():
        _exact(parent[key], required, f"parent_training.{key}")
    for key in (
        "protocol_canonical_fingerprint",
        "protocol_file_sha256",
        "scientific_identity",
        "manifest_sha256",
    ):
        _require(
            isinstance(parent[key], str) and _SHA256_RE.fullmatch(parent[key]) is not None,
            f"parent_training.{key} must be a lowercase SHA-256 hex digest",
        )
    _require(
        parent["campaign_id"].endswith(
            "-" + parent["protocol_canonical_fingerprint"][:12]
        ),
        "parent campaign_id contradicts protocol fingerprint",
    )

    scope = _mapping(root["claim_scope"], "claim_scope")
    _exact_keys(
        scope,
        {
            "claim",
            "confirmatory",
            "may_support_l3_claim",
            "may_support_c4_claim",
            "may_support_main_seed_claim",
            "seed_scope",
            "causal_interpretation",
        },
        "claim_scope",
    )
    _exact(
        scope["claim"],
        "finite_time_approximate_continuous_attractor_only",
        "claim_scope.claim",
    )
    _exact(scope["confirmatory"], False, "claim_scope.confirmatory")
    _exact(scope["may_support_l3_claim"], False, "claim_scope.may_support_l3_claim")
    _exact(scope["may_support_c4_claim"], False, "claim_scope.may_support_c4_claim")
    _exact(
        scope["may_support_main_seed_claim"],
        False,
        "claim_scope.may_support_main_seed_claim",
    )
    _exact(scope["seed_scope"], "parent_pilot_seeds_only", "claim_scope.seed_scope")
    _exact(scope["causal_interpretation"], "none", "claim_scope.causal_interpretation")

    manifold = _mapping(root["manifold"], "manifold")
    _exact_keys(manifold, {"primary", "task_atlas"}, "manifold")
    primary = _mapping(manifold["primary"], "manifold.primary")
    _exact_keys(
        primary,
        {
            "track",
            "source_states",
            "blank_relaxation_steps",
            "interpolation",
            "role",
            "failure_policy",
            "quality_gates",
        },
        "manifold.primary",
    )
    _exact(primary["track"], "track_A", "manifold.primary.track")
    _exact(
        primary["source_states"],
        "task_reachable_trajectory_endpoints",
        "manifold.primary.source_states",
    )
    _exact(
        primary["blank_relaxation_steps"],
        "16_times_task_horizon",
        "manifold.primary.blank_relaxation_steps",
    )
    _exact(
        primary["interpolation"],
        "periodic_cubic_spline",
        "manifold.primary.interpolation",
    )
    _exact(
        primary["role"],
        "primary_for_projection_based_gates",
        "manifold.primary.role",
    )
    _exact(
        primary["failure_policy"],
        "projection_based_gates_inconclusive",
        "manifold.primary.failure_policy",
    )
    quality = _mapping(primary["quality_gates"], "manifold.primary.quality_gates")
    _exact_keys(
        quality,
        {
            "coarse_coverage_bin_count",
            "minimum_coarse_bin_occupancy_fraction",
            "minimum_normalized_tangent_speed",
            "seam_probe_radians",
            "seam_C0_distance_over_Rs_max",
            "seam_C1_relative_difference_max",
        },
        "manifold.primary.quality_gates",
    )
    expected_quality = {
        "coarse_coverage_bin_count": 32,
        "minimum_coarse_bin_occupancy_fraction": 1.0,
        "minimum_normalized_tangent_speed": 0.001,
        "seam_probe_radians": 0.001,
        "seam_C0_distance_over_Rs_max": 1e-5,
        "seam_C1_relative_difference_max": 0.01,
    }
    for key, expected in expected_quality.items():
        observed = quality[key]
        _require(type(observed) in (int, float), f"quality gate {key} must be numeric")
        _exact(float(observed), float(expected), f"manifold.primary.quality_gates.{key}")

    atlas = _mapping(manifold["task_atlas"], "manifold.task_atlas")
    _exact_keys(
        atlas,
        {"track", "role", "may_be_primary", "comparisons"},
        "manifold.task_atlas",
    )
    _exact(atlas["track"], "task_conditioned_atlas", "manifold.task_atlas.track")
    _exact(atlas["role"], "correspondence_only", "manifold.task_atlas.role")
    _exact(atlas["may_be_primary"], False, "manifold.task_atlas.may_be_primary")
    comparisons = tuple(_sequence(atlas["comparisons"], "manifold.task_atlas.comparisons"))
    _exact(
        comparisons,
        (
            "state_distance_to_track_A",
            "decoded_angle_correspondence",
            "fiber_spread",
            "bidirectional_hausdorff_distance",
        ),
        "manifold.task_atlas.comparisons",
    )
    _require(
        primary["role"] != atlas["role"] and atlas["may_be_primary"] is False,
        "task atlas must not replace Track A as the primary manifold",
    )

    c3 = _mapping(root["c3_normal_recovery"], "c3_normal_recovery")
    _exact_keys(
        c3,
        {
            "manifold_distance",
            "clean_adherence",
            "recovery_ratio",
            "registered_horizons",
            "primary_horizon",
            "perturbation_families",
            "aggregation",
            "thresholds",
            "same_memory",
            "legacy_paired_endpoint_metric",
        },
        "c3_normal_recovery",
    )
    _exact(
        c3["manifold_distance"],
        "d_M(x)=norm(x-Pi_M(x))",
        "c3_normal_recovery.manifold_distance",
    )
    _exact(
        c3["clean_adherence"],
        "D_clean(H)=d_M(F0^H(m))/R_s",
        "c3_normal_recovery.clean_adherence",
    )
    _exact(
        c3["recovery_ratio"],
        "Q(H)=d_M(F0^H(m+delta))/max(d_M(m+delta),floor)",
        "c3_normal_recovery.recovery_ratio",
    )
    horizons = tuple(_sequence(c3["registered_horizons"], "registered_horizons"))
    _exact(horizons, REGISTERED_HORIZONS, "c3_normal_recovery.registered_horizons")
    _require(
        all(type(item) is int and item > 0 for item in horizons),
        "registered horizons must be positive integers",
    )
    _require(
        tuple(sorted(set(horizons))) == horizons,
        "registered horizons must be unique and increasing",
    )
    _exact(c3["primary_horizon"], PRIMARY_HORIZON, "c3_normal_recovery.primary_horizon")
    _require(PRIMARY_HORIZON in horizons, "primary horizon must be registered")
    families = tuple(_sequence(c3["perturbation_families"], "perturbation_families"))
    _exact(families, PERTURBATION_FAMILIES, "c3_normal_recovery.perturbation_families")

    aggregation = _mapping(c3["aggregation"], "c3_normal_recovery.aggregation")
    _exact_keys(aggregation, set(PERTURBATION_FAMILIES), "c3_normal_recovery.aggregation")
    _exact(
        aggregation["radial"],
        "summarize_all_registered_radial_perturbations",
        "c3_normal_recovery.aggregation.radial",
    )
    _exact(
        aggregation["ambient_normal"],
        "per_anchor_worst_direction_then_summarize_anchors",
        "c3_normal_recovery.aggregation.ambient_normal",
    )

    thresholds = _mapping(c3["thresholds"], "c3_normal_recovery.thresholds")
    _exact_keys(
        thresholds,
        {"clean_adherence_all_horizons", "primary_horizon_recovery"},
        "c3_normal_recovery.thresholds",
    )
    _validate_threshold(
        thresholds["clean_adherence_all_horizons"],
        label="c3_normal_recovery.thresholds.clean_adherence_all_horizons",
        statistic="q95",
        operator="<=",
        threshold=0.01,
    )
    recovery = _mapping(
        thresholds["primary_horizon_recovery"],
        "c3_normal_recovery.thresholds.primary_horizon_recovery",
    )
    _exact_keys(
        recovery,
        {"horizon", "apply_separately_to", "median", "q95"},
        "c3_normal_recovery.thresholds.primary_horizon_recovery",
    )
    _exact(recovery["horizon"], PRIMARY_HORIZON, "primary recovery horizon")
    _exact(
        tuple(_sequence(recovery["apply_separately_to"], "apply_separately_to")),
        PERTURBATION_FAMILIES,
        "primary recovery perturbation families",
    )
    _validate_threshold(
        recovery["median"],
        label="primary_horizon_recovery.median",
        statistic="median",
        operator="<=",
        threshold=0.5,
    )
    _validate_threshold(
        recovery["q95"],
        label="primary_horizon_recovery.q95",
        statistic="q95",
        operator="<",
        threshold=1.0,
    )

    same_memory = _mapping(c3["same_memory"], "c3_normal_recovery.same_memory")
    _exact_keys(
        same_memory,
        {"separate_gate", "part_of_recovery_ratio", "metric"},
        "c3_normal_recovery.same_memory",
    )
    _exact(same_memory["separate_gate"], True, "same_memory.separate_gate")
    _exact(
        same_memory["part_of_recovery_ratio"],
        False,
        "same_memory.part_of_recovery_ratio",
    )
    _exact(
        same_memory["metric"],
        "decoded_memory_displacement_after_recovery",
        "same_memory.metric",
    )

    legacy = _mapping(
        c3["legacy_paired_endpoint_metric"],
        "c3_normal_recovery.legacy_paired_endpoint_metric",
    )
    _exact_keys(
        legacy,
        {"name", "role", "may_satisfy_c3_gate"},
        "c3_normal_recovery.legacy_paired_endpoint_metric",
    )
    _exact(
        legacy["name"],
        "paired_endpoint_normal_deviation_over_rho",
        "legacy_paired_endpoint_metric.name",
    )
    _exact(legacy["role"], "diagnostic_only", "legacy_paired_endpoint_metric.role")
    _exact(
        legacy["may_satisfy_c3_gate"],
        False,
        "legacy_paired_endpoint_metric.may_satisfy_c3_gate",
    )

    settling = _mapping(root["settling"], "settling")
    _exact_keys(
        settling,
        {
            "variance_definition",
            "absolute_symmetric_relative_change",
            "positive_expansion",
            "mean_sheet_rollout",
        },
        "settling",
    )
    _exact(
        settling["variance_definition"],
        "V_i=mean_path(norm(h_i_path-mean_path(h_i_path))^2)",
        "settling.variance_definition",
    )
    absolute = _mapping(
        settling["absolute_symmetric_relative_change"],
        "settling.absolute_symmetric_relative_change",
    )
    _exact_keys(absolute, {"formula", "threshold"}, "absolute settling change")
    _exact(
        absolute["formula"],
        "A_i=abs(V_next-V_i)/max(V_i,V_next,(100*eps*R_s)^2)",
        "absolute settling formula",
    )
    _validate_threshold(
        absolute["threshold"],
        label="settling.absolute_symmetric_relative_change.threshold",
        statistic="q95",
        operator="<=",
        threshold=0.01,
    )
    expansion = _mapping(settling["positive_expansion"], "settling.positive_expansion")
    _exact_keys(expansion, {"formula", "threshold"}, "positive expansion")
    _exact(
        expansion["formula"],
        "E_i=max(V_next-V_i,0)/max(V_i,(100*eps*R_s)^2)",
        "positive expansion formula",
    )
    _validate_threshold(
        expansion["threshold"],
        label="settling.positive_expansion.threshold",
        statistic="q95",
        operator="<=",
        threshold=0.01,
    )
    rollout = _mapping(settling["mean_sheet_rollout"], "settling.mean_sheet_rollout")
    _exact_keys(
        rollout,
        {"map", "distance", "threshold"},
        "settling.mean_sheet_rollout",
    )
    _exact(rollout["map"], "actual_F0_power_5", "mean sheet rollout map")
    _exact(
        rollout["distance"],
        "d_M(F0^5(mean_sheet_state))/R_s",
        "mean sheet rollout distance",
    )
    _validate_threshold(
        rollout["threshold"],
        label="settling.mean_sheet_rollout.threshold",
        statistic="q95",
        operator="<=",
        threshold=0.01,
    )


def canonical_analysis_freeze_bytes(freeze: Mapping[str, Any]) -> bytes:
    """Return the deterministic JSON encoding used for provenance."""

    return json.dumps(
        freeze,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def analysis_freeze_fingerprint(freeze: Mapping[str, Any]) -> str:
    """Return SHA-256 of the complete canonical analysis freeze object."""

    return hashlib.sha256(canonical_analysis_freeze_bytes(freeze)).hexdigest()
