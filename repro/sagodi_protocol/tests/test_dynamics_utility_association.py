from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from repro.sagodi_protocol.artifacts import atomic_json
from repro.sagodi_protocol.dynamics_utility_association import (
    ANALYSIS_ROLE,
    CAMPAIGN_TYPE,
    FrozenAssociation,
    VerifiedParent,
    _association_statistic,
    _average_ranks,
    _bootstrap_seed,
    _correlation,
    _exact_keys,
    _row_index,
    _shared_main_binding,
    _validate_launch_code_identity,
    build_association_summary,
    load_association_freeze,
)
from repro.sagodi_protocol.engineering_benefit_campaign import (
    EXPECTED_UTILITY_METRIC_KEYS,
)


FREEZE = (
    Path(__file__).resolve().parents[1]
    / "dynamics_utility_association_freeze_v1.json"
)


def _primary_row(model_id: str, seed: int, index: int) -> dict:
    value = float(index + 1)
    return {
        "model_id": model_id,
        "model_seed": seed,
        "analysis_status": "complete_structural_summary_eligible",
        "eligible_by_nmse_rule": True,
        "structurally_estimable": True,
        "included_in_primary_structural_summary": True,
        "training_outcome": {
            "checkpoint_sha256": f"checkpoint-{index}",
            "training_receipt_sha256": f"receipt-{index}",
        },
        "sagodi_metrics": {
            "uniform_flow_norm": value,
            "top_two_real_part_gap": {"mean": value},
            "asymptotic_capacity": {
                "effective_basin_count": value,
                "shannon_entropy_nats": value,
            },
        },
    }


def _engineering_row(model_id: str, seed: int, index: int) -> dict:
    utility = {key: 0.5 for key in EXPECTED_UTILITY_METRIC_KEYS}
    value = float(index + 1)
    utility["temporal/16T/prefix_mean_error_radians"] = 100.0 - value
    utility[
        "perturbation/relative_rms_0/16T/memory_mean_error_radians"
    ] = value
    return {
        "model_id": model_id,
        "model_seed": seed,
        "join_key": f"{model_id}::seed{seed:02d}",
        "seed_excluded": False,
        "utility_metrics": utility,
        "utility_metric_status": {key: "complete" for key in utility},
    }


def _summaries() -> tuple[dict, dict]:
    primary = []
    engineering = []
    for index, (model_id, seed) in enumerate(_exact_keys()):
        primary.append(_primary_row(model_id, seed, index))
        engineering.append(_engineering_row(model_id, seed, index))
    return {"runs": primary}, {"runs": engineering}


def _fast_freeze() -> tuple[dict, tuple[FrozenAssociation, ...]]:
    freeze, associations = load_association_freeze(FREEZE)
    freeze = copy.deepcopy(freeze)
    freeze["statistics"]["bootstrap"]["paired_resamples_with_replacement"] = 50
    return freeze, associations


def test_exact_preregistered_freeze_loads_and_tamper_fails(tmp_path: Path) -> None:
    payload, associations = load_association_freeze(FREEZE)
    assert payload["freeze_id"] == CAMPAIGN_TYPE
    assert payload["analysis_role"] == ANALYSIS_ROLE
    assert len(associations) == 4
    assert associations[0].x_label == "mean_top_two_vector_field_real_part_gap"
    assert "alignment_is_not_measured" in associations[0].x_interpretation

    changed = copy.deepcopy(payload)
    changed["statistics"]["bootstrap"]["base_seed"] += 1
    path = tmp_path / "changed.json"
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="preregistered"):
        load_association_freeze(path)


def test_exact_sixty_key_product_and_order_are_required() -> None:
    primary, _ = _summaries()
    rows = primary["runs"]
    assert tuple(_row_index(rows, label="rows")) == _exact_keys()
    with pytest.raises(RuntimeError, match="exactly 60"):
        _row_index(rows[:-1], label="rows")
    shuffled = list(rows)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    with pytest.raises(RuntimeError, match="order"):
        _row_index(shuffled, label="rows")


def test_average_rank_and_correlations_handle_ties() -> None:
    values = np.asarray([30.0, 10.0, 10.0, 20.0])
    assert np.allclose(_average_ranks(values), [4.0, 1.5, 1.5, 3.0])
    x = np.asarray([1.0, 2.0, 2.0, 4.0])
    y = np.asarray([4.0, 3.0, 3.0, 1.0])
    assert _correlation(x, y, "spearman_average_rank") == pytest.approx(-1.0)
    assert _correlation(x, y, "pearson_product_moment") == pytest.approx(-1.0)
    assert _correlation(np.ones(4), y, "pearson_product_moment") is None


def test_bootstrap_seed_is_stable_and_scope_specific() -> None:
    first = _bootstrap_seed(741203, "a", "pooled", "pearson_product_moment")
    assert first == _bootstrap_seed(
        741203, "a", "pooled", "pearson_product_moment"
    )
    assert first != _bootstrap_seed(
        741203, "a", "model=x", "pearson_product_moment"
    )


def test_summary_preserves_all_missing_pairs_and_signed_associations() -> None:
    freeze, associations = _fast_freeze()
    primary, engineering = _summaries()

    # One ineligible and one eligible-but-structurally-not-estimable seed stay
    # in every 60-row denominator.
    primary["runs"][0].update(
        {
            "analysis_status": "ineligible_for_structural_summary",
            "eligible_by_nmse_rule": False,
            "structurally_estimable": False,
            "included_in_primary_structural_summary": False,
            "sagodi_metrics": None,
        }
    )
    primary["runs"][1].update(
        {
            "analysis_status": "structural_analysis_not_estimable",
            "eligible_by_nmse_rule": True,
            "structurally_estimable": False,
            "included_in_primary_structural_summary": False,
            "sagodi_metrics": None,
        }
    )
    # Capacity can be unavailable even when the other Ságodi metrics are
    # estimable; this pair remains present and is missing only for capacity.
    primary["runs"][2]["sagodi_metrics"]["asymptotic_capacity"][
        "effective_basin_count"
    ] = None
    primary["runs"][2]["sagodi_metrics"]["asymptotic_capacity"][
        "shannon_entropy_nats"
    ] = None
    engineering["runs"][3]["utility_metrics"][
        "temporal/16T/prefix_mean_error_radians"
    ] = None
    engineering["runs"][3]["utility_metric_status"][
        "temporal/16T/prefix_mean_error_radians"
    ] = "engineering_metric_nonfinite"

    result = build_association_summary(
        primary, engineering, freeze, associations
    )
    assert result["registered_join_pair_count"] == 60
    assert result["excluded_registered_pairs"] == 0
    by_id = {item["association_id"]: item for item in result["associations"]}

    gap = by_id["timescale_gap_vs_temporal_16T_error"]
    assert len(gap["rows"]) == 60
    assert gap["included_pair_count"] == 57
    assert gap["missing_reason_counts"]["primary_ineligible"] == 1
    assert gap["missing_reason_counts"]["primary_structurally_not_estimable"] == 1
    ineligible = next(
        row
        for row in gap["rows"]
        if row["primary_analysis_status"] == "ineligible_for_structural_summary"
    )
    assert "primary_ineligible" in ineligible["missing_reasons"]
    assert "primary_structurally_not_estimable" not in ineligible["missing_reasons"]
    assert gap["missing_reason_counts"]["engineering_metric_nonfinite"] == 1
    pearson = gap["scopes"]["pooled_all_six_models"]["statistics"][
        "pearson_product_moment"
    ]
    assert pearson["raw_correlation"] == pytest.approx(-1.0)
    assert pearson["benefit_aligned_correlation"] == pytest.approx(1.0)
    assert pearson["bootstrap"]["paired_resamples_requested"] == 50

    flow = by_id["uniform_flow_vs_clean_16T_blank_retention_error"]
    flow_pearson = flow["scopes"]["pooled_all_six_models"]["statistics"][
        "pearson_product_moment"
    ]
    assert flow_pearson["raw_correlation"] == pytest.approx(1.0)
    assert flow_pearson["benefit_aligned_correlation"] == pytest.approx(1.0)

    capacity = by_id["effective_capacity_vs_temporal_16T_error"]
    assert capacity["included_pair_count"] == 56
    assert capacity["missing_reason_counts"]["primary_metric_missing"] == 3
    assert capacity["missing_reason_counts"]["engineering_metric_nonfinite"] == 1
    assert capacity["rows"][2]["included"] is False
    assert capacity["rows"][2]["missing_reasons"] == ["primary_metric_missing"]


def test_constant_and_small_samples_are_explicit_not_nan() -> None:
    association = FrozenAssociation(
        association_id="a",
        x_source="x",
        x_label="x",
        x_beneficial_direction="higher",
        y_source="y",
        y_label="y",
        y_beneficial_direction="lower",
    )
    small = [
        {"included": True, "x_value_or_null": 1.0, "y_value_or_null": 2.0},
        {"included": True, "x_value_or_null": 2.0, "y_value_or_null": 1.0},
    ]
    result = _association_statistic(
        small,
        association,
        scope="pooled",
        method="pearson_product_moment",
        base_seed=1,
        resamples=10,
        minimum_pairs=3,
    )
    assert result["status"] == "insufficient_complete_pairs"
    constant = [
        {"included": True, "x_value_or_null": 1.0, "y_value_or_null": value}
        for value in (1.0, 2.0, 3.0)
    ]
    result = _association_statistic(
        constant,
        association,
        scope="pooled",
        method="spearman_average_rank",
        base_seed=1,
        resamples=10,
        minimum_pairs=3,
    )
    assert result["status"] == "constant_x"
    assert result["raw_correlation"] is None


def _verified_parents(*, commit: str = "a" * 40) -> tuple[VerifiedParent, VerifiedParent]:
    primary_summary, engineering_summary = _summaries()
    primary_main = {
        "campaign_scientific_identity": "main-id",
        "protocol_canonical_fingerprint": "protocol-id",
        "manifest_sha256": "main-manifest",
        "summary_sha256": "main-summary",
        "complete_sha256": "main-complete",
        "completion_receipt_sha256": "main-receipt",
        "main_code_commit": commit,
    }
    nested = []
    for index, (model_id, seed) in enumerate(_exact_keys()):
        nested.append(
            {
                "model_id": model_id,
                "model_seed": seed,
                "checkpoint_sha256": f"checkpoint-{index}",
                "training_receipt_sha256": f"receipt-{index}",
            }
        )
    engineering_main = {
        "scientific_identity": "main-id",
        "manifest_sha256": "main-manifest",
        "summary_sha256": "main-summary",
        "complete_sha256": "main-complete",
        "completion_receipt_sha256": "main-receipt",
        "main_code_commit": commit,
        "nested_training_artifacts": nested,
    }
    primary = VerifiedParent(
        Path("/primary"),
        {},
        primary_summary,
        {"main_binding": primary_main},
    )
    engineering = VerifiedParent(
        Path("/engineering"),
        {},
        engineering_summary,
        {"parent_main": engineering_main},
    )
    return primary, engineering


def test_cross_parent_binding_checks_all_nested_pairs_and_exact_commit() -> None:
    primary, engineering = _verified_parents()
    shared = _shared_main_binding(primary, engineering)
    assert shared["main_code_commit"] == "a" * 40
    assert shared["verified_nested_training_receipt_count"] == 60
    _validate_launch_code_identity(
        {"worktree_dirty": False, "code_commit": "a" * 40}, shared
    )
    with pytest.raises(RuntimeError, match="exact"):
        _validate_launch_code_identity(
            {"worktree_dirty": False, "code_commit": "b" * 40}, shared
        )
    with pytest.raises(RuntimeError, match="clean"):
        _validate_launch_code_identity(
            {"worktree_dirty": True, "code_commit": "a" * 40}, shared
        )

    changed = copy.deepcopy(engineering)
    changed.binding["parent_main"]["nested_training_artifacts"][4][
        "checkpoint_sha256"
    ] = "changed"
    with pytest.raises(RuntimeError, match="nested binding"):
        _shared_main_binding(primary, changed)


def test_benefit_aligned_interval_has_ordered_sign_flip() -> None:
    association = FrozenAssociation(
        association_id="a",
        x_source="x",
        x_label="x",
        x_beneficial_direction="higher",
        y_source="y",
        y_label="y",
        y_beneficial_direction="lower",
    )
    rows = [
        {
            "included": True,
            "x_value_or_null": float(value),
            "y_value_or_null": float(value * value + (value % 2)),
        }
        for value in range(1, 9)
    ]
    result = _association_statistic(
        rows,
        association,
        scope="pooled",
        method="pearson_product_moment",
        base_seed=741203,
        resamples=100,
        minimum_pairs=3,
    )
    raw = result["bootstrap"]["raw_percentile_95_interval"]
    aligned = result["bootstrap"]["benefit_aligned_percentile_95_interval"]
    assert aligned == pytest.approx([-raw[1], -raw[0]])
    assert all(math.isfinite(value) for value in raw)
