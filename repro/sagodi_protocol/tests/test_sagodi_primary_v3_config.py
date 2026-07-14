from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from repro.sagodi_protocol.config import (
    SAGODI_PRIMARY_LR_SELECTION_FREEZE_ID,
    SAGODI_PRIMARY_MODELS,
    SAGODI_PRIMARY_PARAMETER_COUNTS,
    SAGODI_PRIMARY_SELECTION_SEEDS,
    expand_phase1_runs,
    load_protocol,
    source_protocol_matches,
    validate_protocol,
)
from repro.sagodi_protocol.lr_selection_v3 import (
    build_selection_plan,
    load_selector_spec,
    validate_protocol_binding,
)
from repro.sagodi_protocol.models import build_protocol_model, model_config_from_protocol


PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
PROTOCOL = PACKAGE / "sagodi_primary_lr_selection_v3.yaml"
SELECTOR = PACKAGE / "sagodi_primary_lr_selection_v3.json"


def test_primary_v3_freeze_binds_source_and_exact_120_run_matrix() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol["freeze_id"] == SAGODI_PRIMARY_LR_SELECTION_FREEZE_ID
    assert protocol["source_protocol"]["version"] == "3.1"
    assert source_protocol_matches(protocol, REPOSITORY)
    assert tuple(protocol["phase0_state_audit"]["models"]) == SAGODI_PRIMARY_MODELS
    assert tuple(protocol["seed_policy"]["selection_model_seeds"]) == (
        SAGODI_PRIMARY_SELECTION_SEEDS
    )

    runs = expand_phase1_runs(protocol)
    assert len(runs) == 120
    assert len({run["run_id"] for run in runs}) == 120
    assert {
        (run["model"]["id"], run["model_seed"], run["learning_rate"])
        for run in runs
    } == {
        (model, seed, learning_rate)
        for model in SAGODI_PRIMARY_MODELS
        for seed in SAGODI_PRIMARY_SELECTION_SEEDS
        for learning_rate in (0.01, 0.001, 0.0001, 0.00001)
    }


def test_primary_v3_selector_and_protocol_are_the_same_design() -> None:
    protocol = load_protocol(PROTOCOL)
    selector = load_selector_spec(SELECTOR)
    validate_protocol_binding(protocol, selector)
    plan = build_selection_plan(selector)
    assert len(plan) == 120
    assert len({run.run_id for run in plan}) == 120


def test_primary_v3_runtime_parameter_counts_match_the_freeze() -> None:
    protocol = load_protocol(PROTOCOL)
    for model_id in SAGODI_PRIMARY_MODELS:
        model = build_protocol_model(model_config_from_protocol(protocol, model_id))
        assert model.metadata()["parameters_total"] == SAGODI_PRIMARY_PARAMETER_COUNTS[
            model_id
        ]


@pytest.mark.parametrize(
    "mutation",
    ("model_order", "selection_seed", "lr_grid", "noise", "rp", "run_count"),
)
def test_primary_v3_freeze_rejects_scientific_mutations(mutation: str) -> None:
    protocol = load_protocol(PROTOCOL)
    changed = copy.deepcopy(protocol)
    if mutation == "model_order":
        changed["phase1_ring_pilot"]["models"][0:2] = reversed(
            changed["phase1_ring_pilot"]["models"][0:2]
        )
    elif mutation == "selection_seed":
        changed["seed_policy"]["selection_model_seeds"][0] = 1099
    elif mutation == "lr_grid":
        changed["phase1_ring_pilot"]["training"]["learning_rate"]["grid"][0] = 0.02
    elif mutation == "noise":
        changed["phase1_ring_pilot"]["training"]["state_noise"][
            "coordinate_standard_deviation"
        ] = 0.0
    elif mutation == "rp":
        changed["phase1_ring_pilot"]["training"]["rp_schedule_for_ca_lru"][
            "enabled_during_selector"
        ] = True
    else:
        changed["phase1_ring_pilot"]["run_matrix"]["expected_training_runs"] = 119
    with pytest.raises(ValueError):
        validate_protocol(changed)


def test_primary_v3_selector_json_has_no_analysis_or_claim_gate_fields() -> None:
    payload = json.loads(SELECTOR.read_text(encoding="utf-8"))
    assert payload["scope"] == (
        "training_only_lr_selection_no_manifold_analysis_no_ca_evidence"
    )
    serialized = json.dumps(payload, sort_keys=True)
    assert "manifold" in serialized
    assert "claim_gate" not in serialized
