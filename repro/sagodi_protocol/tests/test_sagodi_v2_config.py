from __future__ import annotations

import copy
import json

import pytest

from repro.sagodi_protocol import config


PROTOCOL_A_V1_FINGERPRINT = (
    "451d95c7078fa0272890b188576fc0c80b98546608e2e1fa15f7f1a0b96c8d7e"
)
NATIVE_V1_FINGERPRINT = (
    "668867dfa36a4d6b4eb57fb63b61236f91c29756334f885f182bc9ed83ce1e1a"
)
LR_SELECTION_V2_FINGERPRINT = (
    "50515ab6be1aa6d1db3000b9d19ea1fd14661eb44805b74a68f44836d9013a96"
)


@pytest.fixture()
def selection_protocol() -> dict:
    return config.load_protocol(config.SAGODI_LR_SELECTION_PROTOCOL_PATH)


def test_v1_canonical_fingerprints_are_unchanged() -> None:
    protocol_a = config.load_protocol(config.DEFAULT_PROTOCOL_PATH)
    native = config.load_protocol(config.NATIVE_RECIPE_PROTOCOL_PATH)
    assert config.protocol_fingerprint(protocol_a) == PROTOCOL_A_V1_FINGERPRINT
    assert config.protocol_fingerprint(native) == NATIVE_V1_FINGERPRINT


def test_selection_freeze_is_json_compatible_and_fingerprinted(
    selection_protocol: dict,
) -> None:
    raw = config.SAGODI_LR_SELECTION_PROTOCOL_PATH.read_text(encoding="utf-8")
    assert json.loads(raw) == selection_protocol
    assert selection_protocol["freeze_id"] == config.SAGODI_LR_SELECTION_FREEZE_ID
    assert (
        config.protocol_fingerprint(selection_protocol)
        == LR_SELECTION_V2_FINGERPRINT
    )
    reporting = selection_protocol["reporting"]
    assert "paper-aligned" in reporting["display_label"]
    assert reporting["method_specific_component"] == "CA-LRU_Retention_Plasticity"
    assert reporting["bit_exact_official_implementation"] is False
    assert reporting["protocol_A_eligible"] is False
    assert reporting["protocol_B_confirmatory_eligible"] is False


def test_selection_run_expansion_is_exactly_eighty(selection_protocol: dict) -> None:
    runs = config.expand_phase1_runs(selection_protocol)
    assert len(runs) == 80
    assert len({run["run_id"] for run in runs}) == 80
    assert {run["model"]["id"] for run in runs} == set(
        config.SAGODI_LR_SELECTION_MODELS
    )
    assert {run["model_seed"] for run in runs} == set(
        config.SAGODI_LR_SELECTION_SEEDS
    )
    assert {run["learning_rate"] for run in runs} == set(
        config.SAGODI_LR_SELECTION_GRID
    )
    assert {
        (run["model"]["id"], run["width"])
        for run in runs
    } == {
        ("ca_lru", 96),
        ("no_rp", 96),
        ("gru_sagodi_width96", 96),
        ("gru_sagodi_param135", 135),
    }
    assert all(not run["confirmatory"] for run in runs)
    assert {run["protocol_track"] for run in runs} == {
        config.SAGODI_LR_SELECTION_TRACK
    }


def test_selector_training_and_decision_rule_are_paper_aligned_only(
    selection_protocol: dict,
) -> None:
    phase1 = selection_protocol["phase1_ring_pilot"]
    training = phase1["training"]
    task = config.AngularTaskSpec.from_protocol(selection_protocol)
    assert task.sequence_steps == 256
    assert task.delta_t == 0.1
    assert task.gp_cholesky_jitter == 1e-6
    assert task.gp_cholesky_jitter_source == "protocol_explicit"
    assert training["batch_size"] == 64
    assert training["optimizer_updates"] == 100
    assert training["optimizer"] == {
        "name": "Adam",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
    }
    assert training["gradient_clipping"] == {
        "policy": "none",
        "frozen_numeric_value": None,
    }
    assert training["state_noise"]["coordinate_standard_deviation"] == 0.1
    assert training["state_noise"]["target"] == "primary_markov_state"
    assert training["rp_schedule_for_ca_lru"]["calls_after_warmup"] == 0
    rule = training["learning_rate"]["selection_rule"]
    assert rule["primary_metric"] == "mean_online_training_loss_at_update_100"
    assert rule["selection_scope"] == "separately_per_model"
    assert rule["validation_metrics_role"] == "secondary_non_selecting"
    assert rule["failed_run_policy"] == (
        "no_scientific_failure_inference_from_nonzero_exit_oom_kill_or_invalid_receipt"
    )
    assert rule["failed_run_retry_policy"] == (
        "infrastructure_or_unknown_failure_aborts_campaign_and_is_resume_eligible"
    )
    assert rule["campaign_completion"] == (
        "all_80_runs_must_have_verified_success_receipts"
    )


def test_selector_freezes_official_style_gru_variants(
    selection_protocol: dict,
) -> None:
    architecture = selection_protocol["phase1_ring_pilot"]["training"][
        "architecture"
    ]
    initializer = architecture["initial_state_encoder"]
    assert initializer["bias"] is False
    assert initializer["weight_initialization"] == {
        "distribution": "normal",
        "mean": 0.0,
        "standard_deviation": "1_over_sqrt_primary_state_dimension",
        "source": "Sagodi_official_W_otr",
    }
    width96 = architecture["gru_sagodi_width96"]
    param135 = architecture["gru_sagodi_param135"]
    assert (width96["hidden_width"], width96["parameter_count"]) == (96, 28898)
    assert (param135["hidden_width"], param135["parameter_count"]) == (135, 56432)
    for block in (width96, param135):
        assert block["initial_state"] == "tanh_of_bias_free_W_otr_times_y0"
        assert block["readout"] == "direct_biased_linear"
        assert block["bias_convention"] == "two_PyTorch_default_random_bias_vectors"
        assert "uninitialized" in block["output_to_hidden_initialization_repair"]


def test_selector_declares_corrected_analysis_semantics_but_disables_ca_evidence(
    selection_protocol: dict,
) -> None:
    evaluation = selection_protocol["evaluation"]
    assert evaluation["analysis_enabled"] is False
    assert evaluation["selection_results_are_approximate_ca_evidence"] is False
    assert evaluation["finite_kick_horizons"] == [1, 5, 20, 100, 500, 1024]
    assert evaluation["primary_manifold_reconstruction"].startswith("track_a_")
    assert evaluation["task_conditioned_atlas_role"] == (
        "correspondence_only_not_primary_projector"
    )
    assert set(evaluation["c3_metric_labels"]) == {
        "clean_adherence",
        "manifold_recovery",
    }
    gates = selection_protocol["claim_gates"]
    assert gates["all_approximate_ca_claims_enabled"] is False
    assert gates["selection_artifacts_may_be_reused_as_ca_evidence"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["seed_policy"]["selection_model_seeds"].__setitem__(
                0, 1099
            ),
            "selection seeds",
        ),
        (
            lambda value: value["phase0_state_audit"]["models"].pop(),
            "Phase 0 models",
        ),
        (
            lambda value: value["phase1_ring_pilot"]["models"][3].__setitem__(
                "hidden_width", 134
            ),
            "model specs",
        ),
        (
            lambda value: value["phase1_ring_pilot"]["training"].__setitem__(
                "optimizer_updates", 101
            ),
            "update 100",
        ),
        (
            lambda value: value["phase1_ring_pilot"]["training"][
                "rp_schedule_for_ca_lru"
            ].__setitem__("calls_after_warmup", 1),
            "zero RP calls",
        ),
        (
            lambda value: value["phase1_ring_pilot"]["training"][
                "learning_rate"
            ]["selection_rule"].__setitem__(
                "validation_metrics_role", "selecting"
            ),
            "decision rule",
        ),
        (
            lambda value: value["evaluation"]["finite_kick_horizons"].pop(),
            "evaluation declaration",
        ),
        (
            lambda value: value["reporting"].__setitem__(
                "bit_exact_official_implementation", True
            ),
            "reporting labels",
        ),
    ],
)
def test_selector_tampering_fails_closed(
    selection_protocol: dict, mutate, message: str
) -> None:
    changed = copy.deepcopy(selection_protocol)
    mutate(changed)
    with pytest.raises(config.ProtocolConfigError, match=message):
        config.validate_protocol(changed)
