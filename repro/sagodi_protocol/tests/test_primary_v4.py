from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from repro.sagodi_protocol.primary_v4 import (
    DEFAULT_CONFIG,
    EXPECTED_PARAMETER_COUNTS,
    MODEL_IDS,
    PAPER_BASELINE_IDS,
    RunSpec,
    _ensure_bank,
    _prelaunch_contract_checks,
    _training_batch,
    build_lr_tune_plan,
    build_main_plan,
    build_rp_tune_plan,
    build_sentinel_plan,
    build_v4_model,
    load_v4_config,
)


def _audit_payload() -> dict:
    return {
        "selection_role": "paper_100_update_online_loss_audit_only",
        "winners": {
            model_id: {"learning_rate": 1e-3} for model_id in MODEL_IDS
        },
    }


def _sentinel_payload() -> dict:
    return {
        "selected_learning_rates": {
            "sagodi_rnn_tanh_n128": 1e-2,
            "sagodi_gru_n128": 1e-2,
            "sagodi_lstm_n64": 1e-2,
            "lru_n52": 1e-3,
            "ca_lru_n52": 1e-4,
            "no_rp_n52": 1e-4,
        }
    }


def test_frozen_config_and_executable_parameter_counts() -> None:
    config = load_v4_config(DEFAULT_CONFIG)
    assert [item["id"] for item in config["models"]] == list(MODEL_IDS)
    for model_id in MODEL_IDS:
        torch.manual_seed(0)
        model = build_v4_model(model_id)
        assert sum(parameter.numel() for parameter in model.parameters()) == (
            EXPECTED_PARAMETER_COUNTS[model_id]
        )


def test_learning_rate_plans_never_tune_no_rp_independently(tmp_path: Path) -> None:
    config = load_v4_config(DEFAULT_CONFIG)
    bank = tmp_path / "bank.npz"
    lr_plan = build_lr_tune_plan(tmp_path, config, bank)
    assert len(lr_plan) == 100
    assert {spec.model_seed for spec in lr_plan} == set(range(100, 105))
    assert {spec.learning_rate for spec in lr_plan} == {1e-2, 1e-3, 1e-4, 1e-5}
    assert all(spec.model_id != "no_rp_n52" for spec in lr_plan)

    sentinel = build_sentinel_plan(tmp_path, config, bank, _audit_payload())
    assert len(sentinel) == 55
    for model_id in PAPER_BASELINE_IDS:
        assert {spec.learning_rate for spec in sentinel if spec.model_id == model_id} == {
            1e-2
        }
    for model_id in ("lru_n52", "ca_lru_n52"):
        assert {spec.learning_rate for spec in sentinel if spec.model_id == model_id} == {
            1e-2,
            1e-3,
            1e-4,
            1e-5,
        }
    assert all(spec.model_id != "no_rp_n52" for spec in sentinel)


def test_rp_and_main_plans_obey_frozen_inheritance(tmp_path: Path) -> None:
    config = load_v4_config(DEFAULT_CONFIG)
    bank = tmp_path / "bank.npz"
    sentinel = _sentinel_payload()
    rp_plan = build_rp_tune_plan(tmp_path, config, bank, sentinel)
    assert len(rp_plan) == 45
    assert {spec.rp_eta_lambda for spec in rp_plan} == {300.0, 1000.0, 3000.0}
    assert {spec.rp_damage_epsilon for spec in rp_plan} == {1e-5, 3e-5, 1e-4}
    assert all(spec.model_id == "ca_lru_n52" and spec.rp_enabled for spec in rp_plan)

    tune = {
        "selected_learning_rates_for_main": sentinel["selected_learning_rates"],
        "rp_selection": {
            "selected": {"eta_lambda": 1000.0, "damage_epsilon": 3e-5}
        },
    }
    main = build_main_plan(tmp_path, config, bank, tune)
    assert len(main) == 60
    assert {spec.model_seed for spec in main} == set(range(10))
    assert all(
        spec.learning_rate == 1e-2
        for spec in main
        if spec.model_id in PAPER_BASELINE_IDS
    )
    ca_rates = {spec.learning_rate for spec in main if spec.model_id == "ca_lru_n52"}
    no_rp_rates = {spec.learning_rate for spec in main if spec.model_id == "no_rp_n52"}
    assert ca_rates == no_rp_rates == {1e-4}
    assert all(spec.rp_enabled == (spec.model_id == "ca_lru_n52") for spec in main)


def test_online_task_batches_are_model_and_model_seed_independent(tmp_path: Path) -> None:
    config = load_v4_config(DEFAULT_CONFIG)
    common = dict(
        run_id="x",
        stage="main",
        learning_rate=1e-3,
        updates=1,
        batch_size=3,
        state_noise_std=0.1,
        evaluation_bank=str(tmp_path / "unused.npz"),
        output_dir=str(tmp_path / "unused"),
    )
    left = RunSpec(model_id="sagodi_rnn_tanh_n128", model_seed=0, **common)
    right = RunSpec(model_id="ca_lru_n52", model_seed=9, **common)
    batch_left = _training_batch(config, left, 37, torch.device("cpu"))
    batch_right = _training_batch(config, right, 37, torch.device("cpu"))
    assert torch.equal(batch_left.inputs, batch_right.inputs)
    assert torch.equal(batch_left.output_targets, batch_right.output_targets)
    assert torch.equal(batch_left.initial_memory, batch_right.initial_memory)


def test_ca_and_no_rp_are_bit_identical_before_rp() -> None:
    torch.manual_seed(123)
    ca = build_v4_model("ca_lru_n52")
    torch.manual_seed(123)
    no_rp = build_v4_model("no_rp_n52")
    assert ca.state_dict().keys() == no_rp.state_dict().keys()
    for key in ca.state_dict():
        assert torch.equal(ca.state_dict()[key], no_rp.state_dict()[key]), key


def test_config_rejects_parameter_count_drift(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text())
    payload["models"][0]["parameter_count"] += 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="parameter count"):
        load_v4_config(path)


def test_tuning_and_main_test_banks_are_disjoint(tmp_path: Path) -> None:
    config = load_v4_config(DEFAULT_CONFIG)
    tuning = _ensure_bank(tmp_path, config, purpose="tuning")
    main_test = _ensure_bank(tmp_path, config, purpose="main_test")
    assert tuning.name == "tuning.npz"
    assert main_test.name == "main_test.npz"
    assert tuning.read_bytes() != main_test.read_bytes()


def test_prelaunch_pairing_and_rp_api_contract() -> None:
    result = _prelaunch_contract_checks(load_v4_config(DEFAULT_CONFIG))
    assert result["passed"] is True
    assert result["ca_no_rp_one_noisy_adam_update_identical"] is True
    assert result["inductive_pairing_contract_through_update"] == 1500
    assert result["rp_reduced_api_smoke_finite"] is True
