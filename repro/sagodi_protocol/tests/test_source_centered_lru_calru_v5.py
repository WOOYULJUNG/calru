from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from repro.sagodi_protocol.artifacts import atomic_json
from repro.sagodi_protocol.source_centered_lru_calru_v5 import (
    DEFAULT_CONFIG,
    EXPECTED_PARAMETER_COUNTS,
    MODEL_IDS,
    STAGE1_MODEL_IDS,
    _source_q1_memory,
    _training_batch,
    build_stage1_fanout_plan,
    build_stage1_sentinel_plan,
    build_stage2_fanout_plan,
    build_stage2_sentinel_plan,
    build_v4_model,
    load_v5_tuning_config,
    screen_stage1_sentinels,
    screen_stage2_sentinels,
    select_stage1_hyperparameters,
    select_stage2_hyperparameters,
    source_angular_integration,
)


def test_worker_progress_is_live_and_receipted(tmp_path: Path) -> None:
    """The CPU smoke worker must leave a final monitorable heartbeat."""

    from repro.sagodi_protocol.source_centered_lru_calru_v5 import run_smoke

    root = tmp_path / "progress_smoke"
    run_smoke(root, DEFAULT_CONFIG, ("cpu",))
    for model_id in MODEL_IDS:
        output = root / "smoke" / "runs" / model_id
        progress = json.loads(
            (output / "progress.json").read_text(encoding="utf-8")
        )
        assert progress["status"] == "complete"
        assert progress["update"] == 2


def _write_result(spec, id_mse: float, blank_mse: float | None = None) -> None:
    output = Path(spec.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "result.json",
        {
            "run_id": spec.run_id,
            "final_metrics": {"masked_mse": float(id_mse)},
            "heldout_blank_memory_mse": blank_mse,
        },
    )


def test_config_registers_source_task_noise_grid_and_parameter_counts() -> None:
    config = load_v5_tuning_config(DEFAULT_CONFIG)
    assert config["task"]["horizon"] == 128
    assert config["task"]["input_sparsity"] == "variable_uniform_0_2"
    assert config["task"]["initial_state_semantics"] == "source_q1_post_update_target"
    observed = config["stage1_lr_noise"]["state_noise_std_grid"]
    assert observed[:2] == [0.0, 0.01]
    assert observed[3] == 0.1
    assert math.isclose(observed[2], math.sqrt(0.1) * 0.1, abs_tol=1e-15)
    for model_id in MODEL_IDS:
        torch.manual_seed(0)
        model = build_v4_model(model_id)
        assert sum(parameter.numel() for parameter in model.parameters()) == (
            EXPECTED_PARAMETER_COUNTS[model_id]
        )


def test_task_is_128_step_variable_sparse_and_uses_clean_q1_initializer() -> None:
    batch = source_angular_integration(16, 0, stream_key=("v5-test", 0))
    assert batch.inputs.shape == (128, 16, 1)
    assert batch.metadata["input_sparsity"] == "variable_uniform_0_2"
    assert torch.equal(_source_q1_memory(batch), batch.output_targets[0])
    assert not torch.equal(_source_q1_memory(batch), batch.initial_memory)
    assert 0 < int((batch.inputs == 0).sum()) < batch.inputs.numel()


def test_online_stream_is_paired_by_seed_and_independent_across_seeds() -> None:
    config = load_v5_tuning_config(DEFAULT_CONFIG)
    left = _training_batch(config, 37, 4, 100, torch.device("cpu"))
    paired = _training_batch(config, 37, 4, 100, torch.device("cpu"))
    independent = _training_batch(config, 37, 4, 101, torch.device("cpu"))
    assert torch.equal(left.inputs, paired.inputs)
    assert torch.equal(left.output_targets, paired.output_targets)
    assert not torch.equal(left.inputs, independent.inputs)


def test_stage1_sentinel_precedes_bounded_fanout_and_no_rp_inherits(
    tmp_path: Path,
) -> None:
    config = load_v5_tuning_config(DEFAULT_CONFIG)
    bank = tmp_path / "bank.npz"
    sentinels = build_stage1_sentinel_plan(tmp_path, config, bank)
    assert len(sentinels) == 32
    assert {spec.model_seed for spec in sentinels} == {100}
    assert all(not spec.rp_enabled for spec in sentinels)

    # Four cells pass for each model; only the best three may fan out.
    for index, spec in enumerate(sentinels):
        within_model = index % 16
        _write_result(spec, 0.001 + within_model * 0.0001 if within_model < 4 else 0.02)
    screening = screen_stage1_sentinels(sentinels, config)
    assert screening["all_model_sentinel_gates_passed"] is True
    assert all(
        len(screening["models"][model_id]["fanned_out_cells"]) == 3
        for model_id in STAGE1_MODEL_IDS
    )
    fanout = build_stage1_fanout_plan(tmp_path, config, bank, screening)
    assert len(fanout) == 24
    assert {spec.model_seed for spec in fanout} == {101, 102, 103, 104}

    for spec in fanout:
        # Make LR=.01/noise=0 the stable five-seed winner for each model.
        penalty = 0.0 if spec.state_noise_std == 0.0 else 0.001
        _write_result(spec, 0.002 + penalty + (spec.model_seed - 101) * 1e-5)
    selected = select_stage1_hyperparameters(
        sentinels, fanout, screening, config
    )
    assert selected["selected_lr_noise"]["lru_n52"] == {
        "learning_rate": 0.01,
        "state_noise_std": 0.0,
    }
    inherited = selected["selected_lr_noise"]["ca_lru_n52"]
    assert inherited["learning_rate"] == selected["selected_lr_noise"]["no_rp_n52"]["learning_rate"]
    assert inherited["state_noise_std"] == selected["selected_lr_noise"]["no_rp_n52"]["state_noise_std"]
    assert inherited["independently_tuned"] is False


def test_failed_stage1_sentinel_blocks_fanout(tmp_path: Path) -> None:
    config = load_v5_tuning_config(DEFAULT_CONFIG)
    bank = tmp_path / "bank.npz"
    sentinels = build_stage1_sentinel_plan(tmp_path, config, bank)
    for spec in sentinels:
        _write_result(spec, 0.001 if spec.model_id == "lru_n52" else 0.02)
    screening = screen_stage1_sentinels(sentinels, config)
    assert screening["failed_models"] == ["no_rp_n52"]
    with pytest.raises(RuntimeError, match="blocked"):
        build_stage1_fanout_plan(tmp_path, config, bank, screening)


def test_stage2_freezes_base_pair_and_screens_before_fanout(tmp_path: Path) -> None:
    config = load_v5_tuning_config(DEFAULT_CONFIG)
    bank = tmp_path / "bank.npz"
    stage1 = {
        "selected_lr_noise": {
            "ca_lru_n52": {"learning_rate": 0.003, "state_noise_std": math.sqrt(0.1) * 0.1}
        }
    }
    sentinels = build_stage2_sentinel_plan(tmp_path, config, bank, stage1)
    assert len(sentinels) == 9
    assert all(spec.rp_enabled for spec in sentinels)
    assert {spec.learning_rate for spec in sentinels} == {0.003}
    assert {spec.state_noise_std for spec in sentinels} == {math.sqrt(0.1) * 0.1}

    for index, spec in enumerate(sentinels):
        _write_result(spec, 0.002, 0.01 + index * 0.01)
    screening = screen_stage2_sentinels(sentinels, config)
    assert screening["sentinel_gate_passed"] is True
    assert len(screening["fanned_out_cells"]) == 3
    fanout = build_stage2_fanout_plan(
        tmp_path, config, bank, stage1, screening
    )
    assert len(fanout) == 12
    for spec in fanout:
        blank = 0.02 if spec.rp_eta_lambda == 300.0 and spec.rp_damage_epsilon == 1e-5 else 0.2
        _write_result(spec, 0.003, blank)
    selected = select_stage2_hyperparameters(
        sentinels, fanout, screening, config
    )
    assert selected["selected"]["eta_lambda"] == 300.0
    assert selected["selected"]["damage_epsilon"] == 1e-5
