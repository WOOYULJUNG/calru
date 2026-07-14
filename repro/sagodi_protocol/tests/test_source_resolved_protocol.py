from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from repro.sagodi_protocol.source_resolved_models import build_source_resolved_model
from repro.sagodi_protocol.source_resolved_protocol import (
    DEFAULT_SOURCE_CONFIG,
    SOURCE_MODEL_IDS,
    UPSTREAM_COMMIT,
    build_source_optimizer,
    load_source_config,
    noisy_training_targets,
    source_angular_integration,
    source_recipe,
)
from repro.sagodi_protocol.source_resolved_worker import run_source_worker
from repro.sagodi_protocol.tasks import save_fixed_bank


def test_source_contract_is_pinned_and_model_specific() -> None:
    config = load_source_config()
    assert config["upstream"]["commit"] == UPSTREAM_COMMIT
    assert config["task"]["horizon"] == 128
    assert config["task"]["duration"] == 12.8
    assert config["task"]["input_sparsity"] == "variable_uniform_0_2"
    assert config["task"]["initial_state_semantics"] == "source_q1_post_update_target"

    rnn = source_recipe("sagodi_rnn_tanh_n128")
    gru = source_recipe("sagodi_gru_n128")
    lstm = source_recipe("sagodi_lstm_n64")
    assert rnn.effective_state_noise_std == pytest.approx(math.sqrt(0.1) * 0.1)
    assert gru.target_noise_std == 0.01 and gru.output_dropout == 0.5
    assert gru.recurrent_weight_decay == 1e-4 and gru.gradient_clip_norm == 100.0
    assert lstm.learning_rate == 1e-3
    assert lstm.recurrent_weight_decay == 1e-2 and lstm.gradient_clip_norm == 1.0


def test_source_contract_rejects_commit_drift(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_SOURCE_CONFIG.read_text(encoding="utf-8"))
    payload["upstream"]["commit"] = "0" * 40
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="upstream commit"):
        load_source_config(path)


def test_variable_sparsity_task_is_deterministic_and_post_update() -> None:
    left = source_angular_integration(24, 7, stream_key=("unit", 3))
    right = source_angular_integration(24, 7, stream_key=("unit", 3))
    other = source_angular_integration(24, 7, stream_key=("unit", 4))
    assert torch.equal(left.inputs, right.inputs)
    assert torch.equal(left.output_targets, right.output_targets)
    assert not torch.equal(left.inputs, other.inputs)
    assert left.inputs.shape == (128, 24, 1)
    assert left.output_targets.shape == (128, 24, 2)
    assert torch.count_nonzero(left.inputs == 0) > 0
    assert torch.count_nonzero(left.inputs != 0) > 0

    q0 = torch.as_tensor(left.metadata["initial_latents"], dtype=left.inputs.dtype)
    q1 = q0[:, 0] + 0.1 * left.inputs[0, :, 0]
    expected = torch.stack((torch.cos(q1), torch.sin(q1)), dim=-1)
    assert torch.allclose(left.output_targets[0], expected, atol=2e-6, rtol=0.0)
    assert left.metadata["initial_state_target_index"] == 0


def test_model_initializations_and_trainable_parameter_counts() -> None:
    for model_id in SOURCE_MODEL_IDS:
        torch.manual_seed(123)
        model = build_source_resolved_model(model_id)
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        assert trainable == source_recipe(model_id).parameter_count
        assert torch.isfinite(model.output_to_hidden).all()

    torch.manual_seed(123)
    rnn = build_source_resolved_model("sagodi_rnn_tanh_n128")
    assert rnn.source_h0 is not None and not rnn.source_h0.requires_grad
    assert sum(parameter.numel() for parameter in rnn.parameters()) == 17154 + 128
    assert rnn.core.wi.std().item() == pytest.approx(1 / math.sqrt(128), rel=0.15)
    assert rnn.core.wo.std().item() == pytest.approx(1 / math.sqrt(128), rel=0.15)
    assert rnn.core.wrec.std().item() == pytest.approx(1.5 / math.sqrt(128), rel=0.08)
    assert rnn.core.brec.abs().max().item() <= math.sqrt(128)
    assert torch.count_nonzero(rnn.core.brec) == 128

    torch.manual_seed(123)
    gru = build_source_resolved_model("sagodi_gru_n128")
    assert gru.core.cell.weight_hh.abs().max().item() <= 0.25 / math.sqrt(128)
    torch.manual_seed(123)
    lstm = build_source_resolved_model("sagodi_lstm_n64")
    assert lstm.core.cell.weight_hh.abs().max().item() <= 1 / math.sqrt(64)


def test_uninitialized_maps_are_repaired_deterministically() -> None:
    for model_id in SOURCE_MODEL_IDS:
        torch.manual_seed(41)
        left = build_source_resolved_model(model_id)
        torch.manual_seed(41)
        right = build_source_resolved_model(model_id)
        assert torch.equal(left.output_to_hidden, right.output_to_hidden)
        if model_id == "sagodi_lstm_n64":
            assert torch.equal(left.output_to_cell, right.output_to_cell)


def test_q1_initialization_and_independent_lstm_cell_repair() -> None:
    model = build_source_resolved_model("sagodi_lstm_n64")
    with torch.no_grad():
        model.output_to_hidden.fill_(0.1)
        assert model.output_to_cell is not None
        model.output_to_cell.fill_(-0.2)
    targets = torch.zeros(2, 3, 2)
    targets[0] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    state = model.initial_state(targets)
    hidden, cell = model.core.split_state(state)
    assert torch.allclose(hidden, torch.tanh(targets[0] @ model.output_to_hidden))
    assert torch.allclose(cell, torch.tanh(targets[0] @ model.output_to_cell))
    assert not torch.equal(hidden, cell)


def test_only_rnn_receives_recurrent_state_noise() -> None:
    inputs = torch.zeros(1, 512, 1)
    targets = torch.zeros(1, 512, 2)
    rnn = build_source_resolved_model("sagodi_rnn_tanh_n128")
    with torch.no_grad():
        rnn.output_to_hidden.zero_()
        for parameter in rnn.core.parameters():
            parameter.zero_()
    _, states = rnn.forward_sequence(
        inputs,
        source_targets=targets,
        state_noise_generator=torch.Generator().manual_seed(5),
        return_states=True,
    )
    assert states[0].std().item() == pytest.approx(math.sqrt(0.1) * 0.1, rel=0.02)

    small_inputs = inputs[:, :4]
    small_targets = targets[:, :4]
    for model_id in ("sagodi_gru_n128", "sagodi_lstm_n64"):
        model = build_source_resolved_model(model_id).eval()
        left = model.forward_sequence(
            small_inputs,
            source_targets=small_targets,
            state_noise_generator=torch.Generator().manual_seed(1),
        )
        right = model.forward_sequence(
            small_inputs,
            source_targets=small_targets,
            state_noise_generator=torch.Generator().manual_seed(2),
        )
        assert torch.equal(left, right)


def test_target_noise_and_optimizer_groups_follow_source() -> None:
    clean = torch.zeros(128, 64, 2)
    rnn_targets = noisy_training_targets(
        clean,
        source_recipe("sagodi_rnn_tanh_n128"),
        generator=torch.Generator().manual_seed(0),
    )
    assert rnn_targets is clean
    gru_targets = noisy_training_targets(
        clean,
        source_recipe("sagodi_gru_n128"),
        generator=torch.Generator().manual_seed(0),
    )
    assert gru_targets.std().item() == pytest.approx(0.01, rel=0.03)

    expected = {
        "sagodi_rnn_tanh_n128": (1e-2, 0.0),
        "sagodi_gru_n128": (1e-2, 1e-4),
        "sagodi_lstm_n64": (1e-3, 1e-2),
    }
    for model_id, (rate, decay) in expected.items():
        model = build_source_resolved_model(model_id)
        optimizer = build_source_optimizer(model)
        groups = {group["group_name"]: group for group in optimizer.param_groups}
        assert groups["recurrent_core"]["lr"] == rate
        assert groups["recurrent_core"]["weight_decay"] == decay
        assert groups["readout_and_initial_map"]["weight_decay"] == 0.0


def test_single_run_worker_writes_complete_receipt(tmp_path: Path) -> None:
    bank = tmp_path / "evaluation.npz"
    save_fixed_bank(bank, source_angular_integration(8, 99, stream_key="evaluation"))
    output = tmp_path / "run"
    run_source_worker(
        run_id="unit__lstm__seed3",
        model_id="sagodi_lstm_n64",
        model_seed=3,
        evaluation_bank=bank,
        output_dir=output,
        device_text="cpu",
        updates=1,
        batch_size=2,
        trace_interval=1,
        validation_interval=1,
    )
    assert (output / "checkpoint_final.pt").is_file()
    assert (output / "completion_receipt.json").is_file()
    assert (output / "progress.json").is_file()
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "complete"
    assert result["updates_completed"] == 1
    assert math.isfinite(result["final_metrics"]["masked_mse"])
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "complete"
    assert progress["update"] == 1
    assert set(manifest["runtime_code_sha256"]) == {
        "source_resolved_worker.py",
        "source_resolved_models.py",
        "source_resolved_protocol.py",
        "tasks.py",
    }
