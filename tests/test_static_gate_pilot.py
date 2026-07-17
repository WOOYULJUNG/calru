from __future__ import annotations

import torch

from repro.static_gate_pilot.cache import (
    build_training_cache,
    cached_training_batch,
)
from repro.static_gate_pilot.models import MODEL_IDS, StaticGateMemory
from repro.static_gate_pilot.run import gate_intervention_rp


class _Probe:
    def __init__(self) -> None:
        self.inputs = torch.randn(4, 3, 1) * 0.1
        self.initial_memory = torch.randn(3, 2)
        self.output_targets = torch.randn(4, 3, 2)


def test_all_static_gate_variants_have_topology_shapes() -> None:
    for model_id in MODEL_IDS:
        model = StaticGateMemory(
            model_id=model_id, topology="t2", width=8
        )
        inputs = torch.randn(5, 3, 2)
        initial = torch.randn(3, 4)
        prediction, states = model.forward_sequence(
            inputs, initial_memory=initial, return_states=True
        )
        assert prediction.shape == (5, 3, 4)
        assert states.shape == (5, 3, 8)
        assert model.theta.requires_grad is (not model.rp_enabled)


def test_untied_input_write_vanishes_exactly_at_zero_input() -> None:
    model = StaticGateMemory(
        model_id="untied_rnn_grad", topology="s1", width=8
    )
    state = torch.randn(4, 8)
    zero = torch.zeros(4, 1)
    retention = model.retention()
    assert model.autonomous_hidden is not None
    autonomous = torch.tanh(model.autonomous_hidden(state))
    expected = retention * state + (1.0 - retention) * autonomous
    torch.testing.assert_close(model.step(zero, state), expected)


def test_retention_override_targets_each_batch_row() -> None:
    model = StaticGateMemory(
        model_id="static_gru_grad", topology="s1", width=4
    )
    state = torch.randn(3, 4)
    inputs = torch.zeros(3, 1)
    override = torch.full((3, 4), 0.8)
    output = model.step(inputs, state, retention_override=override)
    assert output.shape == state.shape
    assert torch.isfinite(output).all()


def test_gate_intervention_rp_updates_frozen_theta() -> None:
    torch.manual_seed(1)
    model = StaticGateMemory(
        model_id="untied_rnn_rp", topology="s1", width=6
    )
    before = model.theta.detach().clone()
    summary = gate_intervention_rp(
        model,
        _Probe(),
        blank_horizon=3,
        lambda_fast=0.5,
        eta_lambda=1.0,
        damage_epsilon=0.0,
    )
    assert not torch.equal(before, model.theta)
    assert torch.isfinite(model.theta).all()
    assert "normalized_damage_mean" in summary


def test_small_training_cache_is_paired_and_reusable(tmp_path) -> None:
    root = build_training_cache(
        tmp_path / "cache", seed=7, trajectories=8, horizon=128
    )
    first = cached_training_batch(
        root,
        topology="s1",
        replicate_seed=10,
        update=3,
        batch_size=4,
        device="cpu",
    )
    repeated = cached_training_batch(
        root,
        topology="s1",
        replicate_seed=10,
        update=3,
        batch_size=4,
        device="cpu",
    )
    torch.testing.assert_close(first.inputs, repeated.inputs)
    torch.testing.assert_close(first.initial_memory, repeated.initial_memory)
    assert first.inputs.shape == (128, 4, 1)
