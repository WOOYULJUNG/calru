from __future__ import annotations

import torch

from repro.sagodi_protocol.config import load_protocol
from repro.sagodi_protocol.models import (
    ModelConfig,
    build_protocol_model,
    model_config_from_protocol,
)


def test_ca_lru_primary_excludes_overwritten_stream():
    model = build_protocol_model(ModelConfig("ca_lru", input_dim=1, output_dim=2, width=16))
    assert model.reported_state_size == 32
    assert model.primary_state_size == 16
    assert model.core.carry_stream is False
    state = model.initial_state(3, "cpu", torch.randn(3, 2))
    assert state.shape == (3, 32)


def test_no_rp_has_identical_scaffold_but_disabled_hook():
    full = build_protocol_model(ModelConfig("ca_lru", 1, 2, width=16))
    control = build_protocol_model(ModelConfig("no_rp", 1, 2, width=16))
    assert type(full.core) is type(control.core)
    assert full.rp_enabled
    assert not control.rp_enabled
    assert full.metadata()["parameters_total"] == control.metadata()["parameters_total"]
    assert (
        full.metadata()["autonomous_primary_map"]
        == "homogeneous_diagonal_linear_F0_h_equals_Lambda_h"
    )
    assert (
        full.metadata()["input_conditioned_writer"]
        == "nonlinear_recurrent_writer_g_h_u_minus_g_h_zero"
    )


def test_forward_sequence_hidden_init_and_noise():
    torch.manual_seed(0)
    model = build_protocol_model(ModelConfig("gru", 1, 2, width=8))
    inputs = torch.zeros(5, 4, 1)
    initial = torch.randn(4, 2)
    output, states = model.forward_sequence(inputs, initial_memory=initial, return_states=True)
    assert output.shape == (5, 4, 2)
    assert states.shape == (5, 4, 8)
    assert torch.isfinite(output).all()


def test_full_block_noise_is_added_after_transition_and_stream_is_recomputed():
    torch.manual_seed(3)
    model = build_protocol_model(ModelConfig("ca_lru", 1, 2, width=8))
    model.eval()
    state = model.initial_state(2, "cpu", torch.randn(2, 2))
    x_t = torch.randn(2, 1)
    noise = torch.full((2, 8), 0.125)
    clean = model.step(x_t, state)
    noisy = model.step(x_t, state, state_noise=noise)
    torch.testing.assert_close(
        model.primary_from_reported(noisy),
        model.primary_from_reported(clean) + noise,
    )
    expected = model._reported_from_noisy_full_block(x_t, model.primary_from_reported(noisy))
    torch.testing.assert_close(noisy, expected)


def test_model_builder_consumes_frozen_architecture_block():
    protocol = load_protocol()
    config = model_config_from_protocol(protocol, "ca_lru", width_override=16)
    assert config.layers == 1
    assert config.pan_lambda_min == 0.90
    assert config.pan_lambda_max == 0.999
    model = build_protocol_model(config)
    assert model.core.variant == "PAN-RNW-Block"
    assert model.core.encoder.bias is None
    assert isinstance(model.core.blocks[0].norm_in, torch.nn.Identity)
    assert model.core.blocks[0].rec.writer_mode == "recurrent"
    assert model.core.carry_stream is False


def test_gru_resolved_architecture_uses_state_size_not_nonexistent_hidden_dim():
    model = build_protocol_model(ModelConfig("gru", 1, 2, width=8))
    assert model.core.state_size == 8
    assert model.core.cell.hidden_size == 8
