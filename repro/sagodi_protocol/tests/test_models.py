from __future__ import annotations

import torch
import pytest

from repro.sagodi_protocol.config import NATIVE_RECIPE_PROTOCOL_PATH, load_protocol
from repro.sagodi_protocol.models import (
    INITIAL_ENCODER_IDENTITY,
    INITIAL_ENCODER_PYTORCH_DEFAULT,
    INITIAL_ENCODER_SAGODI_W_OTR,
    INITIAL_ENCODER_TANH,
    LRU_PARAM96,
    LSTM_PARAM109,
    ModelConfig,
    PROJECT_PARAM_BASELINE_NAMES,
    PROJECT_PARAM_BASELINE_PARAMETER_COUNTS,
    PROJECT_PARAM_BASELINE_WIDTHS,
    RNN_PARAM206,
    SAGODI_GRU_PARAM135,
    SAGODI_GRU_WIDTH96,
    SagodiGRUBaseline,
    build_protocol_model,
    model_config_from_protocol,
)
from repro.sagodi_protocol.state import StateAdapter


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
    assert config.initial_encoder_bias is True
    assert config.initial_encoder_weight_init == INITIAL_ENCODER_PYTORCH_DEFAULT
    assert model.initial_encoder.bias is not None


def test_native_hidden_initializer_uses_bias_free_sagodi_w_otr_policy():
    protocol = load_protocol(NATIVE_RECIPE_PROTOCOL_PATH)
    torch.manual_seed(17)
    config = model_config_from_protocol(protocol, "ca_lru")
    model = build_protocol_model(config)
    assert config.initial_encoder_bias is False
    assert config.initial_encoder_weight_init == INITIAL_ENCODER_SAGODI_W_OTR
    assert model.initial_encoder.bias is None
    expected_std = 1.0 / (model.primary_state_size**0.5)
    observed = model.initial_encoder.weight.detach()
    assert abs(float(observed.mean())) < 0.03
    assert abs(float(observed.std(unbiased=False)) - expected_std) < 0.03
    metadata = model.metadata()["initial_state_encoder_initialization"]
    assert metadata == {
        "policy": INITIAL_ENCODER_SAGODI_W_OTR,
        "bias": False,
        "weight_distribution": "Normal(0, 1/sqrt(primary_state_dimension))",
        "activation": INITIAL_ENCODER_IDENTITY,
        "primary_state_dimension": 96,
    }


def test_gru_resolved_architecture_uses_state_size_not_nonexistent_hidden_dim():
    model = build_protocol_model(ModelConfig("gru", 1, 2, width=8))
    assert model.core.state_size == 8
    assert model.core.cell.hidden_size == 8


def test_sagodi_gru_variants_use_tanh_wotr_and_direct_linear_readout():
    protocol = load_protocol(NATIVE_RECIPE_PROTOCOL_PATH)
    for name, width, expected_count in (
        (SAGODI_GRU_WIDTH96, 96, 28898),
        (SAGODI_GRU_PARAM135, 135, 56432),
    ):
        torch.manual_seed(23)
        config = model_config_from_protocol(protocol, name)
        model = build_protocol_model(config)
        assert isinstance(model.core, SagodiGRUBaseline)
        assert config.width == width
        assert config.initial_encoder_bias is False
        assert config.initial_encoder_weight_init == INITIAL_ENCODER_SAGODI_W_OTR
        assert config.initial_encoder_activation == INITIAL_ENCODER_TANH
        assert model.initial_encoder.bias is None
        assert isinstance(model.core.readout, torch.nn.Linear)
        assert model.core.readout.bias is not None
        assert model.metadata()["parameters_total"] == expected_count
        assert not torch.equal(
            model.core.cell.bias_ih.detach(),
            torch.zeros_like(model.core.cell.bias_ih),
        )
        memory = torch.randn(4, 2)
        state = model.initial_state(4, "cpu", memory)
        expected = torch.tanh(model.initial_encoder(memory))
        torch.testing.assert_close(state, expected)
        output, states = model.forward_sequence(
            torch.zeros(3, 4, 1),
            initial_memory=memory,
            return_states=True,
        )
        assert output.shape == (3, 4, 2)
        assert states.shape == (3, 4, width)
        contract = model.metadata()["sagodi_gru_contract"]
        assert contract["readout"] == "direct_biased_linear"
        assert "uninitialized" in contract["output_to_hidden_initialization_repair"]


def test_sagodi_gru_widths_are_fail_closed():
    with torch.no_grad(), pytest.raises(ValueError, match="freezes hidden width 96"):
        build_protocol_model(
            ModelConfig(
                SAGODI_GRU_WIDTH96,
                1,
                2,
                width=95,
                initial_encoder_bias=False,
                initial_encoder_weight_init=INITIAL_ENCODER_SAGODI_W_OTR,
                initial_encoder_activation=INITIAL_ENCODER_TANH,
            )
        )


def test_sagodi_grus_expose_phase0_step_interface() -> None:
    protocol = load_protocol(NATIVE_RECIPE_PROTOCOL_PATH)
    for name, width in (
        (SAGODI_GRU_WIDTH96, 96),
        (SAGODI_GRU_PARAM135, 135),
    ):
        model = build_protocol_model(model_config_from_protocol(protocol, name))
        adapter = StateAdapter(model.core)
        assert adapter.input_dim == 1
        assert adapter.primary_dim == width


@pytest.mark.parametrize(
    ("name", "width", "primary_dim", "reported_dim", "expected_count", "legacy_variant"),
    (
        (RNN_PARAM206, 206, 206, 206, 56844, "RNN"),
        (LSTM_PARAM109, 109, 218, 218, 56438, "LSTM"),
        (LRU_PARAM96, 96, 192, 288, 56834, "LRU full"),
    ),
)
def test_project_parameter_baselines_are_fixed_and_explicitly_labeled(
    name: str,
    width: int,
    primary_dim: int,
    reported_dim: int,
    expected_count: int,
    legacy_variant: str,
) -> None:
    protocol = load_protocol(NATIVE_RECIPE_PROTOCOL_PATH)
    config = model_config_from_protocol(protocol, name)
    model = build_protocol_model(config)

    assert config.width == width
    assert config.initial_encoder_bias is False
    assert config.initial_encoder_weight_init == INITIAL_ENCODER_SAGODI_W_OTR
    assert model.initial_encoder.bias is None
    assert model.primary_state_size == primary_dim
    assert model.reported_state_size == reported_dim
    assert model.state_size == reported_dim

    metadata = model.metadata()
    assert metadata["legacy_variant"] == legacy_variant
    assert metadata["parameters_total"] == expected_count
    assert metadata["parameters_trainable"] == expected_count
    contract = metadata["project_parameter_matched_baseline_contract"]
    assert contract["official_sagodi_architecture"] is False
    assert contract["architecture_provenance"] == "project_legacy_comparison_scaffold"
    assert contract["matching_role"] == "nearest_parameter_match_to_CA_LRU_56834"
    assert contract["target_parameter_count"] == 56834
    assert contract["actual_parameter_count"] == expected_count
    assert contract["parameter_count_delta"] == expected_count - 56834

    adapter = StateAdapter(model.core)
    assert adapter.primary_dim == primary_dim
    assert adapter.reported_dim == reported_dim
    memory = torch.randn(3, 2)
    output, states = model.forward_sequence(
        torch.zeros(2, 3, 1),
        initial_memory=memory,
        return_states=True,
    )
    assert output.shape == (2, 3, 2)
    assert states.shape == (2, 3, reported_dim)
    assert torch.isfinite(output).all()


def test_project_parameter_baseline_registry_counts_and_widths_are_complete() -> None:
    assert PROJECT_PARAM_BASELINE_NAMES == (RNN_PARAM206, LSTM_PARAM109, LRU_PARAM96)
    assert PROJECT_PARAM_BASELINE_WIDTHS == {
        RNN_PARAM206: 206,
        LSTM_PARAM109: 109,
        LRU_PARAM96: 96,
    }
    assert PROJECT_PARAM_BASELINE_PARAMETER_COUNTS == {
        RNN_PARAM206: 56844,
        LSTM_PARAM109: 56438,
        LRU_PARAM96: 56834,
    }


def test_lstm_param109_initializes_and_evolves_the_full_h_c_markov_state() -> None:
    protocol = load_protocol(NATIVE_RECIPE_PROTOCOL_PATH)
    model = build_protocol_model(model_config_from_protocol(protocol, LSTM_PARAM109))
    memory = torch.randn(4, 2)
    state = model.initial_state(4, "cpu", memory)
    expected = torch.tanh(model.initial_encoder(memory))
    torch.testing.assert_close(state, expected)
    h, c = state.split(109, dim=-1)
    assert h.shape == c.shape == (4, 109)
    next_state = model.step(torch.zeros(4, 1), state)
    assert next_state.shape == (4, 218)
    assert model.metadata()["project_parameter_matched_baseline_contract"][
        "state_semantics"
    ] == "full_concatenated_h_c_Markov_state"


def test_lru_param96_uses_existing_full_complex_project_baseline() -> None:
    protocol = load_protocol(NATIVE_RECIPE_PROTOCOL_PATH)
    model = build_protocol_model(model_config_from_protocol(protocol, LRU_PARAM96))
    core = model.core
    recurrence = core.blocks[0].rec

    assert core.variant == "LRU-Block"
    assert core.encoder.bias is None
    assert isinstance(core.blocks[0].norm_in, torch.nn.Identity)
    assert recurrence.__class__.__name__ == "ComplexLRURec"
    assert recurrence.B_re.shape == recurrence.B_im.shape == (96, 96)
    assert model.primary_state_size == 2 * 96
    assert model.reported_state_size == 3 * 96
    initial = model.initial_state(2, "cpu", torch.randn(2, 2))
    assert initial[:, :192].shape == (2, 192)
    torch.testing.assert_close(initial[:, 192:], torch.zeros(2, 96))
    contract = model.metadata()["project_parameter_matched_baseline_contract"]
    assert contract["state_semantics"] == (
        "full_real_imaginary_complex_carrier_excluding_overwritten_stream"
    )
    assert contract["raw_LRU_Block_excluded_reason"] == (
        "enables_encoder_bias_and_affine_input_LayerNorm"
    )


@pytest.mark.parametrize(
    ("name", "width", "activation"),
    (
        (RNN_PARAM206, 205, INITIAL_ENCODER_IDENTITY),
        (LSTM_PARAM109, 108, INITIAL_ENCODER_TANH),
        (LRU_PARAM96, 95, INITIAL_ENCODER_IDENTITY),
    ),
)
def test_project_parameter_baseline_widths_are_fail_closed(
    name: str, width: int, activation: str
) -> None:
    with pytest.raises(ValueError, match="freezes hidden width"):
        build_protocol_model(
            ModelConfig(
                name,
                1,
                2,
                width=width,
                initial_encoder_bias=False,
                initial_encoder_weight_init=INITIAL_ENCODER_SAGODI_W_OTR,
                initial_encoder_activation=activation,
            )
        )
