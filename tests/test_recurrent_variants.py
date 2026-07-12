from pathlib import Path
import math
import sys

import pytest
import torch


LEGACY_CODE = Path(__file__).resolve().parents[1] / "repro" / "legacy_code"
sys.path.insert(0, str(LEGACY_CODE))

from exp71_pan_block_pulse_hold import (  # noqa: E402
    ADDITIONAL_MODEL_VARIANTS,
    MODEL_VARIANTS,
    build_model_variant,
    normalize_model_variant,
    select_scaffold_matched_rec_dim,
)
from pan_block import (  # noqa: E402
    GRURec,
    RGLRURec,
    build_block_model,
    count_model_parameters,
    count_trainable_parameters,
    normalize_block_variant,
)


def _set_identity(linear):
    with torch.no_grad():
        linear.weight.copy_(torch.eye(linear.out_features, linear.in_features))
        if linear.bias is not None:
            linear.bias.zero_()


def test_rg_lru_step_matches_griffin_equations():
    rec = RGLRURec(
        input_dim=4,
        hidden_dim=4,
        output_dim=4,
        num_blocks=2,
        recurrence_scale=8.0,
    )
    base_radius = 0.95
    softplus_value = -math.log(base_radius)
    a_param = math.log(math.expm1(softplus_value))

    _set_identity(rec.input_proj)
    _set_identity(rec.out_proj)
    with torch.no_grad():
        rec.input_gate.weight.zero_()
        rec.input_gate.bias.zero_()
        rec.recurrence_gate.weight.zero_()
        rec.recurrence_gate.bias.zero_()
        rec.a_param.fill_(a_param)

    x_t = torch.tensor([[0.2, -0.4, 0.6, -0.8]], requires_grad=True)
    state = torch.tensor([[0.3, -0.2, 0.1, 0.5]])
    actual = rec.step(x_t, state)

    gate_x = torch.full_like(x_t, 0.5)
    gate_a = torch.full_like(x_t, 0.5)
    log_a = -8.0 * gate_a * softplus_value
    a = torch.exp(log_a)
    expected = a * state + torch.sqrt(1.0 - a.square()) * gate_x * x_t
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(rec.output(actual), actual)

    (actual.square().sum() + rec.output(actual).square().sum()).backward()
    assert x_t.grad is not None
    assert torch.isfinite(x_t.grad).all()
    for parameter in rec.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_rg_lru_ring_initialization_and_block_validation():
    torch.manual_seed(4)
    rec = RGLRURec(8, 64, 8, num_blocks=16)
    radii = rec.lam_mag().detach()
    assert float(radii.min()) >= 0.90 - 1e-6
    assert float(radii.max()) <= 0.999 + 1e-6

    with pytest.raises(ValueError, match="must be divisible"):
        RGLRURec(8, 30, 8, num_blocks=16)


def test_rg_lru_blank_input_has_no_additive_injection():
    rec = RGLRURec(8, 32, 8, num_blocks=8)
    state = torch.zeros(3, 32)
    blank = torch.zeros(3, 8)
    torch.testing.assert_close(rec.step(blank, state), state)


@pytest.mark.parametrize(
    "variant,rec_dim,num_blocks",
    [
        ("RG-LRU-Block", 32, 4),
        ("GRU-Block", 24, 4),
    ],
)
def test_full_block_step_state_output_interface(variant, rec_dim, num_blocks):
    torch.manual_seed(7)
    model = build_block_model(
        variant=variant,
        input_dim=6,
        output_dim=3,
        d_model=16,
        rec_dim=rec_dim,
        num_layers=2,
        dropout=0.0,
        rglru_num_blocks=num_blocks,
    )
    x = torch.randn(5, 3, 6)
    out, states = model(x, return_states=True)

    assert model.state_size == 2 * rec_dim + 16
    assert out.shape == (5, 3, 3)
    assert states.shape == (5, 3, model.state_size)

    state = model.init_state(batch=3, device=x.device)
    manual_out = []
    for x_t in x:
        state = model.step(x_t, state)
        manual_out.append(model.decode(state))
    torch.testing.assert_close(torch.stack(manual_out), out)
    torch.testing.assert_close(state, states[-1])

    out.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_exp71_aliases_are_opt_in_and_build_shared_scaffold():
    assert normalize_model_variant("rglru") == "RG-LRU-full"
    assert normalize_model_variant("matched-gru") == "GRU-full"
    assert normalize_model_variant("GRU") == "GRU"
    assert normalize_block_variant("griffin-rg-lru") == "RG-LRU-Block"
    assert normalize_block_variant("scaffold-gru") == "GRU-Block"
    assert set(ADDITIONAL_MODEL_VARIANTS).isdisjoint(MODEL_VARIANTS)

    common = dict(
        input_dim=6,
        output_dim=2,
        rank=2,
        d_model=16,
        rec_dim=16,
        layers=1,
        dropout=0.0,
        plru_tau=0.001,
        plru_c=50.0,
        pan_lambda_min=0.90,
        pan_lambda_max=0.999,
        rank_matched_lambda_high=0.999,
        rank_matched_lambda_low=0.0,
        rglru_num_blocks=4,
    )
    rg_lru = build_model_variant("RG-LRU-full", **common)
    gru = build_model_variant("GRU-full", **common)
    assert rg_lru.variant == "RG-LRU-Block"
    assert gru.variant == "GRU-Block"
    assert isinstance(rg_lru.blocks[0].rec, RGLRURec)
    assert isinstance(gru.blocks[0].rec, GRURec)
    assert isinstance(rg_lru.blocks[0].norm_in, torch.nn.Identity)
    assert isinstance(gru.blocks[0].norm_in, torch.nn.Identity)
    assert rg_lru.encoder.bias is None
    assert gru.encoder.bias is None


def test_parameter_matching_hits_exp88_five_percent_budget():
    torch.manual_seed(1234)
    rng_before = torch.random.get_rng_state().clone()
    rg_lru = select_scaffold_matched_rec_dim(
        "RG-LRU-full",
        input_dim=6,
        output_dim=2,
    )
    gru = select_scaffold_matched_rec_dim(
        "GRU-full",
        input_dim=6,
        output_dim=2,
    )

    assert rg_lru.candidate_rec_dim == 176
    assert rg_lru.relative_error < 0.05
    assert gru.candidate_rec_dim == 64
    assert gru.relative_error < 0.05
    assert rg_lru.target_params == gru.target_params == 57122
    assert rg_lru.target_trainable_params == gru.target_trainable_params == 57026
    assert gru.candidate_params == 57122
    assert gru.relative_error == 0.0
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_existing_exp88_ca_lru_parameter_count_is_unchanged():
    model = build_model_variant(
        variant="PAN-RNW-full",
        input_dim=6,
        output_dim=2,
        rank=2,
        d_model=96,
        rec_dim=96,
        layers=1,
        dropout=0.0,
        plru_tau=0.001,
        plru_c=50.0,
        pan_lambda_min=0.90,
        pan_lambda_max=0.999,
        rank_matched_lambda_high=0.999,
        rank_matched_lambda_low=0.0,
    )
    assert count_trainable_parameters(model) == 57026
    assert count_model_parameters(model) == 57122
