from __future__ import annotations

import torch

from repro.state_dependent_retention.models import (
    WRITER_KINDS,
    build_state_dependent_model,
    recurrence,
)


def test_three_writers_share_exact_state_dependent_blank_map() -> None:
    states = torch.randn(5, 52)
    gates = []
    for writer in WRITER_KINDS:
        model = build_state_dependent_model(writer, model_seed=7)
        rec = recurrence(model)
        blank = torch.zeros(5, 52)
        expected = rec.state_dependent_lambda(states) * states
        assert torch.equal(rec.step(blank, states), expected)
        assert rec.theta.requires_grad
        assert not model.rp_enabled
        gates.append(rec.retention_gate.state_dict())
    for key in gates[0]:
        assert torch.equal(gates[0][key], gates[1][key])
        assert torch.equal(gates[0][key], gates[2][key])


def test_writer_equations_and_gamma_placement() -> None:
    states = torch.randn(4, 52)
    inputs = torch.randn(4, 52)
    for writer in WRITER_KINDS:
        rec = recurrence(build_state_dependent_model(writer, model_seed=11))
        retained = rec.state_dependent_lambda(states) * states
        written = rec.write(inputs, states)
        expected = retained + (rec.gamma() * written if writer == "recurrent" else written)
        assert torch.allclose(rec.step(inputs, states), expected)
        assert (rec.gamma_raw is not None) == (writer == "recurrent")


def test_dynamic_retention_starts_at_base_and_can_cross_one() -> None:
    rec = recurrence(build_state_dependent_model("linear", model_seed=3))
    states = torch.randn(6, 52)
    initial = rec.state_dependent_lambda(states)
    assert torch.allclose(initial, rec.lam_mag().expand_as(initial))
    with torch.no_grad():
        rec.retention_gate[2].bias.fill_(10.0)
    modulated = rec.state_dependent_lambda(states)
    assert bool((modulated[:, 0] > 1.0).all())


def test_direct_residual_mlp_starts_at_base_and_is_not_amplitude_bounded() -> None:
    rec = recurrence(
        build_state_dependent_model(
            "recurrent",
            model_seed=3,
            retention_parameterization="direct_residual_mlp",
        )
    )
    states = torch.randn(6, 52)
    initial = rec.state_dependent_lambda(states)
    assert torch.allclose(initial, rec.lam_mag().expand_as(initial))
    with torch.no_grad():
        rec.retention_gate[2].bias.fill_(0.2)
    shifted = rec.state_dependent_lambda(states)
    assert torch.allclose(shifted, rec.lam_mag().expand_as(shifted) + 0.2)


def test_power_tanh_mlp_is_bounded_neutral_and_expansive() -> None:
    rec = recurrence(
        build_state_dependent_model(
            "recurrent",
            model_seed=3,
            retention_parameterization="power_tanh_mlp",
            power_tanh_scale=1.5,
        )
    )
    states = torch.randn(6, 52)
    initial = rec.state_dependent_lambda(states)
    assert torch.allclose(initial, rec.lam_mag().expand_as(initial))
    with torch.no_grad():
        neutral_bias = torch.atanh(torch.tensor(-1.0 / 1.5))
        rec.retention_gate[2].bias.fill_(neutral_bias)
    neutral = rec.state_dependent_lambda(states)
    assert torch.allclose(neutral, torch.ones_like(neutral))
    with torch.no_grad():
        rec.retention_gate[2].bias.fill_(-10.0)
    expansive = rec.state_dependent_lambda(states)
    assert bool((expansive > 1.0).all())
    expected_upper = rec.lam_mag().pow(-0.5).expand_as(expansive)
    assert bool((expansive <= expected_upper + 1.0e-6).all())


def test_explicit_gate_output_initializations_are_seeded_and_bounded() -> None:
    kwargs = {
        "model_seed": 13,
        "max_log_modulation": 0.025,
        "gate_output_weight_std": 0.01,
        "gate_output_bias": -0.1,
    }
    first = recurrence(build_state_dependent_model("recurrent", **kwargs))
    second = recurrence(build_state_dependent_model("recurrent", **kwargs))
    assert torch.equal(first.retention_gate[2].weight, second.retention_gate[2].weight)
    assert bool((first.retention_gate[2].weight != 0.0).any())
    assert torch.allclose(
        first.retention_gate[2].bias,
        torch.full_like(first.retention_gate[2].bias, -0.1),
    )
    values = first.state_dependent_lambda(torch.randn(16, 52)) / first.lam_mag()
    assert bool((values >= torch.exp(torch.tensor(-0.025))).all())
    assert bool((values <= torch.exp(torch.tensor(0.025))).all())


def test_task_gradient_reaches_base_and_state_gate() -> None:
    rec = recurrence(build_state_dependent_model("recurrent", model_seed=5))
    states = torch.randn(8, 52)
    inputs = torch.randn(8, 52)
    rec.step(inputs, states).square().mean().backward()
    assert rec.theta.grad is not None
    assert rec.retention_gate[2].weight.grad is not None
    assert bool(torch.isfinite(rec.theta.grad).all())
    assert bool(torch.isfinite(rec.retention_gate[2].weight.grad).all())


def test_hybrid_rp_detaches_only_base_retention() -> None:
    model = build_state_dependent_model(
        "recurrent", model_seed=5, retention_mode="hybrid_rp"
    )
    rec = recurrence(model)
    states = torch.randn(8, 52)
    inputs = torch.randn(8, 52)
    rec.step(inputs, states).square().mean().backward()
    assert model.rp_enabled
    assert not rec.theta.requires_grad
    assert rec.theta.grad is None
    assert rec.retention_gate[2].weight.grad is not None
    assert bool(torch.isfinite(rec.retention_gate[2].weight.grad).all())
