from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from repro.sagodi_protocol.exact_models import (
    EXACT_MODEL_NAMES,
    SAGODI_GRU,
    SAGODI_LSTM,
    SAGODI_RNN_TANH,
    SagodiExactGRU,
    SagodiExactLSTM,
    SagodiExactRNN,
    build_exact_core,
    exact_core_parameter_count,
    exact_model_parameter_count,
)
from repro.sagodi_protocol.state import StateAdapter


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


@pytest.mark.parametrize("name", EXACT_MODEL_NAMES)
def test_builder_returns_state_adapter_compatible_core(name: str) -> None:
    core = build_exact_core(name, input_dim=1, output_dim=2, hidden=7)
    adapter = StateAdapter(core)
    state = adapter.init_reported(batch=3, device=torch.device("cpu"))
    next_state = adapter.reported_step(state, torch.randn(3, 1))
    output = adapter.decode(next_state)

    expected_state_size = 14 if name == SAGODI_LSTM else 7
    assert core.input_dim == 1
    assert core.output_dim == 2
    assert core.state_size == expected_state_size
    assert state.shape == (3, expected_state_size)
    assert next_state.shape == state.shape
    assert output.shape == (3, 2)


def test_rnn_step_and_decode_match_paper_equations() -> None:
    torch.manual_seed(11)
    core = SagodiExactRNN(2, 3, 5)
    x_t = torch.randn(4, 2)
    state = torch.randn(4, 5)

    proposal = torch.tanh(
        state @ core.wrec.t() + x_t @ core.wi + core.brec
    )
    expected_state = 0.9 * state + 0.1 * proposal
    expected_output = expected_state @ core.wo + core.bo

    torch.testing.assert_close(core.step(x_t, state), expected_state)
    torch.testing.assert_close(core.decode(expected_state), expected_output)
    assert core.dt == 0.1
    assert core.recurrent_gain == 1.5


def test_rnn_initialization_is_deterministic_and_matches_contract() -> None:
    torch.manual_seed(29)
    first = SagodiExactRNN(64, 7, 512)
    torch.manual_seed(29)
    second = SagodiExactRNN(64, 7, 512)

    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)

    assert torch.count_nonzero(first.brec) == 0
    assert torch.count_nonzero(first.bo) == 0
    expected_wrec_std = 1.5 / math.sqrt(512)
    expected_wi_std = math.sqrt(2.0 / (64 + 512))
    expected_wo_std = math.sqrt(2.0 / (512 + 7))
    assert float(first.wrec.std(unbiased=False)) == pytest.approx(
        expected_wrec_std, rel=0.02
    )
    assert float(first.wi.std(unbiased=False)) == pytest.approx(
        expected_wi_std, rel=0.04
    )
    assert float(first.wo.std(unbiased=False)) == pytest.approx(
        expected_wo_std, rel=0.06
    )


def test_gru_initialization_and_step_are_standard_grucell_parity() -> None:
    torch.manual_seed(41)
    core = SagodiExactGRU(3, 2, 6)
    torch.manual_seed(41)
    reference_cell = nn.GRUCell(3, 6, bias=True)
    reference_readout = nn.Linear(6, 2, bias=True)

    for key, value in core.cell.state_dict().items():
        torch.testing.assert_close(value, reference_cell.state_dict()[key], rtol=0, atol=0)
    for key, value in core.readout.state_dict().items():
        torch.testing.assert_close(
            value, reference_readout.state_dict()[key], rtol=0, atol=0
        )

    x_t = torch.randn(4, 3)
    state = torch.randn(4, 6)
    torch.testing.assert_close(core.step(x_t, state), reference_cell(x_t, state))
    torch.testing.assert_close(core.decode(state), reference_readout(state))


def test_lstm_packed_state_matches_standard_lstmcell_and_decodes_h_only() -> None:
    torch.manual_seed(43)
    core = SagodiExactLSTM(3, 2, 6)
    torch.manual_seed(43)
    reference_cell = nn.LSTMCell(3, 6, bias=True)
    reference_readout = nn.Linear(6, 2, bias=True)

    x_t = torch.randn(4, 3)
    hidden = torch.randn(4, 6)
    cell = torch.randn(4, 6)
    packed = torch.cat((hidden, cell), dim=-1)
    expected_hidden, expected_cell = reference_cell(x_t, (hidden, cell))

    for key, value in core.cell.state_dict().items():
        torch.testing.assert_close(value, reference_cell.state_dict()[key], rtol=0, atol=0)
    expected_packed = torch.cat((expected_hidden, expected_cell), dim=-1)
    torch.testing.assert_close(core.step(x_t, packed), expected_packed)
    torch.testing.assert_close(core.decode(packed), reference_readout(hidden))
    assert core.state_size == 12


@pytest.mark.parametrize("name", EXACT_MODEL_NAMES)
def test_analytic_parameter_counts_match_modules(name: str) -> None:
    input_dim, output_dim, hidden = 1, 2, 11
    core = build_exact_core(name, input_dim, output_dim, hidden)
    assert exact_core_parameter_count(name, input_dim, output_dim, hidden) == (
        _parameter_count(core)
    )

    encoded_width = core.state_size
    expected_with_wotr = _parameter_count(core) + 2 * encoded_width
    assert exact_model_parameter_count(
        name, input_dim, output_dim, hidden
    ) == expected_with_wotr


def test_builder_rejects_unknown_exact_model() -> None:
    with pytest.raises(ValueError, match="unknown exact Ságodi model"):
        build_exact_core("project_rnn", 1, 2, 8)
