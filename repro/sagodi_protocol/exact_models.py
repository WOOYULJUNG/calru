"""Paper-first recurrent cores for the Ságodi comparison protocol.

This module deliberately contains only the recurrent core and readout.  The
learned output-to-recurrent initial-state map (``W_otr``), protocol state
noise, sequence unrolling, and artifact metadata belong to the surrounding
``ProtocolModel`` compatibility layer.  Keeping that boundary explicit makes
it possible to audit the baseline recurrence without silently changing it to
match a project-specific scaffold.

The public classes implement the flat-state interface consumed by
``StateAdapter``: ``init_state``, ``step``, ``decode``, and ``state_size``.
For the LSTM the flat Markov state is ordered as ``[h, c]``; only ``h`` is
decoded by the direct linear readout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import nn


SAGODI_RNN_TANH: Final = "sagodi_rnn_tanh"
SAGODI_GRU: Final = "sagodi_gru"
SAGODI_LSTM: Final = "sagodi_lstm"
EXACT_MODEL_NAMES: Final = (SAGODI_RNN_TANH, SAGODI_GRU, SAGODI_LSTM)


@dataclass(frozen=True)
class ExactModelSpec:
    """Static state semantics for a paper-first baseline core."""

    name: str
    state_width_multiplier: int
    decoded_component: str


MODEL_SPECS: Final = {
    SAGODI_RNN_TANH: ExactModelSpec(SAGODI_RNN_TANH, 1, "full_state"),
    SAGODI_GRU: ExactModelSpec(SAGODI_GRU, 1, "full_state"),
    SAGODI_LSTM: ExactModelSpec(SAGODI_LSTM, 2, "h_from_packed_h_c"),
}


def _validate_dimensions(input_dim: int, output_dim: int, hidden: int) -> None:
    for name, value in (
        ("input_dim", input_dim),
        ("output_dim", output_dim),
        ("hidden", hidden),
    ):
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _zero_state(
    reference: torch.Tensor,
    batch: int,
    state_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    if isinstance(batch, bool) or int(batch) != batch or int(batch) <= 0:
        raise ValueError(f"batch must be a positive integer, got {batch!r}")
    return torch.zeros(
        int(batch),
        int(state_size),
        device=device,
        dtype=reference.dtype,
    )


class SagodiExactRNN(nn.Module):
    r"""Ságodi paper-first leaky tanh RNN.

    The one-step recurrence is

    .. math::

       h_{t+1}=(1-\Delta t)h_t + \Delta t\tanh(
       x_t W_{in} + h_t W_{rec}^{\mathsf T} + b_{rec}).

    ``W_rec`` is initialized from ``Normal(0, g / sqrt(H))`` with ``g=1.5``.
    The input and output matrices use Xavier-normal initialization and all
    biases are initialized to zero.  State noise is intentionally not added
    here; the protocol wrapper adds it after this deterministic transition.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: int,
        *,
        dt: float = 0.1,
        recurrent_gain: float = 1.5,
    ) -> None:
        super().__init__()
        _validate_dimensions(input_dim, output_dim, hidden)
        if not math.isfinite(float(dt)) or not 0.0 < float(dt) <= 1.0:
            raise ValueError(f"dt must lie in (0, 1], got {dt!r}")
        if not math.isfinite(float(recurrent_gain)) or float(recurrent_gain) <= 0.0:
            raise ValueError(
                f"recurrent_gain must be finite and positive, got {recurrent_gain!r}"
            )

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden = int(hidden)
        self.hidden_size = self.hidden
        self.dt = float(dt)
        self.recurrent_gain = float(recurrent_gain)

        # The orientations mirror the pinned Ságodi source: wi is I x H,
        # wrec is H x H, and wo is H x O.
        self.wi = nn.Parameter(torch.empty(self.input_dim, self.hidden))
        self.wrec = nn.Parameter(torch.empty(self.hidden, self.hidden))
        self.brec = nn.Parameter(torch.empty(self.hidden))
        self.wo = nn.Parameter(torch.empty(self.hidden, self.output_dim))
        self.bo = nn.Parameter(torch.empty(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.wi)
        nn.init.normal_(
            self.wrec,
            mean=0.0,
            std=self.recurrent_gain / math.sqrt(float(self.hidden)),
        )
        nn.init.zeros_(self.brec)
        nn.init.xavier_normal_(self.wo)
        nn.init.zeros_(self.bo)

    @property
    def state_size(self) -> int:
        return self.hidden

    def init_state(self, batch: int, device: torch.device | str) -> torch.Tensor:
        return _zero_state(self.wi, batch, self.state_size, device)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        recurrent_drive = state.matmul(self.wrec.t())
        input_drive = x_t.matmul(self.wi)
        proposal = torch.tanh(recurrent_drive + input_drive + self.brec)
        return (1.0 - self.dt) * state + self.dt * proposal

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return state.matmul(self.wo) + self.bo


class SagodiExactGRU(nn.Module):
    """One-layer standard GRUCell with a direct linear readout."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int) -> None:
        super().__init__()
        _validate_dimensions(input_dim, output_dim, hidden)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden = int(hidden)
        self.hidden_size = self.hidden
        # Do not reset these modules: their constructors reproduce the
        # standard PyTorch initialization used by the architecture baseline.
        self.cell = nn.GRUCell(self.input_dim, self.hidden, bias=True)
        self.readout = nn.Linear(self.hidden, self.output_dim, bias=True)

    @property
    def state_size(self) -> int:
        return self.hidden

    def init_state(self, batch: int, device: torch.device | str) -> torch.Tensor:
        return _zero_state(self.cell.weight_ih, batch, self.state_size, device)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.cell(x_t, state)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.readout(state)


class SagodiExactLSTM(nn.Module):
    """One-layer standard LSTMCell with flat ``[h, c]`` Markov state."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int) -> None:
        super().__init__()
        _validate_dimensions(input_dim, output_dim, hidden)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden = int(hidden)
        self.hidden_size = self.hidden
        self.cell = nn.LSTMCell(self.input_dim, self.hidden, bias=True)
        self.readout = nn.Linear(self.hidden, self.output_dim, bias=True)

    @property
    def state_size(self) -> int:
        return 2 * self.hidden

    def init_state(self, batch: int, device: torch.device | str) -> torch.Tensor:
        return _zero_state(self.cell.weight_ih, batch, self.state_size, device)

    def split_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if state.shape[-1] != self.state_size:
            raise ValueError(
                f"LSTM state trailing dimension must be {self.state_size}, "
                f"got {state.shape[-1]}"
            )
        return state[..., : self.hidden], state[..., self.hidden :]

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        hidden, cell = self.split_state(state)
        next_hidden, next_cell = self.cell(x_t, (hidden, cell))
        return torch.cat((next_hidden, next_cell), dim=-1)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.split_state(state)
        return self.readout(hidden)


def build_exact_core(
    name: str,
    input_dim: int,
    output_dim: int,
    hidden: int,
) -> nn.Module:
    """Build one of the three frozen Ságodi baseline cores."""

    if name == SAGODI_RNN_TANH:
        return SagodiExactRNN(input_dim, output_dim, hidden)
    if name == SAGODI_GRU:
        return SagodiExactGRU(input_dim, output_dim, hidden)
    if name == SAGODI_LSTM:
        return SagodiExactLSTM(input_dim, output_dim, hidden)
    raise ValueError(
        f"unknown exact Ságodi model {name!r}; expected one of {EXACT_MODEL_NAMES}"
    )


def exact_core_parameter_count(
    name: str,
    input_dim: int,
    output_dim: int,
    hidden: int,
) -> int:
    """Return the analytic trainable-parameter count of an exact core."""

    _validate_dimensions(input_dim, output_dim, hidden)
    i, o, h = int(input_dim), int(output_dim), int(hidden)
    if name == SAGODI_RNN_TANH:
        return i * h + h * h + h + h * o + o
    if name == SAGODI_GRU:
        return 3 * i * h + 3 * h * h + 6 * h + h * o + o
    if name == SAGODI_LSTM:
        return 4 * i * h + 4 * h * h + 8 * h + h * o + o
    raise ValueError(
        f"unknown exact Ságodi model {name!r}; expected one of {EXACT_MODEL_NAMES}"
    )


def exact_model_parameter_count(
    name: str,
    input_dim: int,
    output_dim: int,
    hidden: int,
    *,
    initial_memory_dim: int = 2,
    initial_encoder_bias: bool = False,
) -> int:
    """Count a core plus the protocol-owned flat-state ``W_otr`` encoder.

    This helper does not construct or own ``W_otr``.  It makes the comparison
    count explicit for width matching; an LSTM encoder maps to both packed
    components and therefore has output dimension ``2 * hidden``.
    """

    if (
        isinstance(initial_memory_dim, bool)
        or int(initial_memory_dim) != initial_memory_dim
        or int(initial_memory_dim) <= 0
    ):
        raise ValueError("initial_memory_dim must be a positive integer")
    core_count = exact_core_parameter_count(name, input_dim, output_dim, hidden)
    state_multiplier = MODEL_SPECS[name].state_width_multiplier
    encoded_width = state_multiplier * int(hidden)
    encoder_count = int(initial_memory_dim) * encoded_width
    if bool(initial_encoder_bias):
        encoder_count += encoded_width
    return core_count + encoder_count


__all__ = [
    "EXACT_MODEL_NAMES",
    "MODEL_SPECS",
    "SAGODI_GRU",
    "SAGODI_LSTM",
    "SAGODI_RNN_TANH",
    "ExactModelSpec",
    "SagodiExactGRU",
    "SagodiExactLSTM",
    "SagodiExactRNN",
    "build_exact_core",
    "exact_core_parameter_count",
    "exact_model_parameter_count",
]
