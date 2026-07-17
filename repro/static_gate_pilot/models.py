"""Experimental recurrent cells with static, coordinate-wise retention.

The module is intentionally independent of the registered CA-LRU models.  It
implements two equations from the static-gate side-pilot note:

``static_gru``
    h' = Lambda h + (I - Lambda) tanh(Wx x + Wh (r * h) + b)

``untied_rnn`` (shared-field control)
    h' = Lambda h + (I - Lambda) f0(h)
         + Gamma [fx(h, x) - f0(h)]

``split_rnn``
    h' = Lambda h + (I - Lambda) fh(h)
         + Gamma [fx(h, x) - fx(h, 0)]

Each equation has a gradient-trained retention control and an RP-only
retention variant.  RP variants keep ``theta`` out of task-gradient learning.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from repro.manifold_benchmark.topology_models import topology_dimensions


MODEL_IDS = (
    "static_gru_grad",
    "static_gru_rp",
    "untied_rnn_grad",
    "untied_rnn_rp",
    "split_rnn_grad",
    "split_rnn_rp",
)
CellKind = Literal["static_gru", "untied_rnn", "split_rnn"]


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("initial retention must lie strictly between zero and one")
    return math.log(probability / (1.0 - probability))


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("initial write gain must be positive")
    return math.log(math.expm1(value))


class StaticGateMemory(nn.Module):
    """Topology boundary maps around one experimental static-gate cell."""

    def __init__(
        self,
        *,
        model_id: str,
        topology: str,
        width: int = 52,
        initial_retention: float = 0.95,
        initial_write_gain: float = 1.0,
        recurrent_gain: float = 0.9,
    ) -> None:
        super().__init__()
        if model_id not in MODEL_IDS:
            raise ValueError(f"unknown static-gate model {model_id!r}")
        input_dim, initial_memory_dim, output_dim = topology_dimensions(topology)
        self.model_id = str(model_id)
        self.topology = str(topology)
        if model_id.startswith("static_gru"):
            self.cell_kind: CellKind = "static_gru"
        elif model_id.startswith("split_rnn"):
            self.cell_kind = "split_rnn"
        else:
            self.cell_kind = "untied_rnn"
        self.rp_enabled = model_id.endswith("_rp")
        self.width = int(width)
        self.input_dim = int(input_dim)
        self.initial_memory_dim = int(initial_memory_dim)
        self.output_dim = int(output_dim)
        self.state_size = self.width
        self.initial_retention = float(initial_retention)
        self.initial_write_gain = float(initial_write_gain)
        self.recurrent_gain = float(recurrent_gain)
        if self.recurrent_gain <= 0.0:
            raise ValueError("recurrent gain must be positive")

        self.initial_encoder = nn.Linear(self.initial_memory_dim, self.width, bias=False)
        self.decoder = nn.Linear(self.width, self.output_dim)
        self.theta = nn.Parameter(
            torch.full((self.width,), _logit(float(initial_retention))),
            requires_grad=not self.rp_enabled,
        )

        if self.cell_kind == "static_gru":
            self.reset_input = nn.Linear(self.input_dim, self.width, bias=True)
            self.reset_hidden = nn.Linear(self.width, self.width, bias=False)
            self.candidate_input = nn.Linear(self.input_dim, self.width, bias=False)
            self.candidate_hidden = nn.Linear(self.width, self.width, bias=True)
            self.autonomous_hidden = None
            self.writer_hidden = None
            self.input_write = None
            self.raw_gamma = None
        else:
            self.reset_input = None
            self.reset_hidden = None
            self.candidate_input = None
            self.candidate_hidden = None
            self.autonomous_hidden = nn.Linear(self.width, self.width, bias=True)
            self.writer_hidden = (
                nn.Linear(self.width, self.width, bias=True)
                if self.cell_kind == "split_rnn"
                else None
            )
            self.input_write = nn.Linear(self.input_dim, self.width, bias=False)
            self.raw_gamma = nn.Parameter(
                torch.full((self.width,), _inverse_softplus(initial_write_gain))
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.initial_encoder.weight,
            mean=0.0,
            std=1.0 / math.sqrt(float(self.width)),
        )
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        if self.cell_kind == "static_gru":
            assert self.reset_input is not None
            assert self.reset_hidden is not None
            assert self.candidate_input is not None
            assert self.candidate_hidden is not None
            nn.init.xavier_uniform_(self.reset_input.weight)
            nn.init.zeros_(self.reset_input.bias)
            nn.init.orthogonal_(
                self.reset_hidden.weight, gain=self.recurrent_gain
            )
            nn.init.xavier_uniform_(self.candidate_input.weight)
            nn.init.orthogonal_(
                self.candidate_hidden.weight, gain=self.recurrent_gain
            )
            nn.init.zeros_(self.candidate_hidden.bias)
        else:
            assert self.autonomous_hidden is not None
            assert self.input_write is not None
            nn.init.orthogonal_(
                self.autonomous_hidden.weight, gain=self.recurrent_gain
            )
            nn.init.zeros_(self.autonomous_hidden.bias)
            if self.writer_hidden is not None:
                nn.init.orthogonal_(
                    self.writer_hidden.weight, gain=self.recurrent_gain
                )
                nn.init.zeros_(self.writer_hidden.bias)
            nn.init.xavier_uniform_(self.input_write.weight)

    def retention(self) -> torch.Tensor:
        return torch.sigmoid(self.theta)

    def write_gain(self) -> torch.Tensor | None:
        if self.raw_gamma is None:
            return None
        return F.softplus(self.raw_gamma)

    def initialize(self, initial_memory: torch.Tensor) -> torch.Tensor:
        if initial_memory.ndim != 2 or initial_memory.shape[-1] != self.initial_memory_dim:
            raise ValueError(
                f"initial_memory must be [B,{self.initial_memory_dim}], got "
                f"{tuple(initial_memory.shape)}"
            )
        return torch.tanh(self.initial_encoder(initial_memory))

    def step(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        *,
        retention_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must be [B,{self.input_dim}]")
        if state.ndim != 2 or state.shape[-1] != self.width:
            raise ValueError(f"state must be [B,{self.width}]")
        retention = self.retention() if retention_override is None else retention_override
        retention = retention.to(dtype=state.dtype, device=state.device)
        if self.cell_kind == "static_gru":
            assert self.reset_input is not None
            assert self.reset_hidden is not None
            assert self.candidate_input is not None
            assert self.candidate_hidden is not None
            reset = torch.sigmoid(
                self.reset_input(inputs) + self.reset_hidden(state)
            )
            candidate = torch.tanh(
                self.candidate_input(inputs)
                + self.candidate_hidden(reset * state)
            )
            return retention * state + (1.0 - retention) * candidate

        assert self.autonomous_hidden is not None
        assert self.input_write is not None
        gamma = self.write_gain()
        assert gamma is not None
        autonomous_drive = self.autonomous_hidden(state)
        autonomous = torch.tanh(autonomous_drive)
        writer_drive = (
            autonomous_drive
            if self.writer_hidden is None
            else self.writer_hidden(state)
        )
        writer_zero = torch.tanh(writer_drive)
        writer_conditioned = torch.tanh(writer_drive + self.input_write(inputs))
        return (
            retention * state
            + (1.0 - retention) * autonomous
            + gamma * (writer_conditioned - writer_zero)
        )

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(state)

    def forward_sequence(
        self,
        inputs: torch.Tensor,
        *,
        initial_memory: torch.Tensor,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must be [T,B,{self.input_dim}]")
        state = self.initialize(initial_memory)
        predictions: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for inputs_t in inputs:
            state = self.step(inputs_t, state)
            predictions.append(self.decode(state))
            if return_states:
                states.append(state)
        prediction = torch.stack(predictions)
        if return_states:
            return prediction, torch.stack(states)
        return prediction

    def clamp_theta_(self, lower: float = -8.0, upper: float = 8.0) -> None:
        with torch.no_grad():
            self.theta.clamp_(float(lower), float(upper))

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "cell_kind": self.cell_kind,
            "autonomous_writer_field_sharing": self.cell_kind == "untied_rnn",
            "retention_training": "gate_intervention_rp" if self.rp_enabled else "gradient",
            "topology": self.topology,
            "width": self.width,
            "initial_retention": self.initial_retention,
            "initial_write_gain": self.initial_write_gain,
            "recurrent_gain": self.recurrent_gain,
            "input_dim": self.input_dim,
            "initial_memory_dim": self.initial_memory_dim,
            "output_dim": self.output_dim,
            "parameters_total": sum(p.numel() for p in self.parameters()),
            "parameters_gradient_trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "parameters_rp_updated": self.width if self.rp_enabled else 0,
            "initial_memory_source": "true_pre_update_q0_hidden_initialization_only",
        }


__all__ = ["MODEL_IDS", "StaticGateMemory"]
