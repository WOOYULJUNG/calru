"""State-dependent-retention variants on the frozen CA-LRU outer scaffold.

The registered Ságodi/CA-LRU campaigns are intentionally left untouched.  This
module builds their width-52 No-RP scaffold, then replaces only the recurrent
cell with a subclass that shares one state-dependent retention mechanism across
three writer forms:

* ``linear``: ``B u``;
* ``input_nonlinear``: ``psi(u)`` with ``psi(0)=0``;
* ``recurrent``: ``g(h,u)-g(h,0)``.

The state-dependent multiplier is

    lambda_j(h) = base_lambda_j * exp(a * tanh(r_j(h))).

Unlike the historical LRU parameterization, this permits a small local value
above one.  That is necessary (though not sufficient) for a non-zero attracting
ring: a map constrained to ``0 < lambda <= 1`` cannot push an inward radial
perturbation back out.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

import torch
from torch import nn

from repro.sagodi_protocol.artifacts import derived_seed
from repro.sagodi_protocol.primary_v4 import V4Model, build_v4_model

# Importing the protocol model boundary installs the legacy implementation
# directory on sys.path.  Keep the historical module unmodified because it is
# part of completed, fingerprinted campaigns.
from repro.sagodi_protocol import models as _protocol_models  # noqa: F401
from pan_block import PANNonlinearWriterRec  # type: ignore  # noqa: E402


WriterKind = Literal["linear", "input_nonlinear", "recurrent"]
WRITER_KINDS: tuple[WriterKind, ...] = (
    "linear",
    "input_nonlinear",
    "recurrent",
)
RetentionMode = Literal["gradient_only", "hybrid_rp"]
RETENTION_MODES: tuple[RetentionMode, ...] = ("gradient_only", "hybrid_rp")


def _seeded_module(seed: int, factory):
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        return factory()


class StateDependentRetentionRec(PANNonlinearWriterRec):
    """PAN recurrence with a shared state-dependent diagonal multiplier."""

    def __init__(
        self,
        source: PANNonlinearWriterRec,
        *,
        writer_kind: WriterKind,
        retention_mode: RetentionMode,
        gate_hidden: int,
        max_log_modulation: float,
        gate_seed: int,
        writer_seed: int,
        gate_output_weight_std: float = 0.0,
        gate_output_bias: float = 0.0,
    ) -> None:
        if writer_kind not in WRITER_KINDS:
            raise ValueError(f"unknown writer kind: {writer_kind}")
        if retention_mode not in RETENTION_MODES:
            raise ValueError(f"unknown retention mode: {retention_mode}")
        if gate_hidden <= 0:
            raise ValueError("gate_hidden must be positive")
        if not 0.0 < float(max_log_modulation) < math.log(2.0):
            raise ValueError("max_log_modulation must lie in (0, log(2))")
        if not math.isfinite(float(gate_output_weight_std)) or float(
            gate_output_weight_std
        ) < 0.0:
            raise ValueError("gate_output_weight_std must be finite and nonnegative")
        if not math.isfinite(float(gate_output_bias)):
            raise ValueError("gate_output_bias must be finite")

        super().__init__(
            input_dim=int(source.input_dim),
            hidden_dim=int(source.hidden_dim),
            output_dim=int(source.output_dim),
            lambda_min=0.90,
            lambda_max=0.999,
            gamma_free=source.gamma_raw is not None,
            writer_mode="recurrent",
            writer_hidden=int(source.hidden_dim),
        )
        self.writer_kind: WriterKind = writer_kind
        self.retention_mode: RetentionMode = retention_mode
        self.writer_mode = writer_kind
        self.gate_hidden = int(gate_hidden)
        self.max_log_modulation = float(max_log_modulation)
        self.gate_output_weight_std = float(gate_output_weight_std)
        self.gate_output_bias = float(gate_output_bias)

        with torch.no_grad():
            self.theta.copy_(source.theta)
            if self.gamma_raw is not None and source.gamma_raw is not None:
                self.gamma_raw.copy_(source.gamma_raw)
        # In gradient_only both pieces follow task loss.  In hybrid_rp the
        # static spectrum is detached and updated only by the existing damage
        # intervention, while the state-dependent modulation follows task loss.
        self.theta.requires_grad_(retention_mode == "gradient_only")
        self.out_proj.load_state_dict(copy.deepcopy(source.out_proj.state_dict()))

        def make_gate() -> nn.Sequential:
            gate = nn.Sequential(
                nn.Linear(self.hidden_dim, self.gate_hidden, bias=True),
                nn.GELU(),
                nn.Linear(self.gate_hidden, self.hidden_dim, bias=True),
            )
            nn.init.xavier_uniform_(gate[0].weight)
            nn.init.zeros_(gate[0].bias)
            # The registered comparison keeps both values at zero and thus
            # starts exactly at the historical constant-retention model.  The
            # explicit alternatives are used only by a separate initialization
            # sweep and do not change that default contract.
            if self.gate_output_weight_std == 0.0:
                nn.init.zeros_(gate[2].weight)
            else:
                nn.init.normal_(
                    gate[2].weight,
                    mean=0.0,
                    std=self.gate_output_weight_std,
                )
            nn.init.constant_(gate[2].bias, self.gate_output_bias)
            return gate

        self.retention_gate = _seeded_module(gate_seed, make_gate)

        if writer_kind == "recurrent":
            self.writer = copy.deepcopy(source.writer)
        elif writer_kind == "input_nonlinear":
            # B+ is exactly psi(u), without the separate Gamma used by C.
            self.gamma_raw = None

            def make_input_writer() -> nn.Sequential:
                return nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                )

            self.writer = _seeded_module(writer_seed, make_input_writer)
        else:
            # B is exactly B u; the linear map absorbs any static write scale.
            self.gamma_raw = None
            self.writer = _seeded_module(
                writer_seed,
                lambda: nn.Linear(self.input_dim, self.hidden_dim, bias=False),
            )

    def state_dependent_lambda(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.hidden_dim:
            raise ValueError("state width differs from retention width")
        base = self.lam_mag().to(dtype=state.dtype, device=state.device)
        log_modulation = self.max_log_modulation * torch.tanh(
            self.retention_gate(state)
        )
        return base * torch.exp(log_modulation)

    def write(self, u_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.writer_kind in {"linear", "input_nonlinear"}:
            return self.writer(u_t)
        joined = torch.cat([state, u_t], dim=-1)
        zero_joined = torch.cat([state, torch.zeros_like(u_t)], dim=-1)
        return self.writer(joined) - self.writer(zero_joined)

    def step(self, u_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        retention = self.state_dependent_lambda(state)
        written = self.write(u_t, state)
        if self.writer_kind == "recurrent":
            written = self.gamma() * written
        return retention * state + written

    @torch.no_grad()
    def dynamic_retention_summary(self, state: torch.Tensor) -> dict[str, float]:
        values = self.state_dependent_lambda(state)
        return {
            "minimum": float(values.min().cpu()),
            "median": float(values.median().cpu()),
            "mean": float(values.mean().cpu()),
            "maximum": float(values.max().cpu()),
            "fraction_above_one": float((values > 1.0).float().mean().cpu()),
        }


def build_state_dependent_model(
    writer_kind: WriterKind,
    *,
    model_seed: int,
    retention_mode: RetentionMode = "gradient_only",
    gate_hidden: int = 52,
    max_log_modulation: float = 0.05,
    gate_output_weight_std: float = 0.0,
    gate_output_bias: float = 0.0,
) -> V4Model:
    """Build a paired width-52 model whose only mechanism change is writer form."""

    if writer_kind not in WRITER_KINDS:
        raise ValueError(f"unknown writer kind: {writer_kind}")
    if retention_mode not in RETENTION_MODES:
        raise ValueError(f"unknown retention mode: {retention_mode}")
    torch.manual_seed(int(model_seed))
    model = build_v4_model(
        "no_rp_n52" if retention_mode == "gradient_only" else "ca_lru_n52"
    )
    source = model.core.blocks[0].rec
    if not isinstance(source, PANNonlinearWriterRec):
        raise RuntimeError("CA-LRU/No-RP source recurrence changed unexpectedly")
    replacement = StateDependentRetentionRec(
        source,
        writer_kind=writer_kind,
        retention_mode=retention_mode,
        gate_hidden=gate_hidden,
        max_log_modulation=max_log_modulation,
        gate_seed=derived_seed(model_seed, "state_dependent_retention", "gate"),
        writer_seed=derived_seed(
            model_seed, "state_dependent_retention", "writer", writer_kind
        ),
        gate_output_weight_std=gate_output_weight_std,
        gate_output_bias=gate_output_bias,
    )
    model.core.blocks[0].rec = replacement
    return model


def recurrence(model: V4Model) -> StateDependentRetentionRec:
    value = model.core.blocks[0].rec
    if not isinstance(value, StateDependentRetentionRec):
        raise TypeError("model does not contain StateDependentRetentionRec")
    return value
