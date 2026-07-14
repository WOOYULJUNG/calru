"""Protocol-facing model registry.

The historical experiment files expose useful recurrent implementations but
mix model construction with task generation and evaluation.  This module is
the only compatibility boundary used by the Ságodi protocol.  In particular,
it makes the learned hidden initializer and the minimal recurrent carrier
explicit so they are counted, checkpointed, and audited.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F


_LEGACY = Path(__file__).resolve().parents[1] / "legacy_code"
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))

from exp71_pan_block_pulse_hold import build_model_variant  # noqa: E402


CUSTOM_GRU = "gru"
SAGODI_GRU_WIDTH96 = "gru_sagodi_width96"
SAGODI_GRU_PARAM135 = "gru_sagodi_param135"
SAGODI_GRU_NAMES = (SAGODI_GRU_WIDTH96, SAGODI_GRU_PARAM135)
SAGODI_GRU_WIDTHS = {
    SAGODI_GRU_WIDTH96: 96,
    SAGODI_GRU_PARAM135: 135,
}
MODEL_NAMES = ("ca_lru", "no_rp", CUSTOM_GRU, *SAGODI_GRU_NAMES)
INITIAL_ENCODER_PYTORCH_DEFAULT = "pytorch_default"
INITIAL_ENCODER_SAGODI_W_OTR = "sagodi_W_otr_normal_primary_inverse_sqrt"
INITIAL_ENCODER_IDENTITY = "identity"
INITIAL_ENCODER_TANH = "tanh"
SAGODI_CODE_COMMIT = "cbd7404e9baca4b2dc291560cfc6576bb7b1f078"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    input_dim: int
    output_dim: int
    initial_memory_dim: int = 2
    init_mode: str = "hidden_init"
    width: int = 96
    rank: int = 2
    layers: int = 1
    dropout: float = 0.0
    pan_lambda_min: float = 0.90
    pan_lambda_max: float = 0.999
    plru_tau: float = 0.001
    plru_c: float = 50.0
    rank_matched_lambda_high: float = 0.999
    rank_matched_lambda_low: float = 0.0
    initial_encoder_bias: bool = True
    initial_encoder_weight_init: str = INITIAL_ENCODER_PYTORCH_DEFAULT
    initial_encoder_activation: str = INITIAL_ENCODER_IDENTITY

    def validate(self) -> None:
        if self.name not in MODEL_NAMES:
            raise ValueError(f"unknown protocol model {self.name!r}")
        if self.init_mode not in {"hidden_init", "cue_driven"}:
            raise ValueError(f"unsupported init_mode {self.init_mode!r}")
        for key in ("input_dim", "output_dim", "initial_memory_dim", "width", "rank", "layers"):
            if int(getattr(self, key)) <= 0:
                raise ValueError(f"{key} must be positive")
        frozen = {
            "initial_memory_dim": (int(self.initial_memory_dim), 2),
            "rank": (int(self.rank), 2),
            "layers": (int(self.layers), 1),
            "dropout": (float(self.dropout), 0.0),
            "pan_lambda_min": (float(self.pan_lambda_min), 0.90),
            "pan_lambda_max": (float(self.pan_lambda_max), 0.999),
            "plru_tau": (float(self.plru_tau), 0.001),
            "plru_c": (float(self.plru_c), 50.0),
            "rank_matched_lambda_high": (float(self.rank_matched_lambda_high), 0.999),
            "rank_matched_lambda_low": (float(self.rank_matched_lambda_low), 0.0),
        }
        for key, (actual, expected) in frozen.items():
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"pilot architecture freezes {key}={expected}, got {actual}")
        if self.init_mode != "hidden_init":
            raise ValueError("the Phase-1 pilot freezes hidden_init")
        if self.initial_encoder_activation not in {
            INITIAL_ENCODER_IDENTITY,
            INITIAL_ENCODER_TANH,
        }:
            raise ValueError("unsupported initial-state encoder activation")
        if self.initial_encoder_weight_init not in {
            INITIAL_ENCODER_PYTORCH_DEFAULT,
            INITIAL_ENCODER_SAGODI_W_OTR,
        }:
            raise ValueError(
                "unsupported initial-state encoder initialization "
                f"{self.initial_encoder_weight_init!r}"
            )
        if (
            self.initial_encoder_weight_init == INITIAL_ENCODER_PYTORCH_DEFAULT
            and self.initial_encoder_bias is not True
        ):
            raise ValueError("the legacy Phase-1 freeze uses a biased initial-state encoder")
        if (
            self.initial_encoder_weight_init == INITIAL_ENCODER_SAGODI_W_OTR
            and self.initial_encoder_bias is not False
        ):
            raise ValueError("Ságodi W_otr initialization requires bias=false")
        if self.name in SAGODI_GRU_NAMES:
            expected_width = SAGODI_GRU_WIDTHS[self.name]
            if int(self.width) != expected_width:
                raise ValueError(
                    f"{self.name} freezes hidden width {expected_width}, got {self.width}"
                )
            if self.initial_encoder_weight_init != INITIAL_ENCODER_SAGODI_W_OTR:
                raise ValueError("Ságodi GRU requires the explicit W_otr initialization repair")
            if self.initial_encoder_activation != INITIAL_ENCODER_TANH:
                raise ValueError("Ságodi GRU requires tanh(W_otr y0) initialization")
        elif self.initial_encoder_activation != INITIAL_ENCODER_IDENTITY:
            raise ValueError("legacy v1 models retain identity initial-state activation")


def _legacy_variant(name: str) -> str:
    if name in {"ca_lru", "no_rp"}:
        return "PAN-RNW-full"
    if name == "gru":
        return "GRU"
    raise ValueError(name)


class SagodiGRUBaseline(nn.Module):
    """One-layer official-style GRU with direct linear readout.

    The pinned ``gru.py`` uses ``nn.GRU`` and a direct ``nn.Linear`` head.  A
    single ``GRUCell`` exposes the same one-step recurrence needed by the
    protocol analysis.  We preserve PyTorch's two random bias vectors (the
    official code does not zero the update gate) and explicitly repeat the
    official recurrent-weight uniform initialization.  The pinned GRU forgot
    to initialize ``output_to_hidden``; that deterministic repair is owned by
    :class:`ProtocolModel` rather than silently reproducing uninitialized
    memory.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden: int):
        super().__init__()
        # StateAdapter audits the literal blank-input map through this public
        # step interface, so the input dimension must be explicit rather than
        # hidden inside GRUCell.
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.cell = nn.GRUCell(self.input_dim, self.hidden, bias=True)
        bound = 1.0 / math.sqrt(float(self.hidden))
        with torch.no_grad():
            nn.init.uniform_(self.cell.weight_hh, -bound, bound)
        self.readout = nn.Linear(self.hidden, int(output_dim), bias=True)

    @property
    def state_size(self) -> int:
        return self.hidden

    def init_state(self, batch: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(int(batch), self.hidden, device=device)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.cell(x_t, state)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.readout(state)


def _primary_size(core: nn.Module) -> int:
    # FullBlockSequenceModel stores a decoder stream after recurrent states.
    # With carry_stream=False that stream is overwritten and has no causal
    # influence on the next transition, so it is not part of the Markov state.
    if hasattr(core, "recurrent_state_size") and not bool(getattr(core, "carry_stream", False)):
        return int(core.recurrent_state_size)
    return int(core.state_size)


class ProtocolModel(nn.Module):
    """A recurrent core plus a protocol-controlled initial-state encoder."""

    def __init__(self, config: ModelConfig, core: nn.Module):
        super().__init__()
        config.validate()
        self.config = config
        self.core = core
        self.input_dim = int(config.input_dim)
        self.output_dim = int(config.output_dim)
        self.primary_state_size = _primary_size(core)
        self.reported_state_size = int(core.state_size)
        if config.init_mode == "hidden_init":
            self.initial_encoder: nn.Module | None = nn.Linear(
                int(config.initial_memory_dim),
                self.primary_state_size,
                bias=bool(config.initial_encoder_bias),
            )
            if config.initial_encoder_weight_init == INITIAL_ENCODER_SAGODI_W_OTR:
                nn.init.normal_(
                    self.initial_encoder.weight,
                    mean=0.0,
                    std=1.0 / math.sqrt(float(self.primary_state_size)),
                )
        else:
            self.initial_encoder = None

    @property
    def state_size(self) -> int:
        return self.reported_state_size

    @property
    def is_full_block(self) -> bool:
        return hasattr(self.core, "recurrent_state_size")

    @property
    def rp_enabled(self) -> bool:
        return self.config.name == "ca_lru"

    def primary_from_reported(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., : self.primary_state_size]

    def reported_from_primary(self, primary: torch.Tensor) -> torch.Tensor:
        if self.reported_state_size == self.primary_state_size:
            return primary
        suffix = torch.zeros(
            *primary.shape[:-1],
            self.reported_state_size - self.primary_state_size,
            device=primary.device,
            dtype=primary.dtype,
        )
        return torch.cat([primary, suffix], dim=-1)

    def replace_primary(self, state: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        if state.shape[:-1] != primary.shape[:-1]:
            raise ValueError("state and primary batch dimensions differ")
        if self.reported_state_size == self.primary_state_size:
            return primary
        return torch.cat([primary, state[..., self.primary_state_size :]], dim=-1)

    def initial_state(
        self,
        batch: int,
        device: torch.device | str,
        initial_memory: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.initial_encoder is None:
            return self.core.init_state(int(batch), device)
        if initial_memory is None:
            raise ValueError("hidden_init requires initial_memory")
        if initial_memory.shape != (int(batch), int(self.config.initial_memory_dim)):
            raise ValueError(
                f"initial_memory has shape {tuple(initial_memory.shape)}, expected "
                f"({batch}, {self.config.initial_memory_dim})"
            )
        primary = self.initial_encoder(initial_memory)
        if self.config.initial_encoder_activation == INITIAL_ENCODER_TANH:
            primary = torch.tanh(primary)
        return self.reported_from_primary(primary)

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
        *,
        state_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        next_state = self.core.step(x_t, state)
        # Protocol-A noise is added to the newly computed causal recurrent
        # state.  For the full-block scaffold the decoder stream is not causal,
        # but it must be recomputed from the perturbed carrier so the current
        # output and the state handed to the next step are consistent.
        if state_noise is not None:
            primary = self.primary_from_reported(next_state)
            if state_noise.shape != primary.shape:
                raise ValueError(
                    f"state_noise shape {tuple(state_noise.shape)} != primary {tuple(primary.shape)}"
                )
            primary = primary + state_noise
            if self.is_full_block and not bool(getattr(self.core, "carry_stream", False)):
                next_state = self._reported_from_noisy_full_block(x_t, primary)
            else:
                next_state = self.replace_primary(next_state, primary)
        return next_state

    def _reported_from_noisy_full_block(
        self, x_t: torch.Tensor, primary: torch.Tensor
    ) -> torch.Tensor:
        """Recompute the non-causal block stream after recurrent-state noise."""

        rec_states = [primary[:, state_slice] for state_slice in self.core._rec_slices]
        stream = self.core.encoder(x_t)
        for block, rec_state in zip(self.core.blocks, rec_states):
            rec_out = block.rec.output(rec_state)
            if block.update_mode == "glu":
                update = F.glu(block.glu_proj(F.gelu(rec_out)), dim=-1)
            elif block.update_mode == "gelu":
                update = F.gelu(rec_out)
            elif block.update_mode == "linear":
                update = rec_out
            else:  # pragma: no cover - guarded by the legacy constructor
                raise ValueError(f"unknown update_mode: {block.update_mode}")
            stream_base = stream + block.dropout(update) if block.use_residual else block.dropout(update)
            stream = block.norm_out(stream_base)
        return self.core.merge_state(rec_states, stream)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.core.decode(state)

    def forward_sequence(
        self,
        inputs: torch.Tensor,
        *,
        initial_memory: torch.Tensor | None = None,
        state_noise_std: float = 0.0,
        noise_generator: torch.Generator | None = None,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError("inputs must be time x batch x feature")
        state = self.initial_state(inputs.shape[1], inputs.device, initial_memory)
        outputs: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for x_t in inputs:
            noise = None
            if float(state_noise_std) > 0:
                noise = torch.randn(
                    self.primary_from_reported(state).shape,
                    device=state.device,
                    dtype=state.dtype,
                    generator=noise_generator,
                ) * float(state_noise_std)
            state = self.step(x_t, state, state_noise=noise)
            outputs.append(self.decode(state))
            if return_states:
                states.append(state)
        output = torch.stack(outputs, dim=0)
        if return_states:
            return output, torch.stack(states, dim=0)
        return output

    def pan_recs_with_slices(self) -> Iterable[tuple[nn.Module, slice]]:
        if hasattr(self.core, "pan_recs_with_slices"):
            return self.core.pan_recs_with_slices()
        return ()

    def retention_values(self) -> torch.Tensor:
        if hasattr(self.core, "lam_mag"):
            return self.core.lam_mag()
        return torch.empty(0, device=next(self.parameters()).device)

    def metadata(self) -> dict[str, Any]:
        if self.config.name in {"ca_lru", "no_rp"}:
            autonomous_primary_map = "homogeneous_diagonal_linear_F0_h_equals_Lambda_h"
            input_conditioned_writer = "nonlinear_recurrent_writer_g_h_u_minus_g_h_zero"
        else:
            autonomous_primary_map = "nonlinear_GRU_blank_hidden_map"
            input_conditioned_writer = "standard_GRU_input_conditioning"
        return {
            "model_config": asdict(self.config),
            "initial_state_encoder_initialization": {
                "policy": self.config.initial_encoder_weight_init,
                "bias": bool(self.config.initial_encoder_bias),
                "weight_distribution": (
                    "Normal(0, 1/sqrt(primary_state_dimension))"
                    if self.config.initial_encoder_weight_init
                    == INITIAL_ENCODER_SAGODI_W_OTR
                    else "torch.nn.Linear.reset_parameters"
                ),
                "activation": self.config.initial_encoder_activation,
                "primary_state_dimension": self.primary_state_size,
            },
            "legacy_variant": (
                _legacy_variant(self.config.name)
                if self.config.name not in SAGODI_GRU_NAMES
                else None
            ),
            "autonomous_primary_map": autonomous_primary_map,
            "input_conditioned_writer": input_conditioned_writer,
            "primary_state_size": self.primary_state_size,
            "reported_state_size": self.reported_state_size,
            "parameters_total": sum(p.numel() for p in self.parameters()),
            "parameters_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "rp_enabled": self.rp_enabled,
            "sagodi_gru_contract": (
                {
                    "official_source_commit": SAGODI_CODE_COMMIT,
                    "recurrence": "one_layer_torch_GRUCell_step_equivalent_to_nn_GRU",
                    "bias_convention": "two_PyTorch_default_random_bias_vectors",
                    "initial_state": "tanh(bias_free_W_otr_y0)",
                    "readout": "direct_biased_linear",
                    "output_to_hidden_initialization_repair": (
                        "Normal(0,1/sqrt(hidden)); pinned gru.py left W_otr uninitialized"
                    ),
                    "matching_role": (
                        "width_matched_to_CA_LRU_96"
                        if self.config.name == SAGODI_GRU_WIDTH96
                        else "nearest_parameter_match_to_CA_LRU_56834"
                    ),
                }
                if self.config.name in SAGODI_GRU_NAMES
                else None
            ),
        }


def model_config_from_protocol(
    protocol: Mapping[str, Any],
    model_name: str,
    *,
    width_override: int | None = None,
) -> ModelConfig:
    """Construct a model config from the frozen architecture block.

    This keeps the executable builder parameters tied to the checked protocol
    instead of silently relying on Python defaults.  The resolved scaffold is
    verified again after legacy construction by :func:`build_protocol_model`.
    """

    phase = protocol["phase1_ring_pilot"]
    task = phase["task"]
    training = phase["training"]
    architecture = training["architecture"]
    shared = architecture["shared_builder_kwargs"]
    initializer = architecture["initial_state_encoder"]
    weight_initialization = initializer.get("weight_initialization")
    initial_encoder_weight_init = INITIAL_ENCODER_PYTORCH_DEFAULT
    if weight_initialization is not None:
        if weight_initialization != {
            "distribution": "normal",
            "mean": 0.0,
            "standard_deviation": "1_over_sqrt_primary_state_dimension",
            "source": "Sagodi_official_W_otr",
        }:
            raise ValueError("unsupported initial-state encoder initialization block")
        initial_encoder_weight_init = INITIAL_ENCODER_SAGODI_W_OTR
    resolved_width = int(training["width"] if width_override is None else width_override)
    resolved_encoder_bias = bool(initializer["bias"])
    resolved_encoder_weight_init = initial_encoder_weight_init
    resolved_encoder_activation = INITIAL_ENCODER_IDENTITY
    if model_name in SAGODI_GRU_NAMES:
        frozen_width = SAGODI_GRU_WIDTHS[str(model_name)]
        if width_override is not None and int(width_override) != frozen_width:
            raise ValueError(f"{model_name} width override must be {frozen_width}")
        resolved_width = frozen_width
        resolved_encoder_bias = False
        resolved_encoder_weight_init = INITIAL_ENCODER_SAGODI_W_OTR
        resolved_encoder_activation = INITIAL_ENCODER_TANH
    return ModelConfig(
        name=str(model_name),
        input_dim=int(task["input_dimension"]),
        output_dim=int(task["output_dimension"]),
        initial_memory_dim=int(initializer["input_dimension"]),
        init_mode=str(task["initialization_mode"]),
        width=resolved_width,
        rank=int(shared["rank"]),
        layers=int(shared["layers"]),
        dropout=float(shared["dropout"]),
        pan_lambda_min=float(shared["pan_lambda_min"]),
        pan_lambda_max=float(shared["pan_lambda_max"]),
        plru_tau=float(shared["plru_tau"]),
        plru_c=float(shared["plru_c"]),
        rank_matched_lambda_high=float(shared["rank_matched_lambda_high"]),
        rank_matched_lambda_low=float(shared["rank_matched_lambda_low"]),
        initial_encoder_bias=resolved_encoder_bias,
        initial_encoder_weight_init=resolved_encoder_weight_init,
        initial_encoder_activation=resolved_encoder_activation,
    )


def _verify_resolved_architecture(model: ProtocolModel) -> None:
    """Fail closed if the compatibility builder resolves a different model."""

    config = model.config
    if model.initial_encoder is None or not isinstance(model.initial_encoder, nn.Linear):
        raise RuntimeError("hidden-init model must expose a linear initial-state encoder")
    if config.initial_encoder_bias and model.initial_encoder.bias is None:
        raise RuntimeError("initial-state encoder must include its frozen bias")
    if not config.initial_encoder_bias and model.initial_encoder.bias is not None:
        raise RuntimeError("initial-state encoder must be bias-free")
    core = model.core
    if config.name in {"ca_lru", "no_rp"}:
        if core.__class__.__name__ != "FullBlockSequenceModel":
            raise RuntimeError("CA-LRU must resolve to FullBlockSequenceModel")
        expected_attributes = {
            "variant": "PAN-RNW-Block",
            "d_model": int(config.width),
            "rec_dim": int(config.width),
            "num_layers": int(config.layers),
            "decode_mode": "stream",
            "carry_stream": False,
        }
        for key, expected in expected_attributes.items():
            if getattr(core, key, None) != expected:
                raise RuntimeError(
                    f"resolved CA-LRU scaffold has {key}={getattr(core, key, None)!r}, "
                    f"expected {expected!r}"
                )
        if core.encoder.bias is not None:
            raise RuntimeError("CA-LRU scaffold encoder must be bias-free")
        if len(core.blocks) != 1:
            raise RuntimeError("CA-LRU pilot freezes exactly one recurrent block")
        block = core.blocks[0]
        if not isinstance(block.norm_in, nn.Identity):
            raise RuntimeError("CA-LRU scaffold freezes use_norm_in=False")
        if not isinstance(block.norm_out, nn.LayerNorm) or not block.norm_out.elementwise_affine:
            raise RuntimeError("CA-LRU scaffold requires affine output LayerNorm")
        if block.update_mode != "glu" or block.glu_proj is None:
            raise RuntimeError("CA-LRU scaffold requires the GLU update")
        if not block.use_residual or not math.isclose(float(block.dropout.p), 0.0):
            raise RuntimeError("CA-LRU scaffold requires residual=True and dropout=0")
        recurrence = block.rec
        if recurrence.__class__.__name__ != "PANNonlinearWriterRec":
            raise RuntimeError("CA-LRU recurrence must be PANNonlinearWriterRec")
        if recurrence.writer_mode != "recurrent":
            raise RuntimeError("CA-LRU freezes the recurrent input-conditioned writer")
        if (
            len(recurrence.writer) != 3
            or not isinstance(recurrence.writer[0], nn.Linear)
            or not isinstance(recurrence.writer[1], nn.GELU)
            or not isinstance(recurrence.writer[2], nn.Linear)
            or recurrence.writer[0].bias is None
            or recurrence.writer[2].bias is None
        ):
            raise RuntimeError("CA-LRU writer must be a biased Linear-GELU-Linear map")
        if recurrence.theta.requires_grad:
            raise RuntimeError("retention theta must remain outside task-loss autograd")
        if recurrence.gamma_raw is None:
            raise RuntimeError("CA-LRU freezes gamma_free=True")
        if not torch.equal(recurrence.gamma_raw.detach(), torch.zeros_like(recurrence.gamma_raw)):
            raise RuntimeError("CA-LRU gamma_raw must initialize at zero")
        expected_retention = torch.linspace(
            float(config.pan_lambda_max),
            float(config.pan_lambda_min),
            int(config.width),
            device=recurrence.theta.device,
            dtype=recurrence.theta.dtype,
        )
        if not torch.allclose(
            recurrence.lam_mag().detach(), expected_retention, rtol=1e-6, atol=1e-6
        ):
            raise RuntimeError("CA-LRU retention spectrum must be linear from max to min")
        if recurrence.out_proj.bias is None or block.glu_proj.bias is None:
            raise RuntimeError("CA-LRU recurrence and GLU projections must include biases")
        if (
            len(core.head) != 2
            or not isinstance(core.head[0], nn.LayerNorm)
            or not core.head[0].elementwise_affine
            or not isinstance(core.head[1], nn.Linear)
            or core.head[1].bias is None
        ):
            raise RuntimeError("CA-LRU head must be affine LayerNorm followed by biased Linear")
    elif config.name == CUSTOM_GRU:
        if core.__class__.__name__ != "GRUBaseline":
            raise RuntimeError("GRU pilot baseline must resolve to GRUBaseline")
        if not isinstance(core.cell, nn.GRUCell):
            raise RuntimeError("GRU pilot baseline must use torch.nn.GRUCell")
        if int(core.state_size) != int(config.width) or int(core.cell.hidden_size) != int(
            config.width
        ):
            raise RuntimeError("GRU hidden dimension must equal the frozen width")
        if core.cell.bias_ih is None or core.cell.bias_hh is None:
            raise RuntimeError("GRUCell must retain both bias vectors")
        update = slice(int(config.width), 2 * int(config.width))
        if not torch.equal(
            core.cell.bias_ih[update].detach(), torch.zeros_like(core.cell.bias_ih[update])
        ):
            raise RuntimeError("GRU update input bias must initialize at zero")
        if not torch.equal(
            core.cell.bias_hh[update].detach(), torch.zeros_like(core.cell.bias_hh[update])
        ):
            raise RuntimeError("GRU update hidden bias must initialize at zero")
        readout = core.readout.net
        if (
            len(readout) != 3
            or not isinstance(readout[0], nn.Linear)
            or readout[0].out_features != 64
            or readout[0].bias is None
            or not isinstance(readout[1], nn.Tanh)
            or not isinstance(readout[2], nn.Linear)
            or readout[2].bias is None
        ):
            raise RuntimeError("GRU readout must be the frozen biased 64-unit Tanh MLP")
    else:
        if config.name not in SAGODI_GRU_NAMES:
            raise RuntimeError(f"unverified model architecture {config.name!r}")
        if not isinstance(core, SagodiGRUBaseline):
            raise RuntimeError("Ságodi GRU must resolve to SagodiGRUBaseline")
        if not isinstance(core.cell, nn.GRUCell):
            raise RuntimeError("Ságodi GRU recurrence must use torch.nn.GRUCell")
        if core.cell.bias_ih is None or core.cell.bias_hh is None:
            raise RuntimeError("Ságodi GRU must retain both official bias vectors")
        if not isinstance(core.readout, nn.Linear) or core.readout.bias is None:
            raise RuntimeError("Ságodi GRU must use a direct biased linear readout")
        if core.readout.in_features != int(config.width):
            raise RuntimeError("Ságodi GRU direct readout has the wrong hidden width")
        if config.initial_encoder_activation != INITIAL_ENCODER_TANH:
            raise RuntimeError("Ságodi GRU initial mapping must apply tanh")
        expected_parameters = 3 * int(config.width) ** 2 + 13 * int(config.width) + 2
        actual_parameters = sum(parameter.numel() for parameter in model.parameters())
        if actual_parameters != expected_parameters:
            raise RuntimeError(
                "Ságodi GRU parameter-count contract failed: "
                f"{actual_parameters} != {expected_parameters}"
            )


def build_protocol_model(config: ModelConfig) -> ProtocolModel:
    config.validate()
    if config.name in SAGODI_GRU_NAMES:
        core = SagodiGRUBaseline(
            int(config.input_dim),
            int(config.output_dim),
            int(config.width),
        )
    else:
        core = build_model_variant(
            variant=_legacy_variant(config.name),
            input_dim=int(config.input_dim),
            output_dim=int(config.output_dim),
            rank=int(config.rank),
            d_model=int(config.width),
            rec_dim=int(config.width),
            layers=int(config.layers),
            dropout=float(config.dropout),
            plru_tau=float(config.plru_tau),
            plru_c=float(config.plru_c),
            pan_lambda_min=float(config.pan_lambda_min),
            pan_lambda_max=float(config.pan_lambda_max),
            rank_matched_lambda_high=float(config.rank_matched_lambda_high),
            rank_matched_lambda_low=float(config.rank_matched_lambda_low),
        )
    model = ProtocolModel(config, core)
    _verify_resolved_architecture(model)
    return model


def checkpoint_payload(model: ProtocolModel, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": model.metadata(),
        "state_dict": model.state_dict(),
        "extra": dict(extra or {}),
    }


def load_checkpoint(path: Path | str, device: torch.device | str = "cpu") -> tuple[ProtocolModel, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported checkpoint schema")
    config = ModelConfig(**payload["model"]["model_config"])
    model = build_protocol_model(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload
