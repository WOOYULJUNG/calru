"""Zero-retuning topology adapters for the ring-selected recurrent models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from repro.sagodi_protocol.artifacts import derived_seed, strict_json_load
from repro.sagodi_protocol.exact_models import (
    SAGODI_GRU,
    SAGODI_LSTM,
    SAGODI_RNN_TANH,
    build_exact_core,
)
from repro.sagodi_protocol.models import build_model_variant
from repro.state_dependent_retention.models import StateDependentRetentionRec


CONFIG_PATH = Path(__file__).with_name("topology_transfer_v1.json")
MODEL_IDS = ("rnn", "gru", "lstm", "hc")
TOPOLOGY_DIMS = {
    "s1": (1, 2, 2),
    "t2": (2, 4, 4),
    "s2": (3, 3, 3),
}


@dataclass(frozen=True)
class ModelTransferSpec:
    model_id: str
    ring_model_id: str
    width: int
    learning_rate: float
    recurrent_weight_decay: float
    gradient_clip_norm: float | None


def load_transfer_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    payload = strict_json_load(Path(path).expanduser().resolve(strict=True))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("topology transfer config must be a schema-1 object")
    if payload.get("campaign_id") != "manifold_topology_transfer_v1":
        raise ValueError("topology transfer campaign id differs")
    if tuple(payload.get("models", {})) != MODEL_IDS:
        raise ValueError("topology transfer model order differs")
    return payload


def transfer_spec(model_id: str, config: Mapping[str, Any] | None = None) -> ModelTransferSpec:
    if model_id not in MODEL_IDS:
        raise ValueError(f"unknown topology model {model_id!r}")
    payload = load_transfer_config() if config is None else config
    row = payload["models"][model_id]
    return ModelTransferSpec(
        model_id=model_id,
        ring_model_id=str(row["ring_model_id"]),
        width=int(row["width"]),
        learning_rate=float(row["learning_rate"]),
        recurrent_weight_decay=float(row["recurrent_weight_decay"]),
        gradient_clip_norm=(
            None if row["gradient_clip_norm"] is None else float(row["gradient_clip_norm"])
        ),
    )


def topology_dimensions(topology: str) -> tuple[int, int, int]:
    key = str(topology).lower()
    if key not in TOPOLOGY_DIMS:
        raise ValueError(f"unknown topology {topology!r}")
    return TOPOLOGY_DIMS[key]


def _reset_v6_baseline_core(model_id: str, core: nn.Module, width: int) -> None:
    """Reproduce the selected noise-free v6 ring initialization policy."""

    hidden = float(width)
    if model_id == "rnn":
        nn.init.normal_(core.wi, mean=0.0, std=1.0 / math.sqrt(hidden))
        nn.init.normal_(core.wrec, mean=0.0, std=1.5 / math.sqrt(hidden))
        nn.init.normal_(core.wo, mean=0.0, std=1.0 / math.sqrt(hidden))
        nn.init.zeros_(core.brec)
        nn.init.zeros_(core.bo)
    elif model_id == "gru":
        bound = 0.25 / math.sqrt(hidden)
        nn.init.uniform_(core.cell.weight_hh, -bound, bound)
    elif model_id == "lstm":
        bound = 1.0 / math.sqrt(hidden)
        nn.init.uniform_(core.cell.weight_hh, -bound, bound)
    else:  # pragma: no cover - caller invariant
        raise ValueError(model_id)


def _build_baseline_core(
    model_id: str, input_dim: int, output_dim: int, width: int
) -> nn.Module:
    exact_name = {
        "rnn": SAGODI_RNN_TANH,
        "gru": SAGODI_GRU,
        "lstm": SAGODI_LSTM,
    }[model_id]
    core = build_exact_core(exact_name, input_dim, output_dim, width)
    _reset_v6_baseline_core(model_id, core, width)
    return core


def _build_hc_core(
    *,
    input_dim: int,
    output_dim: int,
    width: int,
    model_seed: int,
    config: Mapping[str, Any],
) -> nn.Module:
    row = config["models"]["hc"]
    core = build_model_variant(
        variant="PAN-RNW-full",
        input_dim=input_dim,
        output_dim=output_dim,
        rank=2,
        d_model=width,
        rec_dim=width,
        layers=1,
        dropout=0.0,
        plru_tau=0.001,
        plru_c=50.0,
        pan_lambda_min=0.90,
        pan_lambda_max=0.999,
        rank_matched_lambda_high=0.999,
        rank_matched_lambda_low=0.0,
    )
    source = core.blocks[0].rec
    core.blocks[0].rec = StateDependentRetentionRec(
        source,
        writer_kind=str(row["writer_kind"]),
        retention_mode=str(row["retention_mode"]),
        gate_hidden=int(row["gate_hidden"]),
        max_log_modulation=float(row["max_log_modulation"]),
        gate_seed=derived_seed(model_seed, "topology_hc", "retention_gate"),
        writer_seed=derived_seed(model_seed, "topology_hc", "writer"),
        gate_output_weight_std=float(row["gate_output_weight_std"]),
        gate_output_bias=float(row["gate_output_bias"]),
    )
    return core


class TopologyRecurrentModel(nn.Module):
    """A frozen ring core recipe with topology-specific boundary dimensions."""

    def __init__(
        self,
        *,
        model_id: str,
        topology: str,
        spec: ModelTransferSpec,
        core: nn.Module,
        initial_memory_dim: int,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.topology = topology
        self.ring_model_id = spec.ring_model_id
        self.width = int(spec.width)
        self.input_dim = int(topology_dimensions(topology)[0])
        self.initial_memory_dim = int(initial_memory_dim)
        self.output_dim = int(topology_dimensions(topology)[2])
        self.learning_rate = float(spec.learning_rate)
        self.recurrent_weight_decay = float(spec.recurrent_weight_decay)
        self.gradient_clip_norm = spec.gradient_clip_norm
        self.core = core
        self.reported_state_size = int(core.state_size)
        self.primary_state_size = int(
            getattr(core, "recurrent_state_size", core.state_size)
        )
        if model_id == "lstm":
            self.initial_encoder_h = nn.Linear(initial_memory_dim, self.width, bias=False)
            self.initial_encoder_c = nn.Linear(initial_memory_dim, self.width, bias=False)
            nn.init.normal_(
                self.initial_encoder_h.weight,
                mean=0.0,
                std=1.0 / math.sqrt(float(self.width)),
            )
            nn.init.normal_(
                self.initial_encoder_c.weight,
                mean=0.0,
                std=1.0 / math.sqrt(float(self.width)),
            )
            self.initial_encoder = None
        else:
            self.initial_encoder = nn.Linear(
                initial_memory_dim, self.primary_state_size, bias=False
            )
            nn.init.normal_(
                self.initial_encoder.weight,
                mean=0.0,
                std=1.0 / math.sqrt(float(self.primary_state_size)),
            )
            self.initial_encoder_h = None
            self.initial_encoder_c = None

    @property
    def state_size(self) -> int:
        return self.reported_state_size

    @property
    def is_full_block(self) -> bool:
        return hasattr(self.core, "recurrent_state_size")

    @property
    def rp_enabled(self) -> bool:
        return self.model_id == "hc"

    def primary_from_reported(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., : self.primary_state_size]

    def reported_from_primary(self, primary: torch.Tensor) -> torch.Tensor:
        if self.primary_state_size == self.reported_state_size:
            return primary
        suffix = torch.zeros(
            *primary.shape[:-1],
            self.reported_state_size - self.primary_state_size,
            device=primary.device,
            dtype=primary.dtype,
        )
        return torch.cat((primary, suffix), dim=-1)

    def initialize(self, initial_memory: torch.Tensor) -> torch.Tensor:
        if initial_memory.ndim != 2 or initial_memory.shape[-1] != self.initial_memory_dim:
            raise ValueError(
                f"initial_memory must be [B,{self.initial_memory_dim}], got "
                f"{tuple(initial_memory.shape)}"
            )
        if self.model_id == "lstm":
            assert self.initial_encoder_h is not None and self.initial_encoder_c is not None
            return torch.cat(
                (
                    torch.tanh(self.initial_encoder_h(initial_memory)),
                    torch.tanh(self.initial_encoder_c(initial_memory)),
                ),
                dim=-1,
            )
        assert self.initial_encoder is not None
        primary = self.initial_encoder(initial_memory)
        if self.model_id == "gru":
            primary = torch.tanh(primary)
        return self.reported_from_primary(primary)

    def step(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.core.step(inputs, state)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.core.decode(state)

    def forward_sequence(
        self,
        inputs: torch.Tensor,
        *,
        initial_memory: torch.Tensor,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"inputs must be [T,B,{self.input_dim}], got {tuple(inputs.shape)}"
            )
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

    def pan_recs_with_slices(self) -> Iterable[tuple[nn.Module, slice]]:
        method = getattr(self.core, "pan_recs_with_slices", None)
        return method() if method is not None else ()

    def dynamic_lambda(self, state: torch.Tensor) -> torch.Tensor | None:
        if self.model_id != "hc":
            return None
        recurrence = self.core.blocks[0].rec
        if not isinstance(recurrence, StateDependentRetentionRec):
            raise TypeError("H-C recurrence type changed")
        return recurrence.state_dependent_lambda(self.primary_from_reported(state))

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "ring_model_id": self.ring_model_id,
            "topology": self.topology,
            "width": self.width,
            "input_dim": self.input_dim,
            "initial_memory_dim": self.initial_memory_dim,
            "output_dim": self.output_dim,
            "reported_state_size": self.reported_state_size,
            "primary_state_size": self.primary_state_size,
            "learning_rate": self.learning_rate,
            "recurrent_weight_decay": self.recurrent_weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "parameters_total": sum(parameter.numel() for parameter in self.parameters()),
            "parameters_gradient_trainable": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "parameters_rp_updated": self.width if self.rp_enabled else 0,
            "topology_specific_retuning": False,
            "initial_memory_source": "true_pre_update_q0_hidden_initialization_only",
        }


def build_topology_model(
    model_id: str,
    topology: str,
    *,
    model_seed: int,
    config: Mapping[str, Any] | None = None,
) -> TopologyRecurrentModel:
    payload = load_transfer_config() if config is None else config
    spec = transfer_spec(model_id, payload)
    input_dim, initial_memory_dim, output_dim = topology_dimensions(topology)
    torch.manual_seed(int(model_seed))
    if model_id == "hc":
        core = _build_hc_core(
            input_dim=input_dim,
            output_dim=output_dim,
            width=spec.width,
            model_seed=model_seed,
            config=payload,
        )
    else:
        core = _build_baseline_core(model_id, input_dim, output_dim, spec.width)
    return TopologyRecurrentModel(
        model_id=model_id,
        topology=topology,
        spec=spec,
        core=core,
        initial_memory_dim=initial_memory_dim,
    )


def build_optimizer(
    model: TopologyRecurrentModel,
    config: Mapping[str, Any] | None = None,
) -> torch.optim.Adam:
    payload = load_transfer_config() if config is None else config
    training = payload["training"]
    if model.model_id in {"gru", "lstm"}:
        recurrent_parameters = list(model.core.cell.parameters())
    elif model.model_id == "rnn":
        recurrent_parameters = list(model.core.parameters())
    else:
        recurrent_parameters = []
    if recurrent_parameters:
        recurrent_ids = {id(parameter) for parameter in recurrent_parameters}
        remaining = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in recurrent_ids
        ]
        groups = [
            {
                "params": [p for p in recurrent_parameters if p.requires_grad],
                "weight_decay": model.recurrent_weight_decay,
                "group_name": "recurrent_core",
            },
            {
                "params": remaining,
                "weight_decay": 0.0,
                "group_name": "readout_and_initial_map",
            },
        ]
    else:
        groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "weight_decay": 0.0,
                "group_name": "hc_gradient_trainable",
            }
        ]
    return torch.optim.Adam(
        groups,
        lr=model.learning_rate,
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["epsilon"]),
    )


def clip_gradients(model: TopologyRecurrentModel) -> float | None:
    if model.gradient_clip_norm is None:
        return None
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=float(model.gradient_clip_norm)
    )
    return float(norm.detach().cpu())


__all__ = [
    "CONFIG_PATH",
    "MODEL_IDS",
    "TOPOLOGY_DIMS",
    "ModelTransferSpec",
    "TopologyRecurrentModel",
    "build_optimizer",
    "build_topology_model",
    "clip_gradients",
    "load_transfer_config",
    "topology_dimensions",
    "transfer_spec",
]
