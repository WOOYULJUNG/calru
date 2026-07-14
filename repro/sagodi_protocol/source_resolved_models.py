"""Deterministic model adapters for the pinned Ságodi public-code paths."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

from .exact_models import (
    SAGODI_GRU,
    SAGODI_LSTM,
    SAGODI_RNN_TANH,
    build_exact_core,
)
from .source_resolved_protocol import (
    SOURCE_MODEL_IDS,
    SourceTrainingRecipe,
    source_recipe,
)


class SourceResolvedBaseline(nn.Module):
    """One baseline with the released initialization, noise, and init path.

    Inputs and targets use the repository-wide ``[time,batch,feature]``
    convention.  ``source_targets[0]`` is the released code's post-update
    target ``q1`` and is used to construct the recurrent initial state.

    The released GRU/LSTM maps are uninitialized ``torch.Tensor`` objects.
    Here they are explicitly initialized from ``N(0, 1/sqrt(H))``.  The LSTM
    cell state uses its own map, repairing the training-forward typo so that
    training and the released sequence-analysis path share one transition.
    """

    def __init__(self, recipe: SourceTrainingRecipe) -> None:
        super().__init__()
        if recipe.model_id not in SOURCE_MODEL_IDS:
            raise ValueError(f"unregistered source-resolved model: {recipe.model_id}")
        self.recipe = recipe
        self.model_id = recipe.model_id
        self.width = int(recipe.width)
        self.input_dim = 1
        self.output_dim = 2

        if self.model_id == "sagodi_rnn_tanh_n128":
            exact_name = SAGODI_RNN_TANH
        elif self.model_id == "sagodi_gru_n128":
            exact_name = SAGODI_GRU
        else:
            exact_name = SAGODI_LSTM
        self.core = build_exact_core(
            exact_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden=self.width,
        )
        self._reset_source_recurrent_weights()

        # ``models.RNN`` always registers h0, then freezes it when
        # map_output_to_hidden=True.  It is inert in this experiment but kept
        # in the state dict for source fidelity; paper parameter matching uses
        # trainable parameters and therefore excludes these H frozen scalars.
        if self.model_id == "sagodi_rnn_tanh_n128":
            self.source_h0 = nn.Parameter(
                torch.empty(self.width), requires_grad=False
            )
            nn.init.uniform_(self.source_h0, -1.0, 1.0)
        else:
            self.register_parameter("source_h0", None)

        # Orientation follows the upstream expression target[:,0,:] @ Wotr.
        self.output_to_hidden = nn.Parameter(torch.empty(self.output_dim, self.width))
        nn.init.normal_(
            self.output_to_hidden,
            mean=0.0,
            std=1.0 / math.sqrt(float(self.width)),
        )
        if self.model_id == "sagodi_lstm_n64":
            self.output_to_cell = nn.Parameter(torch.empty(self.output_dim, self.width))
            nn.init.normal_(
                self.output_to_cell,
                mean=0.0,
                std=1.0 / math.sqrt(float(self.width)),
            )
        else:
            self.register_parameter("output_to_cell", None)
        self.output_dropout = nn.Dropout(p=float(recipe.output_dropout))

        actual = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        if actual != int(recipe.parameter_count):
            raise RuntimeError(
                f"{self.model_id} parameter-count mismatch: "
                f"{actual} != {recipe.parameter_count}"
            )

    @property
    def state_size(self) -> int:
        return int(self.core.state_size)

    @property
    def primary_state_size(self) -> int:
        return self.state_size

    @property
    def reported_state_size(self) -> int:
        return self.state_size

    def _reset_source_recurrent_weights(self) -> None:
        if self.model_id == "sagodi_rnn_tanh_n128":
            # Released models.RNN defaults used by the gain-initialized path.
            nn.init.normal_(
                self.core.wi,
                mean=0.0,
                std=1.0 / math.sqrt(float(self.width)),
            )
            nn.init.normal_(
                self.core.wrec,
                mean=0.0,
                std=1.5 / math.sqrt(float(self.width)),
            )
            nn.init.normal_(
                self.core.wo,
                mean=0.0,
                std=1.0 / math.sqrt(float(self.width)),
            )
            nn.init.uniform_(
                self.core.brec,
                -math.sqrt(float(self.width)),
                math.sqrt(float(self.width)),
            )
            nn.init.zeros_(self.core.bo)
        elif self.model_id == "sagodi_gru_n128":
            bound = 0.25 / math.sqrt(float(self.width))
            nn.init.uniform_(self.core.cell.weight_hh, -bound, bound)
        elif self.model_id == "sagodi_lstm_n64":
            bound = 1.0 / math.sqrt(float(self.width))
            nn.init.uniform_(self.core.cell.weight_hh, -bound, bound)

    def recurrent_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters receiving the source recipe's recurrent weight decay."""

        if self.model_id == "sagodi_rnn_tanh_n128":
            # Upstream RNN applies one global (zero-decay) Adam group.  Keeping
            # the core separate still makes the optimizer partition auditable.
            return self.core.parameters()
        return self.core.cell.parameters()

    def initial_state(self, source_targets: torch.Tensor) -> torch.Tensor:
        if source_targets.ndim != 3:
            raise ValueError("source_targets must be time-major [time,batch,output]")
        if source_targets.shape[0] < 1 or source_targets.shape[-1] != self.output_dim:
            raise ValueError("source_targets must contain the two-dimensional q1 target")
        source_q1 = source_targets[0]
        hidden = source_q1.matmul(self.output_to_hidden)
        if self.model_id in {"sagodi_gru_n128", "sagodi_lstm_n64"}:
            hidden = torch.tanh(hidden)
        if self.model_id != "sagodi_lstm_n64":
            return hidden
        if self.output_to_cell is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("LSTM output-to-cell map is missing")
        cell = torch.tanh(source_q1.matmul(self.output_to_cell))
        return torch.cat((hidden, cell), dim=-1)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        """Decode with output dropout in the same position as public GRU/LSTM."""

        if self.model_id == "sagodi_rnn_tanh_n128":
            return self.core.decode(state)
        if self.model_id == "sagodi_gru_n128":
            return self.core.readout(self.output_dropout(state))
        hidden, _ = self.core.split_state(state)
        return self.core.readout(self.output_dropout(hidden))

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
        *,
        state_noise_generator: torch.Generator | None = None,
        state_noise_std_override: float | None = None,
    ) -> torch.Tensor:
        next_state = self.core.step(x_t, state)
        standard_deviation = (
            float(self.recipe.effective_state_noise_std)
            if state_noise_std_override is None
            else float(state_noise_std_override)
        )
        if not math.isfinite(standard_deviation) or standard_deviation < 0.0:
            raise ValueError("state noise standard deviation must be finite and non-negative")
        if standard_deviation == 0.0:
            return next_state
        noise = torch.randn(
            next_state.shape,
            dtype=next_state.dtype,
            device=next_state.device,
            generator=state_noise_generator,
        )
        return next_state + standard_deviation * noise

    def forward_sequence(
        self,
        inputs: torch.Tensor,
        *,
        source_targets: torch.Tensor,
        state_noise_generator: torch.Generator | None = None,
        state_noise_std_override: float | None = None,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError("inputs must be time-major [time,batch,1]")
        if source_targets.shape[:2] != inputs.shape[:2]:
            raise ValueError("source_targets time/batch dimensions must match inputs")
        state = self.initial_state(source_targets)
        outputs: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for x_t in inputs:
            state = self.step(
                x_t,
                state,
                state_noise_generator=state_noise_generator,
                state_noise_std_override=state_noise_std_override,
            )
            outputs.append(self.decode(state))
            if return_states:
                states.append(state)
        prediction = torch.stack(outputs)
        if return_states:
            return prediction, torch.stack(states)
        return prediction

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "width": self.width,
            "state_size": self.state_size,
            "parameters_total": sum(parameter.numel() for parameter in self.parameters()),
            "parameters_trainable": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "frozen_source_h0_preserved": self.source_h0 is not None,
            "source_initial_state": "post_update_target_q1",
            "effective_state_noise_std": float(self.recipe.effective_state_noise_std),
            "target_noise_std": float(self.recipe.target_noise_std),
            "output_dropout": float(self.recipe.output_dropout),
            "deterministic_repairs": (
                "initialized_output_to_state_maps",
                "lstm_independent_cell_map" if self.model_id == "sagodi_lstm_n64" else None,
            ),
        }


def build_source_resolved_model(model_id: str) -> SourceResolvedBaseline:
    """Build a registered model from the validated source-resolved recipe."""

    return SourceResolvedBaseline(source_recipe(model_id))
