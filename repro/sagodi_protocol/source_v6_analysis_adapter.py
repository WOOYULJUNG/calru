"""Adapters that expose v6/source checkpoints to the Ságodi primary runner."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable

import torch
from torch import nn

from . import source_repaired_baselines_v6 as baseline_v6
from .primary_v4 import V4Model, build_v4_model


SOURCE_MODEL_IDS = (
    "sagodi_rnn_tanh_n128",
    "sagodi_gru_n128",
    "sagodi_lstm_n64",
    "lru_n52",
    "no_rp_n52",
    "ca_lru_n52",
)


class SourceV6AnalysisModel(nn.Module):
    """Present the legacy ProtocolModel surface without altering dynamics."""

    def __init__(self, inner: nn.Module, model_id: str) -> None:
        super().__init__()
        self.inner = inner
        self.config = SimpleNamespace(name=str(model_id), init_mode="hidden_init")
        self.input_dim = 1
        self.output_dim = 2

    @property
    def core(self) -> nn.Module:
        return self.inner.core

    def initial_state(
        self,
        batch: int,
        device: torch.device | str,
        initial_memory: torch.Tensor | None,
    ) -> torch.Tensor:
        if initial_memory is None or tuple(initial_memory.shape) != (int(batch), 2):
            raise ValueError("source-v6 analysis requires q1 [cos,sin] initial memory")
        memory = initial_memory.to(device=device)
        if isinstance(self.inner, V4Model):
            return self.inner.initial_state(int(batch), device, memory)
        # SourceResolvedBaseline.initial_state consumes only targets[0].
        return self.inner.initial_state(memory.unsqueeze(0))

    def step(self, token: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if isinstance(self.inner, V4Model):
            return self.inner.step(token, state, state_noise=None)
        return self.inner.step(
            token,
            state,
            state_noise_generator=None,
            state_noise_std_override=0.0,
        )

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.inner.decode(state)

    def retention_values(self) -> torch.Tensor:
        values: list[torch.Tensor] = []
        for recurrence, _ in self.pan_recs_with_slices():
            if hasattr(recurrence, "retention"):
                values.append(recurrence.retention().reshape(-1))
            elif hasattr(recurrence, "theta"):
                values.append(torch.sqrt(torch.sigmoid(recurrence.theta)).reshape(-1))
        if values:
            return torch.cat(values)
        reference = next(self.parameters())
        return torch.empty(0, device=reference.device, dtype=reference.dtype)

    def pan_recs_with_slices(self) -> Iterable[tuple[nn.Module, slice]]:
        method = getattr(self.inner, "pan_recs_with_slices", None)
        return method() if method is not None else ()

    def _reported_from_noisy_full_block(
        self, x_t: torch.Tensor, primary: torch.Tensor
    ) -> torch.Tensor:
        method = getattr(self.inner, "_reported_from_noisy_full_block", None)
        if method is None:
            raise AttributeError("model has no overwritten full-block stream")
        return method(x_t, primary)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "analysis_adapter": "source_v6_dynamics_preserving",
            "model_id": self.config.name,
            "inner": self.inner.metadata(),
        }


def load_source_v6_checkpoint(
    path: str | torch.serialization.FILE_LIKE,
    device: torch.device | str,
) -> tuple[SourceV6AnalysisModel, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    run = payload.get("run")
    result = payload.get("result")
    if not isinstance(run, dict) or not isinstance(result, dict):
        raise ValueError("source-v6 checkpoint lacks bound run/result")
    model_id = str(run.get("model_id"))
    if model_id not in SOURCE_MODEL_IDS:
        raise ValueError(f"unsupported source-v6 analysis model: {model_id}")
    learning_rate = float(run["learning_rate"])
    state_noise_std = float(run.get("actual_state_noise_std", 0.0))
    if model_id in baseline_v6.MODEL_IDS:
        inner = baseline_v6.build_model(
            baseline_v6.load_config(), model_id, learning_rate, state_noise_std
        )
    else:
        inner = build_v4_model(model_id)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("source-v6 checkpoint lacks a state dict")
    inner.load_state_dict(state_dict, strict=True)
    model = SourceV6AnalysisModel(inner, model_id).to(device)
    return model, payload
