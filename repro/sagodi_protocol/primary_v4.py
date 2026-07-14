"""Paper-first Ságodi baseline versus parameter-matched CA-LRU campaign.

This module is intentionally independent of the retired v3 selector and
training pipeline.  It owns one immutable N=128 experiment matrix, creates
fixed validation data, records every executable run specification, and can
resume a multi-GPU launch from verified child receipts.

The public entry point has three stages::

    python -m repro.sagodi_protocol.primary_v4 --stage smoke ...
    python -m repro.sagodi_protocol.primary_v4 --stage tune ...
    python -m repro.sagodi_protocol.primary_v4 --stage main ...

``tune`` performs the 100-update paper LR audit, runs every LRU/CA-LRU LR
candidate for 5,000 updates (plus paper-LR baseline sentinels), and then
performs the frozen RP grid.  No-RP is never tuned independently: it inherits
the CA-LRU LR.  Main training is blocked until all learning gates and the RP
selection verify.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .artifacts import (
    atomic_json,
    canonical_hash,
    derived_seed,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from .metrics import masked_mse, task_metrics
from .tasks import angular_integration, load_fixed_bank, save_fixed_bank


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "primary_v4_config.json"
FREEZE_DOCUMENT = MODULE_DIR / "SAGODI_PRIMARY_V4_FREEZE_ko.md"
CAMPAIGN_ID = "sagodi_primary_v4_n128"
MODEL_IDS = (
    "sagodi_rnn_tanh_n128",
    "sagodi_gru_n128",
    "sagodi_lstm_n64",
    "lru_n52",
    "no_rp_n52",
    "ca_lru_n52",
)
TUNED_LR_MODEL_IDS = tuple(model for model in MODEL_IDS if model != "no_rp_n52")
PAPER_BASELINE_IDS = MODEL_IDS[:3]
EXPECTED_WIDTHS = {
    "sagodi_rnn_tanh_n128": 128,
    "sagodi_gru_n128": 128,
    "sagodi_lstm_n64": 64,
    "lru_n52": 52,
    "no_rp_n52": 52,
    "ca_lru_n52": 52,
}
EXPECTED_PARAMETER_COUNTS = {
    "sagodi_rnn_tanh_n128": 17154,
    "sagodi_gru_n128": 50818,
    "sagodi_lstm_n64": 17538,
    "lru_n52": 17058,
    "no_rp_n52": 17058,
    "ca_lru_n52": 17058,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_native(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def load_v4_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and fail closed on the registered full-campaign contract."""

    source = Path(path).expanduser().resolve(strict=True)
    payload = strict_json_load(source)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("primary-v4 config must be a schema-1 JSON object")
    if payload.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("primary-v4 campaign_id differs")
    if (
        payload.get("protocol_revision")
        != "paper_matched_sagodi_n128_parameter_matched_calru_n52_v1"
    ):
        raise ValueError("primary-v4 protocol_revision differs")
    models = payload.get("models")
    if not isinstance(models, list) or [item.get("id") for item in models] != list(MODEL_IDS):
        raise ValueError("primary-v4 model order differs")
    for item in models:
        model_id = str(item["id"])
        if int(item.get("width", -1)) != EXPECTED_WIDTHS[model_id]:
            raise ValueError(f"{model_id} width differs from the freeze")
        if int(item.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNTS[model_id]:
            raise ValueError(f"{model_id} parameter count differs from the freeze")
    task = payload.get("task", {})
    if task != {
        "name": "angular_integration",
        "horizon": 256,
        "dt": 0.1,
        "gp_length_scale": 1.0,
        "gp_std": 1.0,
        "gp_jitter": 1e-6,
        "initialization_mode": "hidden-init",
        "target_indexing": "q_t_plus_1_after_velocity_update",
    }:
        raise ValueError("primary-v4 task contract differs")
    training = payload.get("training", {})
    expected_training = {
        "optimizer": "Adam",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "batch_size": 64,
        "updates": 5000,
        "state_noise_std": 0.1,
        "gradient_clipping": None,
        "online_batches": True,
        "validation_interval": 500,
        "trace_interval": 50,
    }
    if training != expected_training:
        raise ValueError("primary-v4 training contract differs")
    tuning = payload.get("learning_rate_tuning", {})
    if tuning.get("grid") != [0.01, 0.001, 0.0001, 0.00001]:
        raise ValueError("primary-v4 LR grid differs")
    if tuning.get("seeds") != [100, 101, 102, 103, 104]:
        raise ValueError("primary-v4 tuning seeds differ")
    if int(tuning.get("paper_selector_updates", -1)) != 100:
        raise ValueError("paper selector must use exactly 100 updates")
    if int(tuning.get("continued_sentinel_updates", -1)) != 5000:
        raise ValueError("continued sentinel must use exactly 5,000 updates")
    if float(tuning.get("paper_baseline_learning_rate", math.nan)) != 0.01:
        raise ValueError("Ságodi baseline learning rate must remain 1e-2")
    if float(tuning.get("success_mse_threshold", math.nan)) != 0.01:
        raise ValueError("primary-v4 task-success MSE threshold differs")
    if tuning.get("no_rp_inherits_from") != "ca_lru_n52":
        raise ValueError("No-RP must inherit the CA-LRU learning rate")
    if payload.get("main", {}).get("seeds") != list(range(10)):
        raise ValueError("main seeds must be exactly 0..9")
    rp = payload.get("retention_plasticity", {})
    required_rp = {
        "warmup_updates": 1500,
        "interval_updates": 50,
        "probe_batch_size": 96,
        "probe_horizon": 256,
        "blank_ablation_horizon": 500,
        "tuning_eta_lambda_grid": [300.0, 1000.0, 3000.0],
        "tuning_damage_epsilon_grid": [1e-5, 3e-5, 1e-4],
        "tuning_seeds": [100, 101, 102, 103, 104],
        "eligibility_id_mse_threshold": 0.01,
        "selection_metric": "heldout_blank_memory_mse",
        "selection_blank_horizon": 4096,
        "theta_clip": [-18.0, 18.0],
        "frozen_for_main": True,
    }
    for key, expected in required_rp.items():
        if rp.get(key) != expected:
            raise ValueError(f"primary-v4 RP contract differs: {key}")
    expected_banks = {
        "tuning": {
            "trials": 1024,
            "task_seed": 0,
            "stream_key": ["primary_v4", "fixed_tuning_bank"],
        },
        "main_test": {
            "trials": 1024,
            "task_seed": 1,
            "stream_key": ["primary_v4", "fixed_main_test_bank"],
        },
    }
    if payload.get("evaluation_banks") != expected_banks:
        raise ValueError("primary-v4 tuning/main evaluation-bank contract differs")
    return payload


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    stage: str
    model_id: str
    model_seed: int
    learning_rate: float
    updates: int
    batch_size: int
    state_noise_std: float
    evaluation_bank: str
    output_dir: str
    rp_enabled: bool = False
    rp_eta_lambda: float | None = None
    rp_damage_epsilon: float | None = None
    smoke: bool = False

    def payload(self) -> dict[str, Any]:
        return _json_native(asdict(self))


class V4Model(nn.Module):
    """Common initial-state/noise/checkpoint adapter around frozen cores."""

    def __init__(self, model_id: str, width: int, core: nn.Module):
        super().__init__()
        self.model_id = str(model_id)
        self.width = int(width)
        self.core = core
        self.input_dim = 1
        self.output_dim = 2
        self.reported_state_size = int(core.state_size)
        self.primary_state_size = int(
            getattr(core, "recurrent_state_size", core.state_size)
        )
        if self.model_id == "sagodi_lstm_n64":
            self.initial_encoder_h = nn.Linear(2, self.width, bias=False)
            self.initial_encoder_c = nn.Linear(2, self.width, bias=False)
            nn.init.xavier_normal_(self.initial_encoder_h.weight)
            nn.init.xavier_normal_(self.initial_encoder_c.weight)
            self.initial_encoder = None
        else:
            self.initial_encoder = nn.Linear(2, self.primary_state_size, bias=False)
            if self.model_id in PAPER_BASELINE_IDS:
                nn.init.xavier_normal_(self.initial_encoder.weight)
            else:
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
        return self.model_id == "ca_lru_n52"

    def primary_from_reported(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., : self.primary_state_size]

    def reported_from_primary(self, primary: torch.Tensor) -> torch.Tensor:
        if self.reported_state_size == self.primary_state_size:
            return primary
        suffix = torch.zeros(
            *primary.shape[:-1],
            self.reported_state_size - self.primary_state_size,
            dtype=primary.dtype,
            device=primary.device,
        )
        return torch.cat([primary, suffix], dim=-1)

    def initial_state(
        self,
        batch: int,
        device: torch.device | str,
        initial_memory: torch.Tensor | None,
    ) -> torch.Tensor:
        if initial_memory is None or tuple(initial_memory.shape) != (int(batch), 2):
            raise ValueError("primary-v4 requires true pre-update [cos(q0),sin(q0)]")
        if self.model_id == "sagodi_lstm_n64":
            assert self.initial_encoder_h is not None and self.initial_encoder_c is not None
            return torch.cat(
                [
                    torch.tanh(self.initial_encoder_h(initial_memory)),
                    torch.tanh(self.initial_encoder_c(initial_memory)),
                ],
                dim=-1,
            )
        assert self.initial_encoder is not None
        primary = self.initial_encoder(initial_memory)
        if self.model_id == "sagodi_gru_n128":
            primary = torch.tanh(primary)
        return self.reported_from_primary(primary)

    def _reported_from_noisy_full_block(
        self, x_t: torch.Tensor, primary: torch.Tensor
    ) -> torch.Tensor:
        rec_states = [primary[:, item] for item in self.core._rec_slices]
        stream = self.core.encoder(x_t)
        for block, rec_state in zip(self.core.blocks, rec_states):
            rec_out = block.rec.output(rec_state)
            if block.update_mode == "glu":
                update = F.glu(block.glu_proj(F.gelu(rec_out)), dim=-1)
            elif block.update_mode == "gelu":
                update = F.gelu(rec_out)
            elif block.update_mode == "linear":
                update = rec_out
            else:  # pragma: no cover - constructor invariant
                raise RuntimeError(f"unsupported block update {block.update_mode!r}")
            base = stream + block.dropout(update) if block.use_residual else block.dropout(update)
            stream = block.norm_out(base)
        return self.core.merge_state(rec_states, stream)

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
        *,
        state_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        next_state = self.core.step(x_t, state)
        if state_noise is None:
            return next_state
        primary = self.primary_from_reported(next_state)
        if state_noise.shape != primary.shape:
            raise ValueError("state-noise shape differs from the causal state")
        primary = primary + state_noise
        if self.is_full_block and not bool(getattr(self.core, "carry_stream", False)):
            return self._reported_from_noisy_full_block(x_t, primary)
        if self.reported_state_size == self.primary_state_size:
            return primary
        return torch.cat([primary, next_state[..., self.primary_state_size :]], dim=-1)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.core.decode(state)

    def forward_sequence(
        self,
        inputs: torch.Tensor,
        *,
        initial_memory: torch.Tensor,
        state_noise_std: float = 0.0,
        noise_generator: torch.Generator | None = None,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        state = self.initial_state(inputs.shape[1], inputs.device, initial_memory)
        outputs: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for x_t in inputs:
            noise = None
            if float(state_noise_std) > 0.0:
                noise = torch.randn(
                    self.primary_from_reported(state).shape,
                    dtype=state.dtype,
                    device=state.device,
                    generator=noise_generator,
                ) * float(state_noise_std)
            state = self.step(x_t, state, state_noise=noise)
            outputs.append(self.decode(state))
            if return_states:
                states.append(state)
        prediction = torch.stack(outputs)
        if return_states:
            return prediction, torch.stack(states)
        return prediction

    def pan_recs_with_slices(self) -> Iterable[tuple[nn.Module, slice]]:
        method = getattr(self.core, "pan_recs_with_slices", None)
        return method() if method is not None else ()

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "width": self.width,
            "primary_state_size": self.primary_state_size,
            "reported_state_size": self.reported_state_size,
            "parameters_total": sum(parameter.numel() for parameter in self.parameters()),
            "parameters_trainable": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
            "initial_state": (
                "independent_tanh_Wotr_h_and_Wotr_c"
                if self.model_id == "sagodi_lstm_n64"
                else (
                    "tanh_Wotr_y0"
                    if self.model_id == "sagodi_gru_n128"
                    else "linear_Wotr_y0"
                )
            ),
            "readout": "direct_linear" if self.model_id in PAPER_BASELINE_IDS else "frozen_full_scaffold",
            "rp_capable": self.rp_enabled,
        }


def _structured_core(model_id: str, width: int) -> nn.Module:
    # ``models`` owns the compatibility sys.path boundary required by the
    # historical module's top-level ``pan_block`` import.
    from .models import build_model_variant

    variant = "LRU full" if model_id == "lru_n52" else "PAN-RNW-full"
    return build_model_variant(
        variant=variant,
        input_dim=1,
        output_dim=2,
        rank=2,
        d_model=int(width),
        rec_dim=int(width),
        layers=1,
        dropout=0.0,
        plru_tau=0.001,
        plru_c=50.0,
        pan_lambda_min=0.90,
        pan_lambda_max=0.999,
        rank_matched_lambda_high=0.999,
        rank_matched_lambda_low=0.0,
    )


def build_v4_model(model_id: str) -> V4Model:
    """Build a registered model and verify the executable parameter count."""

    if model_id not in MODEL_IDS:
        raise ValueError(f"unknown primary-v4 model {model_id!r}")
    width = EXPECTED_WIDTHS[model_id]
    if model_id in PAPER_BASELINE_IDS:
        from .exact_models import (
            SAGODI_GRU,
            SAGODI_LSTM,
            SAGODI_RNN_TANH,
            build_exact_core,
        )

        exact_name = {
            "sagodi_rnn_tanh_n128": SAGODI_RNN_TANH,
            "sagodi_gru_n128": SAGODI_GRU,
            "sagodi_lstm_n64": SAGODI_LSTM,
        }[model_id]
        core = build_exact_core(exact_name, input_dim=1, output_dim=2, hidden=width)
    else:
        core = _structured_core(model_id, width)
    model = V4Model(model_id, width, core)
    actual = sum(parameter.numel() for parameter in model.parameters())
    expected = EXPECTED_PARAMETER_COUNTS[model_id]
    if actual != expected:
        raise RuntimeError(f"{model_id} parameter-count mismatch: {actual} != {expected}")
    return model


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_v4_checkpoint(path: Path, model: V4Model, extra: Mapping[str, Any]) -> None:
    _atomic_torch_save(
        path,
        {
            "schema_version": 1,
            "checkpoint_type": "sagodi_primary_v4",
            "model": model.metadata(),
            "state_dict": model.state_dict(),
            "extra": _json_native(extra),
        },
    )


def load_v4_checkpoint(
    path: Path | str, device: torch.device | str = "cpu"
) -> tuple[V4Model, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("checkpoint_type") != "sagodi_primary_v4":
        raise ValueError("not a primary-v4 checkpoint")
    model = build_v4_model(str(payload["model"]["model_id"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def _configure_determinism(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _initial_memory(batch: Any, device: torch.device) -> torch.Tensor:
    value = batch.initial_memory
    if value is None:
        raise ValueError("angular task bank omitted q0 initial memory")
    return value.to(device=device)


def _to_device_batch(batch: Any, device: torch.device) -> Any:
    from .tasks import Batch

    return Batch(
        inputs=batch.inputs.to(device),
        output_targets=batch.output_targets.to(device),
        latent_targets=batch.latent_targets.to(device),
        mask=batch.mask.to(device),
        metadata=batch.metadata,
    )


def _finite_model(model: nn.Module, *, gradients: bool = False) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise FloatingPointError(f"non-finite parameter: {name}")
        if gradients and parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError(f"non-finite gradient: {name}")


@torch.no_grad()
def _evaluate(model: V4Model, batch: Any) -> dict[str, Any]:
    prediction = model.forward_sequence(
        batch.inputs,
        initial_memory=_initial_memory(batch, batch.inputs.device),
    )
    if not torch.isfinite(prediction).all().item():
        raise FloatingPointError("non-finite validation prediction")
    return _json_native(
        task_metrics(
            prediction,
            batch.output_targets,
            batch.mask,
            batch.latent_targets,
        )
    )


@torch.no_grad()
def _blank_memory_mse(model: V4Model, batch: Any, horizon: int) -> float:
    _, states = model.forward_sequence(
        batch.inputs,
        initial_memory=_initial_memory(batch, batch.inputs.device),
        return_states=True,
    )
    state = states[-1]
    blank = torch.zeros(
        state.shape[0], model.input_dim, dtype=state.dtype, device=state.device
    )
    for _ in range(int(horizon)):
        state = model.step(blank, state)
    prediction = model.decode(state)
    target = batch.output_targets[-1]
    value = (prediction - target).square().mean()
    if not torch.isfinite(value).item():
        raise FloatingPointError("non-finite blank-memory MSE")
    return float(value.cpu())


def _training_batch(config: Mapping[str, Any], spec: RunSpec, update: int, device: torch.device) -> Any:
    task = config["task"]
    return angular_integration(
        spec.batch_size,
        0,
        dimensions=1,
        init_mode=task["initialization_mode"],
        horizon=int(task["horizon"]),
        dt=float(task["dt"]),
        gp_length_scale=float(task["gp_length_scale"]),
        gp_std=float(task["gp_std"]),
        gp_jitter=float(task["gp_jitter"]),
        # The online data stream is model-independent.  The same update index
        # therefore produces the same task batch for every model and seed;
        # model_seed controls only initialization and recurrent-state noise.
        stream_key=("primary_v4", "online_train", int(update)),
        device=device,
    )


def _rp_probe_batch(config: Mapping[str, Any], spec: RunSpec, update: int, device: torch.device) -> Any:
    task = config["task"]
    rp = config["retention_plasticity"]
    return angular_integration(
        int(rp["probe_batch_size"]),
        0,
        init_mode=task["initialization_mode"],
        horizon=int(rp["probe_horizon"]),
        dt=float(task["dt"]),
        gp_length_scale=float(task["gp_length_scale"]),
        gp_std=float(task["gp_std"]),
        gp_jitter=float(task["gp_jitter"]),
        stream_key=("primary_v4", "rp_probe", int(update)),
        device=device,
    )


def _prelaunch_contract_checks(config: Mapping[str, Any]) -> dict[str, Any]:
    """Execute cheap fail-closed checks that unit tests alone cannot register."""

    device = torch.device("cpu")
    seed = 424242
    _configure_determinism(seed)
    torch.manual_seed(seed)
    ca = build_v4_model("ca_lru_n52").to(device)
    torch.manual_seed(seed)
    no_rp = build_v4_model("no_rp_n52").to(device)
    if ca.state_dict().keys() != no_rp.state_dict().keys():
        raise RuntimeError("CA-LRU/No-RP parameter keys differ before RP")
    if not all(
        torch.equal(ca.state_dict()[key], no_rp.state_dict()[key])
        for key in ca.state_dict()
    ):
        raise RuntimeError("CA-LRU/No-RP initialization differs before RP")

    common = dict(
        run_id="prelaunch_pairing",
        stage="prelaunch",
        model_seed=seed,
        learning_rate=1e-3,
        updates=1,
        batch_size=4,
        state_noise_std=float(config["training"]["state_noise_std"]),
        evaluation_bank="unused",
        output_dir="unused",
    )
    ca_spec = RunSpec(model_id="ca_lru_n52", rp_enabled=True, **common)
    no_rp_spec = RunSpec(model_id="no_rp_n52", rp_enabled=False, **common)
    batch_ca = _training_batch(config, ca_spec, 1, device)
    batch_no_rp = _training_batch(config, no_rp_spec, 1, device)
    if not torch.equal(batch_ca.inputs, batch_no_rp.inputs):
        raise RuntimeError("CA-LRU/No-RP paired task streams differ")

    optimizer_ca = torch.optim.Adam(ca.parameters(), lr=1e-3)
    optimizer_no_rp = torch.optim.Adam(no_rp.parameters(), lr=1e-3)
    noise_seed = derived_seed(seed, CAMPAIGN_ID, "state_noise")
    losses: list[torch.Tensor] = []
    for model, optimizer, batch in (
        (ca, optimizer_ca, batch_ca),
        (no_rp, optimizer_no_rp, batch_no_rp),
    ):
        generator = torch.Generator(device="cpu").manual_seed(noise_seed)
        prediction = model.forward_sequence(
            batch.inputs,
            initial_memory=_initial_memory(batch, device),
            state_noise_std=float(common["state_noise_std"]),
            noise_generator=generator,
        )
        loss = masked_mse(prediction, batch.output_targets, batch.mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    if not torch.equal(losses[0], losses[1]):
        raise RuntimeError("CA-LRU/No-RP paired pre-RP losses differ")
    if not all(
        torch.equal(ca.state_dict()[key], no_rp.state_dict()[key])
        for key in ca.state_dict()
    ):
        raise RuntimeError("CA-LRU/No-RP paired optimizer update differs")

    # Exercise the exact RP API on a reduced probe.  The configured full-size
    # call is benchmarked separately before fan-out because it is intentionally
    # expensive, but an interface mismatch must fail here immediately.
    from .train import _retention_plasticity_call

    task = config["task"]
    probe = angular_integration(
        2,
        0,
        init_mode=task["initialization_mode"],
        horizon=8,
        dt=float(task["dt"]),
        gp_length_scale=float(task["gp_length_scale"]),
        gp_std=float(task["gp_std"]),
        gp_jitter=float(task["gp_jitter"]),
        stream_key=("primary_v4", "prelaunch_rp_api"),
        device=device,
    )
    rp_result = _retention_plasticity_call(
        ca,
        probe,
        blank_horizon=2,
        eta_lambda=1.0,
        damage_epsilon=float(config["retention_plasticity"]["damage_epsilon"]),
    )
    if not all(math.isfinite(float(value)) for value in rp_result.values()):
        raise RuntimeError("reduced RP API smoke returned a non-finite value")
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "passed": True,
        "ca_no_rp_initial_state_dict_identical": True,
        "ca_no_rp_one_noisy_adam_update_identical": True,
        "inductive_pairing_contract_through_update": int(
            config["retention_plasticity"]["warmup_updates"]
        ),
        "rp_reduced_api_smoke_finite": True,
        "parameter_counts": dict(EXPECTED_PARAMETER_COUNTS),
    }


def _train_worker(spec: RunSpec, config_path: Path, device_text: str) -> Path:
    config = load_v4_config(config_path)
    campaign_root = Path(spec.evaluation_bank).expanduser().resolve().parents[1]
    expected_identity = strict_json_load(
        campaign_root / ".calru_sagodi_primary_v4_root.json"
    )
    observed_identity = _scientific_identity(
        config_path, require_clean=not spec.smoke
    )
    if observed_identity != expected_identity:
        raise RuntimeError("worker code/config identity differs from the campaign root")
    # Import the RP implementation before training starts so a later worktree
    # edit cannot change the code imported halfway through a long CA-LRU run.
    rp_train_module = None
    if spec.rp_enabled:
        from . import train as rp_train_module

    output = Path(spec.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"worker output is not empty: {output}")
    device = torch.device(device_text)
    _configure_determinism(spec.model_seed)
    model = build_v4_model(spec.model_id).to(device)
    _finite_model(model)
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run": spec.payload(),
        "model": model.metadata(),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "code_sha256": sha256_file(Path(__file__)),
        "exact_models_sha256": sha256_file(MODULE_DIR / "exact_models.py"),
        "runtime_code_sha256": expected_identity["code_sha256"],
        "scientific_identity": expected_identity["scientific_identity"],
        "evaluation_bank_sha256": sha256_file(spec.evaluation_bank),
        "started_at_utc": _utc_now(),
        "device": device_text,
    }
    atomic_json(output / "run_manifest.json", manifest)
    bank = _to_device_batch(load_fixed_bank(spec.evaluation_bank), device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(spec.learning_rate),
        betas=tuple(float(value) for value in config["training"]["betas"]),
        eps=float(config["training"]["epsilon"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    generator = torch.Generator(device=device.type)
    generator.manual_seed(derived_seed(spec.model_seed, CAMPAIGN_ID, "state_noise"))
    trace: list[dict[str, Any]] = []
    rp_trace: list[dict[str, Any]] = []
    best_mse = math.inf
    best_update: int | None = None
    best_path = output / "checkpoint_best_diagnostic.pt"
    trace_interval = int(config["training"]["trace_interval"])
    validation_interval = int(config["training"]["validation_interval"])
    rp = config["retention_plasticity"]
    started = time.time()
    for update in range(1, int(spec.updates) + 1):
        model.train()
        batch = _training_batch(config, spec, update, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_sequence(
            batch.inputs,
            initial_memory=_initial_memory(batch, device),
            state_noise_std=float(spec.state_noise_std),
            noise_generator=generator,
        )
        loss = masked_mse(prediction, batch.output_targets, batch.mask)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()
        _finite_model(model, gradients=True)
        optimizer.step()
        _finite_model(model)

        if (
            spec.rp_enabled
            and update > int(rp["warmup_updates"])
            and update % int(rp["interval_updates"]) == 0
        ):
            probe = _rp_probe_batch(config, spec, update, device)
            if rp_train_module is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("RP implementation was not frozen at worker start")
            result = rp_train_module._retention_plasticity_call(
                model,
                probe,
                blank_horizon=int(rp["blank_ablation_horizon"]),
                eta_lambda=float(spec.rp_eta_lambda),
                damage_epsilon=float(spec.rp_damage_epsilon),
            )
            rp_trace.append({"update": update, **_json_native(result)})

        should_trace = update == 1 or update % trace_interval == 0 or update == spec.updates
        should_validate = update % validation_interval == 0 or update == spec.updates
        row: dict[str, Any] | None = None
        if should_trace:
            row = {
                "update": update,
                "train_mse": float(loss.detach().cpu()),
                "elapsed_seconds": float(time.time() - started),
            }
        if should_validate:
            model.eval()
            validation = _evaluate(model, bank)
            if row is None:
                row = {"update": update, "train_mse": float(loss.detach().cpu())}
            row["validation"] = validation
            validation_mse = float(validation["masked_mse"])
            if validation_mse < best_mse:
                best_mse = validation_mse
                best_update = update
                save_v4_checkpoint(
                    best_path,
                    model,
                    {"run": spec.payload(), "update": update, "diagnostic_only": True},
                )
        if row is not None:
            trace.append(row)

    model.eval()
    final_metrics = _evaluate(model, bank)
    blank_horizon = int(config["retention_plasticity"]["selection_blank_horizon"])
    blank_mse = None
    if spec.stage == "rp_tune":
        blank_mse = _blank_memory_mse(model, bank, blank_horizon)
    result = {
        "schema_version": 1,
        "run_id": spec.run_id,
        "stage": spec.stage,
        "model_id": spec.model_id,
        "model_seed": spec.model_seed,
        "learning_rate": spec.learning_rate,
        "updates_completed": spec.updates,
        "final_metrics": final_metrics,
        "heldout_blank_memory_mse": blank_mse,
        "best_validation_mse_diagnostic": best_mse,
        "best_validation_update_diagnostic": best_update,
        "final_checkpoint_is_primary": True,
        "rp_call_count": len(rp_trace),
        "completed_at_utc": _utc_now(),
    }
    atomic_json(output / "training_trace.json", trace)
    atomic_json(output / "rp_trace.json", rp_trace)
    atomic_json(output / "result.json", result)
    save_v4_checkpoint(
        output / "checkpoint_final.pt",
        model,
        {"run": spec.payload(), "result": result},
    )
    complete = {
        "schema_version": 1,
        "status": "complete",
        "run_id": spec.run_id,
        "updates_completed": spec.updates,
    }
    atomic_json(output / "COMPLETE", complete)
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=spec.run_id,
        artifacts=[
            output / "run_manifest.json",
            output / "training_trace.json",
            output / "rp_trace.json",
            output / "result.json",
            output / "checkpoint_final.pt",
            best_path,
            output / "COMPLETE",
        ],
        metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
        },
    )
    return output


def _model_item(config: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    return next(item for item in config["models"] if item["id"] == model_id)


def build_smoke_plan(root: Path, config: Mapping[str, Any], bank: Path) -> tuple[RunSpec, ...]:
    return tuple(
        RunSpec(
            run_id=f"smoke__{model_id}",
            stage="smoke",
            model_id=model_id,
            model_seed=999,
            learning_rate=1e-3,
            updates=2,
            batch_size=4,
            state_noise_std=float(config["training"]["state_noise_std"]),
            evaluation_bank=str(bank),
            output_dir=str(root / "smoke" / "runs" / model_id),
            smoke=True,
        )
        for model_id in MODEL_IDS
    )


def build_lr_tune_plan(root: Path, config: Mapping[str, Any], bank: Path) -> tuple[RunSpec, ...]:
    tuning = config["learning_rate_tuning"]
    plan: list[RunSpec] = []
    for model_id in TUNED_LR_MODEL_IDS:
        for rate in tuning["grid"]:
            for seed in tuning["seeds"]:
                rate_key = format(float(rate), ".0e").replace("-0", "-")
                run_id = f"lr_tune__{model_id}__lr{rate_key}__seed{int(seed)}"
                plan.append(
                    RunSpec(
                        run_id=run_id,
                        stage="lr_tune",
                        model_id=model_id,
                        model_seed=int(seed),
                        learning_rate=float(rate),
                        updates=int(tuning["paper_selector_updates"]),
                        batch_size=int(config["training"]["batch_size"]),
                        state_noise_std=float(config["training"]["state_noise_std"]),
                        evaluation_bank=str(bank),
                        output_dir=str(root / "tune" / "lr" / "runs" / run_id),
                    )
                )
    return tuple(plan)


def _result_for(spec: RunSpec) -> dict[str, Any]:
    result = strict_json_load(Path(spec.output_dir) / "result.json")
    if result.get("run_id") != spec.run_id:
        raise RuntimeError(f"run result identity mismatch: {spec.run_id}")
    return result


def select_learning_rates(
    plan: Sequence[RunSpec], config: Mapping[str, Any]
) -> dict[str, Any]:
    rates = [float(value) for value in config["learning_rate_tuning"]["grid"]]
    winners: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    for model_id in TUNED_LR_MODEL_IDS:
        rows: list[dict[str, Any]] = []
        for rate in rates:
            matching = [
                spec
                for spec in plan
                if spec.model_id == model_id and spec.learning_rate == rate
            ]
            losses = []
            for spec in matching:
                trace = strict_json_load(Path(spec.output_dir) / "training_trace.json")
                final_rows = [row for row in trace if int(row["update"]) == 100]
                if len(final_rows) != 1:
                    raise RuntimeError(f"missing online update-100 loss: {spec.run_id}")
                losses.append(float(final_rows[0]["train_mse"]))
            if len(losses) != 5 or not all(math.isfinite(value) for value in losses):
                raise RuntimeError(f"incomplete LR cell for {model_id} at {rate:g}")
            rows.append(
                {
                    "learning_rate": rate,
                    "seed_count": len(losses),
                    "mean_masked_mse_at_update_100": float(np.mean(losses)),
                    "per_seed_masked_mse": losses,
                }
            )
        winner = min(rows, key=lambda item: (item["mean_masked_mse_at_update_100"], rates.index(item["learning_rate"])))
        audit[model_id] = rows
        winners[model_id] = winner
    winners["no_rp_n52"] = {
        **winners["ca_lru_n52"],
        "inherited_from": "ca_lru_n52",
        "independently_tuned": False,
    }
    return {
        "schema_version": 1,
        "selection_role": "paper_100_update_online_loss_audit_only",
        "winners": winners,
        "audit": audit,
    }


def build_sentinel_plan(
    root: Path,
    config: Mapping[str, Any],
    bank: Path,
    lr_selection: Mapping[str, Any],
) -> tuple[RunSpec, ...]:
    tuning = config["learning_rate_tuning"]
    plan: list[RunSpec] = []
    # Paper baselines remain fixed at 1e-2.  Every LRU/CA candidate is
    # continued to 5,000 updates; this prevents selection among four
    # chance-level 100-update curves.  lr_selection is retained in the
    # signature to make the audit dependency explicit, but never supplies a
    # scientific winner.
    if lr_selection.get("selection_role") != "paper_100_update_online_loss_audit_only":
        raise ValueError("sentinel plan requires the completed paper LR audit")
    rates_by_model = {
        **{
            model_id: [float(tuning["paper_baseline_learning_rate"])]
            for model_id in PAPER_BASELINE_IDS
        },
        "lru_n52": [float(value) for value in tuning["grid"]],
        "ca_lru_n52": [float(value) for value in tuning["grid"]],
    }
    for model_id, rates in rates_by_model.items():
        for rate in rates:
            for seed in tuning["seeds"]:
                rate_key = format(float(rate), ".0e").replace("-0", "-")
                run_id = f"lr_sentinel__{model_id}__lr{rate_key}__seed{int(seed)}"
                plan.append(
                    RunSpec(
                        run_id=run_id,
                        stage="lr_sentinel",
                        model_id=model_id,
                        model_seed=int(seed),
                        learning_rate=float(rate),
                        updates=int(tuning["continued_sentinel_updates"]),
                        batch_size=int(config["training"]["batch_size"]),
                        state_noise_std=float(config["training"]["state_noise_std"]),
                        evaluation_bank=str(bank),
                        output_dir=str(root / "tune" / "sentinel" / "runs" / run_id),
                    )
                )
    return tuple(plan)


def summarize_sentinels(
    plan: Sequence[RunSpec], config: Mapping[str, Any]
) -> dict[str, Any]:
    threshold = float(config["learning_rate_tuning"]["success_mse_threshold"])
    summary: dict[str, Any] = {}
    selected_rates: dict[str, float] = {}
    for model_id in (*PAPER_BASELINE_IDS, "lru_n52", "ca_lru_n52"):
        rates = sorted({spec.learning_rate for spec in plan if spec.model_id == model_id}, reverse=True)
        cells: list[dict[str, Any]] = []
        for rate in rates:
            matching = [
                spec for spec in plan if spec.model_id == model_id and spec.learning_rate == rate
            ]
            losses = [float(_result_for(spec)["final_metrics"]["masked_mse"]) for spec in matching]
            eligible_count = sum(math.isfinite(value) and value < threshold for value in losses)
            cells.append(
                {
                    "learning_rate": rate,
                    "seed_count": len(losses),
                    "eligible_seed_count": eligible_count,
                    "per_seed_masked_mse": losses,
                    "mean_masked_mse": float(np.mean(losses)),
                    "cell_gate_passed": len(losses) == 5 and eligible_count == 5,
                }
            )
        viable = [cell for cell in cells if cell["cell_gate_passed"]]
        selected = min(viable, key=lambda cell: (cell["mean_masked_mse"], -cell["learning_rate"])) if viable else None
        # Exact baseline LR is paper-fixed even though we still demand that at
        # least one non-main seed learns.  LRU/CA require all five tuning seeds
        # for a candidate before that candidate can win.
        gate_passed = (
            any(cell["eligible_seed_count"] > 0 for cell in cells)
            if model_id in PAPER_BASELINE_IDS
            else selected is not None
        )
        if model_id in PAPER_BASELINE_IDS:
            selected_rates[model_id] = float(config["learning_rate_tuning"]["paper_baseline_learning_rate"])
        elif selected is not None:
            selected_rates[model_id] = float(selected["learning_rate"])
        summary[model_id] = {
            "success_mse_threshold": threshold,
            "cells": cells,
            "selected": selected,
            "gate_passed": gate_passed,
            "selection_role": (
                "paper_fixed_lr_learning_sentinel"
                if model_id in PAPER_BASELINE_IDS
                else "full_5000_update_lr_selection"
            ),
        }
    if "ca_lru_n52" in selected_rates:
        selected_rates["no_rp_n52"] = selected_rates["ca_lru_n52"]
    failed = [model_id for model_id, item in summary.items() if not item["gate_passed"]]
    return {
        "schema_version": 1,
        "models": summary,
        "all_model_learning_gates_passed": not failed,
        "failed_models": failed,
        "selected_learning_rates": selected_rates,
    }


def build_rp_tune_plan(
    root: Path,
    config: Mapping[str, Any],
    bank: Path,
    lr_selection: Mapping[str, Any],
) -> tuple[RunSpec, ...]:
    rp = config["retention_plasticity"]
    rate = float(lr_selection["selected_learning_rates"]["ca_lru_n52"])
    plan: list[RunSpec] = []
    for eta in rp["tuning_eta_lambda_grid"]:
        for epsilon in rp["tuning_damage_epsilon_grid"]:
            for seed in rp["tuning_seeds"]:
                eta_key = format(float(eta), "g")
                epsilon_key = format(float(epsilon), ".0e").replace("-0", "-")
                run_id = f"rp_tune__eta{eta_key}__eps{epsilon_key}__seed{int(seed)}"
                plan.append(
                    RunSpec(
                        run_id=run_id,
                        stage="rp_tune",
                        model_id="ca_lru_n52",
                        model_seed=int(seed),
                        learning_rate=rate,
                        updates=int(config["training"]["updates"]),
                        batch_size=int(config["training"]["batch_size"]),
                        state_noise_std=float(config["training"]["state_noise_std"]),
                        evaluation_bank=str(bank),
                        output_dir=str(root / "tune" / "rp" / "runs" / run_id),
                        rp_enabled=True,
                        rp_eta_lambda=float(eta),
                        rp_damage_epsilon=float(epsilon),
                    )
                )
    return tuple(plan)


def select_rp_hyperparameters(
    plan: Sequence[RunSpec], config: Mapping[str, Any]
) -> dict[str, Any]:
    rp = config["retention_plasticity"]
    threshold = float(rp["eligibility_id_mse_threshold"])
    cells: list[dict[str, Any]] = []
    for eta in rp["tuning_eta_lambda_grid"]:
        for epsilon in rp["tuning_damage_epsilon_grid"]:
            matching = [
                spec
                for spec in plan
                if spec.rp_eta_lambda == float(eta)
                and spec.rp_damage_epsilon == float(epsilon)
            ]
            results = [_result_for(spec) for spec in matching]
            id_losses = [float(item["final_metrics"]["masked_mse"]) for item in results]
            blank_losses = [float(item["heldout_blank_memory_mse"]) for item in results]
            eligible = (
                len(results) == 5
                and all(math.isfinite(value) and value < threshold for value in id_losses)
                and all(math.isfinite(value) for value in blank_losses)
            )
            cells.append(
                {
                    "eta_lambda": float(eta),
                    "damage_epsilon": float(epsilon),
                    "seed_count": len(results),
                    "per_seed_id_mse": id_losses,
                    "per_seed_blank_memory_mse": blank_losses,
                    "mean_id_mse": float(np.mean(id_losses)),
                    "mean_blank_memory_mse": float(np.mean(blank_losses)),
                    "eligible": eligible,
                }
            )
    eligible_cells = [item for item in cells if item["eligible"]]
    if not eligible_cells:
        raise RuntimeError("no RP tuning cell passed the preregistered ID-MSE gate")
    selected = min(
        eligible_cells,
        key=lambda item: (
            item["mean_blank_memory_mse"],
            item["mean_id_mse"],
            item["eta_lambda"],
            item["damage_epsilon"],
        ),
    )
    return {
        "schema_version": 1,
        "selection_metric": rp["selection_metric"],
        "eligibility_id_mse_threshold": threshold,
        "selected": selected,
        "cells": cells,
    }


def build_main_plan(
    root: Path,
    config: Mapping[str, Any],
    bank: Path,
    tune_summary: Mapping[str, Any],
) -> tuple[RunSpec, ...]:
    selected_lr = tune_summary["selected_learning_rates_for_main"]
    selected_rp = tune_summary["rp_selection"]["selected"]
    plan: list[RunSpec] = []
    for model_id in MODEL_IDS:
        for seed in config["main"]["seeds"]:
            run_id = f"main__{model_id}__seed{int(seed):02d}"
            rp_enabled = model_id == "ca_lru_n52"
            plan.append(
                RunSpec(
                    run_id=run_id,
                    stage="main",
                    model_id=model_id,
                    model_seed=int(seed),
                    learning_rate=float(selected_lr[model_id]),
                    updates=int(config["training"]["updates"]),
                    batch_size=int(config["training"]["batch_size"]),
                    state_noise_std=float(config["training"]["state_noise_std"]),
                    evaluation_bank=str(bank),
                    output_dir=str(root / "main" / "runs" / run_id),
                    rp_enabled=rp_enabled,
                    rp_eta_lambda=float(selected_rp["eta_lambda"]) if rp_enabled else None,
                    rp_damage_epsilon=(
                        float(selected_rp["damage_epsilon"]) if rp_enabled else None
                    ),
                )
            )
    return tuple(plan)


def _git_state(repo: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    return {"code_commit": commit, "worktree_dirty": bool(status.strip())}


def _runtime_code_files() -> tuple[Path, ...]:
    legacy = Path(__file__).resolve().parents[1] / "legacy_code"
    return (
        Path(__file__).resolve(),
        MODULE_DIR / "exact_models.py",
        MODULE_DIR / "train.py",
        MODULE_DIR / "tasks.py",
        MODULE_DIR / "metrics.py",
        MODULE_DIR / "artifacts.py",
        MODULE_DIR / "models.py",
        legacy / "exp71_pan_block_pulse_hold.py",
        legacy / "pan_block.py",
        legacy / "plru_regularizers.py",
    )


def _scientific_identity(
    config_source: Path, *, require_clean: bool
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    git = _git_state(repo)
    if require_clean and git["worktree_dirty"]:
        raise RuntimeError("full primary-v4 stages require a clean committed worktree")
    config = load_v4_config(config_source)
    code_hashes = {
        str(path.relative_to(repo)): sha256_file(path)
        for path in _runtime_code_files()
    }
    identity = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protocol_revision": config["protocol_revision"],
        "source_config_sha256": sha256_file(config_source),
        "source_freeze_sha256": sha256_file(FREEZE_DOCUMENT),
        "code_sha256": code_hashes,
        "code_commit": git["code_commit"],
    }
    identity["scientific_identity"] = canonical_hash(identity)
    return identity


def _copy_or_verify(source: Path, destination: Path) -> None:
    payload = source.read_bytes()
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != payload:
            raise RuntimeError(f"immutable campaign input differs: {destination.name}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(raw, destination)
        except BaseException:
            Path(raw).unlink(missing_ok=True)
            raise


def _prepare_root(
    root: Path,
    config_source: Path,
    *,
    require_clean: bool,
) -> tuple[dict[str, Any], Path]:
    root = root.expanduser().resolve()
    config = load_v4_config(config_source)
    identity = _scientific_identity(config_source, require_clean=require_clean)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".calru_sagodi_primary_v4_root.json"
    if marker.exists():
        observed = strict_json_load(marker)
        if observed != identity:
            raise RuntimeError("artifact root belongs to a different primary-v4 identity")
    else:
        unexpected = [item.name for item in root.iterdir()]
        if unexpected:
            raise RuntimeError(f"unmarked artifact root is not empty: {unexpected}")
        atomic_json(marker, identity)
    copied_config = root / "inputs" / "primary_v4_config.json"
    _copy_or_verify(config_source, copied_config)
    _copy_or_verify(FREEZE_DOCUMENT, root / "inputs" / FREEZE_DOCUMENT.name)
    return config, copied_config


def _ensure_bank(
    root: Path, config: Mapping[str, Any], *, purpose: str
) -> Path:
    if purpose not in {"smoke", "tuning", "main_test"}:
        raise ValueError("bank purpose must be smoke, tuning, or main_test")
    if purpose == "smoke":
        bank_contract = config["evaluation_banks"]["tuning"]
        bank_name = "smoke.npz"
        trials = 16
        task_seed = 999
        stream_key = ("primary_v4", "fixed_smoke_bank")
    else:
        bank_contract = config["evaluation_banks"][purpose]
        bank_name = f"{purpose}.npz"
        trials = int(bank_contract["trials"])
        task_seed = int(bank_contract["task_seed"])
        stream_key = tuple(bank_contract["stream_key"])
    bank = root / "banks" / bank_name
    if not bank.exists():
        task = config["task"]
        batch = angular_integration(
            trials,
            task_seed,
            init_mode=task["initialization_mode"],
            horizon=int(task["horizon"]),
            dt=float(task["dt"]),
            gp_length_scale=float(task["gp_length_scale"]),
            gp_std=float(task["gp_std"]),
            gp_jitter=float(task["gp_jitter"]),
            stream_key=stream_key,
            device="cpu",
        )
        save_fixed_bank(bank, batch)
    observed = load_fixed_bank(bank)
    if observed.batch_size != trials or observed.time_steps != 256:
        raise RuntimeError("fixed validation bank shape differs from the stage contract")
    return bank


def _verified_child(spec: RunSpec) -> bool:
    root = Path(spec.output_dir)
    valid, _ = verify_completion_receipt(
        root / "completion_receipt.json",
        expected_job_id=spec.run_id,
        expected_metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
        },
    )
    if not valid or not (root / "COMPLETE").is_file():
        return False
    try:
        manifest = strict_json_load(root / "run_manifest.json")
        result = strict_json_load(root / "result.json")
        campaign_root = Path(spec.evaluation_bank).expanduser().resolve().parents[1]
        identity = strict_json_load(
            campaign_root / ".calru_sagodi_primary_v4_root.json"
        )
        evaluation_bank_sha256 = sha256_file(spec.evaluation_bank)
    except (OSError, ValueError):
        return False
    return (
        manifest.get("run") == spec.payload()
        and result.get("run_id") == spec.run_id
        and manifest.get("scientific_identity") == identity.get("scientific_identity")
        and manifest.get("runtime_code_sha256") == identity.get("code_sha256")
        and manifest.get("config_sha256") == identity.get("source_config_sha256")
        and manifest.get("evaluation_bank_sha256") == evaluation_bank_sha256
    )


def _parse_compute_slots(text: str) -> tuple[str, ...]:
    normalized = str(text).strip().lower()
    if normalized == "cpu":
        return ("cpu",)
    values = tuple(item.strip() for item in normalized.split(",") if item.strip())
    if not values or any(not item.isdigit() for item in values):
        raise ValueError("--gpus must be comma-separated physical ids or 'cpu'")
    if len(set(values)) != len(values):
        raise ValueError("--gpus contains a duplicate physical id")
    return values


def _archive_partial(output: Path, attempts: Path) -> None:
    if not output.exists():
        return
    attempts.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = attempts / f"{output.name}.{stamp}"
    os.replace(output, destination)


def _write_stage_plan(stage_root: Path, specs: Sequence[RunSpec]) -> None:
    payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_count": len(specs),
        "runs": [spec.payload() for spec in specs],
    }
    destination = stage_root / "plan.json"
    if destination.exists():
        if strict_json_load(destination) != payload:
            raise RuntimeError(f"immutable stage plan differs: {stage_root}")
    else:
        atomic_json(destination, payload)


def _run_specs(
    specs: Sequence[RunSpec],
    *,
    config_path: Path,
    compute_slots: Sequence[str],
) -> None:
    if not specs:
        return
    stage_root = Path(specs[0].output_dir).parents[1]
    _write_stage_plan(stage_root, specs)
    pending = [spec for spec in specs if not _verified_child(spec)]
    for spec in pending:
        _archive_partial(Path(spec.output_dir), stage_root / "attempts")
    specs_dir = stage_root / "specs"
    logs_dir = stage_root / "logs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    running: dict[str, tuple[subprocess.Popen[Any], RunSpec, Any, str]] = {}
    queue = list(pending)
    repo = Path(__file__).resolve().parents[2]
    while queue or running:
        free = [slot for slot in compute_slots if slot not in running]
        for slot in free:
            if not queue:
                break
            spec = queue.pop(0)
            spec_path = specs_dir / f"{spec.run_id}.json"
            atomic_json(spec_path, spec.payload())
            log_handle = (logs_dir / f"{spec.run_id}.log").open("ab")
            device = "cpu" if slot == "cpu" else "cuda:0"
            command = [
                sys.executable,
                "-m",
                "repro.sagodi_protocol.primary_v4",
                "--worker-spec",
                str(spec_path),
                "--config",
                str(config_path),
                "--device",
                device,
            ]
            environment = os.environ.copy()
            if slot != "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = slot
            process = subprocess.Popen(
                command,
                cwd=repo,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[slot] = (process, spec, log_handle, " ".join(command))
        if not running:
            continue
        time.sleep(0.2)
        for slot, (process, spec, handle, command) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            del running[slot]
            if return_code != 0 or not _verified_child(spec):
                for other, _, other_handle, _ in running.values():
                    other.terminate()
                    other_handle.close()
                raise RuntimeError(
                    f"primary-v4 child failed ({return_code}): {spec.run_id}; {command}"
                )
        status = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "registered": len(specs),
            "verified_complete": sum(_verified_child(spec) for spec in specs),
            "pending": len(queue),
            "running": [item[1].run_id for item in running.values()],
            "updated_at_utc": _utc_now(),
        }
        atomic_json(stage_root / "status.json", status)


def _finalize_stage(
    stage_root: Path,
    *,
    stage: str,
    specs: Sequence[RunSpec],
    extra_artifacts: Sequence[Path],
) -> Path:
    missing = [spec.run_id for spec in specs if not _verified_child(spec)]
    if missing:
        raise RuntimeError(f"cannot finalize {stage}; unverified children: {missing[:3]}")
    complete = stage_root / "COMPLETE"
    atomic_json(
        complete,
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "stage": stage,
            "verified_run_count": len(specs),
            "completed_at_utc": _utc_now(),
        },
    )
    receipt = stage_root / "completion_receipt.json"
    artifacts = [stage_root / "plan.json", complete, *extra_artifacts]
    write_completion_receipt(
        receipt,
        job_id=f"{CAMPAIGN_ID}__{stage}",
        artifacts=artifacts,
        metadata={"campaign_id": CAMPAIGN_ID, "stage": stage, "run_count": len(specs)},
    )
    return receipt


def _write_or_verify_json(path: Path, payload: Mapping[str, Any]) -> None:
    native = _json_native(payload)
    if path.exists():
        if strict_json_load(path) != native:
            raise RuntimeError(f"immutable derived artifact differs: {path}")
    else:
        atomic_json(path, native)


def _stage_receipt_valid(stage_root: Path, stage: str, run_count: int) -> bool:
    valid, _ = verify_completion_receipt(
        stage_root / "completion_receipt.json",
        expected_job_id=f"{CAMPAIGN_ID}__{stage}",
        expected_metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": stage,
            "run_count": run_count,
        },
    )
    if not valid or not (stage_root / "COMPLETE").is_file():
        return False
    try:
        plan = strict_json_load(stage_root / "plan.json")
        runs = plan.get("runs")
        if (
            plan.get("campaign_id") != CAMPAIGN_ID
            or int(plan.get("run_count", -1)) != int(run_count)
            or not isinstance(runs, list)
            or len(runs) != int(run_count)
        ):
            return False
        specs = tuple(RunSpec(**payload) for payload in runs)
    except (OSError, TypeError, ValueError):
        return False
    return all(_verified_child(spec) for spec in specs)


def _main_metric_summary(specs: Sequence[RunSpec]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        matching = [spec for spec in specs if spec.model_id == model_id]
        mse = [float(_result_for(spec)["final_metrics"]["masked_mse"]) for spec in matching]
        nmse_db = [
            float(_result_for(spec)["final_metrics"]["masked_nmse_db"])
            for spec in matching
        ]
        models[model_id] = {
            "seed_count": len(matching),
            "masked_mse_mean": float(np.mean(mse)),
            "masked_mse_population_std": float(np.std(mse, ddof=0)),
            "masked_nmse_db_mean": float(np.mean(nmse_db)),
            "per_seed_masked_mse": mse,
            "per_seed_masked_nmse_db": nmse_db,
        }
    return {"schema_version": 1, "campaign_id": CAMPAIGN_ID, "models": models}


def run_stage(
    *,
    stage: str,
    artifact_root: Path,
    config_source: Path,
    compute_slots: Sequence[str],
) -> Path:
    if stage not in {"smoke", "tune", "main"}:
        raise ValueError("stage must be smoke, tune, or main")
    if not compute_slots:
        raise ValueError("at least one compute slot is required")
    if compute_slots != ("cpu",) and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU ids were requested but torch.cuda is unavailable")
    root = Path(artifact_root).expanduser().resolve()
    config, copied_config = _prepare_root(
        root,
        Path(config_source).expanduser().resolve(strict=True),
        require_clean=stage != "smoke",
    )
    bank_purpose = {
        "smoke": "smoke",
        "tune": "tuning",
        "main": "main_test",
    }[stage]
    bank = _ensure_bank(root, config, purpose=bank_purpose)
    prelaunch_path = root / stage / "prelaunch_contract_checks.json"
    if stage in {"smoke", "tune"}:
        _write_or_verify_json(prelaunch_path, _prelaunch_contract_checks(config))

    if stage == "smoke":
        stage_root = root / "smoke"
        specs = build_smoke_plan(root, config, bank)
        if _stage_receipt_valid(stage_root, stage, len(specs)):
            return stage_root
        _run_specs(specs, config_path=copied_config, compute_slots=compute_slots)
        summary = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "models": {
                spec.model_id: _result_for(spec)["final_metrics"] for spec in specs
            },
            "scientific_result": False,
        }
        summary_path = stage_root / "summary.json"
        _write_or_verify_json(summary_path, summary)
        _finalize_stage(
            stage_root,
            stage=stage,
            specs=specs,
            extra_artifacts=[summary_path, prelaunch_path],
        )
        return stage_root

    tune_root = root / "tune"
    expected_tune_runs = 100 + 55 + 45
    if stage == "tune" and _stage_receipt_valid(tune_root, "tune", expected_tune_runs):
        return tune_root

    if stage == "main":
        if not _stage_receipt_valid(tune_root, "tune", expected_tune_runs):
            raise RuntimeError("main is blocked until the verified v4 tune stage completes")
        tune_summary = strict_json_load(tune_root / "tune_summary.json")
        stage_root = root / "main"
        specs = build_main_plan(root, config, bank, tune_summary)
        if _stage_receipt_valid(stage_root, stage, len(specs)):
            return stage_root
        _run_specs(specs, config_path=copied_config, compute_slots=compute_slots)
        summary_path = stage_root / "summary.json"
        _write_or_verify_json(summary_path, _main_metric_summary(specs))
        tune_binding = stage_root / "parent_tune_binding.json"
        _write_or_verify_json(
            tune_binding,
            {
                "schema_version": 1,
                "parent_tune_summary_sha256": sha256_file(
                    tune_root / "tune_summary.json"
                ),
                "parent_tune_completion_receipt_sha256": sha256_file(
                    tune_root / "completion_receipt.json"
                ),
            },
        )
        _finalize_stage(
            stage_root,
            stage=stage,
            specs=specs,
            extra_artifacts=[summary_path, tune_binding],
        )
        return stage_root

    lr_plan = build_lr_tune_plan(root, config, bank)
    _run_specs(lr_plan, config_path=copied_config, compute_slots=compute_slots)
    lr_audit = select_learning_rates(lr_plan, config)
    lr_audit_path = tune_root / "lr_audit.json"
    _write_or_verify_json(lr_audit_path, lr_audit)

    sentinel_plan = build_sentinel_plan(root, config, bank, lr_audit)
    _run_specs(sentinel_plan, config_path=copied_config, compute_slots=compute_slots)
    sentinel_summary = summarize_sentinels(sentinel_plan, config)
    sentinel_path = tune_root / "sentinel_summary.json"
    _write_or_verify_json(sentinel_path, sentinel_summary)
    if not sentinel_summary["all_model_learning_gates_passed"]:
        raise RuntimeError(
            "5,000-update learning gate failed for: "
            + ", ".join(sentinel_summary["failed_models"])
        )

    rp_plan = build_rp_tune_plan(root, config, bank, sentinel_summary)
    _run_specs(rp_plan, config_path=copied_config, compute_slots=compute_slots)
    rp_selection = select_rp_hyperparameters(rp_plan, config)
    rp_path = tune_root / "rp_selection.json"
    _write_or_verify_json(rp_path, rp_selection)

    selected_rates = dict(sentinel_summary["selected_learning_rates"])
    # These assertions prevent an audit winner from replacing a paper row.
    for model_id in PAPER_BASELINE_IDS:
        if selected_rates.get(model_id) != 0.01:
            raise RuntimeError(f"paper baseline LR changed for {model_id}")
    if selected_rates.get("no_rp_n52") != selected_rates.get("ca_lru_n52"):
        raise RuntimeError("No-RP did not inherit the CA-LRU learning rate")
    tune_summary = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "paper_lr_audit": lr_audit,
        "learning_sentinel_and_full_lr_selection": sentinel_summary,
        "rp_selection": rp_selection,
        "selected_learning_rates_for_main": selected_rates,
        "no_rp_independently_tuned": False,
        "manifold_geometry_used_for_tuning": False,
        "blank_memory_metric_used_for_rp_tuning": True,
    }
    tune_summary_path = tune_root / "tune_summary.json"
    _write_or_verify_json(tune_summary_path, tune_summary)
    all_specs = (*lr_plan, *sentinel_plan, *rp_plan)
    _write_stage_plan(tune_root, all_specs)
    _finalize_stage(
        tune_root,
        stage="tune",
        specs=all_specs,
        extra_artifacts=[
            prelaunch_path,
            lr_audit_path,
            sentinel_path,
            rp_path,
            tune_summary_path,
        ],
    )
    return tune_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "tune", "main"))
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_spec is not None:
        if args.stage is not None or args.artifact_root is not None:
            raise ValueError("worker mode cannot be combined with stage launch options")
        payload = strict_json_load(args.worker_spec)
        spec = RunSpec(**payload)
        _train_worker(spec, Path(args.config).resolve(strict=True), str(args.device))
        return 0
    if args.stage is None or args.artifact_root is None:
        raise ValueError("campaign mode requires --stage and --artifact-root")
    slots = _parse_compute_slots(args.gpus)
    destination = run_stage(
        stage=args.stage,
        artifact_root=args.artifact_root,
        config_source=args.config,
        compute_slots=slots,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
