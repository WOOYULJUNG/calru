"""Paper-and-source-informed auxiliary controlled baseline benchmark.

This campaign is a new scientific identity.  It does not reinterpret or
overwrite the historical primary-v4 or source-resolved-v5 campaigns.  The
primary comparison is deliberately small: tanh RNN, GRU, and LSTM on one
common 256-step dense angular-integration protocol.  It is not an exact paper
or public-code reproduction; recurrence/noise alternatives are registered as
neutral sensitivity conditions.

Stages::

    python -m repro.sagodi_protocol.sagodi_paper_baselines --stage smoke ...
    python -m repro.sagodi_protocol.sagodi_paper_baselines --stage screen ...
    python -m repro.sagodi_protocol.sagodi_paper_baselines --stage main ...
    python -m repro.sagodi_protocol.sagodi_paper_baselines --stage sensitivity ...

``screen`` evaluates every learning-rate candidate with five seeds for the
full 5,000 updates.  ``main`` reports MSE yield for all ten seeds and marks
individual NMSE < -20 dB seeds as analysis eligible.  Zero eligible seeds for
any model is the only scientific hard failure; low yield is an explicit warning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .artifacts import (
    atomic_json,
    canonical_hash,
    derived_seed,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from .exact_models import SAGODI_GRU, SAGODI_LSTM, build_exact_core
from .metrics import masked_mse, task_metrics
from .tasks import (
    Batch,
    angular_integration,
    keyed_seed,
    load_fixed_bank,
    metadata_payload,
    save_fixed_bank,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "sagodi_paper_baselines_v1.json"
FREEZE_DOCUMENT = MODULE_DIR / "SAGODI_CONTROLLED_RECOVERY_BASELINES_V1_FREEZE_ko.md"
CAMPAIGN_ID = "sagodi_paper_baselines_v1"
PROTOCOL_REVISION = "paper_and_source_informed_controlled_benchmark_v1"
MODEL_IDS = ("rnn_tanh_n128", "gru_n128", "lstm_n64")
WIDTHS = {"rnn_tanh_n128": 128, "gru_n128": 128, "lstm_n64": 64}
PARAMETER_COUNTS = {
    "rnn_tanh_n128": 17154,
    "gru_n128": 50818,
    "lstm_n64": 17538,
}
PRIMARY_RNN_VARIANT = "pure_tanh"
STANDARD_VARIANT = "standard"
LEAKY_RNN_VARIANT = "leaky_tanh_dt_0.1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_native(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the registered contract and reject scientific drift."""

    source = Path(path).expanduser().resolve(strict=True)
    payload = strict_json_load(source)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("baseline config must be a schema-1 JSON object")
    if payload.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("baseline campaign_id differs")
    if payload.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError("baseline protocol revision differs")
    if payload.get("track_classification") != (
        "paper_and_source_informed_controlled_benchmark"
    ):
        raise ValueError("baseline track classification differs")
    if payload.get("protocol_references") != {
        "controlled_recovery_primary": {
            "rnn_recurrence": PRIMARY_RNN_VARIANT,
            "state_noise_std_after_transition": 0.0316227766,
            "provenance_role": "paper_and_source_informed_controlled_benchmark",
            "not_claimed_as": "bit_exact_public_runner_or_exact_paper_protocol",
        },
        "leaky_noise_0p1_reference": {
            "rnn_recurrence": LEAKY_RNN_VARIANT,
            "state_noise_std_after_transition": 0.1,
            "role": "recurrence_noise_sensitivity_only",
        },
        "leaky_noise_0p03162_reference": {
            "rnn_recurrence": LEAKY_RNN_VARIANT,
            "state_noise_std_after_transition": 0.0316227766,
            "role": "recurrence_noise_sensitivity_only",
        },
    }:
        raise ValueError("baseline protocol-reference distinctions differ")
    if payload.get("public_code_audit_context") != {
        "pinned_commit": "cbd7404e9baca4b2dc291560cfc6576bb7b1f078",
        "runner_is_directly_executable_as_published": False,
        "documented_blockers": [
            "root_yaml_runner_missing_29_consumed_keys",
            "task_name_mismatch",
            "missing_imported_modules",
        ],
        "documented_training_conflicts": [
            "rnn_uniform_recurrent_bias_saturates_tanh",
            "model_specific_noise_weight_decay_and_learning_rate_disagree",
        ],
    }:
        raise ValueError("baseline public-code audit context differs")
    if payload.get("task") != {
        "name": "angular_integration",
        "horizon": 256,
        "dt": 0.1,
        "gp_length_scale": 1.0,
        "gp_std": 1.0,
        "gp_jitter": 1e-6,
        "input_process": "dense_gaussian_process",
        "initialization_mode": "hidden-init",
        "target_indexing": "q_t_plus_1_after_velocity_update",
        "loss_support": "all_256_steps",
    }:
        raise ValueError("baseline task contract differs")
    if payload.get("training") != {
        "optimizer": "Adam",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "target_noise_std": 0.0,
        "dropout": 0.0,
        "gradient_clip_norm": 0.0,
        "batch_size": 64,
        "updates": 5000,
        "state_noise_std_after_transition": 0.0316227766,
        "state_noise_location": {
            "rnn_tanh_n128": "h",
            "gru_n128": "h",
            "lstm_n64": "h_and_c",
        },
        "online_batches": True,
        "validation_interval": 500,
        "trace_interval": 50,
    }:
        raise ValueError("baseline training contract differs")
    models = payload.get("models")
    expected_models = [
        {
            "id": model_id,
            "width": WIDTHS[model_id],
            "parameter_count": PARAMETER_COUNTS[model_id],
            "initial_memory_map": (
                "linear_Wotr_q0"
                if model_id == "rnn_tanh_n128"
                else (
                    "tanh_Wotr_q0"
                    if model_id == "gru_n128"
                    else "independent_tanh_Wotr_h_q0_and_Wotr_c_q0"
                )
            ),
            "recurrence": (
                PRIMARY_RNN_VARIANT if model_id == "rnn_tanh_n128" else STANDARD_VARIANT
            ),
            "initialization": (
                "paper_xavier_io_gain1p5_recurrent_zero_bias"
                if model_id == "rnn_tanh_n128"
                else "xavier_normal_all_weights_zero_all_biases"
            ),
        }
        for model_id in MODEL_IDS
    ]
    if models != expected_models:
        raise ValueError("baseline model contract differs")
    if payload.get("learning_rate_screen") != {
        "grid": [0.01, 0.001, 0.0001, 0.00001],
        "seeds": [100, 101, 102, 103, 104],
        "updates_per_run": 5000,
        "selection_rule": (
            "max_success_count_then_min_median_mse_then_min_mean_mse_then_lower_lr"
        ),
        "success_mse_threshold": 0.01,
        "paper_selector_audit_update": 100,
        "paper_selector_audit_metric": "mean_online_training_mse_at_update_100",
        "paper_selector_audit_may_select_primary": False,
        "minimum_valid_completed_seeds_per_candidate": 5,
    }:
        raise ValueError("baseline LR-screen contract differs")
    if payload.get("main") != {
        "seeds": list(range(10)),
        "success_mse_threshold": 0.01,
        "analysis_nmse_db_threshold": -20.0,
        "minimum_analysis_eligible_seeds_for_scientific_pass": 1,
        "low_analysis_yield_warning_below": 8,
        "selection_source": "five_seed_full_length_lr_screen",
    }:
        raise ValueError("baseline main contract differs")
    if payload.get("evaluation_banks") != {
        "tuning": {
            "trials": 1024,
            "task_seed": 31001,
            "stream_key": ["sagodi_paper_baselines_v1", "fixed_tuning_bank"],
        },
        "main_test": {
            "trials": 1024,
            "task_seed": 31002,
            "stream_key": ["sagodi_paper_baselines_v1", "fixed_main_test_bank"],
        },
    }:
        raise ValueError("baseline evaluation-bank contract differs")
    if payload.get("sensitivity") != {
        "seeds": list(range(10)),
        "registered_conditions": [
            {
                "id": "pure_tanh_noise_0p1",
                "models": list(MODEL_IDS),
                "state_noise_std_after_transition": 0.1,
                "rnn_recurrence": PRIMARY_RNN_VARIANT,
                "role": "recurrence_noise_sensitivity_only",
            },
            {
                "id": "leaky_tanh_dt0p1_noise_0p1",
                "models": ["rnn_tanh_n128"],
                "state_noise_std_after_transition": 0.1,
                "rnn_recurrence": LEAKY_RNN_VARIANT,
                "role": "recurrence_noise_sensitivity_only",
            },
            {
                "id": "leaky_tanh_dt0p1_noise_0p03162",
                "models": ["rnn_tanh_n128"],
                "state_noise_std_after_transition": 0.0316227766,
                "rnn_recurrence": LEAKY_RNN_VARIANT,
                "role": "recurrence_noise_sensitivity_only",
            },
        ],
        "may_select_primary_hyperparameters": False,
    }:
        raise ValueError("baseline sensitivity contract differs")
    return payload


class PaperTanhRNN(nn.Module):
    """Zero-bias tanh RNN with paper initialization.

    Primary recurrence is the pure transition ``tanh(W_rec h + W_in x)``.
    The registered leaky variant is sensitivity-only and uses ``dt=0.1``.
    """

    def __init__(self, hidden: int, recurrence_variant: str) -> None:
        super().__init__()
        if recurrence_variant not in {PRIMARY_RNN_VARIANT, LEAKY_RNN_VARIANT}:
            raise ValueError("unsupported RNN recurrence variant")
        self.hidden = int(hidden)
        self.state_size = int(hidden)
        self.recurrence_variant = recurrence_variant
        self.wi = nn.Parameter(torch.empty(1, hidden))
        self.wrec = nn.Parameter(torch.empty(hidden, hidden))
        self.brec = nn.Parameter(torch.zeros(hidden))
        self.wo = nn.Parameter(torch.empty(hidden, 2))
        self.bo = nn.Parameter(torch.zeros(2))
        nn.init.xavier_normal_(self.wi)
        nn.init.normal_(self.wrec, mean=0.0, std=1.5 / math.sqrt(float(hidden)))
        nn.init.xavier_normal_(self.wo)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        proposal = torch.tanh(x_t @ self.wi + state @ self.wrec.t() + self.brec)
        if self.recurrence_variant == PRIMARY_RNN_VARIANT:
            return proposal
        return 0.9 * state + 0.1 * proposal

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return state @ self.wo + self.bo


def _xavier_zero_initialize_gated_core(core: nn.Module) -> None:
    """Apply the registered GRU/LSTM initialization without library defaults."""

    cell = getattr(core, "cell")
    readout = getattr(core, "readout")
    nn.init.xavier_normal_(cell.weight_ih)
    nn.init.xavier_normal_(cell.weight_hh)
    nn.init.zeros_(cell.bias_ih)
    nn.init.zeros_(cell.bias_hh)
    nn.init.xavier_normal_(readout.weight)
    nn.init.zeros_(readout.bias)


class BaselineModel(nn.Module):
    """Protocol-owned q0 maps and post-transition noise adapter."""

    def __init__(self, model_id: str, recurrence_variant: str) -> None:
        super().__init__()
        if model_id not in MODEL_IDS:
            raise ValueError(f"unknown baseline model {model_id!r}")
        self.model_id = model_id
        self.width = WIDTHS[model_id]
        self.recurrence_variant = recurrence_variant
        if model_id == "rnn_tanh_n128":
            self.core = PaperTanhRNN(self.width, recurrence_variant)
            self.initial_encoder = nn.Linear(2, self.width, bias=False)
            self.initial_encoder_h = None
            self.initial_encoder_c = None
            nn.init.xavier_normal_(self.initial_encoder.weight)
            self.state_size = self.width
        elif model_id == "gru_n128":
            if recurrence_variant != STANDARD_VARIANT:
                raise ValueError("GRU recurrence variant must be standard")
            self.core = build_exact_core(SAGODI_GRU, 1, 2, self.width)
            _xavier_zero_initialize_gated_core(self.core)
            self.initial_encoder = nn.Linear(2, self.width, bias=False)
            self.initial_encoder_h = None
            self.initial_encoder_c = None
            nn.init.xavier_normal_(self.initial_encoder.weight)
            self.state_size = self.width
        else:
            if recurrence_variant != STANDARD_VARIANT:
                raise ValueError("LSTM recurrence variant must be standard")
            self.core = build_exact_core(SAGODI_LSTM, 1, 2, self.width)
            _xavier_zero_initialize_gated_core(self.core)
            self.initial_encoder = None
            self.initial_encoder_h = nn.Linear(2, self.width, bias=False)
            self.initial_encoder_c = nn.Linear(2, self.width, bias=False)
            nn.init.xavier_normal_(self.initial_encoder_h.weight)
            nn.init.xavier_normal_(self.initial_encoder_c.weight)
            self.state_size = 2 * self.width

    def initial_state(self, initial_memory: torch.Tensor) -> torch.Tensor:
        if initial_memory.ndim != 2 or initial_memory.shape[-1] != 2:
            raise ValueError("initial_memory must be [batch,2] true q0 memory")
        if self.model_id == "lstm_n64":
            assert self.initial_encoder_h is not None and self.initial_encoder_c is not None
            return torch.cat(
                (
                    torch.tanh(self.initial_encoder_h(initial_memory)),
                    torch.tanh(self.initial_encoder_c(initial_memory)),
                ),
                dim=-1,
            )
        assert self.initial_encoder is not None
        state = self.initial_encoder(initial_memory)
        return torch.tanh(state) if self.model_id == "gru_n128" else state

    def forward_sequence(
        self,
        inputs: torch.Tensor,
        *,
        initial_memory: torch.Tensor,
        state_noise_std: float = 0.0,
        noise_generator: torch.Generator | None = None,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        state = self.initial_state(initial_memory)
        outputs: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for x_t in inputs:
            state = self.core.step(x_t, state)
            # This is the actual additive standard deviation after the full
            # deterministic transition.  No sqrt(dt) rescaling is applied.
            if float(state_noise_std) > 0.0:
                state = state + torch.randn(
                    state.shape,
                    dtype=state.dtype,
                    device=state.device,
                    generator=noise_generator,
                ) * float(state_noise_std)
            outputs.append(self.core.decode(state))
            if return_states:
                states.append(state)
        prediction = torch.stack(outputs)
        if return_states:
            return prediction, torch.stack(states)
        return prediction

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "width": self.width,
            "state_size": self.state_size,
            "recurrence_variant": self.recurrence_variant,
            "parameters_total": sum(p.numel() for p in self.parameters()),
            "rnn_recurrent_bias_initialized_zero": (
                True if self.model_id == "rnn_tanh_n128" else None
            ),
            "initial_memory_source": "true_pre_update_q0_cos_sin",
            "initial_memory_map": (
                "linear"
                if self.model_id == "rnn_tanh_n128"
                else "tanh" if self.model_id == "gru_n128" else "independent_tanh_h_c"
            ),
            "state_noise_application": "additive_after_full_transition",
            "state_noise_components": (
                "h_and_c" if self.model_id == "lstm_n64" else "h"
            ),
            "parameter_initialization": (
                "paper_xavier_io_gain1p5_recurrent_zero_bias"
                if self.model_id == "rnn_tanh_n128"
                else "xavier_normal_all_weights_zero_all_biases"
            ),
            "track_classification": "paper_and_source_informed_controlled_benchmark",
        }


def primary_variant(model_id: str) -> str:
    return PRIMARY_RNN_VARIANT if model_id == "rnn_tanh_n128" else STANDARD_VARIANT


def build_model(model_id: str, recurrence_variant: str | None = None) -> BaselineModel:
    variant = primary_variant(model_id) if recurrence_variant is None else recurrence_variant
    model = BaselineModel(model_id, variant)
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != PARAMETER_COUNTS[model_id]:
        raise RuntimeError(f"parameter count mismatch for {model_id}: {actual}")
    return model


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
    recurrence_variant: str
    evaluation_bank: str
    output_dir: str
    condition_id: str = "primary"
    smoke: bool = False

    def payload(self) -> dict[str, Any]:
        return _json_native(asdict(self))


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


def _assert_finite_model(model: nn.Module, *, gradients: bool = False) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise FloatingPointError(f"non-finite parameter: {name}")
        if gradients and parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError(f"non-finite gradient: {name}")


def _to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        inputs=batch.inputs.to(device),
        output_targets=batch.output_targets.to(device),
        latent_targets=batch.latent_targets.to(device),
        mask=batch.mask.to(device),
        metadata=batch.metadata,
    )


def _q0(batch: Batch, device: torch.device) -> torch.Tensor:
    if batch.initial_memory is None:
        raise ValueError("angular-integration batch omitted true q0")
    return batch.initial_memory.to(device)


def _training_batch(
    config: Mapping[str, Any], update: int, batch_size: int, device: torch.device
) -> Batch:
    task = config["task"]
    return angular_integration(
        int(batch_size),
        0,
        dimensions=1,
        init_mode=task["initialization_mode"],
        horizon=int(task["horizon"]),
        dt=float(task["dt"]),
        gp_length_scale=float(task["gp_length_scale"]),
        gp_std=float(task["gp_std"]),
        gp_jitter=float(task["gp_jitter"]),
        stream_key=(CAMPAIGN_ID, "online_train", int(update)),
        device=device,
    )


@torch.no_grad()
def _evaluate(model: BaselineModel, batch: Batch) -> dict[str, Any]:
    prediction = model.forward_sequence(batch.inputs, initial_memory=_q0(batch, batch.inputs.device))
    if not torch.isfinite(prediction).all().item():
        raise FloatingPointError("non-finite held-out prediction")
    metrics = task_metrics(
        prediction, batch.output_targets, batch.mask, batch.latent_targets
    )

    def finite(value: Any) -> bool:
        if isinstance(value, Mapping):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            return math.isfinite(float(value))
        return True

    if not finite(metrics):
        raise FloatingPointError("non-finite held-out metric")
    return _json_native(metrics)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: Path | str, device: torch.device | str = "cpu"
) -> tuple[BaselineModel, dict[str, Any]]:
    """Load a final campaign checkpoint without guessing model semantics."""

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("checkpoint_type") != CAMPAIGN_ID:
        raise ValueError("not a sagodi-paper-baselines-v1 checkpoint")
    metadata = payload.get("model", {})
    model = build_model(
        str(metadata["model_id"]), str(metadata["recurrence_variant"])
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    if model.metadata() != metadata:
        raise ValueError("checkpoint model metadata differs from executable model")
    return model, payload


def _runtime_files() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        MODULE_DIR / "exact_models.py",
        MODULE_DIR / "tasks.py",
        MODULE_DIR / "metrics.py",
        MODULE_DIR / "artifacts.py",
    )


def _git_state(require_clean: bool) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()
    if require_clean and status:
        raise RuntimeError("full baseline stages require a clean committed worktree")
    return {"code_commit": commit, "worktree_clean": not bool(status)}


def _scientific_identity(config_path: Path, *, require_clean: bool) -> dict[str, Any]:
    git = _git_state(require_clean)
    code_hashes = {path.name: sha256_file(path) for path in _runtime_files()}
    identity = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "source_config_sha256": sha256_file(config_path),
        "source_freeze_sha256": sha256_file(FREEZE_DOCUMENT),
        "runtime_code_sha256": code_hashes,
        "code_commit": git["code_commit"],
    }
    identity["scientific_identity"] = canonical_hash(identity)
    return identity


def _copy_or_verify(source: Path, destination: Path) -> None:
    payload = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError(f"immutable campaign input differs: {destination}")
        return
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
    root: Path, config_source: Path, *, require_clean: bool
) -> tuple[dict[str, Any], Path]:
    root = root.expanduser().resolve()
    config = load_config(config_source)
    identity = _scientific_identity(config_source, require_clean=require_clean)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".sagodi_paper_baselines_v1_root.json"
    if marker.exists():
        if strict_json_load(marker) != identity:
            raise RuntimeError("artifact root belongs to another scientific identity")
    else:
        if any(root.iterdir()):
            raise RuntimeError("unmarked artifact root must be empty")
        atomic_json(marker, identity)
    copied = root / "inputs" / DEFAULT_CONFIG.name
    _copy_or_verify(config_source, copied)
    _copy_or_verify(FREEZE_DOCUMENT, root / "inputs" / FREEZE_DOCUMENT.name)
    return config, copied


def _validate_bank_contract(
    batch: Batch,
    *,
    trials: int,
    task_seed: int,
    stream_key: Sequence[Any],
    config: Mapping[str, Any],
) -> None:
    """Validate semantic task provenance in addition to array shapes."""

    task = config["task"]
    metadata = metadata_payload(batch.metadata)
    expected = {
        "task_name": "angular_integration",
        "task_version": "sagodi-protocol-v1",
        "base_seed": int(task_seed),
        "stream_key": list(stream_key),
        "derived_seed": keyed_seed(
            int(task_seed),
            "angular_integration",
            "sagodi-protocol-v1",
            1,
            int(task["horizon"]),
            float(task["dt"]),
            float(task["gp_length_scale"]),
            float(task["gp_std"]),
            float(task["gp_jitter"]),
            *stream_key,
        ),
        "init_mode": task["initialization_mode"],
        "horizon": int(task["horizon"]),
        "loss_bearing_steps": int(task["horizon"]),
        "delta_t": float(task["dt"]),
        "latent_dimension": 1,
        "input_dimension": 1,
        "output_dimension": 2,
        "gp_grid": "linspace(-1,1,T)",
        "gp_grid_start": -1.0,
        "gp_grid_stop": 1.0,
        "gp_grid_endpoint": True,
        "gp_length_scale": float(task["gp_length_scale"]),
        "gp_std": float(task["gp_std"]),
        "gp_cholesky_jitter": float(task["gp_jitter"]),
        "q0_distribution": "uniform_minus_pi_pi",
        "q0_low": -math.pi,
        "q0_high": math.pi,
        "q0_high_inclusive": False,
        "target_indexing": "post_velocity_update",
        "target_indexing_contract": task["target_indexing"],
        "loss_mask_mode": "all_256_velocity_steps",
        "cue_token_is_loss_masked": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"fixed-bank metadata differs for {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )
    initial = metadata.get("initial_latents")
    if not isinstance(initial, list) or len(initial) != int(trials):
        raise RuntimeError("fixed-bank q0 metadata has the wrong trial count")
    if batch.batch_size != int(trials) or batch.time_steps != int(task["horizon"]):
        raise RuntimeError("fixed-bank tensor shape differs from task contract")
    if tuple(batch.inputs.shape[-1:]) != (1,) or tuple(batch.output_targets.shape[-1:]) != (2,):
        raise RuntimeError("fixed-bank feature dimensions differ from task contract")
    if not torch.all(batch.mask == 1).item():
        raise RuntimeError("fixed-bank mask is not dense all-step supervision")


def _bank_contract(
    config: Mapping[str, Any], purpose: str
) -> tuple[int, int, tuple[Any, ...]]:
    if purpose == "smoke":
        return 16, 31999, (CAMPAIGN_ID, "fixed_smoke_bank")
    elif purpose in {"tuning", "main_test"}:
        contract = config["evaluation_banks"][purpose]
        return (
            int(contract["trials"]),
            int(contract["task_seed"]),
            tuple(contract["stream_key"]),
        )
    raise ValueError("unknown bank purpose")


def _generate_registered_bank(config: Mapping[str, Any], purpose: str) -> Batch:
    trials, seed, stream_key = _bank_contract(config, purpose)
    task = config["task"]
    return angular_integration(
        trials,
        seed,
        dimensions=1,
        init_mode=task["initialization_mode"],
        horizon=int(task["horizon"]),
        dt=float(task["dt"]),
        gp_length_scale=float(task["gp_length_scale"]),
        gp_std=float(task["gp_std"]),
        gp_jitter=float(task["gp_jitter"]),
        stream_key=stream_key,
        device="cpu",
    )


def _load_and_validate_registered_bank(
    path: Path | str, config: Mapping[str, Any], purpose: str
) -> Batch:
    observed = load_fixed_bank(path)
    trials, seed, stream_key = _bank_contract(config, purpose)
    _validate_bank_contract(
        observed,
        trials=trials,
        task_seed=seed,
        stream_key=stream_key,
        config=config,
    )
    source = Path(path).resolve()
    registration = source.parent / f"{purpose}.registration.json"
    receipt_path = source.parent / f"{purpose}.registration_receipt.json"
    valid, _ = verify_completion_receipt(
        receipt_path,
        expected_job_id=f"{CAMPAIGN_ID}__bank__{purpose}",
        expected_metadata={"campaign_id": CAMPAIGN_ID, "purpose": purpose},
    )
    if not valid:
        raise RuntimeError("fixed-bank registered-digest receipt is invalid")
    payload = strict_json_load(registration)
    expected_payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "purpose": purpose,
        "bank_name": source.name,
        "bank_sha256": sha256_file(source),
        "sidecar_sha256": sha256_file(Path(f"{source}.sha256")),
        "task_contract_sha256": canonical_hash(config["task"]),
        "evaluation_contract_sha256": canonical_hash(
            {
                "trials": trials,
                "task_seed": seed,
                "stream_key": list(stream_key),
            }
        ),
    }
    if payload != expected_payload:
        raise RuntimeError("fixed-bank content differs from its registered digest")
    receipt = strict_json_load(receipt_path)
    if set(receipt.get("artifacts", {})) != {
        source.name,
        f"{source.name}.sha256",
        registration.name,
    }:
        raise RuntimeError("fixed-bank registration receipt artifact set differs")
    return observed


def _register_bank(path: Path, config: Mapping[str, Any], purpose: str) -> None:
    source = path.resolve()
    trials, seed, stream_key = _bank_contract(config, purpose)
    registration = source.parent / f"{purpose}.registration.json"
    receipt_path = source.parent / f"{purpose}.registration_receipt.json"
    payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "purpose": purpose,
        "bank_name": source.name,
        "bank_sha256": sha256_file(source),
        "sidecar_sha256": sha256_file(Path(f"{source}.sha256")),
        "task_contract_sha256": canonical_hash(config["task"]),
        "evaluation_contract_sha256": canonical_hash(
            {
                "trials": trials,
                "task_seed": seed,
                "stream_key": list(stream_key),
            }
        ),
    }
    _write_or_verify(registration, payload)
    if not receipt_path.exists():
        write_completion_receipt(
            receipt_path,
            job_id=f"{CAMPAIGN_ID}__bank__{purpose}",
            artifacts=[source, Path(f"{source}.sha256"), registration],
            metadata={"campaign_id": CAMPAIGN_ID, "purpose": purpose},
        )


def _ensure_bank(root: Path, config: Mapping[str, Any], purpose: str) -> Path:
    _bank_contract(config, purpose)
    path = root / "banks" / f"{purpose}.npz"
    if not path.exists():
        save_fixed_bank(path, _generate_registered_bank(config, purpose))
    observed = load_fixed_bank(path)
    trials, seed, stream_key = _bank_contract(config, purpose)
    _validate_bank_contract(
        observed,
        trials=trials,
        task_seed=seed,
        stream_key=stream_key,
        config=config,
    )
    _register_bank(path, config, purpose)
    _load_and_validate_registered_bank(path, config, purpose)
    return path


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
            state_noise_std=float(config["training"]["state_noise_std_after_transition"]),
            recurrence_variant=primary_variant(model_id),
            evaluation_bank=str(bank),
            output_dir=str(root / "smoke" / "runs" / model_id),
            smoke=True,
        )
        for model_id in MODEL_IDS
    )


def build_screen_plan(root: Path, config: Mapping[str, Any], bank: Path) -> tuple[RunSpec, ...]:
    screen = config["learning_rate_screen"]
    specs: list[RunSpec] = []
    for model_id in MODEL_IDS:
        for rate in screen["grid"]:
            for seed in screen["seeds"]:
                key = format(float(rate), ".0e").replace("-0", "-")
                run_id = f"screen__{model_id}__lr{key}__seed{seed}"
                specs.append(
                    RunSpec(
                        run_id=run_id,
                        stage="screen",
                        model_id=model_id,
                        model_seed=int(seed),
                        learning_rate=float(rate),
                        updates=int(screen["updates_per_run"]),
                        batch_size=int(config["training"]["batch_size"]),
                        state_noise_std=float(
                            config["training"]["state_noise_std_after_transition"]
                        ),
                        recurrence_variant=primary_variant(model_id),
                        evaluation_bank=str(bank),
                        output_dir=str(root / "screen" / "runs" / run_id),
                    )
                )
    return tuple(specs)


def _result(spec: RunSpec) -> dict[str, Any]:
    payload = strict_json_load(Path(spec.output_dir) / "result.json")
    if payload.get("run_id") != spec.run_id:
        raise RuntimeError("run/result identity mismatch")
    return payload


def _completed_metric(spec: RunSpec, key: str) -> float | None:
    payload = _result(spec)
    if payload.get("status", "completed") != "completed":
        return None
    metrics = payload.get("final_metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def paper_selector_audit(
    specs: Sequence[RunSpec], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Reconstruct the paper's 100-update selector as a diagnostic only."""

    audit_update = int(config["learning_rate_screen"]["paper_selector_audit_update"])
    models: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        rows: list[dict[str, Any]] = []
        for rate in config["learning_rate_screen"]["grid"]:
            matching = [
                spec
                for spec in specs
                if spec.model_id == model_id and spec.learning_rate == float(rate)
            ]
            values: list[float | None] = []
            for spec in matching:
                value: float | None = None
                trace_path = Path(spec.output_dir) / "training_trace.json"
                if trace_path.is_file():
                    trace = strict_json_load(Path(spec.output_dir) / "training_trace.json")
                    row = next(
                        (item for item in trace if int(item.get("update", -1)) == audit_update),
                        None,
                    )
                    if row is not None and math.isfinite(float(row["train_mse"])):
                        value = float(row["train_mse"])
                values.append(value)
            finite = [value for value in values if value is not None]
            rows.append(
                {
                    "learning_rate": float(rate),
                    "seed_count": len(values),
                    "available_seed_count": len(finite),
                    "mean_online_training_mse_at_update_100": (
                        float(np.mean(finite)) if len(finite) == len(values) else None
                    ),
                    "per_seed_online_training_mse": values,
                }
            )
        winner = min(
            rows,
            key=lambda row: (
                math.inf
                if row["mean_online_training_mse_at_update_100"] is None
                else float(row["mean_online_training_mse_at_update_100"]),
                float(row["learning_rate"]),
            ),
        )
        models[model_id] = {"candidates": rows, "diagnostic_winner": winner}
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "audit_role": "paper_100_update_selector_diagnostic_only",
        "audit_update": audit_update,
        "may_select_primary_hyperparameters": False,
        "full_length_selector_is_separate": True,
        "models": models,
    }


def select_learning_rates(
    specs: Sequence[RunSpec], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Select from all five full runs; no sentinel or single-seed gate."""

    threshold = float(config["learning_rate_screen"]["success_mse_threshold"])
    required_valid = int(
        config["learning_rate_screen"]["minimum_valid_completed_seeds_per_candidate"]
    )
    selected: dict[str, float | None] = {}
    audit: dict[str, Any] = {}
    unavailable: list[str] = []
    for model_id in MODEL_IDS:
        rows: list[dict[str, Any]] = []
        for rate in config["learning_rate_screen"]["grid"]:
            matching = [
                spec
                for spec in specs
                if spec.model_id == model_id and spec.learning_rate == float(rate)
            ]
            if len(matching) != 5:
                raise RuntimeError(f"{model_id} LR {rate} does not have five runs")
            values = [_completed_metric(spec, "masked_mse") for spec in matching]
            finite = [value for value in values if value is not None]
            rows.append(
                {
                    "learning_rate": float(rate),
                    "seed_count": len(values),
                    "completed_seed_count": len(finite),
                    "failed_seed_count": len(values) - len(finite),
                    "selection_eligible": len(finite) >= required_valid,
                    "success_count": sum(value < threshold for value in finite),
                    "median_mse": float(np.median(finite)) if finite else None,
                    "mean_mse": float(np.mean(finite)) if finite else None,
                    "per_seed_mse": values,
                }
            )
        eligible_rows = [row for row in rows if row["selection_eligible"]]
        winner = min(
            eligible_rows,
            key=lambda row: (
                -int(row["success_count"]),
                math.inf if row["median_mse"] is None else float(row["median_mse"]),
                math.inf if row["mean_mse"] is None else float(row["mean_mse"]),
                float(row["learning_rate"]),
            ),
        ) if eligible_rows else None
        selected[model_id] = (
            float(winner["learning_rate"]) if winner is not None else None
        )
        if winner is None:
            unavailable.append(model_id)
        audit[model_id] = {"candidates": rows, "selected": winner}
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "selection_uses_all_five_seeds": True,
        "minimum_valid_completed_seeds_per_candidate": required_valid,
        "single_seed_gate_used": False,
        "selection_available": not unavailable,
        "unavailable_models": unavailable,
        "selected_learning_rates": selected,
        "models": audit,
    }


def build_main_plan(
    root: Path, config: Mapping[str, Any], bank: Path, selection: Mapping[str, Any]
) -> tuple[RunSpec, ...]:
    rates = selection["selected_learning_rates"]
    if selection.get("selection_available") is not True:
        raise RuntimeError("main LR selection is unavailable")
    if set(rates) != set(MODEL_IDS) or any(rates[model_id] is None for model_id in MODEL_IDS):
        raise RuntimeError("main LR selection is incomplete")
    specs: list[RunSpec] = []
    for model_id in MODEL_IDS:
        for seed in config["main"]["seeds"]:
            run_id = f"main__{model_id}__seed{seed}"
            specs.append(
                RunSpec(
                    run_id=run_id,
                    stage="main",
                    model_id=model_id,
                    model_seed=int(seed),
                    learning_rate=float(rates[model_id]),
                    updates=int(config["training"]["updates"]),
                    batch_size=int(config["training"]["batch_size"]),
                    state_noise_std=float(
                        config["training"]["state_noise_std_after_transition"]
                    ),
                    recurrence_variant=primary_variant(model_id),
                    evaluation_bank=str(bank),
                    output_dir=str(root / "main" / "runs" / run_id),
                )
            )
    return tuple(specs)


def build_sensitivity_plan(
    root: Path, config: Mapping[str, Any], bank: Path, selection: Mapping[str, Any]
) -> tuple[RunSpec, ...]:
    rates = selection["selected_learning_rates"]
    if selection.get("selection_available") is not True:
        raise RuntimeError("sensitivity LR selection is unavailable")
    specs: list[RunSpec] = []
    for condition in config["sensitivity"]["registered_conditions"]:
        for model_id in condition["models"]:
            for seed in config["sensitivity"]["seeds"]:
                run_id = f"sensitivity__{condition['id']}__{model_id}__seed{seed}"
                recurrence = (
                    str(condition["rnn_recurrence"])
                    if model_id == "rnn_tanh_n128"
                    else STANDARD_VARIANT
                )
                specs.append(
                    RunSpec(
                        run_id=run_id,
                        stage="sensitivity",
                        model_id=model_id,
                        model_seed=int(seed),
                        learning_rate=float(rates[model_id]),
                        updates=int(config["training"]["updates"]),
                        batch_size=int(config["training"]["batch_size"]),
                        state_noise_std=float(condition["state_noise_std_after_transition"]),
                        recurrence_variant=recurrence,
                        evaluation_bank=str(bank),
                        output_dir=str(root / "sensitivity" / "runs" / run_id),
                        condition_id=str(condition["id"]),
                    )
                )
    return tuple(specs)


def _validate_spec(config: Mapping[str, Any], spec: RunSpec) -> None:
    """Reject a worker spec that falls outside its registered stage."""

    if spec.model_id not in MODEL_IDS:
        raise ValueError("worker spec has an unregistered model")
    primary_noise = float(config["training"]["state_noise_std_after_transition"])
    grid = {float(value) for value in config["learning_rate_screen"]["grid"]}
    common_full = (
        spec.updates == int(config["training"]["updates"])
        and spec.batch_size == int(config["training"]["batch_size"])
        and spec.learning_rate in grid
        and not spec.smoke
    )
    if spec.stage == "smoke":
        valid = (
            spec.smoke
            and spec.model_seed == 999
            and spec.updates == 2
            and spec.batch_size == 4
            and spec.learning_rate == 1e-3
            and spec.condition_id == "primary"
            and spec.state_noise_std == primary_noise
            and spec.recurrence_variant == primary_variant(spec.model_id)
        )
    elif spec.stage == "screen":
        valid = (
            common_full
            and spec.model_seed in config["learning_rate_screen"]["seeds"]
            and spec.condition_id == "primary"
            and spec.state_noise_std == primary_noise
            and spec.recurrence_variant == primary_variant(spec.model_id)
        )
    elif spec.stage == "main":
        valid = (
            common_full
            and spec.model_seed in config["main"]["seeds"]
            and spec.condition_id == "primary"
            and spec.state_noise_std == primary_noise
            and spec.recurrence_variant == primary_variant(spec.model_id)
        )
    elif spec.stage == "sensitivity":
        condition = next(
            (
                item
                for item in config["sensitivity"]["registered_conditions"]
                if item["id"] == spec.condition_id
            ),
            None,
        )
        expected_variant = None
        if condition is not None and spec.model_id in condition["models"]:
            expected_variant = (
                str(condition["rnn_recurrence"])
                if spec.model_id == "rnn_tanh_n128"
                else STANDARD_VARIANT
            )
        valid = (
            common_full
            and spec.model_seed in config["sensitivity"]["seeds"]
            and condition is not None
            and spec.model_id in condition["models"]
            and spec.state_noise_std == float(condition["state_noise_std_after_transition"])
            and spec.recurrence_variant == expected_variant
        )
    else:
        valid = False
    if not valid:
        raise ValueError(f"worker spec violates the frozen {spec.stage!r} contract")


def summarize_main(specs: Sequence[RunSpec], config: Mapping[str, Any]) -> dict[str, Any]:
    threshold = float(config["main"]["success_mse_threshold"])
    nmse_threshold = float(config["main"]["analysis_nmse_db_threshold"])
    nmse_required = int(
        config["main"]["minimum_analysis_eligible_seeds_for_scientific_pass"]
    )
    warning_below = int(config["main"]["low_analysis_yield_warning_below"])
    models: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        matching = [spec for spec in specs if spec.model_id == model_id]
        mse_values = [_completed_metric(spec, "masked_mse") for spec in matching]
        nmse_values = [_completed_metric(spec, "masked_nmse_db") for spec in matching]
        finite_mse = [value for value in mse_values if value is not None]
        successes = sum(value < threshold for value in finite_mse)
        eligible = sum(
            value < nmse_threshold for value in nmse_values if value is not None
        )
        models[model_id] = {
            "seed_count": len(matching),
            "completed_seed_count": len(finite_mse),
            "failed_seed_count": len(matching) - len(finite_mse),
            "successful_seed_count": successes,
            "success_mse_threshold": threshold,
            "descriptive_mse_success_rate": successes / len(matching),
            "mse_success_is_descriptive_not_a_hard_gate": True,
            "analysis_eligible_seed_count": eligible,
            "analysis_eligibility_yield": eligible / len(matching),
            "minimum_analysis_eligible_seeds_for_scientific_pass": nmse_required,
            "analysis_nmse_db_threshold": nmse_threshold,
            "analysis_eligibility_gate_pass": eligible >= nmse_required,
            "low_analysis_yield_warning": 0 < eligible < warning_below,
            "zero_analysis_eligible_hard_fail": eligible == 0,
            "scientific_pass": eligible >= nmse_required,
            "mean_mse": float(np.mean(finite_mse)) if finite_mse else None,
            "median_mse": float(np.median(finite_mse)) if finite_mse else None,
            "population_std_mse": (
                float(np.std(finite_mse, ddof=0)) if finite_mse else None
            ),
            "per_seed_mse": mse_values,
            "per_seed_masked_nmse_db": nmse_values,
        }
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "track_classification": "paper_and_source_informed_controlled_benchmark",
        "single_seed_gate_used": False,
        "mse_reporting": "descriptive_all_10_seed_success_count_and_yield",
        "analysis_eligibility_criterion": (
            "seed_level_final_heldout_masked_nmse_db_below_minus_20"
        ),
        "scientific_hard_fail_criterion": (
            "zero_analysis_eligible_seeds_for_any_model"
        ),
        "low_yield_warning_below": warning_below,
        "any_low_analysis_yield_warning": any(
            row["low_analysis_yield_warning"] for row in models.values()
        ),
        "all_models_scientific_pass": all(
            row["scientific_pass"] for row in models.values()
        ),
        "models": models,
    }


def scientific_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    models = summary["models"]
    passed = bool(summary["all_models_scientific_pass"])
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "track_classification": "paper_and_source_informed_controlled_benchmark",
        "computation_complete_is_not_scientific_pass": True,
        "required_gates": [
            "at_least_one_nmse_below_minus_20_analysis_eligible_seed_per_model",
        ],
        "mse_below_0.01_is_descriptive_not_a_hard_gate": True,
        "low_analysis_yield_is_warning_not_hard_fail": True,
        "per_model": {
            model_id: {
                "analysis_eligibility_gate_pass": bool(
                    models[model_id]["analysis_eligibility_gate_pass"]
                ),
                "analysis_eligible_seed_count": int(
                    models[model_id]["analysis_eligible_seed_count"]
                ),
                "low_analysis_yield_warning": bool(
                    models[model_id]["low_analysis_yield_warning"]
                ),
                "scientific_pass": bool(models[model_id]["scientific_pass"]),
            }
            for model_id in MODEL_IDS
        },
        "all_required_gates_pass": passed,
        "downstream_ca_comparison_authorized": passed,
    }


def summarize_sensitivity(specs: Sequence[RunSpec]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for condition_id in sorted({spec.condition_id for spec in specs}):
        groups[condition_id] = {}
        for model_id in MODEL_IDS:
            matching = [
                spec for spec in specs if spec.condition_id == condition_id and spec.model_id == model_id
            ]
            if not matching:
                continue
            values = [_completed_metric(spec, "masked_mse") for spec in matching]
            finite = [value for value in values if value is not None]
            groups[condition_id][model_id] = {
                "seed_count": len(values),
                "completed_seed_count": len(finite),
                "failed_seed_count": len(values) - len(finite),
                "mean_mse": float(np.mean(finite)) if finite else None,
                "median_mse": float(np.median(finite)) if finite else None,
                "per_seed_mse": values,
            }
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "primary_hyperparameters_changed": False,
        "sensitivity_conditions": groups,
    }


def _train_worker(spec: RunSpec, config_path: Path, device_text: str) -> Path:
    config = load_config(config_path)
    _validate_spec(config, spec)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(spec.evaluation_bank).resolve().parents[1]
    identity = strict_json_load(root / ".sagodi_paper_baselines_v1_root.json")
    if _scientific_identity(config_path, require_clean=not spec.smoke) != identity:
        raise RuntimeError("worker scientific identity differs from campaign root")
    output = Path(spec.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"worker output is not empty: {output}")
    device = torch.device(device_text)
    _configure_determinism(spec.model_seed)
    model = build_model(spec.model_id, spec.recurrence_variant).to(device)
    _assert_finite_model(model)
    bank_purpose = Path(spec.evaluation_bank).stem
    if bank_purpose not in {"smoke", "tuning", "main_test"}:
        raise RuntimeError("worker evaluation bank has an unregistered purpose")
    bank = _to_device(
        _load_and_validate_registered_bank(spec.evaluation_bank, config, bank_purpose),
        device,
    )
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run": spec.payload(),
        "model": model.metadata(),
        "scientific_identity": identity["scientific_identity"],
        "runtime_code_sha256": identity["runtime_code_sha256"],
        "config_sha256": sha256_file(config_path),
        "evaluation_bank_sha256": sha256_file(spec.evaluation_bank),
        "bank_registration_sha256": sha256_file(
            Path(spec.evaluation_bank).parent
            / f"{bank_purpose}.registration.json"
        ),
        "bank_registration_receipt_sha256": sha256_file(
            Path(spec.evaluation_bank).parent
            / f"{bank_purpose}.registration_receipt.json"
        ),
        "optimizer": config["training"],
        "loss": "all_step_clean_target_mse",
        "track_classification": config["track_classification"],
        "cpu_thread_contract": {
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "blas_environment_threads": 1,
        },
        "started_at_utc": _utc_now(),
        "device": device_text,
    }
    atomic_json(output / "run_manifest.json", manifest)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(spec.learning_rate),
        betas=tuple(config["training"]["betas"]),
        eps=float(config["training"]["epsilon"]),
        weight_decay=0.0,
    )
    noise_generator = torch.Generator(device=device.type)
    noise_generator.manual_seed(derived_seed(spec.model_seed, CAMPAIGN_ID, "state_noise"))
    trace: list[dict[str, Any]] = []
    best_mse = math.inf
    best_update: int | None = None
    started = time.time()
    trace_interval = int(config["training"]["trace_interval"])
    validation_interval = int(config["training"]["validation_interval"])
    for update in range(1, spec.updates + 1):
        model.train()
        batch = _training_batch(config, update, spec.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_sequence(
            batch.inputs,
            initial_memory=_q0(batch, device),
            state_noise_std=spec.state_noise_std,
            noise_generator=noise_generator,
        )
        loss = masked_mse(prediction, batch.output_targets, batch.mask)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()
        _assert_finite_model(model, gradients=True)
        # No gradient clipping: gradient_clip_norm=0 is the frozen disabled value.
        optimizer.step()
        _assert_finite_model(model)
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
            if float(validation["masked_mse"]) < best_mse:
                best_mse = float(validation["masked_mse"])
                best_update = update
        if row is not None:
            trace.append(row)
            atomic_json(output / "training_trace.json", trace)
    model.eval()
    final_metrics = _evaluate(model, bank)
    result = {
        "schema_version": 1,
        "status": "completed",
        "run_id": spec.run_id,
        "stage": spec.stage,
        "condition_id": spec.condition_id,
        "model_id": spec.model_id,
        "model_seed": spec.model_seed,
        "learning_rate": spec.learning_rate,
        "updates_completed": spec.updates,
        "final_metrics": final_metrics,
        "best_validation_mse_diagnostic": best_mse,
        "best_validation_update_diagnostic": best_update,
        "final_checkpoint_is_primary": True,
        "completed_at_utc": _utc_now(),
    }
    atomic_json(output / "training_trace.json", trace)
    atomic_json(output / "result.json", result)
    _atomic_torch_save(
        output / "checkpoint_final.pt",
        {
            "schema_version": 1,
            "checkpoint_type": CAMPAIGN_ID,
            "model": model.metadata(),
            "state_dict": model.state_dict(),
            "run": spec.payload(),
            "result": result,
        },
    )
    atomic_json(
        output / "COMPLETE",
        {"schema_version": 1, "status": "complete", "run_id": spec.run_id},
    )
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=spec.run_id,
        artifacts=[
            output / "run_manifest.json",
            output / "training_trace.json",
            output / "result.json",
            output / "checkpoint_final.pt",
            output / "COMPLETE",
        ],
        metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "condition_id": spec.condition_id,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
        },
    )
    return output


def _record_scientific_failure(
    spec: RunSpec,
    config_path: Path,
    device_text: str,
    *,
    failure_kind: str,
    failure_message: str,
    traceback_text: str | None,
) -> Path:
    """Record only a scientific numerical/non-finite outcome in denominators."""

    config = load_config(config_path)
    output = Path(spec.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(spec.evaluation_bank).resolve().parents[1]
    identity = strict_json_load(root / ".sagodi_paper_baselines_v1_root.json")
    manifest_path = output / "run_manifest.json"
    if not manifest_path.exists():
        atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "run": spec.payload(),
                "model": None,
                "scientific_identity": identity["scientific_identity"],
                "runtime_code_sha256": identity["runtime_code_sha256"],
                "config_sha256": sha256_file(config_path),
                "evaluation_bank_sha256": sha256_file(spec.evaluation_bank),
                "bank_registration_sha256": sha256_file(
                    Path(spec.evaluation_bank).parent
                    / f"{Path(spec.evaluation_bank).stem}.registration.json"
                ),
                "bank_registration_receipt_sha256": sha256_file(
                    Path(spec.evaluation_bank).parent
                    / f"{Path(spec.evaluation_bank).stem}.registration_receipt.json"
                ),
                "optimizer": config["training"],
                "loss": "all_step_clean_target_mse",
                "track_classification": config["track_classification"],
                "started_at_utc": None,
                "device": device_text,
                "failure_manifest_synthesized": True,
            },
        )
    failure = {
        "schema_version": 1,
        "status": "failed",
        "run_id": spec.run_id,
        "failure_kind": str(failure_kind),
        "failure_class": "scientific_numerical_nonfinite",
        "failure_message": str(failure_message),
        "traceback": traceback_text,
        "counts_as_failure_in_all_aggregate_denominators": True,
        "recorded_at_utc": _utc_now(),
    }
    atomic_json(output / "failure.json", failure)
    result = {
        "schema_version": 1,
        "status": "failed",
        "run_id": spec.run_id,
        "stage": spec.stage,
        "condition_id": spec.condition_id,
        "model_id": spec.model_id,
        "model_seed": spec.model_seed,
        "learning_rate": spec.learning_rate,
        "updates_completed": None,
        "final_metrics": None,
        "failure_kind": str(failure_kind),
        "failure_class": "scientific_numerical_nonfinite",
        "counts_in_denominator": True,
        "completed_at_utc": _utc_now(),
    }
    atomic_json(output / "result.json", result)
    _atomic_torch_save(
        output / "checkpoint_failure.pt",
        {
            "schema_version": 1,
            "checkpoint_type": f"{CAMPAIGN_ID}_failure",
            "run": spec.payload(),
            "failure": failure,
            "state_dict": None,
        },
    )
    atomic_json(
        output / "FAILED",
        {
            "schema_version": 1,
            "status": "failed",
            "run_id": spec.run_id,
            "counts_in_denominator": True,
        },
    )
    artifacts = [
        manifest_path,
        output / "failure.json",
        output / "result.json",
        output / "checkpoint_failure.pt",
        output / "FAILED",
    ]
    trace_path = output / "training_trace.json"
    if not trace_path.is_file():
        atomic_json(trace_path, [])
    artifacts.append(trace_path)
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=spec.run_id,
        artifacts=artifacts,
        metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "condition_id": spec.condition_id,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
            "outcome": "failed",
            "failure_class": "scientific_numerical_nonfinite",
        },
    )
    return output


def _verified_child(spec: RunSpec) -> bool:
    output = Path(spec.output_dir)
    valid, _ = verify_completion_receipt(
        output / "completion_receipt.json",
        expected_job_id=spec.run_id,
        expected_metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "condition_id": spec.condition_id,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
        },
    )
    if not valid:
        return False
    try:
        receipt = strict_json_load(output / "completion_receipt.json")
        manifest = strict_json_load(output / "run_manifest.json")
        result = strict_json_load(output / "result.json")
        root = Path(spec.evaluation_bank).resolve().parents[1]
        identity = strict_json_load(root / ".sagodi_paper_baselines_v1_root.json")
    except (OSError, ValueError):
        return False
    status = result.get("status", "completed")
    if status == "completed":
        expected_receipt_metadata = {
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "condition_id": spec.condition_id,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
        }
        expected_artifacts = {
            "COMPLETE",
            "checkpoint_final.pt",
            "result.json",
            "run_manifest.json",
            "training_trace.json",
        }
        outcome_files_valid = (
            (output / "COMPLETE").is_file()
            and (output / "checkpoint_final.pt").is_file()
        )
        try:
            _, checkpoint = load_checkpoint(output / "checkpoint_final.pt")
            checkpoint_valid = (
                checkpoint.get("run") == spec.payload()
                and checkpoint.get("result") == result
            )
        except Exception:
            checkpoint_valid = False
    elif status == "failed":
        expected_receipt_metadata = {
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "condition_id": spec.condition_id,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
            "outcome": "failed",
            "failure_class": "scientific_numerical_nonfinite",
        }
        expected_artifacts = {
            "FAILED",
            "checkpoint_failure.pt",
            "failure.json",
            "result.json",
            "run_manifest.json",
            "training_trace.json",
        }
        outcome_files_valid = (
            (output / "FAILED").is_file()
            and (output / "failure.json").is_file()
            and (output / "checkpoint_failure.pt").is_file()
            and result.get("counts_in_denominator") is True
            and result.get("failure_class") == "scientific_numerical_nonfinite"
        )
        try:
            failure = strict_json_load(output / "failure.json")
            checkpoint = torch.load(
                output / "checkpoint_failure.pt", map_location="cpu", weights_only=False
            )
            checkpoint_valid = (
                checkpoint.get("checkpoint_type") == f"{CAMPAIGN_ID}_failure"
                and checkpoint.get("run") == spec.payload()
                and checkpoint.get("failure") == failure
                and checkpoint.get("state_dict") is None
            )
        except Exception:
            checkpoint_valid = False
    else:
        return False
    artifact_set_valid = set(receipt.get("artifacts", {})) == expected_artifacts
    receipt_metadata_valid = receipt.get("metadata") == expected_receipt_metadata
    return (
        outcome_files_valid
        and artifact_set_valid
        and receipt_metadata_valid
        and checkpoint_valid
        and manifest.get("run") == spec.payload()
        and result.get("run_id") == spec.run_id
        and manifest.get("scientific_identity") == identity.get("scientific_identity")
        and manifest.get("evaluation_bank_sha256") == sha256_file(spec.evaluation_bank)
        and manifest.get("bank_registration_sha256")
        == sha256_file(
            Path(spec.evaluation_bank).parent
            / f"{Path(spec.evaluation_bank).stem}.registration.json"
        )
        and manifest.get("bank_registration_receipt_sha256")
        == sha256_file(
            Path(spec.evaluation_bank).parent
            / f"{Path(spec.evaluation_bank).stem}.registration_receipt.json"
        )
    )


def _write_or_verify(path: Path, payload: Mapping[str, Any]) -> None:
    native = _json_native(payload)
    if path.exists():
        if strict_json_load(path) != native:
            raise RuntimeError(f"immutable artifact differs: {path}")
    else:
        atomic_json(path, native)


def _write_plan(stage_root: Path, specs: Sequence[RunSpec]) -> None:
    _write_or_verify(
        stage_root / "plan.json",
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "run_count": len(specs),
            "runs": [spec.payload() for spec in specs],
        },
    )


def _parse_slots(text: str) -> tuple[str, ...]:
    if text.strip().lower() == "cpu":
        return ("cpu",)
    slots = tuple(item.strip() for item in text.split(",") if item.strip())
    if not slots or any(not item.isdigit() for item in slots) or len(set(slots)) != len(slots):
        raise ValueError("--gpus must be unique comma-separated ids or cpu")
    return slots


def _archive_partial(output: Path, attempts: Path) -> None:
    if not output.exists():
        return
    attempts.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    os.replace(output, attempts / f"{output.name}.{stamp}")


def _run_specs(specs: Sequence[RunSpec], config_path: Path, slots: Sequence[str]) -> None:
    if not specs:
        return
    stage_root = Path(specs[0].output_dir).parents[1]
    _write_plan(stage_root, specs)
    pending = [spec for spec in specs if not _verified_child(spec)]
    for spec in pending:
        _archive_partial(Path(spec.output_dir), stage_root / "attempts")
    specs_dir, logs_dir = stage_root / "specs", stage_root / "logs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    queue = list(pending)
    running: dict[str, tuple[subprocess.Popen[Any], RunSpec, Any]] = {}
    repo = Path(__file__).resolve().parents[2]
    while queue or running:
        for slot in [item for item in slots if item not in running]:
            if not queue:
                break
            spec = queue.pop(0)
            spec_path = specs_dir / f"{spec.run_id}.json"
            atomic_json(spec_path, spec.payload())
            handle = (logs_dir / f"{spec.run_id}.log").open("ab")
            command = [
                sys.executable,
                "-m",
                "repro.sagodi_protocol.sagodi_paper_baselines",
                "--worker-spec",
                str(spec_path),
                "--config",
                str(config_path),
                "--device",
                "cpu" if slot == "cpu" else "cuda:0",
            ]
            environment = os.environ.copy()
            for variable in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            ):
                environment[variable] = "1"
            if slot != "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = slot
            process = subprocess.Popen(
                command,
                cwd=repo,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[slot] = (process, spec, handle)
        time.sleep(0.2)
        for slot, (process, spec, handle) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del running[slot]
            if code != 0 or not _verified_child(spec):
                for other, _, other_handle in running.values():
                    other.terminate()
                    other_handle.close()
                raise RuntimeError(
                    f"retryable infrastructure/process child failure "
                    f"(exit={code}): {spec.run_id}"
                )
        atomic_json(
            stage_root / "status.json",
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "registered": len(specs),
                "verified_complete": sum(_verified_child(spec) for spec in specs),
                "pending": len(queue),
                "running": [item[1].run_id for item in running.values()],
                "updated_at_utc": _utc_now(),
            },
        )


def _children_binding(specs: Sequence[RunSpec]) -> dict[str, Any]:
    children: list[dict[str, Any]] = []
    for spec in specs:
        output = Path(spec.output_dir)
        result = strict_json_load(output / "result.json")
        status = str(result.get("status", "completed"))
        checkpoint = (
            output / "checkpoint_final.pt"
            if status == "completed"
            else output / "checkpoint_failure.pt"
        )
        children.append(
            {
                "run_id": spec.run_id,
                "status": status,
                "completion_receipt_sha256": sha256_file(
                    output / "completion_receipt.json"
                ),
                "checkpoint_name": checkpoint.name,
                "checkpoint_sha256": sha256_file(checkpoint),
                "result_sha256": sha256_file(output / "result.json"),
            }
        )
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "transitively_binds_every_child_receipt_and_checkpoint": True,
        "child_count": len(children),
        "children": children,
    }


def _smoke_summary(specs: Sequence[RunSpec]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "scientific_result": False,
        "models": {spec.model_id: _result(spec)["final_metrics"] for spec in specs},
    }


def _parent_screen_binding(root: Path) -> dict[str, Any]:
    screen_root = root / "screen"
    return {
        "schema_version": 1,
        "screen_completion_receipt_sha256": sha256_file(
            screen_root / "completion_receipt.json"
        ),
        "lr_selection_sha256": sha256_file(screen_root / "lr_selection.json"),
    }


def _parent_screen_binding_valid(root: Path, stage_root: Path) -> bool:
    try:
        return strict_json_load(stage_root / "parent_screen_binding.json") == (
            _parent_screen_binding(root)
        )
    except (OSError, ValueError):
        return False


def _finalize(stage_root: Path, stage: str, specs: Sequence[RunSpec], extras: Sequence[Path]) -> None:
    missing = [spec.run_id for spec in specs if not _verified_child(spec)]
    if missing:
        raise RuntimeError(f"cannot finalize {stage}; missing {missing[:3]}")
    children_binding = stage_root / "children_binding.json"
    _write_or_verify(children_binding, _children_binding(specs))
    complete = stage_root / "COMPUTATION_COMPLETE"
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
    write_completion_receipt(
        stage_root / "completion_receipt.json",
        job_id=f"{CAMPAIGN_ID}__{stage}",
        artifacts=[stage_root / "plan.json", children_binding, complete, *extras],
        metadata={"campaign_id": CAMPAIGN_ID, "stage": stage, "run_count": len(specs)},
    )


def _stage_valid(stage_root: Path, stage: str, expected_runs: int) -> bool:
    valid, _ = verify_completion_receipt(
        stage_root / "completion_receipt.json",
        expected_job_id=f"{CAMPAIGN_ID}__{stage}",
        expected_metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": stage,
            "run_count": expected_runs,
        },
    )
    if not valid or not (stage_root / "COMPUTATION_COMPLETE").is_file():
        return False
    try:
        receipt = strict_json_load(stage_root / "completion_receipt.json")
        plan = strict_json_load(stage_root / "plan.json")
        specs = [RunSpec(**item) for item in plan["runs"]]
    except (OSError, KeyError, TypeError, ValueError):
        return False
    if len(specs) != expected_runs or not all(_verified_child(spec) for spec in specs):
        return False
    try:
        observed_binding = strict_json_load(stage_root / "children_binding.json")
        expected_binding = _children_binding(specs)
    except (OSError, ValueError):
        return False
    if observed_binding != expected_binding:
        return False
    try:
        root = stage_root.parent
        config = load_config(root / "inputs" / DEFAULT_CONFIG.name)
        bank_paths = {Path(spec.evaluation_bank).resolve() for spec in specs}
        if len(bank_paths) != 1:
            return False
        bank = next(iter(bank_paths))
        purpose = bank.stem
        _load_and_validate_registered_bank(bank, config, purpose)
        expected_artifacts = {
            "COMPUTATION_COMPLETE",
            "children_binding.json",
            "plan.json",
        }
        if stage == "smoke":
            expected_summary = _smoke_summary(specs)
            if strict_json_load(stage_root / "summary.json") != expected_summary:
                return False
            expected_artifacts.add("summary.json")
        elif stage == "screen":
            expected_selection = select_learning_rates(specs, config)
            expected_audit = paper_selector_audit(specs, config)
            if strict_json_load(stage_root / "lr_selection.json") != expected_selection:
                return False
            if (
                strict_json_load(stage_root / "paper_100_update_selector_audit.json")
                != expected_audit
            ):
                return False
            expected_artifacts.update(
                {"lr_selection.json", "paper_100_update_selector_audit.json"}
            )
        elif stage in {"main", "sensitivity"}:
            screen_root = root / "screen"
            expected_screen_runs = len(MODEL_IDS) * 4 * 5
            if not _stage_valid(screen_root, "screen", expected_screen_runs):
                return False
            selection = strict_json_load(screen_root / "lr_selection.json")
            if not _parent_screen_binding_valid(root, stage_root):
                return False
            expected_plan = (
                build_main_plan(root, config, bank, selection)
                if stage == "main"
                else build_sensitivity_plan(root, config, bank, selection)
            )
            if [spec.payload() for spec in specs] != [
                spec.payload() for spec in expected_plan
            ]:
                return False
            expected_summary = (
                summarize_main(specs, config)
                if stage == "main"
                else summarize_sensitivity(specs)
            )
            if strict_json_load(stage_root / "summary.json") != expected_summary:
                return False
            expected_artifacts.update({"parent_screen_binding.json", "summary.json"})
            if stage == "main":
                expected_gate = scientific_gate(expected_summary)
                if strict_json_load(stage_root / "scientific_gate.json") != expected_gate:
                    return False
                expected_artifacts.add("scientific_gate.json")
                if bool(expected_gate["all_required_gates_pass"]):
                    expected_artifacts.add("SCIENTIFIC_PASS")
        else:
            return False
        if set(receipt.get("artifacts", {})) != expected_artifacts:
            return False
    except (OSError, KeyError, RuntimeError, TypeError, ValueError):
        return False
    if stage == "main":
        try:
            gate = strict_json_load(stage_root / "scientific_gate.json")
        except (OSError, ValueError):
            return False
        pass_marker = stage_root / "SCIENTIFIC_PASS"
        if bool(gate.get("all_required_gates_pass")) != pass_marker.is_file():
            return False
        if pass_marker.is_file():
            marker = strict_json_load(pass_marker)
            if marker.get("scientific_gate_sha256") != sha256_file(
                stage_root / "scientific_gate.json"
            ):
                return False
    return True


def require_scientific_pass(artifact_root: Path | str) -> Path:
    """Fail closed before any downstream CA comparison consumes this track."""

    main_root = Path(artifact_root).expanduser().resolve() / "main"
    expected_runs = len(MODEL_IDS) * 10
    if not _stage_valid(main_root, "main", expected_runs):
        raise RuntimeError("baseline main computation/receipt chain is not valid")
    gate_path = main_root / "scientific_gate.json"
    gate = strict_json_load(gate_path)
    marker = main_root / "SCIENTIFIC_PASS"
    if not bool(gate.get("all_required_gates_pass")) or not marker.is_file():
        raise RuntimeError("baseline computation completed without scientific pass")
    marker_payload = strict_json_load(marker)
    if (
        marker_payload.get("scientific_gate_sha256") != sha256_file(gate_path)
        or marker_payload.get("downstream_ca_comparison_authorized") is not True
    ):
        raise RuntimeError("baseline SCIENTIFIC_PASS marker is not bound to its gate")
    return marker


def run_stage(stage: str, artifact_root: Path, config_source: Path, slots: Sequence[str]) -> Path:
    if stage not in {"smoke", "screen", "main", "sensitivity"}:
        raise ValueError("stage must be smoke, screen, main, or sensitivity")
    root = artifact_root.expanduser().resolve()
    config, copied = _prepare_root(root, config_source, require_clean=stage != "smoke")
    if stage == "smoke":
        bank = _ensure_bank(root, config, "smoke")
        specs = build_smoke_plan(root, config, bank)
        stage_root = root / stage
        if _stage_valid(stage_root, stage, len(specs)):
            return stage_root
        _run_specs(specs, copied, slots)
        summary = stage_root / "summary.json"
        _write_or_verify(summary, _smoke_summary(specs))
        _finalize(stage_root, stage, specs, [summary])
        return stage_root

    screen_root = root / "screen"
    if stage == "screen":
        bank = _ensure_bank(root, config, "tuning")
        specs = build_screen_plan(root, config, bank)
        if _stage_valid(screen_root, stage, len(specs)):
            return screen_root
        _run_specs(specs, copied, slots)
        selection = screen_root / "lr_selection.json"
        _write_or_verify(selection, select_learning_rates(specs, config))
        audit = screen_root / "paper_100_update_selector_audit.json"
        _write_or_verify(audit, paper_selector_audit(specs, config))
        _finalize(screen_root, stage, specs, [selection, audit])
        return screen_root

    expected_screen = len(MODEL_IDS) * 4 * 5
    if not _stage_valid(screen_root, "screen", expected_screen):
        raise RuntimeError(f"{stage} is blocked until the verified five-seed screen completes")
    selection_path = screen_root / "lr_selection.json"
    selection = strict_json_load(selection_path)
    bank = _ensure_bank(root, config, "main_test")
    parent_binding = _parent_screen_binding(root)
    if stage == "main":
        specs = build_main_plan(root, config, bank, selection)
        summary_payload = lambda: summarize_main(specs, config)
    else:
        specs = build_sensitivity_plan(root, config, bank, selection)
        summary_payload = lambda: summarize_sensitivity(specs)
    stage_root = root / stage
    if _stage_valid(stage_root, stage, len(specs)):
        return stage_root
    _run_specs(specs, copied, slots)
    binding_path = stage_root / "parent_screen_binding.json"
    _write_or_verify(binding_path, parent_binding)
    summary_path = stage_root / "summary.json"
    summary = summary_payload()
    _write_or_verify(summary_path, summary)
    extras = [binding_path, summary_path]
    if stage == "main":
        gate_path = stage_root / "scientific_gate.json"
        gate = scientific_gate(summary)
        _write_or_verify(gate_path, gate)
        extras.append(gate_path)
        if bool(gate["all_required_gates_pass"]):
            pass_path = stage_root / "SCIENTIFIC_PASS"
            _write_or_verify(
                pass_path,
                {
                    "schema_version": 1,
                    "campaign_id": CAMPAIGN_ID,
                    "scientific_gate_sha256": sha256_file(gate_path),
                    "downstream_ca_comparison_authorized": True,
                },
            )
            extras.append(pass_path)
        elif (stage_root / "SCIENTIFIC_PASS").exists():
            raise RuntimeError("stale SCIENTIFIC_PASS exists for a failed scientific gate")
    _finalize(stage_root, stage, specs, extras)
    return stage_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "screen", "main", "sensitivity"))
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    return parser


def _execute_worker(spec: RunSpec, config_path: Path, device: str) -> None:
    """Catch scientific non-finite outcomes only; infrastructure remains retryable."""

    try:
        _train_worker(spec, config_path, device)
    except FloatingPointError as error:
        _record_scientific_failure(
            spec,
            config_path,
            device,
            failure_kind=type(error).__name__,
            failure_message=str(error),
            traceback_text=traceback.format_exc(),
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_spec is not None:
        if args.stage is not None or args.artifact_root is not None:
            raise ValueError("worker mode cannot include campaign options")
        spec = RunSpec(**strict_json_load(args.worker_spec))
        config_path = args.config.resolve(strict=True)
        _execute_worker(spec, config_path, str(args.device))
        return 0
    if args.stage is None or args.artifact_root is None:
        raise ValueError("campaign mode requires --stage and --artifact-root")
    destination = run_stage(
        args.stage,
        args.artifact_root,
        args.config.resolve(strict=True),
        _parse_slots(args.gpus),
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
