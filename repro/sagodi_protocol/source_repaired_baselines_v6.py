"""Public-code-centered Ságodi baseline reproduction with documented repairs.

This campaign is intentionally distinct from the historical v4/v5 tracks.  It
uses the released 128-step, variable-sparsity, q1-initialized task and released
RNN/GRU/LSTM model paths.  Only registered repairs are applied.  Results must
be described as a *public-code-centered controlled adaptation with documented
repairs, not exact*, never as an exact reproduction of either the paper prose
or its broken and mutually inconsistent public runners.

Stages are ``smoke -> sentinel -> fanout -> main``.  Every model uses the same
actual post-transition per-coordinate state-noise standard deviation. The pilot
screens all eight learning rates for 2,000 updates with seeds 100 and 101, then
trains the selected setting for 5,000 updates on fresh seeds 0--2. All models use
clean q1 initialization and clean loss targets;
the public GRU/LSTM target-noise value is provenance only. Computation
completeness and scientific success are separate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .artifacts import (
    atomic_json,
    canonical_hash,
    canonical_tensor_mapping_sha256,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from .metrics import task_metrics
from .source_resolved_models import SourceResolvedBaseline
from .source_resolved_protocol import (
    UPSTREAM_COMMIT,
    build_source_optimizer,
    clip_source_gradients,
    source_angular_integration,
    source_masked_mse,
    source_recipe,
)
from .tasks import Batch, load_fixed_bank, save_fixed_bank


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "source_repaired_baselines_v6.json"
FREEZE_DOCUMENT = MODULE_DIR / "SAGODI_SOURCE_REPAIRED_BASELINES_V6_FREEZE_ko.md"
CAMPAIGN_ID = "sagodi_source_repaired_baselines_v6"
PROTOCOL_REVISION = (
    "public_code_architecture_noise_free_controlled_training_lr_only_pilot3_v6"
)
TRACK_CLASSIFICATION = (
    "public_code_architecture_noise_free_controlled_training_with_paper_and_code_noise_"
    "provenance_and_documented_repairs_not_exact"
)
ROOT_MARKER = ".sagodi_source_repaired_baselines_v6_root.json"
CONFIG_CONTRACT_SHA256 = "b10dcb5d14a2c9918edfbdf2b3ed5d5e1d90574cadefaf80fcf8087da5e450ad"
MODEL_IDS = (
    "sagodi_rnn_tanh_n128",
    "sagodi_gru_n128",
    "sagodi_lstm_n64",
)
WIDTHS = {
    "sagodi_rnn_tanh_n128": 128,
    "sagodi_gru_n128": 128,
    "sagodi_lstm_n64": 64,
}
PARAMETER_COUNTS = {
    "sagodi_rnn_tanh_n128": 17154,
    "sagodi_gru_n128": 50818,
    "sagodi_lstm_n64": 17538,
}
UPSTREAM_PUBLIC_TARGET_NOISE = {
    "sagodi_rnn_tanh_n128": 0.0,
    "sagodi_gru_n128": 0.01,
    "sagodi_lstm_n64": 0.01,
}
UPSTREAM_PUBLIC_CONFIG_EFFECTIVE_STATE_NOISE = {
    "sagodi_rnn_tanh_n128": 0.0,
    "sagodi_gru_n128": 0.0,
    "sagodi_lstm_n64": 0.0,
}
UPSTREAM_PUBLIC_CONFIG_NOMINAL_STATE_NOISE = {
    "sagodi_rnn_tanh_n128": 0.0,
    "sagodi_gru_n128": None,
    "sagodi_lstm_n64": None,
}
PRIOR_INTERNAL_RECIPE_NOMINAL_STATE_NOISE = {
    "sagodi_rnn_tanh_n128": 0.1,
    "sagodi_gru_n128": 0.0,
    "sagodi_lstm_n64": 0.0,
}
PRIOR_INTERNAL_RECIPE_EFFECTIVE_STATE_NOISE = {
    "sagodi_rnn_tanh_n128": 0.0316228,
    "sagodi_gru_n128": 0.0,
    "sagodi_lstm_n64": 0.0,
}
HYPOTHETICAL_RNN_SQRT_DT_EFFECTIVE = {
    "sagodi_rnn_tanh_n128": 0.0316228,
    "sagodi_gru_n128": None,
    "sagodi_lstm_n64": None,
}
UPSTREAM_PUBLIC_OUTPUT_DROPOUT = {
    "sagodi_rnn_tanh_n128": 0.0,
    "sagodi_gru_n128": 0.5,
    "sagodi_lstm_n64": 0.0,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _native(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the immutable campaign contract and reject semantic drift."""

    payload = strict_json_load(Path(path).expanduser().resolve(strict=True))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("v6 config must be a schema-1 JSON object")
    if canonical_hash(payload) != CONFIG_CONTRACT_SHA256:
        raise ValueError("v6 config differs from the exact canonical frozen contract")
    fixed = {
        "campaign_id": CAMPAIGN_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "track_classification": TRACK_CLASSIFICATION,
    }
    for key, expected in fixed.items():
        if payload.get(key) != expected:
            raise ValueError(f"v6 {key} differs")
    if payload.get("upstream", {}).get("commit") != UPSTREAM_COMMIT:
        raise ValueError("v6 upstream commit differs")
    task = payload.get("task", {})
    expected_task = {
        "name": "angular_integration",
        "duration": 12.8,
        "horizon": 128,
        "dt": 0.1,
        "gp_length_scale": 1.0,
        "gp_std": 1.0,
        "gp_jitter": 1e-6,
        "input_sparsity": "variable_uniform_0_2",
        "random_angle_init": True,
        "target_indexing": "q_t_plus_1_after_velocity_update",
        "initial_state_semantics": "source_q1_post_update_target",
    }
    if task != expected_task:
        raise ValueError("v6 task contract differs")
    models = payload.get("models")
    if not isinstance(models, list) or [row.get("id") for row in models] != list(MODEL_IDS):
        raise ValueError("v6 model order differs")
    for row in models:
        model_id = str(row["id"])
        if int(row.get("width", -1)) != WIDTHS[model_id]:
            raise ValueError(f"v6 width differs for {model_id}")
        if int(row.get("parameter_count", -1)) != PARAMETER_COUNTS[model_id]:
            raise ValueError(f"v6 parameter count differs for {model_id}")
        if float(row.get("upstream_public_code_target_noise_std", -1.0)) != (
            UPSTREAM_PUBLIC_TARGET_NOISE[model_id]
        ):
            raise ValueError(f"v6 upstream target-noise provenance differs for {model_id}")
        if float(row.get("controlled_target_noise_std", -1.0)) != 0.0:
            raise ValueError(f"v6 controlled target noise differs for {model_id}")
        if float(row.get("upstream_public_config_effective_state_noise_std", -1.0)) != (
            UPSTREAM_PUBLIC_CONFIG_EFFECTIVE_STATE_NOISE[model_id]
        ):
            raise ValueError(f"v6 upstream state-noise provenance differs for {model_id}")
        if row.get("upstream_public_config_nominal_state_noise_std") != (
            UPSTREAM_PUBLIC_CONFIG_NOMINAL_STATE_NOISE[model_id]
        ):
            raise ValueError(f"v6 upstream nominal-noise provenance differs for {model_id}")
        if row.get("prior_internal_source_resolved_recipe_nominal_state_noise_std") != (
            PRIOR_INTERNAL_RECIPE_NOMINAL_STATE_NOISE[model_id]
        ):
            raise ValueError(f"v6 prior internal nominal-noise provenance differs for {model_id}")
        if row.get("prior_internal_source_resolved_recipe_effective_state_noise_std") != (
            PRIOR_INTERNAL_RECIPE_EFFECTIVE_STATE_NOISE[model_id]
        ):
            raise ValueError(f"v6 prior internal effective-noise provenance differs for {model_id}")
        if row.get("hypothetical_nominal_0p1_effective_under_sqrt_dt") != (
            HYPOTHETICAL_RNN_SQRT_DT_EFFECTIVE[model_id]
        ):
            raise ValueError(f"v6 hypothetical sqrt-dt provenance differs for {model_id}")
        if float(row.get("upstream_public_code_output_dropout", -1.0)) != (
            UPSTREAM_PUBLIC_OUTPUT_DROPOUT[model_id]
        ):
            raise ValueError(f"v6 upstream dropout provenance differs for {model_id}")
        if float(row.get("controlled_output_dropout", -1.0)) != 0.0:
            raise ValueError(f"v6 controlled output dropout differs for {model_id}")
    training = payload.get("training", {})
    if training != {
        "optimizer": "Adam",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "batch_size": 64,
        "updates": 5000,
        "online_batches": True,
        "constant_learning_rate": True,
        "early_stopping": False,
        "trace_interval": 50,
        "validation_interval": 500,
        "worker_threads": 1,
        "actual_post_transition_state_noise_std": 0.0,
        "state_noise_distribution": "disabled",
        "state_noise_location": "disabled_no_state_noise_injection",
        "state_noise_scaling": "not_applicable_disabled",
        "paper_state_noise_covariance_provenance_only": "0.01I",
        "evaluation_state_noise_std": 0.0,
        "controlled_target_noise_std": 0.0,
        "controlled_output_dropout": 0.0,
        "training_target_semantics": "clean_cos_sin_target_for_initial_q1_and_loss",
    }:
        raise ValueError("v6 training contract differs")
    tuning = payload.get("hyperparameter_tuning", {})
    if tuning.get("learning_rate_grid") != [
        0.03,
        0.01,
        0.003,
        0.001,
        0.0003,
        0.0001,
        0.00003,
        0.00001,
    ]:
        raise ValueError("v6 LR grid differs")
    if (
        tuning.get("sentinel_seed") != 100
        or tuning.get("fanout_seeds") != [101]
        or tuning.get("screening_updates") != 2000
        or tuning.get("fanout_all_learning_rates") is not True
    ):
        raise ValueError("v6 tuning seed/fanout contract differs")
    if payload.get("main") != {
        "seeds": list(range(3)),
        "fresh_from_tuning": True,
        "mse_threshold_role": "descriptive_success_yield_only",
        "analysis_eligibility_rule": "per_seed_nmse_db_below_minus20",
        "scientific_pass_minimum_eligible_per_model": 1,
        "low_eligible_count_warning_below": 2,
    }:
        raise ValueError("v6 main contract differs")
    return payload


def _model_contract(config: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    for row in config["models"]:
        if row["id"] == model_id:
            return row
    raise ValueError(f"unregistered model: {model_id}")


def _common_state_noise(config: Mapping[str, Any]) -> float:
    return float(config["training"]["actual_post_transition_state_noise_std"])


def _recipe(config: Mapping[str, Any], model_id: str, lr: float, noise: float):
    """Apply only registered v6 overrides to the released model recipe."""

    source = source_recipe(model_id)
    contract = _model_contract(config, model_id)
    if not math.isclose(
        float(source.nominal_state_noise_std),
        float(contract["prior_internal_source_resolved_recipe_nominal_state_noise_std"]),
        rel_tol=1e-6,
        abs_tol=1e-8,
    ) or not math.isclose(
        float(source.effective_state_noise_std),
        float(contract["prior_internal_source_resolved_recipe_effective_state_noise_std"]),
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise RuntimeError("historical source-resolved recipe provenance differs")
    return replace(
        source,
        learning_rate=float(lr),
        nominal_state_noise_std=(
            float(noise) / math.sqrt(0.1)
            if model_id == "sagodi_rnn_tanh_n128"
            else float(source.nominal_state_noise_std)
        ),
        effective_state_noise_std=float(noise),
        target_noise_std=float(contract["controlled_target_noise_std"]),
        output_dropout=float(contract["controlled_output_dropout"]),
        recurrent_weight_decay=float(contract["recurrent_weight_decay"]),
        gradient_clip_norm=(
            None
            if contract["gradient_clip_norm"] is None
            else float(contract["gradient_clip_norm"])
        ),
    )


def noise_metadata(model_id: str, actual_std: float) -> dict[str, Any]:
    """Record disabled execution separately from paper/code provenance."""

    actual = float(actual_std)
    if actual != 0.0:
        raise ValueError("noise-free controlled execution requires actual state noise zero")
    return {
        "actual_post_transition_state_noise_std": actual,
        "state_noise_enabled": False,
        "state_noise_scaling": "not_applicable_disabled",
        "paper_literal_state_noise_std_provenance_only": 0.1,
        "paper_state_noise_covariance_provenance_only": "0.01I",
        "controlled_source_api_nominal_state_noise_std": (
            0.0 if model_id == "sagodi_rnn_tanh_n128" else None
        ),
        "controlled_source_api_nominal_semantics": (
            "disabled_zero_not_executed"
            if model_id == "sagodi_rnn_tanh_n128"
            else "not_applicable_no_upstream_state_noise_api"
        ),
        "upstream_public_config_nominal_state_noise_std_provenance_only": (
            UPSTREAM_PUBLIC_CONFIG_NOMINAL_STATE_NOISE[model_id]
        ),
        "upstream_public_config_effective_state_noise_std_provenance_only": (
            UPSTREAM_PUBLIC_CONFIG_EFFECTIVE_STATE_NOISE[model_id]
        ),
        "prior_internal_source_resolved_recipe_nominal_state_noise_std_provenance_only": (
            PRIOR_INTERNAL_RECIPE_NOMINAL_STATE_NOISE[model_id]
        ),
        "prior_internal_source_resolved_recipe_effective_state_noise_std_provenance_only": (
            PRIOR_INTERNAL_RECIPE_EFFECTIVE_STATE_NOISE[model_id]
        ),
        "hypothetical_nominal_0p1_effective_under_sqrt_dt_provenance_only": (
            HYPOTHETICAL_RNN_SQRT_DT_EFFECTIVE[model_id]
        ),
        "noise_location": "disabled_no_state_noise_injection",
    }


def build_model(config: Mapping[str, Any], model_id: str, lr: float, noise: float) -> SourceResolvedBaseline:
    model = SourceResolvedBaseline(_recipe(config, model_id, lr, noise))
    if model_id == "sagodi_rnn_tanh_n128":
        # The public path zeroes brec and immediately overwrites it with
        # Uniform[-sqrt(H),sqrt(H)].  Registered T128/q1 parity showed that
        # literal oddity misses both gates, whereas the paper-aligned zero-bias
        # repair reaches MSE 0.00576 at the same 5k-update budget.
        nn.init.zeros_(model.core.brec)
    actual = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if actual != PARAMETER_COUNTS[model_id]:
        raise RuntimeError(f"parameter mismatch for {model_id}: {actual}")
    return model


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    stage: str
    model_id: str
    model_seed: int
    learning_rate: float
    actual_state_noise_std: float
    updates: int
    batch_size: int
    evaluation_bank: str
    output_dir: str
    smoke: bool = False

    def payload(self) -> dict[str, Any]:
        return _native(asdict(self))


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
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _finite_model(model: nn.Module, *, gradients: bool = False) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise FloatingPointError(f"non-finite parameter: {name}")
        if gradients and parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            raise FloatingPointError(f"non-finite gradient: {name}")


def _to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        inputs=batch.inputs.to(device),
        output_targets=batch.output_targets.to(device),
        latent_targets=batch.latent_targets.to(device),
        mask=batch.mask.to(device),
        metadata=batch.metadata,
    )


def _training_batch(spec: RunSpec, update: int, device: torch.device) -> Batch:
    return source_angular_integration(
        spec.batch_size,
        0,
        stream_key=(CAMPAIGN_ID, "online_train", spec.model_seed, int(update)),
        device=device,
    )


def _rng_stream_identities(spec: RunSpec, recipe: Any) -> dict[str, Any]:
    return {
        "online_task": {
            "base_seed": 0,
            "stream_key_template": [
                CAMPAIGN_ID,
                "online_train",
                spec.model_seed,
                "<update_1_to_5000>",
            ],
        },
        "target_noise": {
            "enabled": False,
            "generator_seed": None,
            "std": float(recipe.target_noise_std),
            "upstream_public_code_std_provenance_only": float(
                UPSTREAM_PUBLIC_TARGET_NOISE[spec.model_id]
            ),
            "semantics": "clean_q1_initializer_and_clean_loss_target_no_rng_draws",
        },
        "state_noise": {
            "enabled": False,
            "generator_seed": None,
            "std": float(spec.actual_state_noise_std),
            "scaling": "not_applicable_disabled",
            "paper_literal_std_provenance_only": 0.1,
            "upstream_public_config_effective_std_provenance_only": float(
                UPSTREAM_PUBLIC_CONFIG_EFFECTIVE_STATE_NOISE[spec.model_id]
            ),
            "semantics": "controlled_execution_has_no_state_noise_rng_draws",
        },
        "dropout": {
            "enabled": False,
            "global_torch_seed": None,
            "probability": float(recipe.output_dropout),
            "upstream_public_code_probability_provenance_only": float(
                UPSTREAM_PUBLIC_OUTPUT_DROPOUT[spec.model_id]
            ),
            "semantics": "controlled_execution_has_no_dropout_rng_draws",
        },
        "retention_plasticity_probe": {"enabled": False, "stream_key_template": None},
    }


@torch.no_grad()
def _evaluate(model: SourceResolvedBaseline, batch: Batch) -> dict[str, Any]:
    model.eval()
    prediction = model.forward_sequence(
        batch.inputs,
        source_targets=batch.output_targets,
        state_noise_std_override=0.0,
    )
    if not torch.isfinite(prediction).all().item():
        raise FloatingPointError("non-finite held-out prediction")
    metrics = task_metrics(prediction, batch.output_targets, batch.mask, batch.latent_targets)
    metrics["mse"] = metrics["masked_mse"]
    metrics["nmse_db"] = metrics["masked_nmse_db"]
    return _native(metrics)


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


def _environment_versions(device: torch.device | None = None) -> dict[str, Any]:
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device": None if device is None else str(device),
        "worker_threads": torch.get_num_threads(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_slot_from_launcher": os.environ.get("CALRU_PHYSICAL_GPU_SLOT"),
    }
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        payload["cuda_visible_index"] = index
        payload["cuda_device_name"] = torch.cuda.get_device_name(index)
        payload["cuda_compute_capability"] = list(torch.cuda.get_device_capability(index))
    return payload


def _runtime_files() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        MODULE_DIR / "source_resolved_models.py",
        MODULE_DIR / "source_resolved_protocol.py",
        MODULE_DIR / "sagodi_source_resolved_v1.json",
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
        raise RuntimeError("full v6 stages require a clean committed worktree")
    return {"code_commit": commit, "worktree_clean": not bool(status)}


def _scientific_identity(config_path: Path, *, require_clean: bool) -> dict[str, Any]:
    git = _git_state(require_clean)
    payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "track_classification": TRACK_CLASSIFICATION,
        "upstream_commit": UPSTREAM_COMMIT,
        "source_config_sha256": sha256_file(config_path),
        "source_freeze_sha256": sha256_file(FREEZE_DOCUMENT),
        "runtime_code_sha256": {path.name: sha256_file(path) for path in _runtime_files()},
        "code_commit": git["code_commit"],
        "environment_versions": _environment_versions(),
    }
    payload["scientific_identity"] = canonical_hash(payload)
    return payload


def _assert_worker_identity(
    identity: Mapping[str, Any], config_path: Path, *, require_clean: bool
) -> None:
    git = _git_state(require_clean)
    current_hashes = {path.name: sha256_file(path) for path in _runtime_files()}
    identity_core = dict(identity)
    observed_identity_digest = identity_core.pop("scientific_identity", None)
    if canonical_hash(identity_core) != observed_identity_digest:
        raise RuntimeError("worker scientific identity digest differs")
    if identity.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("worker campaign identity differs")
    if identity.get("protocol_revision") != PROTOCOL_REVISION:
        raise RuntimeError("worker protocol revision differs")
    if identity.get("track_classification") != TRACK_CLASSIFICATION:
        raise RuntimeError("worker track classification differs")
    if git["code_commit"] != identity.get("code_commit"):
        raise RuntimeError("worker git commit differs from campaign root")
    if current_hashes != identity.get("runtime_code_sha256"):
        raise RuntimeError("worker runtime code hashes differ from campaign root")
    if sha256_file(config_path) != identity.get("source_config_sha256"):
        raise RuntimeError("worker config hash differs from campaign root")
    if sha256_file(FREEZE_DOCUMENT) != identity.get("source_freeze_sha256"):
        raise RuntimeError("worker freeze-document hash differs from campaign root")


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
    config_source = config_source.expanduser().resolve(strict=True)
    config = load_config(config_source)
    identity = _scientific_identity(config_source, require_clean=require_clean)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ROOT_MARKER
    if marker.exists():
        if strict_json_load(marker) != identity:
            raise RuntimeError("artifact root belongs to another scientific identity")
    else:
        if any(root.iterdir()):
            raise RuntimeError("unmarked v6 artifact root must be empty")
        atomic_json(marker, identity)
    copied = root / "inputs" / DEFAULT_CONFIG.name
    _copy_or_verify(config_source, copied)
    _copy_or_verify(FREEZE_DOCUMENT, root / "inputs" / FREEZE_DOCUMENT.name)
    return config, copied


def _validate_bank(
    batch: Batch,
    *,
    trials: int,
    config: Mapping[str, Any],
    purpose: str,
) -> None:
    metadata = batch.metadata
    expected = {
        "task_name": "angular_integration",
        "task_version": "sagodi-source-resolved-v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "horizon": 128,
        "duration": 12.8,
        "delta_t": 0.1,
        "input_sparsity": "variable_uniform_0_2",
        "target_indexing": "post_velocity_update",
        "initial_state_semantics": "source_q1_post_update_target",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"fixed bank metadata differs for {key}")
    if batch.batch_size != int(trials) or batch.time_steps != 128:
        raise RuntimeError("fixed bank tensor shape differs")
    if tuple(batch.inputs.shape[-1:]) != (1,) or tuple(batch.output_targets.shape[-1:]) != (2,):
        raise RuntimeError("fixed bank feature dimensions differ")
    if not torch.all(batch.mask == 1).item():
        raise RuntimeError("fixed bank mask is not dense")
    if config["task"]["initial_state_semantics"] != metadata["initial_state_semantics"]:
        raise RuntimeError("fixed bank q1 semantics differ")
    if purpose == "smoke":
        expected_seed = 32999
        expected_stream = [CAMPAIGN_ID, "fixed_smoke_bank"]
        expected_trials = 16
    elif purpose in {"tuning", "main_test"}:
        contract = config["evaluation_banks"][purpose]
        expected_seed = int(contract["task_seed"])
        expected_stream = list(contract["stream_key"])
        expected_trials = int(contract["trials"])
    else:
        raise RuntimeError(f"unknown frozen bank purpose: {purpose}")
    if int(trials) != expected_trials:
        raise RuntimeError(f"fixed bank trial count differs for {purpose}")
    if metadata.get("base_seed") != expected_seed:
        raise RuntimeError(f"fixed bank base seed differs for {purpose}")
    if list(metadata.get("stream_key", ())) != expected_stream:
        raise RuntimeError(f"fixed bank stream key differs for {purpose}")
    expected_batch = source_angular_integration(
        int(trials),
        expected_seed,
        stream_key=expected_stream,
        device="cpu",
    )
    # The archive and SHA sidecar are exact. Independent NumPy BLAS builds can
    # differ by a few float32 ulps in ``white @ chol.T``; cumulative angles can
    # amplify the absolute delta when |theta| is large. A one-ulp relative
    # allowance plus 1e-6 absolute allowance is used only for regeneration.
    for name in ("inputs", "output_targets", "latent_targets"):
        observed = getattr(batch, name).detach().cpu()
        expected_tensor = getattr(expected_batch, name)
        if not torch.allclose(
            observed,
            expected_tensor,
            rtol=torch.finfo(observed.dtype).eps,
            atol=1e-6,
        ):
            raise RuntimeError(f"fixed bank deterministic content differs for {name}")
    if not torch.equal(batch.mask.detach().cpu(), expected_batch.mask):
        raise RuntimeError("fixed bank deterministic content differs for mask")


def _ensure_bank(root: Path, config: Mapping[str, Any], purpose: str) -> Path:
    if purpose == "smoke":
        trials, seed = 16, 32999
        stream_key = (CAMPAIGN_ID, "fixed_smoke_bank")
    elif purpose in {"tuning", "main_test"}:
        contract = config["evaluation_banks"][purpose]
        trials, seed = int(contract["trials"]), int(contract["task_seed"])
        stream_key = tuple(contract["stream_key"])
    else:
        raise ValueError("unknown evaluation-bank purpose")
    path = root / "banks" / f"{purpose}.npz"
    if not path.exists():
        save_fixed_bank(
            path,
            source_angular_integration(
                trials, seed, stream_key=stream_key, device="cpu"
            ),
        )
    _validate_bank(load_fixed_bank(path), trials=trials, config=config, purpose=purpose)
    return path


def _cell_key(lr: float, noise: float) -> str:
    lr_text = format(float(lr), ".7g").replace(".", "p").replace("-", "m")
    noise_text = format(float(noise), ".7g").replace(".", "p").replace("-", "m")
    return f"lr{lr_text}__noise{noise_text}"


def build_smoke_plan(root: Path, config: Mapping[str, Any], bank: Path) -> tuple[RunSpec, ...]:
    return tuple(
        RunSpec(
            run_id=f"smoke__{model_id}",
            stage="smoke",
            model_id=model_id,
            model_seed=999,
            learning_rate=1e-3,
            actual_state_noise_std=_common_state_noise(config),
            updates=2,
            batch_size=4,
            evaluation_bank=str(bank),
            output_dir=str(root / "smoke" / "runs" / model_id),
            smoke=True,
        )
        for model_id in MODEL_IDS
    )


def build_sentinel_plan(root: Path, config: Mapping[str, Any], bank: Path) -> tuple[RunSpec, ...]:
    tuning = config["hyperparameter_tuning"]
    noise = _common_state_noise(config)
    specs: list[RunSpec] = []
    for model_id in MODEL_IDS:
        for lr in tuning["learning_rate_grid"]:
            cell = _cell_key(float(lr), noise)
            run_id = f"sentinel__{model_id}__{cell}__seed100"
            specs.append(
                RunSpec(
                    run_id=run_id,
                    stage="sentinel",
                    model_id=model_id,
                    model_seed=int(tuning["sentinel_seed"]),
                    learning_rate=float(lr),
                    actual_state_noise_std=noise,
                    updates=int(tuning["screening_updates"]),
                    batch_size=int(config["training"]["batch_size"]),
                    evaluation_bank=str(bank),
                    output_dir=str(root / "sentinel" / "runs" / run_id),
                )
            )
    return tuple(specs)


def _result(spec: RunSpec) -> dict[str, Any]:
    result = strict_json_load(Path(spec.output_dir) / "result.json")
    if result.get("run_id") != spec.run_id:
        raise RuntimeError(f"run/result identity mismatch: {spec.run_id}")
    return result


def _metric(spec: RunSpec, key: str) -> float | None:
    result = _result(spec)
    if result.get("status") != "completed" or not isinstance(result.get("final_metrics"), dict):
        return None
    value = result["final_metrics"].get(key)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _grid_index(config: Mapping[str, Any], spec: RunSpec) -> int:
    tuning = config["hyperparameter_tuning"]
    return list(tuning["learning_rate_grid"]).index(spec.learning_rate)


def summarize_sentinel(
    specs: Sequence[RunSpec], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Record all one-seed LR results without pruning any LR candidate."""

    tuning = config["hyperparameter_tuning"]
    threshold = float(tuning["success_mse_threshold"])
    nmse_threshold = float(tuning["analysis_nmse_db_threshold"])
    models: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        candidates = [spec for spec in specs if spec.model_id == model_id]
        rows: list[dict[str, Any]] = []
        for spec in candidates:
            mse, nmse = _metric(spec, "mse"), _metric(spec, "nmse_db")
            grid = _grid_index(config, spec)
            rows.append(
                {
                    "learning_rate": spec.learning_rate,
                    "actual_state_noise_std": spec.actual_state_noise_std,
                    "sentinel_run_id": spec.run_id,
                    "completed": mse is not None and nmse is not None,
                    "mse": mse,
                    "nmse_db": nmse,
                    "mse_pass": mse is not None and mse < threshold,
                    "nmse_pass": nmse is not None and nmse < nmse_threshold,
                    "grid_order": grid,
                }
            )
        rows.sort(
            key=lambda row: (
                not row["nmse_pass"],
                not row["mse_pass"],
                math.inf if row["mse"] is None else row["mse"],
                math.inf if row["nmse_db"] is None else row["nmse_db"],
                row["grid_order"],
            )
        )
        if len(rows) != len(tuning["learning_rate_grid"]):
            raise RuntimeError(f"sentinel LR denominator differs for {model_id}")
        models[model_id] = {
            "registered_learning_rates": len(rows),
            "failed_or_nonfinite_learning_rates": sum(not row["completed"] for row in rows),
            "all_ranked_learning_rates": rows,
        }
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "summary_stage": "sentinel_seed100_no_candidate_pruning",
        "failures_remain_in_denominator": True,
        "fanout_all_learning_rates": True,
        "models": models,
    }


def build_fanout_plan(
    root: Path,
    config: Mapping[str, Any],
    bank: Path,
    sentinel_summary: Mapping[str, Any],
) -> tuple[RunSpec, ...]:
    specs: list[RunSpec] = []
    tuning = config["hyperparameter_tuning"]
    noise = _common_state_noise(config)
    for model_id in MODEL_IDS:
        rows = sentinel_summary["models"][model_id]["all_ranked_learning_rates"]
        observed = {float(row["learning_rate"]) for row in rows}
        expected = set(map(float, config["hyperparameter_tuning"]["learning_rate_grid"]))
        if observed != expected or len(rows) != len(expected):
            raise ValueError(f"sentinel summary has wrong LR denominator for {model_id}")
        for lr in config["hyperparameter_tuning"]["learning_rate_grid"]:
            lr = float(lr)
            for seed in config["hyperparameter_tuning"]["fanout_seeds"]:
                cell = _cell_key(lr, noise)
                run_id = f"fanout__{model_id}__{cell}__seed{seed}"
                specs.append(
                    RunSpec(
                        run_id=run_id,
                        stage="fanout",
                        model_id=model_id,
                        model_seed=int(seed),
                        learning_rate=lr,
                        actual_state_noise_std=noise,
                        updates=int(tuning["screening_updates"]),
                        batch_size=int(config["training"]["batch_size"]),
                        evaluation_bank=str(bank),
                        output_dir=str(root / "fanout" / "runs" / run_id),
                    )
                )
    return tuple(specs)


def select_hyperparameters(
    sentinel_specs: Sequence[RunSpec],
    fanout_specs: Sequence[RunSpec],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Select one LR/model after a short two-seed pilot screen."""

    tuning = config["hyperparameter_tuning"]
    mse_threshold = float(tuning["success_mse_threshold"])
    nmse_threshold = float(tuning["analysis_nmse_db_threshold"])
    expected_seeds = {int(tuning["sentinel_seed"]), *map(int, tuning["fanout_seeds"])}
    expected_count = len(expected_seeds)
    models: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        fanout_model = [spec for spec in fanout_specs if spec.model_id == model_id]
        learning_rates = sorted(
            {spec.learning_rate for spec in fanout_model},
            key=list(tuning["learning_rate_grid"]).index,
        )
        if learning_rates != list(map(float, tuning["learning_rate_grid"])):
            raise ValueError(f"fanout LR denominator differs for {model_id}")
        rows: list[dict[str, Any]] = []
        for lr in learning_rates:
            noise = _common_state_noise(config)
            specs = [
                spec
                for spec in (*sentinel_specs, *fanout_specs)
                if spec.model_id == model_id
                and spec.learning_rate == lr
            ]
            if (
                {spec.model_seed for spec in specs} != expected_seeds
                or len(specs) != expected_count
                or {spec.actual_state_noise_std for spec in specs} != {noise}
            ):
                raise ValueError(f"pilot-screen denominator differs for {model_id}/{lr}")
            mses = [_metric(spec, "mse") for spec in specs]
            nmses = [_metric(spec, "nmse_db") for spec in specs]
            finite_mses = [value for value in mses if value is not None]
            finite_nmses = [value for value in nmses if value is not None]
            grid_order = list(tuning["learning_rate_grid"]).index(lr)
            rows.append(
                {
                    "learning_rate": lr,
                    "actual_state_noise_std": noise,
                    "registered_seed_count": expected_count,
                    "completed_seed_count": len(finite_mses),
                    "failed_seed_count": expected_count - len(finite_mses),
                    "mse_success_count": sum(
                        value is not None and value < mse_threshold for value in mses
                    ),
                    "nmse_success_count": sum(
                        value is not None and value < nmse_threshold for value in nmses
                    ),
                    "median_mse": (
                        float(np.median(finite_mses)) if len(finite_mses) == expected_count else None
                    ),
                    "mean_mse": (
                        float(np.mean(finite_mses)) if len(finite_mses) == expected_count else None
                    ),
                    "per_seed": [
                        {
                            "seed": spec.model_seed,
                            "run_id": spec.run_id,
                            "status": _result(spec).get("status"),
                            "mse": _metric(spec, "mse"),
                            "nmse_db": _metric(spec, "nmse_db"),
                        }
                        for spec in sorted(specs, key=lambda item: item.model_seed)
                    ],
                    "grid_order": grid_order,
                }
            )
        rows.sort(
            key=lambda row: (
                -row["nmse_success_count"],
                -row["mse_success_count"],
                math.inf if row["median_mse"] is None else row["median_mse"],
                math.inf if row["mean_mse"] is None else row["mean_mse"],
                row["grid_order"],
            )
        )
        eligible = [row for row in rows if row["completed_seed_count"] == expected_count]
        if not eligible:
            raise RuntimeError(f"no complete pilot-screen tuning cell for {model_id}")
        models[model_id] = {"winner": eligible[0], "ranked_learning_rates": rows}
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "selection_rule": tuning["selection_rule"],
        "registered_denominator_per_learning_rate": expected_count,
        "all_learning_rates_fanned_out": True,
        "failures_remain_in_denominator": True,
        "models": models,
    }


def build_main_plan(
    root: Path,
    config: Mapping[str, Any],
    bank: Path,
    selection: Mapping[str, Any],
) -> tuple[RunSpec, ...]:
    specs: list[RunSpec] = []
    for model_id in MODEL_IDS:
        winner = selection["models"][model_id]["winner"]
        lr, noise = float(winner["learning_rate"]), _common_state_noise(config)
        if float(winner["actual_state_noise_std"]) != noise:
            raise ValueError(f"selected state noise differs from common contract for {model_id}")
        for seed in config["main"]["seeds"]:
            cell = _cell_key(lr, noise)
            run_id = f"main__{model_id}__{cell}__seed{seed}"
            specs.append(
                RunSpec(
                    run_id=run_id,
                    stage="main",
                    model_id=model_id,
                    model_seed=int(seed),
                    learning_rate=lr,
                    actual_state_noise_std=noise,
                    updates=int(config["training"]["updates"]),
                    batch_size=int(config["training"]["batch_size"]),
                    evaluation_bank=str(bank),
                    output_dir=str(root / "main" / "runs" / run_id),
                )
            )
    return tuple(specs)


def summarize_main(specs: Sequence[RunSpec], config: Mapping[str, Any]) -> dict[str, Any]:
    tuning, main = config["hyperparameter_tuning"], config["main"]
    mse_threshold = float(tuning["success_mse_threshold"])
    nmse_threshold = float(tuning["analysis_nmse_db_threshold"])
    eligible_minimum = int(main["scientific_pass_minimum_eligible_per_model"])
    expected_seeds = set(map(int, main["seeds"]))
    expected_count = len(expected_seeds)
    models: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        selected = [spec for spec in specs if spec.model_id == model_id]
        if len(selected) != expected_count or {spec.model_seed for spec in selected} != expected_seeds:
            raise ValueError(f"main denominator differs for {model_id}")
        rows = []
        for spec in sorted(selected, key=lambda item: item.model_seed):
            mse, nmse = _metric(spec, "mse"), _metric(spec, "nmse_db")
            rows.append(
                {
                    "seed": spec.model_seed,
                    "run_id": spec.run_id,
                    "status": _result(spec).get("status"),
                    "mse": mse,
                    "nmse_db": nmse,
                    "mse_pass": mse is not None and mse < mse_threshold,
                    "nmse_pass": nmse is not None and nmse < nmse_threshold,
                }
            )
        mse_count = sum(row["mse_pass"] for row in rows)
        nmse_count = sum(row["nmse_pass"] for row in rows)
        models[model_id] = {
            "registered_seed_count": expected_count,
            "failed_seed_count": sum(row["status"] != "completed" for row in rows),
            "mse_success_count": mse_count,
            "nmse_success_count": nmse_count,
            "mse_success_rate": mse_count / float(expected_count),
            "analysis_eligible_count": nmse_count,
            "analysis_eligible_rate": nmse_count / float(expected_count),
            "analysis_gate_pass": nmse_count >= eligible_minimum,
            "low_eligible_count_warning": nmse_count < int(
                main["low_eligible_count_warning_below"]
            ),
            "mse_threshold_is_descriptive_not_gate": True,
            "per_seed": rows,
        }
    all_pass = all(row["analysis_gate_pass"] for row in models.values())
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "track_classification": TRACK_CLASSIFICATION,
        "failures_remain_in_denominator": True,
        "mse_threshold": mse_threshold,
        "nmse_db_threshold": nmse_threshold,
        "models": models,
        "all_required_gates_pass": all_pass,
    }


def scientific_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "all_required_gates_pass": bool(summary["all_required_gates_pass"]),
        "required_gates": [
            "all_registered_fresh_pilot_seeds_are_trained_and_reported",
            "at_least_one_nmse_below_minus20db_analysis_eligible_seed_per_model",
        ],
        "computation_complete_is_not_scientific_pass": True,
        "model_gate_status": {
            model_id: {
                "analysis_gate_pass": row["analysis_gate_pass"],
                "analysis_eligible_count": row["analysis_eligible_count"],
                "low_eligible_count_warning": row["low_eligible_count_warning"],
                "mse_success_count_descriptive": row["mse_success_count"],
            }
            for model_id, row in summary["models"].items()
        },
    }


def _validate_spec(config: Mapping[str, Any], spec: RunSpec) -> None:
    if spec.model_id not in MODEL_IDS:
        raise ValueError("run spec contains an unknown model")
    if spec.updates <= 0 or spec.batch_size <= 0:
        raise ValueError("run spec updates/batch size must be positive")
    if spec.learning_rate <= 0 or spec.actual_state_noise_std < 0:
        raise ValueError("run spec LR/noise is invalid")
    if spec.actual_state_noise_std != _common_state_noise(config):
        raise ValueError("run spec state noise differs from the common fixed contract")
    if spec.smoke:
        if spec.stage != "smoke":
            raise ValueError("smoke flag/stage mismatch")
        return
    tuning = config["hyperparameter_tuning"]
    if spec.stage not in {"sentinel", "fanout", "main"}:
        raise ValueError("unknown full run stage")
    if spec.learning_rate not in tuning["learning_rate_grid"]:
        raise ValueError("run spec LR is outside frozen grid")
    expected_updates = (
        int(tuning["screening_updates"])
        if spec.stage in {"sentinel", "fanout"}
        else int(config["training"]["updates"])
    )
    if spec.updates != expected_updates:
        raise ValueError("full run update count differs")
    if spec.batch_size != int(config["training"]["batch_size"]):
        raise ValueError("full run batch size differs")
    allowed = {
        "sentinel": {int(tuning["sentinel_seed"])},
        "fanout": set(map(int, tuning["fanout_seeds"])),
        "main": set(map(int, config["main"]["seeds"])),
    }
    if spec.model_seed not in allowed[spec.stage]:
        raise ValueError("run spec seed differs from frozen stage")


def load_checkpoint(
    path: Path | str,
    config_path: Path | str = DEFAULT_CONFIG,
    device: torch.device | str = "cpu",
) -> tuple[SourceResolvedBaseline, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("checkpoint_type") != CAMPAIGN_ID:
        raise ValueError("not a v6 completed checkpoint")
    spec = RunSpec(**payload["run"])
    config = load_config(config_path)
    model = build_model(
        config, spec.model_id, spec.learning_rate, spec.actual_state_noise_std
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def _train_worker(spec: RunSpec, config_path: Path, device_text: str) -> Path:
    config = load_config(config_path)
    _validate_spec(config, spec)
    output = Path(spec.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"worker output is not empty: {output}")
    bank_path = Path(spec.evaluation_bank).expanduser().resolve(strict=True)
    root = bank_path.parents[1]
    identity = strict_json_load(root / ROOT_MARKER)
    _assert_worker_identity(identity, config_path, require_clean=not spec.smoke)
    device = torch.device(device_text)
    _configure_determinism(spec.model_seed)
    model = build_model(
        config, spec.model_id, spec.learning_rate, spec.actual_state_noise_std
    ).to(device)
    initial_state_dict_sha256 = canonical_tensor_mapping_sha256(model.state_dict())
    _finite_model(model)
    recipe = model.recipe
    if float(recipe.target_noise_std) != 0.0:
        raise RuntimeError("controlled primary requires zero target noise")
    if float(recipe.output_dropout) != 0.0:
        raise RuntimeError("controlled primary requires zero output dropout")
    optimizer = build_source_optimizer(model, recipe)
    evaluation = _to_device(load_fixed_bank(bank_path), device)
    _validate_bank(
        evaluation,
        trials=evaluation.batch_size,
        config=config,
        purpose=Path(spec.evaluation_bank).stem,
    )
    rng_stream_identities = _rng_stream_identities(spec, recipe)
    contract = _model_contract(config, spec.model_id)
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "track_classification": TRACK_CLASSIFICATION,
        "run": spec.payload(),
        "scientific_identity": identity["scientific_identity"],
        "runtime_code_sha256": identity["runtime_code_sha256"],
        "code_commit": identity["code_commit"],
        "freeze_sha256": identity["source_freeze_sha256"],
        "config_sha256": sha256_file(config_path),
        "evaluation_bank_sha256": sha256_file(bank_path),
        "upstream_commit": UPSTREAM_COMMIT,
        "model": _native(model.metadata()),
        "parameter_count_trainable": PARAMETER_COUNTS[spec.model_id],
        "initial_state_dict_sha256": initial_state_dict_sha256,
        "rng_stream_identities": rng_stream_identities,
        "recipe": _native(asdict(recipe)),
        "recipe_noise_field_semantics": {
            "nominal_state_noise_std": "controlled_zero_disabled_not_executed",
            "effective_state_noise_std": "controlled_zero_disabled_no_state_noise_draws",
            "paper_literal_state_noise_std_provenance_only": 0.1,
            "upstream_public_config_nominal_state_noise_std_provenance_only": contract[
                "upstream_public_config_nominal_state_noise_std"
            ],
            "upstream_public_config_effective_state_noise_std_provenance_only": contract[
                "upstream_public_config_effective_state_noise_std"
            ],
            "prior_internal_source_resolved_recipe_nominal_state_noise_std_provenance_only": contract[
                "prior_internal_source_resolved_recipe_nominal_state_noise_std"
            ],
            "prior_internal_source_resolved_recipe_effective_state_noise_std_provenance_only": contract[
                "prior_internal_source_resolved_recipe_effective_state_noise_std"
            ],
            "hypothetical_nominal_0p1_effective_under_sqrt_dt_provenance_only": contract[
                "hypothetical_nominal_0p1_effective_under_sqrt_dt"
            ],
        },
        "repair_contract": contract["repair_contract"],
        "recurrent_bias_policy": contract["recurrent_bias_policy"],
        "known_upstream_oddity": contract["known_upstream_oddity"],
        "state_noise": noise_metadata(spec.model_id, spec.actual_state_noise_std),
        "optimizer": {
            "name": "Adam",
            "constant_learning_rate": spec.learning_rate,
            "early_stopping": False,
            "recurrent_weight_decay": recipe.recurrent_weight_decay,
        },
        "source_model_specific_training": {
            "upstream_public_code_target_noise_std_provenance_only": contract[
                "upstream_public_code_target_noise_std"
            ],
            "controlled_target_noise_std": recipe.target_noise_std,
            "upstream_public_code_output_dropout_provenance_only": contract[
                "upstream_public_code_output_dropout"
            ],
            "controlled_output_dropout": recipe.output_dropout,
            "upstream_public_config_effective_state_noise_std_provenance_only": contract[
                "upstream_public_config_effective_state_noise_std"
            ],
            "training_target_semantics": config["training"]["training_target_semantics"],
            "gradient_clip_norm": recipe.gradient_clip_norm,
        },
        "task": config["task"],
        "initial_state_argument": "clean_post_update_q1_target",
        "loss": "clean_all_step_masked_mse_on_cos_sin_targets",
        "evaluation": "clean_fixed_bank_no_dropout_no_state_noise",
        "environment_versions": _environment_versions(device),
        "pairing_policy": "same_seed_update_data_stream_shared_across_models_and_cells",
        "started_at_utc": _utc_now(),
    }
    atomic_json(output / "run_manifest.json", manifest)

    trace: list[dict[str, Any]] = []
    started = time.time()
    trace_interval = int(config["training"]["trace_interval"])
    validation_interval = int(config["training"]["validation_interval"])
    for update in range(1, spec.updates + 1):
        model.train()
        batch = _training_batch(spec, update, device)
        # The paper distinguishes recurrent-state perturbation from the clean
        # target. This controlled run disables that perturbation as well, so
        # clean q1 initializes the state and the loss uses the same clean target.
        training_targets = batch.output_targets
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_sequence(
            batch.inputs,
            source_targets=training_targets,
            state_noise_generator=None,
            state_noise_std_override=spec.actual_state_noise_std,
        )
        loss = source_masked_mse(prediction, training_targets, batch.mask)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"non-finite training loss at update {update}")
        loss.backward()
        _finite_model(model, gradients=True)
        gradient_norm = clip_source_gradients(model, recipe)
        optimizer.step()
        _finite_model(model)
        should_trace = update == 1 or update % trace_interval == 0 or update == spec.updates
        should_validate = update % validation_interval == 0 or update == spec.updates
        if should_trace or should_validate:
            row: dict[str, Any] = {
                "update": update,
                "train_mse": float(loss.detach().cpu()),
                "elapsed_seconds": float(time.time() - started),
            }
            if gradient_norm is not None:
                row["pre_clip_gradient_norm"] = float(gradient_norm.detach().cpu())
            if should_validate:
                row["validation"] = _evaluate(model, evaluation)
            trace.append(row)
            atomic_json(output / "training_trace.json", trace)
            atomic_json(
                output / "progress.json",
                {
                    "schema_version": 1,
                    "status": "running",
                    "run_id": spec.run_id,
                    "update": update,
                    "updates_total": spec.updates,
                    "latest": row,
                    "updated_at_utc": _utc_now(),
                },
            )

    final_metrics = _evaluate(model, evaluation)
    _assert_worker_identity(identity, config_path, require_clean=not spec.smoke)
    result = {
        "schema_version": 1,
        "status": "completed",
        "run_id": spec.run_id,
        "stage": spec.stage,
        "model_id": spec.model_id,
        "model_seed": spec.model_seed,
        "learning_rate": spec.learning_rate,
        "actual_state_noise_std": spec.actual_state_noise_std,
        "updates_completed": spec.updates,
        "final_metrics": final_metrics,
        "counts_in_denominator": True,
        "completed_at_utc": _utc_now(),
    }
    atomic_json(output / "training_trace.json", trace)
    atomic_json(output / "result.json", result)
    checkpoint = output / "checkpoint_final.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "schema_version": 1,
            "checkpoint_type": CAMPAIGN_ID,
            "run": spec.payload(),
            "result": result,
            "model_metadata": _native(model.metadata()),
            "initial_state_dict_sha256": initial_state_dict_sha256,
            "rng_stream_identities": rng_stream_identities,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
    )
    atomic_json(output / "COMPLETE", {"schema_version": 1, "run_id": spec.run_id})
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=spec.run_id,
        artifacts=[
            output / "run_manifest.json",
            output / "progress.json",
            output / "training_trace.json",
            output / "result.json",
            checkpoint,
            output / "COMPLETE",
        ],
        metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
            "outcome": "completed",
        },
    )
    return output


def _record_worker_failure(
    spec: RunSpec,
    config_path: Path,
    device_text: str,
    *,
    failure_kind: str,
    failure_message: str,
    traceback_text: str | None,
) -> Path:
    output = Path(spec.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(spec.evaluation_bank).resolve().parents[1]
    identity = strict_json_load(root / ROOT_MARKER)
    config = load_config(config_path)
    _configure_determinism(spec.model_seed)
    reconstructed_initial_model = build_model(
        config,
        spec.model_id,
        spec.learning_rate,
        spec.actual_state_noise_std,
    )
    initial_state_dict_sha256 = canonical_tensor_mapping_sha256(
        reconstructed_initial_model.state_dict()
    )
    recipe = reconstructed_initial_model.recipe
    rng_stream_identities = _rng_stream_identities(spec, recipe)
    contract = _model_contract(config, spec.model_id)
    manifest_path = output / "run_manifest.json"
    if not manifest_path.exists():
        atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "protocol_revision": PROTOCOL_REVISION,
                "run": spec.payload(),
                "scientific_identity": identity["scientific_identity"],
                "runtime_code_sha256": identity["runtime_code_sha256"],
                "code_commit": identity["code_commit"],
                "freeze_sha256": identity["source_freeze_sha256"],
                "config_sha256": sha256_file(config_path),
                "evaluation_bank_sha256": sha256_file(spec.evaluation_bank),
                "track_classification": TRACK_CLASSIFICATION,
                "upstream_commit": UPSTREAM_COMMIT,
                "model": _native(reconstructed_initial_model.metadata()),
                "parameter_count_trainable": PARAMETER_COUNTS[spec.model_id],
                "initial_state_dict_sha256": initial_state_dict_sha256,
                "rng_stream_identities": rng_stream_identities,
                "recipe": _native(asdict(recipe)),
                "state_noise": noise_metadata(
                    spec.model_id, spec.actual_state_noise_std
                ),
                "source_model_specific_training": {
                    "upstream_public_code_target_noise_std_provenance_only": contract[
                        "upstream_public_code_target_noise_std"
                    ],
                    "controlled_target_noise_std": recipe.target_noise_std,
                    "upstream_public_code_output_dropout_provenance_only": contract[
                        "upstream_public_code_output_dropout"
                    ],
                    "controlled_output_dropout": recipe.output_dropout,
                    "training_target_semantics": config["training"][
                        "training_target_semantics"
                    ],
                },
                "initial_state_argument": "clean_post_update_q1_target",
                "loss": "clean_all_step_masked_mse_on_cos_sin_targets",
                "evaluation": "clean_fixed_bank_no_dropout_no_state_noise",
                "device": device_text,
                "failure_manifest_synthesized": True,
                "environment_versions": _environment_versions(),
            },
        )
    failure = {
        "schema_version": 1,
        "status": "failed",
        "run_id": spec.run_id,
        "failure_kind": str(failure_kind),
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
        "model_id": spec.model_id,
        "model_seed": spec.model_seed,
        "learning_rate": spec.learning_rate,
        "actual_state_noise_std": spec.actual_state_noise_std,
        "updates_completed": None,
        "final_metrics": None,
        "counts_in_denominator": True,
        "completed_at_utc": _utc_now(),
    }
    atomic_json(output / "result.json", result)
    failure_checkpoint = output / "checkpoint_failure.pt"
    _atomic_torch_save(
        failure_checkpoint,
        {
            "schema_version": 1,
            "checkpoint_type": f"{CAMPAIGN_ID}_failure",
            "run": spec.payload(),
            "failure": failure,
            "state_dict": None,
            "initial_state_dict_sha256": initial_state_dict_sha256,
            "rng_stream_identities": rng_stream_identities,
        },
    )
    atomic_json(output / "FAILED", {"schema_version": 1, "run_id": spec.run_id})
    artifacts = [
        manifest_path,
        output / "failure.json",
        output / "result.json",
        failure_checkpoint,
        output / "FAILED",
    ]
    trace = output / "training_trace.json"
    if trace.is_file():
        artifacts.append(trace)
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=spec.run_id,
        artifacts=artifacts,
        metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
            "outcome": "failed",
        },
    )
    return output


def _receipt_artifact_names(output: Path) -> set[str] | None:
    try:
        receipt = strict_json_load(output / "completion_receipt.json")
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, dict):
            return None
        return set(map(str, artifacts))
    except (OSError, ValueError, TypeError):
        return None


def _verified_child(spec: RunSpec) -> bool:
    output = Path(spec.output_dir)
    try:
        result = strict_json_load(output / "result.json")
    except (OSError, ValueError, TypeError):
        return False
    status = result.get("status")
    if status == "completed":
        outcome = "completed"
        required = {
            "run_manifest.json",
            "progress.json",
            "training_trace.json",
            "result.json",
            "checkpoint_final.pt",
            "COMPLETE",
        }
        checkpoint_path = output / "checkpoint_final.pt"
    elif status == "failed":
        outcome = "failed"
        required = {
            "run_manifest.json",
            "failure.json",
            "result.json",
            "checkpoint_failure.pt",
            "FAILED",
        }
        if (output / "training_trace.json").is_file():
            required.add("training_trace.json")
        checkpoint_path = output / "checkpoint_failure.pt"
    else:
        return False
    valid, _ = verify_completion_receipt(
        output / "completion_receipt.json",
        expected_job_id=spec.run_id,
        expected_metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
            "outcome": outcome,
        },
    )
    if not valid or _receipt_artifact_names(output) != required:
        return False
    try:
        manifest = strict_json_load(output / "run_manifest.json")
        root = Path(spec.evaluation_bank).resolve().parents[1]
        identity = strict_json_load(root / ROOT_MARKER)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        return False
    if checkpoint.get("run") != spec.payload():
        return False
    try:
        config = load_config(root / "inputs" / DEFAULT_CONFIG.name)
        _configure_determinism(spec.model_seed)
        reconstructed = build_model(
            config,
            spec.model_id,
            spec.learning_rate,
            spec.actual_state_noise_std,
        )
        expected_initial_hash = canonical_tensor_mapping_sha256(
            reconstructed.state_dict()
        )
        expected_rng_streams = _rng_stream_identities(spec, reconstructed.recipe)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        return False
    if status == "completed":
        checkpoint_result = checkpoint.get("result")
        if checkpoint.get("checkpoint_type") != CAMPAIGN_ID or checkpoint_result != result:
            return False
        try:
            state_dict = checkpoint.get("state_dict")
            if not isinstance(state_dict, Mapping):
                return False
            reconstructed.load_state_dict(state_dict, strict=True)
            _finite_model(reconstructed)
            marker = strict_json_load(output / "COMPLETE")
        except (OSError, ValueError, RuntimeError, TypeError, KeyError):
            return False
        metrics = result.get("final_metrics")
        if not isinstance(metrics, dict):
            return False
        for key in ("mse", "nmse_db", "masked_mse", "masked_nmse_db"):
            if key not in metrics or not math.isfinite(float(metrics[key])):
                return False
        if (
            metrics["mse"] != metrics["masked_mse"]
            or metrics["nmse_db"] != metrics["masked_nmse_db"]
        ):
            return False
        outcome_files = marker.get("run_id") == spec.run_id
        result_identity = (
            result.get("stage") == spec.stage
            and result.get("model_id") == spec.model_id
            and result.get("model_seed") == spec.model_seed
            and result.get("learning_rate") == spec.learning_rate
            and result.get("actual_state_noise_std") == spec.actual_state_noise_std
            and result.get("updates_completed") == spec.updates
            and result.get("counts_in_denominator") is True
            and checkpoint.get("model_metadata") == _native(reconstructed.metadata())
            and manifest.get("model") == _native(reconstructed.metadata())
            and manifest.get("initial_state_dict_sha256") == expected_initial_hash
            and checkpoint.get("initial_state_dict_sha256") == expected_initial_hash
            and manifest.get("rng_stream_identities") == expected_rng_streams
            and checkpoint.get("rng_stream_identities") == expected_rng_streams
        )
    else:
        if checkpoint.get("checkpoint_type") != f"{CAMPAIGN_ID}_failure":
            return False
        try:
            failure = strict_json_load(output / "failure.json")
        except (OSError, ValueError):
            return False
        if checkpoint.get("failure") != failure or result.get("counts_in_denominator") is not True:
            return False
        try:
            marker = strict_json_load(output / "FAILED")
        except (OSError, ValueError):
            return False
        outcome_files = marker.get("run_id") == spec.run_id
        result_identity = (
            result.get("stage") == spec.stage
            and result.get("model_id") == spec.model_id
            and result.get("model_seed") == spec.model_seed
            and result.get("learning_rate") == spec.learning_rate
            and result.get("actual_state_noise_std") == spec.actual_state_noise_std
            and manifest.get("model") == _native(reconstructed.metadata())
            and manifest.get("parameter_count_trainable")
            == PARAMETER_COUNTS[spec.model_id]
            and manifest.get("initial_state_dict_sha256") == expected_initial_hash
            and checkpoint.get("initial_state_dict_sha256") == expected_initial_hash
            and manifest.get("rng_stream_identities") == expected_rng_streams
            and checkpoint.get("rng_stream_identities") == expected_rng_streams
            and manifest.get("recipe") == _native(asdict(reconstructed.recipe))
            and manifest.get("state_noise")
            == noise_metadata(spec.model_id, spec.actual_state_noise_std)
            and isinstance(manifest.get("source_model_specific_training"), Mapping)
            and manifest["source_model_specific_training"].get(
                "controlled_target_noise_std"
            )
            == 0.0
            and manifest["source_model_specific_training"].get(
                "controlled_output_dropout"
            )
            == 0.0
            and manifest.get("initial_state_argument")
            == "clean_post_update_q1_target"
            and manifest.get("loss")
            == "clean_all_step_masked_mse_on_cos_sin_targets"
        )
    return bool(
        outcome_files
        and result_identity
        and manifest.get("run") == spec.payload()
        and result.get("run_id") == spec.run_id
        and manifest.get("protocol_revision") == PROTOCOL_REVISION
        and manifest.get("track_classification") == TRACK_CLASSIFICATION
        and manifest.get("scientific_identity") == identity.get("scientific_identity")
        and manifest.get("runtime_code_sha256") == identity.get("runtime_code_sha256")
        and manifest.get("code_commit") == identity.get("code_commit")
        and manifest.get("config_sha256") == identity.get("source_config_sha256")
        and manifest.get("freeze_sha256") == identity.get("source_freeze_sha256")
        and manifest.get("evaluation_bank_sha256") == sha256_file(spec.evaluation_bank)
    )


def _write_or_verify(path: Path, payload: Mapping[str, Any]) -> None:
    native = _native(payload)
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
    normalized = str(text).strip().lower()
    if normalized == "cpu":
        return ("cpu",)
    slots = tuple(item.strip() for item in normalized.split(",") if item.strip())
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
    """Run/resume children; infrastructure failures remain retryable."""

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
    try:
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
                    "repro.sagodi_protocol.source_repaired_baselines_v6",
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
                    environment["CALRU_PHYSICAL_GPU_SLOT"] = slot
                process = subprocess.Popen(
                    command,
                    cwd=repo,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running[slot] = (process, spec, handle)
            if not running:
                continue
            time.sleep(0.2)
            for slot, (process, spec, handle) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del running[slot]
                if code != 0 or not _verified_child(spec):
                    raise RuntimeError(
                        f"retryable infrastructure/worker failure for {spec.run_id}; "
                        f"exit={code}; no permanent scientific failure was synthesized"
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
    finally:
        for process, _, handle in running.values():
            if process.poll() is None:
                process.terminate()
            handle.close()


def _children_binding(specs: Sequence[RunSpec]) -> dict[str, Any]:
    children: list[dict[str, Any]] = []
    for spec in specs:
        output = Path(spec.output_dir)
        result = strict_json_load(output / "result.json")
        checkpoint = (
            output / "checkpoint_final.pt"
            if result.get("status") == "completed"
            else output / "checkpoint_failure.pt"
        )
        children.append(
            {
                "run_id": spec.run_id,
                "status": result.get("status"),
                "completion_receipt_sha256": sha256_file(output / "completion_receipt.json"),
                "checkpoint_name": checkpoint.name,
                "checkpoint_sha256": sha256_file(checkpoint),
                "result_sha256": sha256_file(output / "result.json"),
            }
        )
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "transitively_binds_every_child_receipt_checkpoint_and_result": True,
        "child_count": len(children),
        "children": children,
    }


def _finalize(
    stage_root: Path,
    stage: str,
    specs: Sequence[RunSpec],
    extras: Sequence[Path],
) -> None:
    missing = [spec.run_id for spec in specs if not _verified_child(spec)]
    if missing:
        raise RuntimeError(f"cannot finalize {stage}; missing {missing[:3]}")
    binding = stage_root / "children_binding.json"
    _write_or_verify(binding, _children_binding(specs))
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
        artifacts=[stage_root / "plan.json", binding, complete, *extras],
        metadata={"campaign_id": CAMPAIGN_ID, "stage": stage, "run_count": len(specs)},
    )


def _plan_specs(stage_root: Path) -> tuple[RunSpec, ...]:
    plan = strict_json_load(stage_root / "plan.json")
    if plan.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("stage plan campaign differs")
    specs = tuple(RunSpec(**item) for item in plan["runs"])
    if int(plan.get("run_count", -1)) != len(specs):
        raise ValueError("stage plan count differs")
    return specs


def _stage_receipt_names(stage_root: Path) -> set[str] | None:
    return _receipt_artifact_names(stage_root)


def _expected_stage_artifacts(stage: str, *, scientific_pass: bool = False) -> set[str]:
    common = {"plan.json", "children_binding.json", "COMPUTATION_COMPLETE"}
    if stage == "smoke":
        return common | {"summary.json"}
    if stage == "sentinel":
        return common | {"sentinel_summary.json"}
    if stage == "fanout":
        return common | {
            "parent_sentinel_binding.json",
            "hyperparameter_selection.json",
        }
    if stage == "main":
        result = common | {
            "parent_fanout_binding.json",
            "summary.json",
            "scientific_gate.json",
        }
        if scientific_pass:
            result.add("SCIENTIFIC_PASS")
        return result
    raise ValueError("unknown stage")


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
        specs = _plan_specs(stage_root)
        if len(specs) != expected_runs or any(spec.stage != stage for spec in specs):
            return False
        if not all(_verified_child(spec) for spec in specs):
            return False
        if strict_json_load(stage_root / "children_binding.json") != _children_binding(specs):
            return False
        root = stage_root.parent
        config = load_config(root / "inputs" / DEFAULT_CONFIG.name)
        for bank_path in sorted({spec.evaluation_bank for spec in specs}):
            bank = load_fixed_bank(bank_path)
            _validate_bank(
                bank,
                trials=bank.batch_size,
                config=config,
                purpose=Path(bank_path).stem,
            )

        scientific_pass = False
        if stage == "smoke":
            expected_summary = {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "scientific_result": False,
                "runs": [
                    {
                        "run_id": spec.run_id,
                        "status": _result(spec).get("status"),
                        "final_metrics": _result(spec).get("final_metrics"),
                    }
                    for spec in specs
                ],
            }
            if strict_json_load(stage_root / "summary.json") != expected_summary:
                return False
        elif stage == "sentinel":
            recomputed = summarize_sentinel(specs, config)
            if strict_json_load(stage_root / "sentinel_summary.json") != recomputed:
                return False
        elif stage == "fanout":
            sentinel_root = root / "sentinel"
            sentinel_expected = len(MODEL_IDS) * len(
                config["hyperparameter_tuning"]["learning_rate_grid"]
            )
            if not _stage_valid(sentinel_root, "sentinel", sentinel_expected):
                return False
            sentinel_specs = _plan_specs(sentinel_root)
            sentinel_summary_path = sentinel_root / "sentinel_summary.json"
            parent_expected = {
                "schema_version": 1,
                "sentinel_completion_receipt_sha256": sha256_file(
                    sentinel_root / "completion_receipt.json"
                ),
                "sentinel_summary_sha256": sha256_file(sentinel_summary_path),
            }
            if strict_json_load(stage_root / "parent_sentinel_binding.json") != parent_expected:
                return False
            recomputed = select_hyperparameters(sentinel_specs, specs, config)
            if strict_json_load(stage_root / "hyperparameter_selection.json") != recomputed:
                return False
        elif stage == "main":
            fanout_root = root / "fanout"
            if not _stage_valid(
                fanout_root,
                "fanout",
                len(MODEL_IDS)
                * len(config["hyperparameter_tuning"]["learning_rate_grid"])
                * len(config["hyperparameter_tuning"]["fanout_seeds"]),
            ):
                return False
            selection_path = fanout_root / "hyperparameter_selection.json"
            parent_expected = {
                "schema_version": 1,
                "fanout_completion_receipt_sha256": sha256_file(
                    fanout_root / "completion_receipt.json"
                ),
                "hyperparameter_selection_sha256": sha256_file(selection_path),
            }
            if strict_json_load(stage_root / "parent_fanout_binding.json") != parent_expected:
                return False
            summary = summarize_main(specs, config)
            gate = scientific_gate(summary)
            if strict_json_load(stage_root / "summary.json") != summary:
                return False
            if strict_json_load(stage_root / "scientific_gate.json") != gate:
                return False
            scientific_pass = bool(gate["all_required_gates_pass"])
            marker = stage_root / "SCIENTIFIC_PASS"
            if marker.is_file() != scientific_pass:
                return False
            if scientific_pass:
                marker_payload = strict_json_load(marker)
                if (
                    marker_payload.get("scientific_gate_sha256")
                    != sha256_file(stage_root / "scientific_gate.json")
                    or marker_payload.get("all_baseline_models_have_analysis_eligible_subset")
                    is not True
                ):
                    return False
        else:
            return False
        if _stage_receipt_names(stage_root) != _expected_stage_artifacts(
            stage, scientific_pass=scientific_pass
        ):
            return False
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        return False
    return True


def require_scientific_pass(artifact_root: Path | str) -> Path:
    root = Path(artifact_root).expanduser().resolve()
    main_root = root / "main"
    if not _stage_valid(main_root, "main", len(MODEL_IDS) * 3):
        raise RuntimeError("v6 main computation/receipt chain is not valid")
    gate_path = main_root / "scientific_gate.json"
    gate = strict_json_load(gate_path)
    marker = main_root / "SCIENTIFIC_PASS"
    if not bool(gate.get("all_required_gates_pass")) or not marker.is_file():
        raise RuntimeError("v6 computation completed without an eligible seed for every model")
    return marker


def require_verified_main(artifact_root: Path | str) -> Path:
    """Require complete, current main receipts without requiring every model eligible."""

    root = Path(artifact_root).expanduser().resolve()
    main_root = root / "main"
    if not _stage_valid(main_root, "main", len(MODEL_IDS) * 3):
        raise RuntimeError("v6 baseline main computation/receipt chain is not valid")
    return main_root / "COMPUTATION_COMPLETE"


def run_stage(
    stage: str,
    artifact_root: Path,
    config_source: Path,
    slots: Sequence[str],
) -> Path:
    if stage not in {"smoke", "sentinel", "fanout", "main"}:
        raise ValueError("stage must be smoke, sentinel, fanout, or main")
    root = artifact_root.expanduser().resolve()
    config, copied = _prepare_root(root, config_source, require_clean=stage != "smoke")

    if stage == "smoke":
        bank = _ensure_bank(root, config, "smoke")
        specs = build_smoke_plan(root, config, bank)
        stage_root = root / "smoke"
        if _stage_valid(stage_root, stage, len(specs)):
            return stage_root
        _run_specs(specs, copied, slots)
        summary_path = stage_root / "summary.json"
        _write_or_verify(
            summary_path,
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "scientific_result": False,
                "runs": [
                    {
                        "run_id": spec.run_id,
                        "status": _result(spec).get("status"),
                        "final_metrics": _result(spec).get("final_metrics"),
                    }
                    for spec in specs
                ],
            },
        )
        _finalize(stage_root, stage, specs, [summary_path])
        return stage_root

    tuning_bank = _ensure_bank(root, config, "tuning")
    sentinel_root = root / "sentinel"
    sentinel_specs = build_sentinel_plan(root, config, tuning_bank)
    if stage == "sentinel":
        if _stage_valid(sentinel_root, stage, len(sentinel_specs)):
            return sentinel_root
        _run_specs(sentinel_specs, copied, slots)
        summary_path = sentinel_root / "sentinel_summary.json"
        _write_or_verify(summary_path, summarize_sentinel(sentinel_specs, config))
        _finalize(sentinel_root, stage, sentinel_specs, [summary_path])
        return sentinel_root

    if not _stage_valid(
        sentinel_root,
        "sentinel",
        len(MODEL_IDS) * len(config["hyperparameter_tuning"]["learning_rate_grid"]),
    ):
        raise RuntimeError(f"{stage} is blocked until verified sentinel completes")
    sentinel_summary_path = sentinel_root / "sentinel_summary.json"
    sentinel_summary = strict_json_load(sentinel_summary_path)
    fanout_specs = build_fanout_plan(root, config, tuning_bank, sentinel_summary)
    fanout_root = root / "fanout"
    if stage == "fanout":
        if _stage_valid(fanout_root, stage, len(fanout_specs)):
            return fanout_root
        _run_specs(fanout_specs, copied, slots)
        parent_path = fanout_root / "parent_sentinel_binding.json"
        _write_or_verify(
            parent_path,
            {
                "schema_version": 1,
                "sentinel_completion_receipt_sha256": sha256_file(
                    sentinel_root / "completion_receipt.json"
                ),
                "sentinel_summary_sha256": sha256_file(sentinel_summary_path),
            },
        )
        selection_path = fanout_root / "hyperparameter_selection.json"
        _write_or_verify(
            selection_path,
            select_hyperparameters(sentinel_specs, fanout_specs, config),
        )
        _finalize(fanout_root, stage, fanout_specs, [parent_path, selection_path])
        return fanout_root

    if not _stage_valid(
        fanout_root,
        "fanout",
        len(MODEL_IDS)
        * len(config["hyperparameter_tuning"]["learning_rate_grid"])
        * len(config["hyperparameter_tuning"]["fanout_seeds"]),
    ):
        raise RuntimeError("main is blocked until verified fanout completes")
    selection_path = fanout_root / "hyperparameter_selection.json"
    selection = strict_json_load(selection_path)
    main_bank = _ensure_bank(root, config, "main_test")
    main_specs = build_main_plan(root, config, main_bank, selection)
    main_root = root / "main"
    if _stage_valid(main_root, stage, len(main_specs)):
        return main_root
    _run_specs(main_specs, copied, slots)
    parent_path = main_root / "parent_fanout_binding.json"
    _write_or_verify(
        parent_path,
        {
            "schema_version": 1,
            "fanout_completion_receipt_sha256": sha256_file(
                fanout_root / "completion_receipt.json"
            ),
            "hyperparameter_selection_sha256": sha256_file(selection_path),
        },
    )
    summary_path = main_root / "summary.json"
    summary = summarize_main(main_specs, config)
    _write_or_verify(summary_path, summary)
    gate_path = main_root / "scientific_gate.json"
    gate = scientific_gate(summary)
    _write_or_verify(gate_path, gate)
    extras = [parent_path, summary_path, gate_path]
    if bool(gate["all_required_gates_pass"]):
        marker = main_root / "SCIENTIFIC_PASS"
        _write_or_verify(
            marker,
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "scientific_gate_sha256": sha256_file(gate_path),
                "all_baseline_models_have_analysis_eligible_subset": True,
                "interpretation": "eligible_seed_subsets_only",
            },
        )
        extras.append(marker)
    elif (main_root / "SCIENTIFIC_PASS").exists():
        raise RuntimeError("stale SCIENTIFIC_PASS exists for failed current gate")
    _finalize(main_root, stage, main_specs, extras)
    return main_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "sentinel", "fanout", "main"))
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
            raise ValueError("worker mode cannot include campaign options")
        spec = RunSpec(**strict_json_load(args.worker_spec))
        config_path = args.config.resolve(strict=True)
        try:
            _train_worker(spec, config_path, str(args.device))
        except FloatingPointError as error:
            # Only a deterministic model numerical failure becomes a
            # denominator-bearing scientific failure. OOM/SIGKILL/interrupt
            # and other infrastructure errors propagate and remain retryable.
            identity = strict_json_load(
                Path(spec.evaluation_bank).resolve().parents[1] / ROOT_MARKER
            )
            _assert_worker_identity(identity, config_path, require_clean=not spec.smoke)
            _record_worker_failure(
                spec,
                config_path,
                str(args.device),
                failure_kind=type(error).__name__,
                failure_message=str(error),
                traceback_text=traceback.format_exc(),
            )
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
