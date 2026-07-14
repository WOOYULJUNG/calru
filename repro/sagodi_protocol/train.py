"""Single-run trainer for the phase-gated Ságodi ring pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import (
    atomic_json,
    canonical_hash,
    derived_seed,
    sha256_file,
    write_completion_receipt,
)
from .config import (
    DEFAULT_PROTOCOL_PATH,
    SAGODI_LR_SELECTION_TRACK,
    SAGODI_PRIMARY_LR_SELECTION_TRACK,
    SAGODI_PRIMARY_MAIN_TRACK,
    AngularTaskSpec,
    load_protocol,
    protocol_fingerprint,
)
from .metrics import masked_mse, task_metrics
from .models import (
    ProtocolModel,
    build_protocol_model,
    checkpoint_payload,
    model_config_from_protocol,
)
from .tasks import Batch, angular_integration, load_fixed_bank, metadata_payload


@dataclass(frozen=True)
class TrainSpec:
    model_name: str
    model_seed: int
    learning_rate: float
    output_dir: Path
    protocol_path: Path = DEFAULT_PROTOCOL_PATH
    evaluation_bank: Path | None = None
    state_spec: Path | None = None
    campaign_identity: str | None = None
    steps_override: int | None = None
    batch_override: int | None = None
    device: str = "cuda:0"
    smoke: bool = False


class NonFiniteTrainingError(RuntimeError):
    """Raised before a corrupt run can be checkpointed or receipted."""


def _frozen_training_model_seeds(protocol: dict[str, Any]) -> tuple[int, ...]:
    """Return the seed set authorized by the protocol's executable track.

    The LR-selection freeze deliberately has no ``pilot_model_seeds`` field.
    Keeping this resolution next to the trainer prevents a validated selector
    run from falling back to the legacy pilot seed namespace.
    """

    track = protocol["phase1_ring_pilot"]["protocol_track"]
    if track in {SAGODI_LR_SELECTION_TRACK, SAGODI_PRIMARY_LR_SELECTION_TRACK}:
        seed_key = "selection_model_seeds"
    elif track == SAGODI_PRIMARY_MAIN_TRACK:
        seed_key = "main_model_seeds"
    else:
        seed_key = "pilot_model_seeds"
    raw = protocol["seed_policy"].get(seed_key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"validated protocol track {track!r} has no nonempty {seed_key!r}"
        )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise ValueError(f"{seed_key} must contain only integer model seeds")
    return tuple(int(value) for value in raw)


def _require_finite_tensor(value: torch.Tensor, label: str) -> None:
    if not torch.isfinite(value).all().item():
        raise NonFiniteTrainingError(f"non-finite tensor detected: {label}")


def _require_finite_model(model: ProtocolModel, label: str, *, gradients: bool = False) -> None:
    for name, parameter in model.named_parameters():
        _require_finite_tensor(parameter.detach(), f"{label}.parameter.{name}")
        if gradients and parameter.grad is not None:
            _require_finite_tensor(parameter.grad.detach(), f"{label}.gradient.{name}")
    for name, buffer in model.named_buffers():
        if buffer.is_floating_point():
            _require_finite_tensor(buffer.detach(), f"{label}.buffer.{name}")


def _require_finite_optimizer(optimizer: torch.optim.Optimizer, label: str) -> None:
    for parameter_index, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                _require_finite_tensor(value.detach(), f"{label}.state{parameter_index}.{key}")


def _require_finite_payload(value: Any, label: str = "payload") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, np.integer)):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise NonFiniteTrainingError(f"non-finite scalar detected: {label}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_payload(item, f"{label}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _require_finite_payload(item, f"{label}[{index}]")
        return
    raise TypeError(f"unsupported payload value at {label}: {type(value).__name__}")


def _configure_determinism(seed: int) -> None:
    # The campaign provenance freezes this exact cuBLAS workspace contract.
    # ``setdefault`` would allow a caller's ambient value to silently change
    # determinism while leaving otherwise identical receipts.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _initial_embedding(batch: Batch, device: torch.device) -> torch.Tensor:
    initial = torch.as_tensor(batch.metadata["initial_latents"], dtype=batch.inputs.dtype, device=device)
    if initial.ndim != 2:
        raise ValueError("initial_latents must be batch x latent dimension")
    return torch.stack([torch.cos(initial), torch.sin(initial)], dim=-1).reshape(initial.shape[0], -1)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _normalize_campaign_identity(value: str | None, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise ValueError("full pilot runs require --campaign-identity")
        return None
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("campaign identity must be a 64-character lowercase SHA-256 digest")
    return normalized


def _build_optimizer(
    model: ProtocolModel,
    optimizer_config: dict[str, Any],
    *,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Construct exactly the optimizer declared by the active freeze."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    kwargs: dict[str, Any] = {
        "lr": float(learning_rate),
        "betas": tuple(float(value) for value in optimizer_config["betas"]),
        "weight_decay": float(optimizer_config["weight_decay"]),
    }
    if "epsilon" in optimizer_config:
        kwargs["eps"] = float(optimizer_config["epsilon"])
    name = str(optimizer_config["name"])
    if name == "Adam":
        return torch.optim.Adam(parameters, **kwargs)
    if name == "AdamW":
        return torch.optim.AdamW(parameters, **kwargs)
    raise ValueError(f"unsupported frozen optimizer {name!r}")


def _load_and_validate_state_spec(
    path: Path | None,
    model: ProtocolModel,
    *,
    required: bool,
) -> tuple[str | None, dict[str, Any] | None]:
    if path is None:
        if required:
            raise ValueError("full pilot runs require --state-spec from the Phase-0 gate")
        return None, None
    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported Phase-0 state-spec schema")
    expected = {
        "external_input_dimension": int(model.input_dim),
        "primary_dimension": int(model.primary_state_size),
        "reported_dimension": int(model.reported_state_size),
        "carry_stream": bool(getattr(model.core, "carry_stream", False)),
        "model_class": model.core.__class__.__name__,
        "primary_state": "full_recurrent_markov_state",
        "zero_input_definition": "literal_all_zero_tensor_passed_to_model.step",
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise ValueError(
                f"Phase-0 state spec mismatch for {key}: "
                f"{payload.get(key)!r} != {expected_value!r}"
            )
    expected_overwritten = ["stream"] if model.reported_state_size > model.primary_state_size else []
    if payload.get("overwritten_components") != expected_overwritten:
        raise ValueError("Phase-0 state spec has inconsistent overwritten components")
    return sha256_file(source), payload


def _validate_evaluation_batch(
    batch: Batch,
    protocol: dict[str, Any],
    *,
    require_full_protocol_shape: bool,
) -> None:
    task_spec = AngularTaskSpec.from_protocol(protocol)
    evaluation = protocol["evaluation"]
    metadata = batch.metadata
    expected_metadata = {
        "task_name": "angular_integration",
        "task_seed": int(protocol["seed_policy"]["task_seed"]),
        "sample_seed": int(protocol["seed_policy"]["evaluation_bank_seed"]),
        "init_mode": task_spec.init_mode_for_generator,
        "horizon": task_spec.sequence_steps,
        "input_dimension": task_spec.input_dimension,
        "output_dimension": task_spec.output_dimension,
        "target_indexing": "post_velocity_update",
        "delta_t": task_spec.delta_t,
        "gp_grid": "linspace(-1,1,T)",
        "gp_grid_start": task_spec.gp_grid_start,
        "gp_grid_stop": task_spec.gp_grid_stop,
        "gp_grid_endpoint": task_spec.gp_grid_endpoint,
        "gp_length_scale": task_spec.gp_length_scale,
        "gp_std": task_spec.gp_marginal_standard_deviation,
        "gp_cholesky_jitter": task_spec.gp_cholesky_jitter,
        "q0_distribution": task_spec.q0_distribution,
        "q0_low": task_spec.q0_low,
        "q0_high": task_spec.q0_high,
        "q0_high_inclusive": task_spec.q0_high_inclusive,
        "target_indexing_contract": task_spec.velocity_token_target,
        "loss_mask_mode": task_spec.loss_mask,
    }
    legacy_metadata = "resolved_task_spec_sha256" not in metadata
    legacy_optional = {
        "gp_grid_start",
        "gp_grid_stop",
        "gp_grid_endpoint",
        "q0_distribution",
        "q0_low",
        "q0_high",
        "q0_high_inclusive",
        "target_indexing_contract",
        "loss_mask_mode",
    }
    for key, expected in expected_metadata.items():
        if legacy_metadata and key in legacy_optional and key not in metadata:
            continue
        if metadata.get(key) != expected:
            raise ValueError(
                f"evaluation bank metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {expected!r}"
            )
    if legacy_metadata:
        if metadata.get("task_version") != "sagodi-protocol-v1":
            raise ValueError("unbound evaluation bank is not a recognized legacy v1 bank")
    else:
        if metadata.get("resolved_task_spec_sha256") != task_spec.fingerprint():
            raise ValueError("evaluation bank resolved task spec SHA-256 mismatch")
        if metadata_payload(metadata.get("resolved_task_spec")) != task_spec.resolved_payload():
            raise ValueError("evaluation bank resolved task spec payload mismatch")
    if require_full_protocol_shape and batch.batch_size != int(evaluation["id_test_trials"]):
        raise ValueError(
            f"evaluation bank has {batch.batch_size} trials, expected "
            f"{evaluation['id_test_trials']}"
        )
    expected_shapes = {
        "inputs": (batch.time_steps, batch.batch_size, task_spec.input_dimension),
        "output_targets": (batch.time_steps, batch.batch_size, task_spec.output_dimension),
        "latent_targets": (batch.time_steps, batch.batch_size, task_spec.latent_dimension),
        "mask": (batch.time_steps, batch.batch_size, task_spec.output_dimension),
    }
    for name, expected in expected_shapes.items():
        value = getattr(batch, name)
        if tuple(value.shape) != expected:
            raise ValueError(f"evaluation bank {name} shape {tuple(value.shape)} != {expected}")
        _require_finite_tensor(value, f"evaluation_bank.{name}")


def _load_evaluation_bank(
    path: Path | None,
    protocol: dict[str, Any],
    *,
    device: torch.device,
    smoke: bool,
) -> tuple[Batch, str | None, str]:
    if path is not None:
        source = Path(path).expanduser().resolve(strict=True)
        digest = sha256_file(source)
        batch = load_fixed_bank(source, device=device)
        # load_fixed_bank verifies the mandatory sidecar before deserializing.
        if sha256_file(source) != digest:
            raise RuntimeError("evaluation bank changed while it was being loaded")
        _validate_evaluation_batch(batch, protocol, require_full_protocol_shape=not smoke)
        return batch, digest, str(source)
    if not smoke:
        raise ValueError("full pilot runs require --evaluation-bank")
    seeds = protocol["seed_policy"]
    task_spec = AngularTaskSpec.from_protocol(protocol)
    batch = angular_integration(
        32,
        int(seeds["task_seed"]),
        **task_spec.generator_kwargs(),
        stream_key=("fixed_evaluation", int(seeds["evaluation_bank_seed"])),
        device=device,
    )
    for name in ("inputs", "output_targets", "latent_targets", "mask"):
        _require_finite_tensor(getattr(batch, name), f"smoke_evaluation.{name}")
    return batch, None, "generated_smoke_only_not_a_shared_bank"


def _expected_rp_steps(
    model: ProtocolModel,
    *,
    steps: int,
    warmup: int,
    interval: int,
    smoke: bool,
    enabled_by_protocol: bool = True,
) -> tuple[int, ...]:
    if not model.rp_enabled or not enabled_by_protocol:
        return ()
    effective_warmup = 0 if smoke else int(warmup)
    effective_interval = 1 if smoke else int(interval)
    return tuple(
        step
        for step in range(1, int(steps) + 1)
        if step > effective_warmup and (step - effective_warmup) % effective_interval == 0
    )


def _roll_blank(model: ProtocolModel, state: torch.Tensor, steps: int) -> torch.Tensor:
    blank = torch.zeros(state.shape[0], model.input_dim, device=state.device, dtype=state.dtype)
    current = state
    for _ in range(int(steps)):
        current = model.step(blank, current)
        _require_finite_tensor(current, "retention_plasticity.blank_roll_state")
    return current


@torch.no_grad()
def _retention_plasticity_call(
    model: ProtocolModel,
    batch: Batch,
    *,
    blank_horizon: int,
    eta_lambda: float,
    damage_epsilon: float,
) -> dict[str, float]:
    if not model.rp_enabled:
        raise ValueError("RP call requested for a model with RP disabled")
    if not math.isfinite(float(eta_lambda)) or float(eta_lambda) <= 0.0:
        raise ValueError("RP eta_lambda must be finite and positive")
    if not math.isfinite(float(damage_epsilon)):
        raise ValueError("RP damage_epsilon must be finite")
    _require_finite_model(model, "rp_pre_update")
    initial = _initial_embedding(batch, batch.inputs.device)
    _, states = model.forward_sequence(batch.inputs, initial_memory=initial, return_states=True)
    _require_finite_tensor(states, "retention_plasticity.probe_states")
    state = states[-1]
    target = batch.output_targets[-1]
    clean_final = _roll_blank(model, state, blank_horizon)
    clean_prediction = model.decode(clean_final)
    clean_energy = (clean_prediction - target).square().sum(dim=-1).mean()
    _require_finite_tensor(clean_prediction, "retention_plasticity.clean_prediction")
    _require_finite_tensor(clean_energy, "retention_plasticity.clean_energy")

    all_scores: list[torch.Tensor] = []
    for recurrence, state_slice in model.pan_recs_with_slices():
        hidden = int(state_slice.stop - state_slice.start)
        batch_size, total_state = state.shape
        ablated = state.unsqueeze(0).expand(hidden, batch_size, total_state).clone()
        coordinate = torch.arange(hidden, device=state.device)
        ablated[coordinate, :, state_slice.start + coordinate] = 0.0
        final = _roll_blank(model, ablated.reshape(hidden * batch_size, total_state), blank_horizon)
        prediction = model.decode(final).reshape(hidden, batch_size, model.output_dim)
        energy = (prediction - target.unsqueeze(0)).square().sum(dim=-1).mean(dim=1)
        damage = energy - clean_energy
        _require_finite_tensor(prediction, "retention_plasticity.ablated_prediction")
        _require_finite_tensor(energy, "retention_plasticity.ablated_energy")
        _require_finite_tensor(damage, "retention_plasticity.damage")
        score = damage - float(damage_epsilon)
        _require_finite_tensor(score, "retention_plasticity.score")
        recurrence.update_theta(score, float(eta_lambda))
        _require_finite_tensor(recurrence.theta, "retention_plasticity.updated_theta")
        _require_finite_tensor(recurrence.lam_mag(), "retention_plasticity.updated_retention")
        all_scores.append(damage)
    if not all_scores:
        raise RuntimeError("CA-LRU exposes no RP coordinates")
    values = torch.cat(all_scores)
    result = {
        "clean_rmse": float(torch.sqrt(clean_energy / model.output_dim).cpu()),
        "damage_mean": float(values.mean().cpu()),
        "damage_max": float(values.max().cpu()),
        "damage_positive_fraction": float((values > 0).float().mean().cpu()),
    }
    _require_finite_model(model, "rp_post_update")
    _require_finite_payload(result, "rp_result")
    return result


@torch.no_grad()
def _evaluate(model: ProtocolModel, batch: Batch, *, device: torch.device) -> dict[str, Any]:
    initial = _initial_embedding(batch, device)
    prediction = model.forward_sequence(batch.inputs, initial_memory=initial)
    _require_finite_tensor(prediction, "final_evaluation.prediction")
    result = task_metrics(prediction, batch.output_targets, batch.mask, batch.latent_targets)
    _require_finite_payload(result, "final_evaluation.metrics")
    return result


def train_one(spec: TrainSpec) -> Path:
    protocol = load_protocol(spec.protocol_path)
    protocol_file_sha256 = sha256_file(spec.protocol_path)
    protocol_canonical_fingerprint = protocol_fingerprint(protocol)
    phase = protocol["phase1_ring_pilot"]
    task_spec = AngularTaskSpec.from_protocol(protocol)
    task_spec_payload = task_spec.resolved_payload()
    task_spec_sha256 = task_spec.fingerprint()
    training = phase["training"]
    seed_policy = protocol["seed_policy"]
    primary_main = phase["protocol_track"] == SAGODI_PRIMARY_MAIN_TRACK
    artifact_role = "primary_main_training" if primary_main else "pilot_training"
    campaign_type = "sagodi_primary_main_v3" if primary_main else "sagodi_protocol_pilot"
    pilot_only = not primary_main
    model_ids = [item["id"] for item in phase["models"]]
    if spec.model_name not in model_ids:
        raise ValueError(f"model {spec.model_name!r} is outside the frozen pilot")
    allowed_model_seeds = _frozen_training_model_seeds(protocol)
    if int(spec.model_seed) not in allowed_model_seeds and not spec.smoke:
        raise ValueError("model seed is outside the frozen training set")
    if float(spec.learning_rate) not in training["learning_rate"]["active_launch_values"] and not spec.smoke:
        raise ValueError("learning rate is outside active frozen values")
    selected_by_model = training["learning_rate"].get("selected_by_model")
    if primary_main:
        if not isinstance(selected_by_model, dict) or set(selected_by_model) != set(model_ids):
            raise ValueError("primary main protocol lacks the complete modelwise LR mapping")
        expected_learning_rate = float(selected_by_model[spec.model_name])
        if not math.isclose(
            float(spec.learning_rate), expected_learning_rate, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError(
                f"primary main {spec.model_name} requires selector-bound "
                f"learning rate {expected_learning_rate:g}"
            )
    if not spec.smoke and (spec.steps_override is not None or spec.batch_override is not None):
        raise ValueError("--steps and --batch-size overrides are smoke-only")
    campaign_identity = _normalize_campaign_identity(
        spec.campaign_identity, required=not spec.smoke
    )

    output_dir = Path(spec.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completion = output_dir / "completion_receipt.json"
    if completion.exists():
        raise FileExistsError(f"completed or partial run already exists: {output_dir}")
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite partial run: {output_dir}")

    model_seed = int(spec.model_seed)
    _configure_determinism(model_seed)
    device = torch.device(spec.device)
    model_config = model_config_from_protocol(protocol, spec.model_name)
    model = build_protocol_model(model_config).to(device)
    _require_finite_model(model, "initialized_model")
    state_spec_sha256, state_spec_payload = _load_and_validate_state_spec(
        spec.state_spec,
        model,
        required=not spec.smoke,
    )
    evaluation_batch, evaluation_bank_sha256, evaluation_bank_source = _load_evaluation_bank(
        spec.evaluation_bank,
        protocol,
        device=device,
        smoke=spec.smoke,
    )
    optimizer = _build_optimizer(
        model,
        dict(training["optimizer"]),
        learning_rate=float(spec.learning_rate),
    )
    steps = int(spec.steps_override or training["optimizer_updates"])
    batch_size = int(spec.batch_override or training["batch_size"])
    if spec.smoke:
        steps = min(steps, 2)
        batch_size = min(batch_size, 4)
    elif steps != int(training["optimizer_updates"]) or batch_size != int(training["batch_size"]):
        raise RuntimeError("full run dimensions differ from the validated protocol freeze")
    state_noise_enabled = bool(training["state_noise"]["enabled"])
    noise_std = (
        float(training["state_noise"]["coordinate_standard_deviation"])
        if state_noise_enabled
        else 0.0
    )
    rp = training["rp_schedule_for_ca_lru"]
    rp_schedule_enabled = bool(
        rp.get("enabled_during_training")
        if primary_main
        else rp.get("enabled_during_selector", True)
    )
    rp_enabled_by_protocol = bool(rp_schedule_enabled and model.rp_enabled)
    # These are inherited pilot defaults from the preceding CA-LRU study and
    # are recorded explicitly.  They are not confirmatory values.
    eta_lambda = float(
        rp.get("eta_lambda")
        if primary_main
        else rp.get("eta_lambda_pilot_default", 3000.0)
    )
    damage_epsilon = float(
        rp.get("damage_epsilon")
        if primary_main
        else rp.get("damage_epsilon_pilot_default", 3e-5)
    )
    rp_probe_batch = min(int(rp["probe_batch_size"]), 4) if spec.smoke else int(rp["probe_batch_size"])
    rp_probe_horizon = min(int(rp["probe_horizon"]), 8) if spec.smoke else int(rp["probe_horizon"])
    configured_blank_horizon = int(rp.get("blank_ablation_horizon", rp["probe_horizon"]))
    rp_blank_horizon = min(configured_blank_horizon, 8) if spec.smoke else configured_blank_horizon
    rp_frozen_contract = dict(rp)
    rp_effective_runtime = {
        "enabled_by_protocol": rp_enabled_by_protocol,
        "probe_batch_size": rp_probe_batch,
        "probe_horizon": rp_probe_horizon,
        "blank_ablation_horizon": rp_blank_horizon,
        "eta_lambda": eta_lambda,
        "damage_epsilon": damage_epsilon,
    }
    expected_rp_steps = _expected_rp_steps(
        model,
        steps=steps,
        warmup=int(rp["warmup_updates"]),
        interval=int(rp["interval_updates"]),
        smoke=spec.smoke,
        enabled_by_protocol=rp_enabled_by_protocol,
    )
    if not spec.smoke and model.rp_enabled:
        frozen_expected = (
            ()
            if not rp_enabled_by_protocol
            else tuple(
                range(
                    int(rp["warmup_updates"]) + int(rp["interval_updates"]),
                    steps + 1,
                    int(rp["interval_updates"]),
                )
            )
        )
        frozen_call_count = (
            0 if not rp_enabled_by_protocol else int(rp["calls_after_warmup"])
        )
        if (
            expected_rp_steps != frozen_expected
            or len(expected_rp_steps) != frozen_call_count
        ):
            raise RuntimeError("full CA-LRU RP schedule differs from the active freeze")
    if not model.rp_enabled and expected_rp_steps:
        raise RuntimeError("No-RP and GRU must have zero RP calls")
    progress_config = training.get("progress_logging")

    serialized_spec = asdict(spec)
    for key in ("output_dir", "protocol_path", "evaluation_bank", "state_spec"):
        value = serialized_spec.get(key)
        serialized_spec[key] = None if value is None else str(value)
    config_payload = {
        "schema_version": 1,
        "protocol_freeze_id": protocol["freeze_id"],
        "protocol_file_sha256": protocol_file_sha256,
        "protocol_canonical_fingerprint": protocol_canonical_fingerprint,
        "protocol_track": phase["protocol_track"],
        "reporting": protocol.get("reporting"),
        "artifact_role": artifact_role,
        "campaign_type": campaign_type,
        "parent_selector": protocol.get("parent_selector"),
        "campaign_identity": campaign_identity,
        "physical_gpu_id": os.environ.get("CALRU_PHYSICAL_GPU_ID"),
        "train_spec": {**serialized_spec, "output_dir": str(output_dir)},
        "model": model.metadata(),
        "architecture_freeze": training["architecture"],
        "resolved_task_spec": task_spec_payload,
        "resolved_task_spec_sha256": task_spec_sha256,
        "phase0_state_spec_sha256": state_spec_sha256,
        "phase0_state_spec": state_spec_payload,
        "evaluation_bank": {
            "source": evaluation_bank_source,
            "sha256": evaluation_bank_sha256,
            "trials": evaluation_batch.batch_size,
            "time_steps": evaluation_batch.time_steps,
            "resolved_task_spec_sha256": task_spec_sha256,
            "bank_metadata_task_spec_sha256": evaluation_batch.metadata.get(
                "resolved_task_spec_sha256"
            ),
            "legacy_v1_metadata_binding": (
                "resolved_task_spec_sha256" not in evaluation_batch.metadata
            ),
        },
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "optimizer": dict(training["optimizer"]),
            "learning_rate": float(spec.learning_rate),
            "gradient_clipping": dict(training["gradient_clipping"]),
            "progress_logging": progress_config,
            "state_noise_enabled": state_noise_enabled,
            "state_noise_coordinate_std": noise_std,
            "rp_eta_lambda_pilot_default": eta_lambda,
            "rp_damage_epsilon_pilot_default": damage_epsilon,
            "rp_probe_batch": rp_probe_batch,
            "rp_probe_horizon": rp_probe_horizon,
            "rp_blank_ablation_horizon": rp_blank_horizon,
            "rp_enabled_by_protocol": rp_enabled_by_protocol,
            "expected_rp_steps": list(expected_rp_steps),
            "rp_frozen_contract": rp_frozen_contract,
            "rp_effective_runtime": rp_effective_runtime,
        },
        "seeds": {
            "task_seed": int(seed_policy["task_seed"]),
            "data_stream_seed": int(seed_policy["data_stream_seed"]),
            "model_seed": model_seed,
            "evaluation_bank_seed": int(seed_policy["evaluation_bank_seed"]),
        },
    }
    _require_finite_payload(config_payload, "config")
    atomic_json(output_dir / "config.json", config_payload)

    loss_trace: list[float] = []
    gradient_norm_trace: list[float] = []
    rp_trace: list[dict[str, Any]] = []
    retention_steps: list[int] = [0]
    initial_retention = model.retention_values().detach().cpu().numpy().astype(np.float32)
    retention_trace: list[np.ndarray] = [initial_retention]
    started = time.time()
    progress_interval = (
        int(progress_config["interval_updates"])
        if progress_config is not None
        else max(1, steps // 4)
    )
    progress_path = (
        output_dir / str(progress_config["atomic_json_path"])
        if progress_config is not None
        else None
    )
    if progress_interval <= 0:
        raise ValueError("progress logging interval must be positive")
    model.train()
    for step in range(1, steps + 1):
        online_task_kwargs = task_spec.generator_kwargs()
        online_task_kwargs["horizon"] = 8 if spec.smoke else task_spec.sequence_steps
        batch = angular_integration(
            batch_size,
            int(seed_policy["task_seed"]),
            **online_task_kwargs,
            stream_key=("online_train", int(seed_policy["data_stream_seed"]), step),
            device=device,
        )
        initial = _initial_embedding(batch, device)
        generator = None
        if state_noise_enabled:
            noise_seed = derived_seed(
                int(seed_policy["data_stream_seed"]), "training_noise", model_seed, step
            )
            generator = torch.Generator(device=device).manual_seed(noise_seed)
        prediction = model.forward_sequence(
            batch.inputs,
            initial_memory=initial,
            state_noise_std=noise_std,
            noise_generator=generator,
        )
        _require_finite_tensor(prediction, f"training.step{step}.prediction")
        loss = masked_mse(prediction, batch.output_targets, batch.mask)
        _require_finite_tensor(loss, f"training.step{step}.loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        _require_finite_model(model, f"training.step{step}.post_backward", gradients=True)
        clip_value = training["gradient_clipping"].get("frozen_numeric_value")
        gradient_norm_value: float
        if clip_value is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(clip_value), error_if_nonfinite=True
            )
            _require_finite_tensor(gradient_norm, f"training.step{step}.clipped_gradient_norm")
            gradient_norm_value = float(gradient_norm.detach().cpu())
            _require_finite_model(model, f"training.step{step}.post_clip", gradients=True)
        else:
            squared_norm = torch.zeros((), device=device)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    squared_norm = squared_norm + parameter.grad.detach().square().sum()
            gradient_norm = torch.sqrt(squared_norm)
            _require_finite_tensor(gradient_norm, f"training.step{step}.gradient_norm")
            gradient_norm_value = float(gradient_norm.cpu())
        optimizer.step()
        _require_finite_model(model, f"training.step{step}.post_optimizer")
        _require_finite_optimizer(optimizer, f"training.step{step}.optimizer")
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            raise NonFiniteTrainingError(f"non-finite scalar detected: training.step{step}.loss")
        loss_trace.append(loss_value)
        gradient_norm_trace.append(gradient_norm_value)

        if step in expected_rp_steps:
            model.eval()
            rp_task_kwargs = task_spec.generator_kwargs()
            rp_task_kwargs["horizon"] = rp_probe_horizon
            probe = angular_integration(
                rp_probe_batch,
                int(seed_policy["task_seed"]),
                **rp_task_kwargs,
                stream_key=("rp_probe", len(rp_trace)),
                device=device,
            )
            details = _retention_plasticity_call(
                model,
                probe,
                blank_horizon=rp_blank_horizon,
                eta_lambda=eta_lambda,
                damage_epsilon=damage_epsilon,
            )
            rp_trace.append({"step": step, **details})
            retention_steps.append(step)
            current_retention = model.retention_values().detach()
            _require_finite_tensor(current_retention, f"training.step{step}.retention")
            retention_trace.append(current_retention.cpu().numpy().astype(np.float32))
            model.train()
        should_report_progress = (
            step == 1 or step == steps or step % progress_interval == 0
        )
        if should_report_progress:
            retention = model.retention_values().detach()
            retained = int((retention > 0.99).sum().cpu()) if retention.numel() else 0
            progress_payload = {
                "schema_version": 1,
                "status": "running" if step < steps else "training_updates_complete",
                "model_id": spec.model_name,
                "model_seed": model_seed,
                "completed_updates": step,
                "total_updates": steps,
                "masked_mse": loss_trace[-1],
                "pre_clip_global_gradient_norm": gradient_norm_trace[-1],
                "retention_gt_0p99": retained,
                "rp_calls_completed": len(rp_trace),
                "elapsed_seconds": time.time() - started,
            }
            _require_finite_payload(progress_payload, "training_progress")
            if progress_path is not None:
                atomic_json(progress_path, progress_payload)
            print(
                f"[{spec.model_name} seed={model_seed}] {step}/{steps} "
                f"loss={loss_trace[-1]:.6g} grad_norm={gradient_norm_trace[-1]:.6g} "
                f"rp_calls={len(rp_trace)} retained_gt_0.99={retained}",
                flush=True,
            )

    model.eval()
    _require_finite_model(model, "pre_final_evaluation")
    actual_rp_steps = tuple(int(item["step"]) for item in rp_trace)
    if actual_rp_steps != expected_rp_steps:
        raise RuntimeError(
            f"RP schedule invariant failed: actual={actual_rp_steps}, expected={expected_rp_steps}"
        )
    if not model.rp_enabled and rp_trace:
        raise RuntimeError("No-RP and GRU completed a forbidden RP call")
    metrics = _evaluate(model, evaluation_batch, device=device)
    retention_tensor = model.retention_values().detach()
    _require_finite_tensor(retention_tensor, "final_retention")
    retention = retention_tensor.cpu().numpy()
    if not retention_steps or retention_steps[-1] != steps:
        retention_steps.append(steps)
        retention_trace.append(np.asarray(retention, dtype=np.float32))
    metrics.update(
        {
            "train_loss_last": loss_trace[-1],
            "train_loss_last100_mean": float(np.mean(loss_trace[-100:])),
            "pre_clip_gradient_norm_last": gradient_norm_trace[-1],
            "pre_clip_gradient_norm_max": max(gradient_norm_trace),
            "elapsed_seconds": time.time() - started,
            "rp_calls": len(rp_trace),
            "retention_gt_0p99": int((retention > 0.99).sum()) if retention.size else 0,
        }
    )
    _require_finite_payload(metrics, "final_metrics")
    if (
        not spec.smoke
        and model.rp_enabled
        and metrics["rp_calls"]
        != (0 if not rp_enabled_by_protocol else int(rp["calls_after_warmup"]))
    ):
        raise RuntimeError("full CA-LRU run completed the wrong number of RP calls")
    if not model.rp_enabled and metrics["rp_calls"] != 0:
        raise RuntimeError("No-RP and GRU must complete exactly zero RP calls")

    checkpoint = output_dir / "checkpoint.pt"
    _atomic_torch_save(
        checkpoint,
        checkpoint_payload(
            model,
            {
                "protocol_freeze_id": protocol["freeze_id"],
                "protocol_file_sha256": protocol_file_sha256,
                "protocol_canonical_fingerprint": protocol_canonical_fingerprint,
                "protocol_track": phase["protocol_track"],
                "reporting": protocol.get("reporting"),
                "artifact_role": artifact_role,
                "campaign_type": campaign_type,
                "parent_selector": protocol.get("parent_selector"),
                "campaign_identity": campaign_identity,
                "evaluation_bank_sha256": evaluation_bank_sha256,
                "resolved_task_spec_sha256": task_spec_sha256,
                "state_spec_sha256": state_spec_sha256,
                "model_seed": model_seed,
                "learning_rate": float(spec.learning_rate),
                "task_metrics": metrics,
            },
        ),
    )
    np.savez_compressed(
        output_dir / "training_trace.npz",
        step=np.arange(1, steps + 1, dtype=np.int64),
        masked_mse=np.asarray(loss_trace, dtype=np.float32),
        pre_clip_global_gradient_norm=np.asarray(gradient_norm_trace, dtype=np.float32),
        retention=np.asarray(retention, dtype=np.float32),
        retention_steps=np.asarray(retention_steps, dtype=np.int64),
        retention_trajectory=np.asarray(retention_trace, dtype=np.float32),
    )
    atomic_json(output_dir / "task_metrics.json", metrics)
    atomic_json(output_dir / "rp_trace.json", rp_trace)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    architecture_metadata = model.metadata()
    manifest = {
        "schema_version": 1,
        "freeze_id": protocol["freeze_id"],
        "protocol_freeze_id": protocol["freeze_id"],
        "protocol_file_sha256": protocol_file_sha256,
        "protocol_canonical_fingerprint": protocol_canonical_fingerprint,
        "protocol_track": phase["protocol_track"],
        "reporting": protocol.get("reporting"),
        "artifact_role": artifact_role,
        "campaign_type": campaign_type,
        "parent_selector": protocol.get("parent_selector"),
        "source_protocol_sha256": protocol["source_protocol"]["sha256"],
        "resolved_task_spec": task_spec_payload,
        "resolved_task_spec_sha256": task_spec_sha256,
        "campaign_identity": campaign_identity,
        "physical_gpu_id": os.environ.get("CALRU_PHYSICAL_GPU_ID"),
        "code_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "checkpoint_sha256": sha256_file(checkpoint),
        "environment": environment,
        "environment_fingerprint": canonical_hash(environment),
        "state_spec_sha256": state_spec_sha256,
        "model_id": spec.model_name,
        "parameter_count": int(architecture_metadata["parameters_total"]),
        "architecture_metadata": architecture_metadata,
        "model": architecture_metadata,
        "seeds": config_payload["seeds"],
        "task_seed": int(seed_policy["task_seed"]),
        "data_stream_seed": int(seed_policy["data_stream_seed"]),
        "model_seed": model_seed,
        "evaluation_bank_sha256": evaluation_bank_sha256,
        "evaluation_bank_task_spec_sha256": evaluation_batch.metadata.get(
            "resolved_task_spec_sha256"
        ),
        "evaluation_bank_legacy_v1_metadata_binding": (
            "resolved_task_spec_sha256" not in evaluation_batch.metadata
        ),
        "perturbation_bank_sha256": None,
        "perturbation_bank_status": "not_applicable_to_training_stage",
        "threshold_version": canonical_hash(protocol["claim_gates"]),
        "rp_schedule": {
            "expected_steps": list(expected_rp_steps),
            "actual_steps": list(actual_rp_steps),
            "calls": len(rp_trace),
        },
        "rp_frozen_contract": rp_frozen_contract,
        "rp_effective_runtime": rp_effective_runtime,
        "task_metrics_path": "task_metrics.json",
        "pilot_only": pilot_only,
        "ca_evidence": False,
        "manifold_analysis_performed": False,
    }
    _require_finite_payload(manifest, "manifest")
    atomic_json(output_dir / "manifest.json", manifest)
    receipt_metadata = {
        "pilot_only": pilot_only,
        "artifact_role": artifact_role,
        "campaign_type": campaign_type,
        "parent_selector": protocol.get("parent_selector"),
        "freeze_id": protocol["freeze_id"],
        "protocol_track": phase["protocol_track"],
        "reporting": protocol.get("reporting"),
        "protocol_canonical_fingerprint": protocol_canonical_fingerprint,
        "campaign_identity": campaign_identity,
        "physical_gpu_id": os.environ.get("CALRU_PHYSICAL_GPU_ID"),
        "model_id": spec.model_name,
        "model_seed": model_seed,
        "parameter_count": int(architecture_metadata["parameters_total"]),
        "architecture_metadata": architecture_metadata,
        "evaluation_bank_sha256": evaluation_bank_sha256,
        "resolved_task_spec_sha256": task_spec_sha256,
        "evaluation_bank_task_spec_sha256": evaluation_batch.metadata.get(
            "resolved_task_spec_sha256"
        ),
        "state_spec_sha256": state_spec_sha256,
        "rp_calls": len(rp_trace),
        "rp_frozen_contract": rp_frozen_contract,
        "rp_effective_runtime": rp_effective_runtime,
    }
    _require_finite_payload(receipt_metadata, "receipt_metadata")
    receipt_artifacts = [
        output_dir / "config.json",
        checkpoint,
        output_dir / "training_trace.npz",
        output_dir / "task_metrics.json",
        output_dir / "rp_trace.json",
        output_dir / "manifest.json",
    ]
    if progress_path is not None:
        receipt_artifacts.append(progress_path)
    write_completion_receipt(
        completion,
        job_id=f"{spec.model_name}-seed{model_seed}-lr{spec.learning_rate:g}",
        artifacts=receipt_artifacts,
        metadata=receipt_metadata,
    )
    return output_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-bank",
        type=Path,
        help="immutable fixed-bank NPZ; mandatory (with its .sha256 sidecar) outside smoke mode",
    )
    parser.add_argument(
        "--state-spec",
        type=Path,
        help="model-specific Phase-0 state_spec.json; mandatory outside smoke mode",
    )
    parser.add_argument(
        "--campaign-identity",
        help="64-character scientific campaign identity; mandatory outside smoke mode",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = train_one(
        TrainSpec(
            model_name=args.model,
            model_seed=args.model_seed,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            protocol_path=args.protocol,
            evaluation_bank=args.evaluation_bank,
            state_spec=args.state_spec,
            campaign_identity=args.campaign_identity,
            steps_override=args.steps,
            batch_override=args.batch_size,
            device=args.device,
            smoke=args.smoke,
        )
    )
    print(json.dumps({"status": "complete", "output_dir": str(output)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
