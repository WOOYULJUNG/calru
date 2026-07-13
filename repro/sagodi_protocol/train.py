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
from .config import DEFAULT_PROTOCOL_PATH, load_protocol, protocol_fingerprint
from .metrics import masked_mse, task_metrics
from .models import (
    ProtocolModel,
    build_protocol_model,
    checkpoint_payload,
    model_config_from_protocol,
)
from .tasks import Batch, angular_integration, load_fixed_bank


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
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
    phase = protocol["phase1_ring_pilot"]
    task = phase["task"]
    evaluation = protocol["evaluation"]
    metadata = batch.metadata
    expected_metadata = {
        "task_name": "angular_integration",
        "task_seed": int(protocol["seed_policy"]["task_seed"]),
        "sample_seed": int(protocol["seed_policy"]["evaluation_bank_seed"]),
        "init_mode": "hidden-init",
        "horizon": int(task["sequence_steps"]),
        "input_dimension": int(task["input_dimension"]),
        "output_dimension": int(task["output_dimension"]),
        "target_indexing": "post_velocity_update",
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"evaluation bank metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {expected!r}"
            )
    if require_full_protocol_shape and batch.batch_size != int(evaluation["id_test_trials"]):
        raise ValueError(
            f"evaluation bank has {batch.batch_size} trials, expected "
            f"{evaluation['id_test_trials']}"
        )
    expected_shapes = {
        "inputs": (batch.time_steps, batch.batch_size, int(task["input_dimension"])),
        "output_targets": (batch.time_steps, batch.batch_size, int(task["output_dimension"])),
        "latent_targets": (batch.time_steps, batch.batch_size, int(task["latent_dimension"])),
        "mask": (batch.time_steps, batch.batch_size, int(task["output_dimension"])),
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
    phase = protocol["phase1_ring_pilot"]
    seeds = protocol["seed_policy"]
    batch = angular_integration(
        32,
        int(seeds["task_seed"]),
        init_mode="hidden-init",
        horizon=int(phase["task"]["sequence_steps"]),
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
) -> tuple[int, ...]:
    if not model.rp_enabled:
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
    training = phase["training"]
    seed_policy = protocol["seed_policy"]
    model_ids = [item["id"] for item in phase["models"]]
    if spec.model_name not in model_ids:
        raise ValueError(f"model {spec.model_name!r} is outside the frozen pilot")
    if int(spec.model_seed) not in seed_policy["pilot_model_seeds"] and not spec.smoke:
        raise ValueError("model seed is outside the frozen pilot set")
    if float(spec.learning_rate) not in training["learning_rate"]["active_launch_values"] and not spec.smoke:
        raise ValueError("learning rate is outside active frozen values")
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
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(spec.learning_rate),
        betas=tuple(float(value) for value in training["optimizer"]["betas"]),
        weight_decay=float(training["optimizer"]["weight_decay"]),
    )
    steps = int(spec.steps_override or training["optimizer_updates"])
    batch_size = int(spec.batch_override or training["batch_size"])
    if spec.smoke:
        steps = min(steps, 2)
        batch_size = min(batch_size, 4)
    elif steps != int(training["optimizer_updates"]) or batch_size != int(training["batch_size"]):
        raise RuntimeError("full run dimensions differ from the validated protocol freeze")
    noise_std = float(training["state_noise"]["coordinate_standard_deviation"])
    rp = training["rp_schedule_for_ca_lru"]
    # These are inherited pilot defaults from the preceding CA-LRU study and
    # are recorded explicitly.  They are not confirmatory values.
    eta_lambda = float(rp.get("eta_lambda_pilot_default", 3000.0))
    damage_epsilon = float(rp.get("damage_epsilon_pilot_default", 3e-5))
    rp_probe_batch = min(int(rp["probe_batch_size"]), 4) if spec.smoke else int(rp["probe_batch_size"])
    rp_probe_horizon = min(int(rp["probe_horizon"]), 8) if spec.smoke else int(rp["probe_horizon"])
    expected_rp_steps = _expected_rp_steps(
        model,
        steps=steps,
        warmup=int(rp["warmup_updates"]),
        interval=int(rp["interval_updates"]),
        smoke=spec.smoke,
    )
    if not spec.smoke and model.rp_enabled:
        frozen_expected = tuple(range(1550, 5001, 50))
        if expected_rp_steps != frozen_expected or len(expected_rp_steps) != int(
            rp["calls_after_warmup"]
        ):
            raise RuntimeError("full CA-LRU RP schedule is not exactly steps 1550..5000 by 50")
    if not model.rp_enabled and expected_rp_steps:
        raise RuntimeError("No-RP and GRU must have zero RP calls")

    serialized_spec = asdict(spec)
    for key in ("output_dir", "protocol_path", "evaluation_bank", "state_spec"):
        value = serialized_spec.get(key)
        serialized_spec[key] = None if value is None else str(value)
    config_payload = {
        "schema_version": 1,
        "protocol_freeze_id": protocol["freeze_id"],
        "protocol_file_sha256": protocol_file_sha256,
        "protocol_canonical_fingerprint": protocol_canonical_fingerprint,
        "campaign_identity": campaign_identity,
        "train_spec": {**serialized_spec, "output_dir": str(output_dir)},
        "model": model.metadata(),
        "architecture_freeze": training["architecture"],
        "phase0_state_spec_sha256": state_spec_sha256,
        "phase0_state_spec": state_spec_payload,
        "evaluation_bank": {
            "source": evaluation_bank_source,
            "sha256": evaluation_bank_sha256,
            "trials": evaluation_batch.batch_size,
            "time_steps": evaluation_batch.time_steps,
        },
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "state_noise_coordinate_std": noise_std,
            "rp_eta_lambda_pilot_default": eta_lambda,
            "rp_damage_epsilon_pilot_default": damage_epsilon,
            "rp_probe_batch": rp_probe_batch,
            "rp_probe_horizon": rp_probe_horizon,
            "expected_rp_steps": list(expected_rp_steps),
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
    rp_trace: list[dict[str, Any]] = []
    retention_steps: list[int] = [0]
    initial_retention = model.retention_values().detach().cpu().numpy().astype(np.float32)
    retention_trace: list[np.ndarray] = [initial_retention]
    started = time.time()
    model.train()
    for step in range(1, steps + 1):
        batch = angular_integration(
            batch_size,
            int(seed_policy["task_seed"]),
            init_mode="hidden-init",
            horizon=8 if spec.smoke else int(phase["task"]["sequence_steps"]),
            stream_key=("online_train", int(seed_policy["data_stream_seed"]), step),
            device=device,
        )
        initial = _initial_embedding(batch, device)
        noise_seed = derived_seed(int(seed_policy["data_stream_seed"]), "training_noise", model_seed, step)
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
        if clip_value is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(clip_value), error_if_nonfinite=True
            )
            _require_finite_tensor(gradient_norm, f"training.step{step}.clipped_gradient_norm")
            _require_finite_model(model, f"training.step{step}.post_clip", gradients=True)
        optimizer.step()
        _require_finite_model(model, f"training.step{step}.post_optimizer")
        _require_finite_optimizer(optimizer, f"training.step{step}.optimizer")
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            raise NonFiniteTrainingError(f"non-finite scalar detected: training.step{step}.loss")
        loss_trace.append(loss_value)

        if step in expected_rp_steps:
            model.eval()
            probe = angular_integration(
                rp_probe_batch,
                int(seed_policy["task_seed"]),
                init_mode="hidden-init",
                horizon=rp_probe_horizon,
                stream_key=("rp_probe", len(rp_trace)),
                device=device,
            )
            details = _retention_plasticity_call(
                model,
                probe,
                blank_horizon=rp_probe_horizon,
                eta_lambda=eta_lambda,
                damage_epsilon=damage_epsilon,
            )
            rp_trace.append({"step": step, **details})
            retention_steps.append(step)
            current_retention = model.retention_values().detach()
            _require_finite_tensor(current_retention, f"training.step{step}.retention")
            retention_trace.append(current_retention.cpu().numpy().astype(np.float32))
            model.train()
        if step == 1 or step == steps or step % max(1, steps // 4) == 0:
            retention = model.retention_values().detach()
            retained = int((retention > 0.99).sum().cpu()) if retention.numel() else 0
            print(
                f"[{spec.model_name} seed={model_seed}] {step}/{steps} "
                f"loss={loss_trace[-1]:.6g} retained_gt_0.99={retained}",
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
            "elapsed_seconds": time.time() - started,
            "rp_calls": len(rp_trace),
            "retention_gt_0p99": int((retention > 0.99).sum()) if retention.size else 0,
        }
    )
    _require_finite_payload(metrics, "final_metrics")
    if not spec.smoke and model.rp_enabled and metrics["rp_calls"] != 70:
        raise RuntimeError("full CA-LRU run must complete exactly 70 RP calls")
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
                "campaign_identity": campaign_identity,
                "evaluation_bank_sha256": evaluation_bank_sha256,
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
        "source_protocol_sha256": protocol["source_protocol"]["sha256"],
        "campaign_identity": campaign_identity,
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
        "perturbation_bank_sha256": None,
        "perturbation_bank_status": "not_applicable_to_training_stage",
        "threshold_version": canonical_hash(protocol["claim_gates"]),
        "rp_schedule": {
            "expected_steps": list(expected_rp_steps),
            "actual_steps": list(actual_rp_steps),
            "calls": len(rp_trace),
        },
        "task_metrics_path": "task_metrics.json",
        "pilot_only": True,
    }
    _require_finite_payload(manifest, "manifest")
    atomic_json(output_dir / "manifest.json", manifest)
    receipt_metadata = {
        "pilot_only": True,
        "freeze_id": protocol["freeze_id"],
        "protocol_canonical_fingerprint": protocol_canonical_fingerprint,
        "campaign_identity": campaign_identity,
        "model_id": spec.model_name,
        "model_seed": model_seed,
        "parameter_count": int(architecture_metadata["parameters_total"]),
        "architecture_metadata": architecture_metadata,
        "evaluation_bank_sha256": evaluation_bank_sha256,
        "state_spec_sha256": state_spec_sha256,
        "rp_calls": len(rp_trace),
    }
    _require_finite_payload(receipt_metadata, "receipt_metadata")
    write_completion_receipt(
        completion,
        job_id=f"{spec.model_name}-seed{model_seed}-lr{spec.learning_rate:g}",
        artifacts=[
            output_dir / "config.json",
            checkpoint,
            output_dir / "training_trace.npz",
            output_dir / "task_metrics.json",
            output_dir / "rp_trace.json",
            output_dir / "manifest.json",
        ],
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
