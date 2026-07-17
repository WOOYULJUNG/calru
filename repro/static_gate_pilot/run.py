"""Run one isolated static-gate side-pilot job.

This is a validation-only exploratory screen.  It reuses generator-v1 and the
q0 initialization convention, but does not modify or register a paper model.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tempfile
import time
from typing import Any

import torch

from repro.manifold_benchmark.topology_training import (
    TOPOLOGIES,
    component_mean_mse,
    generated_batch,
    hold_prediction,
    intrinsic_metrics,
    load_fixed_bank,
)
from repro.sagodi_protocol.artifacts import (
    atomic_json,
    derived_seed,
    write_completion_receipt,
)

from .models import MODEL_IDS, StaticGateMemory
from .cache import cached_training_batch


def _finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@torch.no_grad()
def _evaluate(model: StaticGateMemory, batch) -> dict[str, Any]:
    model.eval()
    prediction, states = model.forward_sequence(
        batch.inputs, initial_memory=batch.initial_memory, return_states=True
    )
    _finite(prediction, "validation prediction")
    _finite(states, "validation states")
    metrics = intrinsic_metrics(batch.topology, prediction, batch.output_targets)
    retention = model.retention()
    target_power = float(batch.output_targets.square().mean().cpu())
    result: dict[str, Any] = {
        **metrics,
        "nmse_db": 10.0
        * math.log10(float(metrics["component_mse"]) / max(target_power, 1e-12)),
        "first_step_component_mse": float(
            component_mean_mse(prediction[0], batch.output_targets[0]).cpu()
        ),
        "final_step_component_mse": float(
            component_mean_mse(prediction[-1], batch.output_targets[-1]).cpu()
        ),
        "hidden_norm_mean": float(
            torch.linalg.vector_norm(states, dim=-1).mean().cpu()
        ),
        "hidden_norm_max": float(
            torch.linalg.vector_norm(states, dim=-1).max().cpu()
        ),
        "lambda_minimum": float(retention.min().cpu()),
        "lambda_mean": float(retention.mean().cpu()),
        "lambda_maximum": float(retention.max().cpu()),
        "lambda_above_0p99": int((retention > 0.99).sum().cpu()),
        "lambda_above_0p999": int((retention > 0.999).sum().cpu()),
        "retention_budget_h2048": float(retention.pow(4096).sum().cpu()),
        "finite": True,
    }
    gamma = model.write_gain()
    result["gamma_mean"] = None if gamma is None else float(gamma.mean().cpu())
    model.train()
    return result


@torch.no_grad()
def _blank_roll(
    model: StaticGateMemory,
    state: torch.Tensor,
    horizon: int,
    *,
    retention_override: torch.Tensor | None = None,
) -> torch.Tensor:
    blank = torch.zeros(
        state.shape[0], model.input_dim, device=state.device, dtype=state.dtype
    )
    current = state
    for _ in range(int(horizon)):
        current = model.step(
            blank, current, retention_override=retention_override
        )
    return current


@torch.no_grad()
def gate_intervention_damage(
    model: StaticGateMemory,
    probe,
    *,
    blank_horizon: int,
    lambda_fast: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Measure one batched counterfactual gate intervention per coordinate."""

    if not model.rp_enabled:
        raise ValueError("gate-intervention RP requires an RP model")
    _, states = model.forward_sequence(
        probe.inputs, initial_memory=probe.initial_memory, return_states=True
    )
    endpoint = states[-1]
    target = probe.output_targets[-1]
    clean_prediction = model.decode(_blank_roll(model, endpoint, blank_horizon))
    clean_error = (clean_prediction - target).square().mean()
    target_power = target.square().mean().clamp_min(torch.finfo(target.dtype).eps)

    width, batch_size = model.width, endpoint.shape[0]
    expanded_state = (
        endpoint.unsqueeze(0)
        .expand(width, batch_size, width)
        .reshape(width * batch_size, width)
        .clone()
    )
    retention = model.retention().detach()
    overrides = retention.unsqueeze(0).expand(width, width).clone()
    coordinate = torch.arange(width, device=endpoint.device)
    overrides[coordinate, coordinate] = float(lambda_fast)
    overrides = (
        overrides.unsqueeze(1)
        .expand(width, batch_size, width)
        .reshape(width * batch_size, width)
    )
    damaged_state = _blank_roll(
        model,
        expanded_state,
        blank_horizon,
        retention_override=overrides,
    )
    damaged_prediction = model.decode(damaged_state).reshape(
        width, batch_size, model.output_dim
    )
    damaged_error = (damaged_prediction - target.unsqueeze(0)).square().mean(
        dim=(1, 2)
    )
    normalized_damage = (damaged_error - clean_error) / target_power
    _finite(normalized_damage, "gate-intervention damage")
    return normalized_damage, {
        "clean_component_mse": float(clean_error.cpu()),
        "normalized_damage_mean": float(normalized_damage.mean().cpu()),
        "normalized_damage_max": float(normalized_damage.max().cpu()),
        "normalized_damage_positive_fraction": float(
            (normalized_damage > 0.0).float().mean().cpu()
        ),
    }


@torch.no_grad()
def gate_intervention_rp(
    model: StaticGateMemory,
    probe,
    *,
    blank_horizon: int,
    lambda_fast: float,
    eta_lambda: float,
    damage_epsilon: float,
    update_rule: str = "signed",
    max_theta_step: float | None = None,
) -> dict[str, float]:
    """Update RP-only theta from counterfactual gate damage."""

    normalized_damage, summary = gate_intervention_damage(
        model,
        probe,
        blank_horizon=blank_horizon,
        lambda_fast=lambda_fast,
    )
    signal = normalized_damage - float(damage_epsilon)
    if update_rule == "positive_only":
        signal = signal.clamp_min(0.0)
    elif update_rule != "signed":
        raise ValueError(f"unknown RP update rule {update_rule!r}")
    theta_step = float(eta_lambda) * signal
    if max_theta_step is not None:
        theta_step = theta_step.clamp(
            min=-float(max_theta_step), max=float(max_theta_step)
        )
    model.theta.add_(theta_step)
    model.clamp_theta_()
    return {
        **summary,
        "theta_step_mean": float(theta_step.mean().cpu()),
        "theta_step_abs_max": float(theta_step.abs().max().cpu()),
        "lambda_mean_after": float(model.retention().mean().cpu()),
        "lambda_max_after": float(model.retention().max().cpu()),
    }


@torch.no_grad()
def _blank_metrics(model: StaticGateMemory, batch, horizons: list[int]) -> dict[str, Any]:
    _, states = model.forward_sequence(
        batch.inputs, initial_memory=batch.initial_memory, return_states=True
    )
    current = states[-1]
    target = batch.output_targets[-1]
    blank = torch.zeros(
        current.shape[0], model.input_dim, device=current.device, dtype=current.dtype
    )
    requested = sorted(set(int(horizon) for horizon in horizons))
    rows: dict[str, Any] = {}
    for step in range(1, requested[-1] + 1):
        current = model.step(blank, current)
        if step not in requested:
            continue
        prediction = model.decode(current)
        metrics = intrinsic_metrics(
            batch.topology, prediction.unsqueeze(0), target.unsqueeze(0)
        )
        rows[str(step)] = {
            **metrics,
            "hidden_norm_mean": float(
                torch.linalg.vector_norm(current, dim=-1).mean().cpu()
            ),
        }
    return rows


@torch.no_grad()
def _command_response(model: StaticGateMemory, batch) -> dict[str, float]:
    initial = model.initialize(batch.initial_memory)
    zero = torch.zeros_like(batch.inputs[0])
    output_zero = model.decode(model.step(zero, initial))
    output_command = model.decode(model.step(batch.inputs[0], initial))
    predicted_move = torch.linalg.vector_norm(output_command - output_zero, dim=-1)
    true_move = torch.linalg.vector_norm(
        batch.output_targets[0] - batch.initial_memory, dim=-1
    )
    ratio = predicted_move / true_move.clamp_min(1e-8)
    return {
        "predicted_move_mean": float(predicted_move.mean().cpu()),
        "target_move_mean": float(true_move.mean().cpu()),
        "response_gain_median": float(ratio.median().cpu()),
        "response_gain_mean": float(ratio.mean().cpu()),
    }


@torch.no_grad()
def _decoder_radial_recovery(
    model: StaticGateMemory,
    batch,
    *,
    horizon: int = 512,
    anchors: int = 32,
    relative_kick: float = 0.05,
) -> dict[str, float]:
    """Exploratory decoder-radial kick; not a certified local normal."""

    _, states = model.forward_sequence(
        batch.inputs, initial_memory=batch.initial_memory, return_states=True
    )
    selected = min(int(anchors), int(states.shape[1]))
    clean0 = states[-1, :selected].clone()
    target = batch.output_targets[-1, :selected]
    radial_output = target.reshape(selected, -1, 2)
    radial_output = (
        radial_output
        / torch.linalg.vector_norm(radial_output, dim=-1, keepdim=True).clamp_min(1e-8)
    ).reshape(selected, model.output_dim)
    hidden_direction = radial_output @ model.decoder.weight
    hidden_direction = hidden_direction / torch.linalg.vector_norm(
        hidden_direction, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    scale = float(relative_kick) * torch.linalg.vector_norm(
        clean0, dim=-1
    ).median()
    kicked0 = clean0 + scale * hidden_direction
    initial_distance = torch.cdist(kicked0, clean0).min(dim=1).values
    clean_h = _blank_roll(model, clean0, horizon)
    kicked_h = _blank_roll(model, kicked0, horizon)
    final_distance = torch.cdist(kicked_h, clean_h).min(dim=1).values
    ratios = final_distance / initial_distance.clamp_min(1e-8)
    same_memory = intrinsic_metrics(
        batch.topology,
        model.decode(kicked_h).unsqueeze(0),
        model.decode(clean_h).unsqueeze(0),
    )
    return {
        "diagnostic": "decoder_radial_vjp_not_certified_local_normal",
        "horizon": int(horizon),
        "relative_kick": float(relative_kick),
        "distance_ratio_median": float(ratios.median().cpu()),
        "distance_ratio_mean": float(ratios.mean().cpu()),
        "same_memory_intrinsic_mean_radians": float(
            same_memory["intrinsic_mean_radians"]
        ),
    }


def run(args: argparse.Namespace) -> Path:
    if args.updates <= 0:
        raise ValueError("updates must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model_seed = derived_seed(
        int(args.seed), "static_gate_side_pilot_v1", args.model
    )
    torch.manual_seed(model_seed)
    model = StaticGateMemory(
        model_id=args.model,
        topology=args.topology,
        width=args.width,
        initial_retention=args.initial_retention,
        initial_write_gain=args.initial_write_gain,
        recurrent_gain=args.recurrent_gain,
    ).to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
    )
    parent_checkpoint: dict[str, Any] | None = None
    if args.load_checkpoint is not None:
        checkpoint_path = args.load_checkpoint.expanduser().resolve(strict=True)
        parent_checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(parent_checkpoint["model_state_dict"], strict=True)
        if args.load_optimizer:
            optimizer.load_state_dict(parent_checkpoint["optimizer_state_dict"])
    output = Path(args.output).expanduser().resolve()
    cell_id = args.cell_id or f"lr{args.learning_rate:g}"
    job_id = (
        f"{args.phase}__{args.model}__{args.topology}"
        f"__{cell_id}__seed{args.seed}"
    ).replace(".", "p")
    run_dir = output / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = run_dir / "COMPLETED.json"
    if receipt_path.is_file() and not args.overwrite:
        return run_dir

    manifest = {
        "schema_version": 1,
        "campaign_id": "static_gate_side_pilot_v1",
        "exploratory_not_registered_paper_model": True,
        "job_id": job_id,
        "phase": args.phase,
        "model": model.metadata(),
        "topology": args.topology,
        "replicate_seed": int(args.seed),
        "model_seed": int(model_seed),
        "updates": int(args.updates),
        "batch_size": int(args.batch_size),
        "horizon": int(args.horizon),
        "learning_rate": float(args.learning_rate),
        "initial_retention": float(args.initial_retention),
        "initial_write_gain": float(args.initial_write_gain),
        "recurrent_gain": float(args.recurrent_gain),
        "data_update_offset": int(args.data_update_offset),
        "state_noise_std": 0.0,
        "target_noise_std": 0.0,
        "output_dropout": 0.0,
        "initial_memory_convention": "q0_hidden_initialization_only",
        "first_prediction_target": "apply_u0_then_compare_to_q1",
        "gate_intervention_rp": {
            "enabled": bool(model.rp_enabled and not args.disable_rp),
            "warmup_updates": int(args.rp_warmup),
            "interval_updates": int(args.rp_interval),
            "probe_batch_size": int(args.rp_probe_batch_size),
            "blank_horizon": int(args.rp_blank_horizon),
            "lambda_fast": float(args.rp_lambda_fast),
            "eta_lambda": float(args.rp_eta_lambda),
            "damage_epsilon": float(args.rp_damage_epsilon),
            "update_rule": str(args.rp_update_rule),
            "max_theta_step": (
                None
                if args.rp_max_theta_step is None
                else float(args.rp_max_theta_step)
            ),
        },
        "selection_split": "validation",
        "test_bank_accessed": False,
        "parent_checkpoint": (
            None
            if args.load_checkpoint is None
            else str(args.load_checkpoint.expanduser().resolve())
        ),
        "parent_optimizer_loaded": bool(args.load_optimizer),
        "training_data": (
            {
                "mode": "shared_memory_mapped_train_pool",
                "cache_root": str(args.train_cache.expanduser().resolve()),
                "paired_indices_exclude_model_id": True,
            }
            if args.train_cache is not None
            else {
                "mode": "legacy_deterministic_online_generation",
                "paired_indices_exclude_model_id": True,
            }
        ),
    }
    atomic_json(run_dir / "manifest.json", manifest)

    validation = load_fixed_bank(
        args.topology,
        split="validation",
        device=device,
        trajectories=args.validation_trajectories,
        horizon=args.horizon,
    )
    hold = intrinsic_metrics(
        args.topology, hold_prediction(validation), validation.output_targets
    )
    initial = _evaluate(model, validation)
    trace: list[dict[str, Any]] = [
        {"update": 0, "train_component_mse": None, "validation": initial}
    ]
    rp_trace: list[dict[str, Any]] = []
    rp_active = bool(model.rp_enabled and not args.disable_rp)
    if args.train_cache is None:
        probe = generated_batch(
            args.topology,
            replicate_seed=int(args.seed),
            update=int(args.data_update_offset),
            batch_size=int(args.rp_probe_batch_size),
            horizon=int(args.horizon),
            device=device,
            stream="static_gate_fixed_rp_probe",
        )
    else:
        probe = cached_training_batch(
            args.train_cache,
            topology=args.topology,
            replicate_seed=int(args.seed),
            update=int(args.data_update_offset),
            batch_size=int(args.rp_probe_batch_size),
            device=device,
        )

    start = time.monotonic()
    last_loss = math.nan
    for update in range(1, int(args.updates) + 1):
        if args.train_cache is None:
            batch = generated_batch(
                args.topology,
                replicate_seed=int(args.seed),
                update=int(args.data_update_offset) + update,
                batch_size=int(args.batch_size),
                horizon=int(args.horizon),
                device=device,
                stream="online_train",
            )
        else:
            batch = cached_training_batch(
                args.train_cache,
                topology=args.topology,
                replicate_seed=int(args.seed),
                update=int(args.data_update_offset) + update,
                batch_size=int(args.batch_size),
                device=device,
            )
        model.train()
        prediction = model.forward_sequence(
            batch.inputs, initial_memory=batch.initial_memory
        )
        loss = component_mean_mse(prediction, batch.output_targets)
        _finite(loss, "training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=5.0,
        )
        optimizer.step()
        model.clamp_theta_()
        last_loss = float(loss.detach().cpu())

        if (
            rp_active
            and update >= int(args.rp_warmup)
            and update % int(args.rp_interval) == 0
        ):
            summary = gate_intervention_rp(
                model,
                probe,
                blank_horizon=int(args.rp_blank_horizon),
                lambda_fast=float(args.rp_lambda_fast),
                eta_lambda=float(args.rp_eta_lambda),
                damage_epsilon=float(args.rp_damage_epsilon),
                update_rule=str(args.rp_update_rule),
                max_theta_step=args.rp_max_theta_step,
            )
            rp_trace.append({"update": update, **summary})

        if update == 1 or update % int(args.report_interval) == 0 or update == args.updates:
            evaluation = _evaluate(model, validation)
            trace.append(
                {
                    "update": update,
                    "train_component_mse": last_loss,
                    "validation": evaluation,
                    "elapsed_seconds": time.monotonic() - start,
                }
            )
            atomic_json(
                run_dir / "trace.json",
                {
                    "schema_version": 1,
                    "job_id": job_id,
                    "trace": trace,
                    "rp_trace": rp_trace,
                },
            )

    final = trace[-1]["validation"]
    final_loss = float(final["component_mse"])
    initial_loss = float(initial["component_mse"])
    result = {
        "schema_version": 1,
        "job_id": job_id,
        "phase": args.phase,
        "model_id": args.model,
        "topology": args.topology,
        "replicate_seed": int(args.seed),
        "updates": int(args.updates),
        "elapsed_seconds": time.monotonic() - start,
        "initial_validation": initial,
        "final_validation": final,
        "validation_loss_ratio_final_over_initial": final_loss
        / max(initial_loss, 1e-12),
        "last_online_train_component_mse": last_loss,
        "hold_baseline": hold,
        "beats_hold_baseline_intrinsic": (
            float(final["intrinsic_mean_radians"])
            < float(hold["intrinsic_mean_radians"])
        ),
        "blank_validation": _blank_metrics(
            model, validation, [128, 512, 2048]
        ),
        "command_response": _command_response(model, validation),
        "decoder_radial_recovery": _decoder_radial_recovery(
            model, validation
        ),
        "rp_calls": len(rp_trace),
        "rp_active": rp_active,
        "parent_checkpoint_loaded": parent_checkpoint is not None,
        "finite": True,
        "test_bank_accessed": False,
    }
    atomic_json(run_dir / "result.json", result)
    _save_checkpoint(
        run_dir / "checkpoint.pt",
        {
            "schema_version": 1,
            "checkpoint_type": "static_gate_side_pilot_v1",
            "manifest": manifest,
            "result": result,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
    )
    write_completion_receipt(
        receipt_path,
        job_id=job_id,
        artifacts=(
            run_dir / "manifest.json",
            run_dir / "trace.json",
            run_dir / "result.json",
            run_dir / "checkpoint.pt",
        ),
        metadata={
            "model_id": args.model,
            "topology": args.topology,
            "phase": args.phase,
            "test_bank_accessed": "false",
        },
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("smoke", "short", "pretrain", "rp"), required=True
    )
    parser.add_argument("--cell-id")
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--topology", choices=("s1", "t2"), required=True)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--width", type=int, default=52)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--initial-retention", type=float, default=0.95)
    parser.add_argument("--initial-write-gain", type=float, default=1.0)
    parser.add_argument("--recurrent-gain", type=float, default=0.9)
    parser.add_argument("--data-update-offset", type=int, default=0)
    parser.add_argument("--validation-trajectories", type=int, default=64)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--rp-warmup", type=int, default=150)
    parser.add_argument("--rp-interval", type=int, default=50)
    parser.add_argument("--rp-probe-batch-size", type=int, default=16)
    parser.add_argument("--rp-blank-horizon", type=int, default=128)
    parser.add_argument("--rp-lambda-fast", type=float, default=0.5)
    parser.add_argument("--rp-eta-lambda", type=float, default=100.0)
    parser.add_argument("--rp-damage-epsilon", type=float, default=3e-5)
    parser.add_argument(
        "--rp-update-rule",
        choices=("signed", "positive_only"),
        default="signed",
    )
    parser.add_argument("--rp-max-theta-step", type=float)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_side_pilot_v1"
        ),
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        help="shared train-only memory-mapped pool; avoids per-update generation",
    )
    parser.add_argument("--disable-rp", action="store_true")
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--load-optimizer", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(run(parse_args()), flush=True)


if __name__ == "__main__":
    main()
