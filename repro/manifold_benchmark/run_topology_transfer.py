"""Run one auditable topology-transfer debug or pilot job.

Examples
--------
python -m repro.manifold_benchmark.run_topology_transfer \
  --stage fixed_overfit --model hc --topology s2 --seed 9 --device cuda:0
python -m repro.manifold_benchmark.run_topology_transfer \
  --stage online_smoke --model gru --topology t2 --seed 9 --device cuda:1
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tempfile
import time
from typing import Any

import torch

from repro.sagodi_protocol.artifacts import (
    atomic_json,
    canonical_hash,
    derived_seed,
    sha256_file,
    write_completion_receipt,
)

from .topology_models import (
    MODEL_IDS,
    build_optimizer,
    build_topology_model,
    clip_gradients,
    load_transfer_config,
)
from .topology_training import (
    TOPOLOGIES,
    TopologyBatch,
    component_mean_mse,
    fixed_debug_batch,
    generated_batch,
    hold_prediction,
    intrinsic_metrics,
    load_fixed_bank,
)


STAGES = ("fixed_overfit", "online_smoke", "pilot")


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _finite_model(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        _finite_tensor(parameter, f"parameter {name}")


@torch.no_grad()
def _evaluate(model, batch: TopologyBatch) -> dict[str, Any]:
    model.eval()
    prediction, states = model.forward_sequence(
        batch.inputs,
        initial_memory=batch.initial_memory,
        return_states=True,
    )
    _finite_tensor(prediction, "evaluation prediction")
    _finite_tensor(states, "evaluation states")
    first_loss = component_mean_mse(prediction[0], batch.output_targets[0])
    final_loss = component_mean_mse(prediction[-1], batch.output_targets[-1])
    primary = model.primary_from_reported(states)
    result: dict[str, Any] = {
        **intrinsic_metrics(batch.topology, prediction, batch.output_targets),
        "first_step_component_mse": float(first_loss.cpu()),
        "final_step_component_mse": float(final_loss.cpu()),
        "prediction_shape": list(prediction.shape),
        "state_shape": list(states.shape),
        "hidden_norm_mean": float(torch.linalg.vector_norm(primary, dim=-1).mean().cpu()),
        "hidden_norm_max": float(torch.linalg.vector_norm(primary, dim=-1).max().cpu()),
        "prediction_component_variance_mean": float(
            prediction.var(dim=(0, 1), unbiased=False).mean().cpu()
        ),
        "target_component_variance_mean": float(
            batch.output_targets.var(dim=(0, 1), unbiased=False).mean().cpu()
        ),
        "finite": True,
    }
    dynamic_lambda = model.dynamic_lambda(states)
    if dynamic_lambda is not None:
        result["lambda_minimum"] = float(dynamic_lambda.min().cpu())
        result["lambda_mean"] = float(dynamic_lambda.mean().cpu())
        result["lambda_maximum"] = float(dynamic_lambda.max().cpu())
        result["lambda_fraction_above_one"] = float(
            (dynamic_lambda > 1.0).float().mean().cpu()
        )
    else:
        result.update(
            {
                "lambda_minimum": None,
                "lambda_mean": None,
                "lambda_maximum": None,
                "lambda_fraction_above_one": None,
            }
        )
    model.train()
    return result


@torch.no_grad()
def _roll_blank(model, state: torch.Tensor, horizon: int) -> torch.Tensor:
    blank = torch.zeros(
        state.shape[0], model.input_dim, device=state.device, dtype=state.dtype
    )
    current = state
    for _ in range(int(horizon)):
        current = model.step(blank, current)
    return current


@torch.no_grad()
def _normalized_retention_plasticity(
    model,
    probe: TopologyBatch,
    *,
    blank_horizon: int,
    eta_lambda: float,
    damage_epsilon: float,
) -> dict[str, float]:
    """Apply the frozen hybrid-RP rule with topology-normalized damage."""

    if not model.rp_enabled:
        raise ValueError("RP is only defined for CA-LRU/H-C search models")
    _, states = model.forward_sequence(
        probe.inputs, initial_memory=probe.initial_memory, return_states=True
    )
    state = states[-1]
    target = probe.output_targets[-1]
    clean = model.decode(_roll_blank(model, state, blank_horizon))
    clean_energy = component_mean_mse(clean, target)
    target_power = target.square().mean()
    denominator = target_power + torch.finfo(target.dtype).eps
    damages: list[torch.Tensor] = []
    for recurrence, state_slice in model.pan_recs_with_slices():
        hidden = int(state_slice.stop - state_slice.start)
        batch_size, total_state = state.shape
        ablated = state.unsqueeze(0).expand(hidden, batch_size, total_state).clone()
        coordinate = torch.arange(hidden, device=state.device)
        ablated[coordinate, :, state_slice.start + coordinate] = 0.0
        final = _roll_blank(
            model,
            ablated.reshape(hidden * batch_size, total_state),
            blank_horizon,
        )
        prediction = model.decode(final).reshape(
            hidden, batch_size, model.output_dim
        )
        ablated_energy = (prediction - target.unsqueeze(0)).square().mean(dim=(1, 2))
        normalized_damage = (ablated_energy - clean_energy) / denominator
        _finite_tensor(normalized_damage, "RP normalized damage")
        recurrence.update_theta(normalized_damage - float(damage_epsilon), eta_lambda)
        damages.append(normalized_damage)
    if not damages:
        raise RuntimeError("search model exposes no RP recurrence")
    values = torch.cat(damages)
    _finite_model(model)
    return {
        "clean_component_mse": float(clean_energy.cpu()),
        "target_component_power": float(target_power.cpu()),
        "normalized_damage_mean": float(values.mean().cpu()),
        "normalized_damage_max": float(values.max().cpu()),
        "normalized_damage_positive_fraction": float((values > 0).float().mean().cpu()),
    }


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


def _stage_defaults(config: dict[str, Any], stage: str) -> tuple[int, int, int]:
    training = config["training"]
    debug = config["debug"]
    if stage == "fixed_overfit":
        return (
            int(debug["fixed_updates"]),
            int(debug["fixed_batch_size"]),
            int(debug["fixed_horizon"]),
        )
    if stage == "online_smoke":
        return (
            int(debug["online_smoke_updates"]),
            int(training["batch_size"]),
            int(training["horizon"]),
        )
    return (
        int(training["updates"]),
        int(training["batch_size"]),
        int(training["horizon"]),
    )


def run(args: argparse.Namespace) -> Path:
    config = load_transfer_config(args.config)
    default_updates, batch_size, horizon = _stage_defaults(config, args.stage)
    updates = default_updates if args.updates is None else int(args.updates)
    if updates <= 0:
        raise ValueError("updates must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model_seed = derived_seed(int(args.seed), "topology_transfer_v1", "model", args.model)
    model = build_topology_model(
        args.model, args.topology, model_seed=model_seed, config=config
    ).to(device)
    optimizer = build_optimizer(model, config)
    output = Path(args.output).expanduser().resolve()
    job_id = f"{args.stage}__{args.model}__{args.topology}__seed{int(args.seed)}"
    run_dir = output / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    trace_path = run_dir / "trace.json"
    result_path = run_dir / "result.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    receipt_path = run_dir / "COMPLETED.json"
    if receipt_path.exists() and not args.overwrite:
        return run_dir

    manifest = {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "job_id": job_id,
        "stage": args.stage,
        "model": model.metadata(),
        "topology": args.topology,
        "replicate_seed": int(args.seed),
        "model_seed": model_seed,
        "updates": updates,
        "batch_size": batch_size,
        "horizon": horizon,
        "online_fresh_sampling": args.stage != "fixed_overfit",
        "same_data_key_excludes_model_id": True,
        "state_noise_std": 0.0,
        "target_noise_std": 0.0,
        "output_dropout": 0.0,
        "initial_memory_convention": "q0_hidden_initialization_only",
        "first_prediction_target": "apply_u0_then_compare_to_q1",
        "loss": config["training"]["loss"],
        "validation_bank_accessed": args.stage != "fixed_overfit",
        "test_bank_accessed": args.stage == "pilot",
        "topology_specific_retuning": False,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "config_canonical_sha256": canonical_hash(config),
    }
    atomic_json(manifest_path, manifest)

    if args.stage == "fixed_overfit":
        fixed = fixed_debug_batch(
            args.topology,
            debug_seed=int(args.seed),
            batch_size=batch_size,
            horizon=horizon,
            device=device,
        )
        validation = fixed
    else:
        fixed = None
        validation = load_fixed_bank(
            args.topology,
            split="validation",
            device=device,
            trajectories=int(args.validation_trajectories),
            horizon=horizon,
        )
    initial_evaluation = _evaluate(model, validation)
    hold_metrics = intrinsic_metrics(
        args.topology, hold_prediction(validation), validation.output_targets
    )
    trace: list[dict[str, Any]] = [
        {"update": 0, "train_component_mse": None, "validation": initial_evaluation}
    ]
    rp_trace: list[dict[str, Any]] = []
    rp = config["retention_plasticity"]
    probe = None
    if args.stage == "pilot" and args.model == "hc":
        probe = generated_batch(
            args.topology,
            replicate_seed=int(args.seed),
            update=0,
            batch_size=int(rp["probe_batch_size"]),
            horizon=horizon,
            device=device,
            stream="fixed_rp_probe",
        )

    start = time.monotonic()
    last_loss = math.nan
    for update in range(1, updates + 1):
        batch = fixed
        if batch is None:
            batch = generated_batch(
                args.topology,
                replicate_seed=int(args.seed),
                update=update,
                batch_size=batch_size,
                horizon=horizon,
                device=device,
            )
        model.train()
        prediction = model.forward_sequence(
            batch.inputs, initial_memory=batch.initial_memory
        )
        loss = component_mean_mse(prediction, batch.output_targets)
        _finite_tensor(loss, "training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_gradients(model)
        optimizer.step()
        _finite_model(model)
        last_loss = float(loss.detach().cpu())

        if (
            probe is not None
            and update >= int(rp["warmup_updates"])
            and update % int(rp["intervention_interval_updates"]) == 0
        ):
            summary = _normalized_retention_plasticity(
                model,
                probe,
                blank_horizon=int(rp["blank_ablation_horizon"]),
                eta_lambda=float(rp["eta_lambda"]),
                damage_epsilon=float(rp["damage_epsilon"]),
            )
            rp_trace.append({"update": update, **summary})

        if update == 1 or update % int(args.report_interval) == 0 or update == updates:
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
                trace_path,
                {"schema_version": 1, "job_id": job_id, "trace": trace, "rp_trace": rp_trace},
            )

    final_evaluation = trace[-1]["validation"]
    initial_loss = float(initial_evaluation["component_mse"])
    final_loss = float(final_evaluation["component_mse"])
    ratio = final_loss / max(initial_loss, torch.finfo(torch.float32).eps)
    result: dict[str, Any] = {
        "schema_version": 1,
        "job_id": job_id,
        "stage": args.stage,
        "model_id": args.model,
        "topology": args.topology,
        "replicate_seed": int(args.seed),
        "updates": updates,
        "elapsed_seconds": time.monotonic() - start,
        "last_online_train_component_mse": last_loss,
        "initial_validation": initial_evaluation,
        "final_validation": final_evaluation,
        "validation_loss_ratio_final_over_initial": ratio,
        "hold_baseline": hold_metrics,
        "beats_hold_baseline_intrinsic": (
            final_evaluation["intrinsic_mean_radians"]
            < hold_metrics["intrinsic_mean_radians"]
        ),
        "fixed_overfit_tens_fold_gate": (
            None if args.stage != "fixed_overfit" else ratio <= 0.1
        ),
        "prediction_nonconstant": (
            final_evaluation["prediction_component_variance_mean"] > 1e-8
        ),
        "finite": True,
        "rp_calls": len(rp_trace),
        "test_bank_accessed": args.stage == "pilot",
    }
    if args.stage == "pilot":
        test = load_fixed_bank(
            args.topology,
            split="test",
            device=device,
            trajectories=int(args.validation_trajectories),
            horizon=horizon,
        )
        result["test"] = _evaluate(model, test)
    atomic_json(result_path, result)
    _save_checkpoint(
        checkpoint_path,
        {
            "schema_version": 1,
            "job_id": job_id,
            "manifest": manifest,
            "result": result,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
    )
    write_completion_receipt(
        receipt_path,
        job_id=job_id,
        artifacts=(manifest_path, trace_path, result_path, checkpoint_path),
        metadata={
            "stage": args.stage,
            "model_id": args.model,
            "topology": args.topology,
            "replicate_seed": str(int(args.seed)),
        },
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--topology", choices=TOPOLOGIES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("topology_transfer_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/manifold_topology_transfer_v1-debug"
        ),
    )
    parser.add_argument("--updates", type=int)
    parser.add_argument("--report-interval", type=int, default=50)
    parser.add_argument("--validation-trajectories", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = run(args)
    print(destination, flush=True)


if __name__ == "__main__":
    main()
