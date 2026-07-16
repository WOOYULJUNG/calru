"""Train one validation-only CA-LRU/H-C topology-search cell.

This runner deliberately never opens the frozen test bank.  Every job is a
full final-checkpoint run and can be resumed at campaign level by its atomic
``COMPLETED.json`` receipt.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import time
from typing import Any

import torch

from repro.sagodi_protocol.artifacts import (
    atomic_json,
    canonical_hash,
    derived_seed,
    sha256_file,
    strict_json_load,
    write_completion_receipt,
)

from .run_topology_transfer import (
    _evaluate,
    _finite_model,
    _finite_tensor,
    _normalized_retention_plasticity,
    _save_checkpoint,
)
from .topology_models import (
    SEARCH_MODEL_IDS,
    build_optimizer,
    build_topology_model,
    clip_gradients,
)
from .topology_training import (
    TOPOLOGIES,
    component_mean_mse,
    generated_batch,
    hold_prediction,
    intrinsic_metrics,
    load_fixed_bank,
)


CONFIG_PATH = Path(__file__).with_name("topology_hparam_v1.json")


def load_search_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    payload = strict_json_load(Path(path).expanduser().resolve(strict=True))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("topology hparam config must be a schema-1 object")
    if payload.get("campaign_id") != "manifold_topology_hparam_v1":
        raise ValueError("topology hparam campaign id differs")
    if tuple(payload.get("models", {})) != SEARCH_MODEL_IDS:
        raise ValueError("topology hparam model order differs")
    if bool(payload["search"]["test_bank_access"]):
        raise ValueError("hyperparameter selection must not access the test bank")
    return payload


def runtime_config(args: argparse.Namespace, frozen: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(frozen)
    row = payload["models"][args.model]
    row["learning_rate"] = float(args.learning_rate)
    if args.model == "hc":
        row["max_log_modulation"] = float(args.max_log_modulation)
        row["gate_output_bias"] = float(args.gate_output_bias)
    rp = payload["retention_plasticity"]
    rp["eta_lambda"] = float(args.rp_eta_lambda)
    rp["intervention_interval_updates"] = int(args.rp_interval)
    rp["warmup_updates"] = int(args.rp_warmup)
    return payload


def _nmse_db(component_mse: float, target: torch.Tensor) -> float:
    target_power = float(target.square().mean().detach().cpu())
    return 10.0 * math.log10(float(component_mse) / max(target_power, 1e-12))


@torch.no_grad()
def _blank_validation(model, batch, horizons: list[int]) -> dict[str, Any]:
    """Measure final-checkpoint blank retention without touching test data."""

    _, states = model.forward_sequence(
        batch.inputs, initial_memory=batch.initial_memory, return_states=True
    )
    current = states[-1]
    target = batch.output_targets[-1]
    blank = torch.zeros(
        current.shape[0], model.input_dim, dtype=current.dtype, device=current.device
    )
    requested = sorted(set(int(value) for value in horizons))
    rows: dict[str, Any] = {}
    diverged_at: int | None = None
    for step in range(1, requested[-1] + 1):
        current = model.step(blank, current)
        if step not in requested:
            continue
        finite = bool(torch.isfinite(current).all())
        prediction = model.decode(current) if finite else None
        finite = finite and prediction is not None and bool(torch.isfinite(prediction).all())
        if not finite:
            diverged_at = step if diverged_at is None else diverged_at
            rows[str(step)] = {
                "finite": False,
                "intrinsic_mean_radians": None,
                "component_mse": None,
                "hidden_norm_mean": None,
                "hidden_norm_max": None,
            }
            continue
        assert prediction is not None
        metrics = intrinsic_metrics(
            batch.topology, prediction.unsqueeze(0), target.unsqueeze(0)
        )
        primary = model.primary_from_reported(current)
        rows[str(step)] = {
            "finite": True,
            "intrinsic_mean_radians": float(metrics["intrinsic_mean_radians"]),
            "component_mse": float(metrics["component_mse"]),
            "hidden_norm_mean": float(
                torch.linalg.vector_norm(primary, dim=-1).mean().cpu()
            ),
            "hidden_norm_max": float(
                torch.linalg.vector_norm(primary, dim=-1).max().cpu()
            ),
        }
    return {
        "horizons": rows,
        "all_finite": all(bool(row["finite"]) for row in rows.values()),
        "diverged_at_or_before_horizon": diverged_at,
    }


def run(args: argparse.Namespace) -> Path:
    frozen = load_search_config(args.config)
    config = runtime_config(args, frozen)
    if args.updates <= 0:
        raise ValueError("updates must be positive")
    if args.report_interval <= 0:
        raise ValueError("report interval must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Cell identity is intentionally excluded so LR/a/b/RP comparisons share
    # the exact same model draw within model/topology/replicate.
    model_seed = derived_seed(
        int(args.seed), "topology_transfer_v1", "model", args.model
    )
    model = build_topology_model(
        args.model, args.topology, model_seed=model_seed, config=config
    ).to(device)
    optimizer = build_optimizer(model, config)

    output = Path(args.output).expanduser().resolve()
    run_dir = output / args.job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    trace_path = run_dir / "trace.json"
    result_path = run_dir / "result.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    progress_path = run_dir / "progress.pt"
    receipt_path = run_dir / "COMPLETED.json"
    if receipt_path.is_file() and not args.overwrite:
        return run_dir

    training = config["training"]
    rp = config["retention_plasticity"]
    model_metadata = model.metadata()
    model_metadata["topology_specific_retuning"] = True
    manifest = {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "job_id": args.job_id,
        "phase": args.phase,
        "cell_id": args.cell_id,
        "model": model_metadata,
        "topology": args.topology,
        "replicate_seed": int(args.seed),
        "model_seed": model_seed,
        "updates": int(args.updates),
        "batch_size": int(training["batch_size"]),
        "horizon": int(training["horizon"]),
        "learning_rate": float(args.learning_rate),
        "max_log_modulation": (
            float(args.max_log_modulation) if args.model == "hc" else None
        ),
        "gate_output_bias": (
            float(args.gate_output_bias) if args.model == "hc" else None
        ),
        "rp_eta_lambda": float(args.rp_eta_lambda),
        "rp_damage_epsilon": float(rp["damage_epsilon"]),
        "rp_interval": int(args.rp_interval),
        "rp_warmup": int(args.rp_warmup),
        "state_noise_std": 0.0,
        "target_noise_std": 0.0,
        "output_dropout": 0.0,
        "initial_memory_convention": "q0_hidden_initialization_only",
        "first_prediction_target": "apply_u0_then_compare_to_q1",
        "online_fresh_sampling": True,
        "same_data_key_excludes_model_and_cell": True,
        "selection_split": "validation",
        "validation_bank_accessed": True,
        "test_bank_accessed": False,
        "topology_specific_retuning": True,
        "final_checkpoint_only": True,
        "early_stopping": False,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "runtime_config_canonical_sha256": canonical_hash(config),
    }
    atomic_json(manifest_path, manifest)

    validation = load_fixed_bank(
        args.topology,
        split="validation",
        device=device,
        trajectories=int(training["validation_trajectories"]),
        horizon=int(training["horizon"]),
    )
    hold_metrics = intrinsic_metrics(
        args.topology, hold_prediction(validation), validation.output_targets
    )
    initial_evaluation = _evaluate(model, validation)
    trace: list[dict[str, Any]] = [
        {"update": 0, "train_component_mse": None, "validation": initial_evaluation}
    ]
    rp_trace: list[dict[str, Any]] = []
    start_update = 0
    elapsed_before_resume = 0.0
    if progress_path.is_file() and not args.overwrite:
        progress = torch.load(progress_path, map_location=device, weights_only=False)
        if progress.get("job_id") != args.job_id:
            raise ValueError("progress checkpoint job identity differs")
        if progress.get("runtime_config_canonical_sha256") != manifest[
            "runtime_config_canonical_sha256"
        ]:
            raise ValueError("progress checkpoint runtime config differs")
        model.load_state_dict(progress["model_state_dict"], strict=True)
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        trace = list(progress["trace"])
        rp_trace = list(progress["rp_trace"])
        start_update = int(progress["update"])
        elapsed_before_resume = float(progress.get("elapsed_seconds", 0.0))
        initial_evaluation = trace[0]["validation"]
    elif args.overwrite:
        progress_path.unlink(missing_ok=True)
    atomic_json(
        trace_path,
        {"schema_version": 1, "job_id": args.job_id, "trace": trace, "rp_trace": rp_trace},
    )

    probe = generated_batch(
        args.topology,
        replicate_seed=int(args.seed),
        update=0,
        batch_size=int(rp["probe_batch_size"]),
        horizon=int(training["horizon"]),
        device=device,
        stream="fixed_rp_probe",
    )
    start = time.monotonic()
    last_loss = math.nan
    for update in range(start_update + 1, int(args.updates) + 1):
        batch = generated_batch(
            args.topology,
            replicate_seed=int(args.seed),
            update=update,
            batch_size=int(training["batch_size"]),
            horizon=int(training["horizon"]),
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

        if update >= int(args.rp_warmup) and update % int(args.rp_interval) == 0:
            summary = _normalized_retention_plasticity(
                model,
                probe,
                blank_horizon=int(rp["blank_ablation_horizon"]),
                eta_lambda=float(args.rp_eta_lambda),
                damage_epsilon=float(rp["damage_epsilon"]),
            )
            rp_trace.append({"update": update, **summary})

        if update == 1 or update % int(args.report_interval) == 0 or update == args.updates:
            evaluation = _evaluate(model, validation)
            trace.append(
                {
                    "update": update,
                    "train_component_mse": last_loss,
                    "validation": evaluation,
                    "elapsed_seconds": elapsed_before_resume + time.monotonic() - start,
                }
            )
            atomic_json(
                trace_path,
                {
                    "schema_version": 1,
                    "job_id": args.job_id,
                    "trace": trace,
                    "rp_trace": rp_trace,
                },
            )
            _save_checkpoint(
                progress_path,
                {
                    "schema_version": 1,
                    "checkpoint_type": "manifold_topology_hparam_progress_v1",
                    "job_id": args.job_id,
                    "runtime_config_canonical_sha256": manifest[
                        "runtime_config_canonical_sha256"
                    ],
                    "update": update,
                    "elapsed_seconds": elapsed_before_resume + time.monotonic() - start,
                    "trace": trace,
                    "rp_trace": rp_trace,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
            )

    final_evaluation = trace[-1]["validation"]
    validation_nmse_db = _nmse_db(
        float(final_evaluation["component_mse"]), validation.output_targets
    )
    blank = _blank_validation(
        model, validation, [int(value) for value in config["search"]["blank_horizons"]]
    )
    final_loss = float(final_evaluation["component_mse"])
    initial_loss = float(initial_evaluation["component_mse"])
    result = {
        "schema_version": 1,
        "job_id": args.job_id,
        "phase": args.phase,
        "cell_id": args.cell_id,
        "model_id": args.model,
        "topology": args.topology,
        "replicate_seed": int(args.seed),
        "updates": int(args.updates),
        "elapsed_seconds": elapsed_before_resume + time.monotonic() - start,
        "last_online_train_component_mse": last_loss,
        "initial_validation": initial_evaluation,
        "final_validation": final_evaluation,
        "validation_nmse_db": validation_nmse_db,
        "validation_loss_ratio_final_over_initial": final_loss / max(initial_loss, 1e-12),
        "hold_baseline": hold_metrics,
        "beats_hold_baseline_intrinsic": (
            final_evaluation["intrinsic_mean_radians"]
            < hold_metrics["intrinsic_mean_radians"]
        ),
        "blank_validation": blank,
        "finite": True,
        "rp_calls": len(rp_trace),
        "test_bank_accessed": False,
    }
    atomic_json(result_path, result)
    _save_checkpoint(
        checkpoint_path,
        {
            "schema_version": 1,
            "checkpoint_type": "manifold_topology_hparam_v1",
            "job_id": args.job_id,
            "manifest": manifest,
            "result": result,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
    )
    write_completion_receipt(
        receipt_path,
        job_id=args.job_id,
        artifacts=(manifest_path, trace_path, result_path, checkpoint_path),
        metadata={
            "phase": args.phase,
            "model_id": args.model,
            "topology": args.topology,
            "replicate_seed": str(int(args.seed)),
            "test_bank_accessed": "false",
        },
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--model", choices=SEARCH_MODEL_IDS, required=True)
    parser.add_argument("--topology", choices=TOPOLOGIES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-log-modulation", type=float, default=0.05)
    parser.add_argument("--gate-output-bias", type=float, default=-0.1)
    parser.add_argument("--rp-eta-lambda", type=float, default=1000.0)
    parser.add_argument("--rp-interval", type=int, default=50)
    parser.add_argument("--rp-warmup", type=int, default=1500)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    destination = run(parse_args())
    print(destination, flush=True)


if __name__ == "__main__":
    main()
