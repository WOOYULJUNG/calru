"""Train one RP-LRU or parameter-matched baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .artifacts import (
    atomic_json,
    atomic_npz,
    atomic_torch_save,
    source_revision,
)
from .config import DEFAULT_PROTOCOL, load_protocol
from .models import (
    MODEL_NAMES,
    ParameterMatch,
    RETENTION_MODES,
    build_model,
    count_parameters,
    parameter_count_formula,
    parameter_matched_width,
)
from .training import TrainSpec, train_model


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def run(args: argparse.Namespace) -> dict:
    protocol = load_protocol(args.protocol)
    if args.dimension not in protocol.dimensions:
        raise ValueError(
            f"dimension {args.dimension} is outside frozen dimensions "
            f"{protocol.dimensions}"
        )
    output = Path(args.output_dir).resolve()
    if (output / "COMPLETED.json").exists():
        raise FileExistsError(f"completed output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    training = protocol.training
    if args.updates is not None or args.batch_size is not None:
        training = replace(
            training,
            optimizer_updates=(
                training.optimizer_updates
                if args.updates is None
                else int(args.updates)
            ),
            batch_size=(
                training.batch_size
                if args.batch_size is None
                else int(args.batch_size)
            ),
            validation_interval=min(
                training.validation_interval,
                training.optimizer_updates
                if args.updates is None
                else int(args.updates),
            ),
            validation_trajectories=(
                training.validation_trajectories
                if args.validation_trajectories is None
                else int(args.validation_trajectories)
            ),
        )
        protocol = replace(protocol, training=training)

    count_trainable = protocol.parameter_count_mode == "trainable_parameters"
    target_count = parameter_count_formula(
        "rp_lru",
        dimension=args.dimension,
        width=protocol.rp_lru_hidden_size,
        trainable_only=count_trainable,
        retention_mode="rp",
    )
    match = None
    if args.width is not None:
        width = int(args.width)
        count = parameter_count_formula(
            args.model,
            dimension=args.dimension,
            width=width,
            trainable_only=count_trainable,
            retention_mode=args.retention_mode,
            fixed_unit_count=args.fixed_unit_count,
        )
        match = ParameterMatch(
            model=args.model,
            width=width,
            parameter_count=count,
            target_count=target_count,
            relative_gap=abs(count - target_count) / float(target_count),
            count_mode=protocol.parameter_count_mode,
        )
    elif args.model == "rp_lru":
        width = protocol.rp_lru_hidden_size
    else:
        match = parameter_matched_width(
            args.model,
            dimension=args.dimension,
            target_count=target_count,
            minimum_width=protocol.minimum_width,
            maximum_width=protocol.maximum_width,
            trainable_only=count_trainable,
        )
        width = match.width
    if match is not None and match.relative_gap > protocol.maximum_parameter_gap:
        raise RuntimeError(
            f"{args.model} parameter gap {match.relative_gap:.3%} exceeds "
            f"{protocol.maximum_parameter_gap:.3%}"
        )

    torch.manual_seed(int(args.seed))
    model = build_model(
        args.model,
        dimension=args.dimension,
        width=width,
        retention_mode=args.retention_mode,
        initial_lambda=args.initial_lambda,
        initial_lambda_low=args.initial_lambda_low,
        initial_lambda_high=args.initial_lambda_high,
        all_slow_lambda=args.all_slow_lambda,
        tau_sat=args.tau_sat,
        fixed_unit_count=args.fixed_unit_count,
        fixed_fast_lambda=args.fixed_fast_lambda,
        fixed_subset_seed=args.fixed_subset_seed,
        chrono_t_max=args.chrono_t_max,
    )
    spec = TrainSpec(
        model=args.model,
        dimension=args.dimension,
        width=width,
        learning_rate=args.learning_rate,
        seed=args.seed,
        optimizer_updates=training.optimizer_updates,
        batch_size=training.batch_size,
        retention_mode=args.retention_mode,
        initial_lambda=args.initial_lambda,
        initial_lambda_low=args.initial_lambda_low,
        initial_lambda_high=args.initial_lambda_high,
        all_slow_lambda=args.all_slow_lambda,
        rp_eta_lambda=args.rp_eta_lambda,
        rp_eta_gradient=args.rp_eta_gradient,
        rp_retention_threshold=args.rp_retention_threshold,
        rp_probe_horizon=args.rp_probe_horizon,
        retention_learning_rate=args.retention_learning_rate,
        tau_sat=args.tau_sat,
        fixed_unit_count=args.fixed_unit_count,
        fixed_fast_lambda=args.fixed_fast_lambda,
        fixed_subset_seed=args.fixed_subset_seed,
        chrono_t_max=args.chrono_t_max,
    )
    revision = source_revision(REPOSITORY_ROOT)
    manifest = {
        "schema_version": 1,
        "experiment_id": protocol.experiment_id,
        "protocol": "protocol.json",
        "protocol_sha256": protocol.sha256,
        "source": revision,
        "device": str(args.device),
        "train_spec": asdict(spec),
        "model": {
            "name": args.model,
            "width": width,
            "state_size": model.state_size,
            "total_parameters": count_parameters(model),
            "trainable_parameters": count_parameters(
                model, trainable_only=True
            ),
            "parameter_match": None if match is None else match.as_dict(),
            "target_parameter_count": target_count,
            "parameter_count_mode": protocol.parameter_count_mode,
            "architecture": model.architecture_metadata(),
            "optimizer_groups": [
                {
                    "name": str(group.get("group_name", "unnamed")),
                    "learning_rate": float(group["lr"]),
                    "weight_decay": float(group["weight_decay"]),
                    "parameter_count": int(
                        sum(parameter.numel() for parameter in group["params"])
                    ),
                }
                for group in model.optimizer_parameter_groups(
                    learning_rate=float(args.learning_rate),
                    weight_decay=float(protocol.training.weight_decay),
                )
            ],
        },
        "started_unix_time": time.time(),
    }
    if args.retention_mode in {"rp_grad", "direct_lambda_grad", "ste_recall_grad"}:
        manifest["rp_grad"] = {
            "auxiliary_objective": "recall_error_EK",
            "reduction": "batch_mean_coordinate_sum",
            "theta_in_task_optimizer": False,
            "states_targets_detached": True,
            "retention_threshold": "not_applied",
            "eta_gradient": float(args.rp_eta_gradient),
            "probe_horizon": int(args.rp_probe_horizon),
            "probe_pairs_max": int(protocol.rp.probe_pairs),
        }
    atomic_json(output / "protocol.json", protocol.raw)
    atomic_json(output / "manifest.json", manifest)
    atomic_json(
        output / "config.json",
        {
            "schema_version": 1,
            "protocol": "protocol.json",
            "protocol_sha256": protocol.sha256,
            "train_spec": asdict(spec),
            "source": revision,
        },
    )

    def progress(payload: dict) -> None:
        atomic_json(output / "progress.json", payload)

    trained = train_model(
        model, protocol, spec, device=args.device, progress=progress
    )
    checkpoint_path = output / "checkpoint.pt"
    atomic_torch_save(
        checkpoint_path,
        {
            "schema_version": 1,
            "model_state_dict": trained.model.state_dict(),
            "train_spec": asdict(spec),
            "model_manifest": manifest["model"],
            "protocol_sha256": protocol.sha256,
            "source": revision,
        },
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    atomic_json(
        output / "artifacts.json",
        {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
        },
    )
    atomic_json(output / "result.json", trained.result)
    atomic_json(
        output / "training_trace.json",
        {
            "loss": trained.loss_trace,
            "validation": trained.validation_trace,
            "rp": trained.rp_trace,
        },
    )
    retention = np.asarray(trained.retention_trace, dtype=np.float32)
    theta = np.asarray(trained.theta_trace, dtype=np.float32)
    if retention.size == 0:
        retention = np.empty((0, 0), dtype=np.float32)
        theta = np.empty((0, 0), dtype=np.float32)
    atomic_npz(
        output / "dynamics_trace.npz",
        retention_steps=np.asarray(trained.retention_steps, dtype=np.int64),
        retention=retention,
        theta=theta,
    )
    if trained.rp_score_trace:
        score_names = (
            "theta_before",
            "theta_after",
            "lambda_before",
            "lambda_after",
            "damage",
            "perturbation_energy",
            "normalized_score",
            "allocation",
        )
        atomic_npz(
            output / "rp_score_diagnostics.npz",
            update=np.asarray(
                [row["update"] for row in trained.rp_score_trace],
                dtype=np.int64,
            ),
            retention_threshold=np.asarray(
                [
                    row["retention_threshold"]
                    for row in trained.rp_score_trace
                ],
                dtype=np.float32,
            ),
            eta_lambda=np.asarray(
                [row["eta_lambda"] for row in trained.rp_score_trace],
                dtype=np.float32,
            ),
            **{
                name: np.stack(
                    [row[name].numpy() for row in trained.rp_score_trace]
                ).astype(np.float32)
                for name in score_names
            },
        )
    if trained.rp_coordinate_trace:
        coordinate_names = (
            "theta_before",
            "theta_after",
            "lambda_before",
            "lambda_after",
            "gradient_theta",
            "gradient_lambda",
            "sigmoid_derivative",
        )
        atomic_npz(
            output / "rp_grad_diagnostics.npz",
            update=np.asarray(
                [row["update"] for row in trained.rp_coordinate_trace],
                dtype=np.int64,
            ),
            **{
                name: np.stack(
                    [row[name].numpy() for row in trained.rp_coordinate_trace]
                ).astype(np.float32)
                for name in coordinate_names
            },
        )
    diagnostics = getattr(trained.model, "diagnostic_tensors", None)
    if diagnostics is not None:
        diagnostic_values = diagnostics(horizon=17017)
        atomic_npz(
            output / "model_diagnostics.npz",
            **{
                name: value.detach().cpu().numpy()
                for name, value in diagnostic_values.items()
            },
        )
    completed = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": protocol.experiment_id,
        "protocol_sha256": protocol.sha256,
        "source": revision,
        "result": trained.result,
    }
    atomic_json(output / "COMPLETED.json", completed)
    (output / "SUMMARY.md").write_text(
        "\n".join(
            [
                f"# {args.retention_mode} d={args.dimension} seed={args.seed}",
                "",
                f"- Final validation NMSE: {trained.result['final_validation']['nmse']:.9g}",
                f"- RP/recall calls: {trained.result['rp_calls']}",
                f"- Checkpoint SHA256: `{checkpoint_sha256}`",
                f"- Git commit: `{revision['commit']}` (dirty={revision['dirty']})",
                f"- Protocol SHA256: `{protocol.sha256}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--validation-trajectories", type=int)
    parser.add_argument("--retention-mode", choices=RETENTION_MODES, default="rp")
    parser.add_argument("--initial-lambda", type=float, default=0.9)
    parser.add_argument("--initial-lambda-low", type=float)
    parser.add_argument("--initial-lambda-high", type=float)
    parser.add_argument("--all-slow-lambda", type=float)
    parser.add_argument("--rp-eta-lambda", type=float)
    parser.add_argument("--rp-eta-gradient", type=float)
    parser.add_argument("--rp-retention-threshold", type=float)
    parser.add_argument("--rp-probe-horizon", type=int)
    parser.add_argument("--retention-learning-rate", type=float)
    parser.add_argument("--tau-sat", type=float, default=16.64)
    parser.add_argument("--fixed-unit-count", type=int, default=0)
    parser.add_argument("--fixed-fast-lambda", type=float, default=0.0)
    parser.add_argument("--fixed-subset-seed", type=int, default=0)
    parser.add_argument("--chrono-t-max", type=int, default=255)
    args = parser.parse_args()
    result = run(args)
    print(
        f"completed {args.model} d={args.dimension} seed={args.seed}: "
        f"NMSE={result['result']['final_validation']['nmse']:.6g}"
    )


if __name__ == "__main__":
    main()
