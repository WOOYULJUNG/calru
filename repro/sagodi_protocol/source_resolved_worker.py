"""Single-run worker for code-resolved Ságodi baseline sentinels.

Example::

    python -m repro.sagodi_protocol.source_resolved_worker \
      --run-id source__gru__seed0 \
      --model-id sagodi_gru_n128 --model-seed 0 \
      --evaluation-bank /path/to/source_eval.npz \
      --output-dir /path/to/run --device cuda:0

The worker performs no hyperparameter selection and no orchestration.  It
writes a relocatable completion receipt so a campaign launcher can safely
resume or atomically promote the completed directory.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .artifacts import (
    atomic_json,
    derived_seed,
    sha256_file,
    write_completion_receipt,
)
from .source_resolved_models import (
    SourceResolvedBaseline,
    build_source_resolved_model,
)
from .source_resolved_protocol import (
    DEFAULT_SOURCE_CONFIG,
    SOURCE_CAMPAIGN_ID,
    SOURCE_MODEL_IDS,
    UPSTREAM_COMMIT,
    build_source_optimizer,
    clip_source_gradients,
    load_source_config,
    noisy_training_targets,
    source_angular_integration,
    source_masked_mse,
    source_recipe,
)
from .tasks import Batch, load_fixed_bank


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


def _finite_model(model: SourceResolvedBaseline, *, gradients: bool = False) -> None:
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


def _validate_evaluation_bank(batch: Batch) -> None:
    metadata = batch.metadata
    expected = {
        "task_name": "angular_integration",
        "task_version": "sagodi-source-resolved-v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "horizon": 128,
        "delta_t": 0.1,
        "input_sparsity": "variable_uniform_0_2",
        "target_indexing": "post_velocity_update",
        "initial_state_semantics": "source_q1_post_update_target",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"source evaluation bank metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )
    if tuple(batch.inputs.shape[1:])[-1] != 1:
        raise ValueError("source evaluation bank input dimension must be one")
    if tuple(batch.output_targets.shape) != (128, batch.batch_size, 2):
        raise ValueError("source evaluation bank target shape differs")


@torch.no_grad()
def _evaluate(model: SourceResolvedBaseline, batch: Batch) -> dict[str, float]:
    model.eval()
    prediction = model.forward_sequence(
        batch.inputs,
        source_targets=batch.output_targets,
        # Evaluation is deterministic; the source state noise is a training
        # perturbation for this registered sentinel metric.
        state_noise_std_override=0.0,
    )
    mse = source_masked_mse(prediction, batch.output_targets, batch.mask)
    predicted_angle = torch.atan2(prediction[..., 1], prediction[..., 0])
    target_angle = torch.atan2(
        batch.output_targets[..., 1], batch.output_targets[..., 0]
    )
    angle_error = torch.remainder(
        predicted_angle - target_angle + math.pi, 2.0 * math.pi
    ) - math.pi
    radius = torch.linalg.vector_norm(prediction, dim=-1)
    values = {
        "masked_mse": float(mse.cpu()),
        "mean_absolute_angular_error": float(angle_error.abs().mean().cpu()),
        "mean_output_radius": float(radius.mean().cpu()),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("non-finite evaluation metric")
    return values


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


def run_source_worker(
    *,
    run_id: str,
    model_id: str,
    model_seed: int,
    evaluation_bank: Path,
    output_dir: Path,
    device_text: str,
    config_path: Path = DEFAULT_SOURCE_CONFIG,
    updates: int | None = None,
    batch_size: int | None = None,
    trace_interval: int = 50,
    validation_interval: int = 500,
    campaign_identity: str | None = None,
) -> Path:
    """Train one final-update sentinel and write a verified output bundle."""

    config = load_source_config(config_path)
    recipe = source_recipe(model_id)
    total_updates = int(config["training"]["updates"] if updates is None else updates)
    train_batch = int(config["training"]["batch_size"] if batch_size is None else batch_size)
    if total_updates <= 0 or train_batch <= 0:
        raise ValueError("updates and batch_size must be positive")
    if int(trace_interval) <= 0 or int(validation_interval) <= 0:
        raise ValueError("trace and validation intervals must be positive")
    if campaign_identity is not None:
        normalized = str(campaign_identity).strip().lower()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError("campaign_identity must be a lowercase SHA-256 digest")
        campaign_identity = normalized

    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"source worker output is not empty: {output}")
    bank_path = evaluation_bank.expanduser().resolve(strict=True)
    device = torch.device(device_text)
    _configure_determinism(int(model_seed))
    model = build_source_resolved_model(model_id).to(device)
    _finite_model(model)
    optimizer = build_source_optimizer(model, recipe)
    evaluation = _to_device(load_fixed_bank(bank_path), device)
    _validate_evaluation_bank(evaluation)

    target_generator = torch.Generator(device=device.type).manual_seed(
        derived_seed(model_seed, SOURCE_CAMPAIGN_ID, "target_noise")
    )
    state_generator = torch.Generator(device=device.type).manual_seed(
        derived_seed(model_seed, SOURCE_CAMPAIGN_ID, "state_noise")
    )
    manifest = {
        "schema_version": 1,
        "run_id": str(run_id),
        "campaign_id": SOURCE_CAMPAIGN_ID,
        "campaign_identity": campaign_identity,
        "upstream_commit": UPSTREAM_COMMIT,
        "model_seed": int(model_seed),
        "model": model.metadata(),
        "recipe": recipe.__dict__,
        "updates": total_updates,
        "batch_size": train_batch,
        "evaluation_bank": str(bank_path),
        "evaluation_bank_sha256": sha256_file(bank_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "worker_sha256": sha256_file(Path(__file__)),
        "runtime_code_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "source_resolved_worker.py",
                "source_resolved_models.py",
                "source_resolved_protocol.py",
                "tasks.py",
            )
        },
        "device": str(device),
        "online_task_stream": ["online_train", int(model_seed), "update"],
        "pairing_policy": "same_model_seed_and_update_share_data_across_models",
    }
    atomic_json(output / "run_manifest.json", manifest)

    trace: list[dict[str, Any]] = []
    started = time.time()
    for update in range(1, total_updates + 1):
        model.train()
        batch = source_angular_integration(
            train_batch,
            0,
            stream_key=("online_train", int(model_seed), int(update)),
            device=device,
            config_path=config_path,
        )
        training_targets = noisy_training_targets(
            batch.output_targets,
            recipe,
            generator=target_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_sequence(
            batch.inputs,
            source_targets=training_targets,
            state_noise_generator=state_generator,
        )
        loss = source_masked_mse(prediction, training_targets, batch.mask)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"non-finite training loss at update {update}")
        loss.backward()
        _finite_model(model, gradients=True)
        gradient_norm = clip_source_gradients(model, recipe)
        optimizer.step()
        _finite_model(model)

        should_trace = update == 1 or update % int(trace_interval) == 0 or update == total_updates
        should_validate = update % int(validation_interval) == 0 or update == total_updates
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

    final_metrics = _evaluate(model, evaluation)
    result = {
        "schema_version": 1,
        "status": "complete",
        "run_id": str(run_id),
        "model_id": model_id,
        "model_seed": int(model_seed),
        "updates_completed": total_updates,
        "learning_rate": float(recipe.learning_rate),
        "final_metrics": final_metrics,
        "final_checkpoint_is_primary": True,
    }
    atomic_json(output / "training_trace.json", trace)
    atomic_json(output / "result.json", result)
    checkpoint = output / "checkpoint_final.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "schema_version": 1,
            "checkpoint_type": "sagodi_source_resolved_v1",
            "run": manifest,
            "result": result,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
    )
    atomic_json(
        output / "COMPLETE",
        {"schema_version": 1, "status": "complete", "run_id": str(run_id)},
    )
    receipt_metadata: dict[str, Any] = {
        "campaign_id": SOURCE_CAMPAIGN_ID,
        "upstream_commit": UPSTREAM_COMMIT,
        "model_id": model_id,
        "model_seed": int(model_seed),
    }
    if campaign_identity is not None:
        receipt_metadata["campaign_scientific_identity"] = campaign_identity
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=str(run_id),
        artifacts=[
            output / "run_manifest.json",
            output / "training_trace.json",
            output / "result.json",
            checkpoint,
            output / "COMPLETE",
        ],
        metadata=receipt_metadata,
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", required=True, choices=SOURCE_MODEL_IDS)
    parser.add_argument("--model-seed", required=True, type=int)
    parser.add_argument("--evaluation-bank", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--trace-interval", type=int, default=50)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--campaign-identity")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run_source_worker(
        run_id=args.run_id,
        model_id=args.model_id,
        model_seed=args.model_seed,
        evaluation_bank=args.evaluation_bank,
        output_dir=args.output_dir,
        device_text=args.device,
        config_path=args.config,
        updates=args.updates,
        batch_size=args.batch_size,
        trace_interval=args.trace_interval,
        validation_interval=args.validation_interval,
        campaign_identity=args.campaign_identity,
    )
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
