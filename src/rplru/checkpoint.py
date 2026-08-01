"""Strict reconstruction of a model from one completed RP-LRU run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import Protocol, load_protocol
from .models import SequenceModel, build_model


def load_checkpoint_model(
    run_dir: str | Path,
    protocol: Protocol,
    *,
    device: torch.device | str,
    allow_protocol_mismatch: bool = False,
) -> tuple[SequenceModel, dict[str, Any]]:
    root = Path(run_dir).resolve()
    if not (root / "COMPLETED.json").exists():
        raise FileNotFoundError(f"run is not complete: {root}")
    payload = torch.load(
        root / "checkpoint.pt", map_location=torch.device(device), weights_only=False
    )
    if (
        payload["protocol_sha256"] != protocol.sha256
        and not allow_protocol_mismatch
    ):
        raise RuntimeError(
            "checkpoint protocol differs from the requested evaluation protocol"
        )
    spec = payload["train_spec"]
    model = build_model(
        spec["model"],
        dimension=int(spec["dimension"]),
        width=int(spec["width"]),
        retention_mode=str(spec["retention_mode"]),
        initial_lambda=float(spec["initial_lambda"]),
        initial_lambda_low=(
            None
            if spec.get("initial_lambda_low") is None
            else float(spec["initial_lambda_low"])
        ),
        initial_lambda_high=(
            None
            if spec.get("initial_lambda_high") is None
            else float(spec["initial_lambda_high"])
        ),
        all_slow_lambda=(
            None
            if spec.get("all_slow_lambda") is None
            else float(spec["all_slow_lambda"])
        ),
        tau_sat=float(spec.get("tau_sat", 16.64)),
        fixed_unit_count=int(spec.get("fixed_unit_count", 0)),
        fixed_fast_lambda=float(spec.get("fixed_fast_lambda", 0.0)),
        fixed_subset_seed=int(spec.get("fixed_subset_seed", spec.get("seed", 0))),
        chrono_t_max=int(spec.get("chrono_t_max", 255)),
    )
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    # ``fixed_unit_mask`` and ``fixed_free_indices`` were added with the
    # reviewer controls.  Pre-control RP checkpoints have the deterministic
    # empty defaults for both buffers and therefore remain exactly
    # reconstructible without storing them.  All other incompatibilities are
    # fatal, and fixed-unit checkpoints themselves must contain the buffers.
    allowed_missing = (
        {"fixed_unit_mask", "fixed_free_indices"}
        if spec["model"] == "rp_lru"
        and str(spec["retention_mode"])
        not in {"fixed_binary", "fixed_unit_bptt"}
        else set()
    )
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing - allowed_missing or unexpected:
        raise RuntimeError(
            "checkpoint state_dict is incompatible: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    model.to(device)
    model.eval()
    return model, payload


def load_checkpoint_protocol(
    run_dir: str | Path, checkpoint: dict[str, Any]
) -> Protocol:
    """Load and hash-check the training protocol recorded by a run."""

    root = Path(run_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_path = Path(manifest["protocol"])
    if not protocol_path.is_absolute():
        protocol_path = root / protocol_path
    protocol_path = protocol_path.resolve()
    training_protocol = load_protocol(protocol_path)
    expected = str(checkpoint["protocol_sha256"])
    if manifest.get("protocol_sha256") != expected:
        raise RuntimeError("manifest and checkpoint protocol hashes differ")
    if training_protocol.sha256 != expected:
        raise RuntimeError("recorded training protocol content has changed")
    return training_protocol


def validate_evaluation_compatibility(
    training_protocol: Protocol,
    evaluation_protocol: Protocol,
    *,
    dimension: int,
) -> None:
    """Fail closed unless a checkpoint's task matches the evaluation task."""

    if training_protocol.task != evaluation_protocol.task:
        raise RuntimeError(
            "checkpoint training task differs from the evaluation task"
        )
    if int(dimension) not in training_protocol.dimensions:
        raise RuntimeError("checkpoint dimension is outside its training protocol")
    if int(dimension) not in evaluation_protocol.dimensions:
        raise RuntimeError("checkpoint dimension is outside the evaluation protocol")
