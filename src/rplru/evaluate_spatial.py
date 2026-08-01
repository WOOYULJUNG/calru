"""Evaluate a spatial-localization checkpoint on a fixed trial bank."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import atomic_json, atomic_npz, derived_seed
from .config import DEFAULT_PROTOCOL, load_protocol
from .spatial import (
    REPRESENTATIVE_CELLS,
    build_classifier,
    class_balance,
    classification_metrics,
    final_logits,
    grid_labels,
)
from .task import TaskSpec, generate_batch


MARGIN_EDGES = np.asarray([0.0, 0.05, 0.10, 0.20, 0.40, np.inf])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_margin(targets: torch.Tensor, bins: int) -> torch.Tensor:
    boundaries = torch.linspace(
        -1.0, 1.0, int(bins) + 1, device=targets.device, dtype=targets.dtype
    )[1:-1]
    distance = (targets.unsqueeze(-1) - boundaries).abs().amin(dim=-1)
    return distance / (2.0 / int(bins))


def _margin_metrics(
    margin: np.ndarray, coordinate_correct: np.ndarray, exact_correct: np.ndarray
) -> list[dict[str, Any]]:
    trial_margin = margin.min(axis=1)
    rows = []
    for left, right in zip(MARGIN_EDGES[:-1], MARGIN_EDGES[1:]):
        mask = (trial_margin >= left) & (trial_margin < right)
        count = int(mask.sum())
        rows.append(
            {
                "left_inclusive": float(left),
                "right_exclusive": None if np.isinf(right) else float(right),
                "trials": count,
                "coordinate_accuracy": (
                    None if count == 0 else float(coordinate_correct[mask].mean())
                ),
                "exact_accuracy": (
                    None if count == 0 else float(exact_correct[mask].mean())
                ),
            }
        )
    return rows


@torch.no_grad()
def evaluate(
    *,
    protocol_path: str | Path,
    run_dir: str | Path,
    output_dir: str | Path,
    device: str,
    trajectories: int,
    batch_size: int,
    bank_seed: int,
    bank_tag: str,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    task = replace(
        protocol.task,
        initial_value_low=-1.0,
        initial_value_high=1.0,
    )
    root = Path(run_dir).resolve()
    checkpoint_path = root / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = checkpoint["manifest"]
    spec = manifest["spec"]
    dimension = int(spec["dimension"])
    bins = int(spec["bins"])
    width = int(manifest["model"]["width"])
    model = build_classifier(
        spec["model"],
        dimension=dimension,
        bins=bins,
        width=width,
        lru_lambda_min=float(spec["lru_lambda_min"]),
        chrono_t_max=int(spec["chrono_t_max"]),
        retention_mode=str(spec.get("retention_mode", "rp")),
        tau_sat=float(spec.get("tau_sat", 16.64)),
        fixed_unit_count=int(spec.get("fixed_unit_count", 0)),
        fixed_fast_lambda=float(spec.get("fixed_fast_lambda", 0.0)),
        fixed_subset_seed=int(
            spec.get("fixed_subset_seed")
            if spec.get("fixed_subset_seed") is not None
            else spec.get("seed", 0)
        ),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    rows = {}
    trial_targets = []
    trial_labels = []
    trial_coordinate_correct = []
    trial_exact_correct = []
    trial_margin = []
    bank_hash = hashlib.sha256()
    for regime, (update_count, hold) in REPRESENTATIVE_CELLS.items():
        logits_chunks = []
        label_chunks = []
        target_chunks = []
        condition_seed = derived_seed(
            bank_seed,
            bank_tag,
            bins,
            dimension,
            update_count,
            hold,
        )
        for start in range(0, int(trajectories), int(batch_size)):
            count = min(int(batch_size), int(trajectories) - start)
            batch = generate_batch(
                TaskSpec.controlled(
                    dimension, count, update_count, hold
                ),
                task,
                seed=condition_seed,
                sample_offset=start,
                device=device,
                audit=False,
            )
            logits_chunks.append(final_logits(model, batch).cpu())
            target_chunks.append(batch.final_targets.cpu())
            label_chunks.append(grid_labels(batch.final_targets, bins).cpu())
        logits = torch.cat(logits_chunks)
        targets = torch.cat(target_chunks)
        labels = torch.cat(label_chunks)
        predictions = logits.reshape(trajectories, dimension, bins).argmax(-1)
        coordinate_correct = (predictions == labels).numpy()
        exact_correct = coordinate_correct.all(axis=1)
        margin = _normalized_margin(targets, bins).numpy()
        bank_hash.update(regime.encode("utf-8"))
        bank_hash.update(targets.numpy().tobytes())
        metrics = classification_metrics(logits, labels, bins)
        rows[regime] = {
            "update_count": update_count,
            "segment_hold": hold,
            **metrics,
            "class_balance": class_balance(labels, bins),
            "margin_strata": _margin_metrics(
                margin, coordinate_correct, exact_correct
            ),
        }
        trial_targets.append(targets.numpy().astype(np.float32))
        trial_labels.append(labels.numpy().astype(np.int16))
        trial_coordinate_correct.append(coordinate_correct)
        trial_exact_correct.append(exact_correct)
        trial_margin.append(margin.astype(np.float32))

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bank_sha256 = bank_hash.hexdigest()
    atomic_npz(
        output / "trial_metrics.npz",
        regime=np.asarray(list(REPRESENTATIVE_CELLS)),
        target=np.stack(trial_targets),
        label=np.stack(trial_labels),
        coordinate_correct=np.stack(trial_coordinate_correct),
        exact_correct=np.stack(trial_exact_correct),
        normalized_boundary_margin=np.stack(trial_margin),
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "run_dir": str(root),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "model": spec["model"],
        "bins": bins,
        "dimension": dimension,
        "seed": int(spec["seed"]),
        "trajectories_per_condition": int(trajectories),
        "evaluation_batch_size": int(batch_size),
        "bank_seed": int(bank_seed),
        "bank_tag": str(bank_tag),
        "bank_sha256": bank_sha256,
        "conditions": rows,
    }
    atomic_json(output / "metrics.json", result)
    atomic_json(
        output / "COMPLETED.json",
        {
            "schema_version": 1,
            "status": "complete",
            "checkpoint_sha256": result["checkpoint_sha256"],
            "bank_sha256": bank_sha256,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trajectories", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bank-seed", type=int, default=20260720)
    parser.add_argument("--bank-tag", default="spatial-evaluation")
    args = parser.parse_args()
    result = evaluate(
        protocol_path=args.protocol,
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        device=args.device,
        trajectories=args.trajectories,
        batch_size=args.batch_size,
        bank_seed=args.bank_seed,
        bank_tag=args.bank_tag,
    )
    print(
        f"model={result['model']} bins={result['bins']} seed={result['seed']} "
        f"bank={result['bank_sha256']}"
    )


if __name__ == "__main__":
    main()
