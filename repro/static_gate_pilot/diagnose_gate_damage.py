"""Measure gate-lesion damage on one fully pretrained checkpoint without updates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from repro.sagodi_protocol.artifacts import atomic_json

from .cache import cached_training_batch
from .models import StaticGateMemory
from .run import gate_intervention_damage


def _floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blank-horizons", default="128,512")
    parser.add_argument("--lambda-fast-values", default="0.5,0.8")
    parser.add_argument("--probe-batch-size", type=int, default=16)
    parser.add_argument("--data-update", type=int, default=5000)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    manifest = checkpoint["manifest"]
    metadata = manifest["model"]
    model = StaticGateMemory(
        model_id=metadata["model_id"],
        topology=metadata["topology"],
        width=int(metadata["width"]),
        initial_retention=float(metadata["initial_retention"]),
        initial_write_gain=float(metadata["initial_write_gain"]),
        recurrent_gain=float(metadata["recurrent_gain"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    probe = cached_training_batch(
        args.train_cache,
        topology=metadata["topology"],
        replicate_seed=int(manifest["replicate_seed"]),
        update=int(args.data_update),
        batch_size=int(args.probe_batch_size),
        device=device,
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    coordinate_rows: list[dict[str, float | int | str]] = []
    summaries: list[dict[str, float | int | str]] = []
    for horizon in _ints(args.blank_horizons):
        for lambda_fast in _floats(args.lambda_fast_values):
            damage, base = gate_intervention_damage(
                model,
                probe,
                blank_horizon=horizon,
                lambda_fast=lambda_fast,
            )
            values = damage.detach().cpu()
            quantiles = torch.quantile(
                values, torch.tensor([0.05, 0.5, 0.95])
            )
            summaries.append(
                {
                    "model": metadata["model_id"],
                    "topology": metadata["topology"],
                    "job_id": manifest["job_id"],
                    "blank_horizon": horizon,
                    "lambda_fast": lambda_fast,
                    **base,
                    "damage_min": float(values.min()),
                    "damage_p05": float(quantiles[0]),
                    "damage_median": float(quantiles[1]),
                    "damage_p95": float(quantiles[2]),
                    "damage_max": float(values.max()),
                    "damage_abs_median": float(values.abs().median()),
                    "damage_negative_fraction": float((values < 0.0).float().mean()),
                }
            )
            for coordinate, value in enumerate(values.tolist()):
                coordinate_rows.append(
                    {
                        "model": metadata["model_id"],
                        "topology": metadata["topology"],
                        "job_id": manifest["job_id"],
                        "blank_horizon": horizon,
                        "lambda_fast": lambda_fast,
                        "coordinate": coordinate,
                        "normalized_damage": float(value),
                        "baseline_lambda": float(
                            model.retention()[coordinate].detach().cpu()
                        ),
                    }
                )
    atomic_json(
        output / "summary.json",
        {
            "schema_version": 1,
            "diagnostic": "gate_intervention_damage_no_parameter_updates",
            "checkpoint": str(checkpoint_path),
            "probe_data_update": int(args.data_update),
            "probe_batch_size": int(args.probe_batch_size),
            "rows": summaries,
        },
    )
    with (output / "coordinate_damage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coordinate_rows[0]))
        writer.writeheader()
        writer.writerows(coordinate_rows)
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
