"""Persistent-homology signature of one split-field checkpoint under blank input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from repro.manifold_benchmark.analyze_persistent_topology import (
    _compute_diagrams,
    _farthest_landmarks,
    _persistences,
    _strong_feature_count,
)
from repro.manifold_benchmark.topology_training import load_fixed_bank
from repro.sagodi_protocol.artifacts import atomic_json

from .analyze_checkpoint_dynamics import (
    _load_model,
    _roll_blank_snapshots,
)


EXPECTED = {
    "s1": {"h1": 1, "h2": 0},
    "t2": {"h1": 2, "h2": 1},
    "s2": {"h1": 0, "h2": 1},
}


def _diagram_summary(
    diagrams: list[np.ndarray],
    *,
    threshold: float,
    expected: dict[str, int],
) -> dict:
    counts = {
        "h1": _strong_feature_count(diagrams[1], threshold),
        "h2": _strong_feature_count(diagrams[2], threshold),
    }
    return {
        "detected_h1": counts["h1"],
        "detected_h2": counts["h2"],
        "expected_h1": expected["h1"],
        "expected_h2": expected["h2"],
        "signature_match": counts == expected,
        "h1_top_persistences": _persistences(diagrams[1])[:5].tolist(),
        "h2_top_persistences": _persistences(diagrams[2])[:5].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=256)
    parser.add_argument("--landmarks", type=int, default=128)
    parser.add_argument("--horizons", default="0,128,512,2048")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, payload = _load_model(checkpoint, device)
    topology = str(payload["manifest"]["topology"])
    batch = load_fixed_bank(
        topology,
        split="validation",
        device=device,
        trajectories=int(args.trajectories),
        horizon=128,
    )
    with torch.no_grad():
        _, sequence = model.forward_sequence(
            batch.inputs,
            initial_memory=batch.initial_memory,
            return_states=True,
        )
        states0 = sequence[-1]
        ideal = batch.output_targets[-1]
    landmark_indices = _farthest_landmarks(
        ideal.detach().cpu().numpy(), int(args.landmarks)
    )
    ideal_points = ideal[landmark_indices].detach().cpu().numpy()
    ideal_diagrams, ideal_scale, ideal_zero = _compute_diagrams(
        ideal_points, maxdim=2, k=5
    )
    expected = EXPECTED[topology]
    required = []
    for dimension in (1, 2):
        count = expected[f"h{dimension}"]
        if count:
            persistence = _persistences(ideal_diagrams[dimension])
            if len(persistence) < count:
                raise RuntimeError("ideal reference lacks expected topology")
            required.append(float(persistence[count - 1]))
    threshold = 0.5 * min(required)
    ideal_summary = _diagram_summary(
        ideal_diagrams, threshold=threshold, expected=expected
    )
    if not ideal_summary["signature_match"]:
        raise RuntimeError(f"ideal signature check failed: {ideal_summary}")

    horizons = sorted(
        {int(value) for value in args.horizons.split(",") if value.strip()}
    )
    positive = tuple(value for value in horizons if value > 0)
    snapshots = (
        _roll_blank_snapshots(model, states0, positive) if positive else {}
    )
    snapshots[0] = states0
    rows = {}
    for horizon in horizons:
        points = (
            snapshots[horizon][landmark_indices].detach().cpu().numpy()
        )
        diagrams, scale, zero_fraction = _compute_diagrams(
            points, maxdim=2, k=5
        )
        rows[str(horizon)] = {
            **_diagram_summary(
                diagrams, threshold=threshold, expected=expected
            ),
            "knn_scale": scale,
            "zero_knn_fraction": zero_fraction,
        }
    result = {
        "schema_version": 1,
        "analysis": "split_field_persistent_topology_v1",
        "checkpoint": str(checkpoint),
        "job_id": payload["manifest"]["job_id"],
        "model": payload["manifest"]["model"]["model_id"],
        "topology": topology,
        "seed": int(payload["manifest"]["replicate_seed"]),
        "trajectories": int(args.trajectories),
        "landmarks": int(args.landmarks),
        "strong_bar_threshold": threshold,
        "ideal_knn_scale": ideal_scale,
        "ideal_zero_knn_fraction": ideal_zero,
        "ideal": ideal_summary,
        "horizons": rows,
    }
    atomic_json(output / "topology.json", result)
    print(output / "topology.json")


if __name__ == "__main__":
    main()
