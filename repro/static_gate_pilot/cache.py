"""Build and read one train-only generator-v1 pool shared by pilot models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from repro.manifold_benchmark.generator import (
    ConditionSpec,
    ParentSpec,
    derive_s1,
    derive_torus,
    make_parent_bank,
)
from repro.sagodi_protocol.artifacts import derived_seed


ARRAY_NAMES = (
    "s1_initial_memory",
    "s1_inputs",
    "s1_output_targets",
    "t2_initial_memory",
    "t2_inputs",
    "t2_output_targets",
)


def build_training_cache(
    root: Path,
    *,
    seed: int,
    trajectories: int = 4096,
    horizon: int = 128,
) -> Path:
    """Generate one disjoint train-only pool, then discard parent auxiliaries."""

    root = root.expanduser().resolve()
    metadata_path = root / "metadata.json"
    if metadata_path.is_file() and all((root / f"{name}.npy").is_file() for name in ARRAY_NAMES):
        metadata = json.loads(metadata_path.read_text())
        if (
            int(metadata["seed"]) == int(seed)
            and int(metadata["trajectories"]) == int(trajectories)
            and int(metadata["horizon"]) == int(horizon)
        ):
            return root
        raise ValueError("existing static-gate training cache has a different specification")

    root.mkdir(parents=True, exist_ok=True)
    sample_seed = derived_seed(int(seed), "static_gate_side_pilot_v1", "train_pool")
    parent = make_parent_bank(
        ParentSpec(
            trajectories=int(trajectories),
            max_horizon=int(horizon),
            max_torus_dimension=8,
            training_horizon=128,
            task_seed=20260716,
            sample_seed=int(sample_seed),
            split=f"static_gate_train_pool_seed_{int(seed)}",
        )
    )
    condition = ConditionSpec(horizon=int(horizon))
    banks = {
        "s1": derive_s1(parent, condition=condition),
        "t2": derive_torus(parent, dimensions=2, condition=condition),
    }
    shapes: dict[str, list[int]] = {}
    for topology, bank in banks.items():
        for field in ("initial_memory", "inputs", "output_targets"):
            name = f"{topology}_{field}"
            value = np.asarray(getattr(bank, field), dtype=np.float32)
            np.save(root / f"{name}.npy", value, allow_pickle=False)
            shapes[name] = list(value.shape)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache_id": "static_gate_train_pool_v1",
                "generator": "manifold-benchmark-generator-v1",
                "split": "train_only_disjoint_from_frozen_validation_and_test",
                "seed": int(seed),
                "sample_seed": int(sample_seed),
                "trajectories": int(trajectories),
                "horizon": int(horizon),
                "task_seed": 20260716,
                "arrays": shapes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return root


def cached_training_batch(
    root: Path,
    *,
    topology: str,
    replicate_seed: int,
    update: int,
    batch_size: int,
    device: torch.device | str,
) -> SimpleNamespace:
    """Sample paired model-independent indices from a memory-mapped train pool."""

    root = root.expanduser().resolve(strict=True)
    metadata = json.loads((root / "metadata.json").read_text())
    if topology not in {"s1", "t2"}:
        raise ValueError("static-gate cache only contains s1 and t2")
    trajectories = int(metadata["trajectories"])
    if not 0 < int(batch_size) <= trajectories:
        raise ValueError("batch size must fit the training cache")
    index_seed = derived_seed(
        int(replicate_seed), "static_gate_train_pool_v1", int(update)
    )
    indices = np.random.default_rng(index_seed).choice(
        trajectories, size=int(batch_size), replace=False
    )
    initial = np.load(
        root / f"{topology}_initial_memory.npy", mmap_mode="r", allow_pickle=False
    )
    inputs = np.load(
        root / f"{topology}_inputs.npy", mmap_mode="r", allow_pickle=False
    )
    targets = np.load(
        root / f"{topology}_output_targets.npy", mmap_mode="r", allow_pickle=False
    )
    return SimpleNamespace(
        topology=topology,
        initial_memory=torch.as_tensor(
            np.array(initial[indices], copy=True), device=device
        ),
        inputs=torch.as_tensor(
            np.array(inputs[:, indices, :], copy=True), device=device
        ),
        output_targets=torch.as_tensor(
            np.array(targets[:, indices, :], copy=True), device=device
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--trajectories", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=128)
    args = parser.parse_args()
    print(
        build_training_cache(
            args.root,
            seed=args.seed,
            trajectories=args.trajectories,
            horizon=args.horizon,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
