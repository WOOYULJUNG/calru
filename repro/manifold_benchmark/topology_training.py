"""Paired training data and metrics for the topology-transfer campaign."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from repro.sagodi_protocol.artifacts import derived_seed

from .artifacts import load_manifold_bank
from .generator import (
    ConditionSpec,
    ManifoldBatch,
    ParentSpec,
    derive_s1,
    derive_s2,
    derive_torus,
    make_parent_bank,
)


VALIDATION_ROOT = Path(
    "/home/biadmin/ca_rnn/experiments/"
    "manifold_benchmark_generator_v1_rodrigues-20260716"
)
TEST_ROOT = Path(
    "/home/biadmin/ca_rnn/experiments/"
    "manifold_benchmark_generator_v1_rodrigues-test-20260716"
)
TOPOLOGIES = ("s1", "t2", "s2")


@dataclass(frozen=True)
class TopologyBatch:
    """Torch view of one generator-v1 batch, keeping the true q0 explicit."""

    topology: str
    initial_memory: torch.Tensor
    inputs: torch.Tensor
    output_targets: torch.Tensor
    latent_targets: torch.Tensor
    latent_path: torch.Tensor
    trajectory_id: torch.Tensor
    generator_metadata: dict[str, Any]

    @property
    def horizon(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.inputs.shape[1])


def _topology_key(topology: str) -> str:
    key = str(topology).lower()
    if key not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}")
    return key


def torch_batch(
    batch: ManifoldBatch,
    *,
    device: torch.device | str,
    trajectories: int | None = None,
    horizon: int | None = None,
) -> TopologyBatch:
    count = batch.trajectories if trajectories is None else int(trajectories)
    if not 0 < count <= batch.trajectories:
        raise ValueError("trajectory subset must be within the bank")
    steps = batch.horizon if horizon is None else int(horizon)
    if not 0 < steps <= batch.horizon:
        raise ValueError("horizon subset must be within the bank")
    topology = str(batch.metadata["topology"]).lower()
    if topology == "t2":
        topology = "t2"
    elif topology == "s1":
        topology = "s1"
    elif topology == "s2":
        topology = "s2"
    else:
        raise ValueError(f"unsupported bank topology {topology!r}")

    def floating(value: np.ndarray, *, path: bool = False) -> torch.Tensor:
        if value.ndim == 3:
            stop = steps + 1 if path else steps
            selected = value[:stop, :count, :]
        else:
            selected = value[:count]
        return torch.as_tensor(
            np.array(selected, copy=True),
            dtype=torch.float32,
            device=device,
        )

    return TopologyBatch(
        topology=topology,
        initial_memory=floating(batch.initial_memory),
        inputs=floating(batch.inputs),
        output_targets=floating(batch.output_targets),
        latent_targets=floating(batch.latent_targets),
        latent_path=floating(batch.latent_path, path=True),
        trajectory_id=torch.as_tensor(
            np.array(batch.trajectory_id[:count], copy=True),
            dtype=torch.int64,
            device=device,
        ),
        generator_metadata=dict(batch.metadata),
    )


def _derive(parent, topology: str, condition: ConditionSpec) -> ManifoldBatch:
    key = _topology_key(topology)
    if key == "s1":
        return derive_s1(parent, condition=condition)
    if key == "t2":
        return derive_torus(parent, dimensions=2, condition=condition)
    return derive_s2(parent, condition=condition)


def generated_batch(
    topology: str,
    *,
    replicate_seed: int,
    update: int,
    batch_size: int,
    horizon: int,
    device: torch.device | str,
    stream: str = "online_train",
) -> TopologyBatch:
    """Make a model-independent paired batch keyed only by replicate/update."""

    sample_seed = derived_seed(
        int(replicate_seed), "topology_transfer_v1", str(stream), int(update)
    )
    parent = make_parent_bank(
        ParentSpec(
            trajectories=int(batch_size),
            max_horizon=max(128, int(horizon)),
            max_torus_dimension=8,
            training_horizon=128,
            task_seed=20260716,
            sample_seed=sample_seed,
            split=f"{stream}_replicate_{int(replicate_seed)}",
        )
    )
    batch = _derive(parent, topology, ConditionSpec(horizon=int(horizon)))
    return torch_batch(batch, device=device)


def fixed_debug_batch(
    topology: str,
    *,
    debug_seed: int,
    batch_size: int,
    horizon: int,
    device: torch.device | str,
) -> TopologyBatch:
    return generated_batch(
        topology,
        replicate_seed=int(debug_seed),
        update=0,
        batch_size=int(batch_size),
        horizon=int(horizon),
        device=device,
        stream="fixed_overfit",
    )


def load_fixed_bank(
    topology: str,
    *,
    split: str,
    device: torch.device | str,
    trajectories: int | None = None,
    horizon: int | None = None,
) -> TopologyBatch:
    key = _topology_key(topology)
    if split == "validation":
        root = VALIDATION_ROOT
    elif split == "test":
        root = TEST_ROOT
    else:
        raise ValueError("split must be validation or test")
    return torch_batch(
        load_manifold_bank(root / f"id_{key}.npz"),
        device=device,
        trajectories=trajectories,
        horizon=horizon,
    )


def component_mean_mse(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape differs: {tuple(prediction.shape)} != "
            f"{tuple(target.shape)}"
        )
    return (prediction - target).square().mean()


def _wrapped_absolute(value: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.remainder(value + math.pi, 2.0 * math.pi) - math.pi)


@torch.no_grad()
def intrinsic_metrics(
    topology: str, prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    key = _topology_key(topology)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shape differs")
    result: dict[str, float] = {
        "component_mse": float(component_mean_mse(prediction, target).cpu())
    }
    if key in {"s1", "t2"}:
        pair_count = 1 if key == "s1" else 2
        prediction_pair = prediction.reshape(*prediction.shape[:-1], pair_count, 2)
        target_pair = target.reshape(*target.shape[:-1], pair_count, 2)
        prediction_angle = torch.atan2(prediction_pair[..., 1], prediction_pair[..., 0])
        target_angle = torch.atan2(target_pair[..., 1], target_pair[..., 0])
        errors = _wrapped_absolute(prediction_angle - target_angle)
        norms = torch.linalg.vector_norm(prediction_pair, dim=-1)
        result.update(
            {
                "intrinsic_mean_radians": float(errors.mean().cpu()),
                "intrinsic_worst_coordinate_mean_radians": float(
                    errors.max(dim=-1).values.mean().cpu()
                ),
                "output_norm_absolute_error": float(
                    torch.abs(norms - 1.0).mean().cpu()
                ),
            }
        )
    else:
        norms = torch.linalg.vector_norm(prediction, dim=-1)
        normalized = prediction / norms.clamp_min(1e-8).unsqueeze(-1)
        target_normalized = target / torch.linalg.vector_norm(
            target, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        cosine = (normalized * target_normalized).sum(dim=-1).clamp(-1.0, 1.0)
        error = torch.acos(cosine)
        result.update(
            {
                "intrinsic_mean_radians": float(error.mean().cpu()),
                "intrinsic_worst_coordinate_mean_radians": float(error.mean().cpu()),
                "output_norm_absolute_error": float(
                    torch.abs(norms - 1.0).mean().cpu()
                ),
            }
        )
    return result


def hold_prediction(batch: TopologyBatch) -> torch.Tensor:
    return batch.initial_memory.unsqueeze(0).expand(
        batch.horizon, batch.batch_size, batch.initial_memory.shape[-1]
    )


__all__ = [
    "TEST_ROOT",
    "TOPOLOGIES",
    "TopologyBatch",
    "VALIDATION_ROOT",
    "component_mean_mse",
    "fixed_debug_batch",
    "generated_batch",
    "hold_prediction",
    "intrinsic_metrics",
    "load_fixed_bank",
    "torch_batch",
]
