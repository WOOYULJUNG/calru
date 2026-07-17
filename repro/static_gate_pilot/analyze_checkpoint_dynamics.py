"""Analyze local tangent/normal dynamics of one static-gate checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from repro.manifold_benchmark.topology_training import (
    intrinsic_metrics,
    load_fixed_bank,
)
from repro.manifold_benchmark.topology_models import (
    build_topology_model,
    load_transfer_config,
)
from repro.sagodi_protocol.artifacts import atomic_json, derived_seed

from .models import StaticGateMemory


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    manifest = checkpoint["manifest"]
    metadata = manifest["model"]
    if checkpoint.get("checkpoint_type") == "static_gate_side_pilot_v1":
        model = StaticGateMemory(
            model_id=metadata["model_id"],
            topology=metadata["topology"],
            width=int(metadata["width"]),
            initial_retention=float(metadata["initial_retention"]),
            initial_write_gain=float(metadata["initial_write_gain"]),
            recurrent_gain=float(metadata["recurrent_gain"]),
        ).to(device)
    elif manifest.get("campaign_id") == "manifold_topology_transfer_v1":
        model = build_topology_model(
            str(metadata["model_id"]),
            str(manifest["topology"]),
            model_seed=int(manifest["model_seed"]),
            config=load_transfer_config(),
        ).to(device)
    else:
        raise ValueError(
            f"unsupported checkpoint schema for dynamics analysis: "
            f"{manifest.get('campaign_id')!r}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _wrapped(value: torch.Tensor) -> torch.Tensor:
    return torch.remainder(value + math.pi, 2.0 * math.pi) - math.pi


def _intrinsic_coordinates(topology: str, target: torch.Tensor) -> torch.Tensor:
    if topology == "s2":
        return target
    pairs = target.reshape(target.shape[0], -1, 2)
    return torch.atan2(pairs[..., 1], pairs[..., 0])


def _local_tangent_bases(
    states: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    topology: str,
    tangent_dimension: int,
    neighbors: int,
    anchor_indices: torch.Tensor,
) -> list[torch.Tensor]:
    bases: list[torch.Tensor] = []
    for index in anchor_indices.tolist():
        if topology == "s2":
            distances = torch.acos(
                (coordinates @ coordinates[index]).clamp(-1.0, 1.0)
            )
        else:
            delta = _wrapped(coordinates - coordinates[index])
            distances = torch.linalg.vector_norm(delta, dim=-1)
        distances[index] = float("inf")
        neighbor_indices = torch.argsort(distances)[:neighbors]
        local = states[neighbor_indices] - states[index]
        left, _, _ = torch.linalg.svd(local.T, full_matrices=False)
        bases.append(left[:, :tangent_dimension])
    return bases


def _local_jacobian(
    model: nn.Module, state: torch.Tensor
) -> torch.Tensor:
    blank = torch.zeros(
        1, model.input_dim, device=state.device, dtype=state.dtype
    )

    def mapping(value: torch.Tensor) -> torch.Tensor:
        return model.step(blank, value.unsqueeze(0)).squeeze(0)

    with torch.enable_grad():
        value = state.detach().requires_grad_(True)
        return torch.autograd.functional.jacobian(
            mapping, value, vectorize=True
        ).detach()


@torch.no_grad()
def _roll_blank(
    model: nn.Module, state: torch.Tensor, horizon: int
) -> torch.Tensor:
    blank = torch.zeros(
        state.shape[0], model.input_dim, device=state.device, dtype=state.dtype
    )
    current = state
    for _ in range(int(horizon)):
        current = model.step(blank, current)
    return current


@torch.no_grad()
def _roll_blank_snapshots(
    model: nn.Module,
    state: torch.Tensor,
    horizons: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    requested = set(int(value) for value in horizons)
    snapshots: dict[int, torch.Tensor] = {}
    blank = torch.zeros(
        state.shape[0], model.input_dim, device=state.device, dtype=state.dtype
    )
    current = state
    for step in range(1, max(requested) + 1):
        current = model.step(blank, current)
        if step in requested:
            snapshots[step] = current
    return snapshots


def _participation_ratio(states: torch.Tensor) -> float:
    centered = states - states.mean(dim=0, keepdim=True)
    values = torch.linalg.eigvalsh(centered.T @ centered).clamp_min(0.0)
    return float((values.sum().square() / values.square().sum().clamp_min(1e-12)).cpu())


def analyze(
    checkpoint_path: Path,
    *,
    device: torch.device,
    trajectories: int,
    anchors: int,
    neighbors: int,
    kick_relative_to_neighbor: float,
) -> dict:
    model, checkpoint = _load_model(checkpoint_path, device)
    manifest = checkpoint["manifest"]
    topology = str(manifest["topology"])
    batch = load_fixed_bank(
        topology,
        split="validation",
        device=device,
        trajectories=trajectories,
        horizon=128,
    )
    with torch.no_grad():
        prediction, sequence = model.forward_sequence(
            batch.inputs,
            initial_memory=batch.initial_memory,
            return_states=True,
        )
        states = sequence[-1]
        target = batch.output_targets[-1]
        coordinates = _intrinsic_coordinates(topology, target)
        pairwise = torch.cdist(states, states)
        pairwise.fill_diagonal_(float("inf"))
        neighbor_scale = pairwise.min(dim=1).values.median()
        blank = torch.zeros(
            states.shape[0], model.input_dim, device=device, dtype=states.dtype
        )
        next_states = model.step(blank, states)
        blank_field = next_states - states
    tangent_dimension = 1 if topology == "s1" else 2
    anchor_indices = torch.linspace(
        0, trajectories - 1, anchors, device=device
    ).round().long()
    tangent_bases = _local_tangent_bases(
        states,
        coordinates,
        topology=topology,
        tangent_dimension=tangent_dimension,
        neighbors=neighbors,
        anchor_indices=anchor_indices,
    )
    tangent_singular: list[float] = []
    normal_max_singular: list[float] = []
    tangent_to_normal: list[float] = []
    field_tangent_fraction: list[float] = []
    worst_normal_directions: list[torch.Tensor] = []
    for anchor_index, tangent in zip(anchor_indices.tolist(), tangent_bases):
        jacobian = _local_jacobian(model, states[anchor_index])
        complete, _ = torch.linalg.qr(tangent, mode="complete")
        normal = complete[:, tangent_dimension:]
        tangent_block = tangent.T @ jacobian @ tangent
        normal_block = normal.T @ jacobian @ normal
        tangent_values = torch.linalg.svdvals(tangent_block)
        normal_u, normal_values, normal_vh = torch.linalg.svd(
            normal_block, full_matrices=False
        )
        del normal_u
        tangent_singular.append(float(tangent_values.mean().cpu()))
        normal_max_singular.append(float(normal_values.max().cpu()))
        tangent_to_normal.append(
            float(torch.linalg.matrix_norm(normal.T @ jacobian @ tangent).cpu())
        )
        field = blank_field[anchor_index]
        tangent_field = tangent @ (tangent.T @ field)
        field_tangent_fraction.append(
            float(
                (
                    torch.linalg.vector_norm(tangent_field)
                    / torch.linalg.vector_norm(field).clamp_min(1e-12)
                ).cpu()
            )
        )
        worst_normal_directions.append(normal @ normal_vh[0])

    with torch.no_grad():
        anchor_states = states[anchor_indices]
        directions = torch.stack(worst_normal_directions)
        directions = directions / torch.linalg.vector_norm(
            directions, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        kick_size = float(kick_relative_to_neighbor) * neighbor_scale
        kicked0 = anchor_states + kick_size * directions
        tangent_directions = torch.stack([basis[:, 0] for basis in tangent_bases])
        tangent_directions = tangent_directions / torch.linalg.vector_norm(
            tangent_directions, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        tangent_kicked0 = anchor_states + kick_size * tangent_directions
        initial_distance = torch.cdist(kicked0, states).min(dim=1).values
        tangent_initial_distance = torch.cdist(
            tangent_kicked0, states
        ).min(dim=1).values
        recovery: dict[str, dict[str, float]] = {}
        tangent_transport: dict[str, dict[str, float]] = {}
        for horizon in (128, 512):
            clean_h = _roll_blank(model, states, horizon)
            kicked_h = _roll_blank(model, kicked0, horizon)
            tangent_kicked_h = _roll_blank(model, tangent_kicked0, horizon)
            final_distance = torch.cdist(kicked_h, clean_h).min(dim=1).values
            tangent_final_distance = torch.cdist(
                tangent_kicked_h, clean_h
            ).min(dim=1).values
            ratio = final_distance / initial_distance.clamp_min(1e-12)
            tangent_ratio = tangent_final_distance / tangent_initial_distance.clamp_min(
                1e-12
            )
            clean_anchor_h = clean_h[anchor_indices]
            same_memory = intrinsic_metrics(
                topology,
                model.decode(kicked_h).unsqueeze(0),
                model.decode(clean_anchor_h).unsqueeze(0),
            )
            recovery[str(horizon)] = {
                "distance_ratio_median": float(ratio.median().cpu()),
                "distance_ratio_mean": float(ratio.mean().cpu()),
                "same_memory_intrinsic_radians": float(
                    same_memory["intrinsic_mean_radians"]
                ),
            }
            tangent_memory = intrinsic_metrics(
                topology,
                model.decode(tangent_kicked_h).unsqueeze(0),
                model.decode(clean_anchor_h).unsqueeze(0),
            )
            tangent_transport[str(horizon)] = {
                "distance_ratio_median": float(tangent_ratio.median().cpu()),
                "distance_ratio_mean": float(tangent_ratio.mean().cpu()),
                "memory_shift_intrinsic_radians": float(
                    tangent_memory["intrinsic_mean_radians"]
                ),
            }

        manifold_evolution: dict[str, dict[str, float]] = {}
        snapshots = _roll_blank_snapshots(model, states, (128, 512, 2048))
        diameter = torch.cdist(states, states).max().clamp_min(1e-12)
        base_norm_square = states.square().sum().clamp_min(1e-12)
        base_pairwise = torch.pdist(states).clamp_min(1e-12)
        for horizon, evolved in snapshots.items():
            global_scale = (evolved * states).sum() / base_norm_square
            scaling_residual = torch.linalg.vector_norm(
                evolved - global_scale * states
            ) / torch.linalg.vector_norm(states).clamp_min(1e-12)
            log_pairwise_scale = torch.log(
                torch.pdist(evolved).clamp_min(1e-12) / base_pairwise
            )
            centered_distortion = (
                log_pairwise_scale - log_pairwise_scale.median()
            )
            state_cosine = torch.nn.functional.cosine_similarity(
                evolved, states, dim=-1
            )
            decoded_error = intrinsic_metrics(
                topology,
                model.decode(evolved).unsqueeze(0),
                target.unsqueeze(0),
            )
            manifold_evolution[str(horizon)] = {
                "raw_stationarity_median": float(
                    (
                        torch.linalg.vector_norm(evolved - states, dim=-1).median()
                        / diameter
                    ).cpu()
                ),
                "best_global_scale": float(global_scale.cpu()),
                "global_scaling_residual": float(scaling_residual.cpu()),
                "pairwise_log_scale_median": float(
                    log_pairwise_scale.median().cpu()
                ),
                "pairwise_shape_distortion_std": float(
                    centered_distortion.std(unbiased=False).cpu()
                ),
                "state_norm_ratio_median": float(
                    (
                        torch.linalg.vector_norm(evolved, dim=-1)
                        / torch.linalg.vector_norm(states, dim=-1).clamp_min(1e-12)
                    ).median().cpu()
                ),
                "state_direction_cosine_median": float(
                    state_cosine.median().cpu()
                ),
                "decoded_memory_intrinsic_radians": float(
                    decoded_error["intrinsic_mean_radians"]
                ),
            }
        retention_method = getattr(model, "retention", None)
        retention = (
            retention_method()
            if callable(retention_method)
            else torch.empty(0, device=device)
        )
        task = intrinsic_metrics(topology, prediction, batch.output_targets)
    return {
        "schema_version": 1,
        "analysis": "local_tangent_normal_dynamics_v1",
        "checkpoint": str(checkpoint_path),
        "job_id": manifest["job_id"],
        "model": model.model_id,
        "topology": topology,
        "seed": manifest["replicate_seed"],
        "phase": manifest.get("phase", manifest.get("stage")),
        "trajectories": trajectories,
        "anchors": anchors,
        "local_neighbors": neighbors,
        "task_intrinsic_radians": task["intrinsic_mean_radians"],
        "endpoint_participation_ratio": _participation_ratio(states),
        "neighbor_scale_median": float(neighbor_scale.cpu()),
        "blank_field_norm_median": float(
            torch.linalg.vector_norm(blank_field, dim=-1).median().cpu()
        ),
        "normalized_fixedness_median": float(
            (
                torch.linalg.vector_norm(blank_field, dim=-1).median()
                / neighbor_scale.clamp_min(1e-12)
            ).cpu()
        ),
        "tangent_singular_mean": float(
            torch.tensor(tangent_singular).mean()
        ),
        "tangent_neutral_error_mean": float(
            torch.abs(torch.tensor(tangent_singular) - 1.0).mean()
        ),
        "normal_max_singular_mean": float(
            torch.tensor(normal_max_singular).mean()
        ),
        "normal_max_singular_worst": float(
            torch.tensor(normal_max_singular).max()
        ),
        "tangent_normal_gap_mean": float(
            torch.tensor(tangent_singular).mean()
            - torch.tensor(normal_max_singular).mean()
        ),
        "tangent_to_normal_coupling_mean": float(
            torch.tensor(tangent_to_normal).mean()
        ),
        "blank_field_tangent_fraction_mean": float(
            torch.tensor(field_tangent_fraction).mean()
        ),
        "finite_local_normal_recovery": recovery,
        "finite_local_tangent_transport": tangent_transport,
        "blank_manifold_evolution": manifold_evolution,
        "lambda_min": (
            float(retention.min().cpu()) if retention.numel() else None
        ),
        "lambda_mean": (
            float(retention.mean().cpu()) if retention.numel() else None
        ),
        "lambda_max": (
            float(retention.max().cpu()) if retention.numel() else None
        ),
        "lambda_std": (
            float(retention.std(unbiased=False).cpu())
            if retention.numel()
            else None
        ),
        "lambda_values": [float(value) for value in retention.cpu().tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=128)
    parser.add_argument("--anchors", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--kick-relative-to-neighbor", type=float, default=0.5)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    result = analyze(
        args.checkpoint.expanduser().resolve(strict=True),
        device=device,
        trajectories=int(args.trajectories),
        anchors=int(args.anchors),
        neighbors=int(args.neighbors),
        kick_relative_to_neighbor=float(args.kick_relative_to_neighbor),
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "dynamics.json", result)
    print(output / "dynamics.json", flush=True)


if __name__ == "__main__":
    main()
