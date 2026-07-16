"""Build frozen initializer, transported-endpoint and closed-path analysis banks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from repro.sagodi_protocol.artifacts import atomic_json, derived_seed

from .artifacts import sha256_file
from .generator import DELTA_T
from .topology_analysis_common import (
    TOPOLOGY_ORDER,
    atomic_npz,
    canonical_sha256,
    load_analysis_config,
)


def _wrap(value: np.ndarray) -> np.ndarray:
    return np.remainder(value + math.pi, 2.0 * math.pi) - math.pi


def _embed_torus(angles: np.ndarray) -> np.ndarray:
    output = np.empty((*angles.shape[:-1], 2 * angles.shape[-1]), dtype=np.float64)
    output[..., 0::2] = np.cos(angles)
    output[..., 1::2] = np.sin(angles)
    return output


def _fibonacci_sphere(count: int) -> np.ndarray:
    index = np.arange(int(count), dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / float(count)
    angle = math.pi * (3.0 - math.sqrt(5.0)) * index
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack((radius * np.cos(angle), radius * np.sin(angle), z), axis=-1)


def _sphere_tangent_basis(points: np.ndarray) -> np.ndarray:
    reference = np.zeros_like(points)
    reference[:, 2] = 1.0
    near = np.abs(points[:, 2]) > 0.9
    reference[near] = np.array([1.0, 0.0, 0.0])
    first = np.cross(reference, points)
    first /= np.linalg.norm(first, axis=-1, keepdims=True)
    second = np.cross(points, first)
    second /= np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack((first, second), axis=0)


def _smooth_commands(
    *,
    seed: int,
    horizon: int,
    batch: int,
    dimension: int,
    smoothing: float,
    std: float,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal((int(horizon), int(batch), int(dimension)))
    commands = np.empty_like(noise)
    innovation = math.sqrt(1.0 - float(smoothing) ** 2)
    commands[0] = noise[0]
    for step in range(1, int(horizon)):
        commands[step] = float(smoothing) * commands[step - 1] + innovation * noise[step]
    return float(std) * commands


def _rodrigues_step(points: np.ndarray, omega: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(omega, axis=-1, keepdims=True)
    alpha = float(DELTA_T) * speed
    result = points.copy()
    moving = speed[:, 0] > 1e-14
    if np.any(moving):
        axis = omega[moving] / speed[moving]
        state = points[moving]
        angle = alpha[moving]
        result[moving] = (
            np.cos(angle) * state
            + np.sin(angle) * np.cross(axis, state)
            + (1.0 - np.cos(angle))
            * np.sum(axis * state, axis=-1, keepdims=True)
            * axis
        )
        result[moving] /= np.linalg.norm(result[moving], axis=-1, keepdims=True)
    return result


def _integrate(
    topology: str, initial_latent: np.ndarray, commands: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if topology in {"s1", "t2"}:
        unwrapped = initial_latent[None] + float(DELTA_T) * np.concatenate(
            (
                np.zeros((1, initial_latent.shape[0], initial_latent.shape[1])),
                np.cumsum(commands, axis=0),
            ),
            axis=0,
        )
        latent_path = _wrap(unwrapped)
        return latent_path, _embed_torus(latent_path)
    path = np.empty((commands.shape[0] + 1, initial_latent.shape[0], 3), dtype=np.float64)
    path[0] = initial_latent
    for step in range(commands.shape[0]):
        path[step + 1] = _rodrigues_step(path[step], commands[step])
    return path, path.copy()


def _anchors(topology: str, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if topology == "s1":
        latent = np.linspace(-math.pi, math.pi, int(count), endpoint=False)[:, None]
        tangent = np.ones((1, int(count), 1), dtype=np.float64)
        return latent, _embed_torus(latent), tangent
    if topology == "t2":
        side = int(round(math.sqrt(int(count))))
        if side * side != int(count):
            raise ValueError("T2 atlas point count must be a square")
        angle = np.linspace(-math.pi, math.pi, side, endpoint=False)
        first, second = np.meshgrid(angle, angle, indexing="ij")
        latent = np.stack((first.reshape(-1), second.reshape(-1)), axis=-1)
        tangent = np.broadcast_to(np.eye(2)[:, None, :], (2, int(count), 2)).copy()
        return latent, _embed_torus(latent), tangent
    latent = _fibonacci_sphere(int(count))
    return latent, latent.copy(), _sphere_tangent_basis(latent)


def _displaced_initial_memory(
    topology: str,
    anchors: np.ndarray,
    tangent: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    dimensions = tangent.shape[0]
    plus: list[np.ndarray] = []
    minus: list[np.ndarray] = []
    for dimension in range(dimensions):
        direction = tangent[dimension]
        if topology in {"s1", "t2"}:
            plus.append(_embed_torus(_wrap(anchors + float(epsilon) * direction)))
            minus.append(_embed_torus(_wrap(anchors - float(epsilon) * direction)))
        else:
            plus_state = math.cos(epsilon) * anchors + math.sin(epsilon) * direction
            minus_state = math.cos(epsilon) * anchors - math.sin(epsilon) * direction
            plus.append(plus_state / np.linalg.norm(plus_state, axis=-1, keepdims=True))
            minus.append(minus_state / np.linalg.norm(minus_state, axis=-1, keepdims=True))
    return np.stack(plus), np.stack(minus)


def make_bank(topology: str, config: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    bank = config["analysis_banks"]
    geometry = config["quick_geometry"]
    count = int(bank["atlas_points"])
    anchors, initial_memory, tangent = _anchors(topology, count)
    input_dim = {"s1": 1, "t2": 2, "s2": 3}[topology]
    # One shared non-zero schedule makes the primary atlas a controlled image
    # of the uniform anchor set.  Path-history variability is measured only by
    # the separate multi-path closed bank below.
    common_transport = _smooth_commands(
        seed=derived_seed(int(bank["seed"]), topology, "transport"),
        horizon=int(bank["transport_horizon"]),
        batch=1,
        dimension=input_dim,
        smoothing=float(bank["command_smoothing"]),
        std=float(bank["command_std"]),
    )
    transport = np.repeat(common_transport, count, axis=1)
    transported_latent_path, transported_output_path = _integrate(
        topology, anchors, transport
    )
    tangent_plus, tangent_minus = _displaced_initial_memory(
        topology,
        anchors,
        tangent,
        float(geometry["tangent_finite_difference"]),
    )

    paths = int(bank["closed_path_count"])
    repeated_anchor = np.repeat(anchors, paths, axis=0)
    repeated_initial = np.repeat(initial_memory, paths, axis=0)
    forward = _smooth_commands(
        seed=derived_seed(int(bank["seed"]), topology, "closed"),
        horizon=int(bank["closed_half_horizon"]),
        batch=count * paths,
        dimension=input_dim,
        smoothing=float(bank["command_smoothing"]),
        std=float(bank["command_std"]),
    )
    closed = np.concatenate((forward, -forward[::-1]), axis=0)
    closed_latent_path, closed_output_path = _integrate(
        topology, repeated_anchor, closed
    )
    if topology in {"s1", "t2"}:
        closure = np.max(np.abs(_wrap(closed_latent_path[-1] - repeated_anchor)))
    else:
        cosine = np.sum(closed_latent_path[-1] * repeated_anchor, axis=-1)
        closure = np.max(np.arccos(np.clip(cosine, -1.0, 1.0)))
    if float(closure) > 1e-6:
        raise RuntimeError(f"{topology} closed path oracle error is {closure}")
    arrays = {
        "anchor_latent": anchors.astype(np.float32),
        "initializer_memory": initial_memory.astype(np.float32),
        "latent_tangent_basis": tangent.astype(np.float32),
        "transport_inputs": transport.astype(np.float32),
        "transport_endpoint_latent": transported_latent_path[-1].astype(np.float32),
        "transport_endpoint_target": transported_output_path[-1].astype(np.float32),
        "tangent_plus_initial_memory": tangent_plus.astype(np.float32),
        "tangent_minus_initial_memory": tangent_minus.astype(np.float32),
        "closed_inputs": closed.astype(np.float32),
        "closed_initial_memory": repeated_initial.astype(np.float32),
        "closed_anchor_index": np.repeat(np.arange(count), paths).astype(np.int64),
        "closed_path_index": np.tile(np.arange(paths), count).astype(np.int64),
        "closed_endpoint_target": closed_output_path[-1].astype(np.float32),
    }
    metadata = {
        "schema_version": 1,
        "analysis_id": config["analysis_id"],
        "topology": topology,
        "atlas_points": count,
        "transport_horizon": int(bank["transport_horizon"]),
        "closed_path_count": paths,
        "closed_half_horizon": int(bank["closed_half_horizon"]),
        "delta_t": float(DELTA_T),
        "oracle_max_closed_path_error_radians": float(closure),
        "initializer_atlas_is_diagnostic_only": True,
        "transported_endpoint_atlas_is_primary": True,
        "transport_control_pairing": (
            "one_common_nonzero_schedule_across_all_anchors_to_isolate_topology"
        ),
        "path_history_variation_source": "separate_closed_path_bank",
        "config_sha256": canonical_sha256(config),
    }
    arrays["metadata_json"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True, allow_nan=False).encode("utf-8"),
        dtype=np.uint8,
    )
    return arrays, metadata


def build(output: Path, config_path: Path) -> None:
    config = load_analysis_config(config_path)
    output.mkdir(parents=True, exist_ok=False)
    rows: dict[str, Any] = {}
    for topology in TOPOLOGY_ORDER:
        arrays, metadata = make_bank(topology, config)
        path = output / f"analysis_bank_{topology}.npz"
        atomic_npz(path, **arrays)
        # Independent round-trip verification; pickle is forbidden.
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise RuntimeError(f"{topology} bank keys changed on round trip")
            for name, value in arrays.items():
                if not np.array_equal(archive[name], value):
                    raise RuntimeError(f"{topology} bank array {name} changed")
        rows[topology] = {"path": path.name, "sha256": sha256_file(path), **metadata}
    atomic_json(
        output / "analysis_banks_manifest.json",
        {
            "schema_version": 1,
            "analysis_id": config["analysis_id"],
            "config_path": str(config_path.resolve()),
            "config_sha256": canonical_sha256(config),
            "banks": rows,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("topology_analysis_v1.json")
    )
    args = parser.parse_args()
    build(args.output.expanduser().resolve(), args.config.expanduser().resolve(strict=True))


if __name__ == "__main__":
    main()
