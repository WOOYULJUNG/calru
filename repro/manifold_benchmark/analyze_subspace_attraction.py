"""Separate slow-subspace retention from manifold-specific attraction.

The common comparison estimates a global finite-time slow-response subspace
from clean-atlas Jacobians.  At every anchor, perturbations are then separated
into the local tangent, the part normal to the manifold but inside that global
subspace, and directions outside both the subspace and tangent.  CA-LRU also
receives an architecture-level split using its explicit high-retention axes.

All finite-kick distances use the time-evolved clean atlas
``M_H = F_0^H(M_0)``.  The script intentionally consumes existing checkpoints
and cached representative atlases; it performs no training and never opens a
test bank.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import jacrev, vmap

from repro.sagodi_protocol.artifacts import atomic_json, derived_seed, strict_json_load

from .artifacts import sha256_file
from .topology_analysis_common import (
    RunRecord,
    atomic_npz,
    blank_snapshots,
    decode_primary,
    load_model,
    normalized_geodesic_errors,
    write_csv,
)


MODEL_ORDER = ("rnn", "gru", "lstm", "calru")
TOPOLOGY_ORDER = ("s1", "t2", "s2")
MODEL_LABEL = {
    "rnn": "RNN",
    "gru": "GRU",
    "lstm": "LSTM",
    "calru": "CA-LRU",
}
TOPOLOGY_LABEL = {"s1": r"$S^1$", "t2": r"$T^2$", "s2": r"$S^2$"}
MODEL_COLOR = {
    "rnn": "#767676",
    "gru": "#3377B5",
    "lstm": "#E08B2C",
    "calru": "#B13B47",
}


@dataclass(frozen=True)
class Target:
    record: RunRecord
    analysis_root: Path
    atlas_cache: Path
    task_success: bool
    validation_error: float
    failed_task_fallback: bool


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _load_config(path: Path) -> dict[str, Any]:
    payload = strict_json_load(path.expanduser().resolve(strict=True))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("analysis_id")
        != "manifold_topology_subspace_attraction_v1"
    ):
        raise ValueError("subspace-attraction analysis config differs")
    if tuple(payload["models"]) != MODEL_ORDER:
        raise ValueError("model order differs")
    if tuple(payload["topologies"]) != TOPOLOGY_ORDER:
        raise ValueError("topology order differs")
    return payload


def _git_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
    ).strip()
    return {"code_commit": commit, "worktree_dirty": bool(status)}


def _target_record(run_root: Path, row: Mapping[str, str]) -> RunRecord:
    job_id = str(row["job_id"])
    run_dir = run_root / job_id
    required = (
        run_dir / "COMPLETED.json",
        run_dir / "manifest.json",
        run_dir / "result.json",
        run_dir / "checkpoint.pt",
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"incomplete checkpoint artifacts for {job_id}")
    manifest = strict_json_load(run_dir / "manifest.json")
    result = strict_json_load(run_dir / "result.json")
    return RunRecord(
        run_dir=run_dir,
        job_id=job_id,
        model_id=str(result["model_id"]),
        topology=str(result["topology"]),
        seed=int(result["replicate_seed"]),
        manifest=manifest,
        result=result,
    )


def discover_representatives(
    *,
    baseline_run_root: Path,
    baseline_analysis_root: Path,
    calru_run_root: Path,
    calru_analysis_root: Path,
    comparison_root: Path,
) -> list[Target]:
    comparison = strict_json_load(comparison_root / "comparison_manifest.json")
    representatives = comparison["display_atlas"]["representative_seeds"]
    baseline_rows = [
        row
        for row in _read_csv(
            baseline_analysis_root / "task" / "task_metrics.csv"
        )
        if row["model"] in {"rnn", "gru", "lstm"}
    ]
    calru_rows = [
        row
        for row in _read_csv(
            calru_analysis_root / "task" / "task_metrics.csv"
        )
        if row["model"] == "calru"
    ]
    rows = baseline_rows + calru_rows
    index: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["topology"]), int(row["seed"]))
        if key in index:
            raise ValueError(f"duplicate task row {key}")
        index[key] = row

    targets: list[Target] = []
    for topology in TOPOLOGY_ORDER:
        for model in MODEL_ORDER:
            seed = int(representatives[topology][model])
            key = (model, topology, seed)
            if key not in index:
                raise KeyError(f"representative task row missing: {key}")
            row = index[key]
            analysis_root = (
                calru_analysis_root if model == "calru" else baseline_analysis_root
            )
            run_root = calru_run_root if model == "calru" else baseline_run_root
            record = _target_record(run_root, row)
            if (
                record.model_id != model
                or record.topology != topology
                or record.seed != seed
            ):
                raise ValueError(f"representative checkpoint mismatch: {key}")
            cache = (
                comparison_root
                / "display_atlas"
                / f"seed{seed}__{model}__{topology}.npz"
            )
            if not cache.is_file():
                raise FileNotFoundError(cache)
            targets.append(
                Target(
                    record=record,
                    analysis_root=analysis_root,
                    atlas_cache=cache,
                    task_success=_as_bool(row["task_success"]),
                    validation_error=float(row["validation_error"]),
                    failed_task_fallback=not _as_bool(row["task_success"]),
                )
            )
    return targets


def orthonormal_columns(raw: torch.Tensor) -> torch.Tensor:
    if raw.ndim != 2:
        raise ValueError("basis input must be rank two")
    if raw.shape[1] == 0:
        return raw
    left, singular, _ = torch.linalg.svd(raw, full_matrices=False)
    scale = float(singular.max().detach().cpu()) if singular.numel() else 0.0
    tolerance = max(raw.shape) * max(scale, 1.0) * 1e-7
    rank = int((singular > tolerance).sum().item())
    return left[:, :rank]


def intersection_normal_basis(
    subspace: torch.Tensor,
    tangent: torch.Tensor,
    *,
    relative_tolerance: float = 1e-6,
) -> torch.Tensor:
    """Return an orthonormal basis for ``S ∩ T^perp``."""

    coupling = tangent.transpose(0, 1) @ subspace
    _, singular, vh = torch.linalg.svd(coupling, full_matrices=True)
    scale = float(singular.max().detach().cpu()) if singular.numel() else 0.0
    tolerance = max(coupling.shape) * max(scale, 1.0) * relative_tolerance
    rank = int((singular > tolerance).sum().item())
    coefficients = vh.transpose(0, 1)[:, rank:]
    return orthonormal_columns(subspace @ coefficients)


def outside_normal_basis(
    subspace: torch.Tensor,
    tangent: torch.Tensor,
    *,
    count: int,
    seed: int,
) -> torch.Tensor:
    """Sample directions in ``(S + T)^perp`` deterministically."""

    combined = orthonormal_columns(torch.cat((subspace, tangent), dim=1))
    available = subspace.shape[0] - combined.shape[1]
    if available < count:
        raise ValueError(
            f"only {available} outside directions are available, requested {count}"
        )
    generator = torch.Generator(device=subspace.device)
    generator.manual_seed(int(seed))
    random = torch.randn(
        subspace.shape[0],
        max(int(count) + 4, int(count)),
        generator=generator,
        device=subspace.device,
        dtype=subspace.dtype,
    )
    projected = random - combined @ (combined.transpose(0, 1) @ random)
    return orthonormal_columns(projected)[:, :count]


def _rollout_single(model, primary: torch.Tensor, horizon: int) -> torch.Tensor:
    state = model.reported_from_primary(primary.unsqueeze(0))
    blank = torch.zeros(
        1, model.input_dim, device=primary.device, dtype=primary.dtype
    )
    for _ in range(int(horizon)):
        state = model.step(blank, state)
    return model.primary_from_reported(state).squeeze(0)


def empirical_slow_subspace(
    model,
    states: torch.Tensor,
    *,
    horizon: int,
    minimum_gain: float,
    chunk_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top eigenspace of mean finite-time Jacobian response energy."""

    if not 0.0 < float(minimum_gain):
        raise ValueError("minimum finite-response gain must be positive")

    def mapping(one: torch.Tensor) -> torch.Tensor:
        return _rollout_single(model, one, horizon)

    gram = torch.zeros(
        states.shape[1],
        states.shape[1],
        device=states.device,
        dtype=torch.float64,
    )
    for start in range(0, states.shape[0], int(chunk_size)):
        stop = min(states.shape[0], start + int(chunk_size))
        jacobian = vmap(jacrev(mapping))(states[start:stop])
        jacobian64 = jacobian.to(torch.float64)
        gram += torch.einsum("aoi,aoj->ij", jacobian64, jacobian64)
    gram /= float(states.shape[0])
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    gains = torch.sqrt(eigenvalues.clamp_min(0.0))
    rank = int((gains >= float(minimum_gain)).sum().item())
    if rank == 0:
        raise RuntimeError("no empirical slow-response direction passes threshold")
    basis = eigenvectors[:, order[:rank]].to(dtype=states.dtype)
    return orthonormal_columns(basis).detach(), eigenvalues.detach()


def explicit_calru_subspace(
    model, state: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    values = model.dynamic_lambda(model.reported_from_primary(state[:1]))
    if values is None:
        raise ValueError("model has no explicit retention coordinates")
    values = values[0]
    indices = torch.nonzero(values >= float(threshold), as_tuple=False).flatten()
    if indices.numel() == 0:
        raise RuntimeError("no CA-LRU coordinates pass the retention threshold")
    basis = torch.eye(
        state.shape[1], device=state.device, dtype=state.dtype
    )[:, indices]
    return basis, values


def _tangent_containment(
    subspace: torch.Tensor, tangents: torch.Tensor
) -> np.ndarray:
    projected = torch.einsum(
        "hk,ahd->akd", subspace, tangents
    )
    energy = projected.square().sum(dim=(1, 2)) / float(tangents.shape[-1])
    return energy.detach().cpu().numpy()


def _load_atlas(
    target: Target, horizons: list[int]
) -> tuple[dict[int, np.ndarray], np.ndarray, str]:
    with np.load(target.atlas_cache, allow_pickle=False) as archive:
        cached_job = str(archive["job_id"].item())
        checkpoint_sha = str(archive["checkpoint_sha256"].item())
        if cached_job != target.record.job_id:
            raise ValueError(f"atlas job mismatch for {target.record.job_id}")
        if checkpoint_sha != sha256_file(target.record.checkpoint_path):
            raise ValueError(f"atlas checkpoint SHA mismatch for {target.record.job_id}")
        available = set(int(value) for value in archive["horizons"].tolist())
        missing = set(horizons).difference(available)
        if missing:
            raise ValueError(f"cached atlas lacks horizons {sorted(missing)}")
        states = {
            horizon: np.array(archive[f"state_h{horizon}"], copy=True)
            for horizon in horizons
        }
        latent = np.array(archive["latent"], copy=True)
    return states, latent, checkpoint_sha


def _decode_in_chunks(model, state: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
    chunks = []
    with torch.no_grad():
        for start in range(0, state.shape[0], int(chunk_size)):
            chunks.append(decode_primary(model, state[start : start + chunk_size]))
    return torch.cat(chunks, dim=0)


def _pairwise_distances(state: torch.Tensor) -> torch.Tensor:
    return torch.pdist(state.to(dtype=torch.float64), p=2)


def stationarity_metrics(
    model,
    topology: str,
    atlas: Mapping[int, torch.Tensor],
    subspace: torch.Tensor,
    *,
    explicit_subspace: torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    horizons = sorted(atlas)
    initial = atlas[0]
    initial_distance = _pairwise_distances(initial)
    diameter = float(initial_distance.max().cpu())
    initial_output = _decode_in_chunks(model, initial)
    initial_norm = torch.linalg.vector_norm(initial, dim=1)
    initial_carrier = initial @ subspace
    initial_carrier_norm = torch.linalg.vector_norm(initial_carrier, dim=1)
    explicit_initial = (
        initial @ explicit_subspace if explicit_subspace is not None else None
    )
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {
        "full_state_norm": [],
        "carrier_norm": [],
        "raw_displacement": [],
        "pairwise_log_distortion": [],
    }
    if explicit_subspace is not None:
        arrays["explicit_carrier_norm"] = []

    for horizon in horizons:
        current = atlas[horizon]
        displacement = torch.linalg.vector_norm(current - initial, dim=1)
        current_norm = torch.linalg.vector_norm(current, dim=1)
        denominator = initial.square().sum().clamp_min(
            torch.finfo(initial.dtype).eps
        )
        scale = (current * initial).sum() / denominator
        residual_squared = (
            (current - scale * initial).square().sum() / denominator
        )
        current_distance = _pairwise_distances(current)
        valid = initial_distance > torch.finfo(initial_distance.dtype).eps
        log_distortion = torch.log(
            current_distance[valid].clamp_min(torch.finfo(torch.float64).eps)
            / initial_distance[valid]
        )
        median_log_scale = torch.median(log_distortion)
        distortion_centered = log_distortion - median_log_scale

        carrier = current @ subspace
        carrier_norm = torch.linalg.vector_norm(carrier, dim=1)
        carrier_cosine = (
            (initial_carrier * carrier).sum(dim=1)
            / (
                initial_carrier_norm
                * carrier_norm
            ).clamp_min(torch.finfo(current.dtype).eps)
        ).clamp(-1.0, 1.0)
        full_cosine = (
            (initial * current).sum(dim=1)
            / (initial_norm * current_norm).clamp_min(
                torch.finfo(current.dtype).eps
            )
        ).clamp(-1.0, 1.0)
        current_output = _decode_in_chunks(model, current)
        memory_error, _ = normalized_geodesic_errors(
            topology,
            current_output.unsqueeze(0),
            initial_output.unsqueeze(0),
        )
        row: dict[str, Any] = {
            "horizon": horizon,
            "atlas_diameter_h0": diameter,
            "raw_stationarity_median": float(
                torch.median(displacement).cpu()
            )
            / max(diameter, np.finfo(np.float64).eps),
            "full_state_norm_median": float(torch.median(current_norm).cpu()),
            "full_state_norm_ratio_median": float(
                torch.median(
                    current_norm
                    / initial_norm.clamp_min(torch.finfo(current.dtype).eps)
                ).cpu()
            ),
            "full_direction_angle_median_rad": float(
                torch.median(torch.acos(full_cosine)).cpu()
            ),
            "best_global_scale": float(scale.cpu()),
            "global_scaling_residual_squared": float(residual_squared.cpu()),
            "global_scaling_residual_root": float(
                torch.sqrt(residual_squared).cpu()
            ),
            "pairwise_log_scale_median": float(median_log_scale.cpu()),
            "pairwise_log_distortion_std": float(
                torch.std(distortion_centered, correction=0).cpu()
            ),
            "carrier_norm_median": float(torch.median(carrier_norm).cpu()),
            "carrier_norm_ratio_median": float(
                torch.median(
                    carrier_norm
                    / initial_carrier_norm.clamp_min(
                        torch.finfo(current.dtype).eps
                    )
                ).cpu()
            ),
            "carrier_direction_angle_median_rad": float(
                torch.median(torch.acos(carrier_cosine)).cpu()
            ),
            "decoded_same_memory_error_mean": float(memory_error.mean().cpu()),
        }
        if explicit_subspace is not None and explicit_initial is not None:
            explicit = current @ explicit_subspace
            explicit_norm = torch.linalg.vector_norm(explicit, dim=1)
            explicit_initial_norm = torch.linalg.vector_norm(
                explicit_initial, dim=1
            )
            explicit_cosine = (
                (explicit_initial * explicit).sum(dim=1)
                / (explicit_initial_norm * explicit_norm).clamp_min(
                    torch.finfo(current.dtype).eps
                )
            ).clamp(-1.0, 1.0)
            row.update(
                {
                    "explicit_carrier_norm_median": float(
                        torch.median(explicit_norm).cpu()
                    ),
                    "explicit_carrier_norm_ratio_median": float(
                        torch.median(
                            explicit_norm
                            / explicit_initial_norm.clamp_min(
                                torch.finfo(current.dtype).eps
                            )
                        ).cpu()
                    ),
                    "explicit_carrier_direction_angle_median_rad": float(
                        torch.median(torch.acos(explicit_cosine)).cpu()
                    ),
                }
            )
            arrays["explicit_carrier_norm"].append(
                explicit_norm.detach().cpu().numpy()
            )
        rows.append(row)
        arrays["full_state_norm"].append(current_norm.detach().cpu().numpy())
        arrays["carrier_norm"].append(carrier_norm.detach().cpu().numpy())
        arrays["raw_displacement"].append(displacement.detach().cpu().numpy())
        arrays["pairwise_log_distortion"].append(
            log_distortion.detach().cpu().numpy()
        )
    return rows, {
        name: np.stack(values).astype(np.float32)
        for name, values in arrays.items()
    }


def _family_bases(
    subspace: torch.Tensor,
    tangents: torch.Tensor,
    *,
    outside_count: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    inside = []
    outside = []
    for anchor in range(tangents.shape[0]):
        tangent = tangents[anchor]
        inside.append(intersection_normal_basis(subspace, tangent))
        outside.append(
            outside_normal_basis(
                subspace,
                tangent,
                count=outside_count,
                seed=derived_seed(seed, "outside", anchor),
            )
        )
    inside_dimensions = {value.shape[1] for value in inside}
    if len(inside_dimensions) != 1:
        raise RuntimeError("inside-normal dimension varies across anchors")
    return {
        "tangent": tangents,
        "in": torch.stack(inside),
        "out": torch.stack(outside),
    }


def _signed(basis: torch.Tensor) -> torch.Tensor:
    return torch.cat((basis, -basis), dim=-1)


def finite_kick_metrics(
    model,
    topology: str,
    anchors: torch.Tensor,
    tangents: torch.Tensor,
    atlas: Mapping[int, torch.Tensor],
    subspace: torch.Tensor,
    *,
    radii_relative: list[float],
    outside_count: int,
    seed: int,
    family_prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    horizons = sorted(atlas)
    centered = atlas[0] - atlas[0].mean(dim=0, keepdim=True)
    state_scale = float(
        torch.median(torch.linalg.vector_norm(centered, dim=1)).cpu()
    )
    if not np.isfinite(state_scale) or state_scale <= 0:
        raise RuntimeError("atlas state scale is not positive")
    clean = blank_snapshots(model, anchors, horizons)
    clean_outputs = {
        horizon: _decode_in_chunks(model, clean[horizon])
        for horizon in horizons
    }
    bases = _family_bases(
        subspace,
        tangents,
        outside_count=outside_count,
        seed=seed,
    )
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}

    # Roll every family and radius in one batch.  This preserves exactly the
    # same perturbations while avoiding nine separate 4,096-step Python loops.
    cases: list[dict[str, Any]] = []
    all_kicked = []
    offset = 0
    for family, unsigned_basis in bases.items():
        basis = _signed(unsigned_basis)
        anchor_count, _, direction_count = basis.shape
        for radius_relative in radii_relative:
            radius = float(radius_relative) * state_scale
            initial = (
                anchors[:, None, :]
                + radius * basis.transpose(1, 2)
            ).reshape(anchor_count * direction_count, anchors.shape[1])
            cases.append(
                {
                    "family": family,
                    "radius_relative": float(radius_relative),
                    "radius": radius,
                    "anchor_count": anchor_count,
                    "direction_count": direction_count,
                    "slice": slice(offset, offset + initial.shape[0]),
                }
            )
            offset += initial.shape[0]
            all_kicked.append(initial)
    kicked_initial_all = torch.cat(all_kicked, dim=0)
    kicked_all = blank_snapshots(model, kicked_initial_all, horizons)

    for case in cases:
        family = str(case["family"])
        radius_relative = float(case["radius_relative"])
        radius = float(case["radius"])
        anchor_count = int(case["anchor_count"])
        direction_count = int(case["direction_count"])
        selection = case["slice"]
        kicked_initial = kicked_initial_all[selection]
        initial_distance = torch.cdist(
            kicked_initial, atlas[0]
        ).min(dim=1).values.reshape(anchor_count, direction_count)
        initial_distance = initial_distance.clamp_min(
            torch.finfo(kicked_initial.dtype).eps
        )
        recovery = []
        distance_relative = []
        same_memory = []
        perturbation_gain = []
        for horizon in horizons:
            current = kicked_all[horizon][selection]
            distance = torch.cdist(current, atlas[horizon]).min(dim=1).values
            distance = distance.reshape(anchor_count, direction_count)
            distance_relative.append((distance / state_scale).cpu().numpy())
            recovery.append((distance / initial_distance).cpu().numpy())
            current_output = _decode_in_chunks(model, current).reshape(
                anchor_count, direction_count, -1
            )
            reference_output = clean_outputs[horizon][:, None, :].expand_as(
                current_output
            )
            memory_error, _ = normalized_geodesic_errors(
                topology, current_output, reference_output
            )
            same_memory.append(memory_error.cpu().numpy())
            clean_repeated = clean[horizon][:, None, :].expand(
                anchor_count, direction_count, anchors.shape[1]
            )
            current_reshaped = current.reshape(
                anchor_count, direction_count, anchors.shape[1]
            )
            perturbation_gain.append(
                (
                    torch.linalg.vector_norm(
                        current_reshaped - clean_repeated, dim=2
                    )
                    / radius
                ).cpu().numpy()
            )
        recovery_array = np.stack(recovery).astype(np.float32)
        distance_array = np.stack(distance_relative).astype(np.float32)
        memory_array = np.stack(same_memory).astype(np.float32)
        gain_array = np.stack(perturbation_gain).astype(np.float32)
        key = (
            f"{family_prefix}_{family}_r"
            f"{str(radius_relative).replace('.', 'p')}"
        )
        arrays[f"{key}_recovery_ratio"] = recovery_array
        arrays[f"{key}_manifold_distance_relative"] = distance_array
        arrays[f"{key}_same_memory_error"] = memory_array
        arrays[f"{key}_perturbation_gain"] = gain_array
        for horizon_index, horizon in enumerate(horizons):
            rows.append(
                {
                    "subspace": family_prefix,
                    "direction_family": family,
                    "radius_relative": float(radius_relative),
                    "horizon": horizon,
                    "directions_per_anchor_signed": direction_count,
                    "state_scale": state_scale,
                    "manifold_distance_relative_median": float(
                        np.median(distance_array[horizon_index])
                    ),
                    "recovery_ratio_median": float(
                        np.median(recovery_array[horizon_index])
                    ),
                    "same_memory_error_mean": float(
                        np.mean(memory_array[horizon_index])
                    ),
                    "perturbation_gain_median": float(
                        np.median(gain_array[horizon_index])
                    ),
                }
            )
    return rows, arrays


def analyze_target(
    target: Target,
    config: Mapping[str, Any],
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    record = target.record
    horizons = [int(value) for value in config["finite_kicks"]["horizons"]]
    atlas_np, latent, checkpoint_sha = _load_atlas(target, horizons)
    model, _ = load_model(record, device)
    atlas = {
        horizon: torch.as_tensor(value, device=device)
        for horizon, value in atlas_np.items()
    }
    geometry_path = (
        target.analysis_root / "geometry" / "runs" / f"{record.job_id}.npz"
    )
    with np.load(geometry_path, allow_pickle=False) as archive:
        geometry_state = np.array(archive["endpoint_state"], copy=True)
        geometry_tangent = np.array(archive["tangent_basis_raw"], copy=True)
        geometry_indices = np.array(archive["anchor_index"], copy=True)

    kick_count = int(config["finite_kicks"]["anchors"])
    kick_selection = np.linspace(
        0, len(geometry_state) - 1, kick_count, dtype=np.int64
    )
    anchors = torch.as_tensor(geometry_state[kick_selection], device=device)
    tangents = torch.linalg.qr(
        torch.as_tensor(geometry_tangent[kick_selection], device=device),
        mode="reduced",
    ).Q

    slow = config["empirical_slow_subspace"]
    jacobian_count = int(slow["jacobian_anchors"])
    jacobian_selection = np.linspace(
        0, len(geometry_state) - 1, jacobian_count, dtype=np.int64
    )
    jacobian_state = torch.as_tensor(
        geometry_state[jacobian_selection], device=device
    )
    dynamic_subspace, eigenvalues = empirical_slow_subspace(
        model,
        jacobian_state,
        horizon=int(slow["jacobian_horizon"]),
        minimum_gain=float(slow["minimum_finite_response_gain"]),
    )
    slow_rank = int(dynamic_subspace.shape[1])
    tangent_containment = _tangent_containment(dynamic_subspace, tangents)

    explicit_subspace = None
    lambda_values = None
    explicit_containment = None
    if record.model_id == "calru":
        explicit_subspace, lambda_values = explicit_calru_subspace(
            model,
            anchors,
            float(config["explicit_calru_subspace"]["lambda_threshold"]),
        )
        explicit_containment = _tangent_containment(
            explicit_subspace, tangents
        )

    stationarity_rows, stationarity_arrays = stationarity_metrics(
        model,
        record.topology,
        atlas,
        dynamic_subspace,
        explicit_subspace=explicit_subspace,
    )
    radii = [
        float(value)
        for value in config["finite_kicks"]["radii_relative_to_atlas_scale"]
    ]
    common_rows, common_arrays = finite_kick_metrics(
        model,
        record.topology,
        anchors,
        tangents,
        atlas,
        dynamic_subspace,
        radii_relative=radii,
        outside_count=int(config["finite_kicks"]["outside_directions"]),
        seed=derived_seed(record.seed, record.job_id, "dynamic_subspace_kicks"),
        family_prefix="dynamic",
    )
    explicit_rows: list[dict[str, Any]] = []
    explicit_arrays: dict[str, np.ndarray] = {}
    if explicit_subspace is not None:
        explicit_rows, explicit_arrays = finite_kick_metrics(
            model,
            record.topology,
            anchors,
            tangents,
            atlas,
            explicit_subspace,
            radii_relative=radii,
            outside_count=int(config["finite_kicks"]["outside_directions"]),
            seed=derived_seed(record.seed, record.job_id, "lambda_subspace_kicks"),
            family_prefix="lambda",
        )

    stationarity_enriched = [
        {
            "job_id": record.job_id,
            "model": record.model_id,
            "topology": record.topology,
            "seed": record.seed,
            "task_success": target.task_success,
            "failed_task_fallback": target.failed_task_fallback,
            **row,
        }
        for row in stationarity_rows
    ]
    kick_enriched = [
        {
            "job_id": record.job_id,
            "model": record.model_id,
            "topology": record.topology,
            "seed": record.seed,
            "task_success": target.task_success,
            "failed_task_fallback": target.failed_task_fallback,
            **row,
        }
        for row in [*common_rows, *explicit_rows]
    ]
    result = {
        "schema_version": 1,
        "job_id": record.job_id,
        "model": record.model_id,
        "topology": record.topology,
        "seed": record.seed,
        "task_success": target.task_success,
        "failed_task_fallback": target.failed_task_fallback,
        "validation_error": target.validation_error,
        "checkpoint_sha256": checkpoint_sha,
        "atlas_cache": str(target.atlas_cache),
        "atlas_points": int(atlas[0].shape[0]),
        "atlas_latent_shape": list(latent.shape),
        "anchors": kick_count,
        "empirical_slow_subspace": {
            "rank": slow_rank,
            "jacobian_horizon": int(slow["jacobian_horizon"]),
            "jacobian_anchors": jacobian_count,
            "minimum_finite_response_gain": float(
                slow["minimum_finite_response_gain"]
            ),
            "response_eigenvalues": eigenvalues.detach().cpu().tolist(),
            "tangent_containment_mean": float(np.mean(tangent_containment)),
            "tangent_containment_min": float(np.min(tangent_containment)),
        },
        "explicit_calru_subspace": (
            {
                "lambda_threshold": float(
                    config["explicit_calru_subspace"]["lambda_threshold"]
                ),
                "rank": int(explicit_subspace.shape[1]),
                "lambda_values": lambda_values.detach().cpu().tolist(),
                "tangent_containment_mean": float(
                    np.mean(explicit_containment)
                ),
                "tangent_containment_min": float(
                    np.min(explicit_containment)
                ),
            }
            if explicit_subspace is not None
            and lambda_values is not None
            and explicit_containment is not None
            else None
        ),
        "stationarity": stationarity_enriched,
        "finite_kicks": kick_enriched,
    }
    run_output = output / "runs" / record.job_id
    run_output.mkdir(parents=True, exist_ok=True)
    atomic_json(run_output / "result.json", result)
    atomic_npz(
        run_output / "arrays.npz",
        horizons=np.asarray(horizons, dtype=np.int64),
        atlas_latent=latent.astype(np.float32),
        geometry_anchor_index=geometry_indices[kick_selection],
        dynamic_subspace=dynamic_subspace.detach().cpu().numpy().astype(np.float32),
        dynamic_tangent_containment=tangent_containment.astype(np.float32),
        explicit_subspace=(
            explicit_subspace.detach().cpu().numpy().astype(np.float32)
            if explicit_subspace is not None
            else np.empty((anchors.shape[1], 0), dtype=np.float32)
        ),
        explicit_tangent_containment=(
            explicit_containment.astype(np.float32)
            if explicit_containment is not None
            else np.empty((0,), dtype=np.float32)
        ),
        **stationarity_arrays,
        **common_arrays,
        **explicit_arrays,
    )
    return result


def _line_value(
    rows: list[dict[str, Any]],
    *,
    horizon: int,
    field: str,
    subspace: str | None = None,
    family: str | None = None,
    radius: float | None = None,
) -> float:
    candidates = [
        row
        for row in rows
        if int(row["horizon"]) == int(horizon)
        and (subspace is None or row.get("subspace") == subspace)
        and (family is None or row.get("direction_family") == family)
        and (
            radius is None
            or math.isclose(float(row.get("radius_relative", -1)), radius)
        )
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one row for H={horizon}, {subspace}/{family}/{radius}"
        )
    return float(candidates[0][field])


def _save_figure(figure: plt.Figure, root: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = root / f"{stem}.{suffix}"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path))
    plt.close(figure)
    return paths


def plot_normal_recovery(
    results: list[dict[str, Any]], config: Mapping[str, Any], figures: Path
) -> list[str]:
    horizons = [int(value) for value in config["finite_kicks"]["horizons"]]
    radius = float(config["finite_kicks"]["primary_radius_relative"])
    lookup = {(row["model"], row["topology"]): row for row in results}
    figure, axes = plt.subplots(
        len(MODEL_ORDER),
        len(TOPOLOGY_ORDER),
        figsize=(13.5, 11.0),
        sharex=True,
        sharey=True,
    )
    for row_index, model in enumerate(MODEL_ORDER):
        for column, topology in enumerate(TOPOLOGY_ORDER):
            axis = axes[row_index, column]
            result = lookup[(model, topology)]
            axis.set_yscale("log")
            for family, label, color in (
                ("in", r"$N_{\rm in}$", "#B13B47"),
                ("out", r"$N_{\rm out}$", "#3377B5"),
            ):
                values = [
                    max(
                        _line_value(
                            result["finite_kicks"],
                            horizon=horizon,
                            field="recovery_ratio_median",
                            subspace="dynamic",
                            family=family,
                            radius=radius,
                        ),
                        1e-7,
                    )
                    for horizon in horizons
                ]
                axis.plot(
                    horizons,
                    values,
                    marker="o",
                    linewidth=2,
                    label=label,
                    color=color,
                )
            axis.axhline(1.0, color="#999999", linestyle=":", linewidth=1)
            axis.set_xscale("symlog", linthresh=128)
            axis.set_ylim(5e-8, 2.0)
            axis.grid(alpha=0.22)
            if row_index == 0:
                axis.set_title(TOPOLOGY_LABEL[topology])
            if column == 0:
                suffix = " · failed-task fallback" if result["failed_task_fallback"] else ""
                axis.set_ylabel(
                    f"{MODEL_LABEL[model]}{suffix}\n"
                    r"$d(h_H^{kick},M_H)/d(h_0^{kick},M_0)$"
                )
            if row_index == len(MODEL_ORDER) - 1:
                axis.set_xlabel("blank horizon H")
    axes[0, 0].legend(frameon=False, loc="best")
    figure.suptitle(
        "Manifold-normal recovery inside vs outside the empirical slow-response subspace\n"
        "5% equal-norm kicks · reference is the time-evolved clean manifold"
    )
    return _save_figure(figure, figures, "fig_A_subspace_normal_recovery")


def plot_same_memory(
    results: list[dict[str, Any]], config: Mapping[str, Any], figures: Path
) -> list[str]:
    horizons = [int(value) for value in config["finite_kicks"]["horizons"]]
    radius = float(config["finite_kicks"]["primary_radius_relative"])
    lookup = {(row["model"], row["topology"]): row for row in results}
    figure, axes = plt.subplots(
        len(MODEL_ORDER),
        len(TOPOLOGY_ORDER),
        figsize=(13.5, 11.0),
        sharex=True,
        sharey=True,
    )
    for row_index, model in enumerate(MODEL_ORDER):
        for column, topology in enumerate(TOPOLOGY_ORDER):
            axis = axes[row_index, column]
            result = lookup[(model, topology)]
            for family, label, color, style in (
                ("tangent", "tangent", "#5B3C88", "-"),
                ("in", r"$N_{\rm in}$", "#B13B47", "-"),
                ("out", r"$N_{\rm out}$", "#3377B5", "--"),
            ):
                values = [
                    _line_value(
                        result["finite_kicks"],
                        horizon=horizon,
                        field="same_memory_error_mean",
                        subspace="dynamic",
                        family=family,
                        radius=radius,
                    )
                    for horizon in horizons
                ]
                axis.plot(
                    horizons,
                    values,
                    marker="o",
                    linewidth=1.8,
                    linestyle=style,
                    label=label,
                    color=color,
                )
            axis.set_xscale("symlog", linthresh=128)
            axis.set_yscale("symlog", linthresh=1e-4)
            axis.grid(alpha=0.22)
            if row_index == 0:
                axis.set_title(TOPOLOGY_LABEL[topology])
            if column == 0:
                suffix = " · fallback" if result["failed_task_fallback"] else ""
                axis.set_ylabel(
                    f"{MODEL_LABEL[model]}{suffix}\nsame-memory error"
                )
            if row_index == len(MODEL_ORDER) - 1:
                axis.set_xlabel("blank horizon H")
    axes[0, 0].legend(frameon=False, loc="best")
    figure.suptitle(
        "Directional selectivity after equal-norm kicks\n"
        "A tangent kick may remain as a memory shift; normal kicks should preserve the original memory"
    )
    return _save_figure(figure, figures, "fig_B_directional_same_memory")


def plot_state_character(
    results: list[dict[str, Any]], config: Mapping[str, Any], figures: Path
) -> list[str]:
    horizons = [int(value) for value in config["state_geometry"]["horizons"]]
    lookup = {(row["model"], row["topology"]): row for row in results}
    panels = (
        ("raw_stationarity_median", "raw stationarity / diam.", "log"),
        ("best_global_scale", r"best global scale $a_H^*$", "linear"),
        ("global_scaling_residual_root", "global-scale residual", "log"),
        ("pairwise_log_distortion_std", "geometry distortion std.", "log"),
        (
            "carrier_direction_angle_median_rad",
            "normalized slow-carrier direction drift (rad)",
            "log",
        ),
        ("decoded_same_memory_error_mean", "decoded memory error", "log"),
    )
    figure, axes = plt.subplots(
        len(TOPOLOGY_ORDER),
        len(panels),
        figsize=(25.0, 9.0),
        sharex=True,
        constrained_layout=True,
    )
    for row_index, topology in enumerate(TOPOLOGY_ORDER):
        for column, (field, label, scale) in enumerate(panels):
            axis = axes[row_index, column]
            for model in MODEL_ORDER:
                result = lookup[(model, topology)]
                values = [
                    max(
                        _line_value(
                            result["stationarity"],
                            horizon=horizon,
                            field=field,
                        ),
                        1e-9,
                    )
                    if scale == "log"
                    else _line_value(
                        result["stationarity"],
                        horizon=horizon,
                        field=field,
                    )
                    for horizon in horizons
                ]
                label_model = MODEL_LABEL[model] + (
                    "*" if result["failed_task_fallback"] else ""
                )
                axis.plot(
                    horizons,
                    values,
                    marker="o",
                    linewidth=1.8,
                    color=MODEL_COLOR[model],
                    label=label_model,
                )
            axis.set_xscale("symlog", linthresh=128)
            if scale == "log":
                axis.set_yscale("log")
            axis.grid(alpha=0.22)
            if row_index == 0:
                axis.set_title(label, fontsize=10)
            if column == 0:
                axis.set_ylabel(TOPOLOGY_LABEL[topology])
            if row_index == len(TOPOLOGY_ORDER) - 1:
                axis.set_xlabel("blank horizon H")
    axes[0, -1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Is the clean hidden manifold stationary, globally shrinking, or anisotropically deforming?\n"
        "* denotes a model-topology pair with no task-success seed"
    )
    return _save_figure(figure, figures, "fig_C_hidden_manifold_character")


def plot_carrier_state_memory(
    results: list[dict[str, Any]], config: Mapping[str, Any], figures: Path
) -> list[str]:
    """Plot the four-way distinction requested by the analysis contract."""

    horizons = [int(value) for value in config["state_geometry"]["horizons"]]
    lookup = {(row["model"], row["topology"]): row for row in results}
    panels = (
        ("full_state_norm_ratio_median", "full recurrent-state norm / H=0", "linear"),
        ("carrier_norm_ratio_median", "slow-carrier norm / H=0", "linear"),
        (
            "carrier_direction_angle_median_rad",
            "normalized carrier direction drift (rad)",
            "log",
        ),
        ("decoded_same_memory_error_mean", "decoded memory error", "log"),
    )
    figure, axes = plt.subplots(
        len(TOPOLOGY_ORDER),
        len(panels),
        figsize=(15.5, 8.5),
        sharex=True,
        constrained_layout=True,
    )
    for row_index, topology in enumerate(TOPOLOGY_ORDER):
        for column, (field, label, scale) in enumerate(panels):
            axis = axes[row_index, column]
            for model in MODEL_ORDER:
                result = lookup[(model, topology)]
                values = []
                for horizon in horizons:
                    value = _line_value(
                        result["stationarity"],
                        horizon=horizon,
                        field=field,
                    )
                    values.append(max(value, 1e-9) if scale == "log" else value)
                label_model = MODEL_LABEL[model] + (
                    "*" if result["failed_task_fallback"] else ""
                )
                axis.plot(
                    horizons,
                    values,
                    marker="o",
                    linewidth=1.8,
                    color=MODEL_COLOR[model],
                    label=label_model,
                )
            axis.set_xscale("symlog", linthresh=128)
            if scale == "log":
                axis.set_yscale("log")
            axis.grid(alpha=0.22)
            if row_index == 0:
                axis.set_title(label, fontsize=10)
            if column == 0:
                axis.set_ylabel(TOPOLOGY_LABEL[topology])
            if row_index == len(TOPOLOGY_ORDER) - 1:
                axis.set_xlabel("blank horizon H")
    axes[0, -1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Carrier norm, normalized direction, full recurrent state, and decoded memory\n"
        "* denotes a failed-task fallback"
    )
    return _save_figure(figure, figures, "fig_D_carrier_state_memory")


def plot_calru_explicit(
    results: list[dict[str, Any]], config: Mapping[str, Any], figures: Path
) -> list[str]:
    horizons = [int(value) for value in config["finite_kicks"]["horizons"]]
    radius = float(config["finite_kicks"]["primary_radius_relative"])
    rows = {row["topology"]: row for row in results if row["model"] == "calru"}
    figure, axes = plt.subplots(2, len(TOPOLOGY_ORDER), figsize=(13.5, 7.0), sharex=True)
    for column, topology in enumerate(TOPOLOGY_ORDER):
        result = rows[topology]
        for row_index, family in enumerate(("in", "out")):
            axis = axes[row_index, column]
            if family == "in":
                axis.set_ylim(0.95, 1.05)
            else:
                axis.set_yscale("log")
                axis.set_ylim(5e-8, 2.0)
            for subspace, label, color, style in (
                ("dynamic", r"empirical $S_{\rm dyn}$", "#444444", "--"),
                ("lambda", r"explicit $S_\lambda$", "#B13B47", "-"),
            ):
                values = [
                    max(
                        _line_value(
                            result["finite_kicks"],
                            horizon=horizon,
                            field="recovery_ratio_median",
                            subspace=subspace,
                            family=family,
                            radius=radius,
                        ),
                        1e-7,
                    )
                    for horizon in horizons
                ]
                axis.plot(
                    horizons,
                    values,
                    marker="o",
                    linewidth=2,
                    linestyle=style,
                    color=color,
                    label=label,
                )
            axis.axhline(1.0, color="#999999", linestyle=":", linewidth=1)
            axis.set_xscale("symlog", linthresh=128)
            axis.grid(alpha=0.22)
            if row_index == 0:
                rank = result["explicit_calru_subspace"]["rank"]
                axis.set_title(f"{TOPOLOGY_LABEL[topology]} · rank($S_\\lambda$)={rank}")
            if column == 0:
                axis.set_ylabel(
                    (r"$N_{\rm in}$" if family == "in" else r"$N_{\rm out}$")
                    + "\nrecovery ratio"
                )
            if row_index == 1:
                axis.set_xlabel("blank horizon H")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        r"CA-LRU: empirical versus explicit high-retention subspace ($\lambda_j\geq0.99$)"
    )
    return _save_figure(figure, figures, "fig_E_calru_explicit_retention_split")


def aggregate(
    results: list[dict[str, Any]],
    config: Mapping[str, Any],
    output: Path,
    *,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    stationarity_rows = [
        row for result in results for row in result["stationarity"]
    ]
    kick_rows = [row for result in results for row in result["finite_kicks"]]

    def uniform(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fields: list[str] = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
        return [{field: row.get(field, "") for field in fields} for row in rows]

    write_csv(output / "stationarity_metrics.csv", uniform(stationarity_rows))
    write_csv(output / "subspace_kick_metrics.csv", uniform(kick_rows))
    summary_rows = []
    primary_radius = float(config["finite_kicks"]["primary_radius_relative"])
    terminal_horizon = max(int(value) for value in config["finite_kicks"]["horizons"])
    for result in results:
        stationarity = result["stationarity"]
        kicks = result["finite_kicks"]
        summary_rows.append(
            {
                "job_id": result["job_id"],
                "model": result["model"],
                "topology": result["topology"],
                "seed": result["seed"],
                "task_success": result["task_success"],
                "failed_task_fallback": result["failed_task_fallback"],
                "validation_error": result["validation_error"],
                "dynamic_slow_rank": result["empirical_slow_subspace"]["rank"],
                "dynamic_tangent_containment_mean": result[
                    "empirical_slow_subspace"
                ]["tangent_containment_mean"],
                "explicit_lambda_rank": (
                    result["explicit_calru_subspace"]["rank"]
                    if result["explicit_calru_subspace"] is not None
                    else ""
                ),
                "raw_stationarity_h4096": _line_value(
                    stationarity,
                    horizon=terminal_horizon,
                    field="raw_stationarity_median",
                ),
                "best_global_scale_h4096": _line_value(
                    stationarity,
                    horizon=terminal_horizon,
                    field="best_global_scale",
                ),
                "global_scaling_residual_h4096": _line_value(
                    stationarity,
                    horizon=terminal_horizon,
                    field="global_scaling_residual_root",
                ),
                "geometry_distortion_h4096": _line_value(
                    stationarity,
                    horizon=terminal_horizon,
                    field="pairwise_log_distortion_std",
                ),
                "decoded_memory_error_h4096": _line_value(
                    stationarity,
                    horizon=terminal_horizon,
                    field="decoded_same_memory_error_mean",
                ),
                "dynamic_in_recovery_h4096": _line_value(
                    kicks,
                    horizon=terminal_horizon,
                    field="recovery_ratio_median",
                    subspace="dynamic",
                    family="in",
                    radius=primary_radius,
                ),
                "dynamic_out_recovery_h4096": _line_value(
                    kicks,
                    horizon=terminal_horizon,
                    field="recovery_ratio_median",
                    subspace="dynamic",
                    family="out",
                    radius=primary_radius,
                ),
                "dynamic_tangent_same_memory_h4096": _line_value(
                    kicks,
                    horizon=terminal_horizon,
                    field="same_memory_error_mean",
                    subspace="dynamic",
                    family="tangent",
                    radius=primary_radius,
                ),
                "dynamic_in_same_memory_h4096": _line_value(
                    kicks,
                    horizon=terminal_horizon,
                    field="same_memory_error_mean",
                    subspace="dynamic",
                    family="in",
                    radius=primary_radius,
                ),
                "dynamic_out_same_memory_h4096": _line_value(
                    kicks,
                    horizon=terminal_horizon,
                    field="same_memory_error_mean",
                    subspace="dynamic",
                    family="out",
                    radius=primary_radius,
                ),
                "lambda_in_recovery_h4096": (
                    _line_value(
                        kicks,
                        horizon=terminal_horizon,
                        field="recovery_ratio_median",
                        subspace="lambda",
                        family="in",
                        radius=primary_radius,
                    )
                    if result["model"] == "calru"
                    else ""
                ),
                "lambda_out_recovery_h4096": (
                    _line_value(
                        kicks,
                        horizon=terminal_horizon,
                        field="recovery_ratio_median",
                        subspace="lambda",
                        family="out",
                        radius=primary_radius,
                    )
                    if result["model"] == "calru"
                    else ""
                ),
            }
        )
    write_csv(output / "representative_summary.csv", summary_rows)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure_paths = [
        *plot_normal_recovery(results, config, figures),
        *plot_same_memory(results, config, figures),
        *plot_state_character(results, config, figures),
        *plot_carrier_state_memory(results, config, figures),
        *plot_calru_explicit(results, config, figures),
    ]
    manifest = {
        "schema_version": 1,
        "analysis_id": config["analysis_id"],
        "training_performed": False,
        "test_bank_accessed": False,
        "models": list(MODEL_ORDER),
        "topologies": list(TOPOLOGY_ORDER),
        "representative_seed_policy": config["representative_seed_policy"],
        "code": _git_provenance(),
        "targets": [
            {
                "job_id": result["job_id"],
                "model": result["model"],
                "topology": result["topology"],
                "seed": result["seed"],
                "task_success": result["task_success"],
                "failed_task_fallback": result["failed_task_fallback"],
                "checkpoint_sha256": result["checkpoint_sha256"],
            }
            for result in results
        ],
        "config": config,
        "source_manifest": source_manifest,
        "figures": figure_paths,
    }
    atomic_json(output / "analysis_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--baseline-analysis-root", type=Path, required=True)
    parser.add_argument("--calru-run-root", type=Path, required=True)
    parser.add_argument("--calru-analysis-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name(
            "topology_subspace_attraction_v1.json"
        ),
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--job-id",
        help="Analyze one representative job. Aggregation requires all 12 jobs.",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline_run_root = args.baseline_run_root.expanduser().resolve(strict=True)
    baseline_analysis_root = args.baseline_analysis_root.expanduser().resolve(
        strict=True
    )
    calru_run_root = args.calru_run_root.expanduser().resolve(strict=True)
    calru_analysis_root = args.calru_analysis_root.expanduser().resolve(
        strict=True
    )
    comparison_root = args.comparison_root.expanduser().resolve(strict=True)
    targets = discover_representatives(
        baseline_run_root=baseline_run_root,
        baseline_analysis_root=baseline_analysis_root,
        calru_run_root=calru_run_root,
        calru_analysis_root=calru_analysis_root,
        comparison_root=comparison_root,
    )
    source_manifest = {
        "baseline_run_root": str(baseline_run_root),
        "baseline_analysis_root": str(baseline_analysis_root),
        "calru_run_root": str(calru_run_root),
        "calru_analysis_root": str(calru_analysis_root),
        "comparison_root": str(comparison_root),
    }
    if args.aggregate_only:
        results = [
            strict_json_load(
                output / "runs" / target.record.job_id / "result.json"
            )
            for target in targets
        ]
        aggregate(
            results,
            config,
            output,
            source_manifest=source_manifest,
        )
        return
    selected = (
        [target for target in targets if target.record.job_id == args.job_id]
        if args.job_id
        else targets
    )
    if not selected:
        raise ValueError("--job-id is not a representative target")
    for target in selected:
        analyze_target(target, config, output, torch.device(args.device))
    if args.job_id is None:
        results = [
            strict_json_load(
                output / "runs" / target.record.job_id / "result.json"
            )
            for target in targets
        ]
        aggregate(
            results,
            config,
            output,
            source_manifest=source_manifest,
        )


if __name__ == "__main__":
    main()
