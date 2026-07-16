"""Post-hoc attractor analysis for state-dependent-retention checkpoints.

This analysis is intentionally checkpoint-only: training and evaluation remain
separate.  It reuses the registered Ságodi slow-manifold reconstruction,
full local Jacobian, projected-flow, and finite carrier-normal recovery code.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from repro.sagodi_protocol.artifacts import atomic_json
from repro.sagodi_protocol.sagodi_primary_analysis import (
    circular_absolute_error,
    circular_difference,
    carrier_ambient_normal_recovery,
    cyclic_flow_reversal_topology,
    dense_full_jacobian_eigenspectrum,
    output_projected_flow,
    signed_angular_flow,
)
from repro.sagodi_protocol.sagodi_primary_runner import (
    NORMAL_RECOVERY_AMBIENT_DIRECTIONS,
    NORMAL_RECOVERY_ANCHOR_COUNT,
    NORMAL_RECOVERY_HORIZONS,
    NORMAL_RECOVERY_RADII_OVER_MANIFOLD_SCALE,
    PrimaryAnalysisSpec,
    _blank_decode_primary,
    reconstruct_slow_manifold,
)
from repro.sagodi_protocol.state import StateAdapter
from repro.sagodi_protocol.tasks import Batch, load_fixed_bank

from .models import build_state_dependent_model


def _native(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _summary(value: np.ndarray | torch.Tensor) -> dict[str, float | int]:
    array = (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    if not flat.size or not np.isfinite(flat).all():
        raise ValueError("summary input must be finite and nonempty")
    return {
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "q05": float(np.quantile(flat, 0.05)),
        "q95": float(np.quantile(flat, 0.95)),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
    }


def _np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _topology_payload(topology: Any) -> dict[str, Any]:
    return {
        "kind": topology.kind,
        "orientation": topology.orientation,
        "stable_count": sum(item.kind == "stable" for item in topology.reversals),
        "saddle_count": sum(item.kind == "saddle" for item in topology.reversals),
        "reversals": [
            {
                "angle": float(item.angle),
                "kind": item.kind,
                "left_flow": float(item.left_flow),
                "right_flow": float(item.right_flow),
            }
            for item in topology.reversals
        ],
    }


def _recovery_payload(recovery: Any) -> dict[str, Any]:
    family = np.asarray(recovery.family, dtype="U32")
    radius = _np(recovery.radius_over_manifold_scale).astype(np.float64)
    horizons = _np(recovery.horizon).astype(np.int64)
    paired = _np(recovery.distance_to_matched_clean_state_ratio)
    fixed = _np(recovery.manifold_distance_ratio)
    memory = _np(recovery.same_memory_error_radians)
    result: dict[str, Any] = {}
    for family_name in ("ambient_normal", "in_plane_radial"):
        by_radius: dict[str, Any] = {}
        for radius_value in sorted(set(float(value) for value in radius)):
            mask = (family == family_name) & np.isclose(
                radius, radius_value, rtol=0.0, atol=1.0e-12
            )
            by_horizon: dict[str, Any] = {}
            for column, horizon in enumerate(horizons):
                by_horizon[str(int(horizon))] = {
                    "matched_clean_state_ratio": _summary(paired[mask, column]),
                    "fixed_spline_distance_ratio": _summary(fixed[mask, column]),
                    "same_memory_error_radians": _summary(memory[mask, column]),
                }
            by_radius[format(float(radius_value), ".12g")] = by_horizon
        result[family_name] = by_radius
    return result


def _tangent_radial_gains(
    manifold: torch.Tensor, jacobian: torch.Tensor, anchor_index: torch.Tensor
) -> dict[str, Any]:
    central = torch.roll(manifold, -1, 0) - torch.roll(manifold, 1, 0)
    tangent = central / torch.linalg.vector_norm(central, dim=1, keepdim=True)
    centered = manifold - manifold.mean(dim=0, keepdim=True)
    _, _, right = torch.linalg.svd(centered, full_matrices=False)
    plane = right[:2]
    tangent_anchor = tangent[anchor_index]
    tangent_plane = tangent_anchor @ plane.transpose(0, 1)
    radial_coordinates = torch.stack(
        (-tangent_plane[:, 1], tangent_plane[:, 0]), dim=1
    )
    radial = radial_coordinates @ plane
    radial = radial / torch.linalg.vector_norm(radial, dim=1, keepdim=True)
    tangent_gain = torch.linalg.vector_norm(
        torch.einsum("nij,nj->ni", jacobian, tangent_anchor), dim=1
    )
    radial_gain = torch.linalg.vector_norm(
        torch.einsum("nij,nj->ni", jacobian, radial), dim=1
    )
    return {
        "tangent_one_step_gain": _summary(tangent_gain),
        "in_plane_radial_one_step_gain": _summary(radial_gain),
        "tangent_minus_radial_gain": _summary(tangent_gain - radial_gain),
    }


@torch.no_grad()
def _finite_time_memory(
    manifold_state: torch.Tensor,
    target_angle: torch.Tensor,
    autonomous_map: Any,
    decoder: Any,
    *,
    horizon: int,
) -> dict[str, torch.Tensor]:
    """Roll the reconstructed carrier and retain the full angular-error curve."""

    state = manifold_state.clone()
    predicted = torch.empty(
        state.shape[0], horizon + 1, device=state.device, dtype=state.dtype
    )
    for step in range(horizon + 1):
        output = decoder(state)
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError(f"non-finite decoded blank rollout at step {step}")
        predicted[:, step] = torch.atan2(output[:, 1], output[:, 0])
        if step < horizon:
            state = autonomous_map(state)
            if not bool(torch.isfinite(state).all()):
                raise FloatingPointError(f"non-finite blank rollout at step {step + 1}")
    signed = circular_difference(predicted, target_angle[:, None])
    absolute = signed.abs()
    return {
        "time": torch.arange(horizon + 1, device=state.device),
        "target_angle": target_angle,
        "predicted_angle": predicted,
        "signed_error": signed,
        "absolute_error": absolute,
        "instantaneous_minimum_error": absolute.min(dim=0).values,
        "instantaneous_mean_error": absolute.mean(dim=0),
        "instantaneous_maximum_error": absolute.max(dim=0).values,
        "terminal_state": state,
    }


def _assigned_stable_angles(
    observed: torch.Tensor, stable_angles: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if not stable_angles.numel():
        return observed.clone(), torch.zeros(0, dtype=torch.int64), float("nan")
    distance = circular_absolute_error(
        observed[:, None], stable_angles.to(observed)[None, :]
    )
    assignment = distance.argmin(dim=1)
    assigned = stable_angles.to(observed)[assignment]
    counts = torch.bincount(assignment, minlength=stable_angles.numel())
    proportions = counts.to(torch.float64) / float(observed.numel())
    positive = proportions[proportions > 0]
    effective = float(torch.exp(-(positive * torch.log(positive)).sum()).cpu())
    return assigned, counts, effective


def analyze_seed(
    *,
    root: Path,
    bank: Batch,
    output: Path,
    seed: int,
    retention_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    run = root / "runs" / f"{retention_mode}__recurrent__seed{seed:02d}"
    checkpoint_path = run / "checkpoint_trained.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_state_dependent_model(
        "recurrent", model_seed=seed, retention_mode=retention_mode
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    batch = Batch(
        bank.inputs.to(device),
        bank.output_targets.to(device),
        bank.latent_targets.to(device),
        bank.mask.to(device),
        bank.metadata,
    )
    if batch.batch_size != 1024 or batch.time_steps != 128:
        raise ValueError("source-v6 attractor analysis requires the 1024xT128 bank")
    with torch.no_grad():
        prediction, states = model.forward_sequence(
            batch.inputs,
            initial_memory=batch.output_targets[0],
            return_states=True,
        )
    endpoint = states[-1]
    adapter = StateAdapter(model.core)
    spec = PrimaryAnalysisSpec(
        task_horizon=128,
        blank_horizon=2048,
        source_v6=True,
    )
    spline_angle = (
        torch.arange(spec.spline_count, device=device, dtype=endpoint.dtype)
        * (2.0 * math.pi / float(spec.spline_count))
    )
    reconstruction = reconstruct_slow_manifold(
        model, adapter, endpoint, spline_angle, spec
    )
    decoder = lambda state: _blank_decode_primary(model, adapter, state)
    anchor_index = torch.div(
        torch.arange(NORMAL_RECOVERY_ANCHOR_COUNT, device=device)
        * int(spec.spline_count),
        NORMAL_RECOVERY_ANCHOR_COUNT,
        rounding_mode="floor",
    )
    spectrum = dense_full_jacobian_eigenspectrum(
        reconstruction.spline_state[anchor_index], adapter.actual_f0
    )
    eigenvalues = spectrum.map_eigenvalues
    magnitudes = eigenvalues.abs()
    flow = output_projected_flow(
        reconstruction.spline_state, adapter.actual_f0, decoder
    )
    angular_flow = signed_angular_flow(flow.output, flow.projected_vector_field)
    topology = cyclic_flow_reversal_topology(
        reconstruction.spline_angle, angular_flow
    )
    gains = _tangent_radial_gains(
        reconstruction.spline_state, spectrum.map_jacobian, anchor_index
    )
    target = batch.output_targets
    mse = (prediction - target).square().mean()
    stable_angles = torch.as_tensor(
        [item.angle for item in topology.reversals if item.kind == "stable"],
        device=device,
        dtype=reconstruction.spline_angle.dtype,
    )
    saddle_angles = torch.as_tensor(
        [item.angle for item in topology.reversals if item.kind == "saddle"],
        device=device,
        dtype=reconstruction.spline_angle.dtype,
    )
    finite_memory = _finite_time_memory(
        reconstruction.spline_state,
        reconstruction.spline_angle,
        adapter.actual_f0,
        decoder,
        horizon=spec.blank_horizon,
    )
    terminal_angle = finite_memory["predicted_angle"][:, -1]
    assigned_angle, basin_counts, effective_basin_count = _assigned_stable_angles(
        terminal_angle, stable_angles
    )
    terminal_error = circular_absolute_error(
        terminal_angle, reconstruction.spline_angle
    )
    seed_output = output / f"seed{seed:02d}"
    seed_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        seed_output / "slow_manifold_reconstruction.npz",
        spline_angle=_np(reconstruction.spline_angle),
        spline_state=_np(reconstruction.spline_state),
        selected_state=_np(reconstruction.selected_state),
        selected_candidate_time=reconstruction.selected_candidate_time,
        selected_candidate_trajectory=reconstruction.selected_candidate_trajectory,
        knot_angle=_np(reconstruction.knot_angle),
        knot_state=_np(reconstruction.knot_state),
    )
    np.savez_compressed(
        seed_output / "local_jacobian.npz",
        anchor_index=_np(anchor_index),
        map_eigenvalues=_np(eigenvalues),
        map_jacobian=_np(spectrum.map_jacobian),
    )
    np.savez_compressed(
        seed_output / "projected_flow_and_topology.npz",
        spline_angle=_np(reconstruction.spline_angle),
        output=_np(flow.output),
        next_output=_np(flow.next_output),
        projected_vector_field=_np(flow.projected_vector_field),
        pointwise_euclidean_norm=_np(flow.pointwise_norm),
        signed_angular_flow=_np(angular_flow),
        stable_fixed_point_angle=_np(stable_angles),
        saddle_fixed_point_angle=_np(saddle_angles),
    )
    vector_field_eigenvalues = eigenvalues - 1.0
    ranked_real = torch.sort(vector_field_eigenvalues.real, dim=1, descending=True).values
    np.savez_compressed(
        seed_output / "full_local_eigenspectrum.npz",
        spline_index=_np(anchor_index),
        spline_angle=_np(reconstruction.spline_angle[anchor_index]),
        map_eigenvalues=_np(eigenvalues),
        vector_field_eigenvalues=_np(vector_field_eigenvalues),
        largest_real_part=_np(ranked_real[:, 0]),
        second_largest_real_part=_np(ranked_real[:, 1]),
        real_part_gap=_np(ranked_real[:, 0] - ranked_real[:, 1]),
        map_spectral_radius=_np(magnitudes.max(dim=1).values),
    )
    np.savez_compressed(
        seed_output / "finite_time_angular_memory.npz",
        **{
            key: _np(value)
            for key, value in finite_memory.items()
            if key != "terminal_state"
        },
    )
    np.savez_compressed(
        seed_output / "asymptotic_structure.npz",
        initial_angle=_np(reconstruction.spline_angle),
        observed_terminal_angle=_np(terminal_angle),
        stable_fixed_point_angle=_np(stable_angles),
        saddle_fixed_point_angle=_np(saddle_angles),
        uniform_grid_basin_counts=_np(basin_counts),
        assigned_stable_angle=_np(assigned_angle),
        asymptotic_absolute_error=_np(terminal_error),
    )
    topology_payload = _topology_payload(topology)
    summary = {
        "schema_version": 1,
        "analysis_status": "complete_extended_structural_analysis",
        "analysis_spec": {
            "trajectory_count": int(spec.trajectory_count),
            "spline_count": int(spec.spline_count),
            "task_horizon": int(spec.task_horizon),
            "blank_horizon": int(spec.blank_horizon),
        },
        "model": (
            "G-C_gradient_only_recurrent_writer"
            if retention_mode == "gradient_only"
            else "H-C_hybrid_rp_recurrent_writer"
        ),
        "seed": seed,
        "task_mse_full_tensor": float(mse.cpu()),
        "reconstruction": reconstruction.qa,
        "projected_flow": {
            "uniform_norm": float(flow.uniform_norm.cpu()),
            "pointwise_norm": _summary(flow.pointwise_norm),
            "absolute_angular_flow": _summary(angular_flow.abs()),
        },
        "topology": topology_payload,
        "fixed_point_topology": topology_payload,
        "finite_time_angular_memory": {
            "blank_horizon": int(spec.blank_horizon),
            "initial_memory_count": int(reconstruction.spline_angle.numel()),
            "terminal_mean_error_radians": float(terminal_error.mean().cpu()),
            "terminal_median_error_radians": float(terminal_error.median().cpu()),
            "terminal_maximum_error_radians": float(terminal_error.max().cpu()),
        },
        "asymptotic_structure": {
            "topology": topology.kind,
            "stable_count": int(stable_angles.numel()),
            "saddle_count": int(saddle_angles.numel()),
            "asymptotic_mean_error_radians": float(terminal_error.mean().cpu()),
            "asymptotic_maximum_error_radians": float(terminal_error.max().cpu()),
            "effective_basin_count": effective_basin_count,
        },
        "jacobian": {
            "spectral_radius": _summary(magnitudes.max(dim=1).values),
            "eigenvalue_magnitude_top1": _summary(magnitudes.max(dim=1).values),
            "unstable_eigenvalue_count": _summary((magnitudes > 1.0).sum(dim=1)),
            "near_unit_eigenvalue_count_abs_tol_1e-3": _summary(
                ((magnitudes - 1.0).abs() <= 1.0e-3).sum(dim=1)
            ),
            **gains,
        },
    }
    # A divergent finite kick is itself an attractor result.  Keep it isolated
    # so it cannot discard otherwise valid reconstruction, flow, and Jacobian
    # diagnostics for the same checkpoint.
    try:
        recovery = carrier_ambient_normal_recovery(
            reconstruction.spline_state,
            reconstruction.spline_angle,
            adapter.actual_f0,
            decoder,
            anchor_count=NORMAL_RECOVERY_ANCHOR_COUNT,
            ambient_directions_per_anchor=NORMAL_RECOVERY_AMBIENT_DIRECTIONS,
            radii_over_manifold_scale=NORMAL_RECOVERY_RADII_OVER_MANIFOLD_SCALE,
            horizons=NORMAL_RECOVERY_HORIZONS,
            seed=314159,
            distance_chunk_size=1024,
        )
        np.savez_compressed(
            seed_output / "normal_recovery.npz",
            family=np.asarray(recovery.family, dtype="U32"),
            anchor_index=_np(recovery.anchor_index),
            radius_over_manifold_scale=_np(recovery.radius_over_manifold_scale),
            horizon=_np(recovery.horizon),
            manifold_distance_ratio=_np(recovery.manifold_distance_ratio),
            distance_to_matched_clean_state_ratio=_np(
                recovery.distance_to_matched_clean_state_ratio
            ),
            same_memory_error_radians=_np(recovery.same_memory_error_radians),
        )
        summary["normal_recovery"] = {
            "status": "estimated",
            **_recovery_payload(recovery),
        }
    except Exception as error:
        # Retry one radius at a time so a large-kick divergence does not hide
        # otherwise finite local-radius recovery cells.
        recovered: dict[str, Any] = {
            "ambient_normal": {},
            "in_plane_radial": {},
        }
        radius_errors: dict[str, str] = {}
        recovered_count = 0
        for radius_value in NORMAL_RECOVERY_RADII_OVER_MANIFOLD_SCALE:
            radius_key = format(float(radius_value), ".12g")
            try:
                radius_recovery = carrier_ambient_normal_recovery(
                    reconstruction.spline_state,
                    reconstruction.spline_angle,
                    adapter.actual_f0,
                    decoder,
                    anchor_count=NORMAL_RECOVERY_ANCHOR_COUNT,
                    ambient_directions_per_anchor=NORMAL_RECOVERY_AMBIENT_DIRECTIONS,
                    radii_over_manifold_scale=(float(radius_value),),
                    horizons=NORMAL_RECOVERY_HORIZONS,
                    seed=314159,
                    distance_chunk_size=1024,
                )
                radius_payload = _recovery_payload(radius_recovery)
                for family_name in recovered:
                    recovered[family_name].update(radius_payload[family_name])
                np.savez_compressed(
                    seed_output / f"normal_recovery_radius_{radius_key}.npz",
                    family=np.asarray(radius_recovery.family, dtype="U32"),
                    anchor_index=_np(radius_recovery.anchor_index),
                    radius_over_manifold_scale=_np(
                        radius_recovery.radius_over_manifold_scale
                    ),
                    horizon=_np(radius_recovery.horizon),
                    manifold_distance_ratio=_np(
                        radius_recovery.manifold_distance_ratio
                    ),
                    distance_to_matched_clean_state_ratio=_np(
                        radius_recovery.distance_to_matched_clean_state_ratio
                    ),
                    same_memory_error_radians=_np(
                        radius_recovery.same_memory_error_radians
                    ),
                )
                recovered_count += 1
            except Exception as radius_error:
                radius_errors[radius_key] = (
                    f"{type(radius_error).__name__}:{radius_error}"
                )
        summary["normal_recovery"] = {
            "status": (
                "partially_estimated"
                if recovered_count
                else "failed_nonfinite_perturbation_rollout"
            ),
            "joint_error": f"{type(error).__name__}:{error}",
            "failed_radii": radius_errors,
            **recovered,
        }
    atomic_json(seed_output / "summary.json", _native(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--retention-mode",
        choices=("gradient_only", "hybrid_rp"),
        default="gradient_only",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    bank = load_fixed_bank(Path(args.bank).expanduser().resolve(strict=True))
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    # Preserve completed seed artifacts when a subset is rerun.
    summaries: dict[str, Any] = {}
    for seed_summary in sorted(output.glob("seed[0-9][0-9]/summary.json")):
        existing = json.loads(seed_summary.read_text(encoding="utf-8"))
        summaries[str(int(existing["seed"]))] = existing
    for seed in seeds:
        label = "G-C" if args.retention_mode == "gradient_only" else "H-C"
        print(f"analyzing {label} seed {seed}", flush=True)
        try:
            summaries[str(seed)] = analyze_seed(
                root=root,
                bank=bank,
                output=output,
                seed=seed,
                retention_mode=args.retention_mode,
                device=torch.device(args.device),
            )
        except Exception as error:
            failure = {"status": "failed", "error": f"{type(error).__name__}:{error}"}
            summaries[str(seed)] = failure
            atomic_json(output / f"seed{seed:02d}_failure.json", failure)
            print(f"seed {seed} failed: {failure['error']}", flush=True)
    atomic_json(
        output / "summary.json",
        {
            "schema_version": 1,
            "model": "G-C" if args.retention_mode == "gradient_only" else "H-C",
            "seeds": _native(summaries),
        },
    )
    return 0 if all(value.get("status") != "failed" for value in summaries.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
