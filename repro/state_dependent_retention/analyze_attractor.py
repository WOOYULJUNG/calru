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


def analyze_seed(
    *,
    root: Path,
    bank: Batch,
    output: Path,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    run = root / "runs" / f"gradient_only__recurrent__seed{seed:02d}"
    checkpoint_path = run / "checkpoint_trained.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_state_dependent_model(
        "recurrent", model_seed=seed, retention_mode="gradient_only"
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
    summary = {
        "schema_version": 1,
        "model": "G-C_gradient_only_recurrent_writer",
        "seed": seed,
        "task_mse_full_tensor": float(mse.cpu()),
        "reconstruction": reconstruction.qa,
        "projected_flow": {
            "uniform_norm": float(flow.uniform_norm.cpu()),
            "pointwise_norm": _summary(flow.pointwise_norm),
            "absolute_angular_flow": _summary(angular_flow.abs()),
        },
        "topology": _topology_payload(topology),
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
        print(f"analyzing G-C seed {seed}", flush=True)
        try:
            summaries[str(seed)] = analyze_seed(
                root=root,
                bank=bank,
                output=output,
                seed=seed,
                device=torch.device(args.device),
            )
        except Exception as error:
            failure = {"status": "failed", "error": f"{type(error).__name__}:{error}"}
            summaries[str(seed)] = failure
            atomic_json(output / f"seed{seed:02d}_failure.json", failure)
            print(f"seed {seed} failed: {failure['error']}", flush=True)
    atomic_json(
        output / "summary.json",
        {"schema_version": 1, "model": "G-C", "seeds": _native(summaries)},
    )
    return 0 if all(value.get("status") != "failed" for value in summaries.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
