"""Checkpoint-only diagnostics for H-C's exp(tanh) retention field.

The registered training and Ságodi analyses are left untouched.  For each H-C
checkpoint this script probes

    lambda_j(h) = base_lambda_j * exp(a * tanh(r_j(h)))

on inward, on-manifold, and outward in-plane radial displacements.  Successful
seeds use the already reconstructed slow manifold.  A seed without a valid
slow manifold is kept as an explicitly labelled task-endpoint proxy and is
also followed along its true autonomous blank rollout to diagnose divergence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from repro.sagodi_protocol.artifacts import atomic_json
from repro.sagodi_protocol.state import StateAdapter
from repro.sagodi_protocol.tasks import Batch, load_fixed_bank

from .models import build_state_dependent_model, recurrence


DEFAULT_OFFSETS = (-0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10)


def _summary(values: torch.Tensor | np.ndarray) -> dict[str, float | int]:
    array = (
        values.detach().cpu().numpy()
        if isinstance(values, torch.Tensor)
        else np.asarray(values)
    )
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if not flat.size:
        raise ValueError("cannot summarize an empty finite array")
    return {
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "q05": float(np.quantile(flat, 0.05)),
        "q95": float(np.quantile(flat, 0.95)),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
    }


def _load_model(root: Path, seed: int, device: torch.device):
    checkpoint_path = (
        root
        / "runs"
        / f"hybrid_rp__recurrent__seed{seed:02d}"
        / "checkpoint_trained.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_state_dependent_model(
        "recurrent", model_seed=seed, retention_mode="hybrid_rp"
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint_path


@torch.no_grad()
def _task_endpoints(model, bank: Batch, device: torch.device):
    batch = Batch(
        bank.inputs.to(device),
        bank.output_targets.to(device),
        bank.latent_targets.to(device),
        bank.mask.to(device),
        bank.metadata,
    )
    _, states = model.forward_sequence(
        batch.inputs,
        initial_memory=batch.output_targets[0],
        return_states=True,
    )
    adapter = StateAdapter(model.core)
    endpoint = adapter.primary_from_reported(states[-1])
    target = batch.output_targets[-1]
    angle = torch.atan2(target[:, 1], target[:, 0])
    return endpoint, angle


def _radial_frame(states: torch.Tensor, angle: torch.Tensor):
    """Construct an oriented in-plane normal from a periodic state curve."""

    order = torch.argsort(angle)
    curve = states[order]
    ordered_angle = angle[order]
    center = curve.mean(dim=0, keepdim=True)
    centered = curve - center
    _, _, right = torch.linalg.svd(centered, full_matrices=False)
    plane = right[:2]
    tangent = torch.roll(curve, -1, 0) - torch.roll(curve, 1, 0)
    tangent_plane = tangent @ plane.transpose(0, 1)
    tangent_norm = torch.linalg.vector_norm(tangent_plane, dim=1, keepdim=True)
    tangent_plane = tangent_plane / tangent_norm.clamp_min(1.0e-12)
    radial_coordinates = torch.stack(
        (-tangent_plane[:, 1], tangent_plane[:, 0]), dim=1
    )
    radial = radial_coordinates @ plane
    projected_centered = (centered @ plane.transpose(0, 1)) @ plane
    outward_score = (radial * projected_centered).sum(dim=1, keepdim=True)
    radial = torch.where(outward_score < 0.0, -radial, radial)

    # Degenerate local tangents can occur for a failed task solution.  Fall
    # back to the PCA-plane direction from the curve centroid at those points.
    radial_norm = torch.linalg.vector_norm(radial, dim=1, keepdim=True)
    fallback_norm = torch.linalg.vector_norm(
        projected_centered, dim=1, keepdim=True
    )
    fallback = projected_centered / fallback_norm.clamp_min(1.0e-12)
    radial = torch.where(radial_norm <= 1.0e-8, fallback, radial)
    radial = radial / torch.linalg.vector_norm(radial, dim=1, keepdim=True).clamp_min(
        1.0e-12
    )
    scale = torch.sqrt(centered.square().sum(dim=1).mean())
    if not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
        raise ValueError("radial reference scale is not positive and finite")
    return curve, ordered_angle, radial, scale, plane


@torch.no_grad()
def _radial_probe(
    model,
    states: torch.Tensor,
    radial: torch.Tensor,
    scale: torch.Tensor,
    offsets: tuple[float, ...],
):
    rec = recurrence(model)
    rows: list[dict[str, Any]] = []
    tensors: dict[str, list[np.ndarray]] = {
        "lambda": [],
        "weighted_lambda": [],
        "weighted_fraction_above_one": [],
        "radial_vector_field": [],
    }
    for offset in offsets:
        perturbed = states + float(offset) * scale * radial
        dynamic_lambda = rec.state_dependent_lambda(perturbed)
        next_state = rec.step(torch.zeros_like(perturbed[:, : rec.input_dim]), perturbed)
        weights = perturbed.square()
        weight_sum = weights.sum(dim=1).clamp_min(1.0e-12)
        weighted_lambda = (weights * dynamic_lambda).sum(dim=1) / weight_sum
        weighted_above = (
            weights * (dynamic_lambda > 1.0).to(weights.dtype)
        ).sum(dim=1) / weight_sum
        radial_flow = ((next_state - perturbed) * radial).sum(dim=1) / scale
        tensors["lambda"].append(dynamic_lambda.detach().cpu().numpy())
        tensors["weighted_lambda"].append(weighted_lambda.detach().cpu().numpy())
        tensors["weighted_fraction_above_one"].append(
            weighted_above.detach().cpu().numpy()
        )
        tensors["radial_vector_field"].append(radial_flow.detach().cpu().numpy())
        rows.append(
            {
                "offset_over_reference_scale": float(offset),
                "lambda": _summary(dynamic_lambda),
                "fraction_above_one": float((dynamic_lambda > 1.0).float().mean()),
                "weighted_lambda": _summary(weighted_lambda),
                "weighted_fraction_above_one": _summary(weighted_above),
                "radial_vector_field_over_reference_scale": _summary(radial_flow),
            }
        )

    lambda_array = np.stack(tensors["lambda"], axis=0)
    weighted_lambda_array = np.stack(tensors["weighted_lambda"], axis=0)
    weighted_above_array = np.stack(
        tensors["weighted_fraction_above_one"], axis=0
    )
    radial_flow_array = np.stack(tensors["radial_vector_field"], axis=0)
    offset_array = np.asarray(offsets, dtype=np.float64)
    zero_index = int(np.flatnonzero(np.isclose(offset_array, 0.0))[0])
    negative_index = int(np.flatnonzero(offset_array < 0.0)[-1])
    positive_index = int(np.flatnonzero(offset_array > 0.0)[0])
    delta = float(offset_array[positive_index])
    if not math.isclose(-float(offset_array[negative_index]), delta):
        raise ValueError("nearest inward/outward offsets must be symmetric")

    # The signed slope of the radial map.  Values in (-1, 1) are locally
    # contractive in this direction; values below 1 shrink a small radial
    # displacement relative to its matched clean state.
    perturbed_plus = states + delta * scale * radial
    perturbed_minus = states - delta * scale * radial
    f_plus = rec.step(torch.zeros_like(perturbed_plus[:, : rec.input_dim]), perturbed_plus)
    f_minus = rec.step(
        torch.zeros_like(perturbed_minus[:, : rec.input_dim]), perturbed_minus
    )
    signed_radial_map_slope = (
        ((f_plus - f_minus) * radial).sum(dim=1) / (2.0 * delta * scale)
    )
    baseline_flow = radial_flow_array[zero_index]
    inward_restoring = radial_flow_array[negative_index] - baseline_flow
    outward_restoring = radial_flow_array[positive_index] - baseline_flow
    restoring_both = (inward_restoring > 0.0) & (outward_restoring < 0.0)

    result = {
        "offsets": rows,
        "nearest_symmetric_offset": delta,
        "signed_local_radial_map_slope": _summary(signed_radial_map_slope),
        "fraction_local_radial_slope_below_one": float(
            (signed_radial_map_slope < 1.0).float().mean()
        ),
        "fraction_local_radial_slope_absolute_below_one": float(
            (signed_radial_map_slope.abs() < 1.0).float().mean()
        ),
        "fraction_bidirectionally_restoring": float(restoring_both.mean()),
        "inward_relative_radial_flow": _summary(inward_restoring),
        "outward_relative_radial_flow": _summary(outward_restoring),
    }
    arrays = {
        "offset_over_reference_scale": offset_array,
        "dynamic_lambda": lambda_array,
        "weighted_lambda": weighted_lambda_array,
        "weighted_fraction_above_one": weighted_above_array,
        "radial_vector_field_over_reference_scale": radial_flow_array,
        "signed_local_radial_map_slope": signed_radial_map_slope.cpu().numpy(),
        "bidirectionally_restoring": restoring_both,
    }
    return result, arrays


@torch.no_grad()
def _coordinate_diagnostics(model, reference_state: torch.Tensor):
    """Identify whether a small expansive coordinate set owns state energy."""

    rec = recurrence(model)
    base_lambda = rec.lam_mag().to(reference_state)
    gate_tanh = torch.tanh(rec.retention_gate(reference_state))
    dynamic_lambda = rec.state_dependent_lambda(reference_state)
    coordinate_energy = reference_state.to(torch.float64).square().sum(dim=0)
    energy_share = coordinate_energy / coordinate_energy.sum().clamp_min(1.0e-24)
    participation_ratio = 1.0 / energy_share.square().sum()
    positive = energy_share[energy_share > 0.0]
    entropy_effective_count = torch.exp(-(positive * positive.log()).sum())
    top_index = torch.argsort(energy_share, descending=True)[:10]
    table = []
    for index in top_index.tolist():
        table.append(
            {
                "coordinate": int(index),
                "state_energy_share": float(energy_share[index]),
                "base_lambda": float(base_lambda[index]),
                "dynamic_lambda_mean": float(dynamic_lambda[:, index].mean()),
                "dynamic_lambda_minimum": float(dynamic_lambda[:, index].min()),
                "dynamic_lambda_maximum": float(dynamic_lambda[:, index].max()),
                "gate_tanh_mean": float(gate_tanh[:, index].mean()),
                "gate_tanh_minimum": float(gate_tanh[:, index].min()),
                "gate_tanh_maximum": float(gate_tanh[:, index].max()),
            }
        )
    energy_on_expansive = (
        reference_state.to(torch.float64).square()
        * (dynamic_lambda > 1.0).to(torch.float64)
    ).sum() / reference_state.to(torch.float64).square().sum().clamp_min(1.0e-24)
    energy_on_saturated_gate = (
        reference_state.to(torch.float64).square()
        * (gate_tanh.abs() >= 0.99).to(torch.float64)
    ).sum() / reference_state.to(torch.float64).square().sum().clamp_min(1.0e-24)
    return {
        "base_lambda": _summary(base_lambda),
        "base_lambda_count_above_0p999": int((base_lambda > 0.999).sum()),
        "state_energy_participation_ratio": float(participation_ratio),
        "state_energy_entropy_effective_coordinate_count": float(
            entropy_effective_count
        ),
        "state_energy_fraction_on_dynamic_lambda_above_one": float(
            energy_on_expansive
        ),
        "state_energy_fraction_on_abs_gate_tanh_at_least_0p99": float(
            energy_on_saturated_gate
        ),
        "top_coordinates_by_state_energy": table,
    }, {
        "base_lambda": base_lambda.cpu().numpy(),
        "reference_gate_tanh": gate_tanh.cpu().numpy(),
        "reference_dynamic_lambda": dynamic_lambda.cpu().numpy(),
        "reference_coordinate_energy_share": energy_share.cpu().numpy(),
    }


@torch.no_grad()
def _blank_trace(model, initial_state: torch.Tensor, horizon: int):
    rec = recurrence(model)
    adapter = StateAdapter(model.core)
    state = initial_state.clone()
    records: list[dict[str, float | int]] = []
    failed_step: int | None = None
    for step in range(horizon + 1):
        finite = bool(torch.isfinite(state).all())
        if not finite:
            failed_step = step
            break
        dynamic_lambda = rec.state_dependent_lambda(state)
        # The failed seed can approach float32's finite limit one transition
        # before the recurrence itself becomes non-finite.  Accumulate only
        # diagnostic norms and energy weights in float64 so that measurement
        # overflow is not mistaken for an earlier model failure.
        state64 = state.to(torch.float64)
        norm = torch.linalg.vector_norm(state64, dim=1)
        weights = state64.square()
        weighted_above = (
            weights * (dynamic_lambda > 1.0).to(torch.float64)
        ).sum() / weights.sum().clamp_min(1.0e-12)
        records.append(
            {
                "step": step,
                "state_norm_median": float(norm.median()),
                "state_norm_q95": float(torch.quantile(norm, 0.95)),
                "state_norm_maximum": float(norm.max()),
                "lambda_mean": float(dynamic_lambda.mean()),
                "lambda_maximum": float(dynamic_lambda.max()),
                "lambda_fraction_above_one": float(
                    (dynamic_lambda > 1.0).float().mean()
                ),
                "state_energy_weighted_fraction_above_one": float(weighted_above),
            }
        )
        if step < horizon:
            state = adapter.actual_f0(state)
    return {"failed_step": failed_step, "records": records}


def _plot_radial_profiles(results: dict[int, dict[str, Any]], output: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)
    colors = {0: "#2171b5", 1: "#cb181d", 2: "#238b45"}
    for seed, result in sorted(results.items()):
        rows = result["radial_probe"]["offsets"]
        x = np.asarray([row["offset_over_reference_scale"] for row in rows])
        label = f"seed {seed}" + (" (endpoint proxy)" if result["reference_kind"] != "slow_manifold" else "")
        axes[0].plot(
            x,
            [row["weighted_lambda"]["mean"] for row in rows],
            marker="o",
            color=colors[seed],
            label=label,
        )
        axes[1].plot(
            x,
            [row["fraction_above_one"] for row in rows],
            marker="o",
            color=colors[seed],
        )
        baseline = next(
            row["radial_vector_field_over_reference_scale"]["mean"]
            for row in rows
            if math.isclose(row["offset_over_reference_scale"], 0.0)
        )
        axes[2].plot(
            x,
            [
                row["radial_vector_field_over_reference_scale"]["mean"] - baseline
                for row in rows
            ],
            marker="o",
            color=colors[seed],
        )
    axes[0].axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    axes[2].axhline(0.0, color="0.35", linestyle="--", linewidth=1)
    for axis in axes:
        axis.axvline(0.0, color="0.75", linewidth=1)
        axis.set_xlabel("radial offset / ring scale")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("state-energy-weighted mean $\\lambda$")
    axes[1].set_ylabel("fraction of coordinates with $\\lambda>1$")
    axes[2].set_ylabel("radial flow relative to on-ring")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("H-C dynamic retention across the radial direction")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"fig_hc_dynamic_retention_radial.{suffix}", dpi=220)
    plt.close(fig)


def _plot_blank_trace(results: dict[int, dict[str, Any]], output: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)
    colors = {0: "#2171b5", 1: "#cb181d", 2: "#238b45"}
    for seed, result in sorted(results.items()):
        records = result["blank_trace"]["records"]
        step = np.asarray([item["step"] for item in records])
        # Plot every recorded point; vector output remains small at 1025 steps.
        axes[0].plot(
            step,
            [item["state_norm_q95"] for item in records],
            color=colors[seed],
            label=f"seed {seed}",
        )
        axes[1].plot(
            step,
            [item["lambda_maximum"] for item in records],
            color=colors[seed],
        )
        axes[2].plot(
            step,
            [item["state_energy_weighted_fraction_above_one"] for item in records],
            color=colors[seed],
        )
        failed = result["blank_trace"]["failed_step"]
        if failed is not None:
            for axis in axes:
                axis.axvline(failed, color=colors[seed], linestyle=":", linewidth=1)
    axes[0].set_yscale("log")
    axes[1].axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    axes[0].set_ylabel("state norm (95th percentile; log scale)")
    axes[1].set_ylabel("maximum dynamic $\\lambda$")
    axes[2].set_ylabel("state-energy fraction at $\\lambda>1$")
    for axis in axes:
        axis.set_xlabel("blank-input step")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("H-C autonomous blank-input stability")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"fig_hc_blank_stability.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--manifold-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blank-horizon", type=int, default=1024)
    parser.add_argument(
        "--offsets", default=",".join(format(value, ".12g") for value in DEFAULT_OFFSETS)
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve(strict=True)
    bank_path = Path(args.bank).expanduser().resolve(strict=True)
    manifold_root = Path(args.manifold_root).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    offsets = tuple(float(value) for value in args.offsets.split(",") if value.strip())
    if 0.0 not in offsets or not any(value < 0.0 for value in offsets) or not any(
        value > 0.0 for value in offsets
    ):
        raise ValueError("offsets must include zero, inward, and outward probes")
    device = torch.device(args.device)
    bank = load_fixed_bank(bank_path)
    results: dict[int, dict[str, Any]] = {}

    for seed in seeds:
        print(f"analyzing H-C dynamic retention seed {seed}", flush=True)
        model, checkpoint_path = _load_model(root, seed, device)
        endpoint, target_angle = _task_endpoints(model, bank, device)
        manifold_path = (
            manifold_root / f"seed{seed:02d}" / "slow_manifold_reconstruction.npz"
        )
        if manifold_path.exists():
            with np.load(manifold_path) as stored:
                reference_state = torch.as_tensor(
                    stored["spline_state"], device=device, dtype=endpoint.dtype
                )
                reference_angle = torch.as_tensor(
                    stored["spline_angle"], device=device, dtype=endpoint.dtype
                )
            reference_kind = "slow_manifold"
        else:
            reference_state = endpoint
            reference_angle = target_angle
            reference_kind = "task_endpoint_proxy_no_valid_slow_manifold"

        curve, ordered_angle, radial, scale, plane = _radial_frame(
            reference_state, reference_angle
        )
        radial_result, radial_arrays = _radial_probe(
            model, curve, radial, scale, offsets
        )
        coordinate_result, coordinate_arrays = _coordinate_diagnostics(model, curve)
        blank_trace = _blank_trace(model, endpoint, int(args.blank_horizon))
        seed_output = output / f"seed{seed:02d}"
        seed_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            seed_output / "dynamic_retention_radial.npz",
            reference_state=curve.detach().cpu().numpy(),
            reference_angle=ordered_angle.detach().cpu().numpy(),
            radial_direction=radial.detach().cpu().numpy(),
            reference_scale=np.asarray(float(scale.cpu()), dtype=np.float64),
            pca_plane=plane.detach().cpu().numpy(),
            **radial_arrays,
            **coordinate_arrays,
        )
        seed_result = {
            "schema_version": 1,
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "reference_kind": reference_kind,
            "reference_count": int(curve.shape[0]),
            "reference_scale": float(scale.cpu()),
            "radial_probe": radial_result,
            "coordinate_diagnostics": coordinate_result,
            "blank_trace": blank_trace,
        }
        atomic_json(seed_output / "summary.json", seed_result)
        results[seed] = seed_result

    _plot_radial_profiles(results, output)
    _plot_blank_trace(results, output)
    aggregate = {
        "schema_version": 1,
        "analysis": "H-C exp(tanh) dynamic-retention radial diagnostic",
        "training_modified": False,
        "bank": str(bank_path),
        "manifold_root": str(manifold_root),
        "offsets_over_reference_scale": list(offsets),
        "seeds": {str(seed): result for seed, result in sorted(results.items())},
    }
    atomic_json(output / "summary.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True)[:12000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
