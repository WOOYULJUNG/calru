#!/usr/bin/env python3
"""OOD normal-kick probes for continuous-attractor geometry.

This is an evaluation-only analysis: it loads existing checkpoints and injects
hidden-state perturbations orthogonal to the local memory tangent.  The task
input remains either zero-input hold or tangent integration input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from exp71_pan_block_pulse_hold import build_model_variant, normalize_model_variant
from exp72_structured_attractor_tasks import (
    RING_TASKS,
    angle_diff,
    make_line_integrate_batch,
    make_ring_hold_batch,
    make_ring_integrate_batch,
    model_rank_for_task,
    ring_xy,
    state_from_line,
    state_from_ring_hold,
    state_from_ring_integrate,
    task_io_dims,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "analysis_normal_kick_ood"


@dataclass(frozen=True)
class EvalSpec:
    task: str
    dim: int
    label: str
    tag: str
    result_dir: str
    intervention: str = "none"


@dataclass(frozen=True)
class AnalysisSpec:
    checkpoint: EvalSpec
    eval_task: str
    label: str


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_list(text: str, cast):
    return [cast(x) for x in str(text).split(",") if x.strip()]


def to_float(value, default):
    if value in (None, "", "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default):
    if value in (None, "", "None"):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def lambda_to_theta(lam: float, device, dtype):
    q = torch.tensor(float(lam) ** 2, device=device, dtype=dtype).clamp(1e-8, 1.0 - 1e-8)
    return torch.logit(q)


@torch.no_grad()
def force_all_slow(model, lam: float = 0.999):
    if not hasattr(model, "pan_recs_with_slices"):
        return
    for rec, _ in model.pan_recs_with_slices():
        if hasattr(rec, "theta"):
            rec.theta.fill_(lambda_to_theta(lam, rec.theta.device, rec.theta.dtype))


def result_path(spec: EvalSpec, seed: int) -> Path:
    return ROOT / f"{spec.result_dir}_results" / f"{spec.task}_d{spec.dim}_{spec.tag}_seed{seed}.json"


def checkpoint_path(spec: EvalSpec, seed: int) -> Path:
    return ROOT / f"checkpoints_{spec.result_dir}" / f"exp72_{spec.task}_d{spec.dim}_{spec.tag}_seed{seed}.pt"


def load_model(spec: EvalSpec, seed: int, device: torch.device):
    rp = result_path(spec, seed)
    cp = checkpoint_path(spec, seed)
    if not rp.exists():
        raise FileNotFoundError(rp)
    if not cp.exists():
        raise FileNotFoundError(cp)
    result = json.loads(rp.read_text())
    input_dim, output_dim = task_io_dims(spec.task, spec.dim)
    variant = normalize_model_variant(result.get("model") or result.get("raw_model"))
    model = build_model_variant(
        variant=variant,
        input_dim=input_dim,
        output_dim=output_dim,
        rank=model_rank_for_task(spec.task, spec.dim),
        d_model=to_int(result.get("d_model"), 96),
        rec_dim=to_int(result.get("rec_dim"), 96),
        layers=to_int(result.get("layers"), 1),
        dropout=to_float(result.get("dropout"), 0.0),
        plru_tau=to_float(result.get("plru_tau"), 0.001),
        plru_c=to_float(result.get("plru_c"), 50.0),
        pan_lambda_min=to_float(result.get("pan_lambda_min"), 0.90),
        pan_lambda_max=to_float(result.get("pan_lambda_max"), 0.999),
        rank_matched_lambda_high=to_float(result.get("rank_matched_lambda_high"), 0.999),
        rank_matched_lambda_low=to_float(result.get("rank_matched_lambda_low"), 0.0),
    ).to(device)
    try:
        state_dict = torch.load(cp, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(cp, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    if spec.intervention == "all_slow":
        force_all_slow(model, lam=0.999)
    return model, result


def decoded_error(model, pred, target, task):
    if task in RING_TASKS:
        pred_theta = torch.atan2(pred[:, 1], pred[:, 0])
        target_theta = torch.atan2(target[:, 1], target[:, 0])
        return float(angle_diff(pred_theta, target_theta).abs().mean().item() * 180.0 / math.pi)
    return float(torch.sqrt(F.mse_loss(pred, target)).item())


def batch_project_tangent(diff: torch.Tensor, basis: torch.Tensor):
    coeff = torch.einsum("bsd,bs->bd", basis, diff)
    return torch.einsum("bsd,bd->bs", basis, coeff)


def default_specs(eval_tasks: set[str]):
    specs: list[AnalysisSpec] = []

    if {"line_hold", "line_integrate"} & eval_tasks:
        rows = [
            ("RNN", "rnn", "exp72_line_integrate_10k_d2_rnn"),
            ("GRU", "gru", "exp72_line_integrate_10k_d2_gru"),
            ("LSTM", "lstm", "exp72_line_integrate_10k_d2_lstm"),
            ("SSM", "ssm", "exp72_line_integrate_10k_d2_ssm"),
            ("LRU", "lru_full", "exp72_line_integrate_10k_d2_lru_full"),
            ("AM-LRU", "pan_full_eps1e-4", "exp72_line_integrate_10k_d2_pan_full_eps1e-4"),
            (
                "AM-LRU all-slow",
                "pan_full_eps1e-4",
                "exp72_line_integrate_10k_d2_pan_full_eps1e-4",
                "all_slow",
            ),
            ("AM-LRU shuffle", "pan_full_eps1e-4_shuffle", "exp75_damage_ablation_line_integrate_d2_pan_full_eps1e-4_shuffle"),
            ("AM-LRU eta=0", "pan_full_eta0", "exp72_line_integrate_10k_d2_pan_full_eta0"),
        ]
        for eval_task in sorted({"line_hold", "line_integrate"} & eval_tasks):
            for row in rows:
                label, tag, result_dir, *rest = row
                specs.append(
                    AnalysisSpec(
                        EvalSpec("line_integrate", 2, label, tag, result_dir, rest[0] if rest else "none"),
                        eval_task,
                        label,
                    )
                )

    if {"ring_hold", "ring_integrate"} & eval_tasks:
        rows = [
            ("RNN", "rnn", "exp73_ring_10k_ring_hold_rnn", "ring_hold"),
            ("GRU", "gru", "exp73_ring_10k_ring_hold_gru", "ring_hold"),
            ("LSTM", "lstm", "exp73_ring_10k_ring_hold_lstm", "ring_hold"),
            ("SSM", "ssm", "exp73_ring_10k_ring_hold_ssm", "ring_hold"),
            ("LRU", "lru_full", "exp73_ring_10k_ring_hold_lru_full", "ring_hold"),
            ("AM-LRU", "pan_full_eps3e-5", "exp73_ring_10k_ring_hold_pan_full_eps3e-5", "ring_hold"),
            (
                "AM-LRU all-slow",
                "pan_full_eps3e-5",
                "exp73_ring_10k_ring_hold_pan_full_eps3e-5",
                "ring_hold",
                "all_slow",
            ),
            (
                "AM-LRU shuffle",
                "pan_full_eps3e-5_shuffle",
                "exp75_damage_ablation_ring_hold_d1_pan_full_eps3e-5_shuffle",
                "ring_hold",
            ),
            ("AM-LRU eta=0", "pan_full_eta0", "exp73_ring_10k_ring_hold_pan_full_eta0", "ring_hold"),
        ]
        for label, tag, result_dir, ckpt_task, *rest in rows:
            if "ring_hold" in eval_tasks:
                specs.append(
                    AnalysisSpec(
                        EvalSpec(ckpt_task, 1, label, tag, result_dir, rest[0] if rest else "none"),
                        "ring_hold",
                        label,
                    )
                )

        integrate_rows = [
            ("RNN", "rnn", "exp73_ring_10k_ring_integrate_rnn"),
            ("GRU", "gru", "exp73_ring_10k_ring_integrate_gru"),
            ("LSTM", "lstm", "exp73_ring_10k_ring_integrate_lstm"),
            ("SSM", "ssm", "exp73_ring_10k_ring_integrate_ssm"),
            ("LRU", "lru_full", "exp73_ring_10k_ring_integrate_lru_full"),
            ("AM-LRU", "pan_full_eps3e-5", "exp73_ring_10k_ring_integrate_pan_full_eps3e-5"),
            (
                "AM-LRU all-slow",
                "pan_full_eps3e-5",
                "exp73_ring_10k_ring_integrate_pan_full_eps3e-5",
                "all_slow",
            ),
            (
                "AM-LRU shuffle",
                "pan_full_eps3e-5_shuffle",
                "exp75_damage_ablation_ring_integrate_d1_pan_full_eps3e-5_shuffle",
            ),
            ("AM-LRU eta=0", "pan_full_eta0", "exp73_ring_10k_ring_integrate_pan_full_eta0"),
        ]
        if "ring_integrate" in eval_tasks:
            for row in integrate_rows:
                label, tag, result_dir, *rest = row
                specs.append(
                    AnalysisSpec(
                        EvalSpec("ring_integrate", 1, label, tag, result_dir, rest[0] if rest else "none"),
                        "ring_integrate",
                        label,
                    )
                )

    return specs


@torch.no_grad()
def make_eval_sequence(eval_task: str, dim: int, batch: int, horizon: int, device, args):
    if eval_task == "line_hold":
        z0 = (torch.rand(batch, dim, device=device) * 2.0 - 1.0) * float(args.z_scale)
        vel = torch.zeros(horizon, batch, dim, device=device)
        x = torch.zeros(horizon + 1, batch, 2 * dim + 1, device=device)
        x[0, :, :dim] = z0
        x[0, :, 2 * dim] = 1.0
        y = z0.unsqueeze(0).expand(horizon + 1, batch, dim).clone()
        return x, y, y[-1], {"z0": z0, "vel": vel}
    if eval_task == "line_integrate":
        return make_line_integrate_batch(
            batch,
            dim,
            horizon,
            device,
            z_scale=float(args.z_scale),
            vel_scale=float(args.vel_scale),
        )
    if eval_task == "ring_hold":
        return make_ring_hold_batch(batch, horizon, device)
    if eval_task == "ring_integrate":
        return make_ring_integrate_batch(
            batch,
            horizon,
            device,
            omega_scale=float(args.omega_scale),
            hold_prob=float(args.omega_hold_prob),
        )
    raise ValueError(eval_task)


@torch.no_grad()
def tangent_basis_at_step(model, eval_task: str, dim: int, aux: dict, step: int, eps: float):
    if eval_task in ("line_hold", "line_integrate"):
        z0 = aux["z0"]
        vel = aux["vel"][:step]
        jac = []
        for j in range(dim):
            dz = torch.zeros_like(z0)
            dz[:, j] = eps
            sp, _ = state_from_line(model, z0 + dz, vel)
            sm, _ = state_from_line(model, z0 - dz, vel)
            jac.append((sp - sm) / (2.0 * eps))
        jac = torch.stack(jac, dim=1)
    elif eval_task == "ring_hold":
        theta = aux["theta0"]
        sp, _ = state_from_ring_hold(model, theta + eps, step)
        sm, _ = state_from_ring_hold(model, theta - eps, step)
        jac = ((sp - sm) / (2.0 * eps)).unsqueeze(1)
    elif eval_task == "ring_integrate":
        theta = aux["theta0"]
        omega = aux["omega"][:step]
        sp, _ = state_from_ring_integrate(model, theta + eps, omega)
        sm, _ = state_from_ring_integrate(model, theta - eps, omega)
        jac = ((sp - sm) / (2.0 * eps)).unsqueeze(1)
    else:
        raise ValueError(eval_task)

    bases = []
    for i in range(jac.shape[0]):
        q, _ = torch.linalg.qr(jac[i].T, mode="reduced")
        bases.append(q)
    return torch.stack(bases, dim=0)


def event_times(horizon: int, count: int):
    if count <= 0:
        return []
    times = []
    for k in range(count):
        frac = (k + 1) / (count + 1)
        t = int(round(frac * max(1, horizon - 1)))
        t = max(0, min(horizon - 1, t))
        if t not in times:
            times.append(t)
    return times


@torch.no_grad()
def run_clean_states(model, x):
    states = []
    state = model.init_state(x.shape[1], x.device)
    for x_t in x:
        state = model.step(x_t, state)
        states.append(state)
    return states


@torch.no_grad()
def evaluate_normal_kicks(model, spec: AnalysisSpec, seed: int, device, args, kick_count: int, radius_scale: float):
    torch.manual_seed(int(args.base_seed) + 1000 * seed + 17 * kick_count + int(radius_scale * 1000))
    x, _, target, aux = make_eval_sequence(spec.eval_task, spec.checkpoint.dim, args.batch, args.horizon, device, args)
    clean_states = run_clean_states(model, x)
    clean_final = clean_states[-1]
    clean_error = decoded_error(model, model.decode(clean_final), target, spec.checkpoint.task)

    events = event_times(args.horizon, kick_count)
    bases = {
        t: tangent_basis_at_step(model, spec.eval_task, spec.checkpoint.dim, aux, t, float(args.tangent_eps))
        for t in events
    }
    final_basis = tangent_basis_at_step(
        model,
        spec.eval_task,
        spec.checkpoint.dim,
        aux,
        args.horizon,
        float(args.tangent_eps),
    )

    pert = model.init_state(x.shape[1], x.device)
    kick_budget_sq = 0.0
    tangent_ratio_sum = 0.0
    for t, x_t in enumerate(x):
        pert = model.step(x_t, pert)
        if t in bases:
            basis = bases[t]
            noise = torch.randn_like(pert)
            tangent = batch_project_tangent(noise, basis)
            normal = noise - tangent
            normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            state_rms = clean_states[t].std(dim=0).pow(2).mean().sqrt().clamp_min(1e-3)
            radius = float(radius_scale) * state_rms
            kick = radius * normal
            tan_part = batch_project_tangent(kick, basis)
            tangent_ratio_sum += float((tan_part.norm(dim=-1) / kick.norm(dim=-1).clamp_min(1e-8)).mean().item())
            kick_budget_sq += float(radius.item() ** 2)
            pert = pert + kick

    pred = model.decode(pert)
    pert_error = decoded_error(model, pred, target, spec.checkpoint.task)
    diff = pert - clean_final
    tan = batch_project_tangent(diff, final_basis)
    norm_part = diff - tan
    denom = math.sqrt(max(kick_budget_sq, 1e-12))
    return {
        "eval_task": spec.eval_task,
        "checkpoint_task": spec.checkpoint.task,
        "dim": spec.checkpoint.dim,
        "label": spec.label,
        "tag": spec.checkpoint.tag,
        "intervention": spec.checkpoint.intervention,
        "seed": seed,
        "horizon": int(args.horizon),
        "kick_count": int(kick_count),
        "normal_radius": float(radius_scale),
        "actual_event_count": len(events),
        "clean_error": clean_error,
        "perturbed_error": pert_error,
        "excess_error": pert_error - clean_error,
        "hidden_total_gain_per_budget": float(diff.norm(dim=-1).mean().item() / denom),
        "hidden_normal_gain_per_budget": float(norm_part.norm(dim=-1).mean().item() / denom),
        "hidden_tangent_leak_per_budget": float(tan.norm(dim=-1).mean().item() / denom),
        "kick_tangent_fraction": tangent_ratio_sum / max(1, len(events)),
    }


@torch.no_grad()
def prepare_eval_context(model, spec: AnalysisSpec, device, args, basis_times: list[int]):
    x, _, target, aux = make_eval_sequence(spec.eval_task, spec.checkpoint.dim, args.batch, args.horizon, device, args)
    clean_states = run_clean_states(model, x)
    clean_final = clean_states[-1]
    clean_error = decoded_error(model, model.decode(clean_final), target, spec.checkpoint.task)
    basis_cache = {
        int(t): tangent_basis_at_step(model, spec.eval_task, spec.checkpoint.dim, aux, int(t), float(args.tangent_eps))
        for t in sorted(set(int(t) for t in basis_times))
    }
    final_basis = tangent_basis_at_step(
        model,
        spec.eval_task,
        spec.checkpoint.dim,
        aux,
        args.horizon,
        float(args.tangent_eps),
    )
    return {
        "x": x,
        "target": target,
        "clean_states": clean_states,
        "clean_final": clean_final,
        "clean_error": clean_error,
        "basis_cache": basis_cache,
        "final_basis": final_basis,
    }


@torch.no_grad()
def evaluate_normal_kicks_fast(
    model,
    spec: AnalysisSpec,
    seed: int,
    args,
    ctx: dict,
    kick_count: int,
    radius_scale: float,
):
    torch.manual_seed(int(args.base_seed) + 1000 * seed + 17 * kick_count + int(radius_scale * 1000))
    x = ctx["x"]
    target = ctx["target"]
    clean_states = ctx["clean_states"]
    clean_final = ctx["clean_final"]
    basis_cache = ctx["basis_cache"]
    final_basis = ctx["final_basis"]
    events = event_times(args.horizon, kick_count)

    pert = model.init_state(x.shape[1], x.device)
    kick_budget_sq = 0.0
    tangent_ratio_sum = 0.0
    for t, x_t in enumerate(x):
        pert = model.step(x_t, pert)
        if t in events:
            basis = basis_cache[t]
            noise = torch.randn_like(pert)
            tangent = batch_project_tangent(noise, basis)
            normal = noise - tangent
            normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            state_rms = clean_states[t].std(dim=0).pow(2).mean().sqrt().clamp_min(1e-3)
            radius = float(radius_scale) * state_rms
            kick = radius * normal
            tan_part = batch_project_tangent(kick, basis)
            tangent_ratio_sum += float((tan_part.norm(dim=-1) / kick.norm(dim=-1).clamp_min(1e-8)).mean().item())
            kick_budget_sq += float(radius.item() ** 2)
            pert = pert + kick

    pred = model.decode(pert)
    pert_error = decoded_error(model, pred, target, spec.checkpoint.task)
    diff = pert - clean_final
    tan = batch_project_tangent(diff, final_basis)
    norm_part = diff - tan
    denom = math.sqrt(max(kick_budget_sq, 1e-12))
    return {
        "eval_task": spec.eval_task,
        "checkpoint_task": spec.checkpoint.task,
        "dim": spec.checkpoint.dim,
        "label": spec.label,
        "tag": spec.checkpoint.tag,
        "intervention": spec.checkpoint.intervention,
        "seed": seed,
        "horizon": int(args.horizon),
        "kick_count": int(kick_count),
        "normal_radius": float(radius_scale),
        "actual_event_count": len(events),
        "clean_error": ctx["clean_error"],
        "perturbed_error": pert_error,
        "excess_error": pert_error - ctx["clean_error"],
        "hidden_total_gain_per_budget": float(diff.norm(dim=-1).mean().item() / denom),
        "hidden_normal_gain_per_budget": float(norm_part.norm(dim=-1).mean().item() / denom),
        "hidden_tangent_leak_per_budget": float(tan.norm(dim=-1).mean().item() / denom),
        "kick_tangent_fraction": tangent_ratio_sum / max(1, len(events)),
    }


def summarize(rows: list[dict]):
    groups = {}
    for r in rows:
        key = (r["eval_task"], r["label"], r["kick_count"], r["normal_radius"])
        groups.setdefault(key, []).append(r)
    out = []
    for (eval_task, label, count, radius), vals in sorted(groups.items()):
        item = {
            "eval_task": eval_task,
            "label": label,
            "kick_count": count,
            "normal_radius": radius,
            "n": len(vals),
        }
        for k in [
            "clean_error",
            "perturbed_error",
            "excess_error",
            "hidden_total_gain_per_budget",
            "hidden_normal_gain_per_budget",
            "hidden_tangent_leak_per_budget",
            "kick_tangent_fraction",
        ]:
            arr = np.asarray([float(v[k]) for v in vals], dtype=float)
            item[f"{k}_mean"] = float(arr.mean())
            item[f"{k}_std"] = float(arr.std(ddof=0))
        out.append(item)
    return out


def write_markdown(summary: list[dict], path: Path):
    lines = [
        "# Normal-Kick OOD Analysis",
        "",
        "Hidden perturbations are projected away from the local memory tangent before injection.",
        "Lower perturbed error and lower hidden normal gain indicate stronger recovery from off-manifold noise.",
        "",
    ]
    for task in sorted({r["eval_task"] for r in summary}):
        lines.extend([f"## {task}", ""])
        hardest = max(
            [r for r in summary if r["eval_task"] == task],
            key=lambda r: (float(r["normal_radius"]), int(r["kick_count"])),
        )
        count = hardest["kick_count"]
        radius = hardest["normal_radius"]
        rows = [
            r
            for r in summary
            if r["eval_task"] == task and int(r["kick_count"]) == int(count) and float(r["normal_radius"]) == float(radius)
        ]
        lines.append(f"Hardest setting shown: count={count}, radius={radius}.")
        lines.append("")
        lines.append("| model | n | clean | perturbed | excess | hidden normal gain | tangent leak |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in sorted(rows, key=lambda x: x["perturbed_error_mean"]):
            lines.append(
                "| {label} | {n} | {clean:.4g} | {pert:.4g} | {excess:.4g} | {normal:.4g} | {leak:.4g} |".format(
                    label=r["label"],
                    n=r["n"],
                    clean=r["clean_error_mean"],
                    pert=r["perturbed_error_mean"],
                    excess=r["excess_error_mean"],
                    normal=r["hidden_normal_gain_per_budget_mean"],
                    leak=r["hidden_tangent_leak_per_budget_mean"],
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--eval-tasks", default="ring_hold,line_hold,line_integrate")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--kick-counts", default="0,1,5,20,100")
    parser.add_argument("--normal-radii", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--tangent-eps", type=float, default=1e-3)
    parser.add_argument("--z-scale", type=float, default=0.50)
    parser.add_argument("--vel-scale", type=float, default=0.08)
    parser.add_argument("--omega-scale", type=float, default=0.18)
    parser.add_argument("--omega-hold-prob", type=float, default=0.25)
    parser.add_argument("--base-seed", type=int, default=93000)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    seeds = parse_list(args.seeds, int)
    kick_counts = parse_list(args.kick_counts, int)
    normal_radii = parse_list(args.normal_radii, float)
    eval_tasks = set(parse_list(args.eval_tasks, str))
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)

    rows = []
    for spec in default_specs(eval_tasks):
        for seed in seeds:
            try:
                model, _ = load_model(spec.checkpoint, seed, device)
            except FileNotFoundError as exc:
                print(f"[skip] {spec.eval_task} {spec.label} seed={seed}: {exc}")
                continue
            print(f"[eval] {spec.eval_task} {spec.label} seed={seed}", flush=True)
            basis_times = sorted({t for count in kick_counts for t in event_times(args.horizon, count)})
            torch.manual_seed(int(args.base_seed) + 10_000 * seed)
            ctx = prepare_eval_context(model, spec, device, args, basis_times)
            for count in kick_counts:
                for radius in normal_radii:
                    rows.append(evaluate_normal_kicks_fast(model, spec, seed, args, ctx, count, radius))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "normal_kick_ood_raw.csv", rows)
    summary = summarize(rows)
    write_csv(out_dir / "normal_kick_ood_summary.csv", summary)
    write_markdown(summary, out_dir / "NORMAL_KICK_OOD_SUMMARY.md")
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
