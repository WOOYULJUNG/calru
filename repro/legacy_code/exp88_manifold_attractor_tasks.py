#!/usr/bin/env python3
"""Exp88: manifold hold/integrate tasks with segment-wise tangent flow.

The task family tests whether recurrent models can hold a point on a manifold,
move along tangent directions under velocity input, stop under zero velocity,
and recover from hidden normal perturbations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from exp71_pan_block_pulse_hold import (
    MODEL_DISPLAY_NAMES,
    apply_slow_lambda_init,
    build_model_variant,
    ensure_dir,
    is_pan_variant,
    normalize_model_variant,
    set_seed,
    slugify,
)
from exp72_structured_attractor_tasks import apply_pan_update_for_task, compute_pan_scores_for_task, pca_stats


TASKS = (
    "ring_hold",
    "ring_integrate",
    "torus_hold",
    "torus_integrate",
    "complex_curve_hold",
    "complex_curve_integrate",
    "surface_hold",
    "surface_integrate",
)


def fixed_rotation(device, dtype, seed: int = 123):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn(3, 3, generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(mat)
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q.to(device=device, dtype=dtype)


@dataclass(frozen=True)
class Geometry:
    name: str
    q_dim: int
    y_dim: int
    angle_velocity: bool

    def phi(self, q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_q0(self, batch: int, device, dtype=torch.float32) -> torch.Tensor:
        raise NotImplementedError

    def wrap_q(self, q: torch.Tensor) -> torch.Tensor:
        return q

    def velocity_features(self, v: torch.Tensor) -> torch.Tensor:
        if self.angle_velocity:
            return torch.cat([v, torch.sin(v), torch.cos(v) - 1.0], dim=-1)
        return v


class RingGeometry(Geometry):
    def __init__(self):
        super().__init__("ring", 1, 2, True)

    def phi(self, q):
        th = q[..., 0]
        return torch.stack([torch.cos(th), torch.sin(th)], dim=-1)

    def sample_q0(self, batch, device, dtype=torch.float32):
        return torch.rand(batch, 1, device=device, dtype=dtype) * (2.0 * math.pi)

    def wrap_q(self, q):
        return torch.remainder(q, 2.0 * math.pi)


class TorusGeometry(Geometry):
    def __init__(self, R=1.0, r=0.35):
        super().__init__("torus", 2, 3, True)
        self.R = float(R)
        self.r = float(r)

    def phi(self, q):
        th, ps = q[..., 0], q[..., 1]
        rho = self.R + self.r * torch.cos(ps)
        return torch.stack([rho * torch.cos(th), rho * torch.sin(th), self.r * torch.sin(ps)], dim=-1)

    def sample_q0(self, batch, device, dtype=torch.float32):
        return torch.rand(batch, 2, device=device, dtype=dtype) * (2.0 * math.pi)

    def wrap_q(self, q):
        return torch.remainder(q, 2.0 * math.pi)


class ComplexCurveGeometry(Geometry):
    def __init__(self, a=0.25, b=0.25, k=3):
        super().__init__("complex_curve", 1, 3, True)
        self.a = float(a)
        self.b = float(b)
        self.k = int(k)

    def phi(self, q):
        s = q[..., 0]
        rho = 1.0 + self.a * torch.cos(self.k * s)
        return torch.stack(
            [rho * torch.cos(s), rho * torch.sin(s), self.b * torch.sin(self.k * s)],
            dim=-1,
        )

    def sample_q0(self, batch, device, dtype=torch.float32):
        return torch.rand(batch, 1, device=device, dtype=dtype) * (2.0 * math.pi)

    def wrap_q(self, q):
        return torch.remainder(q, 2.0 * math.pi)


class SurfaceGeometry(Geometry):
    def __init__(self, a=0.4):
        super().__init__("surface", 2, 3, False)
        self.a = float(a)

    def phi(self, q):
        u, v = q[..., 0], q[..., 1]
        z = self.a * torch.sin(math.pi * u) * torch.sin(math.pi * v)
        raw = torch.stack([u, v, z], dim=-1)
        rot = fixed_rotation(q.device, q.dtype)
        return raw @ rot.T

    def sample_q0(self, batch, device, dtype=torch.float32):
        return torch.empty(batch, 2, device=device, dtype=dtype).uniform_(-0.95, 0.95)

    def wrap_q(self, q):
        return torch.clamp(q, -0.95, 0.95)


GEOMETRIES = {
    "ring": RingGeometry(),
    "torus": TorusGeometry(),
    "complex_curve": ComplexCurveGeometry(),
    "surface": SurfaceGeometry(),
}


def task_geometry(task: str):
    for name, geom in GEOMETRIES.items():
        if task.startswith(name + "_"):
            return geom
    raise ValueError(task)


def is_integrate_task(task: str):
    return task.endswith("_integrate")


def task_io_dims(task: str):
    geom = task_geometry(task)
    vfeat = geom.q_dim * 3 if geom.angle_velocity else geom.q_dim
    return geom.y_dim + vfeat + 1, geom.y_dim


def model_rank_for_task(task: str):
    geom = task_geometry(task)
    if geom.name == "ring":
        return 2
    if geom.name == "complex_curve":
        return 3
    return geom.y_dim


def mode_probabilities(name: str):
    if name == "torus":
        return [("hold", 0.25), ("theta", 0.25), ("psi", 0.25), ("both", 0.25)]
    if name == "surface":
        return [("hold", 0.25), ("u", 0.30), ("v", 0.30), ("both", 0.15)]
    return [("hold", 0.25), ("both", 0.75)]


def velocity_scale(geom: Geometry, args, scale_mult: float = 1.0):
    if geom.name == "ring":
        base = float(args.ring_velocity_deg) * math.pi / 180.0
    elif geom.name == "torus":
        base = float(args.torus_velocity_deg) * math.pi / 180.0
    elif geom.name == "complex_curve":
        base = float(args.curve_velocity_deg) * math.pi / 180.0
    elif geom.name == "surface":
        base = float(args.surface_velocity_scale)
    else:
        raise ValueError(geom.name)
    return base * float(scale_mult)


def choose_mode(probs):
    names, weights = zip(*probs)
    return random.choices(names, weights=weights, k=1)[0]


def make_segment_velocity(
    geom: Geometry,
    horizon: int,
    batch: int,
    device,
    args,
    profile: str = "train",
    scale_mult: float = 1.0,
):
    v = torch.zeros(horizon, batch, geom.q_dim, device=device)
    scale = velocity_scale(geom, args, scale_mult)
    probs = mode_probabilities(geom.name)
    if profile == "temporal_ood":
        hold_len = (int(args.ood_hold_min), int(args.ood_hold_max))
        move_len = (int(args.move_min), int(args.move_max))
        final_hold = (int(args.ood_final_hold_min), int(args.ood_final_hold_max))
    elif profile == "sparse":
        hold_len = (int(args.ood_hold_min), int(args.ood_hold_max))
        move_len = (1, max(1, int(args.move_min)))
        final_hold = (int(args.final_hold_min), int(args.final_hold_max))
    elif profile == "alternating":
        hold_len = (int(args.hold_min), int(args.hold_max))
        move_len = (int(args.move_min), int(args.move_max))
        final_hold = (int(args.final_hold_min), int(args.final_hold_max))
    else:
        hold_len = (int(args.hold_min), int(args.hold_max))
        move_len = (int(args.move_min), int(args.move_max))
        final_hold = (int(args.final_hold_min), int(args.final_hold_max))

    for b in range(batch):
        fh_max = min(max(1, horizon), max(final_hold))
        fh_min = min(fh_max, max(0, min(final_hold)))
        fh = random.randint(fh_min, fh_max) if fh_max > 0 else 0
        active_h = max(0, horizon - fh)
        sign = 1.0
        t = 0
        while t < active_h:
            mode = choose_mode(probs)
            if mode == "hold":
                length = random.randint(max(1, hold_len[0]), max(1, hold_len[1]))
                step = torch.zeros(geom.q_dim, device=device)
            else:
                length = random.randint(max(1, move_len[0]), max(1, move_len[1]))
                step = torch.empty(geom.q_dim, device=device).uniform_(-scale, scale)
                if profile == "alternating":
                    step = step.abs() * sign
                    sign *= -1.0
                if geom.name == "torus":
                    if mode == "theta":
                        step[1] = 0.0
                    elif mode == "psi":
                        step[0] = 0.0
                elif geom.name == "surface":
                    if mode == "u":
                        step[1] = 0.0
                    elif mode == "v":
                        step[0] = 0.0
            end = min(active_h, t + length)
            v[t:end, b] = step
            t = end
    return v


def sequence_from_qv(geom: Geometry, q0: torch.Tensor, v: torch.Tensor):
    horizon, batch, _ = v.shape
    q = torch.zeros(horizon + 1, batch, geom.q_dim, device=q0.device, dtype=q0.dtype)
    q[0] = geom.wrap_q(q0)
    actual_v = torch.zeros_like(v)
    for t in range(horizon):
        nxt = geom.wrap_q(q[t] + v[t])
        actual_v[t] = nxt - q[t]
        if geom.angle_velocity:
            actual_v[t] = torch.atan2(torch.sin(actual_v[t]), torch.cos(actual_v[t]))
        q[t + 1] = nxt
    y = geom.phi(q)
    vfeat = geom.velocity_features(actual_v.reshape(-1, geom.q_dim)).reshape(horizon, batch, -1)
    input_dim = geom.y_dim + vfeat.shape[-1] + 1
    x = torch.zeros(horizon + 1, batch, input_dim, device=q0.device, dtype=q0.dtype)
    x[0, :, : geom.y_dim] = y[0]
    x[0, :, -1] = 1.0
    x[1:, :, geom.y_dim : geom.y_dim + vfeat.shape[-1]] = vfeat
    return x, y, y[-1], {"q0": q0, "q": q, "v": actual_v}


def make_task_batch(task: str, batch: int, horizon: int, device, args, profile: str = "train", scale_mult: float = 1.0):
    geom = task_geometry(task)
    q0 = geom.sample_q0(batch, device)
    if is_integrate_task(task):
        v = make_segment_velocity(geom, horizon, batch, device, args, profile=profile, scale_mult=scale_mult)
    else:
        v = torch.zeros(horizon, batch, geom.q_dim, device=device)
    return sequence_from_qv(geom, q0, v)


@torch.no_grad()
def run_state(model, x_seq):
    state = model.init_state(x_seq.shape[1], x_seq.device)
    for x_t in x_seq:
        state = model.step(x_t, state)
    return state


@torch.no_grad()
def run_states(model, x_seq):
    state = model.init_state(x_seq.shape[1], x_seq.device)
    states = []
    for x_t in x_seq:
        state = model.step(x_t, state)
        states.append(state)
    return torch.stack(states, dim=0)


@torch.no_grad()
def roll_blank(model, state, steps: int):
    blank = torch.zeros(state.shape[0], model.input_dim, device=state.device, dtype=state.dtype)
    cur = state
    for _ in range(int(steps)):
        cur = model.step(blank, cur)
    return cur


def rmse(pred, target):
    return float(torch.sqrt(F.mse_loss(pred, target)).item())


def vec_rmse(pred, target):
    return float(torch.sqrt(((pred - target) ** 2).sum(dim=-1).mean()).item())


@torch.no_grad()
def decoded_metrics(model, state, target, prefix: str):
    pred = model.decode(state)
    return {
        f"{prefix}_rmse": rmse(pred, target),
        f"{prefix}_vec_rmse": vec_rmse(pred, target),
    }


@torch.no_grad()
def state_from_q0(model, geom: Geometry, q0: torch.Tensor, v: torch.Tensor):
    x, _, target, _ = sequence_from_qv(geom, q0, v)
    return run_state(model, x), target


@torch.no_grad()
def local_tangent_basis(model, geom: Geometry, q0: torch.Tensor, v: torch.Tensor, eps: float):
    cols = []
    for j in range(geom.q_dim):
        dq = torch.zeros_like(q0)
        dq[:, j] = float(eps)
        sp, _ = state_from_q0(model, geom, geom.wrap_q(q0 + dq), v)
        sm, _ = state_from_q0(model, geom, geom.wrap_q(q0 - dq), v)
        cols.append((sp - sm) / (2.0 * float(eps)))
    jac = torch.stack(cols, dim=1)
    bases = []
    for i in range(jac.shape[0]):
        q, _ = torch.linalg.qr(jac[i].T, mode="reduced")
        bases.append(q)
    return torch.stack(bases, dim=0)


def batch_project_tangent(diff: torch.Tensor, basis: torch.Tensor):
    coeff = torch.einsum("bsd,bs->bd", basis, diff)
    return torch.einsum("bsd,bd->bs", basis, coeff)


@torch.no_grad()
def eval_final_and_post(model, task: str, horizon: int, args, profile: str, scale_mult: float, prefix: str):
    x, y, target, aux = make_task_batch(task, int(args.eval_batch), horizon, args.device_obj, args, profile, scale_mult)
    out, states = model(x, return_states=True)
    final_state = states[-1]
    metrics = {
        f"{prefix}_final_rmse": rmse(out[-1], target),
        f"{prefix}_final_vec_rmse": vec_rmse(out[-1], target),
        f"{prefix}_nonzero_velocity_fraction": float((aux["v"].norm(dim=-1) > 1e-12).float().mean().item()),
    }
    for post in args.post_holds:
        post_state = roll_blank(model, final_state, int(post))
        metrics.update(decoded_metrics(model, post_state, target, f"{prefix}_postH{post}"))
    if is_integrate_task(task):
        step_err = torch.sqrt(((out[1:] - y[1:]) ** 2).sum(dim=-1))
        moving = aux["v"].norm(dim=-1) > 1e-12
        hold = ~moving
        metrics[f"{prefix}_moving_step_vec_rmse"] = float(step_err[moving].mean().item()) if moving.any() else 0.0
        metrics[f"{prefix}_hold_step_vec_rmse"] = float(step_err[hold].mean().item()) if hold.any() else 0.0
    return metrics, (x, y, target, aux, states)


@torch.no_grad()
def tangent_shift_eval(model, task: str, clean_payload, args, prefix: str):
    geom = task_geometry(task)
    _, _, target, aux, states = clean_payload
    q0, v = aux["q0"], aux["v"]
    direction = torch.randn_like(q0)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    delta = direction * float(args.tangent_delta)
    shifted_state, shifted_target = state_from_q0(model, geom, geom.wrap_q(q0 + delta), v)
    out = {}
    for post in args.recovery_steps:
        pred_state = roll_blank(model, shifted_state, int(post))
        out.update(decoded_metrics(model, pred_state, shifted_target, f"{prefix}_tangentR{post}"))
        out.update(decoded_metrics(model, pred_state, target, f"{prefix}_tangentOldR{post}"))
    return out


@torch.no_grad()
def normal_perturb_eval(model, task: str, clean_payload, args, prefix: str):
    geom = task_geometry(task)
    _, _, target, aux, states = clean_payload
    base = states[-1]
    q0, v = aux["q0"], aux["v"]
    basis = local_tangent_basis(model, geom, q0, v, float(args.tangent_eps))
    clean_post = {int(s): roll_blank(model, base, int(s)) for s in args.recovery_steps}
    state_rms = base.std(dim=0).pow(2).mean().sqrt().clamp_min(1e-3)
    metrics = {}
    for radius_scale in args.normal_radii:
        radius = float(radius_scale) * state_rms
        noise = torch.randn_like(base)
        tangent = batch_project_tangent(noise, basis)
        normal = noise - tangent
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        pert0 = base + radius * normal
        init_dev = float((pert0 - base).norm(dim=-1).mean().item()) + 1e-8
        key = f"{prefix}_normal{radius_scale:g}"
        for post in args.recovery_steps:
            pert = roll_blank(model, pert0, int(post))
            clean = clean_post[int(post)]
            metrics.update(decoded_metrics(model, pert, target, f"{key}_R{post}"))
            metrics[f"{key}_state_dev_ratio_R{post}"] = float((pert - clean).norm(dim=-1).mean().item() / init_dev)

        for count in args.repeated_kicks:
            if int(count) <= 0:
                continue
            pert = base.clone()
            clean = base.clone()
            events = set(np.linspace(1, int(args.repeated_horizon), int(count), dtype=int).tolist())
            kick_budget_sq = 0.0
            blank = torch.zeros(base.shape[0], model.input_dim, device=base.device, dtype=base.dtype)
            for t in range(1, int(args.repeated_horizon) + 1):
                clean = model.step(blank, clean)
                pert = model.step(blank, pert)
                if t in events:
                    noise = torch.randn_like(pert)
                    normal = noise - batch_project_tangent(noise, basis)
                    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    kick = radius * normal
                    kick_budget_sq += float(radius.item() ** 2)
                    pert = pert + kick
            diff = pert - clean
            normal_residue = diff - batch_project_tangent(diff, basis)
            denom = math.sqrt(max(kick_budget_sq, 1e-12))
            rep_prefix = f"{key}_repeated{count}_H{args.repeated_horizon}"
            metrics.update(decoded_metrics(model, pert, target, rep_prefix))
            metrics[f"{rep_prefix}_hidden_normal_gain"] = float(normal_residue.norm(dim=-1).mean().item() / denom)
    return metrics


@torch.no_grad()
def latent_motion_eval(model, task: str, clean_payload, args, prefix: str):
    if not is_integrate_task(task):
        return {}
    geom = task_geometry(task)
    _, _, _, aux, states = clean_payload
    moving = aux["v"].norm(dim=-1) > 1e-12
    if not moving.any():
        return {}
    q0, v = aux["q0"], aux["v"]
    cols = []
    for j in range(geom.q_dim):
        dq = torch.zeros_like(q0)
        dq[:, j] = float(args.tangent_eps)
        xp, _, _, _ = sequence_from_qv(geom, geom.wrap_q(q0 + dq), v)
        xm, _, _, _ = sequence_from_qv(geom, geom.wrap_q(q0 - dq), v)
        sp = run_states(model, xp)
        sm = run_states(model, xm)
        cols.append((sp - sm) / (2.0 * float(args.tangent_eps)))
    jac = torch.stack(cols, dim=-1)
    dh = states[1:] - states[:-1]
    fracs = []
    for t in range(dh.shape[0]):
        idx = moving[t]
        if not bool(idx.any()):
            continue
        for b in torch.where(idx)[0]:
            qbasis, _ = torch.linalg.qr(jac[t, b], mode="reduced")
            delta = dh[t, b]
            proj = qbasis @ (qbasis.T @ delta)
            fracs.append(float(proj.norm().item() / (delta.norm().item() + 1e-8)))
    if not fracs:
        return {}
    arr = np.asarray(fracs, dtype=float)
    return {
        f"{prefix}_latent_tangent_motion_frac_mean": float(arr.mean()),
        f"{prefix}_latent_tangent_motion_frac_min": float(arr.min()),
    }


@torch.no_grad()
def force_all_slow(model, lam: float = 0.999):
    if not hasattr(model, "pan_recs_with_slices"):
        return
    q = torch.tensor(float(lam) ** 2).clamp(1e-8, 1.0 - 1e-8)
    theta = torch.logit(q)
    for rec, _ in model.pan_recs_with_slices():
        if hasattr(rec, "theta"):
            rec.theta.fill_(theta.to(device=rec.theta.device, dtype=rec.theta.dtype))


def evaluate_model(model, task: str, args):
    model.eval()
    metrics = {}
    id_metrics, id_payload = eval_final_and_post(model, task, int(args.id_horizon), args, "id", 1.0, "id")
    metrics.update(id_metrics)
    metrics.update(tangent_shift_eval(model, task, id_payload, args, "id"))
    metrics.update(normal_perturb_eval(model, task, id_payload, args, "id"))
    metrics.update(latent_motion_eval(model, task, id_payload, args, "id"))

    for horizon in args.temporal_horizons:
        temporal_metrics, payload = eval_final_and_post(
            model, task, int(horizon), args, "temporal_ood", 1.0, f"oodT{horizon}"
        )
        metrics.update(temporal_metrics)
        if int(horizon) == max(int(h) for h in args.temporal_horizons):
            metrics.update(normal_perturb_eval(model, task, payload, args, f"oodT{horizon}"))

    if is_integrate_task(task):
        for scale in args.velocity_scales:
            vel_metrics, payload = eval_final_and_post(
                model, task, int(args.id_horizon), args, "id", float(scale), f"vel{scale:g}x"
            )
            metrics.update(vel_metrics)
        for profile in ("sparse", "alternating"):
            vel_metrics, _ = eval_final_and_post(model, task, int(args.id_horizon), args, profile, 1.5, f"vel_{profile}_1p5x")
            metrics.update(vel_metrics)

    x, _, _, _ = make_task_batch(task, int(args.analysis_batch), int(args.id_horizon), args.device_obj, args)
    states = run_states(model, x)
    states_np = states.detach().cpu().numpy().reshape(-1, states.shape[-1])
    metrics.update(pca_stats(states_np))

    lam = model.lam_mag().detach() if hasattr(model, "lam_mag") else torch.empty(0, device=args.device_obj)
    if lam.numel() > 0:
        metrics.update(
            {
                "lambda_sum": float(lam.sum().item()),
                "lambda_max": float(lam.max().item()),
                "lambda_gt_0p9": int((lam > 0.9).sum().item()),
                "lambda_gt_0p95": int((lam > 0.95).sum().item()),
                "lambda_gt_0p99": int((lam > 0.99).sum().item()),
            }
        )
    return metrics


def train_eval_one(args):
    task = str(args.task)
    seed = int(args.seed)
    variant = normalize_model_variant(args.model)
    geom = task_geometry(task)
    rank_for_model = model_rank_for_task(task)

    set_seed(seed)
    ensure_dir(args.out_dir)
    ensure_dir(args.ckpt_dir)
    ensure_dir(args.trace_dir)
    input_dim, output_dim = task_io_dims(task)
    tag = args.tag or slugify(variant)
    json_path = os.path.join(args.out_dir, f"{task}_{tag}_seed{seed}.json")
    ckpt_path = os.path.join(args.ckpt_dir, f"exp88_{task}_{tag}_seed{seed}.pt")
    if os.path.exists(json_path) and os.path.exists(ckpt_path) and not args.force:
        return {"status": "skipped", "path": json_path}

    model = build_model_variant(
        variant=variant,
        input_dim=input_dim,
        output_dim=output_dim,
        rank=rank_for_model,
        d_model=args.d_model,
        rec_dim=args.rec_dim,
        layers=args.layers,
        dropout=args.dropout,
        plru_tau=args.plru_tau,
        plru_c=args.plru_c,
        pan_lambda_min=args.pan_lambda_min,
        pan_lambda_max=args.pan_lambda_max,
        rank_matched_lambda_high=args.rank_matched_lambda_high,
        rank_matched_lambda_low=args.rank_matched_lambda_low,
    ).to(args.device_obj)
    apply_slow_lambda_init(
        model,
        mode=args.slow_lambda_init_mode,
        lambda_min=args.slow_lambda_min,
        lambda_max=args.slow_lambda_max,
        lambda_fixed=args.slow_lambda_fixed,
        lambda_mean=args.slow_lambda_mean,
        lambda_std=args.slow_lambda_std,
        lambda_low_mean=args.slow_lambda_low_mean,
        lambda_high_mean=args.slow_lambda_high_mean,
        lambda_high_prob=args.slow_lambda_high_prob,
        shuffle=args.slow_lambda_shuffle,
        seed=args.slow_lambda_init_seed,
    )
    if args.force_all_slow:
        force_all_slow(model, float(args.all_slow_lambda))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-5)
    warmup_steps = int(round(args.steps * args.pan_warmup_frac)) if is_pan_variant(variant) else 0

    lambda_trace = None
    if args.log_lambda_trajectory and hasattr(model, "lam_mag"):
        lam0 = model.lam_mag().detach().cpu().numpy().astype(np.float32)
        lambda_trace = {
            "steps": [0],
            "lambdas": [lam0],
            "lambda_gt_0p99": [int((lam0 > 0.99).sum())],
            "pan_score_mean": [float("nan")],
            "pan_score_max": [float("nan")],
        }

    losses, task_losses = [], []
    pan_last_score_mean = float("nan")
    pan_last_score_max = float("nan")
    t0 = time.time()
    model.train()
    for step in range(1, int(args.steps) + 1):
        horizon = random.randint(int(args.train_min), int(args.train_max))
        x, y, _, _ = make_task_batch(task, int(args.batch), horizon, args.device_obj, args)
        out = model(x)
        task_loss = F.mse_loss(out, y)
        reg = model.regularization_loss() if hasattr(model, "regularization_loss") else None
        loss = task_loss + (reg if reg is not None else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if args.force_all_slow:
            force_all_slow(model, float(args.all_slow_lambda))

        if is_pan_variant(variant) and not args.force_all_slow and step % int(args.pan_probe_every) == 0:
            model.eval()
            probe_x, _, probe_target, _ = make_task_batch(
                task,
                int(args.pan_probe_batch),
                int(args.pan_probe_horizon),
                args.device_obj,
                args,
                profile="train",
            )
            score_payload, _ = compute_pan_scores_for_task(model, probe_x, probe_target, int(args.pan_h_probe))
            all_scores = torch.cat([scores.detach().flatten() for _, scores in score_payload])
            pan_last_score_mean = float(all_scores.mean().item())
            pan_last_score_max = float(all_scores.max().item())
            if step > warmup_steps:
                apply_pan_update_for_task(score_payload, float(args.pan_eta_lambda), args)
            model.train()

        losses.append(float(loss.item()))
        task_losses.append(float(task_loss.item()))
        if step == 1 or step % max(1, int(args.steps) // 4) == 0:
            lam_msg = ""
            if hasattr(model, "lam_mag"):
                lam = model.lam_mag().detach()
                lam_msg = f" sum_lambda={lam.sum().item():.1f} n>.99={(lam > .99).sum().item()}"
            print(
                f"[{task} {variant:14s} seed={seed}] {step:5d}/{args.steps} "
                f"task={task_loss.item():.5f}{lam_msg}",
                flush=True,
            )
        if lambda_trace is not None and (step == 1 or step == args.steps or step % int(args.lambda_log_every) == 0):
            lam = model.lam_mag().detach().cpu().numpy().astype(np.float32)
            lambda_trace["steps"].append(step)
            lambda_trace["lambdas"].append(lam)
            lambda_trace["lambda_gt_0p99"].append(int((lam > 0.99).sum()))
            lambda_trace["pan_score_mean"].append(float(pan_last_score_mean))
            lambda_trace["pan_score_max"].append(float(pan_last_score_max))

    metrics = evaluate_model(model, task, args)
    result = {
        "task": task,
        "geometry": geom.name,
        "q_dim": geom.q_dim,
        "output_dim": output_dim,
        "input_dim": input_dim,
        "rank_for_model": rank_for_model,
        "model": variant,
        "model_display": MODEL_DISPLAY_NAMES.get(variant, variant),
        "tag": tag,
        "seed": seed,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_steps": int(args.steps),
        "train_min": int(args.train_min),
        "train_max": int(args.train_max),
        "id_horizon": int(args.id_horizon),
        "temporal_horizons": list(map(int, args.temporal_horizons)),
        "post_holds": list(map(int, args.post_holds)),
        "normal_radii": list(map(float, args.normal_radii)),
        "repeated_kicks": list(map(int, args.repeated_kicks)),
        "velocity_scales": list(map(float, args.velocity_scales)),
        "train_loss_final": float(np.mean(losses[-min(50, len(losses)):])),
        "train_task_loss_final": float(np.mean(task_losses[-min(50, len(task_losses)):])),
        "seconds": time.time() - t0,
        "d_model": int(args.d_model),
        "rec_dim": int(args.rec_dim),
        "layers": int(args.layers),
        "lr": float(args.lr),
        "slow_lambda_init_mode": args.slow_lambda_init_mode,
        "slow_lambda_min": float(args.slow_lambda_min),
        "slow_lambda_max": float(args.slow_lambda_max),
        "pan_lambda_min": args.pan_lambda_min if is_pan_variant(variant) else "",
        "pan_lambda_max": args.pan_lambda_max if is_pan_variant(variant) else "",
        "pan_eta_lambda": args.pan_eta_lambda if is_pan_variant(variant) else "",
        "pan_score_eps": args.pan_score_eps if is_pan_variant(variant) else "",
        "pan_eps_mode": args.pan_eps_mode if is_pan_variant(variant) else "",
        "pan_score_mode": args.pan_score_mode if is_pan_variant(variant) else "",
        "force_all_slow": bool(args.force_all_slow),
        "all_slow_lambda": float(args.all_slow_lambda) if args.force_all_slow else "",
        "ring_velocity_deg": float(args.ring_velocity_deg),
        "torus_velocity_deg": float(args.torus_velocity_deg),
        "curve_velocity_deg": float(args.curve_velocity_deg),
        "surface_velocity_scale": float(args.surface_velocity_scale),
    }
    result.update(metrics)

    torch.save(model.state_dict(), ckpt_path)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    if lambda_trace is not None:
        np.savez_compressed(
            os.path.join(args.trace_dir, f"{task}_{tag}_seed{seed}_lambda_trace.npz"),
            steps=np.asarray(lambda_trace["steps"], dtype=np.int64),
            lambdas=np.asarray(lambda_trace["lambdas"], dtype=np.float32),
            lambda_gt_0p99=np.asarray(lambda_trace["lambda_gt_0p99"], dtype=np.int64),
            pan_score_mean=np.asarray(lambda_trace["pan_score_mean"], dtype=np.float32),
            pan_score_max=np.asarray(lambda_trace["pan_score_max"], dtype=np.float32),
        )
    return {"status": "done", "path": json_path}


def aggregate(out_dir, csv_path):
    rows = []
    for path in sorted(Path(out_dir).glob("*.json")):
        rows.append(json.loads(path.read_text()))
    if not rows:
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--analysis-batch", type=int, default=96)
    p.add_argument("--train-min", type=int, default=80)
    p.add_argument("--train-max", type=int, default=260)
    p.add_argument("--id-horizon", type=int, default=260)
    p.add_argument("--temporal-horizons", type=int, nargs="+", default=[500, 1000, 2000])
    p.add_argument("--post-holds", type=int, nargs="+", default=[500, 1000])
    p.add_argument("--recovery-steps", type=int, nargs="+", default=[0, 20, 100, 500])
    p.add_argument("--normal-radii", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--repeated-kicks", type=int, nargs="+", default=[1, 20, 100])
    p.add_argument("--repeated-horizon", type=int, default=500)
    p.add_argument("--velocity-scales", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    p.add_argument("--hold-min", type=int, default=5)
    p.add_argument("--hold-max", type=int, default=20)
    p.add_argument("--move-min", type=int, default=3)
    p.add_argument("--move-max", type=int, default=10)
    p.add_argument("--final-hold-min", type=int, default=20)
    p.add_argument("--final-hold-max", type=int, default=80)
    p.add_argument("--ood-hold-min", type=int, default=30)
    p.add_argument("--ood-hold-max", type=int, default=120)
    p.add_argument("--ood-final-hold-min", type=int, default=100)
    p.add_argument("--ood-final-hold-max", type=int, default=250)
    p.add_argument("--ring-velocity-deg", type=float, default=3.0)
    p.add_argument("--torus-velocity-deg", type=float, default=2.5)
    p.add_argument("--curve-velocity-deg", type=float, default=2.5)
    p.add_argument("--surface-velocity-scale", type=float, default=0.018)
    p.add_argument("--tangent-eps", type=float, default=1e-3)
    p.add_argument("--tangent-delta", type=float, default=0.10)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--rec-dim", type=int, default=96)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--plru-tau", type=float, default=0.001)
    p.add_argument("--plru-c", type=float, default=50.0)
    p.add_argument("--rank-matched-lambda-high", type=float, default=0.999)
    p.add_argument("--rank-matched-lambda-low", type=float, default=0.0)
    p.add_argument("--slow-lambda-init-mode", choices=["default", "linspace", "random", "gaussian", "bimodal", "fixed"], default="linspace")
    p.add_argument("--slow-lambda-min", type=float, default=0.90)
    p.add_argument("--slow-lambda-max", type=float, default=0.999)
    p.add_argument("--slow-lambda-fixed", type=float, default=0.999)
    p.add_argument("--slow-lambda-mean", type=float, default=0.5)
    p.add_argument("--slow-lambda-std", type=float, default=0.2)
    p.add_argument("--slow-lambda-low-mean", type=float, default=0.05)
    p.add_argument("--slow-lambda-high-mean", type=float, default=0.95)
    p.add_argument("--slow-lambda-high-prob", type=float, default=0.5)
    p.add_argument("--slow-lambda-shuffle", action="store_true")
    p.add_argument("--slow-lambda-init-seed", type=int, default=-1)
    p.add_argument("--pan-lambda-min", type=float, default=0.90)
    p.add_argument("--pan-lambda-max", type=float, default=0.999)
    p.add_argument("--pan-eta-lambda", type=float, default=3000.0)
    p.add_argument("--pan-score-eps", type=float, default=0.0)
    p.add_argument("--pan-eps-mode", choices=["fixed", "mean_abs", "median_pos"], default="fixed")
    p.add_argument("--pan-score-mode", choices=["damage", "shuffle", "shuffle_fixed", "random"], default="damage")
    p.add_argument("--pan-warmup-frac", type=float, default=0.3)
    p.add_argument("--pan-probe-every", type=int, default=100)
    p.add_argument("--pan-probe-batch", type=int, default=96)
    p.add_argument("--pan-probe-horizon", type=int, default=260)
    p.add_argument("--pan-h-probe", type=int, default=500)
    p.add_argument("--force-all-slow", action="store_true")
    p.add_argument("--all-slow-lambda", type=float, default=0.999)
    p.add_argument("--log-lambda-trajectory", action="store_true")
    p.add_argument("--lambda-log-every", type=int, default=1000)
    p.add_argument("--gpu", type=int, default=-1)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="exp88_manifold_results")
    p.add_argument("--ckpt-dir", default="checkpoints_exp88_manifold")
    p.add_argument("--trace-dir", default="traces_exp88_manifold")
    p.add_argument("--summary-csv", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.plot_only:
        aggregate(args.out_dir, args.summary_csv or os.path.join(args.out_dir, "exp88_manifold_metrics.csv"))
        return
    if args.gpu >= 0 and args.device == "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.smoke:
        args.steps = min(args.steps, 2)
        args.batch = min(args.batch, 8)
        args.eval_batch = min(args.eval_batch, 8)
        args.analysis_batch = min(args.analysis_batch, 8)
        args.train_min = 8
        args.train_max = 12
        args.id_horizon = 12
        args.temporal_horizons = [16]
        args.post_holds = [4]
        args.recovery_steps = [0, 4]
        args.repeated_kicks = [1, 3]
        args.repeated_horizon = 8
        args.pan_probe_every = 1
        args.pan_probe_batch = 8
        args.pan_probe_horizon = 12
        args.pan_h_probe = 8
    if args.device == "auto":
        args.device_obj = torch.device("cuda:0" if torch.cuda.is_available() and not args.smoke else "cpu")
    else:
        args.device_obj = torch.device(args.device)
    print(train_eval_one(args), flush=True)


if __name__ == "__main__":
    main()
