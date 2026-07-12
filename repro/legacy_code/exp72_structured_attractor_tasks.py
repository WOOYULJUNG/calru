"""Exp72/73: structured attractor tasks on the Exp71 model scaffold.

This file intentionally does not define new model architectures.  It imports
the Exp71 model builder and changes only the task generator and diagnostics:

  line_hold:      z_0 -> hold z_0 in R^d
  line_integrate: z_t = z_0 + sum u_t in R^d
  ring_hold:      theta_0 -> hold (cos theta_0, sin theta_0)
                  with the same zero-drive raw+trig layout as ring_integrate
  noisy_ring_hold:
                  noisy cue of a clean ring state, then hold the clean target
  ring_integrate: theta_t = theta_0 + sum omega_t on S^1
                  with raw+trig zero-drive update channels
  perturbed_ring_integrate:
                  ring integration with nuisance cue-channel perturbations
                  during the driven trajectory
  ring_distractor_recall:
                  cue a ring state, inject later distractor ring cues, recall
                  the original cue
  line_distractor_recall:
                  cue a vector state, inject later distractor vectors, recall
                  the original cue
  line_distractor_stream_recall:
                  cue a vector state, then inject a distractor vector at every
                  timestep for the whole horizon; recall the original cue
  line_sparse_distractor_recall:
                  cue a vector state, then inject randomly timed sparse
                  distractor pulses; recall the original cue
  ring_latent_control:
                  drive a ring state toward a latent target using angular
                  control inputs, then hold
"""

import argparse
import csv
import json
import math
import os
import random
import time

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


TASKS = (
    "line_hold",
    "line_integrate",
    "ring_hold",
    "noisy_ring_hold",
    "ring_integrate",
    "perturbed_ring_integrate",
    "line_distractor_recall",
    "line_distractor_stream_recall",
    "line_sparse_distractor_recall",
    "ring_distractor_recall",
    "ring_latent_control",
)
RING_TASKS = (
    "ring_hold",
    "noisy_ring_hold",
    "ring_integrate",
    "perturbed_ring_integrate",
    "ring_distractor_recall",
    "ring_latent_control",
)
RING_HOLD_TASKS = ("ring_hold", "noisy_ring_hold", "ring_distractor_recall")
RING_INTEGRATE_TASKS = ("ring_integrate", "perturbed_ring_integrate")
RING_MODEL_RANK = 2


def rmse(pred, target):
    return float(torch.sqrt(F.mse_loss(pred, target)).item())


def vec_rmse(pred, target):
    return float(torch.sqrt(((pred - target) ** 2).sum(dim=-1).mean()).item())


def pca_stats(states):
    x = states - states.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    var = s ** 2
    total = float(var.sum()) + 1e-12
    ratio = var / total
    csum = np.cumsum(ratio)
    return {
        "latent_pca_pr": float(total ** 2 / (np.square(var).sum() + 1e-12)),
        "latent_pca_dim90": int(np.searchsorted(csum, 0.90) + 1),
        "latent_pca_dim95": int(np.searchsorted(csum, 0.95) + 1),
        "latent_pca_var1": float(ratio[0]) if ratio.size else 0.0,
        "latent_pca_var2": float(ratio[:2].sum()) if ratio.size >= 2 else float(ratio.sum()),
    }


def angle_diff(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def ring_xy(theta):
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


def angle_from_xy(xy):
    return torch.atan2(xy[:, 1], xy[:, 0])


def add_ring_metrics(metrics, prefix, pred, target):
    pred_theta = angle_from_xy(pred)
    target_theta = angle_from_xy(target)
    err = angle_diff(pred_theta, target_theta)
    radius = pred.norm(dim=-1)
    metrics[f"{prefix}_rmse"] = rmse(pred, target)
    metrics[f"{prefix}_vec_rmse"] = vec_rmse(pred, target)
    metrics[f"{prefix}_angle_mae_deg"] = float(err.abs().mean().item() * 180.0 / math.pi)
    metrics[f"{prefix}_angle_rmse_deg"] = float(torch.sqrt((err ** 2).mean()).item() * 180.0 / math.pi)
    metrics[f"{prefix}_radius_mean"] = float(radius.mean().item())
    metrics[f"{prefix}_radius_rmse"] = float(torch.sqrt(((radius - 1.0) ** 2).mean()).item())


def task_io_dims(task, dim):
    if task == "line_hold":
        return dim + 1, dim
    if task == "line_integrate":
        return 2 * dim + 1, dim
    if task in ("line_distractor_recall", "line_distractor_stream_recall", "line_sparse_distractor_recall"):
        return 2 * dim + 2, dim
    if task in RING_HOLD_TASKS:
        return 6, 2
    if task == "ring_latent_control":
        return 6, 2
    if task in RING_INTEGRATE_TASKS:
        return 6, 2
    raise ValueError(task)


def model_rank_for_task(task, dim):
    return RING_MODEL_RANK if task in RING_TASKS else int(dim)


def make_line_hold_batch(batch, dim, horizon, device, z_scale=0.50):
    z0 = (torch.rand(batch, dim, device=device) * 2.0 - 1.0) * float(z_scale)
    x = torch.zeros(horizon + 1, batch, dim + 1, device=device)
    x[0, :, :dim] = z0
    x[0, :, dim] = 1.0
    y = z0.unsqueeze(0).expand(horizon + 1, batch, dim).clone()
    return x, y, z0, {"z0": z0}


def make_line_integrate_batch(batch, dim, horizon, device, z_scale=0.50, vel_scale=0.08):
    z0 = (torch.rand(batch, dim, device=device) * 2.0 - 1.0) * float(z_scale)
    scale = float(vel_scale) / math.sqrt(max(1, dim))
    vel = (torch.rand(horizon, batch, dim, device=device) * 2.0 - 1.0) * scale
    x = torch.zeros(horizon + 1, batch, 2 * dim + 1, device=device)
    x[0, :, :dim] = z0
    x[0, :, 2 * dim] = 1.0
    x[1:, :, dim : 2 * dim] = vel
    y = torch.cat([z0.unsqueeze(0), z0.unsqueeze(0) + torch.cumsum(vel, dim=0)], dim=0)
    return x, y, y[-1], {"z0": z0, "vel": vel}


def make_ring_hold_batch(batch, horizon, device):
    theta = torch.rand(batch, device=device) * (2.0 * math.pi)
    target = ring_xy(theta)
    x = torch.zeros(horizon + 1, batch, 6, device=device)
    x[0, :, :2] = target
    x[0, :, 5] = 1.0
    y = target.unsqueeze(0).expand(horizon + 1, batch, 2).clone()
    return x, y, target, {"theta0": theta}


def make_noisy_ring_hold_batch(batch, horizon, device, cue_noise_std=0.25):
    theta = torch.rand(batch, device=device) * (2.0 * math.pi)
    target = ring_xy(theta)
    cue = target + torch.randn_like(target) * float(cue_noise_std)
    x = torch.zeros(horizon + 1, batch, 6, device=device)
    x[0, :, :2] = cue
    x[0, :, 5] = 1.0
    y = target.unsqueeze(0).expand(horizon + 1, batch, 2).clone()
    return x, y, target, {"theta0": theta, "cue_noise": cue - target}


def make_ring_integrate_batch(batch, horizon, device, omega_scale=0.18, hold_prob=0.25):
    theta0 = torch.rand(batch, device=device) * (2.0 * math.pi)
    omega = (torch.rand(horizon, batch, device=device) * 2.0 - 1.0) * float(omega_scale)
    if float(hold_prob) > 0.0:
        omega = omega.masked_fill(torch.rand_like(omega) < float(hold_prob), 0.0)
    theta = torch.cat([theta0.unsqueeze(0), theta0.unsqueeze(0) + torch.cumsum(omega, dim=0)], dim=0)
    x = torch.zeros(horizon + 1, batch, 6, device=device)
    x[0, :, :2] = ring_xy(theta0)
    x[0, :, 5] = 1.0
    x[1:, :, 2] = omega
    x[1:, :, 3] = torch.sin(omega)
    x[1:, :, 4] = torch.cos(omega) - 1.0
    y = ring_xy(theta.reshape(-1)).reshape(horizon + 1, batch, 2)
    return x, y, y[-1], {"theta0": theta0, "omega": omega, "theta_final": theta[-1]}


def make_perturbed_ring_integrate_batch(
    batch,
    horizon,
    device,
    omega_scale=0.18,
    hold_prob=0.25,
    nuisance_std=0.20,
    nuisance_prob=1.0,
):
    theta0 = torch.rand(batch, device=device) * (2.0 * math.pi)
    omega = (torch.rand(horizon, batch, device=device) * 2.0 - 1.0) * float(omega_scale)
    if float(hold_prob) > 0.0:
        omega = omega.masked_fill(torch.rand_like(omega) < float(hold_prob), 0.0)
    theta = torch.cat([theta0.unsqueeze(0), theta0.unsqueeze(0) + torch.cumsum(omega, dim=0)], dim=0)
    nuisance = torch.randn(horizon, batch, 2, device=device) * float(nuisance_std)
    if float(nuisance_prob) < 1.0:
        mask = (torch.rand(horizon, batch, 1, device=device) < float(nuisance_prob)).to(nuisance.dtype)
        nuisance = nuisance * mask
    x = torch.zeros(horizon + 1, batch, 6, device=device)
    x[0, :, :2] = ring_xy(theta0)
    x[0, :, 5] = 1.0
    x[1:, :, :2] = nuisance
    x[1:, :, 2] = omega
    x[1:, :, 3] = torch.sin(omega)
    x[1:, :, 4] = torch.cos(omega) - 1.0
    y = ring_xy(theta.reshape(-1)).reshape(horizon + 1, batch, 2)
    return x, y, y[-1], {
        "theta0": theta0,
        "omega": omega,
        "theta_final": theta[-1],
        "nuisance": nuisance,
    }


def distractor_times_for_horizon(horizon, count):
    count = max(0, int(count))
    if count == 0 or horizon <= 1:
        return []
    times = []
    for k in range(count):
        frac = (k + 1) / (count + 1)
        t = int(round(frac * horizon))
        t = max(1, min(int(horizon), t))
        if t not in times:
            times.append(t)
    return times


def make_ring_distractor_recall_batch(batch, horizon, device, distractor_count=2):
    theta = torch.rand(batch, device=device) * (2.0 * math.pi)
    target = ring_xy(theta)
    times = distractor_times_for_horizon(horizon, distractor_count)
    distractor_theta = torch.rand(len(times), batch, device=device) * (2.0 * math.pi)
    x = torch.zeros(horizon + 1, batch, 6, device=device)
    x[0, :, :2] = target
    x[0, :, 5] = 1.0
    for idx, t in enumerate(times):
        x[t, :, :2] = ring_xy(distractor_theta[idx])
        x[t, :, 4] = 1.0
    y = target.unsqueeze(0).expand(horizon + 1, batch, 2).clone()
    return x, y, target, {
        "theta0": theta,
        "distractor_theta": distractor_theta,
        "distractor_times": times,
    }


def make_line_distractor_recall_batch(batch, dim, horizon, device, z_scale=0.50, distractor_count=2):
    z0 = (torch.rand(batch, dim, device=device) * 2.0 - 1.0) * float(z_scale)
    times = distractor_times_for_horizon(horizon, distractor_count)
    distractors = (torch.rand(len(times), batch, dim, device=device) * 2.0 - 1.0) * float(z_scale)
    x = torch.zeros(horizon + 1, batch, 2 * dim + 2, device=device)
    x[0, :, :dim] = z0
    x[0, :, 2 * dim] = 1.0
    for idx, t in enumerate(times):
        x[t, :, dim : 2 * dim] = distractors[idx]
        x[t, :, 2 * dim + 1] = 1.0
    y = z0.unsqueeze(0).expand(horizon + 1, batch, dim).clone()
    return x, y, z0, {
        "z0": z0,
        "distractors": distractors,
        "distractor_times": times,
    }


def make_line_distractor_stream_recall_batch(batch, dim, horizon, device, z_scale=0.50):
    z0 = (torch.rand(batch, dim, device=device) * 2.0 - 1.0) * float(z_scale)
    distractors = (torch.rand(horizon, batch, dim, device=device) * 2.0 - 1.0) * float(z_scale)
    x = torch.zeros(horizon + 1, batch, 2 * dim + 2, device=device)
    x[0, :, :dim] = z0
    x[0, :, 2 * dim] = 1.0
    if horizon > 0:
        x[1:, :, dim : 2 * dim] = distractors
        x[1:, :, 2 * dim + 1] = 1.0
    y = z0.unsqueeze(0).expand(horizon + 1, batch, dim).clone()
    return x, y, z0, {
        "z0": z0,
        "distractors": distractors,
    }


def sparse_distractor_times_for_horizon(horizon, rate=0.05, min_count=1):
    horizon = int(horizon)
    if horizon <= 0:
        return []
    count = int(round(float(rate) * horizon))
    count = max(int(min_count), count)
    count = max(0, min(horizon, count))
    if count == 0:
        return []
    return sorted(random.sample(range(1, horizon + 1), count))


def make_line_sparse_distractor_recall_batch(
    batch,
    dim,
    horizon,
    device,
    z_scale=0.50,
    sparse_rate=0.05,
    sparse_min_count=1,
):
    z0 = (torch.rand(batch, dim, device=device) * 2.0 - 1.0) * float(z_scale)
    times = sparse_distractor_times_for_horizon(horizon, sparse_rate, sparse_min_count)
    distractors = (torch.rand(len(times), batch, dim, device=device) * 2.0 - 1.0) * float(z_scale)
    x = torch.zeros(horizon + 1, batch, 2 * dim + 2, device=device)
    x[0, :, :dim] = z0
    x[0, :, 2 * dim] = 1.0
    for idx, t in enumerate(times):
        x[t, :, dim : 2 * dim] = distractors[idx]
        x[t, :, 2 * dim + 1] = 1.0
    y = z0.unsqueeze(0).expand(horizon + 1, batch, dim).clone()
    return x, y, z0, {
        "z0": z0,
        "distractors": distractors,
        "distractor_times": times,
        "sparse_distractor_rate": float(sparse_rate),
    }


def make_ring_latent_control_batch(
    batch,
    horizon,
    device,
    omega_max=0.18,
    control_gain=0.35,
    control_frac=0.70,
):
    theta0 = torch.rand(batch, device=device) * (2.0 * math.pi)
    delta = (torch.rand(batch, device=device) * 2.0 - 1.0) * math.pi
    theta_goal = theta0 + delta
    control_steps = max(1, min(int(horizon), int(round(float(control_frac) * int(horizon)))))
    omega = torch.zeros(horizon, batch, device=device)
    theta_vals = [theta0]
    theta_cur = theta0
    for t in range(horizon):
        if t < control_steps:
            err = angle_diff(theta_goal, theta_cur)
            omega_t = torch.clamp(float(control_gain) * torch.sin(err), -float(omega_max), float(omega_max))
        else:
            omega_t = torch.zeros_like(theta_cur)
        omega[t] = omega_t
        theta_cur = theta_cur + omega_t
        theta_vals.append(theta_cur)
    theta = torch.stack(theta_vals, dim=0)
    x = torch.zeros(horizon + 1, batch, 6, device=device)
    x[0, :, :2] = ring_xy(theta0)
    x[0, :, 5] = 1.0
    x[1:, :, 2] = omega
    x[1:, :, 3] = torch.sin(omega)
    x[1:, :, 4] = torch.cos(omega) - 1.0
    y = ring_xy(theta.reshape(-1)).reshape(horizon + 1, batch, 2)
    return x, y, y[-1], {
        "theta0": theta0,
        "theta_goal": theta_goal,
        "omega": omega,
        "control_steps": control_steps,
        "theta_final": theta[-1],
    }


def make_task_batch(task, batch, dim, horizon, device, args):
    if task == "line_hold":
        return make_line_hold_batch(batch, dim, horizon, device, args.z_scale)
    if task == "line_integrate":
        return make_line_integrate_batch(batch, dim, horizon, device, args.z_scale, args.vel_scale)
    if task == "line_distractor_recall":
        return make_line_distractor_recall_batch(
            batch,
            dim,
            horizon,
            device,
            args.z_scale,
            args.distractor_count,
        )
    if task == "line_distractor_stream_recall":
        return make_line_distractor_stream_recall_batch(batch, dim, horizon, device, args.z_scale)
    if task == "line_sparse_distractor_recall":
        return make_line_sparse_distractor_recall_batch(
            batch,
            dim,
            horizon,
            device,
            args.z_scale,
            args.sparse_distractor_rate,
            args.sparse_distractor_min_count,
        )
    if task == "ring_hold":
        return make_ring_hold_batch(batch, horizon, device)
    if task == "noisy_ring_hold":
        return make_noisy_ring_hold_batch(batch, horizon, device, args.cue_noise_std)
    if task == "ring_integrate":
        return make_ring_integrate_batch(batch, horizon, device, args.omega_scale, args.omega_hold_prob)
    if task == "perturbed_ring_integrate":
        return make_perturbed_ring_integrate_batch(
            batch,
            horizon,
            device,
            args.omega_scale,
            args.omega_hold_prob,
            args.ring_nuisance_std,
            args.ring_nuisance_prob,
        )
    if task == "ring_distractor_recall":
        return make_ring_distractor_recall_batch(batch, horizon, device, args.distractor_count)
    if task == "ring_latent_control":
        return make_ring_latent_control_batch(
            batch,
            horizon,
            device,
            args.control_omega_max,
            args.control_gain,
            args.control_frac,
        )
    raise ValueError(task)


def append_hold_suffix(x, y, target, hold_steps):
    hold_steps = int(hold_steps)
    if hold_steps <= 0:
        return x, y
    blank = torch.zeros(hold_steps, x.shape[1], x.shape[2], device=x.device, dtype=x.dtype)
    y_hold = target.unsqueeze(0).expand(hold_steps, y.shape[1], y.shape[2]).clone()
    return torch.cat([x, blank], dim=0), torch.cat([y, y_hold], dim=0)


def maybe_append_ring_integrate_train_hold(task, x, y, target, args):
    if task not in RING_INTEGRATE_TASKS:
        return x, y
    hold_max = int(args.ring_integrate_train_hold_max)
    hold_min = int(args.ring_integrate_train_hold_min)
    if hold_max <= 0:
        return x, y
    hold_min = max(0, min(hold_min, hold_max))
    hold_steps = random.randint(hold_min, hold_max)
    return append_hold_suffix(x, y, target, hold_steps)


def make_pan_probe_batch(task, batch, dim, horizon, device, args):
    probe_task = str(args.pan_probe_task)
    if probe_task == "task":
        return make_task_batch(task, batch, dim, horizon, device, args)
    if probe_task == "line_hold":
        z0 = (torch.rand(batch, dim, device=device) * 2.0 - 1.0) * float(args.z_scale)
        input_dim, _ = task_io_dims(task, dim)
        x = torch.zeros(horizon + 1, batch, input_dim, device=device)
        x[0, :, :dim] = z0
        if task == "line_hold":
            x[0, :, dim] = 1.0
        elif task in ("line_integrate",):
            x[0, :, 2 * dim] = 1.0
        elif task in ("line_distractor_recall", "line_distractor_stream_recall", "line_sparse_distractor_recall"):
            x[0, :, 2 * dim] = 1.0
        else:
            raise ValueError(f"line_hold pan probe is incompatible with task: {task}")
        y = z0.unsqueeze(0).expand(horizon + 1, batch, dim).clone()
        return x, y, z0, {"z0": z0}
    if probe_task == "ring_hold":
        return make_ring_hold_batch(batch, horizon, device)
    if probe_task == "blank_hold":
        if task in RING_TASKS:
            return make_ring_hold_batch(batch, horizon, device)
        return make_line_hold_batch(batch, dim, horizon, device, args.z_scale)
    raise ValueError(f"unknown pan probe task: {probe_task}")


@torch.no_grad()
def run_state(model, x_seq):
    state = model.init_state(x_seq.shape[1], x_seq.device)
    for x_t in x_seq:
        state = model.step(x_t, state)
    return state


@torch.no_grad()
def roll_blank(model, state, steps):
    blank = torch.zeros(state.shape[0], model.input_dim, device=state.device)
    cur = state
    for _ in range(int(steps)):
        cur = model.step(blank, cur)
    return cur


@torch.no_grad()
def compute_pan_scores_for_task(model, x_seq, target, h_probe):
    state = run_state(model, x_seq)
    clean_final = roll_blank(model, state, h_probe)
    clean_pred = model.decode(clean_final)
    clean_energy = ((clean_pred - target) ** 2).sum(dim=-1).mean()

    score_payload = []
    batch = state.shape[0]
    total_state = state.shape[1]
    for rec, rec_slice in model.pan_recs_with_slices():
        if hasattr(rec, "ablate_pan_coordinates"):
            ablated = rec.ablate_pan_coordinates(state, rec_slice)
            hidden = ablated.shape[0]
        else:
            hidden = rec_slice.stop - rec_slice.start
            ablated = state.unsqueeze(0).expand(hidden, batch, total_state).clone()
            idx = torch.arange(hidden, device=state.device)
            ablated[idx, :, rec_slice.start + idx] = 0.0
        final = roll_blank(model, ablated.reshape(hidden * batch, total_state), h_probe)
        pred = model.decode(final).reshape(hidden, batch, model.output_dim)
        energy = ((pred - target.unsqueeze(0)) ** 2).sum(dim=-1).mean(dim=1)
        score_payload.append((rec, (energy - clean_energy).detach()))
    return score_payload, float(torch.sqrt(clean_energy / model.output_dim).item())


_FIXED_SHUFFLE_PERMS = {}


@torch.no_grad()
def transform_pan_scores(scores, mode):
    mode = str(mode)
    if mode == "damage":
        return scores
    if mode == "shuffle":
        flat = scores.flatten()
        perm = torch.randperm(flat.numel(), device=flat.device)
        return flat[perm].reshape_as(scores)
    if mode == "shuffle_fixed":
        # Count-matched shuffle control: one permutation fixed for the whole
        # run, so persistent support stays as compact as damage-mode updates
        # but lands on functionally mismatched coordinates.
        flat = scores.flatten()
        key = flat.numel()
        perm = _FIXED_SHUFFLE_PERMS.get(key)
        if perm is None:
            gen = torch.Generator(device="cpu").manual_seed(torch.initial_seed() % (2**31))
            perm = torch.randperm(key, generator=gen)
            _FIXED_SHUFFLE_PERMS[key] = perm
        return flat[perm.to(flat.device)].reshape_as(scores)
    if mode == "random":
        mean = scores.mean()
        std = scores.std(unbiased=False)
        if float(std.item()) < 1e-12:
            return torch.full_like(scores, float(mean.item()))
        return torch.randn_like(scores) * std + mean
    raise ValueError(f"unknown PAN score mode: {mode}")


@torch.no_grad()
def pan_epsilon_for_scores(scores, args):
    mode = str(args.pan_eps_mode)
    base = float(args.pan_score_eps)
    if mode == "fixed":
        return torch.as_tensor(base, dtype=scores.dtype, device=scores.device)
    if mode == "mean_abs":
        return base * scores.abs().mean()
    if mode == "median_pos":
        pos = scores[scores > 0]
        if pos.numel() == 0:
            return torch.zeros((), dtype=scores.dtype, device=scores.device)
        return base * pos.median()
    raise ValueError(f"unknown PAN epsilon mode: {mode}")


@torch.no_grad()
def apply_pan_update_for_task(score_payload, eta_lambda, args):
    for rec, scores in score_payload:
        transformed = transform_pan_scores(scores, args.pan_score_mode)
        eps = pan_epsilon_for_scores(transformed, args)
        rec.update_theta(transformed - eps, eta_lambda)


@torch.no_grad()
def state_from_line_hold(model, z0, horizon):
    batch, dim = z0.shape
    x = torch.zeros(horizon + 1, batch, dim + 1, device=z0.device)
    x[0, :, :dim] = z0
    x[0, :, dim] = 1.0
    state = run_state(model, x)
    return state, z0


@torch.no_grad()
def state_from_line(model, z0, vel):
    horizon, batch, dim = vel.shape
    x = torch.zeros(horizon + 1, batch, 2 * dim + 1, device=z0.device)
    x[0, :, :dim] = z0
    x[0, :, 2 * dim] = 1.0
    x[1:, :, dim : 2 * dim] = vel
    state = run_state(model, x)
    target = z0 + vel.sum(dim=0)
    return state, target


@torch.no_grad()
def state_from_line_distractor_recall(model, z0, distractors, distractor_times, horizon):
    batch, dim = z0.shape
    x = torch.zeros(horizon + 1, batch, 2 * dim + 2, device=z0.device)
    x[0, :, :dim] = z0
    x[0, :, 2 * dim] = 1.0
    for idx, t in enumerate(distractor_times):
        if idx >= distractors.shape[0]:
            break
        if 0 <= int(t) <= horizon:
            x[int(t), :, dim : 2 * dim] = distractors[idx]
            x[int(t), :, 2 * dim + 1] = 1.0
    state = run_state(model, x)
    return state, z0


@torch.no_grad()
def state_from_line_distractor_stream_recall(model, z0, distractors):
    horizon, batch, dim = distractors.shape
    x = torch.zeros(horizon + 1, batch, 2 * dim + 2, device=z0.device)
    x[0, :, :dim] = z0
    x[0, :, 2 * dim] = 1.0
    if horizon > 0:
        x[1:, :, dim : 2 * dim] = distractors
        x[1:, :, 2 * dim + 1] = 1.0
    state = run_state(model, x)
    return state, z0


@torch.no_grad()
def state_from_ring_hold(model, theta, horizon):
    batch = theta.numel()
    x = torch.zeros(horizon + 1, batch, 6, device=theta.device)
    x[0, :, :2] = ring_xy(theta)
    x[0, :, 5] = 1.0
    state = run_state(model, x)
    return state, ring_xy(theta)


@torch.no_grad()
def state_from_noisy_ring_hold(model, theta, cue_noise, horizon):
    batch = theta.numel()
    clean = ring_xy(theta)
    x = torch.zeros(horizon + 1, batch, 6, device=theta.device)
    x[0, :, :2] = clean + cue_noise
    x[0, :, 5] = 1.0
    state = run_state(model, x)
    return state, clean


@torch.no_grad()
def state_from_ring_integrate(model, theta0, omega):
    horizon, batch = omega.shape
    theta = torch.cat([theta0.unsqueeze(0), theta0.unsqueeze(0) + torch.cumsum(omega, dim=0)], dim=0)
    x = torch.zeros(horizon + 1, batch, 6, device=theta0.device)
    x[0, :, :2] = ring_xy(theta0)
    x[0, :, 5] = 1.0
    x[1:, :, 2] = omega
    x[1:, :, 3] = torch.sin(omega)
    x[1:, :, 4] = torch.cos(omega) - 1.0
    state = run_state(model, x)
    return state, ring_xy(theta[-1])


@torch.no_grad()
def state_from_perturbed_ring_integrate(model, theta0, omega, nuisance):
    horizon, batch = omega.shape
    theta = torch.cat([theta0.unsqueeze(0), theta0.unsqueeze(0) + torch.cumsum(omega, dim=0)], dim=0)
    x = torch.zeros(horizon + 1, batch, 6, device=theta0.device)
    x[0, :, :2] = ring_xy(theta0)
    x[0, :, 5] = 1.0
    x[1:, :, :2] = nuisance
    x[1:, :, 2] = omega
    x[1:, :, 3] = torch.sin(omega)
    x[1:, :, 4] = torch.cos(omega) - 1.0
    state = run_state(model, x)
    return state, ring_xy(theta[-1])


@torch.no_grad()
def state_from_ring_distractor_recall(model, theta, distractor_theta, distractor_times, horizon):
    batch = theta.numel()
    x = torch.zeros(horizon + 1, batch, 6, device=theta.device)
    x[0, :, :2] = ring_xy(theta)
    x[0, :, 5] = 1.0
    for idx, t in enumerate(distractor_times):
        if idx >= distractor_theta.shape[0]:
            break
        if 0 <= int(t) <= horizon:
            x[int(t), :, :2] = ring_xy(distractor_theta[idx])
            x[int(t), :, 4] = 1.0
    state = run_state(model, x)
    return state, ring_xy(theta)


@torch.no_grad()
def state_from_ring_latent_control(model, theta0, omega):
    horizon, batch = omega.shape
    theta = torch.cat([theta0.unsqueeze(0), theta0.unsqueeze(0) + torch.cumsum(omega, dim=0)], dim=0)
    x = torch.zeros(horizon + 1, batch, 6, device=theta0.device)
    x[0, :, :2] = ring_xy(theta0)
    x[0, :, 5] = 1.0
    x[1:, :, 2] = omega
    x[1:, :, 3] = torch.sin(omega)
    x[1:, :, 4] = torch.cos(omega) - 1.0
    state = run_state(model, x)
    return state, ring_xy(theta[-1])


@torch.no_grad()
def local_tangent_basis(model, task, dim, aux, horizon, eps=1e-3):
    if task == "line_hold":
        z0 = aux["z0"]
        jac = []
        for j in range(dim):
            dz = torch.zeros_like(z0)
            dz[:, j] = eps
            sp, _ = state_from_line_hold(model, z0 + dz, horizon)
            sm, _ = state_from_line_hold(model, z0 - dz, horizon)
            jac.append((sp - sm) / (2.0 * eps))
        jac = torch.stack(jac, dim=1)
    elif task == "line_integrate":
        z0 = aux["z0"]
        vel = aux["vel"]
        jac = []
        for j in range(dim):
            dz = torch.zeros_like(z0)
            dz[:, j] = eps
            sp, _ = state_from_line(model, z0 + dz, vel)
            sm, _ = state_from_line(model, z0 - dz, vel)
            jac.append((sp - sm) / (2.0 * eps))
        jac = torch.stack(jac, dim=1)
    elif task == "line_distractor_recall":
        z0 = aux["z0"]
        distractors = aux["distractors"]
        distractor_times = aux["distractor_times"]
        jac = []
        for j in range(dim):
            dz = torch.zeros_like(z0)
            dz[:, j] = eps
            sp, _ = state_from_line_distractor_recall(model, z0 + dz, distractors, distractor_times, horizon)
            sm, _ = state_from_line_distractor_recall(model, z0 - dz, distractors, distractor_times, horizon)
            jac.append((sp - sm) / (2.0 * eps))
        jac = torch.stack(jac, dim=1)
    elif task == "line_sparse_distractor_recall":
        z0 = aux["z0"]
        distractors = aux["distractors"]
        distractor_times = aux["distractor_times"]
        jac = []
        for j in range(dim):
            dz = torch.zeros_like(z0)
            dz[:, j] = eps
            sp, _ = state_from_line_distractor_recall(model, z0 + dz, distractors, distractor_times, horizon)
            sm, _ = state_from_line_distractor_recall(model, z0 - dz, distractors, distractor_times, horizon)
            jac.append((sp - sm) / (2.0 * eps))
        jac = torch.stack(jac, dim=1)
    elif task == "line_distractor_stream_recall":
        z0 = aux["z0"]
        distractors = aux["distractors"]
        jac = []
        for j in range(dim):
            dz = torch.zeros_like(z0)
            dz[:, j] = eps
            sp, _ = state_from_line_distractor_stream_recall(model, z0 + dz, distractors)
            sm, _ = state_from_line_distractor_stream_recall(model, z0 - dz, distractors)
            jac.append((sp - sm) / (2.0 * eps))
        jac = torch.stack(jac, dim=1)
    elif task == "ring_hold":
        theta = aux["theta0"]
        sp, _ = state_from_ring_hold(model, theta + eps, horizon)
        sm, _ = state_from_ring_hold(model, theta - eps, horizon)
        jac = ((sp - sm) / (2.0 * eps)).unsqueeze(1)
    elif task == "noisy_ring_hold":
        theta = aux["theta0"]
        cue_noise = aux["cue_noise"]
        sp, _ = state_from_noisy_ring_hold(model, theta + eps, cue_noise, horizon)
        sm, _ = state_from_noisy_ring_hold(model, theta - eps, cue_noise, horizon)
        jac = ((sp - sm) / (2.0 * eps)).unsqueeze(1)
    elif task == "ring_integrate":
        theta = aux["theta0"]
        omega = aux["omega"]
        sp, _ = state_from_ring_integrate(model, theta + eps, omega)
        sm, _ = state_from_ring_integrate(model, theta - eps, omega)
        jac = ((sp - sm) / (2.0 * eps)).unsqueeze(1)
    elif task == "perturbed_ring_integrate":
        theta = aux["theta0"]
        omega = aux["omega"]
        nuisance = aux["nuisance"]
        sp, _ = state_from_perturbed_ring_integrate(model, theta + eps, omega, nuisance)
        sm, _ = state_from_perturbed_ring_integrate(model, theta - eps, omega, nuisance)
        jac = ((sp - sm) / (2.0 * eps)).unsqueeze(1)
    elif task == "ring_distractor_recall":
        theta = aux["theta0"]
        distractor_theta = aux["distractor_theta"]
        distractor_times = aux["distractor_times"]
        sp, _ = state_from_ring_distractor_recall(model, theta + eps, distractor_theta, distractor_times, horizon)
        sm, _ = state_from_ring_distractor_recall(model, theta - eps, distractor_theta, distractor_times, horizon)
        jac = ((sp - sm) / (2.0 * eps)).unsqueeze(1)
    elif task == "ring_latent_control":
        theta = aux["theta0"]
        omega = aux["omega"]
        sp, _ = state_from_ring_latent_control(model, theta + eps, omega)
        sm, _ = state_from_ring_latent_control(model, theta - eps, omega)
        jac = ((sp - sm) / (2.0 * eps)).unsqueeze(1)
    else:
        raise ValueError(task)

    bases = []
    for i in range(jac.shape[0]):
        q, _ = torch.linalg.qr(jac[i].T, mode="reduced")
        bases.append(q)
    return torch.stack(bases, dim=0)


@torch.no_grad()
def tangent_shift_state(model, task, dim, aux, horizon, delta_scale):
    if task == "line_hold":
        z0 = aux["z0"]
        delta = torch.randn_like(z0)
        delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = delta * (float(delta_scale) / math.sqrt(max(1, dim)))
        return state_from_line_hold(model, z0 + delta, horizon)
    if task == "line_integrate":
        z0 = aux["z0"]
        vel = aux["vel"]
        delta = torch.randn_like(z0)
        delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = delta * (float(delta_scale) / math.sqrt(max(1, dim)))
        return state_from_line(model, z0 + delta, vel)
    if task == "line_distractor_recall":
        z0 = aux["z0"]
        delta = torch.randn_like(z0)
        delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = delta * (float(delta_scale) / math.sqrt(max(1, dim)))
        return state_from_line_distractor_recall(
            model,
            z0 + delta,
            aux["distractors"],
            aux["distractor_times"],
            horizon,
        )
    if task == "line_sparse_distractor_recall":
        z0 = aux["z0"]
        delta = torch.randn_like(z0)
        delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = delta * (float(delta_scale) / math.sqrt(max(1, dim)))
        return state_from_line_distractor_recall(
            model,
            z0 + delta,
            aux["distractors"],
            aux["distractor_times"],
            horizon,
        )
    if task == "line_distractor_stream_recall":
        z0 = aux["z0"]
        delta = torch.randn_like(z0)
        delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = delta * (float(delta_scale) / math.sqrt(max(1, dim)))
        return state_from_line_distractor_stream_recall(
            model,
            z0 + delta,
            aux["distractors"],
        )
    if task == "ring_hold":
        theta = aux["theta0"] + float(delta_scale)
        return state_from_ring_hold(model, theta, horizon)
    if task == "noisy_ring_hold":
        theta = aux["theta0"] + float(delta_scale)
        return state_from_noisy_ring_hold(model, theta, aux["cue_noise"], horizon)
    if task == "ring_integrate":
        theta = aux["theta0"] + float(delta_scale)
        return state_from_ring_integrate(model, theta, aux["omega"])
    if task == "perturbed_ring_integrate":
        theta = aux["theta0"] + float(delta_scale)
        return state_from_perturbed_ring_integrate(model, theta, aux["omega"], aux["nuisance"])
    if task == "ring_distractor_recall":
        theta = aux["theta0"] + float(delta_scale)
        return state_from_ring_distractor_recall(
            model,
            theta,
            aux["distractor_theta"],
            aux["distractor_times"],
            horizon,
        )
    if task == "ring_latent_control":
        theta = aux["theta0"] + float(delta_scale)
        return state_from_ring_latent_control(model, theta, aux["omega"])
    raise ValueError(task)


@torch.no_grad()
def evaluate_geometry(model, task, dim, args, device):
    batch = int(args.analysis_batch)
    horizon = int(args.analysis_horizon)
    x, _, target, aux = make_task_batch(task, batch, dim, horizon, device, args)
    states = run_state(model, x)
    pred = model.decode(states)

    metrics = {}
    if task in RING_TASKS:
        add_ring_metrics(metrics, "manifold", pred, target)
    else:
        metrics["manifold_rmse"] = rmse(pred, target)
        metrics["manifold_vec_rmse"] = vec_rmse(pred, target)
    metrics.update(pca_stats(states.detach().cpu().numpy()))

    jac_batch = min(int(args.jacobian_batch), batch)
    sub_aux = {}
    if task == "line_hold":
        sub_aux["z0"] = aux["z0"][:jac_batch]
        target_sub = target[:jac_batch]
    elif task == "line_integrate":
        sub_aux["z0"] = aux["z0"][:jac_batch]
        sub_aux["vel"] = aux["vel"][:, :jac_batch]
        target_sub = target[:jac_batch]
    elif task == "line_distractor_recall":
        sub_aux["z0"] = aux["z0"][:jac_batch]
        sub_aux["distractors"] = aux["distractors"][:, :jac_batch]
        sub_aux["distractor_times"] = aux["distractor_times"]
        target_sub = target[:jac_batch]
    elif task == "line_sparse_distractor_recall":
        sub_aux["z0"] = aux["z0"][:jac_batch]
        sub_aux["distractors"] = aux["distractors"][:, :jac_batch]
        sub_aux["distractor_times"] = aux["distractor_times"]
        target_sub = target[:jac_batch]
    elif task == "line_distractor_stream_recall":
        sub_aux["z0"] = aux["z0"][:jac_batch]
        sub_aux["distractors"] = aux["distractors"][:, :jac_batch]
        target_sub = target[:jac_batch]
    elif task == "ring_hold":
        sub_aux["theta0"] = aux["theta0"][:jac_batch]
        target_sub = target[:jac_batch]
    elif task == "noisy_ring_hold":
        sub_aux["theta0"] = aux["theta0"][:jac_batch]
        sub_aux["cue_noise"] = aux["cue_noise"][:jac_batch]
        target_sub = target[:jac_batch]
    elif task == "ring_integrate":
        sub_aux["theta0"] = aux["theta0"][:jac_batch]
        sub_aux["omega"] = aux["omega"][:, :jac_batch]
        target_sub = target[:jac_batch]
    elif task == "perturbed_ring_integrate":
        sub_aux["theta0"] = aux["theta0"][:jac_batch]
        sub_aux["omega"] = aux["omega"][:, :jac_batch]
        sub_aux["nuisance"] = aux["nuisance"][:, :jac_batch]
        target_sub = target[:jac_batch]
    elif task == "ring_distractor_recall":
        sub_aux["theta0"] = aux["theta0"][:jac_batch]
        sub_aux["distractor_theta"] = aux["distractor_theta"][:, :jac_batch]
        sub_aux["distractor_times"] = aux["distractor_times"]
        target_sub = target[:jac_batch]
    elif task == "ring_latent_control":
        sub_aux["theta0"] = aux["theta0"][:jac_batch]
        sub_aux["omega"] = aux["omega"][:, :jac_batch]
        target_sub = target[:jac_batch]
    else:
        raise ValueError(task)

    base = states[:jac_batch]
    basis = local_tangent_basis(model, task, dim, sub_aux, horizon)
    noise = torch.randn(base.shape, device=device, dtype=base.dtype)
    proj = torch.einsum("bsk,bs->bk", basis, noise)
    tangent = torch.einsum("bsk,bk->bs", basis, proj)
    normal = noise - tangent
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    state_rms = base.std(dim=0).pow(2).mean().sqrt()
    radius = torch.clamp(float(args.normal_radius) * state_rms, min=1e-3)
    pert_normal0 = base + radius * normal
    initial_dev = float((pert_normal0 - base).norm(dim=-1).mean().item()) + 1e-8

    tangent_state, tangent_target = tangent_shift_state(
        model,
        task,
        dim,
        sub_aux,
        horizon,
        args.tangent_delta,
    )
    pert_tangent0 = base + (tangent_state - base)

    for step in args.recovery_steps:
        clean = roll_blank(model, base, step)
        pert = roll_blank(model, pert_normal0, step)
        normal_pred = model.decode(pert)
        metrics[f"normal_state_dev_ratio_R{step}"] = float((pert - clean).norm(dim=-1).mean().item() / initial_dev)
        if task in RING_TASKS:
            add_ring_metrics(metrics, f"normal_R{step}", normal_pred, target_sub)
        else:
            metrics[f"normal_rmse_R{step}"] = rmse(normal_pred, target_sub)
            metrics[f"normal_vec_rmse_R{step}"] = vec_rmse(normal_pred, target_sub)

    for step in args.tangent_steps:
        pert = roll_blank(model, pert_tangent0, step)
        tangent_pred = model.decode(pert)
        if task in RING_TASKS:
            add_ring_metrics(metrics, f"tangent_new_R{step}", tangent_pred, tangent_target)
            add_ring_metrics(metrics, f"tangent_old_R{step}", tangent_pred, target_sub)
        else:
            metrics[f"tangent_new_rmse_R{step}"] = rmse(tangent_pred, tangent_target)
            metrics[f"tangent_old_rmse_R{step}"] = rmse(tangent_pred, target_sub)
            metrics[f"tangent_new_vec_rmse_R{step}"] = vec_rmse(tangent_pred, tangent_target)
    return metrics


@torch.no_grad()
def evaluate_closed_loop_ring_control(model, args, device, prefix="closed_loop"):
    batch = int(args.eval_batch)
    horizon = max(1, max(int(h) for h in args.eval_horizons))
    control_steps = max(1, min(horizon, int(round(float(args.control_frac) * horizon))))
    theta0 = torch.rand(batch, device=device) * (2.0 * math.pi)
    delta = (torch.rand(batch, device=device) * 2.0 - 1.0) * math.pi
    theta_goal = theta0 + delta

    x0 = torch.zeros(batch, 6, device=device)
    x0[:, :2] = ring_xy(theta0)
    x0[:, 5] = 1.0
    state = model.step(x0, model.init_state(batch, device))
    pred = model.decode(state)
    traj_err = []
    for t in range(horizon):
        theta_hat = angle_from_xy(pred)
        if t < control_steps:
            err = angle_diff(theta_goal, theta_hat)
            omega = torch.clamp(
                float(args.control_gain) * torch.sin(err),
                -float(args.control_omega_max),
                float(args.control_omega_max),
            )
        else:
            omega = torch.zeros(batch, device=device)
        x_t = torch.zeros(batch, 6, device=device)
        x_t[:, 2] = omega
        x_t[:, 3] = torch.sin(omega)
        x_t[:, 4] = torch.cos(omega) - 1.0
        state = model.step(x_t, state)
        pred = model.decode(state)
        traj_err.append(angle_diff(angle_from_xy(pred), theta_goal).abs())

    final_pred = pred
    final_err = angle_diff(angle_from_xy(final_pred), theta_goal)
    hold_state = roll_blank(model, state, int(args.post_hold))
    hold_pred = model.decode(hold_state)
    hold_err = angle_diff(angle_from_xy(hold_pred), theta_goal)
    traj = torch.stack(traj_err, dim=0)
    metrics = {
        f"{prefix}_angle_mae_deg": float(final_err.abs().mean().item() * 180.0 / math.pi),
        f"{prefix}_angle_rmse_deg": float(torch.sqrt((final_err ** 2).mean()).item() * 180.0 / math.pi),
        f"{prefix}_postH{args.post_hold}_angle_mae_deg": float(hold_err.abs().mean().item() * 180.0 / math.pi),
        f"{prefix}_postH{args.post_hold}_angle_rmse_deg": float(torch.sqrt((hold_err ** 2).mean()).item() * 180.0 / math.pi),
        f"{prefix}_median_traj_angle_deg": float(traj.median().item() * 180.0 / math.pi),
    }
    return metrics


@torch.no_grad()
def evaluate_model(model, task, dim, args, device):
    model.eval()
    metrics = {}
    for horizon in args.eval_horizons:
        x, y, final_target, _ = make_task_batch(task, int(args.eval_batch), dim, int(horizon), device, args)
        out, states = model(x, return_states=True)
        final_pred = out[-1]
        prefix = (
            f"H{horizon}"
            if task in ("line_hold", "line_distractor_recall", "line_distractor_stream_recall", "line_sparse_distractor_recall") or task in RING_HOLD_TASKS
            else f"T{horizon}"
        )
        if task in RING_TASKS:
            add_ring_metrics(metrics, prefix, final_pred, final_target)
        else:
            metrics[f"rmse_{prefix}"] = rmse(final_pred, final_target)
            metrics[f"vec_rmse_{prefix}"] = vec_rmse(final_pred, final_target)

        post = roll_blank(model, states[-1], int(args.post_hold))
        post_pred = model.decode(post)
        post_prefix = f"{prefix}_postH{args.post_hold}"
        if task in RING_TASKS:
            add_ring_metrics(metrics, post_prefix, post_pred, final_target)
        else:
            metrics[f"rmse_{post_prefix}"] = rmse(post_pred, final_target)
            metrics[f"vec_rmse_{post_prefix}"] = vec_rmse(post_pred, final_target)

    metrics.update(evaluate_geometry(model, task, dim, args, device))
    if task == "ring_latent_control":
        metrics.update(evaluate_closed_loop_ring_control(model, args, device))

    lam = model.lam_mag().detach() if hasattr(model, "lam_mag") else torch.empty(0, device=device)
    if lam.numel() > 0:
        metrics.update({
            "lambda_sum": float(lam.sum().item()),
            "lambda_max": float(lam.max().item()),
            "lambda_gt_0p9": int((lam > 0.9).sum().item()),
            "lambda_gt_0p95": int((lam > 0.95).sum().item()),
            "lambda_gt_0p99": int((lam > 0.99).sum().item()),
        })

    if hasattr(model, "pan_recs_with_slices") and model.pan_recs_with_slices():
        x, _, target, _ = make_pan_probe_batch(
            task,
            int(args.pan_probe_batch),
            dim,
            int(args.pan_probe_horizon),
            device,
            args,
        )
        score_payload, pan_probe_rmse = compute_pan_scores_for_task(model, x, target, int(args.pan_h_probe))
        scores = torch.cat([score.detach().flatten() for _, score in score_payload]).cpu().numpy()
        scores_pos = np.clip(scores, 0.0, None)
        pan_lams = torch.cat([rec.lam_mag().detach().flatten() for rec, _ in score_payload]).cpu().numpy()
        corr = 0.0
        if scores.size and np.std(scores) > 1e-12 and np.std(pan_lams) > 1e-12:
            corr = float(np.corrcoef(scores, pan_lams)[0, 1])
        metrics.update({
            "pan_eval_probe_rmse": pan_probe_rmse,
            "pan_score_mean": float(scores.mean()) if scores.size else 0.0,
            "pan_score_max": float(scores.max()) if scores.size else 0.0,
            "pan_score_pos_sum": float(scores_pos.sum()),
            "pan_score_lambda_corr": corr,
        })
    return metrics


def should_preserve_rank_matched_init(variant):
    return variant in {
        "rank-matched LRU unit",
        "rank-matched LRU-Block",
        "rank-matched real-diag full",
        "rm-real full zero-drive",
        "rm-real full zero-input-ln",
        "rm-real full minimal",
        "rm-real full no-enc-bias",
        "rm-real full no-ln",
        "rm-real full no-glu",
        "rm-real full no-residual",
        "rm-real full rec-decode",
    }


def train_eval_one(args):
    task = str(args.task)
    dim = int(args.dim)
    seed = int(args.seed)
    variant = normalize_model_variant(args.model)
    rank_for_model = model_rank_for_task(task, dim)

    set_seed(seed)
    ensure_dir(args.out_dir)
    ensure_dir(args.ckpt_dir)
    ensure_dir(args.trace_dir)
    input_dim, output_dim = task_io_dims(task, dim)

    tag = args.tag or slugify(variant)
    json_path = os.path.join(args.out_dir, f"{task}_d{dim}_{tag}_seed{seed}.json")
    ckpt_path = os.path.join(args.ckpt_dir, f"exp72_{task}_d{dim}_{tag}_seed{seed}.pt")
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
        pan_rnn_form=args.pan_rnn_form,
        pan_rnn_gamma_init=args.pan_rnn_gamma_init,
        pan_gru_alpha_form=args.pan_gru_alpha_form,
        pan_gru_keep_bias_init=args.pan_gru_keep_bias_init,
        pan_gru_reset_bias_init=args.pan_gru_reset_bias_init,
        pan_lstm_alpha_form=args.pan_lstm_alpha_form,
        pan_lstm_forget_bias_init=args.pan_lstm_forget_bias_init,
        pan_lstm_input_bias_init=args.pan_lstm_input_bias_init,
        pan_lstm_ablation_state=args.pan_lstm_ablation_state,
    ).to(args.device_obj)
    if not should_preserve_rank_matched_init(variant):
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

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-5)
    warmup_steps = int(round(args.steps * args.pan_warmup_frac)) if is_pan_variant(variant) else 0
    plru_warmup = int(round(args.steps * args.plru_warmup_frac)) if variant in ("P-LRU unit", "P-LRU-Block") else 0

    lambda_trace = None
    if args.log_lambda_trajectory and hasattr(model, "lam_mag"):
        lam0 = model.lam_mag().detach().cpu().numpy().astype(np.float32)
        lambda_trace = {
            "steps": [0],
            "lambdas": [lam0],
            "lambda_gt_0p9": [int((lam0 > 0.9).sum())],
            "lambda_gt_0p95": [int((lam0 > 0.95).sum())],
            "lambda_gt_0p99": [int((lam0 > 0.99).sum())],
            "pan_score_mean": [float("nan")],
            "pan_score_max": [float("nan")],
            "pan_probe_rmse": [float("nan")],
        }

    losses, task_losses, reg_losses = [], [], []
    pan_probe_rmse = float("nan")
    pan_last_score_mean = float("nan")
    pan_last_score_max = float("nan")
    t0 = time.time()
    model.train()

    for step in range(1, int(args.steps) + 1):
        horizon = random.randint(int(args.train_min), int(args.train_max))
        x, y, _, _ = make_task_batch(task, int(args.batch), dim, horizon, args.device_obj, args)
        x, y = maybe_append_ring_integrate_train_hold(task, x, y, y[-1], args)
        use_aux_this_step = (
            float(args.aux_blank_weight) > 0.0
            and int(args.aux_blank_every) > 0
            and step % int(args.aux_blank_every) == 0
        )
        need_states = use_aux_this_step
        if need_states:
            out, states = model(x, return_states=True)
        else:
            out = model(x)
            states = None
        task_loss = F.mse_loss(out, y)
        reg = model.regularization_loss() if hasattr(model, "regularization_loss") else None
        reg_value = 0.0
        aux_value = 0.0
        loss = task_loss
        if need_states:
            aux_state = roll_blank(model, states[-1], int(args.aux_blank_horizon))
            aux_pred = model.decode(aux_state)
            aux_loss = F.mse_loss(aux_pred, y[-1])
            loss = loss + float(args.aux_blank_weight) * aux_loss
            aux_value = float(aux_loss.item())
        if reg is not None:
            reg_weight = 1.0
            if variant in ("P-LRU unit", "P-LRU-Block"):
                reg_weight = 1.0 if plru_warmup == 0 else min(1.0, step / plru_warmup)
            loss = loss + reg_weight * reg
            reg_value = float(reg.item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if is_pan_variant(variant) and step % int(args.pan_probe_every) == 0:
            model.eval()
            probe_x, _, probe_target, _ = make_pan_probe_batch(
                task,
                int(args.pan_probe_batch),
                dim,
                int(args.pan_probe_horizon),
                args.device_obj,
                args,
            )
            probe_x, _ = maybe_append_ring_integrate_train_hold(task, probe_x, probe_target.unsqueeze(0), probe_target, args)
            score_payload, pan_probe_rmse = compute_pan_scores_for_task(
                model,
                probe_x,
                probe_target,
                int(args.pan_h_probe),
            )
            all_scores = torch.cat([scores.detach().flatten() for _, scores in score_payload])
            pan_last_score_mean = float(all_scores.mean().item())
            pan_last_score_max = float(all_scores.max().item())
            if step > warmup_steps:
                apply_pan_update_for_task(score_payload, float(args.pan_eta_lambda), args)
            model.train()

        losses.append(float(loss.item()))
        task_losses.append(float(task_loss.item()))
        reg_losses.append(reg_value)

        if step == 1 or step % max(1, int(args.steps) // 4) == 0:
            lam_msg = ""
            if hasattr(model, "lam_mag"):
                lam = model.lam_mag().detach()
                lam_msg = f" sum_lambda={lam.sum().item():.1f} n>.99={(lam > .99).sum().item()}"
            print(
                f"[{task} {variant:22s} d={dim:<2d} seed={seed}] "
                f"{step:5d}/{args.steps} task={task_loss.item():.5f} aux={aux_value:.5f} "
                f"reg={reg_value:.4f}{lam_msg}",
                flush=True,
            )

        if lambda_trace is not None and (step == 1 or step == args.steps or step % int(args.lambda_log_every) == 0):
            lam = model.lam_mag().detach().cpu().numpy().astype(np.float32)
            lambda_trace["steps"].append(step)
            lambda_trace["lambdas"].append(lam)
            lambda_trace["lambda_gt_0p9"].append(int((lam > 0.9).sum()))
            lambda_trace["lambda_gt_0p95"].append(int((lam > 0.95).sum()))
            lambda_trace["lambda_gt_0p99"].append(int((lam > 0.99).sum()))
            lambda_trace["pan_score_mean"].append(float(pan_last_score_mean))
            lambda_trace["pan_score_max"].append(float(pan_last_score_max))
            lambda_trace["pan_probe_rmse"].append(float(pan_probe_rmse))

    metrics = evaluate_model(model, task, dim, args, args.device_obj)
    result = {
        "task": task,
        "dim": dim,
        "rank_for_model": rank_for_model,
        "model": variant,
        "model_display": MODEL_DISPLAY_NAMES.get(variant, variant),
        "tag": tag,
        "seed": seed,
        "device": str(args.device_obj),
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_steps": int(args.steps),
        "train_min": int(args.train_min),
        "train_max": int(args.train_max),
        "ring_integrate_train_hold_min": int(args.ring_integrate_train_hold_min) if task in RING_INTEGRATE_TASKS else "",
        "ring_integrate_train_hold_max": int(args.ring_integrate_train_hold_max) if task in RING_INTEGRATE_TASKS else "",
        "cue_noise_std": float(args.cue_noise_std) if task == "noisy_ring_hold" else "",
        "ring_nuisance_std": float(args.ring_nuisance_std) if task == "perturbed_ring_integrate" else "",
        "ring_nuisance_prob": float(args.ring_nuisance_prob) if task == "perturbed_ring_integrate" else "",
        "distractor_count": int(args.distractor_count) if task in ("line_distractor_recall", "ring_distractor_recall") else "",
        "distractor_stream": bool(task == "line_distractor_stream_recall"),
        "sparse_distractor_rate": float(args.sparse_distractor_rate) if task == "line_sparse_distractor_recall" else "",
        "sparse_distractor_min_count": int(args.sparse_distractor_min_count) if task == "line_sparse_distractor_recall" else "",
        "control_gain": float(args.control_gain) if task == "ring_latent_control" else "",
        "control_omega_max": float(args.control_omega_max) if task == "ring_latent_control" else "",
        "control_frac": float(args.control_frac) if task == "ring_latent_control" else "",
        "train_loss_final": float(np.mean(losses[-min(50, len(losses)):])),
        "train_task_loss_final": float(np.mean(task_losses[-min(50, len(task_losses)):])),
        "train_reg_loss_final": float(np.mean(reg_losses[-min(50, len(reg_losses)):])),
        "seconds": time.time() - t0,
        "d_model": int(args.d_model),
        "rec_dim": int(args.rec_dim),
        "layers": int(args.layers),
        "slow_lambda_init_mode": args.slow_lambda_init_mode if not should_preserve_rank_matched_init(variant) else "rank_matched_preserved",
        "slow_lambda_min": args.slow_lambda_min if not should_preserve_rank_matched_init(variant) else "",
        "slow_lambda_max": args.slow_lambda_max if not should_preserve_rank_matched_init(variant) else "",
        "rank_matched_lambda_high": args.rank_matched_lambda_high if should_preserve_rank_matched_init(variant) else "",
        "rank_matched_lambda_low": args.rank_matched_lambda_low if should_preserve_rank_matched_init(variant) else "",
        "plru_tau": args.plru_tau if variant in ("P-LRU unit", "P-LRU-Block") else "",
        "plru_c": args.plru_c if variant in ("P-LRU unit", "P-LRU-Block") else "",
        "plru_warmup_frac": args.plru_warmup_frac if variant in ("P-LRU unit", "P-LRU-Block") else "",
        "pan_lambda_min": args.pan_lambda_min if is_pan_variant(variant) else "",
        "pan_lambda_max": args.pan_lambda_max if is_pan_variant(variant) else "",
        "pan_eta_lambda": args.pan_eta_lambda if is_pan_variant(variant) else "",
        "pan_score_eps": args.pan_score_eps if is_pan_variant(variant) else "",
        "pan_eps_mode": args.pan_eps_mode if is_pan_variant(variant) else "",
        "pan_score_mode": args.pan_score_mode if is_pan_variant(variant) else "",
        "pan_probe_task": args.pan_probe_task if is_pan_variant(variant) else "",
        "pan_h_probe": args.pan_h_probe if is_pan_variant(variant) else "",
        "pan_rnn_form": args.pan_rnn_form if variant == "RNN-PAN" else "",
        "pan_rnn_gamma_init": args.pan_rnn_gamma_init if variant == "RNN-PAN" and args.pan_rnn_form != "leaky" else "",
        "pan_gru_alpha_form": args.pan_gru_alpha_form if variant == "GRU-PAN" else "",
        "pan_gru_keep_bias_init": args.pan_gru_keep_bias_init if variant == "GRU-PAN" else "",
        "pan_gru_reset_bias_init": args.pan_gru_reset_bias_init if variant == "GRU-PAN" else "",
        "pan_lstm_alpha_form": args.pan_lstm_alpha_form if variant == "LSTM-PAN" else "",
        "pan_lstm_forget_bias_init": args.pan_lstm_forget_bias_init if variant == "LSTM-PAN" else "",
        "pan_lstm_input_bias_init": args.pan_lstm_input_bias_init if variant == "LSTM-PAN" else "",
        "pan_lstm_ablation_state": args.pan_lstm_ablation_state if variant == "LSTM-PAN" else "",
        "aux_blank_weight": float(args.aux_blank_weight),
        "aux_blank_horizon": int(args.aux_blank_horizon),
        "aux_blank_every": int(args.aux_blank_every),
    }
    result.update(metrics)

    torch.save(model.state_dict(), ckpt_path)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    if lambda_trace is not None:
        trace_path = os.path.join(args.trace_dir, f"{task}_d{dim}_{tag}_seed{seed}_lambda_trace.npz")
        np.savez_compressed(
            trace_path,
            steps=np.asarray(lambda_trace["steps"], dtype=np.int64),
            lambdas=np.asarray(lambda_trace["lambdas"], dtype=np.float32),
            lambda_gt_0p9=np.asarray(lambda_trace["lambda_gt_0p9"], dtype=np.int64),
            lambda_gt_0p95=np.asarray(lambda_trace["lambda_gt_0p95"], dtype=np.int64),
            lambda_gt_0p99=np.asarray(lambda_trace["lambda_gt_0p99"], dtype=np.int64),
            pan_score_mean=np.asarray(lambda_trace["pan_score_mean"], dtype=np.float32),
            pan_score_max=np.asarray(lambda_trace["pan_score_max"], dtype=np.float32),
            pan_probe_rmse=np.asarray(lambda_trace["pan_probe_rmse"], dtype=np.float32),
        )
    return {"status": "done", "path": json_path}


def aggregate(out_dir, csv_path):
    rows = []
    for name in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        if not name.endswith(".json"):
            continue
        with open(os.path.join(out_dir, name)) as f:
            rows.append(json.load(f))
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
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=512)
    p.add_argument("--analysis-batch", type=int, default=96)
    p.add_argument("--jacobian-batch", type=int, default=64)
    p.add_argument("--train-min", type=int, default=10)
    p.add_argument("--train-max", type=int, default=50)
    p.add_argument("--eval-horizons", type=int, nargs="+", default=[50, 100, 200, 500, 1000])
    p.add_argument("--analysis-horizon", type=int, default=50)
    p.add_argument("--post-hold", type=int, default=500)
    p.add_argument("--recovery-steps", type=int, nargs="+", default=[0, 20, 100, 500])
    p.add_argument("--tangent-steps", type=int, nargs="+", default=[0, 20, 100, 500])
    p.add_argument("--normal-radius", type=float, default=0.25)
    p.add_argument("--tangent-delta", type=float, default=0.10)
    p.add_argument("--z-scale", type=float, default=0.50)
    p.add_argument("--vel-scale", type=float, default=0.08)
    p.add_argument("--cue-noise-std", type=float, default=0.25)
    p.add_argument("--omega-scale", type=float, default=0.18)
    p.add_argument("--omega-hold-prob", type=float, default=0.25)
    p.add_argument("--ring-nuisance-std", type=float, default=0.20)
    p.add_argument("--ring-nuisance-prob", type=float, default=1.0)
    p.add_argument("--distractor-count", type=int, default=2)
    p.add_argument("--sparse-distractor-rate", type=float, default=0.05)
    p.add_argument("--sparse-distractor-min-count", type=int, default=1)
    p.add_argument("--control-gain", type=float, default=0.35)
    p.add_argument("--control-omega-max", type=float, default=0.18)
    p.add_argument("--control-frac", type=float, default=0.70)
    p.add_argument("--ring-integrate-train-hold-min", type=int, default=0)
    p.add_argument("--ring-integrate-train-hold-max", type=int, default=0)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--rec-dim", type=int, default=96)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--plru-tau", type=float, default=0.001)
    p.add_argument("--plru-c", type=float, default=50.0)
    p.add_argument("--plru-warmup-frac", type=float, default=0.3)
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
    p.add_argument("--pan-probe-task", choices=["task", "line_hold", "ring_hold", "blank_hold"], default="task")
    p.add_argument("--pan-probe-every", type=int, default=100)
    p.add_argument("--pan-probe-batch", type=int, default=96)
    p.add_argument("--pan-probe-horizon", type=int, default=50)
    p.add_argument("--pan-h-probe", type=int, default=500)
    p.add_argument("--pan-rnn-form", choices=["leaky", "decoupled", "decoupled_write"], default="leaky")
    p.add_argument("--pan-rnn-gamma-init", type=float, default=0.3)
    p.add_argument("--pan-gru-alpha-form", choices=["upper_cap", "lower_bound", "gate_bias"], default="upper_cap")
    p.add_argument("--pan-gru-keep-bias-init", type=float, default=0.0)
    p.add_argument("--pan-gru-reset-bias-init", type=float, default=0.0)
    p.add_argument("--pan-lstm-alpha-form", choices=["upper_cap", "lower_bound", "gate_bias"], default="upper_cap")
    p.add_argument("--pan-lstm-forget-bias-init", type=float, default=1.0)
    p.add_argument("--pan-lstm-input-bias-init", type=float, default=0.0)
    p.add_argument("--pan-lstm-ablation-state", choices=["c_only", "h_and_c"], default="c_only")
    p.add_argument("--log-lambda-trajectory", action="store_true")
    p.add_argument("--lambda-log-every", type=int, default=100)
    p.add_argument("--aux-blank-weight", type=float, default=0.0)
    p.add_argument("--aux-blank-horizon", type=int, default=500)
    p.add_argument("--aux-blank-every", type=int, default=100)
    p.add_argument("--gpu", type=int, default=-1)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="exp72_structured_attractor_results")
    p.add_argument("--ckpt-dir", default="checkpoints_exp72_structured_attractor")
    p.add_argument("--trace-dir", default="traces_exp72_structured_attractor")
    p.add_argument("--summary-csv", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.plot_only:
        aggregate(args.out_dir, args.summary_csv or os.path.join(args.out_dir, "exp72_structured_attractor_metrics.csv"))
        return
    if args.gpu >= 0 and args.device == "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.smoke:
        args.steps = min(args.steps, 2)
        args.batch = min(args.batch, 8)
        args.eval_batch = min(args.eval_batch, 8)
        args.analysis_batch = min(args.analysis_batch, 8)
        args.jacobian_batch = min(args.jacobian_batch, 4)
        args.train_min = 2
        args.train_max = 3
        args.eval_horizons = [2]
        args.analysis_horizon = 2
        args.post_hold = 2
        args.recovery_steps = [0, 2]
        args.tangent_steps = [0, 2]
        args.pan_probe_every = 1
        args.pan_probe_batch = 8
        args.pan_probe_horizon = 2
        args.pan_h_probe = 2

    if args.device == "auto":
        args.device_obj = torch.device("cuda:0" if torch.cuda.is_available() and not args.smoke else "cpu")
    else:
        args.device_obj = torch.device(args.device)
    print(train_eval_one(args), flush=True)


if __name__ == "__main__":
    main()
