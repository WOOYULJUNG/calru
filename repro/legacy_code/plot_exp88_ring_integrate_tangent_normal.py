#!/usr/bin/env python3
"""Local tangent-normal visualization for ring integration checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp71_pan_block_pulse_hold import build_model_variant, normalize_model_variant
from exp88_manifold_attractor_tasks import (
    RingGeometry,
    batch_project_tangent,
    local_tangent_basis,
    model_rank_for_task,
    roll_blank,
    sequence_from_qv,
    task_io_dims,
)


TAGS = [
    "rnn",
    "gru",
    "lstm",
    "ssm",
    "lru",
    "am_lru_eps0",
    "am_lru_eps3e-5",
    "am_lru_eps1e-4",
    "am_lru_eta0",
    "am_lru_allslow",
]

PRETTY = {
    "rnn": "RNN",
    "gru": "GRU",
    "lstm": "LSTM",
    "ssm": "SSM",
    "lru": "LRU",
    "am_lru_eps0": "AM-LRU eps=0",
    "am_lru_eps3e-5": "AM-LRU eps=3e-5",
    "am_lru_eps1e-4": "AM-LRU eps=1e-4",
    "am_lru_eta0": "AM-LRU eta=0",
    "am_lru_allslow": "AM-LRU all-slow",
}


def load_model(result_path: Path, ckpt_path: Path, device: torch.device):
    result = json.loads(result_path.read_text())
    input_dim, output_dim = task_io_dims("ring_integrate")
    model = build_model_variant(
        variant=normalize_model_variant(result["model"]),
        input_dim=input_dim,
        output_dim=output_dim,
        rank=model_rank_for_task("ring_integrate"),
        d_model=int(result.get("d_model", 96)),
        rec_dim=int(result.get("rec_dim", 96)),
        layers=int(result.get("layers", 1)),
        dropout=0.0,
        plru_tau=float(result.get("plru_tau", 0.001) or 0.001),
        plru_c=float(result.get("plru_c", 50.0) or 50.0),
        pan_lambda_min=float(result.get("pan_lambda_min", 0.90) or 0.90),
        pan_lambda_max=float(result.get("pan_lambda_max", 0.999) or 0.999),
        rank_matched_lambda_high=float(result.get("rank_matched_lambda_high", 0.999) or 0.999),
        rank_matched_lambda_low=float(result.get("rank_matched_lambda_low", 0.0) or 0.0),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model, result


@torch.no_grad()
def run_state(model, x_seq):
    state = model.init_state(x_seq.shape[1], x_seq.device)
    for x_t in x_seq:
        state = model.step(x_t, state)
    return state


@torch.no_grad()
def final_state_from_q(model, geom, q0, horizon: int):
    v = torch.zeros(horizon, q0.shape[0], geom.q_dim, device=q0.device)
    x, _, _, _ = sequence_from_qv(geom, q0, v)
    return run_state(model, x), v


@torch.no_grad()
def roll_blank_steps(model, state, steps):
    blank = torch.zeros(state.shape[0], model.input_dim, device=state.device, dtype=state.dtype)
    cur = state.clone()
    out = {}
    last = 0
    for step in sorted(set(int(s) for s in steps)):
        for _ in range(step - last):
            cur = model.step(blank, cur)
        out[step] = cur.clone()
        last = step
    return out


@torch.no_grad()
def driven_ring_traj(model, geom, base_state, velocity: float, steps: int):
    cur = base_state.clone()
    states = [cur[0].clone()]
    v = torch.tensor([[float(velocity)]], device=cur.device, dtype=cur.dtype)
    vfeat = geom.velocity_features(v)
    for _ in range(int(steps)):
        x = torch.zeros(1, model.input_dim, device=cur.device, dtype=cur.dtype)
        x[:, geom.y_dim : geom.y_dim + vfeat.shape[-1]] = vfeat
        cur = model.step(x, cur)
        states.append(cur[0].clone())
    return torch.stack(states, dim=0)


def local_ring_grid(base_theta: float, span: float, points: int, device):
    d = torch.linspace(-float(span), float(span), int(points), device=device)
    return torch.remainder(torch.tensor([[base_theta]], device=device) + d[:, None], 2.0 * math.pi)


def coords(x, base, tangent_axis, normal_axis):
    diff = x - base[None, :]
    return np.stack([diff @ tangent_axis, diff @ normal_axis], axis=-1)


def payload_for_tag(tag, args, device):
    result_path = Path(args.result_dir) / f"ring_integrate_{tag}_seed{args.seed}.json"
    ckpt_path = Path(args.ckpt_dir) / f"exp88_ring_integrate_{tag}_seed{args.seed}.pt"
    if not result_path.exists() or not ckpt_path.exists():
        return None
    model, result = load_model(result_path, ckpt_path, device)
    geom = RingGeometry()
    base_q = torch.tensor([[args.base_theta]], device=device, dtype=torch.float32)
    base_state, v_zero = final_state_from_q(model, geom, base_q, int(args.horizon))
    basis = local_tangent_basis(model, geom, base_q, v_zero, float(args.tangent_eps))[0]
    dq = torch.full_like(base_q, float(args.tangent_eps))
    sp, _ = final_state_from_q(model, geom, geom.wrap_q(base_q + dq), int(args.horizon))
    sm, _ = final_state_from_q(model, geom, geom.wrap_q(base_q - dq), int(args.horizon))
    tangent_scale = float(((sp - sm) / (2.0 * float(args.tangent_eps))).norm(dim=-1).mean().item())
    tangent_scale = max(tangent_scale, 1e-8)

    q_grid = local_ring_grid(args.base_theta, args.local_span, args.local_points, device)
    clean_state, _ = final_state_from_q(model, geom, q_grid, int(args.horizon))

    gen = torch.Generator(device="cpu").manual_seed(int(args.noise_seed))
    noise = torch.randn(base_state.shape, generator=gen, dtype=base_state.dtype).to(device)
    tangent = batch_project_tangent(noise, basis[None, :, :])
    normal = noise - tangent
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    state_rms = clean_state.std(dim=0).pow(2).mean().sqrt().clamp_min(1e-3)
    perturbed = base_state + float(args.normal_radius) * state_rms * normal
    nstates = roll_blank_steps(model, perturbed, args.recovery_steps)
    ntraj = torch.stack([nstates[int(s)][0] for s in args.recovery_steps], dim=0)
    btraj = driven_ring_traj(model, geom, base_state, args.velocity, args.integration_steps)

    base_np = base_state[0].detach().cpu().numpy()
    clean_np = clean_state.detach().cpu().numpy()
    ntraj_np = ntraj.detach().cpu().numpy()
    btraj_np = btraj.detach().cpu().numpy()
    t_axis = basis[:, 0].detach().cpu().numpy()
    t_axis = t_axis / (np.linalg.norm(t_axis) + 1e-12)
    n_axis = normal[0].detach().cpu().numpy()
    n_axis = n_axis - t_axis * np.dot(n_axis, t_axis)
    n_axis = n_axis / (np.linalg.norm(n_axis) + 1e-12)

    clean_xy = coords(clean_np, base_np, t_axis, n_axis)
    normal_xy = coords(ntraj_np, base_np, t_axis, n_axis)
    blue_xy = coords(btraj_np, base_np, t_axis, n_axis)
    if args.angle_units:
        unit = 180.0 / math.pi / tangent_scale
        clean_xy = clean_xy * unit
        normal_xy = normal_xy * unit
        blue_xy = blue_xy * unit

    return {
        "tag": tag,
        "pretty": PRETTY.get(tag, tag),
        "result": result,
        "clean": clean_xy,
        "normal": normal_xy,
        "blue": blue_xy,
    }


def plot(payloads, args, out_dir):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.2), constrained_layout=True)
    for ax, p in zip(axes.flat, payloads):
        c = p["clean"]
        n = p["normal"]
        b = p["blue"]
        bd = b[-1] - b[0]
        blue_norm_frac = abs(float(bd[1])) / (float(np.linalg.norm(bd)) + 1e-12)
        ax.scatter(c[:, 0], c[:, 1], s=10, c="0.70", alpha=0.42, linewidths=0)
        ax.axhline(0.0, color="#83A980", lw=1.4, alpha=0.75)
        ax.plot(n[:, 0], n[:, 1], "-o", color="#D55E00", lw=2.0, ms=3.3, label="normal recovery")
        ax.plot(b[:, 0], b[:, 1], "-o", color="#0072B2", lw=2.0, ms=3.3, label="tangent integration")
        ax.scatter([0.0], [0.0], marker="*", s=90, c="black", zorder=6)
        ax.scatter([n[0, 0]], [n[0, 1]], marker="x", s=75, c="#D55E00", zorder=6)
        ax.scatter([b[-1, 0]], [b[-1, 1]], marker="*", s=80, c="#0072B2", zorder=6)
        pts = np.concatenate([c, n, b, np.zeros((1, 2))], axis=0)
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        dx, dy = np.maximum(hi - lo, 1e-3)
        if args.square_window:
            center = 0.5 * (lo + hi)
            half = 0.5 * max(float(dx), float(dy)) * (1.0 + 2.0 * float(args.pad))
            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
        else:
            ax.set_xlim(lo[0] - args.pad * dx, hi[0] + args.pad * dx)
            ax.set_ylim(lo[1] - args.pad * dy, hi[1] + args.pad * dy)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{p['pretty']}\n"
            f"T2000={p['result'].get('oodT2000_postH500_vec_rmse', float('nan')):.3g}, "
            f"blueN={blue_norm_frac:.2f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if args.angle_units:
            ax.set_xlabel("local tangent axis (deg-equivalent)", fontsize=8)
            ax.set_ylabel("normal axis (deg-equivalent)", fontsize=8)
        else:
            ax.set_xlabel("local tangent axis", fontsize=8)
            ax.set_ylabel("normal kick axis", fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    path = out_dir / f"ring_integrate_local_tangent_normal_seed{args.seed}.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tags", nargs="+", default=TAGS)
    parser.add_argument("--result-dir", default="exp88_manifold_main_results")
    parser.add_argument("--ckpt-dir", default="checkpoints_exp88_manifold_main")
    parser.add_argument("--out-dir", default="figures_exp88_ring_integrate_tangent_normal")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, default=260)
    parser.add_argument("--base-theta", type=float, default=1.15)
    parser.add_argument("--local-span", type=float, default=1.35)
    parser.add_argument("--local-points", type=int, default=120)
    parser.add_argument("--normal-radius", type=float, default=1.0)
    parser.add_argument("--recovery-steps", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20, 50, 100, 200, 500])
    parser.add_argument("--integration-steps", type=int, default=24)
    parser.add_argument("--velocity", type=float, default=0.05235987755982989)
    parser.add_argument("--tangent-eps", type=float, default=1e-3)
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--pad", type=float, default=0.18)
    parser.add_argument("--square-window", action="store_true")
    parser.add_argument("--angle-units", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    payloads = []
    missing = []
    for tag in args.tags:
        p = payload_for_tag(tag, args, device)
        if p is None:
            missing.append(tag)
            continue
        payloads.append(p)
        print(tag, "loaded")
    if missing:
        print("missing:", ",".join(missing))
    if not payloads:
        raise SystemExit("no payloads")
    print(plot(payloads, args, out_dir))


if __name__ == "__main__":
    main()
