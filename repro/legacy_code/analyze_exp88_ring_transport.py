#!/usr/bin/env python3
"""Hidden-manifold transport probe for Exp88 ring integration checkpoints."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

import plot_exp88_ring_integrate_tangent_normal as viz
from exp88_manifold_attractor_tasks import RingGeometry, sequence_from_qv


DEFAULT_SPECS = [
    ("exp88_manifold_main_results", "checkpoints_exp88_manifold_main", "gru"),
    ("exp88_manifold_main_results", "checkpoints_exp88_manifold_main", "lru"),
    ("exp88_manifold_main_results", "checkpoints_exp88_manifold_main", "am_lru_eps0"),
    ("exp88_manifold_main_results", "checkpoints_exp88_manifold_main", "am_lru_eps3e-5"),
    ("exp88_manifold_main_results", "checkpoints_exp88_manifold_main", "am_lru_eps1e-4"),
    ("exp88_writer_sweep_results", "checkpoints_exp88_writer_sweep", "am_lru_nw_eps0"),
    ("exp88_writer_sweep_results", "checkpoints_exp88_writer_sweep", "am_lru_nw_eps3e-5"),
    ("exp88_writer_sweep_results", "checkpoints_exp88_writer_sweep", "am_lru_nw_eps1e-4"),
    ("exp88_writer_sweep_results", "checkpoints_exp88_writer_sweep", "am_lru_rnw_eps0"),
    ("exp88_writer_sweep_results", "checkpoints_exp88_writer_sweep", "am_lru_rnw_eps3e-5"),
    ("exp88_writer_sweep_results", "checkpoints_exp88_writer_sweep", "am_lru_rnw_eps1e-4"),
]


def deg(x):
    return float(x) * 180.0 / math.pi


def angle(y):
    return math.atan2(float(y[1]), float(y[0]))


def wrapdiff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def load_if_exists(result_dir: Path, ckpt_dir: Path, tag: str, seed: int, device: torch.device):
    result_path = result_dir / f"ring_integrate_{tag}_seed{seed}.json"
    ckpt_path = ckpt_dir / f"exp88_ring_integrate_{tag}_seed{seed}.pt"
    if not result_path.exists() or not ckpt_path.exists():
        return None, None
    return viz.load_model(result_path, ckpt_path, device)


@torch.no_grad()
def blank_roll(model, state, steps: int):
    blank = torch.zeros(state.shape[0], model.input_dim, device=state.device, dtype=state.dtype)
    cur = state
    for _ in range(int(steps)):
        cur = model.step(blank, cur)
    return cur


@torch.no_grad()
def probe_model(model, tag: str, args, device: torch.device):
    geom = RingGeometry()
    base_theta = float(args.base_theta)
    velocity = float(args.velocity_deg) * math.pi / 180.0
    target_theta = (base_theta + int(args.move_steps) * velocity) % (2.0 * math.pi)
    base_q = torch.tensor([[base_theta]], device=device, dtype=torch.float32)
    target_q = torch.tensor([[target_theta]], device=device, dtype=torch.float32)
    q_grid = torch.linspace(0.0, 2.0 * math.pi, int(args.grid) + 1, device=device)[:-1, None]

    clean, _ = viz.final_state_from_q(model, geom, q_grid, int(args.clean_horizon))
    target_clean, _ = viz.final_state_from_q(model, geom, target_q, int(args.clean_horizon))

    v = torch.zeros(int(args.move_steps), 1, 1, device=device)
    v[:, 0, 0] = velocity
    x, _, _, _ = sequence_from_qv(geom, base_q, v)
    state = model.init_state(1, device)
    states = []
    for x_t in x:
        state = model.step(x_t, state)
        states.append(state.clone())
    drive_state = states[-1]
    base_angle = angle(model.decode(states[0])[0])

    rows = []
    for blank_steps in args.blank_steps:
        st = blank_roll(model, drive_state, int(blank_steps))
        pred = model.decode(st)[0]
        d = torch.linalg.norm(clean - st, dim=-1)
        idx = int(torch.argmin(d).item())
        nearest_q = float(q_grid[idx, 0].item())
        decoded_delta = deg(wrapdiff(angle(pred), base_angle))
        nearest_delta = deg(wrapdiff(nearest_q, base_theta))
        target_delta = deg(wrapdiff(target_theta, base_theta))
        rows.append(
            {
                "tag": tag,
                "move_steps": int(args.move_steps),
                "velocity_deg": float(args.velocity_deg),
                "target_delta_deg": target_delta,
                "blank_steps": int(blank_steps),
                "decoded_delta_deg": decoded_delta,
                "nearest_clean_delta_deg": nearest_delta,
                "nearest_angle_error_deg": abs(deg(wrapdiff(nearest_q, target_theta))),
                "dist_to_nearest_clean": float(d[idx].item()),
                "dist_to_target_clean": float(torch.linalg.norm(st - target_clean, dim=-1).item()),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-theta", type=float, default=1.15)
    parser.add_argument("--velocity-deg", type=float, default=3.0)
    parser.add_argument("--move-steps", type=int, default=5)
    parser.add_argument("--clean-horizon", type=int, default=260)
    parser.add_argument("--grid", type=int, default=1440)
    parser.add_argument("--blank-steps", type=int, nargs="+", default=[0, 1, 5, 20, 100, 500, 1000, 2000])
    parser.add_argument("--out", default="analysis_exp88_ring_transport/ring_transport_seed0.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    rows = []
    for result_dir, ckpt_dir, tag in DEFAULT_SPECS:
        model, _ = load_if_exists(Path(result_dir), Path(ckpt_dir), tag, int(args.seed), device)
        if model is None:
            continue
        print(f"loaded {tag}", flush=True)
        rows.extend(probe_model(model, tag, args, device))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = sorted(set().union(*(row.keys() for row in rows)))
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
