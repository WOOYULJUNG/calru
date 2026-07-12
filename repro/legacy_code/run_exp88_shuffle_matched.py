#!/usr/bin/env python3
"""E4: count-matched shuffle control for full CAMN (PAN-RNW-full).

`shuffle_fixed` applies one fixed score permutation for the whole run, so the
persistent support stays as compact as damage-mode RP but lands on
functionally mismatched coordinates. Plain `shuffle` rows are included as the
dense-shuffle comparison. Results go into the writer-sweep directories so the
standard aggregation picks them up.
"""

from __future__ import annotations

import argparse

import run_exp88_manifold_main as base


base.MAIN_TASKS = [
    "ring_hold",
    "ring_integrate",
    "torus_integrate",
]

base.MODELS = [
    (
        "camn_shufmatch_eps3e-5",
        "PAN-RNW-full",
        ["--pan-score-eps", "3e-5", "--pan-score-mode", "shuffle_fixed"],
    ),
    (
        "camn_shuffle_eps3e-5",
        "PAN-RNW-full",
        ["--pan-score-eps", "3e-5", "--pan-score-mode", "shuffle"],
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--gpus", default="3,4")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--analysis-batch", type=int, default=96)
    parser.add_argument("--include-surface", action="store_true")
    parser.add_argument("--out-dir", default="exp88_writer_sweep_results")
    parser.add_argument("--ckpt-dir", default="checkpoints_exp88_writer_sweep")
    parser.add_argument("--trace-dir", default="traces_exp88_writer_sweep")
    parser.add_argument("--log-dir", default="logs_exp88_shuffle_matched")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    jobs = base.build_jobs(seeds, include_surface=False)
    return base.run_jobs(jobs, gpus, args)


if __name__ == "__main__":
    raise SystemExit(main())
