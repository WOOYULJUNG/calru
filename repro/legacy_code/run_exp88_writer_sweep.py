#!/usr/bin/env python3
"""Launch Exp88 nonlinear-writer AM-LRU sweep.

This reuses the Exp88 launcher settings and changes only the model list.
"""

from __future__ import annotations

import argparse

import run_exp88_manifold_main as base


MODELS = [
    ("am_lru_nw_eps0", "PAN-NW-full", ["--pan-score-eps", "0"]),
    ("am_lru_nw_eps3e-5", "PAN-NW-full", ["--pan-score-eps", "3e-5"]),
    ("am_lru_nw_eps1e-4", "PAN-NW-full", ["--pan-score-eps", "1e-4"]),
    ("am_lru_rnw_eps0", "PAN-RNW-full", ["--pan-score-eps", "0"]),
    ("am_lru_rnw_eps3e-5", "PAN-RNW-full", ["--pan-score-eps", "3e-5"]),
    ("am_lru_rnw_eps1e-4", "PAN-RNW-full", ["--pan-score-eps", "1e-4"]),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--analysis-batch", type=int, default=96)
    parser.add_argument("--include-surface", action="store_true")
    parser.add_argument("--out-dir", default="exp88_writer_sweep_results")
    parser.add_argument("--ckpt-dir", default="checkpoints_exp88_writer_sweep")
    parser.add_argument("--trace-dir", default="traces_exp88_writer_sweep")
    parser.add_argument("--log-dir", default="logs_exp88_writer_sweep")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base.MODELS = MODELS
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    jobs = base.build_jobs(seeds, include_surface=bool(args.include_surface))
    return base.run_jobs(jobs, gpus, args)


if __name__ == "__main__":
    raise SystemExit(main())
