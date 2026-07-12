#!/usr/bin/env python3
"""E1: surface_hold / surface_integrate rows for the main baseline grid.

Reuses the Exp88 launcher and restricts the task list to the surface tasks,
so it is disjoint from the currently running main-grid queue. Completed jobs
are skipped via the shared out/ckpt directories.
"""

from __future__ import annotations

import argparse

import run_exp88_manifold_main as base


base.MAIN_TASKS = list(base.SURFACE_TASKS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--analysis-batch", type=int, default=96)
    parser.add_argument("--include-surface", action="store_true")
    parser.add_argument("--out-dir", default="exp88_manifold_main_results")
    parser.add_argument("--ckpt-dir", default="checkpoints_exp88_manifold_main")
    parser.add_argument("--trace-dir", default="traces_exp88_manifold_main")
    parser.add_argument("--log-dir", default="logs_exp88_surface_main")
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
