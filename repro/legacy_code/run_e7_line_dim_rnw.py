#!/usr/bin/env python3
"""E7 (plane part): full CAMN (PAN-RNW-full) line/plane integration dim sweep.

Reuses the priority1-4 exp72 command template and directory naming so the new
rows sit next to the existing linear-writer rows. Dims 1..16, eps 1e-4,
3 seeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import run_priority1_4_experiments as pri


DIMS = [1, 2, 4, 8, 16]

MODELS = [
    ("camn_rnw_eps1e-4", "PAN-RNW-full", ["--pan-score-eps", "1e-4"]),
    ("camn_rnw_eps3e-5", "PAN-RNW-full", ["--pan-score-eps", "3e-5"]),
]


def build_jobs(seeds):
    jobs = []
    for dim in DIMS:
        for seed in seeds:
            for tag, model, extra in MODELS:
                dirs = pri.base_dirs("exp72_line_integrate_10k", tag, "line_integrate", dim, include_task=False)
                pri.add_job(jobs, "e7_rnw_line", "line_integrate", dim, model, tag, seed, **dirs, extra=list(extra))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--gpus", default="5")
    parser.add_argument("--log-dir", default="logs_e7_line_dim_rnw")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    jobs = build_jobs(seeds)
    return pri.run_jobs(jobs, gpus, Path(args.log_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
