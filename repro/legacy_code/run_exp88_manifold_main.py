#!/usr/bin/env python3
"""Launch Exp88 manifold hold/integrate main sweep.

The sweep trains standard recurrent baselines and AM-LRU controls on the new
manifold tasks.  Jobs are spread dynamically across GPUs: whenever a GPU
finishes, it takes the next pending job.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "exp88_manifold_attractor_tasks.py"


MAIN_TASKS = [
    "ring_hold",
    "ring_integrate",
    "torus_hold",
    "torus_integrate",
    "complex_curve_hold",
    "complex_curve_integrate",
]

SURFACE_TASKS = [
    "surface_hold",
    "surface_integrate",
]


MODELS = [
    ("rnn", "RNN", []),
    ("gru", "GRU", []),
    ("lstm", "LSTM", []),
    ("ssm", "SSM", []),
    ("lru", "lru-full", []),
    ("am_lru_eps0", "PAN-full", ["--pan-score-eps", "0"]),
    ("am_lru_eps3e-5", "PAN-full", ["--pan-score-eps", "3e-5"]),
    ("am_lru_eps1e-4", "PAN-full", ["--pan-score-eps", "1e-4"]),
    (
        "am_lru_eta0",
        "PAN-full",
        ["--pan-score-eps", "3e-5", "--pan-eta-lambda", "0", "--pan-probe-every", "100000"],
    ),
    (
        "am_lru_allslow",
        "PAN-full",
        [
            "--pan-score-eps",
            "3e-5",
            "--pan-eta-lambda",
            "0",
            "--pan-probe-every",
            "100000",
            "--force-all-slow",
            "--all-slow-lambda",
            "0.999",
        ],
    ),
]


def build_jobs(seeds: list[int], include_surface: bool) -> list[dict]:
    tasks = list(MAIN_TASKS)
    if include_surface:
        tasks.extend(SURFACE_TASKS)
    jobs = []
    for task in tasks:
        for seed in seeds:
            for tag, model, extra in MODELS:
                jobs.append(
                    {
                        "task": task,
                        "seed": int(seed),
                        "tag": tag,
                        "model": model,
                        "extra": list(extra),
                    }
                )
    return jobs


def expected_paths(job: dict, out_dir: Path, ckpt_dir: Path):
    json_path = out_dir / f"{job['task']}_{job['tag']}_seed{job['seed']}.json"
    ckpt_path = ckpt_dir / f"exp88_{job['task']}_{job['tag']}_seed{job['seed']}.pt"
    return json_path, ckpt_path


def command(job: dict, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--task",
        job["task"],
        "--model",
        job["model"],
        "--tag",
        job["tag"],
        "--seed",
        str(job["seed"]),
        "--steps",
        str(args.steps),
        "--batch",
        str(args.batch),
        "--eval-batch",
        str(args.eval_batch),
        "--analysis-batch",
        str(args.analysis_batch),
        "--train-min",
        "80",
        "--train-max",
        "260",
        "--id-horizon",
        "260",
        "--temporal-horizons",
        "500",
        "1000",
        "2000",
        "--post-holds",
        "500",
        "1000",
        "--recovery-steps",
        "0",
        "20",
        "100",
        "500",
        "--normal-radii",
        "0.25",
        "0.5",
        "1.0",
        "--repeated-kicks",
        "1",
        "20",
        "100",
        "--repeated-horizon",
        "500",
        "--velocity-scales",
        "1.0",
        "1.5",
        "2.0",
        "--hold-min",
        "5",
        "--hold-max",
        "20",
        "--move-min",
        "3",
        "--move-max",
        "10",
        "--final-hold-min",
        "20",
        "--final-hold-max",
        "80",
        "--ood-hold-min",
        "30",
        "--ood-hold-max",
        "120",
        "--ood-final-hold-min",
        "100",
        "--ood-final-hold-max",
        "250",
        "--ring-velocity-deg",
        "3.0",
        "--torus-velocity-deg",
        "2.5",
        "--curve-velocity-deg",
        "2.5",
        "--surface-velocity-scale",
        "0.018",
        "--d-model",
        "96",
        "--rec-dim",
        "96",
        "--layers",
        "1",
        "--lr",
        "0.001",
        "--grad-clip",
        "1.0",
        "--plru-tau",
        "0.001",
        "--plru-c",
        "50.0",
        "--rank-matched-lambda-high",
        "0.999",
        "--rank-matched-lambda-low",
        "0.0",
        "--slow-lambda-init-mode",
        "linspace",
        "--slow-lambda-min",
        "0.90",
        "--slow-lambda-max",
        "0.999",
        "--pan-lambda-min",
        "0.90",
        "--pan-lambda-max",
        "0.999",
        "--pan-eta-lambda",
        "3000",
        "--pan-eps-mode",
        "fixed",
        "--pan-score-mode",
        "damage",
        "--pan-warmup-frac",
        "0.3",
        "--pan-probe-every",
        "100",
        "--pan-probe-batch",
        "96",
        "--pan-probe-horizon",
        "260",
        "--pan-h-probe",
        "500",
        "--log-lambda-trajectory",
        "--lambda-log-every",
        "1000",
        "--out-dir",
        args.out_dir,
        "--ckpt-dir",
        args.ckpt_dir,
        "--trace-dir",
        args.trace_dir,
        "--force",
    ]
    cmd.extend(job["extra"])
    return cmd


def run_jobs(jobs: list[dict], gpus: list[int], args: argparse.Namespace) -> int:
    log_dir = ROOT / args.log_dir
    out_dir = ROOT / args.out_dir
    ckpt_dir = ROOT / args.ckpt_dir
    trace_dir = ROOT / args.trace_dir
    for path in (log_dir, out_dir, ckpt_dir, trace_dir):
        path.mkdir(parents=True, exist_ok=True)

    pending = []
    with (log_dir / "manifest.jsonl").open("w") as f:
        for idx, job in enumerate(jobs):
            json_path, ckpt_path = expected_paths(job, out_dir, ckpt_dir)
            exists = json_path.exists() and ckpt_path.exists()
            row = {"idx": idx, "exists": exists, **job}
            f.write(json.dumps(row, sort_keys=True) + "\n")
            if not exists or args.force:
                pending.append((idx, job))

    print(
        f"tasks={len(set(j['task'] for j in jobs))} models={len(MODELS)} "
        f"total={len(jobs)} pending={len(pending)} seeds={args.seeds} gpus={gpus}",
        flush=True,
    )
    print(f"out_dir={args.out_dir} ckpt_dir={args.ckpt_dir} trace_dir={args.trace_dir}", flush=True)
    if args.dry_run:
        for idx, job in pending:
            print(idx, " ".join(command(job, args)))
        return 0

    active: list[tuple[int, int, dict, subprocess.Popen, object]] = []
    free = list(gpus)
    next_job = 0
    failed = 0
    while next_job < len(pending) or active:
        while free and next_job < len(pending):
            idx, job = pending[next_job]
            next_job += 1
            gpu = free.pop(0)
            log_path = log_dir / f"{idx:04d}_{job['task']}_{job['tag']}_seed{job['seed']}_gpu{gpu}.log"
            log_f = log_path.open("w")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(
                f"[launch] idx={idx} gpu={gpu} task={job['task']} tag={job['tag']} seed={job['seed']}",
                flush=True,
            )
            proc = subprocess.Popen(command(job, args), cwd=ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            active.append((idx, gpu, job, proc, log_f))

        time.sleep(float(args.poll_seconds))
        still = []
        for idx, gpu, job, proc, log_f in active:
            code = proc.poll()
            if code is None:
                still.append((idx, gpu, job, proc, log_f))
                continue
            log_f.close()
            free.append(gpu)
            status = "done" if code == 0 else f"failed({code})"
            print(f"[{status}] idx={idx} gpu={gpu} task={job['task']} tag={job['tag']} seed={job['seed']}", flush=True)
            if code != 0:
                failed += 1
        active = still

    if failed:
        print(f"[failed] {failed} jobs failed", flush=True)
        return 1
    print("[done] all jobs completed", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--analysis-batch", type=int, default=96)
    parser.add_argument("--include-surface", action="store_true")
    parser.add_argument("--out-dir", default="exp88_manifold_main_results")
    parser.add_argument("--ckpt-dir", default="checkpoints_exp88_manifold_main")
    parser.add_argument("--trace-dir", default="traces_exp88_manifold_main")
    parser.add_argument("--log-dir", default="logs_exp88_manifold_main")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    jobs = build_jobs(seeds, include_surface=bool(args.include_surface))
    return run_jobs(jobs, gpus, args)


if __name__ == "__main__":
    raise SystemExit(main())
