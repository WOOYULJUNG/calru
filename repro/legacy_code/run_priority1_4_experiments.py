"""Launch priority 1-4 continuous-attractor experiments across local GPUs.

This runner intentionally excludes flip-flop. It covers:

1. Seed-averaged line/ring experiments.
2. PAN damage shuffle/random controls and compute-matched auxiliary loss controls.
3. Normalized epsilon policy sweeps.

Each job calls exp72_structured_attractor_tasks.py and writes to its own
result/checkpoint/trace directory. Existing completed jobs are skipped when both
the JSON result and checkpoint are present.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "exp72_structured_attractor_tasks.py"


def slug_num(x):
    return str(x).replace("-", "m").replace(".", "p")


def base_dirs(prefix, tag, task, dim=None, include_task=True):
    if dim is None:
        stem = f"{prefix}_{task}_{tag}" if include_task else f"{prefix}_{tag}"
    else:
        stem = f"{prefix}_{task}_d{dim}_{tag}" if include_task else f"{prefix}_d{dim}_{tag}"
    return {
        "out_dir": f"{stem}_results",
        "ckpt_dir": f"checkpoints_{stem}",
        "trace_dir": f"traces_{stem}",
    }


def expected_paths(job):
    out = ROOT / job["out_dir"] / f"{job['task']}_d{job['dim']}_{job['tag']}_seed{job['seed']}.json"
    ckpt = ROOT / job["ckpt_dir"] / f"exp72_{job['task']}_d{job['dim']}_{job['tag']}_seed{job['seed']}.pt"
    return out, ckpt


def add_job(jobs, category, task, dim, model, tag, seed, out_dir, ckpt_dir, trace_dir, extra=None):
    job = {
        "category": category,
        "task": task,
        "dim": int(dim),
        "model": model,
        "tag": tag,
        "seed": int(seed),
        "out_dir": out_dir,
        "ckpt_dir": ckpt_dir,
        "trace_dir": trace_dir,
        "extra": list(extra or []),
    }
    jobs.append(job)


def common_args(job):
    args = [
        sys.executable,
        str(SCRIPT),
        "--task",
        job["task"],
        "--dim",
        str(job["dim"]),
        "--model",
        job["model"],
        "--tag",
        job["tag"],
        "--seed",
        str(job["seed"]),
        "--steps",
        "10000",
        "--batch",
        "256",
        "--eval-batch",
        "512",
        "--analysis-batch",
        "96",
        "--jacobian-batch",
        "64",
        "--train-min",
        "10",
        "--train-max",
        "50",
        "--eval-horizons",
        "50",
        "100",
        "200",
        "500",
        "1000",
        "2000",
        "--analysis-horizon",
        "50",
        "--post-hold",
        "500",
        "--recovery-steps",
        "0",
        "20",
        "100",
        "500",
        "--tangent-steps",
        "0",
        "20",
        "100",
        "500",
        "--slow-lambda-init-mode",
        "linspace",
        "--slow-lambda-min",
        "0.90",
        "--slow-lambda-max",
        "0.999",
        "--log-lambda-trajectory",
        "--lambda-log-every",
        "1000",
        "--out-dir",
        job["out_dir"],
        "--ckpt-dir",
        job["ckpt_dir"],
        "--trace-dir",
        job["trace_dir"],
    ]
    args.extend(job["extra"])
    return args


def build_jobs(seeds):
    jobs = []

    pan_eps = [
        ("pan_full_eps0", "0"),
        ("pan_full_eps3e-5", "3e-5"),
        ("pan_full_eps1e-4", "1e-4"),
    ]
    baselines = [
        ("gru", "GRU", []),
        ("lstm", "LSTM", []),
        ("lru_full", "lru-full", []),
        ("rank_matched_lru_full", "rm-real-full-zero-drive", []),
        ("plru_full", "p-lru-full", []),
    ]

    # Priority 1: seed averages.
    for dim in [1, 2, 4, 8, 16]:
        for seed in seeds:
            for tag, eps in pan_eps:
                dirs = base_dirs(f"exp72_line_integrate_10k", tag, "line_integrate", dim, include_task=False)
                add_job(jobs, "p1_seed_line", "line_integrate", dim, "PAN-full", tag, seed, **dirs, extra=["--pan-score-eps", eps])
            for tag, model, extra in baselines:
                dirs = base_dirs(f"exp72_line_integrate_10k", tag, "line_integrate", dim, include_task=False)
                add_job(jobs, "p1_seed_line", "line_integrate", dim, model, tag, seed, **dirs, extra=extra)

    for seed in seeds:
        for tag, eps in [("pan_full_eps3e-5", "3e-5"), ("pan_full_eps1e-4", "1e-4")]:
            dirs = base_dirs("exp73_ring_10k", tag, "ring_hold")
            add_job(jobs, "p1_seed_ring_hold", "ring_hold", 1, "PAN-full", tag, seed, **dirs, extra=["--pan-score-eps", eps])
        for tag, model, extra in baselines:
            dirs = base_dirs("exp73_ring_10k", tag, "ring_hold")
            add_job(jobs, "p1_seed_ring_hold", "ring_hold", 1, model, tag, seed, **dirs, extra=extra)

    for seed in seeds:
        for setting, prefix, hold_extra in [
            ("active", "exp73_ring_10k", []),
            ("longhold", "exp73_ring_integrate_longhold_10k", ["--ring-integrate-train-hold-min", "50", "--ring-integrate-train-hold-max", "150"]),
        ]:
            for tag, eps in pan_eps:
                dirs = base_dirs(prefix, tag, "ring_integrate", include_task=(setting != "longhold"))
                add_job(
                    jobs,
                    f"p1_seed_ring_integrate_{setting}",
                    "ring_integrate",
                    1,
                    "PAN-full",
                    tag,
                    seed,
                    **dirs,
                    extra=["--pan-score-eps", eps] + hold_extra,
                )
            for tag, model, extra in baselines:
                dirs = base_dirs(prefix, tag, "ring_integrate", include_task=(setting != "longhold"))
                add_job(
                    jobs,
                    f"p1_seed_ring_integrate_{setting}",
                    "ring_integrate",
                    1,
                    model,
                    tag,
                    seed,
                    **dirs,
                    extra=extra + hold_extra,
                )

    # Priority 2A/B: coordinate-specific damage controls on representative tasks.
    damage_reps = [
        ("line_integrate", 2, "1e-4"),
        ("line_integrate", 8, "3e-5"),
        ("ring_hold", 1, "3e-5"),
        ("ring_integrate", 1, "3e-5"),
    ]
    for seed in seeds:
        for task, dim, eps in damage_reps:
            for mode in ["shuffle", "random"]:
                tag = f"pan_full_eps{eps}_{mode}".replace("eps3e-5", "eps3e-5").replace("eps1e-4", "eps1e-4")
                dirs = base_dirs("exp75_damage_ablation", tag, task, dim)
                add_job(
                    jobs,
                    "p2_damage_control",
                    task,
                    dim,
                    "PAN-full",
                    tag,
                    seed,
                    **dirs,
                    extra=["--pan-score-eps", eps, "--pan-score-mode", mode],
                )

    # Priority 2C: compute-matched auxiliary blank-rollout loss controls.
    aux_models = [
        ("gru_auxH500w1", "GRU", []),
        ("lstm_auxH500w1", "LSTM", []),
        ("lru_full_auxH500w1", "lru-full", []),
        ("plru_full_auxH500w1", "p-lru-full", []),
        ("pan_full_eta0_auxH500w1", "PAN-full", ["--pan-eta-lambda", "0", "--pan-probe-every", "100000"]),
    ]
    aux_extra = ["--aux-blank-weight", "1.0", "--aux-blank-horizon", "500", "--aux-blank-every", "100"]
    for seed in seeds:
        for task, dim, _eps in damage_reps:
            for tag, model, extra in aux_models:
                dirs = base_dirs("exp75_aux_blank", tag, task, dim)
                add_job(jobs, "p2_aux_blank", task, dim, model, tag, seed, **dirs, extra=extra + aux_extra)

    # Priority 3: normalized epsilon policy. Use mean absolute damage scaling.
    for seed in seeds:
        for task, dims in [("line_integrate", [1, 2, 4, 8, 16]), ("ring_hold", [1]), ("ring_integrate", [1])]:
            for dim in dims:
                for c in [0.1, 0.3, 1.0]:
                    tag = f"pan_full_epmean{slug_num(c)}"
                    dirs = base_dirs("exp75_epsilon_policy", tag, task, dim)
                    add_job(
                        jobs,
                        "p3_epsilon_policy",
                        task,
                        dim,
                        "PAN-full",
                        tag,
                        seed,
                        **dirs,
                        extra=["--pan-eps-mode", "mean_abs", "--pan-score-eps", str(c)],
                    )
    return jobs


def run_jobs(jobs, gpus, log_dir, dry_run=False):
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    skipped = []
    for idx, job in enumerate(jobs):
        out, ckpt = expected_paths(job)
        if out.exists() and ckpt.exists():
            skipped.append((idx, job))
        else:
            pending.append((idx, job))

    manifest = log_dir / "manifest.jsonl"
    with manifest.open("w") as f:
        for idx, job in enumerate(jobs):
            out, ckpt = expected_paths(job)
            row = {"idx": idx, "skipped_existing": out.exists() and ckpt.exists(), **job}
            f.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"total jobs: {len(jobs)}")
    print(f"skipped existing: {len(skipped)}")
    print(f"pending: {len(pending)}")
    print(f"manifest: {manifest}")
    if dry_run:
        for idx, job in pending[:20]:
            print(idx, shlex.join(common_args(job)))
        return 0

    active = {}
    done_path = log_dir / "completed.jsonl"
    fail_path = log_dir / "failed.jsonl"
    queue = list(pending)
    failures = 0

    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            idx, job = queue.pop(0)
            args = common_args(job)
            log_path = log_dir / f"{idx:04d}_{job['category']}_{job['task']}_d{job['dim']}_{job['tag']}_seed{job['seed']}_gpu{gpu}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("PYTHONUNBUFFERED", "1")
            log_f = log_path.open("w")
            print(f"[launch gpu{gpu}] job {idx}: {job['category']} {job['task']} d={job['dim']} {job['tag']} seed={job['seed']}")
            proc = subprocess.Popen(args, cwd=str(ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            active[gpu] = (proc, log_f, log_path, idx, job, time.time())

        time.sleep(5)

        for gpu, item in list(active.items()):
            proc, log_f, log_path, idx, job, start = item
            ret = proc.poll()
            if ret is None:
                continue
            log_f.close()
            elapsed = time.time() - start
            row = {"idx": idx, "returncode": ret, "seconds": elapsed, "log": str(log_path), **job}
            if ret == 0:
                with done_path.open("a") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                print(f"[done gpu{gpu}] job {idx} in {elapsed/60:.1f} min")
            else:
                failures += 1
                with fail_path.open("a") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                print(f"[FAILED gpu{gpu}] job {idx} ret={ret}; log={log_path}")
            del active[gpu]

    print(f"all pending jobs finished; failures={failures}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--log-dir", default="logs_priority1_4_continuous")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    jobs = build_jobs(seeds)
    raise SystemExit(run_jobs(jobs, gpus, ROOT / args.log_dir, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
