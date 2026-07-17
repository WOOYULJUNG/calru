"""Launch the isolated static-gate smoke and short-pilot campaign.

The launcher intentionally caps concurrency because the main CA-LRU tuning
campaign is CPU-bound and may be running at the same time.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .cache import build_training_cache
from .models import MODEL_IDS


TOPOLOGIES = ("s1", "t2", "s2")


def _job_dir(
    output: Path,
    *,
    phase: str,
    model: str,
    topology: str,
    learning_rate: float,
    seed: int,
) -> Path:
    name = (
        f"{phase}__{model}__{topology}__lr{learning_rate:g}__seed{seed}"
    ).replace(".", "p")
    return output / name


def _run_job(
    *,
    output: Path,
    phase: str,
    model: str,
    topology: str,
    seed: int,
    learning_rate: float,
    updates: int,
    device: str,
    train_cache: Path,
    rp_eta_lambda: float,
) -> dict[str, Any]:
    run_dir = _job_dir(
        output,
        phase=phase,
        model=model,
        topology=topology,
        learning_rate=learning_rate,
        seed=seed,
    )
    if (run_dir / "COMPLETED.json").is_file():
        return {
            "phase": phase,
            "model": model,
            "topology": topology,
            "device": device,
            "returncode": 0,
            "skipped_completed": True,
            "run_dir": str(run_dir),
        }
    command = [
        sys.executable,
        "-m",
        "repro.static_gate_pilot.run",
        "--phase",
        phase,
        "--model",
        model,
        "--topology",
        topology,
        "--seed",
        str(seed),
        "--learning-rate",
        str(learning_rate),
        "--updates",
        str(updates),
        "--report-interval",
        "100",
        "--device",
        device,
        "--output",
        str(output),
        "--train-cache",
        str(train_cache),
        "--rp-eta-lambda",
        str(rp_eta_lambda),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    return {
        "phase": phase,
        "model": model,
        "topology": topology,
        "device": device,
        "returncode": int(completed.returncode),
        "skipped_completed": False,
        "run_dir": str(run_dir),
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-8000:],
    }


def _smoke_passed(run_dir: Path) -> bool:
    result_path = run_dir / "result.json"
    if not result_path.is_file():
        return False
    result = json.loads(result_path.read_text())
    return bool(result.get("finite")) and float(
        result["validation_loss_ratio_final_over_initial"]
    ) < 0.9


def _run_phase(
    *,
    output: Path,
    phase: str,
    cells: list[tuple[str, str]],
    seed: int,
    learning_rate: float,
    updates: int,
    devices: list[str],
    workers: int,
    train_cache: Path,
    rp_eta_lambda: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for index, (model, topology) in enumerate(cells):
            device = devices[index % len(devices)]
            future = executor.submit(
                _run_job,
                output=output,
                phase=phase,
                model=model,
                topology=topology,
                seed=seed,
                learning_rate=learning_rate,
                updates=updates,
                device=device,
                train_cache=train_cache,
                rp_eta_lambda=rp_eta_lambda,
            )
            futures[future] = (model, topology)
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            status = "ok" if record["returncode"] == 0 else "FAILED"
            print(
                f"[{phase}] {record['model']}/{record['topology']} "
                f"{status} on {record['device']}",
                flush=True,
            )
            if record["returncode"] != 0:
                print(record["stderr"], flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "short", "all"), default="all")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--smoke-updates", type=int, default=300)
    parser.add_argument("--short-updates", type=int, default=1500)
    parser.add_argument("--models", default=",".join(MODEL_IDS))
    parser.add_argument("--topologies", default=",".join(TOPOLOGIES))
    parser.add_argument("--rp-eta-lambda", type=float, default=100.0)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_side_pilot_v1"
        ),
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        help="reuse an existing shared train pool instead of creating one under output",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("at least one device is required")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_root = (
        output / f"train_pool_seed{int(args.seed)}"
        if args.train_cache is None
        else args.train_cache
    )
    train_cache = build_training_cache(
        cache_root,
        seed=int(args.seed),
        trajectories=4096,
        horizon=128,
    )
    print(f"[cache] shared train pool ready: {train_cache}", flush=True)
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    topologies = tuple(
        value.strip() for value in args.topologies.split(",") if value.strip()
    )
    unknown_models = set(models) - set(MODEL_IDS)
    unknown_topologies = set(topologies) - set(TOPOLOGIES)
    if unknown_models or unknown_topologies:
        raise ValueError(
            f"unknown models/topologies: {sorted(unknown_models)}, "
            f"{sorted(unknown_topologies)}"
        )
    all_cells = [(model, topology) for model in models for topology in topologies]
    records: list[dict[str, Any]] = []

    if args.phase in {"smoke", "all"}:
        records.extend(
            _run_phase(
                output=output,
                phase="smoke",
                cells=all_cells,
                seed=args.seed,
                learning_rate=args.learning_rate,
                updates=args.smoke_updates,
                devices=devices,
                workers=args.workers,
                train_cache=train_cache,
                rp_eta_lambda=args.rp_eta_lambda,
            )
        )

    short_cells = all_cells
    if args.phase == "all":
        diagnostic_passes = [
            (model, topology)
            for model, topology in all_cells
            if _smoke_passed(
                _job_dir(
                    output,
                    phase="smoke",
                    model=model,
                    topology=topology,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                )
            )
        ]
        print(
            f"[smoke diagnostic] {len(diagnostic_passes)}/{len(all_cells)} cells "
            "met finite + loss-ratio<0.9; running all cells as requested",
            flush=True,
        )
    if args.phase in {"short", "all"} and short_cells:
        records.extend(
            _run_phase(
                output=output,
                phase="short",
                cells=short_cells,
                seed=args.seed,
                learning_rate=args.learning_rate,
                updates=args.short_updates,
                devices=devices,
                workers=args.workers,
                train_cache=train_cache,
                rp_eta_lambda=args.rp_eta_lambda,
            )
        )
    (output / "launcher_records.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "static_gate_side_pilot_v1",
                "phase": args.phase,
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "smoke_updates": args.smoke_updates,
                "short_updates": args.short_updates,
                "devices": devices,
                "workers": args.workers,
                "train_cache": str(train_cache),
                "rp_eta_lambda": args.rp_eta_lambda,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    failures = [record for record in records if record["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} static-gate jobs failed")


if __name__ == "__main__":
    main()
