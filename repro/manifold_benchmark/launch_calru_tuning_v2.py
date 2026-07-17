"""Restartable validation-only CA-LRU topology tuning v2.

The campaign preserves the ring-selected width and RP threshold, searches a
small neighborhood on S1/T2, and broadens LR/RP dose on S2. It never opens a
test bank. Screening uses one development seed; finalists are then evaluated
on two additional development seeds before a frozen final selection is
written.
"""

from __future__ import annotations

import argparse
from itertools import product
import math
from pathlib import Path
from statistics import median
import subprocess
import sys
from typing import Any

from repro.sagodi_protocol.artifacts import atomic_json, strict_json_load

from .launch_topology_hparam import _base_job, run_phase
from .run_topology_hparam import load_search_config


CONFIG_PATH = Path(__file__).with_name("topology_calru_tuning_v2.json")
CAMPAIGN_ID = "manifold_calru_topology_tuning_v2"


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    return {"code_commit": commit, "worktree_dirty": bool(status)}


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    payload = load_search_config(path)
    if payload["campaign_id"] != CAMPAIGN_ID:
        raise ValueError("CA-LRU topology tuning v2 campaign id differs")
    if tuple(payload["search"]["topologies"]) != ("s1", "t2", "s2"):
        raise ValueError("CA-LRU topology tuning v2 topology order differs")
    if float(payload["retention_plasticity"]["damage_epsilon"]) != 3e-5:
        raise ValueError("v2 freezes RP epsilon at 3e-5")
    if int(payload["models"]["calru"]["width"]) != 52:
        raise ValueError("v2 freezes CA-LRU width at 52")
    return payload


def _slug(value: float) -> str:
    return (
        format(float(value), ".6g")
        .replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
    )


def _cell_id(learning_rate: float, eta_lambda: float, interval: int) -> str:
    return f"lr{_slug(learning_rate)}_eta{_slug(eta_lambda)}_i{int(interval)}"


def smoke_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    row = config["search"]["smoke"]
    rp = config["retention_plasticity"]
    return [
        _base_job(
            phase="smoke",
            model="calru",
            topology=topology,
            seed=int(row["seed"]),
            learning_rate=float(config["models"]["calru"]["learning_rate"]),
            rp_eta_lambda=float(rp["eta_lambda"]),
            rp_interval=int(rp["intervention_interval_updates"]),
            rp_warmup=200,
            updates=int(row["updates"]),
            cell_id="registered_debug",
        )
        for topology in config["search"]["topologies"]
    ]


def screen_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    screen = config["search"]["screen"]
    warmup = int(config["retention_plasticity"]["warmup_updates"])
    updates = int(config["training"]["updates"])
    jobs: list[dict[str, Any]] = []
    for topology in config["search"]["topologies"]:
        grid = screen[topology]
        for learning_rate, eta_lambda, interval in product(
            grid["learning_rates"],
            grid["eta_lambdas"],
            grid["intervention_intervals"],
        ):
            jobs.append(
                _base_job(
                    phase="screen",
                    model="calru",
                    topology=topology,
                    seed=int(screen["seed"]),
                    learning_rate=float(learning_rate),
                    rp_eta_lambda=float(eta_lambda),
                    rp_interval=int(interval),
                    rp_warmup=warmup,
                    updates=updates,
                    cell_id=_cell_id(learning_rate, eta_lambda, interval),
                )
            )
    return jobs


def _result(root: Path, job: dict[str, Any]) -> dict[str, Any] | None:
    run_dir = root / job["job_id"]
    if not (run_dir / "COMPLETED.json").is_file():
        return None
    return strict_json_load(run_dir / "result.json")


def _accounted(root: Path, job: dict[str, Any]) -> bool:
    run_dir = root / job["job_id"]
    return (run_dir / "COMPLETED.json").is_file() or (
        run_dir / "FAILED.json"
    ).is_file()


def _metrics(
    result: dict[str, Any] | None, config: dict[str, Any]
) -> dict[str, Any]:
    if result is None:
        return {
            "completed": False,
            "task_gate": False,
            "blank_h4096_finite": False,
            "task_intrinsic_mean_radians": None,
            "blank_h4096_intrinsic_mean_radians": None,
            "validation_nmse_db": None,
        }
    blank = result["blank_validation"]["horizons"]["4096"]
    gate = (
        bool(result["beats_hold_baseline_intrinsic"])
        and float(result["validation_nmse_db"])
        < float(config["search"]["task_gate_nmse_db_strictly_below"])
    )
    return {
        "completed": True,
        "task_gate": gate,
        "blank_h4096_finite": bool(blank["finite"]),
        "task_intrinsic_mean_radians": float(
            result["final_validation"]["intrinsic_mean_radians"]
        ),
        "blank_h4096_intrinsic_mean_radians": (
            float(blank["intrinsic_mean_radians"]) if blank["finite"] else None
        ),
        "validation_nmse_db": float(result["validation_nmse_db"]),
    }


def _finite_or_inf(value: Any) -> float:
    if value is None:
        return math.inf
    number = float(value)
    return number if math.isfinite(number) else math.inf


def _screen_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    return (
        not bool(metrics["task_gate"]),
        not bool(metrics["blank_h4096_finite"]),
        _finite_or_inf(metrics["blank_h4096_intrinsic_mean_radians"]),
        _finite_or_inf(metrics["task_intrinsic_mean_radians"]),
        tuple(row["grid_order"]),
    )


def select_screen(
    root: Path,
    jobs: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    path = root / "selection" / "screen_selection.json"
    if path.is_file():
        return strict_json_load(path)
    missing = [job["job_id"] for job in jobs if not _accounted(root, job)]
    if missing:
        raise RuntimeError(f"screen selection has {len(missing)} unaccounted jobs")
    selected: dict[str, list[dict[str, Any]]] = {}
    ranked: dict[str, list[dict[str, Any]]] = {}
    for topology in config["search"]["topologies"]:
        topology_jobs = [job for job in jobs if job["topology"] == topology]
        rows = []
        for grid_index, job in enumerate(topology_jobs):
            rows.append(
                {
                    "job": job,
                    "grid_order": [grid_index],
                    "metrics": _metrics(_result(root, job), config),
                }
            )
        rows.sort(key=_screen_key)
        count = int(config["search"]["screen"][topology]["finalists"])
        completed = [row for row in rows if row["metrics"]["completed"]]
        if len(completed) < count:
            raise RuntimeError(f"too few completed screen candidates for {topology}")
        selected[topology] = [row["job"] for row in completed[:count]]
        ranked[topology] = rows
    payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "selection_split": "validation",
        "test_bank_accessed": False,
        "selection_rule": config["search"]["screen_selection_rule"],
        "selected": selected,
        "ranked_candidates": ranked,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload)
    return payload


def confirm_jobs(
    config: dict[str, Any], selection: dict[str, Any]
) -> list[dict[str, Any]]:
    seeds = [int(seed) for seed in config["search"]["confirm"]["additional_seeds"]]
    jobs: list[dict[str, Any]] = []
    for topology in config["search"]["topologies"]:
        for finalist in selection["selected"][topology]:
            for seed in seeds:
                jobs.append(
                    _base_job(
                        phase="confirm",
                        model="calru",
                        topology=topology,
                        seed=seed,
                        learning_rate=float(finalist["learning_rate"]),
                        rp_eta_lambda=float(finalist["rp_eta_lambda"]),
                        rp_interval=int(finalist["rp_interval"]),
                        rp_warmup=int(finalist["rp_warmup"]),
                        updates=int(finalist["updates"]),
                        cell_id=str(finalist["cell_id"]),
                    )
                )
    return jobs


def _median(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(median(finite)) if finite else None


def _aggregate_finalist(
    root: Path,
    screen_job: dict[str, Any],
    confirm: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    jobs = [screen_job, *confirm]
    rows = [
        {
            "job_id": job["job_id"],
            "seed": int(job["seed"]),
            **_metrics(_result(root, job), config),
        }
        for job in jobs
    ]
    task_values = [
        float(row["task_intrinsic_mean_radians"])
        for row in rows
        if row["task_intrinsic_mean_radians"] is not None
    ]
    blank_values = [
        float(row["blank_h4096_intrinsic_mean_radians"])
        for row in rows
        if row["blank_h4096_intrinsic_mean_radians"] is not None
    ]
    return {
        "cell": {
            "cell_id": screen_job["cell_id"],
            "topology": screen_job["topology"],
            "learning_rate": screen_job["learning_rate"],
            "rp_eta_lambda": screen_job["rp_eta_lambda"],
            "rp_interval": screen_job["rp_interval"],
            "rp_warmup": screen_job["rp_warmup"],
            "updates": screen_job["updates"],
        },
        "rows": rows,
        "summary": {
            "registered_seed_count": len(rows),
            "completed_count": sum(bool(row["completed"]) for row in rows),
            "task_gate_count": sum(bool(row["task_gate"]) for row in rows),
            "blank_h4096_finite_count": sum(
                bool(row["blank_h4096_finite"]) for row in rows
            ),
            "median_task_intrinsic_mean_radians": _median(task_values),
            "median_blank_h4096_intrinsic_mean_radians": _median(blank_values),
        },
    }


def _final_key(row: dict[str, Any], grid_order: int) -> tuple[Any, ...]:
    summary = row["summary"]
    return (
        -int(summary["task_gate_count"]),
        -int(summary["completed_count"]),
        -int(summary["blank_h4096_finite_count"]),
        _finite_or_inf(summary["median_blank_h4096_intrinsic_mean_radians"]),
        _finite_or_inf(summary["median_task_intrinsic_mean_radians"]),
        int(grid_order),
    )


def select_final(
    root: Path,
    screen_selection: dict[str, Any],
    confirm: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    path = root / "selection" / "FINAL_SELECTION.json"
    if path.is_file():
        return strict_json_load(path)
    missing = [job["job_id"] for job in confirm if not _accounted(root, job)]
    if missing:
        raise RuntimeError(f"final selection has {len(missing)} unaccounted jobs")
    selected: dict[str, Any] = {}
    ranked: dict[str, Any] = {}
    for topology in config["search"]["topologies"]:
        rows: list[dict[str, Any]] = []
        for finalist in screen_selection["selected"][topology]:
            matching = [
                job
                for job in confirm
                if job["topology"] == topology
                and job["cell_id"] == finalist["cell_id"]
            ]
            rows.append(_aggregate_finalist(root, finalist, matching, config))
        order = {row["cell"]["cell_id"]: index for index, row in enumerate(rows)}
        rows.sort(
            key=lambda row: _final_key(row, order[row["cell"]["cell_id"]])
        )
        selected[topology] = rows[0]
        ranked[topology] = rows
    payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "selection_split": "validation",
        "test_bank_accessed": False,
        "selection_rule": config["search"]["final_selection_rule"],
        "selected": selected,
        "ranked_finalists": ranked,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload)
    return payload


def _run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = args.config.expanduser().resolve(strict=True)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("at least one CUDA device is required")
    git = _git_state()
    if (
        bool(config["execution"].get("require_clean_git_worktree", False))
        and git["worktree_dirty"]
    ):
        raise RuntimeError("CA-LRU topology tuning v2 requires a clean worktree")
    atomic_json(
        root / "campaign_manifest.json",
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "config": config,
            "config_path": str(config_path),
            **git,
            "devices": devices,
            "selection_split": "validation",
            "test_bank_accessed": False,
            "resume_command": (
                f"PYTHONPATH=. {sys.executable} -m "
                "repro.manifold_benchmark.launch_calru_tuning_v2 "
                f"--phase all --devices {','.join(devices)} "
                f"--config {config_path} --output {root}"
            ),
        },
    )

    smoke = smoke_jobs(config)
    screen = screen_jobs(config)
    if args.phase in {"smoke", "all"}:
        status = run_phase(
            phase="smoke",
            jobs=smoke,
            root=root,
            devices=devices,
            config_path=config_path,
            config=config,
        )
        if status["completed_or_preexisting"] != len(smoke):
            raise RuntimeError("v2 smoke phase did not complete all three topologies")
        if args.phase == "smoke":
            return
    if args.phase in {"screen", "all"}:
        run_phase(
            phase="screen",
            jobs=screen,
            root=root,
            devices=devices,
            config_path=config_path,
            config=config,
        )
        if args.phase == "screen":
            select_screen(root, screen, config)
            return
    screen_selection = select_screen(root, screen, config)
    confirm = confirm_jobs(config, screen_selection)
    if args.phase in {"confirm", "all"}:
        run_phase(
            phase="confirm",
            jobs=confirm,
            root=root,
            devices=devices,
            config_path=config_path,
            config=config,
        )
    final_selection = select_final(root, screen_selection, confirm, config)
    expected = smoke + screen + confirm
    atomic_json(
        root / "CAMPAIGN_TUNING_FINISHED.json",
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "expected_jobs_including_smoke": len(expected),
            "completed_jobs": sum(
                (root / job["job_id"] / "COMPLETED.json").is_file()
                for job in expected
            ),
            "failed_jobs": [
                job["job_id"]
                for job in expected
                if (root / job["job_id"] / "FAILED.json").is_file()
            ],
            "selection_split": "validation",
            "test_bank_accessed": False,
            "selected": final_selection["selected"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("smoke", "screen", "confirm", "all"), default="all"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    _run(args, load_config(args.config))


if __name__ == "__main__":
    main()
