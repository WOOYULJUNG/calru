"""Fresh-seed confirmation and final topology-aware CA-LRU selection."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
from statistics import median
import subprocess
import sys
from typing import Any

from repro.sagodi_protocol.artifacts import atomic_json

from .analyze_calru_topology_screen_v3 import (
    TOPOLOGIES,
    _task_metrics,
    _topology_metrics,
    _write_csv,
)
from .launch_calru_tuning_v2 import CONFIG_PATH, load_config
from .launch_topology_hparam import _base_job, run_phase


FRESH_CONFIRMATION_SEEDS = (23, 24, 25)
TOP_RANKED_COUNTS = {"s1": 4, "t2": 4, "s2": 4}
S2_PRIOR_CELLS = (
    # Previous seeds 10--12 formed a strong S2 H2 feature at this cell.
    "lr0p003_eta1000_i50",
    # Best moderate-LR task-gated S2 screen cell in tuning-v2.
    "lr0p003_eta3000_i50",
)


def _selected_candidates(ranking_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(ranking_path.read_text())
    selected: list[dict[str, Any]] = []
    for topology in TOPOLOGIES:
        ranking = payload["rankings"][topology]
        chosen = list(ranking[: TOP_RANKED_COUNTS[topology]])
        if topology == "s2":
            by_cell = {row["cell_id"]: row for row in ranking}
            for cell_id in S2_PRIOR_CELLS:
                if cell_id not in by_cell:
                    raise ValueError(f"missing registered S2 prior cell {cell_id}")
                if all(row["cell_id"] != cell_id for row in chosen):
                    chosen.append(by_cell[cell_id])
        for row in chosen:
            selected.append(
                {
                    "topology": topology,
                    "cell_id": row["cell_id"],
                    "learning_rate": float(row["learning_rate"]),
                    "rp_eta_lambda": float(row["rp_eta_lambda"]),
                    "rp_interval": int(row["rp_interval"]),
                    "rp_warmup": int(row["rp_warmup"]),
                    "screen_rank": int(row["rank"]),
                    "screen_task_gate": bool(row["task_gate"]),
                    "screen_topology_penalty": float(
                        row["topology_penalty_total"]
                    ),
                    "screen_topology_exact": bool(
                        row["topology_exact_h0_h2048"]
                    ),
                    "included_by_prior": (
                        topology == "s2" and row["cell_id"] in S2_PRIOR_CELLS
                    ),
                }
            )
    expected = sum(TOP_RANKED_COUNTS.values()) + len(S2_PRIOR_CELLS)
    if len(selected) != expected:
        raise ValueError(f"expected {expected} unique candidates, found {len(selected)}")
    return selected


def _jobs(candidates: list[dict[str, Any]], updates: int) -> list[dict[str, Any]]:
    return [
        _base_job(
            phase="topology_confirm_v3",
            model="calru",
            topology=row["topology"],
            seed=seed,
            learning_rate=row["learning_rate"],
            rp_eta_lambda=row["rp_eta_lambda"],
            rp_interval=row["rp_interval"],
            rp_warmup=row["rp_warmup"],
            updates=updates,
            cell_id=row["cell_id"],
        )
        for row in candidates
        for seed in FRESH_CONFIRMATION_SEEDS
    ]


def _median(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(median(finite)) if finite else math.inf


def _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(row["joint_eligible_count"]),
        -int(row["topology_exact_count"]),
        -int(row["task_gate_count"]),
        float(row["median_topology_penalty"]),
        float(row["median_blank_h4096_intrinsic_radians"]),
        float(row["median_task_intrinsic_radians"]),
        str(row["cell_id"]),
    )


def _report(
    seed_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    lines = [
        "# CA-LRU topology-aware confirmation v3",
        "",
        "Candidates were screened on seed 20, then retrained from scratch on "
        "fresh seeds 23, 24, and 25. Final ranking uses validation task gates "
        "and exact persistent-homology signatures at both H=0 and H=2048. "
        "Test data are not accessed.",
        "",
        "| Topology | Rank | Cell | Joint exact | Topology exact | Task gate | "
        "Median PH penalty | Task rad | Blank-4096 rad | Robust joint |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for topology in TOPOLOGIES:
        group = sorted(
            (row for row in summaries if row["topology"] == topology),
            key=_selection_key,
        )
        for rank, row in enumerate(group, 1):
            lines.append(
                f"| {topology.upper()} | {rank} | {row['cell_id']} | "
                f"{row['joint_eligible_count']}/3 | "
                f"{row['topology_exact_count']}/3 | "
                f"{row['task_gate_count']}/3 | "
                f"{row['median_topology_penalty']:.4g} | "
                f"{row['median_task_intrinsic_radians']:.4g} | "
                f"{row['median_blank_h4096_intrinsic_radians']:.4g} | "
                f"{int(bool(row['robust_joint_success']))} |"
            )
    lines.extend(["", "## Final selections", ""])
    for topology in TOPOLOGIES:
        row = selected[topology]
        lines.append(
            f"- {topology.upper()}: `{row['cell_id']}`; "
            f"joint {row['joint_eligible_count']}/3, topology "
            f"{row['topology_exact_count']}/3, task {row['task_gate_count']}/3."
        )
    if not any(bool(row["robust_joint_success"]) for row in selected.values()):
        lines.extend(
            [
                "",
                "No selected topology has a 2/3 fresh-seed majority that "
                "simultaneously passes the task gate and exact H0/H2048 "
                "topology criterion.",
            ]
        )
    lines.extend(
        [
            "",
            f"Seed-level rows: {len(seed_rows)}. Final selection never uses "
            "the test bank.",
            "",
        ]
    )
    (output / "CONFIRMATION_RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen-ranking",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "manifold_calru_topology_selection_v3/"
            "screen_topology_ranking.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "manifold_calru_topology_selection_v3"
        ),
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--devices", default="0,1,2,3,4,5")
    parser.add_argument("--analysis-workers", type=int, default=12)
    args = parser.parse_args()

    ranking_path = args.screen_ranking.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = args.config.expanduser().resolve(strict=True)
    config = load_config(config_path)
    candidates = _selected_candidates(ranking_path)
    _write_csv(output / "confirmation_candidates.csv", candidates)
    jobs = _jobs(candidates, int(config["training"]["updates"]))
    if len(jobs) != 42:
        raise ValueError(f"expected 42 fresh confirmation jobs, found {len(jobs)}")
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    state = run_phase(
        phase="topology_confirmation_v3",
        jobs=jobs,
        root=output,
        devices=devices,
        config_path=config_path,
        config=config,
    )
    if int(state["failed"]):
        raise SystemExit(f"{state['failed']} confirmation training jobs failed")

    def analyze(index: int, job: dict[str, Any]) -> dict[str, Any]:
        checkpoint = output / job["job_id"] / "checkpoint.pt"
        target = output / "confirmation_topology" / job["job_id"]
        target.mkdir(parents=True, exist_ok=True)
        topology_path = target / "topology.json"
        if topology_path.is_file():
            return {
                "job_id": job["job_id"],
                "returncode": 0,
                "skipped_completed": True,
            }
        command = [
            sys.executable,
            "-m",
            "repro.static_gate_pilot.analyze_checkpoint_topology",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(target),
            "--device",
            f"cuda:{index % len(devices)}",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        return {
            "job_id": job["job_id"],
            "returncode": int(completed.returncode),
            "skipped_completed": False,
            "stderr": completed.stderr[-8000:],
        }

    analysis_records = []
    with ThreadPoolExecutor(max_workers=int(args.analysis_workers)) as executor:
        futures = {
            executor.submit(analyze, index, job): job
            for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            record = future.result()
            analysis_records.append(record)
            print(
                f"[PH {len(analysis_records):02d}/42] {record['job_id']} "
                f"{'ok' if record['returncode'] == 0 else 'FAILED'}",
                flush=True,
            )
            if record["returncode"]:
                print(record["stderr"], flush=True)
    failures = [record for record in analysis_records if record["returncode"]]
    if failures:
        raise SystemExit(f"{len(failures)} confirmation PH analyses failed")

    candidate_index = {
        (row["topology"], row["cell_id"]): row for row in candidates
    }
    seed_rows = []
    for job in jobs:
        run_dir = output / job["job_id"]
        result = json.loads((run_dir / "result.json").read_text())
        topology_payload = json.loads(
            (
                output
                / "confirmation_topology"
                / job["job_id"]
                / "topology.json"
            ).read_text()
        )
        source = candidate_index[(job["topology"], job["cell_id"])]
        row = {
            "job_id": job["job_id"],
            "topology": job["topology"],
            "cell_id": job["cell_id"],
            "seed": int(job["seed"]),
            "learning_rate": float(job["learning_rate"]),
            "rp_eta_lambda": float(job["rp_eta_lambda"]),
            "rp_interval": int(job["rp_interval"]),
            "screen_rank": int(source["screen_rank"]),
            "included_by_prior": bool(source["included_by_prior"]),
            **_task_metrics(result),
            **_topology_metrics(topology_payload),
            "checkpoint": str(run_dir / "checkpoint.pt"),
        }
        row["joint_eligible"] = bool(
            row["task_gate"]
            and row["blank_h4096_finite"]
            and row["topology_exact_h0_h2048"]
        )
        seed_rows.append(row)
    _write_csv(output / "confirmation_seed_metrics.csv", seed_rows)

    summaries = []
    for candidate in candidates:
        group = [
            row
            for row in seed_rows
            if row["topology"] == candidate["topology"]
            and row["cell_id"] == candidate["cell_id"]
        ]
        if len(group) != 3:
            raise ValueError(
                f"expected 3 rows for {candidate['topology']}/{candidate['cell_id']}"
            )
        summary = {
            **candidate,
            "seed_count": 3,
            "joint_eligible_count": sum(
                bool(row["joint_eligible"]) for row in group
            ),
            "topology_exact_count": sum(
                bool(row["topology_exact_h0_h2048"]) for row in group
            ),
            "task_gate_count": sum(bool(row["task_gate"]) for row in group),
            "blank_finite_count": sum(
                bool(row["blank_h4096_finite"]) for row in group
            ),
            "median_topology_penalty": _median(
                [row["topology_penalty_total"] for row in group]
            ),
            "median_task_intrinsic_radians": _median(
                [row["task_intrinsic_radians"] for row in group]
            ),
            "median_blank_h4096_intrinsic_radians": _median(
                [row["blank_h4096_intrinsic_radians"] for row in group]
            ),
        }
        summary["robust_joint_success"] = bool(
            summary["joint_eligible_count"] >= 2
            and summary["topology_exact_count"] >= 2
            and summary["task_gate_count"] >= 2
        )
        summaries.append(summary)
    summaries.sort(
        key=lambda row: (TOPOLOGIES.index(row["topology"]), _selection_key(row))
    )
    _write_csv(output / "confirmation_cell_summary.csv", summaries)
    selected = {
        topology: min(
            (row for row in summaries if row["topology"] == topology),
            key=_selection_key,
        )
        for topology in TOPOLOGIES
    }
    atomic_json(
        output / "FINAL_TOPOLOGY_SELECTION.json",
        {
            "schema_version": 1,
            "selection_split": "validation",
            "test_bank_accessed": False,
            "screen_seed": 20,
            "fresh_confirmation_seeds": list(FRESH_CONFIRMATION_SEEDS),
            "topology_selection_horizons": [0, 2048],
            "robust_joint_rule": (
                "at least 2/3 seeds jointly pass task gate, finite H4096, "
                "and exact target PH signature at H0 and H2048"
            ),
            "selection_rule": (
                "max joint count, max topology count, max task count, "
                "min median PH penalty, min blank error, min task error"
            ),
            "selected": selected,
            "ranked_candidates": {
                topology: sorted(
                    (row for row in summaries if row["topology"] == topology),
                    key=_selection_key,
                )
                for topology in TOPOLOGIES
            },
        },
    )
    atomic_json(
        output / "confirmation_analysis_records.json",
        {
            "schema_version": 1,
            "records": analysis_records,
        },
    )
    _report(seed_rows, summaries, selected, output)
    print(output / "CONFIRMATION_RESULTS.md")


if __name__ == "__main__":
    main()
