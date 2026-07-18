"""Topology-aware reanalysis of every CA-LRU tuning-v2 screen checkpoint."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

from repro.sagodi_protocol.artifacts import atomic_json


TOPOLOGIES = ("s1", "t2", "s2")
EXPECTED = {
    "s1": {"h1": 1, "h2": 0},
    "t2": {"h1": 2, "h2": 1},
    "s2": {"h1": 0, "h2": 1},
}
SELECTION_HORIZONS = ("0", "2048")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _task_metrics(result: dict[str, Any]) -> dict[str, Any]:
    blank = result["blank_validation"]["horizons"]["4096"]
    task_gate = bool(result["beats_hold_baseline_intrinsic"]) and float(
        result["validation_nmse_db"]
    ) < -20.0
    return {
        "task_gate": task_gate,
        "task_intrinsic_radians": float(
            result["final_validation"]["intrinsic_mean_radians"]
        ),
        "blank_h4096_finite": bool(blank["finite"]),
        "blank_h4096_intrinsic_radians": (
            float(blank["intrinsic_mean_radians"])
            if bool(blank["finite"])
            else math.inf
        ),
        "validation_nmse_db": float(result["validation_nmse_db"]),
    }


def _dimension_penalty(
    persistences: list[float],
    *,
    expected_count: int,
    threshold: float,
) -> float:
    """Continuous exact-signature violation in units of the ideal threshold."""

    scale = max(float(threshold), 1e-12)
    values = sorted((float(value) for value in persistences), reverse=True)
    penalty = 0.0
    for index in range(expected_count):
        value = values[index] if index < len(values) else 0.0
        penalty += max(0.0, scale - value) / scale
    for value in values[expected_count:]:
        penalty += max(0.0, value - scale) / scale
    return penalty


def _topology_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    expected = EXPECTED[str(payload["topology"])]
    thresholds = payload["strong_bar_thresholds"]
    rows: dict[str, Any] = {}
    mismatch_total = 0
    penalty_total = 0.0
    exact_both = True
    for horizon in SELECTION_HORIZONS:
        item = payload["horizons"][horizon]
        exact = bool(item["signature_match"])
        exact_both = exact_both and exact
        mismatch = abs(int(item["detected_h1"]) - int(expected["h1"])) + abs(
            int(item["detected_h2"]) - int(expected["h2"])
        )
        penalty = _dimension_penalty(
            item["h1_top_persistences"],
            expected_count=int(expected["h1"]),
            threshold=float(thresholds["h1"]),
        ) + _dimension_penalty(
            item["h2_top_persistences"],
            expected_count=int(expected["h2"]),
            threshold=float(thresholds["h2"]),
        )
        mismatch_total += mismatch
        penalty_total += penalty
        rows.update(
            {
                f"topology_exact_h{horizon}": exact,
                f"detected_h1_h{horizon}": int(item["detected_h1"]),
                f"detected_h2_h{horizon}": int(item["detected_h2"]),
                f"topology_mismatch_h{horizon}": mismatch,
                f"topology_penalty_h{horizon}": penalty,
            }
        )
    return {
        **rows,
        "topology_exact_h0_h2048": exact_both,
        "topology_mismatch_total": mismatch_total,
        "topology_penalty_total": penalty_total,
        "h1_threshold": float(thresholds["h1"]),
        "h2_threshold": float(thresholds["h2"]),
    }


def _ranking_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Topology-first joint selection with task/blank eligibility."""

    joint_exact = (
        bool(row["topology_exact_h0_h2048"])
        and bool(row["task_gate"])
        and bool(row["blank_h4096_finite"])
    )
    return (
        not joint_exact,
        not bool(row["topology_exact_h0_h2048"]),
        int(row["topology_mismatch_total"]),
        float(row["topology_penalty_total"]),
        not bool(row["task_gate"]),
        float(row["blank_h4096_intrinsic_radians"]),
        float(row["task_intrinsic_radians"]),
        str(row["cell_id"]),
    )


def _report(rows: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "# CA-LRU topology-aware screen v3",
        "",
        "All tuning-v2 seed-20 screen checkpoints are ranked using validation "
        "persistent homology at H=0 and H=2048. Test data are not accessed.",
        "",
        "| Topology | Rank | Cell | Task gate | Exact H0 | Exact H2048 | "
        "Mismatch | PH penalty | Task rad | Blank-4096 rad |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for topology in TOPOLOGIES:
        group = sorted(
            (row for row in rows if row["topology"] == topology),
            key=_ranking_key,
        )
        for rank, row in enumerate(group, 1):
            lines.append(
                f"| {topology.upper()} | {rank} | {row['cell_id']} | "
                f"{int(bool(row['task_gate']))} | "
                f"{int(bool(row['topology_exact_h0']))} | "
                f"{int(bool(row['topology_exact_h2048']))} | "
                f"{row['topology_mismatch_total']} | "
                f"{row['topology_penalty_total']:.4g} | "
                f"{row['task_intrinsic_radians']:.4g} | "
                f"{row['blank_h4096_intrinsic_radians']:.4g} |"
            )
    (output / "SCREEN_RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "manifold_calru_topology_tuning_v2-e722ccf"
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
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(run_root.glob("screen__calru__*/checkpoint.pt"))
    if len(checkpoints) != 40:
        raise ValueError(f"expected 40 screen checkpoints, found {len(checkpoints)}")
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("at least one device is required")

    def run(index: int, checkpoint: Path) -> dict[str, Any]:
        target = output / "screen_topology" / checkpoint.parent.name
        target.mkdir(parents=True, exist_ok=True)
        topology_path = target / "topology.json"
        if topology_path.is_file():
            return {
                "job_id": checkpoint.parent.name,
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
            devices[index % len(devices)],
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        return {
            "job_id": checkpoint.parent.name,
            "returncode": int(completed.returncode),
            "skipped_completed": False,
            "stderr": completed.stderr[-8000:],
        }

    records = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {
            executor.submit(run, index, checkpoint): checkpoint
            for index, checkpoint in enumerate(checkpoints)
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"[{len(records):02d}/40] {record['job_id']} "
                f"{'ok' if record['returncode'] == 0 else 'FAILED'}",
                flush=True,
            )
            if record["returncode"]:
                print(record["stderr"], flush=True)
    failures = [record for record in records if record["returncode"]]
    if failures:
        raise SystemExit(f"{len(failures)} topology analyses failed")

    rows = []
    for checkpoint in checkpoints:
        result = json.loads((checkpoint.parent / "result.json").read_text())
        manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
        topology_payload = json.loads(
            (
                output
                / "screen_topology"
                / checkpoint.parent.name
                / "topology.json"
            ).read_text()
        )
        row = {
            "job_id": checkpoint.parent.name,
            "cell_id": manifest["cell_id"],
            "topology": manifest["topology"],
            "seed": int(manifest["replicate_seed"]),
            "learning_rate": float(manifest["learning_rate"]),
            "rp_eta_lambda": float(manifest["rp_eta_lambda"]),
            "rp_interval": int(manifest["rp_interval"]),
            "rp_warmup": int(manifest["rp_warmup"]),
            **_task_metrics(result),
            **_topology_metrics(topology_payload),
            "checkpoint": str(checkpoint),
        }
        row["joint_eligible"] = bool(
            row["task_gate"]
            and row["blank_h4096_finite"]
            and row["topology_exact_h0_h2048"]
        )
        rows.append(row)
    rows.sort(key=lambda row: (TOPOLOGIES.index(row["topology"]), _ranking_key(row)))
    _write_csv(output / "screen_topology_metrics.csv", rows)
    rankings = {
        topology: [
            {
                "rank": rank,
                **row,
            }
            for rank, row in enumerate(
                sorted(
                    (item for item in rows if item["topology"] == topology),
                    key=_ranking_key,
                ),
                1,
            )
        ]
        for topology in TOPOLOGIES
    }
    atomic_json(
        output / "screen_topology_ranking.json",
        {
            "schema_version": 1,
            "selection_split": "validation",
            "test_bank_accessed": False,
            "selection_horizons": [int(value) for value in SELECTION_HORIZONS],
            "ranking_rule": (
                "joint exact topology and task gate; exact topology; count "
                "mismatch; continuous PH penalty; task gate; blank; task"
            ),
            "rankings": rankings,
        },
    )
    (output / "pipeline_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    _report(rows, output)
    print(output / "SCREEN_RESULTS.md")


if __name__ == "__main__":
    main()
