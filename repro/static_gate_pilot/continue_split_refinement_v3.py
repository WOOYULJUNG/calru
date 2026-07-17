"""Continue the v3 split-field refinement through dynamics and 3-seed analysis."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


EXPERIMENTS = Path("/home/biadmin/ca_rnn/experiments")
SCREEN = EXPERIMENTS / "static_gate_split_refinement_v3"
FULL = EXPERIMENTS / "static_gate_split_refinement_full_v3"
DYNAMICS = EXPERIMENTS / "static_gate_split_refinement_dynamics_v3"
RP = EXPERIMENTS / "static_gate_split_refinement_rp_v3"
RP_DYNAMICS = EXPERIMENTS / "static_gate_split_refinement_rp_dynamics_v3"
FINAL_SEEDS = EXPERIMENTS / "static_gate_split_refinement_final_seeds_v3"
FINAL_ANALYSIS = EXPERIMENTS / "static_gate_split_refinement_final_analysis_v3"
STATUS = EXPERIMENTS / "static_gate_split_refinement_pipeline_v3_status.json"


def _save_status(payload: dict) -> None:
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATUS)


def _run(stage: str, command: list[str], status: dict) -> None:
    status["stage"] = stage
    status["state"] = "running"
    status["stage_started_unix"] = time.time()
    _save_status(status)
    print(f"[pipeline] {stage}", flush=True)
    completed = subprocess.run(command)
    status.setdefault("completed_stages", []).append(
        {
            "stage": stage,
            "returncode": int(completed.returncode),
            "finished_unix": time.time(),
        }
    )
    if completed.returncode:
        status["state"] = "failed"
        status["returncode"] = int(completed.returncode)
        _save_status(status)
        raise SystemExit(f"{stage} failed with {completed.returncode}")
    _save_status(status)


def main() -> None:
    status = {
        "schema_version": 1,
        "campaign": "static_gate_split_refinement_v3",
        "state": "waiting_for_screen",
        "stage": "screen",
        "started_unix": time.time(),
        "completed_stages": [],
    }
    _save_status(status)
    marker = SCREEN / "launcher_records.json"
    while not marker.is_file():
        print("[pipeline] waiting for 504-cell refinement screen", flush=True)
        time.sleep(30)

    python = sys.executable
    _run(
        "full_pretrain_6000",
        [
            python,
            "-m",
            "repro.static_gate_pilot.launch_split_refinement_full_v3",
        ],
        status,
    )
    _run(
        "candidate_dynamics",
        [
            python,
            "-m",
            "repro.static_gate_pilot.launch_split_candidate_dynamics_v2",
            "--candidates",
            str(FULL / "full_summary.csv"),
            "--output",
            str(DYNAMICS),
        ],
        status,
    )
    _run(
        "rp_and_same_update_controls",
        [
            python,
            "-m",
            "repro.static_gate_pilot.launch_split_candidate_rp_v2",
            "--candidates",
            str(DYNAMICS / "candidate_dynamics_summary.csv"),
            "--output",
            str(RP),
            "--data-update-offset",
            "6000",
            "--updates",
            "2000",
        ],
        status,
    )
    _run(
        "rp_dynamics_and_selection",
        [
            python,
            "-m",
            "repro.static_gate_pilot.analyze_split_candidate_rp_v2",
            "--candidates",
            str(RP / "rp_results.csv"),
            "--output",
            str(RP_DYNAMICS),
        ],
        status,
    )
    _run(
        "final_seeds_11_12",
        [
            python,
            "-m",
            "repro.static_gate_pilot.launch_split_final_seeds_v2",
            "--selection",
            str(RP_DYNAMICS / "finalists.csv"),
            "--output",
            str(FINAL_SEEDS),
            "--pretrain-updates",
            "6000",
            "--final-updates",
            "2000",
        ],
        status,
    )
    _run(
        "final_dynamics_topology_baseline_comparison",
        [
            python,
            "-m",
            "repro.static_gate_pilot.launch_split_final_analysis_v2",
            "--selection",
            str(RP_DYNAMICS / "finalists.csv"),
            "--final-root",
            str(FINAL_SEEDS / "final"),
            "--output",
            str(FINAL_ANALYSIS),
        ],
        status,
    )
    status["state"] = "complete"
    status["stage"] = "complete"
    status["finished_unix"] = time.time()
    _save_status(status)
    print(FINAL_ANALYSIS / "RESULTS.md")


if __name__ == "__main__":
    main()
