from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


V2_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = V2_DIR.parents[1]
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

import launch_p0 as launch
import status_p0 as status


def _job(name: str) -> launch.Job:
    condition = name
    return launch.Job(
        job_id=f"{name}-ring-hold-s0-deadbeef00",
        family="test",
        condition=condition,
        task="ring_hold",
        model="test",
        worker_model="test",
        seed=0,
        runner="legacy",
        command=("<PYTHON>", "worker.py"),
        implemented=True,
        implementation_note="test fixture",
        expected=(
            f"results/{name}/result.json",
            f"checkpoints/{name}/model.pt",
            f"traces/{name}/ring_hold_v2_{name}_seed0_loss_trace.npz",
        ),
        log_dir=f"logs/{name}",
        metadata={},
    )


def _write_valid_artifacts(job: launch.Job, campaign: Path) -> None:
    for path in launch._expected_paths(job, campaign):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(
                json.dumps({"task": job.task, "seed": job.seed, "tag": f"v2_{job.condition}"}),
                encoding="utf-8",
            )
        elif path.suffix == ".pt":
            torch.save({"weight": torch.ones(1)}, path)
        else:
            np.savez_compressed(
                path,
                steps=np.asarray([1], dtype=np.int64),
                train_total_loss=np.asarray([0.2], dtype=np.float32),
                train_task_loss=np.asarray([0.2], dtype=np.float32),
            )


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, launch.Job]]:
    root = tmp_path / "artifacts"
    launch._prepare_artifact_root(root)
    jobs = {name: _job(name) for name in ("complete", "running", "failed", "partial", "pending")}
    config = json.loads((V2_DIR / "campaign.example.json").read_text(encoding="utf-8"))
    campaign_id, manifest = launch.build_manifest(config, list(jobs.values()), smoke=False)
    campaign = root / campaign_id
    launch._write_or_check_manifest(campaign, manifest)

    _write_valid_artifacts(jobs["complete"], campaign)
    launch._write_completion_receipt(jobs["complete"], campaign)
    partial_json = launch._expected_paths(jobs["partial"], campaign)[0]
    partial_json.parent.mkdir(parents=True, exist_ok=True)
    partial_json.write_text(
        json.dumps({"task": "ring_hold", "seed": 0, "tag": "v2_partial"}), encoding="utf-8"
    )

    for name in ("running", "failed", "partial"):
        log = campaign / jobs[name].log_dir / f"{jobs[name].job_id}.attempt001.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"{name}\n", encoding="utf-8")
    launch._atomic_write_json(
        campaign / "status.json",
        {
            "jobs": {
                jobs["complete"].job_id: {"state": "complete", "exit_code": 0},
                jobs["running"].job_id: {
                    "state": "running",
                    "gpu": 3,
                    "pid": 4242,
                    "log": f"{jobs['running'].log_dir}/{jobs['running'].job_id}.attempt001.log",
                },
                jobs["failed"].job_id: {
                    "state": "failed",
                    "exit_code": 1,
                    "log": f"{jobs['failed'].log_dir}/{jobs['failed'].job_id}.attempt001.log",
                },
                jobs["partial"].job_id: {
                    "state": "blocked_partial",
                    "log": f"{jobs['partial'].log_dir}/{jobs['partial'].job_id}.attempt001.log",
                },
                jobs["pending"].job_id: {"state": "pending"},
            }
        },
    )
    return root, campaign, jobs


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    snapshot = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        stat = path.stat()
        snapshot[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns, digest)
    return snapshot


def test_status_summary_counts_details_and_read_only_cli(tmp_path: Path):
    root, campaign, jobs = _fixture(tmp_path)
    before = _tree_snapshot(root)

    summary = status.summarize_path(root)
    assert summary["campaign_count"] == 1
    item = summary["campaigns"][0]
    assert item["counts"] == {
        "complete": 1,
        "running": 1,
        "failed": 1,
        "partial": 1,
        "pending": 1,
    }
    assert item["progress"]["percent"] == 20.0
    assert item["receipt_counts"] == {"valid": 1, "invalid": 0, "missing": 4, "orphan": 0}
    assert item["running"] == [
        {
            "job_id": jobs["running"].job_id,
            "condition": "running",
            "task": "ring_hold",
            "seed": 0,
            "gpu": 3,
            "pid": 4242,
            "log": f"{jobs['running'].log_dir}/{jobs['running'].job_id}.attempt001.log",
        }
    ]
    attention = {entry["state"]: entry for entry in item["attention"]}
    assert attention["failed"]["log"].endswith("attempt001.log")
    assert attention["partial"]["log"].endswith("attempt001.log")
    assert item["validation_ok"] is True

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(V2_DIR / "status_p0.py"), str(campaign), "--json"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["aggregate"]["counts"] == item["counts"]
    assert _tree_snapshot(root) == before


def test_human_output_and_corrupt_receipt_detection(tmp_path: Path):
    _, campaign, jobs = _fixture(tmp_path)
    healthy = status.summarize_path(campaign)
    rendered = status._format_human(healthy)
    assert "progress: 1/5 (20.0%)" in rendered
    assert f"gpu=3 pid=4242" in rendered
    assert "failed/partial:" in rendered

    result_json = launch._expected_paths(jobs["complete"], campaign)[0]
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    payload["mutated"] = True
    result_json.write_text(json.dumps(payload), encoding="utf-8")
    corrupted = status.summarize_path(campaign)
    item = corrupted["campaigns"][0]
    assert item["counts"]["complete"] == 0
    assert item["counts"]["partial"] == 2
    assert item["receipt_counts"]["invalid"] == 1
    assert item["validation_ok"] is False
    assert any("invalid completion receipt" in error for error in item["validation_errors"])


def test_invalid_input_is_reported_without_writes(tmp_path: Path):
    before = _tree_snapshot(tmp_path)
    result = subprocess.run(
        [sys.executable, str(V2_DIR / "status_p0.py"), str(tmp_path), "--json"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 2
    assert "error" in json.loads(result.stdout)
    assert _tree_snapshot(tmp_path) == before
