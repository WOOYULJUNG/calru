from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


V2_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = V2_DIR.parents[1]
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

import launch_p0 as launch


def load_config() -> dict:
    return json.loads((V2_DIR / "campaign.example.json").read_text(encoding="utf-8"))


def make_campaign(tmp_path: Path, config: dict, jobs: list[launch.Job]) -> tuple[Path, Path]:
    root = tmp_path / "artifacts"
    launch._prepare_artifact_root(root)
    campaign_id, manifest = launch.build_manifest(config, jobs, smoke=False)
    campaign = root / campaign_id
    launch._write_or_check_manifest(campaign, manifest)
    return root, campaign


def write_valid_artifacts(job: launch.Job, campaign: Path) -> None:
    for path in launch._expected_paths(job, campaign):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(
                json.dumps({"task": job.task, "seed": job.seed, "tag": f"v2_{job.condition}"}),
                encoding="utf-8",
            )
        elif path.suffix == ".pt":
            torch.save({"weight": torch.ones(2)}, path)
        elif path.name.endswith("_loss_trace.npz"):
            np.savez_compressed(
                path,
                steps=np.asarray([1, 2], dtype=np.int64),
                train_total_loss=np.asarray([0.4, 0.3], dtype=np.float32),
                train_task_loss=np.asarray([0.35, 0.25], dtype=np.float32),
            )
        elif path.name.endswith("_lambda_trace.npz"):
            np.savez_compressed(
                path,
                steps=np.asarray([0, 2], dtype=np.int64),
                lambdas=np.asarray([[0.9, 0.99], [0.91, 0.999]], dtype=np.float32),
                lambda_gt_0p99=np.asarray([0, 1], dtype=np.int64),
                pan_score_mean=np.asarray([np.nan, 0.1], dtype=np.float32),
                pan_score_max=np.asarray([np.nan, 0.2], dtype=np.float32),
            )
        elif path.name.endswith("_aux_trace.npz"):
            np.savez_compressed(
                path,
                train_total_loss=np.asarray([0.4, 0.3], dtype=np.float32),
                train_task_loss=np.asarray([0.35, 0.25], dtype=np.float32),
                aux_steps=np.asarray([1, 2], dtype=np.int64),
                aux_loss=np.asarray([0.05, 0.04], dtype=np.float32),
            )
        else:  # pragma: no cover - protects this test helper as formats evolve
            raise AssertionError(path)


def test_default_job_matrix_and_safety_flags():
    config = load_config()
    launch._validate_config(config)
    jobs = launch.build_jobs(config)

    assert len(jobs) == 30
    assert sum(job.family == "rp_control" for job in jobs) == 18
    assert sum(job.family == "horizon_information_control" for job in jobs) == 12
    assert all(job.implemented for job in jobs)
    assert all("--force" not in job.command for job in jobs)
    assert all(not Path(path).is_absolute() for job in jobs for path in job.expected)
    assert all("--log-loss-trajectory" in job.command for job in jobs)
    assert all("--deterministic-training" in job.command for job in jobs)
    for job in jobs:
        for flag, expected in (
            ("--train-data-seed-base", "880071"),
            ("--probe-seed-base", "880072"),
            ("--eval-seed-base", "880073"),
        ):
            index = job.command.index(flag)
            assert job.command[index + 1] == expected
    assert all(
        any(path.endswith("_loss_trace.npz") for path in job.expected)
        for job in jobs
        if job.runner != "aux"
    )

    gradient_aux = [job for job in jobs if job.condition == "gradient_lambda_aux"]
    assert len(gradient_aux) == 6
    assert all("--train-pan-theta" in job.command for job in gradient_aux)

    rp_off = [job for job in jobs if job.condition == "rp_off_frozen_retention"]
    assert len(rp_off) == 6
    assert all("--force-all-slow" not in job.command for job in rp_off)

    uniform = [job for job in jobs if job.condition == "uniform_retention_theta_cap"]
    assert len(uniform) == 6
    assert all("--force-all-slow" in job.command for job in uniform)
    for job in uniform:
        index = job.command.index("--all-slow-lambda")
        assert float(job.command[index + 1]) == pytest.approx(launch.UNIFORM_CAP_LAMBDA)


def test_manifest_is_deterministic():
    config = load_config()
    jobs_a = launch.build_jobs(config)
    jobs_b = launch.build_jobs(copy.deepcopy(config))
    campaign_a, manifest_a = launch.build_manifest(config, jobs_a, smoke=False)
    campaign_b, manifest_b = launch.build_manifest(config, jobs_b, smoke=False)
    assert campaign_a == campaign_b
    assert manifest_a == manifest_b


def test_manifest_has_transitive_source_closure_and_software_versions():
    config = load_config()
    _, manifest = launch.build_manifest(config, launch.build_jobs(config), smoke=False)
    required_sources = {
        "repro/legacy_code/exp88_manifold_attractor_tasks.py",
        "repro/legacy_code/exp72_structured_attractor_tasks.py",
        "repro/legacy_code/exp71_pan_block_pulse_hold.py",
        "repro/legacy_code/pan_block.py",
        "repro/legacy_code/plru_regularizers.py",
        "repro/experimental_v2/launch_p0.py",
        "repro/experimental_v2/train_aux_blank.py",
        "repro/experimental_v2/train_fixed_retention.py",
    }
    assert required_sources == set(manifest["source_hash_closure"])
    assert all(len(value) == 64 for value in manifest["source_hash_closure"].values())
    assert {"python", "python_implementation", "numpy", "torch", "torch_cuda_build", "cudnn_build"} <= set(
        manifest["software_versions"]
    )


def test_gpu_runtime_is_outside_campaign_identity(tmp_path: Path):
    config_a = load_config()
    config_b = copy.deepcopy(config_a)
    config_b["gpus"] = [4, 5]
    jobs_a = launch.build_jobs(config_a)
    jobs_b = launch.build_jobs(config_b)
    campaign_a, manifest_a = launch.build_manifest(config_a, jobs_a, smoke=False)
    campaign_b, manifest_b = launch.build_manifest(config_b, jobs_b, smoke=False)
    assert campaign_a == campaign_b
    assert manifest_a == manifest_b

    root = tmp_path / "artifacts"
    launch._prepare_artifact_root(root)
    campaign = root / campaign_a
    launch._write_or_check_manifest(campaign, manifest_a)
    receipt_a = launch._write_or_check_runtime_receipt(
        campaign, launch._runtime_receipt(config_a, max_parallel=3, dry_run=False)
    )
    receipt_b = launch._write_or_check_runtime_receipt(
        campaign, launch._runtime_receipt(config_b, max_parallel=2, dry_run=False)
    )
    assert receipt_a != receipt_b
    assert json.loads(receipt_a.read_text(encoding="utf-8"))["gpus"] == [0, 1, 2]
    assert json.loads(receipt_b.read_text(encoding="utf-8"))["gpus"] == [4, 5]


def test_uniform_cap_is_not_legacy_point_999():
    assert launch.UNIFORM_CAP_LAMBDA > 0.99999
    assert launch.UNIFORM_CAP_LAMBDA**500 > 0.999
    assert 0.999**500 == pytest.approx(0.6063789448611847)
    assert 0.999**1000 == pytest.approx(0.36769542477096373)


def test_modern_models_are_parameter_matched_when_enabled():
    config = load_config()
    launch._apply_enable_overrides(config, {"rg_lru_main", "matched_gru_anchor"}, set())
    jobs = launch.build_jobs(config)
    rg_lru = [job for job in jobs if job.condition == "rg_lru_main"]
    matched_gru = [job for job in jobs if job.condition == "matched_gru_anchor"]
    assert len(rg_lru) == 8 * 3
    assert len(matched_gru) == 2 * 3
    assert all(job.implemented for job in [*rg_lru, *matched_gru])
    assert all(job.worker_model == "RG-LRU-full" for job in rg_lru)
    assert all(job.worker_model == "GRU-full" for job in matched_gru)
    assert {job.metadata["parameter_match"]["candidate_rec_dim"] for job in rg_lru} == {176}
    assert {job.metadata["parameter_match"]["candidate_rec_dim"] for job in matched_gru} == {64}
    assert all(job.metadata["parameter_match"]["relative_error"] < 0.05 for job in rg_lru)
    assert all(job.metadata["parameter_match"]["relative_error"] < 0.05 for job in matched_gru)


def test_compute_matched_label_is_rejected():
    config = load_config()
    config["compute_matched_controls"] = [{"name": "fake_equal_compute"}]
    with pytest.raises(ValueError, match="deliberately unsupported"):
        launch._validate_config(config)


def test_artifact_root_requires_marker_if_nonempty(tmp_path: Path):
    root = tmp_path / "unmarked"
    root.mkdir()
    (root / "unrelated.txt").write_text("do not touch\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-empty unmarked"):
        launch._prepare_artifact_root(root)
    assert (root / "unrelated.txt").read_text(encoding="utf-8") == "do not touch\n"


def test_complete_and_partial_artifact_detection(tmp_path: Path):
    config = load_config()
    job = next(item for item in launch.build_jobs(config) if item.condition == "rp_aligned")
    _, campaign = make_campaign(tmp_path, config, [job])
    state, paths = launch._artifact_state(job, campaign)
    assert state == "missing"
    write_valid_artifacts(job, campaign)
    state, present = launch._artifact_state(job, campaign)
    assert state == "partial"
    assert present == launch._expected_paths(job, campaign)

    receipt = launch._write_completion_receipt(job, campaign)
    assert receipt.is_file()
    state, complete_paths = launch._artifact_state(job, campaign)
    assert state == "complete"
    assert receipt in complete_paths

    result_path = next(path for path in launch._expected_paths(job, campaign) if path.suffix == ".json")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["post_receipt_mutation"] = True
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    state, _ = launch._artifact_state(job, campaign)
    assert state == "partial"


@pytest.mark.parametrize("corruption", ["json", "checkpoint", "npz_missing_key", "npz_empty"])
def test_structural_artifact_validation_rejects_corruption(tmp_path: Path, corruption: str):
    config = load_config()
    job = next(item for item in launch.build_jobs(config) if item.condition == "rp_aligned")
    _, campaign = make_campaign(tmp_path, config, [job])
    write_valid_artifacts(job, campaign)
    paths = launch._expected_paths(job, campaign)
    if corruption == "json":
        next(path for path in paths if path.suffix == ".json").write_text("{broken", encoding="utf-8")
    elif corruption == "checkpoint":
        next(path for path in paths if path.suffix == ".pt").write_bytes(b"not a checkpoint")
    else:
        loss_path = next(path for path in paths if path.name.endswith("_loss_trace.npz"))
        if corruption == "npz_missing_key":
            np.savez_compressed(
                loss_path,
                steps=np.asarray([1]),
                train_total_loss=np.asarray([0.1]),
            )
        else:
            np.savez_compressed(
                loss_path,
                steps=np.asarray([], dtype=np.int64),
                train_total_loss=np.asarray([], dtype=np.float32),
                train_task_loss=np.asarray([], dtype=np.float32),
            )
    with pytest.raises(RuntimeError):
        launch._validate_expected_artifacts(job, campaign)


def test_successful_worker_publishes_receipt_last(tmp_path: Path):
    script = """
from pathlib import Path
import json
import numpy as np
import torch
root = Path(r'<CAMPAIGN_DIR>')
(root / 'results/test').mkdir(parents=True, exist_ok=True)
(root / 'checkpoints/test').mkdir(parents=True, exist_ok=True)
(root / 'traces/test').mkdir(parents=True, exist_ok=True)
(root / 'results/test/result.json').write_text(json.dumps({'task':'ring_hold','seed':0,'tag':'v2_test'}))
torch.save({'weight': torch.ones(1)}, root / 'checkpoints/test/model.pt')
np.savez_compressed(root / 'traces/test/ring_hold_v2_test_seed0_loss_trace.npz',
    steps=np.asarray([1]), train_total_loss=np.asarray([0.2]), train_task_loss=np.asarray([0.2]))
"""
    job = launch.Job(
        job_id="test-ring-hold-s0-deadbeef00",
        family="test",
        condition="test",
        task="ring_hold",
        model="test",
        worker_model="test",
        seed=0,
        runner="legacy",
        command=("<PYTHON>", "-c", script),
        implemented=True,
        implementation_note="test fixture",
        expected=(
            "results/test/result.json",
            "checkpoints/test/model.pt",
            "traces/test/ring_hold_v2_test_seed0_loss_trace.npz",
        ),
        log_dir="logs/test",
        metadata={},
    )
    config = load_config()
    config["gpus"] = [0]
    root, campaign = make_campaign(tmp_path, config, [job])
    code = launch.run_jobs(
        jobs=[job],
        root=root,
        campaign_dir=campaign,
        gpus=[0],
        max_parallel=1,
        poll_seconds=0.05,
    )
    assert code == 0
    receipt = launch._completion_receipt_path(job, campaign)
    assert receipt.is_file()
    assert launch._artifact_state(job, campaign)[0] == "complete"
    status = json.loads((campaign / "status.json").read_text(encoding="utf-8"))
    assert status["jobs"][job.job_id]["state"] == "complete"
    assert status["jobs"][job.job_id]["completion_receipt"] == str(receipt.relative_to(campaign))


def test_process_group_shutdown_escalates_and_reaps():
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ready', flush=True); time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        launch._terminate_process_groups([process], grace_seconds=0.05)
        assert process.poll() is not None
        assert process.returncode == -signal.SIGKILL
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)


def test_cli_dry_run_writes_only_v2_root(tmp_path: Path):
    artifact_root = tmp_path / "v2-artifacts"
    result = subprocess.run(
        [
            sys.executable,
            str(V2_DIR / "launch_p0.py"),
            "--config",
            str(V2_DIR / "campaign.example.json"),
            "--artifact-root",
            str(artifact_root),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "jobs=30 implemented=30 blocked=0" in result.stdout
    assert (artifact_root / launch.ROOT_MARKER).read_text(encoding="utf-8") == launch.ROOT_MARKER_CONTENT
    manifests = list(artifact_root.glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert len(manifest["jobs"]) == 30

    subset = subprocess.run(
        [
            sys.executable,
            str(V2_DIR / "launch_p0.py"),
            "--config",
            str(V2_DIR / "campaign.example.json"),
            "--artifact-root",
            str(artifact_root),
            "--only",
            "rp_aligned",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert subset.returncode == 0, subset.stdout
    assert "jobs=6 implemented=6 blocked=0" in subset.stdout
    assert f"campaign={manifest['campaign_id']}" in subset.stdout
    assert len(list(artifact_root.glob("*/manifest.json"))) == 1
    runtime_receipts = list(manifests[0].parent.glob("runtime_receipts/*.json"))
    assert len(runtime_receipts) == 2
    selected_counts = sorted(
        len(json.loads(path.read_text(encoding="utf-8"))["selected_job_ids"])
        for path in runtime_receipts
    )
    assert selected_counts == [6, 30]
