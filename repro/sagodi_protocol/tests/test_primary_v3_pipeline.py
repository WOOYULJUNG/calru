from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from repro.sagodi_protocol.artifacts import (
    strict_json_load,
    write_completion_receipt,
)
from repro.sagodi_protocol import primary_v3_pipeline as pipeline


def _default_sources() -> pipeline.PipelineSources:
    return pipeline.PipelineSources(
        selector=pipeline.DEFAULT_SELECTOR,
        protocol=pipeline.DEFAULT_PROTOCOL,
        main_template=pipeline.DEFAULT_MAIN_TEMPLATE,
        engineering_freeze=pipeline.DEFAULT_ENGINEERING_FREEZE,
        association_freeze=pipeline.DEFAULT_ASSOCIATION_FREEZE,
    )


def test_stage_plan_has_exact_order_roots_and_no_smoke(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    sources = pipeline._copied_sources(root)
    stages = pipeline.build_stage_plan(
        artifact_root=root,
        python="/opt/frozen/python",
        gpus=(5, 2),
        sources=sources,
    )

    assert tuple(stage.name for stage in stages) == pipeline.STAGE_ORDER
    assert tuple(stage.artifact_root.name for stage in stages) == (
        "selector",
        "main",
        "primary_analysis",
        "engineering_benefit",
        "dynamics_utility_association",
    )
    for stage in stages:
        assert stage.command[:3] == (
            "/opt/frozen/python",
            "-m",
            stage.module,
        )
        assert "--smoke" not in stage.command
    for stage in stages[:4]:
        gpu_index = stage.command.index("--gpus")
        assert stage.command[gpu_index + 1] == "5,2"
    assert "--gpus" not in stages[-1].command
    assert "--python" not in stages[-1].command


def test_dry_run_is_side_effect_free_and_uses_immutable_copy_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "not-created"
    plan = pipeline.build_dry_run_plan(
        artifact_root=root,
        python=sys.executable,
        gpus=(0, 1),
        sources=_default_sources(),
    )

    assert plan["dry_run"] is True
    assert plan["execution_performed"] is False
    assert plan["stage_order"] == list(pipeline.STAGE_ORDER)
    assert not root.exists()
    selector_command = plan["stages"][0]["command"]
    assert str(root / "inputs" / "selector.json") in selector_command
    assert all(
        len(item["sha256"]) == 64 for item in plan["source_files"].values()
    )


def test_cli_dry_run_does_not_create_artifact_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "planned"
    assert (
        pipeline.main(
            [
                "--artifact-root",
                str(root),
                "--python",
                sys.executable,
                "--gpus",
                "3,1",
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["physical_gpu_ids"] == [3, 1]
    assert payload["clean_committed_worktree_check"] == "deferred_to_full_execution"
    assert not root.exists()


def test_run_stage_records_exact_command_timestamps_return_and_receipts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pipeline"
    root.mkdir()
    child = root / "selector"
    child.mkdir()
    payload = child / "payload.json"
    payload.write_text("{}\n")
    (child / "COMPLETE").write_text("complete\n")
    write_completion_receipt(
        child / "completion_receipt.json",
        job_id="fake-child",
        artifacts=[payload, child / "COMPLETE"],
    )
    stage = pipeline.PipelineStage(
        name="selector",
        module="fake.module",
        artifact_root=child,
        command=(sys.executable, "-c", "print('ok')"),
    )
    manifest = {"scientific_identity": "a" * 64}
    status = pipeline._new_status(manifest, (stage,))
    # _new_status normally receives all five stages; narrow it for this direct
    # execution unit and retain the same command/attempt contract.
    status["stage_order"] = ["selector"]

    pipeline._run_stage(
        root=root,
        repo_root=Path(__file__).resolve().parents[3],
        stage=stage,
        status=status,
    )

    attempt = status["stages"]["selector"]["attempts"][0]
    assert attempt["command"] == list(stage.command)
    assert attempt["return_code"] == 0
    assert attempt["started_at_utc"].endswith("Z")
    assert attempt["ended_at_utc"].endswith("Z")
    assert len(attempt["log_sha256"]) == 64
    assert len(attempt["complete_sha256"]) == 64
    assert status["stages"]["selector"]["status"] == "complete"
    assert strict_json_load(root / pipeline.STATUS)["state"] == "running"


def test_run_stage_fails_on_child_return_code_and_records_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pipeline"
    root.mkdir()
    child = root / "main"
    stage = pipeline.PipelineStage(
        name="main",
        module="fake.module",
        artifact_root=child,
        command=(sys.executable, "-c", "raise SystemExit(7)"),
    )
    status = {
        "schema_version": 1,
        "pipeline_id": pipeline.PIPELINE_ID,
        "pipeline_scientific_identity": "b" * 64,
        "state": "pending",
        "current_stage": None,
        "stage_order": ["main"],
        "stages": {
            "main": {
                "status": "pending",
                "command": list(stage.command),
                "command_display": "ignored",
                "attempts": [],
            }
        },
    }

    with pytest.raises(pipeline.PipelineStageError) as captured:
        pipeline._run_stage(
            root=root,
            repo_root=Path(__file__).resolve().parents[3],
            stage=stage,
            status=status,
        )

    assert captured.value.return_code == 7
    on_disk = strict_json_load(root / pipeline.STATUS)
    assert on_disk["state"] == "failed"
    assert on_disk["current_stage"] == "main"
    assert on_disk["stages"]["main"]["attempts"][0]["return_code"] == 7


def test_pipeline_stops_at_first_failed_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        pipeline,
        "_git_state",
        lambda _repo: {"code_commit": "c" * 40, "worktree_dirty": False},
    )
    monkeypatch.setattr(pipeline, "_verify_frozen_inputs", lambda **_kwargs: None)

    def fake_run_stage(**kwargs: object) -> None:
        stage = kwargs["stage"]
        assert isinstance(stage, pipeline.PipelineStage)
        calls.append(stage.name)
        if stage.name == "main":
            raise pipeline.PipelineStageError("main", 9, "synthetic failure")

    monkeypatch.setattr(pipeline, "_run_stage", fake_run_stage)

    with pytest.raises(pipeline.PipelineStageError):
        pipeline.run_primary_v3_pipeline(
            artifact_root=tmp_path / "campaign",
            python=sys.executable,
            gpus=(0,),
            sources=_default_sources(),
        )

    assert calls == ["selector", "main"]


def test_full_pipeline_rejects_dirty_worktree_before_creating_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pipeline,
        "_git_state",
        lambda _repo: {"code_commit": "d" * 40, "worktree_dirty": True},
    )
    root = tmp_path / "campaign"
    with pytest.raises(RuntimeError, match="clean committed worktree"):
        pipeline.run_primary_v3_pipeline(
            artifact_root=root,
            python=sys.executable,
            gpus=(0,),
            sources=_default_sources(),
        )
    assert not root.exists()


def test_missing_status_cannot_discard_prior_attempt_history(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    (root / "logs").mkdir(parents=True)
    manifest = {"scientific_identity": "e" * 64}
    stages = pipeline.build_stage_plan(
        artifact_root=root,
        python=sys.executable,
        gpus=(0,),
        sources=pipeline._copied_sources(root),
    )

    with pytest.raises(RuntimeError, match="refusing to discard attempt history"):
        pipeline._load_or_initialize_status(
            root / pipeline.STATUS,
            manifest,
            stages,
        )
