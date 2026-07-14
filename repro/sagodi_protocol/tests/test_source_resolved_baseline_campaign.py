from __future__ import annotations

import json
from pathlib import Path

import pytest

from repro.sagodi_protocol.artifacts import atomic_json
from repro.sagodi_protocol.source_resolved_baseline_campaign import (
    SENTINEL_SEED,
    build_plan,
    run_campaign,
    summarize_gate,
)
from repro.sagodi_protocol.source_resolved_protocol import SOURCE_MODEL_IDS


def test_baseline_plan_registers_one_full_sentinel_per_model(tmp_path: Path) -> None:
    bank = tmp_path / "bank.npz"
    plan = build_plan(tmp_path, bank, smoke=False)
    assert [spec.model_id for spec in plan] == list(SOURCE_MODEL_IDS)
    assert {spec.model_seed for spec in plan} == {SENTINEL_SEED}
    assert {spec.updates for spec in plan} == {5000}
    assert {spec.batch_size for spec in plan} == {64}


def _write_result(spec, mse: float) -> None:
    output = Path(spec.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "result.json",
        {
            "run_id": spec.run_id,
            "model_id": spec.model_id,
            "final_metrics": {"masked_mse": mse},
        },
    )


def test_gate_requires_all_three_models_below_threshold(tmp_path: Path) -> None:
    plan = build_plan(tmp_path, tmp_path / "bank.npz", smoke=False)
    for index, spec in enumerate(plan):
        _write_result(spec, 0.001 if index < 2 else 0.02)
    summary = summarize_gate(plan)
    assert summary["all_source_baselines_passed"] is False
    assert summary["failed_models"] == [plan[-1].model_id]


def test_cpu_smoke_writes_verified_stage_bundle(tmp_path: Path) -> None:
    stage = run_campaign(
        stage="smoke",
        artifact_root=tmp_path / "campaign",
        config_source=Path(
            "repro/sagodi_protocol/sagodi_source_resolved_v1.json"
        ).resolve(),
        compute_slots=("cpu",),
    )
    assert (stage / "EXECUTION_COMPLETE").is_file()
    assert (stage / "completion_receipt.json").is_file()
    status = json.loads((stage / "status.json").read_text(encoding="utf-8"))
    assert status["verified_complete"] == 3


def test_unknown_stage_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stage"):
        run_campaign(
            stage="main",
            artifact_root=tmp_path / "campaign",
            config_source=Path(
                "repro/sagodi_protocol/sagodi_source_resolved_v1.json"
            ).resolve(),
            compute_slots=("cpu",),
        )
