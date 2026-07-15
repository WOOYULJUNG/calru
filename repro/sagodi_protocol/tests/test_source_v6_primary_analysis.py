from __future__ import annotations

from pathlib import Path

import torch

from repro.sagodi_protocol.artifacts import strict_json_load
from repro.sagodi_protocol.primary_v4 import build_v4_model
from repro.sagodi_protocol.sagodi_primary_runner import (
    PrimaryAnalysisSpec,
    _structural_analysis_requested,
    run_primary_analysis,
)
from repro.sagodi_protocol.source_v6_primary_analysis_campaign import load_config


def test_source_v6_core_campaign_freezes_eight_runs() -> None:
    config = load_config()
    assert config["seeds"] == [0]
    assert config["expected_runs"] == 8
    assert config["analysis"]["trajectory_count"] == 256
    assert config["analysis"]["spline_count"] == 128
    assert config["analysis"]["finite_time_and_asymptotic_memory"] is False
    assert config["analysis"]["eligibility_policy"] == "label_only_analyze_all_checkpoints"


def test_source_v6_core_spec_requires_frozen_reduced_sizes() -> None:
    spec = PrimaryAnalysisSpec(
        trajectory_count=256,
        spline_count=128,
        task_horizon=128,
        blank_horizon=2048,
        source_v6=True,
        core_only=True,
    )
    spec.validate()
    assert _structural_analysis_requested(eligible=False, spec=spec) is True
    assert (
        _structural_analysis_requested(
            eligible=False, spec=PrimaryAnalysisSpec(source_v6=True)
        )
        is False
    )


def test_source_v6_lru_checkpoint_runs_primary_smoke(tmp_path: Path) -> None:
    model = build_v4_model("lru_n52")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "checkpoint_type": "unit_source_v6",
            "run": {
                "model_id": "lru_n52",
                "learning_rate": 0.001,
                "actual_state_noise_std": 0.0,
            },
            "result": {
                "status": "completed",
                "final_metrics": {
                    "mse": 0.001,
                    "nmse_db": -25.0,
                    "masked_mse": 0.001,
                    "masked_nmse_db": -25.0,
                },
            },
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    protocol = Path(__file__).resolve().parents[1] / "analysis_protocol.yaml"
    summary_path = run_primary_analysis(
        checkpoint_path=checkpoint,
        protocol_path=protocol,
        output_dir=tmp_path / "analysis",
        device="cpu",
        spec=PrimaryAnalysisSpec(
            trajectory_count=8,
            spline_count=8,
            task_horizon=128,
            blank_horizon=4,
            spectrum_chunk_size=4,
            candidate_distance_chunk_size=64,
            normal_recovery_anchor_count=2,
            normal_recovery_ambient_directions=1,
            normal_recovery_radii_over_manifold_scale=(0.01,),
            normal_recovery_horizons=(0, 1, 4),
            source_v6=True,
            core_only=True,
            smoke=True,
        ),
    )
    summary = strict_json_load(summary_path)
    assert summary["smoke"] is True
    assert summary["analysis_status"] in {
        "complete_core_structural_analysis",
        "structural_analysis_not_estimable",
    }
    assert summary["project_resolutions"]["source_v6_public_code_task"] is True
    if summary["analysis_status"] == "complete_core_structural_analysis":
        assert summary["fixed_point_topology"]["status"] == "omitted_from_pilot_core"
        assert summary["finite_time_angular_memory"]["status"] == "omitted_from_pilot_core"
