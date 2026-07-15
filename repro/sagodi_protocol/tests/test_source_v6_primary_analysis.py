from __future__ import annotations

from pathlib import Path

import torch

from repro.sagodi_protocol.artifacts import strict_json_load
from repro.sagodi_protocol.primary_v4 import build_v4_model
from repro.sagodi_protocol.sagodi_primary_runner import PrimaryAnalysisSpec, run_primary_analysis


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
            smoke=True,
        ),
    )
    summary = strict_json_load(summary_path)
    assert summary["smoke"] is True
    assert summary["analysis_status"] in {
        "complete_structural_summary_eligible",
        "structural_analysis_not_estimable",
    }
    assert summary["project_resolutions"]["source_v6_public_code_task"] is True
