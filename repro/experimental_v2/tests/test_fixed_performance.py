from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from repro.experimental_v2.evaluate_fixed_performance import (
    TASKS,
    apply_retention_permutation,
    build_profiles,
    create_assets,
    deterministic_velocity,
    discover_jobs,
    fixed_q0,
    read_model_metadata,
    run_evaluation,
    sha256_array,
    task_io_dims,
)


LEGACY_TRAINING = {
    "id_horizon": 6,
    "temporal_horizons": [9],
    "velocity_scales": [1.0, 1.5],
    "hold_min": 1,
    "hold_max": 2,
    "move_min": 1,
    "move_max": 2,
    "final_hold_min": 1,
    "final_hold_max": 2,
    "ood_hold_min": 2,
    "ood_hold_max": 3,
    "ood_final_hold_min": 2,
    "ood_final_hold_max": 3,
    "ring_velocity_deg": 3.0,
    "torus_velocity_deg": 2.5,
    "curve_velocity_deg": 2.5,
    "surface_velocity_scale": 0.018,
}


def _build_variant(name: str, task: str, d_model: int = 8, rec_dim: int = 8):
    from repro.legacy_code.exp71_pan_block_pulse_hold import build_model_variant, normalize_model_variant
    from repro.legacy_code.exp88_manifold_attractor_tasks import model_rank_for_task

    input_dim, output_dim = task_io_dims(task)
    return build_model_variant(
        variant=normalize_model_variant(name),
        input_dim=input_dim,
        output_dim=output_dim,
        rank=model_rank_for_task(task),
        d_model=d_model,
        rec_dim=rec_dim,
        layers=1,
        dropout=0.0,
        plru_tau=0.001,
        plru_c=50.0,
        pan_lambda_min=0.90,
        pan_lambda_max=0.999,
        rank_matched_lambda_high=0.999,
        rank_matched_lambda_low=0.0,
    )


def _fake_campaign(root: Path, model_name: str = "GRU-full") -> tuple[Path, dict]:
    task, condition, tag, seed = "ring_hold", "test_condition", "v2_test_condition", 0
    campaign = root / "campaign"
    result_rel = Path("results") / condition / f"{task}_{tag}_seed{seed}.json"
    checkpoint_rel = Path("checkpoints") / condition / f"exp88_{task}_{tag}_seed{seed}.pt"
    (campaign / result_rel).parent.mkdir(parents=True)
    (campaign / checkpoint_rel).parent.mkdir(parents=True)
    model = _build_variant(model_name, task)
    torch.save(model.state_dict(), campaign / checkpoint_rel)
    input_dim, output_dim = task_io_dims(task)
    metadata = {
        "task": task,
        "model": model_name,
        "tag": tag,
        "seed": seed,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "rank_for_model": 2,
        "d_model": 8,
        "rec_dim": 8,
        "layers": 1,
        "params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        # Poisoned training-time metrics must never enter fixed evaluation.
        "postH1000_ambient_component_rmse": 987654321.0,
        "train_loss_final": 987654321.0,
    }
    (campaign / result_rel).write_text(json.dumps(metadata), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "campaign_id": "unit-campaign",
        "scientific_config": {"config": {"training": LEGACY_TRAINING}, "smoke": True},
        "jobs": [
            {
                "job_id": "test-job",
                "family": "unit",
                "condition": condition,
                "task": task,
                "model": model_name,
                "worker_model": model_name,
                "seed": seed,
                "implemented": True,
                "expected": [result_rel.as_posix(), checkpoint_rel.as_posix()],
            }
        ],
    }
    (campaign / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return campaign, metadata


class FixedPerformanceTests(unittest.TestCase):
    def test_fixed_assets_are_deterministic_and_velocity_profiles_share_schedule(self):
        profiles = build_profiles(
            LEGACY_TRAINING,
            id_horizon=None,
            temporal_horizons=None,
            velocity_scales=None,
            smoke=False,
        )
        q_first = fixed_q0("torus_integrate", 11, 42)
        q_second = fixed_q0("torus_integrate", 11, 42)
        np.testing.assert_array_equal(q_first, q_second)
        id_profile = next(profile for profile in profiles if profile.name == "id")
        scaled_profile = next(profile for profile in profiles if profile.name == "velocity_1p5x")
        v_id, schedule_id = deterministic_velocity(
            "torus_integrate", q_first, id_profile, LEGACY_TRAINING, 42
        )
        v_scaled, schedule_scaled = deterministic_velocity(
            "torus_integrate", q_first, scaled_profile, LEGACY_TRAINING, 42
        )
        np.testing.assert_array_equal(schedule_id, schedule_scaled)
        moving = np.abs(v_id) > 1e-10
        np.testing.assert_allclose(v_scaled[moving], 1.5 * v_id[moving], rtol=2e-5, atol=2e-6)
        self.assertEqual(sha256_array(q_first), sha256_array(q_second))

    def test_asset_writer_covers_all_eight_tasks_with_q_velocity_schedule(self):
        profiles = build_profiles(
            LEGACY_TRAINING,
            id_horizon=5,
            temporal_horizons=[7],
            velocity_scales=[1.0],
            smoke=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fresh"
            output.mkdir()
            _, manifest = create_assets(output, profiles, LEGACY_TRAINING, eval_points=4, base_seed=7)
            self.assertEqual(set(manifest["tasks"]), set(TASKS))
            for task in TASKS:
                task_row = manifest["tasks"][task]
                self.assertTrue((output / task_row["q0"]["path"]).is_file())
                self.assertIn("id", task_row["profiles"])
                for profile_row in task_row["profiles"].values():
                    self.assertTrue((output / profile_row["velocity"]["path"]).is_file())
                    self.assertTrue((output / profile_row["schedule"]["path"]).is_file())

    def test_manifest_discovery_filters_and_strict_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _ = _fake_campaign(Path(tmp))
            _, jobs = discover_jobs(
                campaign,
                conditions=["test_condition"],
                models=["GRU-full"],
                tasks=["ring_hold"],
                seeds=[0],
            )
            self.assertEqual([job.job_id for job in jobs], ["test-job"])
            read_model_metadata(jobs[0])
            payload = json.loads(jobs[0].result_path.read_text())
            payload["rec_dim"] = "not-an-integer"
            jobs[0].result_path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                read_model_metadata(jobs[0])

    def test_retention_permutation_preserves_exact_lambda_multiset(self):
        model = _build_variant("PAN-RNW-full", "ring_hold")
        before = model.lam_mag().detach().cpu()
        payload = apply_retention_permutation(model, seed=12345)
        after = model.lam_mag().detach().cpu()
        self.assertTrue(torch.equal(torch.sort(before).values, torch.sort(after).values))
        self.assertTrue(payload["retention_multiset_preserved"])
        self.assertEqual(payload["retention_coordinate_count"], before.numel())
        self.assertNotEqual(payload["retention_order_sha256_before"], payload["retention_order_sha256_after"])

    def test_end_to_end_is_fresh_atomic_and_ignores_training_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign, _ = _fake_campaign(root)
            output = root / "fixed-eval"
            result = run_evaluation(
                campaign_dir=campaign,
                output_dir=output,
                eval_points=4,
                blank_horizon=1000,
                id_horizon=4,
                temporal_horizons=[5],
                velocity_scales=[1.0],
                device_name="cpu",
                smoke=False,
            )
            self.assertEqual(result["status"], "complete")
            self.assertTrue((output / "COMPLETE").is_file())
            self.assertTrue((output / "SHA256SUMS").is_file())
            self.assertFalse((output / "INCOMPLETE").exists())
            with (output / "fixed_performance.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)  # ID plus one temporal profile.
            self.assertNotIn("train_loss_final", rows[0])
            self.assertNotEqual(float(rows[0]["post_blank_component_rmse"]), 987654321.0)
            self.assertEqual(
                float(rows[0]["postH1000_ambient_component_rmse"]),
                float(rows[0]["post_blank_component_rmse"]),
            )
            self.assertIn("paired_H0_to_H1000_component_drift_rmse", rows[0])
            run_manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertFalse(run_manifest["training_metrics_reused"])
            with self.assertRaises(FileExistsError):
                run_evaluation(campaign_dir=campaign, output_dir=output, dry_run=True)

    def test_retention_permutation_rejects_non_pan_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign, _ = _fake_campaign(root, model_name="GRU-full")
            output = root / "must-not-exist"
            with self.assertRaises(ValueError):
                run_evaluation(
                    campaign_dir=campaign,
                    output_dir=output,
                    retention_permutation=True,
                    dry_run=True,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
