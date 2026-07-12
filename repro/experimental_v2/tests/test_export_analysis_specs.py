from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from repro.experimental_v2.export_analysis_specs import (
    campaign_payload,
    legacy_payload,
)


def campaign_job(seed: int, expected_prefix: str = "") -> dict:
    return {
        "condition": "rp_aligned",
        "family": "rp_control",
        "implemented": True,
        "model": "PAN-RNW-full",
        "task": "ring_hold",
        "seed": seed,
        "expected": [
            f"{expected_prefix}results/rp_aligned/ring_hold_v2_rp_aligned_seed{seed}.json",
            f"checkpoints/rp_aligned/exp88_ring_hold_v2_rp_aligned_seed{seed}.pt",
        ],
    }


class ExportAnalysisSpecsTests(unittest.TestCase):
    def test_campaign_manifest_groups_seed_jobs_into_one_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "campaign_id": "unit-campaign",
                        "jobs": [campaign_job(0), campaign_job(1)],
                    }
                ),
                encoding="utf-8",
            )
            payload, artifact_root, coverage = campaign_payload(manifest, seeds=[0, 1])
            self.assertEqual(artifact_root, root.resolve())
            self.assertEqual(len(payload["checkpoints"]), 1)
            spec = payload["checkpoints"][0]
            self.assertEqual(spec["paper_model"], "rp_aligned")
            self.assertEqual(spec["tag"], "v2_rp_aligned")
            self.assertEqual(spec["result_dir"], "results/rp_aligned")
            self.assertEqual(coverage["rp_aligned|ring_hold|v2_rp_aligned"], [0, 1])

    def test_campaign_manifest_rejects_escaped_expected_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"schema_version": 1, "jobs": [campaign_job(0, "../")]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "without '\\.\\.'"):
                campaign_payload(manifest)

    def test_legacy_mode_can_require_real_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "results"
            checkpoint_dir = root / "checkpoints"
            result_dir.mkdir()
            checkpoint_dir.mkdir()
            (result_dir / "ring_hold_x_seed0.json").write_text("{}", encoding="utf-8")
            (checkpoint_dir / "exp88_ring_hold_x_seed0.pt").write_bytes(b"checkpoint")
            template = root / "template.json"
            template.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "checkpoints": [
                            {
                                "paper_model": "X",
                                "task": "ring_hold",
                                "tag": "x",
                                "result_dir": "results",
                                "checkpoint_dir": "checkpoints",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload, artifact_root, coverage = legacy_payload(
                root,
                template=template,
                seeds=[0],
                require_artifacts=True,
            )
            self.assertEqual(artifact_root, root.resolve())
            self.assertEqual(len(payload["checkpoints"]), 1)
            self.assertEqual(coverage["X|ring_hold|x"], [0])


if __name__ == "__main__":
    unittest.main()
