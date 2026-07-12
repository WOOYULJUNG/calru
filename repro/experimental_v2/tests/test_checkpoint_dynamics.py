from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from repro.experimental_v2.evaluate_checkpoint_dynamics import (
    CheckpointSpec,
    PlanItem,
    StateLayout,
    finalize_output,
    deterministic_directions,
    load_specs,
    make_assets,
    snapshot_file,
    validate_plan,
    verify_source_snapshot,
)


class DummyFullBlock:
    state_size = 7
    recurrent_state_size = 4
    readout_slice = slice(4, 7)


class CheckpointDynamicsTests(unittest.TestCase):
    def test_layout_keeps_stream_exactly_unchanged(self):
        layout = StateLayout.from_model(DummyFullBlock())
        state = torch.arange(14, dtype=torch.float32).reshape(2, 7)
        replacement = torch.full((2, 4), -3.0)
        changed = layout.replace_carrier(state, replacement)
        self.assertTrue(torch.equal(changed[:, :4], replacement))
        self.assertTrue(torch.equal(changed[:, 4:], state[:, 4:]))
        embedded = layout.embed_direction(torch.ones(2, 4))
        self.assertTrue(torch.equal(embedded[:, 4:], torch.zeros(2, 3)))

    def test_directions_are_unit_and_carrier_normal(self):
        basis = torch.zeros(3, 6, 2)
        basis[:, 0, 0] = 1.0
        basis[:, 1, 1] = 1.0
        tangent, normal = deterministic_directions(basis, 4, 5, seed=17)
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(tangent, dim=-1), torch.ones(3, 4)))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(normal, dim=-1), torch.ones(3, 5)))
        overlap = torch.einsum("bkc,bcq->bkq", normal, basis)
        self.assertLess(float(overlap.abs().max()), 1e-6)

    def test_assets_are_deterministic_and_task_specific(self):
        first = make_assets("torus_integrate", 16, 7, 20, 1234)
        second = make_assets("torus_integrate", 16, 7, 20, 1234)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first["q_family"].shape, (16, 2))
        self.assertEqual(first["q_eval"].shape, (7, 2))
        self.assertEqual(first["v_eval"].shape, (20, 7, 2))
        self.assertGreater(float(np.abs(first["v_eval"]).sum()), 0.0)
        # The deterministic schedule reserves a final hold segment.
        np.testing.assert_array_equal(first["v_eval"][-4:], np.zeros((4, 7, 2), dtype=np.float32))

    def test_output_cannot_be_inside_legacy_input_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "results"
            checkpoint_dir = root / "checkpoints"
            result_dir.mkdir()
            checkpoint_dir.mkdir()
            result = result_dir / "ring_hold_x_seed0.json"
            checkpoint = checkpoint_dir / "exp88_ring_hold_x_seed0.pt"
            result.write_text("{}")
            checkpoint.write_bytes(b"checkpoint")
            spec = CheckpointSpec("X", "ring_hold", "x", "results", "checkpoints")
            plan = [PlanItem(spec, 0, result, checkpoint)]
            with self.assertRaises(ValueError):
                validate_plan(plan, root, result_dir / "new-analysis")

    def test_spec_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "checkpoints": [
                            {
                                "paper_model": "X",
                                "task": "ring_hold",
                                "tag": "x",
                                "result_dir": "../escaped-results",
                                "checkpoint_dir": "checkpoints",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "without '\\.\\.'"):
                load_specs(path)

    def test_frozen_source_change_blocks_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            path.write_bytes(b"initial")
            frozen = snapshot_file(path, "checkpoint")
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "source snapshot verification failed"):
                verify_source_snapshot([frozen])

    def test_complete_marker_is_last_and_binds_hash_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            output.mkdir()
            (output / "INCOMPLETE").write_text("in progress\n", encoding="utf-8")
            (output / "metrics.csv").write_text("value\n1\n", encoding="utf-8")
            finalize_output(output)
            self.assertFalse((output / "INCOMPLETE").exists())
            self.assertTrue((output / "COMPLETE").is_file())
            sums = (output / "SHA256SUMS").read_bytes()
            marker = json.loads((output / "COMPLETE").read_text(encoding="utf-8"))
            self.assertEqual(marker["sha256sums_sha256"], hashlib.sha256(sums).hexdigest())
            manifest_text = sums.decode("utf-8")
            self.assertIn("metrics.csv", manifest_text)
            self.assertNotIn("INCOMPLETE", manifest_text)
            self.assertNotIn("COMPLETE", manifest_text)


if __name__ == "__main__":
    unittest.main()
