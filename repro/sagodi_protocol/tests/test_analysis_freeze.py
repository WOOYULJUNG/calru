from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

from repro.sagodi_protocol import analysis_freeze


class AnalysisFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.freeze = analysis_freeze.load_analysis_freeze()

    def _write(self, payload: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "freeze.yaml"
        path.write_text(payload, encoding="utf-8")
        return path

    def test_default_freeze_is_valid_and_parent_bound(self) -> None:
        self.assertEqual(self.freeze["freeze_status"], "pilot_reanalysis_only")
        self.assertEqual(
            self.freeze["parent_training"],
            analysis_freeze.PARENT_TRAINING_BINDING,
        )
        self.assertEqual(
            self.freeze["c3_normal_recovery"]["registered_horizons"],
            [1, 5, 20, 100, 500, 1024],
        )
        fingerprint = analysis_freeze.analysis_freeze_fingerprint(self.freeze)
        self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")
        self.assertEqual(
            fingerprint,
            analysis_freeze.analysis_freeze_fingerprint(
                analysis_freeze.load_analysis_freeze()
            ),
        )

    def test_tamper_changes_canonical_fingerprint(self) -> None:
        tampered = copy.deepcopy(self.freeze)
        tampered["c3_normal_recovery"]["primary_horizon"] = 100
        self.assertNotEqual(
            analysis_freeze.analysis_freeze_fingerprint(self.freeze),
            analysis_freeze.analysis_freeze_fingerprint(tampered),
        )
        with self.assertRaisesRegex(
            analysis_freeze.AnalysisFreezeError, "primary_horizon"
        ):
            analysis_freeze.validate_analysis_freeze(tampered)

    def test_loader_rejects_duplicate_key(self) -> None:
        path = self._write(
            '{"schema_version":"1.0.0","schema_version":"1.0.0"}'
        )
        with self.assertRaisesRegex(
            analysis_freeze.AnalysisFreezeError, "duplicate JSON key"
        ):
            analysis_freeze.load_analysis_freeze(path)

    def test_loader_rejects_nonfinite_constant(self) -> None:
        path = self._write('{"schema_version":"1.0.0","value":NaN}')
        with self.assertRaisesRegex(
            analysis_freeze.AnalysisFreezeError, "non-finite JSON constant"
        ):
            analysis_freeze.load_analysis_freeze(path)
        in_memory = copy.deepcopy(self.freeze)
        in_memory["settling"]["positive_expansion"]["threshold"]["value"] = math.inf
        with self.assertRaisesRegex(
            analysis_freeze.AnalysisFreezeError, "non-finite"
        ):
            analysis_freeze.validate_analysis_freeze(in_memory)

    def test_missing_and_extra_keys_are_rejected(self) -> None:
        missing = copy.deepcopy(self.freeze)
        del missing["claim_scope"]["may_support_c4_claim"]
        with self.assertRaisesRegex(analysis_freeze.AnalysisFreezeError, "missing keys"):
            analysis_freeze.validate_analysis_freeze(missing)

        extra = copy.deepcopy(self.freeze)
        extra["manifold"]["primary"]["fallback"] = "task_atlas"
        with self.assertRaisesRegex(analysis_freeze.AnalysisFreezeError, "extra keys"):
            analysis_freeze.validate_analysis_freeze(extra)

    def test_invalid_hash_and_parent_contradiction_are_rejected(self) -> None:
        invalid = copy.deepcopy(self.freeze)
        invalid["parent_training"]["manifest_sha256"] = "not-a-sha"
        with self.assertRaisesRegex(
            analysis_freeze.AnalysisFreezeError, "manifest_sha256"
        ):
            analysis_freeze.validate_analysis_freeze(invalid)

        mismatched = copy.deepcopy(self.freeze)
        mismatched["parent_training"]["campaign_id"] = (
            "calru_native_sagodi_ring_pilot_v1-deadbeefdead"
        )
        with self.assertRaisesRegex(
            analysis_freeze.AnalysisFreezeError, "campaign_id"
        ):
            analysis_freeze.validate_analysis_freeze(mismatched)

    def test_semantic_contradictions_are_rejected(self) -> None:
        weak_geometry = copy.deepcopy(self.freeze)
        weak_geometry["manifold"]["primary"]["quality_gates"][
            "minimum_coarse_bin_occupancy_fraction"
        ] = 0.5
        with self.assertRaisesRegex(
            analysis_freeze.AnalysisFreezeError,
            "minimum_coarse_bin_occupancy_fraction",
        ):
            analysis_freeze.validate_analysis_freeze(weak_geometry)

        task_atlas_primary = copy.deepcopy(self.freeze)
        task_atlas_primary["manifold"]["task_atlas"]["may_be_primary"] = True
        with self.assertRaisesRegex(
            analysis_freeze.AnalysisFreezeError, "may_be_primary"
        ):
            analysis_freeze.validate_analysis_freeze(task_atlas_primary)

        legacy_gate = copy.deepcopy(self.freeze)
        legacy_gate["c3_normal_recovery"]["legacy_paired_endpoint_metric"][
            "may_satisfy_c3_gate"
        ] = True
        with self.assertRaisesRegex(
            analysis_freeze.AnalysisFreezeError, "may_satisfy_c3_gate"
        ):
            analysis_freeze.validate_analysis_freeze(legacy_gate)

        confirmatory = copy.deepcopy(self.freeze)
        confirmatory["claim_scope"]["confirmatory"] = True
        with self.assertRaisesRegex(
            analysis_freeze.AnalysisFreezeError, "confirmatory"
        ):
            analysis_freeze.validate_analysis_freeze(confirmatory)


if __name__ == "__main__":
    unittest.main()
