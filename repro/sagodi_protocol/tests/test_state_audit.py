from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_DIR = REPO_ROOT / "repro" / "legacy_code"
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

from exp71_pan_block_pulse_hold import RealDiagLRUUnitBaseline  # noqa: E402
from pan_block import FullBlockSequenceModel  # noqa: E402
from repro.sagodi_protocol.audit import (  # noqa: E402
    AuditConfig,
    architecture_exactness_screen,
    check_autograd_jvp,
    check_no_external_hidden_cache,
    run_phase0_audit,
)
from repro.sagodi_protocol.state import StateAdapter, StateParts  # noqa: E402


class LinearStepModel(nn.Module):
    """Small exact map used to make the numerical audit independently testable."""

    def __init__(self):
        super().__init__()
        self.input_dim = 2
        self.output_dim = 1
        self.register_buffer("A", torch.diag(torch.tensor([0.7, 0.8, 0.9])))
        self.register_buffer(
            "B",
            torch.tensor(
                [
                    [0.25, -0.1],
                    [0.0, 0.3],
                    [-0.2, 0.15],
                ]
            ),
        )
        self.readout = nn.Linear(3, 1, bias=False)

    @property
    def state_size(self):
        return 3

    def init_state(self, batch, device):
        return torch.zeros(int(batch), self.state_size, device=device, dtype=self.A.dtype)

    def step(self, x_t, state):
        return state @ self.A.t() + x_t @ self.B.t()

    def decode(self, state):
        return self.readout(state)


class HiddenCounterStepModel(LinearStepModel):
    """Deliberately invalid recurrence with transition state outside ``state``."""

    def __init__(self):
        super().__init__()
        self.register_buffer("external_step_counter", torch.zeros(()))

    def step(self, x_t, state):
        self.external_step_counter.add_(1.0)
        return super().step(x_t, state) + self.external_step_counter


class StateAdapterTests(unittest.TestCase):
    def test_actual_legacy_stepbaseline_uses_whole_state(self):
        torch.manual_seed(11)
        model = RealDiagLRUUnitBaseline(input_dim=2, output_dim=1, hidden=4)
        adapter = StateAdapter(model)
        self.assertFalse(adapter.is_full_block)
        self.assertEqual(adapter.primary_dim, model.state_size)
        self.assertEqual(adapter.reported_dim, model.state_size)

        state = torch.randn(3, model.state_size)
        x_t = torch.randn(3, model.input_dim)
        expected = model.step(x_t, state)
        actual = adapter.primary_step(state, x_t)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

        parts = adapter.unpack(state)
        self.assertIsInstance(parts, StateParts)
        self.assertIsNone(parts.stream)
        self.assertTrue(torch.equal(adapter.pack(parts), state))
        self.assertEqual(adapter.state_spec()["overwritten_components"], [])

    def test_full_block_without_carried_stream_has_minimal_carrier_primary(self):
        torch.manual_seed(7)
        model = FullBlockSequenceModel(
            input_dim=2,
            output_dim=2,
            variant="PAN-Block",
            d_model=4,
            rec_dim=4,
            num_layers=1,
            dropout=0.0,
            carry_stream=False,
            decode_mode="stream",
        ).eval()
        adapter = StateAdapter(model)
        self.assertTrue(adapter.is_full_block)
        self.assertEqual(adapter.carrier_dim, 4)
        self.assertEqual(adapter.stream_dim, 4)
        self.assertEqual(adapter.primary_dim, 4)
        self.assertEqual(adapter.reported_dim, 8)

        carrier = torch.randn(3, adapter.carrier_dim)
        stream = torch.randn(3, adapter.stream_dim)
        reported = adapter.pack(carrier, stream)
        unpacked_carrier, unpacked_stream = adapter.unpack(reported)
        self.assertTrue(torch.equal(unpacked_carrier, carrier))
        self.assertTrue(torch.equal(unpacked_stream, stream))
        self.assertTrue(torch.equal(adapter.primary_from_reported(reported), carrier))

        x_t = torch.randn(3, model.input_dim)
        direct_reported = model.step(x_t, reported)
        adapted_primary = adapter.primary_step(carrier, x_t)
        torch.testing.assert_close(
            adapted_primary,
            direct_reported[:, : adapter.carrier_dim],
            rtol=0.0,
            atol=0.0,
        )

        intervention = adapter.stream_intervention(reported, x_t)
        self.assertTrue(intervention["required_no_feedback"])
        self.assertTrue(intervention["overwritten_without_feedback"])
        self.assertEqual(intervention["next_primary_max_abs_difference"], 0.0)
        self.assertEqual(intervention["next_reported_max_abs_difference"], 0.0)
        self.assertTrue(intervention["passed"])

        decoded = adapter.decode(direct_reported)
        torch.testing.assert_close(decoded, model.decode(direct_reported))
        with self.assertRaisesRegex(ValueError, "matching reported stream"):
            adapter.decode(adapted_primary)
        direct_parts = adapter.unpack(direct_reported)
        decoded_from_parts = adapter.decode(adapted_primary, stream=direct_parts.stream)
        torch.testing.assert_close(decoded_from_parts, decoded)

        spec = adapter.state_spec()
        self.assertEqual(spec["primary_dimension"], 4)
        self.assertEqual(spec["reported_dimension"], 8)
        self.assertEqual(spec["overwritten_components"], ["stream"])
        self.assertFalse(spec["components"][1]["feeds_next_step"])
        json.dumps(spec, allow_nan=False)

    def test_carried_stream_is_part_of_primary_state(self):
        torch.manual_seed(9)
        model = FullBlockSequenceModel(
            input_dim=2,
            output_dim=1,
            variant="PAN-Block",
            d_model=4,
            rec_dim=4,
            num_layers=1,
            dropout=0.0,
            carry_stream=True,
        ).eval()
        adapter = StateAdapter(model)
        self.assertEqual(adapter.primary_dim, adapter.reported_dim)
        self.assertTrue(adapter.stream_is_primary)
        spec = adapter.state_spec()
        self.assertTrue(spec["components"][1]["feeds_next_step"])
        self.assertEqual(spec["overwritten_components"], [])

        state = adapter.init_reported(2)
        intervention = adapter.stream_intervention(state)
        self.assertFalse(intervention["required_no_feedback"])
        self.assertTrue(intervention["passed"])


class PhaseZeroAuditTests(unittest.TestCase):
    def test_hidden_cache_intervention_rejects_module_side_step_counter(self):
        model = HiddenCounterStepModel().eval()
        adapter = StateAdapter(model)
        state = torch.zeros(2, 3)
        check = check_no_external_hidden_cache(adapter, state, seed=11)
        self.assertFalse(check["passed"])
        self.assertFalse(check["bitwise_equal_after_unrelated_trajectory"])

    def test_linear_map_passes_jvp_and_exactness_screen(self):
        model = LinearStepModel().eval()
        adapter = StateAdapter(model)
        state = torch.tensor(
            [[0.1, -0.3, 0.5], [-0.4, 0.2, 0.7]], dtype=torch.float32
        )
        jvp = check_autograd_jvp(
            adapter,
            state,
            directions=8,
            seed=123,
        )
        self.assertTrue(jvp["passed"], jvp)
        self.assertEqual(jvp["passed_directions"], 8)

        screen = architecture_exactness_screen(adapter, seed=99)
        self.assertTrue(screen["completed"])
        self.assertTrue(screen["affine_consistent"])
        self.assertTrue(screen["diagonal_in_reported_coordinates"])
        self.assertTrue(screen["strict_euclidean_contraction"])
        self.assertTrue(screen["unique_affine_fixed_point"])
        self.assertTrue(screen["exact_continuum_ruled_out"])
        self.assertLess(screen["spectral_radius"], 1.0)
        json.dumps(screen, allow_nan=False)

    def test_complete_phase0_report_is_json_serializable_and_restores_mode(self):
        model = LinearStepModel()
        model.train()
        report = run_phase0_audit(
            model,
            config=AuditConfig(
                batch_size=2,
                seed=321,
                random_directions=6,
                architecture_max_dimension=16,
            ),
        )
        self.assertTrue(report["passed"], json.dumps(report, indent=2))
        self.assertTrue(model.training)
        checks = report["checks"]
        self.assertTrue(checks["state_transition_inventory"]["passed"])
        self.assertTrue(checks["pack_unpack_round_trip"]["passed"])
        self.assertTrue(checks["required_blank_maps"]["passed"])
        self.assertTrue(checks["analysis_mode"]["passed"])
        self.assertTrue(checks["zero_input"]["passed"])
        self.assertTrue(checks["determinism"]["passed"])
        self.assertTrue(checks["no_external_hidden_cache"]["passed"])
        self.assertTrue(checks["actual_f0"]["passed"])
        self.assertTrue(checks["autograd_vs_central_finite_difference"]["passed"])
        self.assertTrue(checks["float64_subset"]["passed"])
        self.assertEqual(
            checks["float64_subset"]["analysis_dtype"], "float64"
        )
        self.assertTrue(checks["architecture_exactness_screen"]["passed"])
        self.assertFalse(checks["stream_intervention"]["applicable"])
        encoded = json.dumps(report, sort_keys=True, allow_nan=False)
        self.assertIn("sagodi_phase0_state_audit", encoded)

    def test_actual_full_block_phase0_audits_overwritten_stream(self):
        torch.manual_seed(17)
        model = FullBlockSequenceModel(
            input_dim=2,
            output_dim=2,
            variant="PAN-Block",
            d_model=4,
            rec_dim=4,
            num_layers=1,
            dropout=0.0,
            carry_stream=False,
        )
        report = run_phase0_audit(
            model,
            batch_size=2,
            seed=42,
            random_directions=4,
            architecture_max_dimension=16,
        )
        self.assertTrue(report["passed"], json.dumps(report, indent=2))
        stream = report["checks"]["stream_intervention"]
        self.assertTrue(stream["overwritten_without_feedback"])
        self.assertEqual(stream["next_primary_max_abs_difference"], 0.0)
        self.assertEqual(stream["next_reported_max_abs_difference"], 0.0)
        self.assertTrue(
            report["checks"]["architecture_exactness_screen"]
            ["structural_classification"]["recognized"]
        )

    def test_pan_rnw_is_diagonal_blank_map_when_actual_drive_is_zero(self):
        torch.manual_seed(23)
        model = FullBlockSequenceModel(
            input_dim=2,
            output_dim=1,
            variant="PAN-RNW-Block",
            d_model=4,
            rec_dim=4,
            num_layers=1,
            dropout=0.0,
            encoder_bias=False,
            carry_stream=False,
        ).eval()
        adapter = StateAdapter(model)
        screen = architecture_exactness_screen(
            adapter,
            seed=101,
            max_dimension=16,
        )
        structural = screen["structural_classification"]
        self.assertTrue(structural["recognized"], structural)
        self.assertEqual(structural["form"], "diagonal_linear_primary_blank_map")
        self.assertTrue(structural["blank_recurrent_drive_exactly_zero"])
        self.assertEqual(structural["blank_recurrent_drive_max_abs"], 0.0)
        self.assertTrue(screen["affine_consistent"], screen)
        self.assertTrue(screen["diagonal_in_reported_coordinates"], screen)
        self.assertTrue(screen["strict_euclidean_contraction"], screen)
        self.assertTrue(screen["exact_continuum_ruled_out"], screen)


if __name__ == "__main__":
    unittest.main()
