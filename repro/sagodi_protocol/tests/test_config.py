from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from repro.sagodi_protocol import config


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class ProtocolFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = config.load_protocol()
        cls.native_protocol = config.load_protocol(config.NATIVE_RECIPE_PROTOCOL_PATH)

    def test_protocol_file_is_json_compatible_yaml(self) -> None:
        raw = config.DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        self.assertEqual(parsed["schema_version"], "1.0.0")
        self.assertEqual(parsed["freeze_status"], "pilot_only")

    def test_loader_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text(
                '{"schema_version":"1.0.0","schema_version":"1.0.0"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(config.ProtocolConfigError, "duplicate JSON key"):
                config.load_protocol(path)

    def test_only_phase0_and_phase1_pilot_are_active(self) -> None:
        scope = self.protocol["scope"]
        self.assertEqual(
            scope["active_phase_ids"],
            ["phase0_state_audit", "phase1_ring_pilot"],
        )
        self.assertFalse(scope["pilot_results_are_confirmatory"])
        self.assertFalse(scope["pilot_results_may_support_l3_claim"])
        self.assertFalse(scope["old_campaign_results_may_be_pooled"])
        self.assertTrue(scope["later_phases"])
        self.assertTrue(all(not item["enabled"] for item in scope["later_phases"]))

    def test_phase0_blocks_phase1(self) -> None:
        phase0 = self.protocol["phase0_state_audit"]
        phase1 = self.protocol["phase1_ring_pilot"]
        self.assertTrue(phase0["blocks_dependents_on_failure"])
        self.assertEqual(phase1["depends_on"], ["phase0_state_audit"])
        self.assertEqual(phase0["primary_state_rule"], "minimum_full_markov_recurrent_state")

    def test_phase1_matrix_is_exactly_fifteen_runs(self) -> None:
        runs = config.expand_phase1_runs(self.protocol)
        self.assertEqual(len(runs), 15)
        self.assertEqual(len({run["run_id"] for run in runs}), 15)
        self.assertEqual({run["model"]["id"] for run in runs}, {"ca_lru", "no_rp", "gru"})
        self.assertEqual({run["model_seed"] for run in runs}, {100, 101, 102, 103, 104})
        self.assertEqual({run["learning_rate"] for run in runs}, {0.01})
        self.assertEqual({run["width"] for run in runs}, {96})
        self.assertTrue(all(run["phase0_gate_required"] for run in runs))
        self.assertTrue(all(not run["confirmatory"] for run in runs))

    def test_native_recipe_freeze_is_separate_and_exact(self) -> None:
        protocol = self.native_protocol
        self.assertEqual(protocol["freeze_id"], "calru_native_sagodi_ring_pilot_v1")
        self.assertEqual(
            protocol["phase1_ring_pilot"]["protocol_track"],
            config.NATIVE_RECIPE_TRACK,
        )
        self.assertEqual(
            protocol["reporting"],
            {
                "training_track": "calru_native_recipe_transfer",
                "display_label": "CA-LRU native training on Ságodi ring task",
                "protocol_A_eligible": False,
                "protocol_B_confirmatory_eligible": False,
            },
        )
        self.assertEqual(
            protocol["phase1_ring_pilot"]["launch_policy"],
            {
                "mode": "sentinel_then_remaining",
                "sentinel_model": "ca_lru",
                "sentinel_seed": 100,
                "gate": "verified_training_completion_receipt",
            },
        )
        task = protocol["phase1_ring_pilot"]["task"]
        self.assertEqual(task["sequence_steps"], 256)
        self.assertEqual(task["initialization_mode"], "hidden_init")
        self.assertEqual(task["input_feature"], "raw_angular_velocity")
        self.assertEqual(task["velocity_process"]["gp_cholesky_jitter"], 1e-6)
        training = protocol["phase1_ring_pilot"]["training"]
        self.assertEqual(training["batch_size"], 256)
        self.assertEqual(training["optimizer_updates"], 10000)
        self.assertEqual(
            training["optimizer"],
            {
                "name": "AdamW",
                "betas": [0.9, 0.999],
                "epsilon": 1e-8,
                "weight_decay": 1e-5,
            },
        )
        self.assertEqual(training["learning_rate"]["active_launch_values"], [0.001])
        self.assertFalse(training["state_noise"]["enabled"])
        self.assertEqual(
            training["gradient_clipping"],
            {"policy": "global_norm", "frozen_numeric_value": 1.0},
        )
        self.assertEqual(
            training["progress_logging"],
            {
                "interval_updates": 100,
                "atomic_json_path": "progress.json",
                "record_pre_clip_global_gradient_norm": True,
            },
        )
        self.assertEqual(
            training["architecture"]["initial_state_encoder"]["weight_initialization"],
            {
                "distribution": "normal",
                "mean": 0.0,
                "standard_deviation": "1_over_sqrt_primary_state_dimension",
                "source": "Sagodi_official_W_otr",
            },
        )
        self.assertFalse(
            training["architecture"]["initial_state_encoder"]["bias"]
        )
        rp = training["rp_schedule_for_ca_lru"]
        self.assertEqual(rp["warmup_updates"], 3000)
        self.assertEqual(rp["interval_updates"], 100)
        self.assertEqual(rp["calls_after_warmup"], 70)
        self.assertEqual(rp["probe_batch_size"], 96)
        self.assertEqual(rp["probe_horizon"], 256)
        self.assertEqual(rp["blank_ablation_horizon"], 500)
        self.assertEqual(rp["eta_lambda_pilot_default"], 3000.0)
        self.assertEqual(rp["damage_epsilon_pilot_default"], 1e-4)

    def test_native_recipe_matrix_uses_only_pilot_seeds_and_inherited_lr(self) -> None:
        runs = config.expand_phase1_runs(self.native_protocol)
        self.assertEqual(len(runs), 15)
        self.assertEqual({run["model_seed"] for run in runs}, {100, 101, 102, 103, 104})
        self.assertEqual({run["learning_rate"] for run in runs}, {0.001})
        self.assertEqual(
            {run["protocol_track"] for run in runs},
            {config.NATIVE_RECIPE_TRACK},
        )
        self.assertTrue(all(not run["confirmatory"] for run in runs))

    def test_training_freeze_matches_requested_protocol_a_pilot(self) -> None:
        training = self.protocol["phase1_ring_pilot"]["training"]
        self.assertEqual(training["batch_size"], 64)
        self.assertEqual(training["optimizer_updates"], 5000)
        self.assertEqual(training["optimizer"]["name"], "Adam")
        self.assertEqual(training["optimizer"]["betas"], [0.9, 0.999])
        self.assertEqual(training["state_noise"]["coordinate_standard_deviation"], 0.1)
        self.assertEqual(
            training["learning_rate"]["pilot_grid"],
            [0.01, 0.001, 0.0001, 0.00001],
        )
        self.assertEqual(training["learning_rate"]["active_launch_values"], [0.01])
        self.assertEqual(
            training["learning_rate"]["selection_status"],
            "pilot_default_not_selected_from_current_results",
        )
        self.assertFalse(training["learning_rate"]["future_grid_sweep_enabled"])
        architecture = training["architecture"]
        self.assertEqual(architecture["shared_builder_kwargs"]["layers"], 1)
        self.assertEqual(architecture["shared_builder_kwargs"]["pan_lambda_min"], 0.90)
        self.assertEqual(architecture["shared_builder_kwargs"]["pan_lambda_max"], 0.999)
        self.assertEqual(
            architecture["ca_lru_and_no_rp"]["blank_primary_map"],
            "homogeneous_diagonal_linear_F0_h_equals_Lambda_h",
        )
        self.assertEqual(
            architecture["ca_lru_and_no_rp"]["writer_formula"],
            "g_h_u_minus_g_h_zero",
        )

    def test_phase0_artifacts_match_json_npz_backend(self) -> None:
        required = self.protocol["phase0_state_audit"]["required_artifacts"]
        self.assertIn("blank_map_trace.npz", required)
        self.assertIn("blank_map_trace_metadata.json", required)
        self.assertIn("determinism_check.json", required)
        self.assertNotIn("blank_map_trace.parquet", required)

    def test_all_section_3_13_2_gates_are_encoded(self) -> None:
        gates = self.protocol["claim_gates"]
        self.assertTrue(config.REQUIRED_CLAIM_GATES.issubset(gates))
        self.assertFalse(gates["c4"]["evaluated_in_phase1_pilot"])
        self.assertFalse(gates["model_seeds"]["evaluated_in_phase1_pilot"])

    def test_conservative_ambiguity_resolutions_are_frozen(self) -> None:
        gates = self.protocol["claim_gates"]
        self.assertEqual(
            gates["c1_decoding"]["primary_decoder"],
            "held_out_linear_diagnostic_decoder",
        )
        self.assertEqual(gates["c2_drift"]["horizon_steps"], 1024)
        self.assertIn(1024, self.protocol["evaluation"]["blank_horizons"])
        self.assertEqual(
            set(gates["c3_normal_recovery"]["required_direction_families"]),
            {"radial", "ambient"},
        )
        self.assertEqual(
            gates["c3_normal_recovery"]["aggregation_resolution"],
            "each_required_direction_family_must_pass_separately",
        )
        self.assertEqual(gates["tangent_non_expansion"]["within_seed_quantile"], 0.95)
        self.assertEqual(gates["tangent_non_expansion"]["q95_max"], 1.05)

    def test_qa_thresholds_are_separate_from_claim_gates(self) -> None:
        qa = self.protocol["qa_thresholds"]
        self.assertEqual(qa["status"], "analysis_quality_controls_not_claim_gates")
        self.assertEqual(qa["projection"]["implied_q95_error_max"], 0.002)
        self.assertEqual(qa["projection"]["ring_dense_spline_factor"], 8)
        self.assertTrue(qa["clean_paired_recovery_required"])
        settling = qa["settling"]
        self.assertEqual(settling["within_path_next5_relative_decrease_less_than"], 0.01)
        self.assertEqual(settling["normalized_geodesic_drift_q95_less_than"], 0.01)
        self.assertEqual(settling["systematic_transverse_decrease_fraction_max"], 0.5)
        self.assertEqual(
            settling["source_status"],
            "protocol_criterion_3_operationalized_for_pilot_not_claim_threshold",
        )

    def test_source_protocol_hash_matches_current_note(self) -> None:
        self.assertTrue(config.source_protocol_matches(self.protocol, REPOSITORY_ROOT))

    def test_fingerprint_is_deterministic(self) -> None:
        first = config.protocol_fingerprint(self.protocol)
        second = config.protocol_fingerprint(copy.deepcopy(self.protocol))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_validator_rejects_pooled_radial_and_ambient_gate(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["claim_gates"]["c3_normal_recovery"][
            "aggregation_resolution"
        ] = "pooled"
        with self.assertRaisesRegex(config.ProtocolConfigError, "must not be pooled"):
            config.validate_protocol(mutated)

    def test_validator_rejects_endpoint_only_primary_normal_gap(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["claim_gates"]["sampled_normal_gap"][
            "projection_schedule"
        ] = "endpoint_only"
        with self.assertRaisesRegex(
            config.ProtocolConfigError, "per-step projected normal cocycle"
        ):
            config.validate_protocol(mutated)

    def test_validator_rejects_approximate_four_t_horizon(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["claim_gates"]["c2_drift"]["horizon_steps"] = 1000
        with self.assertRaisesRegex(config.ProtocolConfigError, "must equal exact 4T"):
            config.validate_protocol(mutated)

    def test_validator_rejects_enabling_later_phase(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["scope"]["later_phases"][0]["enabled"] = True
        with self.assertRaisesRegex(config.ProtocolConfigError, "must be disabled"):
            config.validate_protocol(mutated)

    def test_validator_rejects_missing_claim_gate(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        del mutated["claim_gates"]["c1_rank"]
        with self.assertRaisesRegex(config.ProtocolConfigError, "missing Section 3.13.2 gates"):
            config.validate_protocol(mutated)

    def test_validator_rejects_architecture_drift(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["phase1_ring_pilot"]["training"]["architecture"][
            "ca_lru_and_no_rp"
        ]["carry_stream"] = True
        with self.assertRaisesRegex(config.ProtocolConfigError, "resolved scaffold"):
            config.validate_protocol(mutated)

    def test_native_validator_rejects_protocol_a_label_or_wotr_drift(self) -> None:
        mislabeled = copy.deepcopy(self.native_protocol)
        mislabeled["reporting"]["protocol_A_eligible"] = True
        with self.assertRaisesRegex(config.ProtocolConfigError, "reporting labels"):
            config.validate_protocol(mislabeled)

        biased = copy.deepcopy(self.native_protocol)
        biased["phase1_ring_pilot"]["training"]["architecture"][
            "initial_state_encoder"
        ]["bias"] = True
        with self.assertRaisesRegex(config.ProtocolConfigError, "initial-state encoder"):
            config.validate_protocol(biased)

    def test_protocol_a_rejects_native_identity_and_execution_fields(self) -> None:
        mislabeled = copy.deepcopy(self.protocol)
        mislabeled["reporting"] = copy.deepcopy(self.native_protocol["reporting"])
        with self.assertRaisesRegex(config.ProtocolConfigError, "reporting labels"):
            config.validate_protocol(mislabeled)

        staged = copy.deepcopy(self.protocol)
        staged["phase1_ring_pilot"]["launch_policy"] = copy.deepcopy(
            self.native_protocol["phase1_ring_pilot"]["launch_policy"]
        )
        with self.assertRaisesRegex(config.ProtocolConfigError, "sentinel launch policy"):
            config.validate_protocol(staged)

        instrumented = copy.deepcopy(self.protocol)
        instrumented["phase1_ring_pilot"]["training"]["progress_logging"] = (
            copy.deepcopy(
                self.native_protocol["phase1_ring_pilot"]["training"][
                    "progress_logging"
                ]
            )
        )
        with self.assertRaisesRegex(config.ProtocolConfigError, "progress logging"):
            config.validate_protocol(instrumented)

        renamed = copy.deepcopy(self.protocol)
        renamed["freeze_id"] = "calru_native_sagodi_ring_pilot_v1"
        with self.assertRaisesRegex(config.ProtocolConfigError, "dedicated freeze_id"):
            config.validate_protocol(renamed)

    def test_native_validator_rejects_gp_jitter_drift(self) -> None:
        mutated = copy.deepcopy(self.native_protocol)
        mutated["phase1_ring_pilot"]["task"]["velocity_process"][
            "gp_cholesky_jitter"
        ] = 1e-5
        with self.assertRaisesRegex(config.ProtocolConfigError, "Cholesky jitter"):
            config.validate_protocol(mutated)


if __name__ == "__main__":
    unittest.main()
