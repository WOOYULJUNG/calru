"""Frozen configuration loader for the Ságodi Phase 0/1 ring pilot.

The ``analysis_protocol.yaml`` file intentionally contains JSON.  JSON is a
strict subset of YAML, so the freeze remains YAML-compatible while this module
can load and validate it with Python's standard library only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("analysis_protocol.yaml")

REQUIRED_CLAIM_GATES = frozenset(
    {
        "task",
        "c1_decoding",
        "c1_rank",
        "c1_neighborhood",
        "c1_sheet_fiber",
        "invariance",
        "c2_drift",
        "c3_normal_recovery",
        "c3_same_memory",
        "c3_paper_noise",
        "tangent_equivariance",
        "tangent_non_expansion",
        "sampled_normal_gap",
        "c4",
        "model_seeds",
    }
)

ACTIVE_PHASES = ("phase0_state_audit", "phase1_ring_pilot")
PILOT_MODELS = ("ca_lru", "no_rp", "gru")
PILOT_MODEL_SEEDS = (100, 101, 102, 103, 104)


class ProtocolConfigError(ValueError):
    """Raised when a protocol freeze is malformed or internally inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolConfigError(message)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be an array",
    )
    return value


def _unique(values: Iterable[Any], label: str) -> tuple[Any, ...]:
    result = tuple(values)
    _require(len(result) == len(set(result)), f"{label} contains duplicates")
    return result


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the JSON-compatible YAML protocol freeze."""

    protocol_path = Path(path) if path is not None else DEFAULT_PROTOCOL_PATH
    try:
        with protocol_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ProtocolConfigError(f"protocol file not found: {protocol_path}") from exc
    except json.JSONDecodeError as exc:
        raise ProtocolConfigError(
            f"protocol is not JSON-compatible YAML: {protocol_path}: {exc}"
        ) from exc

    _require(isinstance(data, dict), "protocol root must be an object")
    validate_protocol(data)
    return data


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate the frozen scope and conservative ambiguity resolutions."""

    _require(protocol.get("schema_version") == "1.0.0", "unsupported schema_version")
    _require(protocol.get("freeze_status") == "pilot_only", "freeze must be pilot_only")

    scope = _require_mapping(protocol.get("scope"), "scope")
    active = tuple(_require_sequence(scope.get("active_phase_ids"), "active_phase_ids"))
    _require(active == ACTIVE_PHASES, f"active phases must be {ACTIVE_PHASES!r}")
    _require(scope.get("pilot_results_are_confirmatory") is False, "pilot is not confirmatory")
    _require(scope.get("pilot_results_may_support_l3_claim") is False, "pilot cannot support L3")
    _require(scope.get("old_campaign_results_may_be_pooled") is False, "old runs cannot be pooled")
    later_phases = _require_sequence(scope.get("later_phases"), "scope.later_phases")
    _require(later_phases, "at least one gated later phase must be declared")
    for phase in later_phases:
        item = _require_mapping(phase, "later phase")
        _require(item.get("enabled") is False, f"later phase {item.get('id')} must be disabled")
        _require(bool(item.get("activation_gate")), "later phase must declare activation_gate")

    seeds = _require_mapping(protocol.get("seed_policy"), "seed_policy")
    pilot_seeds = tuple(
        _require_sequence(seeds.get("pilot_model_seeds"), "pilot_model_seeds")
    )
    _require(pilot_seeds == PILOT_MODEL_SEEDS, f"pilot seeds must be {PILOT_MODEL_SEEDS!r}")
    _unique(pilot_seeds, "pilot_model_seeds")
    _require(seeds.get("main_model_seeds") == [], "main seeds must not be launched")
    _require(
        seeds.get("independent_statistical_unit") == "trained_model_seed",
        "trained model seed must be the statistical unit",
    )

    phase0 = _require_mapping(protocol.get("phase0_state_audit"), "phase0_state_audit")
    _require(phase0.get("enabled") is True, "Phase 0 must be enabled")
    _require(phase0.get("blocks_dependents_on_failure") is True, "Phase 0 must block on failure")
    _require(
        tuple(phase0.get("models", [])) == PILOT_MODELS,
        f"Phase 0 models must be {PILOT_MODELS!r}",
    )
    _require(
        phase0.get("primary_state_rule") == "minimum_full_markov_recurrent_state",
        "Phase 0 must audit the minimum full Markov state",
    )
    _require(
        phase0.get("required_artifacts")
        == [
            "state_spec.json",
            "state_transition_audit.json",
            "blank_map_trace.npz",
            "blank_map_trace_metadata.json",
            "exactness_screen.json",
            "determinism_check.json",
            "jacobian_check.json",
            "float64_subset_check.json",
            "hidden_cache_check.json",
            "phase0_gate.json",
        ],
        "Phase 0 artifacts must use the pilot JSON/NPZ backend",
    )
    trace_scope = _require_mapping(phase0.get("trace_scope"), "phase0.trace_scope")
    _require(
        trace_scope.get("architecture_phase0")
        == "untrained_architecture_blank_map_trace_without_manifold_distances",
        "Phase 0 must label its trace as an untrained architecture trace",
    )
    _require(
        trace_scope.get("trained_checkpoint_followup")
        == "phase1_checkpoint_autonomous_trace_with_task_conditioned_atlas_and_radial_perturbations",
        "trained-checkpoint autonomous tracing must remain a Phase-1 follow-up",
    )

    phase1 = _require_mapping(protocol.get("phase1_ring_pilot"), "phase1_ring_pilot")
    _require(phase1.get("enabled") is True, "Phase 1 pilot must be enabled")
    _require(phase1.get("confirmatory") is False, "Phase 1 must remain a pilot")
    _require(
        tuple(phase1.get("depends_on", [])) == ("phase0_state_audit",),
        "Phase 1 must depend on Phase 0",
    )
    _require(
        phase1.get("protocol_track") == "A_sagodi_training_rules",
        "Phase 1 must use Protocol A training rules",
    )

    task = _require_mapping(phase1.get("task"), "phase1.task")
    _require(task.get("id") == "angular_velocity_integration", "Phase 1 task must be angular integration")
    _require(task.get("topology") == "S1", "Phase 1 must be a ring task")
    _require(task.get("latent_dimension") == 1, "Phase 1 latent dimension must be one")
    _require(task.get("sequence_steps") == 256, "Phase 1 T must be 256")
    _require(task.get("initialization_mode") == "hidden_init", "Protocol A uses hidden initialization")
    _require(task.get("input_feature") == "raw_angular_velocity", "Protocol A uses raw velocity")
    indexing = _require_mapping(task.get("indexing"), "phase1.task.indexing")
    _require(
        indexing.get("velocity_token_t_target") == "q_t_plus_1_after_velocity_update",
        "target indexing must be post-update",
    )

    model_specs = _require_sequence(phase1.get("models"), "phase1.models")
    model_ids = tuple(_require_mapping(item, "model spec").get("id") for item in model_specs)
    _require(model_ids == PILOT_MODELS, f"pilot models must be {PILOT_MODELS!r}")
    _unique(model_ids, "phase1 model ids")
    for item in model_specs[:2]:
        _require(item.get("variant") == "PAN-RNW-full", "CA-LRU variants must be PAN-RNW-full")
        _require(
            item.get("autonomous_primary_map")
            == "homogeneous_diagonal_linear_F0_h_equals_Lambda_h",
            "CA-LRU blank carrier map must be labeled homogeneous diagonal linear",
        )
        _require(
            item.get("input_conditioned_writer")
            == "nonlinear_recurrent_writer_g_h_u_minus_g_h_zero",
            "CA-LRU nonlinearity must be attributed to the input-conditioned writer",
        )

    training = _require_mapping(phase1.get("training"), "phase1.training")
    _require(training.get("width") == 96, "pilot width must be 96")
    architecture = _require_mapping(training.get("architecture"), "phase1.training.architecture")
    initializer = _require_mapping(
        architecture.get("initial_state_encoder"), "architecture.initial_state_encoder"
    )
    _require(
        initializer
        == {
            "input_dimension": 2,
            "output_dimension": "primary_state_dimension",
            "module": "torch.nn.Linear",
            "bias": True,
        },
        "initial-state encoder architecture differs from the pilot freeze",
    )
    builder = _require_mapping(
        architecture.get("shared_builder_kwargs"), "architecture.shared_builder_kwargs"
    )
    _require(
        builder
        == {
            "rank": 2,
            "d_model": "width",
            "rec_dim": "width",
            "layers": 1,
            "dropout": 0.0,
            "plru_tau": 0.001,
            "plru_c": 50.0,
            "pan_lambda_min": 0.90,
            "pan_lambda_max": 0.999,
            "rank_matched_lambda_high": 0.999,
            "rank_matched_lambda_low": 0.0,
        },
        "shared model builder kwargs differ from the pilot freeze",
    )
    ca_architecture = _require_mapping(
        architecture.get("ca_lru_and_no_rp"), "architecture.ca_lru_and_no_rp"
    )
    _require(
        ca_architecture
        == {
            "legacy_variant": "PAN-RNW-full",
            "resolved_recurrence_variant": "PAN-RNW-Block",
            "writer_mode": "recurrent",
            "writer_hidden": "max_d_model_rec_dim",
            "writer_formula": "g_h_u_minus_g_h_zero",
            "writer_activation": "GELU",
            "writer_linear_bias": True,
            "blank_primary_map": "homogeneous_diagonal_linear_F0_h_equals_Lambda_h",
            "retention_initialization": "linear_lambda_max_to_lambda_min",
            "theta_requires_grad": False,
            "gamma_free": True,
            "gamma_raw_initial_value": 0.0,
            "recurrence_output_projection_bias": True,
            "encoder_bias": False,
            "use_norm_in": False,
            "norm_in_affine": True,
            "use_norm_out": True,
            "norm_out_affine": True,
            "update_mode": "glu",
            "glu_projection_bias": True,
            "use_residual": True,
            "decode_mode": "stream",
            "carry_stream": False,
            "head_layer_norm_affine": True,
            "head_linear_bias": True,
        },
        "CA-LRU resolved scaffold differs from the pilot freeze",
    )
    gru_architecture = _require_mapping(architecture.get("gru"), "architecture.gru")
    _require(
        gru_architecture
        == {
            "legacy_variant": "GRU",
            "module": "torch.nn.GRUCell",
            "hidden_dimension": "width",
            "keep_bias_init": 0.0,
            "gru_cell_bias": True,
            "readout_module": "two_layer_MLP",
            "readout_hidden_dimension": 64,
            "readout_activation": "Tanh",
            "readout_linear_bias": True,
        },
        "GRU architecture differs from the pilot freeze",
    )
    _require(training.get("batch_size") == 64, "Protocol A batch size must be 64")
    _require(training.get("optimizer_updates") == 5000, "Protocol A updates must be 5000")
    optimizer = _require_mapping(training.get("optimizer"), "phase1.training.optimizer")
    _require(optimizer.get("name") == "Adam", "Protocol A optimizer must be Adam")
    _require(optimizer.get("betas") == [0.9, 0.999], "Adam betas must be frozen")

    learning_rate = _require_mapping(training.get("learning_rate"), "learning_rate")
    _require(
        learning_rate.get("pilot_grid") == [0.01, 0.001, 0.0001, 0.00001],
        "Protocol A LR grid must be preserved",
    )
    _require(
        learning_rate.get("active_launch_values") == [0.01],
        "initial pilot launch must use only 1e-2",
    )
    _require(
        learning_rate.get("frozen_pilot_default") == 0.01,
        "1e-2 must be marked as the frozen pilot default",
    )
    _require(
        learning_rate.get("selection_status")
        == "pilot_default_not_selected_from_current_results",
        "pilot LR must not be described as result-selected",
    )
    _require(
        learning_rate.get("future_grid_sweep_enabled") is False,
        "LR grid sweep must not be launched in this freeze",
    )

    noise = _require_mapping(training.get("state_noise"), "phase1.training.state_noise")
    _require(noise.get("enabled") is True, "Protocol A state noise must be enabled")
    _require(noise.get("coordinate_standard_deviation") == 0.1, "noise std must be 0.1")
    _require(noise.get("analysis_noise_enabled") is False, "analysis noise must be disabled")

    rp = _require_mapping(training.get("rp_schedule_for_ca_lru"), "rp schedule")
    _require(rp.get("warmup_updates") == 1500, "Protocol A RP warm-up must be 1500")
    _require(rp.get("interval_updates") == 50, "Protocol A RP interval must be 50")
    _require(rp.get("calls_after_warmup") == 70, "Protocol A must make 70 RP calls")
    _require(rp.get("probe_batch_size") == 256, "RP probe batch must be 256")
    _require(rp.get("probe_horizon") == 256, "RP probe horizon must be 256")
    _require(rp.get("eta_lambda_pilot_default") == 3000.0, "pilot eta_lambda must be explicit")
    _require(rp.get("damage_epsilon_pilot_default") == 0.00003, "pilot damage epsilon must be explicit")

    run_matrix = _require_mapping(phase1.get("run_matrix"), "phase1.run_matrix")
    expected_runs = len(model_ids) * len(pilot_seeds) * len(learning_rate["active_launch_values"])
    _require(run_matrix.get("expected_training_runs") == expected_runs, "run count is inconsistent")

    evaluation = _require_mapping(protocol.get("evaluation"), "evaluation")
    exact_four_t = 4 * int(task["sequence_steps"])
    _require(evaluation.get("exact_four_T_horizon") == exact_four_t, "evaluation must include exact 4T")
    _require(exact_four_t in evaluation.get("blank_horizons", []), "blank horizons must include exact 4T")

    gates = _require_mapping(protocol.get("claim_gates"), "claim_gates")
    missing_gates = REQUIRED_CLAIM_GATES.difference(gates)
    _require(not missing_gates, f"missing Section 3.13.2 gates: {sorted(missing_gates)}")
    _validate_conservative_gate_resolutions(gates, exact_four_t)

    qa = _require_mapping(protocol.get("qa_thresholds"), "qa_thresholds")
    _require(
        qa.get("status") == "analysis_quality_controls_not_claim_gates",
        "QA thresholds must be separate from claim gates",
    )
    _require(qa.get("clean_paired_recovery_required") is True, "clean-paired recovery is required")
    settling = _require_mapping(qa.get("settling"), "qa_thresholds.settling")
    _require(settling.get("candidate_horizons") == [0, 5, 20, 100], "settling horizons changed")
    _require(
        settling.get("within_path_next5_relative_decrease_less_than") == 0.01,
        "settling relative-decrease QA must be 0.01",
    )
    _require(
        settling.get("normalized_geodesic_drift_q95_less_than") == 0.01,
        "settling geodesic-drift QA must be 0.01",
    )
    _require(
        settling.get("systematic_transverse_decrease_fraction_max") == 0.5,
        "settling systematic-decrease QA must be 0.5",
    )
    _require(
        settling.get("source_status")
        == "protocol_criterion_3_operationalized_for_pilot_not_claim_threshold",
        "settling operationalization must remain a pilot QA control",
    )
    projection = _require_mapping(qa.get("projection"), "qa_thresholds.projection")
    _require(projection.get("ring_dense_spline_factor") == 8, "ring projection density must be 8x")
    _require(
        qa.get("tangent_normal_orthogonality_error_max") == 0.000001,
        "sampled tangent-normal orthogonality QA must remain 1e-6",
    )

    statistics = _require_mapping(protocol.get("statistics"), "statistics")
    pilot_statistics = _require_mapping(statistics.get("pilot"), "statistics.pilot")
    _require(
        pilot_statistics.get("confirmatory_tests_enabled") is False,
        "confirmatory tests must be disabled for pilot seeds",
    )


def _validate_conservative_gate_resolutions(
    gates: Mapping[str, Any], exact_four_t: int
) -> None:
    decoding = _require_mapping(gates["c1_decoding"], "claim_gates.c1_decoding")
    _require(
        decoding.get("primary_decoder") == "held_out_linear_diagnostic_decoder",
        "the linear diagnostic decoder must be primary",
    )

    rank = _require_mapping(gates["c1_rank"], "claim_gates.c1_rank")
    _require(
        rank.get("metric") == "normalized_sigma_d_over_sigma_1",
        "C1 rank must use the normalized local singular-value ratio",
    )
    _require(rank.get("minimum") == 0.001, "C1 rank ratio floor must equal 1e-3")
    _require(
        rank.get("required_atlas_fraction") == 0.99,
        "C1 rank gate must cover at least 99% of atlas anchors",
    )

    drift = _require_mapping(gates["c2_drift"], "claim_gates.c2_drift")
    _require(drift.get("horizon_rule") == "exact_4T", "C2 must use exact 4T")
    _require(drift.get("horizon_steps") == exact_four_t, "C2 horizon must equal exact 4T")

    recovery = _require_mapping(gates["c3_normal_recovery"], "claim_gates.c3_normal_recovery")
    _require(
        set(recovery.get("required_direction_families", [])) == {"radial", "ambient"},
        "C3 must require both radial and ambient recovery",
    )
    _require(
        recovery.get("aggregation_resolution")
        == "each_required_direction_family_must_pass_separately",
        "radial and ambient gates must not be pooled",
    )

    non_expansion = _require_mapping(gates["tangent_non_expansion"], "claim_gates.tangent_non_expansion")
    _require(non_expansion.get("within_seed_quantile") == 0.95, "tangent gate must use q95 coverage")
    _require(non_expansion.get("q95_max") == 1.05, "tangent q95 amplification must be <= 1.05")

    sampled_gap = _require_mapping(
        gates["sampled_normal_gap"], "claim_gates.sampled_normal_gap"
    )
    _require(
        sampled_gap.get("projection_schedule") == "P_N_after_every_clean_rollout_JVP_step",
        "sampled normal gap must use the per-step projected normal cocycle",
    )
    _require(
        sampled_gap.get("tangent_block")
        == "endpoint_T_qH_transpose_J_product_T_q0_without_intermediate_P_T",
        "sampled gap tangent block must not insert intermediate tangent projections",
    )

    _require(
        gates["c4"].get("evaluated_in_phase1_pilot") is False,
        "C4 must not be claimed by this pilot",
    )
    _require(
        gates["model_seeds"].get("evaluated_in_phase1_pilot") is False,
        "the 8/10 main-seed gate must not be applied to pilot seeds",
    )


def canonical_protocol_bytes(protocol: Mapping[str, Any]) -> bytes:
    """Return a deterministic canonical JSON representation for provenance."""

    validate_protocol(protocol)
    return json.dumps(
        protocol,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def protocol_fingerprint(protocol: Mapping[str, Any]) -> str:
    """SHA-256 of the validated canonical protocol object."""

    return hashlib.sha256(canonical_protocol_bytes(protocol)).hexdigest()


def source_protocol_matches(
    protocol: Mapping[str, Any], repository_root: str | Path
) -> bool:
    """Check the source-note hash recorded by the freeze, if the note is present."""

    validate_protocol(protocol)
    source = _require_mapping(protocol["source_protocol"], "source_protocol")
    source_path = Path(repository_root) / str(source["path"])
    if not source_path.is_file():
        return False
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return digest == source["sha256"]


def expand_phase1_runs(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Expand the only training matrix authorized by this freeze (15 runs)."""

    validate_protocol(protocol)
    phase1 = protocol["phase1_ring_pilot"]
    training = phase1["training"]
    seed_policy = protocol["seed_policy"]
    runs: list[dict[str, Any]] = []

    for model_spec in phase1["models"]:
        for model_seed in seed_policy["pilot_model_seeds"]:
            for learning_rate in training["learning_rate"]["active_launch_values"]:
                lr_token = _float_token(float(learning_rate))
                run_id = (
                    f"phase1__angle_integrate__{model_spec['id']}__"
                    f"w{training['width']:03d}__seed{model_seed:03d}__lr{lr_token}"
                )
                runs.append(
                    {
                        "run_id": run_id,
                        "freeze_id": protocol["freeze_id"],
                        "phase_id": "phase1_ring_pilot",
                        "phase0_gate_required": True,
                        "confirmatory": False,
                        "protocol_track": phase1["protocol_track"],
                        "task": copy.deepcopy(phase1["task"]),
                        "model": copy.deepcopy(model_spec),
                        "model_seed": model_seed,
                        "task_seed": seed_policy["task_seed"],
                        "data_stream_seed": seed_policy["data_stream_seed"],
                        "evaluation_bank_seed": seed_policy["evaluation_bank_seed"],
                        "perturbation_bank_seed": seed_policy["perturbation_bank_seed"],
                        "width": training["width"],
                        "batch_size": training["batch_size"],
                        "optimizer_updates": training["optimizer_updates"],
                        "optimizer": copy.deepcopy(training["optimizer"]),
                        "learning_rate": learning_rate,
                        "learning_rate_status": training["learning_rate"]["selection_status"],
                        "state_noise": copy.deepcopy(training["state_noise"]),
                        "checkpoint_selection": training["checkpoint_selection"],
                    }
                )

    expected = phase1["run_matrix"]["expected_training_runs"]
    _require(len(runs) == expected, "expanded run count does not match the freeze")
    _unique((run["run_id"] for run in runs), "expanded run ids")
    return tuple(runs)


def _float_token(value: float) -> str:
    text = format(value, ".10g")
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument(
        "--print-runs",
        action="store_true",
        help="print the authorized Phase 1 run matrix as JSON",
    )
    parser.add_argument(
        "--print-fingerprint",
        action="store_true",
        help="print the canonical protocol SHA-256",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.print_runs:
        print(json.dumps(expand_phase1_runs(protocol), indent=2, sort_keys=True))
    elif args.print_fingerprint:
        print(protocol_fingerprint(protocol))
    else:
        print(
            json.dumps(
                {
                    "freeze_id": protocol["freeze_id"],
                    "status": "valid",
                    "active_phases": protocol["scope"]["active_phase_ids"],
                    "authorized_training_runs": len(expand_phase1_runs(protocol)),
                    "fingerprint": protocol_fingerprint(protocol),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
