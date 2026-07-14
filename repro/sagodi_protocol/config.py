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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("analysis_protocol.yaml")
NATIVE_RECIPE_PROTOCOL_PATH = Path(__file__).with_name(
    "calru_native_sagodi_ring_pilot_v1.yaml"
)
SAGODI_LR_SELECTION_PROTOCOL_PATH = Path(__file__).with_name(
    "sagodi_ring_lr_selection_v2.yaml"
)

PROTOCOL_A_TRACK = "A_sagodi_training_rules"
NATIVE_RECIPE_TRACK = "calru_native_recipe_transfer"
SAGODI_LR_SELECTION_TRACK = "sagodi_paper_aligned_lr_selection"
SAGODI_LR_SELECTION_FREEZE_ID = "sagodi_ring_lr_selection_v2"
SAGODI_PRIMARY_LR_SELECTION_TRACK = "sagodi_primary_v3_lr_selection"
SAGODI_PRIMARY_LR_SELECTION_FREEZE_ID = "sagodi_primary_lr_selection_v3"
SAGODI_PRIMARY_MAIN_TRACK = "sagodi_primary_v3_main"
SAGODI_PRIMARY_MAIN_FREEZE_ID = "sagodi_primary_main_v3_resolved"
SAGODI_PRIMARY_MODELS = (
    "rnn_param206",
    "gru_sagodi_param135",
    "lstm_param109",
    "lru_param96",
    "no_rp",
    "ca_lru",
)
SAGODI_PRIMARY_MODEL_WIDTHS = {
    "rnn_param206": 206,
    "gru_sagodi_param135": 135,
    "lstm_param109": 109,
    "lru_param96": 96,
    "no_rp": 96,
    "ca_lru": 96,
}
SAGODI_PRIMARY_PARAMETER_COUNTS = {
    "rnn_param206": 56844,
    "gru_sagodi_param135": 56432,
    "lstm_param109": 56438,
    "lru_param96": 56834,
    "no_rp": 56834,
    "ca_lru": 56834,
}
SAGODI_PRIMARY_SELECTION_SEEDS = (1100, 1101, 1102, 1103, 1104)
SAGODI_PRIMARY_MAIN_SEEDS = tuple(range(10))

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
SAGODI_LR_SELECTION_MODELS = (
    "ca_lru",
    "no_rp",
    "gru_sagodi_width96",
    "gru_sagodi_param135",
)
SAGODI_LR_SELECTION_SEEDS = (1100, 1101, 1102, 1103, 1104)
SAGODI_LR_SELECTION_GRID = (0.01, 0.001, 0.0001, 0.00001)

LEGACY_ANGULAR_GP_CHOLESKY_JITTER = 1e-6


class ProtocolConfigError(ValueError):
    """Raised when a protocol freeze is malformed or internally inconsistent."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently keeping the last value."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolConfigError(f"protocol contains duplicate JSON key: {key!r}")
        result[key] = value
    return result


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


@dataclass(frozen=True)
class AngularTaskSpec:
    """Fully resolved executable contract for the Phase-1 angular task.

    The v1 protocol files deliberately remain immutable.  Protocol A omitted
    the numerical Cholesky jitter, so that one legacy value is resolved here
    and labelled explicitly.  Every other task field is read from the freeze;
    callers must use :meth:`generator_kwargs` rather than Python defaults.
    """

    task_id: str
    topology: str
    latent_dimension: int
    sequence_steps: int
    delta_t: float
    input_dimension: int
    output_dimension: int
    input_feature: str
    initialization_mode: str
    initial_state_encoder_input: tuple[str, ...]
    q0_distribution: str
    q0_low: float
    q0_high: float
    q0_high_inclusive: bool
    gp_family: str
    gp_grid: str
    gp_grid_start: float
    gp_grid_stop: float
    gp_grid_endpoint: bool
    gp_length_scale: float
    gp_marginal_standard_deviation: float
    gp_cholesky_jitter: float
    gp_cholesky_jitter_source: str
    training_sampling: str
    initial_state_represents: str
    velocity_token_target: str
    loss_mask: str
    target: tuple[str, ...]

    @classmethod
    def from_protocol(cls, protocol: Mapping[str, Any]) -> "AngularTaskSpec":
        phase = _require_mapping(
            protocol.get("phase1_ring_pilot"), "phase1_ring_pilot"
        )
        task = _require_mapping(phase.get("task"), "phase1.task")
        velocity = _require_mapping(
            task.get("velocity_process"), "phase1.task.velocity_process"
        )
        indexing = _require_mapping(task.get("indexing"), "phase1.task.indexing")

        jitter_is_explicit = "gp_cholesky_jitter" in velocity
        jitter = (
            velocity.get("gp_cholesky_jitter")
            if jitter_is_explicit
            else LEGACY_ANGULAR_GP_CHOLESKY_JITTER
        )
        spec = cls(
            task_id=str(task.get("id")),
            topology=str(task.get("topology")),
            latent_dimension=int(task.get("latent_dimension", -1)),
            sequence_steps=int(task.get("sequence_steps", -1)),
            delta_t=float(task.get("delta_t", float("nan"))),
            input_dimension=int(task.get("input_dimension", -1)),
            output_dimension=int(task.get("output_dimension", -1)),
            input_feature=str(task.get("input_feature")),
            initialization_mode=str(task.get("initialization_mode")),
            initial_state_encoder_input=tuple(
                _require_sequence(
                    task.get("initial_state_encoder_input"),
                    "phase1.task.initial_state_encoder_input",
                )
            ),
            q0_distribution=str(task.get("theta0_distribution")),
            q0_low=-math.pi,
            q0_high=math.pi,
            q0_high_inclusive=False,
            gp_family=str(velocity.get("family")),
            gp_grid=str(velocity.get("normalized_time_grid")),
            gp_grid_start=-1.0,
            gp_grid_stop=1.0,
            gp_grid_endpoint=True,
            gp_length_scale=float(velocity.get("length_scale", float("nan"))),
            gp_marginal_standard_deviation=float(
                velocity.get("marginal_standard_deviation", float("nan"))
            ),
            gp_cholesky_jitter=float(jitter),
            gp_cholesky_jitter_source=(
                "protocol_explicit"
                if jitter_is_explicit
                else "legacy_v1_implementation_default"
            ),
            training_sampling=str(velocity.get("training_sampling")),
            initial_state_represents=str(indexing.get("initial_state_represents")),
            velocity_token_target=str(indexing.get("velocity_token_t_target")),
            loss_mask=str(task.get("loss_mask")),
            target=tuple(
                _require_sequence(task.get("target"), "phase1.task.target")
            ),
        )
        spec._validate()
        return spec

    def _validate(self) -> None:
        expected = {
            "task_id": (self.task_id, "angular_velocity_integration"),
            "topology": (self.topology, "S1"),
            "latent_dimension": (self.latent_dimension, 1),
            "sequence_steps": (self.sequence_steps, 256),
            "delta_t": (self.delta_t, 0.1),
            "input_dimension": (self.input_dimension, 1),
            "output_dimension": (self.output_dimension, 2),
            "input_feature": (self.input_feature, "raw_angular_velocity"),
            "initialization_mode": (self.initialization_mode, "hidden_init"),
            "initial_state_encoder_input": (
                self.initial_state_encoder_input,
                ("cos_theta0", "sin_theta0"),
            ),
            "q0_distribution": (self.q0_distribution, "uniform_minus_pi_pi"),
            "gp_family": (self.gp_family, "gaussian_process"),
            "gp_grid": (
                self.gp_grid,
                "linspace_minus1_plus1_inclusive",
            ),
            "gp_length_scale": (self.gp_length_scale, 1.0),
            "gp_marginal_standard_deviation": (
                self.gp_marginal_standard_deviation,
                1.0,
            ),
            "training_sampling": (self.training_sampling, "online_fresh"),
            "initial_state_represents": (
                self.initial_state_represents,
                "q0_before_velocity",
            ),
            "velocity_token_target": (
                self.velocity_token_target,
                "q_t_plus_1_after_velocity_update",
            ),
            "loss_mask": (self.loss_mask, "all_256_velocity_steps"),
            "target": (self.target, ("cos_theta", "sin_theta")),
        }
        for label, (actual, required) in expected.items():
            _require(actual == required, f"angular task {label} must be {required!r}")
        _require(
            math.isfinite(self.gp_cholesky_jitter)
            and self.gp_cholesky_jitter > 0.0,
            "angular task GP Cholesky jitter must be finite and positive",
        )
        _require(
            self.gp_cholesky_jitter == LEGACY_ANGULAR_GP_CHOLESKY_JITTER,
            "angular task GP Cholesky jitter must remain 1e-6",
        )

    @property
    def init_mode_for_generator(self) -> str:
        return self.initialization_mode.replace("_", "-")

    def generator_kwargs(self) -> dict[str, Any]:
        """Return every non-random generator argument, without defaults."""

        return {
            "dimensions": self.latent_dimension,
            "init_mode": self.init_mode_for_generator,
            "horizon": self.sequence_steps,
            "dt": self.delta_t,
            "gp_length_scale": self.gp_length_scale,
            "gp_std": self.gp_marginal_standard_deviation,
            "gp_jitter": self.gp_cholesky_jitter,
            "gp_grid_start": self.gp_grid_start,
            "gp_grid_stop": self.gp_grid_stop,
            "gp_grid_endpoint": self.gp_grid_endpoint,
            "q0_distribution": self.q0_distribution,
            "q0_low": self.q0_low,
            "q0_high": self.q0_high,
            "q0_high_inclusive": self.q0_high_inclusive,
            "target_indexing": self.velocity_token_target,
            "loss_mask_mode": self.loss_mask,
            "resolved_task_spec": self.resolved_payload(),
            "resolved_task_spec_sha256": self.fingerprint(),
        }

    def resolved_payload(self) -> dict[str, Any]:
        """Canonical JSON-compatible scientific task contract."""

        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "topology": self.topology,
            "latent_dimension": self.latent_dimension,
            "sequence_steps": self.sequence_steps,
            "delta_t": self.delta_t,
            "input_dimension": self.input_dimension,
            "output_dimension": self.output_dimension,
            "input_feature": self.input_feature,
            "initialization_mode": self.initialization_mode,
            "initial_state_encoder_input": list(self.initial_state_encoder_input),
            "q0": {
                "distribution": self.q0_distribution,
                "low": self.q0_low,
                "high": self.q0_high,
                "high_inclusive": self.q0_high_inclusive,
            },
            "velocity_process": {
                "family": self.gp_family,
                "grid": self.gp_grid,
                "grid_start": self.gp_grid_start,
                "grid_stop": self.gp_grid_stop,
                "grid_endpoint": self.gp_grid_endpoint,
                "length_scale": self.gp_length_scale,
                "marginal_standard_deviation": self.gp_marginal_standard_deviation,
                "gp_cholesky_jitter": self.gp_cholesky_jitter,
                "gp_cholesky_jitter_source": self.gp_cholesky_jitter_source,
                "training_sampling": self.training_sampling,
            },
            "indexing": {
                "initial_state_represents": self.initial_state_represents,
                "velocity_token_t_target": self.velocity_token_target,
            },
            "loss_mask": self.loss_mask,
            "target": list(self.target),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.resolved_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the JSON-compatible YAML protocol freeze."""

    protocol_path = Path(path) if path is not None else DEFAULT_PROTOCOL_PATH
    try:
        with protocol_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=_unique_json_object)
    except FileNotFoundError as exc:
        raise ProtocolConfigError(f"protocol file not found: {protocol_path}") from exc
    except json.JSONDecodeError as exc:
        raise ProtocolConfigError(
            f"protocol is not JSON-compatible YAML: {protocol_path}: {exc}"
        ) from exc

    _require(isinstance(data, dict), "protocol root must be an object")
    validate_protocol(data)
    return data


def _validate_sagodi_lr_selection_protocol(protocol: Mapping[str, Any]) -> None:
    """Fail-closed validation for the 100-update Ságodi LR selector.

    This freeze is deliberately not an attractor-analysis or confirmatory
    freeze.  It exists only to select one learning rate per model from the
    paper's four-value grid using the paper's update-100 training-loss rule.
    Keeping it on a dedicated validation branch prevents later selector
    conveniences from weakening either immutable v1 protocol.
    """

    _require(
        protocol.get("freeze_id") == SAGODI_LR_SELECTION_FREEZE_ID,
        "Ságodi LR selector must use its dedicated freeze_id",
    )
    _require(
        protocol.get("frozen_at_utc") == "2026-07-14",
        "Ságodi LR selector frozen date changed",
    )
    source = _require_mapping(protocol.get("source_protocol"), "source_protocol")
    _require(
        source
        == {
            "path": "CA_LRU_Sagodi_Experimental_Protocol_ko.md",
            "version": "1.0",
            "sha256": "39487213422eafdd071b541680662fd9b417be0342081c435785fd406f05b4f4",
            "normative_claim_gate_section": "3.13.2",
        },
        "Ságodi LR selector source protocol binding changed",
    )

    reporting = _require_mapping(protocol.get("reporting"), "reporting")
    _require(
        reporting
        == {
            "training_track": SAGODI_LR_SELECTION_TRACK,
            "display_label": (
                "Ságodi paper-aligned task/training shell + method-specific "
                "CA-LRU RP"
            ),
            "protocol_A_eligible": False,
            "protocol_B_confirmatory_eligible": False,
            "paper_alignment": "task_and_100_update_learning_rate_selection_shell",
            "method_specific_component": "CA-LRU_Retention_Plasticity",
            "bit_exact_official_implementation": False,
        },
        "Ságodi LR selector reporting labels changed",
    )

    scope = _require_mapping(protocol.get("scope"), "scope")
    _require(
        tuple(_require_sequence(scope.get("active_phase_ids"), "active_phase_ids"))
        == ACTIVE_PHASES,
        f"active phases must be {ACTIVE_PHASES!r}",
    )
    for key in (
        "pilot_results_are_confirmatory",
        "pilot_results_may_support_l3_claim",
        "selection_results_are_approximate_ca_evidence",
        "old_campaign_results_may_be_pooled",
    ):
        _require(scope.get(key) is False, f"selector scope {key} must be false")
    _require(
        scope.get("later_phases")
        == [
            {
                "id": "phase1_ring_main_training",
                "enabled": False,
                "activation_gate": (
                    "modelwise_learning_rate_selection_receipt_and_new_training_freeze"
                ),
            }
        ],
        "selector must not authorize main training",
    )

    seeds = _require_mapping(protocol.get("seed_policy"), "seed_policy")
    selection_seeds = tuple(
        _require_sequence(
            seeds.get("selection_model_seeds"), "selection_model_seeds"
        )
    )
    _require(
        selection_seeds == SAGODI_LR_SELECTION_SEEDS,
        f"selection seeds must be {SAGODI_LR_SELECTION_SEEDS!r}",
    )
    _unique(selection_seeds, "selection_model_seeds")
    _require(seeds.get("main_model_seeds") == [], "main seeds must be empty")
    _require(
        {key: seeds.get(key) for key in (
            "task_seed",
            "data_stream_seed",
            "evaluation_bank_seed",
            "perturbation_bank_seed",
        )}
        == {
            "task_seed": 0,
            "data_stream_seed": 0,
            "evaluation_bank_seed": 0,
            "perturbation_bank_seed": 0,
        },
        "selector shared task/data/evaluation seeds changed",
    )
    _require(
        seeds.get("independent_statistical_unit") == "trained_model_seed",
        "selector statistical unit must be the trained model seed",
    )
    _require(
        seeds.get("same_online_batch_key_across_models")
        == [
            "task",
            "latent_dimension",
            "geometry",
            "initialization_mode",
            "task_seed",
            "update_index",
        ],
        "selector common-random-number batch key changed",
    )

    phase0 = _require_mapping(protocol.get("phase0_state_audit"), "phase0_state_audit")
    _require(phase0.get("enabled") is True, "selector Phase 0 must be enabled")
    _require(
        phase0.get("blocks_dependents_on_failure") is True,
        "selector Phase 0 must block dependent training",
    )
    _require(
        tuple(phase0.get("models", [])) == SAGODI_LR_SELECTION_MODELS,
        f"selector Phase 0 models must be {SAGODI_LR_SELECTION_MODELS!r}",
    )
    _require(
        phase0.get("primary_state_rule") == "minimum_full_markov_recurrent_state",
        "selector Phase 0 primary-state rule changed",
    )
    _require(
        phase0.get("required_maps") == [
            "F0_primary",
            "F0_carrier_if_applicable",
            "F0_reported_full",
        ],
        "selector Phase 0 map inventory changed",
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
        "selector Phase 0 artifact contract changed",
    )
    declared_checks = tuple(
        _require_mapping(item, "phase0 required check").get("id")
        for item in _require_sequence(
            phase0.get("required_checks"), "phase0.required_checks"
        )
    )
    _require(
        declared_checks
        == (
            "state_transition_inventory",
            "pack_unpack_round_trip",
            "actual_blank_map",
            "blank_input_trace",
            "determinism",
            "analysis_mode",
            "float64_subset",
            "jacobian_finite_difference",
            "hidden_cache_audit",
        ),
        "selector Phase 0 check inventory changed",
    )

    phase1 = _require_mapping(protocol.get("phase1_ring_pilot"), "phase1_ring_pilot")
    _require(phase1.get("enabled") is True, "selector Phase 1 must be enabled")
    _require(phase1.get("confirmatory") is False, "selector is not confirmatory")
    _require(
        tuple(phase1.get("depends_on", [])) == ("phase0_state_audit",),
        "selector Phase 1 must depend on Phase 0",
    )
    _require(
        phase1.get("protocol_track") == SAGODI_LR_SELECTION_TRACK,
        "selector protocol_track changed",
    )
    _require(
        phase1.get("purpose") == "modelwise_learning_rate_selection_only",
        "selector purpose must remain LR selection only",
    )

    task = _require_mapping(phase1.get("task"), "phase1.task")
    AngularTaskSpec.from_protocol(protocol)
    _require(
        task
        == {
            "id": "angular_velocity_integration",
            "topology": "S1",
            "latent_dimension": 1,
            "sequence_steps": 256,
            "delta_t": 0.1,
            "input_dimension": 1,
            "output_dimension": 2,
            "input_feature": "raw_angular_velocity",
            "initialization_mode": "hidden_init",
            "initial_state_encoder_input": ["cos_theta0", "sin_theta0"],
            "theta0_distribution": "uniform_minus_pi_pi",
            "velocity_process": {
                "family": "gaussian_process",
                "normalized_time_grid": "linspace_minus1_plus1_inclusive",
                "length_scale": 1.0,
                "marginal_standard_deviation": 1.0,
                "gp_cholesky_jitter": 0.000001,
                "training_sampling": "online_fresh",
            },
            "indexing": {
                "initial_state_represents": "q0_before_velocity",
                "velocity_token_t_target": "q_t_plus_1_after_velocity_update",
            },
            "loss_mask": "all_256_velocity_steps",
            "target": ["cos_theta", "sin_theta"],
        },
        "selector angular task contract changed",
    )

    model_specs = _require_sequence(phase1.get("models"), "phase1.models")
    model_ids = tuple(
        _require_mapping(item, "selector model spec").get("id")
        for item in model_specs
    )
    _require(
        model_ids == SAGODI_LR_SELECTION_MODELS,
        f"selector models must be {SAGODI_LR_SELECTION_MODELS!r}",
    )
    _unique(model_ids, "selector model ids")
    expected_model_specs = [
        {
            "id": "ca_lru",
            "role": "proposed",
            "variant": "PAN-RNW-full",
            "hidden_width": 96,
            "parameter_count": 56834,
            "trainable_parameter_count": 56738,
            "retention_plasticity": True,
            "retention_plasticity_calls_during_selector": 0,
            "state_noise_target": "primary_markov_state",
        },
        {
            "id": "no_rp",
            "role": "mechanism_control",
            "base_model": "ca_lru",
            "variant": "PAN-RNW-full",
            "hidden_width": 96,
            "parameter_count": 56834,
            "trainable_parameter_count": 56738,
            "retention_plasticity": False,
            "outer_scaffold_identical_to_base": True,
            "state_noise_target": "primary_markov_state",
        },
        {
            "id": "gru_sagodi_width96",
            "role": "official_style_width_matched_baseline",
            "variant": "Sagodi_official_style_GRU",
            "hidden_width": 96,
            "parameter_count": 28898,
            "state_noise_target": "recurrent_hidden_state",
        },
        {
            "id": "gru_sagodi_param135",
            "role": "official_style_parameter_matched_baseline",
            "variant": "Sagodi_official_style_GRU",
            "hidden_width": 135,
            "parameter_count": 56432,
            "state_noise_target": "recurrent_hidden_state",
        },
    ]
    _require(list(model_specs) == expected_model_specs, "selector model specs changed")

    training = _require_mapping(phase1.get("training"), "phase1.training")
    _require(training.get("width") == 96, "selector CA/default width must be 96")
    architecture = _require_mapping(training.get("architecture"), "training.architecture")
    _require(
        architecture.get("initial_state_encoder")
        == {
            "input_dimension": 2,
            "output_dimension": "primary_state_dimension",
            "module": "torch.nn.Linear",
            "bias": False,
            "weight_initialization": {
                "distribution": "normal",
                "mean": 0.0,
                "standard_deviation": "1_over_sqrt_primary_state_dimension",
                "source": "Sagodi_official_W_otr",
            },
            "activation_by_model": {
                "ca_lru": "identity",
                "no_rp": "identity",
                "gru_sagodi_width96": "tanh",
                "gru_sagodi_param135": "tanh",
            },
        },
        "selector bias-free W_otr initializer contract changed",
    )
    _require(
        architecture.get("shared_builder_kwargs")
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
        "selector shared model-builder kwargs changed",
    )
    _require(
        architecture.get("ca_lru_and_no_rp")
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
        "selector CA-LRU scaffold changed",
    )
    expected_gru_common = {
        "source_code_commit": "cbd7404e9baca4b2dc291560cfc6576bb7b1f078",
        "recurrence": "one_layer_torch_GRUCell_step_equivalent_to_nn_GRU",
        "input_dimension": 1,
        "gru_bias": True,
        "bias_convention": "two_PyTorch_default_random_bias_vectors",
        "initial_state": "tanh_of_bias_free_W_otr_times_y0",
        "readout": "direct_biased_linear",
        "output_to_hidden_initialization_repair": (
            "Normal_0_1_over_sqrt_hidden_because_pinned_gru_py_left_W_otr_uninitialized"
        ),
    }
    _require(
        architecture.get("gru_sagodi_width96")
        == {
            **expected_gru_common,
            "hidden_width": 96,
            "parameter_count": 28898,
            "matching_role": "width_matched_to_CA_LRU_96",
        },
        "width-matched official-style GRU architecture changed",
    )
    _require(
        architecture.get("gru_sagodi_param135")
        == {
            **expected_gru_common,
            "hidden_width": 135,
            "parameter_count": 56432,
            "matching_role": "nearest_parameter_match_to_CA_LRU_56834",
        },
        "parameter-matched official-style GRU architecture changed",
    )

    _require(training.get("batch_size") == 64, "selector batch size must be 64")
    _require(
        training.get("optimizer_updates") == 100,
        "selector must stop at update 100",
    )
    _require(
        training.get("optimizer")
        == {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "epsilon": 0.00000001,
            "weight_decay": 0.0,
        },
        "selector Adam contract changed",
    )
    learning_rate = _require_mapping(training.get("learning_rate"), "learning_rate")
    _require(
        tuple(learning_rate.get("grid", [])) == SAGODI_LR_SELECTION_GRID,
        f"selector LR grid must be {SAGODI_LR_SELECTION_GRID!r}",
    )
    _require(
        tuple(learning_rate.get("active_launch_values", []))
        == SAGODI_LR_SELECTION_GRID,
        "all four selector learning rates must be active",
    )
    _unique(learning_rate["grid"], "selector LR grid")
    _require(
        learning_rate.get("selection_status") == "pending_frozen_selection",
        "selector results must remain pending before execution",
    )
    _require(
        learning_rate.get("selection_rule")
        == {
            "primary_metric": "mean_online_training_loss_at_update_100",
            "seed_aggregation": "arithmetic_mean_across_five_selection_model_seeds",
            "selection_scope": "separately_per_model",
            "winner": "minimum_primary_metric",
            "validation_metrics_role": "secondary_non_selecting",
            "tie_policy": "smaller_numeric_learning_rate",
            "failed_run_policy": (
                "no_scientific_failure_inference_from_nonzero_exit_oom_kill_or_invalid_receipt"
            ),
            "failed_run_retry_policy": (
                "infrastructure_or_unknown_failure_aborts_campaign_and_is_resume_eligible"
            ),
            "campaign_completion": "all_80_runs_must_have_verified_success_receipts",
        },
        "selector learning-rate decision rule changed",
    )
    _require(
        training.get("state_noise")
        == {
            "enabled": True,
            "distribution": "normal",
            "coordinate_standard_deviation": 0.1,
            "coordinate_variance": 0.01,
            "target": "primary_markov_state",
            "analysis_noise_enabled": False,
        },
        "selector state-noise contract changed",
    )
    _require(
        training.get("gradient_clipping")
        == {"policy": "none", "frozen_numeric_value": None},
        "selector must not use gradient clipping",
    )
    _require(
        training.get("checkpoint_selection")
        == "final_update_100_for_lr_selection_only",
        "selector checkpoint semantics changed",
    )
    _require(
        training.get("rp_schedule_for_ca_lru")
        == {
            "enabled_during_selector": False,
            "selector_policy": "disabled_during_100_update_lr_selection",
            "warmup_updates": 100,
            "interval_updates": 1,
            "calls_after_warmup": 0,
            "probe_batch_size": 256,
            "probe_horizon": 256,
            "probe_noise_enabled": False,
            "probe_bank": "not_materialized_during_selector",
            "eta_lambda_pilot_default": 3000.0,
            "damage_epsilon_pilot_default": 0.00003,
            "numeric_values_status": "inactive_method_specific_metadata_only",
        },
        "selector must make exactly zero RP calls",
    )

    run_matrix = _require_mapping(phase1.get("run_matrix"), "phase1.run_matrix")
    _require(
        run_matrix
        == {
            "cross_product": [
                "models",
                "selection_model_seeds",
                "active_launch_values",
            ],
            "expected_training_runs": 80,
            "task_seed_fixed": 0,
            "data_stream_seed_fixed": 0,
        },
        "selector run matrix changed",
    )
    _require(
        len(model_ids) * len(selection_seeds) * len(learning_rate["active_launch_values"])
        == 80,
        "selector run count must be exactly 80",
    )

    evaluation = _require_mapping(protocol.get("evaluation"), "evaluation")
    _require(
        evaluation
        == {
            "validation_trials": 2048,
            "id_test_trials": 2048,
            "analysis_enabled": False,
            "selection_results_are_approximate_ca_evidence": False,
            "finite_kick_horizons": [1, 5, 20, 100, 500, 1024],
            "primary_recovery_horizon": 500,
            "primary_manifold_reconstruction": (
                "track_a_task_reachable_endpoint_then_blank_periodic_spline"
            ),
            "task_conditioned_atlas_role": "correspondence_only_not_primary_projector",
            "c3_metric_labels": {
                "clean_adherence": "D_clean(H)=d_M(F0^H(m))/R_s",
                "manifold_recovery": (
                    "Q(H)=d_M(F0^H(m+delta))/max(d_M(m+delta),floor)"
                ),
            },
            "disabled_reason": "learning_rate_selection_only_not_CA_analysis",
        },
        "selector evaluation declaration changed",
    )
    _require(
        protocol.get("claim_gates")
        == {
            "status": "disabled_for_learning_rate_selection",
            "all_approximate_ca_claims_enabled": False,
            "selection_artifacts_may_be_reused_as_ca_evidence": False,
        },
        "selector must not expose approximate-CA claim gates",
    )
    _require(
        protocol.get("statistics")
        == {
            "selection": {
                "confirmatory_tests_enabled": False,
                "unit": "trained_model_seed",
                "primary_summary": "mean_online_training_loss_at_update_100",
                "validation_metrics_role": "secondary_non_selecting",
            }
        },
        "selector statistics declaration changed",
    )


def _validate_sagodi_primary_lr_selection_protocol(
    protocol: Mapping[str, Any],
) -> None:
    """Fail closed on the six-model, 120-run primary LR-selection freeze.

    This is deliberately a new schema branch.  The historical v1/v2 freezes
    remain byte-for-byte interpretable and cannot silently inherit the wider
    model registry or the primary-analysis scope introduced here.
    """

    _require(
        protocol.get("schema_version") == "2.0.0",
        "primary selector must use schema_version 2.0.0",
    )
    _require(
        protocol.get("freeze_id") == SAGODI_PRIMARY_LR_SELECTION_FREEZE_ID,
        "primary selector freeze_id changed",
    )
    _require(
        protocol.get("freeze_status") == "preregistered_training_only",
        "primary selector must remain training-only",
    )
    _require(
        protocol.get("frozen_at_utc") == "2026-07-14",
        "primary selector frozen date changed",
    )
    source = _require_mapping(protocol.get("source_protocol"), "source_protocol")
    _require(
        source.get("path")
        == "repro/sagodi_protocol/SAGODI_PRIMARY_V3_FREEZE_ko.md",
        "primary selector source path changed",
    )
    _require(source.get("version") == "3.1", "primary selector source version changed")
    digest = str(source.get("sha256", ""))
    _require(
        len(digest) == 64 and all(character in "0123456789abcdef" for character in digest),
        "primary selector source SHA-256 must be lowercase hexadecimal",
    )
    _require(
        source.get("normative_sections") == ["3", "4", "5"],
        "primary selector normative sections changed",
    )

    reporting = _require_mapping(protocol.get("reporting"), "reporting")
    _require(
        reporting.get("training_track") == SAGODI_PRIMARY_LR_SELECTION_TRACK,
        "primary selector reporting track changed",
    )
    _require(
        reporting.get("analysis_role") == "training_only_no_CA_evidence",
        "primary selector may not be labelled as CA evidence",
    )
    for key in (
        "protocol_A_eligible",
        "protocol_B_confirmatory_eligible",
        "bit_exact_official_implementation",
    ):
        _require(reporting.get(key) is False, f"primary selector {key} must be false")

    scope = _require_mapping(protocol.get("scope"), "scope")
    _require(
        tuple(_require_sequence(scope.get("active_phase_ids"), "active_phase_ids"))
        == ACTIVE_PHASES,
        f"primary selector active phases must be {ACTIVE_PHASES!r}",
    )
    _require(
        scope.get("selection_results_are_approximate_ca_evidence") is False,
        "LR selection is not approximate-CA evidence",
    )
    _require(
        scope.get("old_campaign_results_may_be_pooled") is False,
        "historical selector results may not be pooled into v3",
    )
    later = _require_sequence(scope.get("later_phases"), "scope.later_phases")
    _require(len(later) == 1, "primary selector must declare one gated main phase")
    _require(
        _require_mapping(later[0], "later phase").get("enabled") is False,
        "main phase must remain disabled in the selector freeze",
    )

    seeds = _require_mapping(protocol.get("seed_policy"), "seed_policy")
    selection_seeds = tuple(
        _require_sequence(seeds.get("selection_model_seeds"), "selection_model_seeds")
    )
    _require(
        selection_seeds == SAGODI_PRIMARY_SELECTION_SEEDS,
        f"primary selection seeds must be {SAGODI_PRIMARY_SELECTION_SEEDS!r}",
    )
    _unique(selection_seeds, "primary selection seeds")
    _require(seeds.get("main_model_seeds") == [], "selector cannot launch main seeds")
    for key in (
        "task_seed",
        "data_stream_seed",
        "evaluation_bank_seed",
        "perturbation_bank_seed",
    ):
        _require(seeds.get(key) == 0, f"primary selector {key} must be zero")
    _require(
        seeds.get("independent_statistical_unit") == "trained_model_seed",
        "trained model seed must remain the statistical unit",
    )

    phase0 = _require_mapping(protocol.get("phase0_state_audit"), "phase0_state_audit")
    _require(phase0.get("enabled") is True, "primary selector Phase 0 must be enabled")
    _require(
        phase0.get("blocks_dependents_on_failure") is True,
        "primary selector Phase 0 must block training on failure",
    )
    _require(
        tuple(phase0.get("models", [])) == SAGODI_PRIMARY_MODELS,
        f"primary Phase-0 models must be {SAGODI_PRIMARY_MODELS!r}",
    )
    _require(
        phase0.get("primary_state_rule") == "minimum_full_markov_recurrent_state",
        "primary selector must audit the minimum full Markov state",
    )
    _require(
        phase0.get("required_maps")
        == ["F0_primary", "F0_carrier_if_applicable", "F0_reported_full"],
        "primary selector blank-map inventory changed",
    )
    required_check_ids = tuple(
        _require_mapping(item, "Phase-0 check").get("id")
        for item in _require_sequence(phase0.get("required_checks"), "required_checks")
    )
    _require(
        required_check_ids
        == (
            "state_transition_inventory",
            "pack_unpack_round_trip",
            "actual_blank_map",
            "blank_input_trace",
            "determinism",
            "analysis_mode",
            "float64_subset",
            "jacobian_finite_difference",
            "hidden_cache_audit",
        ),
        "primary selector Phase-0 check inventory changed",
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
        "primary selector Phase-0 artifact inventory changed",
    )

    phase1 = _require_mapping(protocol.get("phase1_ring_pilot"), "phase1_ring_pilot")
    _require(phase1.get("enabled") is True, "primary selector training must be enabled")
    _require(phase1.get("confirmatory") is False, "LR selection is not confirmatory")
    _require(
        phase1.get("protocol_track") == SAGODI_PRIMARY_LR_SELECTION_TRACK,
        "primary selector protocol track changed",
    )
    _require(
        phase1.get("purpose") == "modelwise_learning_rate_selection_only",
        "primary selector purpose changed",
    )
    AngularTaskSpec.from_protocol(protocol)

    model_specs = _require_sequence(phase1.get("models"), "phase1.models")
    model_ids = tuple(_require_mapping(item, "model spec").get("id") for item in model_specs)
    _require(
        model_ids == SAGODI_PRIMARY_MODELS,
        f"primary selector models must be {SAGODI_PRIMARY_MODELS!r}",
    )
    _unique(model_ids, "primary selector model ids")
    for model_spec in model_specs:
        item = _require_mapping(model_spec, "primary model spec")
        model_id = str(item["id"])
        _require(
            item.get("hidden_width") == SAGODI_PRIMARY_MODEL_WIDTHS[model_id],
            f"{model_id} hidden width changed",
        )
        _require(
            item.get("parameter_count") == SAGODI_PRIMARY_PARAMETER_COUNTS[model_id],
            f"{model_id} parameter count changed",
        )

    training = _require_mapping(phase1.get("training"), "phase1.training")
    _require(training.get("width") == 96, "primary selector default width must be 96")
    architecture = _require_mapping(training.get("architecture"), "training.architecture")
    initializer = _require_mapping(
        architecture.get("initial_state_encoder"), "initial_state_encoder"
    )
    _require(initializer.get("input_dimension") == 2, "memory encoder input must be 2D")
    _require(initializer.get("bias") is False, "primary memory encoder must be bias-free")
    _require(
        initializer.get("activation_by_model")
        == {
            "rnn_param206": "identity",
            "gru_sagodi_param135": "tanh",
            "lstm_param109": "tanh",
            "lru_param96": "identity",
            "no_rp": "identity",
            "ca_lru": "identity",
        },
        "primary model initial-state activations changed",
    )
    _require(
        architecture.get("shared_builder_kwargs")
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
        "primary selector shared builder arguments changed",
    )
    _require(training.get("batch_size") == 64, "primary selector batch size must be 64")
    _require(
        training.get("optimizer_updates") == 100,
        "primary selector must run exactly 100 updates",
    )
    _require(
        training.get("optimizer")
        == {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "epsilon": 0.00000001,
            "weight_decay": 0.0,
        },
        "primary selector Adam configuration changed",
    )
    learning_rate = _require_mapping(training.get("learning_rate"), "learning_rate")
    _require(
        tuple(learning_rate.get("grid", [])) == SAGODI_LR_SELECTION_GRID,
        f"primary selector LR grid must be {SAGODI_LR_SELECTION_GRID!r}",
    )
    _require(
        tuple(learning_rate.get("active_launch_values", []))
        == SAGODI_LR_SELECTION_GRID,
        "all four primary selector LRs must be active",
    )
    rule = _require_mapping(learning_rate.get("selection_rule"), "selection_rule")
    _require(
        rule.get("primary_metric") == "mean_online_training_loss_at_update_100",
        "primary selector metric changed",
    )
    _require(rule.get("all_runs_required") is True, "all 120 runs must be required")
    _require(
        rule.get("tie_policy") == "smaller_numeric_learning_rate",
        "primary selector tie policy changed",
    )
    noise = _require_mapping(training.get("state_noise"), "state_noise")
    _require(noise.get("enabled") is True, "primary selector state noise must be enabled")
    _require(
        noise.get("coordinate_standard_deviation") == 0.1,
        "primary selector state-noise std must be 0.1",
    )
    _require(
        noise.get("target") == "minimum_full_markov_recurrent_state",
        "primary selector noise target changed",
    )
    _require(
        training.get("gradient_clipping")
        == {"policy": "none", "frozen_numeric_value": None},
        "primary selector must not clip gradients",
    )
    rp = _require_mapping(training.get("rp_schedule_for_ca_lru"), "RP schedule")
    _require(rp.get("enabled_during_selector") is False, "RP must be disabled in selection")
    _require(rp.get("calls_after_warmup") == 0, "selector must make zero RP calls")

    run_matrix = _require_mapping(phase1.get("run_matrix"), "run_matrix")
    _require(
        run_matrix.get("expected_training_runs") == 120,
        "primary selector must contain exactly 120 training runs",
    )
    _require(
        len(model_ids)
        * len(selection_seeds)
        * len(learning_rate["active_launch_values"])
        == 120,
        "primary selector cross-product is not 120 runs",
    )
    evaluation = _require_mapping(protocol.get("evaluation"), "evaluation")
    _require(evaluation.get("id_test_trials") == 2048, "selector ID bank must use 2048 trials")
    _require(evaluation.get("analysis_enabled") is False, "selector may not run CA analysis")
    gates = _require_mapping(protocol.get("claim_gates"), "claim_gates")
    _require(
        gates.get("all_approximate_ca_claims_enabled") is False,
        "primary selector CA claim gates must be disabled",
    )


def _validate_sha256(value: Any, label: str) -> None:
    text = str(value)
    _require(
        len(text) == 64
        and all(character in "0123456789abcdef" for character in text),
        f"{label} must be a lowercase SHA-256 digest",
    )


def _validate_sagodi_primary_main_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate the selector-bound 60-run main protocol materialization."""

    _require(protocol.get("schema_version") == "2.0.0", "main schema must be 2.0.0")
    _require(
        protocol.get("freeze_id") == SAGODI_PRIMARY_MAIN_FREEZE_ID,
        "main freeze_id changed",
    )
    _require(
        protocol.get("freeze_status") == "resolved_before_training",
        "main protocol must be resolved before training",
    )
    _require(
        protocol.get("campaign_mode") == "sagodi_primary_main_v3",
        "main campaign mode changed",
    )
    source = _require_mapping(protocol.get("source_protocol"), "source_protocol")
    _require(
        source.get("path")
        == "repro/sagodi_protocol/SAGODI_PRIMARY_V3_FREEZE_ko.md",
        "main source protocol path changed",
    )
    _validate_sha256(source.get("sha256"), "main source protocol hash")

    parent = _require_mapping(protocol.get("parent_selector"), "parent_selector")
    _require(
        parent.get("campaign_id") == "sagodi_six_model_lr_selection_v3",
        "main parent selector campaign id changed",
    )
    for key in (
        "scientific_identity",
        "manifest_sha256",
        "summary_sha256",
        "selection_receipt_sha256",
        "completion_receipt_sha256",
    ):
        _validate_sha256(parent.get(key), f"parent_selector.{key}")

    reporting = _require_mapping(protocol.get("reporting"), "reporting")
    _require(
        reporting.get("training_track") == SAGODI_PRIMARY_MAIN_TRACK,
        "main reporting track changed",
    )
    _require(
        reporting.get("analysis_role") == "primary_comparative_training",
        "main reporting role changed",
    )
    _require(
        reporting.get("bit_exact_official_implementation") is False,
        "mixed project baselines are not a bit-exact official implementation",
    )

    seeds = _require_mapping(protocol.get("seed_policy"), "seed_policy")
    _require(
        tuple(seeds.get("selection_model_seeds", []))
        == SAGODI_PRIMARY_SELECTION_SEEDS,
        "main must preserve selector seeds as provenance",
    )
    _require(
        tuple(seeds.get("main_model_seeds", [])) == SAGODI_PRIMARY_MAIN_SEEDS,
        f"main seeds must be {SAGODI_PRIMARY_MAIN_SEEDS!r}",
    )
    _unique(seeds["main_model_seeds"], "main model seeds")
    for key in (
        "task_seed",
        "data_stream_seed",
        "evaluation_bank_seed",
        "perturbation_bank_seed",
    ):
        _require(seeds.get(key) == 0, f"main {key} must be zero")

    phase0 = _require_mapping(protocol.get("phase0_state_audit"), "phase0_state_audit")
    _require(phase0.get("enabled") is True, "main Phase 0 must be enabled")
    _require(
        phase0.get("blocks_dependents_on_failure") is True,
        "main Phase 0 must block training on failure",
    )
    _require(
        tuple(phase0.get("models", [])) == SAGODI_PRIMARY_MODELS,
        f"main Phase-0 models must be {SAGODI_PRIMARY_MODELS!r}",
    )
    _require(
        phase0.get("primary_state_rule") == "minimum_full_markov_recurrent_state",
        "main must audit the minimum full Markov state",
    )

    phase1 = _require_mapping(protocol.get("phase1_ring_pilot"), "phase1_ring_pilot")
    _require(phase1.get("enabled") is True, "main training must be enabled")
    _require(
        phase1.get("confirmatory") is True,
        "resolved main task/training comparison must remain preregistered confirmatory",
    )
    _require(
        phase1.get("protocol_track") == SAGODI_PRIMARY_MAIN_TRACK,
        "main protocol track changed",
    )
    _require(
        phase1.get("purpose") == "six_model_primary_main_training",
        "main purpose changed",
    )
    AngularTaskSpec.from_protocol(protocol)
    model_specs = _require_sequence(phase1.get("models"), "phase1.models")
    model_ids = tuple(_require_mapping(item, "model spec").get("id") for item in model_specs)
    _require(model_ids == SAGODI_PRIMARY_MODELS, "main model order changed")
    for model_spec in model_specs:
        item = _require_mapping(model_spec, "main model spec")
        model_id = str(item["id"])
        _require(
            item.get("hidden_width") == SAGODI_PRIMARY_MODEL_WIDTHS[model_id],
            f"main {model_id} width changed",
        )
        _require(
            item.get("parameter_count") == SAGODI_PRIMARY_PARAMETER_COUNTS[model_id],
            f"main {model_id} parameter count changed",
        )

    training = _require_mapping(phase1.get("training"), "phase1.training")
    _require(training.get("batch_size") == 64, "main batch size must be 64")
    _require(training.get("optimizer_updates") == 5000, "main must run 5000 updates")
    _require(
        training.get("optimizer")
        == {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "epsilon": 0.00000001,
            "weight_decay": 0.0,
        },
        "main Adam configuration changed",
    )
    noise = _require_mapping(training.get("state_noise"), "state_noise")
    _require(noise.get("enabled") is True, "main state noise must be enabled")
    _require(
        noise.get("coordinate_standard_deviation") == 0.1,
        "main state-noise std must be 0.1",
    )
    _require(
        noise.get("target") == "minimum_full_markov_recurrent_state",
        "main state-noise target changed",
    )
    _require(
        training.get("gradient_clipping")
        == {"policy": "none", "frozen_numeric_value": None},
        "main training must not clip gradients",
    )
    learning_rate = _require_mapping(training.get("learning_rate"), "learning_rate")
    _require(
        tuple(learning_rate.get("grid", [])) == SAGODI_LR_SELECTION_GRID,
        "main must preserve the selector grid",
    )
    _require(
        learning_rate.get("source") == "selector_bound",
        "main learning rates must be selector-bound",
    )
    selected = _require_mapping(
        learning_rate.get("selected_by_model"), "selected_by_model"
    )
    _require(
        set(selected) == set(SAGODI_PRIMARY_MODELS),
        "main selected-LR model set changed",
    )
    selected_values = tuple(float(selected[model]) for model in SAGODI_PRIMARY_MODELS)
    _require(
        all(value in SAGODI_LR_SELECTION_GRID for value in selected_values),
        "main selected LR lies outside the preregistered grid",
    )
    active = tuple(float(value) for value in learning_rate.get("active_launch_values", []))
    _require(len(active) == len(set(active)), "main active LR values contain duplicates")
    _require(
        set(active) == set(selected_values),
        "main active LR set must equal the modelwise selector winners",
    )

    rp = _require_mapping(training.get("rp_schedule_for_ca_lru"), "RP schedule")
    expected_rp = {
        "enabled_during_training": True,
        "warmup_updates": 1500,
        "interval_updates": 50,
        "calls_after_warmup": 70,
        "probe_batch_size": 96,
        "probe_horizon": 256,
        "blank_ablation_horizon": 500,
        "probe_noise_enabled": False,
        "eta_lambda": 3000.0,
        "damage_epsilon": 0.00003,
    }
    _require(dict(rp) == expected_rp, "main CA-LRU RP schedule changed")

    run_matrix = _require_mapping(phase1.get("run_matrix"), "run_matrix")
    _require(
        run_matrix.get("expected_training_runs") == 60,
        "main run matrix must contain exactly 60 runs",
    )
    evaluation = _require_mapping(protocol.get("evaluation"), "evaluation")
    _require(evaluation.get("id_test_trials") == 2048, "main ID bank must use 2048 trials")
    gates = _require_mapping(protocol.get("claim_gates"), "claim_gates")
    _require(
        gates.get("all_approximate_ca_claims_enabled") is False,
        "main training cannot enable numerical CA claim gates",
    )


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate the frozen scope and conservative ambiguity resolutions."""

    if protocol.get("freeze_id") == SAGODI_PRIMARY_LR_SELECTION_FREEZE_ID:
        _validate_sagodi_primary_lr_selection_protocol(protocol)
        return
    if protocol.get("freeze_id") == SAGODI_PRIMARY_MAIN_FREEZE_ID:
        _validate_sagodi_primary_main_protocol(protocol)
        return
    _require(protocol.get("schema_version") == "1.0.0", "unsupported schema_version")
    _require(protocol.get("freeze_status") == "pilot_only", "freeze must be pilot_only")
    if protocol.get("freeze_id") == SAGODI_LR_SELECTION_FREEZE_ID:
        _validate_sagodi_lr_selection_protocol(protocol)
        return

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
    protocol_track = phase1.get("protocol_track")
    _require(
        protocol_track in {PROTOCOL_A_TRACK, NATIVE_RECIPE_TRACK},
        "Phase 1 protocol_track is unsupported",
    )
    native_recipe = protocol_track == NATIVE_RECIPE_TRACK
    if native_recipe:
        _require(
            protocol.get("freeze_id") == "calru_native_sagodi_ring_pilot_v1",
            "native-recipe track must use its dedicated freeze_id",
        )
        reporting = _require_mapping(protocol.get("reporting"), "reporting")
        _require(
            reporting
            == {
                "training_track": NATIVE_RECIPE_TRACK,
                "display_label": "CA-LRU native training on Ságodi ring task",
                "protocol_A_eligible": False,
                "protocol_B_confirmatory_eligible": False,
            },
            "native-recipe reporting labels differ from the freeze",
        )
        launch_policy = _require_mapping(
            phase1.get("launch_policy"), "phase1.launch_policy"
        )
        _require(
            launch_policy
            == {
                "mode": "sentinel_then_remaining",
                "sentinel_model": "ca_lru",
                "sentinel_seed": 100,
                "gate": "verified_training_completion_receipt",
            },
            "native-recipe launch policy differs from the freeze",
        )
    else:
        _require(
            protocol.get("freeze_id") == "sagodi_phase01_ring_pilot_v1",
            "Protocol A track must use its dedicated freeze_id",
        )
        _require(
            "reporting" not in protocol,
            "Protocol A must not declare native-recipe reporting labels",
        )
        _require(
            "launch_policy" not in phase1,
            "Protocol A must not declare the native sentinel launch policy",
        )

    task = _require_mapping(phase1.get("task"), "phase1.task")
    # Resolve and validate the complete executable task contract.  This closes
    # the v1 gap where dt/GP fields changed the protocol fingerprint without
    # changing the actual Python generator defaults.
    AngularTaskSpec.from_protocol(protocol)
    _require(task.get("id") == "angular_velocity_integration", "Phase 1 task must be angular integration")
    _require(task.get("topology") == "S1", "Phase 1 must be a ring task")
    _require(task.get("latent_dimension") == 1, "Phase 1 latent dimension must be one")
    _require(task.get("sequence_steps") == 256, "Phase 1 T must be 256")
    _require(task.get("initialization_mode") == "hidden_init", "Phase 1 uses hidden initialization")
    _require(task.get("input_feature") == "raw_angular_velocity", "Phase 1 uses raw velocity")
    velocity_process = _require_mapping(
        task.get("velocity_process"), "phase1.task.velocity_process"
    )
    if native_recipe:
        _require(
            velocity_process.get("gp_cholesky_jitter") == 0.000001,
            "native-recipe GP Cholesky jitter must remain 1e-6",
        )
    else:
        _require(
            "gp_cholesky_jitter" not in velocity_process,
            "Protocol A freeze must not be retroactively changed with GP jitter metadata",
        )
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
    expected_initializer = {
        "input_dimension": 2,
        "output_dimension": "primary_state_dimension",
        "module": "torch.nn.Linear",
        "bias": True,
    }
    if native_recipe:
        expected_initializer = {
            "input_dimension": 2,
            "output_dimension": "primary_state_dimension",
            "module": "torch.nn.Linear",
            "bias": False,
            "weight_initialization": {
                "distribution": "normal",
                "mean": 0.0,
                "standard_deviation": "1_over_sqrt_primary_state_dimension",
                "source": "Sagodi_official_W_otr",
            },
        }
    _require(
        initializer == expected_initializer,
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
    optimizer = _require_mapping(training.get("optimizer"), "phase1.training.optimizer")
    learning_rate = _require_mapping(training.get("learning_rate"), "learning_rate")
    noise = _require_mapping(training.get("state_noise"), "phase1.training.state_noise")
    rp = _require_mapping(training.get("rp_schedule_for_ca_lru"), "rp schedule")
    clipping = _require_mapping(
        training.get("gradient_clipping"), "phase1.training.gradient_clipping"
    )
    if native_recipe:
        _require(training.get("batch_size") == 256, "native recipe batch size must be 256")
        _require(
            training.get("optimizer_updates") == 10000,
            "native recipe updates must be 10000",
        )
        _require(
            optimizer
            == {
                "name": "AdamW",
                "betas": [0.9, 0.999],
                "epsilon": 0.00000001,
                "weight_decay": 0.00001,
            },
            "native recipe optimizer differs from AdamW freeze",
        )
        _require(
            learning_rate
            == {
                "active_launch_values": [0.001],
                "frozen_pilot_default": 0.001,
                "selection_status": "inherited_from_successful_exp88_recipe_not_selected_on_sagodi_results",
                "future_grid_sweep_enabled": False,
            },
            "native recipe learning-rate block differs from the freeze",
        )
        _require(
            noise
            == {
                "enabled": False,
                "distribution": "none",
                "coordinate_standard_deviation": 0.0,
                "coordinate_variance": 0.0,
                "analysis_noise_enabled": False,
            },
            "native recipe must disable training state noise",
        )
        _require(
            clipping
            == {"policy": "global_norm", "frozen_numeric_value": 1.0},
            "native recipe gradient clipping must be global norm 1.0",
        )
        _require(
            training.get("progress_logging")
            == {
                "interval_updates": 100,
                "atomic_json_path": "progress.json",
                "record_pre_clip_global_gradient_norm": True,
            },
            "native recipe progress logging differs from the freeze",
        )
        _require(
            rp
            == {
                "warmup_updates": 3000,
                "interval_updates": 100,
                "calls_after_warmup": 70,
                "probe_batch_size": 96,
                "probe_horizon": 256,
                "blank_ablation_horizon": 500,
                "probe_noise_enabled": False,
                "probe_bank": "deterministic_by_rp_call_index",
                "pre_warmup_score_only_probes": "not_executed",
                "legacy_pre_warmup_probe_difference": "Exp88_computed_score_only_probes_steps_100_through_3000",
                "eta_lambda_pilot_default": 3000.0,
                "damage_epsilon_pilot_default": 0.0001,
                "default_status": "numeric_values_inherited_with_declared_task_and_probe_stream_adaptations",
            },
            "native recipe RP schedule differs from the freeze",
        )
    else:
        _require(
            "progress_logging" not in training,
            "Protocol A must not declare native-recipe progress logging",
        )
        _require(training.get("batch_size") == 64, "Protocol A batch size must be 64")
        _require(training.get("optimizer_updates") == 5000, "Protocol A updates must be 5000")
        _require(optimizer.get("name") == "Adam", "Protocol A optimizer must be Adam")
        _require(optimizer.get("betas") == [0.9, 0.999], "Adam betas must be frozen")
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
        _require(noise.get("enabled") is True, "Protocol A state noise must be enabled")
        _require(noise.get("coordinate_standard_deviation") == 0.1, "noise std must be 0.1")
        _require(noise.get("analysis_noise_enabled") is False, "analysis noise must be disabled")
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
    """Expand the exact training matrix authorized by the selected freeze."""

    validate_protocol(protocol)
    phase1 = protocol["phase1_ring_pilot"]
    training = phase1["training"]
    seed_policy = protocol["seed_policy"]
    selection_freeze = phase1["protocol_track"] in {
        SAGODI_LR_SELECTION_TRACK,
        SAGODI_PRIMARY_LR_SELECTION_TRACK,
    }
    model_seeds = seed_policy[
        "selection_model_seeds" if selection_freeze else "pilot_model_seeds"
    ]
    runs: list[dict[str, Any]] = []

    for model_spec in phase1["models"]:
        width = int(model_spec.get("hidden_width", training["width"]))
        for model_seed in model_seeds:
            for learning_rate in training["learning_rate"]["active_launch_values"]:
                lr_token = _float_token(float(learning_rate))
                run_id = (
                    f"phase1__angle_integrate__{model_spec['id']}__"
                    f"w{width:03d}__seed{model_seed:03d}__lr{lr_token}"
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
                        "width": width,
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
