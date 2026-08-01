"""Validated configuration for the primary RP-LRU protocol."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_PROTOCOL = Path(__file__).with_name("protocol.json")


def _tuple_int(values: Sequence[Any], name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value < 0 for value in result):
        raise ValueError(f"{name} must contain non-negative integers")
    return result


def _tuple_float(values: Sequence[Any], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


@dataclass(frozen=True)
class TaskConfig:
    initial_value_low: float
    initial_value_high: float
    delta_low: float
    delta_high: float
    trajectory_low: float
    trajectory_high: float
    maximum_rejection_attempts: int
    train_update_count_min: int
    train_update_count_max: int
    train_hold_min: int
    train_hold_max: int

    def validate(self) -> None:
        if not self.initial_value_low < self.initial_value_high:
            raise ValueError("initial-value range is empty")
        if not self.delta_low < self.delta_high:
            raise ValueError("delta range is empty")
        if not self.trajectory_low < self.trajectory_high:
            raise ValueError("trajectory range is empty")
        if self.maximum_rejection_attempts <= 0:
            raise ValueError("maximum_rejection_attempts must be positive")
        if not 0 <= self.train_update_count_min <= self.train_update_count_max:
            raise ValueError("invalid training update-count range")
        if not 0 <= self.train_hold_min <= self.train_hold_max:
            raise ValueError("invalid training hold range")
        if (
            self.initial_value_low < self.trajectory_low
            or self.initial_value_high > self.trajectory_high
        ):
            raise ValueError("initial values must fit inside the trajectory bounds")


@dataclass(frozen=True)
class TrainingConfig:
    optimizer_updates: int
    batch_size: int
    gradient_clip_norm: float
    weight_decay: float
    validation_interval: int
    validation_trajectories: int
    baseline_learning_rates: tuple[float, ...]
    replicate_seeds: tuple[int, ...]

    def validate(self) -> None:
        for name in (
            "optimizer_updates",
            "batch_size",
            "validation_interval",
            "validation_trajectories",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if any(value <= 0 for value in self.baseline_learning_rates):
            raise ValueError("learning rates must be positive")


@dataclass(frozen=True)
class RPConfig:
    initial_lambdas: tuple[float, ...]
    initial_lambda_ranges: tuple[tuple[float, float], ...]
    eta_lambdas: tuple[float, ...]
    retention_thresholds: tuple[float, ...]
    probe_horizons: tuple[int, ...]
    warmup_updates: int
    interval_updates: int
    probe_pairs: int
    normalization_epsilon: float

    def validate(self) -> None:
        if not self.initial_lambdas and not self.initial_lambda_ranges:
            raise ValueError("RP requires scalar lambdas or uniform lambda ranges")
        if any(not 0 < value < 1 for value in self.initial_lambdas):
            raise ValueError("RP initial lambda must be strictly between zero and one")
        if any(
            not 0 < low < high < 1
            for low, high in self.initial_lambda_ranges
        ):
            raise ValueError(
                "RP uniform lambda ranges must satisfy 0 < low < high < 1"
            )
        if any(value <= 0 for value in self.eta_lambdas):
            raise ValueError("RP eta_lambda must be positive")
        if any(value < 0 for value in self.retention_thresholds):
            raise ValueError("RP thresholds must be non-negative")
        if any(value <= 0 for value in self.probe_horizons):
            raise ValueError("RP horizons must be positive")
        if self.warmup_updates < 0 or self.interval_updates <= 0:
            raise ValueError("invalid RP schedule")
        if self.probe_pairs <= 0 or self.normalization_epsilon <= 0:
            raise ValueError("invalid RP probe normalization")


@dataclass(frozen=True)
class EvaluationConfig:
    update_counts: tuple[int, ...]
    segment_hold_lengths: tuple[int, ...]
    trajectories: int
    bank_seed: int
    id_update_count_max: int
    id_hold_max: int

    def validate(self) -> None:
        if any(value < 0 for value in self.update_counts):
            raise ValueError("evaluation update counts must be non-negative")
        if any(value < 0 for value in self.segment_hold_lengths):
            raise ValueError("evaluation hold lengths must be non-negative")
        if self.trajectories <= 0:
            raise ValueError("evaluation trajectories must be positive")

    def regime(self, update_count: int, segment_hold: int) -> str:
        update_ood = int(update_count) > self.id_update_count_max
        horizon_ood = int(segment_hold) > self.id_hold_max
        if update_ood and horizon_ood:
            return "joint_ood"
        if update_ood:
            return "update_ood"
        if horizon_ood:
            return "horizon_ood"
        return "id"


@dataclass(frozen=True)
class Protocol:
    raw: Mapping[str, Any]
    source: Path
    sha256: str
    schema_version: int
    experiment_id: str
    dimensions: tuple[int, ...]
    rp_lru_hidden_size: int
    parameter_count_mode: str
    maximum_parameter_gap: float
    minimum_width: int
    maximum_width: int
    task: TaskConfig
    training: TrainingConfig
    rp: RPConfig
    evaluation: EvaluationConfig
    retention_modes: tuple[str, ...]
    all_slow_lambdas: tuple[float, ...]
    near_unit_exponents: tuple[int, ...]
    pca_components: int

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema version {self.schema_version}")
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if any(value <= 0 for value in self.dimensions):
            raise ValueError("memory dimensions must be positive")
        if self.rp_lru_hidden_size <= 0:
            raise ValueError("RP-LRU hidden size must be positive")
        if self.parameter_count_mode not in {"total_parameters", "trainable_parameters"}:
            raise ValueError("unsupported parameter count mode")
        if not 0 <= self.maximum_parameter_gap < 1:
            raise ValueError("maximum parameter gap must be in [0, 1)")
        if not 1 <= self.minimum_width <= self.maximum_width:
            raise ValueError("invalid parameter-matching width range")
        allowed_modes = {"frozen", "bptt", "all_slow"}
        if set(self.retention_modes) - allowed_modes:
            raise ValueError("unknown retention ablation")
        if any(not 0 < value <= 1 for value in self.all_slow_lambdas):
            raise ValueError("all-slow lambda must be in (0, 1]")
        if any(value <= 0 for value in self.near_unit_exponents):
            raise ValueError("near-unit exponents must be positive")
        if self.pca_components <= 0:
            raise ValueError("pca_components must be positive")
        self.task.validate()
        self.training.validate()
        self.rp.validate()
        self.evaluation.validate()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> Protocol:
    source = Path(path).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    task = raw["task"]
    training = raw["training"]
    rp = raw["rp"]
    evaluation = raw["evaluation"]
    matching = raw["parameter_matching"]
    ablations = raw["ablations"]
    analysis = raw["analysis"]
    protocol = Protocol(
        raw=raw,
        source=source,
        sha256=hashlib.sha256(_canonical_bytes(raw)).hexdigest(),
        schema_version=int(raw["schema_version"]),
        experiment_id=str(raw["experiment_id"]),
        dimensions=_tuple_int(raw["dimensions"], "dimensions"),
        rp_lru_hidden_size=int(raw["rp_lru_hidden_size"]),
        parameter_count_mode=str(matching["count"]),
        maximum_parameter_gap=float(matching["maximum_relative_gap"]),
        minimum_width=int(matching["minimum_width"]),
        maximum_width=int(matching["maximum_width"]),
        task=TaskConfig(**task),
        training=TrainingConfig(
            optimizer_updates=int(training["optimizer_updates"]),
            batch_size=int(training["batch_size"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            weight_decay=float(training["weight_decay"]),
            validation_interval=int(training["validation_interval"]),
            validation_trajectories=int(training["validation_trajectories"]),
            baseline_learning_rates=_tuple_float(
                training["baseline_learning_rates"], "baseline_learning_rates"
            ),
            replicate_seeds=_tuple_int(
                training["replicate_seeds"], "replicate_seeds"
            ),
        ),
        rp=RPConfig(
            initial_lambdas=tuple(
                float(value) for value in rp.get("initial_lambdas", [])
            ),
            initial_lambda_ranges=tuple(
                (float(bounds[0]), float(bounds[1]))
                for bounds in rp.get("initial_lambda_ranges", [])
            ),
            eta_lambdas=_tuple_float(rp["eta_lambdas"], "eta_lambdas"),
            retention_thresholds=_tuple_float(
                rp["retention_thresholds"], "retention_thresholds"
            ),
            probe_horizons=_tuple_int(rp["probe_horizons"], "probe_horizons"),
            warmup_updates=int(rp["warmup_updates"]),
            interval_updates=int(rp["interval_updates"]),
            probe_pairs=int(rp["probe_pairs"]),
            normalization_epsilon=float(rp["normalization_epsilon"]),
        ),
        evaluation=EvaluationConfig(
            update_counts=_tuple_int(
                evaluation["update_counts"], "evaluation.update_counts"
            ),
            segment_hold_lengths=_tuple_int(
                evaluation["segment_hold_lengths"],
                "evaluation.segment_hold_lengths",
            ),
            trajectories=int(evaluation["trajectories"]),
            bank_seed=int(evaluation["bank_seed"]),
            id_update_count_max=int(evaluation["id_update_count_max"]),
            id_hold_max=int(evaluation["id_hold_max"]),
        ),
        retention_modes=tuple(str(value) for value in ablations["retention_modes"]),
        all_slow_lambdas=_tuple_float(
            ablations["all_slow_lambdas"], "all_slow_lambdas"
        ),
        near_unit_exponents=_tuple_int(
            analysis["near_unit_exponents"], "near_unit_exponents"
        ),
        pca_components=int(analysis["pca_components"]),
    )
    protocol.validate()
    return protocol
