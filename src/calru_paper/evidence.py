#!/usr/bin/env python3
"""Build deterministic, auditable evidence tables for the CA-LRU paper.

The raw result files predate the final paper terminology.  This module keeps
paper-facing labels (for example, ``CA-LRU``) separate from the historical
``model`` and ``tag`` values stored in those files.  Every JSON group is
validated before aggregation, and all tables are built and validated in
memory before any output file is replaced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "paper" / "evidence" / "raw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper" / "evidence" / "tables"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "configs" / "evidence_manifest.json"

EXPECTED_SEED_IDS = (0, 1, 2)
TASK_ORDER = (
    "ring_hold",
    "ring_integrate",
    "torus_hold",
    "torus_integrate",
    "complex_curve_hold",
    "complex_curve_integrate",
    "surface_hold",
    "surface_integrate",
)

TABLE_ORDER = (
    "main_manifold_metrics.csv",
    "update_ablation_metrics.csv",
    "ring_integrate_ood_metrics.csv",
    "epsilon_sensitivity.csv",
    "rp_control_metrics.csv",
    "retention_scaling_metrics.csv",
    "ring_transport_seed_values.csv",
    "ring_transport_summary.csv",
    "discrete_control_metrics.csv",
)

REQUIRED_METRIC_GROUPS = (
    "main",
    "ood",
    "retention_scaling",
    "ring_transport",
    "discrete_kway",
    "discrete_flipflop",
)

CSV_FLOAT_SIGNIFICANT_DIGITS = 12

BASELINE_SPECS = {
    "rnn": ("RNN", "tanh", "RNN"),
    "gru": ("GRU", "", "GRU"),
    "lstm": ("LSTM", "", "LSTM"),
    "lru": ("LRU", "full", "LRU full"),
    "ssm": ("SSM", "diagonal", "SSM"),
}

AGGREGATE_COLUMNS = (
    "run_id",
    "paper_task",
    "paper_model",
    "paper_variant",
    "epsilon",
    "legacy_task",
    "legacy_model",
    "legacy_tag",
    "n_seeds",
    "seed_ids",
    "source_pattern",
    "source_files",
)

TRANSPORT_SEED_COLUMNS = (
    "run_id",
    "paper_task",
    "paper_model",
    "paper_variant",
    "epsilon",
    "legacy_task",
    "legacy_model",
    "legacy_tag",
    "seed_id",
    "blank_steps",
    "source_file",
)

LEGACY_PAPER_LABEL_TOKENS = ("pan", "camn", "am-lru", "am_lru")


class EvidenceError(ValueError):
    """Raised when evidence input or generated output violates the contract."""


@dataclass(frozen=True)
class MetricSpec:
    """A canonical output metric and its historical raw-data key."""

    canonical_name: str
    raw_key: str


@dataclass(frozen=True)
class EvidenceConfig:
    """Validated subset of the evidence manifest used by the builder."""

    expected_seed_ids: Tuple[int, ...]
    selected_epsilon_by_task: Mapping[str, str]
    expected_rows: Mapping[str, int]
    metric_groups: Mapping[str, Tuple[MetricSpec, ...]]
    manifest_path: Optional[Path] = None


@dataclass(frozen=True)
class RunSpec:
    """Expected identity and provenance for one three-seed run group."""

    run_id: str
    paper_task: str
    paper_model: str
    paper_variant: str
    epsilon: str
    legacy_task: str
    legacy_model: str
    legacy_tag: str
    source_pattern: str
    metrics: Tuple[MetricSpec, ...]


@dataclass(frozen=True)
class EvidenceTable:
    """A deterministic in-memory CSV table."""

    filename: str
    fieldnames: Tuple[str, ...]
    rows: Tuple[Mapping[str, object], ...]


def _read_manifest_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read evidence manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"evidence manifest must contain a JSON object: {path}")
    return value


def _parse_expected_seeds(raw: object) -> Tuple[int, ...]:
    if not isinstance(raw, list) or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw):
        raise EvidenceError("manifest expected_seed_ids must be a list of integers")
    seeds = tuple(raw)
    if len(set(seeds)) != len(seeds):
        raise EvidenceError(f"manifest expected_seed_ids contains duplicates: {seeds}")
    if seeds != EXPECTED_SEED_IDS:
        raise EvidenceError(
            f"this evidence release requires expected_seed_ids={EXPECTED_SEED_IDS}, got {seeds}"
        )
    return seeds


def _parse_metric_groups(raw: object) -> Dict[str, Tuple[MetricSpec, ...]]:
    if raw is None:
        raise EvidenceError("manifest is missing metric_raw_key_mapping")
    if not isinstance(raw, dict):
        raise EvidenceError("manifest metric_raw_key_mapping must be an object")
    groups: Dict[str, Tuple[MetricSpec, ...]] = {}
    for group_name, mapping in raw.items():
        if not isinstance(group_name, str) or not isinstance(mapping, dict) or not mapping:
            raise EvidenceError(f"invalid metric mapping for group {group_name!r}")
        specs: List[MetricSpec] = []
        seen_raw = set()
        for canonical_name, raw_key in mapping.items():
            if not isinstance(canonical_name, str) or not canonical_name:
                raise EvidenceError(f"invalid canonical metric name in group {group_name!r}")
            if not isinstance(raw_key, str) or not raw_key:
                raise EvidenceError(f"invalid raw metric key for {canonical_name!r}")
            if raw_key in seen_raw:
                raise EvidenceError(f"duplicate raw metric key {raw_key!r} in group {group_name!r}")
            seen_raw.add(raw_key)
            specs.append(MetricSpec(canonical_name, raw_key))
        groups[group_name] = tuple(specs)
    missing = set(REQUIRED_METRIC_GROUPS) - set(groups)
    if missing:
        raise EvidenceError(f"manifest is missing metric groups: {sorted(missing)}")
    return groups


def load_config(manifest_path: Optional[Path] = None) -> EvidenceConfig:
    """Load and strictly validate the checked-in aggregation contract."""

    path = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST_PATH
    if not path.is_file():
        raise EvidenceError(f"manifest path is not a file: {path}")
    manifest = _read_manifest_object(path)
    resolved_manifest = path.resolve()
    if manifest.get("schema_version") != 1:
        raise EvidenceError(
            f"unsupported or missing manifest schema_version: "
            f"{manifest.get('schema_version')!r}"
        )

    seeds = _parse_expected_seeds(manifest.get("expected_seed_ids"))
    aggregation = manifest.get("aggregation")
    if not isinstance(aggregation, dict):
        raise EvidenceError("manifest aggregation must be an object")
    digits = aggregation.get("float_serialization_significant_digits")
    if digits != CSV_FLOAT_SIGNIFICANT_DIGITS:
        raise EvidenceError(
            "manifest float_serialization_significant_digits must be "
            f"{CSV_FLOAT_SIGNIFICANT_DIGITS}, got {digits!r}"
        )

    raw_selected = manifest.get("selected_epsilon_by_task")
    if not isinstance(raw_selected, dict):
        raise EvidenceError("manifest selected_epsilon_by_task must be an object")
    selected_keys = set(raw_selected)
    expected_task_keys = set(TASK_ORDER)
    if selected_keys != expected_task_keys:
        raise EvidenceError(
            "manifest selected_epsilon_by_task keys must exactly match the release tasks; "
            f"missing={sorted(expected_task_keys - selected_keys)}, "
            f"extra={sorted(selected_keys - expected_task_keys)}"
        )
    selected: Dict[str, str] = {}
    for task in TASK_ORDER:
        epsilon = raw_selected[task]
        if not isinstance(epsilon, str) or not epsilon:
            raise EvidenceError(f"invalid selected epsilon entry: {task!r} -> {epsilon!r}")
        selected[task] = epsilon

    raw_tables = manifest.get("tables")
    if not isinstance(raw_tables, dict):
        raise EvidenceError("manifest tables must be an object")
    missing_tables = set(TABLE_ORDER) - set(raw_tables)
    if missing_tables:
        raise EvidenceError(
            f"manifest is missing table contracts: {sorted(missing_tables)}"
        )
    expected_rows: Dict[str, int] = {}
    for table_name in TABLE_ORDER:
        table_config = raw_tables[table_name]
        if not isinstance(table_config, dict):
            raise EvidenceError(f"invalid table manifest entry: {table_name!r}")
        row_count = table_config.get("expected_rows")
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
            raise EvidenceError(f"invalid expected_rows for {table_name}: {row_count!r}")
        expected_rows[table_name] = row_count

    metric_groups = _parse_metric_groups(manifest.get("metric_raw_key_mapping"))
    return EvidenceConfig(
        expected_seed_ids=seeds,
        selected_epsilon_by_task=selected,
        expected_rows=expected_rows,
        metric_groups=metric_groups,
        manifest_path=resolved_manifest,
    )


def sample_sd(values: Sequence[float]) -> float:
    """Return sample standard deviation (ddof=1), or zero for one value."""

    if not values:
        raise EvidenceError("sample_sd requires at least one value")
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _json_finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{context} must be a JSON number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceError(f"{context} must be finite, got {value!r}")
    return number


def _csv_finite_number(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise EvidenceError(f"{context} must be numeric, got {value!r}")
    try:
        number = float(value)  # CSV values are strings by definition.
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{context} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise EvidenceError(f"{context} must be finite, got {value!r}")
    return number


def _validate_json_identity(record: Mapping[str, object], spec: RunSpec, path: Path) -> int:
    expected = {
        "task": spec.legacy_task,
        "model": spec.legacy_model,
        "tag": spec.legacy_tag,
    }
    for key, expected_value in expected.items():
        if key not in record:
            raise EvidenceError(f"{path}: missing identity field {key!r}")
        if record[key] != expected_value:
            raise EvidenceError(
                f"{path}: expected raw {key}={expected_value!r}, got {record[key]!r}"
            )
    seed = record.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvidenceError(f"{path}: seed must be an integer, got {seed!r}")
    return seed


def load_json_runs(
    source_root: Path,
    spec: RunSpec,
    expected_seed_ids: Sequence[int] = EXPECTED_SEED_IDS,
) -> List[Dict[str, object]]:
    """Load and strictly validate one JSON run group."""

    root = Path(source_root)
    if not root.is_dir():
        raise EvidenceError(f"source root is not a directory: {root}")
    paths = sorted(
        (path for path in root.glob(spec.source_pattern) if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths:
        raise EvidenceError(f"{spec.run_id}: source pattern matched no files: {spec.source_pattern}")

    by_seed: Dict[int, Dict[str, object]] = {}
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvidenceError(f"cannot read JSON evidence file {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise EvidenceError(f"{path}: expected a JSON object")
        seed = _validate_json_identity(raw, spec, path)
        if seed in by_seed:
            previous = by_seed[seed]["_source_file"]
            raise EvidenceError(
                f"{spec.run_id}: duplicate seed {seed} in {previous!r} and "
                f"{path.relative_to(root).as_posix()!r}"
            )
        record: Dict[str, object] = dict(raw)
        record["_source_file"] = path.relative_to(root).as_posix()
        for metric in spec.metrics:
            if metric.raw_key not in record:
                raise EvidenceError(f"{path}: missing required metric {metric.raw_key!r}")
            record[metric.raw_key] = _json_finite_number(
                record[metric.raw_key], f"{path}:{metric.raw_key}"
            )
        by_seed[seed] = record

    expected = tuple(expected_seed_ids)
    actual = tuple(sorted(by_seed))
    if actual != expected:
        raise EvidenceError(f"{spec.run_id}: expected exact seed ids {expected}, got {actual}")
    return [by_seed[seed] for seed in expected]


def _paper_identity(spec: RunSpec) -> Dict[str, object]:
    return {
        "run_id": spec.run_id,
        "paper_task": spec.paper_task,
        "paper_model": spec.paper_model,
        "paper_variant": spec.paper_variant,
        "epsilon": spec.epsilon,
        "legacy_task": spec.legacy_task,
        "legacy_model": spec.legacy_model,
        "legacy_tag": spec.legacy_tag,
    }


def summarize_runs(
    rows: Sequence[Mapping[str, object]],
    spec: RunSpec,
    expected_seed_ids: Sequence[int] = EXPECTED_SEED_IDS,
) -> Dict[str, object]:
    """Aggregate a validated run group without performing file I/O."""

    by_seed: Dict[int, Mapping[str, object]] = {}
    for row in rows:
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise EvidenceError(f"{spec.run_id}: invalid seed value {seed!r}")
        if seed in by_seed:
            raise EvidenceError(f"{spec.run_id}: duplicate seed {seed}")
        by_seed[seed] = row
    expected = tuple(expected_seed_ids)
    actual = tuple(sorted(by_seed))
    if actual != expected:
        raise EvidenceError(f"{spec.run_id}: expected exact seed ids {expected}, got {actual}")

    output = _paper_identity(spec)
    output.update(
        {
            "n_seeds": len(expected),
            "seed_ids": ",".join(str(seed) for seed in expected),
            "source_pattern": spec.source_pattern,
            "source_files": ";".join(str(by_seed[seed].get("_source_file", "")) for seed in expected),
        }
    )
    for metric in spec.metrics:
        values: List[float] = []
        for seed in expected:
            row = by_seed[seed]
            if metric.raw_key not in row:
                raise EvidenceError(
                    f"{spec.run_id}: seed {seed} is missing required metric {metric.raw_key!r}"
                )
            values.append(_json_finite_number(row[metric.raw_key], f"{spec.run_id}:{metric.raw_key}"))
        output[f"{metric.canonical_name}_mean"] = statistics.mean(values)
        output[f"{metric.canonical_name}_sd"] = sample_sd(values)
    return output


def _load_and_summarize(source_root: Path, config: EvidenceConfig, spec: RunSpec) -> Dict[str, object]:
    rows = load_json_runs(source_root, spec, config.expected_seed_ids)
    return summarize_runs(rows, spec, config.expected_seed_ids)


def _metric_columns(metrics: Sequence[MetricSpec]) -> Tuple[str, ...]:
    return tuple(column for metric in metrics for column in (f"{metric.canonical_name}_mean", f"{metric.canonical_name}_sd"))


def _without_metrics(
    metrics: Sequence[MetricSpec], canonical_names: Sequence[str]
) -> Tuple[MetricSpec, ...]:
    """Return the metrics applicable to a model that lacks named diagnostics."""

    excluded = set(canonical_names)
    return tuple(metric for metric in metrics if metric.canonical_name not in excluded)


def _normalise_rows(
    filename: str,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> EvidenceTable:
    schema = tuple(fieldnames)
    if len(schema) != len(set(schema)):
        raise EvidenceError(f"{filename}: duplicate columns in schema")
    allowed = set(schema)
    normalised: List[Mapping[str, object]] = []
    for index, row in enumerate(rows):
        extras = set(row) - allowed
        if extras:
            raise EvidenceError(f"{filename}: row {index} has unexpected fields: {sorted(extras)}")
        normalised.append({field: row.get(field, "") for field in schema})
    return EvidenceTable(filename, schema, tuple(normalised))


def _baseline_run_spec(
    *,
    run_id: str,
    paper_task: str,
    legacy_task: str,
    tag: str,
    source_pattern: str,
    metrics: Tuple[MetricSpec, ...],
) -> RunSpec:
    paper_model, paper_variant, legacy_model = BASELINE_SPECS[tag]
    return RunSpec(
        run_id=run_id,
        paper_task=paper_task,
        paper_model=paper_model,
        paper_variant=paper_variant,
        epsilon="",
        legacy_task=legacy_task,
        legacy_model=legacy_model,
        legacy_tag=tag,
        source_pattern=source_pattern,
        metrics=metrics,
    )


def build_main_table(source_root: Path, config: EvidenceConfig) -> EvidenceTable:
    metrics = config.metric_groups["main"]
    rows: List[Mapping[str, object]] = []
    for task in TASK_ORDER:
        epsilon = config.selected_epsilon_by_task[task]
        calru = RunSpec(
            run_id=f"main.{task}.ca_lru",
            paper_task=task,
            paper_model="CA-LRU",
            paper_variant="state-dependent update",
            epsilon=epsilon,
            legacy_task=task,
            legacy_model="PAN-RNW-full",
            legacy_tag=f"am_lru_rnw_eps{epsilon}",
            source_pattern=f"exp88_writer_sweep_results/{task}_am_lru_rnw_eps{epsilon}_seed*.json",
            metrics=metrics,
        )
        rows.append(_load_and_summarize(source_root, config, calru))
        for tag in ("gru", "lru", "lstm", "ssm", "rnn"):
            # Retention coordinates are defined for diagonal LRU/SSM-style
            # state updates, but not for gated or tanh recurrent baselines.
            # Those two cells are explicitly N/A; every metric that is
            # applicable to the selected baseline remains strictly required.
            baseline_metrics = metrics
            if tag in ("gru", "lstm", "rnn"):
                baseline_metrics = _without_metrics(
                    metrics, ("persistent_coordinate_count", "retention_sum")
                )
            spec = _baseline_run_spec(
                run_id=f"main.{task}.{tag}",
                paper_task=task,
                legacy_task=task,
                tag=tag,
                source_pattern=f"exp88_manifold_main_results/{task}_{tag}_seed*.json",
                metrics=baseline_metrics,
            )
            rows.append(_load_and_summarize(source_root, config, spec))
    return _normalise_rows(
        "main_manifold_metrics.csv", AGGREGATE_COLUMNS + _metric_columns(metrics), rows
    )


def build_update_ablation_table(source_root: Path, config: EvidenceConfig) -> EvidenceTable:
    metrics = config.metric_groups["main"]
    rows: List[Mapping[str, object]] = []
    for task in ("ring_integrate", "torus_integrate", "complex_curve_integrate", "surface_integrate"):
        epsilon = config.selected_epsilon_by_task[task]
        variants = (
            (
                "state_dependent",
                "state-dependent update",
                "PAN-RNW-full",
                f"am_lru_rnw_eps{epsilon}",
                f"exp88_writer_sweep_results/{task}_am_lru_rnw_eps{epsilon}_seed*.json",
            ),
            (
                "input_only",
                "input-only update",
                "PAN-NW-full",
                f"am_lru_nw_eps{epsilon}",
                f"exp88_writer_sweep_results/{task}_am_lru_nw_eps{epsilon}_seed*.json",
            ),
            (
                "linear",
                "linear update",
                "PAN-full",
                f"am_lru_eps{epsilon}",
                f"exp88_manifold_main_results/{task}_am_lru_eps{epsilon}_seed*.json",
            ),
        )
        for slug, variant, legacy_model, legacy_tag, pattern in variants:
            spec = RunSpec(
                run_id=f"update_ablation.{task}.{slug}",
                paper_task=task,
                paper_model="CA-LRU",
                paper_variant=variant,
                epsilon=epsilon,
                legacy_task=task,
                legacy_model=legacy_model,
                legacy_tag=legacy_tag,
                source_pattern=pattern,
                metrics=metrics,
            )
            rows.append(_load_and_summarize(source_root, config, spec))
    return _normalise_rows(
        "update_ablation_metrics.csv", AGGREGATE_COLUMNS + _metric_columns(metrics), rows
    )


def build_ood_table(source_root: Path, config: EvidenceConfig) -> EvidenceTable:
    metrics = config.metric_groups["ood"]
    task = "ring_integrate"
    specs = [
        RunSpec(
            run_id="ood.ring_integrate.ca_lru",
            paper_task=task,
            paper_model="CA-LRU",
            paper_variant="state-dependent update",
            epsilon="1e-4",
            legacy_task=task,
            legacy_model="PAN-RNW-full",
            legacy_tag="am_lru_rnw_eps1e-4",
            source_pattern="exp88_writer_sweep_results/ring_integrate_am_lru_rnw_eps1e-4_seed*.json",
            metrics=metrics,
        ),
        RunSpec(
            run_id="ood.ring_integrate.linear_update",
            paper_task=task,
            paper_model="CA-LRU",
            paper_variant="linear update",
            epsilon="3e-5",
            legacy_task=task,
            legacy_model="PAN-full",
            legacy_tag="am_lru_eps3e-5",
            source_pattern="exp88_manifold_main_results/ring_integrate_am_lru_eps3e-5_seed*.json",
            metrics=metrics,
        ),
    ]
    for tag in ("gru", "lstm", "lru", "ssm", "rnn"):
        specs.append(
            _baseline_run_spec(
                run_id=f"ood.ring_integrate.{tag}",
                paper_task=task,
                legacy_task=task,
                tag=tag,
                source_pattern=f"exp88_manifold_main_results/ring_integrate_{tag}_seed*.json",
                metrics=metrics,
            )
        )
    rows = [_load_and_summarize(source_root, config, spec) for spec in specs]
    return _normalise_rows(
        "ring_integrate_ood_metrics.csv", AGGREGATE_COLUMNS + _metric_columns(metrics), rows
    )


def build_epsilon_sensitivity_table(source_root: Path, config: EvidenceConfig) -> EvidenceTable:
    metrics = config.metric_groups["main"]
    rows: List[Mapping[str, object]] = []
    for task in TASK_ORDER:
        for epsilon in ("0", "3e-5", "1e-4"):
            legacy_tag = f"am_lru_rnw_eps{epsilon}"
            spec = RunSpec(
                run_id=f"epsilon_sensitivity.{task}.eps{epsilon}",
                paper_task=task,
                paper_model="CA-LRU",
                paper_variant="state-dependent update",
                epsilon=epsilon,
                legacy_task=task,
                legacy_model="PAN-RNW-full",
                legacy_tag=legacy_tag,
                source_pattern=f"exp88_writer_sweep_results/{task}_{legacy_tag}_seed*.json",
                metrics=metrics,
            )
            rows.append(_load_and_summarize(source_root, config, spec))
    return _normalise_rows(
        "epsilon_sensitivity.csv", AGGREGATE_COLUMNS + _metric_columns(metrics), rows
    )


def build_rp_control_table(source_root: Path, config: EvidenceConfig) -> EvidenceTable:
    metrics = config.metric_groups["main"]
    rows: List[Mapping[str, object]] = []

    # These are historical linear-scaffold controls, not matched final-scaffold
    # ablations.  The paper variant says so explicitly while provenance retains
    # the original model and tag.
    for task in ("ring_hold", "ring_integrate"):
        controls = (
            ("no_rp_linear", "no RP (linear scaffold)", "am_lru_eta0"),
            ("all_slow_linear", "all-slow retention 0.999 (linear scaffold)", "am_lru_allslow"),
        )
        for slug, variant, legacy_tag in controls:
            spec = RunSpec(
                run_id=f"rp_control.{task}.{slug}",
                paper_task=task,
                paper_model="CA-LRU control",
                paper_variant=variant,
                epsilon="",
                legacy_task=task,
                legacy_model="PAN-full",
                legacy_tag=legacy_tag,
                source_pattern=f"exp88_manifold_main_results/{task}_{legacy_tag}_seed*.json",
                metrics=metrics,
            )
            rows.append(_load_and_summarize(source_root, config, spec))

    for task in ("ring_hold", "ring_integrate", "torus_integrate"):
        controls = (
            ("aligned_rp", "aligned RP", f"{task}_am_lru_rnw_eps3e-5", "am_lru_rnw_eps3e-5"),
            (
                "fixed_permutation",
                "fixed-permutation RP scores",
                f"{task}_camn_shufmatch_eps3e-5",
                "camn_shufmatch_eps3e-5",
            ),
            (
                "fresh_shuffle",
                "freshly shuffled RP scores",
                f"{task}_camn_shuffle_eps3e-5",
                "camn_shuffle_eps3e-5",
            ),
        )
        for slug, variant, file_stem, legacy_tag in controls:
            spec = RunSpec(
                run_id=f"rp_control.{task}.{slug}",
                paper_task=task,
                paper_model="CA-LRU",
                paper_variant=variant,
                epsilon="3e-5",
                legacy_task=task,
                legacy_model="PAN-RNW-full",
                legacy_tag=legacy_tag,
                source_pattern=f"exp88_writer_sweep_results/{file_stem}_seed*.json",
                metrics=metrics,
            )
            rows.append(_load_and_summarize(source_root, config, spec))
    return _normalise_rows(
        "rp_control_metrics.csv", AGGREGATE_COLUMNS + _metric_columns(metrics), rows
    )


def build_retention_scaling_table(source_root: Path, config: EvidenceConfig) -> EvidenceTable:
    metrics = config.metric_groups["retention_scaling"]
    rows: List[Mapping[str, object]] = []
    for dimension in (1, 2, 4, 8, 16):
        spec = RunSpec(
            run_id=f"retention_scaling.line_integrate.d{dimension}",
            paper_task="line_integrate",
            paper_model="CA-LRU",
            paper_variant="state-dependent update",
            epsilon="1e-4",
            legacy_task="line_integrate",
            legacy_model="PAN-RNW-full",
            legacy_tag="camn_rnw_eps1e-4",
            source_pattern=f"exp72_line_integrate_10k_d{dimension}_camn_rnw_eps1e-4_results/*.json",
            metrics=metrics,
        )
        row = _load_and_summarize(source_root, config, spec)
        row["memory_dimension"] = dimension
        rows.append(row)
    columns = AGGREGATE_COLUMNS + ("memory_dimension",) + _metric_columns(metrics)
    return _normalise_rows("retention_scaling_metrics.csv", columns, rows)


@dataclass(frozen=True)
class _TransportSpec:
    run_id: str
    paper_model: str
    paper_variant: str
    epsilon: str
    legacy_model: str
    legacy_tag: str


TRANSPORT_SPECS = (
    _TransportSpec(
        "ring_transport.ca_lru",
        "CA-LRU",
        "state-dependent update",
        "1e-4",
        "PAN-RNW-full",
        "am_lru_rnw_eps1e-4",
    ),
    _TransportSpec(
        "ring_transport.input_only",
        "CA-LRU",
        "input-only update",
        "1e-4",
        "PAN-NW-full",
        "am_lru_nw_eps1e-4",
    ),
    _TransportSpec(
        "ring_transport.linear",
        "CA-LRU",
        "linear update",
        "1e-4",
        "PAN-full",
        "am_lru_eps1e-4",
    ),
    _TransportSpec("ring_transport.gru", "GRU", "", "", "GRU", "gru"),
    _TransportSpec("ring_transport.lru", "LRU", "full", "", "LRU full", "lru"),
)

TRANSPORT_SOURCE_FILES = {
    0: "analysis_exp88_ring_transport/ring_transport_writer_seed0.csv",
    1: "analysis_exp88_ring_transport/ring_transport_seed1.csv",
    2: "analysis_exp88_ring_transport/ring_transport_seed2.csv",
}


def _transport_identity(spec: _TransportSpec) -> Dict[str, object]:
    return {
        "run_id": spec.run_id,
        "paper_task": "ring_integrate",
        "paper_model": spec.paper_model,
        "paper_variant": spec.paper_variant,
        "epsilon": spec.epsilon,
        "legacy_task": "ring_integrate",
        "legacy_model": spec.legacy_model,
        "legacy_tag": spec.legacy_tag,
    }


def _load_transport_seed_rows(source_root: Path, config: EvidenceConfig) -> List[Dict[str, object]]:
    root = Path(source_root)
    if not root.is_dir():
        raise EvidenceError(f"source root is not a directory: {root}")
    metrics = config.metric_groups["ring_transport"]
    output: List[Dict[str, object]] = []
    for seed in config.expected_seed_ids:
        relative = TRANSPORT_SOURCE_FILES.get(seed)
        if relative is None:
            raise EvidenceError(f"no transport source file is configured for seed {seed}")
        path = root / relative
        if not path.is_file():
            raise EvidenceError(f"missing transport source file: {path}")
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                raw_rows = list(csv.DictReader(handle))
        except OSError as exc:
            raise EvidenceError(f"cannot read transport CSV {path}: {exc}") from exc
        if not raw_rows:
            raise EvidenceError(f"transport CSV contains no rows: {path}")
        required_columns = {"tag", "blank_steps"} | {metric.raw_key for metric in metrics}
        missing_columns = required_columns - set(raw_rows[0])
        if missing_columns:
            raise EvidenceError(f"{path}: missing columns {sorted(missing_columns)}")

        for spec in TRANSPORT_SPECS:
            matches = [
                row
                for row in raw_rows
                if row.get("tag") == spec.legacy_tag and row.get("blank_steps") == "2000"
            ]
            if len(matches) != 1:
                raise EvidenceError(
                    f"{path}: expected exactly one row for tag={spec.legacy_tag!r}, "
                    f"blank_steps=2000; got {len(matches)}"
                )
            raw = matches[0]
            row = _transport_identity(spec)
            row.update(
                {
                    "seed_id": seed,
                    "blank_steps": 2000,
                    "source_file": relative,
                }
            )
            for metric in metrics:
                row[metric.canonical_name] = _csv_finite_number(
                    raw.get(metric.raw_key), f"{path}:{metric.raw_key}"
                )
            output.append(row)
    # Stable model-major, then seed-major order is easier to audit than the
    # historical CSV row order.
    run_order = {spec.run_id: index for index, spec in enumerate(TRANSPORT_SPECS)}
    output.sort(key=lambda row: (run_order[str(row["run_id"])], int(row["seed_id"])))
    return output


def build_transport_tables(
    source_root: Path, config: EvidenceConfig
) -> Tuple[EvidenceTable, EvidenceTable]:
    metrics = config.metric_groups["ring_transport"]
    seed_rows = _load_transport_seed_rows(source_root, config)
    seed_table = _normalise_rows(
        "ring_transport_seed_values.csv",
        TRANSPORT_SEED_COLUMNS + tuple(metric.canonical_name for metric in metrics),
        seed_rows,
    )

    summary_rows: List[Mapping[str, object]] = []
    for spec in TRANSPORT_SPECS:
        subset = [row for row in seed_rows if row["run_id"] == spec.run_id]
        seed_map = {int(row["seed_id"]): row for row in subset}
        actual = tuple(sorted(seed_map))
        if actual != config.expected_seed_ids or len(subset) != len(config.expected_seed_ids):
            raise EvidenceError(
                f"{spec.run_id}: expected one transport row for seeds {config.expected_seed_ids}, got {actual}"
            )
        row = _transport_identity(spec)
        source_files = [str(seed_map[seed]["source_file"]) for seed in config.expected_seed_ids]
        row.update(
            {
                "n_seeds": len(config.expected_seed_ids),
                "seed_ids": ",".join(str(seed) for seed in config.expected_seed_ids),
                "source_pattern": "analysis_exp88_ring_transport/ring_transport*_seed*.csv",
                "source_files": ";".join(source_files),
            }
        )
        for metric in metrics:
            values = [float(seed_map[seed][metric.canonical_name]) for seed in config.expected_seed_ids]
            row[f"{metric.canonical_name}_mean"] = statistics.mean(values)
            row[f"{metric.canonical_name}_sd"] = sample_sd(values)
        summary_rows.append(row)
    summary_table = _normalise_rows(
        "ring_transport_summary.csv",
        AGGREGATE_COLUMNS + _metric_columns(metrics),
        summary_rows,
    )
    return seed_table, summary_table


def build_discrete_table(source_root: Path, config: EvidenceConfig) -> EvidenceTable:
    kway_metrics = config.metric_groups["discrete_kway"]
    flipflop_metrics = config.metric_groups["discrete_flipflop"]
    rows: List[Mapping[str, object]] = []
    variants = (
        ("rnn", "RNN", "tanh", "", "RNN", "rnn"),
        ("gru", "GRU", "", "", "GRU", "gru"),
        ("lstm", "LSTM", "", "", "LSTM", "lstm"),
        ("lru", "LRU", "full", "", "LRU full", "lru"),
        ("ca_lru_eps3e-5", "CA-LRU", "state-dependent update", "3e-5", "PAN-RNW-full", "camn_eps3e-5"),
        ("ca_lru_eps1e-4", "CA-LRU", "state-dependent update", "1e-4", "PAN-RNW-full", "camn_eps1e-4"),
    )
    task_specs = (
        ("kway_hold_k16", "kway_hold", kway_metrics),
        ("flipflop_n4", "flipflop", flipflop_metrics),
    )
    for slug, paper_model, paper_variant, epsilon, legacy_model, legacy_tag in variants:
        for paper_task, legacy_task, metrics in task_specs:
            spec = RunSpec(
                run_id=f"discrete.{paper_task}.{slug}",
                paper_task=paper_task,
                paper_model=paper_model,
                paper_variant=paper_variant,
                epsilon=epsilon,
                legacy_task=legacy_task,
                legacy_model=legacy_model,
                legacy_tag=legacy_tag,
                source_pattern=f"exp89_discrete_kway_quick_results/{paper_task}_{legacy_tag}_seed*.json",
                metrics=metrics,
            )
            rows.append(_load_and_summarize(source_root, config, spec))
    all_metrics = kway_metrics + flipflop_metrics
    return _normalise_rows(
        "discrete_control_metrics.csv", AGGREGATE_COLUMNS + _metric_columns(all_metrics), rows
    )


def _validate_paper_labels(filename: str, row_index: int, row: Mapping[str, object]) -> None:
    for field in ("paper_model", "paper_variant"):
        value = str(row.get(field, "")).lower()
        for token in LEGACY_PAPER_LABEL_TOKENS:
            if token in value:
                raise EvidenceError(
                    f"{filename}: row {row_index} leaks legacy token {token!r} into {field}"
                )


def validate_tables(tables: Mapping[str, EvidenceTable], config: EvidenceConfig) -> None:
    """Validate all nine complete tables before any file is written."""

    actual_names = tuple(tables)
    if actual_names != TABLE_ORDER:
        raise EvidenceError(f"expected table order {TABLE_ORDER}, got {actual_names}")
    for filename in TABLE_ORDER:
        table = tables[filename]
        if table.filename != filename:
            raise EvidenceError(f"table key {filename!r} disagrees with filename {table.filename!r}")
        expected_rows = config.expected_rows.get(filename)
        if expected_rows is None:
            raise EvidenceError(f"no expected row count configured for {filename}")
        if len(table.rows) != expected_rows:
            raise EvidenceError(
                f"{filename}: expected {expected_rows} rows, generated {len(table.rows)}"
            )
        if len(table.fieldnames) != len(set(table.fieldnames)):
            raise EvidenceError(f"{filename}: duplicate field names")
        seen_keys = set()
        for index, row in enumerate(table.rows):
            if tuple(row) != table.fieldnames:
                raise EvidenceError(f"{filename}: row {index} does not match the fixed schema")
            _validate_paper_labels(filename, index, row)
            if filename == "ring_transport_seed_values.csv":
                key = (row["run_id"], row["seed_id"])
            else:
                key = row["run_id"]
                if row.get("n_seeds") != len(config.expected_seed_ids):
                    raise EvidenceError(f"{filename}: row {index} has invalid n_seeds")
                expected_seed_text = ",".join(str(seed) for seed in config.expected_seed_ids)
                if row.get("seed_ids") != expected_seed_text:
                    raise EvidenceError(f"{filename}: row {index} has invalid seed_ids")
            if key in seen_keys:
                raise EvidenceError(f"{filename}: duplicate row key {key!r}")
            seen_keys.add(key)
            if not row.get("legacy_tag") or not row.get("legacy_model"):
                raise EvidenceError(f"{filename}: row {index} is missing legacy provenance")
            for field, value in row.items():
                if field.endswith("_mean") or field.endswith("_sd"):
                    if value == "":  # Task-inapplicable cells in the discrete table.
                        continue
                    _json_finite_number(value, f"{filename}:row {index}:{field}")


def build_all_tables(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    config: Optional[EvidenceConfig] = None,
) -> Dict[str, EvidenceTable]:
    """Build and validate all tables in memory, without writing files."""

    active_config = config if config is not None else load_config()
    source = Path(source_root)
    transport_seed, transport_summary = build_transport_tables(source, active_config)
    tables = {
        "main_manifold_metrics.csv": build_main_table(source, active_config),
        "update_ablation_metrics.csv": build_update_ablation_table(source, active_config),
        "ring_integrate_ood_metrics.csv": build_ood_table(source, active_config),
        "epsilon_sensitivity.csv": build_epsilon_sensitivity_table(source, active_config),
        "rp_control_metrics.csv": build_rp_control_table(source, active_config),
        "retention_scaling_metrics.csv": build_retention_scaling_table(source, active_config),
        "ring_transport_seed_values.csv": transport_seed,
        "ring_transport_summary.csv": transport_summary,
        "discrete_control_metrics.csv": build_discrete_table(source, active_config),
    }
    validate_tables(tables, active_config)
    return tables


def _canonical_csv_value(value: object) -> object:
    """Serialize floats identically across supported Python minor versions."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceError(f"cannot render non-finite CSV value: {value!r}")
        return format(value, f".{CSV_FLOAT_SIGNIFICANT_DIGITS}g")
    return value


def render_csv(table: EvidenceTable) -> bytes:
    """Render one table with deterministic UTF-8 and Unix newlines."""

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=table.fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(
        {
            field: _canonical_csv_value(row[field])
            for field in table.fieldnames
        }
        for row in table.rows
    )
    return stream.getvalue().encode("utf-8")


def write_tables_atomic(
    tables: Mapping[str, EvidenceTable],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Tuple[Path, ...]:
    """Stage every table, then atomically replace each destination file."""

    output = Path(output_dir)
    if output.exists() and not output.is_dir():
        raise EvidenceError(f"output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".evidence-stage-", dir=str(output)))
    written: List[Path] = []
    try:
        for filename in TABLE_ORDER:
            if filename not in tables:
                raise EvidenceError(f"cannot write incomplete table set; missing {filename}")
            (stage / filename).write_bytes(render_csv(tables[filename]))
        for filename in TABLE_ORDER:
            destination = output / filename
            os.replace(stage / filename, destination)
            written.append(destination)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return tuple(written)


def build_and_write(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Optional[Path] = None,
) -> Tuple[Path, ...]:
    """High-level aggregation entry point used by the CLI."""

    config = load_config(manifest_path)
    tables = build_all_tables(source_root, config)
    return write_tables_atomic(tables, output_dir)


def check_reproducibility(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Optional[Path] = None,
) -> Tuple[str, ...]:
    """Compare a clean in-memory rebuild with the committed CSV bytes."""

    config = load_config(manifest_path)
    tables = build_all_tables(source_root, config)
    output = Path(output_dir)
    differences: List[str] = []
    for filename in TABLE_ORDER:
        path = output / filename
        expected = render_csv(tables[filename])
        if not path.is_file():
            differences.append(f"{filename}: missing")
            continue
        actual = path.read_bytes()
        if actual != expected:
            expected_hash = hashlib.sha256(expected).hexdigest()
            actual_hash = hashlib.sha256(actual).hexdigest()
            differences.append(
                f"{filename}: differs (generated sha256={expected_hash}, existing sha256={actual_hash})"
            )
    if output.is_dir():
        extras = sorted(path.name for path in output.glob("*.csv") if path.name not in TABLE_ORDER)
        differences.extend(f"{filename}: unexpected CSV" for filename in extras)
    return tuple(differences)


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"raw evidence root (default: {DEFAULT_SOURCE_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"generated/committed table directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"evidence manifest (default: {DEFAULT_MANIFEST_PATH})",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _common_parser("Build all CA-LRU paper evidence tables.")
    args = parser.parse_args(argv)
    try:
        written = build_and_write(args.source_root, args.output_dir, args.manifest)
    except EvidenceError as exc:
        print(f"evidence build failed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {len(written)} evidence tables to {Path(args.output_dir).resolve()}")
    return 0


def reproducibility_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _common_parser("Check committed CA-LRU evidence tables against a clean rebuild.")
    args = parser.parse_args(argv)
    try:
        differences = check_reproducibility(args.source_root, args.output_dir, args.manifest)
    except EvidenceError as exc:
        print(f"reproducibility check failed: {exc}", file=sys.stderr)
        return 2
    if differences:
        print("evidence tables are not reproducible:", file=sys.stderr)
        for difference in differences:
            print(f"  - {difference}", file=sys.stderr)
        return 1
    print(f"all {len(TABLE_ORDER)} evidence tables reproduce byte-for-byte")
    return 0


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SOURCE_ROOT",
    "EXPECTED_SEED_IDS",
    "EvidenceConfig",
    "EvidenceError",
    "EvidenceTable",
    "MetricSpec",
    "RunSpec",
    "TABLE_ORDER",
    "build_all_tables",
    "build_and_write",
    "check_reproducibility",
    "load_config",
    "load_json_runs",
    "main",
    "render_csv",
    "reproducibility_main",
    "sample_sd",
    "summarize_runs",
    "validate_tables",
    "write_tables_atomic",
]


if __name__ == "__main__":
    raise SystemExit(main())
