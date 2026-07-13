"""Deterministic, descriptive-only aggregation for the Phase-1 ring pilot.

The pilot aggregation deliberately does not perform hypothesis tests.  Every
pre-registered pilot run remains in the all-started denominator, while
manifold summaries are also reported conditionally on passing the task gate.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import canonical_bytes, sha256_file, strict_json_load


AGGREGATION_SCHEMA_VERSION = 1
AGGREGATION_DIRECTORY = "pilot_aggregation"
SUMMARY_NAME = "pilot_summary.json"
MATRIX_NAME = "pilot_run_matrix.csv"


def aggregation_specification() -> dict[str, Any]:
    """Return the immutable paths and schema signed by the campaign manifest."""

    return {
        "schema_version": AGGREGATION_SCHEMA_VERSION,
        "directory": AGGREGATION_DIRECTORY,
        "summary": SUMMARY_NAME,
        "run_matrix_csv": MATRIX_NAME,
        "denominator": "all_preregistered_pilot_runs_that_reached_valid_analysis_receipts",
        "inference": "descriptive_nonconfirmatory_no_p_values",
    }


def _claim_path(root: Path, manifest: Mapping[str, Any], run_id: str) -> Path:
    expectation = manifest["receipt_expectations"].get(f"analysis:{run_id}")
    if not isinstance(expectation, Mapping):
        raise ValueError(f"missing analysis receipt expectation for {run_id}")
    return root / str(expectation["output"]) / "claim_gate.json"


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _flatten_scalars(value: Any, prefix: str = "value") -> dict[str, float]:
    """Flatten finite numeric leaves of gate values, excluding booleans/lists."""

    scalar = _finite_number(value)
    if scalar is not None:
        return {prefix: scalar}
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key in sorted(value, key=str):
        result.update(_flatten_scalars(value[key], f"{prefix}.{key}"))
    return result


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    position = (len(sorted_values) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        (1.0 - weight) * sorted_values[lower] + weight * sorted_values[upper]
    )


def _numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("numeric summary requires at least one value")
    mean = math.fsum(ordered) / len(ordered)
    sample_std = None
    if len(ordered) >= 2:
        sample_std = math.sqrt(
            math.fsum((value - mean) ** 2 for value in ordered)
            / (len(ordered) - 1)
        )
    q25 = _linear_quantile(ordered, 0.25)
    median = _linear_quantile(ordered, 0.5)
    q75 = _linear_quantile(ordered, 0.75)
    return {
        "n": len(ordered),
        "mean": mean,
        "sample_std": sample_std,
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
    }


def _status_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(record["status"]) for record in records)
    return {key: counts[key] for key in sorted(counts)}


def _gate_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gate_ids = sorted(
        {gate_id for row in rows for gate_id in row["gates"]}
    )
    result: dict[str, Any] = {}
    for gate_id in gate_ids:
        records = [row["gates"][gate_id] for row in rows if gate_id in row["gates"]]
        scalar_values: dict[str, list[float]] = defaultdict(list)
        for record in records:
            for path, value in _flatten_scalars(record.get("value")).items():
                scalar_values[path].append(value)
        result[gate_id] = {
            "available_run_count": len(records),
            "status_counts": _status_counts(records),
            "passed_true_count": sum(record.get("passed") is True for record in records),
            "passed_false_count": sum(record.get("passed") is False for record in records),
            "passed_null_count": sum(record.get("passed") is None for record in records),
            "numeric_value_summaries": {
                path: _numeric_summary(values)
                for path, values in sorted(scalar_values.items())
            },
        }
    return result


def _model_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["task_success"]]
    return {
        "all_started": {
            "denominator": len(rows),
            "task_gate_status_counts": _status_counts(
                [row["gates"]["task"] for row in rows]
            ),
            "task_success_count": len(successful),
            "task_success_fraction": len(successful) / len(rows) if rows else None,
            "gate_summaries": _gate_aggregate(rows),
        },
        "task_success_conditional": {
            "denominator": len(successful),
            "conditioning_rule": "task gate status=passed and passed=true",
            "gate_summaries": _gate_aggregate(successful),
        },
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], gate_ids: Sequence[str]) -> bytes:
    fields = [
        "run_id",
        "model",
        "model_seed",
        "learning_rate",
        "task_status",
        "task_passed",
        "task_value_json",
    ]
    for gate_id in gate_ids:
        fields.extend(
            [
                f"{gate_id}__status",
                f"{gate_id}__passed",
                f"{gate_id}__value_json",
            ]
        )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        task = row["gates"]["task"]
        record: dict[str, Any] = {
            "run_id": row["run_id"],
            "model": row["model"],
            "model_seed": row["model_seed"],
            "learning_rate": format(float(row["learning_rate"]), ".17g"),
            "task_status": task["status"],
            "task_passed": "" if task.get("passed") is None else str(bool(task["passed"])).lower(),
            "task_value_json": json.dumps(
                task.get("value"), sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
        }
        for gate_id in gate_ids:
            gate = row["gates"].get(gate_id)
            if gate is None:
                record[f"{gate_id}__status"] = "missing"
                record[f"{gate_id}__passed"] = ""
                record[f"{gate_id}__value_json"] = "null"
                continue
            record[f"{gate_id}__status"] = gate["status"]
            record[f"{gate_id}__passed"] = (
                "" if gate.get("passed") is None else str(bool(gate["passed"])).lower()
            )
            record[f"{gate_id}__value_json"] = json.dumps(
                gate.get("value"), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        writer.writerow(record)
    return handle.getvalue().encode("utf-8")


def build_pilot_aggregation(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes]:
    """Build canonical JSON data and CSV bytes from all frozen pilot runs."""

    root = Path(root).resolve()
    run_matrix = manifest.get("run_matrix")
    if not isinstance(run_matrix, list) or not run_matrix:
        raise ValueError("campaign manifest has no non-empty run_matrix")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run in run_matrix:
        run_id = str(run["run_id"])
        if run_id in seen:
            raise ValueError(f"duplicate run_id in campaign manifest: {run_id}")
        seen.add(run_id)
        claim_path = _claim_path(root, manifest, run_id)
        claim = strict_json_load(claim_path)
        gates = claim.get("gates")
        if not isinstance(gates, dict) or not isinstance(gates.get("task"), dict):
            raise ValueError(f"claim gate artifact lacks a task gate: {claim_path}")
        normalized_gates: dict[str, dict[str, Any]] = {}
        for gate_id, gate in sorted(gates.items()):
            if not isinstance(gate, dict):
                raise ValueError(f"gate {gate_id!r} is not an object in {claim_path}")
            if gate.get("gate_id") != gate_id:
                raise ValueError(f"gate id mismatch for {gate_id!r} in {claim_path}")
            if not isinstance(gate.get("status"), str):
                raise ValueError(f"gate {gate_id!r} has no status in {claim_path}")
            if gate.get("passed") not in (True, False, None):
                raise ValueError(f"gate {gate_id!r} has invalid passed value in {claim_path}")
            normalized_gates[gate_id] = {
                "metric": gate.get("metric"),
                "status": gate["status"],
                "passed": gate.get("passed"),
                "value": gate.get("value"),
            }
        task_gate = normalized_gates["task"]
        model = run.get("model", {})
        rows.append(
            {
                "run_id": run_id,
                "model": str(model["id"]),
                "model_seed": int(run["model_seed"]),
                "learning_rate": float(run["learning_rate"]),
                "task_success": bool(
                    task_gate["status"] == "passed" and task_gate["passed"] is True
                ),
                "claim_gate_sha256": sha256_file(claim_path),
                "gates": normalized_gates,
            }
        )

    gate_ids = sorted(rows[0]["gates"])
    expected_gate_ids = set(gate_ids)
    for row in rows[1:]:
        observed = set(row["gates"])
        if observed != expected_gate_ids:
            raise ValueError(
                f"gate set mismatch for {row['run_id']}: "
                f"missing={sorted(expected_gate_ids - observed)}, "
                f"extra={sorted(observed - expected_gate_ids)}"
            )
    csv_payload = _csv_bytes(rows, gate_ids)
    csv_sha256 = hashlib.sha256(csv_payload).hexdigest()
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    task_successful = [row for row in rows if row["task_success"]]
    summary = {
        "schema_version": AGGREGATION_SCHEMA_VERSION,
        "campaign_id": manifest["campaign_id"],
        "scientific_identity": manifest["scientific_identity"],
        "scope": "nonconfirmatory_phase1_ring_pilot",
        "inference": {
            "confirmatory": False,
            "p_values_computed": False,
            "multiplicity_adjustment": "not_applicable_no_hypothesis_tests",
            "interpretation": "descriptive_pilot_summary_only",
        },
        "denominator_policy": {
            "all_started": "all frozen pilot runs with valid analysis receipts; task failures are retained",
            "task_success_conditional": "status=passed and passed=true on the task gate",
        },
        "pilot_seeds": sorted({int(row["model_seed"]) for row in rows}),
        "started_run_count": len(rows),
        "task_success_count": len(task_successful),
        "task_success_fraction": len(task_successful) / len(rows),
        "gate_ids": gate_ids,
        "run_matrix_csv_sha256": csv_sha256,
        "runs": rows,
        "all_models": _model_aggregate(rows),
        "per_model": {
            model: {
                "pilot_seeds": sorted(int(row["model_seed"]) for row in model_rows),
                **_model_aggregate(model_rows),
            }
            for model, model_rows in sorted(by_model.items())
        },
    }
    return summary, csv_payload


def expected_aggregation_bytes(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[bytes, bytes]:
    summary, matrix = build_pilot_aggregation(root, manifest)
    return canonical_bytes(summary) + b"\n", matrix
