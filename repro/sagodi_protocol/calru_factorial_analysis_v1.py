"""Analyze the completed CA-LRU 2x2 factorial with extended dynamics diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import calru_factorial_v1 as factorial
from .artifacts import (
    atomic_json,
    canonical_hash,
    sha256_file,
    strict_json_load,
    write_completion_receipt,
)
from .source_v6_primary_analysis_campaign import (
    AnalysisSpec,
    _complete,
    _git_state,
    _parse_slots,
    _run,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "calru_factorial_analysis_v1.json"
FREEZE_DOCUMENT = MODULE_DIR / "CALRU_FACTORIAL_ANALYSIS_V1_FREEZE_ko.md"
PROTOCOL = MODULE_DIR / "analysis_protocol.yaml"
CAMPAIGN_ID = "calru_factorial_extended_analysis_v1"
PROTOCOL_REVISION = "calru_four_cell_three_seed_extended_dynamics_pilot_v1"
CONFIG_CONTRACT_SHA256 = "c13c34fa9af1532d6fd6205e3e8d12a498f51be9e9281724af3ee4b0e1393ed1"
ROOT_MARKER = ".calru_factorial_extended_analysis_v1_root.json"
CONDITIONS = factorial.CONDITIONS
SEEDS = (0, 1, 2)
EXPECTED_RUNS = len(CONDITIONS) * len(SEEDS)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = strict_json_load(Path(path).expanduser().resolve(strict=True))
    if canonical_hash(payload) != CONFIG_CONTRACT_SHA256:
        raise ValueError("CA-LRU factorial analysis config differs")
    if payload.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("CA-LRU factorial analysis campaign id differs")
    if payload.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError("CA-LRU factorial analysis protocol revision differs")
    if payload.get("conditions") != list(CONDITIONS):
        raise ValueError("CA-LRU factorial analysis conditions differ")
    if payload.get("seeds") != list(SEEDS) or payload.get("expected_runs") != EXPECTED_RUNS:
        raise ValueError("CA-LRU factorial analysis denominator differs")
    analysis = payload.get("analysis")
    reference = strict_json_load(MODULE_DIR / "source_v6_primary_analysis.json")["analysis"]
    if analysis != reference:
        raise ValueError("CA-LRU and baseline extended analysis settings differ")
    return payload


def _factorial_specs(factorial_root: Path) -> tuple[Any, ...]:
    if not factorial._stage_valid(factorial_root, "factorial_main", EXPECTED_RUNS):
        raise RuntimeError("verified CA-LRU factorial_main is required")
    specs = factorial._read_specs(factorial_root / "factorial_main")
    observed = {(str(spec.condition_id), int(spec.model_seed)) for spec in specs}
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if observed != expected:
        raise RuntimeError("CA-LRU factorial checkpoint matrix differs")
    return tuple(sorted(specs, key=lambda item: (CONDITIONS.index(item.condition_id), item.model_seed)))


def build_plan(root: Path, factorial_root: Path) -> tuple[AnalysisSpec, ...]:
    planned: list[AnalysisSpec] = []
    for source in _factorial_specs(factorial_root):
        output = Path(source.output_dir).resolve()
        checkpoint = output / "checkpoint_final.pt"
        receipt = output / "completion_receipt.json"
        if not checkpoint.is_file() or not receipt.is_file():
            raise RuntimeError(f"factorial child artifacts are missing: {source.run_id}")
        run_id = f"{source.condition_id}__seed{int(source.model_seed):02d}"
        planned.append(
            AnalysisSpec(
                run_id=run_id,
                model_id=str(source.model_id),
                condition=str(source.condition_id),
                model_seed=int(source.model_seed),
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha256_file(checkpoint),
                training_receipt_sha256=sha256_file(receipt),
                output_dir=str(root / "runs" / run_id),
            )
        )
    return tuple(planned)


def _prepare_root(
    root: Path,
    factorial_root: Path,
    config_path: Path,
) -> tuple[dict[str, Any], tuple[AnalysisSpec, ...]]:
    root = root.expanduser().resolve()
    factorial_root = factorial_root.expanduser().resolve()
    config = load_config(config_path)
    specs = build_plan(root, factorial_root)
    parent = factorial_root / "factorial_main"
    identity: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "code_commit": _git_state(True),
        "config_sha256": sha256_file(config_path),
        "freeze_sha256": sha256_file(FREEZE_DOCUMENT),
        "protocol_sha256": sha256_file(PROTOCOL),
        "parent_binding": {
            "factorial_root": str(factorial_root),
            "factorial_main_receipt_sha256": sha256_file(parent / "completion_receipt.json"),
            "factorial_main_summary_sha256": sha256_file(parent / "summary.json"),
            "factorial_main_plan_sha256": sha256_file(parent / "plan.json"),
        },
        "run_plan": [json.loads(json.dumps(asdict(spec), sort_keys=True)) for spec in specs],
    }
    identity["scientific_identity"] = canonical_hash(identity)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ROOT_MARKER
    if marker.exists():
        if strict_json_load(marker) != identity:
            raise RuntimeError("CA-LRU factorial analysis root identity differs")
    elif any(root.iterdir()):
        raise RuntimeError("unmarked CA-LRU factorial analysis root must be empty")
    else:
        atomic_json(marker, identity)
        atomic_json(
            root / "plan.json",
            {"schema_version": 1, "run_count": len(specs), "runs": [asdict(spec) for spec in specs]},
        )
    return config, specs


def _extended_fields(summary: Mapping[str, Any]) -> dict[str, Any]:
    topology = summary["fixed_point_topology"]
    recovery = summary["carrier_ambient_normal_recovery"]
    final_recovery: dict[str, Any] = {}
    for family in ("ambient_normal", "in_plane_radial"):
        by_radius = recovery["metrics_by_family"][family]["by_radius"]
        final_recovery[family] = {
            radius: {
                metric: values["by_horizon"]["4096"][metric]
                for metric in (
                    "manifold_distance_ratio",
                    "same_memory_error_radians",
                    "excess_same_memory_error_radians",
                )
            }
            for radius, values in by_radius.items()
        }
    return {
        "task_performance_eligible": summary["structural_summary_eligibility"]["eligible"],
        "uniform_flow_norm": summary["projected_flow"]["uniform_norm"],
        "largest_real_part_mean": summary["full_local_eigenspectrum"]["largest_real_part"]["mean"],
        "top_two_real_part_gap_mean": summary["full_local_eigenspectrum"]["top_two_real_part_gap"]["mean"],
        "fixed_point_topology": {
            "kind": topology["kind"],
            "stable_count": topology["stable_count"],
            "saddle_count": topology["saddle_count"],
            "stable_angles": topology["stable_angles"],
            "saddle_angles": topology["saddle_angles"],
        },
        "finite_time_terminal_mean_error_radians": summary["finite_time_angular_memory"]["terminal_mean_error_radians"],
        "finite_time_terminal_maximum_error_radians": summary["finite_time_angular_memory"]["terminal_maximum_error_radians"],
        "asymptotic_structure": summary["asymptotic_structure"],
        "finite_normal_recovery_at_4096": final_recovery,
        "manifold_reconstruction_qa": summary["manifold_reconstruction"]["qa"],
    }


def _aggregate(specs: Sequence[AnalysisSpec]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        summary = strict_json_load(Path(spec.output_dir) / "summary.json")
        row: dict[str, Any] = {
            "run_id": spec.run_id,
            "model_id": spec.model_id,
            "condition": spec.condition,
            "seed": spec.model_seed,
            "analysis_status": summary["analysis_status"],
        }
        if summary["analysis_status"] == "complete_extended_structural_analysis":
            row.update(_extended_fields(summary))
        rows.append(row)
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "registered_run_count": len(rows),
        "complete_extended_count": sum(
            row["analysis_status"] == "complete_extended_structural_analysis" for row in rows
        ),
        "structural_not_estimable_count": sum(
            row["analysis_status"] == "structural_analysis_not_estimable" for row in rows
        ),
        "runs": rows,
    }


def run_campaign(
    root: Path,
    factorial_root: Path,
    config_path: Path,
    slots: Sequence[str],
) -> Path:
    config, specs = _prepare_root(root, factorial_root, config_path)
    _run(specs, slots, config)
    if not all(_complete(spec) for spec in specs):
        raise RuntimeError("CA-LRU factorial extended analysis did not complete")
    summary = root / "summary.json"
    atomic_json(summary, _aggregate(specs))
    atomic_json(
        root / "COMPLETE",
        {"schema_version": 1, "registered_run_count": len(specs), "completed_at_utc": _utc_now()},
    )
    write_completion_receipt(
        root / "completion_receipt.json",
        job_id=f"{CAMPAIGN_ID}__complete",
        artifacts=[
            root / "plan.json",
            summary,
            root / "COMPLETE",
            *[Path(spec.output_dir) / "completion_receipt.json" for spec in specs],
        ],
        metadata={"campaign_id": CAMPAIGN_ID, "run_count": len(specs)},
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--factorial-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    args = parser.parse_args(argv)
    print(
        run_campaign(
            args.artifact_root.resolve(),
            args.factorial_root.resolve(),
            args.config.resolve(strict=True),
            _parse_slots(args.gpus),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
