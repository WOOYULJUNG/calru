"""Run the core Ságodi analysis for four baselines with/without noise training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import source_repaired_baselines_v6 as baseline_v6
from . import source_repaired_lru_calru_v6 as lru_v6
from . import state_noise_search_v1 as noise_v1
from .artifacts import atomic_json, canonical_hash, sha256_file, strict_json_load, verify_completion_receipt, write_completion_receipt


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "source_v6_primary_analysis.json"
FREEZE_DOCUMENT = MODULE_DIR / "SOURCE_V6_PRIMARY_ANALYSIS_FREEZE_ko.md"
PROTOCOL = MODULE_DIR / "analysis_protocol.yaml"
CAMPAIGN_ID = "source_v6_primary_analysis"
PROTOCOL_REVISION = "four_baselines_noise_free_vs_positive_noise_sagodi_core_pilot1_v3"
ROOT_MARKER = ".source_v6_primary_analysis_root.json"
CONFIG_CONTRACT_SHA256 = "10f0e721569ac7ea4db4a2dca64f21cb201ecacc8ff17492f910671bf30ccce0"
MODEL_IDS = (*baseline_v6.MODEL_IDS, "lru_n52")
CONDITIONS = ("noise_free", "positive_state_noise_training")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = strict_json_load(Path(path).expanduser().resolve(strict=True))
    if canonical_hash(payload) != CONFIG_CONTRACT_SHA256:
        raise ValueError("source-v6 analysis config differs")
    if payload.get("campaign_id") != CAMPAIGN_ID or payload.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError("source-v6 analysis identity differs")
    if (
        payload["models"] != list(MODEL_IDS)
        or payload["conditions"] != list(CONDITIONS)
        or payload["seeds"] != [0]
        or payload["expected_runs"] != 8
    ):
        raise ValueError("source-v6 analysis denominator differs")
    expected_analysis = {
        "scope": "core_slow_manifold_timescale_separation_projected_drift",
        "trajectory_count": 256,
        "spline_count": 128,
        "task_horizon": 128,
        "blank_horizon": 2048,
        "slow_relative_speed": 0.001,
        "full_local_jacobian_eigenspectrum": True,
        "projected_flow": True,
        "flow_reversal_fixed_point_topology": False,
        "finite_time_and_asymptotic_memory": False,
        "carrier_ambient_normal_recovery": False,
        "analysis_state_noise_disabled": True,
        "eligibility_nmse_db_below": -20.0,
        "eligibility_policy": "label_only_analyze_all_checkpoints",
    }
    if payload["analysis"] != expected_analysis:
        raise ValueError("source-v6 core analysis contract differs")
    return payload


@dataclass(frozen=True)
class AnalysisSpec:
    run_id: str
    model_id: str
    condition: str
    model_seed: int
    checkpoint: str
    checkpoint_sha256: str
    training_receipt_sha256: str
    output_dir: str

    def payload(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), sort_keys=True))


def _git_state(require_clean: bool) -> str:
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
    if require_clean and dirty:
        raise RuntimeError("source-v6 analysis requires a clean committed worktree")
    return commit


def _parse_slots(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values or any(not item.isdigit() for item in values) or len(set(values)) != len(values):
        raise ValueError("--gpus must contain unique comma-separated ids")
    return values


def _plan_rows(path: Path) -> list[dict[str, Any]]:
    payload = strict_json_load(path)
    rows = payload.get("runs")
    if not isinstance(rows, list):
        raise RuntimeError(f"training plan is unreadable: {path}")
    return rows


def _checkpoint_binding(row: Mapping[str, Any]) -> tuple[Path, str, str]:
    output = Path(str(row["output_dir"])).resolve()
    checkpoint = output / "checkpoint_final.pt"
    receipt = output / "completion_receipt.json"
    if not checkpoint.is_file() or not receipt.is_file():
        raise RuntimeError(f"training checkpoint/receipt missing: {output}")
    return checkpoint, sha256_file(checkpoint), sha256_file(receipt)


def build_plan(
    root: Path,
    baseline_root: Path,
    lru_root: Path,
    noise_root: Path,
    seeds: Sequence[int] = (0,),
) -> tuple[AnalysisSpec, ...]:
    baseline_v6.require_verified_main(baseline_root)
    lru_v6._require_stage(lru_root, "lru_main", 3)
    if not noise_v1._stage_valid(noise_root, "main", 12):
        raise RuntimeError("verified positive-noise main is required")
    source_rows = _plan_rows(baseline_root / "main" / "plan.json")
    lru_rows = _plan_rows(lru_root / "lru_main" / "plan.json")
    noise_rows = _plan_rows(noise_root / "main" / "plan.json")
    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in (*source_rows, *lru_rows):
        seed = int(row["model_seed"])
        if seed in seeds:
            indexed[(str(row["model_id"]), "noise_free", seed)] = row
    for row in noise_rows:
        seed = int(row["model_seed"])
        if seed in seeds:
            indexed[(str(row["model_id"]), "positive_state_noise_training", seed)] = row
    expected = {
        (model, condition, seed)
        for model in MODEL_IDS
        for condition in CONDITIONS
        for seed in seeds
    }
    if set(indexed) != expected:
        missing, extra = sorted(expected - set(indexed)), sorted(set(indexed) - expected)
        raise RuntimeError(f"analysis checkpoint matrix differs; missing={missing[:3]}, extra={extra[:3]}")
    specs = []
    for model in MODEL_IDS:
        for condition in CONDITIONS:
            for seed in seeds:
                checkpoint, checkpoint_sha, receipt_sha = _checkpoint_binding(indexed[(model, condition, seed)])
                run_id = f"{model}__{condition}__seed{seed:02d}"
                specs.append(AnalysisSpec(run_id, model, condition, seed, str(checkpoint), checkpoint_sha, receipt_sha, str(root / "runs" / run_id)))
    return tuple(specs)


def _prepare_root(root: Path, baseline_root: Path, lru_root: Path, noise_root: Path, config_path: Path) -> tuple[dict[str, Any], tuple[AnalysisSpec, ...]]:
    root = root.expanduser().resolve()
    config = load_config(config_path)
    specs = build_plan(
        root,
        baseline_root.resolve(),
        lru_root.resolve(),
        noise_root.resolve(),
        tuple(int(seed) for seed in config["seeds"]),
    )
    parent = {
        "baseline_root": str(baseline_root.resolve()), "lru_root": str(lru_root.resolve()), "state_noise_root": str(noise_root.resolve()),
        "baseline_main_receipt": sha256_file(baseline_root / "main" / "completion_receipt.json"),
        "lru_main_receipt": sha256_file(lru_root / "lru_main" / "completion_receipt.json"),
        "state_noise_main_receipt": sha256_file(noise_root / "main" / "completion_receipt.json"),
    }
    identity = {
        "schema_version": 1, "campaign_id": CAMPAIGN_ID, "protocol_revision": PROTOCOL_REVISION,
        "code_commit": _git_state(True), "config_sha256": sha256_file(config_path), "freeze_sha256": sha256_file(FREEZE_DOCUMENT),
        "protocol_sha256": sha256_file(PROTOCOL), "parent_binding": parent, "run_plan": [spec.payload() for spec in specs],
    }
    identity["scientific_identity"] = canonical_hash(identity)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ROOT_MARKER
    if marker.exists():
        if strict_json_load(marker) != identity:
            raise RuntimeError("source-v6 analysis root identity differs")
    elif any(root.iterdir()):
        raise RuntimeError("unmarked source-v6 analysis root must be empty")
    else:
        atomic_json(marker, identity)
        atomic_json(root / "plan.json", {"schema_version": 1, "run_count": len(specs), "runs": [spec.payload() for spec in specs]})
    return config, specs


def _complete(spec: AnalysisSpec) -> bool:
    output = Path(spec.output_dir)
    try:
        summary = strict_json_load(output / "summary.json")
        identity = strict_json_load(output / "analysis_identity.json")["analysis_identity"]
    except (OSError, ValueError, TypeError, KeyError):
        return False
    valid, _ = verify_completion_receipt(output / "completion_receipt.json", expected_job_id=f"sagodi-primary-{spec.model_id}-{identity[:12]}", expected_metadata={"analysis_identity": identity})
    return bool(valid and summary.get("analysis_status") in {"complete_core_structural_analysis", "structural_analysis_not_estimable"} and sha256_file(spec.checkpoint) == spec.checkpoint_sha256)


def _run(
    specs: Sequence[AnalysisSpec],
    slots: Sequence[str],
    config: Mapping[str, Any],
) -> None:
    queue = [spec for spec in specs if not _complete(spec)]
    running: dict[str, tuple[subprocess.Popen[Any], AnalysisSpec, Any]] = {}
    repo = Path(__file__).resolve().parents[2]
    logs = Path(specs[0].output_dir).parents[1] / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        while queue or running:
            for slot in [value for value in slots if value not in running]:
                if not queue:
                    break
                spec = queue.pop(0)
                output = Path(spec.output_dir)
                if output.exists() and not _complete(spec):
                    attempts = output.parents[1] / "attempts"
                    attempts.mkdir(parents=True, exist_ok=True)
                    os.replace(output, attempts / f"{output.name}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}")
                handle = (logs / f"{spec.run_id}.log").open("ab")
                analysis = config["analysis"]
                command = [
                    sys.executable,
                    "-m",
                    "repro.sagodi_protocol.sagodi_primary_runner",
                    "--source-v6",
                    "--core-only",
                    "--trajectory-count",
                    str(analysis["trajectory_count"]),
                    "--spline-count",
                    str(analysis["spline_count"]),
                    "--task-horizon",
                    str(analysis["task_horizon"]),
                    "--blank-horizon",
                    str(analysis["blank_horizon"]),
                    "--checkpoint",
                    spec.checkpoint,
                    "--protocol",
                    str(PROTOCOL),
                    "--output",
                    spec.output_dir,
                    "--device",
                    "cuda:0",
                ]
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = slot
                process = subprocess.Popen(command, cwd=repo, env=environment, stdout=handle, stderr=subprocess.STDOUT)
                running[slot] = (process, spec, handle)
            time.sleep(0.5)
            for slot, (process, spec, handle) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del running[slot]
                if code != 0 or not _complete(spec):
                    raise RuntimeError(f"source-v6 analysis failed: {spec.run_id}; exit={code}")
            status_root = Path(specs[0].output_dir).parents[1]
            atomic_json(status_root / "status.json", {"schema_version": 1, "registered": len(specs), "verified_complete": sum(_complete(spec) for spec in specs), "pending": len(queue), "running": [row[1].run_id for row in running.values()], "updated_at_utc": _utc_now()})
    finally:
        for process, _, handle in running.values():
            if process.poll() is None:
                process.terminate()
            handle.close()


def _aggregate(specs: Sequence[AnalysisSpec]) -> dict[str, Any]:
    rows = []
    for spec in specs:
        summary = strict_json_load(Path(spec.output_dir) / "summary.json")
        row = {"run_id": spec.run_id, "model_id": spec.model_id, "condition": spec.condition, "seed": spec.model_seed, "analysis_status": summary["analysis_status"]}
        if summary["analysis_status"] == "complete_core_structural_analysis":
            row.update({
                "task_performance_eligible": summary["structural_summary_eligibility"]["eligible"],
                "uniform_flow_norm": summary["projected_flow"]["uniform_norm"],
                "largest_real_part_mean": summary["full_local_eigenspectrum"]["largest_real_part"]["mean"],
                "top_two_real_part_gap_mean": summary["full_local_eigenspectrum"]["top_two_real_part_gap"]["mean"],
                "manifold_reconstruction_qa": summary["manifold_reconstruction"]["qa"],
            })
        rows.append(row)
    statuses = (
        "complete_core_structural_analysis",
        "structural_analysis_not_estimable",
    )
    counts = {model: {condition: {status: sum(row["model_id"] == model and row["condition"] == condition and row["analysis_status"] == status for row in rows) for status in statuses} for condition in CONDITIONS} for model in MODEL_IDS}
    return {"schema_version": 1, "campaign_id": CAMPAIGN_ID, "registered_run_count": len(rows), "status_counts": counts, "runs": rows}


def run_campaign(root: Path, baseline_root: Path, lru_root: Path, noise_root: Path, config_path: Path, slots: Sequence[str]) -> Path:
    config, specs = _prepare_root(root, baseline_root, lru_root, noise_root, config_path)
    _run(specs, slots, config)
    if not all(_complete(spec) for spec in specs):
        raise RuntimeError("source-v6 analysis did not complete all registered outcomes")
    summary = root / "summary.json"
    atomic_json(summary, _aggregate(specs))
    run_count = len(specs)
    atomic_json(root / "COMPLETE", {"schema_version": 1, "registered_run_count": run_count, "completed_at_utc": _utc_now()})
    write_completion_receipt(root / "completion_receipt.json", job_id=f"{CAMPAIGN_ID}__complete", artifacts=[root / "plan.json", summary, root / "COMPLETE", *[Path(spec.output_dir) / "completion_receipt.json" for spec in specs]], metadata={"campaign_id": CAMPAIGN_ID, "run_count": run_count})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--lru-root", type=Path, required=True)
    parser.add_argument("--state-noise-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    args = parser.parse_args(argv)
    print(run_campaign(args.artifact_root.resolve(), args.baseline_root.resolve(), args.lru_root.resolve(), args.state_noise_root.resolve(), args.config.resolve(strict=True), _parse_slots(args.gpus)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
