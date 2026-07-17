"""Stage 1: verify the complete 36-run topology pilot before test access."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from repro.sagodi_protocol.artifacts import (
    atomic_json,
    strict_json_load,
    verify_completion_receipt,
)

from .topology_analysis_common import (
    discover_completed_runs,
    expected_jobs,
    file_sha256,
    load_analysis_config,
)


def analyze(run_root: Path, output: Path, config_path: Path) -> dict[str, Any]:
    config = load_analysis_config(config_path)
    expected = config["expected_training"]
    search_analysis = config["analysis_id"] == "manifold_topology_hparam_analysis_v1"
    banks = config["frozen_banks"]
    records, missing = discover_completed_runs(run_root, config)
    launcher_path = run_root / "launcher_manifest.json"
    launcher = strict_json_load(launcher_path) if launcher_path.is_file() else None
    bank_checks: dict[str, Any] = {}
    for split in ("validation", "test"):
        root = Path(banks[f"{split}_root"])
        for topology, digest in banks[f"{split}_sha256"].items():
            path = root / f"id_{topology}.npz"
            actual = file_sha256(path)
            bank_checks[f"{split}_{topology}"] = {
                "path": str(path),
                "expected_sha256": digest,
                "actual_sha256": actual,
                "matches": actual == digest,
            }

    run_checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for record in records:
        receipt_ok, receipt_reason = verify_completion_receipt(
            record.run_dir / "COMPLETED.json", expected_job_id=record.job_id
        )
        checkpoint = torch.load(
            record.checkpoint_path, map_location="cpu", weights_only=False
        )
        tensors_finite = all(
            torch.isfinite(value).all().item()
            for value in checkpoint["model_state_dict"].values()
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
        )
        manifest = record.manifest
        phase_ok = (
            manifest.get("phase") in {"broad", "refine", "robust"}
            if search_analysis
            else manifest.get("stage") == "pilot"
        )
        expected_config_sha256 = expected.get(
            "training_config_sha256",
            expected.get("topology_transfer_config_sha256"),
        )
        per_run_test_policy_ok = (
            not bool(record.result.get("test_bank_accessed", False))
            if search_analysis
            else bool(record.result.get("test_bank_accessed", False))
        )
        checks = {
            "job_id": record.job_id,
            "receipt_ok": bool(receipt_ok),
            "receipt_reason": receipt_reason,
            "campaign_matches": manifest.get("campaign_id") == expected["campaign_id"],
            "training_phase_matches": phase_ok,
            "final_update_is_5000": int(record.result.get("updates", -1)) == 5000,
            "config_sha256_matches": (
                manifest.get("config_sha256")
                == expected_config_sha256
            ),
            "checkpoint_job_id_matches": checkpoint.get("job_id") == record.job_id,
            "checkpoint_tensors_finite": bool(tensors_finite),
            "result_finite": bool(record.result.get("finite", False)),
            "per_run_test_access_policy_ok": per_run_test_policy_ok,
            "parameter_count": int(manifest["model"]["parameters_total"]),
            "checkpoint_sha256": file_sha256(record.checkpoint_path),
        }
        checks["all_run_checks_pass"] = all(
            value
            for key, value in checks.items()
            if key
            not in {
                "job_id",
                "receipt_reason",
                "parameter_count",
                "checkpoint_sha256",
            }
        )
        if not checks["all_run_checks_pass"]:
            failures.append(record.job_id)
        run_checks.append(checks)

    root_suffix_matches = run_root.name.endswith(str(expected["run_root_suffix"]))
    launcher_jobs = set()
    if launcher is not None:
        launcher_jobs = {
            f"{item['stage']}__{item['model']}__{item['topology']}__seed{item['seed']}"
            for item in launcher.get("jobs", [])
        }
    all_expected_jobs_registered = (
        set(record.job_id for record in records) == set(expected_jobs(config))
        if "job_ids" in expected
        else launcher_jobs == set(expected_jobs(config))
    )
    payload = {
        "schema_version": 1,
        "analysis_id": config["analysis_id"],
        "run_root": str(run_root),
        "expected_run_count": int(expected["expected_runs"]),
        "completed_run_count": len(records),
        "missing_run_count": len(missing),
        "missing_job_ids": missing,
        "failed_integrity_job_ids": failures,
        "all_expected_jobs_in_launcher": all_expected_jobs_registered,
        "run_root_commit_suffix_matches": root_suffix_matches,
        "expected_training_git_commit": expected["git_commit"],
        "training_commit_directly_recorded_in_run_manifest": False,
        "training_commit_verification_scope": (
            "expected commit and run-root suffix plus frozen config hash; the v1 "
            "training manifest did not record git HEAD directly"
        ),
        "campaign_global_test_sequestering_satisfied": search_analysis,
        "campaign_global_test_sequestering_limitation": (
            None
            if search_analysis
            else "the frozen v1 runner evaluates test after each individual final "
            "checkpoint rather than behind a 36-run campaign barrier; those values "
            "did not select checkpoints, learning rates, eligibility, or stopping"
        ),
        "analysis_test_access_policy": (
            f"fresh task analysis remains blocked until all {int(expected['expected_runs'])} "
            "selected final checkpoints pass integrity"
        ),
        "all_bank_hashes_match": all(item["matches"] for item in bank_checks.values()),
        "bank_checks": bank_checks,
        "run_checks": run_checks,
    }
    payload["ready_for_test_analysis"] = bool(
        len(records) == int(expected["expected_runs"])
        and not missing
        and not failures
        and payload["all_expected_jobs_in_launcher"]
        and root_suffix_matches
        and payload["all_bank_hashes_match"]
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "run_integrity.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("topology_analysis_v1.json")
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    payload = analyze(
        args.run_root.expanduser().resolve(strict=True),
        args.output.expanduser().resolve(),
        args.config.expanduser().resolve(strict=True),
    )
    if args.require_complete and not payload["ready_for_test_analysis"]:
        raise SystemExit("pilot is not ready for test analysis")
    print(
        f"completed={payload['completed_run_count']}/{payload['expected_run_count']} "
        f"ready={payload['ready_for_test_analysis']}"
    )


if __name__ == "__main__":
    main()
