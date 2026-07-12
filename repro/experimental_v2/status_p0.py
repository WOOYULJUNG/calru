#!/usr/bin/env python3
"""Read-only validation and progress summary for P0 campaign artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# Importing the launcher gives this monitor the exact completion-receipt
# validator without creating Python bytecode beside a live campaign.
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import launch_p0 as launch  # noqa: E402


STATES = ("complete", "running", "failed", "partial", "pending")
FAILED_STATUS_STATES = {"failed", "terminated"}
COMPLETE_STATUS_STATES = {"complete", "skipped_complete"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_campaigns(path: Path) -> list[Path]:
    candidate = path.expanduser().resolve()
    if (candidate / "manifest.json").is_file():
        return [candidate]
    marker = candidate / launch.ROOT_MARKER
    if not marker.is_file() or marker.read_text(encoding="utf-8") != launch.ROOT_MARKER_CONTENT:
        raise RuntimeError(
            f"{candidate} is neither a campaign directory nor a marked experimental_v2 artifact root"
        )
    campaigns = sorted(
        child.resolve()
        for child in candidate.iterdir()
        if child.is_dir() and (child / "manifest.json").is_file()
    )
    if not campaigns:
        raise RuntimeError(f"marked artifact root contains no campaign manifests: {candidate}")
    return campaigns


def _job_from_manifest(payload: Any) -> launch.Job:
    if not isinstance(payload, dict):
        raise ValueError("job entry is not an object")
    required = {
        "job_id",
        "family",
        "condition",
        "task",
        "model",
        "worker_model",
        "seed",
        "runner",
        "command",
        "implemented",
        "implementation_note",
        "expected",
        "log_dir",
        "metadata",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"job entry is missing fields: {missing}")
    if not isinstance(payload["command"], list) or not isinstance(payload["expected"], list):
        raise ValueError("job command and expected fields must be arrays")
    return launch.Job(
        job_id=str(payload["job_id"]),
        family=str(payload["family"]),
        condition=str(payload["condition"]),
        task=str(payload["task"]),
        model=str(payload["model"]),
        worker_model=str(payload["worker_model"]),
        seed=int(payload["seed"]),
        runner=str(payload["runner"]),
        command=tuple(str(item) for item in payload["command"]),
        implemented=bool(payload["implemented"]),
        implementation_note=str(payload["implementation_note"]),
        expected=tuple(str(item) for item in payload["expected"]),
        log_dir=str(payload["log_dir"]),
        metadata=dict(payload["metadata"]),
    )


def _load_status(campaign_dir: Path) -> tuple[dict[str, dict[str, Any]], bool, str | None]:
    path = campaign_dir / "status.json"
    if not path.exists():
        return {}, False, None
    try:
        payload = _read_json(path)
        if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), dict):
            raise ValueError("status must be an object containing a jobs object")
        jobs: dict[str, dict[str, Any]] = {}
        for job_id, entry in payload["jobs"].items():
            if not isinstance(entry, dict):
                raise ValueError(f"status entry {job_id!r} is not an object")
            jobs[str(job_id)] = entry
        return jobs, True, None
    except Exception as exc:
        return {}, True, f"cannot parse status.json: {type(exc).__name__}: {exc}"


def _latest_log(job: launch.Job, campaign_dir: Path, status_entry: dict[str, Any]) -> str | None:
    status_log = status_entry.get("log")
    if isinstance(status_log, str) and status_log:
        return status_log
    directory = campaign_dir / job.log_dir
    matches = sorted(directory.glob(f"{job.job_id}.attempt*.log")) if directory.is_dir() else []
    if not matches:
        return None
    return str(matches[-1].relative_to(campaign_dir))


def _artifact_presence(job: launch.Job, campaign_dir: Path) -> tuple[str, list[str]]:
    try:
        paths = launch._expected_paths(job, campaign_dir)
    except Exception as exc:
        return "invalid", [f"{type(exc).__name__}: {exc}"]
    present = [str(path.relative_to(campaign_dir.resolve())) for path in paths if path.exists()]
    if not present:
        return "missing", []
    if len(present) == len(paths):
        return "unreceipted_full_set", present
    return "partial", present


def _effective_state(
    *,
    receipt_state: str,
    artifact_state: str,
    status_state: str,
) -> str:
    if receipt_state == "valid":
        return "complete"
    if receipt_state == "invalid":
        return "partial"
    if status_state == "running":
        return "running"
    if status_state in FAILED_STATUS_STATES:
        return "failed"
    if status_state == "blocked_partial" or artifact_state != "missing":
        return "partial"
    return "pending"


def summarize_campaign(campaign_dir: Path) -> dict[str, Any]:
    campaign_dir = campaign_dir.expanduser().resolve()
    errors: list[str] = []
    manifest_path = campaign_dir / "manifest.json"
    try:
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError("manifest is not a JSON object")
    except Exception as exc:
        return {
            "campaign_dir": str(campaign_dir),
            "campaign_id": campaign_dir.name,
            "manifest_valid": False,
            "status_present": (campaign_dir / "status.json").exists(),
            "total_jobs": 0,
            "counts": {state: 0 for state in STATES},
            "progress": {"complete": 0, "total": 0, "fraction": 0.0, "percent": 0.0},
            "receipt_counts": {"valid": 0, "invalid": 0, "missing": 0, "orphan": 0},
            "running": [],
            "attention": [],
            "jobs": [],
            "validation_errors": [f"cannot parse manifest.json: {type(exc).__name__}: {exc}"],
            "validation_ok": False,
        }

    campaign_id = str(manifest.get("campaign_id", campaign_dir.name))
    try:
        launch._verified_manifest_identity(campaign_dir)
        manifest_valid = True
    except Exception as exc:
        manifest_valid = False
        errors.append(f"manifest validation failed: {type(exc).__name__}: {exc}")

    raw_jobs = manifest.get("jobs")
    jobs: list[launch.Job] = []
    if not isinstance(raw_jobs, list):
        errors.append("manifest jobs field is not an array")
    else:
        for index, payload in enumerate(raw_jobs):
            try:
                jobs.append(_job_from_manifest(payload))
            except Exception as exc:
                errors.append(f"manifest job[{index}] is invalid: {type(exc).__name__}: {exc}")
    job_ids = [job.job_id for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        errors.append("manifest contains duplicate job IDs")

    status_jobs, status_present, status_error = _load_status(campaign_dir)
    if status_error:
        errors.append(status_error)
    unknown_status_ids = sorted(set(status_jobs) - set(job_ids))
    if unknown_status_ids:
        errors.append(f"status contains unknown job IDs: {unknown_status_ids}")

    expected_receipts: set[Path] = set()
    records: list[dict[str, Any]] = []
    receipt_counts: Counter[str] = Counter()
    for job in jobs:
        status_entry = status_jobs.get(job.job_id, {})
        status_state = str(status_entry.get("state", "pending"))
        receipt_path = launch._completion_receipt_path(job, campaign_dir)
        expected_receipts.add(receipt_path)
        receipt_error: str | None = None
        if receipt_path.exists():
            try:
                launch._verify_completion_receipt(job, campaign_dir)
                receipt_state = "valid"
            except Exception as exc:
                receipt_state = "invalid"
                receipt_error = f"{type(exc).__name__}: {exc}"
                errors.append(f"{job.job_id}: invalid completion receipt: {receipt_error}")
        else:
            receipt_state = "missing"
        receipt_counts[receipt_state] += 1

        artifact_state, present_artifacts = _artifact_presence(job, campaign_dir)
        if receipt_state == "valid":
            artifact_state = "validated"
        state = _effective_state(
            receipt_state=receipt_state,
            artifact_state=artifact_state,
            status_state=status_state,
        )
        if status_state in COMPLETE_STATUS_STATES and receipt_state != "valid":
            errors.append(
                f"{job.job_id}: status claims {status_state!r} without a valid completion receipt"
            )
        known_status_states = {
            "pending",
            "running",
            "failed",
            "terminated",
            "blocked_partial",
            "complete",
            "skipped_complete",
        }
        if status_state not in known_status_states:
            errors.append(f"{job.job_id}: unknown status state {status_state!r}")

        log = _latest_log(job, campaign_dir, status_entry)
        records.append(
            {
                "job_id": job.job_id,
                "family": job.family,
                "condition": job.condition,
                "task": job.task,
                "seed": job.seed,
                "state": state,
                "status_state": status_state,
                "artifact_state": artifact_state,
                "present_artifacts": present_artifacts,
                "receipt_state": receipt_state,
                "receipt": (
                    str(receipt_path.relative_to(campaign_dir)) if receipt_path.exists() else None
                ),
                "receipt_error": receipt_error,
                "gpu": status_entry.get("gpu"),
                "pid": status_entry.get("pid"),
                "log": log,
                "exit_code": status_entry.get("exit_code"),
            }
        )

    receipt_root = campaign_dir / "completion_receipts"
    actual_receipts = {
        path.resolve() for path in receipt_root.rglob("*.json")
    } if receipt_root.is_dir() else set()
    orphan_receipts = sorted(actual_receipts - expected_receipts)
    receipt_counts["orphan"] = len(orphan_receipts)
    if orphan_receipts:
        relative = [str(path.relative_to(campaign_dir)) for path in orphan_receipts]
        errors.append(f"orphan completion receipts: {relative}")

    counts_counter = Counter(record["state"] for record in records)
    counts = {state: int(counts_counter.get(state, 0)) for state in STATES}
    total = len(records)
    complete = counts["complete"]
    fraction = float(complete / total) if total else 0.0
    running = [
        {
            "job_id": record["job_id"],
            "condition": record["condition"],
            "task": record["task"],
            "seed": record["seed"],
            "gpu": record["gpu"],
            "pid": record["pid"],
            "log": record["log"],
        }
        for record in records
        if record["state"] == "running"
    ]
    attention = [
        {
            "state": record["state"],
            "job_id": record["job_id"],
            "condition": record["condition"],
            "task": record["task"],
            "seed": record["seed"],
            "log": record["log"],
            "exit_code": record["exit_code"],
            "artifact_state": record["artifact_state"],
            "receipt_error": record["receipt_error"],
        }
        for record in records
        if record["state"] in {"failed", "partial"}
    ]
    return {
        "campaign_dir": str(campaign_dir),
        "campaign_id": campaign_id,
        "manifest_valid": manifest_valid,
        "status_present": status_present,
        "total_jobs": total,
        "counts": counts,
        "progress": {
            "complete": complete,
            "total": total,
            "fraction": fraction,
            "percent": 100.0 * fraction,
        },
        "receipt_counts": {
            "valid": int(receipt_counts.get("valid", 0)),
            "invalid": int(receipt_counts.get("invalid", 0)),
            "missing": int(receipt_counts.get("missing", 0)),
            "orphan": int(receipt_counts.get("orphan", 0)),
        },
        "running": running,
        "attention": attention,
        "jobs": records,
        "validation_errors": errors,
        "validation_ok": not errors,
    }


def summarize_path(path: Path) -> dict[str, Any]:
    campaigns = [summarize_campaign(campaign) for campaign in _discover_campaigns(path)]
    counts = {state: sum(item["counts"][state] for item in campaigns) for state in STATES}
    total = sum(item["total_jobs"] for item in campaigns)
    complete = counts["complete"]
    fraction = float(complete / total) if total else 0.0
    receipt_counts = {
        state: sum(item["receipt_counts"][state] for item in campaigns)
        for state in ("valid", "invalid", "missing", "orphan")
    }
    error_count = sum(len(item["validation_errors"]) for item in campaigns)
    return {
        "input": str(path.expanduser().resolve()),
        "campaign_count": len(campaigns),
        "campaigns": campaigns,
        "aggregate": {
            "total_jobs": total,
            "counts": counts,
            "progress": {
                "complete": complete,
                "total": total,
                "fraction": fraction,
                "percent": 100.0 * fraction,
            },
            "receipt_counts": receipt_counts,
            "validation_error_count": error_count,
            "validation_ok": error_count == 0,
        },
    }


def _format_human(summary: dict[str, Any]) -> str:
    lines = [f"P0 status: {summary['input']}"]
    for campaign in summary["campaigns"]:
        progress = campaign["progress"]
        counts = campaign["counts"]
        receipts = campaign["receipt_counts"]
        lines.extend(
            [
                "",
                f"Campaign {campaign['campaign_id']}",
                f"  path: {campaign['campaign_dir']}",
                (
                    f"  progress: {progress['complete']}/{progress['total']} "
                    f"({progress['percent']:.1f}%)"
                ),
                "  states: " + " ".join(f"{state}={counts[state]}" for state in STATES),
                (
                    "  receipts: "
                    + " ".join(
                        f"{state}={receipts[state]}"
                        for state in ("valid", "invalid", "missing", "orphan")
                    )
                ),
                (
                    "  validation: OK"
                    if campaign["validation_ok"]
                    else f"  validation: ERROR ({len(campaign['validation_errors'])})"
                ),
            ]
        )
        if campaign["running"]:
            lines.append("  running:")
            for item in campaign["running"]:
                lines.append(
                    f"    {item['job_id']} gpu={item['gpu']} pid={item['pid']} "
                    f"log={item['log'] or '(none)'}"
                )
        if campaign["attention"]:
            lines.append("  failed/partial:")
            for item in campaign["attention"]:
                lines.append(
                    f"    [{item['state']}] {item['job_id']} "
                    f"log={item['log'] or '(none)'}"
                )
        if campaign["validation_errors"]:
            lines.append("  validation errors:")
            lines.extend(f"    - {error}" for error in campaign["validation_errors"])

    if summary["campaign_count"] > 1:
        aggregate = summary["aggregate"]
        progress = aggregate["progress"]
        lines.extend(
            [
                "",
                f"Aggregate: {progress['complete']}/{progress['total']} ({progress['percent']:.1f}%)",
                "  states: "
                + " ".join(f"{state}={aggregate['counts'][state]}" for state in STATES),
            ]
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="marked artifact root or one campaign directory")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        summary = summarize_path(args.path)
    except Exception as exc:
        error = {
            "input": str(args.path.expanduser().resolve()),
            "error": f"{type(exc).__name__}: {exc}",
        }
        if args.json:
            print(json.dumps(error, indent=2, sort_keys=True))
        else:
            print(f"P0 status error: {error['error']}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(_format_human(summary), end="")
    return 0 if summary["aggregate"]["validation_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
