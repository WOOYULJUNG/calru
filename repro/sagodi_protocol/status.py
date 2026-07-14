"""Read-only status and independent receipt validation for a pilot campaign."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .artifacts import (
    canonical_hash,
    sha256_file,
    strict_json_load,
)
from .orchestrate import (
    _process_identity,
    _verify_campaign_inputs,
    compute_campaign_completion,
    verify_campaign_output_receipt,
)


JOB_STATES = (
    "complete",
    "running",
    "pending",
    "invalid",
    "stale",
    "failed",
    "terminated",
    "abandoned",
)


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_identity_matches(pid: Any, expected: Any) -> bool:
    if not _pid_alive(pid) or not isinstance(expected, dict):
        return False
    current = _process_identity(int(pid))
    return current is not None and current == expected


def _validate_manifest(root: Path, manifest: Any) -> tuple[bool, str]:
    if not isinstance(manifest, dict):
        return False, "manifest is not an object"
    if manifest.get("schema_version") != 2:
        return False, "unsupported campaign manifest schema"
    payload = manifest.get("scientific_identity_payload")
    if not isinstance(payload, dict):
        return False, "scientific_identity_payload is missing"
    if canonical_hash(payload) != manifest.get("scientific_identity"):
        return False, "scientific identity hash mismatch"
    expectations = manifest.get("receipt_expectations")
    if not isinstance(expectations, dict) or "phase0" not in expectations:
        return False, "receipt expectations are missing"
    for key, expected in payload.items():
        if manifest.get(key) != expected:
            return False, f"manifest field {key!r} differs from scientific identity payload"
    return True, "ok"


def _validate_complete_marker(
    root: Path,
    manifest: dict[str, Any],
    *,
    computed_complete: bool,
    completion_hashes: dict[str, Any],
) -> tuple[bool, str]:
    path = root / "COMPLETE"
    if not path.is_file():
        return False, "missing COMPLETE marker"
    try:
        marker = strict_json_load(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid COMPLETE marker: {exc}"
    if not computed_complete:
        return False, "COMPLETE exists but receipts/gate are incomplete"
    expected = {
        "schema_version": 3,
        "campaign_id": manifest["campaign_id"],
        "scientific_identity": manifest["scientific_identity"],
        "protocol_fingerprint": manifest["protocol_canonical_fingerprint"],
        "receipt_sha256": completion_hashes["receipts"],
        "pilot_aggregation_sha256": completion_hashes["pilot_aggregation"],
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            return False, f"COMPLETE marker mismatch for {key}"
    reporting = manifest.get("reporting")
    if isinstance(reporting, dict):
        expected_scope = f"phase0_and_{reporting['training_track']}_pilot_only"
        if marker.get("scope") != expected_scope:
            return False, "COMPLETE marker mismatch for scope"
        if marker.get("reporting") != reporting:
            return False, "COMPLETE marker mismatch for reporting"
    elif marker.get("scope") != "phase0_and_nonconfirmatory_phase1_ring_pilot_only":
        return False, "COMPLETE marker mismatch for legacy scope"
    return True, "verified against all current receipts"


def summarize(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"campaign manifest not found: {manifest_path}")
    try:
        manifest = strict_json_load(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"campaign manifest is invalid: {exc}") from exc
    manifest_valid, manifest_reason = _validate_manifest(root, manifest)
    status = {}
    status_path = root / "status.json"
    if status_path.is_file():
        try:
            status = strict_json_load(status_path)
        except (OSError, ValueError, json.JSONDecodeError):
            status = {"status_file_invalid": True}

    input_valid = False
    input_reason = "manifest invalid"
    if manifest_valid:
        try:
            _verify_campaign_inputs(root, manifest, Path(__file__).resolve().parents[2])
        except Exception as exc:
            input_reason = str(exc)
        else:
            input_valid = True
            input_reason = "all protocol, source, and evaluation-bank hashes match"

    expectations = manifest.get("receipt_expectations", {}) if manifest_valid else {}
    rows: list[dict[str, Any]] = []
    for key, expectation in expectations.items():
        if expectation.get("stage") == "phase0":
            continue
        stage = str(expectation["stage"])
        run_id = str(expectation["run_id"])
        output = root / str(expectation["output"])
        valid, reason = verify_campaign_output_receipt(root, manifest, key)
        status_entry = status.get("jobs", {}).get(key, {})
        pid = status_entry.get("pid")
        alive = _pid_alive(pid)
        identity_matches = _pid_identity_matches(
            pid, status_entry.get("process_identity")
        )
        if valid:
            state = "complete"
        elif status_entry.get("state") == "running":
            state = "running" if identity_matches else "stale"
            if not alive:
                reason = f"recorded running PID {pid!r} is not alive; {reason}"
            elif not identity_matches:
                reason = (
                    f"PID {pid!r} was reused or its process-start identity is missing; {reason}"
                )
        elif output.exists():
            state = "invalid"
        elif status_entry.get("state") in {"failed", "terminated", "abandoned"}:
            state = status_entry["state"]
        else:
            state = "pending"
        rows.append(
            {
                "stage": stage,
                "run_id": run_id,
                "state": state,
                "receipt_valid": valid,
                "reason": reason,
                "gpu": status_entry.get("gpu"),
                "pid": pid,
                "pid_alive": alive,
                "pid_identity_matches": identity_matches,
                "attempt_dir": status_entry.get("attempt_dir"),
                "log": status_entry.get("log"),
            }
        )

    phase0_valid = False
    phase0_reason = "manifest invalid"
    gate_passed = False
    if manifest_valid:
        phase0 = expectations["phase0"]
        phase0_valid, phase0_reason = verify_campaign_output_receipt(
            root, manifest, "phase0"
        )
        try:
            gate_passed = (
                strict_json_load(root / phase0["output"] / "phase0_gate.json").get(
                    "passed"
                )
                is True
            )
        except (OSError, ValueError, json.JSONDecodeError):
            gate_passed = False
    phase0_status = status.get("phase0", {})
    phase0_pid = phase0_status.get("pid")
    phase0_alive = _pid_alive(phase0_pid)
    phase0_identity_matches = _pid_identity_matches(
        phase0_pid, phase0_status.get("process_identity")
    )
    if not phase0_valid and phase0_status.get("state") == "running":
        if not phase0_alive:
            phase0_reason = (
                f"recorded running PID {phase0_pid!r} is not alive; {phase0_reason}"
            )
        elif not phase0_identity_matches:
            phase0_reason = (
                f"Phase-0 PID {phase0_pid!r} was reused or start identity is missing; "
                f"{phase0_reason}"
            )

    counts = Counter((row["stage"], row["state"]) for row in rows)
    computed_receipts_complete = False
    completion_reasons: dict[str, str] = {}
    completion_hashes: dict[str, Any] = {
        "receipts": {},
        "pilot_aggregation": {},
    }
    if manifest_valid:
        computed_receipts_complete, completion_reasons, completion_hashes = (
            compute_campaign_completion(root, manifest)
        )
    computed_complete = manifest_valid and input_valid and computed_receipts_complete
    marker_valid, marker_reason = _validate_complete_marker(
        root,
        manifest,
        computed_complete=computed_complete,
        completion_hashes=completion_hashes,
    ) if manifest_valid else (False, "manifest invalid")
    receipts_and_gate_valid = bool(
        phase0_valid
        and gate_passed
        and rows
        and all(row["receipt_valid"] for row in rows)
    )

    return {
        "campaign_id": manifest.get("campaign_id", "unknown"),
        "scientific_identity": manifest.get("scientific_identity"),
        "artifact_root": str(root),
        "stage": status.get("stage", "not_started"),
        "manifest": {"valid": manifest_valid, "reason": manifest_reason},
        "campaign_inputs": {"valid": input_valid, "reason": input_reason},
        "phase0": {
            "receipt_valid": phase0_valid,
            "gate_passed": gate_passed,
            "reason": phase0_reason,
            "pid": phase0_pid,
            "pid_alive": phase0_alive,
            "pid_identity_matches": phase0_identity_matches,
        },
        "training": {
            state: counts[("training", state)] for state in JOB_STATES
        },
        "analysis": {state: counts[("analysis", state)] for state in JOB_STATES},
        "jobs": rows,
        "completion_receipts": {
            "all_valid_and_gate_passed": receipts_and_gate_valid,
            "reasons": {
                key: value
                for key, value in completion_reasons.items()
                if key != "pilot_aggregation"
            },
        },
        "pilot_aggregation": {
            "required": "pilot_aggregation" in manifest,
            "valid": (
                computed_receipts_complete
                and (
                    "pilot_aggregation" not in manifest
                    or bool(completion_hashes["pilot_aggregation"])
                )
            ),
            "reason": completion_reasons.get("pilot_aggregation", "not checked"),
            "sha256": completion_hashes["pilot_aggregation"],
        },
        "computed_complete": computed_complete,
        "complete_marker": {"valid": marker_valid, "reason": marker_reason},
        "campaign_complete": computed_complete and marker_valid,
    }


def format_human(summary: dict[str, Any]) -> str:
    lines = [
        f"Campaign {summary['campaign_id']}",
        f"  stage: {summary['stage']}",
        f"  manifest/inputs: {summary['manifest']['valid']}/{summary['campaign_inputs']['valid']}",
        (
            "  phase0: receipt="
            f"{summary['phase0']['receipt_valid']} gate={summary['phase0']['gate_passed']}"
        ),
    ]
    for stage in ("training", "analysis"):
        counts = summary[stage]
        total = sum(counts.values())
        lines.append(
            f"  {stage}: {counts['complete']}/{total} complete, "
            f"running={counts['running']} pending={counts['pending']} "
            f"invalid={counts['invalid']} stale={counts['stale']} "
            f"failed={counts['failed']} terminated={counts['terminated']} "
            f"abandoned={counts['abandoned']}"
        )
    running = [row for row in summary["jobs"] if row["state"] == "running"]
    for row in running:
        lines.append(
            f"    {row['stage']} {row['run_id']} gpu={row['gpu']} "
            f"pid={row['pid']} alive={row['pid_alive']}"
            f" identity={row['pid_identity_matches']}"
        )
    lines.append(
        f"  computed_complete/COMPLETE_valid: "
        f"{summary['computed_complete']}/{summary['complete_marker']['valid']}"
    )
    lines.append(
        "  pilot aggregation: "
        f"valid={summary['pilot_aggregation']['valid']} "
        f"({summary['pilot_aggregation']['reason']})"
    )
    lines.append(f"  campaign_complete: {summary['campaign_complete']}")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = summarize(args.artifact_root)
    except Exception as exc:
        print(f"invalid campaign: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else format_human(report))
    return 0 if report["campaign_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
