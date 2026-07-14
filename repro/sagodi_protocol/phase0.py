"""Materialize the mandatory Phase-0 state/blank-map gate."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import atomic_json, sha256_file, write_completion_receipt
from .audit import AuditConfig, run_phase0_audit
from .config import DEFAULT_PROTOCOL_PATH, load_protocol, protocol_fingerprint
from .models import (
    PROJECT_PARAM_BASELINE_NAMES,
    SAGODI_GRU_NAMES,
    build_protocol_model,
    model_config_from_protocol,
)
from .state import StateAdapter


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _phase0_width_override(model_name: str, *, smoke: bool) -> int | None:
    """Keep architecture-locked Ságodi GRUs at their frozen dimensions."""

    if not smoke or model_name in (*SAGODI_GRU_NAMES, *PROJECT_PARAM_BASELINE_NAMES):
        return None
    return 8


@torch.no_grad()
def _blank_trace(
    adapter: StateAdapter, *, seed: int, steps: int = 20
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Trace the literal blank map without pretending a training atlas exists.

    Phase 0 precedes training, so no learned manifold is available for a
    nearest-manifold distance or a geometrically defined radial kick.  The
    checkpoint-specific preflight in Phase 1 supplies those two fields.  This
    architecture trace nevertheless materializes every actual state/input and
    decoder tensor needed to audit overwrite and reset semantics.
    """

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    primary = torch.randn(2, adapter.primary_dim, generator=generator, dtype=torch.float32)
    primary = primary.to(device=adapter.device, dtype=adapter.dtype) * 0.25
    reported = adapter.reported_from_primary(primary)
    rows: dict[str, list[np.ndarray]] = {
        "actual_input": [],
        "pre_primary": [],
        "post_primary": [],
        "pre_carrier": [],
        "post_carrier": [],
        "pre_stream": [],
        "post_stream": [],
        "reset_mask": [],
        "overwrite_mask": [],
        "decoder_input": [],
        "decoder_output": [],
        "f0_residual": [],
    }

    def cpu(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()

    for _ in range(int(steps)):
        blank = adapter.zero_input(reported)
        pre_parts = adapter.unpack(reported)
        next_reported = adapter.reported_step(reported, blank)
        next_primary = adapter.primary_from_reported(next_reported)
        post_parts = adapter.unpack(next_reported)
        pre_stream = (
            pre_parts.stream
            if pre_parts.stream is not None
            else torch.empty(reported.shape[0], 0, device=reported.device, dtype=reported.dtype)
        )
        post_stream = (
            post_parts.stream
            if post_parts.stream is not None
            else torch.empty(reported.shape[0], 0, device=reported.device, dtype=reported.dtype)
        )
        decode_mode = str(getattr(adapter.model, "decode_mode", "model_defined"))
        decoder_input = post_stream if adapter.is_full_block and decode_mode == "stream" else next_reported
        rows["actual_input"].append(cpu(blank))
        rows["pre_primary"].append(cpu(primary))
        rows["post_primary"].append(cpu(next_primary))
        rows["pre_carrier"].append(cpu(pre_parts.carrier))
        rows["post_carrier"].append(cpu(post_parts.carrier))
        rows["pre_stream"].append(cpu(pre_stream))
        rows["post_stream"].append(cpu(post_stream))
        rows["reset_mask"].append(np.zeros((reported.shape[0], 1), dtype=np.bool_))
        rows["overwrite_mask"].append(
            np.full((reported.shape[0], max(1, adapter.stream_dim)), bool(adapter.is_full_block and not adapter.carry_stream), dtype=np.bool_)
        )
        rows["decoder_input"].append(cpu(decoder_input))
        rows["decoder_output"].append(cpu(adapter.decode(next_reported)))
        rows["f0_residual"].append(cpu(next_primary - primary))
        primary = next_primary
        reported = next_reported
    arrays = {
        "step": np.arange(1, int(steps) + 1, dtype=np.int64),
        **{key: np.stack(value, axis=0) for key, value in rows.items()},
        "nearest_manifold_distance": np.full((int(steps), reported.shape[0]), np.nan, dtype=np.float64),
    }
    arrays["actual_input_norm"] = np.linalg.norm(arrays["actual_input"], axis=-1)
    arrays["f0_residual_norm"] = np.linalg.norm(arrays["f0_residual"], axis=-1)
    metadata = {
        "schema_version": 1,
        "steps": int(steps),
        "external_input_exactly_zero": bool(np.count_nonzero(arrays["actual_input"]) == 0),
        "external_reset_present": False,
        "nearest_manifold_distance_status": "not_available_before_training",
        "radial_perturbation_status": "not_defined_before_task_conditioned_atlas",
        "required_checkpoint_followup": "phase1_checkpoint_autonomous_trace",
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
    }
    return arrays, metadata


def run_phase0(
    protocol_path: Path,
    output_root: Path,
    *,
    device: str = "cpu",
    smoke: bool = False,
) -> Path:
    protocol = load_protocol(protocol_path)
    primary_main = protocol.get("campaign_mode") == "sagodi_primary_main_v3"
    pilot_only = not primary_main
    root = Path(output_root).resolve()
    completion = root / "completion_receipt.json"
    if completion.exists() or (root.exists() and any(root.iterdir())):
        raise FileExistsError(f"refusing to overwrite Phase-0 output: {root}")
    root.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha256_file(protocol_path)
    reports: dict[str, Any] = {}
    artifacts: list[Path] = []
    for index, model_name in enumerate(protocol["phase0_state_audit"]["models"]):
        torch.manual_seed(100)
        random.seed(100)
        np.random.seed(100)
        protocol_model = build_protocol_model(
            model_config_from_protocol(
                protocol,
                model_name,
                width_override=_phase0_width_override(model_name, smoke=smoke),
            )
        ).to(device)
        adapter = StateAdapter(protocol_model.core)
        report = run_phase0_audit(
            adapter,
            config=AuditConfig(
                batch_size=2,
                seed=20260713 + index,
                random_directions=2 if smoke else 16,
                architecture_max_dimension=32 if smoke else 512,
            ),
        )
        implemented_maps = {
            key
            for key in report["checks"]["required_blank_maps"]
            if key.startswith("F0_")
        }
        declared_maps = set(protocol["phase0_state_audit"]["required_maps"])
        if implemented_maps != declared_maps:
            raise RuntimeError(
                "Phase-0 implementation/freeze blank-map mismatch: "
                f"implemented={sorted(implemented_maps)}, declared={sorted(declared_maps)}"
            )
        model_dir = root / f"model={model_name}"
        model_dir.mkdir(parents=True, exist_ok=False)
        report_path = model_dir / "state_transition_audit.json"
        state_path = model_dir / "state_spec.json"
        trace_path = model_dir / "blank_map_trace.npz"
        trace_metadata_path = model_dir / "blank_map_trace_metadata.json"
        exact_path = model_dir / "exactness_screen.json"
        determinism_path = model_dir / "determinism_check.json"
        jacobian_path = model_dir / "jacobian_check.json"
        float64_path = model_dir / "float64_subset_check.json"
        hidden_cache_path = model_dir / "hidden_cache_check.json"
        atomic_json(report_path, report)
        atomic_json(state_path, report["state_spec"])
        atomic_json(exact_path, report["checks"]["architecture_exactness_screen"])
        atomic_json(determinism_path, report["checks"]["determinism"])
        atomic_json(
            jacobian_path,
            report["checks"]["autograd_vs_central_finite_difference"],
        )
        atomic_json(float64_path, report["checks"]["float64_subset"])
        atomic_json(hidden_cache_path, report["checks"]["no_external_hidden_cache"])
        trace_arrays, trace_metadata = _blank_trace(adapter, seed=20260713 + index)
        _atomic_npz(trace_path, **trace_arrays)
        atomic_json(trace_metadata_path, trace_metadata)
        artifacts.extend(
            [
                report_path,
                state_path,
                trace_path,
                trace_metadata_path,
                exact_path,
                determinism_path,
                jacobian_path,
                float64_path,
                hidden_cache_path,
            ]
        )
        required_checks = {
            "state_transition_inventory": bool(
                report["checks"]["state_transition_inventory"]["passed"]
            ),
            "pack_unpack_round_trip": bool(
                report["checks"]["pack_unpack_round_trip"]["passed"]
            ),
            "actual_blank_map": bool(
                report["checks"]["zero_input"]["passed"]
                and report["checks"]["actual_f0"]["passed"]
                and report["checks"]["required_blank_maps"]["passed"]
            ),
            "blank_input_trace": bool(
                trace_metadata["external_input_exactly_zero"]
                and not trace_metadata["external_reset_present"]
                and int(trace_metadata["steps"]) == 20
            ),
            "determinism": bool(report["checks"]["determinism"]["passed"]),
            "analysis_mode": bool(report["checks"]["analysis_mode"]["passed"]),
            "float64_subset": bool(report["checks"]["float64_subset"]["passed"]),
            "jacobian_finite_difference": bool(
                report["checks"]["autograd_vs_central_finite_difference"]["passed"]
            ),
            "hidden_cache_audit": bool(
                report["checks"]["no_external_hidden_cache"]["passed"]
            ),
        }
        declared_check_ids = {
            str(item["id"]) for item in protocol["phase0_state_audit"]["required_checks"]
        }
        if set(required_checks) != declared_check_ids:
            raise RuntimeError(
                "Phase-0 implementation/freeze required-check mismatch: "
                f"implemented={sorted(required_checks)}, declared={sorted(declared_check_ids)}"
            )
        reports[model_name] = {
            "passed": bool(report["passed"] and all(required_checks.values())),
            "required_checks": required_checks,
            "trained_checkpoint_followup_required": True,
            "trained_checkpoint_followup": "enforced_before_phase1_claim_artifacts",
            "primary_dimension": report["state_spec"]["primary_dimension"],
            "reported_dimension": report["state_spec"]["reported_dimension"],
            "overwritten_components": report["state_spec"]["overwritten_components"],
            "exact_continuum_ruled_out": report["checks"]["architecture_exactness_screen"].get(
                "exact_continuum_ruled_out", False
            ),
            "audit_path": str(report_path.relative_to(root)),
        }
    required_artifact_names = tuple(
        str(value) for value in protocol["phase0_state_audit"]["required_artifacts"]
    )
    artifact_presence: dict[str, bool] = {}
    for name in required_artifact_names:
        if name == "phase0_gate.json":
            # The gate is written atomically immediately below after embedding
            # this validation record.
            artifact_presence[name] = True
        else:
            artifact_presence[name] = all(
                (root / f"model={model_name}" / name).is_file()
                for model_name in reports
            )
    passed = bool(
        all(item["passed"] for item in reports.values())
        and all(artifact_presence.values())
    )
    gate = {
        "schema_version": 1,
        "phase_id": "phase0_state_audit",
        "passed": passed,
        "blocks_phase1_on_failure": True,
        "models": reports,
        "required_artifacts": artifact_presence,
        "interpretation": (
            "primary Markov maps are auditable; exactness screens do not block approximate-CA analysis"
            if passed
            else "Phase 1 is blocked"
        ),
    }
    gate_path = root / "phase0_gate.json"
    manifest_path = root / "manifest.json"
    atomic_json(gate_path, gate)
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "protocol_freeze_id": protocol["freeze_id"],
            "protocol_file_sha256": protocol_hash,
            "protocol_canonical_fingerprint": protocol_fingerprint(protocol),
            "source_protocol_sha256": protocol["source_protocol"]["sha256"],
            "pilot_only": pilot_only,
            "campaign_type": (
                "sagodi_primary_main_v3" if primary_main else "sagodi_protocol_pilot"
            ),
            "parent_selector": protocol.get("parent_selector"),
        },
    )
    artifacts.extend([gate_path, manifest_path])
    write_completion_receipt(
        completion,
        job_id="phase0_state_audit",
        artifacts=artifacts,
        metadata={
            "passed": passed,
            "pilot_only": pilot_only,
            "campaign_type": (
                "sagodi_primary_main_v3" if primary_main else "sagodi_protocol_pilot"
            ),
            "parent_selector": protocol.get("parent_selector"),
        },
    )
    if not passed:
        raise RuntimeError("Phase-0 audit failed; Phase 1 must not run")
    return root


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = run_phase0(args.protocol, args.output_root, device=args.device, smoke=args.smoke)
    print(json.dumps({"status": "complete", "output_root": str(output)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
