"""Run the no-gate engineering-benefit evaluation over all 60 main checkpoints."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import (
    RECEIPT_IDENTITY_ENV,
    atomic_bytes,
    atomic_json,
    canonical_hash,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from .engineering_benefit_runner import (
    ANALYSIS_NAME,
    ANALYSIS_ROLE,
    MODEL_IDS,
    MODEL_SEEDS,
    EngineeringBenefitSpec,
    load_engineering_freeze,
)
from .orchestrate import (
    _acquire_campaign_lock,
    _environment_fingerprint,
    _git_state,
    _preserve_invalid_output,
    _resolve_python,
    _source_hashes,
    _unique_attempt_path,
    _validated_gpu_ids,
)
from .primary_main_campaign import (
    COMPLETE as MAIN_COMPLETE,
    COMPLETION_RECEIPT as MAIN_COMPLETION_RECEIPT,
    MANIFEST as MAIN_MANIFEST,
    SUMMARY as MAIN_SUMMARY,
    MainRun,
    _training_output,
    _verify_training_output,
    build_main_plan,
    verify_main_completion,
    verify_parent_selector,
)


CAMPAIGN_TYPE = "calru_engineering_benefit_v1"
CAMPAIGN_SCOPE = "project_engineering_utility_not_sagodi_matched_no_binary_ca_gates"
EXPECTED_RUNS = 60
ROOT_MARKER = ".calru_engineering_benefit_v1_root.json"
FREEZE_COPY = "engineering_benefit_freeze.json"
MANIFEST = "engineering_benefit_manifest.json"
STATUS = "status.json"
SUMMARY = "engineering_benefit_summary.json"
COMPLETE = "COMPLETE"
COMPLETION_RECEIPT = "completion_receipt.json"


@dataclass(frozen=True)
class EngineeringRun:
    model_id: str
    model_seed: int
    checkpoint: Path
    checkpoint_sha256: str
    training_receipt: Path
    training_receipt_sha256: str

    @property
    def run_id(self) -> str:
        return f"engineering_benefit__{self.model_id}__seed{self.model_seed:02d}"

    @property
    def main_run_id(self) -> str:
        return f"primary_main__{self.model_id}__seed{self.model_seed:02d}"

    @property
    def join_key(self) -> str:
        return f"{self.model_id}::seed{self.model_seed:02d}"


@dataclass(frozen=True)
class VerifiedMainParent:
    main_root: Path
    selector_root: Path
    parent_selector: Any
    main_manifest: dict[str, Any]
    main_plan: tuple[MainRun, ...]
    runs: tuple[EngineeringRun, ...]
    binding: dict[str, Any]


def expected_utility_metric_keys() -> tuple[str, ...]:
    keys: list[str] = []
    for label in ("1T", "2T", "4T", "8T", "16T"):
        keys.extend(
            (
                f"temporal/{label}/final_mean_error_radians",
                f"temporal/{label}/prefix_mean_error_radians",
            )
        )
    for label in ("1x", "2x", "4x"):
        keys.extend(
            (
                f"velocity/{label}/final_mean_error_radians",
                f"velocity/{label}/sequence_mean_error_radians",
            )
        )
    for magnitude in ("0", "0.01", "0.1", "1"):
        for horizon in ("1T", "4T", "16T"):
            keys.extend(
                (
                    f"perturbation/relative_rms_{magnitude}/{horizon}/memory_mean_error_radians",
                    f"perturbation/relative_rms_{magnitude}/{horizon}/"
                    "clean_paired_mean_error_radians",
                )
            )
    return tuple(keys)


EXPECTED_UTILITY_METRIC_KEYS = expected_utility_metric_keys()


def verify_main_parent(
    main_root: Path | str, selector_root: Path | str
) -> VerifiedMainParent:
    """Recursively verify the selector, main campaign, and all 60 checkpoints."""

    main = Path(main_root).expanduser().resolve(strict=True)
    selector = Path(selector_root).expanduser().resolve(strict=True)
    parent_selector = verify_parent_selector(selector)
    main_manifest = strict_json_load(main / MAIN_MANIFEST)
    if not isinstance(main_manifest, dict):
        raise RuntimeError("main manifest is malformed")
    main_plan = build_main_plan(parent_selector)
    valid, reason = verify_main_completion(
        main, main_manifest, parent_selector, main_plan
    )
    if not valid:
        raise RuntimeError(f"main parent recursive verification failed: {reason}")
    if main_manifest.get("campaign_type") != "sagodi_primary_main_v3":
        raise RuntimeError("parent is not the frozen primary-main campaign")
    if main_manifest.get("smoke") is not False:
        raise RuntimeError("engineering campaign requires the full 5000-update parent")
    if int(main_manifest.get("expected_training_runs", -1)) != EXPECTED_RUNS:
        raise RuntimeError("main parent does not declare 60 runs")
    if main_manifest.get("failed_seed_replacement_policy") != "forbidden":
        raise RuntimeError("main parent seed-replacement policy differs")
    main_code = main_manifest.get("code")
    if (
        not isinstance(main_code, Mapping)
        or main_code.get("worktree_dirty") is not False
        or not isinstance(main_code.get("code_commit"), str)
        or len(main_code["code_commit"]) != 40
    ):
        raise RuntimeError("main parent was not produced from a clean exact commit")
    if tuple(run.model.model_id for run in main_plan[:: len(MODEL_SEEDS)]) != MODEL_IDS:
        raise RuntimeError("main parent model order differs")
    runs: list[EngineeringRun] = []
    nested: list[dict[str, Any]] = []
    for run in main_plan:
        output = _training_output(main, run)
        child_valid, child_reason, _ = _verify_training_output(
            output, run, main_manifest, parent=parent_selector
        )
        if not child_valid:
            raise RuntimeError(f"main child no longer verifies: {run.run_id}: {child_reason}")
        checkpoint = output / "checkpoint.pt"
        receipt = output / "completion_receipt.json"
        engineering = EngineeringRun(
            model_id=run.model.model_id,
            model_seed=run.model_seed,
            checkpoint=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            training_receipt=receipt,
            training_receipt_sha256=sha256_file(receipt),
        )
        runs.append(engineering)
        nested.append(
            {
                "main_run_id": engineering.main_run_id,
                "model_id": engineering.model_id,
                "model_seed": engineering.model_seed,
                "join_key": engineering.join_key,
                "checkpoint_sha256": engineering.checkpoint_sha256,
                "training_receipt_sha256": engineering.training_receipt_sha256,
            }
        )
    if len(runs) != EXPECTED_RUNS or len({run.join_key for run in runs}) != EXPECTED_RUNS:
        raise RuntimeError("main parent is not exactly 60 unique model/seed checkpoints")
    binding = {
        "schema_version": 1,
        "campaign_type": "sagodi_primary_main_v3",
        "scientific_identity": main_manifest["scientific_identity"],
        "manifest_sha256": sha256_file(main / MAIN_MANIFEST),
        "summary_sha256": sha256_file(main / MAIN_SUMMARY),
        "complete_sha256": sha256_file(main / MAIN_COMPLETE),
        "completion_receipt_sha256": sha256_file(main / MAIN_COMPLETION_RECEIPT),
        "main_code_commit": main_code["code_commit"],
        "selector_scientific_identity": parent_selector.binding[
            "scientific_identity"
        ],
        "verified_training_run_count": EXPECTED_RUNS,
        "failed_seed_replacements": 0,
        "nested_training_artifacts": nested,
    }
    return VerifiedMainParent(
        main,
        selector,
        parent_selector,
        main_manifest,
        tuple(main_plan),
        tuple(runs),
        binding,
    )


def _prepare_root(
    root: Path,
    freeze_source: Path,
    freeze: Mapping[str, Any],
    parent: VerifiedMainParent,
    *,
    smoke: bool,
) -> None:
    marker_payload = {
        "schema_version": 1,
        "campaign_type": CAMPAIGN_TYPE,
        "scope": CAMPAIGN_SCOPE,
        "freeze_source_sha256": sha256_file(freeze_source),
        "freeze_canonical_fingerprint": canonical_hash(freeze),
        "parent_main": parent.binding,
        "smoke": bool(smoke),
    }
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ROOT_MARKER
    if marker.exists():
        if strict_json_load(marker) != marker_payload:
            raise RuntimeError("engineering campaign root marker differs")
    else:
        if any(root.iterdir()):
            raise RuntimeError("engineering campaign root is nonempty without its marker")
        atomic_json(marker, marker_payload)
    copy = root / FREEZE_COPY
    if copy.exists():
        if copy.read_bytes() != freeze_source.read_bytes():
            raise RuntimeError("immutable engineering freeze copy differs")
    else:
        atomic_bytes(copy, freeze_source.read_bytes())


def _build_manifest(
    *,
    parent: VerifiedMainParent,
    freeze: Mapping[str, Any],
    freeze_copy: Path,
    spec: EngineeringBenefitSpec,
    git_state: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    environment: Mapping[str, Any],
    python: str,
    gpus: Sequence[int],
    smoke: bool,
) -> dict[str, Any]:
    run_matrix = [
        {
            "run_id": run.run_id,
            "main_run_id": run.main_run_id,
            "model_id": run.model_id,
            "model_seed": run.model_seed,
            "join_key": run.join_key,
            "checkpoint_sha256": run.checkpoint_sha256,
            "training_receipt_sha256": run.training_receipt_sha256,
        }
        for run in parent.runs
    ]
    normalized_spec = json.loads(
        json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    payload = {
        "campaign_type": CAMPAIGN_TYPE,
        "scope": CAMPAIGN_SCOPE,
        "analysis_role": ANALYSIS_ROLE,
        "sagodi_matched_analysis": False,
        "binary_ca_gates": False,
        "expected_direction_pass_thresholds": False,
        "utility_scalar_nonfinite_policy": (
            "null_unless_all_registered_trial_values_are_finite"
        ),
        "angle_utility_radius_policy": (
            "null_unless_all_corresponding_estimate_and_reference_radii_are_"
            "finite_and_at_least_epsilon"
        ),
        "seed_exclusion_or_replacement": "forbidden",
        "smoke": bool(smoke),
        "parent_main": parent.binding,
        "freeze_file": FREEZE_COPY,
        "freeze_file_sha256": sha256_file(freeze_copy),
        "freeze_canonical_fingerprint": canonical_hash(freeze),
        "spec": normalized_spec,
        "code": dict(git_state),
        "source_hashes": dict(source_hashes),
        "environment": dict(environment),
        "python": str(Path(python).resolve()),
        "gpus": [int(gpu) for gpu in gpus],
        "run_matrix": run_matrix,
        "expected_runs": EXPECTED_RUNS,
        "utility_metric_keys": list(EXPECTED_UTILITY_METRIC_KEYS),
        "association_join_keys": ["model_id", "model_seed"],
    }
    return {
        "schema_version": 1,
        **payload,
        "scientific_identity_payload": payload,
        "scientific_identity": canonical_hash(payload),
    }


def _write_or_check(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if strict_json_load(path) != dict(payload):
            raise RuntimeError(f"immutable campaign file differs: {path.name}")
    else:
        atomic_json(path, dict(payload))


def _fast_verify_parent(parent: VerifiedMainParent) -> None:
    root_paths = {
        "manifest_sha256": parent.main_root / MAIN_MANIFEST,
        "summary_sha256": parent.main_root / MAIN_SUMMARY,
        "complete_sha256": parent.main_root / MAIN_COMPLETE,
        "completion_receipt_sha256": parent.main_root / MAIN_COMPLETION_RECEIPT,
    }
    for key, path in root_paths.items():
        if not path.is_file() or sha256_file(path) != parent.binding[key]:
            raise RuntimeError(f"parent main binding changed: {key}")
    expected = {
        item["join_key"]: item
        for item in parent.binding["nested_training_artifacts"]
    }
    for run in parent.runs:
        binding = expected[run.join_key]
        if sha256_file(run.checkpoint) != binding["checkpoint_sha256"]:
            raise RuntimeError(f"parent checkpoint changed: {run.join_key}")
        if sha256_file(run.training_receipt) != binding["training_receipt_sha256"]:
            raise RuntimeError(f"parent training receipt changed: {run.join_key}")


def _validate_launch_code_identity(
    git_state: Mapping[str, Any],
    parent: VerifiedMainParent,
    *,
    smoke: bool,
) -> None:
    """Require the exact training commit for every full downstream launch."""

    if smoke:
        return
    if git_state.get("worktree_dirty") is not False:
        raise RuntimeError("full engineering campaign requires a clean committed worktree")
    if git_state.get("code_commit") != parent.binding["main_code_commit"]:
        raise RuntimeError(
            "full engineering campaign must use the exact main-training code commit"
        )


def _verify_campaign_inputs(
    root: Path,
    manifest: Mapping[str, Any],
    parent: VerifiedMainParent,
    repo_root: Path,
) -> None:
    signed = manifest.get("scientific_identity_payload")
    if not isinstance(signed, Mapping) or canonical_hash(signed) != manifest.get(
        "scientific_identity"
    ):
        raise RuntimeError("engineering campaign scientific identity is invalid")
    if strict_json_load(root / MANIFEST) != dict(manifest):
        raise RuntimeError("on-disk engineering manifest differs")
    if _git_state(repo_root) != manifest["code"]:
        raise RuntimeError("git state changed after engineering manifest creation")
    if _source_hashes(repo_root, Path(__file__).resolve().parent) != manifest[
        "source_hashes"
    ]:
        raise RuntimeError("engineering campaign source hashes changed")
    if sha256_file(root / FREEZE_COPY) != manifest["freeze_file_sha256"]:
        raise RuntimeError("engineering freeze copy changed")
    freeze, spec = load_engineering_freeze(
        root / FREEZE_COPY, smoke=bool(manifest["smoke"])
    )
    if canonical_hash(freeze) != manifest["freeze_canonical_fingerprint"]:
        raise RuntimeError("engineering freeze semantics changed")
    if canonical_hash(asdict(spec)) != canonical_hash(manifest["spec"]):
        raise RuntimeError("resolved engineering spec changed")
    _fast_verify_parent(parent)


def _output_dir(root: Path, run: EngineeringRun) -> Path:
    return root / "evaluation" / f"model={run.model_id}" / f"seed={run.model_seed}"


def _child_metadata(
    manifest: Mapping[str, Any], run: EngineeringRun
) -> dict[str, Any]:
    return {
        "campaign_scientific_identity": str(manifest["scientific_identity"]),
        "protocol_fingerprint": str(manifest["freeze_canonical_fingerprint"]),
        "run_id": run.run_id,
        "stage": "engineering_benefit_evaluation",
        "analysis_role": ANALYSIS_ROLE,
        "model_id": run.model_id,
        "model_seed": run.model_seed,
    }


def _expected_child_identity(
    run: EngineeringRun,
    manifest: Mapping[str, Any],
) -> tuple[str, str]:
    payload = {
        "analysis": ANALYSIS_NAME,
        "schema_version": 1,
        "analysis_role": ANALYSIS_ROLE,
        "checkpoint_sha256": run.checkpoint_sha256,
        "freeze_file_sha256": manifest["freeze_file_sha256"],
        "freeze_canonical_fingerprint": manifest["freeze_canonical_fingerprint"],
        "model_id": run.model_id,
        "model_seed": run.model_seed,
        "spec": manifest["spec"],
    }
    identity = canonical_hash(payload)
    job_id = f"engineering-benefit-{run.model_id}-seed{run.model_seed}-{identity[:12]}"
    return identity, job_id


def verify_engineering_output(
    output: Path,
    run: EngineeringRun,
    manifest: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any] | None]:
    identity, job_id = _expected_child_identity(run, manifest)
    expected_metadata = {
        **_child_metadata(manifest, run),
        "analysis_identity": identity,
        "join_key": run.join_key,
    }
    valid, reason = verify_completion_receipt(
        output / "completion_receipt.json",
        expected_job_id=job_id,
        expected_metadata=expected_metadata,
    )
    if not valid:
        return False, reason, None
    try:
        summary = strict_json_load(output / "summary.json")
        child_manifest = strict_json_load(output / "manifest.json")
        complete = strict_json_load(output / "COMPLETE")
        receipt = strict_json_load(output / "completion_receipt.json")
        with np.load(output / "individual_errors.npz", allow_pickle=False) as arrays:
            trial_count = int(manifest["spec"]["trial_count"])
            task_horizon = int(manifest["spec"]["task_horizon"])
            shapes = {
                "temporal_absolute_error": (16 * task_horizon, trial_count),
                "temporal_final_absolute_error": (5, trial_count),
                "temporal_prefix_mean_absolute_error": (5, trial_count),
                "velocity_absolute_error": (3, task_horizon, trial_count),
                "velocity_final_absolute_error": (3, trial_count),
                "velocity_sequence_mean_absolute_error": (3, trial_count),
                "perturbation_memory_absolute_error": (4, 3, trial_count),
                "perturbation_clean_paired_absolute_error": (4, 3, trial_count),
                "perturbation_output_radius": (4, 3, trial_count),
                "perturbation_clean_output_radius": (3, trial_count),
                "perturbation_pre_memory_output_radius": (trial_count,),
                "perturbation_realized_relative_l2": (4, trial_count),
            }
            for key, shape in shapes.items():
                if key not in arrays or arrays[key].shape != shape:
                    return False, f"individual error shape mismatch: {key}", None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, f"engineering child is unreadable: {exc}", None
    required = {
        "engineering_benefit_freeze.json",
        "individual_errors.npz",
        "manifest.json",
        "summary.json",
        "COMPLETE",
    }
    if not isinstance(receipt.get("artifacts"), Mapping) or not required.issubset(
        receipt["artifacts"]
    ):
        return False, "engineering child receipt lacks required artifacts", None
    for payload, label in ((summary, "summary"), (child_manifest, "manifest")):
        if not isinstance(payload, Mapping):
            return False, f"engineering child {label} is not an object", None
        checks = {
            "analysis_identity": identity,
            "analysis_role": ANALYSIS_ROLE,
            "model_id": run.model_id,
            "model_seed": run.model_seed,
            "join_key": run.join_key,
        }
        for key, expected in checks.items():
            if payload.get(key) != expected:
                return False, f"engineering child {label} {key} mismatch", None
    if child_manifest.get("checkpoint_sha256") != run.checkpoint_sha256:
        return False, "engineering child checkpoint binding mismatch", None
    if child_manifest.get("freeze_file_sha256") != manifest["freeze_file_sha256"]:
        return False, "engineering child freeze binding mismatch", None
    if child_manifest.get("binary_ca_gates") is not False:
        return False, "engineering child unexpectedly contains CA gates", None
    if child_manifest.get("expected_direction_pass_thresholds") is not False:
        return False, "engineering child unexpectedly contains pass thresholds", None
    expected_nonfinite_policy = (
        "null_unless_all_registered_trial_values_are_finite"
    )
    if child_manifest.get("utility_scalar_nonfinite_policy") != expected_nonfinite_policy:
        return False, "engineering child manifest nonfinite policy mismatch", None
    if summary.get("utility_scalar_nonfinite_policy") != expected_nonfinite_policy:
        return False, "engineering child summary nonfinite policy mismatch", None
    expected_radius_policy = (
        "null_unless_all_corresponding_estimate_and_reference_radii_are_"
        "finite_and_at_least_epsilon"
    )
    if child_manifest.get("angle_utility_radius_policy") != expected_radius_policy:
        return False, "engineering child manifest radius policy mismatch", None
    if summary.get("angle_utility_radius_policy") != expected_radius_policy:
        return False, "engineering child summary radius policy mismatch", None
    metrics = summary.get("utility_metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(
        EXPECTED_UTILITY_METRIC_KEYS
    ):
        return False, "engineering child utility metric keys differ", None
    for value in metrics.values():
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return False, "engineering child utility metric is malformed", None
    if complete.get("analysis_identity") != identity:
        return False, "engineering child COMPLETE identity mismatch", None
    if complete.get("model_id") != run.model_id:
        return False, "engineering child COMPLETE model id mismatch", None
    if complete.get("model_seed") != run.model_seed:
        return False, "engineering child COMPLETE model seed mismatch", None
    if complete.get("join_key") != run.join_key:
        return False, "engineering child COMPLETE join key mismatch", None
    if complete.get("summary_sha256") != sha256_file(output / "summary.json"):
        return False, "engineering child COMPLETE summary hash mismatch", None
    if complete.get("individual_errors_sha256") != sha256_file(
        output / "individual_errors.npz"
    ):
        return False, "engineering child COMPLETE individual-error hash mismatch", None
    bindings = {
        "summary_sha256": sha256_file(output / "summary.json"),
        "individual_errors_sha256": sha256_file(output / "individual_errors.npz"),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "completion_receipt_sha256": sha256_file(output / "completion_receipt.json"),
    }
    return True, "verified engineering child receipt and individual arrays", bindings


def _child_environment(
    manifest: Mapping[str, Any], run: EngineeringRun, gpu: int
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    environment["CALRU_PHYSICAL_GPU_ID"] = str(int(gpu))
    metadata = _child_metadata(manifest, run)
    for key, variable in RECEIPT_IDENTITY_ENV.items():
        environment[variable] = str(metadata[key])
    return environment


def _child_preexec(expected_parent: int) -> Any:
    def configure() -> None:
        os.setsid()
        if not sys.platform.startswith("linux"):
            os._exit(125)
        libc = ctypes.CDLL(None, use_errno=True)
        if int(libc.prctl(1, signal.SIGKILL, 0, 0, 0)) != 0:
            os._exit(125)
        if os.getppid() != expected_parent:
            os.kill(os.getpid(), signal.SIGKILL)

    return configure


def _command(
    root: Path,
    attempt: Path,
    run: EngineeringRun,
    python: str,
    *,
    smoke: bool,
) -> list[str]:
    command = [
        python,
        "-m",
        "repro.sagodi_protocol.engineering_benefit_runner",
        "--checkpoint",
        str(run.checkpoint),
        "--freeze",
        str(root / FREEZE_COPY),
        "--output-dir",
        str(attempt),
        "--model-id",
        run.model_id,
        "--model-seed",
        str(run.model_seed),
        "--device",
        "cuda:0",
    ]
    if smoke:
        command.append("--smoke")
    return command


def _terminate_active(active: Mapping[int, Mapping[str, Any]]) -> None:
    for item in active.values():
        process = item["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 10.0
    while time.time() < deadline and any(
        item["process"].poll() is None for item in active.values()
    ):
        time.sleep(0.1)
    for item in active.values():
        process = item["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
        item["handle"].close()


def _recover_attempt(
    root: Path, run: EngineeringRun, manifest: Mapping[str, Any]
) -> tuple[bool, str]:
    parent = root / "attempts" / "engineering_benefit" / run.run_id
    if not parent.is_dir():
        return False, "no prior attempts"
    valid: list[Path] = []
    for candidate in sorted(path for path in parent.iterdir() if path.is_dir()):
        ok, _, _ = verify_engineering_output(candidate, run, manifest)
        if ok:
            valid.append(candidate)
    if len(valid) > 1:
        raise RuntimeError(f"multiple valid attempts for {run.run_id}; audit required")
    if not valid:
        return False, "no valid prior attempts"
    output = _output_dir(root, run)
    if output.exists():
        raise RuntimeError(f"cannot recover {run.run_id} over existing output")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(valid[0], output)
    return True, "recovered verified prior attempt"


def _run_jobs(
    root: Path,
    manifest: Mapping[str, Any],
    parent: VerifiedMainParent,
    python: str,
    gpus: Sequence[int],
    status: dict[str, Any],
    repo_root: Path,
) -> None:
    jobs = status.setdefault("jobs", {})
    pending: list[EngineeringRun] = []
    for run in parent.runs:
        output = _output_dir(root, run)
        valid, reason, _ = verify_engineering_output(output, run, manifest)
        if valid:
            jobs[run.run_id] = {"state": "complete", "reason": reason}
            continue
        if output.exists():
            preserved = _preserve_invalid_output(
                root, "engineering_benefit", run.run_id, output
            )
            reason = f"invalid output preserved at {preserved}: {reason}"
        recovered, recovery_reason = _recover_attempt(root, run, manifest)
        if recovered:
            jobs[run.run_id] = {"state": "complete", "reason": recovery_reason}
            continue
        jobs[run.run_id] = {"state": "pending", "reason": reason}
        pending.append(run)
    atomic_json(root / STATUS, status)
    active: dict[int, dict[str, Any]] = {}
    try:
        while pending or active:
            while pending and len(active) < len(gpus):
                _verify_campaign_inputs(root, manifest, parent, repo_root)
                used = {int(item["gpu"]) for item in active.values()}
                gpu = next(value for value in gpus if value not in used)
                run = pending.pop(0)
                attempt = _unique_attempt_path(
                    root, "engineering_benefit", run.run_id
                )
                log_dir = root / "logs" / "engineering_benefit" / run.run_id
                log_dir.mkdir(parents=True, exist_ok=True)
                log = log_dir / f"attempt-{attempt.name}.log"
                handle = log.open("ab", buffering=0)
                process = subprocess.Popen(
                    _command(
                        root,
                        attempt,
                        run,
                        python,
                        smoke=bool(manifest["smoke"]),
                    ),
                    cwd=repo_root,
                    env=_child_environment(manifest, run, gpu),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    preexec_fn=_child_preexec(os.getpid()),
                )
                active[process.pid] = {
                    "process": process,
                    "handle": handle,
                    "run": run,
                    "attempt": attempt,
                    "gpu": gpu,
                    "log": log,
                }
                jobs[run.run_id] = {
                    "state": "running",
                    "gpu": int(gpu),
                    "attempt": str(attempt),
                    "log": str(log),
                    "pid": process.pid,
                }
                atomic_json(root / STATUS, status)
            completed = [
                pid
                for pid, item in active.items()
                if item["process"].poll() is not None
            ]
            if not completed:
                time.sleep(0.25)
                continue
            for pid in completed:
                item = active.pop(pid)
                process = item["process"]
                item["handle"].close()
                run = item["run"]
                attempt = item["attempt"]
                if process.returncode != 0:
                    jobs[run.run_id] = {
                        "state": "failed",
                        "return_code": process.returncode,
                        "attempt": str(attempt),
                        "log": str(item["log"]),
                    }
                    atomic_json(root / STATUS, status)
                    raise RuntimeError(
                        f"engineering evaluation {run.run_id} exited {process.returncode}; "
                        f"see {item['log']}"
                    )
                valid, reason, _ = verify_engineering_output(
                    attempt, run, manifest
                )
                if not valid:
                    jobs[run.run_id] = {
                        "state": "failed_verification",
                        "reason": reason,
                        "attempt": str(attempt),
                    }
                    atomic_json(root / STATUS, status)
                    raise RuntimeError(
                        f"engineering output {run.run_id} invalid: {reason}"
                    )
                output = _output_dir(root, run)
                if output.exists():
                    raise RuntimeError(f"publish target appeared during run: {output}")
                output.parent.mkdir(parents=True, exist_ok=True)
                os.replace(attempt, output)
                valid, reason, _ = verify_engineering_output(
                    output, run, manifest
                )
                if not valid:
                    raise RuntimeError(f"published engineering output invalid: {reason}")
                jobs[run.run_id] = {"state": "complete", "reason": reason}
                atomic_json(root / STATUS, status)
    except BaseException:
        _terminate_active(active)
        raise


def _metric_summary(values: Sequence[float | None]) -> dict[str, Any]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    result: dict[str, Any] = {
        "seed_count": len(values),
        "finite_seed_count": int(finite.size),
        "nonfinite_seed_count": int(len(values) - finite.size),
    }
    if not finite.size:
        result.update({"mean": None, "std": None, "median": None, "min": None, "max": None})
    else:
        result.update(
            {
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=0)),
                "median": float(np.median(finite)),
                "min": float(finite.min()),
                "max": float(finite.max()),
            }
        )
    return result


def aggregate_engineering_results(
    root: Path,
    manifest: Mapping[str, Any],
    runs: Sequence[EngineeringRun],
) -> dict[str, Any]:
    """Build model/seed rows and per-model descriptive summaries, with no ranking."""

    rows: list[dict[str, Any]] = []
    for run in runs:
        output = _output_dir(root, run)
        valid, reason, bindings = verify_engineering_output(output, run, manifest)
        if not valid or bindings is None:
            raise RuntimeError(f"cannot aggregate {run.run_id}: {reason}")
        child = strict_json_load(output / "summary.json")
        metric_status = {
            key: (
                "complete"
                if child["utility_metrics"][key] is not None
                else "engineering_metric_nonfinite"
            )
            for key in EXPECTED_UTILITY_METRIC_KEYS
        }
        rows.append(
            {
                "model_id": run.model_id,
                "model_seed": run.model_seed,
                "join_key": run.join_key,
                "main_run_id": run.main_run_id,
                "engineering_run_id": run.run_id,
                "utility_metrics": child["utility_metrics"],
                "utility_metric_status": metric_status,
                "nonfinite_utility_metric_count": sum(
                    value == "engineering_metric_nonfinite"
                    for value in metric_status.values()
                ),
                "artifact_bindings": bindings,
                "seed_excluded": False,
            }
        )
    models: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        if tuple(row["model_seed"] for row in model_rows) != MODEL_SEEDS:
            raise RuntimeError(f"model seed order differs for {model_id}")
        models[model_id] = {
            "seed_count": len(model_rows),
            "excluded_seed_count": 0,
            "utility_metrics": {
                key: _metric_summary(
                    [row["utility_metrics"][key] for row in model_rows]
                )
                for key in EXPECTED_UTILITY_METRIC_KEYS
            },
        }
    return {
        "schema_version": 1,
        "campaign_type": CAMPAIGN_TYPE,
        "scope": CAMPAIGN_SCOPE,
        "analysis_role": ANALYSIS_ROLE,
        "campaign_scientific_identity": manifest["scientific_identity"],
        "smoke": bool(manifest["smoke"]),
        "expected_runs": EXPECTED_RUNS,
        "verified_run_count": len(rows),
        "failed_seed_replacements": 0,
        "excluded_seed_count": 0,
        "binary_ca_gates": False,
        "expected_direction_pass_thresholds": False,
        "utility_scalar_nonfinite_policy": (
            "null_unless_all_registered_trial_values_are_finite"
        ),
        "angle_utility_radius_policy": (
            "null_unless_all_corresponding_estimate_and_reference_radii_are_"
            "finite_and_at_least_epsilon"
        ),
        "claim_thresholds": None,
        "association_join_keys": ["model_id", "model_seed"],
        "parent_main": manifest["parent_main"],
        "utility_metric_keys": list(EXPECTED_UTILITY_METRIC_KEYS),
        "runs": rows,
        "models": models,
    }


def _root_receipt_metadata(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "campaign_scientific_identity": manifest["scientific_identity"],
        "protocol_fingerprint": manifest["freeze_canonical_fingerprint"],
        "run_id": "engineering_benefit_v1",
        "stage": "finalization",
        "analysis_role": ANALYSIS_ROLE,
        "run_count": EXPECTED_RUNS,
        "binary_ca_gates": False,
        "expected_direction_pass_thresholds": False,
        "utility_scalar_nonfinite_policy": (
            "null_unless_all_registered_trial_values_are_finite"
        ),
        "angle_utility_radius_policy": (
            "null_unless_all_corresponding_estimate_and_reference_radii_are_"
            "finite_and_at_least_epsilon"
        ),
        "claim_thresholds": None,
    }


def verify_engineering_campaign_completion(
    root: Path,
    manifest: Mapping[str, Any],
    runs: Sequence[EngineeringRun],
) -> tuple[bool, str]:
    try:
        summary = strict_json_load(root / SUMMARY)
        complete = strict_json_load(root / COMPLETE)
        receipt = strict_json_load(root / COMPLETION_RECEIPT)
        for run in runs:
            valid, reason, _ = verify_engineering_output(
                _output_dir(root, run), run, manifest
            )
            if not valid:
                return False, f"invalid nested engineering run {run.run_id}: {reason}"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, f"engineering completion is unreadable: {exc}"
    valid, reason = verify_completion_receipt(
        root / COMPLETION_RECEIPT,
        expected_job_id="calru_engineering_benefit_v1_campaign_complete",
        expected_metadata=_root_receipt_metadata(manifest),
    )
    if not valid:
        return False, reason
    required_receipt_artifacts = {
        MANIFEST,
        FREEZE_COPY,
        SUMMARY,
        COMPLETE,
        *{
            (
                f"evaluation/model={run.model_id}/seed={run.model_seed}/"
                "completion_receipt.json"
            )
            for run in runs
        },
    }
    if not isinstance(receipt.get("artifacts"), Mapping) or not (
        required_receipt_artifacts.issubset(receipt["artifacts"])
    ):
        return False, "engineering root receipt lacks nested receipt bindings"
    if summary.get("verified_run_count") != EXPECTED_RUNS:
        return False, "engineering summary run count mismatch"
    if summary.get("campaign_scientific_identity") != manifest["scientific_identity"]:
        return False, "engineering summary identity mismatch"
    if summary.get("binary_ca_gates") is not False:
        return False, "engineering summary unexpectedly contains CA gates"
    if summary.get("expected_direction_pass_thresholds") is not False:
        return False, "engineering summary unexpectedly contains pass thresholds"
    if summary.get("utility_scalar_nonfinite_policy") != (
        "null_unless_all_registered_trial_values_are_finite"
    ):
        return False, "engineering summary nonfinite utility policy mismatch"
    if summary.get("angle_utility_radius_policy") != (
        "null_unless_all_corresponding_estimate_and_reference_radii_are_"
        "finite_and_at_least_epsilon"
    ):
        return False, "engineering summary angle-radius utility policy mismatch"
    if complete.get("campaign_scientific_identity") != manifest["scientific_identity"]:
        return False, "engineering COMPLETE identity mismatch"
    if complete.get("summary_sha256") != sha256_file(root / SUMMARY):
        return False, "engineering COMPLETE summary hash mismatch"
    if complete.get("failed_seed_replacements") != 0:
        return False, "engineering completion contains seed replacement"
    return True, "verified all 60 engineering-benefit child receipts"


def _finalize(
    root: Path,
    manifest: Mapping[str, Any],
    runs: Sequence[EngineeringRun],
) -> dict[str, Any]:
    summary = aggregate_engineering_results(root, manifest, runs)
    atomic_json(root / SUMMARY, summary)
    atomic_json(
        root / COMPLETE,
        {
            "schema_version": 1,
            "status": "complete",
            "campaign_scientific_identity": manifest["scientific_identity"],
            "verified_run_count": EXPECTED_RUNS,
            "failed_seed_replacements": 0,
            "excluded_seed_count": 0,
            "summary_sha256": sha256_file(root / SUMMARY),
        },
    )
    write_completion_receipt(
        root / COMPLETION_RECEIPT,
        job_id="calru_engineering_benefit_v1_campaign_complete",
        artifacts=[
            root / MANIFEST,
            root / FREEZE_COPY,
            root / SUMMARY,
            root / COMPLETE,
            *[
                _output_dir(root, run) / "completion_receipt.json"
                for run in runs
            ],
        ],
        metadata=_root_receipt_metadata(manifest),
    )
    valid, reason = verify_engineering_campaign_completion(root, manifest, runs)
    if not valid:
        raise RuntimeError(f"final engineering recursive verification failed: {reason}")
    return summary


def run_engineering_campaign(
    *,
    main_root: Path,
    selector_root: Path,
    freeze_path: Path,
    artifact_root: Path,
    python: str,
    gpus: Sequence[int],
    smoke: bool = False,
) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    parent = verify_main_parent(main_root, selector_root)
    freeze_source = Path(freeze_path).expanduser().resolve(strict=True)
    freeze, spec = load_engineering_freeze(freeze_source, smoke=smoke)
    root = Path(artifact_root).expanduser().resolve()
    resolved_python = _resolve_python(python)
    selected_gpus = _validated_gpu_ids(gpus)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES; --gpus uses physical GPU ids")
    git_state = _git_state(repo_root)
    _validate_launch_code_identity(git_state, parent, smoke=smoke)
    _prepare_root(
        root, freeze_source, freeze, parent, smoke=bool(smoke)
    )
    lock = _acquire_campaign_lock(root)
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"engineering campaign interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        manifest = _build_manifest(
            parent=parent,
            freeze=freeze,
            freeze_copy=root / FREEZE_COPY,
            spec=spec,
            git_state=git_state,
            source_hashes=_source_hashes(repo_root, Path(__file__).resolve().parent),
            environment=_environment_fingerprint(resolved_python, selected_gpus),
            python=resolved_python,
            gpus=selected_gpus,
            smoke=smoke,
        )
        _write_or_check(root / MANIFEST, manifest)
        _verify_campaign_inputs(root, manifest, parent, repo_root)
        complete, _ = verify_engineering_campaign_completion(
            root, manifest, parent.runs
        )
        if complete:
            return root
        status: dict[str, Any] = {
            "schema_version": 1,
            "campaign_type": CAMPAIGN_TYPE,
            "scope": CAMPAIGN_SCOPE,
            "analysis_role": ANALYSIS_ROLE,
            "campaign_scientific_identity": manifest["scientific_identity"],
            "smoke": bool(smoke),
            "stage": "engineering_benefit_evaluation",
            "jobs": {},
        }
        if (root / STATUS).exists():
            existing = strict_json_load(root / STATUS)
            if isinstance(existing, Mapping):
                status["jobs"] = dict(existing.get("jobs", {}))
        atomic_json(root / STATUS, status)
        _run_jobs(
            root,
            manifest,
            parent,
            resolved_python,
            selected_gpus,
            status,
            repo_root,
        )
        # Full recursive parent verification is repeated at the scientific
        # finalization boundary, not merely the fast hash check used per launch.
        refreshed = verify_main_parent(parent.main_root, parent.selector_root)
        if refreshed.binding != parent.binding:
            raise RuntimeError("parent main identity changed during engineering run")
        _verify_campaign_inputs(root, manifest, parent, repo_root)
        status["stage"] = "finalizing"
        atomic_json(root / STATUS, status)
        summary = _finalize(root, manifest, parent.runs)
        status["stage"] = "complete"
        status["verified_run_count"] = summary["verified_run_count"]
        status["completed_at"] = time.time()
        atomic_json(root / STATUS, status)
        return root
    finally:
        lock.release()
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--selector-root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = run_engineering_campaign(
        main_root=args.main_root,
        selector_root=args.selector_root,
        freeze_path=args.freeze,
        artifact_root=args.artifact_root,
        python=args.python,
        gpus=tuple(int(value) for value in args.gpus.split(",") if value.strip()),
        smoke=args.smoke,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "artifact_root": str(root),
                "scope": CAMPAIGN_SCOPE,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
