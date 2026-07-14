"""Run the complete receipt-bound Ságodi-primary v3 experiment chain.

The launcher is deliberately operational: it does not change any scientific
freeze and it does not infer that a partially written child is complete.  It
invokes the five independently fail-closed campaign entry points in their
registered order.  Re-running this module re-invokes each entry point, which
causes completed children to be recursively verified and incomplete children
to resume from their own receipts.

There is no pipeline-level smoke mode.  The selector's smoke campaign does not
select learning rates, so it cannot be a scientifically valid parent of the
remaining four stages.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import (
    atomic_bytes,
    atomic_json,
    canonical_hash,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from .orchestrate import (
    _acquire_campaign_lock,
    _git_state,
    _resolve_python,
    _validated_gpu_ids,
)


PIPELINE_ID = "calru_sagodi_primary_v3_pipeline"
PIPELINE_SCOPE = (
    "sequential_launcher_lr_selection_main_primary_analysis_"
    "engineering_benefit_descriptive_association"
)
STAGE_ORDER = (
    "selector",
    "main",
    "primary_analysis",
    "engineering_benefit",
    "dynamics_utility_association",
)

ROOT_MARKER = ".calru_sagodi_primary_v3_pipeline_root.json"
MANIFEST = "pipeline_manifest.json"
STATUS = "pipeline_status.json"
COMPLETE = "COMPLETE"
COMPLETION_RECEIPT = "completion_receipt.json"
INPUT_DIRECTORY = "inputs"
LOG_DIRECTORY = "logs"

_MODULE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_SELECTOR = _MODULE_DIRECTORY / "sagodi_primary_lr_selection_v3.json"
DEFAULT_PROTOCOL = _MODULE_DIRECTORY / "sagodi_primary_lr_selection_v3.yaml"
DEFAULT_MAIN_TEMPLATE = _MODULE_DIRECTORY / "primary_main_template_v3.json"
DEFAULT_ENGINEERING_FREEZE = _MODULE_DIRECTORY / "engineering_benefit_freeze_v1.json"
DEFAULT_ASSOCIATION_FREEZE = (
    _MODULE_DIRECTORY / "dynamics_utility_association_freeze_v1.json"
)


class PipelineStageError(RuntimeError):
    """A child stage returned unsuccessfully or omitted its receipt boundary."""

    def __init__(self, stage: str, return_code: int, reason: str) -> None:
        super().__init__(f"pipeline stage {stage!r} failed: {reason}")
        self.stage = stage
        self.return_code = int(return_code)
        self.reason = reason


@dataclass(frozen=True)
class PipelineSources:
    selector: Path
    protocol: Path
    main_template: Path
    engineering_freeze: Path
    association_freeze: Path

    def items(self) -> tuple[tuple[str, Path], ...]:
        return (
            ("selector", self.selector),
            ("protocol", self.protocol),
            ("main_template", self.main_template),
            ("engineering_freeze", self.engineering_freeze),
            ("association_freeze", self.association_freeze),
        )


@dataclass(frozen=True)
class PipelineStage:
    name: str
    module: str
    artifact_root: Path
    command: tuple[str, ...]

    @property
    def completion_receipt(self) -> Path:
        return self.artifact_root / "completion_receipt.json"

    @property
    def complete_marker(self) -> Path:
        return self.artifact_root / "COMPLETE"

    def payload(self, pipeline_root: Path) -> dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "artifact_root_relative_to_pipeline": self.artifact_root.relative_to(
                pipeline_root
            ).as_posix(),
            "command": list(self.command),
            "command_display": shlex.join(self.command),
            "required_terminal_files": ["COMPLETE", "completion_receipt.json"],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_sources(sources: PipelineSources) -> PipelineSources:
    values: dict[str, Path] = {}
    for name, source in sources.items():
        try:
            values[name] = Path(source).expanduser().resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"pipeline source {name!r} is missing: {source}") from exc
        if not values[name].is_file():
            raise ValueError(f"pipeline source {name!r} is not a file: {values[name]}")
    return PipelineSources(**values)


def _source_payload(sources: PipelineSources) -> dict[str, dict[str, str]]:
    return {
        name: {"source_path": str(path), "sha256": sha256_file(path)}
        for name, path in sources.items()
    }


def _copied_sources(root: Path) -> PipelineSources:
    inputs = root / INPUT_DIRECTORY
    return PipelineSources(
        selector=inputs / "selector.json",
        protocol=inputs / "protocol.yaml",
        main_template=inputs / "primary_main_template.json",
        engineering_freeze=inputs / "engineering_benefit_freeze.json",
        association_freeze=inputs / "dynamics_utility_association_freeze.json",
    )


def _materialize_sources(
    root: Path,
    sources: PipelineSources,
    expected: Mapping[str, Mapping[str, str]],
) -> PipelineSources:
    copies = _copied_sources(root)
    for (name, source), (_, destination) in zip(sources.items(), copies.items()):
        expected_hash = expected[name]["sha256"]
        if sha256_file(source) != expected_hash:
            raise RuntimeError(f"pipeline source changed before copy: {name}")
        if destination.exists():
            if not destination.is_file() or sha256_file(destination) != expected_hash:
                raise RuntimeError(f"immutable pipeline input copy differs: {name}")
        else:
            atomic_bytes(destination, source.read_bytes())
        if sha256_file(destination) != expected_hash:
            raise RuntimeError(f"pipeline input copy hash mismatch: {name}")
    return copies


def build_stage_plan(
    *,
    artifact_root: Path,
    python: str,
    gpus: Sequence[int],
    sources: PipelineSources,
) -> tuple[PipelineStage, ...]:
    """Build the exact argv arrays used by the five sequential child stages."""

    root = Path(artifact_root).expanduser().resolve()
    gpu_text = ",".join(str(value) for value in _validated_gpu_ids(gpus))
    python = str(python)
    selector_root = root / "selector"
    main_root = root / "main"
    primary_root = root / "primary_analysis"
    engineering_root = root / "engineering_benefit"
    association_root = root / "dynamics_utility_association"
    return (
        PipelineStage(
            "selector",
            "repro.sagodi_protocol.lr_selection_v3",
            selector_root,
            (
                python,
                "-m",
                "repro.sagodi_protocol.lr_selection_v3",
                "--selector",
                str(sources.selector),
                "--protocol",
                str(sources.protocol),
                "--artifact-root",
                str(selector_root),
                "--python",
                python,
                "--gpus",
                gpu_text,
            ),
        ),
        PipelineStage(
            "main",
            "repro.sagodi_protocol.primary_main_campaign",
            main_root,
            (
                python,
                "-m",
                "repro.sagodi_protocol.primary_main_campaign",
                "--selector-root",
                str(selector_root),
                "--template",
                str(sources.main_template),
                "--artifact-root",
                str(main_root),
                "--python",
                python,
                "--gpus",
                gpu_text,
            ),
        ),
        PipelineStage(
            "primary_analysis",
            "repro.sagodi_protocol.primary_analysis_campaign",
            primary_root,
            (
                python,
                "-m",
                "repro.sagodi_protocol.primary_analysis_campaign",
                "--main-root",
                str(main_root),
                "--artifact-root",
                str(primary_root),
                "--python",
                python,
                "--gpus",
                gpu_text,
            ),
        ),
        PipelineStage(
            "engineering_benefit",
            "repro.sagodi_protocol.engineering_benefit_campaign",
            engineering_root,
            (
                python,
                "-m",
                "repro.sagodi_protocol.engineering_benefit_campaign",
                "--main-root",
                str(main_root),
                "--selector-root",
                str(selector_root),
                "--freeze",
                str(sources.engineering_freeze),
                "--artifact-root",
                str(engineering_root),
                "--python",
                python,
                "--gpus",
                gpu_text,
            ),
        ),
        PipelineStage(
            "dynamics_utility_association",
            "repro.sagodi_protocol.dynamics_utility_association",
            association_root,
            (
                python,
                "-m",
                "repro.sagodi_protocol.dynamics_utility_association",
                "--primary-root",
                str(primary_root),
                "--engineering-root",
                str(engineering_root),
                "--main-root",
                str(main_root),
                "--selector-root",
                str(selector_root),
                "--freeze",
                str(sources.association_freeze),
                "--artifact-root",
                str(association_root),
            ),
        ),
    )


def build_dry_run_plan(
    *,
    artifact_root: Path,
    python: str,
    gpus: Sequence[int],
    sources: PipelineSources,
) -> dict[str, Any]:
    """Return a side-effect-free plan; no child command or git command is run."""

    root = Path(artifact_root).expanduser().resolve()
    resolved_sources = _resolve_sources(sources)
    resolved_python = _resolve_python(python)
    selected_gpus = _validated_gpu_ids(gpus)
    # A real run uses immutable copies.  The dry plan shows those exact paths
    # without creating them.
    stages = build_stage_plan(
        artifact_root=root,
        python=resolved_python,
        gpus=selected_gpus,
        sources=_copied_sources(root),
    )
    return {
        "schema_version": 1,
        "pipeline_id": PIPELINE_ID,
        "dry_run": True,
        "artifact_root": str(root),
        "python": resolved_python,
        "physical_gpu_ids": list(selected_gpus),
        "source_files": _source_payload(resolved_sources),
        "stage_order": list(STAGE_ORDER),
        "stages": [stage.payload(root) for stage in stages],
        "execution_performed": False,
        "clean_committed_worktree_check": "deferred_to_full_execution",
    }


def _pipeline_identity_payload(
    *,
    root: Path,
    repo_root: Path,
    resolved_python: str,
    selected_gpus: Sequence[int],
    source_payload: Mapping[str, Mapping[str, str]],
    git_state: Mapping[str, Any],
    stages: Sequence[PipelineStage],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline_id": PIPELINE_ID,
        "scope": PIPELINE_SCOPE,
        "artifact_root": str(root),
        "repository_root": str(repo_root),
        "code_commit": git_state["code_commit"],
        "worktree_dirty": False,
        "python": resolved_python,
        "physical_gpu_ids": list(selected_gpus),
        "source_files": dict(source_payload),
        "stage_order": list(STAGE_ORDER),
        "stages": [stage.payload(root) for stage in stages],
    }


def _prepare_root(root: Path, marker_payload: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ROOT_MARKER
    if marker.exists():
        if strict_json_load(marker) != dict(marker_payload):
            raise RuntimeError("pipeline artifact-root marker differs; choose a new root")
        return
    unexpected = [item.name for item in root.iterdir() if item.name != ROOT_MARKER]
    if unexpected:
        raise RuntimeError(
            "unmarked pipeline artifact root is not empty: " + ", ".join(sorted(unexpected))
        )
    atomic_json(marker, dict(marker_payload))


def _write_or_check_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    if path.exists():
        if strict_json_load(path) != dict(manifest):
            raise RuntimeError("immutable pipeline manifest differs")
    else:
        root = path.parent
        prior_execution_state = [
            root / STATUS,
            root / COMPLETE,
            root / COMPLETION_RECEIPT,
            root / LOG_DIRECTORY,
            *(root / name for name in STAGE_ORDER),
        ]
        if any(item.exists() for item in prior_execution_state):
            raise RuntimeError(
                "pipeline manifest is missing beside prior execution state; "
                "refusing to reconstruct provenance"
            )
        atomic_json(path, dict(manifest))


def _verify_frozen_inputs(
    *,
    manifest: Mapping[str, Any],
    sources: PipelineSources,
    copied_sources: PipelineSources,
    repo_root: Path,
) -> None:
    current_git = _git_state(repo_root)
    if current_git.get("worktree_dirty") is not False:
        raise RuntimeError("pipeline requires a clean committed worktree at every stage")
    if current_git.get("code_commit") != manifest.get("code_commit"):
        raise RuntimeError("repository commit changed during the pipeline")
    registered = manifest.get("source_files")
    if not isinstance(registered, Mapping):
        raise RuntimeError("pipeline source-file manifest is malformed")
    for (name, source), (_, copied) in zip(sources.items(), copied_sources.items()):
        item = registered.get(name)
        if not isinstance(item, Mapping):
            raise RuntimeError(f"pipeline source binding is missing: {name}")
        expected = item.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError(f"pipeline source hash is malformed: {name}")
        if sha256_file(source) != expected:
            raise RuntimeError(f"repository pipeline source changed: {name}")
        if not copied.is_file() or sha256_file(copied) != expected:
            raise RuntimeError(f"immutable pipeline source copy changed: {name}")


def _new_status(manifest: Mapping[str, Any], stages: Sequence[PipelineStage]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline_id": PIPELINE_ID,
        "pipeline_scientific_identity": manifest["scientific_identity"],
        "state": "pending",
        "current_stage": None,
        "stage_order": list(STAGE_ORDER),
        "stages": {
            stage.name: {
                "status": "pending",
                "command": list(stage.command),
                "command_display": shlex.join(stage.command),
                "attempts": [],
            }
            for stage in stages
        },
    }


def _load_or_initialize_status(
    path: Path, manifest: Mapping[str, Any], stages: Sequence[PipelineStage]
) -> dict[str, Any]:
    expected = _new_status(manifest, stages)
    if not path.exists():
        root = path.parent
        prior_execution_state = [
            root / COMPLETE,
            root / COMPLETION_RECEIPT,
            root / LOG_DIRECTORY,
            *(stage.artifact_root for stage in stages),
        ]
        if any(item.exists() for item in prior_execution_state):
            raise RuntimeError(
                "pipeline status is missing beside prior execution state; "
                "refusing to discard attempt history"
            )
        atomic_json(path, expected)
        return expected
    observed = strict_json_load(path)
    if not isinstance(observed, dict):
        raise RuntimeError("pipeline status is not a JSON object")
    for key in ("schema_version", "pipeline_id", "pipeline_scientific_identity", "stage_order"):
        if observed.get(key) != expected[key]:
            raise RuntimeError(f"pipeline status identity differs: {key}")
    stage_status = observed.get("stages")
    if not isinstance(stage_status, dict) or set(stage_status) != set(STAGE_ORDER):
        raise RuntimeError("pipeline status stage set differs")
    for stage in stages:
        item = stage_status.get(stage.name)
        if not isinstance(item, dict) or item.get("command") != list(stage.command):
            raise RuntimeError(f"pipeline status command differs: {stage.name}")
        if not isinstance(item.get("attempts"), list):
            raise RuntimeError(f"pipeline status attempts are malformed: {stage.name}")
    return observed


def _run_stage(
    *,
    root: Path,
    repo_root: Path,
    stage: PipelineStage,
    status: dict[str, Any],
) -> None:
    stage_status = status["stages"][stage.name]
    attempts = stage_status["attempts"]
    attempt_number = len(attempts) + 1
    log_relative = f"{LOG_DIRECTORY}/{stage.name}.attempt{attempt_number:03d}.log"
    log_path = root / log_relative
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempt: dict[str, Any] = {
        "attempt": attempt_number,
        "command": list(stage.command),
        "command_display": shlex.join(stage.command),
        "cwd": str(repo_root),
        "log_relative_to_pipeline": log_relative,
        "started_at_utc": _utc_now(),
        "started_at_unix": time.time(),
        "ended_at_utc": None,
        "ended_at_unix": None,
        "return_code": None,
    }
    attempts.append(attempt)
    stage_status["status"] = "running"
    status["state"] = "running"
    status["current_stage"] = stage.name
    atomic_json(root / STATUS, status)

    process: subprocess.Popen[Any] | None = None
    received_signal: list[int] = []
    previous_handlers: dict[int, Any] = {}

    def forward(signum: int, _frame: Any) -> None:
        received_signal.append(signum)
        if process is not None and process.poll() is None:
            process.send_signal(signum)

    return_code = -1
    launch_error: OSError | None = None
    try:
        with log_path.open("wb") as log_handle:
            header = (
                f"pipeline_stage={stage.name}\n"
                f"started_at_utc={attempt['started_at_utc']}\n"
                f"command={shlex.join(stage.command)}\n\n"
            ).encode("utf-8")
            log_handle.write(header)
            log_handle.flush()
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, forward)
            process = subprocess.Popen(
                stage.command,
                cwd=repo_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            return_code = int(process.wait())
    except OSError as exc:
        launch_error = exc
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        attempt["ended_at_utc"] = _utc_now()
        attempt["ended_at_unix"] = time.time()
        attempt["return_code"] = return_code
        if log_path.is_file():
            attempt["log_sha256"] = sha256_file(log_path)

    if launch_error is not None:
        stage_status["status"] = "failed"
        status["state"] = "failed"
        status["current_stage"] = stage.name
        attempt["launch_error"] = f"{type(launch_error).__name__}: {launch_error}"
        atomic_json(root / STATUS, status)
        raise PipelineStageError(
            stage.name,
            1,
            f"could not launch exact command: {launch_error}",
        ) from launch_error
    if received_signal:
        stage_status["status"] = "interrupted"
        status["state"] = "interrupted"
        status["current_stage"] = stage.name
        atomic_json(root / STATUS, status)
        signum = received_signal[-1]
        raise PipelineStageError(
            stage.name,
            128 + signum,
            f"received signal {signum}; child return code {return_code}",
        )
    if return_code != 0:
        stage_status["status"] = "failed"
        status["state"] = "failed"
        status["current_stage"] = stage.name
        atomic_json(root / STATUS, status)
        raise PipelineStageError(stage.name, return_code, f"return code {return_code}")

    valid, reason = verify_completion_receipt(stage.completion_receipt)
    if not stage.complete_marker.is_file() or not valid:
        stage_status["status"] = "failed"
        status["state"] = "failed"
        status["current_stage"] = stage.name
        atomic_json(root / STATUS, status)
        detail = "missing COMPLETE" if not stage.complete_marker.is_file() else reason
        raise PipelineStageError(
            stage.name,
            1,
            f"zero exit without a valid terminal receipt boundary: {detail}",
        )
    attempt["complete_sha256"] = sha256_file(stage.complete_marker)
    attempt["completion_receipt_sha256"] = sha256_file(stage.completion_receipt)
    stage_status["status"] = "complete"
    stage_status["last_successful_attempt"] = attempt_number
    status["current_stage"] = None
    atomic_json(root / STATUS, status)


def _finalize(
    root: Path, manifest: Mapping[str, Any], stages: Sequence[PipelineStage], status: dict[str, Any]
) -> None:
    terminal_children = {
        stage.name: {
            "complete_sha256": sha256_file(stage.complete_marker),
            "completion_receipt_sha256": sha256_file(stage.completion_receipt),
        }
        for stage in stages
    }
    complete = {
        "schema_version": 1,
        "status": "complete",
        "pipeline_id": PIPELINE_ID,
        "pipeline_scientific_identity": manifest["scientific_identity"],
        "code_commit": manifest["code_commit"],
        "stage_order": list(STAGE_ORDER),
        "terminal_children": terminal_children,
    }
    atomic_json(root / COMPLETE, complete)
    status["state"] = "complete"
    status["current_stage"] = None
    status["completed_at_utc"] = _utc_now()
    status["completed_at_unix"] = time.time()
    atomic_json(root / STATUS, status)
    receipt_metadata = {
        "pipeline_id": PIPELINE_ID,
        "pipeline_scientific_identity": manifest["scientific_identity"],
        "code_commit": manifest["code_commit"],
        "stage_count": len(STAGE_ORDER),
    }
    write_completion_receipt(
        root / COMPLETION_RECEIPT,
        job_id=f"{PIPELINE_ID}_complete",
        artifacts=[root / MANIFEST, root / STATUS, root / COMPLETE],
        metadata=receipt_metadata,
    )
    valid, reason = verify_completion_receipt(
        root / COMPLETION_RECEIPT,
        expected_job_id=f"{PIPELINE_ID}_complete",
        expected_metadata=receipt_metadata,
    )
    if not valid:
        raise RuntimeError(f"pipeline completion receipt failed verification: {reason}")


def run_primary_v3_pipeline(
    *,
    artifact_root: Path,
    python: str,
    gpus: Sequence[int],
    sources: PipelineSources,
) -> Path:
    """Run or resume the complete full-mode chain, stopping on first failure."""

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES; --gpus uses physical GPU ids")
    root = Path(artifact_root).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    resolved_sources = _resolve_sources(sources)
    source_payload = _source_payload(resolved_sources)
    resolved_python = _resolve_python(python)
    selected_gpus = _validated_gpu_ids(gpus)
    git_state = _git_state(repo_root)
    if git_state.get("worktree_dirty") is not False:
        raise RuntimeError("full v3 pipeline requires a clean committed worktree")
    commit = git_state.get("code_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("full v3 pipeline requires a valid 40-character git commit")

    copied_sources = _copied_sources(root)
    stages = build_stage_plan(
        artifact_root=root,
        python=resolved_python,
        gpus=selected_gpus,
        sources=copied_sources,
    )
    identity = _pipeline_identity_payload(
        root=root,
        repo_root=repo_root,
        resolved_python=resolved_python,
        selected_gpus=selected_gpus,
        source_payload=source_payload,
        git_state=git_state,
        stages=stages,
    )
    scientific_identity = canonical_hash(identity)
    marker_payload = {
        "schema_version": 1,
        "pipeline_id": PIPELINE_ID,
        "pipeline_scientific_identity": scientific_identity,
    }
    _prepare_root(root, marker_payload)
    copied_sources = _materialize_sources(
        root, resolved_sources, source_payload
    )
    manifest = dict(identity)
    manifest["scientific_identity_payload"] = identity
    manifest["scientific_identity"] = scientific_identity
    _write_or_check_manifest(root / MANIFEST, manifest)
    lock = _acquire_campaign_lock(root)
    try:
        status = _load_or_initialize_status(root / STATUS, manifest, stages)
        active_stage: str | None = None
        try:
            for stage in stages:
                active_stage = stage.name
                _verify_frozen_inputs(
                    manifest=manifest,
                    sources=resolved_sources,
                    copied_sources=copied_sources,
                    repo_root=repo_root,
                )
                _run_stage(root=root, repo_root=repo_root, stage=stage, status=status)
            active_stage = "final_verification"
            _verify_frozen_inputs(
                manifest=manifest,
                sources=resolved_sources,
                copied_sources=copied_sources,
                repo_root=repo_root,
            )
            _finalize(root, manifest, stages, status)
            return root
        except BaseException as exc:
            # _run_stage already records detailed child failures.  This outer
            # boundary additionally records failures that occur between child
            # invocations (for example a changed commit or frozen input).
            if status.get("state") not in {"failed", "interrupted"}:
                interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
                status["state"] = "interrupted" if interrupted else "failed"
                status["current_stage"] = active_stage
            status["last_failure"] = {
                "stage": active_stage,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "recorded_at_utc": _utc_now(),
                "recorded_at_unix": time.time(),
            }
            atomic_json(root / STATUS, status)
            raise
    finally:
        lock.release()


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpus must be comma-separated integers") from exc
    try:
        return _validated_gpu_ids(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", type=_parse_gpu_ids, default=_parse_gpu_ids("0,1,2,3,4,5"))
    parser.add_argument("--selector", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--main-template", type=Path, default=DEFAULT_MAIN_TEMPLATE)
    parser.add_argument(
        "--engineering-freeze", type=Path, default=DEFAULT_ENGINEERING_FREEZE
    )
    parser.add_argument(
        "--association-freeze", type=Path, default=DEFAULT_ASSOCIATION_FREEZE
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact plan without creating artifacts or executing a child stage",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    sources = PipelineSources(
        selector=args.selector,
        protocol=args.protocol,
        main_template=args.main_template,
        engineering_freeze=args.engineering_freeze,
        association_freeze=args.association_freeze,
    )
    if args.dry_run:
        plan = build_dry_run_plan(
            artifact_root=args.artifact_root,
            python=args.python,
            gpus=args.gpus,
            sources=sources,
        )
        print(json.dumps(plan, sort_keys=True), flush=True)
        return 0
    try:
        output = run_primary_v3_pipeline(
            artifact_root=args.artifact_root,
            python=args.python,
            gpus=args.gpus,
            sources=sources,
        )
    except PipelineStageError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "stage": exc.stage,
                    "return_code": exc.return_code,
                    "reason": exc.reason,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return exc.return_code if 0 < exc.return_code < 256 else 1
    print(
        json.dumps(
            {
                "status": "complete",
                "pipeline_id": PIPELINE_ID,
                "artifact_root": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
