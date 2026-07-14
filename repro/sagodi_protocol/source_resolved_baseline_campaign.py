"""Fail-closed campaign for the three code-resolved Ságodi baselines.

The campaign deliberately does one thing before any LRU/CA-LRU search: train
one full 5,000-update sentinel for each released baseline recipe and require
all three final clean held-out MSEs to be below 0.01.  Child runs are delegated
to :mod:`source_resolved_worker`; this module owns only immutable planning,
multi-GPU scheduling, receipt verification, and the learning gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .artifacts import (
    atomic_json,
    canonical_hash,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from .source_resolved_protocol import (
    DEFAULT_SOURCE_CONFIG,
    SOURCE_CAMPAIGN_ID,
    SOURCE_MODEL_IDS,
    UPSTREAM_COMMIT,
    load_source_config,
    source_angular_integration,
)
from .tasks import load_fixed_bank, save_fixed_bank


MODULE_DIR = Path(__file__).resolve().parent
FREEZE_DOCUMENT = MODULE_DIR / "SAGODI_SOURCE_RESOLVED_V1_FREEZE_ko.md"
CAMPAIGN_ID = "sagodi_source_resolved_baseline_sentinel_v1"
SUCCESS_MSE_THRESHOLD = 0.01
SENTINEL_SEED = 100


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _native(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class BaselineRunSpec:
    run_id: str
    stage: str
    model_id: str
    model_seed: int
    updates: int
    batch_size: int
    evaluation_bank: str
    output_dir: str

    def payload(self) -> dict[str, Any]:
        return _native(asdict(self))


def build_plan(root: Path, bank: Path, *, smoke: bool) -> tuple[BaselineRunSpec, ...]:
    stage = "smoke" if smoke else "sentinel"
    seed = 999 if smoke else SENTINEL_SEED
    updates = 2 if smoke else 5000
    batch_size = 4 if smoke else 64
    return tuple(
        BaselineRunSpec(
            run_id=f"{stage}__{model_id}__seed{seed}",
            stage=stage,
            model_id=model_id,
            model_seed=seed,
            updates=updates,
            batch_size=batch_size,
            evaluation_bank=str(bank),
            output_dir=str(root / stage / "runs" / f"{model_id}__seed{seed}"),
        )
        for model_id in SOURCE_MODEL_IDS
    )


def summarize_gate(
    specs: Sequence[BaselineRunSpec],
    *,
    threshold: float = SUCCESS_MSE_THRESHOLD,
) -> dict[str, Any]:
    if {spec.model_id for spec in specs} != set(SOURCE_MODEL_IDS):
        raise ValueError("baseline gate requires exactly the three registered models")
    models: dict[str, Any] = {}
    failed: list[str] = []
    for spec in specs:
        result = strict_json_load(Path(spec.output_dir) / "result.json")
        if result.get("run_id") != spec.run_id or result.get("model_id") != spec.model_id:
            raise RuntimeError(f"baseline result identity mismatch: {spec.run_id}")
        mse = float(result["final_metrics"]["masked_mse"])
        passed = math.isfinite(mse) and mse < float(threshold)
        models[spec.model_id] = {
            "model_seed": spec.model_seed,
            "final_clean_heldout_mse": mse,
            "passed": passed,
        }
        if not passed:
            failed.append(spec.model_id)
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "metric": "final_clean_heldout_masked_mse",
        "success_mse_threshold": float(threshold),
        "models": models,
        "all_source_baselines_passed": not failed,
        "failed_models": failed,
    }


def _identity(config_path: Path) -> dict[str, Any]:
    files = (
        Path(__file__),
        MODULE_DIR / "source_resolved_worker.py",
        MODULE_DIR / "source_resolved_models.py",
        MODULE_DIR / "source_resolved_protocol.py",
        MODULE_DIR / "tasks.py",
        MODULE_DIR / "artifacts.py",
        FREEZE_DOCUMENT,
    )
    payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "upstream_commit": UPSTREAM_COMMIT,
        "config_sha256": sha256_file(config_path),
        "code_sha256": {path.name: sha256_file(path) for path in files},
    }
    payload["scientific_identity"] = canonical_hash(payload)
    return payload


def _copy_or_verify(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise RuntimeError(f"immutable campaign input differs: {destination}")
        return
    shutil.copy2(source, destination)


def _prepare_root(root: Path, config_source: Path) -> tuple[dict[str, Any], Path]:
    root = root.expanduser().resolve()
    config_source = config_source.expanduser().resolve(strict=True)
    config = load_source_config(config_source)
    identity = _identity(config_source)
    marker = root / ".sagodi_source_resolved_baseline_v1_root.json"
    root.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        if strict_json_load(marker) != identity:
            raise RuntimeError("artifact root belongs to another scientific identity")
    else:
        if any(root.iterdir()):
            raise RuntimeError("unmarked baseline artifact root is not empty")
        atomic_json(marker, identity)
    copied_config = root / "inputs" / DEFAULT_SOURCE_CONFIG.name
    _copy_or_verify(config_source, copied_config)
    _copy_or_verify(FREEZE_DOCUMENT, root / "inputs" / FREEZE_DOCUMENT.name)
    return config, copied_config


def _ensure_bank(root: Path, *, smoke: bool) -> Path:
    purpose = "smoke" if smoke else "sentinel"
    trials = 16 if smoke else 1024
    seed = 999 if smoke else 0
    bank = root / "banks" / f"{purpose}.npz"
    if not bank.exists():
        batch = source_angular_integration(
            trials,
            seed,
            stream_key=(CAMPAIGN_ID, purpose, "fixed_clean_bank"),
            device="cpu",
        )
        save_fixed_bank(bank, batch)
    observed = load_fixed_bank(bank)
    if observed.time_steps != 128 or observed.batch_size != trials:
        raise RuntimeError("source baseline evaluation bank shape differs")
    if observed.metadata.get("upstream_commit") != UPSTREAM_COMMIT:
        raise RuntimeError("source baseline evaluation bank commit differs")
    return bank


def _verified(spec: BaselineRunSpec, identity: Mapping[str, Any]) -> bool:
    output = Path(spec.output_dir)
    valid, _ = verify_completion_receipt(
        output / "completion_receipt.json",
        expected_job_id=spec.run_id,
        expected_metadata={
            "campaign_id": SOURCE_CAMPAIGN_ID,
            "upstream_commit": UPSTREAM_COMMIT,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
            "campaign_scientific_identity": identity["scientific_identity"],
        },
    )
    if not valid or not (output / "COMPLETE").is_file():
        return False
    try:
        manifest = strict_json_load(output / "run_manifest.json")
        result = strict_json_load(output / "result.json")
    except (OSError, ValueError):
        return False
    return (
        manifest.get("campaign_identity") == identity["scientific_identity"]
        and manifest.get("config_sha256") == identity["config_sha256"]
        and manifest.get("runtime_code_sha256")
        == {
            name: identity["code_sha256"][name]
            for name in (
                "source_resolved_worker.py",
                "source_resolved_models.py",
                "source_resolved_protocol.py",
                "tasks.py",
            )
        }
        and manifest.get("evaluation_bank_sha256") == sha256_file(spec.evaluation_bank)
        and result.get("run_id") == spec.run_id
        and result.get("model_id") == spec.model_id
        and int(result.get("model_seed", -1)) == spec.model_seed
        and int(result.get("updates_completed", -1)) == spec.updates
    )


def _archive_partial(output: Path, attempts: Path) -> None:
    if not output.exists():
        return
    attempts.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    os.replace(output, attempts / f"{output.name}.{stamp}")


def _write_plan(stage_root: Path, specs: Sequence[BaselineRunSpec]) -> None:
    payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "runs": [spec.payload() for spec in specs],
    }
    path = stage_root / "plan.json"
    if path.exists():
        if strict_json_load(path) != payload:
            raise RuntimeError("immutable source baseline plan differs")
    else:
        atomic_json(path, payload)


def _run_specs(
    specs: Sequence[BaselineRunSpec],
    *,
    config_path: Path,
    identity: Mapping[str, Any],
    compute_slots: Sequence[str],
) -> None:
    if not specs:
        return
    stage_root = Path(specs[0].output_dir).parents[1]
    _write_plan(stage_root, specs)
    pending = [spec for spec in specs if not _verified(spec, identity)]
    for spec in pending:
        _archive_partial(Path(spec.output_dir), stage_root / "attempts")
    logs = stage_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    queue = list(pending)
    running: dict[str, tuple[subprocess.Popen[Any], BaselineRunSpec, Any]] = {}
    repo = Path(__file__).resolve().parents[2]
    while queue or running:
        for slot in (item for item in compute_slots if item not in running):
            if not queue:
                break
            spec = queue.pop(0)
            output = Path(spec.output_dir)
            command = [
                sys.executable,
                "-m",
                "repro.sagodi_protocol.source_resolved_worker",
                "--run-id",
                spec.run_id,
                "--model-id",
                spec.model_id,
                "--model-seed",
                str(spec.model_seed),
                "--evaluation-bank",
                spec.evaluation_bank,
                "--output-dir",
                str(output),
                "--device",
                "cpu" if slot == "cpu" else "cuda:0",
                "--config",
                str(config_path),
                "--updates",
                str(spec.updates),
                "--batch-size",
                str(spec.batch_size),
                "--campaign-identity",
                str(identity["scientific_identity"]),
            ]
            environment = os.environ.copy()
            if slot != "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = slot
            handle = (logs / f"{spec.run_id}.log").open("ab")
            process = subprocess.Popen(
                command,
                cwd=repo,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[slot] = (process, spec, handle)
        if not running:
            continue
        time.sleep(0.2)
        for slot, (process, spec, handle) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del running[slot]
            if code != 0 or not _verified(spec, identity):
                for other, _, other_handle in running.values():
                    other.terminate()
                    other_handle.close()
                raise RuntimeError(f"source baseline child failed ({code}): {spec.run_id}")
        atomic_json(
            stage_root / "status.json",
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "registered": len(specs),
                "verified_complete": sum(_verified(spec, identity) for spec in specs),
                "pending": len(queue),
                "running": [item[1].run_id for item in running.values()],
                "updated_at_utc": _utc_now(),
            },
        )


def _parse_slots(text: str) -> tuple[str, ...]:
    normalized = str(text).strip().lower()
    if normalized == "cpu":
        return ("cpu",)
    values = tuple(item.strip() for item in normalized.split(",") if item.strip())
    if not values or any(not item.isdigit() for item in values):
        raise ValueError("--gpus must be comma-separated physical ids or 'cpu'")
    if len(set(values)) != len(values):
        raise ValueError("--gpus contains duplicate ids")
    return values


def run_campaign(
    *,
    stage: str,
    artifact_root: Path,
    config_source: Path,
    compute_slots: Sequence[str],
) -> Path:
    if stage not in {"smoke", "sentinel"}:
        raise ValueError("stage must be smoke or sentinel")
    _, copied_config = _prepare_root(artifact_root, config_source)
    root = artifact_root.expanduser().resolve()
    marker = strict_json_load(root / ".sagodi_source_resolved_baseline_v1_root.json")
    smoke = stage == "smoke"
    bank = _ensure_bank(root, smoke=smoke)
    specs = build_plan(root, bank, smoke=smoke)
    _run_specs(
        specs,
        config_path=copied_config,
        identity=marker,
        compute_slots=compute_slots,
    )
    stage_root = root / stage
    execution_complete = stage_root / "EXECUTION_COMPLETE"
    atomic_json(
        execution_complete,
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "stage": stage,
            "verified_run_count": len(specs),
            "completed_at_utc": _utc_now(),
        },
    )
    extra: list[Path] = [stage_root / "plan.json", execution_complete]
    if not smoke:
        summary = summarize_gate(specs)
        summary_path = stage_root / "gate_summary.json"
        atomic_json(summary_path, summary)
        extra.append(summary_path)
    write_completion_receipt(
        stage_root / "completion_receipt.json",
        job_id=f"{CAMPAIGN_ID}__{stage}",
        artifacts=extra,
        metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": stage,
            "scientific_identity": marker["scientific_identity"],
        },
    )
    if not smoke and not summary["all_source_baselines_passed"]:
        raise RuntimeError(
            "source baseline gate failed for: " + ", ".join(summary["failed_models"])
        )
    if not smoke:
        atomic_json(
            stage_root / "GATE_PASS",
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "success_mse_threshold": SUCCESS_MSE_THRESHOLD,
            },
        )
    return stage_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "sentinel"))
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--gpus", default="0,1,2")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    slots = _parse_slots(args.gpus)
    if slots != ("cpu",) and not torch.cuda.is_available():
        raise RuntimeError("CUDA ids requested but torch.cuda is unavailable")
    path = run_campaign(
        stage=args.stage,
        artifact_root=args.artifact_root,
        config_source=args.config,
        compute_slots=slots,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
