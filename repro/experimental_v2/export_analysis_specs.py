#!/usr/bin/env python3
"""Export checkpoint-dynamics specs from legacy or P0 campaign artifacts.

The dynamics evaluator intentionally accepts one artifact root per run.  This
tool creates a matching portable spec for either the legacy artifact root or a
single P0 campaign directory; it never copies or modifies a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_LEGACY_TEMPLATE = HERE / "dynamics_specs.json"
SCHEMA_VERSION = 1
TASKS = {
    "ring_hold",
    "ring_integrate",
    "torus_hold",
    "torus_integrate",
    "complex_curve_hold",
    "complex_curve_integrate",
    "surface_hold",
    "surface_integrate",
}
SPEC_FIELDS = {"paper_model", "task", "tag", "result_dir", "checkpoint_dir"}


def parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique non-negative integers")
    return seeds


def safe_relative_path(value: str, field: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or path == Path(".") or ".." in path.parts:
        raise ValueError(f"{field} must be a non-empty relative path without '..': {value!r}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_specs(payload: dict[str, Any]) -> list[dict[str, str]]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"spec schema_version must equal {SCHEMA_VERSION}")
    raw = payload.get("checkpoints")
    if not isinstance(raw, list) or not raw:
        raise ValueError("spec must contain a non-empty checkpoints list")
    output = []
    seen = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != SPEC_FIELDS:
            raise ValueError(f"checkpoint spec {index} must have exactly {sorted(SPEC_FIELDS)}")
        spec = {name: str(item[name]) for name in SPEC_FIELDS}
        if spec["task"] not in TASKS:
            raise ValueError(f"unknown task in checkpoint spec {index}: {spec['task']!r}")
        spec["result_dir"] = safe_relative_path(spec["result_dir"], "result_dir").as_posix()
        spec["checkpoint_dir"] = safe_relative_path(
            spec["checkpoint_dir"], "checkpoint_dir"
        ).as_posix()
        key = tuple(spec[name] for name in sorted(SPEC_FIELDS))
        if key in seen:
            raise ValueError(f"duplicate checkpoint spec {index}: {key}")
        seen.add(key)
        output.append(spec)
    return output


def assert_within_root(root: Path, relative: Path, field: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes artifact root: {relative}") from exc
    return candidate


def legacy_payload(
    legacy_root: Path,
    template: Path = DEFAULT_LEGACY_TEMPLATE,
    seeds: Sequence[int] = (),
    require_artifacts: bool = False,
) -> tuple[dict[str, Any], Path, dict[str, list[int]]]:
    root = legacy_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise FileNotFoundError(f"legacy artifact root is not a directory: {root}")
    specs = validate_checkpoint_specs(load_json(template.expanduser().resolve(strict=True)))
    coverage: dict[str, list[int]] = {}
    for spec in specs:
        key = f"{spec['paper_model']}|{spec['task']}|{spec['tag']}"
        coverage[key] = list(map(int, seeds))
        if not require_artifacts:
            continue
        for seed in seeds:
            result = assert_within_root(
                root,
                Path(spec["result_dir"]) / f"{spec['task']}_{spec['tag']}_seed{seed}.json",
                "result",
            )
            checkpoint = assert_within_root(
                root,
                Path(spec["checkpoint_dir"])
                / f"exp88_{spec['task']}_{spec['tag']}_seed{seed}.pt",
                "checkpoint",
            )
            for path in (result, checkpoint):
                if not path.is_file():
                    raise FileNotFoundError(path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "description": "Checkpoint-dynamics specs exported for the legacy Exp88 artifact root.",
        "source": {
            "kind": "legacy-template",
            "name": template.name,
            "sha256": sha256_file(template),
        },
        "checkpoints": specs,
    }
    return payload, root, coverage


def _job_artifacts(job: dict[str, Any]) -> tuple[Path, Path, str]:
    task = str(job.get("task", ""))
    seed = job.get("seed")
    if task not in TASKS or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"invalid campaign job task/seed: {task!r}/{seed!r}")
    expected = job.get("expected")
    if not isinstance(expected, list):
        raise ValueError("campaign job expected field must be a list")
    paths = [safe_relative_path(str(value), "job expected path") for value in expected]
    results = [path for path in paths if path.parts[0] == "results" and path.suffix == ".json"]
    checkpoints = [
        path for path in paths if path.parts[0] == "checkpoints" and path.suffix == ".pt"
    ]
    if len(results) != 1 or len(checkpoints) != 1:
        raise ValueError("each campaign job must declare exactly one result JSON and checkpoint PT")
    result, checkpoint = results[0], checkpoints[0]
    prefix = f"{task}_"
    suffix = f"_seed{seed}.json"
    if not result.name.startswith(prefix) or not result.name.endswith(suffix):
        raise ValueError(f"result filename does not match job task/seed: {result}")
    tag = result.name[len(prefix) : -len(suffix)]
    if not tag:
        raise ValueError(f"empty tag in result path: {result}")
    expected_checkpoint = f"exp88_{task}_{tag}_seed{seed}.pt"
    if checkpoint.name != expected_checkpoint:
        raise ValueError(
            f"checkpoint filename {checkpoint.name!r} does not match {expected_checkpoint!r}"
        )
    return result, checkpoint, tag


def campaign_payload(
    campaign_manifest: Path,
    seeds: Sequence[int] = (),
    require_artifacts: bool = False,
) -> tuple[dict[str, Any], Path, dict[str, list[int]]]:
    manifest_path = campaign_manifest.expanduser().resolve(strict=True)
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"campaign schema_version must equal {SCHEMA_VERSION}")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("campaign manifest must contain a non-empty jobs list")
    root = manifest_path.parent.resolve(strict=True)
    grouped: dict[tuple[str, str, str, str, str], set[int]] = {}
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("campaign jobs must be JSON objects")
        if not bool(job.get("implemented", False)):
            continue
        result, checkpoint, tag = _job_artifacts(job)
        result_path = assert_within_root(root, result, "campaign result")
        checkpoint_path = assert_within_root(root, checkpoint, "campaign checkpoint")
        if require_artifacts:
            for path in (result_path, checkpoint_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
        condition = str(job.get("condition", "")).strip()
        if not condition:
            raise ValueError("implemented campaign job has no condition")
        task = str(job["task"])
        seed = int(job["seed"])
        key = (
            condition,
            task,
            tag,
            result.parent.as_posix(),
            checkpoint.parent.as_posix(),
        )
        group_seeds = grouped.setdefault(key, set())
        if seed in group_seeds:
            raise ValueError(f"duplicate campaign analysis job for {condition}/{task}/seed{seed}")
        group_seeds.add(seed)
    if not grouped:
        raise ValueError("campaign manifest contains no implemented analysis checkpoints")

    required_seeds = set(map(int, seeds))
    specs = []
    coverage = {}
    for key in sorted(grouped):
        paper_model, task, tag, result_dir, checkpoint_dir = key
        available = grouped[key]
        if required_seeds and not required_seeds.issubset(available):
            missing = sorted(required_seeds - available)
            raise ValueError(
                f"campaign spec {paper_model}/{task}/{tag} is missing manifest seeds {missing}"
            )
        specs.append(
            {
                "paper_model": paper_model,
                "task": task,
                "tag": tag,
                "result_dir": result_dir,
                "checkpoint_dir": checkpoint_dir,
            }
        )
        coverage[f"{paper_model}|{task}|{tag}"] = sorted(available)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Checkpoint-dynamics specs exported from campaign "
            f"{manifest.get('campaign_id', manifest_path.parent.name)}."
        ),
        "source": {
            "kind": "campaign-manifest",
            "campaign_id": str(manifest.get("campaign_id", manifest_path.parent.name)),
            "name": manifest_path.name,
            "sha256": sha256_file(manifest_path),
        },
        "checkpoints": specs,
    }
    validate_checkpoint_specs(payload)
    return payload, root, coverage


def atomic_write_new(path: Path, payload: dict[str, Any]) -> None:
    output = path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--legacy-root", type=Path)
    source.add_argument("--campaign-manifest", type=Path)
    parser.add_argument("--legacy-template", type=Path, default=DEFAULT_LEGACY_TEMPLATE)
    parser.add_argument("--seeds", default="", help="optional required seed coverage, e.g. 0,1,2")
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.seeds = parse_seeds(args.seeds)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.legacy_root is not None:
            payload, root, coverage = legacy_payload(
                args.legacy_root,
                template=args.legacy_template,
                seeds=args.seeds,
                require_artifacts=args.require_artifacts,
            )
            source_kind = "legacy"
        else:
            payload, root, coverage = campaign_payload(
                args.campaign_manifest,
                seeds=args.seeds,
                require_artifacts=args.require_artifacts,
            )
            source_kind = "campaign"
        atomic_write_new(args.output, payload)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "source_kind": source_kind,
                    "artifact_root": str(root),
                    "output": str(args.output.expanduser().resolve()),
                    "checkpoint_specs": len(payload["checkpoints"]),
                    "seed_coverage": coverage,
                },
                indent=2,
                sort_keys=True,
            )
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
