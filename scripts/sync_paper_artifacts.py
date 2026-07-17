#!/usr/bin/env python3
"""Copy frozen experiment outputs into the canonical paper artifact tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "configs" / "paper_artifacts.json"
DEFAULT_EXPERIMENT_ROOT = Path("/home/biadmin/ca_rnn/experiments")
LOCK_PATH = REPOSITORY_ROOT / "paper" / "artifact_checksums.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path: {value}")
    return path


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("paper artifact manifest must use schema_version 1")
    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("paper artifact manifest has no groups")
    ids = [str(group["id"]) for group in groups]
    if len(ids) != len(set(ids)):
        raise ValueError("paper artifact group IDs must be unique")
    destinations: set[Path] = set()
    for group in groups:
        _safe_relative(str(group["source_root"]), "source_root")
        files = group.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"group {group['id']} has no files")
        for item in files:
            _safe_relative(str(item["source"]), "source")
            destination = _safe_relative(
                str(item["destination"]), "destination"
            )
            if destination in destinations:
                raise ValueError(f"duplicate destination: {destination}")
            destinations.add(destination)
    return payload


def _selected_groups(
    manifest: dict[str, Any], requested: list[str]
) -> list[dict[str, Any]]:
    groups = list(manifest["groups"])
    if not requested:
        return groups
    known = {str(group["id"]): group for group in groups}
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(f"unknown artifact groups: {', '.join(unknown)}")
    return [known[group_id] for group_id in requested]


def _resolve_experiment_root(
    manifest: dict[str, Any], argument: Path | None
) -> Path:
    if argument is not None:
        return argument.expanduser().resolve(strict=True)
    environment = str(manifest["experiment_root_environment"])
    configured = os.environ.get(environment)
    root = Path(configured) if configured else DEFAULT_EXPERIMENT_ROOT
    return root.expanduser().resolve(strict=True)


def _records(
    groups: list[dict[str, Any]], experiment_root: Path
) -> list[dict[str, Any]]:
    records = []
    for group in groups:
        source_root_relative = _safe_relative(
            str(group["source_root"]), "source_root"
        )
        source_root = experiment_root / source_root_relative
        for item in group["files"]:
            source_relative = _safe_relative(str(item["source"]), "source")
            destination_relative = _safe_relative(
                str(item["destination"]), "destination"
            )
            records.append(
                {
                    "group": str(group["id"]),
                    "generator": str(group["generator"]),
                    "source_root": source_root_relative.as_posix(),
                    "source_relative": source_relative.as_posix(),
                    "source": source_root / source_relative,
                    "destination_relative": destination_relative.as_posix(),
                    "destination": REPOSITORY_ROOT / destination_relative,
                }
            )
    return records


def _write_lock(records: list[dict[str, Any]]) -> None:
    artifacts = []
    for record in records:
        destination = record["destination"]
        artifacts.append(
            {
                "group": record["group"],
                "generator": record["generator"],
                "source_root": record["source_root"],
                "source": record["source_relative"],
                "destination": record["destination_relative"],
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    payload = {"schema_version": 1, "artifacts": artifacts}
    LOCK_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sync(
    records: list[dict[str, Any]],
    lock_records: list[dict[str, Any]],
) -> int:
    for record in records:
        source = record["source"]
        destination = record["destination"]
        if not source.is_file():
            raise FileNotFoundError(f"missing source artifact: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
        print(f"sync  {record['group']}: {record['destination_relative']}")
    _write_lock(lock_records)
    return 0


def _check(records: list[dict[str, Any]]) -> int:
    failures = []
    for record in records:
        source = record["source"]
        destination = record["destination"]
        if not source.is_file():
            failures.append(f"missing source: {source}")
            continue
        if not destination.is_file():
            failures.append(
                f"missing destination: {record['destination_relative']}"
            )
            continue
        if _sha256(source) != _sha256(destination):
            failures.append(
                f"content mismatch: {record['destination_relative']}"
            )
    if failures:
        for failure in failures:
            print(f"FAIL  {failure}", file=sys.stderr)
        return 1
    print(f"OK    {len(records)} paper artifacts match their frozen sources")
    return 0


def _list(groups: list[dict[str, Any]]) -> int:
    for group in groups:
        print(
            f"{group['id']}: {group['status']} · "
            f"{len(group['files'])} files · {group['generator']}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--group", action="append", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--list", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest.expanduser().resolve(strict=True))
    groups = _selected_groups(manifest, args.group)
    if args.list:
        return _list(groups)
    experiment_root = _resolve_experiment_root(
        manifest, args.experiment_root
    )
    records = _records(groups, experiment_root)
    if args.check:
        return _check(records)
    lock_records = _records(list(manifest["groups"]), experiment_root)
    return _sync(records, lock_records)


if __name__ == "__main__":
    raise SystemExit(main())
