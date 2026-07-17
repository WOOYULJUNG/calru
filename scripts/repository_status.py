#!/usr/bin/env python3
"""Print and validate the canonical CA-LRU repository index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "configs" / "paper_artifacts.json"
REQUIRED_DOCUMENTS = (
    "README.md",
    "docs/PROJECT_STATUS.md",
    "docs/CODE_MAP.md",
    "docs/EXPERIMENT_CATALOG.md",
    "docs/REPOSITORY_WORKFLOW.md",
    "paper/README.md",
    "paper/figures/README.md",
    "paper/evidence/topology/README.md",
    "repro/README.md",
)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("paper artifact manifest must use schema_version 1")
    if not isinstance(value.get("groups"), list):
        raise ValueError("paper artifact manifest groups must be a list")
    return value


def _category(status: str) -> str:
    for prefix, label in (
        ("current", "CURRENT"),
        ("auxiliary", "AUXILIARY"),
        ("archived", "ARCHIVED"),
    ):
        if status.startswith(prefix):
            return label
    return "OTHER"


def validate_repository(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    group_ids: set[str] = set()
    destinations: set[str] = set()
    for index, group in enumerate(manifest["groups"]):
        if not isinstance(group, dict):
            errors.append(f"group {index} is not an object")
            continue
        group_id = str(group.get("id", ""))
        if not group_id:
            errors.append(f"group {index} has no id")
        elif group_id in group_ids:
            errors.append(f"duplicate group id: {group_id}")
        group_ids.add(group_id)
        status = str(group.get("status", ""))
        if _category(status) == "OTHER":
            errors.append(f"{group_id}: unclassified status {status!r}")
        source_root = Path(str(group.get("source_root", "")))
        if not str(source_root) or source_root.is_absolute():
            errors.append(f"{group_id}: source_root must be relative")
        files = group.get("files")
        if not isinstance(files, list) or not files:
            errors.append(f"{group_id}: files must be a nonempty list")
            continue
        for mapping in files:
            if not isinstance(mapping, dict):
                errors.append(f"{group_id}: file mapping is not an object")
                continue
            source = Path(str(mapping.get("source", "")))
            destination_text = str(mapping.get("destination", ""))
            destination = Path(destination_text)
            if not str(source) or source.is_absolute():
                errors.append(f"{group_id}: source must be relative: {source}")
            if (
                not destination_text
                or destination.is_absolute()
                or not destination.parts
                or destination.parts[0] != "paper"
            ):
                errors.append(
                    f"{group_id}: destination must be a relative paper/ path: "
                    f"{destination_text!r}"
                )
            if destination_text in destinations:
                errors.append(f"duplicate artifact destination: {destination_text}")
            destinations.add(destination_text)
            if destination_text and not (REPOSITORY_ROOT / destination).is_file():
                errors.append(f"missing committed artifact: {destination_text}")
    for relative in REQUIRED_DOCUMENTS:
        document = REPOSITORY_ROOT / relative
        if not document.is_file():
            errors.append(f"missing canonical document: {relative}")
            continue
        for target in MARKDOWN_LINK.findall(
            document.read_text(encoding="utf-8")
        ):
            cleaned = target.strip().strip("<>")
            if (
                not cleaned
                or cleaned.startswith(("http://", "https://", "mailto:", "#"))
            ):
                continue
            path_text = cleaned.split("#", 1)[0]
            candidate = (document.parent / path_text).resolve()
            try:
                candidate.relative_to(REPOSITORY_ROOT)
            except ValueError:
                errors.append(f"{relative}: link escapes repository: {target}")
                continue
            if not candidate.exists():
                errors.append(f"{relative}: broken local link: {target}")
    return errors


def print_status(manifest: dict[str, Any]) -> None:
    grouped: dict[str, list[tuple[str, str]]] = {
        "CURRENT": [],
        "AUXILIARY": [],
        "ARCHIVED": [],
        "OTHER": [],
    }
    for group in manifest["groups"]:
        status = str(group["status"])
        grouped[_category(status)].append((str(group["id"]), status))
    print("CA-LRU repository status")
    for category in ("CURRENT", "AUXILIARY", "ARCHIVED", "OTHER"):
        if not grouped[category]:
            continue
        print(f"\n{category}")
        for group_id, status in grouped[category]:
            print(f"  {group_id:<34} {status}")
    print("\nCanonical docs")
    for relative in REQUIRED_DOCUMENTS:
        print(f"  {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = _load_manifest(args.manifest.resolve(strict=True))
    errors = validate_repository(manifest)
    if args.check:
        if errors:
            for error in errors:
                print(f"ERROR {error}")
            return 1
        print(
            f"OK    {len(manifest['groups'])} artifact groups and "
            f"{len(REQUIRED_DOCUMENTS)} canonical documents"
        )
        return 0
    print_status(manifest)
    if errors:
        print(f"\nWARNING: {len(errors)} repository-layout issue(s); run make check-layout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
