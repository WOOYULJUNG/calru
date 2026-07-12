#!/usr/bin/env python3
"""Build deterministic provenance records for the immutable raw snapshot.

The directory layout below ``paper/evidence/raw`` mirrors the source layout in
the legacy ``ca_rnn`` workspace.  Consequently, the raw-root-relative path is
also the legacy workspace-relative source path.  The generated manifest keeps
both that origin and the new repository-relative location explicit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_ROOT = REPOSITORY_ROOT / "paper" / "evidence" / "raw"
FILE_MANIFEST_PATH = RAW_DATA_ROOT / "file_manifest.csv"
CHECKSUM_PATH = RAW_DATA_ROOT / "checksums.sha256"
GENERATED_PATHS = frozenset(
    {
        FILE_MANIFEST_PATH,
        CHECKSUM_PATH,
    }
)
FILE_MANIFEST_COLUMNS = (
    "repo_relative_path",
    "bytes",
    "sha256",
    "legacy_source_path",
)


@dataclass(frozen=True)
class RawFileRecord:
    """A content-addressed raw file and its old and new relative paths."""

    repo_relative_path: str
    bytes: int
    sha256: str
    legacy_source_path: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of *path* without loading it whole."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def raw_input_paths() -> list[Path]:
    """Return provenance inputs in deterministic repository-relative order."""

    paths = [
        path
        for path in RAW_DATA_ROOT.rglob("*")
        if path.is_file() and path not in GENERATED_PATHS
    ]
    return sorted(
        paths,
        key=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix(),
    )


def build_records() -> list[RawFileRecord]:
    """Hash every raw input and return its deterministic provenance record."""

    records: list[RawFileRecord] = []
    for raw_path in raw_input_paths():
        records.append(
            RawFileRecord(
                repo_relative_path=raw_path.relative_to(REPOSITORY_ROOT).as_posix(),
                bytes=raw_path.stat().st_size,
                sha256=sha256_file(raw_path),
                legacy_source_path=raw_path.relative_to(RAW_DATA_ROOT).as_posix(),
            )
        )
    return records


def render_file_manifest(records: Sequence[RawFileRecord]) -> str:
    """Render the CSV manifest with stable columns and Unix line endings."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=FILE_MANIFEST_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                "repo_relative_path": record.repo_relative_path,
                "bytes": record.bytes,
                "sha256": record.sha256,
                "legacy_source_path": record.legacy_source_path,
            }
        )
    return output.getvalue()


def render_checksums(records: Sequence[RawFileRecord]) -> str:
    """Render sha256sum-compatible lines using repository-relative paths."""

    return "".join(
        f"{record.sha256}  {record.repo_relative_path}\n" for record in records
    )


def check_current(path: Path, expected_content: str) -> bool:
    """Return whether an existing UTF-8 file exactly matches generated content."""

    return path.is_file() and path.read_text(encoding="utf-8") == expected_content


def write_outputs(file_manifest: str, checksums: str) -> None:
    """Write both generated provenance files as UTF-8 with stable newlines."""

    for output_path, content in (
        (FILE_MANIFEST_PATH, file_manifest),
        (CHECKSUM_PATH, checksums),
    ):
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if either committed provenance file is missing or stale",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = build_records()
    file_manifest = render_file_manifest(records)
    checksums = render_checksums(records)

    if args.check:
        stale_paths = [
            path.relative_to(REPOSITORY_ROOT).as_posix()
            for path, expected_content in (
                (FILE_MANIFEST_PATH, file_manifest),
                (CHECKSUM_PATH, checksums),
            )
            if not check_current(path, expected_content)
        ]
        if stale_paths:
            print("Stale raw provenance files:")
            for stale_path in stale_paths:
                print(f"  {stale_path}")
            return 1
        print(f"Raw provenance is current for {len(records)} files.")
        return 0

    write_outputs(file_manifest, checksums)
    print(f"Wrote raw provenance for {len(records)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
