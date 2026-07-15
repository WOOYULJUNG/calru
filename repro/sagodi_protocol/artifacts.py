"""Small, dependency-free provenance and atomic-write helpers.

Completion receipts are intentionally relocatable.  A child writes into an
attempt directory and the orchestrator atomically renames that directory only
after the receipt has been verified.  Schema-2 receipts therefore store paths
relative to the receipt directory; schema-1 absolute-path receipts remain
readable for old exploratory artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


RECEIPT_IDENTITY_ENV = {
    "campaign_scientific_identity": "CALRU_CAMPAIGN_SCIENTIFIC_IDENTITY",
    "protocol_fingerprint": "CALRU_PROTOCOL_FINGERPRINT",
    "run_id": "CALRU_RUN_ID",
    "stage": "CALRU_STAGE",
}


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous objects instead of silently keeping the last key."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def strict_json_loads(payload: str | bytes | bytearray) -> Any:
    """Parse JSON while rejecting non-finite constants and duplicate object keys."""

    return json.loads(
        payload,
        parse_constant=_reject_nonfinite_constant,
        object_pairs_hook=_unique_json_object,
    )


def strict_json_load(path: Path | str) -> Any:
    return strict_json_loads(Path(path).read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def canonical_tensor_mapping_sha256(tensors: Mapping[str, Any]) -> str:
    """Hash named dense tensors independent of mapping and serialization order."""

    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        try:
            value = tensor.detach().cpu().contiguous()
            raw = value.numpy().tobytes(order="C")
            header = canonical_bytes(
                {
                    "name": str(name),
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "byte_length": len(raw),
                }
            )
        except (AttributeError, TypeError, RuntimeError) as error:
            raise TypeError(f"tensor mapping value {name!r} is not a supported dense tensor") from error
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(raw)
    return digest.hexdigest()


def derived_seed(base: int, *parts: Any) -> int:
    payload = canonical_bytes([int(base), *parts])
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path | str, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path | str, payload: Any) -> None:
    atomic_bytes(Path(path), canonical_bytes(payload) + b"\n")


def write_completion_receipt(
    path: Path | str,
    *,
    job_id: str,
    artifacts: Iterable[Path],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    receipt_path = Path(path).resolve()
    receipt_parent = receipt_path.parent
    artifact_hashes: dict[str, str] = {}
    for item in sorted(Path(value).resolve() for value in artifacts):
        try:
            relative = item.relative_to(receipt_parent)
        except ValueError as exc:
            raise ValueError(
                f"receipt artifact must be inside its output directory: {item}"
            ) from exc
        artifact_hashes[relative.as_posix()] = sha256_file(item)

    receipt_metadata = dict(metadata or {})
    inherited = {
        key: os.environ[environment]
        for key, environment in RECEIPT_IDENTITY_ENV.items()
        if environment in os.environ
    }
    if inherited and len(inherited) != len(RECEIPT_IDENTITY_ENV):
        missing = sorted(set(RECEIPT_IDENTITY_ENV).difference(inherited))
        raise RuntimeError(f"incomplete campaign receipt identity environment: {missing}")
    for key, value in inherited.items():
        if key in receipt_metadata and receipt_metadata[key] != value:
            raise ValueError(f"receipt metadata conflicts with campaign identity: {key}")
        receipt_metadata[key] = value
    receipt = {
        "schema_version": 2,
        "job_id": str(job_id),
        "artifact_paths_relative_to": "receipt_parent",
        "artifacts": artifact_hashes,
        "metadata": receipt_metadata,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def verify_completion_receipt(
    path: Path | str,
    *,
    expected_job_id: str | None = None,
    expected_metadata: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    receipt_path = Path(path)
    if not receipt_path.is_file():
        return False, "missing receipt"
    try:
        payload = strict_json_load(receipt_path)
        schema = int(payload.get("schema_version", -1))
        if schema not in (1, 2):
            return False, "unsupported schema"
        if expected_job_id is not None and payload.get("job_id") != str(expected_job_id):
            return False, (
                f"job_id mismatch: expected {expected_job_id!r}, "
                f"got {payload.get('job_id')!r}"
            )
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return False, "receipt metadata is not an object"
        for key, expected in (expected_metadata or {}).items():
            if metadata.get(key) != expected:
                return False, (
                    f"metadata mismatch for {key}: expected {expected!r}, "
                    f"got {metadata.get(key)!r}"
                )
        if not isinstance(payload.get("artifacts"), dict) or not payload["artifacts"]:
            return False, "receipt artifacts must be a nonempty object"
        for raw_path, expected in payload["artifacts"].items():
            artifact = Path(raw_path)
            if schema == 2:
                if artifact.is_absolute():
                    return False, f"schema-2 artifact path is absolute: {artifact}"
                artifact = (receipt_path.parent / artifact).resolve()
                try:
                    artifact.relative_to(receipt_path.parent.resolve())
                except ValueError:
                    return False, f"artifact path escapes receipt directory: {raw_path}"
            if not artifact.is_file():
                return False, f"missing artifact: {artifact}"
            if sha256_file(artifact) != expected:
                return False, f"hash mismatch: {artifact}"
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return False, str(exc)
    return True, "ok"
