"""Safe NPZ artifacts for manifold benchmark parent and derived banks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .generator import BANK_SCHEMA_VERSION, ManifoldBatch, ParentBank


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(int(chunk_size)):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    header = canonical_bytes({"shape": list(array.shape), "dtype": str(array.dtype)})
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sidecar(path: Path) -> Path:
    return Path(f"{path}.sha256")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: str | os.PathLike[str], value: Any) -> None:
    """Write canonical JSON atomically."""

    _atomic_bytes(Path(path), canonical_bytes(value) + b"\n")


def _save_archive(
    path: str | os.PathLike[str],
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    artifact_kind: str,
    overwrite: bool,
) -> str:
    destination = Path(path).expanduser().resolve()
    sidecar = _sidecar(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and (destination.exists() or sidecar.exists()):
        raise FileExistsError(f"bank or checksum already exists: {destination}")
    if not arrays or "metadata_json" in arrays:
        raise ValueError("arrays must be nonempty and cannot contain metadata_json")
    materialized = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    envelope = {
        "bank_schema_version": BANK_SCHEMA_VERSION,
        "artifact_kind": str(artifact_kind),
        "metadata": dict(metadata),
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": array_sha256(value),
            }
            for name, value in sorted(materialized.items())
        },
    }
    metadata_array = np.frombuffer(canonical_bytes(envelope), dtype=np.uint8)
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            # Random Gaussian arrays do not compress meaningfully.  An uncompressed
            # ZIP keeps creation and reload fast while remaining pickle-free.
            np.savez(handle, **materialized, metadata_json=metadata_array)
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, destination)
        _atomic_bytes(sidecar, f"{digest}  {destination.name}\n".encode("ascii"))
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return digest


def _expected_digest(path: Path) -> str:
    sidecar = _sidecar(path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing SHA-256 sidecar: {sidecar}")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise ValueError(f"malformed SHA-256 sidecar: {sidecar}")
    digest = fields[0].lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"malformed digest in {sidecar}")
    return digest


def _load_archive(
    path: str | os.PathLike[str], *, expected_kind: str
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    source = Path(path).expanduser().resolve(strict=True)
    expected = _expected_digest(source)
    actual = sha256_file(source)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {source}: expected {expected}, got {actual}")
    with np.load(source, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("bank has no metadata_json")
        envelope = json.loads(
            np.asarray(archive["metadata_json"], dtype=np.uint8).tobytes().decode("utf-8")
        )
        specifications = envelope.get("arrays")
        if not isinstance(specifications, dict):
            raise ValueError("bank metadata has no array specifications")
        expected_keys = set(specifications) | {"metadata_json"}
        if set(archive.files) != expected_keys:
            raise ValueError(
                f"bank keys differ from metadata: {sorted(archive.files)} vs "
                f"{sorted(expected_keys)}"
            )
        arrays = {name: np.array(archive[name], copy=True) for name in specifications}
    if envelope.get("bank_schema_version") != BANK_SCHEMA_VERSION:
        raise ValueError("unsupported bank schema version")
    if envelope.get("artifact_kind") != expected_kind:
        raise ValueError("bank artifact kind mismatch")
    for name, value in arrays.items():
        specification = specifications[name]
        if specification.get("shape") != list(value.shape):
            raise ValueError(f"shape metadata mismatch for {name}")
        if specification.get("dtype") != str(value.dtype):
            raise ValueError(f"dtype metadata mismatch for {name}")
        if specification.get("sha256") != array_sha256(value):
            raise ValueError(f"array digest mismatch for {name}")
    metadata = envelope.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("bank metadata payload must be an object")
    metadata = dict(metadata)
    metadata["archive_sha256"] = actual
    return arrays, metadata, actual


def save_parent_bank(
    path: str | os.PathLike[str], parent: ParentBank, *, overwrite: bool = False
) -> str:
    if not isinstance(parent, ParentBank):
        raise TypeError("parent must be ParentBank")
    arrays = {
        "white_noise": parent.white_noise,
        "id_base_drive": parent.id_base_drive,
        "q0_angles": parent.q0_angles,
        "q0_sphere_gaussian": parent.q0_sphere_gaussian,
        "sparsity_parameters": parent.sparsity_parameters,
        "mask_uniform_randoms": parent.mask_uniform_randoms,
        "trajectory_id": parent.trajectory_id,
    }
    metadata = dict(parent.metadata)
    metadata["white_noise_sha256"] = array_sha256(parent.white_noise)
    return _save_archive(
        path,
        arrays=arrays,
        metadata=metadata,
        artifact_kind="parent_stochastic_bank",
        overwrite=overwrite,
    )


def load_parent_bank(path: str | os.PathLike[str]) -> ParentBank:
    arrays, metadata, _ = _load_archive(path, expected_kind="parent_stochastic_bank")
    if metadata.get("white_noise_sha256") != array_sha256(arrays["white_noise"]):
        raise ValueError("parent white-noise identity mismatch")
    return ParentBank(metadata=metadata, **arrays)


def _batch_arrays(batch: ManifoldBatch) -> dict[str, np.ndarray]:
    arrays = {
        "initial_memory": batch.initial_memory,
        "inputs": batch.inputs,
        "output_targets": batch.output_targets,
        "latent_targets": batch.latent_targets,
        "latent_path": batch.latent_path,
        "base_drive": batch.base_drive,
        "effective_velocity": batch.effective_velocity,
        "dwell_mask": batch.dwell_mask,
        "trajectory_id": batch.trajectory_id,
        "mask": batch.mask,
    }
    if batch.latent_unwrapped is not None:
        arrays["latent_unwrapped"] = batch.latent_unwrapped
    return arrays


def save_manifold_bank(
    path: str | os.PathLike[str], batch: ManifoldBatch, *, overwrite: bool = False
) -> str:
    if not isinstance(batch, ManifoldBatch):
        raise TypeError("batch must be ManifoldBatch")
    if not batch.metadata.get("parent_bank_sha256"):
        raise ValueError("derived bank must bind a saved parent_bank_sha256")
    return _save_archive(
        path,
        arrays=_batch_arrays(batch),
        metadata=batch.metadata,
        artifact_kind="derived_condition_bank",
        overwrite=overwrite,
    )


def load_manifold_bank(path: str | os.PathLike[str]) -> ManifoldBatch:
    arrays, metadata, _ = _load_archive(path, expected_kind="derived_condition_bank")
    latent_unwrapped = arrays.pop("latent_unwrapped", None)
    return ManifoldBatch(metadata=metadata, latent_unwrapped=latent_unwrapped, **arrays)


__all__ = [
    "array_sha256",
    "atomic_json",
    "canonical_bytes",
    "load_manifold_bank",
    "load_parent_bank",
    "save_manifold_bank",
    "save_parent_bank",
    "sha256_file",
]
