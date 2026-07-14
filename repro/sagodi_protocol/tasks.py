"""Deterministic task generators and fixed banks for the Ságodi protocol.

The generators in this module deliberately have no dependency on the legacy
Exp88 task code.  Every returned tensor is time-major (``time, batch,
feature``), and every stochastic draw is derived from an explicit, stable key.

The primary conventions frozen here are:

* memory-guided saccade has 512 steps and integer delays 50--399;
* angular integration has 256 velocity steps, ``dt=0.1``, and GP length scale
  one on ``linspace(-1, 1, T)``;
* a velocity token at step ``t`` is supervised against the post-update latent
  and output at ``t + 1``;
* cue-driven integration prepends one masked cue token, while hidden-init does
  not alter the 256 loss-bearing velocity tokens.

Fixed banks are stored as safe, pickle-free ``.npz`` files.  Their canonical
JSON metadata is embedded in the archive, and a mandatory SHA-256 sidecar is
verified before loading.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .config import AngularTaskSpec


TASK_VERSION = "sagodi-protocol-v1"
BANK_SCHEMA_VERSION = 1
MGS_HORIZON = 512
ANGULAR_HORIZON = 256
ANGULAR_DT = 0.1
GP_LENGTH_SCALE = 1.0
GP_STD = 1.0
GP_CHOLESKY_JITTER = 1e-6


def _freeze(value: Any) -> Any:
    """Recursively convert metadata containers to immutable equivalents."""

    if isinstance(value, Mapping):
        frozen = {str(key): _freeze(item) for key, item in value.items()}
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return _freeze(value.tolist())
    if isinstance(value, torch.Tensor):
        return _freeze(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"metadata value is not JSON-compatible: {type(value).__name__}")


def _thaw(value: Any) -> Any:
    """Convert frozen metadata into canonical-JSON-compatible containers."""

    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"metadata value is not JSON-compatible: {type(value).__name__}")


def metadata_payload(value: Any) -> Any:
    """Return immutable task metadata as ordinary JSON-compatible objects."""

    return _thaw(value)


@dataclass(frozen=True)
class Batch:
    """An immutable task-batch record with time-major tensors.

    Structural fields and metadata are immutable.  PyTorch tensors themselves
    remain ordinary tensors, so callers should treat them as read-only assets.
    All four tensors have shape ``[time, batch, feature]``; ``mask`` exactly
    matches ``output_targets``.
    """

    inputs: torch.Tensor
    output_targets: torch.Tensor
    latent_targets: torch.Tensor
    mask: torch.Tensor
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        names = ("inputs", "output_targets", "latent_targets", "mask")
        tensors = tuple(getattr(self, name) for name in names)
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("Batch array fields must be torch.Tensor instances")
        for name, value in zip(names, tensors):
            if value.ndim != 3:
                raise ValueError(f"{name} must be time-major rank 3, got {tuple(value.shape)}")
        time_batch = self.inputs.shape[:2]
        for name, value in zip(names[1:], tensors[1:]):
            if value.shape[:2] != time_batch:
                raise ValueError(
                    f"{name} time/batch dimensions {tuple(value.shape[:2])} "
                    f"do not match inputs {tuple(time_batch)}"
                )
        if self.mask.shape != self.output_targets.shape:
            raise ValueError("mask must have exactly the output_targets shape")
        if not all(value.is_floating_point() for value in tensors):
            raise TypeError("Batch tensors must use floating-point dtypes")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @property
    def time_steps(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.inputs.shape[1])

    @property
    def initial_memory(self) -> torch.Tensor | None:
        """Return the canonical ``Phi_d(q0)`` memory, shaped ``[B, 2d]``.

        Protocol generators record ``q0`` in immutable metadata.  Keeping the
        canonical memory as a derived property avoids duplicating a large
        tensor in fixed banks.  A custom Batch without ``initial_latents`` may
        legitimately return ``None``.
        """

        initial = self.metadata.get("initial_latents")
        if initial is None:
            return None
        q0 = torch.as_tensor(initial, dtype=self.inputs.dtype, device=self.inputs.device)
        if q0.ndim != 2 or q0.shape[0] != self.batch_size:
            raise ValueError("metadata initial_latents must have shape [batch, latent_dimension]")
        embedded = torch.empty(
            q0.shape[0], 2 * q0.shape[1], dtype=q0.dtype, device=q0.device
        )
        embedded[:, 0::2] = torch.cos(q0)
        embedded[:, 1::2] = torch.sin(q0)
        return embedded


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _thaw(_freeze(value)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def keyed_seed(base_seed: int, *keys: Any) -> int:
    """Derive a stable 63-bit seed from a base seed and typed JSON keys.

    Python's randomized ``hash`` is intentionally not used.  The same inputs
    therefore identify the same data stream across processes and machines.
    """

    if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)):
        raise TypeError("base_seed must be an integer")
    if int(base_seed) < 0:
        raise ValueError("base_seed must be non-negative")
    payload = {
        "namespace": "calru-sagodi-keyed-seed-v1",
        "base_seed": int(base_seed),
        "keys": list(keys),
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def _stream_key(value: Sequence[Any] | Any | None) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value.decode("utf-8") if isinstance(value, bytes) else value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _torch_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("task generators support torch.float32 or torch.float64")
    return dtype


def _tensor(array: np.ndarray, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    return torch.as_tensor(array, dtype=_torch_dtype(dtype), device=device).contiguous()


def _wrap_angles(values: np.ndarray) -> np.ndarray:
    return np.remainder(values + math.pi, 2.0 * math.pi) - math.pi


def _torus_embedding(q: np.ndarray) -> np.ndarray:
    """Interleaved canonical embedding [cos q1, sin q1, ...]."""

    output = np.empty((*q.shape[:-1], 2 * q.shape[-1]), dtype=q.dtype)
    output[..., 0::2] = np.cos(q)
    output[..., 1::2] = np.sin(q)
    return output


def memory_guided_saccade(
    batch_size: int,
    base_seed: int,
    *,
    stream_key: Sequence[Any] | Any | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Batch:
    """Generate the frozen Ságodi memory-guided-saccade task.

    Each 512-step trial contains 5 blank, 5 target-cue, 50--399 delay,
    5 go-cue, 5 masked transition, and a response period.  Pre-response
    supervised targets are zero, matching the paper generator convention.
    """

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    keys = _stream_key(stream_key)
    derived = keyed_seed(base_seed, "memory_guided_saccade", TASK_VERSION, *keys)
    rng = np.random.default_rng(derived)
    batch = int(batch_size)
    theta = rng.uniform(0.0, 2.0 * math.pi, size=(batch, 1))
    delays = rng.integers(50, 400, size=batch, endpoint=False, dtype=np.int64)
    cue = _torus_embedding(theta).reshape(batch, 2)

    inputs = np.zeros((MGS_HORIZON, batch, 3), dtype=np.float64)
    targets = np.zeros((MGS_HORIZON, batch, 2), dtype=np.float64)
    latent = np.broadcast_to(theta[None, :, :], (MGS_HORIZON, batch, 1)).copy()
    mask = np.ones_like(targets)
    response_starts: list[int] = []
    for sample, delay in enumerate(delays.tolist()):
        inputs[5:10, sample, :2] = cue[sample]
        go_start = 10 + int(delay)
        inputs[go_start : go_start + 5, sample, 2] = 1.0
        transition_start = go_start + 5
        response_start = transition_start + 5
        mask[transition_start:response_start, sample, :] = 0.0
        targets[response_start:, sample, :] = cue[sample]
        response_starts.append(response_start)

    metadata = {
        "task_name": "memory_guided_saccade",
        "task_version": TASK_VERSION,
        "base_seed": int(base_seed),
        "derived_seed": int(derived),
        "stream_key": keys,
        "horizon": MGS_HORIZON,
        "input_dimension": 3,
        "output_dimension": 2,
        "latent_dimension": 1,
        "initial_blank_steps": 5,
        "target_cue_steps": 5,
        "delay_support": (50, 399),
        "go_cue_steps": 5,
        "response_transition_steps": 5,
        "delays": tuple(int(value) for value in delays),
        "response_starts": tuple(response_starts),
        "initial_latents": theta,
        "target_indexing": "response_only_after_masked_transition",
    }
    return Batch(
        inputs=_tensor(inputs, dtype=dtype, device=device),
        output_targets=_tensor(targets, dtype=dtype, device=device),
        latent_targets=_tensor(latent, dtype=dtype, device=device),
        mask=_tensor(mask, dtype=dtype, device=device),
        metadata=metadata,
    )


@lru_cache(maxsize=32)
def _gp_cholesky(
    horizon: int,
    length_scale: float,
    gp_std: float,
    jitter: float,
    grid_start: float,
    grid_stop: float,
    grid_endpoint: bool,
) -> np.ndarray:
    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    if float(length_scale) <= 0.0 or float(gp_std) <= 0.0:
        raise ValueError("GP length scale and standard deviation must be positive")
    if float(jitter) <= 0.0:
        raise ValueError("GP Cholesky jitter must be positive")
    if not math.isfinite(float(grid_start)) or not math.isfinite(float(grid_stop)):
        raise ValueError("GP grid bounds must be finite")
    if float(grid_start) >= float(grid_stop):
        raise ValueError("GP grid start must be smaller than grid stop")
    grid = np.linspace(
        float(grid_start),
        float(grid_stop),
        int(horizon),
        endpoint=bool(grid_endpoint),
        dtype=np.float64,
    )
    delta = grid[:, None] - grid[None, :]
    covariance = float(gp_std) ** 2 * np.exp(
        -(delta**2) / (2.0 * float(length_scale) ** 2)
    )
    covariance.flat[:: int(horizon) + 1] += float(jitter)
    factor = np.linalg.cholesky(covariance)
    factor.setflags(write=False)
    return factor


def _normalize_init_mode(init_mode: str) -> str:
    key = str(init_mode).strip().lower().replace("_", "-")
    aliases = {
        "hidden-init": "hidden-init",
        "hidden": "hidden-init",
        "cue-driven": "cue-driven",
        "cue": "cue-driven",
    }
    if key not in aliases:
        raise ValueError("init_mode must be 'hidden-init' or 'cue-driven'")
    return aliases[key]


def angular_integration(
    batch_size: int,
    base_seed: int,
    *,
    dimensions: int = 1,
    init_mode: str = "hidden-init",
    horizon: int = ANGULAR_HORIZON,
    dt: float = ANGULAR_DT,
    gp_length_scale: float = GP_LENGTH_SCALE,
    gp_std: float = GP_STD,
    gp_jitter: float = GP_CHOLESKY_JITTER,
    gp_grid_start: float = -1.0,
    gp_grid_stop: float = 1.0,
    gp_grid_endpoint: bool = True,
    q0_distribution: str = "uniform_minus_pi_pi",
    q0_low: float = -math.pi,
    q0_high: float = math.pi,
    q0_high_inclusive: bool = False,
    target_indexing: str = "q_t_plus_1_after_velocity_update",
    loss_mask_mode: str = "all_256_velocity_steps",
    resolved_task_spec: Mapping[str, Any] | None = None,
    resolved_task_spec_sha256: str | None = None,
    stream_key: Sequence[Any] | Any | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Batch:
    """Generate one- or multi-angle GP velocity integration.

    Hidden-init returns exactly ``horizon`` raw-velocity tokens.  Cue-driven
    prepends ``[Phi(q0), zeros(d), 1]`` and masks that cue token.  Crucially,
    ``init_mode`` is not part of the stochastic seed: the two tracks are paired
    views of the identical latent trajectories.
    """

    batch = int(batch_size)
    dim = int(dimensions)
    steps = int(horizon)
    if batch <= 0 or dim <= 0 or steps <= 0:
        raise ValueError("batch_size, dimensions, and horizon must be positive")
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive")
    if str(q0_distribution) != "uniform_minus_pi_pi":
        raise ValueError("unsupported q0 distribution")
    if not math.isfinite(float(q0_low)) or not math.isfinite(float(q0_high)):
        raise ValueError("q0 bounds must be finite")
    if float(q0_low) >= float(q0_high):
        raise ValueError("q0 lower bound must be smaller than upper bound")
    if bool(q0_high_inclusive):
        raise ValueError("NumPy uniform q0 sampling uses an exclusive high endpoint")
    if str(target_indexing) != "q_t_plus_1_after_velocity_update":
        raise ValueError("unsupported angular target indexing")
    if str(loss_mask_mode) != "all_256_velocity_steps":
        raise ValueError("unsupported angular loss mask")
    if (resolved_task_spec is None) != (resolved_task_spec_sha256 is None):
        raise ValueError("resolved task spec payload and SHA-256 must be supplied together")
    if resolved_task_spec is not None:
        expected_task_spec_sha256 = hashlib.sha256(
            _canonical_json_bytes(resolved_task_spec)
        ).hexdigest()
        if str(resolved_task_spec_sha256) != expected_task_spec_sha256:
            raise ValueError("resolved task spec SHA-256 does not match its payload")
    mode = _normalize_init_mode(init_mode)
    keys = _stream_key(stream_key)
    derived = keyed_seed(
        base_seed,
        "angular_integration",
        TASK_VERSION,
        dim,
        steps,
        float(dt),
        float(gp_length_scale),
        float(gp_std),
        float(gp_jitter),
        *keys,
    )
    rng = np.random.default_rng(derived)
    q0 = rng.uniform(float(q0_low), float(q0_high), size=(batch, dim))
    white = rng.standard_normal(size=(steps, batch * dim))
    factor = _gp_cholesky(
        steps,
        gp_length_scale,
        gp_std,
        gp_jitter,
        gp_grid_start,
        gp_grid_stop,
        gp_grid_endpoint,
    )
    velocity = (factor @ white).reshape(steps, batch, dim)

    q = np.empty((steps + 1, batch, dim), dtype=np.float64)
    q[0] = q0
    for step in range(steps):
        q[step + 1] = _wrap_angles(q[step] + float(dt) * velocity[step])
    embedded = _torus_embedding(q)

    if mode == "hidden-init":
        inputs = velocity
        latent_targets = q[1:]
        output_targets = embedded[1:]
        mask = np.ones_like(output_targets)
    else:
        input_dim = 3 * dim + 1
        inputs = np.zeros((steps + 1, batch, input_dim), dtype=np.float64)
        inputs[0, :, : 2 * dim] = embedded[0]
        inputs[0, :, -1] = 1.0
        inputs[1:, :, 2 * dim : 3 * dim] = velocity
        latent_targets = q
        output_targets = embedded
        mask = np.ones_like(output_targets)
        mask[0] = 0.0

    metadata = {
        "task_name": "angular_integration" if dim == 1 else "double_angular_integration",
        "task_version": TASK_VERSION,
        "base_seed": int(base_seed),
        "derived_seed": int(derived),
        "stream_key": keys,
        "init_mode": mode,
        "horizon": steps,
        "loss_bearing_steps": steps,
        "delta_t": float(dt),
        "latent_dimension": dim,
        "input_dimension": int(inputs.shape[-1]),
        "output_dimension": 2 * dim,
        "gp_grid": "linspace(-1,1,T)",
        "gp_grid_start": float(gp_grid_start),
        "gp_grid_stop": float(gp_grid_stop),
        "gp_grid_endpoint": bool(gp_grid_endpoint),
        "gp_length_scale": float(gp_length_scale),
        "gp_std": float(gp_std),
        "gp_cholesky_jitter": float(gp_jitter),
        "q0_distribution": str(q0_distribution),
        "q0_low": float(q0_low),
        "q0_high": float(q0_high),
        "q0_high_inclusive": bool(q0_high_inclusive),
        "initial_latents": q0,
        "target_indexing": "post_velocity_update",
        "target_indexing_contract": str(target_indexing),
        "loss_mask_mode": str(loss_mask_mode),
        "cue_token_is_loss_masked": mode == "cue-driven",
    }
    if resolved_task_spec is not None:
        metadata["resolved_task_spec"] = resolved_task_spec
        metadata["resolved_task_spec_sha256"] = str(resolved_task_spec_sha256)
    return Batch(
        inputs=_tensor(inputs, dtype=dtype, device=device),
        output_targets=_tensor(output_targets, dtype=dtype, device=device),
        latent_targets=_tensor(latent_targets, dtype=dtype, device=device),
        mask=_tensor(mask, dtype=dtype, device=device),
        metadata=metadata,
    )


def double_angular_integration(
    batch_size: int,
    base_seed: int,
    **kwargs: Any,
) -> Batch:
    """Convenience wrapper for the frozen two-angle integration task."""

    if "dimensions" in kwargs:
        raise TypeError("double_angular_integration fixes dimensions=2")
    return angular_integration(batch_size, base_seed, dimensions=2, **kwargs)


def _annotate_seed_roles(batch: Batch, task_seed: int, sample_seed: int) -> Batch:
    metadata = _thaw(batch.metadata)
    metadata["task_seed"] = int(task_seed)
    metadata["sample_seed"] = int(sample_seed)
    return Batch(
        inputs=batch.inputs,
        output_targets=batch.output_targets,
        latent_targets=batch.latent_targets,
        mask=batch.mask,
        metadata=metadata,
    )


def sample_memory_guided_saccade(
    batch_size: int,
    task_seed: int,
    sample_seed: int,
    *,
    horizon: int = MGS_HORIZON,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Batch:
    """Stable two-seed API for the frozen 512-step saccade task."""

    if int(horizon) != MGS_HORIZON:
        raise ValueError(f"memory-guided saccade horizon is frozen at {MGS_HORIZON}")
    batch = memory_guided_saccade(
        batch_size,
        task_seed,
        stream_key=("sample_seed", int(sample_seed)),
        device=device,
        dtype=dtype,
    )
    return _annotate_seed_roles(batch, task_seed, sample_seed)


def sample_angular_integration(
    batch_size: int,
    horizon: int,
    task_seed: int,
    sample_seed: int,
    init_mode: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    *,
    dt: float = ANGULAR_DT,
    gp_length_scale: float = GP_LENGTH_SCALE,
    gp_std: float = GP_STD,
    gp_jitter: float = GP_CHOLESKY_JITTER,
    task_spec: AngularTaskSpec | None = None,
    allow_horizon_override: bool = False,
) -> Batch:
    """Stable two-seed API for single-angle integration."""

    if task_spec is None:
        generator_kwargs = {
            "dimensions": 1,
            "init_mode": init_mode,
            "horizon": horizon,
            "dt": dt,
            "gp_length_scale": gp_length_scale,
            "gp_std": gp_std,
            "gp_jitter": gp_jitter,
        }
    else:
        if (
            int(horizon) != int(task_spec.sequence_steps)
            and not bool(allow_horizon_override)
        ):
            raise ValueError("sample horizon differs from resolved task spec")
        if _normalize_init_mode(init_mode) != task_spec.init_mode_for_generator:
            raise ValueError("sample init mode differs from resolved task spec")
        generator_kwargs = task_spec.generator_kwargs()
        if bool(allow_horizon_override):
            generator_kwargs["horizon"] = int(horizon)
    batch = angular_integration(
        batch_size,
        task_seed,
        **generator_kwargs,
        stream_key=("sample_seed", int(sample_seed)),
        device=device,
        dtype=dtype,
    )
    return _annotate_seed_roles(batch, task_seed, sample_seed)


def sample_double_angular_integration(
    batch_size: int,
    horizon: int,
    task_seed: int,
    sample_seed: int,
    init_mode: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    *,
    dt: float = ANGULAR_DT,
    gp_length_scale: float = GP_LENGTH_SCALE,
    gp_std: float = GP_STD,
    gp_jitter: float = GP_CHOLESKY_JITTER,
) -> Batch:
    """Stable two-seed API for the independent double-angle task."""

    batch = angular_integration(
        batch_size,
        task_seed,
        dimensions=2,
        init_mode=init_mode,
        horizon=horizon,
        dt=dt,
        gp_length_scale=gp_length_scale,
        gp_std=gp_std,
        gp_jitter=gp_jitter,
        stream_key=("sample_seed", int(sample_seed)),
        device=device,
        dtype=dtype,
    )
    return _annotate_seed_roles(batch, task_seed, sample_seed)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_sidecar(path: Path) -> Path:
    return Path(f"{path}.sha256")


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_fixed_bank(
    path: str | os.PathLike[str],
    batch: Batch,
    *,
    overwrite: bool = False,
) -> str:
    """Atomically save a Batch and return its SHA-256 digest.

    ``path`` and ``path + '.sha256'`` must both be absent unless ``overwrite``
    is explicitly requested.  The archive never contains pickled objects.
    """

    if not isinstance(batch, Batch):
        raise TypeError("batch must be a Batch")
    destination = Path(path).expanduser().resolve()
    sidecar = _sha256_sidecar(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and (destination.exists() or sidecar.exists()):
        raise FileExistsError(f"fixed bank or checksum already exists: {destination}")

    arrays = {
        "inputs": batch.inputs.detach().cpu().numpy(),
        "output_targets": batch.output_targets.detach().cpu().numpy(),
        "latent_targets": batch.latent_targets.detach().cpu().numpy(),
        "mask": batch.mask.detach().cpu().numpy(),
    }
    envelope = {
        "bank_schema_version": BANK_SCHEMA_VERSION,
        "batch_metadata": _thaw(batch.metadata),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
    }
    metadata_bytes = _canonical_json_bytes(envelope)

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
            np.savez_compressed(
                handle,
                **arrays,
                metadata_json=np.frombuffer(metadata_bytes, dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, destination)
        checksum_line = f"{digest}  {destination.name}\n".encode("ascii")
        _atomic_replace_bytes(sidecar, checksum_line)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return digest


def _expected_digest(path: Path) -> str:
    sidecar = _sha256_sidecar(path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"fixed-bank SHA-256 sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise ValueError(f"malformed fixed-bank SHA-256 sidecar: {sidecar}")
    digest = fields[0].lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"malformed SHA-256 digest in {sidecar}")
    return digest


def load_fixed_bank(
    path: str | os.PathLike[str],
    *,
    device: torch.device | str = "cpu",
) -> Batch:
    """Load a fixed bank after mandatory SHA-256 and schema validation."""

    source = Path(path).expanduser().resolve(strict=True)
    expected = _expected_digest(source)
    actual = sha256_file(source)
    if actual != expected:
        raise ValueError(
            f"fixed-bank SHA-256 mismatch for {source}: expected {expected}, got {actual}"
        )
    required = {"inputs", "output_targets", "latent_targets", "mask", "metadata_json"}
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError(
                f"fixed bank keys must be {sorted(required)}, got {sorted(archive.files)}"
            )
        metadata_raw = np.asarray(archive["metadata_json"], dtype=np.uint8).tobytes()
        envelope = json.loads(metadata_raw.decode("utf-8"))
        if envelope.get("bank_schema_version") != BANK_SCHEMA_VERSION:
            raise ValueError("unsupported fixed-bank schema version")
        arrays = {name: np.array(archive[name], copy=True) for name in required - {"metadata_json"}}

    specifications = envelope.get("arrays")
    if not isinstance(specifications, dict):
        raise ValueError("fixed-bank metadata has no array specifications")
    for name, array in arrays.items():
        spec = specifications.get(name, {})
        if spec.get("shape") != list(array.shape) or spec.get("dtype") != str(array.dtype):
            raise ValueError(f"fixed-bank metadata mismatch for {name}")
    metadata = envelope.get("batch_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("fixed-bank batch_metadata must be an object")
    return Batch(
        inputs=torch.as_tensor(arrays["inputs"], device=device),
        output_targets=torch.as_tensor(arrays["output_targets"], device=device),
        latent_targets=torch.as_tensor(arrays["latent_targets"], device=device),
        mask=torch.as_tensor(arrays["mask"], device=device),
        metadata=metadata,
    )


__all__ = [
    "ANGULAR_DT",
    "ANGULAR_HORIZON",
    "BANK_SCHEMA_VERSION",
    "Batch",
    "GP_CHOLESKY_JITTER",
    "GP_LENGTH_SCALE",
    "GP_STD",
    "MGS_HORIZON",
    "TASK_VERSION",
    "angular_integration",
    "double_angular_integration",
    "keyed_seed",
    "load_fixed_bank",
    "metadata_payload",
    "memory_guided_saccade",
    "sample_angular_integration",
    "sample_double_angular_integration",
    "sample_memory_guided_saccade",
    "save_fixed_bank",
    "sha256_file",
]
