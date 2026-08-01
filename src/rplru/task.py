"""Intermittent Vector Integration task generation and invariant checks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np
import torch

from .config import TaskConfig


HOLD = 0
WRITE = 1
ADD = 2
PADDING = -1


@dataclass(frozen=True)
class TaskSpec:
    dimension: int
    batch_size: int
    update_count_min: int
    update_count_max: int
    hold_min: int
    hold_max: int
    controlled_update_count: int | None = None
    controlled_segment_hold: int | None = None

    def validate(self) -> None:
        if self.dimension <= 0 or self.batch_size <= 0:
            raise ValueError("dimension and batch_size must be positive")
        if not 0 <= self.update_count_min <= self.update_count_max:
            raise ValueError("invalid update-count range")
        if not 0 <= self.hold_min <= self.hold_max:
            raise ValueError("invalid hold range")
        if self.controlled_update_count is not None:
            if self.controlled_update_count < 0:
                raise ValueError("controlled update count must be non-negative")
            if self.update_count_min != self.update_count_max:
                raise ValueError("controlled update count requires a fixed range")
            if self.update_count_min != self.controlled_update_count:
                raise ValueError("controlled update count differs from fixed range")
        if self.controlled_segment_hold is not None:
            if self.controlled_segment_hold < 0:
                raise ValueError("controlled hold must be non-negative")
            if self.hold_min != self.hold_max:
                raise ValueError("controlled hold requires a fixed range")
            if self.hold_min != self.controlled_segment_hold:
                raise ValueError("controlled hold differs from fixed range")

    @classmethod
    def training(
        cls, dimension: int, batch_size: int, config: TaskConfig
    ) -> "TaskSpec":
        return cls(
            dimension=int(dimension),
            batch_size=int(batch_size),
            update_count_min=int(config.train_update_count_min),
            update_count_max=int(config.train_update_count_max),
            hold_min=int(config.train_hold_min),
            hold_max=int(config.train_hold_max),
        )

    @classmethod
    def controlled(
        cls, dimension: int, batch_size: int, update_count: int, segment_hold: int
    ) -> "TaskSpec":
        return cls(
            dimension=int(dimension),
            batch_size=int(batch_size),
            update_count_min=int(update_count),
            update_count_max=int(update_count),
            hold_min=int(segment_hold),
            hold_max=int(segment_hold),
            controlled_update_count=int(update_count),
            controlled_segment_hold=int(segment_hold),
        )


@dataclass
class IntermittentBatch:
    """Padded time-major batch with analysis-only memory annotations."""

    inputs: torch.Tensor
    final_targets: torch.Tensor
    memory_targets: torch.Tensor
    valid_mask: torch.Tensor
    boundary_mask: torch.Tensor
    hold_end_mask: torch.Tensor
    operation: torch.Tensor
    lengths: torch.Tensor
    update_counts: torch.Tensor
    segment_holds: torch.Tensor
    sampling_attempts: torch.Tensor
    metadata: Mapping[str, Any]

    @property
    def time_steps(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.inputs.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.final_targets.shape[-1])

    @property
    def total_blank_steps(self) -> torch.Tensor:
        return self.segment_holds.sum(dim=-1)

    def to(self, device: torch.device | str) -> "IntermittentBatch":
        fields = {}
        for name in (
            "inputs",
            "final_targets",
            "memory_targets",
            "valid_mask",
            "boundary_mask",
            "hold_end_mask",
            "operation",
            "lengths",
            "update_counts",
            "segment_holds",
            "sampling_attempts",
        ):
            fields[name] = getattr(self, name).to(device)
        return IntermittentBatch(**fields, metadata=dict(self.metadata))

    def numpy_payload(self) -> dict[str, np.ndarray]:
        return {
            name: getattr(self, name).detach().cpu().numpy()
            for name in (
                "inputs",
                "final_targets",
                "memory_targets",
                "valid_mask",
                "boundary_mask",
                "hold_end_mask",
                "operation",
                "lengths",
                "update_counts",
                "segment_holds",
                "sampling_attempts",
            )
        }

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.numpy_payload().items()):
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()


def _sample_trajectory(
    rng: np.random.Generator,
    *,
    dimension: int,
    updates: int,
    task: TaskConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    initial = rng.uniform(
        task.initial_value_low, task.initial_value_high, size=dimension
    ).astype(np.float32)
    deltas = np.empty((updates, dimension), dtype=np.float32)
    current = initial.astype(np.float64)
    attempts = 0
    for update in range(updates):
        accepted = False
        for _ in range(task.maximum_rejection_attempts):
            attempts += 1
            delta = rng.uniform(task.delta_low, task.delta_high, size=dimension)
            candidate = current + delta
            if np.all(candidate >= task.trajectory_low) and np.all(
                candidate <= task.trajectory_high
            ):
                deltas[update] = delta.astype(np.float32)
                current = candidate
                accepted = True
                break
        if not accepted:
            raise RuntimeError(
                "trajectory rejection sampler exhausted its maximum attempts"
            )
    return initial, deltas, attempts


def _sample_seed(seed: int, sample_index: int) -> int:
    digest = hashlib.sha256(
        f"ivi_sample_v1\0{int(seed)}\0{int(sample_index)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def generate_batch(
    spec: TaskSpec,
    task: TaskConfig,
    *,
    seed: int,
    sample_offset: int = 0,
    device: torch.device | str = "cpu",
    audit: bool = True,
) -> IntermittentBatch:
    """Generate one deterministic IVI batch.

    The first event is always ``write``.  Each write/add event is followed by
    its hold segment.  Padding is zero-valued but excluded by ``valid_mask``.
    """

    spec.validate()
    task.validate()
    if int(sample_offset) < 0:
        raise ValueError("sample_offset must be non-negative")
    updates = np.empty(spec.batch_size, dtype=np.int64)
    max_segments = spec.update_count_max + 1
    holds = np.zeros((spec.batch_size, max_segments), dtype=np.int64)
    trajectories = []
    for sample in range(spec.batch_size):
        rng = np.random.default_rng(
            _sample_seed(int(seed), int(sample_offset) + sample)
        )
        count = int(
            rng.integers(spec.update_count_min, spec.update_count_max + 1)
        )
        updates[sample] = count
        holds[sample, : count + 1] = rng.integers(
            spec.hold_min, spec.hold_max + 1, size=count + 1
        )
        trajectories.append(
            _sample_trajectory(
                rng, dimension=spec.dimension, updates=count, task=task
            )
        )
    lengths = 1 + updates + holds.sum(axis=1)
    max_time = int(lengths.max())
    input_dim = spec.dimension + 2
    inputs = np.zeros((max_time, spec.batch_size, input_dim), dtype=np.float32)
    final_targets = np.zeros((spec.batch_size, spec.dimension), dtype=np.float32)
    memory_targets = np.zeros(
        (max_time, spec.batch_size, spec.dimension), dtype=np.float32
    )
    valid = np.zeros((max_time, spec.batch_size), dtype=np.bool_)
    boundary = np.zeros_like(valid)
    hold_end = np.zeros_like(valid)
    operation = np.full(
        (max_time, spec.batch_size), fill_value=PADDING, dtype=np.int8
    )
    attempts = np.zeros(spec.batch_size, dtype=np.int64)

    for sample, count_value in enumerate(updates):
        count = int(count_value)
        initial, deltas, attempt_count = trajectories[sample]
        attempts[sample] = attempt_count
        current = initial.copy()
        time = 0
        inputs[time, sample, : spec.dimension] = current
        inputs[time, sample, spec.dimension] = 1.0
        valid[time, sample] = True
        boundary[time, sample] = True
        operation[time, sample] = WRITE
        memory_targets[time, sample] = current
        time += 1

        for segment in range(count + 1):
            hold_length = int(holds[sample, segment])
            for hold_offset in range(hold_length):
                valid[time, sample] = True
                operation[time, sample] = HOLD
                memory_targets[time, sample] = current
                if hold_offset == hold_length - 1:
                    hold_end[time, sample] = True
                time += 1
            if segment < count:
                delta = deltas[segment]
                current = current + delta
                inputs[time, sample, : spec.dimension] = delta
                inputs[time, sample, spec.dimension + 1] = 1.0
                valid[time, sample] = True
                boundary[time, sample] = True
                operation[time, sample] = ADD
                memory_targets[time, sample] = current
                time += 1

        if time != int(lengths[sample]):
            raise AssertionError("internal sequence-length mismatch")
        final_targets[sample] = current

    metadata = {
        "schema_version": 1,
        "task": "intermittent_vector_integration",
        "dimension": spec.dimension,
        "batch_size": spec.batch_size,
        "seed": int(seed),
        "sample_offset": int(sample_offset),
        "controlled_update_count": spec.controlled_update_count,
        "controlled_segment_hold": spec.controlled_segment_hold,
        "input_layout": {
            "value": [0, spec.dimension],
            "write_flag": spec.dimension,
            "add_flag": spec.dimension + 1,
        },
    }
    batch = IntermittentBatch(
        inputs=torch.from_numpy(inputs),
        final_targets=torch.from_numpy(final_targets),
        memory_targets=torch.from_numpy(memory_targets),
        valid_mask=torch.from_numpy(valid),
        boundary_mask=torch.from_numpy(boundary),
        hold_end_mask=torch.from_numpy(hold_end),
        operation=torch.from_numpy(operation),
        lengths=torch.from_numpy(lengths),
        update_counts=torch.from_numpy(updates),
        segment_holds=torch.from_numpy(holds),
        sampling_attempts=torch.from_numpy(attempts),
        metadata=metadata,
    )
    if audit:
        audit_batch(batch, task)
    if torch.device(device).type != "cpu":
        batch = batch.to(device)
    return batch


def audit_batch(batch: IntermittentBatch, task: TaskConfig) -> dict[str, Any]:
    """Fail closed on task leakage or target-construction errors."""

    inputs = batch.inputs.detach().cpu()
    target = batch.memory_targets.detach().cpu()
    valid = batch.valid_mask.detach().cpu()
    boundary = batch.boundary_mask.detach().cpu()
    hold_end = batch.hold_end_mask.detach().cpu()
    operation = batch.operation.detach().cpu()
    dimension = batch.dimension
    if inputs.ndim != 3 or inputs.shape[-1] != dimension + 2:
        raise ValueError("invalid input tensor shape")
    if target.shape != (*inputs.shape[:2], dimension):
        raise ValueError("invalid memory-target shape")
    if not torch.equal(inputs[operation == HOLD], torch.zeros_like(inputs[operation == HOLD])):
        raise ValueError("hold input is not exactly zero")
    if not torch.equal(
        inputs[operation == PADDING], torch.zeros_like(inputs[operation == PADDING])
    ):
        raise ValueError("padding input is not exactly zero")
    if torch.any(boundary & ~valid) or torch.any(hold_end & ~valid):
        raise ValueError("analysis mask selects padding")
    if not torch.equal(operation != PADDING, valid):
        raise ValueError("operation and valid masks disagree")
    if not torch.equal(
        boundary.sum(dim=0).to(batch.update_counts.device),
        batch.update_counts + 1,
    ):
        raise ValueError("boundary count differs from write plus add count")
    reconstructed = torch.zeros_like(batch.final_targets.detach().cpu())
    for sample in range(batch.batch_size):
        current = None
        for time in range(int(batch.lengths[sample])):
            op = int(operation[time, sample])
            value = inputs[time, sample, :dimension]
            if op == WRITE:
                current = value.clone()
            elif op == ADD:
                if current is None:
                    raise ValueError("add appeared before write")
                current = current + value
            elif op != HOLD:
                raise ValueError("unexpected operation inside valid sequence")
            if not torch.allclose(target[time, sample], current, rtol=0.0, atol=1e-6):
                raise ValueError("memory target disagrees with operation history")
        reconstructed[sample] = current
    if not torch.allclose(
        reconstructed, batch.final_targets.detach().cpu(), rtol=0.0, atol=1e-6
    ):
        raise ValueError("final target disagrees with accumulated updates")
    valid_targets = target[valid]
    if torch.any(valid_targets < task.trajectory_low - 1e-6) or torch.any(
        valid_targets > task.trajectory_high + 1e-6
    ):
        raise ValueError("trajectory escaped the configured bounds")
    return {
        "fingerprint": batch.fingerprint(),
        "time_steps": batch.time_steps,
        "batch_size": batch.batch_size,
        "dimension": batch.dimension,
        "blank_steps": int((operation == HOLD).sum()),
        "boundary_states": int(boundary.sum()),
        "hold_end_states": int(hold_end.sum()),
        "target_min": float(valid_targets.min()),
        "target_max": float(valid_targets.max()),
        "mean_sampling_attempts": float(batch.sampling_attempts.float().mean()),
        "rejected_delta_draws": int(
            (
                batch.sampling_attempts
                - batch.update_counts
            ).sum().cpu()
        ),
    }
