from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from repro.manifold_benchmark.artifacts import (
    load_manifold_bank,
    load_parent_bank,
    save_manifold_bank,
    save_parent_bank,
)
from repro.manifold_benchmark.generator import (
    ConditionSpec,
    ParentSpec,
    derive_s1,
    derive_s2,
    derive_torus,
    make_parent_bank,
)


def small_parent():
    return make_parent_bank(
        ParentSpec(
            trajectories=16,
            max_horizon=32,
            max_torus_dimension=8,
            training_horizon=16,
            gp_grid_spacing=2.0 / 15.0,
            task_seed=17,
            sample_seed=23,
            split="unit_test",
        )
    )


def assert_batch_equal(left, right):
    for name in (
        "initial_memory",
        "inputs",
        "output_targets",
        "latent_targets",
        "latent_path",
        "base_drive",
        "effective_velocity",
        "dwell_mask",
        "trajectory_id",
        "mask",
    ):
        assert np.array_equal(getattr(left, name), getattr(right, name)), name
    if left.latent_unwrapped is None:
        assert right.latent_unwrapped is None
    else:
        assert np.array_equal(left.latent_unwrapped, right.latent_unwrapped)


def test_parent_replay_is_bit_identical_and_streams_are_separate():
    left = small_parent()
    right = small_parent()
    for name in (
        "white_noise",
        "id_base_drive",
        "q0_angles",
        "q0_sphere_gaussian",
        "sparsity_parameters",
        "mask_uniform_randoms",
        "trajectory_id",
    ):
        assert np.array_equal(getattr(left, name), getattr(right, name)), name

    changed = make_parent_bank(
        ParentSpec(
            trajectories=16,
            max_horizon=32,
            max_torus_dimension=8,
            training_horizon=16,
            gp_grid_spacing=2.0 / 15.0,
            task_seed=17,
            sample_seed=24,
            split="unit_test",
        )
    )
    assert not np.array_equal(left.white_noise, changed.white_noise)
    assert not np.array_equal(left.trajectory_id, changed.trajectory_id)


def test_q0_q1_indexing_and_torus_oracle():
    parent = small_parent()
    condition = ConditionSpec(horizon=16, dwell_profile="dense")
    batch = derive_torus(
        parent, dimensions=2, condition=condition, storage_dtype=np.float64
    )
    q0 = parent.q0_angles[:, :2]
    assert np.allclose(batch.initial_memory[:, 0::2], np.cos(q0), atol=1e-14)
    assert np.allclose(batch.initial_memory[:, 1::2], np.sin(q0), atol=1e-14)

    # Independent loop oracle: it does not call the production torus integrator.
    oracle = np.empty_like(batch.latent_unwrapped)
    oracle[0] = q0
    for step in range(batch.horizon):
        oracle[step + 1] = oracle[step] + 0.1 * batch.inputs[step]
    wrapped = (oracle + math.pi) % (2.0 * math.pi) - math.pi
    expected_output = np.empty_like(batch.output_targets)
    expected_output[..., 0::2] = np.cos(wrapped[1:])
    expected_output[..., 1::2] = np.sin(wrapped[1:])
    assert np.max(np.abs(batch.latent_unwrapped - oracle)) < 1e-12
    assert np.max(np.abs(batch.output_targets - expected_output)) < 1e-12
    assert not np.array_equal(batch.initial_memory, batch.output_targets[0])


def test_zero_control_oracle_and_global_dwell_mask():
    parent = small_parent()
    blank = ConditionSpec(horizon=16, dwell_profile="all_blank")
    for batch in (
        derive_s1(parent, condition=blank, storage_dtype=np.float64),
        derive_torus(
            parent, dimensions=8, condition=blank, storage_dtype=np.float64
        ),
        derive_s2(parent, condition=blank, storage_dtype=np.float64),
    ):
        assert np.count_nonzero(batch.inputs) == 0
        assert np.count_nonzero(batch.effective_velocity) == 0
        assert np.array_equal(
            batch.latent_path,
            np.broadcast_to(batch.latent_path[0], batch.latent_path.shape),
        )
        assert np.array_equal(
            batch.output_targets,
            np.broadcast_to(batch.initial_memory, batch.output_targets.shape),
        )

    variable = derive_torus(
        parent,
        dimensions=8,
        condition=ConditionSpec(horizon=32),
        storage_dtype=np.float64,
    )
    zero = variable.dwell_mask[..., 0] == 0
    assert np.all(variable.inputs[zero] == 0)
    assert np.all(variable.effective_velocity[zero] == 0)


def test_topology_constraints_dimension_nesting_and_prefix_identity():
    parent = small_parent()
    short = ConditionSpec(horizon=16)
    long = ConditionSpec(horizon=32)
    s1_short = derive_s1(parent, condition=short, storage_dtype=np.float64)
    s1_long = derive_s1(parent, condition=long, storage_dtype=np.float64)
    t8 = derive_torus(
        parent, dimensions=8, condition=long, storage_dtype=np.float64
    )

    for name in (
        "inputs",
        "output_targets",
        "latent_targets",
        "base_drive",
        "effective_velocity",
        "dwell_mask",
        "mask",
    ):
        assert np.array_equal(getattr(s1_short, name), getattr(s1_long, name)[:16])
    assert np.array_equal(s1_short.latent_path, s1_long.latent_path[:17])
    assert np.array_equal(s1_short.latent_unwrapped, s1_long.latent_unwrapped[:17])
    assert np.array_equal(s1_long.latent_path[..., 0], t8.latent_path[..., 0])
    assert np.array_equal(s1_long.inputs[..., 0], t8.inputs[..., 0])
    assert np.array_equal(s1_long.output_targets, t8.output_targets[..., :2])

    pairs = t8.output_targets.reshape(*t8.output_targets.shape[:2], 8, 2)
    assert np.max(np.abs(np.sum(pairs**2, axis=-1) - 1.0)) < 1e-12


def test_sphere_tangency_rodrigues_rotation_and_norm_oracle():
    parent = small_parent()
    batch = derive_s2(
        parent,
        condition=ConditionSpec(horizon=32, dwell_profile="dense"),
        storage_dtype=np.float64,
    )
    norm_error = np.max(np.abs(np.linalg.norm(batch.latent_path, axis=-1) - 1.0))
    tangency = np.max(
        np.abs(np.sum(batch.latent_path[:-1] * batch.effective_velocity, axis=-1))
    )
    assert norm_error < 1e-12
    assert tangency < 1e-12

    assert np.max(
        np.abs(
            batch.effective_velocity
            - np.cross(batch.inputs, batch.latent_path[:-1])
        )
    ) < 1e-12

    # Independent Rodrigues loop from q0 and model inputs.
    oracle = np.empty_like(batch.latent_path)
    oracle[0] = batch.initial_memory
    for step in range(batch.horizon):
        current = oracle[step]
        omega = batch.inputs[step]
        speed = np.linalg.norm(omega, axis=1, keepdims=True)
        alpha = 0.1 * speed
        axis = np.divide(omega, speed, out=np.zeros_like(omega), where=speed > 0)
        rotated = (
            np.cos(alpha) * current
            + np.sin(alpha) * np.cross(axis, current)
            + (1.0 - np.cos(alpha))
            * np.sum(axis * current, axis=1, keepdims=True)
            * axis
        )
        moving = speed[:, 0] > 0
        oracle[step + 1] = current
        oracle[step + 1, moving] = rotated[moving]
    assert np.max(np.abs(batch.latent_path - oracle)) < 1e-12


def test_ood_scale_pairing_and_energy_matched_rule():
    parent = small_parent()
    base = derive_torus(
        parent,
        dimensions=8,
        condition=ConditionSpec(horizon=16, dwell_profile="dense"),
        storage_dtype=np.float64,
    )
    scaled = derive_torus(
        parent,
        dimensions=8,
        condition=ConditionSpec(
            horizon=16,
            dwell_profile="dense",
            velocity_scale=2.0,
            condition_axis="velocity_scale",
            condition_value="2",
        ),
        storage_dtype=np.float64,
    )
    energy = derive_torus(
        parent,
        dimensions=8,
        condition=ConditionSpec(horizon=16, dwell_profile="dense"),
        energy_mode="energy_matched",
        storage_dtype=np.float64,
    )
    assert np.array_equal(scaled.base_drive, 2.0 * base.base_drive)
    assert np.array_equal(scaled.initial_memory, base.initial_memory)
    assert np.array_equal(scaled.dwell_mask, base.dwell_mask)
    assert np.array_equal(energy.base_drive, base.base_drive / math.sqrt(8))


def test_safe_parent_and_derived_bank_roundtrip(tmp_path: Path):
    parent_path = tmp_path / "parent.npz"
    parent = small_parent()
    parent_digest = save_parent_bank(parent_path, parent)
    loaded_parent = load_parent_bank(parent_path)
    assert loaded_parent.metadata["archive_sha256"] == parent_digest
    assert np.array_equal(parent.white_noise, loaded_parent.white_noise)

    batch = derive_s2(
        loaded_parent,
        condition=ConditionSpec(horizon=16),
        storage_dtype=np.float32,
    )
    batch_path = tmp_path / "id_s2.npz"
    save_manifold_bank(batch_path, batch)
    loaded_batch = load_manifold_bank(batch_path)
    assert_batch_equal(batch, loaded_batch)
    assert loaded_batch.metadata["parent_bank_sha256"] == parent_digest
