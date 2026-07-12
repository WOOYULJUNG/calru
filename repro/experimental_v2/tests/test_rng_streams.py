from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


LEGACY_DIR = Path(__file__).resolve().parents[2] / "legacy_code"
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

import exp88_manifold_attractor_tasks as exp88  # noqa: E402


def _task_args() -> SimpleNamespace:
    return SimpleNamespace(
        ring_velocity_deg=3.0,
        torus_velocity_deg=2.5,
        curve_velocity_deg=2.5,
        surface_velocity_scale=0.018,
        hold_min=5,
        hold_max=20,
        move_min=3,
        move_max=10,
        final_hold_min=20,
        final_hold_max=80,
        ood_hold_min=30,
        ood_hold_max=120,
        ood_final_hold_min=100,
        ood_final_hold_max=250,
    )


def test_isolated_rng_restores_python_numpy_and_torch_state():
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    with exp88.isolated_experiment_rng(777):
        random.random()
        np.random.rand(4)
        torch.rand(4)

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_isolated_task_asset_is_independent_of_caller_rng():
    args = _task_args()
    with exp88.isolated_experiment_rng(12345):
        first = exp88.make_task_batch("ring_integrate", 5, 24, torch.device("cpu"), args)

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    random.random()
    np.random.rand(100)
    torch.rand(100)

    with exp88.isolated_experiment_rng(12345):
        second = exp88.make_task_batch("ring_integrate", 5, 24, torch.device("cpu"), args)

    for left, right in zip(first[:3], second[:3]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for key in first[3]:
        torch.testing.assert_close(first[3][key], second[3][key], rtol=0, atol=0)


def test_seed_derivation_is_stable_and_namespaced():
    first = exp88.deterministic_experiment_seed(880071, "ring_hold", 0, "train_batch", 7)
    assert first == exp88.deterministic_experiment_seed(
        880071, "ring_hold", 0, "train_batch", 7
    )
    assert first != exp88.deterministic_experiment_seed(
        880071, "ring_hold", 0, "long_horizon_probe", 7
    )
    assert first != exp88.deterministic_experiment_seed(
        880071, "torus_hold", 0, "train_batch", 7
    )
