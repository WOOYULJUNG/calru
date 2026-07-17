from __future__ import annotations

from repro.manifold_benchmark.launch_calru_tuning_v2 import (
    _final_key,
    _screen_key,
    confirm_jobs,
    load_config,
    screen_jobs,
    smoke_jobs,
)


def test_calru_topology_tuning_v2_grid_and_frozen_contract():
    config = load_config()
    smoke = smoke_jobs(config)
    screen = screen_jobs(config)
    assert len(smoke) == 3
    assert len(screen) == 40
    assert sum(job["topology"] == "s1" for job in screen) == 8
    assert sum(job["topology"] == "t2" for job in screen) == 8
    assert sum(job["topology"] == "s2" for job in screen) == 24
    assert {job["seed"] for job in screen} == {20}
    assert {job["updates"] for job in screen} == {5000}
    assert config["retention_plasticity"]["damage_epsilon"] == 3e-5
    assert config["training"]["state_noise_std"] == 0.0
    assert config["training"]["target_noise_std"] == 0.0
    assert config["training"]["output_dropout"] == 0.0


def test_calru_topology_tuning_v2_confirm_denominator():
    config = load_config()
    screen = screen_jobs(config)
    selection = {
        "selected": {
            "s1": [job for job in screen if job["topology"] == "s1"][:2],
            "t2": [job for job in screen if job["topology"] == "t2"][:2],
            "s2": [job for job in screen if job["topology"] == "s2"][:4],
        }
    }
    confirm = confirm_jobs(config, selection)
    assert len(confirm) == 16
    assert {job["seed"] for job in confirm} == {21, 22}
    assert len({job["job_id"] for job in confirm}) == 16


def test_calru_topology_tuning_v2_selection_prioritizes_gates_then_blank():
    passing = {
        "grid_order": [1],
        "metrics": {
            "task_gate": True,
            "blank_h4096_finite": True,
            "blank_h4096_intrinsic_mean_radians": 0.2,
            "task_intrinsic_mean_radians": 0.02,
        },
    }
    lower_blank_but_failed_task = {
        "grid_order": [0],
        "metrics": {
            "task_gate": False,
            "blank_h4096_finite": True,
            "blank_h4096_intrinsic_mean_radians": 0.01,
            "task_intrinsic_mean_radians": 0.5,
        },
    }
    assert _screen_key(passing) < _screen_key(lower_blank_but_failed_task)

    all_seed_gate = {
        "summary": {
            "task_gate_count": 3,
            "completed_count": 3,
            "blank_h4096_finite_count": 3,
            "median_blank_h4096_intrinsic_mean_radians": 0.2,
            "median_task_intrinsic_mean_radians": 0.02,
        }
    }
    two_seed_gate = {
        "summary": {
            "task_gate_count": 2,
            "completed_count": 3,
            "blank_h4096_finite_count": 3,
            "median_blank_h4096_intrinsic_mean_radians": 0.01,
            "median_task_intrinsic_mean_radians": 0.01,
        }
    }
    assert _final_key(all_seed_gate, 1) < _final_key(two_seed_gate, 0)
