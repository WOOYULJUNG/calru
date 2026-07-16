from __future__ import annotations

import pytest
import torch

from repro.manifold_benchmark.topology_models import (
    MODEL_IDS,
    TOPOLOGY_DIMS,
    build_topology_model,
    load_transfer_config,
)
from repro.manifold_benchmark.launch_topology_hparam import (
    broad_jobs,
    refine_jobs,
    robust_jobs,
    smoke_jobs,
)
from repro.manifold_benchmark.run_topology_hparam import load_search_config
from repro.manifold_benchmark.launch_topology_transfer import (
    _balanced_queues,
    _jobs,
)
from repro.manifold_benchmark.topology_training import (
    TOPOLOGIES,
    component_mean_mse,
    fixed_debug_batch,
    generated_batch,
    hold_prediction,
    intrinsic_metrics,
)


@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_all_model_topology_adapters_have_correct_shapes(model_id, topology):
    input_dim, memory_dim, output_dim = TOPOLOGY_DIMS[topology]
    model = build_topology_model(model_id, topology, model_seed=91)
    inputs = torch.randn(4, 3, input_dim)
    initial_memory = torch.randn(3, memory_dim)
    prediction, states = model.forward_sequence(
        inputs, initial_memory=initial_memory, return_states=True
    )
    assert prediction.shape == (4, 3, output_dim)
    assert states.shape == (4, 3, model.state_size)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(states).all()
    assert model.metadata()["topology_specific_retuning"] is False


@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_first_prediction_is_one_application_of_u0_after_q0(model_id, topology):
    batch = fixed_debug_batch(
        topology, debug_seed=9, batch_size=3, horizon=4, device="cpu"
    )
    model = build_topology_model(model_id, topology, model_seed=92)
    prediction = model.forward_sequence(
        batch.inputs, initial_memory=batch.initial_memory
    )
    q0_state = model.initialize(batch.initial_memory)
    expected_first = model.decode(model.step(batch.inputs[0], q0_state))
    torch.testing.assert_close(prediction[0], expected_first, rtol=0, atol=0)


def test_online_stream_is_model_independent_and_reproducible():
    first = generated_batch(
        "t2",
        replicate_seed=10,
        update=37,
        batch_size=5,
        horizon=7,
        device="cpu",
    )
    repeated = generated_batch(
        "t2",
        replicate_seed=10,
        update=37,
        batch_size=5,
        horizon=7,
        device="cpu",
    )
    torch.testing.assert_close(first.initial_memory, repeated.initial_memory, rtol=0, atol=0)
    torch.testing.assert_close(first.inputs, repeated.inputs, rtol=0, atol=0)
    torch.testing.assert_close(first.output_targets, repeated.output_targets, rtol=0, atol=0)
    torch.testing.assert_close(first.trajectory_id, repeated.trajectory_id, rtol=0, atol=0)


def test_component_mean_mse_is_output_dimension_invariant():
    prediction = torch.tensor([[[1.0, -2.0]]])
    target = torch.zeros_like(prediction)
    duplicated_prediction = prediction.repeat(1, 1, 2)
    duplicated_target = target.repeat(1, 1, 2)
    torch.testing.assert_close(
        component_mean_mse(prediction, target),
        component_mean_mse(duplicated_prediction, duplicated_target),
    )


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_oracle_intrinsic_metrics_and_hold_baseline(topology):
    batch = fixed_debug_batch(
        topology, debug_seed=9, batch_size=8, horizon=16, device="cpu"
    )
    oracle = intrinsic_metrics(topology, batch.output_targets, batch.output_targets)
    hold = intrinsic_metrics(topology, hold_prediction(batch), batch.output_targets)
    assert oracle["component_mse"] == 0.0
    assert oracle["intrinsic_mean_radians"] < 1e-3
    assert hold["intrinsic_mean_radians"] > oracle["intrinsic_mean_radians"]


def test_frozen_ring_selected_settings_are_explicit():
    config = load_transfer_config()
    expected = {
        "rnn": (128, 0.003),
        "gru": (128, 0.0003),
        "lstm": (64, 0.001),
        "hc": (52, 0.01),
    }
    assert {
        name: (int(row["width"]), float(row["learning_rate"]))
        for name, row in config["models"].items()
    } == expected
    assert config["training"]["state_noise_std"] == 0.0
    assert config["training"]["target_noise_std"] == 0.0
    assert config["training"]["output_dropout"] == 0.0


def test_pilot_launcher_balances_hc_across_six_gpus():
    queues, loads = _balanced_queues(_jobs("pilot", load_transfer_config()), 6)
    assert sum(map(len, queues)) == 36
    assert all(any(job["model"] == "hc" for job in queue) for queue in queues)
    assert max(loads) - min(loads) <= 1.5


@pytest.mark.parametrize("model_id", ("calru", "hc"))
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_search_models_support_all_topologies_with_rp(model_id, topology):
    config = load_search_config()
    model = build_topology_model(model_id, topology, model_seed=93, config=config)
    input_dim, memory_dim, output_dim = TOPOLOGY_DIMS[topology]
    prediction = model.forward_sequence(
        torch.randn(3, 2, input_dim), initial_memory=torch.randn(2, memory_dim)
    )
    assert prediction.shape == (3, 2, output_dim)
    assert model.rp_enabled
    assert model.dynamic_lambda(model.initialize(torch.randn(2, memory_dim))) is not None


def test_hparam_campaign_has_frozen_75_full_runs_and_six_smokes():
    config = load_search_config()
    smoke = smoke_jobs(config)
    broad = broad_jobs(config)
    assert len(smoke) == 6
    assert len(broad) == 39
    selected = {
        "selected": {
            topology: {
                "hc": next(job for job in broad if job["model"] == "hc" and job["topology"] == topology),
                "calru": next(job for job in broad if job["model"] == "calru" and job["topology"] == topology),
            }
            for topology in TOPOLOGIES
        }
    }
    refine = refine_jobs(config, selected)
    assert len(refine) == 24
    refine_selected = {
        "selected_hc_finalists": {
            topology: [
                job
                for job in refine
                if job["model"] == "hc" and job["topology"] == topology
            ][:2]
            for topology in TOPOLOGIES
        }
    }
    robust = robust_jobs(config, refine_selected)
    assert len(robust) == 12
    assert len(broad) + len(refine) + len(robust) == 75
    assert all(job["updates"] == 5000 for job in broad + refine + robust)
