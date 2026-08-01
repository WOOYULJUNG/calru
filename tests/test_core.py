import torch

from rplru.config import load_protocol
from rplru.models import (
    MODEL_NAMES,
    RPLRUModel,
    build_model,
    count_parameters,
    parameter_count_formula,
    saturated_sigmoid,
)
from rplru.spatial import classification_metrics, coordinate_ce, grid_labels
from rplru.task import HOLD, TaskSpec, generate_batch


def test_protocol_and_task_are_deterministic_and_leak_free():
    protocol = load_protocol()
    assert protocol.rp_lru_hidden_size == 91
    assert protocol.rp.probe_horizons == (50,)
    assert protocol.rp.retention_thresholds == (0.2,)
    assert protocol.evaluation.bank_seed == 20260720

    spec = TaskSpec.controlled(
        dimension=2,
        batch_size=8,
        update_count=4,
        segment_hold=10,
    )
    first = generate_batch(spec, protocol.task, seed=7)
    second = generate_batch(spec, protocol.task, seed=7)
    assert first.fingerprint() == second.fingerprint()
    assert torch.count_nonzero(first.inputs[first.operation == HOLD]) == 0
    assert torch.allclose(first.final_targets, second.final_targets)


def test_rp_lru_zero_input_is_exactly_diagonal_autonomous_dynamics():
    torch.manual_seed(0)
    model = RPLRUModel(
        input_dim=4,
        output_dim=2,
        width=7,
        initial_lambda_low=0.98,
        initial_lambda_high=0.999,
    )
    state = torch.randn(5, model.state_size)
    zero = torch.zeros(5, model.input_dim)
    expected = model.retention() * state
    assert torch.equal(model.step(zero, state), expected)


def test_saturated_retention_has_an_explicit_unit_branch():
    theta = torch.tensor([0.0, 16.63, 16.64, 30.0], dtype=torch.float64)
    retention = saturated_sigmoid(theta)
    assert retention[0] == 0.5
    assert retention[1] < 1.0
    assert torch.equal(retention[2:], torch.ones(2, dtype=torch.float64))


def test_parameter_count_formulas_match_constructed_models():
    for name in MODEL_NAMES:
        model = build_model(name, dimension=2, width=8)
        assert parameter_count_formula(name, dimension=2, width=8) == count_parameters(
            model
        )
    rp = build_model("rp_lru", dimension=2, width=8, retention_mode="rp")
    assert parameter_count_formula(
        "rp_lru",
        dimension=2,
        width=8,
        trainable_only=True,
        retention_mode="rp",
    ) == count_parameters(rp, trainable_only=True)


def test_spatial_labels_loss_and_metrics():
    values = torch.tensor([[-1.0, -0.2], [0.0, 0.4], [1.0, 0.8]])
    labels = grid_labels(values, bins=5)
    assert labels.tolist() == [[0, 2], [2, 3], [4, 4]]

    logits = torch.full((3, 10), -10.0)
    for sample in range(3):
        for coordinate in range(2):
            logits[sample, coordinate * 5 + labels[sample, coordinate]] = 10.0
    assert coordinate_ce(logits, labels, bins=5) < 1e-6
    metrics = classification_metrics(logits, labels, bins=5)
    assert metrics["coordinate_accuracy"] == 1.0
    assert metrics["exact_accuracy"] == 1.0
