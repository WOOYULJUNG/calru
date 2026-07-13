from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from repro.sagodi_protocol import train
from repro.sagodi_protocol.config import load_protocol
from repro.sagodi_protocol.models import ModelConfig, build_protocol_model
from repro.sagodi_protocol.tasks import sample_angular_integration, save_fixed_bank


def test_nonfinite_guards_reject_tensor_and_nested_metric():
    with pytest.raises(train.NonFiniteTrainingError, match="loss"):
        train._require_finite_tensor(torch.tensor(float("nan")), "loss")
    with pytest.raises(train.NonFiniteTrainingError, match="metrics.geodesic.q95"):
        train._require_finite_payload(
            {"geodesic": {"q95": float("inf")}}, "metrics"
        )
    model = build_protocol_model(ModelConfig("gru", 1, 2, width=8))
    next(model.parameters()).data.fill_(float("inf"))
    with pytest.raises(train.NonFiniteTrainingError, match="parameter"):
        train._require_finite_model(model, "model")


def test_full_rp_schedule_is_exact_and_controls_have_no_calls():
    ca_lru = build_protocol_model(ModelConfig("ca_lru", 1, 2, width=8))
    no_rp = build_protocol_model(ModelConfig("no_rp", 1, 2, width=8))
    gru = build_protocol_model(ModelConfig("gru", 1, 2, width=8))
    expected = tuple(range(1550, 5001, 50))
    assert train._expected_rp_steps(
        ca_lru, steps=5000, warmup=1500, interval=50, smoke=False
    ) == expected
    assert len(expected) == 70
    for control in (no_rp, gru):
        assert train._expected_rp_steps(
            control, steps=5000, warmup=1500, interval=50, smoke=False
        ) == ()


def test_shared_evaluation_bank_is_verified_and_returned_exactly(tmp_path: Path):
    protocol = load_protocol()
    protocol["evaluation"]["id_test_trials"] = 3
    batch = sample_angular_integration(3, 256, 0, 0, "hidden-init")
    path = tmp_path / "angular_id_seed000.npz"
    digest = save_fixed_bank(path, batch)
    loaded, actual, source = train._load_evaluation_bank(
        path, protocol, device=torch.device("cpu"), smoke=False
    )
    assert actual == digest
    assert source == str(path.resolve())
    assert loaded.batch_size == 3
    assert torch.equal(loaded.inputs, batch.inputs)


def test_full_run_rejects_training_overrides_before_writing(tmp_path: Path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="smoke-only"):
        train.train_one(
            train.TrainSpec(
                model_name="gru",
                model_seed=100,
                learning_rate=0.01,
                output_dir=output,
                device="cpu",
                steps_override=2,
                smoke=False,
            )
        )
    assert not output.exists()


def test_nonfinite_loss_never_writes_completion_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "nan_run"

    def nonfinite_loss(
        prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        del target, mask
        return prediction.sum() * torch.tensor(float("nan"), device=prediction.device)

    monkeypatch.setattr(train, "masked_mse", nonfinite_loss)
    with pytest.raises(train.NonFiniteTrainingError, match="loss"):
        train.train_one(
            train.TrainSpec(
                model_name="gru",
                model_seed=100,
                learning_rate=0.01,
                output_dir=output,
                device="cpu",
                steps_override=1,
                batch_override=2,
                smoke=True,
            )
        )
    assert (output / "config.json").is_file()
    assert not (output / "completion_receipt.json").exists()


def test_gru_smoke_receipt_records_zero_rp_calls(tmp_path: Path):
    output = tmp_path / "gru_smoke"
    train.train_one(
        train.TrainSpec(
            model_name="gru",
            model_seed=100,
            learning_rate=0.01,
            output_dir=output,
            device="cpu",
            steps_override=1,
            batch_override=2,
            smoke=True,
        )
    )
    receipt = json.loads((output / "completion_receipt.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert receipt["metadata"]["rp_calls"] == 0
    assert manifest["rp_schedule"]["actual_steps"] == []
    assert manifest["rp_schedule"]["expected_steps"] == []
    assert manifest["model_id"] == "gru"
    assert manifest["protocol_canonical_fingerprint"]
    assert manifest["architecture_metadata"]["parameters_total"] > 0
