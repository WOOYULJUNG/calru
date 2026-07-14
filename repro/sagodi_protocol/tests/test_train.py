from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from repro.sagodi_protocol import train
from repro.sagodi_protocol.artifacts import verify_completion_receipt
from repro.sagodi_protocol.config import (
    NATIVE_RECIPE_PROTOCOL_PATH,
    SAGODI_LR_SELECTION_PROTOCOL_PATH,
    AngularTaskSpec,
    load_protocol,
)
from repro.sagodi_protocol.models import ModelConfig, build_protocol_model
from repro.sagodi_protocol.tasks import (
    Batch,
    metadata_payload,
    sample_angular_integration,
    save_fixed_bank,
)


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
    assert train._expected_rp_steps(
        ca_lru,
        steps=100,
        warmup=100,
        interval=1,
        smoke=False,
        enabled_by_protocol=False,
    ) == ()
    for control in (no_rp, gru):
        assert train._expected_rp_steps(
            control, steps=5000, warmup=1500, interval=50, smoke=False
        ) == ()


def test_native_rp_schedule_is_exactly_3100_through_10000():
    ca_lru = build_protocol_model(ModelConfig("ca_lru", 1, 2, width=8))
    expected = tuple(range(3100, 10001, 100))
    assert train._expected_rp_steps(
        ca_lru, steps=10000, warmup=3000, interval=100, smoke=False
    ) == expected
    assert len(expected) == 70


def test_optimizer_factory_supports_legacy_adam_and_native_adamw():
    model = build_protocol_model(ModelConfig("gru", 1, 2, width=8))
    legacy = train._build_optimizer(
        model,
        {"name": "Adam", "betas": [0.9, 0.999], "weight_decay": 0.0},
        learning_rate=0.01,
    )
    assert isinstance(legacy, torch.optim.Adam)
    assert legacy.param_groups[0]["lr"] == 0.01
    native = train._build_optimizer(
        model,
        {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 1e-5,
        },
        learning_rate=0.001,
    )
    assert isinstance(native, torch.optim.AdamW)
    assert native.param_groups[0]["lr"] == 0.001
    assert native.param_groups[0]["weight_decay"] == 1e-5
    assert native.param_groups[0]["eps"] == 1e-8


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


def test_evaluation_bank_rejects_task_metadata_drift():
    protocol = load_protocol(NATIVE_RECIPE_PROTOCOL_PATH)
    spec = AngularTaskSpec.from_protocol(protocol)
    batch = sample_angular_integration(
        3,
        256,
        0,
        0,
        "hidden-init",
        task_spec=spec,
    )
    metadata = metadata_payload(batch.metadata)
    metadata["delta_t"] = 0.2
    tampered = Batch(
        inputs=batch.inputs,
        output_targets=batch.output_targets,
        latent_targets=batch.latent_targets,
        mask=batch.mask,
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="delta_t"):
        train._validate_evaluation_batch(
            tampered,
            protocol,
            require_full_protocol_shape=False,
        )


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


def test_trainer_resolves_track_specific_frozen_model_seeds() -> None:
    native = load_protocol(NATIVE_RECIPE_PROTOCOL_PATH)
    selector = load_protocol(SAGODI_LR_SELECTION_PROTOCOL_PATH)
    assert train._frozen_training_model_seeds(native) == (100, 101, 102, 103, 104)
    assert train._frozen_training_model_seeds(selector) == (
        1100,
        1101,
        1102,
        1103,
        1104,
    )


def test_selector_rejects_nonselection_seed_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "selection_bad_seed"
    with pytest.raises(ValueError, match="frozen training set"):
        train.train_one(
            train.TrainSpec(
                model_name="gru_sagodi_width96",
                model_seed=100,
                learning_rate=0.01,
                output_dir=output,
                protocol_path=SAGODI_LR_SELECTION_PROTOCOL_PATH,
                device="cpu",
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


def test_selector_ca_lru_smoke_executes_zero_rp_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "selector_ca_lru_smoke"
    monkeypatch.setenv("CALRU_PHYSICAL_GPU_ID", "0")
    train.train_one(
        train.TrainSpec(
            model_name="ca_lru",
            model_seed=1100,
            learning_rate=0.01,
            output_dir=output,
            protocol_path=SAGODI_LR_SELECTION_PROTOCOL_PATH,
            device="cpu",
            steps_override=1,
            batch_override=2,
            smoke=True,
        )
    )
    config_payload = json.loads((output / "config.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    receipt = json.loads((output / "completion_receipt.json").read_text())
    assert config_payload["training"]["rp_enabled_by_protocol"] is False
    assert config_payload["training"]["expected_rp_steps"] == []
    assert manifest["rp_schedule"] == {
        "actual_steps": [],
        "calls": 0,
        "expected_steps": [],
    }
    assert json.loads((output / "rp_trace.json").read_text()) == []
    assert receipt["metadata"]["rp_calls"] == 0


def test_native_gru_smoke_records_recipe_transfer_training_config(tmp_path: Path):
    output = tmp_path / "native_gru_smoke"
    train.train_one(
        train.TrainSpec(
            model_name="gru",
            model_seed=100,
            learning_rate=0.001,
            output_dir=output,
            protocol_path=NATIVE_RECIPE_PROTOCOL_PATH,
            device="cpu",
            steps_override=1,
            batch_override=2,
            smoke=True,
        )
    )
    payload = json.loads((output / "config.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    frozen = payload["training"]
    assert frozen["optimizer"] == {
        "name": "AdamW",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 1e-5,
    }
    assert frozen["learning_rate"] == 0.001
    assert frozen["gradient_clipping"] == {
        "policy": "global_norm",
        "frozen_numeric_value": 1.0,
    }
    assert frozen["state_noise_enabled"] is False
    assert frozen["state_noise_coordinate_std"] == 0.0
    assert frozen["rp_probe_horizon"] == 8
    assert frozen["rp_blank_ablation_horizon"] == 8
    progress = json.loads((output / "progress.json").read_text())
    receipt = json.loads((output / "completion_receipt.json").read_text())
    assert progress["completed_updates"] == 1
    assert progress["total_updates"] == 1
    assert progress["status"] == "training_updates_complete"
    assert progress["pre_clip_global_gradient_norm"] >= 0.0
    assert manifest["protocol_track"] == "calru_native_recipe_transfer"
    assert manifest["reporting"]["protocol_A_eligible"] is False
    assert payload["resolved_task_spec_sha256"] == manifest["resolved_task_spec_sha256"]
    assert receipt["metadata"]["resolved_task_spec_sha256"] == manifest[
        "resolved_task_spec_sha256"
    ]
    assert manifest["evaluation_bank_task_spec_sha256"] == manifest[
        "resolved_task_spec_sha256"
    ]
    assert "progress.json" in receipt["artifacts"]
    valid, reason = verify_completion_receipt(output / "completion_receipt.json")
    assert valid, reason

    progress["completed_updates"] = 0
    (output / "progress.json").write_text(json.dumps(progress))
    valid, reason = verify_completion_receipt(output / "completion_receipt.json")
    assert not valid
    assert "hash mismatch" in reason
