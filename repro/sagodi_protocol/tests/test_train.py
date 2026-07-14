from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from repro.sagodi_protocol import train
from repro.sagodi_protocol.artifacts import (
    RECEIPT_IDENTITY_ENV,
    verify_completion_receipt,
)
from repro.sagodi_protocol.config import (
    NATIVE_RECIPE_PROTOCOL_PATH,
    SAGODI_LR_SELECTION_PROTOCOL_PATH,
    SAGODI_PRIMARY_MODELS,
    SAGODI_PRIMARY_MODEL_WIDTHS,
    SAGODI_PRIMARY_PARAMETER_COUNTS,
    AngularTaskSpec,
    load_protocol,
    protocol_fingerprint,
)
from repro.sagodi_protocol.models import ModelConfig, build_protocol_model
from repro.sagodi_protocol.primary_main_campaign import (
    EXPECTED_RP_CONTRACT,
    MainModel,
    MainRun,
    ParentSelector,
    _verify_training_output,
    load_main_template,
    materialize_resolved_protocol,
)
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


def test_rp_update_equals_batch_mean_ablation_damage_logit_rule():
    """Regression-test the paper equation, not only the RP call schedule."""

    torch.manual_seed(17)
    model = build_protocol_model(ModelConfig("ca_lru", 1, 2, width=4))
    model.eval()
    batch = sample_angular_integration(3, 4, 23, 29, "hidden-init")
    initial = train._initial_embedding(batch, batch.inputs.device)
    with torch.no_grad():
        _, states = model.forward_sequence(
            batch.inputs, initial_memory=initial, return_states=True
        )
        state = states[-1]
        target = batch.output_targets[-1]
        clean = train._roll_blank(model, state, 3)
        clean_error = (model.decode(clean) - target).square().sum(dim=-1).mean()
        recurrence, state_slice = tuple(model.pan_recs_with_slices())[0]
        hidden = state_slice.stop - state_slice.start
        ablated = state.unsqueeze(0).expand(hidden, *state.shape).clone()
        coordinate = torch.arange(hidden)
        ablated[coordinate, :, state_slice.start + coordinate] = 0.0
        delayed = train._roll_blank(model, ablated.reshape(-1, state.shape[1]), 3)
        ablated_error = (
            model.decode(delayed)
            .reshape(hidden, batch.batch_size, model.output_dim)
            .sub(target.unsqueeze(0))
            .square()
            .sum(dim=-1)
            .mean(dim=1)
        )
        damage = ablated_error - clean_error
        theta_before = recurrence.theta.detach().clone()
        eta = 7.0
        epsilon = 1.0e-4
        expected_theta = (theta_before + eta * (damage - epsilon)).clamp(-18.0, 18.0)

    train._retention_plasticity_call(
        model,
        batch,
        blank_horizon=3,
        eta_lambda=eta,
        damage_epsilon=epsilon,
    )

    torch.testing.assert_close(recurrence.theta, expected_theta, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        recurrence.lam_mag(),
        torch.sqrt(torch.sigmoid(expected_theta).clamp(1.0e-8, 1.0 - 1.0e-8)),
        rtol=0.0,
        atol=0.0,
    )


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
    assert frozen["rp_frozen_contract"]["probe_batch_size"] == 96
    assert frozen["rp_frozen_contract"]["probe_horizon"] == 256
    assert frozen["rp_frozen_contract"]["blank_ablation_horizon"] == 500
    assert frozen["rp_effective_runtime"] == {
        "enabled_by_protocol": False,
        "probe_batch_size": 4,
        "probe_horizon": 8,
        "blank_ablation_horizon": 8,
        "eta_lambda": 3000.0,
        "damage_epsilon": 1e-4,
    }
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
    assert manifest["rp_frozen_contract"] == frozen["rp_frozen_contract"]
    assert manifest["rp_effective_runtime"] == frozen["rp_effective_runtime"]
    assert receipt["metadata"]["rp_frozen_contract"] == frozen["rp_frozen_contract"]
    assert receipt["metadata"]["rp_effective_runtime"] == frozen[
        "rp_effective_runtime"
    ]
    assert "progress.json" in receipt["artifacts"]
    valid, reason = verify_completion_receipt(output / "completion_receipt.json")
    assert valid, reason

    progress["completed_updates"] = 0
    (output / "progress.json").write_text(json.dumps(progress))
    valid, reason = verify_completion_receipt(output / "completion_receipt.json")
    assert not valid
    assert "hash mismatch" in reason


def test_primary_main_receipt_binds_established_rp_numeric_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = Path(train.__file__).resolve().parent
    selector_protocol = load_protocol(package / "sagodi_primary_lr_selection_v3.yaml")
    rates = (0.01, 0.001, 0.0001, 0.00001, 0.001, 0.01)
    models = tuple(
        MainModel(
            model_id=model_id,
            hidden_width=SAGODI_PRIMARY_MODEL_WIDTHS[model_id],
            parameter_count=SAGODI_PRIMARY_PARAMETER_COUNTS[model_id],
            learning_rate=rate,
        )
        for model_id, rate in zip(SAGODI_PRIMARY_MODELS, rates)
    )
    binding = {
        "schema_version": 1,
        "campaign_id": "sagodi_six_model_lr_selection_v3",
        "scientific_identity": "0" * 64,
        "selector_protocol_canonical_fingerprint": "1" * 64,
        "manifest_sha256": "2" * 64,
        "summary_sha256": "3" * 64,
        "selection_receipt_sha256": "4" * 64,
        "completion_receipt_sha256": "5" * 64,
        "selector_complete_sha256": "6" * 64,
        "selector_code_commit": "7" * 40,
        "verified_nested_training_receipts": 120,
        "selected_learning_rates": {
            model.model_id: model.learning_rate for model in models
        },
    }
    parent = ParentSelector(
        root=tmp_path / "selector",
        protocol=selector_protocol,
        manifest={"scientific_identity": "0" * 64},
        models=models,
        binding=binding,
    )
    resolved = materialize_resolved_protocol(
        load_main_template(package / "primary_main_template_v3.json"), parent
    )
    protocol_path = tmp_path / "resolved_primary_main_protocol.json"
    protocol_path.write_text(json.dumps(resolved))
    scientific_identity = "a" * 64
    run = MainRun(models[-1], 0)
    expected_receipt_identity = {
        "campaign_scientific_identity": scientific_identity,
        "protocol_fingerprint": protocol_fingerprint(resolved),
        "run_id": run.run_id,
        "stage": "primary_main_training",
    }
    for key, environment in RECEIPT_IDENTITY_ENV.items():
        monkeypatch.setenv(environment, expected_receipt_identity[key])
    output = tmp_path / "primary_ca_lru_smoke"
    train.train_one(
        train.TrainSpec(
            model_name="ca_lru",
            model_seed=0,
            learning_rate=models[-1].learning_rate,
            output_dir=output,
            protocol_path=protocol_path,
            campaign_identity=scientific_identity,
            device="cpu",
            steps_override=2,
            batch_override=4,
            smoke=True,
        )
    )
    config = json.loads((output / "config.json").read_text())
    child = json.loads((output / "manifest.json").read_text())
    receipt = json.loads((output / "completion_receipt.json").read_text())
    expected_effective = {
        "enabled_by_protocol": True,
        "probe_batch_size": 4,
        "probe_horizon": 8,
        "blank_ablation_horizon": 8,
        "eta_lambda": 3000.0,
        "damage_epsilon": 3e-5,
    }
    for payload in (config["training"], child, receipt["metadata"]):
        assert payload["rp_frozen_contract"] == EXPECTED_RP_CONTRACT
        assert payload["rp_effective_runtime"] == expected_effective

    campaign_manifest = {
        "smoke": True,
        "scientific_identity": scientific_identity,
        "protocol_canonical_fingerprint": protocol_fingerprint(resolved),
        "evaluation_bank": {"sha256": child["evaluation_bank_sha256"]},
    }
    valid, reason, _ = _verify_training_output(
        output, run, campaign_manifest, parent=parent
    )
    assert valid, reason

    receipt["metadata"]["rp_frozen_contract"]["blank_ablation_horizon"] = 256
    (output / "completion_receipt.json").write_text(json.dumps(receipt))
    valid, reason, _ = _verify_training_output(
        output, run, campaign_manifest, parent=parent
    )
    assert not valid
    assert "RP numeric contract" in reason
