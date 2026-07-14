from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import repro.sagodi_protocol.sagodi_paper_baselines as benchmark
from repro.sagodi_protocol.artifacts import write_completion_receipt

from repro.sagodi_protocol.sagodi_paper_baselines import (
    CAMPAIGN_ID,
    DEFAULT_CONFIG,
    LEAKY_RNN_VARIANT,
    MODEL_IDS,
    PARAMETER_COUNTS,
    PRIMARY_RNN_VARIANT,
    PaperTanhRNN,
    _children_binding,
    _ensure_bank,
    _execute_worker,
    _finalize,
    _generate_registered_bank,
    _load_and_validate_registered_bank,
    _parent_screen_binding,
    _parent_screen_binding_valid,
    _prepare_root,
    _record_scientific_failure,
    _smoke_summary,
    _stage_valid,
    _validate_bank_contract,
    _verified_child,
    _write_plan,
    build_smoke_plan,
    build_main_plan,
    build_model,
    build_screen_plan,
    build_sensitivity_plan,
    load_config,
    load_checkpoint,
    paper_selector_audit,
    scientific_gate,
    select_learning_rates,
    summarize_main,
)


from repro.sagodi_protocol.tasks import Batch, angular_integration, save_fixed_bank


def _write_result(
    path: Path,
    run_id: str,
    mse: float | None,
    *,
    nmse_db: float | None = -21.0,
    status: str = "completed",
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metrics = None if status == "failed" else {"masked_mse": mse, "masked_nmse_db": nmse_db}
    (path / "result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "status": status,
                "counts_in_denominator": status == "failed",
                "final_metrics": metrics,
            }
        )
    )


def test_frozen_config_and_model_counts() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config["training"]["weight_decay"] == 0.0
    assert config["training"]["target_noise_std"] == 0.0
    assert config["training"]["state_noise_std_after_transition"] == 0.0316227766
    for model_id in MODEL_IDS:
        torch.manual_seed(0)
        model = build_model(model_id)
        assert sum(parameter.numel() for parameter in model.parameters()) == (
            PARAMETER_COUNTS[model_id]
        )
        assert model.metadata()["initial_memory_source"] == "true_pre_update_q0_cos_sin"
        assert model.metadata()["track_classification"] == (
            "paper_and_source_informed_controlled_benchmark"
        )


def test_gru_lstm_use_explicit_xavier_weights_and_zero_biases() -> None:
    for model_id in ("gru_n128", "lstm_n64"):
        torch.manual_seed(5)
        model = build_model(model_id)
        assert model.metadata()["parameter_initialization"] == (
            "xavier_normal_all_weights_zero_all_biases"
        )
        assert torch.count_nonzero(model.core.cell.bias_ih).item() == 0
        assert torch.count_nonzero(model.core.cell.bias_hh).item() == 0
        assert torch.count_nonzero(model.core.readout.bias).item() == 0
        for weight in (
            model.core.cell.weight_ih,
            model.core.cell.weight_hh,
            model.core.readout.weight,
        ):
            assert torch.isfinite(weight).all().item()
            assert weight.std().item() > 0


def test_rnn_primary_is_pure_tanh_zero_bias_and_leaky_is_explicit() -> None:
    torch.manual_seed(7)
    pure = PaperTanhRNN(5, PRIMARY_RNN_VARIANT)
    leaky = PaperTanhRNN(5, LEAKY_RNN_VARIANT)
    leaky.load_state_dict(pure.state_dict())
    assert torch.count_nonzero(pure.brec).item() == 0
    x_t = torch.randn(3, 1)
    state = torch.randn(3, 5)
    proposal = torch.tanh(x_t @ pure.wi + state @ pure.wrec.t())
    assert torch.allclose(pure.step(x_t, state), proposal)
    assert torch.allclose(leaky.step(x_t, state), 0.9 * state + 0.1 * proposal)


def test_true_q0_maps_and_lstm_noise_cover_h_and_c() -> None:
    torch.manual_seed(11)
    model = build_model("lstm_n64")
    q0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    initial = model.initial_state(q0)
    assert initial.shape == (2, 128)
    assert not torch.equal(initial[:, :64], initial[:, 64:])

    inputs = torch.zeros(1, 2, 1)
    _, clean_states = model.forward_sequence(
        inputs, initial_memory=q0, return_states=True
    )
    generator = torch.Generator().manual_seed(123)
    _, noisy_states = model.forward_sequence(
        inputs,
        initial_memory=q0,
        state_noise_std=0.0316227766,
        noise_generator=generator,
        return_states=True,
    )
    delta = noisy_states[0] - clean_states[0]
    expected_generator = torch.Generator().manual_seed(123)
    expected = torch.randn((2, 128), generator=expected_generator) * 0.0316227766
    assert torch.allclose(delta, expected, atol=1e-8, rtol=1e-6)
    assert torch.count_nonzero(delta[:, :64]).item() > 0
    assert torch.count_nonzero(delta[:, 64:]).item() > 0


def test_screen_main_and_sensitivity_plans_are_separate(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    bank = tmp_path / "bank.npz"
    screen = build_screen_plan(tmp_path, config, bank)
    assert len(screen) == 60
    assert {spec.model_seed for spec in screen} == set(range(100, 105))
    assert {spec.learning_rate for spec in screen} == {1e-2, 1e-3, 1e-4, 1e-5}
    assert all(spec.updates == 5000 for spec in screen)
    assert all(spec.state_noise_std == 0.0316227766 for spec in screen)

    selection = {
        "selection_available": True,
        "selected_learning_rates": {model_id: 1e-3 for model_id in MODEL_IDS},
    }
    main = build_main_plan(tmp_path, config, bank, selection)
    assert len(main) == 30
    assert {spec.model_seed for spec in main} == set(range(10))
    assert all(spec.condition_id == "primary" for spec in main)

    sensitivity = build_sensitivity_plan(tmp_path, config, bank, selection)
    assert len(sensitivity) == 50
    assert sum(spec.condition_id == "pure_tanh_noise_0p1" for spec in sensitivity) == 30
    assert sum(
        spec.condition_id == "leaky_tanh_dt0p1_noise_0p1"
        for spec in sensitivity
    ) == 10
    assert sum(
        spec.condition_id == "leaky_tanh_dt0p1_noise_0p03162"
        for spec in sensitivity
    ) == 10
    assert all(
        spec.recurrence_variant == LEAKY_RNN_VARIANT
        for spec in sensitivity
        if spec.condition_id
        in {
            "leaky_tanh_dt0p1_noise_0p1",
            "leaky_tanh_dt0p1_noise_0p03162",
        }
    )
    assert {spec.learning_rate for spec in sensitivity} == {1e-3}


def test_lr_selection_uses_all_five_seeds_not_one_sentinel(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    specs = build_screen_plan(tmp_path, config, tmp_path / "bank.npz")
    for spec in specs:
        if spec.learning_rate == 1e-3:
            mse = 0.005
        elif spec.learning_rate == 1e-2 and spec.model_seed == 100:
            mse = 0.0001
        else:
            mse = 0.1 + spec.learning_rate
        _write_result(Path(spec.output_dir), spec.run_id, mse)
    selection = select_learning_rates(specs, config)
    assert selection["single_seed_gate_used"] is False
    assert selection["selection_uses_all_five_seeds"] is True
    assert selection["selection_available"] is True
    assert selection["selected_learning_rates"] == {
        model_id: 1e-3 for model_id in MODEL_IDS
    }


def test_failed_screen_child_stays_in_denominator(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    specs = build_screen_plan(tmp_path, config, tmp_path / "bank.npz")
    failed_run = next(
        spec
        for spec in specs
        if spec.model_id == "rnn_tanh_n128"
        and spec.learning_rate == 1e-3
        and spec.model_seed == 100
    )
    for spec in specs:
        if spec.run_id == failed_run.run_id:
            _write_result(Path(spec.output_dir), spec.run_id, None, status="failed")
        else:
            mse = 0.005 if spec.learning_rate == 1e-3 else 0.05
            _write_result(Path(spec.output_dir), spec.run_id, mse)
    selection = select_learning_rates(specs, config)
    row = next(
        item
        for item in selection["models"]["rnn_tanh_n128"]["candidates"]
        if item["learning_rate"] == 1e-3
    )
    assert row["seed_count"] == 5
    assert row["completed_seed_count"] == 4
    assert row["failed_seed_count"] == 1
    assert row["success_count"] == 4
    assert row["selection_eligible"] is False


def test_lr_selection_unavailable_without_five_valid_seeds(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    specs = build_screen_plan(tmp_path, config, tmp_path / "bank.npz")
    for spec in specs:
        status = "failed" if spec.model_seed == 100 else "completed"
        _write_result(
            Path(spec.output_dir),
            spec.run_id,
            None if status == "failed" else 0.005,
            status=status,
        )
    selection = select_learning_rates(specs, config)
    assert selection["selection_available"] is False
    assert selection["unavailable_models"] == list(MODEL_IDS)
    with pytest.raises(RuntimeError, match="selection is unavailable"):
        build_main_plan(tmp_path, config, tmp_path / "bank.npz", selection)


def test_paper_100_update_audit_cannot_select_primary(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    specs = build_screen_plan(tmp_path, config, tmp_path / "bank.npz")
    for spec in specs:
        _write_result(Path(spec.output_dir), spec.run_id, 0.005)
        value = 0.001 if spec.learning_rate == 1e-2 else 0.1
        (Path(spec.output_dir) / "training_trace.json").write_text(
            json.dumps([{"update": 100, "train_mse": value}])
        )
    audit = paper_selector_audit(specs, config)
    assert audit["may_select_primary_hyperparameters"] is False
    assert audit["full_length_selector_is_separate"] is True
    assert all(
        row["diagnostic_winner"]["learning_rate"] == 1e-2
        for row in audit["models"].values()
    )


def test_main_reports_yield_and_only_zero_eligible_is_hard_fail(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    selection = {
        "selection_available": True,
        "selected_learning_rates": {model_id: 1e-3 for model_id in MODEL_IDS},
    }
    specs = build_main_plan(tmp_path, config, tmp_path / "bank.npz", selection)
    passing = {"rnn_tanh_n128": 8, "gru_n128": 1, "lstm_n64": 0}
    for spec in specs:
        mse = 0.009 if spec.model_seed < passing[spec.model_id] else 0.011
        nmse = -21.0 if spec.model_seed < passing[spec.model_id] else -19.0
        _write_result(Path(spec.output_dir), spec.run_id, mse, nmse_db=nmse)
    summary = summarize_main(specs, config)
    assert summary["single_seed_gate_used"] is False
    assert summary["models"]["rnn_tanh_n128"]["analysis_eligibility_yield"] == 0.8
    assert summary["models"]["gru_n128"]["scientific_pass"] is True
    assert summary["models"]["gru_n128"]["low_analysis_yield_warning"] is True
    assert summary["models"]["lstm_n64"]["zero_analysis_eligible_hard_fail"] is True
    assert summary["all_models_scientific_pass"] is False
    gate = scientific_gate(summary)
    assert gate["all_required_gates_pass"] is False
    assert gate["downstream_ca_comparison_authorized"] is False


def test_tuning_and_main_banks_are_disjoint(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    tuning = _ensure_bank(tmp_path, config, "tuning")
    main = _ensure_bank(tmp_path, config, "main_test")
    assert tuning.read_bytes() != main.read_bytes()


def test_bank_validation_checks_seed_stream_and_task_metadata() -> None:
    config = load_config(DEFAULT_CONFIG)
    batch = angular_integration(
        4,
        31001,
        horizon=256,
        stream_key=("wrong", "stream"),
    )
    try:
        _validate_bank_contract(
            batch,
            trials=4,
            task_seed=31001,
            stream_key=("sagodi_paper_baselines_v1", "fixed_tuning_bank"),
            config=config,
        )
    except RuntimeError as error:
        assert "stream_key" in str(error)
    else:
        raise AssertionError("wrong bank stream metadata was accepted")


def test_bank_validation_rejects_resigned_tensor_tampering(tmp_path: Path) -> None:
    config, _ = _prepare_root(tmp_path, DEFAULT_CONFIG, require_clean=False)
    path = _ensure_bank(tmp_path, config, "smoke")
    expected = _generate_registered_bank(config, "smoke")
    inputs = expected.inputs.clone()
    inputs[0, 0, 0] += 1.0
    tampered = Batch(
        inputs=inputs,
        output_targets=expected.output_targets,
        latent_targets=expected.latent_targets,
        mask=expected.mask,
        metadata=expected.metadata,
    )
    save_fixed_bank(path, tampered, overwrite=True)
    with pytest.raises(RuntimeError, match="registered-digest|registered digest"):
        _load_and_validate_registered_bank(path, config, "smoke")


def test_checkpoint_round_trip_preserves_registered_semantics(tmp_path: Path) -> None:
    torch.manual_seed(17)
    model = build_model("rnn_tanh_n128")
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "checkpoint_type": CAMPAIGN_ID,
            "model": model.metadata(),
            "state_dict": model.state_dict(),
        },
        path,
    )
    restored, payload = load_checkpoint(path)
    assert payload["model"]["recurrence_variant"] == PRIMARY_RNN_VARIANT
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_scientific_failure_has_verified_receipt_checkpoint_and_binding(tmp_path: Path) -> None:
    config, copied = _prepare_root(tmp_path, DEFAULT_CONFIG, require_clean=False)
    bank = _ensure_bank(tmp_path, config, "smoke")
    spec = build_smoke_plan(tmp_path, config, bank)[0]
    _record_scientific_failure(
        spec,
        copied,
        "cpu",
        failure_kind="SyntheticNonFinite",
        failure_message="test-only failure",
        traceback_text=None,
    )
    assert _verified_child(spec)
    result = json.loads((Path(spec.output_dir) / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["counts_in_denominator"] is True
    assert result["failure_class"] == "scientific_numerical_nonfinite"
    binding = _children_binding([spec])
    assert binding["transitively_binds_every_child_receipt_and_checkpoint"] is True
    assert binding["children"][0]["checkpoint_name"] == "checkpoint_failure.pt"
    stage_root = tmp_path / "smoke"
    _write_plan(stage_root, [spec])
    summary = stage_root / "summary.json"
    summary.write_text(json.dumps(_smoke_summary([spec])))
    _finalize(stage_root, "smoke", [spec], [summary])
    assert (stage_root / "COMPUTATION_COMPLETE").is_file()
    assert not (stage_root / "SCIENTIFIC_PASS").exists()
    assert _stage_valid(stage_root, "smoke", 1)
    receipt = json.loads((stage_root / "completion_receipt.json").read_text())
    assert "children_binding.json" in receipt["artifacts"]


def test_infrastructure_and_interrupt_exceptions_are_not_denominator_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, copied = _prepare_root(tmp_path, DEFAULT_CONFIG, require_clean=False)
    bank = _ensure_bank(tmp_path, config, "smoke")
    spec = build_smoke_plan(tmp_path, config, bank)[0]

    def raise_io(*args: object, **kwargs: object) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(benchmark, "_train_worker", raise_io)
    with pytest.raises(OSError, match="disk unavailable"):
        _execute_worker(spec, copied, "cpu")
    assert not (Path(spec.output_dir) / "completion_receipt.json").exists()

    def raise_interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(benchmark, "_train_worker", raise_interrupt)
    with pytest.raises(KeyboardInterrupt):
        _execute_worker(spec, copied, "cpu")
    assert not (Path(spec.output_dir) / "completion_receipt.json").exists()


def test_child_receipt_rejects_extra_artifact_and_checkpoint_identity_tamper(
    tmp_path: Path,
) -> None:
    config, copied = _prepare_root(tmp_path, DEFAULT_CONFIG, require_clean=False)
    bank = _ensure_bank(tmp_path, config, "smoke")
    spec = build_smoke_plan(tmp_path, config, bank)[0]
    _record_scientific_failure(
        spec,
        copied,
        "cpu",
        failure_kind="SyntheticNonFinite",
        failure_message="test-only failure",
        traceback_text=None,
    )
    output = Path(spec.output_dir)
    extra = output / "unexpected.txt"
    extra.write_text("unexpected")
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=spec.run_id,
        artifacts=[
            output / "run_manifest.json",
            output / "training_trace.json",
            output / "result.json",
            output / "failure.json",
            output / "checkpoint_failure.pt",
            output / "FAILED",
            extra,
        ],
        metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "condition_id": spec.condition_id,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
            "outcome": "failed",
            "failure_class": "scientific_numerical_nonfinite",
        },
    )
    assert not _verified_child(spec)

    _record_scientific_failure(
        spec,
        copied,
        "cpu",
        failure_kind="SyntheticNonFinite",
        failure_message="test-only failure",
        traceback_text=None,
    )
    checkpoint_path = output / "checkpoint_failure.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["run"] = {**checkpoint["run"], "run_id": "tampered"}
    torch.save(checkpoint, checkpoint_path)
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=spec.run_id,
        artifacts=[
            output / "run_manifest.json",
            output / "training_trace.json",
            output / "result.json",
            output / "failure.json",
            checkpoint_path,
            output / "FAILED",
        ],
        metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "condition_id": spec.condition_id,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
            "outcome": "failed",
            "failure_class": "scientific_numerical_nonfinite",
        },
    )
    assert not _verified_child(spec)


def test_stage_recomputation_rejects_resigned_summary_tamper(tmp_path: Path) -> None:
    config, copied = _prepare_root(tmp_path, DEFAULT_CONFIG, require_clean=False)
    bank = _ensure_bank(tmp_path, config, "smoke")
    spec = build_smoke_plan(tmp_path, config, bank)[0]
    _record_scientific_failure(
        spec,
        copied,
        "cpu",
        failure_kind="SyntheticNonFinite",
        failure_message="test-only failure",
        traceback_text=None,
    )
    stage_root = tmp_path / "smoke"
    _write_plan(stage_root, [spec])
    summary = stage_root / "summary.json"
    summary.write_text(json.dumps(_smoke_summary([spec])))
    _finalize(stage_root, "smoke", [spec], [summary])
    assert _stage_valid(stage_root, "smoke", 1)

    summary.write_text(json.dumps({"schema_version": 1, "tampered": True}))
    write_completion_receipt(
        stage_root / "completion_receipt.json",
        job_id=f"{CAMPAIGN_ID}__smoke",
        artifacts=[
            stage_root / "plan.json",
            stage_root / "children_binding.json",
            stage_root / "COMPUTATION_COMPLETE",
            summary,
        ],
        metadata={"campaign_id": CAMPAIGN_ID, "stage": "smoke", "run_count": 1},
    )
    assert not _stage_valid(stage_root, "smoke", 1)


def test_parent_screen_binding_detects_current_selection_change(tmp_path: Path) -> None:
    screen = tmp_path / "screen"
    stage = tmp_path / "main"
    screen.mkdir()
    stage.mkdir()
    (screen / "completion_receipt.json").write_text("receipt-v1")
    (screen / "lr_selection.json").write_text("selection-v1")
    (stage / "parent_screen_binding.json").write_text(
        json.dumps(_parent_screen_binding(tmp_path))
    )
    assert _parent_screen_binding_valid(tmp_path, stage)
    (screen / "lr_selection.json").write_text("selection-v2")
    assert not _parent_screen_binding_valid(tmp_path, stage)
