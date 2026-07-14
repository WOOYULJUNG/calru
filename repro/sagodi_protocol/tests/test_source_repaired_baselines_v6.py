from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from repro.sagodi_protocol.artifacts import atomic_json, sha256_file, strict_json_load
from repro.sagodi_protocol.source_repaired_baselines_v6 import (
    CAMPAIGN_ID,
    MODEL_IDS,
    PARAMETER_COUNTS,
    ROOT_MARKER,
    RunSpec,
    _train_worker,
    _scientific_identity,
    _stage_valid,
    _validate_bank,
    _verified_child,
    build_fanout_plan,
    build_main_plan,
    build_model,
    build_sentinel_plan,
    load_config,
    noise_metadata,
    scientific_gate,
    select_hyperparameters,
    select_sentinel_top3,
    summarize_main,
)
from repro.sagodi_protocol.source_resolved_protocol import source_angular_integration
from repro.sagodi_protocol.tasks import Batch, save_fixed_bank


def _write_result(spec: RunSpec, mse: float, nmse_db: float, *, status: str = "completed") -> None:
    output = Path(spec.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "result.json",
        {
            "run_id": spec.run_id,
            "status": status,
            "final_metrics": (
                {"mse": mse, "nmse_db": nmse_db} if status == "completed" else None
            ),
        },
    )


def _populate_sentinel(specs: tuple[RunSpec, ...]) -> None:
    for index, spec in enumerate(specs):
        _write_result(spec, 0.002 + index * 1e-5, -25.0 + index * 1e-3)


def test_config_is_t128_source_q1_and_frozen_repair_contract() -> None:
    config = load_config()
    assert config["task"]["horizon"] == 128
    assert config["task"]["input_sparsity"] == "variable_uniform_0_2"
    assert config["task"]["initial_state_semantics"] == "source_q1_post_update_target"
    assert config["training"]["updates"] == 5000
    assert config["training"]["batch_size"] == 64
    assert config["training"]["constant_learning_rate"] is True
    assert config["training"]["early_stopping"] is False
    paths = config["upstream"]["paths"]
    assert paths["sagodi_rnn_tanh_n128"]["model"] == "models.py:104-319"
    rnn = config["models"][0]
    assert rnn["recurrent_bias_policy"].startswith("paper_aligned_zero_bias_repair")
    assert "seed100" in rnn["bias_parity_evidence"]["task"]
    assert config["models"][2]["recurrent_weight_decay"] == 0.0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda cfg: cfg["upstream"].update(repository="https://evil.invalid"),
        lambda cfg: cfg["models"][1].update(recurrent_weight_decay=123.0),
        lambda cfg: cfg["hyperparameter_tuning"].update(analysis_nmse_db_threshold=999.0),
        lambda cfg: cfg["main"].update(scientific_pass_minimum_eligible_per_model=9),
        lambda cfg: cfg["evaluation_banks"]["main_test"].update(task_seed=1),
    ],
)
def test_config_rejects_any_major_contract_mutation(
    tmp_path: Path, mutation
) -> None:
    config = deepcopy(load_config())
    mutation(config)
    path = tmp_path / "mutated.json"
    atomic_json(path, config)
    with pytest.raises(ValueError, match="canonical frozen contract"):
        load_config(path)


def test_models_have_expected_counts_maps_bias_and_lstm_wd() -> None:
    config = load_config()
    for model_id in MODEL_IDS:
        model = build_model(config, model_id, 1e-3, 0.01)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == PARAMETER_COUNTS[model_id]
        assert torch.isfinite(model.output_to_hidden).all()
    rnn = build_model(config, MODEL_IDS[0], 1e-3, 0.0)
    assert torch.count_nonzero(rnn.core.brec) == 0
    lstm = build_model(config, MODEL_IDS[2], 1e-3, 0.0)
    assert lstm.output_to_cell is not None
    assert lstm.output_to_cell.data_ptr() != lstm.output_to_hidden.data_ptr()
    assert lstm.recipe.recurrent_weight_decay == 0.0


def test_source_task_is_variable_sparsity_t128_and_q1_initialized() -> None:
    batch = source_angular_integration(64, 7, stream_key=(CAMPAIGN_ID, "unit"))
    assert batch.inputs.shape == (128, 64, 1)
    assert batch.output_targets.shape == (128, 64, 2)
    assert batch.metadata["initial_state_semantics"] == "source_q1_post_update_target"
    counts = torch.as_tensor(batch.metadata["zero_token_counts"])
    assert counts.min() == 0
    assert counts.max() > 0
    q1 = batch.latent_targets[0, :, 0]
    assert torch.allclose(batch.output_targets[0, :, 0], torch.cos(q1))
    assert torch.allclose(batch.output_targets[0, :, 1], torch.sin(q1))


def test_noise_metadata_separates_nominal_and_actual() -> None:
    rnn = noise_metadata(MODEL_IDS[0], 0.01)
    assert rnn["actual_post_transition_state_noise_std"] == 0.01
    assert rnn["source_api_nominal_state_noise_std"] == pytest.approx(0.01 / (0.1**0.5))
    for model_id in MODEL_IDS[1:]:
        row = noise_metadata(model_id, 0.01)
        assert row["actual_post_transition_state_noise_std"] == 0.01
        assert row["source_api_nominal_state_noise_std"] is None
        assert row["source_api_nominal_semantics"].startswith("not_applicable")


def test_bank_validation_recomputes_deterministic_content() -> None:
    config = load_config()
    batch = source_angular_integration(
        16, 32999, stream_key=(CAMPAIGN_ID, "fixed_smoke_bank")
    )
    _validate_bank(batch, trials=16, config=config, purpose="smoke")
    tampered_inputs = batch.inputs.clone()
    tampered_inputs[0, 0, 0] += 1.0
    tampered = Batch(
        inputs=tampered_inputs,
        output_targets=batch.output_targets,
        latent_targets=batch.latent_targets,
        mask=batch.mask,
        metadata=batch.metadata,
    )
    with pytest.raises(RuntimeError, match="deterministic content differs"):
        _validate_bank(tampered, trials=16, config=config, purpose="smoke")
    wrong_but_self_consistent = source_angular_integration(
        16, 5, stream_key=(CAMPAIGN_ID, "wrong-but-self-consistent")
    )
    with pytest.raises(RuntimeError, match="base seed differs"):
        _validate_bank(
            wrong_but_self_consistent, trials=16, config=config, purpose="smoke"
        )


def test_plan_counts_and_fresh_seed_boundaries(tmp_path: Path) -> None:
    config = load_config()
    bank = tmp_path / "bank.npz"
    sentinel = build_sentinel_plan(tmp_path, config, bank)
    assert len(sentinel) == 48
    assert {spec.model_seed for spec in sentinel} == {100}
    _populate_sentinel(sentinel)
    top = select_sentinel_top3(sentinel, config)
    fanout = build_fanout_plan(tmp_path, config, bank, top)
    assert len(fanout) == 36
    assert {spec.model_seed for spec in fanout} == {101, 102, 103, 104}
    for index, spec in enumerate(fanout):
        _write_result(spec, 0.003 + index * 1e-5, -24.0 + index * 1e-3)
    selection = select_hyperparameters(sentinel, fanout, config)
    main = build_main_plan(tmp_path, config, bank, selection)
    assert len(main) == 30
    assert {spec.model_seed for spec in main} == set(range(10))


def test_sentinel_and_fanout_never_select_missing_seed(tmp_path: Path) -> None:
    config = load_config()
    bank = tmp_path / "bank.npz"
    sentinel = build_sentinel_plan(tmp_path, config, bank)
    _populate_sentinel(sentinel)
    top = select_sentinel_top3(sentinel, config)
    fanout = build_fanout_plan(tmp_path, config, bank, top)
    for index, spec in enumerate(fanout):
        _write_result(spec, 0.003 + index * 1e-5, -24.0)
    first_model = MODEL_IDS[0]
    top_cell = top["models"][first_model]["top_cells"][0]
    broken = next(
        spec
        for spec in fanout
        if spec.model_id == first_model
        and spec.learning_rate == top_cell["learning_rate"]
        and spec.actual_state_noise_std == top_cell["actual_state_noise_std"]
    )
    _write_result(broken, 0.0, -100.0, status="failed")
    selection = select_hyperparameters(sentinel, fanout, config)
    winner = selection["models"][first_model]["winner"]
    assert (winner["learning_rate"], winner["actual_state_noise_std"]) != (
        broken.learning_rate,
        broken.actual_state_noise_std,
    )
    for row in top["models"][first_model]["top_cells"]:
        target = next(
            spec
            for spec in fanout
            if spec.model_id == first_model
            and spec.learning_rate == row["learning_rate"]
            and spec.actual_state_noise_std == row["actual_state_noise_std"]
        )
        _write_result(target, 0.0, -100.0, status="failed")
    with pytest.raises(RuntimeError, match="no complete five-seed"):
        select_hyperparameters(sentinel, fanout, config)


def test_main_policy_reports_mse_but_gates_only_zero_eligible(tmp_path: Path) -> None:
    config = load_config()
    specs: list[RunSpec] = []
    for model_id in MODEL_IDS:
        for seed in range(10):
            spec = RunSpec(
                run_id=f"{model_id}-{seed}",
                stage="main",
                model_id=model_id,
                model_seed=seed,
                learning_rate=1e-3,
                actual_state_noise_std=0.0,
                updates=5000,
                batch_size=64,
                evaluation_bank=str(tmp_path / "bank.npz"),
                output_dir=str(tmp_path / model_id / str(seed)),
            )
            specs.append(spec)
            _write_result(spec, 0.02, -21.0 if seed == 0 else -10.0)
    summary = summarize_main(specs, config)
    assert summary["all_required_gates_pass"] is True
    for row in summary["models"].values():
        assert row["mse_success_count"] == 0
        assert row["analysis_eligible_count"] == 1
        assert row["low_eligible_count_warning"] is True
    assert scientific_gate(summary)["all_required_gates_pass"] is True
    victim = specs[0]
    _write_result(victim, 0.02, -10.0)
    failed = summarize_main(specs, config)
    assert failed["all_required_gates_pass"] is False


def test_worker_receipt_exact_set_and_checkpoint_identity(tmp_path: Path) -> None:
    config = load_config()
    root = tmp_path / "campaign"
    bank_path = root / "banks" / "smoke.npz"
    bank = source_angular_integration(
        16, 32999, stream_key=(CAMPAIGN_ID, "fixed_smoke_bank")
    )
    save_fixed_bank(bank_path, bank)
    config_path = Path(__file__).parents[1] / "source_repaired_baselines_v6.json"
    atomic_json(root / ROOT_MARKER, _scientific_identity(config_path, require_clean=False))
    (root / "inputs").mkdir(parents=True)
    shutil.copy2(
        config_path,
        root / "inputs" / "source_repaired_baselines_v6.json",
    )
    spec = RunSpec(
        run_id="receipt-smoke",
        stage="smoke",
        model_id=MODEL_IDS[0],
        model_seed=999,
        learning_rate=1e-3,
        actual_state_noise_std=0.0,
        updates=1,
        batch_size=2,
        evaluation_bank=str(bank_path),
        output_dir=str(root / "smoke" / "runs" / "receipt-smoke"),
        smoke=True,
    )
    _train_worker(spec, config_path, "cpu")
    assert _verified_child(spec)

    receipt_path = Path(spec.output_dir) / "completion_receipt.json"
    receipt = strict_json_load(receipt_path)
    receipt["artifacts"]["unexpected.txt"] = "0" * 64
    atomic_json(receipt_path, receipt)
    assert not _verified_child(spec)

    del receipt["artifacts"]["unexpected.txt"]
    checkpoint_path = Path(spec.output_dir) / "checkpoint_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["state_dict"] = {"garbage": torch.tensor([float("nan")])}
    torch.save(checkpoint, checkpoint_path)
    receipt["artifacts"]["checkpoint_final.pt"] = sha256_file(checkpoint_path)
    atomic_json(receipt_path, receipt)
    assert not _verified_child(spec)


def test_real_cli_cpu_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    root = tmp_path / "cli-smoke"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "repro.sagodi_protocol.source_repaired_baselines_v6",
            "--stage",
            "smoke",
            "--artifact-root",
            str(root),
            "--gpus",
            "cpu",
        ],
        cwd=repo,
        env=environment,
        check=True,
        timeout=120,
    )
    assert _stage_valid(root / "smoke", "smoke", 3)
