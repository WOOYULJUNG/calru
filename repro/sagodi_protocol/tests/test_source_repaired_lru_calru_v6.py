from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import torch

import repro.sagodi_protocol.source_repaired_lru_calru_v6 as downstream_v6
from repro.sagodi_protocol.artifacts import (
    atomic_json,
    canonical_hash,
    derived_seed,
    sha256_file,
    strict_json_load,
)
from repro.sagodi_protocol.metrics import masked_mse
from repro.sagodi_protocol.primary_v4 import _configure_determinism, build_v4_model
from repro.sagodi_protocol.source_repaired_baselines_v6 import (
    CAMPAIGN_ID as BASELINE_CAMPAIGN_ID,
    DEFAULT_CONFIG as BASELINE_CONFIG,
    load_config as load_baseline_config,
)
from repro.sagodi_protocol.source_repaired_lru_calru_v6 import (
    DEFAULT_CONFIG,
    FREEZE_DOCUMENT,
    GRADIENT_COUNTS,
    ROOT_MARKER,
    TOTAL_COUNTS,
    RunSpec,
    _git_state,
    _expected_rp_updates,
    _runtime_files,
    _smoke_plan,
    _source_q1,
    _train_worker,
    _validate_spec,
    _verified,
    _write_plan,
    build_lr_fanout_plan,
    build_lr_sentinel_plan,
    build_main_plan,
    build_rp_fanout_plan,
    build_rp_sentinel_plan,
    load_config,
    screen_rp_sentinels,
    screen_sentinels,
    select_lr_noise,
    select_rp,
    summarize_main,
)
from repro.sagodi_protocol.source_resolved_protocol import source_angular_integration
from repro.sagodi_protocol.tasks import save_fixed_bank
from repro.sagodi_protocol.train import _retention_plasticity_call


def _write_result(
    spec: RunSpec,
    mse: float,
    nmse: float,
    *,
    blank: float | None = None,
    status: str = "completed",
) -> None:
    output = Path(spec.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "result.json",
        {
            "run_id": spec.run_id,
            "status": status,
            "final_metrics": {"mse": mse, "nmse_db": nmse} if status == "completed" else None,
            "heldout_blank_memory_mse": blank,
        },
    )


def test_config_freezes_dependency_graph_and_fairness() -> None:
    config = load_config()
    assert config["task_and_data_contract"]["online_stream_namespace"] == BASELINE_CAMPAIGN_ID
    assert config["dependencies"]["no_rp_sentinel_requires"] == "lru_main_scientific_pass"
    assert config["dependencies"]["ca_rp_sentinel_requires"].startswith("verified_no_rp_main")
    assert config["ca_fairness"]["inherits_lr_noise_from"] == "no_rp_n52"
    assert config["ca_fairness"]["ca_lr_noise_independently_tuned"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda cfg: cfg["training"].update(updates=100),
        lambda cfg: cfg["main"].update(analysis_eligibility_nmse_db_threshold=-5),
        lambda cfg: cfg["dependencies"].update(no_rp_sentinel_requires="nothing"),
        lambda cfg: cfg["retention_plasticity"].update(warmup_updates=0),
        lambda cfg: cfg["task_and_data_contract"].update(online_stream_namespace="wrong"),
    ],
)
def test_downstream_config_rejects_major_mutations(tmp_path: Path, mutation) -> None:
    config = deepcopy(load_config())
    mutation(config)
    path = tmp_path / "mutated.json"
    atomic_json(path, config)
    with pytest.raises(ValueError, match="canonical frozen contract"):
        load_config(path)


def test_model_counts_and_causal_noise_dimensions() -> None:
    for model_id in TOTAL_COUNTS:
        model = build_v4_model(model_id)
        assert sum(p.numel() for p in model.parameters()) == TOTAL_COUNTS[model_id]
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == GRADIENT_COUNTS[model_id]
    assert build_v4_model("lru_n52").metadata()["primary_state_size"] == 104
    assert build_v4_model("no_rp_n52").metadata()["primary_state_size"] == 52
    assert build_v4_model("ca_lru_n52").metadata()["primary_state_size"] == 52


def test_lru_plans_are_independent_and_nmse_first(tmp_path: Path) -> None:
    config = load_config()
    bank = tmp_path / "tuning.npz"
    sentinel = build_lr_sentinel_plan(tmp_path, config, bank, "lru")
    assert len(sentinel) == 16
    assert {spec.model_id for spec in sentinel} == {"lru_n52"}
    assert {spec.model_seed for spec in sentinel} == {100}
    for index, spec in enumerate(sentinel):
        _write_result(spec, 0.002 + index * 1e-4, -10.0)
    screening = screen_sentinels(sentinel, config)
    fanout = build_lr_fanout_plan(tmp_path, config, bank, "lru", screening)
    assert len(fanout) == 12
    assert {spec.model_seed for spec in fanout} == {101, 102, 103, 104}

    cells = screening["top_cells"]
    for spec in fanout:
        cell_index = next(
            i
            for i, cell in enumerate(cells)
            if cell["learning_rate"] == spec.learning_rate
            and cell["actual_state_noise_std"] == spec.actual_state_noise_std
        )
        # Cell 0 has better/descriptive MSE but no eligible seed. Cell 1 has
        # worse MSE and five eligible seeds, so NMSE-first must select cell 1.
        if cell_index == 0:
            _write_result(spec, 0.001, -10.0)
        elif cell_index == 1:
            _write_result(spec, 0.02, -25.0)
        else:
            _write_result(spec, 0.03, -8.0)
    # Include the sentinel seed in the same intended pattern.
    for i, cell in enumerate(cells):
        spec = next(
            row
            for row in sentinel
            if row.learning_rate == cell["learning_rate"]
            and row.actual_state_noise_std == cell["actual_state_noise_std"]
        )
        _write_result(spec, 0.001 if i == 0 else 0.02, -10.0 if i != 1 else -25.0)
    selection = select_lr_noise(sentinel, fanout, config)
    assert selection["winner"]["learning_rate"] == cells[1]["learning_rate"]
    assert selection["winner"]["actual_state_noise_std"] == cells[1]["actual_state_noise_std"]


def test_missing_seed_cell_is_never_selected(tmp_path: Path) -> None:
    config = load_config()
    bank = tmp_path / "tuning.npz"
    sentinel = build_lr_sentinel_plan(tmp_path, config, bank, "lru")
    for index, spec in enumerate(sentinel):
        _write_result(spec, 0.002 + index * 1e-4, -25.0)
    screen = screen_sentinels(sentinel, config)
    fanout = build_lr_fanout_plan(tmp_path, config, bank, "lru", screen)
    for spec in fanout:
        _write_result(spec, 0.003, -24.0)
    best = screen["top_cells"][0]
    broken = next(
        spec
        for spec in fanout
        if spec.learning_rate == best["learning_rate"]
        and spec.actual_state_noise_std == best["actual_state_noise_std"]
    )
    _write_result(broken, 0.0, -100.0, status="failed")
    selected = select_lr_noise(sentinel, fanout, config)
    assert (
        selected["winner"]["learning_rate"],
        selected["winner"]["actual_state_noise_std"],
    ) != (broken.learning_rate, broken.actual_state_noise_std)


def test_ca_inherits_no_rp_lr_noise_and_only_rp_is_tuned(tmp_path: Path) -> None:
    config = load_config()
    bank = tmp_path / "tuning.npz"
    no_rp = {"winner": {"learning_rate": 0.003, "actual_state_noise_std": 0.01}}
    sentinel = build_rp_sentinel_plan(tmp_path, config, bank, no_rp)
    assert len(sentinel) == 9
    assert {spec.learning_rate for spec in sentinel} == {0.003}
    assert {spec.actual_state_noise_std for spec in sentinel} == {0.01}
    for index, spec in enumerate(sentinel):
        _write_result(spec, 0.004, -23.0, blank=0.1 + index * 0.01)
    screening = screen_rp_sentinels(sentinel, config)
    fanout = build_rp_fanout_plan(tmp_path, config, bank, no_rp, screening)
    assert len(fanout) == 12
    for spec in fanout:
        _write_result(spec, 0.004, -23.0, blank=0.2)
    selected = select_rp(sentinel, fanout, config)
    main = build_main_plan(
        tmp_path,
        config,
        tmp_path / "main_test.npz",
        "ca_rp",
        "ca_lru_n52",
        no_rp,
        rp=selected["winner"],
    )
    assert len(main) == 10
    assert {spec.learning_rate for spec in main} == {0.003}
    assert {spec.actual_state_noise_std for spec in main} == {0.01}
    assert all(spec.rp_enabled for spec in main)


def test_main_gate_needs_one_eligible_seed_only(tmp_path: Path) -> None:
    config = load_config()
    specs = build_main_plan(
        tmp_path,
        config,
        tmp_path / "main_test.npz",
        "lru",
        "lru_n52",
        {"winner": {"learning_rate": 0.001, "actual_state_noise_std": 0.0}},
    )
    for spec in specs:
        _write_result(spec, 0.02, -21.0 if spec.model_seed == 0 else -10.0)
    summary = summarize_main(specs, config)
    assert summary["scientific_pass"] is True
    assert summary["analysis_eligible_count"] == 1
    assert summary["mse_success_count_descriptive"] == 0


def test_no_rp_ca_are_identical_through_one_pre_rp_update() -> None:
    seed = 17
    _configure_determinism(seed)
    no_rp = build_v4_model("no_rp_n52")
    _configure_determinism(seed)
    ca = build_v4_model("ca_lru_n52")
    assert no_rp.state_dict().keys() == ca.state_dict().keys()
    for key in no_rp.state_dict():
        assert torch.equal(no_rp.state_dict()[key], ca.state_dict()[key])

    batch = source_angular_integration(
        4,
        0,
        stream_key=(BASELINE_CAMPAIGN_ID, "online_train", seed, 1),
    )
    optimizers = [
        torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        for model in (no_rp, ca)
    ]
    for model, optimizer in zip((no_rp, ca), optimizers):
        generator = torch.Generator().manual_seed(
            derived_seed(seed, BASELINE_CAMPAIGN_ID, "state_noise")
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_sequence(
            batch.inputs,
            initial_memory=_source_q1(batch),
            state_noise_std=0.01,
            noise_generator=generator,
        )
        loss = masked_mse(prediction, batch.output_targets, batch.mask)
        loss.backward()
        optimizer.step()
    for key in no_rp.state_dict():
        assert torch.equal(no_rp.state_dict()[key], ca.state_dict()[key])


def test_rp_schedule_and_reduced_call_are_exact_and_finite(tmp_path: Path) -> None:
    config = load_config()
    full = RunSpec(
        run_id="ca-rp-full",
        stage="ca_rp_main",
        model_id="ca_lru_n52",
        model_seed=0,
        learning_rate=1e-3,
        actual_state_noise_std=0.01,
        updates=5000,
        batch_size=64,
        evaluation_bank=str(tmp_path / "main_test.npz"),
        campaign_root=str(tmp_path),
        output_dir=str(tmp_path / "ca_rp_main" / "runs" / "ca-rp-full"),
        rp_enabled=True,
        rp_eta_lambda=300.0,
        rp_damage_epsilon=1e-5,
    )
    expected = _expected_rp_updates(full, config)
    assert len(expected) == 70
    assert expected[0] == 1550 and expected[-1] == 5000
    assert _expected_rp_updates(
        RunSpec(**{**full.payload(), "model_id": "no_rp_n52", "rp_enabled": False,
                   "rp_eta_lambda": None, "rp_damage_epsilon": None}),
        config,
    ) == ()

    _configure_determinism(71)
    model = build_v4_model("ca_lru_n52")
    batch = source_angular_integration(2, 0, stream_key=("rp_unit", 71))
    recurrences = list(model.pan_recs_with_slices())
    before = [recurrence.theta.clone() for recurrence, _ in recurrences]
    result = _retention_plasticity_call(
        model,
        batch,
        blank_horizon=2,
        eta_lambda=300.0,
        damage_epsilon=1e-5,
        initial_memory=_source_q1(batch),
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
    assert all(
        torch.isfinite(recurrence.theta).all()
        and torch.count_nonzero(recurrence.theta - old).item() == old.numel()
        for old, (recurrence, _) in zip(before, recurrences)
    )


def test_worker_rejects_smoke_bypass_on_full_stage(tmp_path: Path) -> None:
    invalid = RunSpec(
        run_id="malicious-short-main",
        stage="lru_main",
        model_id="lru_n52",
        model_seed=0,
        learning_rate=1e-3,
        actual_state_noise_std=0.0,
        updates=1,
        batch_size=2,
        evaluation_bank=str(tmp_path / "main_test.npz"),
        campaign_root=str(tmp_path.resolve()),
        output_dir=str(tmp_path / "lru_main" / "runs" / "malicious-short-main"),
        smoke=True,
    )
    with pytest.raises(ValueError, match="smoke=True"):
        _train_worker(invalid, DEFAULT_CONFIG, "cpu")


def test_parent_stage_rejects_corrupted_receipt_and_child_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dependency gate consumes a verified parent, not just its plan."""

    root = tmp_path.resolve()
    (root / "inputs").mkdir(parents=True)
    shutil.copy2(DEFAULT_CONFIG, root / "inputs" / DEFAULT_CONFIG.name)
    shutil.copy2(BASELINE_CONFIG, root / "inputs" / BASELINE_CONFIG.name)
    atomic_json(root / "baseline_parent_binding.json", {"unit_test": True})
    baseline = load_baseline_config()
    contract = baseline["evaluation_banks"]["tuning"]
    bank_path = root / "banks" / "tuning.npz"
    save_fixed_bank(
        bank_path,
        source_angular_integration(
            int(contract["trials"]),
            int(contract["task_seed"]),
            stream_key=contract["stream_key"],
        ),
    )
    config = load_config()
    specs = build_lr_sentinel_plan(root, config, bank_path, "lru")
    _write_plan(root / "lru_sentinel", specs)
    for index, spec in enumerate(specs):
        _write_result(spec, 0.01 + index * 1e-4, -21.0)
        output = Path(spec.output_dir)
        torch.save({"unit_test": True}, output / "checkpoint_final.pt")
        atomic_json(output / "completion_receipt.json", {"unit_test": True})
    monkeypatch.setattr(downstream_v6, "_verified", lambda spec: True)
    downstream_v6._finish_stage(
        root,
        "lru_sentinel",
        specs,
        "screening.json",
        screen_sentinels(specs, config),
    )
    assert downstream_v6._stage_valid(root, "lru_sentinel", 16)

    receipt_path = root / "lru_sentinel" / "completion_receipt.json"
    receipt = strict_json_load(receipt_path)
    corrupted_receipt = deepcopy(receipt)
    corrupted_receipt["metadata"]["stage"] = "corrupted"
    atomic_json(receipt_path, corrupted_receipt)
    assert not downstream_v6._stage_valid(root, "lru_sentinel", 16)
    atomic_json(receipt_path, receipt)

    child_result = Path(specs[0].output_dir) / "result.json"
    payload = strict_json_load(child_result)
    payload["final_metrics"]["nmse_db"] = -99.0
    atomic_json(child_result, payload)
    assert not downstream_v6._stage_valid(root, "lru_sentinel", 16)


def test_downstream_worker_smoke_and_checkpoint_adversary(tmp_path: Path) -> None:
    root = tmp_path / "downstream"
    (root / "inputs").mkdir(parents=True)
    shutil.copy2(DEFAULT_CONFIG, root / "inputs" / DEFAULT_CONFIG.name)
    shutil.copy2(BASELINE_CONFIG, root / "inputs" / BASELINE_CONFIG.name)
    bank_path = root / "banks" / "tuning.npz"
    baseline = load_baseline_config()
    contract = baseline["evaluation_banks"]["tuning"]
    bank = source_angular_integration(
        int(contract["trials"]),
        int(contract["task_seed"]),
        stream_key=contract["stream_key"],
    )
    save_fixed_bank(bank_path, bank)
    identity = {
        "schema_version": 1,
        "campaign_id": load_config()["campaign_id"],
        "protocol_revision": load_config()["protocol_revision"],
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "freeze_sha256": sha256_file(FREEZE_DOCUMENT),
        "runtime_code_sha256": {path.name: sha256_file(path) for path in _runtime_files()},
        "code_commit": _git_state(False),
        "parent_binding": {"unit_test": True},
    }
    identity["scientific_identity"] = canonical_hash(identity)
    atomic_json(root / ROOT_MARKER, identity)
    atomic_json(root / "baseline_parent_binding.json", identity["parent_binding"])
    smoke_specs = _smoke_plan(root, bank_path)
    _write_plan(root / "smoke", smoke_specs)
    spec = next(row for row in smoke_specs if row.model_id == "lru_n52")
    _train_worker(spec, DEFAULT_CONFIG, "cpu")
    assert _verified(spec)
    manifest = strict_json_load(Path(spec.output_dir) / "run_manifest.json")
    assert manifest["downstream_initial_state_semantics"] == "source_q1_post_update_target"
    assert manifest["state_noise_location"] == "post_transition_full_104d_real_imag_carrier"
    assert "y0" not in manifest["model"]["initial_state"]

    ca_spec = next(row for row in smoke_specs if row.model_id == "ca_lru_n52")
    _train_worker(ca_spec, DEFAULT_CONFIG, "cpu")
    assert _verified(ca_spec)
    ca_result = strict_json_load(Path(ca_spec.output_dir) / "result.json")
    assert strict_json_load(Path(ca_spec.output_dir) / "rp_trace.json") == []
    assert ca_result["rp_call_count"] == 0
    assert ca_result["heldout_blank_memory_mse"] is None

    checkpoint_path = Path(spec.output_dir) / "checkpoint_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["state_dict"] = {"garbage": torch.tensor([float("nan")])}
    torch.save(checkpoint, checkpoint_path)
    receipt_path = Path(spec.output_dir) / "completion_receipt.json"
    receipt = strict_json_load(receipt_path)
    receipt["artifacts"]["checkpoint_final.pt"] = sha256_file(checkpoint_path)
    atomic_json(receipt_path, receipt)
    assert not _verified(spec)
