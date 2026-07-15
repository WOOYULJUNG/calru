from __future__ import annotations

import shutil
from pathlib import Path

from repro.sagodi_protocol.artifacts import atomic_json, canonical_hash, sha256_file, strict_json_load
from repro.sagodi_protocol.calru_factorial_v1 import (
    BASELINE_CONFIG,
    CAMPAIGN_ID,
    CONFIG_CONTRACT_SHA256,
    DEFAULT_CONFIG,
    FREEZE_DOCUMENT,
    PROTOCOL_REVISION,
    ROOT_MARKER,
    _copy_bound_bank,
    _git_state,
    _rng_identities,
    _runtime_files,
    _train_worker,
    _verified,
    build_factorial_main_plan,
    build_rp_fanout_plan,
    build_rp_sentinel_plan,
    build_smoke_plan,
    load_config,
    screen_rp_sentinels,
    select_rp,
)
from repro.sagodi_protocol.source_resolved_protocol import source_angular_integration
from repro.sagodi_protocol.tasks import load_fixed_bank, save_fixed_bank


def _parent() -> dict:
    return {"learning_rate": 0.001, "positive_state_noise_std": 0.01}


def _write_result(spec, score: float) -> None:
    output = Path(spec.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "result.json", {"run_id": spec.run_id, "status": "completed", "final_metrics": {"mse": score, "nmse_db": -30.0 + score}, "blank_memory_mse": score})


def test_config_and_plan_denominators(tmp_path: Path) -> None:
    config, parent = load_config(), _parent()
    assert canonical_hash(strict_json_load(DEFAULT_CONFIG)) == CONFIG_CONTRACT_SHA256
    smoke = build_smoke_plan(tmp_path, config, parent, tmp_path / "tuning.npz")
    sentinel = build_rp_sentinel_plan(tmp_path, config, parent, tmp_path / "tuning.npz")
    assert len(smoke) == 4
    assert len(sentinel) == 27
    assert {spec.rp_interval_updates for spec in sentinel} == {25, 50, 100}
    assert {spec.learning_rate for spec in (*smoke, *sentinel)} == {0.001}


def test_copy_bound_bank_keeps_mandatory_checksum_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "source" / "tuning.npz"
    destination = tmp_path / "destination" / "tuning.npz"
    save_fixed_bank(
        source,
        source_angular_integration(
            16, 32999, stream_key=(CAMPAIGN_ID, "copy_bound_bank")
        ),
    )

    _copy_bound_bank(source, destination)

    assert destination.is_file()
    assert destination.with_suffix(".npz.sha256").is_file()
    assert sha256_file(destination) == sha256_file(source)
    assert sha256_file(destination.with_suffix(".npz.sha256")) == sha256_file(
        source.with_suffix(".npz.sha256")
    )
    loaded = load_fixed_bank(destination)
    assert loaded.batch_size == 16


def test_rp_selection_and_factorial_are_frozen(tmp_path: Path) -> None:
    config, parent = load_config(), _parent()
    sentinel = build_rp_sentinel_plan(tmp_path, config, parent, tmp_path / "tuning.npz")
    for index, spec in enumerate(sentinel):
        _write_result(spec, 0.001 + index / 1000.0)
    screening = screen_rp_sentinels(sentinel, config)
    assert len(screening["top_cells"]) == 5
    fanout = build_rp_fanout_plan(tmp_path, config, parent, tmp_path / "tuning.npz", screening)
    assert len(fanout) == 10
    for spec in fanout:
        _write_result(spec, 0.002)
    selection = select_rp(sentinel, fanout, config)
    main = build_factorial_main_plan(tmp_path, config, parent, tmp_path / "main_test.npz", selection)
    assert len(main) == 12
    assert {spec.condition_id for spec in main} == {"no_rp_no_noise", "no_rp_with_noise", "rp_no_noise", "rp_with_noise"}
    assert {spec.actual_state_noise_std for spec in main} == {0.0, 0.01}
    assert len({(spec.rp_eta_lambda, spec.rp_damage_epsilon, spec.rp_interval_updates) for spec in main if spec.rp_enabled}) == 1
    paired = [spec for spec in main if spec.model_seed == 0 and spec.actual_state_noise_std > 0]
    assert len({_rng_identities(spec)["state_noise"]["generator_seed"] for spec in paired}) == 1


def test_smoke_worker_records_rp_interval_and_noise(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "inputs").mkdir(parents=True)
    shutil.copy2(DEFAULT_CONFIG, root / "inputs" / DEFAULT_CONFIG.name)
    shutil.copy2(BASELINE_CONFIG, root / "inputs" / BASELINE_CONFIG.name)
    bank = root / "banks" / "tuning.npz"
    save_fixed_bank(bank, source_angular_integration(16, 32999, stream_key=(CAMPAIGN_ID, "test_bank")))
    parent = _parent()
    identity = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "code_commit": _git_state(False),
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "freeze_sha256": sha256_file(FREEZE_DOCUMENT),
        "runtime_code_sha256": {path.name: sha256_file(path) for path in _runtime_files()},
        "parent_binding": parent,
    }
    identity["scientific_identity"] = canonical_hash(identity)
    atomic_json(root / "parent_binding.json", parent)
    atomic_json(root / ROOT_MARKER, identity)
    spec = next(row for row in build_smoke_plan(root, load_config(), parent, bank) if row.condition_id == "rp_with_noise")
    _train_worker(spec, DEFAULT_CONFIG, "cpu")
    assert _verified(spec)
    result = strict_json_load(Path(spec.output_dir) / "result.json")
    assert result["actual_state_noise_std"] == 0.01
    assert result["rp_interval_updates"] == 25
    assert result["rp_call_count"] == 0
