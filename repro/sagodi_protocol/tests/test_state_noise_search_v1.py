from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from repro.sagodi_protocol.artifacts import (
    atomic_json,
    canonical_hash,
    sha256_file,
    strict_json_load,
)
from repro.sagodi_protocol.source_resolved_protocol import source_angular_integration
from repro.sagodi_protocol.state_noise_search_v1 import (
    BASELINE_CONFIG,
    CAMPAIGN_ID,
    CONFIG_CONTRACT_SHA256,
    DEFAULT_CONFIG,
    DOWNSTREAM_CONFIG,
    FREEZE_DOCUMENT,
    MODEL_IDS,
    PROTOCOL_REVISION,
    ROOT_MARKER,
    TRACK_CLASSIFICATION,
    _git_state,
    _rng_identities,
    _runtime_files,
    _train_worker,
    _verified,
    build_main_plan,
    build_smoke_plan,
    build_tuning_plan,
    load_config,
    select_state_noise,
)
from repro.sagodi_protocol.tasks import save_fixed_bank


def _parent() -> dict:
    selected = {}
    rates = [0.001, 0.001, 0.0003, 0.01]
    for model_id, rate in zip(MODEL_IDS, rates):
        selected[model_id] = {
            "learning_rate": rate,
            "learning_rate_source": "unit_test",
        }
    return {"schema_version": 1, "selected_hyperparameters": selected}


def _write_result(spec, mse: float, nmse: float, *, status: str = "completed") -> None:
    output = Path(spec.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "result.json",
        {
            "run_id": spec.run_id,
            "status": status,
            "final_metrics": (
                {"mse": mse, "nmse_db": nmse} if status == "completed" else None
            ),
        },
    )


def test_config_and_plan_denominators_are_frozen(tmp_path: Path) -> None:
    config = load_config()
    assert canonical_hash(strict_json_load(DEFAULT_CONFIG)) == CONFIG_CONTRACT_SHA256
    assert config["training"]["target_noise_std"] == 0.0
    assert config["training"]["output_dropout"] == 0.0
    assert config["state_noise_search"]["actual_post_transition_state_noise_std_grid"] == [
        0.0,
        0.003,
        0.01,
        0.0316228,
        0.1,
    ]
    parent = _parent()
    smoke = build_smoke_plan(tmp_path, config, parent, tmp_path / "tuning.npz")
    tuning = build_tuning_plan(tmp_path, config, parent, tmp_path / "tuning.npz")
    assert len(smoke) == 4
    assert len(tuning) == 100
    assert all(spec.learning_rate == parent["selected_hyperparameters"][spec.model_id]["learning_rate"] for spec in tuning)
    for model_id in MODEL_IDS:
        rows = [spec for spec in tuning if spec.model_id == model_id]
        assert len(rows) == 25
        assert {spec.model_seed for spec in rows} == {100, 101, 102, 103, 104}
        assert {spec.actual_state_noise_std for spec in rows} == {
            0.0,
            0.003,
            0.01,
            0.0316228,
            0.1,
        }


def test_selection_uses_all_five_seeds_and_main_is_fresh(tmp_path: Path) -> None:
    config, parent = load_config(), _parent()
    tuning = build_tuning_plan(tmp_path, config, parent, tmp_path / "tuning.npz")
    for spec in tuning:
        distance = abs(spec.actual_state_noise_std - 0.01)
        _write_result(spec, 0.002 + distance, -25.0 + 100.0 * distance)
    selection = select_state_noise(tuning, config, parent)
    assert {
        row["winner"]["actual_state_noise_std"]
        for row in selection["models"].values()
    } == {0.01}
    main = build_main_plan(
        tmp_path, config, parent, tmp_path / "main_test.npz", selection
    )
    assert len(main) == 40
    assert {spec.model_seed for spec in main} == set(range(10))
    assert {spec.actual_state_noise_std for spec in main} == {0.01}

    victim = next(
        spec
        for spec in tuning
        if spec.model_id == MODEL_IDS[0]
        and spec.actual_state_noise_std == 0.01
        and spec.model_seed == 104
    )
    _write_result(victim, 0.0, -100.0, status="failed")
    changed = select_state_noise(tuning, config, parent)
    assert changed["models"][MODEL_IDS[0]]["winner"]["actual_state_noise_std"] != 0.01

    for spec in tuning:
        _write_result(
            spec,
            0.001 + spec.actual_state_noise_std,
            -30.0 + 100.0 * spec.actual_state_noise_std,
        )
    zero_best = select_state_noise(tuning, config, parent)
    assert all(
        row["overall_winner"]["actual_state_noise_std"] == 0.0
        and row["overall_winner_is_zero"] is True
        and row["positive_noise_winner"]["actual_state_noise_std"] > 0.0
        for row in zero_best["models"].values()
    )


def test_zero_noise_disables_generator_and_positive_noise_pairs_stream(tmp_path: Path) -> None:
    config, parent = load_config(), _parent()
    tuning = build_tuning_plan(tmp_path, config, parent, tmp_path / "tuning.npz")
    zero = next(spec for spec in tuning if spec.model_id == MODEL_IDS[0] and spec.actual_state_noise_std == 0.0)
    positive = [
        spec
        for spec in tuning
        if spec.model_id == MODEL_IDS[0]
        and spec.model_seed == 100
        and spec.actual_state_noise_std > 0
    ]
    zero_rng = _rng_identities(zero)["state_noise"]
    assert zero_rng["enabled"] is False and zero_rng["generator_seed"] is None
    seeds = {_rng_identities(spec)["state_noise"]["generator_seed"] for spec in positive}
    assert len(seeds) == 1


def _worker_root(tmp_path: Path) -> tuple[Path, Path, dict]:
    root = tmp_path / "worker-root"
    (root / "inputs").mkdir(parents=True)
    shutil.copy2(DEFAULT_CONFIG, root / "inputs" / DEFAULT_CONFIG.name)
    shutil.copy2(BASELINE_CONFIG, root / "inputs" / BASELINE_CONFIG.name)
    shutil.copy2(DOWNSTREAM_CONFIG, root / "inputs" / DOWNSTREAM_CONFIG.name)
    bank_path = root / "banks" / "tuning.npz"
    save_fixed_bank(
        bank_path,
        source_angular_integration(
            16,
            32999,
            stream_key=("sagodi_source_repaired_baselines_v6", "fixed_smoke_bank"),
        ),
    )
    parent = _parent()
    identity = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "track_classification": TRACK_CLASSIFICATION,
        "code_commit": _git_state(False),
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "freeze_sha256": sha256_file(FREEZE_DOCUMENT),
        "runtime_code_sha256": {path.name: sha256_file(path) for path in _runtime_files()},
        "parent_binding": parent,
    }
    identity["scientific_identity"] = canonical_hash(identity)
    atomic_json(root / "parent_binding.json", parent)
    atomic_json(root / ROOT_MARKER, identity)
    return root, bank_path, parent


def test_source_and_lru_smoke_workers_record_positive_state_noise(tmp_path: Path) -> None:
    root, bank, parent = _worker_root(tmp_path)
    specs = build_smoke_plan(root, load_config(), parent, bank)
    for model_id in ("sagodi_rnn_tanh_n128", "lru_n52"):
        spec = next(row for row in specs if row.model_id == model_id)
        _train_worker(spec, DEFAULT_CONFIG, "cpu")
        assert _verified(spec)
        manifest = strict_json_load(Path(spec.output_dir) / "run_manifest.json")
        assert manifest["actual_post_transition_state_noise_std"] == 0.01
        assert manifest["target_noise_std"] == 0.0
        assert manifest["output_dropout"] == 0.0
        assert manifest["rng_stream_identities"]["state_noise"]["enabled"] is True
        assert manifest["rng_stream_identities"]["state_noise"]["generator_seed"] is not None
        checkpoint = torch.load(
            Path(spec.output_dir) / "checkpoint_final.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["rng_stream_identities"] == manifest["rng_stream_identities"]
