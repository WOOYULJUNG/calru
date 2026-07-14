from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from repro.sagodi_protocol.artifacts import (
    atomic_bytes,
    atomic_json,
    sha256_file,
    write_completion_receipt,
)
from repro.sagodi_protocol.lr_selection_v3 import (
    EXPECTED_LEARNING_RATES,
    EXPECTED_MODEL_IDS,
    EXPECTED_RUN_COUNT,
    EXPECTED_SELECTION_RULE,
    IncompleteSelectionError,
    PROTOCOL_COPY,
    SCOPE,
    SelectorSpecError,
    _receipt_metadata,
    _recover_training_attempt,
    _training_output,
    _training_command,
    _verify_training_output,
    aggregate_selection,
    build_selection_plan,
    parse_selector_spec,
)


def _selector_payload() -> dict:
    widths = (206, 135, 109, 96, 96, 96)
    counts = (1001, 1002, 1003, 1004, 56834, 56834)
    return {
        "schema_version": 1,
        "campaign_id": "sagodi_six_model_lr_selection_v3",
        "scope": SCOPE,
        "models": [
            {"id": model, "hidden_width": width, "parameter_count": count}
            for model, width, count in zip(EXPECTED_MODEL_IDS, widths, counts)
        ],
        "selection_model_seeds": [1100, 1101, 1102, 1103, 1104],
        "learning_rates": list(EXPECTED_LEARNING_RATES),
        "optimizer_updates": 100,
        "batch_size": 64,
        "expected_runs": EXPECTED_RUN_COUNT,
        "selection_rule": dict(EXPECTED_SELECTION_RULE),
    }


def _rows(spec, loss_by_model_lr=None):
    loss_by_model_lr = loss_by_model_lr or {}
    rows = []
    for model_index, model in enumerate(spec.models):
        for rate_index, rate in enumerate(spec.learning_rates):
            base = loss_by_model_lr.get(
                (model.model_id, rate), float(model_index + rate_index + 1)
            )
            for seed_index, seed in enumerate(spec.selection_model_seeds):
                rows.append(
                    {
                        "model_id": model.model_id,
                        "model_seed": seed,
                        "learning_rate": rate,
                        "status": "complete",
                        "completed_updates": 100,
                        "loss_at_required_update": base + 0.01 * seed_index,
                    }
                )
    return rows


def test_selector_json_expands_exact_six_by_five_by_four_matrix():
    spec = parse_selector_spec(_selector_payload())
    plan = build_selection_plan(spec)

    assert len(plan) == 120
    assert len({run.run_id for run in plan}) == 120
    assert tuple(dict.fromkeys(run.model_id for run in plan)) == EXPECTED_MODEL_IDS
    assert {run.model_seed for run in plan} == {1100, 1101, 1102, 1103, 1104}
    assert {run.learning_rate for run in plan} == set(EXPECTED_LEARNING_RATES)
    assert all(run.required_update == 100 and run.batch_size == 64 for run in plan)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["models"].reverse(), "model ids/order"),
        (
            lambda value: value["selection_model_seeds"].__setitem__(
                4, value["selection_model_seeds"][0]
            ),
            "unique",
        ),
        (
            lambda value: value["learning_rates"].__setitem__(0, 0.02),
            "learning rates/order",
        ),
        (lambda value: value.__setitem__("optimizer_updates", 99), "exactly 100"),
        (lambda value: value.__setitem__("batch_size", 32), "exactly 64"),
        (
            lambda value: value["selection_rule"].__setitem__(
                "tie_break", "larger_numeric_learning_rate"
            ),
            "selection_rule",
        ),
    ],
)
def test_selector_json_rejects_matrix_or_rule_tamper(mutate, message):
    payload = copy.deepcopy(_selector_payload())
    mutate(payload)
    with pytest.raises(SelectorSpecError, match=message):
        parse_selector_spec(payload)


def test_parameter_counts_are_declarative_not_hard_coded():
    payload = _selector_payload()
    payload["models"][3]["parameter_count"] = 987654
    spec = parse_selector_spec(payload)
    run = next(item for item in build_selection_plan(spec) if item.model_id == "lru_param96")
    assert run.parameter_count == 987654


def test_aggregation_uses_arithmetic_mean_of_update_100_losses_only():
    spec = parse_selector_spec(_selector_payload())
    losses = {}
    for model in EXPECTED_MODEL_IDS:
        losses[(model, 1e-2)] = 4.0
        losses[(model, 1e-3)] = 1.0
        losses[(model, 1e-4)] = 2.0
        losses[(model, 1e-5)] = 3.0
    rows = _rows(spec, losses)
    for row in rows:
        # These fields must never influence the selector.
        row["validation_nmse_db"] = -1000.0 if row["learning_rate"] == 1e-2 else 1000.0

    result = aggregate_selection(rows, spec, strict=True)

    assert result["complete"] is True
    assert len(result["winners"]) == 6
    for winner in result["winners"].values():
        assert winner["learning_rate"] == pytest.approx(1e-3)
        assert winner["mean_online_training_loss_at_update_100"] == pytest.approx(1.02)


def test_exact_mean_tie_uses_smaller_numeric_learning_rate():
    spec = parse_selector_spec(_selector_payload())
    losses = {}
    for model in EXPECTED_MODEL_IDS:
        losses[(model, 1e-2)] = 5.0
        losses[(model, 1e-3)] = 1.0
        losses[(model, 1e-4)] = 1.0
        losses[(model, 1e-5)] = 4.0
    result = aggregate_selection(_rows(spec, losses), spec, strict=True)
    assert all(
        winner["learning_rate"] == pytest.approx(1e-4)
        for winner in result["winners"].values()
    )


@pytest.mark.parametrize("failure", ["missing", "failed", "nonfinite"])
def test_strict_aggregation_requires_every_matrix_cell(failure):
    spec = parse_selector_spec(_selector_payload())
    rows = _rows(spec)
    target = next(
        row
        for row in rows
        if row["model_id"] == "ca_lru"
        and row["model_seed"] == 1102
        and row["learning_rate"] == 1e-3
    )
    if failure == "missing":
        rows.remove(target)
    elif failure == "failed":
        target["status"] = "failed"
    else:
        target["loss_at_required_update"] = float("nan")

    with pytest.raises(IncompleteSelectionError, match="120 unique successful"):
        aggregate_selection(rows, spec, strict=True)


def _minimal_manifest(smoke=False):
    return {
        "scientific_identity": "a" * 64,
        "protocol_canonical_fingerprint": "b" * 64,
        "smoke": smoke,
        "evaluation_bank": {"sha256": "c" * 64, "path": "evaluation_bank/bank.npz"},
    }


def _write_fake_training_output(output: Path, run, manifest, root: Path):
    output.mkdir(parents=True)
    atomic_bytes(output / "checkpoint.pt", b"checkpoint")
    steps = 2 if manifest["smoke"] else 100
    np.savez_compressed(
        output / "training_trace.npz",
        step=np.arange(1, steps + 1, dtype=np.int64),
        masked_mse=np.linspace(1.0, 0.25, steps, dtype=np.float32),
    )
    atomic_json(output / "task_metrics.json", {"train_loss_last": 0.25})
    atomic_json(output / "rp_trace.json", [])
    state_hash = None
    if not manifest["smoke"]:
        state = root / "phase0" / f"model={run.model_id}" / "state_spec.json"
        atomic_json(state, {"schema_version": 1})
        state_hash = sha256_file(state)
    model_metadata = {
        "model_config": {"width": run.hidden_width},
        "parameters_total": run.parameter_count,
    }
    atomic_json(
        output / "config.json",
        {
            "train_spec": {
                "model_name": run.model_id,
                "model_seed": run.model_seed,
                "learning_rate": run.learning_rate,
            },
            "training": {
                "steps": steps,
                "batch_size": 4 if manifest["smoke"] else 64,
                "rp_enabled_by_protocol": False,
                "expected_rp_steps": [],
            },
            "model": model_metadata,
        },
    )
    atomic_json(
        output / "manifest.json",
        {
            "campaign_identity": manifest["scientific_identity"],
            "protocol_canonical_fingerprint": manifest[
                "protocol_canonical_fingerprint"
            ],
            "model_id": run.model_id,
            "model_seed": run.model_seed,
            "parameter_count": run.parameter_count,
            "evaluation_bank_sha256": manifest["evaluation_bank"]["sha256"],
            "state_spec_sha256": state_hash,
            "rp_schedule": {"expected_steps": [], "actual_steps": [], "calls": 0},
        },
    )
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=run.receipt_job_id,
        artifacts=[
            output / name
            for name in (
                "config.json",
                "checkpoint.pt",
                "training_trace.npz",
                "task_metrics.json",
                "rp_trace.json",
                "manifest.json",
            )
        ],
        metadata=_receipt_metadata(manifest, "training", run.run_id),
    )


def test_training_verifier_binds_declared_parameter_count_and_receipt(tmp_path: Path):
    spec = parse_selector_spec(_selector_payload())
    run = build_selection_plan(spec)[0]
    manifest = _minimal_manifest()
    output = tmp_path / "output"
    _write_fake_training_output(output, run, manifest, tmp_path)

    valid, reason, bindings = _verify_training_output(
        output, run, manifest, root=tmp_path
    )
    assert valid, reason
    assert bindings is not None

    config = json.loads((output / "config.json").read_text())
    config["model"]["parameters_total"] += 1
    atomic_json(output / "config.json", config)
    valid, reason, _ = _verify_training_output(output, run, manifest, root=tmp_path)
    assert not valid
    assert "hash mismatch" in reason


def test_valid_unpublished_attempt_is_recovered_without_retraining(tmp_path: Path):
    spec = parse_selector_spec(_selector_payload())
    run = build_selection_plan(spec)[0]
    manifest = _minimal_manifest()
    attempt = (
        tmp_path
        / "attempts"
        / "lr_selection_training"
        / run.run_id
        / "attempt-crash-window"
    )
    _write_fake_training_output(attempt, run, manifest, tmp_path)

    recovered, reason = _recover_training_attempt(tmp_path, run, manifest)

    assert recovered, reason
    assert _training_output(tmp_path, run).is_dir()
    assert not attempt.exists()


def test_training_command_is_training_only_and_never_invokes_analysis(tmp_path: Path):
    spec = parse_selector_spec(_selector_payload())
    run = build_selection_plan(spec)[0]
    manifest = _minimal_manifest()
    command = _training_command(tmp_path, tmp_path / "attempt", run, manifest, "python")

    assert command[:3] == ["python", "-m", "repro.sagodi_protocol.train"]
    assert "analysis" not in " ".join(command).lower()
    assert "--state-spec" in command
    assert str(tmp_path / PROTOCOL_COPY) in command
