from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repro.sagodi_protocol.config import load_protocol, validate_protocol
from repro.sagodi_protocol.lr_selection_v3 import EXPECTED_MODEL_IDS
from repro.sagodi_protocol.primary_main_campaign import (
    CAMPAIGN_MODE,
    EXPECTED_MAIN_SEEDS,
    EXPECTED_RP_CONTRACT,
    EXPECTED_RP_STEPS,
    MainModel,
    MainRun,
    MainTemplateError,
    ParentSelector,
    ParentSelectorError,
    _expected_child_rp,
    _training_command,
    build_main_plan,
    load_main_template,
    materialize_resolved_protocol,
    verify_parent_selector,
)


PACKAGE = Path(__file__).resolve().parents[1]
TEMPLATE = PACKAGE / "primary_main_template_v3.json"
SELECTOR_PROTOCOL = PACKAGE / "sagodi_primary_lr_selection_v3.yaml"
WIDTHS = (206, 135, 109, 96, 96, 96)
COUNTS = (56844, 56432, 56438, 56834, 56834, 56834)


def _models() -> tuple[MainModel, ...]:
    rates = (0.01, 0.001, 0.0001, 0.00001, 0.001, 0.01)
    return tuple(
        MainModel(model_id, width, count, rate)
        for model_id, width, count, rate in zip(
            EXPECTED_MODEL_IDS, WIDTHS, COUNTS, rates
        )
    )


def _binding() -> dict[str, object]:
    models = _models()
    return {
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


def _parent(root: Path = Path("/selector")) -> ParentSelector:
    return ParentSelector(
        root=root,
        protocol=load_protocol(SELECTOR_PROTOCOL),
        manifest={"scientific_identity": "0" * 64},
        models=_models(),
        binding=_binding(),
    )


def test_committed_template_encodes_exact_main_contract() -> None:
    template = load_main_template(TEMPLATE)
    assert template["campaign_mode"] == CAMPAIGN_MODE
    assert tuple(template["main_model_seeds"]) == EXPECTED_MAIN_SEEDS
    assert template["training"]["optimizer_updates"] == 5000
    assert template["training"]["batch_size"] == 64
    assert template["retention_plasticity"]["calls_after_warmup"] == 70
    assert template["retention_plasticity"] == {
        "enabled_model": "ca_lru",
        **{
            key: value
            for key, value in EXPECTED_RP_CONTRACT.items()
            if key != "enabled_during_training"
        },
    }


def test_template_validation_fails_closed_on_training_change(tmp_path: Path) -> None:
    payload = json.loads(TEMPLATE.read_text())
    payload["training"]["optimizer_updates"] = 4999
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload))
    with pytest.raises(MainTemplateError, match="training contract"):
        load_main_template(changed)


@pytest.mark.parametrize(
    ("field", "changed_value"),
    (
        ("warmup_updates", 1499),
        ("interval_updates", 49),
        ("calls_after_warmup", 69),
        ("probe_batch_size", 95),
        ("probe_horizon", 255),
        ("blank_ablation_horizon", 499),
        ("probe_noise_enabled", True),
        ("eta_lambda", 2999.0),
        ("damage_epsilon", 4e-5),
    ),
)
def test_template_validation_fails_closed_on_rp_numeric_change(
    tmp_path: Path, field: str, changed_value: object
) -> None:
    payload = json.loads(TEMPLATE.read_text())
    payload["retention_plasticity"][field] = changed_value
    changed = tmp_path / f"changed_{field}.json"
    changed.write_text(json.dumps(payload))
    with pytest.raises(MainTemplateError, match="Retention Plasticity"):
        load_main_template(changed)


def test_materialized_protocol_is_valid_and_selector_bound() -> None:
    template = load_main_template(TEMPLATE)
    resolved = materialize_resolved_protocol(template, _parent())
    validate_protocol(resolved)
    phase = resolved["phase1_ring_pilot"]
    training = phase["training"]
    assert resolved["freeze_id"] == "sagodi_primary_main_v3_resolved"
    assert resolved["freeze_status"] == "resolved_before_training"
    assert phase["confirmatory"] is True
    assert phase["purpose"] == "six_model_primary_main_training"
    assert resolved["seed_policy"]["main_model_seeds"] == list(range(10))
    assert training["learning_rate"]["selected_by_model"] == {
        model.model_id: model.learning_rate for model in _models()
    }
    assert training["learning_rate"]["source"] == "selector_bound"
    assert training["gradient_clipping"]["frozen_numeric_value"] is None
    assert training["rp_schedule_for_ca_lru"] == EXPECTED_RP_CONTRACT


@pytest.mark.parametrize(
    ("field", "changed_value"),
    (("probe_batch_size", 256), ("blank_ablation_horizon", 256)),
)
def test_resolved_protocol_validator_fails_closed_on_old_rp_values(
    field: str, changed_value: int
) -> None:
    resolved = materialize_resolved_protocol(load_main_template(TEMPLATE), _parent())
    resolved["phase1_ring_pilot"]["training"]["rp_schedule_for_ca_lru"][
        field
    ] = changed_value
    with pytest.raises(ValueError, match="RP schedule"):
        validate_protocol(resolved)


def test_materialization_does_not_mutate_selector_protocol() -> None:
    parent = _parent()
    before = copy.deepcopy(parent.protocol)
    materialize_resolved_protocol(load_main_template(TEMPLATE), parent)
    assert parent.protocol == before


def test_main_plan_is_exact_ordered_six_by_ten_cross_product() -> None:
    plan = build_main_plan(_parent())
    assert len(plan) == 60
    assert len({run.run_id for run in plan}) == 60
    assert [run.model.model_id for run in plan[::10]] == list(EXPECTED_MODEL_IDS)
    assert tuple(run.model_seed for run in plan[:10]) == EXPECTED_MAIN_SEEDS
    assert all(run.model.learning_rate == _models()[0].learning_rate for run in plan[:10])


def test_rp_schedule_applies_only_to_ca_lru() -> None:
    models = {model.model_id: model for model in _models()}
    assert _expected_child_rp(MainRun(models["ca_lru"], 0), smoke=False) == EXPECTED_RP_STEPS
    assert len(EXPECTED_RP_STEPS) == 70
    assert EXPECTED_RP_STEPS[0] == 1550
    assert EXPECTED_RP_STEPS[-1] == 5000
    assert _expected_child_rp(MainRun(models["no_rp"], 0), smoke=False) == ()
    assert _expected_child_rp(MainRun(models["ca_lru"], 0), smoke=True) == (1, 2)


def test_training_command_uses_selected_lr_parent_phase0_and_no_analysis(tmp_path: Path) -> None:
    parent = _parent(tmp_path / "selector")
    run = MainRun(parent.models[-1], 7)
    manifest = {
        "evaluation_bank": {"path": "evaluation_bank/angular_integration_id.npz"},
        "scientific_identity": "a" * 64,
        "smoke": False,
    }
    command = _training_command(
        tmp_path / "main",
        tmp_path / "attempt",
        run,
        manifest,
        parent,
        "/python",
    )
    assert command[0:3] == ["/python", "-m", "repro.sagodi_protocol.train"]
    assert command[command.index("--learning-rate") + 1] == str(run.model.learning_rate)
    assert command[command.index("--model-seed") + 1] == "7"
    state_path = command[command.index("--state-spec") + 1]
    assert state_path.endswith("selector/phase0/model=ca_lru/state_spec.json")
    assert not any("analysis" in value for value in command)


def _write_parent_shell(root: Path) -> None:
    root.mkdir()
    for name in (
        "selector_spec.json",
        "protocol.yaml",
        "lr_selection_manifest.json",
        "lr_selection_summary.json",
        "lr_selection_receipt.json",
        "completion_receipt.json",
        "COMPLETE",
    ):
        (root / name).write_text("{}")


def test_parent_selector_requires_recursive_completion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import repro.sagodi_protocol.primary_main_campaign as campaign

    root = tmp_path / "selector"
    _write_parent_shell(root)
    selector_models = tuple(
        SimpleNamespace(
            model_id=model.model_id,
            hidden_width=model.hidden_width,
            parameter_count=model.parameter_count,
        )
        for model in _models()
    )
    spec = SimpleNamespace(
        models=selector_models,
        learning_rates=(0.01, 0.001, 0.0001, 0.00001),
    )
    manifest = {
        "campaign_id": "sagodi_six_model_lr_selection_v3",
        "campaign_type": "six_model_lr_selection_v3",
        "scope": "training_only_lr_selection_no_manifold_analysis_no_ca_evidence",
        "smoke": False,
        "scientific_identity": "a" * 64,
        "protocol_canonical_fingerprint": "b" * 64,
        "code": {"code_commit": "c" * 40, "worktree_dirty": False},
    }
    winners = {
        model.model_id: {
            "learning_rate": model.learning_rate,
            "selection_seed_count": 5,
        }
        for model in _models()
    }
    summary = {"selection_performed": True, "winners": winners}
    complete = {"verified_success_receipt_count": 120}

    monkeypatch.setattr(campaign, "load_selector_spec", lambda _: spec)
    monkeypatch.setattr(campaign, "load_protocol", lambda _: {"protocol": True})
    monkeypatch.setattr(campaign, "validate_protocol_binding", lambda *_: None)
    monkeypatch.setattr(campaign, "build_selection_plan", lambda _: ("plan",))

    def load(path: Path):
        return {
            "lr_selection_manifest.json": manifest,
            "lr_selection_summary.json": summary,
            "COMPLETE": complete,
        }[Path(path).name]

    monkeypatch.setattr(campaign, "strict_json_load", load)
    called = []
    monkeypatch.setattr(
        campaign,
        "verify_selection_completion",
        lambda *args: (called.append(args) is None, "verified"),
    )
    parent = verify_parent_selector(root)
    assert len(called) == 1
    assert parent.binding["verified_nested_training_receipts"] == 120
    assert tuple(model.model_id for model in parent.models) == EXPECTED_MODEL_IDS


def test_parent_selector_rejects_failed_recursive_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import repro.sagodi_protocol.primary_main_campaign as campaign

    root = tmp_path / "selector"
    _write_parent_shell(root)
    monkeypatch.setattr(campaign, "load_selector_spec", lambda _: SimpleNamespace())
    monkeypatch.setattr(campaign, "load_protocol", lambda _: {})
    monkeypatch.setattr(campaign, "validate_protocol_binding", lambda *_: None)
    monkeypatch.setattr(campaign, "build_selection_plan", lambda _: ())
    monkeypatch.setattr(
        campaign,
        "strict_json_load",
        lambda path: {
            "lr_selection_manifest.json": {},
            "lr_selection_summary.json": {},
            "COMPLETE": {},
        }[Path(path).name],
    )
    monkeypatch.setattr(
        campaign,
        "verify_selection_completion",
        lambda *_: (False, "nested receipt hash mismatch"),
    )
    with pytest.raises(ParentSelectorError, match="nested receipt hash mismatch"):
        verify_parent_selector(root)
