"""LRU-LR-fixed CA-LRU RP search and paired RP-by-state-noise factorial."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from . import source_repaired_baselines_v6 as baseline_v6
from . import source_repaired_lru_calru_v6 as lru_v6
from . import state_noise_search_v1 as noise_v1
from .artifacts import (
    atomic_json,
    canonical_hash,
    canonical_tensor_mapping_sha256,
    derived_seed,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from .metrics import masked_mse, task_metrics
from .primary_v4 import build_v4_model
from .source_resolved_protocol import source_angular_integration
from .tasks import Batch, load_fixed_bank
from .train import _retention_plasticity_call


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "calru_factorial_v1.json"
FREEZE_DOCUMENT = MODULE_DIR / "CALRU_FACTORIAL_V1_FREEZE_ko.md"
BASELINE_CONFIG = MODULE_DIR / "source_repaired_baselines_v6.json"
CAMPAIGN_ID = "calru_factorial_v1"
PROTOCOL_REVISION = "lru_lr_fixed_rp_only_search_then_rp_by_state_noise_factorial_v1"
ROOT_MARKER = ".calru_factorial_v1_root.json"
CONFIG_CONTRACT_SHA256 = "f8b160c6b04123cb22c72721ea4ce071af3d6b3c7f4077bb2af3729a5e01e14f"
STAGES = ("smoke", "rp_sentinel", "rp_fanout", "factorial_main")
CONDITIONS = (
    "no_rp_no_noise",
    "no_rp_with_noise",
    "rp_no_noise",
    "rp_with_noise",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _native(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = strict_json_load(Path(path).expanduser().resolve(strict=True))
    if canonical_hash(payload) != CONFIG_CONTRACT_SHA256:
        raise ValueError("CA-LRU factorial config differs from frozen contract")
    if payload.get("campaign_id") != CAMPAIGN_ID or payload.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError("CA-LRU factorial identity differs")
    rp = payload["retention_plasticity_search"]
    if (
        rp["eta_lambda_grid"] != [300.0, 1000.0, 3000.0]
        or rp["damage_epsilon_grid"] != [1e-5, 3e-5, 1e-4]
        or rp["intervention_interval_updates_grid"] != [25, 50, 100]
        or rp["sentinel_seed"] != 100
        or rp["fanout_seeds"] != [101, 102, 103, 104]
        or rp["top_k"] != 5
    ):
        raise ValueError("CA-LRU RP search grid differs")
    main = payload["factorial_main"]
    if main["seeds"] != list(range(10)) or [row["id"] for row in main["conditions"]] != list(CONDITIONS):
        raise ValueError("CA-LRU factorial main contract differs")
    training = payload["training"]
    if (
        training["updates"] != 5000
        or training["batch_size"] != 64
        or training["target_noise_std"] != 0.0
        or training["output_dropout"] != 0.0
        or training["evaluation_state_noise_std"] != 0.0
    ):
        raise ValueError("CA-LRU factorial training contract differs")
    return payload


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    stage: str
    condition_id: str
    model_id: str
    model_seed: int
    learning_rate: float
    actual_state_noise_std: float
    rp_enabled: bool
    rp_eta_lambda: float | None
    rp_damage_epsilon: float | None
    rp_interval_updates: int | None
    updates: int
    batch_size: int
    evaluation_bank: str
    campaign_root: str
    output_dir: str
    smoke: bool = False

    def payload(self) -> dict[str, Any]:
        return _native(asdict(self))


def _float_key(value: float) -> str:
    return format(float(value), ".8g").replace(".", "p").replace("-", "m")


def _parse_slots(text: str) -> tuple[str, ...]:
    if str(text).strip().lower() == "cpu":
        return ("cpu",)
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values or any(not item.isdigit() for item in values) or len(set(values)) != len(values):
        raise ValueError("--gpus must be unique comma-separated ids or cpu")
    return values


def _runtime_files() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        MODULE_DIR / "primary_v4.py",
        MODULE_DIR / "models.py",
        MODULE_DIR / "train.py",
        MODULE_DIR / "source_resolved_protocol.py",
        MODULE_DIR / "source_repaired_baselines_v6.py",
        MODULE_DIR / "source_repaired_lru_calru_v6.py",
        MODULE_DIR / "state_noise_search_v1.py",
        MODULE_DIR / "tasks.py",
        MODULE_DIR / "metrics.py",
        MODULE_DIR / "artifacts.py",
    )


def _git_state(require_clean: bool) -> str:
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
    if require_clean and dirty:
        raise RuntimeError("full CA-LRU factorial stages require a clean committed worktree")
    return commit


def _parent_binding(baseline_root: Path, lru_root: Path, noise_root: Path) -> dict[str, Any]:
    baseline_root = baseline_root.expanduser().resolve()
    lru_root = lru_root.expanduser().resolve()
    noise_root = noise_root.expanduser().resolve()
    baseline_v6.require_verified_main(baseline_root)
    lru_v6._require_stage(lru_root, "lru_main", 10)
    if not noise_v1._stage_valid(noise_root, "main", 40):
        raise RuntimeError("verified four-baseline state-noise main is required")
    lru_selection_path = lru_root / "lru_fanout" / "selection.json"
    noise_selection_path = noise_root / "tuning" / "selection.json"
    lru_selection = strict_json_load(lru_selection_path)
    noise_selection = strict_json_load(noise_selection_path)["models"]["lru_n52"]
    learning_rate = float(lru_selection["winner"]["learning_rate"])
    positive_noise = float(noise_selection["positive_noise_winner"]["actual_state_noise_std"])
    if positive_noise <= 0.0:
        raise RuntimeError("factorial +Noise condition requires a strictly-positive LRU std")
    if float(noise_selection["positive_noise_winner"]["learning_rate"]) != learning_rate:
        raise RuntimeError("state-noise parent did not freeze the selected LRU LR")
    banks: dict[str, Any] = {}
    for purpose in ("tuning", "main_test"):
        archive = baseline_root / "banks" / f"{purpose}.npz"
        banks[purpose] = {"archive": str(archive), "sha256": sha256_file(archive)}
    return {
        "schema_version": 1,
        "baseline_root": str(baseline_root),
        "lru_root": str(lru_root),
        "state_noise_root": str(noise_root),
        "learning_rate": learning_rate,
        "learning_rate_source": "noise_free_lru_fanout_winner",
        "positive_state_noise_std": positive_noise,
        "positive_state_noise_source": "lru_positive_noise_winner",
        "receipts": {
            "baseline_main": sha256_file(baseline_root / "main" / "completion_receipt.json"),
            "lru_main": sha256_file(lru_root / "lru_main" / "completion_receipt.json"),
            "state_noise_main": sha256_file(noise_root / "main" / "completion_receipt.json"),
        },
        "selections": {
            "lru": sha256_file(lru_selection_path),
            "state_noise": sha256_file(noise_selection_path),
        },
        "banks": banks,
    }


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError(f"bound copy differs: {destination}")
    else:
        shutil.copy2(source, destination)


def _prepare_root(root: Path, baseline_root: Path, lru_root: Path, noise_root: Path, config_source: Path, *, require_clean: bool) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = root.expanduser().resolve()
    config_source = config_source.expanduser().resolve(strict=True)
    config = load_config(config_source)
    parent = _parent_binding(baseline_root, lru_root, noise_root)
    identity = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "code_commit": _git_state(require_clean),
        "config_sha256": sha256_file(config_source),
        "freeze_sha256": sha256_file(FREEZE_DOCUMENT),
        "runtime_code_sha256": {path.name: sha256_file(path) for path in _runtime_files()},
        "parent_binding": parent,
    }
    identity["scientific_identity"] = canonical_hash(identity)
    marker = root / ROOT_MARKER
    root.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        if strict_json_load(marker) != identity:
            raise RuntimeError("CA-LRU factorial root identity differs")
    elif any(root.iterdir()):
        raise RuntimeError("unmarked CA-LRU factorial root must be empty")
    else:
        _copy_exact(config_source, root / "inputs" / DEFAULT_CONFIG.name)
        _copy_exact(FREEZE_DOCUMENT, root / "inputs" / FREEZE_DOCUMENT.name)
        _copy_exact(BASELINE_CONFIG, root / "inputs" / BASELINE_CONFIG.name)
        for purpose, row in parent["banks"].items():
            _copy_exact(Path(row["archive"]), root / "banks" / f"{purpose}.npz")
        atomic_json(root / "parent_binding.json", parent)
        atomic_json(marker, identity)
    return config, parent, root / "inputs" / DEFAULT_CONFIG.name


def _assert_identity(root: Path, config_path: Path, *, require_clean: bool) -> dict[str, Any]:
    identity = strict_json_load(root / ROOT_MARKER)
    core = dict(identity)
    observed = core.pop("scientific_identity", None)
    if canonical_hash(core) != observed or identity.get("code_commit") != _git_state(require_clean):
        raise RuntimeError("CA-LRU factorial worker identity differs")
    if identity.get("config_sha256") != sha256_file(config_path) or identity.get("freeze_sha256") != sha256_file(FREEZE_DOCUMENT):
        raise RuntimeError("CA-LRU factorial worker inputs differ")
    if identity.get("runtime_code_sha256") != {path.name: sha256_file(path) for path in _runtime_files()}:
        raise RuntimeError("CA-LRU factorial runtime code differs")
    return identity


def _spec(root: Path, parent: Mapping[str, Any], stage: str, condition: str, seed: int, noise: float, rp: bool, eta: float | None, epsilon: float | None, interval: int | None, updates: int, batch_size: int, bank: Path, *, smoke: bool = False) -> RunSpec:
    model_id = "ca_lru_n52" if rp else "no_rp_n52"
    suffix = f"__eta{_float_key(eta)}__eps{_float_key(epsilon)}__int{interval}" if rp else ""
    run_id = f"{stage}__{condition}__noise{_float_key(noise)}{suffix}__seed{seed}"
    return RunSpec(run_id, stage, condition, model_id, int(seed), float(parent["learning_rate"]), float(noise), bool(rp), eta, epsilon, interval, int(updates), int(batch_size), str(bank), str(root), str(root / stage / "runs" / run_id), smoke)


def build_smoke_plan(root: Path, config: Mapping[str, Any], parent: Mapping[str, Any], bank: Path) -> tuple[RunSpec, ...]:
    rp = config["retention_plasticity_search"]
    rows = []
    for condition in config["factorial_main"]["conditions"]:
        enabled = bool(condition["rp"])
        noise = float(parent["positive_state_noise_std"]) if condition["state_noise"] else 0.0
        rows.append(_spec(root, parent, "smoke", condition["id"], 999, noise, enabled, float(rp["eta_lambda_grid"][0]) if enabled else None, float(rp["damage_epsilon_grid"][0]) if enabled else None, int(rp["intervention_interval_updates_grid"][0]) if enabled else None, 2, 2, bank, smoke=True))
    return tuple(rows)


def build_rp_sentinel_plan(root: Path, config: Mapping[str, Any], parent: Mapping[str, Any], bank: Path) -> tuple[RunSpec, ...]:
    rp, training = config["retention_plasticity_search"], config["training"]
    return tuple(
        _spec(root, parent, "rp_sentinel", "rp_no_noise", int(rp["sentinel_seed"]), 0.0, True, float(eta), float(epsilon), int(interval), int(training["updates"]), int(training["batch_size"]), bank)
        for eta in rp["eta_lambda_grid"]
        for epsilon in rp["damage_epsilon_grid"]
        for interval in rp["intervention_interval_updates_grid"]
    )


def _result(spec: RunSpec) -> dict[str, Any]:
    try:
        payload = strict_json_load(Path(spec.output_dir) / "result.json")
    except (OSError, ValueError, TypeError):
        return {"status": "missing"}
    if payload.get("run_id") != spec.run_id:
        raise RuntimeError(f"result identity differs: {spec.run_id}")
    return payload


def _metric(spec: RunSpec, name: str) -> float | None:
    result = _result(spec)
    value = (result.get("final_metrics") or {}).get(name) if result.get("status") == "completed" else None
    return float(value) if value is not None and math.isfinite(float(value)) else None


def screen_rp_sentinels(specs: Sequence[RunSpec], config: Mapping[str, Any]) -> dict[str, Any]:
    rp, main = config["retention_plasticity_search"], config["factorial_main"]
    rows = []
    for spec in specs:
        mse, nmse, blank = _metric(spec, "mse"), _metric(spec, "nmse_db"), _result(spec).get("blank_memory_mse")
        complete = mse is not None and nmse is not None and blank is not None and math.isfinite(float(blank))
        rows.append({
            "eta_lambda": spec.rp_eta_lambda,
            "damage_epsilon": spec.rp_damage_epsilon,
            "intervention_interval_updates": spec.rp_interval_updates,
            "completed": complete,
            "mse": mse,
            "nmse_db": nmse,
            "blank_memory_mse": float(blank) if complete else None,
            "nmse_eligible": nmse is not None and nmse < float(main["analysis_nmse_db_threshold"]),
            "mse_success": mse is not None and mse < float(main["mse_success_threshold"]),
            "grid_order": [rp["eta_lambda_grid"].index(spec.rp_eta_lambda), rp["damage_epsilon_grid"].index(spec.rp_damage_epsilon), rp["intervention_interval_updates_grid"].index(spec.rp_interval_updates)],
        })
    rows.sort(key=lambda row: (not row["nmse_eligible"], not row["mse_success"], math.inf if row["blank_memory_mse"] is None else row["blank_memory_mse"], math.inf if row["mse"] is None else row["mse"], *row["grid_order"]))
    complete = [row for row in rows if row["completed"]]
    if len(complete) < int(rp["top_k"]):
        raise RuntimeError("fewer than top_k completed RP sentinel cells")
    return {"schema_version": 1, "single_seed_pruning": True, "top_cells": complete[: int(rp["top_k"])], "all_cells": rows}


def build_rp_fanout_plan(root: Path, config: Mapping[str, Any], parent: Mapping[str, Any], bank: Path, screening: Mapping[str, Any]) -> tuple[RunSpec, ...]:
    rp, training = config["retention_plasticity_search"], config["training"]
    return tuple(
        _spec(root, parent, "rp_fanout", "rp_no_noise", int(seed), 0.0, True, float(cell["eta_lambda"]), float(cell["damage_epsilon"]), int(cell["intervention_interval_updates"]), int(training["updates"]), int(training["batch_size"]), bank)
        for cell in screening["top_cells"]
        for seed in rp["fanout_seeds"]
    )


def select_rp(sentinel: Sequence[RunSpec], fanout: Sequence[RunSpec], config: Mapping[str, Any]) -> dict[str, Any]:
    rp, main = config["retention_plasticity_search"], config["factorial_main"]
    expected = {int(rp["sentinel_seed"]), *map(int, rp["fanout_seeds"])}
    cells = sorted({(spec.rp_eta_lambda, spec.rp_damage_epsilon, spec.rp_interval_updates) for spec in fanout}, key=lambda cell: (rp["eta_lambda_grid"].index(cell[0]), rp["damage_epsilon_grid"].index(cell[1]), rp["intervention_interval_updates_grid"].index(cell[2])))
    rows = []
    for eta, epsilon, interval in cells:
        group = [spec for spec in (*sentinel, *fanout) if (spec.rp_eta_lambda, spec.rp_damage_epsilon, spec.rp_interval_updates) == (eta, epsilon, interval)]
        if len(group) != 5 or {spec.model_seed for spec in group} != expected:
            raise RuntimeError("RP finalist does not have five registered seeds")
        mses, nmses = [_metric(spec, "mse") for spec in group], [_metric(spec, "nmse_db") for spec in group]
        blanks = [_result(spec).get("blank_memory_mse") if _result(spec).get("status") == "completed" else None for spec in group]
        complete = all(value is not None and math.isfinite(float(value)) for value in (*mses, *nmses, *blanks))
        rows.append({
            "eta_lambda": eta, "damage_epsilon": epsilon, "intervention_interval_updates": interval,
            "completed_seed_count": 5 if complete else 0,
            "nmse_eligible_count": sum(value is not None and value < float(main["analysis_nmse_db_threshold"]) for value in nmses),
            "mse_success_count": sum(value is not None and value < float(main["mse_success_threshold"]) for value in mses),
            "mean_blank_memory_mse": float(np.mean(blanks)) if complete else None,
            "mean_mse": float(np.mean(mses)) if complete else None,
            "grid_order": [rp["eta_lambda_grid"].index(eta), rp["damage_epsilon_grid"].index(epsilon), rp["intervention_interval_updates_grid"].index(interval)],
        })
    rows.sort(key=lambda row: (-row["nmse_eligible_count"], -row["mse_success_count"], math.inf if row["mean_blank_memory_mse"] is None else row["mean_blank_memory_mse"], math.inf if row["mean_mse"] is None else row["mean_mse"], *row["grid_order"]))
    complete = [row for row in rows if row["completed_seed_count"] == 5]
    if not complete:
        raise RuntimeError("no complete five-seed RP finalist")
    return {"schema_version": 1, "selection_rule": rp["selection_rule"], "winner": complete[0], "cells": rows}


def build_factorial_main_plan(root: Path, config: Mapping[str, Any], parent: Mapping[str, Any], bank: Path, selection: Mapping[str, Any]) -> tuple[RunSpec, ...]:
    winner, training = selection["winner"], config["training"]
    rows = []
    for condition in config["factorial_main"]["conditions"]:
        enabled = bool(condition["rp"])
        noise = float(parent["positive_state_noise_std"]) if condition["state_noise"] else 0.0
        for seed in config["factorial_main"]["seeds"]:
            rows.append(_spec(root, parent, "factorial_main", condition["id"], int(seed), noise, enabled, float(winner["eta_lambda"]) if enabled else None, float(winner["damage_epsilon"]) if enabled else None, int(winner["intervention_interval_updates"]) if enabled else None, int(training["updates"]), int(training["batch_size"]), bank))
    return tuple(rows)


def _to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(batch.inputs.to(device), batch.output_targets.to(device), batch.latent_targets.to(device), batch.mask.to(device), batch.metadata)


def _source_q1(batch: Batch) -> torch.Tensor:
    return batch.output_targets[0]


def _training_batch(spec: RunSpec, update: int, device: torch.device) -> Batch:
    return source_angular_integration(spec.batch_size, 0, stream_key=(baseline_v6.CAMPAIGN_ID, "online_train", spec.model_seed, int(update)), device=device)


def _rp_probe(config: Mapping[str, Any], spec: RunSpec, update: int, device: torch.device) -> Batch:
    rp = config["retention_plasticity_search"]
    return source_angular_integration(int(rp["probe_batch_size"]), 0, stream_key=(baseline_v6.CAMPAIGN_ID, "rp_probe", spec.model_seed, int(update)), device=device)


def _state_noise_seed(spec: RunSpec) -> int | None:
    return derived_seed(spec.model_seed, CAMPAIGN_ID, "paired_state_noise") if spec.actual_state_noise_std > 0.0 else None


def _rng_identities(spec: RunSpec) -> dict[str, Any]:
    noise_seed = _state_noise_seed(spec)
    return {
        "online_task": {"base_seed": 0, "stream_key_template": [baseline_v6.CAMPAIGN_ID, "online_train", spec.model_seed, "<update_1_to_5000>"]},
        "state_noise": {"enabled": noise_seed is not None, "generator_seed": noise_seed, "std": spec.actual_state_noise_std, "paired_across_rp_within_seed": True},
        "target_noise": {"enabled": False, "std": 0.0},
        "dropout": {"enabled": False, "probability": 0.0},
        "retention_plasticity_probe": {"enabled": spec.rp_enabled, "base_seed": 0 if spec.rp_enabled else None},
    }


@torch.no_grad()
def _evaluate(model: Any, batch: Batch) -> dict[str, Any]:
    model.eval()
    prediction = model.forward_sequence(batch.inputs, initial_memory=_source_q1(batch))
    if not torch.isfinite(prediction).all().item():
        raise FloatingPointError("non-finite held-out prediction")
    values = task_metrics(prediction, batch.output_targets, batch.mask, batch.latent_targets)
    values["mse"] = values["masked_mse"]
    values["nmse_db"] = values["masked_nmse_db"]
    return _native(values)


@torch.no_grad()
def _blank_mse(model: Any, batch: Batch, horizon: int) -> float:
    _, states = model.forward_sequence(batch.inputs, initial_memory=_source_q1(batch), return_states=True)
    state = states[-1]
    blank = torch.zeros(state.shape[0], model.input_dim, device=state.device, dtype=state.dtype)
    for _ in range(int(horizon)):
        state = model.step(blank, state)
    value = (model.decode(state) - batch.output_targets[-1]).square().mean()
    if not torch.isfinite(value).item():
        raise FloatingPointError("non-finite blank-memory MSE")
    return float(value.cpu())


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _expected_rp_updates(spec: RunSpec, config: Mapping[str, Any]) -> tuple[int, ...]:
    if not spec.rp_enabled:
        return ()
    warmup = int(config["retention_plasticity_search"]["warmup_updates"])
    interval = int(spec.rp_interval_updates)
    return tuple(update for update in range(1, spec.updates + 1) if update > warmup and update % interval == 0)


def _train_worker(spec: RunSpec, config_path: Path, device_text: str) -> Path:
    config = load_config(config_path)
    root = Path(spec.campaign_root).resolve()
    identity = _assert_identity(root, config_path, require_clean=not spec.smoke)
    if spec.stage not in STAGES or spec.condition_id not in CONDITIONS:
        raise ValueError("unregistered CA-LRU factorial worker")
    if spec.learning_rate != float(identity["parent_binding"]["learning_rate"]):
        raise ValueError("CA-LRU factorial LR differs from LRU winner")
    allowed_noise = {0.0, float(identity["parent_binding"]["positive_state_noise_std"])}
    if spec.actual_state_noise_std not in allowed_noise:
        raise ValueError("CA-LRU factorial noise differs from frozen LRU values")
    if spec.rp_enabled:
        rp = config["retention_plasticity_search"]
        if spec.model_id != "ca_lru_n52" or spec.rp_eta_lambda not in rp["eta_lambda_grid"] or spec.rp_damage_epsilon not in rp["damage_epsilon_grid"] or spec.rp_interval_updates not in rp["intervention_interval_updates_grid"]:
            raise ValueError("CA-LRU RP worker settings differ from frozen grid")
    elif spec.model_id != "no_rp_n52" or any(value is not None for value in (spec.rp_eta_lambda, spec.rp_damage_epsilon, spec.rp_interval_updates)):
        raise ValueError("No-RP worker contains RP settings")

    output = Path(spec.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("CA-LRU factorial worker output is not empty")
    device = torch.device(device_text)
    baseline_v6._configure_determinism(spec.model_seed)
    torch.set_num_threads(1)
    model = build_v4_model(spec.model_id).to(device)
    initial_hash = canonical_tensor_mapping_sha256(model.state_dict())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != int(config["models"]["parameters_gradient_trainable"]):
        raise RuntimeError(f"trainable parameter count differs: {trainable}")
    training = config["training"]
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=spec.learning_rate, betas=tuple(float(value) for value in training["betas"]), eps=float(training["epsilon"]), weight_decay=float(training["weight_decay"]))
    bank = _to_device(load_fixed_bank(spec.evaluation_bank), device)
    rng = _rng_identities(spec)
    state_generator = torch.Generator(device=device.type).manual_seed(int(rng["state_noise"]["generator_seed"])) if rng["state_noise"]["enabled"] else None
    manifest = {
        "schema_version": 1, "campaign_id": CAMPAIGN_ID, "protocol_revision": PROTOCOL_REVISION,
        "run": spec.payload(), "scientific_identity": identity["scientific_identity"], "code_commit": identity["code_commit"],
        "runtime_code_sha256": identity["runtime_code_sha256"], "config_sha256": identity["config_sha256"], "freeze_sha256": identity["freeze_sha256"],
        "parent_binding_sha256": canonical_hash(identity["parent_binding"]), "initial_state_dict_sha256": initial_hash,
        "parameters_gradient_trainable": trainable, "rng_stream_identities": rng,
        "learning_rate_fixed_from_lru": True, "state_noise_fixed_from_lru_positive_winner": spec.actual_state_noise_std > 0.0,
        "target_noise_std": 0.0, "output_dropout": 0.0, "evaluation_state_noise_std": 0.0,
        "evaluation_bank_sha256": sha256_file(spec.evaluation_bank), "started_at_utc": _utc_now(), "device": device_text,
    }
    atomic_json(output / "run_manifest.json", manifest)
    trace, rp_trace = [], []
    started = time.time()
    scheduled_rp_updates = set(_expected_rp_updates(spec, config))
    for update in range(1, spec.updates + 1):
        model.train()
        batch = _training_batch(spec, update, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_sequence(batch.inputs, initial_memory=_source_q1(batch), state_noise_std=spec.actual_state_noise_std, noise_generator=state_generator)
        loss = masked_mse(prediction, batch.output_targets, batch.mask)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()
        baseline_v6._finite_model(model, gradients=True)
        optimizer.step()
        baseline_v6._finite_model(model)
        if update in scheduled_rp_updates:
            probe = _rp_probe(config, spec, update, device)
            values = _retention_plasticity_call(model, probe, blank_horizon=int(config["retention_plasticity_search"]["blank_ablation_horizon"]), eta_lambda=float(spec.rp_eta_lambda), damage_epsilon=float(spec.rp_damage_epsilon), initial_memory=_source_q1(probe))
            rp_trace.append({"update": update, **_native(values)})
        should_trace = update == 1 or update % int(training["trace_interval"]) == 0 or update == spec.updates
        should_validate = update % int(training["validation_interval"]) == 0 or update == spec.updates
        if should_trace or should_validate:
            row = {"update": update, "train_mse": float(loss.detach().cpu()), "elapsed_seconds": float(time.time() - started)}
            if should_validate:
                row["validation"] = _evaluate(model, bank)
            trace.append(row)
            atomic_json(output / "training_trace.json", trace)
            atomic_json(output / "rp_trace.json", rp_trace)
            atomic_json(output / "progress.json", {"status": "running", "run_id": spec.run_id, "update": update, "updates_total": spec.updates, "latest": row, "updated_at_utc": _utc_now()})
    final_metrics = _evaluate(model, bank)
    blank_horizon = int(config["factorial_main"]["blank_horizon"] if spec.stage == "factorial_main" else config["retention_plasticity_search"]["selection_blank_horizon"])
    blank = None if spec.smoke else _blank_mse(model, bank, blank_horizon)
    result = {
        "schema_version": 1, "status": "completed", "run_id": spec.run_id, "stage": spec.stage,
        "condition_id": spec.condition_id, "model_id": spec.model_id, "model_seed": spec.model_seed,
        "learning_rate": spec.learning_rate, "actual_state_noise_std": spec.actual_state_noise_std,
        "rp_enabled": spec.rp_enabled, "rp_eta_lambda": spec.rp_eta_lambda, "rp_damage_epsilon": spec.rp_damage_epsilon,
        "rp_interval_updates": spec.rp_interval_updates, "rp_call_count": len(rp_trace), "updates_completed": spec.updates,
        "final_metrics": final_metrics, "blank_memory_mse": blank, "counts_in_denominator": True,
    }
    atomic_json(output / "training_trace.json", trace)
    atomic_json(output / "rp_trace.json", rp_trace)
    atomic_json(output / "result.json", result)
    atomic_json(output / "progress.json", {"status": "completed", "run_id": spec.run_id, "update": spec.updates})
    checkpoint = output / "checkpoint_final.pt"
    _atomic_torch_save(checkpoint, {"schema_version": 1, "checkpoint_type": CAMPAIGN_ID, "run": spec.payload(), "result": result, "initial_state_dict_sha256": initial_hash, "rng_stream_identities": rng, "state_dict": model.state_dict()})
    atomic_json(output / "COMPLETE", {"schema_version": 1, "run_id": spec.run_id})
    write_completion_receipt(output / "completion_receipt.json", job_id=spec.run_id, artifacts=[output / "run_manifest.json", output / "progress.json", output / "training_trace.json", output / "rp_trace.json", output / "result.json", checkpoint, output / "COMPLETE"], metadata={"campaign_id": CAMPAIGN_ID, "stage": spec.stage, "condition_id": spec.condition_id, "model_seed": spec.model_seed})
    return output


def _verified(spec: RunSpec) -> bool:
    output = Path(spec.output_dir)
    try:
        result = strict_json_load(output / "result.json")
        checkpoint = torch.load(output / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        manifest = strict_json_load(output / "run_manifest.json")
        marker = strict_json_load(output / "COMPLETE")
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        return False
    required = {"run_manifest.json", "progress.json", "training_trace.json", "rp_trace.json", "result.json", "checkpoint_final.pt", "COMPLETE"}
    valid, _ = verify_completion_receipt(output / "completion_receipt.json", expected_job_id=spec.run_id, expected_metadata={"campaign_id": CAMPAIGN_ID, "stage": spec.stage, "condition_id": spec.condition_id, "model_seed": spec.model_seed})
    try:
        receipt_names = set(strict_json_load(output / "completion_receipt.json")["artifacts"])
    except (OSError, ValueError, TypeError, KeyError):
        return False
    if not valid or receipt_names != required:
        return False
    return bool(
        result.get("status") == "completed" and result.get("run_id") == spec.run_id and result.get("updates_completed") == spec.updates
        and result.get("rp_call_count") == len(_expected_rp_updates(spec, load_config(Path(spec.campaign_root) / "inputs" / DEFAULT_CONFIG.name)))
        and checkpoint.get("checkpoint_type") == CAMPAIGN_ID and checkpoint.get("run") == spec.payload() and checkpoint.get("result") == result
        and manifest.get("run") == spec.payload() and marker.get("run_id") == spec.run_id
    )


def _write_or_verify(path: Path, payload: Mapping[str, Any]) -> None:
    native = _native(payload)
    if path.exists():
        if strict_json_load(path) != native:
            raise RuntimeError(f"immutable artifact differs: {path}")
    else:
        atomic_json(path, native)


def _write_plan(stage_root: Path, specs: Sequence[RunSpec]) -> None:
    _write_or_verify(stage_root / "plan.json", {"schema_version": 1, "campaign_id": CAMPAIGN_ID, "run_count": len(specs), "runs": [spec.payload() for spec in specs]})


def _run_specs(specs: Sequence[RunSpec], config_path: Path, slots: Sequence[str]) -> None:
    if not specs:
        return
    stage_root = Path(specs[0].output_dir).parents[1]
    _write_plan(stage_root, specs)
    pending = [spec for spec in specs if not _verified(spec)]
    for spec in pending:
        output = Path(spec.output_dir)
        if output.exists():
            attempts = stage_root / "attempts"
            attempts.mkdir(parents=True, exist_ok=True)
            os.replace(output, attempts / f"{output.name}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}")
    specs_dir, logs_dir = stage_root / "specs", stage_root / "logs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    queue, running = list(pending), {}
    repo = Path(__file__).resolve().parents[2]
    try:
        while queue or running:
            for slot in [slot for slot in slots if slot not in running]:
                if not queue:
                    break
                spec = queue.pop(0)
                spec_path = specs_dir / f"{spec.run_id}.json"
                atomic_json(spec_path, spec.payload())
                handle = (logs_dir / f"{spec.run_id}.log").open("ab")
                command = [sys.executable, "-m", "repro.sagodi_protocol.calru_factorial_v1", "--worker-spec", str(spec_path), "--config", str(config_path), "--device", "cpu" if slot == "cpu" else "cuda:0"]
                environment = os.environ.copy()
                for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    environment[variable] = "1"
                if slot != "cpu":
                    environment["CUDA_VISIBLE_DEVICES"] = slot
                process = subprocess.Popen(command, cwd=repo, env=environment, stdout=handle, stderr=subprocess.STDOUT)
                running[slot] = (process, spec, handle)
            time.sleep(0.2)
            for slot, (process, spec, handle) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del running[slot]
                if code != 0 or not _verified(spec):
                    raise RuntimeError(f"retryable CA-LRU factorial worker failure: {spec.run_id}; exit={code}")
            atomic_json(stage_root / "status.json", {"schema_version": 1, "registered": len(specs), "verified_complete": sum(_verified(spec) for spec in specs), "pending": len(queue), "running": [row[1].run_id for row in running.values()], "updated_at_utc": _utc_now()})
    finally:
        for process, _, handle in running.values():
            if process.poll() is None:
                process.terminate()
            handle.close()


def summarize_factorial(specs: Sequence[RunSpec], config: Mapping[str, Any]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for condition in CONDITIONS:
        group = sorted((spec for spec in specs if spec.condition_id == condition), key=lambda spec: spec.model_seed)
        if len(group) != 10 or {spec.model_seed for spec in group} != set(range(10)):
            raise RuntimeError(f"factorial denominator differs for {condition}")
        per_seed = []
        for spec in group:
            result = _result(spec)
            metrics = result.get("final_metrics") or {}
            per_seed.append({
                "seed": spec.model_seed, "status": result.get("status"), "mse": metrics.get("mse"), "nmse_db": metrics.get("nmse_db"),
                "blank_memory_mse": result.get("blank_memory_mse"),
                "mse_success": metrics.get("mse") is not None and float(metrics["mse"]) < float(config["factorial_main"]["mse_success_threshold"]),
                "analysis_eligible": metrics.get("nmse_db") is not None and float(metrics["nmse_db"]) < float(config["factorial_main"]["analysis_nmse_db_threshold"]),
            })
        models[condition] = {
            "learning_rate": group[0].learning_rate, "actual_state_noise_std": group[0].actual_state_noise_std,
            "rp_enabled": group[0].rp_enabled, "rp_eta_lambda": group[0].rp_eta_lambda,
            "rp_damage_epsilon": group[0].rp_damage_epsilon, "rp_interval_updates": group[0].rp_interval_updates,
            "completed_seed_count": sum(row["status"] == "completed" for row in per_seed),
            "mse_success_count": sum(row["mse_success"] for row in per_seed),
            "analysis_eligible_count": sum(row["analysis_eligible"] for row in per_seed), "per_seed": per_seed,
        }
    paired = []
    for seed in range(10):
        value = {condition: models[condition]["per_seed"][seed] for condition in CONDITIONS}
        paired.append({
            "seed": seed,
            "noise_effect_without_rp_mse": value["no_rp_with_noise"]["mse"] - value["no_rp_no_noise"]["mse"],
            "noise_effect_with_rp_mse": value["rp_with_noise"]["mse"] - value["rp_no_noise"]["mse"],
            "rp_effect_without_noise_mse": value["rp_no_noise"]["mse"] - value["no_rp_no_noise"]["mse"],
            "rp_effect_with_noise_mse": value["rp_with_noise"]["mse"] - value["no_rp_with_noise"]["mse"],
        })
    return {"schema_version": 1, "campaign_id": CAMPAIGN_ID, "models": models, "paired_contrasts": paired, "expected_ranking_is_hypothesis_not_gate": ["rp_with_noise", "rp_no_noise", "no_rp_with_noise", "no_rp_no_noise"]}


def _children_binding(specs: Sequence[RunSpec]) -> dict[str, Any]:
    return {"schema_version": 1, "children": [{"run_id": spec.run_id, "result_sha256": sha256_file(Path(spec.output_dir) / "result.json"), "checkpoint_sha256": sha256_file(Path(spec.output_dir) / "checkpoint_final.pt"), "receipt_sha256": sha256_file(Path(spec.output_dir) / "completion_receipt.json")} for spec in specs]}


def _finish_stage(root: Path, stage: str, specs: Sequence[RunSpec], name: str, payload: Mapping[str, Any]) -> Path:
    missing = [spec.run_id for spec in specs if not _verified(spec)]
    if missing:
        raise RuntimeError(f"cannot finalize {stage}; missing {missing[:3]}")
    stage_root = root / stage
    primary, children, complete = stage_root / name, stage_root / "children_binding.json", stage_root / "COMPUTATION_COMPLETE"
    _write_or_verify(primary, payload)
    _write_or_verify(children, _children_binding(specs))
    atomic_json(complete, {"schema_version": 1, "stage": stage, "verified_run_count": len(specs), "completed_at_utc": _utc_now()})
    write_completion_receipt(stage_root / "completion_receipt.json", job_id=f"{CAMPAIGN_ID}__{stage}", artifacts=[stage_root / "plan.json", primary, children, complete], metadata={"campaign_id": CAMPAIGN_ID, "stage": stage, "run_count": len(specs)})
    return stage_root


def _read_specs(stage_root: Path) -> tuple[RunSpec, ...]:
    return tuple(RunSpec(**row) for row in strict_json_load(stage_root / "plan.json")["runs"])


def _stage_valid(root: Path, stage: str, expected: int) -> bool:
    stage_root = root / stage
    valid, _ = verify_completion_receipt(stage_root / "completion_receipt.json", expected_job_id=f"{CAMPAIGN_ID}__{stage}", expected_metadata={"campaign_id": CAMPAIGN_ID, "stage": stage, "run_count": expected})
    if not valid or not (stage_root / "COMPUTATION_COMPLETE").exists():
        return False
    try:
        specs = _read_specs(stage_root)
        return len(specs) == expected and all(_verified(spec) for spec in specs)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        return False


def run_stage(stage: str, artifact_root: Path, baseline_root: Path, lru_root: Path, noise_root: Path, config_source: Path, slots: Sequence[str]) -> Path:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    root = artifact_root.expanduser().resolve()
    config, parent, copied_config = _prepare_root(root, baseline_root, lru_root, noise_root, config_source, require_clean=stage != "smoke")
    tuning_bank, main_bank = root / "banks" / "tuning.npz", root / "banks" / "main_test.npz"
    smoke = build_smoke_plan(root, config, parent, tuning_bank)
    if stage == "smoke":
        if _stage_valid(root, stage, 4):
            return root / stage
        _run_specs(smoke, copied_config, slots)
        return _finish_stage(root, stage, smoke, "summary.json", {"schema_version": 1, "scientific_result": False, "runs": [spec.run_id for spec in smoke]})
    if not _stage_valid(root, "smoke", 4):
        raise RuntimeError("RP search is blocked until verified smoke completes")
    sentinel = build_rp_sentinel_plan(root, config, parent, tuning_bank)
    if stage == "rp_sentinel":
        if _stage_valid(root, stage, 27):
            return root / stage
        _run_specs(sentinel, copied_config, slots)
        return _finish_stage(root, stage, sentinel, "screening.json", screen_rp_sentinels(sentinel, config))
    if not _stage_valid(root, "rp_sentinel", 27):
        raise RuntimeError("RP fanout is blocked until verified sentinel completes")
    screening = strict_json_load(root / "rp_sentinel" / "screening.json")
    fanout = build_rp_fanout_plan(root, config, parent, tuning_bank, screening)
    if stage == "rp_fanout":
        if _stage_valid(root, stage, 20):
            return root / stage
        _run_specs(fanout, copied_config, slots)
        return _finish_stage(root, stage, fanout, "selection.json", select_rp(sentinel, fanout, config))
    if not _stage_valid(root, "rp_fanout", 20):
        raise RuntimeError("factorial main is blocked until verified RP fanout completes")
    selection = strict_json_load(root / "rp_fanout" / "selection.json")
    main = build_factorial_main_plan(root, config, parent, main_bank, selection)
    if _stage_valid(root, stage, 40):
        return root / stage
    _run_specs(main, copied_config, slots)
    return _finish_stage(root, stage, main, "summary.json", summarize_factorial(main, config))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--lru-root", type=Path)
    parser.add_argument("--state-noise-root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_spec is not None:
        spec = RunSpec(**strict_json_load(args.worker_spec))
        _train_worker(spec, args.config.resolve(strict=True), str(args.device))
        return 0
    required = (args.stage, args.artifact_root, args.baseline_root, args.lru_root, args.state_noise_root)
    if any(value is None for value in required):
        raise ValueError("campaign mode requires stage, artifact root, and all three parent roots")
    print(run_stage(str(args.stage), args.artifact_root, args.baseline_root, args.lru_root, args.state_noise_root, args.config.resolve(strict=True), _parse_slots(args.gpus)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
