"""Baseline-bound staged LRU -> No-RP -> CA-LRU campaign.

This downstream campaign cannot start before the repaired RNN/GRU/LSTM v6
campaign has a verified completed main stage and per-model eligibility report.
Individual ineligible baseline models do not block it. It byte-copies and hash-binds the
parent tuning/main banks and uses the parent's online input namespace.  LRU is
tuned and tested first; No-RP is blocked until LRU has at least one fresh main
seed with NMSE < -20 dB. CA-LRU needs verified No-RP completion but may run
even when No-RP itself has no eligible seed, because RP may create eligibility.
All downstream models use the baseline-controlled clean q1 and clean loss
targets, with target-noise standard deviation fixed to zero.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .artifacts import (
    atomic_json,
    canonical_hash,
    canonical_tensor_mapping_sha256,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from .metrics import masked_mse, task_metrics
from .primary_v4 import _configure_determinism, _finite_model, _to_device_batch, build_v4_model
from .source_repaired_baselines_v6 import (
    CAMPAIGN_ID as BASELINE_CAMPAIGN_ID,
    DEFAULT_CONFIG as BASELINE_CONFIG,
    ROOT_MARKER as BASELINE_ROOT_MARKER,
    _validate_bank,
    load_config as load_baseline_config,
    require_verified_main as require_baseline_main,
)
from .source_resolved_protocol import source_angular_integration
from .tasks import load_fixed_bank


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "source_repaired_lru_calru_v6.json"
FREEZE_DOCUMENT = MODULE_DIR / "SAGODI_SOURCE_REPAIRED_LRU_CALRU_V6_FREEZE_ko.md"
CAMPAIGN_ID = "sagodi_source_repaired_lru_calru_v6"
PROTOCOL_REVISION = (
    "baseline_bound_noise_free_controlled_training_lr_only_pilot3_v6"
)
TRACK_CLASSIFICATION = (
    "baseline_bound_noise_free_controlled_training_with_paper_and_code_noise_provenance"
)
ROOT_MARKER = ".sagodi_source_repaired_lru_calru_v6_root.json"
CONFIG_CONTRACT_SHA256 = "d352733fc31c7889a725488944ef46e805ffe2584202438be8c0d6003621cd06"
MODEL_IDS = ("lru_n52", "no_rp_n52", "ca_lru_n52")
TOTAL_COUNTS = {model_id: 17058 for model_id in MODEL_IDS}
GRADIENT_COUNTS = {"lru_n52": 17058, "no_rp_n52": 17006, "ca_lru_n52": 17006}
LR_STAGES = {
    "lru": "lru_n52",
    "no_rp": "no_rp_n52",
}
STAGES = (
    "smoke",
    "lru_sentinel",
    "lru_fanout",
    "lru_main",
    "no_rp_sentinel",
    "no_rp_fanout",
    "no_rp_main",
    "ca_rp_sentinel",
    "ca_rp_fanout",
    "ca_rp_main",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _native(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = strict_json_load(Path(path).expanduser().resolve(strict=True))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("downstream v6 config must be schema 1")
    if canonical_hash(payload) != CONFIG_CONTRACT_SHA256:
        raise ValueError("downstream config differs from exact canonical frozen contract")
    if payload.get("campaign_id") != CAMPAIGN_ID or payload.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError("downstream v6 identity differs")
    if payload.get("track_classification") != TRACK_CLASSIFICATION:
        raise ValueError("downstream v6 track classification differs")
    if payload.get("baseline_parent_campaign") != BASELINE_CAMPAIGN_ID:
        raise ValueError("baseline parent identity differs")
    if payload.get("task_and_data_contract") != {
        "reuse_exact_parent_tuning_and_main_banks": True,
        "online_stream_namespace": BASELINE_CAMPAIGN_ID,
        "horizon": 128,
        "initial_state_semantics": "source_q1_post_update_target",
    }:
        raise ValueError("downstream task/data contract differs")
    if [row.get("id") for row in payload.get("models", [])] != list(MODEL_IDS):
        raise ValueError("downstream model order differs")
    for row in payload["models"]:
        model_id = row["id"]
        if row.get("parameters_total") != TOTAL_COUNTS[model_id]:
            raise ValueError(f"total parameter count differs for {model_id}")
        if row.get("parameters_gradient_trainable") != GRADIENT_COUNTS[model_id]:
            raise ValueError(f"gradient parameter count differs for {model_id}")
    training = payload["training"]
    if training["actual_post_transition_state_noise_std"] != 0.0:
        raise ValueError("common actual state-noise standard deviation differs")
    if training["state_noise_distribution"] != "disabled":
        raise ValueError("common state-noise distribution differs")
    if training["state_noise_location"] != "disabled_no_state_noise_injection":
        raise ValueError("common state-noise location differs")
    if training["state_noise_scaling"] != "not_applicable_disabled":
        raise ValueError("disabled state-noise scaling differs")
    if training["paper_state_noise_covariance_provenance_only"] != "0.01I":
        raise ValueError("paper state-noise provenance differs")
    if training["evaluation_state_noise_std"] != 0.0:
        raise ValueError("evaluation state-noise contract differs")
    if training["controlled_target_noise_std"] != 0.0:
        raise ValueError("controlled target-noise contract differs")
    if training["controlled_output_dropout"] != 0.0:
        raise ValueError("controlled output-dropout contract differs")
    if training["training_target_semantics"] != (
        "clean_cos_sin_target_for_initial_q1_and_loss"
    ):
        raise ValueError("controlled training-target semantics differ")
    tuning = payload["learning_rate_tuning"]
    if tuning["learning_rate_grid"] != [
        0.03,
        0.01,
        0.003,
        0.001,
        0.0003,
        0.0001,
        0.00003,
        0.00001,
    ]:
        raise ValueError("LR grid differs")
    if (
        tuning["sentinel_seed"] != 100
        or tuning["fanout_seeds"] != [101]
        or tuning["screening_updates"] != 2000
        or tuning["fanout_all_learning_rates"] is not True
    ):
        raise ValueError("LR tuning seeds/fanout policy differ")
    if payload["main"]["seeds"] != list(range(3)):
        raise ValueError("main seeds differ")
    if payload["ca_fairness"] != {
        "inherits_learning_rate_from": "no_rp_n52",
        "inherits_common_state_noise_from": "global_training_contract",
        "same_seed_initialization_and_online_batches": True,
        "ca_learning_rate_independently_tuned": False,
    }:
        raise ValueError("CA pairing/fairness contract differs")
    return payload


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    stage: str
    model_id: str
    model_seed: int
    learning_rate: float
    actual_state_noise_std: float
    updates: int
    batch_size: int
    evaluation_bank: str
    campaign_root: str
    output_dir: str
    rp_enabled: bool = False
    rp_eta_lambda: float | None = None
    rp_damage_epsilon: float | None = None
    smoke: bool = False

    def payload(self) -> dict[str, Any]:
        return _native(asdict(self))


def _common_state_noise(config: Mapping[str, Any]) -> float:
    return float(config["training"]["actual_post_transition_state_noise_std"])


def _float_key(value: float) -> str:
    return format(float(value), ".7g").replace(".", "p").replace("-", "m")


def _source_q1(batch: Any) -> torch.Tensor:
    if batch.output_targets.shape[0] != 128:
        raise ValueError("source q1 initializer requires T128")
    return batch.output_targets[0]


def _training_batch(spec: RunSpec, update: int, device: torch.device) -> Any:
    return source_angular_integration(
        spec.batch_size,
        0,
        stream_key=(BASELINE_CAMPAIGN_ID, "online_train", spec.model_seed, int(update)),
        device=device,
    )


def _rp_probe(spec: RunSpec, config: Mapping[str, Any], update: int, device: torch.device) -> Any:
    rp = config["retention_plasticity"]
    return source_angular_integration(
        int(rp["probe_batch_size"]),
        0,
        stream_key=(BASELINE_CAMPAIGN_ID, "rp_probe", spec.model_seed, int(update)),
        device=device,
    )


def _rng_stream_identities(spec: RunSpec) -> dict[str, Any]:
    return {
        "online_task": {
            "base_seed": 0,
            "stream_key_template": [
                BASELINE_CAMPAIGN_ID,
                "online_train",
                spec.model_seed,
                "<update_1_to_5000>",
            ],
        },
        "target_noise": {
            "enabled": False,
            "generator_seed": None,
            "std": 0.0,
            "semantics": "clean_q1_initializer_and_clean_loss_target_no_rng_draws",
        },
        "state_noise": {
            "enabled": False,
            "generator_seed": None,
            "std": float(spec.actual_state_noise_std),
            "scaling": "not_applicable_disabled",
            "paper_literal_std_provenance_only": 0.1,
            "paper_covariance_provenance_only": "0.01I",
            "semantics": "controlled_execution_has_no_state_noise_rng_draws",
        },
        "dropout": {
            "enabled": False,
            "global_torch_seed": None,
            "probability": 0.0,
            "semantics": "controlled_execution_has_no_dropout_rng_draws",
        },
        "retention_plasticity_probe": {
            "enabled": bool(spec.rp_enabled),
            "base_seed": 0 if spec.rp_enabled else None,
            "stream_key_template": (
                [
                    BASELINE_CAMPAIGN_ID,
                    "rp_probe",
                    spec.model_seed,
                    "<scheduled_update>",
                ]
                if spec.rp_enabled
                else None
            ),
        },
    }


@torch.no_grad()
def _evaluate(model: Any, batch: Any) -> dict[str, Any]:
    model.eval()
    prediction = model.forward_sequence(batch.inputs, initial_memory=_source_q1(batch))
    if not torch.isfinite(prediction).all().item():
        raise FloatingPointError("non-finite held-out prediction")
    result = task_metrics(prediction, batch.output_targets, batch.mask, batch.latent_targets)
    result["mse"] = result["masked_mse"]
    result["nmse_db"] = result["masked_nmse_db"]
    return _native(result)


@torch.no_grad()
def _blank_mse(model: Any, batch: Any, horizon: int) -> float:
    _, states = model.forward_sequence(
        batch.inputs, initial_memory=_source_q1(batch), return_states=True
    )
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


def _device_provenance(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_slot_from_launcher": os.environ.get("CALRU_PHYSICAL_GPU_SLOT"),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        payload.update(
            {
                "cuda_visible_index": index,
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_compute_capability": list(torch.cuda.get_device_capability(index)),
                "torch_cuda": torch.version.cuda,
            }
        )
    return payload


def build_lr_sentinel_plan(
    root: Path,
    config: Mapping[str, Any],
    bank: Path,
    prefix: str,
) -> tuple[RunSpec, ...]:
    model_id = LR_STAGES[prefix]
    tuning, training = config["learning_rate_tuning"], config["training"]
    noise = _common_state_noise(config)
    specs: list[RunSpec] = []
    for lr in tuning["learning_rate_grid"]:
        run_id = (
            f"{prefix}_sentinel__lr{_float_key(lr)}__noise{_float_key(noise)}"
            f"__seed{tuning['sentinel_seed']}"
        )
        specs.append(
            RunSpec(
                run_id=run_id,
                stage=f"{prefix}_sentinel",
                model_id=model_id,
                model_seed=int(tuning["sentinel_seed"]),
                learning_rate=float(lr),
                actual_state_noise_std=noise,
                updates=int(tuning["screening_updates"]),
                batch_size=int(training["batch_size"]),
                evaluation_bank=str(bank),
                campaign_root=str(root),
                output_dir=str(root / f"{prefix}_sentinel" / "runs" / run_id),
            )
        )
    return tuple(specs)


def _result(spec: RunSpec) -> dict[str, Any]:
    payload = strict_json_load(Path(spec.output_dir) / "result.json")
    if payload.get("run_id") != spec.run_id:
        raise RuntimeError("run/result identity differs")
    return payload


def _metric(spec: RunSpec, key: str) -> float | None:
    result = _result(spec)
    metrics = result.get("final_metrics")
    if result.get("status") != "completed" or not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def screen_sentinels(specs: Sequence[RunSpec], config: Mapping[str, Any]) -> dict[str, Any]:
    """Record one-seed LR results without pruning any LR candidate."""

    tuning, main = config["learning_rate_tuning"], config["main"]
    mse_threshold = float(main["mse_success_threshold"])
    nmse_threshold = float(main["analysis_eligibility_nmse_db_threshold"])
    rows: list[dict[str, Any]] = []
    for spec in specs:
        mse, nmse = _metric(spec, "mse"), _metric(spec, "nmse_db")
        rows.append(
            {
                "learning_rate": spec.learning_rate,
                "actual_state_noise_std": spec.actual_state_noise_std,
                "run_id": spec.run_id,
                "completed": mse is not None and nmse is not None,
                "mse": mse,
                "nmse_db": nmse,
                "nmse_eligible": nmse is not None and nmse < nmse_threshold,
                "mse_success": mse is not None and mse < mse_threshold,
                "grid_order": tuning["learning_rate_grid"].index(spec.learning_rate),
            }
        )
    rows.sort(
        key=lambda row: (
            not row["nmse_eligible"],
            not row["mse_success"],
            math.inf if row["mse"] is None else row["mse"],
            math.inf if row["nmse_db"] is None else row["nmse_db"],
            row["grid_order"],
        )
    )
    if len(rows) != len(tuning["learning_rate_grid"]):
        raise RuntimeError("sentinel LR denominator differs")
    return {
        "schema_version": 1,
        "selection_priority": "nmse_eligibility_before_descriptive_mse",
        "fanout_all_learning_rates": True,
        "all_learning_rates": rows,
    }


def build_lr_fanout_plan(
    root: Path,
    config: Mapping[str, Any],
    bank: Path,
    prefix: str,
    screening: Mapping[str, Any],
) -> tuple[RunSpec, ...]:
    model_id = LR_STAGES[prefix]
    tuning, training = config["learning_rate_tuning"], config["training"]
    noise = _common_state_noise(config)
    specs: list[RunSpec] = []
    observed = {float(row["learning_rate"]) for row in screening["all_learning_rates"]}
    expected = set(map(float, tuning["learning_rate_grid"]))
    if observed != expected or len(screening["all_learning_rates"]) != len(expected):
        raise ValueError("fanout requires every registered learning rate")
    for lr in tuning["learning_rate_grid"]:
        lr = float(lr)
        for seed in tuning["fanout_seeds"]:
            run_id = (
                f"{prefix}_fanout__lr{_float_key(lr)}__noise{_float_key(noise)}"
                f"__seed{seed}"
            )
            specs.append(
                RunSpec(
                    run_id=run_id,
                    stage=f"{prefix}_fanout",
                    model_id=model_id,
                    model_seed=int(seed),
                    learning_rate=lr,
                    actual_state_noise_std=noise,
                    updates=int(tuning["screening_updates"]),
                    batch_size=int(training["batch_size"]),
                    evaluation_bank=str(bank),
                    campaign_root=str(root),
                    output_dir=str(root / f"{prefix}_fanout" / "runs" / run_id),
                )
            )
    return tuple(specs)


def select_learning_rate(
    sentinel_specs: Sequence[RunSpec],
    fanout_specs: Sequence[RunSpec],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tuning, main = config["learning_rate_tuning"], config["main"]
    expected_seeds = {int(tuning["sentinel_seed"]), *map(int, tuning["fanout_seeds"])}
    expected_count = len(expected_seeds)
    learning_rates = sorted(
        {spec.learning_rate for spec in fanout_specs},
        key=tuning["learning_rate_grid"].index,
    )
    if learning_rates != list(map(float, tuning["learning_rate_grid"])):
        raise RuntimeError("fanout LR denominator differs")
    rows: list[dict[str, Any]] = []
    for lr in learning_rates:
        noise = _common_state_noise(config)
        specs = [
            spec
            for spec in (*sentinel_specs, *fanout_specs)
            if spec.learning_rate == lr
        ]
        if (
            len(specs) != expected_count
            or {spec.model_seed for spec in specs} != expected_seeds
            or {spec.actual_state_noise_std for spec in specs} != {noise}
        ):
            raise RuntimeError("LR candidate does not have the registered pilot-screen seeds")
        mses = [_metric(spec, "mse") for spec in specs]
        nmses = [_metric(spec, "nmse_db") for spec in specs]
        complete = all(value is not None for value in (*mses, *nmses))
        rows.append(
            {
                "learning_rate": lr,
                "actual_state_noise_std": noise,
                "completed_seed_count": sum(value is not None for value in mses),
                "nmse_eligible_count": sum(
                    value is not None
                    and value < float(main["analysis_eligibility_nmse_db_threshold"])
                    for value in nmses
                ),
                "mse_success_count": sum(
                    value is not None and value < float(main["mse_success_threshold"])
                    for value in mses
                ),
                "median_mse": float(np.median(mses)) if complete else None,
                "mean_mse": float(np.mean(mses)) if complete else None,
                "grid_order": tuning["learning_rate_grid"].index(lr),
                "per_seed": [
                    {
                        "seed": spec.model_seed,
                        "status": _result(spec).get("status"),
                        "mse": _metric(spec, "mse"),
                        "nmse_db": _metric(spec, "nmse_db"),
                    }
                    for spec in sorted(specs, key=lambda item: item.model_seed)
                ],
            }
        )
    rows.sort(
        key=lambda row: (
            -row["nmse_eligible_count"],
            -row["mse_success_count"],
            math.inf if row["median_mse"] is None else row["median_mse"],
            math.inf if row["mean_mse"] is None else row["mean_mse"],
            row["grid_order"],
        )
    )
    complete = [row for row in rows if row["completed_seed_count"] == expected_count]
    if not complete:
        raise RuntimeError("no complete pilot-screen LR candidate")
    return {
        "schema_version": 1,
        "selection_rule": tuning["selection_rule"],
        "all_learning_rates_fanned_out": True,
        "winner": complete[0],
        "cells": rows,
    }


def build_main_plan(
    root: Path,
    config: Mapping[str, Any],
    bank: Path,
    prefix: str,
    model_id: str,
    learning_rate_selection: Mapping[str, Any],
    *,
    rp: Mapping[str, Any] | None = None,
) -> tuple[RunSpec, ...]:
    winner = (
        learning_rate_selection["winner"]
        if "winner" in learning_rate_selection
        else learning_rate_selection
    )
    noise = _common_state_noise(config)
    if float(winner["actual_state_noise_std"]) != noise:
        raise ValueError("selected state noise differs from the common fixed contract")
    specs: list[RunSpec] = []
    for seed in config["main"]["seeds"]:
        run_id = f"{prefix}_main__seed{seed}"
        specs.append(
            RunSpec(
                run_id=run_id,
                stage=f"{prefix}_main",
                model_id=model_id,
                model_seed=int(seed),
                learning_rate=float(winner["learning_rate"]),
                actual_state_noise_std=noise,
                updates=int(config["training"]["updates"]),
                batch_size=int(config["training"]["batch_size"]),
                evaluation_bank=str(bank),
                campaign_root=str(root),
                output_dir=str(root / f"{prefix}_main" / "runs" / run_id),
                rp_enabled=rp is not None,
                rp_eta_lambda=None if rp is None else float(rp["eta_lambda"]),
                rp_damage_epsilon=None if rp is None else float(rp["damage_epsilon"]),
            )
        )
    return tuple(specs)


def summarize_main(specs: Sequence[RunSpec], config: Mapping[str, Any]) -> dict[str, Any]:
    main = config["main"]
    expected_seeds = set(map(int, config["main"]["seeds"]))
    if len(specs) != len(expected_seeds) or {spec.model_seed for spec in specs} != expected_seeds:
        raise ValueError("main must contain all registered fresh pilot seeds")
    rows = []
    for spec in sorted(specs, key=lambda item: item.model_seed):
        mse, nmse = _metric(spec, "mse"), _metric(spec, "nmse_db")
        rows.append(
            {
                "seed": spec.model_seed,
                "status": _result(spec).get("status"),
                "mse": mse,
                "nmse_db": nmse,
                "mse_success": mse is not None and mse < float(main["mse_success_threshold"]),
                "analysis_eligible": nmse is not None
                and nmse < float(main["analysis_eligibility_nmse_db_threshold"]),
            }
        )
    eligible = sum(row["analysis_eligible"] for row in rows)
    mse_success = sum(row["mse_success"] for row in rows)
    return {
        "schema_version": 1,
        "model_id": specs[0].model_id,
        "registered_seed_count": len(specs),
        "failed_seed_count": sum(row["status"] != "completed" for row in rows),
        "analysis_eligible_count": eligible,
        "analysis_eligible_rate": eligible / float(len(specs)),
        "mse_success_count_descriptive": mse_success,
        "mse_success_rate_descriptive": mse_success / float(len(specs)),
        "scientific_pass": eligible >= int(main["scientific_pass_minimum_eligible_per_model"]),
        "low_eligible_count_warning": eligible < int(main["low_eligible_count_warning_below"]),
        "per_seed": rows,
    }


def build_rp_sentinel_plan(
    root: Path,
    config: Mapping[str, Any],
    bank: Path,
    no_rp_selection: Mapping[str, Any],
) -> tuple[RunSpec, ...]:
    winner = no_rp_selection["winner"]
    rp, training = config["retention_plasticity"], config["training"]
    noise = _common_state_noise(config)
    if float(winner["actual_state_noise_std"]) != noise:
        raise ValueError("No-RP selection state noise differs from the common contract")
    specs: list[RunSpec] = []
    for eta in rp["eta_lambda_grid"]:
        for epsilon in rp["damage_epsilon_grid"]:
            run_id = (
                f"ca_rp_sentinel__eta{_float_key(eta)}__eps{_float_key(epsilon)}"
                f"__seed{rp['sentinel_seed']}"
            )
            specs.append(
                RunSpec(
                    run_id=run_id,
                    stage="ca_rp_sentinel",
                    model_id="ca_lru_n52",
                    model_seed=int(rp["sentinel_seed"]),
                    learning_rate=float(winner["learning_rate"]),
                    actual_state_noise_std=noise,
                    updates=int(training["updates"]),
                    batch_size=int(training["batch_size"]),
                    evaluation_bank=str(bank),
                    campaign_root=str(root),
                    output_dir=str(root / "ca_rp_sentinel" / "runs" / run_id),
                    rp_enabled=True,
                    rp_eta_lambda=float(eta),
                    rp_damage_epsilon=float(epsilon),
                )
            )
    return tuple(specs)


def screen_rp_sentinels(specs: Sequence[RunSpec], config: Mapping[str, Any]) -> dict[str, Any]:
    rp, main = config["retention_plasticity"], config["main"]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        mse, nmse = _metric(spec, "mse"), _metric(spec, "nmse_db")
        result = _result(spec)
        blank = result.get("heldout_blank_memory_mse") if result.get("status") == "completed" else None
        if blank is not None and not math.isfinite(float(blank)):
            blank = None
        rows.append(
            {
                "eta_lambda": spec.rp_eta_lambda,
                "damage_epsilon": spec.rp_damage_epsilon,
                "completed": mse is not None and nmse is not None and blank is not None,
                "mse": mse,
                "nmse_db": nmse,
                "blank_memory_mse": blank,
                "nmse_eligible": nmse is not None
                and nmse < float(main["analysis_eligibility_nmse_db_threshold"]),
                "mse_success": mse is not None and mse < float(main["mse_success_threshold"]),
                "grid_order": [
                    rp["eta_lambda_grid"].index(spec.rp_eta_lambda),
                    rp["damage_epsilon_grid"].index(spec.rp_damage_epsilon),
                ],
            }
        )
    rows.sort(
        key=lambda row: (
            not row["nmse_eligible"],
            not row["mse_success"],
            math.inf if row["blank_memory_mse"] is None else row["blank_memory_mse"],
            math.inf if row["mse"] is None else row["mse"],
            *row["grid_order"],
        )
    )
    complete = [row for row in rows if row["completed"]]
    if len(complete) < int(rp["top_k"]):
        raise RuntimeError("RP sentinel has fewer than three completed cells")
    return {"schema_version": 1, "top_cells": complete[: int(rp["top_k"])], "all_cells": rows}


def build_rp_fanout_plan(
    root: Path,
    config: Mapping[str, Any],
    bank: Path,
    no_rp_selection: Mapping[str, Any],
    screening: Mapping[str, Any],
) -> tuple[RunSpec, ...]:
    winner = no_rp_selection["winner"]
    rp, training = config["retention_plasticity"], config["training"]
    noise = _common_state_noise(config)
    if float(winner["actual_state_noise_std"]) != noise:
        raise ValueError("No-RP selection state noise differs from the common contract")
    specs: list[RunSpec] = []
    for cell in screening["top_cells"]:
        for seed in rp["fanout_seeds"]:
            run_id = (
                f"ca_rp_fanout__eta{_float_key(cell['eta_lambda'])}"
                f"__eps{_float_key(cell['damage_epsilon'])}__seed{seed}"
            )
            specs.append(
                RunSpec(
                    run_id=run_id,
                    stage="ca_rp_fanout",
                    model_id="ca_lru_n52",
                    model_seed=int(seed),
                    learning_rate=float(winner["learning_rate"]),
                    actual_state_noise_std=noise,
                    updates=int(training["updates"]),
                    batch_size=int(training["batch_size"]),
                    evaluation_bank=str(bank),
                    campaign_root=str(root),
                    output_dir=str(root / "ca_rp_fanout" / "runs" / run_id),
                    rp_enabled=True,
                    rp_eta_lambda=float(cell["eta_lambda"]),
                    rp_damage_epsilon=float(cell["damage_epsilon"]),
                )
            )
    return tuple(specs)


def select_rp(
    sentinel_specs: Sequence[RunSpec], fanout_specs: Sequence[RunSpec], config: Mapping[str, Any]
) -> dict[str, Any]:
    rp, main = config["retention_plasticity"], config["main"]
    expected_seeds = {int(rp["sentinel_seed"]), *map(int, rp["fanout_seeds"])}
    expected_count = len(expected_seeds)
    cells = sorted(
        {(spec.rp_eta_lambda, spec.rp_damage_epsilon) for spec in fanout_specs},
        key=lambda pair: (
            rp["eta_lambda_grid"].index(pair[0]), rp["damage_epsilon_grid"].index(pair[1])
        ),
    )
    rows: list[dict[str, Any]] = []
    for eta, epsilon in cells:
        specs = [
            spec
            for spec in (*sentinel_specs, *fanout_specs)
            if spec.rp_eta_lambda == eta and spec.rp_damage_epsilon == epsilon
        ]
        if len(specs) != expected_count or {spec.model_seed for spec in specs} != expected_seeds:
            raise RuntimeError("RP cell does not have all registered pilot seeds")
        mses = [_metric(spec, "mse") for spec in specs]
        nmses = [_metric(spec, "nmse_db") for spec in specs]
        blanks = [
            _result(spec).get("heldout_blank_memory_mse")
            if _result(spec).get("status") == "completed"
            else None
            for spec in specs
        ]
        complete = all(value is not None and math.isfinite(float(value)) for value in (*mses, *nmses, *blanks))
        rows.append(
            {
                "eta_lambda": eta,
                "damage_epsilon": epsilon,
                "completed_seed_count": expected_count if complete else 0,
                "nmse_eligible_count": sum(
                    value is not None
                    and value < float(main["analysis_eligibility_nmse_db_threshold"])
                    for value in nmses
                ),
                "mse_success_count": sum(
                    value is not None and value < float(main["mse_success_threshold"])
                    for value in mses
                ),
                "mean_blank_memory_mse": float(np.mean(blanks)) if complete else None,
                "mean_mse": float(np.mean(mses)) if complete else None,
                "grid_order": [
                    rp["eta_lambda_grid"].index(eta), rp["damage_epsilon_grid"].index(epsilon)
                ],
            }
        )
    rows.sort(
        key=lambda row: (
            -row["nmse_eligible_count"],
            -row["mse_success_count"],
            math.inf if row["mean_blank_memory_mse"] is None else row["mean_blank_memory_mse"],
            math.inf if row["mean_mse"] is None else row["mean_mse"],
            *row["grid_order"],
        )
    )
    complete = [row for row in rows if row["completed_seed_count"] == expected_count]
    if not complete:
        raise RuntimeError("no complete pilot RP cell")
    return {
        "schema_version": 1,
        "selection_rule": rp["selection_rule"],
        "winner": complete[0],
        "cells": rows,
    }


def _runtime_files() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        MODULE_DIR / "primary_v4.py",
        MODULE_DIR / "models.py",
        MODULE_DIR / "train.py",
        MODULE_DIR / "source_resolved_protocol.py",
        MODULE_DIR / "source_repaired_baselines_v6.py",
        MODULE_DIR / "tasks.py",
        MODULE_DIR / "metrics.py",
        MODULE_DIR / "artifacts.py",
    )


def _git_state(require_clean: bool) -> str:
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()
    if require_clean and dirty:
        raise RuntimeError("full downstream stages require a clean committed worktree")
    return commit


def _assert_worker_identity(
    identity: Mapping[str, Any],
    config_path: Path,
    root: Path,
    *,
    require_clean: bool,
) -> None:
    root = root.expanduser().resolve()
    if identity.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("downstream root campaign identity differs")
    if identity.get("protocol_revision") != PROTOCOL_REVISION:
        raise RuntimeError("downstream root protocol revision differs")
    if identity.get("track_classification") != TRACK_CLASSIFICATION:
        raise RuntimeError("downstream root track classification differs")
    identity_core = dict(identity)
    observed_scientific_identity = identity_core.pop("scientific_identity", None)
    if canonical_hash(identity_core) != observed_scientific_identity:
        raise RuntimeError("downstream root scientific identity digest differs")
    if strict_json_load(root / ROOT_MARKER) != identity:
        raise RuntimeError("downstream root marker changed")
    if strict_json_load(root / "baseline_parent_binding.json") != identity.get(
        "parent_binding"
    ):
        raise RuntimeError("downstream baseline-parent binding differs")
    if _git_state(require_clean) != identity.get("code_commit"):
        raise RuntimeError("downstream worker commit differs from campaign root")
    current = {path.name: sha256_file(path) for path in _runtime_files()}
    if current != identity.get("runtime_code_sha256"):
        raise RuntimeError("downstream worker code hashes differ from campaign root")
    if sha256_file(config_path) != identity.get("config_sha256"):
        raise RuntimeError("downstream worker config differs from campaign root")
    if sha256_file(FREEZE_DOCUMENT) != identity.get("freeze_sha256"):
        raise RuntimeError("downstream worker freeze differs from campaign root")


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError(f"bound copy differs: {destination}")
    else:
        shutil.copy2(source, destination)


def _prepare_root(
    root: Path,
    baseline_root: Path,
    config_source: Path,
    *,
    require_clean: bool,
) -> tuple[dict[str, Any], Path]:
    root = root.expanduser().resolve()
    baseline_root = baseline_root.expanduser().resolve()
    config_source = config_source.expanduser().resolve(strict=True)
    config = load_config(config_source)
    require_baseline_main(baseline_root)
    baseline_marker = baseline_root / BASELINE_ROOT_MARKER
    baseline_main = baseline_root / "main"
    baseline_config = load_baseline_config(baseline_root / "inputs" / BASELINE_CONFIG.name)
    if (
        float(baseline_config["training"]["actual_post_transition_state_noise_std"])
        != _common_state_noise(config)
    ):
        raise RuntimeError("downstream common state noise differs from baseline parent")
    if (
        float(baseline_config["training"]["controlled_target_noise_std"])
        != float(config["training"]["controlled_target_noise_std"])
        or float(config["training"]["controlled_target_noise_std"]) != 0.0
    ):
        raise RuntimeError("downstream zero target-noise contract differs from baseline parent")
    if (
        float(baseline_config["training"]["controlled_output_dropout"])
        != float(config["training"]["controlled_output_dropout"])
        or float(config["training"]["controlled_output_dropout"]) != 0.0
    ):
        raise RuntimeError("downstream zero-dropout contract differs from baseline parent")
    parent_binding = {
        "schema_version": 1,
        "baseline_campaign_id": BASELINE_CAMPAIGN_ID,
        "baseline_root_marker_sha256": sha256_file(baseline_marker),
        "baseline_main_completion_receipt_sha256": sha256_file(
            baseline_main / "completion_receipt.json"
        ),
        "baseline_scientific_gate_sha256": sha256_file(baseline_main / "scientific_gate.json"),
        "baseline_scientific_pass_sha256": (
            sha256_file(baseline_main / "SCIENTIFIC_PASS")
            if (baseline_main / "SCIENTIFIC_PASS").is_file()
            else None
        ),
        "baseline_all_models_scientifically_eligible": (
            baseline_main / "SCIENTIFIC_PASS"
        ).is_file(),
        "banks": {},
    }
    for purpose in ("tuning", "main_test"):
        source = baseline_root / "banks" / f"{purpose}.npz"
        sidecar = source.with_suffix(source.suffix + ".sha256")
        batch = load_fixed_bank(source)
        _validate_bank(
            batch, trials=batch.batch_size, config=baseline_config, purpose=purpose
        )
        parent_binding["banks"][purpose] = {
            "source_archive_sha256": sha256_file(source),
            "source_sidecar_sha256": sha256_file(sidecar),
        }
    identity = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "track_classification": TRACK_CLASSIFICATION,
        "config_sha256": sha256_file(config_source),
        "freeze_sha256": sha256_file(FREEZE_DOCUMENT),
        "runtime_code_sha256": {path.name: sha256_file(path) for path in _runtime_files()},
        "code_commit": _git_state(require_clean),
        "parent_binding": parent_binding,
    }
    identity["scientific_identity"] = canonical_hash(identity)
    marker = root / ROOT_MARKER
    root.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        if strict_json_load(marker) != identity:
            raise RuntimeError("downstream root scientific identity differs")
    elif any(root.iterdir()):
        raise RuntimeError("unmarked downstream root must be empty")
    else:
        for purpose in ("tuning", "main_test"):
            source = baseline_root / "banks" / f"{purpose}.npz"
            destination = root / "banks" / source.name
            _copy_exact(source, destination)
            _copy_exact(
                source.with_suffix(source.suffix + ".sha256"),
                destination.with_suffix(destination.suffix + ".sha256"),
            )
        _copy_exact(config_source, root / "inputs" / DEFAULT_CONFIG.name)
        _copy_exact(
            baseline_root / "inputs" / BASELINE_CONFIG.name,
            root / "inputs" / BASELINE_CONFIG.name,
        )
        _copy_exact(FREEZE_DOCUMENT, root / "inputs" / FREEZE_DOCUMENT.name)
        atomic_json(root / "baseline_parent_binding.json", parent_binding)
        atomic_json(marker, identity)
    _copy_exact(config_source, root / "inputs" / DEFAULT_CONFIG.name)
    _copy_exact(
        baseline_root / "inputs" / BASELINE_CONFIG.name,
        root / "inputs" / BASELINE_CONFIG.name,
    )
    _copy_exact(FREEZE_DOCUMENT, root / "inputs" / FREEZE_DOCUMENT.name)
    if strict_json_load(root / "baseline_parent_binding.json") != parent_binding:
        raise RuntimeError("baseline parent binding changed")
    for purpose in ("tuning", "main_test"):
        source = baseline_root / "banks" / f"{purpose}.npz"
        copied = root / "banks" / source.name
        if sha256_file(source) != sha256_file(copied):
            raise RuntimeError("copied fixed bank is not byte-identical to baseline")
        if sha256_file(source.with_suffix(source.suffix + ".sha256")) != sha256_file(
            copied.with_suffix(copied.suffix + ".sha256")
        ):
            raise RuntimeError("copied fixed-bank sidecar differs from baseline")
        batch = load_fixed_bank(copied)
        _validate_bank(
            batch, trials=batch.batch_size, config=baseline_config, purpose=purpose
        )
    return config, root / "inputs" / DEFAULT_CONFIG.name


def _train_worker(spec: RunSpec, config_path: Path, device_text: str) -> Path:
    config = load_config(config_path)
    root = Path(spec.campaign_root).resolve()
    _validate_spec(config, spec, require_registered_plan=True)
    identity = strict_json_load(root / ROOT_MARKER)
    _assert_worker_identity(
        identity, config_path, root, require_clean=not spec.smoke
    )
    output = Path(spec.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("worker output is not empty")
    device = torch.device(device_text)
    _configure_determinism(spec.model_seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model = build_v4_model(spec.model_id).to(device)
    initial_state_dict_sha256 = canonical_tensor_mapping_sha256(model.state_dict())
    rng_stream_identities = _rng_stream_identities(spec)
    _finite_model(model)
    total = sum(parameter.numel() for parameter in model.parameters())
    gradient = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if total != TOTAL_COUNTS[spec.model_id] or gradient != GRADIENT_COUNTS[spec.model_id]:
        raise RuntimeError(f"parameter count mismatch for {spec.model_id}: {total}/{gradient}")
    bank = _to_device_batch(load_fixed_bank(spec.evaluation_bank), device)
    baseline_config = load_baseline_config(root / "inputs" / BASELINE_CONFIG.name)
    _validate_bank(
        bank,
        trials=bank.batch_size,
        config=baseline_config,
        purpose=Path(spec.evaluation_bank).stem,
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(spec.learning_rate),
        betas=tuple(float(value) for value in config["training"]["betas"]),
        eps=float(config["training"]["epsilon"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    downstream_model_metadata = dict(model.metadata())
    downstream_model_metadata["initial_state"] = (
        "linear_Wotr_source_q1_post_update_target"
    )
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "track_classification": TRACK_CLASSIFICATION,
        "run": spec.payload(),
        "scientific_identity": identity["scientific_identity"],
        "runtime_code_sha256": identity["runtime_code_sha256"],
        "code_commit": identity["code_commit"],
        "config_sha256": identity["config_sha256"],
        "freeze_sha256": identity["freeze_sha256"],
        "baseline_parent_binding": identity["parent_binding"],
        "model": downstream_model_metadata,
        "downstream_initial_state_semantics": "source_q1_post_update_target",
        "initial_state_argument_passed_to_model": "clean_batch.output_targets[0]",
        "controlled_target_noise_std": config["training"][
            "controlled_target_noise_std"
        ],
        "controlled_output_dropout": config["training"]["controlled_output_dropout"],
        "training_target_semantics": config["training"]["training_target_semantics"],
        "loss": "clean_masked_mse_on_cos_sin_targets",
        "parameters_total": total,
        "parameters_gradient_trainable": gradient,
        "initial_state_dict_sha256": initial_state_dict_sha256,
        "rng_stream_identities": rng_stream_identities,
        "actual_post_transition_state_noise_std": spec.actual_state_noise_std,
        "state_noise_location": "disabled_no_state_noise_injection",
        "state_carrier_provenance": (
            "post_transition_full_104d_real_imag_carrier"
            if spec.model_id == "lru_n52"
            else "post_transition_full_52d_real_carrier"
        ),
        "state_noise_scale_semantics": "not_applicable_disabled",
        "paper_literal_state_noise_std_provenance_only": 0.1,
        "paper_state_noise_covariance_provenance_only": "0.01I",
        "evaluation_bank_sha256": sha256_file(spec.evaluation_bank),
        "online_stream_namespace": BASELINE_CAMPAIGN_ID,
        "ca_learning_rate_independently_tuned": (
            False if spec.model_id == "ca_lru_n52" else None
        ),
        "started_at_utc": _utc_now(),
        "device": device_text,
        "device_provenance": _device_provenance(device),
        "worker_threads": torch.get_num_threads(),
    }
    atomic_json(output / "run_manifest.json", _native(manifest))
    trace: list[dict[str, Any]] = []
    rp_trace: list[dict[str, Any]] = []
    started = time.time()
    rp = config["retention_plasticity"]
    for update in range(1, spec.updates + 1):
        model.train()
        batch = _training_batch(spec, update, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_sequence(
            batch.inputs,
            initial_memory=_source_q1(batch),
            state_noise_std=float(spec.actual_state_noise_std),
            noise_generator=None,
        )
        loss = masked_mse(prediction, batch.output_targets, batch.mask)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()
        _finite_model(model, gradients=True)
        optimizer.step()
        _finite_model(model)
        if (
            spec.rp_enabled
            and update > int(rp["warmup_updates"])
            and update % int(rp["interval_updates"]) == 0
        ):
            from .train import _retention_plasticity_call

            probe = _rp_probe(spec, config, update, device)
            values = _retention_plasticity_call(
                model,
                probe,
                blank_horizon=int(rp["blank_ablation_horizon"]),
                eta_lambda=float(spec.rp_eta_lambda),
                damage_epsilon=float(spec.rp_damage_epsilon),
                initial_memory=_source_q1(probe),
            )
            rp_trace.append({"update": update, **_native(values)})
        should_trace = update == 1 or update % int(config["training"]["trace_interval"]) == 0 or update == spec.updates
        should_validate = update % int(config["training"]["validation_interval"]) == 0 or update == spec.updates
        if should_trace or should_validate:
            row: dict[str, Any] = {
                "update": update,
                "train_mse": float(loss.detach().cpu()),
                "elapsed_seconds": float(time.time() - started),
            }
            if should_validate:
                row["validation"] = _evaluate(model, bank)
            trace.append(row)
            atomic_json(output / "training_trace.json", trace)
            atomic_json(output / "rp_trace.json", rp_trace)
            atomic_json(
                output / "progress.json",
                {
                    "status": "running",
                    "run_id": spec.run_id,
                    "update": update,
                    "updates_total": spec.updates,
                    "latest": row,
                    "updated_at_utc": _utc_now(),
                },
            )
    final_metrics = _evaluate(model, bank)
    _assert_worker_identity(
        identity, config_path, root, require_clean=not spec.smoke
    )
    # Smoke has only two updates (therefore zero scheduled RP calls).  Running
    # the full 4,096-step blank metric over the 1,024-example bank would make
    # smoke slower without exercising RP; the reduced RP API has a direct test.
    blank = (
        _blank_mse(model, bank, int(rp["selection_blank_horizon"]))
        if spec.rp_enabled and not spec.smoke
        else None
    )
    result = {
        "schema_version": 1,
        "status": "completed",
        "run_id": spec.run_id,
        "stage": spec.stage,
        "model_id": spec.model_id,
        "model_seed": spec.model_seed,
        "learning_rate": spec.learning_rate,
        "actual_state_noise_std": spec.actual_state_noise_std,
        "rp_enabled": spec.rp_enabled,
        "rp_eta_lambda": spec.rp_eta_lambda,
        "rp_damage_epsilon": spec.rp_damage_epsilon,
        "updates_completed": spec.updates,
        "final_metrics": final_metrics,
        "heldout_blank_memory_mse": blank,
        "rp_call_count": len(rp_trace),
        "counts_in_denominator": True,
    }
    atomic_json(output / "training_trace.json", trace)
    atomic_json(output / "rp_trace.json", rp_trace)
    atomic_json(output / "result.json", result)
    atomic_json(output / "progress.json", {"status": "completed", "run_id": spec.run_id, "update": spec.updates})
    checkpoint = output / "checkpoint_final.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "schema_version": 1,
            "checkpoint_type": CAMPAIGN_ID,
            "run": spec.payload(),
            "result": result,
            "initial_state_dict_sha256": initial_state_dict_sha256,
            "rng_stream_identities": rng_stream_identities,
            "state_dict": model.state_dict(),
        },
    )
    atomic_json(output / "COMPLETE", {"run_id": spec.run_id, "status": "completed"})
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=spec.run_id,
        artifacts=[
            output / "run_manifest.json",
            output / "progress.json",
            output / "training_trace.json",
            output / "rp_trace.json",
            output / "result.json",
            checkpoint,
            output / "COMPLETE",
        ],
        metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
        },
    )
    return output


def _record_numerical_failure(
    spec: RunSpec, config_path: Path, device_text: str, error: BaseException
) -> Path:
    output = Path(spec.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(spec.campaign_root).resolve()
    config = load_config(config_path)
    _validate_spec(config, spec, require_registered_plan=True)
    identity = strict_json_load(root / ROOT_MARKER)
    _assert_worker_identity(
        identity, config_path, root, require_clean=not spec.smoke
    )
    _configure_determinism(spec.model_seed)
    reconstructed_initial_model = build_v4_model(spec.model_id)
    initial_state_dict_sha256 = canonical_tensor_mapping_sha256(
        reconstructed_initial_model.state_dict()
    )
    rng_stream_identities = _rng_stream_identities(spec)
    manifest = output / "run_manifest.json"
    if not manifest.exists():
        model_metadata = dict(reconstructed_initial_model.metadata())
        model_metadata["initial_state"] = (
            "linear_Wotr_source_q1_post_update_target"
        )
        atomic_json(
            manifest,
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "protocol_revision": PROTOCOL_REVISION,
                "track_classification": TRACK_CLASSIFICATION,
                "run": spec.payload(),
                "scientific_identity": identity["scientific_identity"],
                "runtime_code_sha256": identity["runtime_code_sha256"],
                "code_commit": identity["code_commit"],
                "config_sha256": identity["config_sha256"],
                "freeze_sha256": identity["freeze_sha256"],
                "baseline_parent_binding": identity["parent_binding"],
                "model": model_metadata,
                "downstream_initial_state_semantics": "source_q1_post_update_target",
                "initial_state_argument_passed_to_model": "clean_batch.output_targets[0]",
                "controlled_target_noise_std": 0.0,
                "controlled_output_dropout": 0.0,
                "training_target_semantics": "clean_cos_sin_target_for_initial_q1_and_loss",
                "loss": "clean_masked_mse_on_cos_sin_targets",
                "parameters_total": TOTAL_COUNTS[spec.model_id],
                "parameters_gradient_trainable": GRADIENT_COUNTS[spec.model_id],
                "initial_state_dict_sha256": initial_state_dict_sha256,
                "rng_stream_identities": rng_stream_identities,
                "actual_post_transition_state_noise_std": spec.actual_state_noise_std,
                "state_noise_location": "disabled_no_state_noise_injection",
                "state_carrier_provenance": (
                    "post_transition_full_104d_real_imag_carrier"
                    if spec.model_id == "lru_n52"
                    else "post_transition_full_52d_real_carrier"
                ),
                "state_noise_scale_semantics": "not_applicable_disabled",
                "paper_literal_state_noise_std_provenance_only": 0.1,
                "paper_state_noise_covariance_provenance_only": "0.01I",
                "evaluation_bank_sha256": sha256_file(spec.evaluation_bank),
                "online_stream_namespace": BASELINE_CAMPAIGN_ID,
                "ca_learning_rate_independently_tuned": (
                    False if spec.model_id == "ca_lru_n52" else None
                ),
                "device": device_text,
                "failure_manifest_synthesized": True,
            },
        )
    failure = {
        "schema_version": 1,
        "status": "failed",
        "run_id": spec.run_id,
        "failure_kind": type(error).__name__,
        "failure_message": str(error),
        "traceback": traceback.format_exc(),
        "counts_as_failure_in_denominator": True,
    }
    result = {
        "schema_version": 1,
        "status": "failed",
        "run_id": spec.run_id,
        "stage": spec.stage,
        "model_id": spec.model_id,
        "model_seed": spec.model_seed,
        "learning_rate": spec.learning_rate,
        "actual_state_noise_std": spec.actual_state_noise_std,
        "rp_enabled": spec.rp_enabled,
        "rp_eta_lambda": spec.rp_eta_lambda,
        "rp_damage_epsilon": spec.rp_damage_epsilon,
        "updates_completed": None,
        "rp_call_count": None,
        "final_metrics": None,
        "heldout_blank_memory_mse": None,
        "counts_in_denominator": True,
    }
    atomic_json(output / "failure.json", failure)
    atomic_json(output / "result.json", result)
    checkpoint = output / "checkpoint_failure.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "checkpoint_type": f"{CAMPAIGN_ID}_failure",
            "run": spec.payload(),
            "failure": failure,
            "initial_state_dict_sha256": initial_state_dict_sha256,
            "rng_stream_identities": rng_stream_identities,
        },
    )
    atomic_json(output / "FAILED", {"run_id": spec.run_id, "status": "failed"})
    artifacts = [manifest, output / "failure.json", output / "result.json", checkpoint, output / "FAILED"]
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=spec.run_id,
        artifacts=artifacts,
        metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
        },
    )
    return output


def _receipt_names(output: Path) -> set[str] | None:
    try:
        receipt = strict_json_load(output / "completion_receipt.json")
        return set(receipt["artifacts"])
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _finite_json_numbers(value: Any) -> bool:
    """Return false when a nested JSON-compatible payload contains NaN/Inf."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_json_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_json_numbers(item) for item in value)
    return False


def _expected_rp_updates(
    spec: RunSpec, config: Mapping[str, Any]
) -> tuple[int, ...]:
    """Exact downstream RP schedule (smoke has no post-warmup update)."""

    if not spec.rp_enabled:
        return ()
    rp = config["retention_plasticity"]
    warmup = int(rp["warmup_updates"])
    interval = int(rp["interval_updates"])
    return tuple(
        update
        for update in range(1, int(spec.updates) + 1)
        if update > warmup and update % interval == 0
    )


def _verified(spec: RunSpec) -> bool:
    output = Path(spec.output_dir)
    try:
        root = Path(spec.campaign_root).resolve()
        config = load_config(root / "inputs" / DEFAULT_CONFIG.name)
        # Stage validation owns recursive parent/plan validation once. Child
        # receipt validation repeats only the local immutable spec contract.
        _validate_spec(config, spec, require_registered_plan=False)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        return False
    try:
        result = strict_json_load(output / "result.json")
    except (OSError, ValueError):
        return False
    if result.get("status") == "completed":
        required = {
            "run_manifest.json",
            "progress.json",
            "training_trace.json",
            "rp_trace.json",
            "result.json",
            "checkpoint_final.pt",
            "COMPLETE",
        }
        checkpoint_path = output / "checkpoint_final.pt"
        checkpoint_type = CAMPAIGN_ID
    elif result.get("status") == "failed":
        required = {
            "run_manifest.json",
            "failure.json",
            "result.json",
            "checkpoint_failure.pt",
            "FAILED",
        }
        checkpoint_path = output / "checkpoint_failure.pt"
        checkpoint_type = f"{CAMPAIGN_ID}_failure"
    else:
        return False
    valid, _ = verify_completion_receipt(
        output / "completion_receipt.json",
        expected_job_id=spec.run_id,
        expected_metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
        },
    )
    if not valid or _receipt_names(output) != required:
        return False
    try:
        manifest = strict_json_load(output / "run_manifest.json")
        identity = strict_json_load(root / ROOT_MARKER)
        parent_binding = strict_json_load(root / "baseline_parent_binding.json")
        identity_core = dict(identity)
        observed_identity_digest = identity_core.pop("scientific_identity")
        if canonical_hash(identity_core) != observed_identity_digest:
            return False
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _configure_determinism(spec.model_seed)
        reconstructed = build_v4_model(spec.model_id)
        expected_initial_hash = canonical_tensor_mapping_sha256(
            reconstructed.state_dict()
        )
        expected_rng_streams = _rng_stream_identities(spec)
        expected_model_metadata = dict(reconstructed.metadata())
        expected_model_metadata["initial_state"] = (
            "linear_Wotr_source_q1_post_update_target"
        )
        evaluation_bank_sha256 = sha256_file(spec.evaluation_bank)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        return False
    expected_carrier = (
        "post_transition_full_104d_real_imag_carrier"
        if spec.model_id == "lru_n52"
        else "post_transition_full_52d_real_carrier"
    )
    manifest_identity = (
        manifest.get("campaign_id") == CAMPAIGN_ID
        and manifest.get("protocol_revision") == PROTOCOL_REVISION
        and manifest.get("track_classification") == TRACK_CLASSIFICATION
        and manifest.get("run") == spec.payload()
        and manifest.get("scientific_identity") == identity.get("scientific_identity")
        and manifest.get("runtime_code_sha256") == identity.get("runtime_code_sha256")
        and manifest.get("code_commit") == identity.get("code_commit")
        and manifest.get("config_sha256") == identity.get("config_sha256")
        and manifest.get("freeze_sha256") == identity.get("freeze_sha256")
        and identity.get("parent_binding") == parent_binding
        and manifest.get("baseline_parent_binding") == parent_binding
        and manifest.get("model") == expected_model_metadata
        and manifest.get("parameters_total") == TOTAL_COUNTS[spec.model_id]
        and manifest.get("parameters_gradient_trainable")
        == GRADIENT_COUNTS[spec.model_id]
        and manifest.get("downstream_initial_state_semantics")
        == "source_q1_post_update_target"
        and manifest.get("initial_state_argument_passed_to_model")
        == "clean_batch.output_targets[0]"
        and manifest.get("controlled_target_noise_std") == 0.0
        and manifest.get("controlled_output_dropout") == 0.0
        and manifest.get("training_target_semantics")
        == "clean_cos_sin_target_for_initial_q1_and_loss"
        and manifest.get("loss") == "clean_masked_mse_on_cos_sin_targets"
        and manifest.get("actual_post_transition_state_noise_std")
        == spec.actual_state_noise_std
        and manifest.get("state_noise_location") == "disabled_no_state_noise_injection"
        and manifest.get("state_carrier_provenance") == expected_carrier
        and manifest.get("state_noise_scale_semantics") == "not_applicable_disabled"
        and manifest.get("paper_literal_state_noise_std_provenance_only") == 0.1
        and manifest.get("paper_state_noise_covariance_provenance_only") == "0.01I"
        and manifest.get("online_stream_namespace") == BASELINE_CAMPAIGN_ID
        and manifest.get("ca_learning_rate_independently_tuned")
        == (False if spec.model_id == "ca_lru_n52" else None)
        and manifest.get("evaluation_bank_sha256") == evaluation_bank_sha256
    )
    if not manifest_identity:
        return False
    if checkpoint.get("checkpoint_type") != checkpoint_type or checkpoint.get("run") != spec.payload():
        return False
    if result.get("status") == "completed":
        try:
            state_dict = checkpoint.get("state_dict")
            if not isinstance(state_dict, Mapping):
                return False
            reconstructed.load_state_dict(state_dict, strict=True)
            _finite_model(reconstructed)
            marker = strict_json_load(output / "COMPLETE")
            rp_trace = strict_json_load(output / "rp_trace.json")
        except (OSError, ValueError, RuntimeError, TypeError, KeyError):
            return False
        metrics = result.get("final_metrics")
        if not isinstance(metrics, dict) or any(
            key not in metrics or not math.isfinite(float(metrics[key]))
            for key in ("mse", "nmse_db", "masked_mse", "masked_nmse_db")
        ):
            return False
        if (
            metrics["mse"] != metrics["masked_mse"]
            or metrics["nmse_db"] != metrics["masked_nmse_db"]
        ):
            return False
        expected_rp_updates = _expected_rp_updates(spec, config)
        expected_rp_keys = {
            "update",
            "clean_rmse",
            "damage_mean",
            "damage_max",
            "damage_positive_fraction",
        }
        if (
            not isinstance(rp_trace, list)
            or not _finite_json_numbers(rp_trace)
            or any(
                not isinstance(item, dict) or set(item) != expected_rp_keys
                for item in rp_trace
            )
            or tuple(item.get("update") for item in rp_trace if isinstance(item, dict))
            != expected_rp_updates
            or len(rp_trace) != len(expected_rp_updates)
            or result.get("rp_call_count") != len(expected_rp_updates)
        ):
            return False
        blank = result.get("heldout_blank_memory_mse")
        if spec.rp_enabled and not spec.smoke:
            if not isinstance(blank, (int, float)) or not math.isfinite(float(blank)):
                return False
        elif blank is not None:
            return False
        checkpoint_identity = (
            checkpoint.get("result") == result
            and manifest.get("initial_state_dict_sha256") == expected_initial_hash
            and checkpoint.get("initial_state_dict_sha256") == expected_initial_hash
            and manifest.get("rng_stream_identities") == expected_rng_streams
            and checkpoint.get("rng_stream_identities") == expected_rng_streams
        )
        outcome_identity = marker.get("run_id") == spec.run_id
    else:
        try:
            failure = strict_json_load(output / "failure.json")
            marker = strict_json_load(output / "FAILED")
        except (OSError, ValueError):
            return False
        checkpoint_identity = (
            checkpoint.get("failure") == failure
            and manifest.get("initial_state_dict_sha256") == expected_initial_hash
            and checkpoint.get("initial_state_dict_sha256") == expected_initial_hash
            and manifest.get("rng_stream_identities") == expected_rng_streams
            and checkpoint.get("rng_stream_identities") == expected_rng_streams
        )
        outcome_identity = (
            marker.get("run_id") == spec.run_id
            and failure.get("run_id") == spec.run_id
            and failure.get("status") == "failed"
            and failure.get("counts_as_failure_in_denominator") is True
            and result.get("counts_in_denominator") is True
        )
    result_identity = (
        result.get("run_id") == spec.run_id
        and result.get("stage") == spec.stage
        and result.get("model_id") == spec.model_id
        and result.get("model_seed") == spec.model_seed
        and result.get("learning_rate") == spec.learning_rate
        and result.get("actual_state_noise_std") == spec.actual_state_noise_std
        and result.get("rp_enabled") is spec.rp_enabled
        and result.get("rp_eta_lambda") == spec.rp_eta_lambda
        and result.get("rp_damage_epsilon") == spec.rp_damage_epsilon
        and result.get("counts_in_denominator") is True
    )
    if result.get("status") == "completed":
        result_identity = bool(
            result_identity
            and result.get("updates_completed") == spec.updates
        )
    else:
        result_identity = bool(
            result_identity
            and result.get("updates_completed") is None
            and result.get("rp_call_count") is None
            and result.get("final_metrics") is None
            and result.get("heldout_blank_memory_mse") is None
        )
    return bool(
        checkpoint_identity
        and outcome_identity
        and result_identity
    )


def _write_or_verify(path: Path, payload: Mapping[str, Any]) -> None:
    native = _native(payload)
    if path.exists():
        if strict_json_load(path) != native:
            raise RuntimeError(f"immutable artifact differs: {path}")
    else:
        atomic_json(path, native)


def _write_plan(stage_root: Path, specs: Sequence[RunSpec]) -> None:
    _write_or_verify(
        stage_root / "plan.json",
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "run_count": len(specs),
            "runs": [spec.payload() for spec in specs],
        },
    )


def _archive_partial(output: Path, attempts: Path) -> None:
    if not output.exists():
        return
    attempts.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    os.replace(output, attempts / f"{output.name}.{stamp}")


def _run_specs(specs: Sequence[RunSpec], config_path: Path, slots: Sequence[str]) -> None:
    if not specs:
        return
    stage_root = Path(specs[0].output_dir).parents[1]
    _write_plan(stage_root, specs)
    pending = [spec for spec in specs if not _verified(spec)]
    for spec in pending:
        _archive_partial(Path(spec.output_dir), stage_root / "attempts")
    specs_dir, logs_dir = stage_root / "specs", stage_root / "logs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    queue = list(pending)
    running: dict[str, tuple[subprocess.Popen[Any], RunSpec, Any]] = {}
    repo = Path(__file__).resolve().parents[2]
    try:
        while queue or running:
            for slot in [value for value in slots if value not in running]:
                if not queue:
                    break
                spec = queue.pop(0)
                spec_file = specs_dir / f"{spec.run_id}.json"
                atomic_json(spec_file, spec.payload())
                handle = (logs_dir / f"{spec.run_id}.log").open("ab")
                command = [
                    sys.executable,
                    "-m",
                    "repro.sagodi_protocol.source_repaired_lru_calru_v6",
                    "--worker-spec",
                    str(spec_file),
                    "--config",
                    str(config_path),
                    "--device",
                    "cpu" if slot == "cpu" else "cuda:0",
                ]
                environment = os.environ.copy()
                for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    environment[variable] = "1"
                if slot != "cpu":
                    environment["CUDA_VISIBLE_DEVICES"] = slot
                    environment["CALRU_PHYSICAL_GPU_SLOT"] = slot
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
                    raise RuntimeError(
                        f"retryable infrastructure/worker failure: {spec.run_id}; exit={code}"
                    )
    finally:
        for process, _, handle in running.values():
            if process.poll() is None:
                process.terminate()
            handle.close()


def _children_binding(specs: Sequence[RunSpec]) -> dict[str, Any]:
    children = []
    for spec in specs:
        output = Path(spec.output_dir)
        result = strict_json_load(output / "result.json")
        checkpoint = output / (
            "checkpoint_final.pt" if result.get("status") == "completed" else "checkpoint_failure.pt"
        )
        children.append(
            {
                "run_id": spec.run_id,
                "status": result.get("status"),
                "receipt_sha256": sha256_file(output / "completion_receipt.json"),
                "result_sha256": sha256_file(output / "result.json"),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
    return {"schema_version": 1, "campaign_id": CAMPAIGN_ID, "children": children}


def _finalize(stage_root: Path, stage: str, specs: Sequence[RunSpec], extras: Sequence[Path]) -> None:
    if not all(_verified(spec) for spec in specs):
        raise RuntimeError(f"cannot finalize incomplete {stage}")
    binding = stage_root / "children_binding.json"
    _write_or_verify(binding, _children_binding(specs))
    complete = stage_root / "COMPUTATION_COMPLETE"
    atomic_json(complete, {"campaign_id": CAMPAIGN_ID, "stage": stage, "run_count": len(specs)})
    write_completion_receipt(
        stage_root / "completion_receipt.json",
        job_id=f"{CAMPAIGN_ID}__{stage}",
        artifacts=[stage_root / "plan.json", binding, complete, *extras],
        metadata={"campaign_id": CAMPAIGN_ID, "stage": stage, "run_count": len(specs)},
    )


def _parse_slots(text: str) -> tuple[str, ...]:
    if text.strip().lower() == "cpu":
        return ("cpu",)
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values or any(not value.isdigit() for value in values) or len(values) != len(set(values)):
        raise ValueError("--gpus must be unique comma-separated ids or cpu")
    return values


def _read_specs(stage_root: Path) -> tuple[RunSpec, ...]:
    plan = strict_json_load(stage_root / "plan.json")
    specs = tuple(RunSpec(**row) for row in plan["runs"])
    if plan.get("campaign_id") != CAMPAIGN_ID or plan.get("run_count") != len(specs):
        raise ValueError("stage plan identity/count differs")
    return specs


def _parent_payload(root: Path, stage: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "baseline_parent_binding_sha256": sha256_file(root / "baseline_parent_binding.json"),
    }
    if stage in {"smoke", "lru_sentinel"}:
        return payload
    if stage == "no_rp_sentinel":
        parent_stage, artifact = "lru_main", "summary.json"
    elif stage == "ca_rp_sentinel":
        parent_stage, artifact = "no_rp_main", "summary.json"
    elif stage.endswith("_fanout"):
        parent_stage, artifact = stage.replace("_fanout", "_sentinel"), "screening.json"
    elif stage.endswith("_main"):
        parent_stage, artifact = stage.replace("_main", "_fanout"), "selection.json"
    else:
        raise ValueError(f"unknown parent for {stage}")
    parent = root / parent_stage
    payload.update(
        {
            "immediate_parent_stage": parent_stage,
            "immediate_parent_completion_receipt_sha256": sha256_file(
                parent / "completion_receipt.json"
            ),
            "immediate_parent_artifact": artifact,
            "immediate_parent_artifact_sha256": sha256_file(parent / artifact),
        }
    )
    return payload


def _main_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_id": summary["model_id"],
        "scientific_pass": bool(summary["scientific_pass"]),
        "gate": "at_least_one_fresh_seed_nmse_below_minus20db",
        "mse_below_0p01_is_descriptive_only": True,
        "low_eligible_count_warning": bool(summary["low_eligible_count_warning"]),
    }


def _stage_expected_names(stage: str, scientific_pass: bool = False) -> set[str]:
    names = {"plan.json", "children_binding.json", "COMPUTATION_COMPLETE", "parent_binding.json"}
    if stage == "smoke":
        names.add("summary.json")
    elif stage.endswith("_sentinel"):
        names.add("screening.json")
    elif stage.endswith("_fanout"):
        names.add("selection.json")
    elif stage.endswith("_main"):
        names.update({"summary.json", "scientific_gate.json"})
        if scientific_pass:
            names.add("SCIENTIFIC_PASS")
    return names


def _stage_valid(root: Path, stage: str, expected_runs: int) -> bool:
    stage_root = root / stage
    valid, _ = verify_completion_receipt(
        stage_root / "completion_receipt.json",
        expected_job_id=f"{CAMPAIGN_ID}__{stage}",
        expected_metadata={"campaign_id": CAMPAIGN_ID, "stage": stage, "run_count": expected_runs},
    )
    if not valid or not (stage_root / "COMPUTATION_COMPLETE").is_file():
        return False
    try:
        config = load_config(root / "inputs" / DEFAULT_CONFIG.name)
        _validate_stage_dependencies(root, stage)
        specs = _read_specs(stage_root)
        expected_specs = _expected_plan_for_stage(root, config, stage)
        if (
            len(specs) != expected_runs
            or len(expected_specs) != expected_runs
            or specs != expected_specs
        ):
            return False
        if not all(_verified(spec) for spec in specs):
            return False
        if strict_json_load(stage_root / "children_binding.json") != _children_binding(specs):
            return False
        if strict_json_load(stage_root / "parent_binding.json") != _parent_payload(root, stage):
            return False
        baseline_config = load_baseline_config(root / "inputs" / BASELINE_CONFIG.name)
        for path in {spec.evaluation_bank for spec in specs}:
            bank = load_fixed_bank(path)
            _validate_bank(
                bank,
                trials=bank.batch_size,
                config=baseline_config,
                purpose=Path(path).stem,
            )
        science = False
        if stage == "smoke":
            expected = {
                "schema_version": 1,
                "runs": [
                    {"run_id": spec.run_id, "status": _result(spec).get("status")}
                    for spec in specs
                ],
                "scientific_result": False,
            }
            if strict_json_load(stage_root / "summary.json") != expected:
                return False
        elif stage.endswith("_sentinel"):
            expected = (
                screen_rp_sentinels(specs, config)
                if stage == "ca_rp_sentinel"
                else screen_sentinels(specs, config)
            )
            if strict_json_load(stage_root / "screening.json") != expected:
                return False
        elif stage.endswith("_fanout"):
            sentinel_specs = _read_specs(root / stage.replace("_fanout", "_sentinel"))
            expected = (
                select_rp(sentinel_specs, specs, config)
                if stage == "ca_rp_fanout"
                else select_learning_rate(sentinel_specs, specs, config)
            )
            if strict_json_load(stage_root / "selection.json") != expected:
                return False
        elif stage.endswith("_main"):
            summary = summarize_main(specs, config)
            gate = _main_gate(summary)
            if strict_json_load(stage_root / "summary.json") != summary:
                return False
            if strict_json_load(stage_root / "scientific_gate.json") != gate:
                return False
            science = bool(summary["scientific_pass"])
            marker = stage_root / "SCIENTIFIC_PASS"
            if marker.is_file() != science:
                return False
            if science:
                marker_payload = strict_json_load(marker)
                if marker_payload.get("scientific_gate_sha256") != sha256_file(
                    stage_root / "scientific_gate.json"
                ):
                    return False
        else:
            return False
        if _receipt_names(stage_root) != _stage_expected_names(stage, science):
            return False
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        return False
    return True


def _require_stage(root: Path, stage: str, count: int) -> Path:
    if not _stage_valid(root, stage, count):
        raise RuntimeError(f"{stage} verified computation is required")
    return root / stage


def _require_lru_scientific_pass(root: Path) -> Path:
    stage_root = _require_stage(root, "lru_main", 3)
    summary = strict_json_load(stage_root / "summary.json")
    if not summary.get("scientific_pass") or not (stage_root / "SCIENTIFIC_PASS").is_file():
        raise RuntimeError("No-RP/CA stages are blocked: LRU has zero NMSE<-20 eligible seeds")
    return stage_root


def _finish_stage(
    root: Path,
    stage: str,
    specs: Sequence[RunSpec],
    primary_name: str,
    primary_payload: Mapping[str, Any],
    *,
    main_summary: bool = False,
) -> Path:
    stage_root = root / stage
    primary = stage_root / primary_name
    _write_or_verify(primary, primary_payload)
    parent = stage_root / "parent_binding.json"
    _write_or_verify(parent, _parent_payload(root, stage))
    extras = [parent, primary]
    if main_summary:
        gate_path = stage_root / "scientific_gate.json"
        gate = _main_gate(primary_payload)
        _write_or_verify(gate_path, gate)
        extras.append(gate_path)
        if bool(primary_payload["scientific_pass"]):
            marker = stage_root / "SCIENTIFIC_PASS"
            marker_payload = {
                "schema_version": 1,
                "model_id": primary_payload["model_id"],
                "scientific_gate_sha256": sha256_file(gate_path),
                "has_analysis_eligible_subset": True,
                "unlocks_no_rp_and_ca_stages": stage == "lru_main",
            }
            _write_or_verify(marker, marker_payload)
            extras.append(marker)
        elif (stage_root / "SCIENTIFIC_PASS").exists():
            raise RuntimeError("stale scientific pass marker exists")
    _finalize(stage_root, stage, specs, extras)
    return stage_root


def _smoke_plan(
    root: Path, config: Mapping[str, Any], bank: Path
) -> tuple[RunSpec, ...]:
    specs = []
    for model_id in MODEL_IDS:
        specs.append(
            RunSpec(
                run_id=f"smoke__{model_id}",
                stage="smoke",
                model_id=model_id,
                model_seed=999,
                learning_rate=1e-3,
                actual_state_noise_std=_common_state_noise(config),
                updates=2,
                batch_size=2,
                evaluation_bank=str(bank),
                campaign_root=str(root),
                output_dir=str(root / "smoke" / "runs" / model_id),
                rp_enabled=model_id == "ca_lru_n52",
                rp_eta_lambda=300.0 if model_id == "ca_lru_n52" else None,
                rp_damage_epsilon=1e-5 if model_id == "ca_lru_n52" else None,
                smoke=True,
            )
        )
    return tuple(specs)


def _recomputed_lr_selection(
    root: Path, config: Mapping[str, Any], prefix: str
) -> dict[str, Any]:
    sentinel_specs = _read_specs(root / f"{prefix}_sentinel")
    fanout_specs = _read_specs(root / f"{prefix}_fanout")
    screening = screen_sentinels(sentinel_specs, config)
    if strict_json_load(root / f"{prefix}_sentinel" / "screening.json") != screening:
        raise RuntimeError(f"{prefix} sentinel screening differs from current children")
    selection = select_learning_rate(sentinel_specs, fanout_specs, config)
    if strict_json_load(root / f"{prefix}_fanout" / "selection.json") != selection:
        raise RuntimeError(f"{prefix} LR selection differs from current children")
    return selection


def _expected_plan_for_stage(
    root: Path, config: Mapping[str, Any], stage: str
) -> tuple[RunSpec, ...]:
    """Rebuild a stage plan solely from the freeze and verified-parent outputs."""

    root = root.expanduser().resolve()
    tuning_bank = root / "banks" / "tuning.npz"
    main_bank = root / "banks" / "main_test.npz"
    if stage == "smoke":
        return _smoke_plan(root, config, tuning_bank)
    if stage == "lru_sentinel":
        return build_lr_sentinel_plan(root, config, tuning_bank, "lru")
    if stage == "lru_fanout":
        sentinel = _read_specs(root / "lru_sentinel")
        screening = screen_sentinels(sentinel, config)
        if strict_json_load(root / "lru_sentinel" / "screening.json") != screening:
            raise RuntimeError("LRU sentinel screening differs from current children")
        return build_lr_fanout_plan(root, config, tuning_bank, "lru", screening)
    if stage == "lru_main":
        selection = _recomputed_lr_selection(root, config, "lru")
        return build_main_plan(root, config, main_bank, "lru", "lru_n52", selection)
    if stage == "no_rp_sentinel":
        return build_lr_sentinel_plan(root, config, tuning_bank, "no_rp")
    if stage == "no_rp_fanout":
        sentinel = _read_specs(root / "no_rp_sentinel")
        screening = screen_sentinels(sentinel, config)
        if strict_json_load(root / "no_rp_sentinel" / "screening.json") != screening:
            raise RuntimeError("No-RP sentinel screening differs from current children")
        return build_lr_fanout_plan(root, config, tuning_bank, "no_rp", screening)
    if stage == "no_rp_main":
        selection = _recomputed_lr_selection(root, config, "no_rp")
        return build_main_plan(
            root, config, main_bank, "no_rp", "no_rp_n52", selection
        )
    if stage == "ca_rp_sentinel":
        no_rp_selection = _recomputed_lr_selection(root, config, "no_rp")
        return build_rp_sentinel_plan(root, config, tuning_bank, no_rp_selection)
    if stage == "ca_rp_fanout":
        no_rp_selection = _recomputed_lr_selection(root, config, "no_rp")
        sentinel = _read_specs(root / "ca_rp_sentinel")
        screening = screen_rp_sentinels(sentinel, config)
        if strict_json_load(root / "ca_rp_sentinel" / "screening.json") != screening:
            raise RuntimeError("CA-RP sentinel screening differs from current children")
        return build_rp_fanout_plan(
            root, config, tuning_bank, no_rp_selection, screening
        )
    if stage == "ca_rp_main":
        no_rp_selection = _recomputed_lr_selection(root, config, "no_rp")
        rp_sentinel = _read_specs(root / "ca_rp_sentinel")
        rp_fanout = _read_specs(root / "ca_rp_fanout")
        rp_selection = select_rp(rp_sentinel, rp_fanout, config)
        if strict_json_load(root / "ca_rp_fanout" / "selection.json") != rp_selection:
            raise RuntimeError("CA-RP selection differs from current children")
        return build_main_plan(
            root,
            config,
            main_bank,
            "ca_rp",
            "ca_lru_n52",
            no_rp_selection,
            rp=rp_selection["winner"],
        )
    raise ValueError(f"unknown registered stage: {stage}")


def _validate_stage_dependencies(root: Path, stage: str) -> None:
    """Recursively require the exact immediate parent before consuming it."""

    config = load_config(root / "inputs" / DEFAULT_CONFIG.name)
    lr_sentinel_count = len(config["learning_rate_tuning"]["learning_rate_grid"])
    lr_fanout_count = lr_sentinel_count * len(
        config["learning_rate_tuning"]["fanout_seeds"]
    )
    if stage in {"smoke", "lru_sentinel"}:
        return
    if stage == "lru_fanout":
        _require_stage(root, "lru_sentinel", lr_sentinel_count)
    elif stage == "lru_main":
        _require_stage(root, "lru_fanout", lr_fanout_count)
    elif stage == "no_rp_sentinel":
        _require_lru_scientific_pass(root)
    elif stage == "no_rp_fanout":
        _require_stage(root, "no_rp_sentinel", lr_sentinel_count)
    elif stage == "no_rp_main":
        _require_stage(root, "no_rp_fanout", lr_fanout_count)
    elif stage == "ca_rp_sentinel":
        _require_stage(root, "no_rp_main", 3)
    elif stage == "ca_rp_fanout":
        _require_stage(root, "ca_rp_sentinel", 9)
    elif stage == "ca_rp_main":
        _require_stage(root, "ca_rp_fanout", 12)
    else:
        raise ValueError(f"unknown registered stage: {stage}")


def _validate_spec(
    config: Mapping[str, Any],
    spec: RunSpec,
    *,
    require_registered_plan: bool,
) -> None:
    """Reject any worker spec not equal to its recomputed registered plan row."""

    if spec.stage not in STAGES or spec.model_id not in MODEL_IDS:
        raise ValueError("worker spec model/stage is not registered")
    if spec.smoke != (spec.stage == "smoke"):
        raise ValueError("smoke=True is allowed only for stage=smoke")
    root = Path(spec.campaign_root).expanduser().resolve()
    if str(root) != spec.campaign_root:
        raise ValueError("campaign_root must be the canonical absolute root")
    expected_output = root / spec.stage / "runs" / (
        spec.model_id if spec.stage == "smoke" else spec.run_id
    )
    if Path(spec.output_dir).expanduser().resolve() != expected_output:
        raise ValueError("worker output path differs from the registered stage layout")
    expected_bank = root / "banks" / (
        "main_test.npz" if spec.stage.endswith("_main") else "tuning.npz"
    )
    if Path(spec.evaluation_bank).expanduser().resolve() != expected_bank:
        raise ValueError("worker evaluation bank differs from the registered stage purpose")
    if spec.stage == "smoke":
        if (
            spec.model_seed != 999
            or spec.learning_rate != 1e-3
            or spec.actual_state_noise_std != _common_state_noise(config)
            or spec.updates != 2
            or spec.batch_size != 2
        ):
            raise ValueError("smoke spec settings differ from the registered smoke")
    else:
        if spec.actual_state_noise_std != _common_state_noise(config):
            raise ValueError("full worker state noise differs from the common contract")
        lr_screen_stage = spec.stage in {
            "lru_sentinel",
            "lru_fanout",
            "no_rp_sentinel",
            "no_rp_fanout",
        }
        expected_updates = (
            int(config["learning_rate_tuning"]["screening_updates"])
            if lr_screen_stage
            else int(config["training"]["updates"])
        )
        if spec.updates != expected_updates:
            raise ValueError("full worker update count differs")
        if spec.batch_size != int(config["training"]["batch_size"]):
            raise ValueError("full worker batch size differs")
    if spec.model_id == "ca_lru_n52":
        if not spec.rp_enabled:
            raise ValueError("CA-LRU worker must have RP enabled")
        if spec.rp_eta_lambda not in config["retention_plasticity"]["eta_lambda_grid"]:
            raise ValueError("CA-LRU eta_lambda is outside the frozen grid")
        if spec.rp_damage_epsilon not in config["retention_plasticity"]["damage_epsilon_grid"]:
            raise ValueError("CA-LRU damage_epsilon is outside the frozen grid")
    elif (
        spec.rp_enabled
        or spec.rp_eta_lambda is not None
        or spec.rp_damage_epsilon is not None
    ):
        raise ValueError("LRU/No-RP specs cannot contain RP settings")
    if not require_registered_plan:
        return
    _validate_stage_dependencies(root, spec.stage)
    expected_specs = _expected_plan_for_stage(root, config, spec.stage)
    matches = [row for row in expected_specs if row.run_id == spec.run_id]
    if len(matches) != 1 or matches[0] != spec:
        raise ValueError("worker spec is not the exact recomputed registered plan row")
    stored_specs = _read_specs(root / spec.stage)
    if stored_specs != expected_specs:
        raise ValueError("stored stage plan differs from the recomputed registered plan")


def run_stage(
    stage: str,
    artifact_root: Path,
    baseline_root: Path,
    config_source: Path,
    slots: Sequence[str],
) -> Path:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    root = artifact_root.expanduser().resolve()
    config, copied_config = _prepare_root(
        root, baseline_root, config_source, require_clean=stage != "smoke"
    )
    tuning_bank = root / "banks" / "tuning.npz"
    main_bank = root / "banks" / "main_test.npz"
    if stage == "smoke":
        specs = _smoke_plan(root, config, tuning_bank)
        if _stage_valid(root, stage, len(specs)):
            return root / stage
        _run_specs(specs, copied_config, slots)
        payload = {
            "schema_version": 1,
            "runs": [{"run_id": spec.run_id, "status": _result(spec).get("status")} for spec in specs],
            "scientific_result": False,
        }
        return _finish_stage(root, stage, specs, "summary.json", payload)

    # Reconstruct LRU stages deterministically on every invocation.
    lru_sentinel = build_lr_sentinel_plan(root, config, tuning_bank, "lru")
    if stage == "lru_sentinel":
        if _stage_valid(root, stage, len(lru_sentinel)):
            return root / stage
        _run_specs(lru_sentinel, copied_config, slots)
        return _finish_stage(
            root, stage, lru_sentinel, "screening.json", screen_sentinels(lru_sentinel, config)
        )
    _require_stage(root, "lru_sentinel", len(lru_sentinel))
    lru_screen = strict_json_load(root / "lru_sentinel" / "screening.json")
    lru_fanout = build_lr_fanout_plan(root, config, tuning_bank, "lru", lru_screen)
    if stage == "lru_fanout":
        if _stage_valid(root, stage, len(lru_fanout)):
            return root / stage
        _run_specs(lru_fanout, copied_config, slots)
        return _finish_stage(
            root,
            stage,
            lru_fanout,
            "selection.json",
            select_learning_rate(lru_sentinel, lru_fanout, config),
        )
    _require_stage(root, "lru_fanout", len(lru_fanout))
    lru_selection = strict_json_load(root / "lru_fanout" / "selection.json")
    lru_main = build_main_plan(root, config, main_bank, "lru", "lru_n52", lru_selection)
    if stage == "lru_main":
        if _stage_valid(root, stage, len(lru_main)):
            return root / stage
        _run_specs(lru_main, copied_config, slots)
        return _finish_stage(
            root, stage, lru_main, "summary.json", summarize_main(lru_main, config), main_summary=True
        )

    _require_lru_scientific_pass(root)
    no_rp_sentinel = build_lr_sentinel_plan(root, config, tuning_bank, "no_rp")
    if stage == "no_rp_sentinel":
        if _stage_valid(root, stage, len(no_rp_sentinel)):
            return root / stage
        _run_specs(no_rp_sentinel, copied_config, slots)
        return _finish_stage(
            root, stage, no_rp_sentinel, "screening.json", screen_sentinels(no_rp_sentinel, config)
        )
    _require_stage(root, "no_rp_sentinel", len(no_rp_sentinel))
    no_rp_screen = strict_json_load(root / "no_rp_sentinel" / "screening.json")
    no_rp_fanout = build_lr_fanout_plan(root, config, tuning_bank, "no_rp", no_rp_screen)
    if stage == "no_rp_fanout":
        if _stage_valid(root, stage, len(no_rp_fanout)):
            return root / stage
        _run_specs(no_rp_fanout, copied_config, slots)
        return _finish_stage(
            root,
            stage,
            no_rp_fanout,
            "selection.json",
            select_learning_rate(no_rp_sentinel, no_rp_fanout, config),
        )
    _require_stage(root, "no_rp_fanout", len(no_rp_fanout))
    no_rp_selection = strict_json_load(root / "no_rp_fanout" / "selection.json")
    no_rp_main = build_main_plan(
        root, config, main_bank, "no_rp", "no_rp_n52", no_rp_selection
    )
    if stage == "no_rp_main":
        if _stage_valid(root, stage, len(no_rp_main)):
            return root / stage
        _run_specs(no_rp_main, copied_config, slots)
        return _finish_stage(
            root,
            stage,
            no_rp_main,
            "summary.json",
            summarize_main(no_rp_main, config),
            main_summary=True,
        )

    # Verified completion, not No-RP scientific eligibility, unlocks CA-RP.
    _require_stage(root, "no_rp_main", 3)
    rp_sentinel = build_rp_sentinel_plan(root, config, tuning_bank, no_rp_selection)
    if stage == "ca_rp_sentinel":
        if _stage_valid(root, stage, 9):
            return root / stage
        _run_specs(rp_sentinel, copied_config, slots)
        return _finish_stage(
            root, stage, rp_sentinel, "screening.json", screen_rp_sentinels(rp_sentinel, config)
        )
    _require_stage(root, "ca_rp_sentinel", 9)
    rp_screen = strict_json_load(root / "ca_rp_sentinel" / "screening.json")
    rp_fanout = build_rp_fanout_plan(
        root, config, tuning_bank, no_rp_selection, rp_screen
    )
    if stage == "ca_rp_fanout":
        if _stage_valid(root, stage, len(rp_fanout)):
            return root / stage
        _run_specs(rp_fanout, copied_config, slots)
        return _finish_stage(
            root, stage, rp_fanout, "selection.json", select_rp(rp_sentinel, rp_fanout, config)
        )
    _require_stage(root, "ca_rp_fanout", len(rp_fanout))
    rp_selection = strict_json_load(root / "ca_rp_fanout" / "selection.json")
    ca_main = build_main_plan(
        root,
        config,
        main_bank,
        "ca_rp",
        "ca_lru_n52",
        no_rp_selection,
        rp=rp_selection["winner"],
    )
    if _stage_valid(root, "ca_rp_main", len(ca_main)):
        return root / "ca_rp_main"
    _run_specs(ca_main, copied_config, slots)
    return _finish_stage(
        root,
        "ca_rp_main",
        ca_main,
        "summary.json",
        summarize_main(ca_main, config),
        main_summary=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_spec is not None:
        spec = RunSpec(**strict_json_load(args.worker_spec))
        try:
            _train_worker(spec, args.config.resolve(strict=True), args.device)
        except FloatingPointError as error:
            identity = strict_json_load(Path(spec.campaign_root).resolve() / ROOT_MARKER)
            _assert_worker_identity(
                identity,
                args.config.resolve(strict=True),
                Path(spec.campaign_root).resolve(),
                require_clean=not spec.smoke,
            )
            _record_numerical_failure(spec, args.config.resolve(strict=True), args.device, error)
        return 0
    if args.stage is None or args.artifact_root is None or args.baseline_root is None:
        raise ValueError("campaign mode requires --stage, --artifact-root, and --baseline-root")
    destination = run_stage(
        args.stage,
        args.artifact_root,
        args.baseline_root,
        args.config.resolve(strict=True),
        _parse_slots(args.gpus),
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
