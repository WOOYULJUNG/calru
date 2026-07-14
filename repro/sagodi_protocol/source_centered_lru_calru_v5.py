"""Two-stage source-centered hyperparameter tuning for LRU and CA-LRU.

Stage 1 selects learning-rate/state-noise pairs with RP disabled.  Every grid
cell first has to pass a full 5,000-update sentinel run; only the best passing
cells are fanned out to four additional seeds.  Stage 2 freezes the selected
CA-LRU pair and applies the same sentinel-before-fanout rule to RP parameters.

This module deliberately owns a separate artifact identity from primary-v4.
It does not mutate or reuse v4 result directories.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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

from .artifacts import (
    atomic_json,
    canonical_hash,
    derived_seed,
    sha256_file,
    strict_json_load,
    verify_completion_receipt,
    write_completion_receipt,
)
from .metrics import masked_mse, task_metrics
from .primary_v4 import (
    _configure_determinism,
    _finite_model,
    _to_device_batch,
    build_v4_model,
)
from .source_resolved_protocol import source_angular_integration
from .tasks import load_fixed_bank, save_fixed_bank


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "source_centered_lru_calru_v5_config.json"
FREEZE_DOCUMENT = MODULE_DIR / "SOURCE_CENTERED_LRU_CALRU_V5_FREEZE_ko.md"
CAMPAIGN_ID = "source_centered_lru_calru_tuning_v5"
MODEL_IDS = ("lru_n52", "no_rp_n52", "ca_lru_n52")
STAGE1_MODEL_IDS = ("lru_n52", "no_rp_n52")
EXPECTED_PARAMETER_COUNTS = {
    "lru_n52": 17058,
    "no_rp_n52": 17058,
    "ca_lru_n52": 17058,
}
EXPECTED_LR_GRID = [0.01, 0.003, 0.001, 0.0003]
EXPECTED_NOISE_GRID = [0.0, 0.01, 0.03162277660168379, 0.1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _native(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def load_v5_tuning_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and strictly validate the registered tuning contract."""

    payload = strict_json_load(Path(path).expanduser().resolve(strict=True))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("v5 tuning config must be a schema-1 object")
    if payload.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("v5 tuning campaign_id differs")
    if payload.get("protocol_revision") != "source_centered_two_stage_lru_calru_tuning_v1":
        raise ValueError("v5 tuning protocol_revision differs")
    upstream = payload.get("upstream_reference", {})
    if upstream.get("commit") != "cbd7404e9baca4b2dc291560cfc6576bb7b1f078":
        raise ValueError("upstream commit differs")
    if not math.isclose(
        float(upstream.get("effective_rnn_state_noise_std", math.nan)),
        math.sqrt(0.1) * 0.1,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("upstream effective RNN noise differs")
    task = payload.get("task", {})
    if task != {
        "name": "angular_integration",
        "duration": 12.8,
        "horizon": 128,
        "dt": 0.1,
        "gp_length_scale": 1.0,
        "gp_std": 1.0,
        "gp_jitter": 1e-6,
        "input_sparsity": "variable_uniform_0_2",
        "random_angle_init": True,
        "target_indexing": "q_t_plus_1_after_velocity_update",
        "initial_state_target_index": 0,
        "initial_state_semantics": "source_q1_post_update_target",
    }:
        raise ValueError("v5 task contract differs")
    models = payload.get("models")
    if not isinstance(models, list) or [item.get("id") for item in models] != list(MODEL_IDS):
        raise ValueError("v5 model order differs")
    for item in models:
        model_id = str(item["id"])
        if int(item.get("width", -1)) != 52:
            raise ValueError(f"{model_id} width differs")
        if int(item.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNTS[model_id]:
            raise ValueError(f"{model_id} parameter count differs")
    training = payload.get("training", {})
    required_training = {
        "optimizer": "Adam",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "batch_size": 64,
        "updates": 5000,
        "online_batches": True,
        "validation_interval": 500,
        "trace_interval": 50,
        "state_noise_semantics": "post_update_additive_per_coordinate_std",
    }
    if training != required_training:
        raise ValueError("v5 training contract differs")
    stage1 = payload.get("stage1_lr_noise", {})
    if stage1.get("learning_rate_grid") != EXPECTED_LR_GRID:
        raise ValueError("v5 LR grid differs")
    if stage1.get("state_noise_std_grid") != EXPECTED_NOISE_GRID:
        raise ValueError("v5 state-noise grid differs")
    required_stage1 = {
        "sentinel_seed": 100,
        "fanout_seeds": [101, 102, 103, 104],
        "max_fanout_cells_per_model": 3,
        "success_mse_threshold": 0.01,
        "screening_metric": "final_heldout_masked_mse",
        "selection_metric": "five_seed_mean_final_heldout_masked_mse",
        "ca_lru_inherits_no_rp_pair": True,
    }
    for key, expected in required_stage1.items():
        if stage1.get(key) != expected:
            raise ValueError(f"v5 stage-1 contract differs: {key}")
    rp = payload.get("stage2_retention_plasticity", {})
    required_rp = {
        "warmup_updates": 1500,
        "interval_updates": 50,
        "probe_batch_size": 96,
        "probe_horizon": 128,
        "blank_ablation_horizon": 500,
        "eta_lambda_grid": [300.0, 1000.0, 3000.0],
        "damage_epsilon_grid": [1e-5, 3e-5, 1e-4],
        "sentinel_seed": 100,
        "fanout_seeds": [101, 102, 103, 104],
        "max_fanout_cells": 3,
        "eligibility_id_mse_threshold": 0.01,
        "selection_metric": "five_seed_mean_heldout_blank_memory_mse",
        "selection_blank_horizon": 4096,
        "theta_clip": [-18.0, 18.0],
    }
    for key, expected in required_rp.items():
        if rp.get(key) != expected:
            raise ValueError(f"v5 stage-2 contract differs: {key}")
    if payload.get("evaluation_bank") != {
        "trials": 1024,
        "task_seed": 0,
        "stream_key": ["source_centered_lru_calru_v5", "fixed_tuning_bank"],
    }:
        raise ValueError("v5 evaluation-bank contract differs")
    return payload


@dataclass(frozen=True)
class TuningRunSpec:
    run_id: str
    stage: str
    model_id: str
    model_seed: int
    learning_rate: float
    state_noise_std: float
    updates: int
    batch_size: int
    evaluation_bank: str
    output_dir: str
    rp_enabled: bool = False
    rp_eta_lambda: float | None = None
    rp_damage_epsilon: float | None = None
    smoke: bool = False

    def payload(self) -> dict[str, Any]:
        return _native(asdict(self))


def _float_key(value: float) -> str:
    return format(float(value), ".8g").replace(".", "p").replace("-", "m")


def _result_for(spec: TuningRunSpec) -> dict[str, Any]:
    result = strict_json_load(Path(spec.output_dir) / "result.json")
    if result.get("run_id") != spec.run_id:
        raise RuntimeError(f"run result identity mismatch: {spec.run_id}")
    return result


def _id_mse(spec: TuningRunSpec) -> float:
    return float(_result_for(spec)["final_metrics"]["masked_mse"])


def build_stage1_sentinel_plan(
    root: Path, config: Mapping[str, Any], bank: Path
) -> tuple[TuningRunSpec, ...]:
    tuning = config["stage1_lr_noise"]
    training = config["training"]
    plan: list[TuningRunSpec] = []
    for model_id in STAGE1_MODEL_IDS:
        for rate in tuning["learning_rate_grid"]:
            for noise in tuning["state_noise_std_grid"]:
                run_id = (
                    f"stage1_sentinel__{model_id}__lr{_float_key(rate)}"
                    f"__noise{_float_key(noise)}__seed{tuning['sentinel_seed']}"
                )
                plan.append(
                    TuningRunSpec(
                        run_id=run_id,
                        stage="stage1_sentinel",
                        model_id=model_id,
                        model_seed=int(tuning["sentinel_seed"]),
                        learning_rate=float(rate),
                        state_noise_std=float(noise),
                        updates=int(training["updates"]),
                        batch_size=int(training["batch_size"]),
                        evaluation_bank=str(bank),
                        output_dir=str(root / "tune" / "stage1_sentinel" / "runs" / run_id),
                    )
                )
    return tuple(plan)


def screen_stage1_sentinels(
    plan: Sequence[TuningRunSpec], config: Mapping[str, Any]
) -> dict[str, Any]:
    tuning = config["stage1_lr_noise"]
    threshold = float(tuning["success_mse_threshold"])
    limit = int(tuning["max_fanout_cells_per_model"])
    models: dict[str, Any] = {}
    failed: list[str] = []
    for model_id in STAGE1_MODEL_IDS:
        cells = []
        for spec in (item for item in plan if item.model_id == model_id):
            mse = _id_mse(spec)
            cells.append(
                {
                    "learning_rate": spec.learning_rate,
                    "state_noise_std": spec.state_noise_std,
                    "sentinel_seed": spec.model_seed,
                    "sentinel_id_mse": mse,
                    "passed": math.isfinite(mse) and mse < threshold,
                }
            )
        passing = [cell for cell in cells if cell["passed"]]
        passing.sort(
            key=lambda cell: (
                cell["sentinel_id_mse"],
                tuning["learning_rate_grid"].index(cell["learning_rate"]),
                tuning["state_noise_std_grid"].index(cell["state_noise_std"]),
            )
        )
        screened = passing[:limit]
        if not screened:
            failed.append(model_id)
        models[model_id] = {
            "cells": cells,
            "passing_cell_count": len(passing),
            "fanned_out_cells": screened,
            "gate_passed": bool(screened),
        }
    return {
        "schema_version": 1,
        "success_mse_threshold": threshold,
        "models": models,
        "all_model_sentinel_gates_passed": not failed,
        "failed_models": failed,
    }


def build_stage1_fanout_plan(
    root: Path,
    config: Mapping[str, Any],
    bank: Path,
    screening: Mapping[str, Any],
) -> tuple[TuningRunSpec, ...]:
    if not screening.get("all_model_sentinel_gates_passed"):
        raise RuntimeError("stage-1 fan-out is blocked by a failed sentinel gate")
    tuning = config["stage1_lr_noise"]
    training = config["training"]
    plan: list[TuningRunSpec] = []
    for model_id in STAGE1_MODEL_IDS:
        for cell in screening["models"][model_id]["fanned_out_cells"]:
            for seed in tuning["fanout_seeds"]:
                rate = float(cell["learning_rate"])
                noise = float(cell["state_noise_std"])
                run_id = (
                    f"stage1_fanout__{model_id}__lr{_float_key(rate)}"
                    f"__noise{_float_key(noise)}__seed{int(seed)}"
                )
                plan.append(
                    TuningRunSpec(
                        run_id=run_id,
                        stage="stage1_fanout",
                        model_id=model_id,
                        model_seed=int(seed),
                        learning_rate=rate,
                        state_noise_std=noise,
                        updates=int(training["updates"]),
                        batch_size=int(training["batch_size"]),
                        evaluation_bank=str(bank),
                        output_dir=str(root / "tune" / "stage1_fanout" / "runs" / run_id),
                    )
                )
    return tuple(plan)


def select_stage1_hyperparameters(
    sentinel_plan: Sequence[TuningRunSpec],
    fanout_plan: Sequence[TuningRunSpec],
    screening: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tuning = config["stage1_lr_noise"]
    threshold = float(tuning["success_mse_threshold"])
    selected: dict[str, Any] = {}
    models: dict[str, Any] = {}
    sentinel_seed = int(tuning["sentinel_seed"])
    expected_seeds = [sentinel_seed, *[int(value) for value in tuning["fanout_seeds"]]]
    for model_id in STAGE1_MODEL_IDS:
        cells: list[dict[str, Any]] = []
        for screened in screening["models"][model_id]["fanned_out_cells"]:
            rate = float(screened["learning_rate"])
            noise = float(screened["state_noise_std"])
            matching = [
                spec
                for spec in (*sentinel_plan, *fanout_plan)
                if spec.model_id == model_id
                and spec.learning_rate == rate
                and spec.state_noise_std == noise
            ]
            by_seed = {spec.model_seed: _id_mse(spec) for spec in matching}
            complete = sorted(by_seed) == expected_seeds
            losses = [by_seed[seed] for seed in expected_seeds] if complete else []
            eligible = complete and all(
                math.isfinite(value) and value < threshold for value in losses
            )
            cells.append(
                {
                    "learning_rate": rate,
                    "state_noise_std": noise,
                    "seed_count": len(by_seed),
                    "per_seed_id_mse": by_seed,
                    "mean_id_mse": float(np.mean(losses)) if losses else None,
                    "eligible": eligible,
                }
            )
        viable = [cell for cell in cells if cell["eligible"]]
        if not viable:
            raise RuntimeError(f"no five-seed LR/noise cell passed for {model_id}")
        winner = min(
            viable,
            key=lambda cell: (
                cell["mean_id_mse"],
                tuning["learning_rate_grid"].index(cell["learning_rate"]),
                tuning["state_noise_std_grid"].index(cell["state_noise_std"]),
            ),
        )
        selected[model_id] = {
            "learning_rate": winner["learning_rate"],
            "state_noise_std": winner["state_noise_std"],
        }
        models[model_id] = {"cells": cells, "selected": winner}
    selected["ca_lru_n52"] = {
        **selected["no_rp_n52"],
        "inherited_from": "no_rp_n52",
        "independently_tuned": False,
    }
    return {
        "schema_version": 1,
        "selection_metric": tuning["selection_metric"],
        "models": models,
        "selected_lr_noise": selected,
    }


def build_stage2_sentinel_plan(
    root: Path,
    config: Mapping[str, Any],
    bank: Path,
    stage1_selection: Mapping[str, Any],
) -> tuple[TuningRunSpec, ...]:
    rp = config["stage2_retention_plasticity"]
    training = config["training"]
    pair = stage1_selection["selected_lr_noise"]["ca_lru_n52"]
    plan: list[TuningRunSpec] = []
    for eta in rp["eta_lambda_grid"]:
        for epsilon in rp["damage_epsilon_grid"]:
            run_id = (
                f"stage2_sentinel__eta{_float_key(eta)}__eps{_float_key(epsilon)}"
                f"__seed{rp['sentinel_seed']}"
            )
            plan.append(
                TuningRunSpec(
                    run_id=run_id,
                    stage="stage2_sentinel",
                    model_id="ca_lru_n52",
                    model_seed=int(rp["sentinel_seed"]),
                    learning_rate=float(pair["learning_rate"]),
                    state_noise_std=float(pair["state_noise_std"]),
                    updates=int(training["updates"]),
                    batch_size=int(training["batch_size"]),
                    evaluation_bank=str(bank),
                    output_dir=str(root / "tune" / "stage2_sentinel" / "runs" / run_id),
                    rp_enabled=True,
                    rp_eta_lambda=float(eta),
                    rp_damage_epsilon=float(epsilon),
                )
            )
    return tuple(plan)


def screen_stage2_sentinels(
    plan: Sequence[TuningRunSpec], config: Mapping[str, Any]
) -> dict[str, Any]:
    rp = config["stage2_retention_plasticity"]
    threshold = float(rp["eligibility_id_mse_threshold"])
    cells: list[dict[str, Any]] = []
    for spec in plan:
        result = _result_for(spec)
        id_mse = float(result["final_metrics"]["masked_mse"])
        blank_mse = float(result["heldout_blank_memory_mse"])
        cells.append(
            {
                "eta_lambda": spec.rp_eta_lambda,
                "damage_epsilon": spec.rp_damage_epsilon,
                "sentinel_seed": spec.model_seed,
                "sentinel_id_mse": id_mse,
                "sentinel_blank_memory_mse": blank_mse,
                "passed": (
                    math.isfinite(id_mse)
                    and id_mse < threshold
                    and math.isfinite(blank_mse)
                ),
            }
        )
    passing = [cell for cell in cells if cell["passed"]]
    passing.sort(
        key=lambda cell: (
            cell["sentinel_blank_memory_mse"],
            cell["sentinel_id_mse"],
            rp["eta_lambda_grid"].index(cell["eta_lambda"]),
            rp["damage_epsilon_grid"].index(cell["damage_epsilon"]),
        )
    )
    screened = passing[: int(rp["max_fanout_cells"])]
    return {
        "schema_version": 1,
        "eligibility_id_mse_threshold": threshold,
        "cells": cells,
        "passing_cell_count": len(passing),
        "fanned_out_cells": screened,
        "sentinel_gate_passed": bool(screened),
    }


def build_stage2_fanout_plan(
    root: Path,
    config: Mapping[str, Any],
    bank: Path,
    stage1_selection: Mapping[str, Any],
    screening: Mapping[str, Any],
) -> tuple[TuningRunSpec, ...]:
    if not screening.get("sentinel_gate_passed"):
        raise RuntimeError("stage-2 fan-out is blocked by a failed sentinel gate")
    rp = config["stage2_retention_plasticity"]
    training = config["training"]
    pair = stage1_selection["selected_lr_noise"]["ca_lru_n52"]
    plan: list[TuningRunSpec] = []
    for cell in screening["fanned_out_cells"]:
        for seed in rp["fanout_seeds"]:
            eta = float(cell["eta_lambda"])
            epsilon = float(cell["damage_epsilon"])
            run_id = (
                f"stage2_fanout__eta{_float_key(eta)}__eps{_float_key(epsilon)}"
                f"__seed{int(seed)}"
            )
            plan.append(
                TuningRunSpec(
                    run_id=run_id,
                    stage="stage2_fanout",
                    model_id="ca_lru_n52",
                    model_seed=int(seed),
                    learning_rate=float(pair["learning_rate"]),
                    state_noise_std=float(pair["state_noise_std"]),
                    updates=int(training["updates"]),
                    batch_size=int(training["batch_size"]),
                    evaluation_bank=str(bank),
                    output_dir=str(root / "tune" / "stage2_fanout" / "runs" / run_id),
                    rp_enabled=True,
                    rp_eta_lambda=eta,
                    rp_damage_epsilon=epsilon,
                )
            )
    return tuple(plan)


def select_stage2_hyperparameters(
    sentinel_plan: Sequence[TuningRunSpec],
    fanout_plan: Sequence[TuningRunSpec],
    screening: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rp = config["stage2_retention_plasticity"]
    threshold = float(rp["eligibility_id_mse_threshold"])
    expected_seeds = [int(rp["sentinel_seed"]), *[int(v) for v in rp["fanout_seeds"]]]
    cells: list[dict[str, Any]] = []
    for screened in screening["fanned_out_cells"]:
        eta = float(screened["eta_lambda"])
        epsilon = float(screened["damage_epsilon"])
        matching = [
            spec
            for spec in (*sentinel_plan, *fanout_plan)
            if spec.rp_eta_lambda == eta and spec.rp_damage_epsilon == epsilon
        ]
        results = {spec.model_seed: _result_for(spec) for spec in matching}
        complete = sorted(results) == expected_seeds
        id_by_seed = {
            seed: float(result["final_metrics"]["masked_mse"])
            for seed, result in results.items()
        }
        blank_by_seed = {
            seed: float(result["heldout_blank_memory_mse"])
            for seed, result in results.items()
        }
        eligible = complete and all(
            math.isfinite(id_by_seed[seed]) and id_by_seed[seed] < threshold
            and math.isfinite(blank_by_seed[seed])
            for seed in expected_seeds
        )
        cells.append(
            {
                "eta_lambda": eta,
                "damage_epsilon": epsilon,
                "seed_count": len(results),
                "per_seed_id_mse": id_by_seed,
                "per_seed_blank_memory_mse": blank_by_seed,
                "mean_id_mse": (
                    float(np.mean([id_by_seed[s] for s in expected_seeds]))
                    if complete else None
                ),
                "mean_blank_memory_mse": (
                    float(np.mean([blank_by_seed[s] for s in expected_seeds]))
                    if complete else None
                ),
                "eligible": eligible,
            }
        )
    viable = [cell for cell in cells if cell["eligible"]]
    if not viable:
        raise RuntimeError("no five-seed RP cell passed the ID-MSE gate")
    winner = min(
        viable,
        key=lambda cell: (
            cell["mean_blank_memory_mse"],
            cell["mean_id_mse"],
            rp["eta_lambda_grid"].index(cell["eta_lambda"]),
            rp["damage_epsilon_grid"].index(cell["damage_epsilon"]),
        ),
    )
    return {
        "schema_version": 1,
        "selection_metric": rp["selection_metric"],
        "eligibility_id_mse_threshold": threshold,
        "selected": winner,
        "cells": cells,
    }


def _training_batch(
    config: Mapping[str, Any],
    update: int,
    batch_size: int,
    model_seed: int,
    device: torch.device,
) -> Any:
    if int(config["task"]["horizon"]) != 128:
        raise ValueError("source-centered training requires the 128-step task")
    return source_angular_integration(
        int(batch_size),
        0,
        # Same seed/update is paired across models, while different training
        # seeds receive independent online data as in independent source runs.
        stream_key=(CAMPAIGN_ID, "online_train", int(model_seed), int(update)),
        device=device,
    )


def _rp_probe_batch(
    config: Mapping[str, Any], update: int, model_seed: int, device: torch.device
) -> Any:
    rp = config["stage2_retention_plasticity"]
    if int(rp["probe_horizon"]) != 128:
        raise ValueError("source-centered RP probes must use the 128-step task")
    return source_angular_integration(
        int(rp["probe_batch_size"]),
        0,
        stream_key=(CAMPAIGN_ID, "rp_probe", int(model_seed), int(update)),
        device=device,
    )


def _source_q1_memory(batch: Any) -> torch.Tensor:
    """Return the clean first post-update target used by upstream as h0 input."""

    if batch.output_targets.ndim != 3 or batch.output_targets.shape[0] != 128:
        raise ValueError("source q1 initializer requires a 128-step output target")
    return batch.output_targets[0]


@torch.no_grad()
def _evaluate_source(model: Any, batch: Any) -> dict[str, Any]:
    prediction = model.forward_sequence(
        batch.inputs,
        initial_memory=_source_q1_memory(batch),
    )
    if not torch.isfinite(prediction).all().item():
        raise FloatingPointError("non-finite validation prediction")
    return _native(
        task_metrics(
            prediction,
            batch.output_targets,
            batch.mask,
            batch.latent_targets,
        )
    )


@torch.no_grad()
def _blank_memory_mse_source(model: Any, batch: Any, horizon: int) -> float:
    _, states = model.forward_sequence(
        batch.inputs,
        initial_memory=_source_q1_memory(batch),
        return_states=True,
    )
    state = states[-1]
    blank = torch.zeros(
        state.shape[0], model.input_dim, dtype=state.dtype, device=state.device
    )
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


def _identity(config_path: Path) -> dict[str, Any]:
    files = (
        Path(__file__),
        MODULE_DIR / "primary_v4.py",
        MODULE_DIR / "models.py",
        MODULE_DIR / "train.py",
        MODULE_DIR / "tasks.py",
        MODULE_DIR / "source_resolved_protocol.py",
        MODULE_DIR / "sagodi_source_resolved_v1.json",
        FREEZE_DOCUMENT,
    )
    payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "config_sha256": sha256_file(config_path),
        "code_sha256": {path.name: sha256_file(path) for path in files},
    }
    payload["scientific_identity"] = canonical_hash(payload)
    return payload


def _train_worker(spec: TuningRunSpec, config_path: Path, device_text: str) -> Path:
    config = load_v5_tuning_config(config_path)
    root = Path(spec.evaluation_bank).expanduser().resolve().parents[1]
    expected_identity = strict_json_load(root / ".source_centered_lru_calru_v5_root.json")
    if _identity(config_path) != expected_identity:
        raise RuntimeError("worker code/config identity differs from campaign root")
    output = Path(spec.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"worker output is not empty: {output}")
    device = torch.device(device_text)
    _configure_determinism(spec.model_seed)
    model = build_v4_model(spec.model_id).to(device)
    _finite_model(model)
    bank = _to_device_batch(load_fixed_bank(spec.evaluation_bank), device)
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run": spec.payload(),
        "model": model.metadata(),
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "scientific_identity": expected_identity["scientific_identity"],
        "config_sha256": sha256_file(config_path),
        "evaluation_bank_sha256": sha256_file(spec.evaluation_bank),
        "task_contract": {
            "horizon": int(config["task"]["horizon"]),
            "input_sparsity": config["task"]["input_sparsity"],
            "initial_state_semantics": config["task"]["initial_state_semantics"],
        },
        "state_noise_contract": {
            "actual_per_step_coordinate_std": float(spec.state_noise_std),
            "injection_point": config["training"]["state_noise_semantics"],
            "upstream_rnn_reference_std": float(
                config["upstream_reference"]["effective_rnn_state_noise_std"]
            ),
        },
        "online_rng_pairing": "same_model_seed_and_update_paired_across_models",
        "started_at_utc": _utc_now(),
        "device": device_text,
    }
    atomic_json(output / "run_manifest.json", manifest)
    training = config["training"]
    rp = config["stage2_retention_plasticity"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(spec.learning_rate),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    generator = torch.Generator(device=device.type).manual_seed(
        derived_seed(spec.model_seed, CAMPAIGN_ID, "state_noise")
    )
    rp_module = None
    if spec.rp_enabled:
        from . import train as rp_module
    trace: list[dict[str, Any]] = []
    rp_trace: list[dict[str, Any]] = []
    started = time.time()
    for update in range(1, int(spec.updates) + 1):
        model.train()
        batch = _training_batch(
            config, update, spec.batch_size, spec.model_seed, device
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_sequence(
            batch.inputs,
            initial_memory=_source_q1_memory(batch),
            state_noise_std=float(spec.state_noise_std),
            noise_generator=generator,
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
            if rp_module is None:  # pragma: no cover
                raise RuntimeError("RP module was not frozen at worker start")
            probe = _rp_probe_batch(config, update, spec.model_seed, device)
            values = rp_module._retention_plasticity_call(
                model,
                probe,
                blank_horizon=int(rp["blank_ablation_horizon"]),
                eta_lambda=float(spec.rp_eta_lambda),
                damage_epsilon=float(spec.rp_damage_epsilon),
                initial_memory=_source_q1_memory(probe),
            )
            rp_trace.append({"update": update, **_native(values)})
        if (
            update == 1
            or update % int(training["trace_interval"]) == 0
            or update == spec.updates
        ):
            row: dict[str, Any] = {
                "update": update,
                "train_mse": float(loss.detach().cpu()),
                "elapsed_seconds": float(time.time() - started),
            }
            if update % int(training["validation_interval"]) == 0 or update == spec.updates:
                model.eval()
                row["validation"] = _evaluate_source(model, bank)
            trace.append(row)
    model.eval()
    final_metrics = _evaluate_source(model, bank)
    blank_mse = None
    if spec.rp_enabled:
        blank_mse = _blank_memory_mse_source(
            model, bank, int(rp["selection_blank_horizon"])
        )
    result = {
        "schema_version": 1,
        "run_id": spec.run_id,
        "stage": spec.stage,
        "model_id": spec.model_id,
        "model_seed": spec.model_seed,
        "learning_rate": spec.learning_rate,
        "state_noise_std": spec.state_noise_std,
        "updates_completed": spec.updates,
        "final_metrics": final_metrics,
        "heldout_blank_memory_mse": blank_mse,
        "rp_call_count": len(rp_trace),
        "completed_at_utc": _utc_now(),
    }
    atomic_json(output / "training_trace.json", trace)
    atomic_json(output / "rp_trace.json", rp_trace)
    atomic_json(output / "result.json", result)
    _atomic_torch_save(
        output / "checkpoint_final.pt",
        {
            "schema_version": 1,
            "checkpoint_type": CAMPAIGN_ID,
            "run": spec.payload(),
            "result": result,
            "state_dict": model.state_dict(),
        },
    )
    atomic_json(
        output / "COMPLETE",
        {"schema_version": 1, "status": "complete", "run_id": spec.run_id},
    )
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=spec.run_id,
        artifacts=[
            output / "run_manifest.json",
            output / "training_trace.json",
            output / "rp_trace.json",
            output / "result.json",
            output / "checkpoint_final.pt",
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


def _verified(spec: TuningRunSpec) -> bool:
    output = Path(spec.output_dir)
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
    if not valid:
        return False
    try:
        manifest = strict_json_load(output / "run_manifest.json")
        result = strict_json_load(output / "result.json")
    except (OSError, ValueError):
        return False
    return manifest.get("run") == spec.payload() and result.get("run_id") == spec.run_id


def _write_plan(stage_root: Path, specs: Sequence[TuningRunSpec]) -> None:
    payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_count": len(specs),
        "runs": [spec.payload() for spec in specs],
    }
    path = stage_root / "plan.json"
    if path.exists() and strict_json_load(path) != payload:
        raise RuntimeError(f"immutable plan differs: {stage_root}")
    if not path.exists():
        atomic_json(path, payload)


def _write_or_verify(path: Path, payload: Mapping[str, Any]) -> None:
    value = _native(payload)
    if path.exists() and strict_json_load(path) != value:
        raise RuntimeError(f"immutable derived artifact differs: {path}")
    if not path.exists():
        atomic_json(path, value)


def _run_specs(
    specs: Sequence[TuningRunSpec], config_path: Path, compute_slots: Sequence[str]
) -> None:
    if not specs:
        return
    stage_root = Path(specs[0].output_dir).parents[1]
    _write_plan(stage_root, specs)
    queue = [spec for spec in specs if not _verified(spec)]
    attempts = stage_root / "attempts"
    for spec in queue:
        output = Path(spec.output_dir)
        if output.exists():
            attempts.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            os.replace(output, attempts / f"{output.name}.{stamp}")
    specs_dir = stage_root / "specs"
    logs_dir = stage_root / "logs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    running: dict[str, tuple[subprocess.Popen[Any], TuningRunSpec, Any]] = {}
    repo = Path(__file__).resolve().parents[2]
    while queue or running:
        for slot in (item for item in compute_slots if item not in running):
            if not queue:
                break
            spec = queue.pop(0)
            spec_path = specs_dir / f"{spec.run_id}.json"
            atomic_json(spec_path, spec.payload())
            handle = (logs_dir / f"{spec.run_id}.log").open("ab")
            command = [
                sys.executable,
                "-m",
                "repro.sagodi_protocol.source_centered_lru_calru_v5",
                "--worker-spec",
                str(spec_path),
                "--config",
                str(config_path),
                "--device",
                "cpu" if slot == "cpu" else "cuda:0",
            ]
            environment = os.environ.copy()
            if slot != "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = slot
            running[slot] = (
                subprocess.Popen(
                    command,
                    cwd=repo,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                ),
                spec,
                handle,
            )
        if not running:
            continue
        time.sleep(0.2)
        for slot, (process, spec, handle) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del running[slot]
            if code != 0 or not _verified(spec):
                for other, _, other_handle in running.values():
                    other.terminate()
                    other_handle.close()
                raise RuntimeError(f"v5 tuning child failed ({code}): {spec.run_id}")
        atomic_json(
            stage_root / "status.json",
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "registered": len(specs),
                "verified_complete": sum(_verified(spec) for spec in specs),
                "pending": len(queue),
                "running": [item[1].run_id for item in running.values()],
                "updated_at_utc": _utc_now(),
            },
        )


def _ensure_root(root: Path, config_source: Path) -> tuple[dict[str, Any], Path]:
    root = root.expanduser().resolve()
    config = load_v5_tuning_config(config_source)
    identity = _identity(config_source)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".source_centered_lru_calru_v5_root.json"
    if marker.exists():
        if strict_json_load(marker) != identity:
            raise RuntimeError("artifact root belongs to another v5 tuning identity")
    else:
        if any(root.iterdir()):
            raise RuntimeError("unmarked v5 tuning artifact root is not empty")
        atomic_json(marker, identity)
    copied = root / "inputs" / DEFAULT_CONFIG.name
    copied.parent.mkdir(parents=True, exist_ok=True)
    if copied.exists() and copied.read_bytes() != config_source.read_bytes():
        raise RuntimeError("copied v5 config differs")
    if not copied.exists():
        copied.write_bytes(config_source.read_bytes())
    freeze = root / "inputs" / FREEZE_DOCUMENT.name
    if freeze.exists() and freeze.read_bytes() != FREEZE_DOCUMENT.read_bytes():
        raise RuntimeError("copied v5 freeze differs")
    if not freeze.exists():
        freeze.write_bytes(FREEZE_DOCUMENT.read_bytes())
    return config, copied


def _ensure_bank(root: Path, config: Mapping[str, Any], smoke: bool = False) -> Path:
    contract = config["evaluation_bank"]
    bank = root / "banks" / ("smoke.npz" if smoke else "tuning.npz")
    trials = 16 if smoke else int(contract["trials"])
    if not bank.exists():
        batch = source_angular_integration(
            trials,
            999 if smoke else int(contract["task_seed"]),
            stream_key=(CAMPAIGN_ID, "smoke_bank") if smoke else tuple(contract["stream_key"]),
            device="cpu",
        )
        save_fixed_bank(bank, batch)
    observed = load_fixed_bank(bank)
    if observed.batch_size != trials or observed.time_steps != 128:
        raise RuntimeError("v5 tuning bank shape differs")
    return bank


def _parse_slots(text: str) -> tuple[str, ...]:
    normalized = text.strip().lower()
    if normalized == "cpu":
        return ("cpu",)
    slots = tuple(value.strip() for value in normalized.split(",") if value.strip())
    if not slots or any(not value.isdigit() for value in slots) or len(set(slots)) != len(slots):
        raise ValueError("--gpus must be unique comma-separated ids or 'cpu'")
    return slots


def run_tuning(
    artifact_root: Path, config_source: Path, compute_slots: Sequence[str]
) -> Path:
    config, copied = _ensure_root(artifact_root, config_source.resolve(strict=True))
    root = artifact_root.expanduser().resolve()
    bank = _ensure_bank(root, config)

    stage1_sentinel = build_stage1_sentinel_plan(root, config, bank)
    _run_specs(stage1_sentinel, copied, compute_slots)
    stage1_screen = screen_stage1_sentinels(stage1_sentinel, config)
    stage1_screen_path = root / "tune" / "stage1_screening.json"
    _write_or_verify(stage1_screen_path, stage1_screen)
    if not stage1_screen["all_model_sentinel_gates_passed"]:
        raise RuntimeError(
            "stage-1 sentinel gate failed for: " + ", ".join(stage1_screen["failed_models"])
        )

    stage1_fanout = build_stage1_fanout_plan(root, config, bank, stage1_screen)
    _run_specs(stage1_fanout, copied, compute_slots)
    stage1_selection = select_stage1_hyperparameters(
        stage1_sentinel, stage1_fanout, stage1_screen, config
    )
    stage1_selection_path = root / "tune" / "stage1_selection.json"
    _write_or_verify(stage1_selection_path, stage1_selection)

    stage2_sentinel = build_stage2_sentinel_plan(
        root, config, bank, stage1_selection
    )
    _run_specs(stage2_sentinel, copied, compute_slots)
    stage2_screen = screen_stage2_sentinels(stage2_sentinel, config)
    stage2_screen_path = root / "tune" / "stage2_screening.json"
    _write_or_verify(stage2_screen_path, stage2_screen)
    if not stage2_screen["sentinel_gate_passed"]:
        raise RuntimeError("stage-2 RP sentinel gate failed")

    stage2_fanout = build_stage2_fanout_plan(
        root, config, bank, stage1_selection, stage2_screen
    )
    _run_specs(stage2_fanout, copied, compute_slots)
    rp_selection = select_stage2_hyperparameters(
        stage2_sentinel, stage2_fanout, stage2_screen, config
    )
    rp_selection_path = root / "tune" / "stage2_rp_selection.json"
    _write_or_verify(rp_selection_path, rp_selection)
    all_specs = (
        *stage1_sentinel,
        *stage1_fanout,
        *stage2_sentinel,
        *stage2_fanout,
    )
    summary = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "selected_lr_noise_for_main": stage1_selection["selected_lr_noise"],
        "selected_rp_for_main": {
            "eta_lambda": rp_selection["selected"]["eta_lambda"],
            "damage_epsilon": rp_selection["selected"]["damage_epsilon"],
        },
        "ca_lru_lr_noise_independently_tuned": False,
        "ca_lru_inherits_lr_noise_from": "no_rp_n52",
        "sentinel_before_fanout": True,
        "manifold_or_ood_metrics_used_for_selection": False,
        "actual_run_count": len(all_specs),
        "maximum_registered_run_count": 77,
    }
    tune_root = root / "tune"
    summary_path = tune_root / "tuning_summary.json"
    _write_or_verify(summary_path, summary)
    atomic_json(
        tune_root / "COMPLETE",
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "verified_run_count": len(all_specs),
            "completed_at_utc": _utc_now(),
        },
    )
    write_completion_receipt(
        tune_root / "completion_receipt.json",
        job_id=f"{CAMPAIGN_ID}__tune",
        artifacts=[
            stage1_screen_path,
            stage1_selection_path,
            stage2_screen_path,
            rp_selection_path,
            summary_path,
            tune_root / "COMPLETE",
        ],
        metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": "tune",
            "run_count": len(all_specs),
        },
    )
    return tune_root


def run_smoke(
    artifact_root: Path, config_source: Path, compute_slots: Sequence[str]
) -> Path:
    config, copied = _ensure_root(artifact_root, config_source.resolve(strict=True))
    root = artifact_root.expanduser().resolve()
    bank = _ensure_bank(root, config, smoke=True)
    pair = float(config["upstream_reference"]["effective_rnn_state_noise_std"])
    specs = tuple(
        TuningRunSpec(
            run_id=f"smoke__{model_id}",
            stage="smoke",
            model_id=model_id,
            model_seed=999,
            learning_rate=1e-3,
            state_noise_std=pair,
            updates=2,
            batch_size=4,
            evaluation_bank=str(bank),
            output_dir=str(root / "smoke" / "runs" / model_id),
            smoke=True,
        )
        for model_id in MODEL_IDS
    )
    _run_specs(specs, copied, compute_slots)
    return root / "smoke"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "tune"))
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_spec is not None:
        if args.stage is not None or args.artifact_root is not None:
            raise ValueError("worker mode cannot be combined with campaign options")
        spec = TuningRunSpec(**strict_json_load(args.worker_spec))
        _train_worker(spec, args.config.resolve(strict=True), str(args.device))
        return 0
    if args.stage is None or args.artifact_root is None:
        raise ValueError("campaign mode requires --stage and --artifact-root")
    slots = _parse_slots(args.gpus)
    if slots != ("cpu",) and not torch.cuda.is_available():
        raise RuntimeError("CUDA ids requested but torch.cuda is unavailable")
    destination = (
        run_smoke(args.artifact_root, args.config, slots)
        if args.stage == "smoke"
        else run_tuning(args.artifact_root, args.config, slots)
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
