"""Four-baseline recurrent-state-noise search after noise-free LR selection.

The verified noise-free campaigns supply one learning rate for each of RNN,
GRU, LSTM, and LRU.  This campaign freezes those values and searches only the
actual post-transition per-coordinate state-noise standard deviation.  Target
noise and output dropout remain zero.  No-RP and CA-LRU are deliberately
excluded and handled by a later LRU-LR-fixed campaign.
"""

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
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from . import source_repaired_baselines_v6 as baseline_v6
from . import source_repaired_lru_calru_v6 as downstream_v6
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
from .source_resolved_protocol import (
    build_source_optimizer,
    clip_source_gradients,
    source_angular_integration,
    source_masked_mse,
)
from .tasks import Batch, load_fixed_bank


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "state_noise_search_v1.json"
FREEZE_DOCUMENT = MODULE_DIR / "STATE_NOISE_SEARCH_V1_FREEZE_ko.md"
BASELINE_CONFIG = MODULE_DIR / "source_repaired_baselines_v6.json"
DOWNSTREAM_CONFIG = MODULE_DIR / "source_repaired_lru_calru_v6.json"
CAMPAIGN_ID = "sagodi_state_noise_search_v1"
PROTOCOL_REVISION = "four_baseline_fixed_noise_free_lr_state_noise_only_pilot3_v3"
TRACK_CLASSIFICATION = (
    "four_baseline_model_specific_training_state_noise_search_before_calru_tuning"
)
ROOT_MARKER = ".sagodi_state_noise_search_v1_root.json"
CONFIG_CONTRACT_SHA256 = "5f59be730393cde736ec620a512a67138c7493411dcd245c5b18a3e032588f72"
SOURCE_BASELINE_MODEL_IDS = tuple(baseline_v6.MODEL_IDS)
LRU_MODEL_ID = "lru_n52"
MODEL_IDS = (*SOURCE_BASELINE_MODEL_IDS, LRU_MODEL_ID)
STAGES = ("smoke", "tuning", "main")
EXPECTED_TRAINABLE = {
    **baseline_v6.PARAMETER_COUNTS,
    LRU_MODEL_ID: downstream_v6.GRADIENT_COUNTS[LRU_MODEL_ID],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _native(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = strict_json_load(Path(path).expanduser().resolve(strict=True))
    if canonical_hash(payload) != CONFIG_CONTRACT_SHA256:
        raise ValueError("state-noise-search config differs from the canonical contract")
    if (
        payload.get("campaign_id") != CAMPAIGN_ID
        or payload.get("protocol_revision") != PROTOCOL_REVISION
        or payload.get("track_classification") != TRACK_CLASSIFICATION
    ):
        raise ValueError("state-noise-search identity differs")
    models = payload.get("models")
    if not isinstance(models, list) or [row.get("id") for row in models] != list(MODEL_IDS):
        raise ValueError("state-noise-search model order differs")
    for row in models:
        model_id = str(row["id"])
        if int(row.get("parameters_trainable", -1)) != EXPECTED_TRAINABLE[model_id]:
            raise ValueError(f"trainable parameter count differs for {model_id}")
        expected_family = (
            "source_baseline" if model_id in SOURCE_BASELINE_MODEL_IDS else "lru_baseline"
        )
        if row.get("family") != expected_family:
            raise ValueError(f"model family differs for {model_id}")
    training = payload["training"]
    if (
        training["batch_size"] != 64
        or training["updates"] != 5000
        or training["target_noise_std"] != 0.0
        or training["output_dropout"] != 0.0
        or training["evaluation_state_noise_std"] != 0.0
        or training["state_noise_location"] != "post_transition_full_recurrent_state"
        or training["training_target_semantics"]
        != "clean_cos_sin_target_for_initial_q1_and_loss"
    ):
        raise ValueError("state-noise-search training contract differs")
    search = payload["state_noise_search"]
    if search["actual_post_transition_state_noise_std_grid"] != [
        0.0,
        0.003,
        0.01,
        0.0316228,
        0.1,
    ]:
        raise ValueError("state-noise grid differs")
    if search["tuning_seeds"] != [100, 101] or search["screening_updates"] != 2000:
        raise ValueError("state-noise tuning seeds differ")
    if search["all_noise_values_all_seeds"] is not True:
        raise ValueError("state-noise search cannot prune using one seed")
    if (
        search["record_overall_winner_including_zero"] is not True
        or search["record_best_strictly_positive_noise"] is not True
        or payload["main"][
            "train_best_strictly_positive_noise_for_noise_vs_no_noise_analysis"
        ]
        is not True
    ):
        raise ValueError("noise/no-noise comparison contract differs")
    if payload["main"]["seeds"] != list(range(3)):
        raise ValueError("state-noise main seeds differ")
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
    smoke: bool = False

    def payload(self) -> dict[str, Any]:
        return _native(asdict(self))


def _float_key(value: float) -> str:
    return format(float(value), ".8g").replace(".", "p").replace("-", "m")


def _parse_slots(text: str) -> tuple[str, ...]:
    normalized = str(text).strip().lower()
    if normalized == "cpu":
        return ("cpu",)
    slots = tuple(item.strip() for item in normalized.split(",") if item.strip())
    if not slots or any(not item.isdigit() for item in slots) or len(set(slots)) != len(slots):
        raise ValueError("--gpus must be unique comma-separated ids or cpu")
    return slots


def _runtime_files() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        MODULE_DIR / "artifacts.py",
        MODULE_DIR / "source_repaired_baselines_v6.py",
        MODULE_DIR / "source_repaired_lru_calru_v6.py",
        MODULE_DIR / "source_resolved_models.py",
        MODULE_DIR / "source_resolved_protocol.py",
        MODULE_DIR / "primary_v4.py",
        MODULE_DIR / "models.py",
        MODULE_DIR / "tasks.py",
        MODULE_DIR / "metrics.py",
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
        raise RuntimeError("full state-noise-search stages require a clean committed worktree")
    return commit


def _selection_binding(baseline_root: Path, downstream_root: Path) -> dict[str, Any]:
    baseline_root = baseline_root.expanduser().resolve()
    downstream_root = downstream_root.expanduser().resolve()
    baseline_v6.require_verified_main(baseline_root)
    downstream_v6._require_stage(downstream_root, "lru_main", 3)

    baseline_selection_path = baseline_root / "fanout" / "hyperparameter_selection.json"
    lru_selection_path = downstream_root / "lru_fanout" / "selection.json"
    baseline_selection = strict_json_load(baseline_selection_path)
    lru_selection = strict_json_load(lru_selection_path)

    selected: dict[str, Any] = {}
    for model_id in SOURCE_BASELINE_MODEL_IDS:
        selected[model_id] = {
            "learning_rate": float(baseline_selection["models"][model_id]["winner"]["learning_rate"]),
            "learning_rate_source": "baseline_fanout_selection",
        }
    selected[LRU_MODEL_ID] = {
        "learning_rate": float(lru_selection["winner"]["learning_rate"]),
        "learning_rate_source": "lru_fanout_selection",
    }
    banks: dict[str, Any] = {}
    for purpose in ("tuning", "main_test"):
        archive = baseline_root / "banks" / f"{purpose}.npz"
        sidecar = archive.with_suffix(archive.suffix + ".sha256")
        banks[purpose] = {
            "archive": str(archive),
            "archive_sha256": sha256_file(archive),
            "sidecar_sha256": sha256_file(sidecar),
        }
    return {
        "schema_version": 1,
        "baseline_root": str(baseline_root),
        "downstream_root": str(downstream_root),
        "baseline_main_receipt_sha256": sha256_file(
            baseline_root / "main" / "completion_receipt.json"
        ),
        "downstream_lru_main_receipt_sha256": sha256_file(
            downstream_root / "lru_main" / "completion_receipt.json"
        ),
        "selection_files": {
            "baseline": sha256_file(baseline_selection_path),
            "lru": sha256_file(lru_selection_path),
        },
        "selected_hyperparameters": selected,
        "banks": banks,
    }


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
    downstream_root: Path,
    config_source: Path,
    *,
    require_clean: bool,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = root.expanduser().resolve()
    config_source = config_source.expanduser().resolve(strict=True)
    config = load_config(config_source)
    parent = _selection_binding(baseline_root, downstream_root)
    identity = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "track_classification": TRACK_CLASSIFICATION,
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
            raise RuntimeError("state-noise-search root scientific identity differs")
    elif any(root.iterdir()):
        raise RuntimeError("unmarked state-noise-search root must be empty")
    else:
        _copy_exact(config_source, root / "inputs" / DEFAULT_CONFIG.name)
        _copy_exact(FREEZE_DOCUMENT, root / "inputs" / FREEZE_DOCUMENT.name)
        _copy_exact(BASELINE_CONFIG, root / "inputs" / BASELINE_CONFIG.name)
        _copy_exact(DOWNSTREAM_CONFIG, root / "inputs" / DOWNSTREAM_CONFIG.name)
        for purpose, row in parent["banks"].items():
            source = Path(row["archive"])
            target = root / "banks" / f"{purpose}.npz"
            _copy_exact(source, target)
            _copy_exact(
                source.with_suffix(source.suffix + ".sha256"),
                target.with_suffix(target.suffix + ".sha256"),
            )
        atomic_json(root / "parent_binding.json", parent)
        atomic_json(marker, identity)
    if strict_json_load(root / "parent_binding.json") != parent:
        raise RuntimeError("state-noise-search parent binding changed")
    for purpose, row in parent["banks"].items():
        copied = root / "banks" / f"{purpose}.npz"
        if sha256_file(copied) != row["archive_sha256"]:
            raise RuntimeError("copied parent bank differs")
    return config, parent, root / "inputs" / DEFAULT_CONFIG.name


def _assert_worker_identity(root: Path, config_path: Path, *, require_clean: bool) -> dict[str, Any]:
    identity = strict_json_load(root / ROOT_MARKER)
    core = dict(identity)
    observed = core.pop("scientific_identity", None)
    if canonical_hash(core) != observed:
        raise RuntimeError("state-noise-search identity digest differs")
    if (
        identity.get("campaign_id") != CAMPAIGN_ID
        or identity.get("protocol_revision") != PROTOCOL_REVISION
        or identity.get("track_classification") != TRACK_CLASSIFICATION
    ):
        raise RuntimeError("state-noise-search worker identity differs")
    if identity.get("code_commit") != _git_state(require_clean):
        raise RuntimeError("state-noise-search worker commit differs")
    if identity.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("state-noise-search worker config differs")
    if identity.get("freeze_sha256") != sha256_file(FREEZE_DOCUMENT):
        raise RuntimeError("state-noise-search freeze differs")
    if identity.get("runtime_code_sha256") != {
        path.name: sha256_file(path) for path in _runtime_files()
    }:
        raise RuntimeError("state-noise-search runtime code differs")
    if strict_json_load(root / "parent_binding.json") != identity.get("parent_binding"):
        raise RuntimeError("state-noise-search parent binding differs")
    return identity


def _selected(parent: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    try:
        return parent["selected_hyperparameters"][model_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"parent selection missing for {model_id}") from error


def _spec(
    root: Path,
    parent: Mapping[str, Any],
    model_id: str,
    stage: str,
    seed: int,
    noise: float,
    updates: int,
    batch_size: int,
    bank: Path,
    *,
    smoke: bool = False,
) -> RunSpec:
    choice = _selected(parent, model_id)
    run_id = (
        f"{stage}__{model_id}__lr{_float_key(choice['learning_rate'])}"
        f"__noise{_float_key(noise)}__seed{seed}"
    )
    return RunSpec(
        run_id=run_id,
        stage=stage,
        model_id=model_id,
        model_seed=int(seed),
        learning_rate=float(choice["learning_rate"]),
        actual_state_noise_std=float(noise),
        updates=int(updates),
        batch_size=int(batch_size),
        evaluation_bank=str(bank),
        campaign_root=str(root),
        output_dir=str(root / stage / "runs" / run_id),
        smoke=smoke,
    )


def build_smoke_plan(
    root: Path, config: Mapping[str, Any], parent: Mapping[str, Any], bank: Path
) -> tuple[RunSpec, ...]:
    return tuple(
        _spec(root, parent, model_id, "smoke", 999, 0.01, 2, 2, bank, smoke=True)
        for model_id in MODEL_IDS
    )


def build_tuning_plan(
    root: Path, config: Mapping[str, Any], parent: Mapping[str, Any], bank: Path
) -> tuple[RunSpec, ...]:
    search, training = config["state_noise_search"], config["training"]
    return tuple(
        _spec(
            root,
            parent,
            model_id,
            "tuning",
            int(seed),
            float(noise),
            int(search["screening_updates"]),
            int(training["batch_size"]),
            bank,
        )
        for model_id in MODEL_IDS
        for noise in search["actual_post_transition_state_noise_std_grid"]
        for seed in search["tuning_seeds"]
    )


def _result(spec: RunSpec) -> dict[str, Any]:
    try:
        result = strict_json_load(Path(spec.output_dir) / "result.json")
    except (OSError, ValueError, TypeError):
        return {"status": "missing"}
    if result.get("run_id") != spec.run_id:
        raise RuntimeError(f"result identity differs: {spec.run_id}")
    return result


def _metric(spec: RunSpec, key: str) -> float | None:
    result = _result(spec)
    metrics = result.get("final_metrics")
    if result.get("status") != "completed" or not isinstance(metrics, Mapping):
        return None
    value = metrics.get(key)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def select_state_noise(
    specs: Sequence[RunSpec], config: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, Any]:
    search = config["state_noise_search"]
    expected_seeds = set(map(int, search["tuning_seeds"]))
    models: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        rows: list[dict[str, Any]] = []
        for noise in search["actual_post_transition_state_noise_std_grid"]:
            cell = [
                spec
                for spec in specs
                if spec.model_id == model_id
                and spec.actual_state_noise_std == float(noise)
            ]
            if len(cell) != len(expected_seeds) or {spec.model_seed for spec in cell} != expected_seeds:
                raise RuntimeError(f"state-noise denominator differs for {model_id}/{noise}")
            if {spec.learning_rate for spec in cell} != {
                float(_selected(parent, model_id)["learning_rate"])
            }:
                raise RuntimeError(f"parent-selected LR differs for {model_id}")
            mses = [_metric(spec, "mse") for spec in cell]
            nmses = [_metric(spec, "nmse_db") for spec in cell]
            complete = all(value is not None for value in (*mses, *nmses))
            rows.append(
                {
                    "learning_rate": float(cell[0].learning_rate),
                    "actual_state_noise_std": float(noise),
                    "registered_seed_count": len(expected_seeds),
                    "completed_seed_count": sum(value is not None for value in mses),
                    "nmse_eligible_count": sum(
                        value is not None
                        and value < float(search["analysis_nmse_db_threshold"])
                        for value in nmses
                    ),
                    "mse_success_count": sum(
                        value is not None
                        and value < float(search["descriptive_mse_threshold"])
                        for value in mses
                    ),
                    "median_mse": float(np.median(mses)) if complete else None,
                    "mean_mse": float(np.mean(mses)) if complete else None,
                    "grid_order": search[
                        "actual_post_transition_state_noise_std_grid"
                    ].index(noise),
                    "per_seed": [
                        {
                            "seed": spec.model_seed,
                            "status": _result(spec).get("status"),
                            "mse": _metric(spec, "mse"),
                            "nmse_db": _metric(spec, "nmse_db"),
                        }
                        for spec in sorted(cell, key=lambda row: row.model_seed)
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
        complete_rows = [
            row for row in rows if row["completed_seed_count"] == len(expected_seeds)
        ]
        if not complete_rows:
            raise RuntimeError(f"no complete state-noise cell for {model_id}")
        positive_rows = [
            row for row in complete_rows if row["actual_state_noise_std"] > 0.0
        ]
        if not positive_rows:
            raise RuntimeError(f"no complete strictly-positive state-noise cell for {model_id}")
        models[model_id] = {
            "overall_winner": complete_rows[0],
            "positive_noise_winner": positive_rows[0],
            "winner": positive_rows[0],
            "overall_winner_is_zero": complete_rows[0]["actual_state_noise_std"] == 0.0,
            "ranked_noise_values": rows,
        }
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "selection_rule": search["selection_rule"],
        "learning_rates_frozen_from_parent": True,
        "target_noise_std": 0.0,
        "output_dropout": 0.0,
        "models": models,
    }


def build_main_plan(
    root: Path,
    config: Mapping[str, Any],
    parent: Mapping[str, Any],
    bank: Path,
    selection: Mapping[str, Any],
) -> tuple[RunSpec, ...]:
    training = config["training"]
    rows: list[RunSpec] = []
    for model_id in MODEL_IDS:
        winner = selection["models"][model_id]["positive_noise_winner"]
        if float(winner["actual_state_noise_std"]) <= 0.0:
            raise RuntimeError(f"main requires strictly-positive noise for {model_id}")
        if float(winner["learning_rate"]) != float(_selected(parent, model_id)["learning_rate"]):
            raise RuntimeError(f"main LR differs from parent for {model_id}")
        for seed in config["main"]["seeds"]:
            rows.append(
                _spec(
                    root,
                    parent,
                    model_id,
                    "main",
                    int(seed),
                    float(winner["actual_state_noise_std"]),
                    int(training["updates"]),
                    int(training["batch_size"]),
                    bank,
                )
            )
    return tuple(rows)


def summarize_main(specs: Sequence[RunSpec], config: Mapping[str, Any]) -> dict[str, Any]:
    search = config["state_noise_search"]
    models: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        rows = [spec for spec in specs if spec.model_id == model_id]
        expected_seeds = set(map(int, config["main"]["seeds"]))
        if len(rows) != len(expected_seeds) or {spec.model_seed for spec in rows} != expected_seeds:
            raise RuntimeError(f"main denominator differs for {model_id}")
        per_seed = []
        for spec in sorted(rows, key=lambda row: row.model_seed):
            mse, nmse = _metric(spec, "mse"), _metric(spec, "nmse_db")
            per_seed.append(
                {
                    "seed": spec.model_seed,
                    "status": _result(spec).get("status"),
                    "mse": mse,
                    "nmse_db": nmse,
                    "mse_success": mse is not None
                    and mse < float(search["descriptive_mse_threshold"]),
                    "analysis_eligible": nmse is not None
                    and nmse < float(search["analysis_nmse_db_threshold"]),
                }
            )
        models[model_id] = {
            "learning_rate": rows[0].learning_rate,
            "selected_actual_state_noise_std": rows[0].actual_state_noise_std,
            "completed_seed_count": sum(row["status"] == "completed" for row in per_seed),
            "mse_success_count": sum(row["mse_success"] for row in per_seed),
            "analysis_eligible_count": sum(row["analysis_eligible"] for row in per_seed),
            "per_seed": per_seed,
        }
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "track_classification": TRACK_CLASSIFICATION,
        "independent_model_specific_noise_optima": True,
        "calru_models_excluded": True,
        "models": models,
    }


def _to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        inputs=batch.inputs.to(device),
        output_targets=batch.output_targets.to(device),
        latent_targets=batch.latent_targets.to(device),
        mask=batch.mask.to(device),
        metadata=batch.metadata,
    )


def _training_batch(spec: RunSpec, update: int, device: torch.device) -> Batch:
    return source_angular_integration(
        spec.batch_size,
        0,
        stream_key=(
            baseline_v6.CAMPAIGN_ID,
            "online_train",
            spec.model_seed,
            int(update),
        ),
        device=device,
    )


def _source_q1(batch: Batch) -> torch.Tensor:
    return batch.output_targets[0]


def _state_noise_seed(spec: RunSpec) -> int | None:
    if spec.actual_state_noise_std == 0.0:
        return None
    return derived_seed(spec.model_seed, CAMPAIGN_ID, "state_noise")


def _rng_identities(spec: RunSpec) -> dict[str, Any]:
    seed = _state_noise_seed(spec)
    return {
        "online_task": {
            "base_seed": 0,
            "stream_key_template": [
                baseline_v6.CAMPAIGN_ID,
                "online_train",
                spec.model_seed,
                "<update_1_to_5000>",
            ],
        },
        "state_noise": {
            "enabled": seed is not None,
            "generator_seed": seed,
            "actual_post_transition_per_coordinate_std": spec.actual_state_noise_std,
            "same_standard_normal_stream_across_std_within_seed": True,
        },
        "target_noise": {"enabled": False, "generator_seed": None, "std": 0.0},
        "dropout": {"enabled": False, "generator_seed": None, "probability": 0.0},
        "retention_plasticity_probe": {"enabled": False, "base_seed": None},
    }


def _build_model(
    spec: RunSpec,
    baseline_config: Mapping[str, Any],
    device: torch.device,
) -> tuple[Any, Any | None]:
    if spec.model_id in SOURCE_BASELINE_MODEL_IDS:
        model = baseline_v6.build_model(
            baseline_config,
            spec.model_id,
            spec.learning_rate,
            spec.actual_state_noise_std,
        ).to(device)
        if model.recipe.target_noise_std != 0.0 or model.recipe.output_dropout != 0.0:
            raise RuntimeError("source baseline target-noise/dropout contract differs")
        return model, model.recipe
    model = build_v4_model(spec.model_id).to(device)
    return model, None


@torch.no_grad()
def _evaluate(model: Any, spec: RunSpec, batch: Batch) -> dict[str, Any]:
    model.eval()
    if spec.model_id in SOURCE_BASELINE_MODEL_IDS:
        prediction = model.forward_sequence(
            batch.inputs,
            source_targets=batch.output_targets,
            state_noise_generator=None,
            state_noise_std_override=0.0,
        )
    else:
        prediction = model.forward_sequence(batch.inputs, initial_memory=_source_q1(batch))
    if not torch.isfinite(prediction).all().item():
        raise FloatingPointError("non-finite held-out prediction")
    metrics = task_metrics(prediction, batch.output_targets, batch.mask, batch.latent_targets)
    metrics["mse"] = metrics["masked_mse"]
    metrics["nmse_db"] = metrics["masked_nmse_db"]
    return _native(metrics)


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


def _train_worker(spec: RunSpec, config_path: Path, device_text: str) -> Path:
    config = load_config(config_path)
    root = Path(spec.campaign_root).resolve()
    identity = _assert_worker_identity(root, config_path, require_clean=not spec.smoke)
    if spec.stage not in STAGES or spec.model_id not in MODEL_IDS:
        raise ValueError("unregistered state-noise-search worker spec")
    if spec.actual_state_noise_std not in config["state_noise_search"][
        "actual_post_transition_state_noise_std_grid"
    ]:
        raise ValueError("worker state-noise std is outside frozen grid")
    choice = _selected(identity["parent_binding"], spec.model_id)
    if spec.learning_rate != float(choice["learning_rate"]):
        raise ValueError("worker LR differs from verified parent selection")

    output = Path(spec.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("state-noise-search worker output is not empty")
    device = torch.device(device_text)
    baseline_v6._configure_determinism(spec.model_seed)
    model, recipe = _build_model(
        spec,
        baseline_v6.load_config(root / "inputs" / BASELINE_CONFIG.name),
        device,
    )
    initial_hash = canonical_tensor_mapping_sha256(model.state_dict())
    rng = _rng_identities(spec)
    state_generator = None
    if rng["state_noise"]["enabled"]:
        state_generator = torch.Generator(device=device.type).manual_seed(
            int(rng["state_noise"]["generator_seed"])
        )
    baseline_v6._finite_model(model)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != EXPECTED_TRAINABLE[spec.model_id]:
        raise RuntimeError(f"trainable parameter count differs: {trainable}")
    downstream_config = downstream_v6.load_config(root / "inputs" / DOWNSTREAM_CONFIG.name)
    if spec.model_id in SOURCE_BASELINE_MODEL_IDS:
        optimizer = build_source_optimizer(model, recipe)
    else:
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=spec.learning_rate,
            betas=tuple(float(value) for value in downstream_config["training"]["betas"]),
            eps=float(downstream_config["training"]["epsilon"]),
            weight_decay=float(downstream_config["training"]["weight_decay"]),
        )
    bank = _to_device(load_fixed_bank(spec.evaluation_bank), device)
    parent_choice = _selected(identity["parent_binding"], spec.model_id)
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "track_classification": TRACK_CLASSIFICATION,
        "run": spec.payload(),
        "scientific_identity": identity["scientific_identity"],
        "code_commit": identity["code_commit"],
        "runtime_code_sha256": identity["runtime_code_sha256"],
        "config_sha256": identity["config_sha256"],
        "freeze_sha256": identity["freeze_sha256"],
        "parent_binding_sha256": canonical_hash(identity["parent_binding"]),
        "parent_selected_hyperparameters": parent_choice,
        "model_metadata": _native(model.metadata()),
        "parameters_trainable": trainable,
        "initial_state_dict_sha256": initial_hash,
        "rng_stream_identities": rng,
        "actual_post_transition_state_noise_std": spec.actual_state_noise_std,
        "state_noise_location": "post_transition_full_recurrent_state",
        "state_noise_semantics": "iid_gaussian_per_coordinate_standard_deviation",
        "target_noise_std": 0.0,
        "output_dropout": 0.0,
        "evaluation_state_noise_std": 0.0,
        "training_target_semantics": "clean_cos_sin_target_for_initial_q1_and_loss",
        "calru_model": False,
        "evaluation_bank_sha256": sha256_file(spec.evaluation_bank),
        "started_at_utc": _utc_now(),
        "device": device_text,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    atomic_json(output / "run_manifest.json", manifest)

    trace: list[dict[str, Any]] = []
    started = time.time()
    trace_interval = int(config["training"]["trace_interval"])
    validation_interval = int(config["training"]["validation_interval"])
    for update in range(1, spec.updates + 1):
        model.train()
        batch = _training_batch(spec, update, device)
        optimizer.zero_grad(set_to_none=True)
        if spec.model_id in SOURCE_BASELINE_MODEL_IDS:
            prediction = model.forward_sequence(
                batch.inputs,
                source_targets=batch.output_targets,
                state_noise_generator=state_generator,
                state_noise_std_override=spec.actual_state_noise_std,
            )
            loss = source_masked_mse(prediction, batch.output_targets, batch.mask)
        else:
            prediction = model.forward_sequence(
                batch.inputs,
                initial_memory=_source_q1(batch),
                state_noise_std=spec.actual_state_noise_std,
                noise_generator=state_generator,
            )
            loss = masked_mse(prediction, batch.output_targets, batch.mask)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()
        baseline_v6._finite_model(model, gradients=True)
        if spec.model_id in SOURCE_BASELINE_MODEL_IDS:
            clip_source_gradients(model, recipe)
        optimizer.step()
        baseline_v6._finite_model(model)
        should_trace = update == 1 or update % trace_interval == 0 or update == spec.updates
        should_validate = update % validation_interval == 0 or update == spec.updates
        if should_trace or should_validate:
            row: dict[str, Any] = {
                "update": update,
                "train_mse": float(loss.detach().cpu()),
                "elapsed_seconds": float(time.time() - started),
            }
            if should_validate:
                row["validation"] = _evaluate(model, spec, bank)
            trace.append(row)
            atomic_json(output / "training_trace.json", trace)
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
    final_metrics = _evaluate(model, spec, bank)
    result = {
        "schema_version": 1,
        "status": "completed",
        "run_id": spec.run_id,
        "stage": spec.stage,
        "model_id": spec.model_id,
        "model_seed": spec.model_seed,
        "learning_rate": spec.learning_rate,
        "actual_state_noise_std": spec.actual_state_noise_std,
        "updates_completed": spec.updates,
        "final_metrics": final_metrics,
        "counts_in_denominator": True,
    }
    atomic_json(output / "training_trace.json", trace)
    atomic_json(output / "result.json", result)
    atomic_json(
        output / "progress.json",
        {"status": "completed", "run_id": spec.run_id, "update": spec.updates},
    )
    checkpoint = output / "checkpoint_final.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "schema_version": 1,
            "checkpoint_type": CAMPAIGN_ID,
            "run": spec.payload(),
            "result": result,
            "initial_state_dict_sha256": initial_hash,
            "rng_stream_identities": rng,
            "state_dict": model.state_dict(),
        },
    )
    atomic_json(output / "COMPLETE", {"schema_version": 1, "run_id": spec.run_id})
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=spec.run_id,
        artifacts=[
            output / "run_manifest.json",
            output / "progress.json",
            output / "training_trace.json",
            output / "result.json",
            checkpoint,
            output / "COMPLETE",
        ],
        metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
            "outcome": "completed",
        },
    )
    return output


def _record_numerical_failure(
    spec: RunSpec, config_path: Path, device_text: str, error: BaseException
) -> Path:
    output = Path(spec.output_dir).resolve()
    root = Path(spec.campaign_root).resolve()
    identity = _assert_worker_identity(root, config_path, require_clean=not spec.smoke)
    baseline_v6._configure_determinism(spec.model_seed)
    model, _ = _build_model(
        spec,
        baseline_v6.load_config(root / "inputs" / BASELINE_CONFIG.name),
        torch.device("cpu"),
    )
    initial_hash = canonical_tensor_mapping_sha256(model.state_dict())
    rng = _rng_identities(spec)
    manifest_path = output / "run_manifest.json"
    if not manifest_path.exists():
        atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "protocol_revision": PROTOCOL_REVISION,
                "track_classification": TRACK_CLASSIFICATION,
                "run": spec.payload(),
                "scientific_identity": identity["scientific_identity"],
                "code_commit": identity["code_commit"],
                "runtime_code_sha256": identity["runtime_code_sha256"],
                "config_sha256": identity["config_sha256"],
                "freeze_sha256": identity["freeze_sha256"],
                "parent_binding_sha256": canonical_hash(identity["parent_binding"]),
                "parent_selected_hyperparameters": _selected(
                    identity["parent_binding"], spec.model_id
                ),
                "model_metadata": _native(model.metadata()),
                "parameters_trainable": EXPECTED_TRAINABLE[spec.model_id],
                "initial_state_dict_sha256": initial_hash,
                "rng_stream_identities": rng,
                "actual_post_transition_state_noise_std": spec.actual_state_noise_std,
                "state_noise_location": "post_transition_full_recurrent_state",
                "target_noise_std": 0.0,
                "output_dropout": 0.0,
                "evaluation_state_noise_std": 0.0,
                "evaluation_bank_sha256": sha256_file(spec.evaluation_bank),
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
        "updates_completed": None,
        "final_metrics": None,
        "counts_in_denominator": True,
    }
    atomic_json(output / "failure.json", failure)
    atomic_json(output / "result.json", result)
    checkpoint = output / "checkpoint_failure.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "schema_version": 1,
            "checkpoint_type": f"{CAMPAIGN_ID}_failure",
            "run": spec.payload(),
            "failure": failure,
            "initial_state_dict_sha256": initial_hash,
            "rng_stream_identities": rng,
        },
    )
    atomic_json(output / "FAILED", {"schema_version": 1, "run_id": spec.run_id})
    artifacts = [
        manifest_path,
        output / "failure.json",
        output / "result.json",
        checkpoint,
        output / "FAILED",
    ]
    for optional in (output / "training_trace.json",):
        if optional.exists():
            artifacts.append(optional)
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=spec.run_id,
        artifacts=artifacts,
        metadata={
            "campaign_id": CAMPAIGN_ID,
            "stage": spec.stage,
            "model_id": spec.model_id,
            "model_seed": spec.model_seed,
            "outcome": "failed",
        },
    )
    return output


def _receipt_names(output: Path) -> set[str] | None:
    try:
        receipt = strict_json_load(output / "completion_receipt.json")
        return set(map(str, receipt["artifacts"]))
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _verified(spec: RunSpec) -> bool:
    output = Path(spec.output_dir)
    try:
        result = strict_json_load(output / "result.json")
    except (OSError, ValueError, TypeError):
        return False
    status = result.get("status")
    if status == "completed":
        outcome = "completed"
        checkpoint_path = output / "checkpoint_final.pt"
        required = {
            "run_manifest.json",
            "progress.json",
            "training_trace.json",
            "result.json",
            "checkpoint_final.pt",
            "COMPLETE",
        }
    elif status == "failed":
        outcome = "failed"
        checkpoint_path = output / "checkpoint_failure.pt"
        required = {
            "run_manifest.json",
            "failure.json",
            "result.json",
            "checkpoint_failure.pt",
            "FAILED",
        }
        for optional in ("training_trace.json",):
            if (output / optional).exists():
                required.add(optional)
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
            "outcome": outcome,
        },
    )
    if not valid or _receipt_names(output) != required:
        return False
    try:
        root = Path(spec.campaign_root).resolve()
        identity = strict_json_load(root / ROOT_MARKER)
        manifest = strict_json_load(output / "run_manifest.json")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        baseline_v6._configure_determinism(spec.model_seed)
        reconstructed, _ = _build_model(
            spec,
            baseline_v6.load_config(root / "inputs" / BASELINE_CONFIG.name),
            torch.device("cpu"),
        )
        initial_hash = canonical_tensor_mapping_sha256(reconstructed.state_dict())
        rng = _rng_identities(spec)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        return False
    common = (
        manifest.get("campaign_id") == CAMPAIGN_ID
        and manifest.get("protocol_revision") == PROTOCOL_REVISION
        and manifest.get("track_classification") == TRACK_CLASSIFICATION
        and manifest.get("run") == spec.payload()
        and manifest.get("scientific_identity") == identity.get("scientific_identity")
        and manifest.get("code_commit") == identity.get("code_commit")
        and manifest.get("runtime_code_sha256") == identity.get("runtime_code_sha256")
        and manifest.get("config_sha256") == identity.get("config_sha256")
        and manifest.get("freeze_sha256") == identity.get("freeze_sha256")
        and manifest.get("parent_binding_sha256") == canonical_hash(identity["parent_binding"])
        and manifest.get("parent_selected_hyperparameters")
        == _selected(identity["parent_binding"], spec.model_id)
        and manifest.get("initial_state_dict_sha256") == initial_hash
        and manifest.get("rng_stream_identities") == rng
        and manifest.get("actual_post_transition_state_noise_std")
        == spec.actual_state_noise_std
        and manifest.get("target_noise_std") == 0.0
        and manifest.get("output_dropout") == 0.0
        and manifest.get("evaluation_state_noise_std") == 0.0
        and manifest.get("evaluation_bank_sha256") == sha256_file(spec.evaluation_bank)
        and checkpoint.get("run") == spec.payload()
        and checkpoint.get("initial_state_dict_sha256") == initial_hash
        and checkpoint.get("rng_stream_identities") == rng
        and result.get("run_id") == spec.run_id
        and result.get("stage") == spec.stage
        and result.get("model_id") == spec.model_id
        and result.get("model_seed") == spec.model_seed
        and result.get("learning_rate") == spec.learning_rate
        and result.get("actual_state_noise_std") == spec.actual_state_noise_std
        and result.get("counts_in_denominator") is True
    )
    if not common:
        return False
    if status == "completed":
        if checkpoint.get("checkpoint_type") != CAMPAIGN_ID or checkpoint.get("result") != result:
            return False
        metrics = result.get("final_metrics")
        if not isinstance(metrics, Mapping) or any(
            key not in metrics or not math.isfinite(float(metrics[key]))
            for key in ("mse", "nmse_db", "masked_mse", "masked_nmse_db")
        ):
            return False
        try:
            state_dict = checkpoint.get("state_dict")
            if not isinstance(state_dict, Mapping):
                return False
            reconstructed.load_state_dict(state_dict, strict=True)
            baseline_v6._finite_model(reconstructed)
            marker = strict_json_load(output / "COMPLETE")
        except (OSError, ValueError, RuntimeError, TypeError, KeyError):
            return False
        return bool(
            marker.get("run_id") == spec.run_id
            and result.get("updates_completed") == spec.updates
        )
    if checkpoint.get("checkpoint_type") != f"{CAMPAIGN_ID}_failure":
        return False
    try:
        failure = strict_json_load(output / "failure.json")
        marker = strict_json_load(output / "FAILED")
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        checkpoint.get("failure") == failure
        and marker.get("run_id") == spec.run_id
        and failure.get("counts_as_failure_in_denominator") is True
        and result.get("updates_completed") is None
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
    verified_complete = len(specs) - len(pending)
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
                spec_path = specs_dir / f"{spec.run_id}.json"
                atomic_json(spec_path, spec.payload())
                handle = (logs_dir / f"{spec.run_id}.log").open("ab")
                command = [
                    sys.executable,
                    "-m",
                    "repro.sagodi_protocol.state_noise_search_v1",
                    "--worker-spec",
                    str(spec_path),
                    "--config",
                    str(config_path),
                    "--device",
                    "cpu" if slot == "cpu" else "cuda:0",
                ]
                environment = os.environ.copy()
                for variable in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                ):
                    environment[variable] = "1"
                if slot != "cpu":
                    environment["CUDA_VISIBLE_DEVICES"] = slot
                    environment["CALRU_PHYSICAL_GPU_SLOT"] = slot
                process = subprocess.Popen(
                    command,
                    cwd=repo,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running[slot] = (process, spec, handle)
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
                    raise RuntimeError(
                        f"retryable state-noise-search worker failure: {spec.run_id}; exit={code}"
                    )
                verified_complete += 1
            atomic_json(
                stage_root / "status.json",
                {
                    "schema_version": 1,
                    "campaign_id": CAMPAIGN_ID,
                    "registered": len(specs),
                    "verified_complete": verified_complete,
                    "pending": len(queue),
                    "running": [item[1].run_id for item in running.values()],
                    "updated_at_utc": _utc_now(),
                },
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
            "checkpoint_final.pt"
            if result.get("status") == "completed"
            else "checkpoint_failure.pt"
        )
        children.append(
            {
                "run_id": spec.run_id,
                "status": result.get("status"),
                "completion_receipt_sha256": sha256_file(
                    output / "completion_receipt.json"
                ),
                "result_sha256": sha256_file(output / "result.json"),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "child_count": len(children),
        "children": children,
    }


def _finish_stage(
    root: Path,
    stage: str,
    specs: Sequence[RunSpec],
    primary_name: str,
    primary_payload: Mapping[str, Any],
) -> Path:
    stage_root = root / stage
    missing = [spec.run_id for spec in specs if not _verified(spec)]
    if missing:
        raise RuntimeError(f"cannot finalize {stage}; missing {missing[:3]}")
    primary = stage_root / primary_name
    binding = stage_root / "children_binding.json"
    complete = stage_root / "COMPUTATION_COMPLETE"
    _write_or_verify(primary, primary_payload)
    _write_or_verify(binding, _children_binding(specs))
    atomic_json(
        complete,
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "stage": stage,
            "verified_run_count": len(specs),
            "completed_at_utc": _utc_now(),
        },
    )
    write_completion_receipt(
        stage_root / "completion_receipt.json",
        job_id=f"{CAMPAIGN_ID}__{stage}",
        artifacts=[stage_root / "plan.json", primary, binding, complete],
        metadata={"campaign_id": CAMPAIGN_ID, "stage": stage, "run_count": len(specs)},
    )
    return stage_root


def _read_specs(stage_root: Path) -> tuple[RunSpec, ...]:
    plan = strict_json_load(stage_root / "plan.json")
    return tuple(RunSpec(**row) for row in plan["runs"])


def _stage_valid(root: Path, stage: str, expected_runs: int) -> bool:
    stage_root = root / stage
    valid, _ = verify_completion_receipt(
        stage_root / "completion_receipt.json",
        expected_job_id=f"{CAMPAIGN_ID}__{stage}",
        expected_metadata={"campaign_id": CAMPAIGN_ID, "stage": stage, "run_count": expected_runs},
    )
    if not valid or not (stage_root / "COMPUTATION_COMPLETE").exists():
        return False
    try:
        specs = _read_specs(stage_root)
        if len(specs) != expected_runs or not all(_verified(spec) for spec in specs):
            return False
        if strict_json_load(stage_root / "children_binding.json") != _children_binding(specs):
            return False
        config = load_config(root / "inputs" / DEFAULT_CONFIG.name)
        parent = strict_json_load(root / "parent_binding.json")
        if stage == "smoke":
            expected = {
                "schema_version": 1,
                "scientific_result": False,
                "runs": [
                    {"run_id": spec.run_id, "status": _result(spec).get("status")}
                    for spec in specs
                ],
            }
            return strict_json_load(stage_root / "summary.json") == expected
        if stage == "tuning":
            expected = select_state_noise(specs, config, parent)
            return strict_json_load(stage_root / "selection.json") == expected
        if stage == "main":
            expected = summarize_main(specs, config)
            return strict_json_load(stage_root / "summary.json") == expected
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        return False
    return False


def run_stage(
    stage: str,
    artifact_root: Path,
    baseline_root: Path,
    downstream_root: Path,
    config_source: Path,
    slots: Sequence[str],
) -> Path:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    root = artifact_root.expanduser().resolve()
    config, parent, copied_config = _prepare_root(
        root,
        baseline_root,
        downstream_root,
        config_source,
        require_clean=stage != "smoke",
    )
    tuning_bank = root / "banks" / "tuning.npz"
    main_bank = root / "banks" / "main_test.npz"
    smoke = build_smoke_plan(root, config, parent, tuning_bank)
    if stage == "smoke":
        if _stage_valid(root, stage, len(smoke)):
            return root / stage
        _run_specs(smoke, copied_config, slots)
        payload = {
            "schema_version": 1,
            "scientific_result": False,
            "runs": [
                {"run_id": spec.run_id, "status": _result(spec).get("status")}
                for spec in smoke
            ],
        }
        return _finish_stage(root, stage, smoke, "summary.json", payload)

    if not _stage_valid(root, "smoke", len(smoke)):
        raise RuntimeError("tuning is blocked until verified smoke completes")
    tuning = build_tuning_plan(root, config, parent, tuning_bank)
    if stage == "tuning":
        if _stage_valid(root, stage, len(tuning)):
            return root / stage
        _run_specs(tuning, copied_config, slots)
        return _finish_stage(
            root,
            stage,
            tuning,
            "selection.json",
            select_state_noise(tuning, config, parent),
        )

    if not _stage_valid(root, "tuning", len(tuning)):
        raise RuntimeError("main is blocked until verified tuning completes")
    selection = strict_json_load(root / "tuning" / "selection.json")
    main = build_main_plan(root, config, parent, main_bank, selection)
    if _stage_valid(root, stage, len(main)):
        return root / stage
    _run_specs(main, copied_config, slots)
    return _finish_stage(root, stage, main, "summary.json", summarize_main(main, config))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--downstream-root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_spec is not None:
        if any(value is not None for value in (args.stage, args.artifact_root, args.baseline_root, args.downstream_root)):
            raise ValueError("worker mode cannot include campaign options")
        spec = RunSpec(**strict_json_load(args.worker_spec))
        config_path = args.config.resolve(strict=True)
        try:
            _train_worker(spec, config_path, str(args.device))
        except FloatingPointError as error:
            _record_numerical_failure(spec, config_path, str(args.device), error)
        return 0
    if any(value is None for value in (args.stage, args.artifact_root, args.baseline_root, args.downstream_root)):
        raise ValueError("campaign mode requires stage, artifact-root, and both parent roots")
    destination = run_stage(
        str(args.stage),
        args.artifact_root,
        args.baseline_root,
        args.downstream_root,
        args.config.resolve(strict=True),
        _parse_slots(args.gpus),
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
