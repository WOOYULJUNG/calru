"""Executable contract for the code-resolved Ságodi baselines.

The NeurIPS paper and the released training code do not specify one identical
recipe.  This module deliberately follows the *executed public-code paths*
for RNN/GRU/LSTM, pinned to commit
``cbd7404e9baca4b2dc291560cfc6576bb7b1f078``.  The small repairs required to
turn those paths into deterministic experiments are registered in the JSON
freeze next to this file; they are never inferred silently at run time.

This is a baseline-reproduction contract.  It is intentionally separate from
the later controlled CA-LRU comparison, in which hyperparameters are tuned
under a common, explicitly registered search budget.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import torch

from .tasks import Batch, keyed_seed

if TYPE_CHECKING:  # pragma: no cover - import only for static checking
    from .source_resolved_models import SourceResolvedBaseline


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_CONFIG = MODULE_DIR / "sagodi_source_resolved_v1.json"
UPSTREAM_COMMIT = "cbd7404e9baca4b2dc291560cfc6576bb7b1f078"
SOURCE_CAMPAIGN_ID = "sagodi_source_resolved_v1"
SOURCE_MODEL_IDS = (
    "sagodi_rnn_tanh_n128",
    "sagodi_gru_n128",
    "sagodi_lstm_n64",
)


@dataclass(frozen=True)
class SourceTrainingRecipe:
    """Model-specific settings used by the released source training path."""

    model_id: str
    width: int
    parameter_count: int
    learning_rate: float
    nominal_state_noise_std: float
    effective_state_noise_std: float
    target_noise_std: float
    output_dropout: float
    recurrent_weight_decay: float
    gradient_clip_norm: float | None
    recurrent_initialization: str


def _require_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise ValueError(f"{label} must be finite" + (" and non-negative" if nonnegative else ""))
    return number


def load_source_config(path: Path | str = DEFAULT_SOURCE_CONFIG) -> dict[str, Any]:
    """Load the immutable source-resolved contract and fail closed on drift."""

    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("source-resolved config must be a schema-1 JSON object")
    if payload.get("campaign_id") != SOURCE_CAMPAIGN_ID:
        raise ValueError("source-resolved campaign_id differs")
    if payload.get("protocol_revision") != "upstream_code_cbd7404_deterministic_repairs_v1":
        raise ValueError("source-resolved protocol revision differs")
    upstream = payload.get("upstream", {})
    if upstream.get("commit") != UPSTREAM_COMMIT:
        raise ValueError("source-resolved upstream commit differs")
    if upstream.get("repository") != (
        "https://github.com/catniplab/back_to_the_continuous_attractor"
    ):
        raise ValueError("source-resolved upstream repository differs")

    task = payload.get("task", {})
    expected_task = {
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
    }
    if task != expected_task:
        raise ValueError("source-resolved task contract differs")

    training = payload.get("training", {})
    if training != {
        "optimizer": "Adam",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "batch_size": 64,
        "updates": 5000,
        "online_batches": True,
    }:
        raise ValueError("source-resolved common training contract differs")
    if payload.get("evaluation") != {
        "trials": 1024,
        "target_noise_std": 0.0,
        "state_noise_std": 0.0,
        "dropout": False,
        "checkpoint": "final_update",
    }:
        raise ValueError("source-resolved evaluation contract differs")

    expected_models = {
        "sagodi_rnn_tanh_n128": {
            "width": 128,
            "parameter_count": 17154,
            "learning_rate": 1e-2,
            "nominal_state_noise_std": 0.1,
            "effective_state_noise_std": math.sqrt(0.1) * 0.1,
            "target_noise_std": 0.0,
            "output_dropout": 0.0,
            "recurrent_weight_decay": 0.0,
            "gradient_clip_norm": None,
            "recurrent_initialization": "normal_gain_1p5_over_sqrt_h",
        },
        "sagodi_gru_n128": {
            "width": 128,
            "parameter_count": 50818,
            "learning_rate": 1e-2,
            "nominal_state_noise_std": 0.0,
            "effective_state_noise_std": 0.0,
            "target_noise_std": 1e-2,
            "output_dropout": 0.5,
            "recurrent_weight_decay": 1e-4,
            "gradient_clip_norm": 100.0,
            "recurrent_initialization": "uniform_plus_minus_0p25_over_sqrt_h",
        },
        "sagodi_lstm_n64": {
            "width": 64,
            "parameter_count": 17538,
            "learning_rate": 1e-3,
            "nominal_state_noise_std": 0.0,
            "effective_state_noise_std": 0.0,
            "target_noise_std": 1e-2,
            "output_dropout": 0.0,
            "recurrent_weight_decay": 1e-2,
            "gradient_clip_norm": 1.0,
            "recurrent_initialization": "uniform_plus_minus_1_over_sqrt_h",
        },
    }
    models = payload.get("models")
    if not isinstance(models, list) or [item.get("id") for item in models] != list(SOURCE_MODEL_IDS):
        raise ValueError("source-resolved model order differs")
    for item in models:
        model_id = str(item["id"])
        expected = expected_models[model_id]
        if set(item) != {"id", *expected.keys()}:
            raise ValueError(f"source-resolved fields differ for {model_id}")
        for key, expected_value in expected.items():
            observed = item[key]
            if isinstance(expected_value, float):
                if not math.isclose(
                    _require_number(observed, f"{model_id}.{key}"),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise ValueError(f"source-resolved {model_id}.{key} differs")
            elif observed != expected_value:
                raise ValueError(f"source-resolved {model_id}.{key} differs")

    repair_ids = [item.get("id") for item in payload.get("deterministic_repairs", [])]
    if repair_ids != [
        "initialize_uninitialized_output_to_state_maps",
        "lstm_cell_initialization_typo",
        "keyed_rng_streams",
        "gp_factorization_jitter",
    ]:
        raise ValueError("source-resolved deterministic-repair registry differs")
    return payload


@lru_cache(maxsize=1)
def source_recipes() -> Mapping[str, SourceTrainingRecipe]:
    """Return immutable, validated model recipes keyed by registered ID."""

    config = load_source_config()
    recipes = {
        item["id"]: SourceTrainingRecipe(
            model_id=str(item["id"]),
            width=int(item["width"]),
            parameter_count=int(item["parameter_count"]),
            learning_rate=float(item["learning_rate"]),
            nominal_state_noise_std=float(item["nominal_state_noise_std"]),
            effective_state_noise_std=float(item["effective_state_noise_std"]),
            target_noise_std=float(item["target_noise_std"]),
            output_dropout=float(item["output_dropout"]),
            recurrent_weight_decay=float(item["recurrent_weight_decay"]),
            gradient_clip_norm=(
                None
                if item["gradient_clip_norm"] is None
                else float(item["gradient_clip_norm"])
            ),
            recurrent_initialization=str(item["recurrent_initialization"]),
        )
        for item in config["models"]
    }
    return MappingProxyType(recipes)


def source_recipe(model_id: str) -> SourceTrainingRecipe:
    try:
        return source_recipes()[str(model_id)]
    except KeyError as error:
        raise ValueError(
            f"unknown source-resolved model {model_id!r}; expected one of {SOURCE_MODEL_IDS}"
        ) from error


@lru_cache(maxsize=4)
def _source_gp_factor(
    horizon: int,
    length_scale: float,
    gp_std: float,
    jitter: float,
) -> np.ndarray:
    grid = np.linspace(-float(length_scale), float(length_scale), int(horizon))
    delta = grid[:, None] - grid[None, :]
    covariance = float(gp_std) ** 2 * np.exp(-0.5 * np.square(delta))
    covariance.flat[:: int(horizon) + 1] += float(jitter)
    factor = np.linalg.cholesky(covariance)
    factor.setflags(write=False)
    return factor


def source_angular_integration(
    batch_size: int,
    base_seed: int,
    *,
    stream_key: Any = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    config_path: Path | str = DEFAULT_SOURCE_CONFIG,
) -> Batch:
    """Generate the released 128-step variable-sparsity angular task.

    Draw order mirrors ``tasks.angularintegration_task``: sparsity and mask,
    GP velocities, then the random initial angle.  Arrays are returned in the
    repository-wide time-major convention, without changing the source task's
    post-update target indexing.
    """

    if isinstance(batch_size, bool) or int(batch_size) != batch_size or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    config = load_source_config(config_path)
    task = config["task"]
    horizon = int(task["horizon"])
    dt = float(task["dt"])
    seed = keyed_seed(
        int(base_seed),
        SOURCE_CAMPAIGN_ID,
        "source_angular_integration",
        stream_key,
    )
    rng = np.random.default_rng(seed)
    batch = int(batch_size)

    # Upstream variable sparsity samples s~U(0,2) and zeros a token when
    # U(0,1) < 1-s.  Values s>1 therefore produce a fully dense trajectory.
    sparsities = rng.uniform(0.0, 2.0, size=batch)
    zero_mask = rng.random(size=(batch, horizon)) < (1.0 - sparsities[:, None])
    white = rng.standard_normal(size=(batch, horizon))
    factor = _source_gp_factor(
        horizon,
        float(task["gp_length_scale"]),
        float(task["gp_std"]),
        float(task["gp_jitter"]),
    )
    velocity = white @ factor.T
    velocity[zero_mask] = 0.0
    q0 = rng.uniform(-math.pi, math.pi, size=(batch, 1))
    angles = np.cumsum(velocity, axis=1) * dt + q0
    outputs = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    inputs_tm = np.ascontiguousarray(velocity.T[:, :, None])
    outputs_tm = np.ascontiguousarray(outputs.transpose(1, 0, 2))
    latent_tm = np.ascontiguousarray(angles.T[:, :, None])
    mask_tm = np.ones_like(outputs_tm)
    tensor = lambda value: torch.as_tensor(value, dtype=dtype, device=device).contiguous()
    return Batch(
        inputs=tensor(inputs_tm),
        output_targets=tensor(outputs_tm),
        latent_targets=tensor(latent_tm),
        mask=tensor(mask_tm),
        metadata={
            "task_name": "angular_integration",
            "task_version": "sagodi-source-resolved-v1",
            "upstream_commit": UPSTREAM_COMMIT,
            "base_seed": int(base_seed),
            "derived_seed": int(seed),
            "stream_key": stream_key,
            "horizon": horizon,
            "duration": float(task["duration"]),
            "delta_t": dt,
            "input_dimension": 1,
            "output_dimension": 2,
            "latent_dimension": 1,
            "input_sparsity": str(task["input_sparsity"]),
            "sampled_sparsities": sparsities,
            "zero_token_counts": zero_mask.sum(axis=1),
            "initial_latents": q0,
            "target_indexing": "post_velocity_update",
            "initial_state_target_index": 0,
            "initial_state_semantics": "source_q1_post_update_target",
        },
    )


def noisy_training_targets(
    clean_targets: torch.Tensor,
    recipe: SourceTrainingRecipe,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Apply the source model's target noise before both init and loss."""

    if clean_targets.ndim != 3:
        raise ValueError("clean_targets must be time-major [time,batch,output]")
    standard_deviation = float(recipe.target_noise_std)
    if standard_deviation == 0.0:
        return clean_targets
    noise = torch.randn(
        clean_targets.shape,
        dtype=clean_targets.dtype,
        device=clean_targets.device,
        generator=generator,
    )
    return clean_targets + standard_deviation * noise


def source_masked_mse(
    prediction: torch.Tensor,
    training_targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Match upstream ``MSELoss(output * mask, target * mask)`` exactly."""

    if prediction.shape != training_targets.shape or mask.shape != prediction.shape:
        raise ValueError("prediction, training_targets, and mask must have the same shape")
    return ((prediction * mask) - (training_targets * mask)).square().mean()


def build_source_optimizer(
    model: "SourceResolvedBaseline",
    recipe: SourceTrainingRecipe | None = None,
) -> torch.optim.Adam:
    """Build the source model's Adam parameter groups without hidden decay."""

    selected = source_recipe(model.model_id) if recipe is None else recipe
    if selected.model_id != model.model_id:
        raise ValueError("optimizer recipe/model mismatch")
    common = load_source_config()["training"]
    recurrent_ids = {id(parameter) for parameter in model.recurrent_parameters()}
    recurrent = [parameter for parameter in model.parameters() if id(parameter) in recurrent_ids]
    remaining = [parameter for parameter in model.parameters() if id(parameter) not in recurrent_ids]
    if not recurrent or not remaining:
        raise RuntimeError("source optimizer partition is incomplete")
    groups = [
        {
            "params": recurrent,
            "weight_decay": float(selected.recurrent_weight_decay),
            "group_name": "recurrent_core",
        },
        {"params": remaining, "weight_decay": 0.0, "group_name": "readout_and_initial_map"},
    ]
    return torch.optim.Adam(
        groups,
        lr=float(selected.learning_rate),
        betas=tuple(float(value) for value in common["betas"]),
        eps=float(common["epsilon"]),
    )


def clip_source_gradients(
    model: "SourceResolvedBaseline",
    recipe: SourceTrainingRecipe | None = None,
) -> torch.Tensor | None:
    """Apply the released model-specific clipping policy after ``backward``."""

    selected = source_recipe(model.model_id) if recipe is None else recipe
    if selected.gradient_clip_norm is None:
        return None
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=float(selected.gradient_clip_norm)
    )
