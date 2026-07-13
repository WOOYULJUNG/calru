"""Normative task and latent metrics from the frozen protocol."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    weights = torch.broadcast_to(mask, target.shape).to(dtype=target.dtype)
    denom = weights.sum().clamp_min(1.0)
    return (weights * (prediction - target).square()).sum() / denom


def masked_nmse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    weights = torch.broadcast_to(mask, target.shape).to(dtype=target.dtype)
    numerator = (weights * (prediction - target).square()).sum()
    denominator = (weights * target.square()).sum().clamp_min(torch.finfo(target.dtype).eps)
    return numerator / denominator


def nmse_db(value: torch.Tensor | float) -> float:
    scalar = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
    return 10.0 * math.log10(max(scalar, 1e-30))


def wrap_angle(delta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def normalized_torus_geodesic(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    if q_a.shape != q_b.shape or q_a.ndim < 1:
        raise ValueError("latent tensors must have equal shape")
    delta = wrap_angle(q_a - q_b)
    return torch.sqrt(delta.square().mean(dim=-1)) / math.pi


def angle_from_embedding(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] % 2:
        raise ValueError("cos/sin embedding must have an even final dimension")
    pairs = values.reshape(*values.shape[:-1], -1, 2)
    return torch.atan2(pairs[..., 1], pairs[..., 0])


def distribution_summary(values: torch.Tensor | np.ndarray) -> dict[str, float]:
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    if not flat.size or not np.isfinite(flat).all():
        raise ValueError("metric distribution is empty or non-finite")
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(ddof=0)),
        "median": float(np.median(flat)),
        "q95": float(np.quantile(flat, 0.95)),
        "q99": float(np.quantile(flat, 0.99)),
        "max": float(flat.max()),
    }


def task_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    latent_target: torch.Tensor | None = None,
) -> dict[str, Any]:
    nmse = masked_nmse(prediction, target, mask)
    result: dict[str, Any] = {
        "masked_mse": float(masked_mse(prediction, target, mask).detach().cpu()),
        "masked_nmse": float(nmse.detach().cpu()),
        "masked_nmse_db": nmse_db(nmse),
    }
    if latent_target is not None and target.shape[-1] == 2 * latent_target.shape[-1]:
        decoded = angle_from_embedding(prediction)
        distance = normalized_torus_geodesic(decoded, latent_target)
        result["geodesic"] = distribution_summary(distance)
    return result
