"""Measure tangent preservation and normal contraction for one checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import torch

from .artifacts import atomic_json, derived_seed
from .checkpoint import (
    load_checkpoint_model,
    load_checkpoint_protocol,
    validate_evaluation_compatibility,
)
from .config import DEFAULT_PROTOCOL, load_protocol


def _write_state(model, values: torch.Tensor) -> torch.Tensor:
    inputs = torch.zeros(
        values.shape[0], model.input_dim, device=values.device, dtype=values.dtype
    )
    inputs[:, : model.output_dim] = values
    inputs[:, model.output_dim] = 1.0
    state = model.initial_state(
        values.shape[0], device=values.device, dtype=values.dtype
    )
    return model.step(inputs, state)


def _hold(model, state: torch.Tensor, steps: int) -> torch.Tensor:
    zero = torch.zeros(
        state.shape[0], model.input_dim, device=state.device, dtype=state.dtype
    )
    for _ in range(int(steps)):
        state = model.step(zero, state)
    return state


def _task_basis(
    model, *, dimension: int, scale: float, hold_steps: int, device: torch.device
) -> tuple[torch.Tensor, float]:
    dtype = next(model.parameters()).dtype
    targets = torch.zeros(dimension + 1, dimension, device=device, dtype=dtype)
    targets[1:] = torch.eye(dimension, device=device, dtype=dtype) * float(scale)
    states = _hold(model, _write_state(model, targets), hold_steps)
    raw = (states[1:] - states[0]).T.contiguous()
    basis, triangular = torch.linalg.qr(raw, mode="reduced")
    tolerance = max(raw.shape) * torch.finfo(raw.dtype).eps
    if int((triangular.diagonal().abs() > tolerance).sum()) != dimension:
        raise RuntimeError("task basis is rank deficient")
    amplitude = 0.25 * float(torch.pdist(states).median().cpu())
    if amplitude <= 0:
        raise RuntimeError("task states have zero separation")
    return basis, amplitude


def _directions(
    basis: torch.Tensor, *, count: int, seed: int, normal: bool
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if normal:
        values = torch.randn(count, basis.shape[0], generator=generator)
        values = values.to(device=basis.device, dtype=basis.dtype)
        values = values - (values @ basis) @ basis.T
    else:
        coefficients = torch.randn(count, basis.shape[1], generator=generator)
        coefficients = coefficients.to(device=basis.device, dtype=basis.dtype)
        values = coefficients @ basis.T
    return values / torch.clamp(
        torch.linalg.vector_norm(values, dim=-1, keepdim=True), min=1e-12
    )


def _parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(sorted({int(item) for item in value.split(",")}))
    if not horizons or horizons[0] < 0:
        raise ValueError("horizons must be non-negative")
    return horizons


@torch.no_grad()
def run(
    *,
    protocol_path: str | Path,
    run_dir: str | Path,
    output_dir: str | Path,
    device: str,
    trials: int,
    basis_scale: float,
    settle_steps: int,
    horizons: Iterable[int],
) -> dict:
    protocol = load_protocol(protocol_path)
    model, checkpoint = load_checkpoint_model(
        run_dir, protocol, device=device, allow_protocol_mismatch=True
    )
    dimension = int(checkpoint["train_spec"]["dimension"])
    training_protocol = load_checkpoint_protocol(run_dir, checkpoint)
    validate_evaluation_compatibility(
        training_protocol, protocol, dimension=dimension
    )
    device_object = torch.device(device)
    basis, amplitude = _task_basis(
        model,
        dimension=dimension,
        scale=basis_scale,
        hold_steps=settle_steps,
        device=device_object,
    )

    generator = torch.Generator(device="cpu").manual_seed(
        derived_seed(protocol.evaluation.bank_seed, "geometry", dimension)
    )
    low, high = protocol.task.initial_value_low, protocol.task.initial_value_high
    targets = low + (high - low) * torch.rand(
        trials, dimension, generator=generator
    )
    targets = targets.to(device_object)
    clean = _hold(model, _write_state(model, targets), settle_steps)
    tangent = _directions(
        basis,
        count=trials,
        seed=derived_seed(protocol.evaluation.bank_seed, "tangent", dimension),
        normal=False,
    )
    normal = _directions(
        basis,
        count=trials,
        seed=derived_seed(protocol.evaluation.bank_seed, "normal", dimension),
        normal=True,
    )
    states = torch.cat(
        [clean, clean + amplitude * tangent, clean + amplitude * normal], dim=0
    )
    horizons = tuple(sorted(set(int(value) for value in horizons)))
    snapshots = {0: states.clone()} if 0 in horizons else {}
    zero = torch.zeros(
        states.shape[0], model.input_dim, device=device_object, dtype=states.dtype
    )
    for step in range(1, max(horizons) + 1):
        states = model.step(zero, states)
        if step in horizons:
            snapshots[step] = states.clone()

    count = int(trials)
    initial = snapshots[min(horizons)] if min(horizons) == 0 else torch.cat(
        [clean, clean + amplitude * tangent, clean + amplitude * normal], dim=0
    )
    clean_output_0 = model.decode(initial[:count])
    tangent_output_0 = torch.linalg.vector_norm(
        model.decode(initial[count : 2 * count]) - clean_output_0, dim=-1
    )
    normal_output_0 = torch.linalg.vector_norm(
        model.decode(initial[2 * count :]) - clean_output_0, dim=-1
    )

    rows = []
    for horizon in horizons:
        current = snapshots[horizon]
        clean_h = current[:count]
        tangent_delta = current[count : 2 * count] - clean_h
        normal_delta = current[2 * count :] - clean_h
        clean_output = model.decode(clean_h)
        tangent_output = torch.linalg.vector_norm(
            model.decode(current[count : 2 * count]) - clean_output, dim=-1
        )
        normal_output = torch.linalg.vector_norm(
            model.decode(current[2 * count :]) - clean_output, dim=-1
        )
        rows.append(
            {
                "horizon": horizon,
                "tangent_hidden_gain": float(
                    (torch.linalg.vector_norm(tangent_delta, dim=-1) / amplitude)
                    .mean()
                    .cpu()
                ),
                "normal_hidden_gain": float(
                    (torch.linalg.vector_norm(normal_delta, dim=-1) / amplitude)
                    .mean()
                    .cpu()
                ),
                "normal_to_tangent_gain": float(
                    (torch.linalg.vector_norm(normal_delta @ basis, dim=-1) / amplitude)
                    .mean()
                    .cpu()
                ),
                "tangent_decoded_gain": float(
                    (
                        tangent_output
                        / torch.clamp(tangent_output_0, min=1e-12)
                    )
                    .mean()
                    .cpu()
                ),
                "normal_decoded_gain": float(
                    (normal_output / torch.clamp(normal_output_0, min=1e-12))
                    .mean()
                    .cpu()
                ),
            }
        )

    spectrum = None
    if hasattr(model, "retention"):
        retention = model.retention().detach().cpu()
        spectrum = {
            "exact_count": int((retention == 1).sum()),
            "effective_count": int((retention.double().pow(17017) >= 0.99).sum()),
            "below_half_count": int((retention < 0.5).sum()),
        }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "geometry.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dimension": dimension,
        "trials": trials,
        "basis_scale": basis_scale,
        "settle_steps": settle_steps,
        "kick_amplitude": amplitude,
        "spectrum": spectrum,
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--basis-scale", type=float, default=0.25)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument(
        "--horizons", default="0,50,200,1000,2000,5000,10000"
    )
    args = parser.parse_args()
    run(
        protocol_path=args.protocol,
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        device=args.device,
        trials=args.trials,
        basis_scale=args.basis_scale,
        settle_steps=args.settle_steps,
        horizons=_parse_horizons(args.horizons),
    )


if __name__ == "__main__":
    main()
