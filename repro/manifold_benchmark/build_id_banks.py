"""Build the first fixed S1/T2/S2 banks from one paired stochastic parent."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import (
    atomic_json,
    load_manifold_bank,
    load_parent_bank,
    save_manifold_bank,
    save_parent_bank,
    sha256_file,
)
from .audit import (
    dimension_nesting_audit,
    exact_bank_audit,
    parent_distribution_audit,
    topology_calibration,
)
from .generator import (
    GENERATOR_VERSION,
    ConditionSpec,
    ParentSpec,
    derive_s1,
    derive_s2,
    derive_torus,
    make_parent_bank,
)


def _arrays_equal(left: Any, right: Any, fields: tuple[str, ...]) -> bool:
    return all(np.array_equal(getattr(left, field), getattr(right, field)) for field in fields)


def build(output: Path, spec: ParentSpec) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output must be a fresh path: {output}")
    output.mkdir(parents=True)
    (output / "INCOMPLETE").write_text("manifold benchmark ID banks in progress\n")

    atomic_json(
        output / "spec.json",
        {
            "generator_version": GENERATOR_VERSION,
            "scope": "parent_plus_T128_ID_S1_T2_S2",
            "parent_spec": asdict(spec),
            "sphere_control": {
                "input": "state-independent 3D angular velocity omega_t",
                "update": "Rodrigues rotation R(delta_t * omega_t) n_t",
                "effective_velocity": "omega_t cross n_t",
            },
        },
    )

    original_parent = make_parent_bank(spec)
    parent_path = output / "parent_bank.npz"
    parent_digest = save_parent_bank(parent_path, original_parent)
    parent = load_parent_bank(parent_path)
    parent_roundtrip = _arrays_equal(
        original_parent,
        parent,
        (
            "white_noise",
            "id_base_drive",
            "q0_angles",
            "q0_sphere_gaussian",
            "sparsity_parameters",
            "mask_uniform_randoms",
            "trajectory_id",
        ),
    )

    condition = ConditionSpec(
        horizon=spec.training_horizon,
        condition_axis="id",
        condition_value="training_distribution",
    )
    generated = {
        "s1": derive_s1(parent, condition=condition, storage_dtype=np.float32),
        "t2": derive_torus(
            parent,
            dimensions=2,
            condition=condition,
            energy_mode="coordinate_matched",
            storage_dtype=np.float32,
        ),
        "s2": derive_s2(parent, condition=condition, storage_dtype=np.float32),
    }
    loaded = {}
    bank_digests = {}
    bank_roundtrip = {}
    batch_fields = (
        "initial_memory",
        "inputs",
        "output_targets",
        "latent_targets",
        "latent_path",
        "base_drive",
        "effective_velocity",
        "dwell_mask",
        "trajectory_id",
        "mask",
    )
    for name, batch in generated.items():
        path = output / f"id_{name}.npz"
        bank_digests[name] = save_manifold_bank(path, batch)
        loaded[name] = load_manifold_bank(path)
        bank_roundtrip[name] = _arrays_equal(batch, loaded[name], batch_fields)
        if batch.latent_unwrapped is not None:
            bank_roundtrip[name] = bank_roundtrip[name] and np.array_equal(
                batch.latent_unwrapped, loaded[name].latent_unwrapped
            )

    exact = exact_bank_audit(loaded, float_tolerance=2e-6)
    t8 = derive_torus(
        parent,
        dimensions=8,
        condition=condition,
        energy_mode="coordinate_matched",
        storage_dtype=np.float32,
    )
    nesting = dimension_nesting_audit(loaded["s1"], t8)
    distribution = parent_distribution_audit(parent)
    calibration = topology_calibration(loaded["s1"], loaded["t2"], loaded["s2"])
    roundtrip = {
        "pass": bool(parent_roundtrip and all(bank_roundtrip.values())),
        "parent": bool(parent_roundtrip),
        "banks": bank_roundtrip,
    }

    validation = output / "validation"
    validation.mkdir()
    atomic_json(validation / "exact_tests.json", exact)
    atomic_json(validation / "dimension_nesting.json", nesting)
    atomic_json(validation / "distribution_audit.json", distribution)
    atomic_json(validation / "topology_calibration.json", calibration)
    atomic_json(validation / "artifact_roundtrip.json", roundtrip)

    passed = bool(
        exact["pass"]
        and nesting["pass"]
        and distribution["pass"]
        and calibration["pass"]
        and roundtrip["pass"]
    )
    summary = {
        "generator_version": GENERATOR_VERSION,
        "id_banks_ready": passed,
        "scope": "parent_plus_T128_ID_S1_T2_S2",
        "not_yet_claimed": [
            "OOD pairing complete",
            "input leakage probe complete",
            "GENERATOR_VERIFIED",
        ],
        "parent_bank_sha256": parent_digest,
        "bank_sha256": bank_digests,
        "checks": {
            "exact": exact["pass"],
            "dimension_nesting": nesting["pass"],
            "distribution": distribution["pass"],
            "topology_calibration": calibration["pass"],
            "artifact_roundtrip": roundtrip["pass"],
        },
        "artifact_sha256": {
            "spec.json": sha256_file(output / "spec.json"),
            "validation/exact_tests.json": sha256_file(
                validation / "exact_tests.json"
            ),
            "validation/dimension_nesting.json": sha256_file(
                validation / "dimension_nesting.json"
            ),
            "validation/distribution_audit.json": sha256_file(
                validation / "distribution_audit.json"
            ),
            "validation/topology_calibration.json": sha256_file(
                validation / "topology_calibration.json"
            ),
            "validation/artifact_roundtrip.json": sha256_file(
                validation / "artifact_roundtrip.json"
            ),
        },
    }
    atomic_json(output / "summary.json", summary)
    if passed:
        atomic_json(output / "GENERATOR_ID_BANKS_READY.json", summary)
        (output / "INCOMPLETE").unlink()
    else:
        atomic_json(output / "GENERATOR_ID_BANKS_FAILED.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=1024)
    parser.add_argument("--max-horizon", type=int, default=2048)
    parser.add_argument("--training-horizon", type=int, default=128)
    parser.add_argument("--max-torus-dimension", type=int, default=8)
    parser.add_argument("--task-seed", type=int, default=20260716)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--split", default="validation_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = ParentSpec(
        trajectories=args.trajectories,
        max_horizon=args.max_horizon,
        max_torus_dimension=args.max_torus_dimension,
        training_horizon=args.training_horizon,
        gp_grid_spacing=2.0 / (args.training_horizon - 1),
        task_seed=args.task_seed,
        sample_seed=args.sample_seed,
        split=args.split,
    )
    summary = build(args.output, spec)
    if not summary["id_banks_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
