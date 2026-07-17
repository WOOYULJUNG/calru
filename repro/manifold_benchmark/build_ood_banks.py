"""Build paired fixed OOD banks from the frozen generator-v1 test parent."""

from __future__ import annotations

import argparse
from dataclasses import fields
import os
from pathlib import Path
from typing import Any

import numpy as np

from repro.sagodi_protocol.artifacts import strict_json_load

from .artifacts import (
    atomic_json,
    load_manifold_bank,
    load_parent_bank,
    save_manifold_bank,
    sha256_file,
)
from .audit import exact_bank_audit, prefix_pairing_audit, topology_statistics
from .generator import ConditionSpec, ManifoldBatch, derive_s1, derive_s2, derive_torus
from .topology_analysis_common import canonical_sha256


TOPOLOGIES = ("s1", "t2", "s2")
CONDITION_FIELDS = {item.name for item in fields(ConditionSpec)}


def _condition(row: dict[str, Any]) -> ConditionSpec:
    payload = {
        name: row[name]
        for name in CONDITION_FIELDS
        if name in row
    }
    payload["condition_axis"] = str(row["axis"])
    payload["condition_value"] = str(row["value"])
    return ConditionSpec(**payload)


def _derive(parent, topology: str, condition: ConditionSpec) -> ManifoldBatch:
    if topology == "s1":
        return derive_s1(parent, condition=condition)
    if topology == "t2":
        return derive_torus(parent, dimensions=2, condition=condition)
    if topology == "s2":
        return derive_s2(parent, condition=condition)
    raise ValueError(topology)


def _source_parent_path(config: dict[str, Any]) -> Path:
    configured = Path(str(config["source_parent"])).expanduser()
    if configured.is_absolute():
        return configured.resolve(strict=True)
    environment = str(
        config.get("experiment_root_environment", "CALRU_EXPERIMENT_ROOT")
    )
    root = os.environ.get(environment)
    if not root:
        raise RuntimeError(
            f"{environment} must point to the frozen experiment root"
        )
    return (Path(root).expanduser() / configured).resolve(strict=True)


def _subset(batch: ManifoldBatch, count: int, condition_id: str) -> ManifoldBatch:
    trajectories = int(count)
    metadata = {
        **dict(batch.metadata),
        "ood_condition_id": str(condition_id),
        "evaluation_trajectories": trajectories,
        "trajectory_subset": "first_N_from_frozen_test_parent",
    }
    return ManifoldBatch(
        initial_memory=batch.initial_memory[:trajectories],
        inputs=batch.inputs[:, :trajectories],
        output_targets=batch.output_targets[:, :trajectories],
        latent_targets=batch.latent_targets[:, :trajectories],
        latent_path=batch.latent_path[:, :trajectories],
        base_drive=batch.base_drive[:, :trajectories],
        effective_velocity=batch.effective_velocity[:, :trajectories],
        dwell_mask=batch.dwell_mask[:, :trajectories],
        trajectory_id=batch.trajectory_id[:trajectories],
        mask=batch.mask[:, :trajectories],
        latent_unwrapped=(
            None
            if batch.latent_unwrapped is None
            else batch.latent_unwrapped[:, :trajectories]
        ),
        metadata=metadata,
    )


def _paired_checks(
    condition_id: str,
    row: dict[str, Any],
    banks: dict[str, ManifoldBatch],
    id_banks: dict[str, ManifoldBatch] | None,
    length_banks: dict[int, dict[str, ManifoldBatch]],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    if id_banks is None:
        return checks
    for topology, bank in banks.items():
        reference = id_banks[topology]
        checks[f"{topology}.same_q0"] = bool(
            np.array_equal(bank.initial_memory, reference.initial_memory)
        )
        checks[f"{topology}.same_trajectory_id"] = bool(
            np.array_equal(bank.trajectory_id, reference.trajectory_id)
        )
    axis = str(row["axis"])
    if axis == "length":
        prefix = prefix_pairing_audit(id_banks, banks)
        checks["id_prefix_exact"] = bool(prefix["pass"])
    elif axis == "velocity":
        scale = float(row["value"])
        for topology, bank in banks.items():
            reference = id_banks[topology]
            checks[f"{topology}.same_dwell"] = bool(
                np.array_equal(bank.dwell_mask, reference.dwell_mask)
            )
            checks[f"{topology}.paired_scaled_input"] = bool(
                np.allclose(
                    bank.inputs,
                    scale * reference.inputs,
                    rtol=2e-6,
                    atol=2e-7,
                )
            )
    elif axis == "dwell":
        for topology, bank in banks.items():
            reference = id_banks[topology]
            checks[f"{topology}.same_base_drive"] = bool(
                np.array_equal(bank.base_drive, reference.base_drive)
            )
    elif axis == "smoothness":
        for topology, bank in banks.items():
            reference = id_banks[topology]
            checks[f"{topology}.same_dwell"] = bool(
                np.array_equal(bank.dwell_mask, reference.dwell_mask)
            )
    elif axis == "combined":
        horizon = int(row["horizon"])
        scale = float(row.get("velocity_scale", 1.0))
        reference_set = length_banks[horizon]
        for topology, bank in banks.items():
            reference = reference_set[topology]
            checks[f"{topology}.same_dwell_as_length"] = bool(
                np.array_equal(bank.dwell_mask, reference.dwell_mask)
            )
            checks[f"{topology}.paired_scaled_length_input"] = bool(
                np.allclose(
                    bank.inputs,
                    scale * reference.inputs,
                    rtol=2e-6,
                    atol=2e-7,
                )
            )
    return checks


def build(output: Path, config_path: Path) -> dict[str, Any]:
    config = strict_json_load(config_path)
    if config.get("schema_version") != 1:
        raise ValueError("OOD config must use schema version 1")
    if tuple(config["topologies"]) != TOPOLOGIES:
        raise ValueError("OOD topology order differs")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output must be a fresh path: {output}")
    output.mkdir(parents=True)
    (output / "INCOMPLETE").write_text("paired OOD banks in progress\n")
    banks_root = output / "banks"
    banks_root.mkdir()

    parent_path = _source_parent_path(config)
    parent = load_parent_bank(parent_path)
    count = int(config["trajectories"])
    if count > parent.trajectories:
        raise ValueError("OOD trajectory count exceeds frozen parent")
    if max(int(row["horizon"]) for row in config["conditions"]) > parent.max_horizon:
        raise ValueError("OOD horizon exceeds frozen parent")

    condition_ids = [str(row["id"]) for row in config["conditions"]]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("OOD condition IDs must be unique")

    manifest_rows: dict[str, Any] = {}
    id_banks: dict[str, ManifoldBatch] | None = None
    length_banks: dict[int, dict[str, ManifoldBatch]] = {}
    all_checks: dict[str, bool] = {}
    for row in config["conditions"]:
        condition_id = str(row["id"])
        condition = _condition(row)
        derived: dict[str, ManifoldBatch] = {}
        files: dict[str, Any] = {}
        for topology in TOPOLOGIES:
            full = _derive(parent, topology, condition)
            batch = _subset(full, count, condition_id)
            path = banks_root / f"{condition_id}__{topology}.npz"
            digest = save_manifold_bank(path, batch)
            loaded = load_manifold_bank(path)
            derived[topology] = loaded
            files[topology] = {
                "path": str(path.relative_to(output)),
                "sha256": digest,
                "statistics": topology_statistics(loaded),
            }
        exact = exact_bank_audit(derived, float_tolerance=2e-6)
        all_checks[f"{condition_id}.exact_bank"] = bool(exact["pass"])
        paired = _paired_checks(
            condition_id,
            row,
            derived,
            id_banks,
            length_banks,
        )
        for name, passed in paired.items():
            all_checks[f"{condition_id}.{name}"] = bool(passed)
        if condition_id == "id_h128":
            id_banks = derived
        if str(row["axis"]) == "length":
            length_banks[int(row["horizon"])] = derived
        manifest_rows[condition_id] = {
            "axis": row["axis"],
            "value": row["value"],
            "label": row["label"],
            "condition_spec": {
                name: getattr(condition, name) for name in CONDITION_FIELDS
            },
            "files": files,
            "exact_audit": exact,
            "paired_checks": paired,
        }
    if id_banks is None:
        raise RuntimeError("OOD config has no id_h128 condition")
    passed = bool(all(all_checks.values()))
    manifest = {
        "schema_version": 1,
        "analysis_id": config["analysis_id"],
        "config_file": config_path.name,
        "config_sha256": canonical_sha256(config),
        "source_parent": str(config["source_parent"]),
        "source_parent_experiment_id": parent_path.parent.name,
        "source_parent_file": parent_path.name,
        "source_parent_sha256": sha256_file(parent_path),
        "trajectories": count,
        "conditions": manifest_rows,
        "checks": all_checks,
        "pass": passed,
    }
    atomic_json(output / "ood_banks_manifest.json", manifest)
    completion = {
        "schema_version": 1,
        "analysis_id": config["analysis_id"],
        "ood_banks_ready": passed,
        "condition_count": len(manifest_rows),
        "topology_count": len(TOPOLOGIES),
        "bank_count": len(manifest_rows) * len(TOPOLOGIES),
        "manifest_sha256": sha256_file(output / "ood_banks_manifest.json"),
    }
    atomic_json(output / "OOD_BANKS_READY.json", completion)
    if not passed:
        raise RuntimeError("paired OOD bank audit failed")
    (output / "INCOMPLETE").unlink()
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("topology_ood_v1.json"),
    )
    args = parser.parse_args()
    build(
        args.output.expanduser().resolve(),
        args.config.expanduser().resolve(strict=True),
    )


if __name__ == "__main__":
    main()
