"""Compare calibrated RP branches with their mature no-RP parents."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parents",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_split_pretrain_full_v1/"
            "rp_parent_checkpoints.csv"
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_rp_sweep_v1"
        ),
    )
    args = parser.parse_args()
    parents = list(csv.DictReader(args.parents.expanduser().resolve(strict=True).open()))
    parent_lookup = {
        (row["model"], row["topology"], int(row["seed"])): row for row in parents
    }
    root = args.root.expanduser().resolve(strict=True)
    rows: list[dict[str, Any]] = []
    for result_path in sorted(root.glob("rp__*/result.json")):
        result = json.loads(result_path.read_text())
        manifest = json.loads((result_path.parent / "manifest.json").read_text())
        key = (result["model_id"], result["topology"], int(result["replicate_seed"]))
        parent = parent_lookup[key]
        final = result["final_validation"]
        blank2048 = result["blank_validation"]["2048"]["intrinsic_mean_radians"]
        rows.append(
            {
                "job_id": result["job_id"],
                "model": result["model_id"],
                "topology": result["topology"],
                "seed": result["replicate_seed"],
                "eta": manifest["gate_intervention_rp"]["eta_lambda"],
                "update_rule": manifest["gate_intervention_rp"]["update_rule"],
                "max_theta_step": manifest["gate_intervention_rp"]["max_theta_step"],
                "id_intrinsic_rad": final["intrinsic_mean_radians"],
                "blank128_rad": result["blank_validation"]["128"][
                    "intrinsic_mean_radians"
                ],
                "blank512_rad": result["blank_validation"]["512"][
                    "intrinsic_mean_radians"
                ],
                "blank2048_rad": blank2048,
                "parent_id_rad": float(parent["id_intrinsic_rad"]),
                "parent_blank2048_rad": float(parent["blank2048_rad"]),
                "id_ratio_to_parent": float(final["intrinsic_mean_radians"])
                / max(float(parent["id_intrinsic_rad"]), 1e-12),
                "blank2048_ratio_to_parent": float(blank2048)
                / max(float(parent["blank2048_rad"]), 1e-12),
                "lambda_min": final["lambda_minimum"],
                "lambda_mean": final["lambda_mean"],
                "lambda_max": final["lambda_maximum"],
                "rp_calls": result["rp_calls"],
                "checkpoint": str(result_path.parent / "checkpoint.pt"),
                "parent_checkpoint": parent["checkpoint"],
                "learning_rate": parent["learning_rate"],
                "initial_retention": parent["initial_retention"],
                "initial_write_gain": parent["initial_write_gain"],
                "recurrent_gain": parent["recurrent_gain"],
            }
        )
    if len(rows) != 32:
        raise ValueError(f"expected 32 RP results, found {len(rows)}")
    with (root / "rp_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected: list[dict[str, Any]] = []
    absolute_gates = {"s1": 0.2, "t2": 0.3}
    for key, parent in parent_lookup.items():
        model, topology, seed = key
        group = [
            row
            for row in rows
            if row["model"] == model
            and row["topology"] == topology
            and int(row["seed"]) == seed
        ]
        passing = [
            row
            for row in group
            if float(row["id_intrinsic_rad"]) < absolute_gates[topology]
            and float(row["id_ratio_to_parent"]) <= 1.5
        ]
        if passing:
            best = min(
                passing,
                key=lambda row: (
                    float(row["blank2048_ratio_to_parent"]),
                    float(row["id_intrinsic_rad"]),
                ),
            )
            use_rp = float(best["blank2048_ratio_to_parent"]) < 1.0
        else:
            best = min(group, key=lambda row: float(row["id_intrinsic_rad"]))
            use_rp = False
        selected.append(
            {
                **best,
                "task_gate_passed": bool(passing),
                "rp_improves_parent_blank2048": (
                    float(best["blank2048_ratio_to_parent"]) < 1.0
                ),
                "recommended_use_rp": use_rp,
            }
        )
    with (root / "rp_finalists.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    (root / "selection.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "static_gate_rp_sweep_v1",
                "rows": len(rows),
                "selection": selected,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(root / "rp_finalists.csv")


if __name__ == "__main__":
    main()
