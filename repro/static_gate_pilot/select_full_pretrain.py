"""Select one mature no-RP checkpoint per model and topology."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_split_pretrain_full_v1"
        ),
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    rows: list[dict[str, object]] = []
    for result_path in sorted(root.glob("pretrain__*/result.json")):
        result = json.loads(result_path.read_text())
        manifest = json.loads((result_path.parent / "manifest.json").read_text())
        final = result["final_validation"]
        rows.append(
            {
                "job_id": result["job_id"],
                "model": result["model_id"],
                "topology": result["topology"],
                "seed": result["replicate_seed"],
                "learning_rate": manifest["learning_rate"],
                "initial_retention": manifest["initial_retention"],
                "initial_write_gain": manifest["initial_write_gain"],
                "recurrent_gain": manifest["recurrent_gain"],
                "id_intrinsic_rad": final["intrinsic_mean_radians"],
                "blank128_rad": result["blank_validation"]["128"][
                    "intrinsic_mean_radians"
                ],
                "blank512_rad": result["blank_validation"]["512"][
                    "intrinsic_mean_radians"
                ],
                "blank2048_rad": result["blank_validation"]["2048"][
                    "intrinsic_mean_radians"
                ],
                "command_gain": result["command_response"]["response_gain_median"],
                "checkpoint": str(result_path.parent / "checkpoint.pt"),
            }
        )
    if len(rows) != 8:
        raise ValueError(f"expected 8 mature checkpoints, found {len(rows)}")
    selected: list[dict[str, object]] = []
    gates = {"s1": 0.2, "t2": 0.3}
    for model in ("untied_rnn_rp", "split_rnn_rp"):
        for topology in ("s1", "t2"):
            group = [
                row
                for row in rows
                if row["model"] == model and row["topology"] == topology
            ]
            passing = [
                row
                for row in group
                if float(row["id_intrinsic_rad"]) < gates[topology]
            ]
            pool = passing if passing else group
            best = min(
                pool,
                key=lambda row: (
                    float(row["blank512_rad"]),
                    float(row["id_intrinsic_rad"]),
                ),
            )
            selected.append(
                {
                    **best,
                    "task_gate_passed": bool(passing),
                    "selection_rule": (
                        "task_gate_then_blank512"
                        if passing
                        else "fallback_blank512"
                    ),
                }
            )
    with (root / "rp_parent_checkpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(root / "rp_parent_checkpoints.csv")


if __name__ == "__main__":
    main()
