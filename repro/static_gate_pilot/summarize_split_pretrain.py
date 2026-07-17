"""Summarize and select finalists from the RP-disabled pretraining screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from repro.sagodi_protocol.artifacts import strict_json_load

from .launch_split_pretrain import CONFIG_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_pretrain_screen_v1"
        ),
    )
    args = parser.parse_args()
    config = strict_json_load(args.config.expanduser().resolve(strict=True))
    root = args.root.expanduser().resolve(strict=True)
    rows: list[dict[str, Any]] = []
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
                "parameters_total": manifest["model"]["parameters_total"],
                "id_intrinsic_rad": final["intrinsic_mean_radians"],
                "id_nmse_db": final["nmse_db"],
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
                "lambda_mean": final["lambda_mean"],
                "checkpoint": str(result_path.parent / "checkpoint.pt"),
            }
        )
    expected = int(config["screen"]["run_count"])
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} results, found {len(rows)}")
    fields = list(rows[0])
    with (root / "screen_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    selected: list[dict[str, Any]] = []
    finalist_count = int(config["selection"]["finalists_per_model_topology"])
    gates = config["selection"]["task_gate_intrinsic_radians"]
    for model in config["models"]:
        for topology in config["topologies"]:
            group = [
                row
                for row in rows
                if row["model"] == model and row["topology"] == topology
            ]
            passing = [
                row
                for row in group
                if float(row["id_intrinsic_rad"]) < float(gates[topology])
            ]
            passing_sorted = sorted(
                passing,
                key=lambda row: (
                    float(row["blank512_rad"]),
                    float(row["id_intrinsic_rad"]),
                ),
            )
            passing_ids = {row["job_id"] for row in passing}
            fallback = sorted(
                [row for row in group if row["job_id"] not in passing_ids],
                key=lambda row: (
                    float(row["id_intrinsic_rad"]),
                    float(row["blank512_rad"]),
                ),
            )
            pool = (passing_sorted + fallback)[:finalist_count]
            for rank, row in enumerate(pool, 1):
                row_passed = row["job_id"] in passing_ids
                selected.append(
                    {
                        **row,
                        "rank": rank,
                        "task_gate_passed": row_passed,
                        "selection_rule": (
                            "task_gate_then_blank512"
                            if row_passed
                            else "fallback_lowest_id_error"
                        ),
                    }
                )
    with (root / "finalists.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    (root / "selection.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": config["campaign_id"],
                "expected_runs": expected,
                "completed_runs": len(rows),
                "finalists": selected,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(root / "finalists.csv")


if __name__ == "__main__":
    main()
