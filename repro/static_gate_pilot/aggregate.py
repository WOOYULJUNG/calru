"""Aggregate completed static-gate side-pilot cells into compact tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/result.json")):
        result = json.loads(path.read_text())
        if result.get("campaign_id") not in {None, "static_gate_side_pilot_v1"}:
            continue
        final = result["final_validation"]
        blank = result["blank_validation"]
        radial = result["decoder_radial_recovery"]
        command = result["command_response"]
        rows.append(
            {
                "phase": result["phase"],
                "model": result["model_id"],
                "topology": result["topology"],
                "seed": result["replicate_seed"],
                "updates": result["updates"],
                "id_nmse_db": final["nmse_db"],
                "id_intrinsic_rad": final["intrinsic_mean_radians"],
                "blank128_rad": blank["128"]["intrinsic_mean_radians"],
                "blank512_rad": blank["512"]["intrinsic_mean_radians"],
                "blank2048_rad": blank["2048"]["intrinsic_mean_radians"],
                "command_response_gain": command["response_gain_median"],
                "decoder_radial_ratio512": radial["distance_ratio_median"],
                "decoder_radial_same_memory_rad512": radial[
                    "same_memory_intrinsic_mean_radians"
                ],
                "lambda_min": final["lambda_minimum"],
                "lambda_mean": final["lambda_mean"],
                "lambda_max": final["lambda_maximum"],
                "lambda_gt_0p99": final["lambda_above_0p99"],
                "retention_budget_h2048": final["retention_budget_h2048"],
                "rp_calls": result["rp_calls"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/static_gate_side_pilot_v1"
        ),
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    rows = _rows(root)
    if not rows:
        raise SystemExit("no completed static-gate results found")
    fields = list(rows[0])
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "static_gate_side_pilot_v1",
                "exploratory_only": True,
                "decoder_radial_warning": (
                    "decoder-output radial VJP is not a certified local "
                    "manifold-normal direction"
                ),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(root / "summary.csv")


if __name__ == "__main__":
    main()
