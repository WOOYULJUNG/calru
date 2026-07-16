"""Aggregate completed topology-transfer jobs without loading checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any

from repro.sagodi_protocol.artifacts import atomic_json, strict_json_load


def summarize(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(root.glob("*/result.json")):
        run_dir = result_path.parent
        if not (run_dir / "COMPLETED.json").is_file():
            continue
        result = strict_json_load(result_path)
        trace = strict_json_load(run_dir / "trace.json")["trace"]
        final = result["final_validation"]
        hold = result["hold_baseline"]
        late = [
            item["validation"]["intrinsic_mean_radians"]
            for item in trace
            if int(item["update"]) >= max(1, int(result["updates"]) * 3 // 5)
        ]
        rows.append(
            {
                "job_id": result["job_id"],
                "stage": result["stage"],
                "model_id": result["model_id"],
                "topology": result["topology"],
                "seed": result["replicate_seed"],
                "updates": result["updates"],
                "final_component_mse": final["component_mse"],
                "final_intrinsic_mean_radians": final["intrinsic_mean_radians"],
                "late_intrinsic_median_radians": statistics.median(late),
                "late_intrinsic_best_radians": min(late),
                "hold_intrinsic_mean_radians": hold["intrinsic_mean_radians"],
                "final_over_hold_intrinsic_ratio": (
                    final["intrinsic_mean_radians"]
                    / hold["intrinsic_mean_radians"]
                ),
                "beats_hold_baseline": result["beats_hold_baseline_intrinsic"],
                "first_step_component_mse": final["first_step_component_mse"],
                "final_step_component_mse": final["final_step_component_mse"],
                "output_norm_absolute_error": final["output_norm_absolute_error"],
                "prediction_component_variance_mean": final[
                    "prediction_component_variance_mean"
                ],
                "hidden_norm_max": final["hidden_norm_max"],
                "lambda_minimum": final["lambda_minimum"],
                "lambda_mean": final["lambda_mean"],
                "lambda_maximum": final["lambda_maximum"],
                "finite": result["finite"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    rows = summarize(root)
    atomic_json(
        root / "aggregate_summary.json",
        {"schema_version": 1, "completed_job_count": len(rows), "rows": rows},
    )
    if rows:
        with (root / "aggregate_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"completed_job_count": len(rows), "root": str(root)}))


if __name__ == "__main__":
    main()
