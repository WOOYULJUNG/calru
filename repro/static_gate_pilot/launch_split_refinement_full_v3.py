"""Select diverse refinement finalists and continue them to 6,000 updates."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

from repro.sagodi_protocol.artifacts import strict_json_load


CONFIG_PATH = Path(__file__).with_name("split_field_refinement_v3.json")


def _read_runs(root: Path) -> list[dict]:
    rows = []
    for result_path in sorted(root.glob("pretrain__*/result.json")):
        result = json.loads(result_path.read_text())
        manifest = json.loads((result_path.parent / "manifest.json").read_text())
        rows.append(
            {
                "job_id": result["job_id"],
                "model": result["model_id"],
                "topology": result["topology"],
                "seed": int(result["replicate_seed"]),
                "width": int(manifest["model"]["width"]),
                "learning_rate": float(manifest["learning_rate"]),
                "initial_retention": float(manifest["initial_retention"]),
                "initial_write_gain": float(manifest["initial_write_gain"]),
                "recurrent_gain": float(manifest["recurrent_gain"]),
                "id_intrinsic_rad": float(
                    result["final_validation"]["intrinsic_mean_radians"]
                ),
                "blank512_rad": float(
                    result["blank_validation"]["512"][
                        "intrinsic_mean_radians"
                    ]
                ),
                "blank2048_rad": float(
                    result["blank_validation"]["2048"][
                        "intrinsic_mean_radians"
                    ]
                ),
                "checkpoint": str(result_path.parent / "checkpoint.pt"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--screen-root",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_refinement_v3"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_split_refinement_full_v3"
        ),
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=Path(
            "/home/biadmin/ca_rnn/experiments/"
            "static_gate_side_pilot_v1/train_pool_seed10"
        ),
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    config = strict_json_load(args.config.expanduser().resolve(strict=True))
    if config["campaign_id"] != "static_gate_split_field_refinement_v3":
        raise ValueError("unexpected campaign")
    screen_root = args.screen_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_cache = args.train_cache.expanduser().resolve(strict=True)
    rows = _read_runs(screen_root)
    expected = int(config["screen"]["run_count"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} screen rows, found {len(rows)}")
    _write_csv(output / "screen_summary.csv", rows)

    selection = config["selection"]
    selected = []
    for topology in config["screen"]["by_topology"]:
        group = [row for row in rows if row["topology"] == topology]
        task_gate = float(
            selection["task_gate_intrinsic_radians"][topology]
        )
        passing = [
            row for row in group if row["id_intrinsic_rad"] < task_gate
        ]
        pool = passing if passing else group
        task_ranked = sorted(
            pool, key=lambda row: row["id_intrinsic_rad"]
        )[: int(selection["task_ranked_per_topology"])]
        blank_ranked = sorted(
            pool,
            key=lambda row: (
                row["blank512_rad"],
                row["id_intrinsic_rad"],
            ),
        )[: int(selection["blank_ranked_per_topology"])]
        low_gain_pool = [
            row
            for row in pool
            if row["recurrent_gain"] <= 0.5
            and row["initial_retention"] <= (0.9 if topology == "s2" else 0.8)
        ]
        low_gain_ranked = sorted(
            low_gain_pool,
            key=lambda row: (
                row["id_intrinsic_rad"] / task_gate
                + row["blank512_rad"],
                row["recurrent_gain"],
            ),
        )[: int(selection["low_gain_ranked_per_topology"])]
        large_width_ranked = []
        if topology == "s2":
            large_width_ranked = sorted(
                [row for row in pool if row["width"] >= 128],
                key=lambda row: (
                    row["id_intrinsic_rad"],
                    row["blank512_rad"],
                ),
            )[: int(selection["large_width_ranked_s2"])]
        selected_ids = {
            row["job_id"]
            for row in (
                task_ranked
                + blank_ranked
                + low_gain_ranked
                + large_width_ranked
            )
        }
        for row in group:
            if row["job_id"] in selected_ids:
                selected.append(
                    {
                        **row,
                        "screen_task_gate_passed": row in passing,
                        "selected_by_task": row in task_ranked,
                        "selected_by_blank": row in blank_ranked,
                        "selected_by_low_gain": row in low_gain_ranked,
                        "selected_by_large_width": row in large_width_ranked,
                    }
                )
    _write_csv(output / "screen_finalists.csv", selected)

    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    screen_updates = int(config["screen"]["updates"])
    total_updates = int(selection["full_pretrain_total_updates"])
    additional_updates = total_updates - screen_updates

    def run(index: int, row: dict) -> dict:
        original_cell = row["job_id"].split("__")[3]
        cell_id = f"full_{original_cell}"
        job_id = (
            f"pretrain__{row['model']}__{row['topology']}__{cell_id}"
            f"__seed{row['seed']}"
        )
        device = devices[index % len(devices)]
        if (output / job_id / "COMPLETED.json").is_file():
            return {
                "job_id": job_id,
                "returncode": 0,
                "device": device,
                "skipped_completed": True,
            }
        command = [
            sys.executable,
            "-m",
            "repro.static_gate_pilot.run",
            "--phase",
            "pretrain",
            "--cell-id",
            cell_id,
            "--model",
            row["model"],
            "--topology",
            row["topology"],
            "--seed",
            str(row["seed"]),
            "--width",
            str(row["width"]),
            "--updates",
            str(additional_updates),
            "--learning-rate",
            str(row["learning_rate"]),
            "--initial-retention",
            str(row["initial_retention"]),
            "--initial-write-gain",
            str(row["initial_write_gain"]),
            "--recurrent-gain",
            str(row["recurrent_gain"]),
            "--report-interval",
            "100",
            "--disable-rp",
            "--load-checkpoint",
            row["checkpoint"],
            "--load-optimizer",
            "--data-update-offset",
            str(screen_updates),
            "--train-cache",
            str(train_cache),
            "--device",
            device,
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        return {
            "job_id": job_id,
            "returncode": int(completed.returncode),
            "device": device,
            "skipped_completed": False,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-8000:],
        }

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run, index, row): row
            for index, row in enumerate(selected)
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"[{len(records):02d}/{len(selected)}] {record['job_id']} "
                f"{'ok' if record['returncode'] == 0 else 'FAILED'}",
                flush=True,
            )
            if record["returncode"] != 0:
                print(record["stderr"], flush=True)
    failures = [record for record in records if record["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} refinement finalists failed")

    full_rows = _read_runs(output)
    if len(full_rows) != len(selected):
        raise ValueError(
            f"expected {len(selected)} full rows, found {len(full_rows)}"
        )
    _write_csv(output / "full_summary.csv", full_rows)
    (output / "pipeline_records.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "static_gate_split_field_refinement_full_v3",
                "screen_root": str(screen_root),
                "screen_rows": len(rows),
                "finalists": len(selected),
                "total_updates": total_updates,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(output / "full_summary.csv")


if __name__ == "__main__":
    main()
