"""Tests for strict, deterministic evidence aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calru_paper.evidence import (  # noqa: E402
    DEFAULT_SOURCE_ROOT,
    EvidenceError,
    MetricSpec,
    RunSpec,
    TABLE_ORDER,
    build_all_tables,
    build_transport_tables,
    check_reproducibility,
    load_config,
    load_json_runs,
    sample_sd,
    write_tables_atomic,
)


@pytest.fixture(scope="session")
def evidence_config():
    return load_config()


@pytest.fixture(scope="session")
def generated_tables(evidence_config):
    return build_all_tables(DEFAULT_SOURCE_ROOT, evidence_config)


def _toy_spec(pattern: str = "runs/*.json") -> RunSpec:
    return RunSpec(
        run_id="toy.run",
        paper_task="toy_task",
        paper_model="Toy",
        paper_variant="",
        epsilon="",
        legacy_task="legacy_task",
        legacy_model="LegacyModel",
        legacy_tag="legacy_tag",
        source_pattern=pattern,
        metrics=(MetricSpec("score", "raw_score"),),
    )


def _write_toy_runs(root: Path, seeds, *, missing_metric_seed=None, special_value=None) -> None:
    directory = root / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    for index, seed in enumerate(seeds):
        payload = {
            "task": "legacy_task",
            "model": "LegacyModel",
            "tag": "legacy_tag",
            "seed": seed,
            "raw_score": float(index + 1),
        }
        if seed == missing_metric_seed:
            payload.pop("raw_score")
        if special_value is not None and index == 1:
            payload["raw_score"] = special_value
        (directory / f"run_{index}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def _find_row(table, **identity):
    matches = [
        row
        for row in table.rows
        if all(row.get(field) == value for field, value in identity.items())
    ]
    assert len(matches) == 1, identity
    return matches[0]


def test_sample_sd_uses_ddof_one():
    assert sample_sd([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert sample_sd([7.0]) == 0.0


@pytest.mark.parametrize(
    "seeds, expected_message",
    [
        ([0, 1], "expected exact seed ids"),
        ([0, 0, 1], "duplicate seed"),
        ([0, 1, 2, 3], "expected exact seed ids"),
    ],
)
def test_seed_set_must_be_exact_and_unique(tmp_path, seeds, expected_message):
    _write_toy_runs(tmp_path, seeds)
    with pytest.raises(EvidenceError, match=expected_message):
        load_json_runs(tmp_path, _toy_spec())


def test_empty_glob_fails(tmp_path):
    with pytest.raises(EvidenceError, match="matched no files"):
        load_json_runs(tmp_path, _toy_spec())


def test_missing_manifest_fails_instead_of_using_hidden_defaults(tmp_path):
    with pytest.raises(EvidenceError, match="manifest path is not a file"):
        load_config(tmp_path / "missing_manifest.json")


def test_manifest_must_define_every_release_task(tmp_path):
    manifest = json.loads(
        (REPO_ROOT / "configs" / "evidence_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["selected_epsilon_by_task"].pop("surface_integrate")
    path = tmp_path / "incomplete_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(EvidenceError, match="keys must exactly match"):
        load_config(path)


@pytest.mark.parametrize(
    "field,bad_value,expected_message",
    [
        ("task", "wrong_task", "expected raw task"),
        ("model", "WrongModel", "expected raw model"),
        ("tag", "wrong_tag", "expected raw tag"),
    ],
)
def test_raw_identity_must_match_spec(tmp_path, field, bad_value, expected_message):
    _write_toy_runs(tmp_path, [0, 1, 2])
    path = tmp_path / "runs" / "run_1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = bad_value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvidenceError, match=expected_message):
        load_json_runs(tmp_path, _toy_spec())


def test_required_metric_must_exist_on_every_seed(tmp_path):
    _write_toy_runs(tmp_path, [0, 1, 2], missing_metric_seed=1)
    with pytest.raises(EvidenceError, match="missing required metric"):
        load_json_runs(tmp_path, _toy_spec())


@pytest.mark.parametrize(
    "bad_value,expected_message",
    [
        ("not-a-number", "must be a JSON number"),
        (float("nan"), "must be finite"),
        (float("inf"), "must be finite"),
        (-float("inf"), "must be finite"),
    ],
)
def test_required_metric_must_be_numeric_and_finite(
    tmp_path, bad_value, expected_message
):
    _write_toy_runs(tmp_path, [0, 1, 2], special_value=bad_value)
    with pytest.raises(EvidenceError, match=expected_message):
        load_json_runs(tmp_path, _toy_spec())


def test_transport_key_is_unique_per_tag_seed_and_protocol(tmp_path, evidence_config):
    relative_files = (
        "analysis_exp88_ring_transport/ring_transport_writer_seed0.csv",
        "analysis_exp88_ring_transport/ring_transport_seed1.csv",
        "analysis_exp88_ring_transport/ring_transport_seed2.csv",
    )
    for relative in relative_files:
        source = DEFAULT_SOURCE_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    seed_zero = tmp_path / relative_files[0]
    with seed_zero.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    duplicate = next(
        row
        for row in rows
        if row["tag"] == "am_lru_rnw_eps1e-4" and row["blank_steps"] == "2000"
    )
    rows.append(dict(duplicate))
    with seed_zero.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(EvidenceError, match="expected exactly one row"):
        build_transport_tables(tmp_path, evidence_config)


def test_all_nine_tables_have_manifest_row_counts(generated_tables, evidence_config):
    assert tuple(generated_tables) == TABLE_ORDER
    for filename in TABLE_ORDER:
        assert len(generated_tables[filename].rows) == evidence_config.expected_rows[filename]


def test_canonical_labels_and_legacy_provenance_are_separate(generated_tables):
    row = _find_row(
        generated_tables["main_manifold_metrics.csv"],
        paper_task="ring_hold",
        paper_model="CA-LRU",
    )
    assert row["paper_variant"] == "state-dependent update"
    assert row["epsilon"] == "1e-4"
    assert row["legacy_model"] == "PAN-RNW-full"
    assert row["legacy_tag"] == "am_lru_rnw_eps1e-4"

    legacy_tokens = ("pan", "camn", "am-lru", "am_lru")
    for table in generated_tables.values():
        for candidate in table.rows:
            paper_text = f"{candidate['paper_model']} {candidate['paper_variant']}".lower()
            assert not any(token in paper_text for token in legacy_tokens)


def test_full_regeneration_is_byte_identical(
    tmp_path, generated_tables, evidence_config
):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    write_tables_atomic(generated_tables, first_dir)

    independently_rebuilt = build_all_tables(DEFAULT_SOURCE_ROOT, evidence_config)
    write_tables_atomic(independently_rebuilt, second_dir)

    for filename in TABLE_ORDER:
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()
    assert check_reproducibility(DEFAULT_SOURCE_ROOT, first_dir) == ()


def test_paper_regression_values(generated_tables):
    ring_hold = _find_row(
        generated_tables["main_manifold_metrics.csv"],
        paper_task="ring_hold",
        paper_model="CA-LRU",
    )
    assert ring_hold["task_rmse_mean"] == pytest.approx(0.0007350305386353284)

    ring_integrate = _find_row(
        generated_tables["main_manifold_metrics.csv"],
        paper_task="ring_integrate",
        paper_model="CA-LRU",
    )
    assert ring_integrate["task_rmse_mean"] == pytest.approx(0.0021815363434143364)

    torus_input_only = _find_row(
        generated_tables["update_ablation_metrics.csv"],
        paper_task="torus_integrate",
        paper_variant="input-only update",
    )
    assert torus_input_only["task_rmse_mean"] == pytest.approx(0.015149755713840326)

    ring_transport = _find_row(
        generated_tables["ring_transport_summary.csv"],
        paper_model="CA-LRU",
        paper_variant="state-dependent update",
    )
    assert ring_transport["nearest_angle_error_deg_mean"] == pytest.approx(
        0.14014618510970342
    )
