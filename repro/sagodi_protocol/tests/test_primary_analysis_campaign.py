from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from repro.sagodi_protocol.artifacts import atomic_json, canonical_hash, write_completion_receipt
from repro.sagodi_protocol.lr_selection_v3 import EXPECTED_MODEL_IDS
from repro.sagodi_protocol.primary_analysis_campaign import (
    AnalysisRun,
    NORMAL_RECOVERY_DIRECTIONS,
    NORMAL_RECOVERY_METRICS,
    NORMAL_RECOVERY_RADII,
    VerifiedMain,
    _analysis_command,
    _analysis_identity,
    _analysis_output,
    _analysis_spec,
    _require_exact_main_commit,
    _normal_recovery_design,
    _prepare_root,
    _validate_carrier_normal_recovery_artifact,
    _validate_carrier_normal_recovery_summary,
    _recover_or_resume_attempt,
    aggregate_results,
    verify_analysis_output,
)


def _stats(value: float, count: int) -> dict:
    return {
        "registered_count": count,
        "finite_count": count,
        "missing_or_nonfinite_count": 0,
        "mean": value,
        "population_std": 0.0,
        "median": value,
        "q05": value,
        "q95": value,
        "min": value,
        "max": value,
    }


def _normal_recovery_fixture(output: Path, spec) -> dict:
    design = _normal_recovery_design(spec)
    metrics_by_family = {}
    rows = []
    for family, direction_count in NORMAL_RECOVERY_DIRECTIONS.items():
        by_radius = {}
        for radius in NORMAL_RECOVERY_RADII:
            by_horizon = {}
            for horizon in design["horizons"]:
                count = design["anchor_count"] * direction_count
                by_horizon[str(horizon)] = {
                    metric: _stats(0.5, count) for metric in NORMAL_RECOVERY_METRICS
                }
            for anchor in range(design["anchor_count"]):
                for direction in range(direction_count):
                    rows.append((family, anchor, direction, radius))
            by_radius[format(radius, "g")] = {"by_horizon": by_horizon}
        metrics_by_family[family] = {"by_radius": by_radius}

    family = np.asarray([row[0] for row in rows], dtype="U32")
    anchor_index = np.asarray([row[1] for row in rows], dtype=np.int64)
    direction_index = np.asarray([row[2] for row in rows], dtype=np.int64)
    radius = np.asarray([row[3] for row in rows], dtype=np.float64)
    horizon = np.asarray(design["horizons"], dtype=np.int64)
    count = len(rows)
    directions = np.tile(np.asarray([[0.0, 1.0, 0.0]]), (count, 1))
    tangents = np.tile(np.asarray([[1.0, 0.0, 0.0]]), (count, 1))
    numeric = np.full((count, len(horizon)), 0.5, dtype=np.float64)
    np.savez(
        output / "carrier_ambient_normal_recovery.npz",
        family=family,
        anchor_index=anchor_index,
        anchor_angle=np.linspace(-3.0, 3.0, count),
        direction_index=direction_index,
        radius_over_manifold_scale=radius,
        radius_absolute=radius * 2.0,
        direction=directions,
        tangent=tangents,
        direction_norm_error=np.zeros(count),
        absolute_tangent_dot_direction=np.zeros(count),
        horizon=horizon,
        manifold_distance=numeric,
        manifold_distance_ratio=numeric,
        decoded_angle=numeric,
        same_memory_error_radians=numeric,
        nearest_manifold_index=np.tile(anchor_index[:, None], (1, len(horizon))),
        clean_manifold_distance=numeric,
        clean_decoded_angle=numeric,
        clean_same_memory_error_radians=numeric,
        excess_same_memory_error_radians=numeric,
        distance_to_matched_clean_state=numeric,
        distance_to_matched_clean_state_ratio=numeric,
        manifold_distance_minus_clean=numeric,
        manifold_scale=np.asarray(2.0),
    )
    return {
        "role": "project_defined_descriptive_primary_extension_no_threshold",
        "state_space": "minimum_causal_primary_carrier_state",
        "manifold_source": "reconstructed_carrier_spline",
        "distance_definition": "nearest_spline_euclidean",
        "tangent_definition": "normalized_periodic_spline_derivative",
        "deterministic_design": design,
        "manifold_scale": 2.0,
        "numerical_qa": {
            "unique_anchor_count": design["anchor_count"],
            "expected_anchor_count": design["anchor_count"],
            "maximum_direction_norm_error": 0.0,
            "maximum_absolute_tangent_dot_direction": 0.0,
            "initial_manifold_distance": _stats(
                0.5, design["registered_base_perturbation_count"]
            ),
            "initial_manifold_distance_all_finite": True,
            "ambient_normal_base_perturbation_count": design[
                "registered_base_perturbation_count_by_family"
            ]["ambient_normal"],
            "in_plane_radial_base_perturbation_count": design[
                "registered_base_perturbation_count_by_family"
            ]["in_plane_radial"],
            "manifold_scale_finite_positive": True,
        },
        "metrics_by_family": metrics_by_family,
        "claim_gate": False,
    }


def _main_and_manifest(tmp_path: Path, *, all_runs: bool = False):
    main_root = tmp_path / "main"
    main_root.mkdir()
    (main_root / "resolved_primary_main_protocol.yaml").write_text("protocol\n")
    model_seed_pairs = (
        [(model, seed) for model in EXPECTED_MODEL_IDS for seed in range(10)]
        if all_runs
        else [(EXPECTED_MODEL_IDS[0], 0)]
    )
    runs = []
    for index, (model, seed) in enumerate(model_seed_pairs):
        runs.append(
            AnalysisRun(
                model_id=model,
                model_seed=seed,
                learning_rate=1e-3,
                hidden_width=96,
                parameter_count=1000 + index,
                training_run_id=f"primary_main__{model}__seed{seed:02d}",
                training_output=(
                    main_root / "training" / f"model={model}" / f"seed={seed}"
                ),
                checkpoint_sha256=f"{index + 1:064x}",
                training_receipt_sha256=f"{index + 101:064x}",
                training_outcome={
                    "status": "complete",
                    "validation_masked_nmse_db": -30.0,
                    "structural_summary_eligible_nmse_lt_minus20db": True,
                    "train_loss_last": 0.01,
                    "rp_calls": 70 if model == "ca_lru" else 0,
                },
            )
        )
    binding = {
        "resolved_protocol_sha256": "a" * 64,
        "protocol_canonical_fingerprint": "b" * 64,
        "main_code_commit": "d" * 40,
    }
    main = VerifiedMain(
        root=main_root,
        manifest={"smoke": True},
        summary={},
        protocol={},
        runs=tuple(runs),
        binding=binding,
    )
    manifest = {
        "scientific_identity": "c" * 64,
        "protocol_canonical_fingerprint": binding[
            "protocol_canonical_fingerprint"
        ],
    }
    return main, manifest


def test_full_analysis_requires_the_exact_clean_main_commit(tmp_path: Path):
    main, _ = _main_and_manifest(tmp_path)
    matching = {"code_commit": "d" * 40, "worktree_dirty": False}
    _require_exact_main_commit(matching, main, smoke=False)

    with pytest.raises(RuntimeError, match="exact selector/main"):
        _require_exact_main_commit(
            {"code_commit": "e" * 40, "worktree_dirty": False},
            main,
            smoke=False,
        )
    with pytest.raises(RuntimeError, match="clean committed"):
        _require_exact_main_commit(
            {"code_commit": "d" * 40, "worktree_dirty": True},
            main,
            smoke=False,
        )
    _require_exact_main_commit(
        {"code_commit": "e" * 40, "worktree_dirty": True}, main, smoke=True
    )


def _write_analysis(
    output: Path,
    run: AnalysisRun,
    main: VerifiedMain,
    manifest: dict,
    spec,
    *,
    status: str,
    eligible: bool,
):
    output.mkdir(parents=True)
    identity, payload = _analysis_identity(run, main, spec)
    atomic_json(
        output / "analysis_identity.json",
        {
            "schema_version": 1,
            "analysis_identity": identity,
            "identity_payload": payload,
        },
    )
    summary = {
        "schema_version": 1,
        "analysis_status": status,
        "smoke": True,
        "checkpoint": {"sha256": run.checkpoint_sha256},
        "protocol": {
            "sha256": main.binding["resolved_protocol_sha256"],
            "fingerprint": main.binding["protocol_canonical_fingerprint"],
        },
        "structural_summary_eligibility": {"eligible": eligible},
    }
    if status == "complete_structural_summary_eligible":
        recovery = _normal_recovery_fixture(output, spec)
        summary.update(
            {
                "projected_flow": {"uniform_norm": 0.0125},
                "fixed_point_topology": {
                    "kind": "fixed_points",
                    "stable_count": 4,
                    "saddle_count": 4,
                },
                "full_local_eigenspectrum": {
                    "point_count": 8,
                    "largest_real_part": {
                        "count": 8,
                        "mean": -0.01,
                        "min": -0.02,
                        "max": -0.001,
                    },
                    "second_largest_real_part": {
                        "count": 8,
                        "mean": -0.51,
                        "min": -0.77,
                        "max": -0.25,
                    },
                    "top_two_real_part_gap": {
                        "count": 8,
                        "mean": 0.5,
                        "min": 0.25,
                        "max": 0.75,
                    },
                    "map_spectral_radius": {
                        "count": 8,
                        "mean": 0.99,
                        "min": 0.90,
                        "max": 0.999,
                    },
                    "map_spectral_radius_below_one_fraction": 1.0,
                },
                "finite_time_angular_memory": {
                    "named_horizons": {},
                    "terminal_mean_error_radians": 0.1,
                    "terminal_maximum_error_radians": 0.2,
                },
                "asymptotic_structure": {
                    "topology": "fixed_points",
                    "capacity_status": "estimated",
                    "stable_count": 4,
                    "saddle_count": 4,
                    "shannon_entropy_nats": 1.2,
                    "effective_basin_count": 3.3,
                    "asymptotic_mean_error_radians": 0.2,
                    "asymptotic_maximum_error_radians": 0.4,
                },
                "carrier_ambient_normal_recovery": recovery,
            }
        )
    elif status == "structural_analysis_not_estimable":
        summary["structural_numerical_failure"] = {
            "failed_stage": "slow_manifold_reconstruction",
            "error_type": "StructuralNotEstimableError",
            "message": "synthetic structural fixture",
        }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "progress.json", {"stage": "terminal"})
    artifact_paths = [
        output / "analysis_identity.json",
        output / "progress.json",
        output / "direct_task_trajectories.npz",
        output / "summary.json",
    ]
    (output / "direct_task_trajectories.npz").write_bytes(b"fixture")
    if status == "complete_structural_summary_eligible":
        for name in (
            "slow_manifold_reconstruction.npz",
            "projected_flow_and_topology.npz",
            "full_local_eigenspectrum.npz",
            "finite_time_angular_memory.npz",
            "asymptotic_structure.npz",
            "carrier_ambient_normal_recovery.npz",
        ):
            path = output / name
            if not path.exists():
                path.write_bytes(b"fixture")
            artifact_paths.append(path)
    write_completion_receipt(
        output / "completion_receipt.json",
        job_id=f"sagodi-primary-{run.model_id}-{identity[:12]}",
        artifacts=artifact_paths,
        metadata={
            "campaign_scientific_identity": manifest["scientific_identity"],
            "protocol_fingerprint": manifest["protocol_canonical_fingerprint"],
            "run_id": run.run_id,
            "stage": "sagodi_primary_analysis",
            "analysis_identity": identity,
            "analysis_status": status,
        },
    )


def _contains_key(value, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return bool(set(value) & forbidden) or any(
            _contains_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def test_smoke_command_uses_reduced_fixture_sizes_without_changing_single_runner(
    tmp_path: Path,
):
    spec = _analysis_spec(smoke=True)
    main, _ = _main_and_manifest(tmp_path)
    run = main.runs[0]
    command = _analysis_command(run, main, Path("/attempt"), "/python", spec)

    assert "--smoke" in command
    assert command[command.index("--trajectory-count") + 1] == "8"
    assert command[command.index("--spline-count") + 1] == "8"
    assert command[command.index("--task-horizon") + 1] == "8"
    assert command[command.index("--blank-horizon") + 1] == "16"
    assert spec.candidate_distance_chunk_size == 16384


def test_campaign_root_spec_round_trips_with_json_native_tuple_fields(
    tmp_path: Path,
) -> None:
    main, _ = _main_and_manifest(tmp_path)
    spec = _analysis_spec(smoke=True)
    root = tmp_path / "analysis-root"

    _prepare_root(root, main, spec, smoke=True)
    # A second invocation performs the immutable on-disk equality check that
    # previously compared JSON lists against in-memory tuples.
    _prepare_root(root, main, spec, smoke=True)

    marker = __import__("json").loads(
        (root / ".calru_sagodi_primary_analysis_v3_root.json").read_text()
    )
    assert marker["analysis_spec"]["normal_recovery_radii_over_manifold_scale"] == [
        0.01,
        0.05,
        0.1,
    ]
    assert marker["analysis_spec"]["normal_recovery_horizons"] == [
        0,
        1,
        4,
        16,
        64,
        256,
        1024,
        4096,
    ]


@pytest.mark.parametrize(
    ("status", "eligible"),
    [
        ("complete_structural_summary_eligible", True),
        ("ineligible_for_structural_summary", False),
        ("structural_analysis_not_estimable", True),
    ],
)
def test_terminal_outcomes_are_valid_and_bound_to_checkpoint(
    tmp_path: Path, status: str, eligible: bool
):
    main, manifest = _main_and_manifest(tmp_path)
    run = main.runs[0]
    spec = _analysis_spec(smoke=True)
    output = tmp_path / "output"
    _write_analysis(
        output, run, main, manifest, spec, status=status, eligible=eligible
    )

    valid, reason, _ = verify_analysis_output(
        output, run, main, manifest, spec
    )
    assert valid, reason

    summary = __import__("json").loads((output / "summary.json").read_text())
    summary["checkpoint"]["sha256"] = "f" * 64
    atomic_json(output / "summary.json", summary)
    valid, reason, _ = verify_analysis_output(
        output, run, main, manifest, spec
    )
    assert not valid
    assert "hash mismatch" in reason or "checkpoint hash" in reason


def test_aggregation_keeps_all_ten_seed_outcomes_and_no_binary_c1_c4_gate(
    tmp_path: Path,
):
    main, manifest = _main_and_manifest(tmp_path, all_runs=True)
    root = tmp_path / "analysis-campaign"
    spec = _analysis_spec(smoke=True)
    for run in main.runs:
        if run.model_seed == 0:
            status, eligible = "ineligible_for_structural_summary", False
        elif run.model_seed == 1:
            status, eligible = "structural_analysis_not_estimable", True
        else:
            status, eligible = "complete_structural_summary_eligible", True
        _write_analysis(
            _analysis_output(root, run),
            run,
            main,
            manifest,
            spec,
            status=status,
            eligible=eligible,
        )

    summary = aggregate_results(root, main, manifest, spec)

    assert summary["registered_training_outcome_count"] == 60
    assert summary["eligibility_count"] == 54
    assert summary["eligible_and_estimable_count"] == 48
    assert len(summary["runs"]) == 60
    for model in EXPECTED_MODEL_IDS:
        item = summary["model_summaries"][model]
        assert item["registered_training_seed_count"] == 10
        assert item["eligibility_rate_all_10_registered_seeds"] == pytest.approx(0.9)
        assert item["eligible_and_estimable_rate_all_10_registered_seeds"] == pytest.approx(
            0.8
        )
        numeric = item["numeric_descriptive_summaries"]
        validation = numeric["validation_masked_nmse_db_all_registered_seeds"]
        assert validation["registered_seed_denominator"] == 10
        assert validation["finite_value_count"] == 10
        conditional = numeric[
            "sagodi_metrics_conditional_on_eligible_and_estimable"
        ]["uniform_projected_flow_norm"]
        assert conditional["registered_seed_denominator"] == 10
        assert conditional["eligible_and_estimable_conditional_denominator"] == 8
        assert conditional["finite_value_count"] == 8
        assert conditional["mean"] == pytest.approx(0.0125)
        recovery = numeric[
            "sagodi_metrics_conditional_on_eligible_and_estimable"
        ]["carrier_recovery_ambient_normal_r0.01_h0_manifold_distance_ratio_trial_mean"]
        assert recovery["registered_seed_denominator"] == 10
        assert recovery["eligible_and_estimable_conditional_denominator"] == 8
        assert recovery["finite_value_count"] == 8
        assert recovery["missing_or_nonfinite_within_conditional_denominator"] == 0
        assert recovery["mean"] == pytest.approx(0.5)
        assert recovery["q05"] == pytest.approx(0.5)
        assert recovery["q95"] == pytest.approx(0.5)
    assert not _contains_key(
        summary,
        {
            "c1",
            "c2",
            "c3",
            "c4",
            "C1",
            "C2",
            "C3",
            "C4",
            "c1_c4_gate",
            "custom_C1_C4_gates",
            "binary_claim_gate",
        },
    )
    included = [
        row for row in summary["runs"] if row["included_in_primary_structural_summary"]
    ]
    assert all(row["sagodi_metrics"]["uniform_flow_norm"] == 0.0125 for row in included)
    assert all(
        row["sagodi_metrics"]["carrier_ambient_normal_recovery"]["claim_gate"]
        is False
        for row in included
    )


def test_carrier_normal_recovery_missing_or_nonfinite_fails_closed(tmp_path: Path):
    spec = _analysis_spec(smoke=True)
    output = tmp_path / "recovery"
    output.mkdir()
    summary = _normal_recovery_fixture(output, spec)
    _validate_carrier_normal_recovery_summary(summary, spec)
    _validate_carrier_normal_recovery_artifact(
        output / "carrier_ambient_normal_recovery.npz", spec, summary
    )

    damaged = dict(summary)
    damaged_families = {
        key: dict(value) for key, value in summary["metrics_by_family"].items()
    }
    damaged["metrics_by_family"] = damaged_families
    ambient = dict(damaged_families["ambient_normal"])
    damaged_families["ambient_normal"] = ambient
    radii = dict(ambient["by_radius"])
    ambient["by_radius"] = radii
    radius = dict(radii["0.01"])
    radii["0.01"] = radius
    horizons = dict(radius["by_horizon"])
    radius["by_horizon"] = horizons
    horizon = dict(horizons["0"])
    horizons["0"] = horizon
    metric = dict(horizon["manifold_distance_ratio"])
    horizon["manifold_distance_ratio"] = metric
    metric["finite_count"] -= 1
    metric["missing_or_nonfinite_count"] = 1
    with pytest.raises(RuntimeError, match="omits a finite registered value"):
        _validate_carrier_normal_recovery_summary(damaged, spec)

    with np.load(output / "carrier_ambient_normal_recovery.npz") as arrays:
        payload = {name: arrays[name].copy() for name in arrays.files}

    nonorthogonal = {name: value.copy() for name, value in payload.items()}
    nonorthogonal["direction"] = nonorthogonal["tangent"].copy()
    nonorthogonal["direction_norm_error"] = np.abs(
        np.linalg.norm(nonorthogonal["direction"], axis=1) - 1.0
    )
    nonorthogonal["absolute_tangent_dot_direction"] = np.abs(
        np.sum(nonorthogonal["direction"] * nonorthogonal["tangent"], axis=1)
    )
    np.savez(output / "carrier_ambient_normal_recovery.npz", **nonorthogonal)
    with pytest.raises(RuntimeError, match="tangent-orthogonality tolerance"):
        _validate_carrier_normal_recovery_artifact(
            output / "carrier_ambient_normal_recovery.npz", spec, summary
        )

    np.savez(output / "carrier_ambient_normal_recovery.npz", **payload)
    tampered_summary = copy.deepcopy(summary)
    tampered_summary["metrics_by_family"]["ambient_normal"]["by_radius"]["0.01"][
        "by_horizon"
    ]["0"]["manifold_distance_ratio"]["mean"] += 0.125
    with pytest.raises(RuntimeError, match="summary statistic differs from NPZ"):
        _validate_carrier_normal_recovery_artifact(
            output / "carrier_ambient_normal_recovery.npz", spec, tampered_summary
        )

    payload["same_memory_error_radians"][0] = np.nan
    np.savez(output / "carrier_ambient_normal_recovery.npz", **payload)
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        _validate_carrier_normal_recovery_artifact(
            output / "carrier_ambient_normal_recovery.npz", spec, summary
        )


def test_normal_recovery_artifact_accepts_float32_directions_with_canonical_qa(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    spec = _analysis_spec(smoke=True)
    summary = _normal_recovery_fixture(output, spec)
    artifact = output / "carrier_ambient_normal_recovery.npz"
    with np.load(artifact) as arrays:
        payload = {name: arrays[name].copy() for name in arrays.files}

    count = int(payload["direction"].shape[0])
    root_half = np.float32(np.sqrt(0.5))
    direction = np.tile(
        np.asarray([[root_half, root_half, 0.0]], dtype=np.float32),
        (count, 1),
    )
    tangent = np.tile(
        np.asarray([[-root_half, root_half, 0.0]], dtype=np.float32),
        (count, 1),
    )
    direction64 = direction.astype(np.float64)
    tangent64 = tangent.astype(np.float64)
    payload["direction"] = direction
    payload["tangent"] = tangent
    payload["direction_norm_error"] = np.abs(
        np.linalg.norm(direction64, axis=1) - 1.0
    )
    payload["absolute_tangent_dot_direction"] = np.abs(
        np.sum(direction64 * tangent64, axis=1)
    )
    summary["numerical_qa"]["maximum_direction_norm_error"] = float(
        payload["direction_norm_error"].max()
    )
    summary["numerical_qa"]["maximum_absolute_tangent_dot_direction"] = float(
        payload["absolute_tangent_dot_direction"].max()
    )
    np.savez(artifact, **payload)

    _validate_carrier_normal_recovery_artifact(artifact, spec, summary)


def test_partial_attempt_with_matching_identity_is_resumed_not_replaced(tmp_path: Path):
    main, manifest = _main_and_manifest(tmp_path)
    run = main.runs[0]
    spec = _analysis_spec(smoke=True)
    parent = (
        tmp_path
        / "campaign"
        / "attempts"
        / "sagodi_primary_analysis"
        / run.run_id
    )
    attempt = parent / "attempt-0001"
    attempt.mkdir(parents=True)
    identity, payload = _analysis_identity(run, main, spec)
    atomic_json(
        attempt / "analysis_identity.json",
        {"analysis_identity": identity, "identity_payload": payload},
    )
    atomic_json(attempt / "progress.json", {"stage": "full_local_jacobian"})

    resumed, reason = _recover_or_resume_attempt(
        tmp_path / "campaign", run, main, manifest, spec
    )

    assert resumed == attempt
    assert "resume existing attempt" in reason


def test_analysis_identity_binds_exact_resolved_protocol_and_spec(tmp_path: Path):
    main, _ = _main_and_manifest(tmp_path)
    identity, payload = _analysis_identity(
        main.runs[0], main, _analysis_spec(smoke=True)
    )
    assert identity == canonical_hash(payload)
    assert payload["protocol_sha256"] == main.binding["resolved_protocol_sha256"]
    assert payload["protocol_fingerprint"] == main.binding[
        "protocol_canonical_fingerprint"
    ]
    assert payload["spec"]["trajectory_count"] == 8
