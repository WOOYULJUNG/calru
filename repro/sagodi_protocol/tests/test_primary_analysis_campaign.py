from __future__ import annotations

from pathlib import Path

import pytest

from repro.sagodi_protocol.artifacts import atomic_json, canonical_hash, write_completion_receipt
from repro.sagodi_protocol.lr_selection_v3 import EXPECTED_MODEL_IDS
from repro.sagodi_protocol.primary_analysis_campaign import (
    AnalysisRun,
    VerifiedMain,
    _analysis_command,
    _analysis_identity,
    _analysis_output,
    _analysis_spec,
    _require_exact_main_commit,
    _recover_or_resume_attempt,
    aggregate_results,
    verify_analysis_output,
)


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
        ):
            path = output / name
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
