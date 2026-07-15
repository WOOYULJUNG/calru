from __future__ import annotations

from repro.sagodi_protocol.calru_factorial_analysis_v1 import (
    CONDITIONS,
    EXPECTED_RUNS,
    SEEDS,
    _extended_fields,
    load_config,
)


def _stats(value: float) -> dict[str, float | int]:
    return {
        "registered_count": 1,
        "finite_count": 1,
        "missing_or_nonfinite_count": 0,
        "mean": value,
        "population_std": 0.0,
        "median": value,
        "q05": value,
        "q95": value,
        "min": value,
        "max": value,
    }


def test_factorial_analysis_freezes_four_cells_and_three_seeds() -> None:
    config = load_config()
    assert config["conditions"] == list(CONDITIONS)
    assert config["seeds"] == list(SEEDS)
    assert config["expected_runs"] == EXPECTED_RUNS == 12
    assert config["analysis"]["carrier_ambient_normal_recovery"] is True
    assert config["analysis"]["flow_reversal_fixed_point_topology"] is True


def test_extended_aggregate_keeps_topology_memory_and_recovery() -> None:
    by_radius = {
        radius: {
            "by_horizon": {
                "4096": {
                    "manifold_distance_ratio": _stats(0.2),
                    "same_memory_error_radians": _stats(0.1),
                    "excess_same_memory_error_radians": _stats(0.05),
                }
            }
        }
        for radius in ("0.01", "0.05", "0.1")
    }
    summary = {
        "structural_summary_eligibility": {"eligible": True},
        "projected_flow": {"uniform_norm": 0.01},
        "full_local_eigenspectrum": {
            "largest_real_part": {"mean": -0.001},
            "top_two_real_part_gap": {"mean": 0.1},
        },
        "fixed_point_topology": {
            "kind": "fixed_points",
            "stable_count": 2,
            "saddle_count": 2,
            "stable_angles": [1.0, 4.0],
            "saddle_angles": [2.0, 5.0],
        },
        "finite_time_angular_memory": {
            "terminal_mean_error_radians": 0.2,
            "terminal_maximum_error_radians": 0.4,
        },
        "asymptotic_structure": {"effective_basin_count": 2.0},
        "carrier_ambient_normal_recovery": {
            "metrics_by_family": {
                "ambient_normal": {"by_radius": by_radius},
                "in_plane_radial": {"by_radius": by_radius},
            }
        },
        "manifold_reconstruction": {"qa": {"candidate_count": 10}},
    }

    fields = _extended_fields(summary)

    assert fields["fixed_point_topology"]["stable_count"] == 2
    assert fields["asymptotic_structure"]["effective_basin_count"] == 2.0
    assert (
        fields["finite_normal_recovery_at_4096"]["ambient_normal"]["0.1"]
        ["manifold_distance_ratio"]["median"]
        == 0.2
    )
