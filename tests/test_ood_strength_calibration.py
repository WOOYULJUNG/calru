from repro.manifold_benchmark.calibrate_ood_strength import _classify


POLICY = {
    "collapsed_if_chance_ratio_at_least": 0.75,
    "collapsed_if_id_error_ratio_above": 8.0,
    "transition_if_chance_ratio_at_least": 0.5,
    "transition_if_id_error_ratio_above": 4.0,
}


def test_calibration_distinguishes_alive_transition_and_collapse() -> None:
    assert (
        _classify(
            finite_fraction=1.0,
            chance_ratio=0.2,
            id_error_ratio=2.0,
            policy=POLICY,
        )
        == "alive"
    )
    assert (
        _classify(
            finite_fraction=1.0,
            chance_ratio=0.55,
            id_error_ratio=3.0,
            policy=POLICY,
        )
        == "transition"
    )
    assert (
        _classify(
            finite_fraction=1.0,
            chance_ratio=0.3,
            id_error_ratio=9.0,
            policy=POLICY,
        )
        == "severely_degraded"
    )
    assert (
        _classify(
            finite_fraction=0.0,
            chance_ratio=0.0,
            id_error_ratio=1.0,
            policy=POLICY,
        )
        == "chance_collapsed"
    )
