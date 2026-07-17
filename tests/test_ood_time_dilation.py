import numpy as np

from repro.manifold_benchmark.build_ood_banks import (
    _dilation_schedule,
    _time_dilate,
)
from repro.manifold_benchmark.generator import (
    ConditionSpec,
    ParentSpec,
    derive_s1,
    derive_s2,
    derive_torus,
    make_parent_bank,
)


def test_time_dilation_preserves_commands_path_and_endpoint() -> None:
    parent = make_parent_bank(
        ParentSpec(
            trajectories=5,
            max_horizon=8,
            max_torus_dimension=3,
            training_horizon=8,
            gp_grid_spacing=2.0 / 7.0,
        )
    )
    condition = ConditionSpec(horizon=8)
    sources = (
        derive_s1(parent, condition=condition),
        derive_torus(parent, dimensions=2, condition=condition),
        derive_s2(parent, condition=condition),
    )
    schedule = _dilation_schedule(8, 32)
    inserted = np.ones(32, dtype=bool)
    inserted[schedule] = False

    for source in sources:
        dilated = _time_dilate(
            source,
            target_horizon=32,
            condition_id="temporal_x4",
        )
        assert np.array_equal(dilated.inputs[schedule], source.inputs)
        assert np.count_nonzero(dilated.inputs[inserted]) == 0
        assert np.count_nonzero(dilated.dwell_mask[inserted]) == 0
        assert np.array_equal(
            dilated.output_targets[schedule], source.output_targets
        )
        assert np.array_equal(
            dilated.output_targets[-1], source.output_targets[-1]
        )
        assert np.allclose(
            np.linalg.norm(dilated.effective_velocity, axis=-1).sum(axis=0),
            np.linalg.norm(source.effective_velocity, axis=-1).sum(axis=0),
        )
