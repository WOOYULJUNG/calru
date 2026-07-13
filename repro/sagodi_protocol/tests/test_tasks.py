from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
import torch

from repro.sagodi_protocol.tasks import (
    ANGULAR_DT,
    ANGULAR_HORIZON,
    MGS_HORIZON,
    Batch,
    angular_integration,
    double_angular_integration,
    keyed_seed,
    load_fixed_bank,
    memory_guided_saccade,
    sample_angular_integration,
    sample_double_angular_integration,
    sample_memory_guided_saccade,
    save_fixed_bank,
    sha256_file,
)


def _wrapped_delta(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(left - right), torch.cos(left - right))


def test_batch_is_structurally_immutable_and_validates_time_major_shapes():
    batch = memory_guided_saccade(2, 7)
    assert isinstance(batch, Batch)
    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.inputs = torch.empty(0)  # type: ignore[misc]
    with pytest.raises(TypeError):
        batch.metadata["new"] = "forbidden"  # type: ignore[index]
    with pytest.raises(ValueError, match="time-major rank 3"):
        Batch(
            inputs=torch.zeros(2, 3),
            output_targets=torch.zeros(2, 3, 1),
            latent_targets=torch.zeros(2, 3, 1),
            mask=torch.zeros(2, 3, 1),
            metadata={},
        )


def test_keyed_seed_is_stable_and_namespaced():
    assert keyed_seed(17, "task", 3) == keyed_seed(17, "task", 3)
    assert keyed_seed(17, "task", 3) != keyed_seed(17, "task", 4)
    assert keyed_seed(17, "task", 3) != keyed_seed(18, "task", 3)
    with pytest.raises(ValueError):
        keyed_seed(-1, "task")


def test_stable_sampler_api_separates_task_and_sample_seeds():
    first = sample_angular_integration(3, 16, 7, 11, "hidden-init")
    repeat = sample_angular_integration(3, 16, 7, 11, "hidden-init")
    changed_sample = sample_angular_integration(3, 16, 7, 12, "hidden-init")
    assert torch.equal(first.inputs, repeat.inputs)
    assert not torch.equal(first.inputs, changed_sample.inputs)
    assert first.metadata["task_seed"] == 7
    assert first.metadata["sample_seed"] == 11
    assert first.initial_memory is not None
    assert first.initial_memory.shape == (3, 2)

    saccade = sample_memory_guided_saccade(2, 7, 11)
    double = sample_double_angular_integration(2, 16, 7, 11, "cue-driven")
    assert saccade.initial_memory is not None
    assert saccade.initial_memory.shape == (2, 2)
    assert double.initial_memory is not None
    assert double.initial_memory.shape == (2, 4)


def test_memory_guided_saccade_exact_segments_delay_and_mask():
    batch = memory_guided_saccade(24, 123, dtype=torch.float64)
    repeat = memory_guided_saccade(24, 123, dtype=torch.float64)
    assert torch.equal(batch.inputs, repeat.inputs)
    assert torch.equal(batch.output_targets, repeat.output_targets)
    assert batch.inputs.shape == (MGS_HORIZON, 24, 3)
    assert batch.output_targets.shape == (MGS_HORIZON, 24, 2)
    assert batch.latent_targets.shape == (MGS_HORIZON, 24, 1)
    assert batch.mask.shape == batch.output_targets.shape
    assert tuple(batch.metadata["delay_support"]) == (50, 399)

    delays = tuple(int(value) for value in batch.metadata["delays"])
    assert min(delays) >= 50
    assert max(delays) <= 399
    for sample, delay in enumerate(delays):
        theta = batch.latent_targets[0, sample, 0]
        cue = torch.stack((torch.cos(theta), torch.sin(theta)))
        go_start = 10 + delay
        transition_start = go_start + 5
        response_start = transition_start + 5
        assert torch.equal(batch.inputs[:5, sample], torch.zeros(5, 3, dtype=torch.float64))
        torch.testing.assert_close(
            batch.inputs[5:10, sample, :2], cue.expand(5, 2), rtol=0, atol=1e-14
        )
        assert torch.count_nonzero(batch.inputs[go_start : go_start + 5, sample, 2]) == 5
        assert torch.equal(
            batch.mask[transition_start:response_start, sample],
            torch.zeros(5, 2, dtype=torch.float64),
        )
        assert torch.equal(batch.mask[:transition_start, sample], torch.ones(transition_start, 2, dtype=torch.float64))
        assert torch.count_nonzero(batch.output_targets[:response_start, sample]) == 0
        torch.testing.assert_close(
            batch.output_targets[response_start:, sample],
            cue.expand(MGS_HORIZON - response_start, 2),
            rtol=0,
            atol=1e-14,
        )


def test_hidden_init_angular_targets_are_post_update():
    batch = angular_integration(5, 991, init_mode="hidden-init", dtype=torch.float64)
    assert batch.inputs.shape == (ANGULAR_HORIZON, 5, 1)
    assert batch.output_targets.shape == (ANGULAR_HORIZON, 5, 2)
    assert batch.latent_targets.shape == (ANGULAR_HORIZON, 5, 1)
    assert torch.equal(batch.mask, torch.ones_like(batch.mask))
    assert batch.metadata["gp_grid"] == "linspace(-1,1,T)"
    assert batch.metadata["gp_length_scale"] == 1.0
    assert batch.metadata["target_indexing"] == "post_velocity_update"

    q0 = torch.tensor(batch.metadata["initial_latents"], dtype=torch.float64)
    expected_q1 = q0 + ANGULAR_DT * batch.inputs[0]
    assert float(_wrapped_delta(batch.latent_targets[0], expected_q1).abs().max()) < 1e-12
    expected_output = torch.cat(
        (torch.cos(batch.latent_targets), torch.sin(batch.latent_targets)), dim=-1
    )
    torch.testing.assert_close(batch.output_targets, expected_output, rtol=0, atol=1e-14)


def test_cue_driven_is_paired_view_of_hidden_init_path():
    hidden = angular_integration(4, 2026, init_mode="hidden-init", dtype=torch.float64)
    cue = angular_integration(4, 2026, init_mode="cue-driven", dtype=torch.float64)
    assert cue.inputs.shape == (ANGULAR_HORIZON + 1, 4, 4)
    assert cue.output_targets.shape == (ANGULAR_HORIZON + 1, 4, 2)
    assert torch.equal(cue.mask[0], torch.zeros_like(cue.mask[0]))
    assert torch.equal(cue.mask[1:], torch.ones_like(cue.mask[1:]))
    assert torch.equal(cue.inputs[1:, :, 2:3], hidden.inputs)
    assert torch.count_nonzero(cue.inputs[1:, :, :2]) == 0
    assert torch.count_nonzero(cue.inputs[1:, :, 3]) == 0
    assert torch.equal(cue.latent_targets[1:], hidden.latent_targets)
    assert torch.equal(cue.output_targets[1:], hidden.output_targets)
    assert cue.metadata["derived_seed"] == hidden.metadata["derived_seed"]
    assert torch.equal(cue.inputs[0, :, 3], torch.ones(4, dtype=torch.float64))
    q0 = cue.latent_targets[0, :, 0]
    expected_cue = torch.stack((torch.cos(q0), torch.sin(q0)), dim=-1)
    torch.testing.assert_close(cue.inputs[0, :, :2], expected_cue, rtol=0, atol=1e-14)


def test_double_angular_integration_shapes_and_independent_gp_coordinates():
    batch = double_angular_integration(8, 55, dtype=torch.float64)
    assert batch.inputs.shape == (ANGULAR_HORIZON, 8, 2)
    assert batch.output_targets.shape == (ANGULAR_HORIZON, 8, 4)
    assert batch.latent_targets.shape == (ANGULAR_HORIZON, 8, 2)
    assert batch.metadata["task_name"] == "double_angular_integration"
    assert not torch.equal(batch.inputs[..., 0], batch.inputs[..., 1])
    q = batch.latent_targets
    expected = torch.stack(
        (torch.cos(q[..., 0]), torch.sin(q[..., 0]), torch.cos(q[..., 1]), torch.sin(q[..., 1])),
        dim=-1,
    )
    torch.testing.assert_close(batch.output_targets, expected, rtol=0, atol=1e-14)


def test_fixed_bank_round_trip_and_sha256_tamper_detection(tmp_path: Path):
    batch = double_angular_integration(
        3,
        81,
        init_mode="cue-driven",
        stream_key=("id", 0),
        dtype=torch.float64,
    )
    path = tmp_path / "double_id_seed000.npz"
    digest = save_fixed_bank(path, batch)
    assert digest == sha256_file(path)
    assert Path(f"{path}.sha256").is_file()
    loaded = load_fixed_bank(path)
    assert torch.equal(loaded.inputs, batch.inputs)
    assert torch.equal(loaded.output_targets, batch.output_targets)
    assert torch.equal(loaded.latent_targets, batch.latent_targets)
    assert torch.equal(loaded.mask, batch.mask)
    assert loaded.metadata == batch.metadata
    with pytest.raises(FileExistsError):
        save_fixed_bank(path, batch)

    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_fixed_bank(path)
