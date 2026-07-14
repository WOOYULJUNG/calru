from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from repro.sagodi_protocol.artifacts import verify_completion_receipt
from repro.sagodi_protocol.config import DEFAULT_PROTOCOL_PATH
from repro.sagodi_protocol.phase0 import _phase0_width_override, run_phase0


def test_phase0_smoke_preserves_architecture_locked_sagodi_gru_widths() -> None:
    assert _phase0_width_override("ca_lru", smoke=True) == 8
    assert _phase0_width_override("no_rp", smoke=True) == 8
    assert _phase0_width_override("gru_sagodi_width96", smoke=True) is None
    assert _phase0_width_override("gru_sagodi_param135", smoke=True) is None
    assert _phase0_width_override("gru_sagodi_width96", smoke=False) is None


def test_phase0_materializes_required_tensor_trace_and_gate(tmp_path: Path) -> None:
    output = run_phase0(DEFAULT_PROTOCOL_PATH, tmp_path / "phase0", smoke=True)
    gate = json.loads((output / "phase0_gate.json").read_text())
    assert gate["passed"] is True
    for model_name, model_gate in gate["models"].items():
        assert model_gate["passed"] is True
        assert all(model_gate["required_checks"].values())
        model_dir = output / f"model={model_name}"
        metadata = json.loads((model_dir / "blank_map_trace_metadata.json").read_text())
        assert metadata["steps"] == 20
        assert metadata["external_input_exactly_zero"] is True
        assert metadata["external_reset_present"] is False
        assert metadata["nearest_manifold_distance_status"] == "not_available_before_training"
        with np.load(model_dir / "blank_map_trace.npz", allow_pickle=False) as trace:
            assert trace["actual_input"].shape[0] == 20
            assert trace["pre_primary"].shape == trace["post_primary"].shape
            assert trace["pre_carrier"].shape == trace["post_carrier"].shape
            assert trace["decoder_output"].shape[0] == 20
            assert trace["f0_residual"].shape[0] == 20
            assert np.count_nonzero(trace["actual_input"]) == 0
            assert np.isnan(trace["nearest_manifold_distance"]).all()

    valid, reason = verify_completion_receipt(
        output / "completion_receipt.json",
        expected_job_id="phase0_state_audit",
    )
    assert valid, reason
