from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "repro"
    / "experimental_v2"
    / "evaluate_ring_transport_v2.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_ring_transport_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_condition_count_matches_protocol():
    assert MODULE.condition_count(32, [-1, 1], [1.0, 2.0], [5, 20], [1, 2, 3], [0, 1, 2]) == 2304


def test_circular_delta_wraps_short_way():
    left = torch.tensor([math_value(179.0)])
    right = torch.tensor([math_value(-179.0)])
    delta = MODULE.circular_delta(left, right)
    assert torch.allclose(delta, torch.tensor([math_value(-2.0)]), atol=1e-6)


def math_value(degrees: float) -> float:
    return degrees * 3.141592653589793 / 180.0


class FullStateModel:
    recurrent_state_size = 3


class PlainStateModel:
    pass


def test_carrier_view_excludes_recomputed_stream():
    state = torch.arange(10).reshape(2, 5)
    assert MODULE.carrier_view(FullStateModel(), state).shape == (2, 3)
    assert torch.equal(MODULE.carrier_view(PlainStateModel(), state), state)


def test_existing_empty_output_is_still_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        result = root / "results" / "x.json"
        checkpoint = root / "checkpoints" / "x.pt"
        result.parent.mkdir()
        checkpoint.parent.mkdir()
        result.write_text("{}", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
        output = root / "already-created"
        output.mkdir()
        records = [(MODULE.DEFAULT_SPECS[0], 0, result, checkpoint)]
        with pytest.raises(FileExistsError):
            MODULE.validate_fresh_output(output, records)


def test_atomic_completion_marker_binds_hash_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "analysis"
        output.mkdir()
        (output / "INCOMPLETE").write_text("in progress\n", encoding="utf-8")
        (output / "ring_transport_v2.csv").write_text("x\n1\n", encoding="utf-8")
        MODULE.atomic_write_json(output / "manifest.json", {"schema_version": 1})
        MODULE.finalize_output(output)
        sums = (output / "SHA256SUMS").read_bytes()
        marker = json.loads((output / "COMPLETE").read_text(encoding="utf-8"))
        assert marker["sha256sums_sha256"] == hashlib.sha256(sums).hexdigest()
        assert not (output / "INCOMPLETE").exists()
