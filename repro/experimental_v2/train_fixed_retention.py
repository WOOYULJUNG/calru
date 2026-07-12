#!/usr/bin/env python3
"""Run Exp88 with every PAN retention theta fixed to a finite value.

The legacy ``force_all_slow(lambda)`` helper creates its intermediate scalar in
float32.  At the RP theta cap, the mathematically valid lambda rounds to 1 and
can produce an infinite theta.  This wrapper preserves the legacy training
loop but replaces that helper with a direct finite-theta assignment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LEGACY_DIR = REPO_ROOT / "repro" / "legacy_code"
ROOT_MARKER = ".calru_experimental_v2_root"

if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

import exp88_manifold_attractor_tasks as legacy  # noqa: E402


def _parse_args():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--fixed-pan-theta", type=float, required=True)
    extra.add_argument("--v2-artifact-root", required=True)
    extra_args, remaining = extra.parse_known_args()
    saved = sys.argv
    try:
        sys.argv = [saved[0], *remaining]
        args = legacy.parse_args()
    finally:
        sys.argv = saved
    args.fixed_pan_theta = float(extra_args.fixed_pan_theta)
    args.v2_artifact_root = str(extra_args.v2_artifact_root)
    return args


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate(args) -> Path:
    root = Path(args.v2_artifact_root).expanduser().resolve()
    marker = root / ROOT_MARKER
    if not marker.is_file() or marker.read_text(encoding="utf-8") != "calru-experimental-v2\n":
        raise RuntimeError(f"invalid v2 artifact root marker: {marker}")
    for raw in (args.out_dir, args.ckpt_dir, args.trace_dir):
        path = Path(raw).expanduser().resolve()
        if not _is_relative_to(path, root):
            raise RuntimeError(f"refusing output outside v2 artifact root: {path}")
    if args.force:
        raise RuntimeError("--force is forbidden in experimental_v2")
    if not args.force_all_slow:
        raise RuntimeError("fixed-theta runner requires --force-all-slow")
    if not -18.0 <= args.fixed_pan_theta <= 18.0:
        raise ValueError("--fixed-pan-theta must lie in the RP clip range [-18,18]")
    return root


@torch.no_grad()
def _force_finite_theta(model, _ignored_lambda: float = 0.0):
    if not hasattr(model, "pan_recs_with_slices"):
        raise RuntimeError("fixed retention requires a model exposing pan_recs_with_slices")
    found = 0
    for rec, _ in model.pan_recs_with_slices():
        theta = getattr(rec, "theta", None)
        if theta is not None:
            theta.fill_(_force_finite_theta.value)
            found += int(theta.numel())
    if found == 0:
        raise RuntimeError("fixed retention found no PAN theta parameters")


_force_finite_theta.value = 0.0


def _configure_device(args) -> None:
    if args.gpu >= 0 and args.device == "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.smoke:
        args.steps = min(args.steps, 2)
        args.batch = min(args.batch, 8)
        args.eval_batch = min(args.eval_batch, 8)
        args.analysis_batch = min(args.analysis_batch, 8)
        args.train_min = 8
        args.train_max = 12
        args.id_horizon = 12
        args.temporal_horizons = [16]
        args.post_holds = [4]
        args.recovery_steps = [0, 4]
        args.repeated_kicks = [1, 3]
        args.repeated_horizon = 8
        args.pan_probe_every = 1
        args.pan_probe_batch = 8
        args.pan_probe_horizon = 12
        args.pan_h_probe = 8
    if args.device == "auto":
        args.device_obj = torch.device("cuda:0" if torch.cuda.is_available() and not args.smoke else "cpu")
    else:
        args.device_obj = torch.device(args.device)


def _add_metadata(args, result_status: dict) -> None:
    if result_status.get("status") != "done":
        return
    path = Path(result_status["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    theta64 = torch.tensor(args.fixed_pan_theta, dtype=torch.float64)
    effective64 = float(torch.sqrt(torch.sigmoid(theta64)).item())
    effective32 = float(torch.sqrt(torch.sigmoid(theta64.float())).item())
    payload.update(
        {
            "fixed_retention_v2": True,
            "fixed_pan_theta": float(args.fixed_pan_theta),
            "requested_uniform_lambda": float(args.all_slow_lambda),
            "effective_uniform_lambda_float64": effective64,
            "effective_uniform_lambda_model_dtype": effective32,
            "finite_theta_assignment": True,
        }
    )
    tmp = path.with_name(f".{path.name}.metadata.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    args = _parse_args()
    _validate(args)
    if args.plot_only:
        raise RuntimeError("--plot-only is not supported by the fixed-retention runner")
    _configure_device(args)
    _force_finite_theta.value = float(args.fixed_pan_theta)
    legacy.force_all_slow = _force_finite_theta
    result = legacy.train_eval_one(args)
    _add_metadata(args, result)
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
