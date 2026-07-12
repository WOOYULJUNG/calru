#!/usr/bin/env python3
"""Train an Exp88 model with explicit long-horizon blank supervision.

This is a horizon-information control, not a compute-matched control.  It uses
the legacy Exp88 task/model implementation without modifying the legacy model
builder.  PAN/CA-LRU variants are rejected unless ``--train-pan-theta`` is
given, because a frozen-retention auxiliary control would not test whether
ordinary gradients can learn the retention coefficients.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LEGACY_DIR = REPO_ROOT / "repro" / "legacy_code"
ROOT_MARKER = ".calru_experimental_v2_root"

if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

import exp88_manifold_attractor_tasks as legacy  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse v2-only flags, then delegate all common flags to Exp88."""

    aux = argparse.ArgumentParser(add_help=False)
    aux.add_argument("--aux-loss-weight", type=float, default=1.0)
    aux.add_argument("--aux-every", type=int, default=100)
    aux.add_argument("--aux-start-step", type=int, default=-1)
    aux.add_argument("--aux-batch", type=int, default=96)
    aux.add_argument("--aux-sequence-horizon", type=int, default=260)
    aux.add_argument("--aux-blank-horizon", type=int, default=500)
    aux.add_argument("--train-pan-theta", action="store_true")
    aux.add_argument("--pan-theta-clip", type=float, default=18.0)
    aux.add_argument("--v2-artifact-root", required=True)
    aux_args, remaining = aux.parse_known_args()

    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0], *remaining]
        args = legacy.parse_args()
    finally:
        sys.argv = saved_argv

    for key, value in vars(aux_args).items():
        setattr(args, key.replace("-", "_"), value)
    return args


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_v2_paths(args: argparse.Namespace) -> Path:
    root = Path(args.v2_artifact_root).expanduser().resolve()
    marker = root / ROOT_MARKER
    if not marker.is_file() or marker.read_text(encoding="utf-8") != "calru-experimental-v2\n":
        raise RuntimeError(
            f"refusing to write without a valid v2 artifact marker: {marker}. "
            "Run this trainer through launch_p0.py."
        )
    for raw in (args.out_dir, args.ckpt_dir, args.trace_dir):
        path = Path(raw).expanduser().resolve()
        if not _is_relative_to(path, root):
            raise RuntimeError(f"refusing non-v2 output path outside {root}: {path}")
    if args.force:
        raise RuntimeError("--force is forbidden in experimental_v2")
    return root


def _configure_device(args: argparse.Namespace) -> None:
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
        args.aux_every = 1
        args.aux_start_step = 1
        args.aux_batch = min(args.aux_batch, 4)
        args.aux_sequence_horizon = 8
        args.aux_blank_horizon = 4
    if args.device == "auto":
        args.device_obj = torch.device("cuda:0" if torch.cuda.is_available() and not args.smoke else "cpu")
    else:
        args.device_obj = torch.device(args.device)


def _pan_theta_parameters(model) -> list[torch.nn.Parameter]:
    if not hasattr(model, "pan_recs_with_slices"):
        return []
    params = []
    for rec, _ in model.pan_recs_with_slices():
        theta = getattr(rec, "theta", None)
        if theta is not None:
            params.append(theta)
    return params


def _run_state_with_grad(model, x_seq: torch.Tensor) -> torch.Tensor:
    state = model.init_state(x_seq.shape[1], x_seq.device)
    for x_t in x_seq:
        state = model.step(x_t, state)
    return state


def _roll_blank_with_grad(model, state: torch.Tensor, steps: int) -> torch.Tensor:
    blank = torch.zeros(state.shape[0], model.input_dim, device=state.device, dtype=state.dtype)
    cur = state
    for _ in range(int(steps)):
        cur = model.step(blank, cur)
    return cur


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if path.exists():
        tmp.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to overwrite {path}")
    os.replace(tmp, path)


def _atomic_torch_save(path: Path, state_dict: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(state_dict, tmp)
    if path.exists():
        tmp.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to overwrite {path}")
    os.replace(tmp, path)


def _atomic_trace(path: Path, **arrays) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    if path.exists():
        tmp.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to overwrite {path}")
    os.replace(tmp, path)


def train_eval_aux(args: argparse.Namespace) -> dict:
    _validate_v2_paths(args)
    task = str(args.task)
    seed = int(args.seed)
    variant = legacy.normalize_model_variant(args.model)
    is_pan = legacy.is_pan_variant(variant)

    if is_pan and not args.train_pan_theta:
        raise RuntimeError(
            "PAN/CA-LRU auxiliary controls must pass --train-pan-theta; "
            "a frozen-retention auxiliary is not a horizon-learning control"
        )
    if args.force_all_slow:
        raise RuntimeError("auxiliary-gradient controls cannot also use --force-all-slow")
    if args.aux_loss_weight <= 0:
        raise ValueError("--aux-loss-weight must be positive")
    if args.aux_every <= 0 or args.aux_batch <= 0:
        raise ValueError("aux cadence and batch must be positive")
    if args.aux_sequence_horizon <= 0 or args.aux_blank_horizon <= 0:
        raise ValueError("auxiliary horizons must be positive")

    legacy.configure_deterministic_runtime(bool(args.deterministic_training))
    legacy.set_seed(seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.trace_dir).mkdir(parents=True, exist_ok=True)

    input_dim, output_dim = legacy.task_io_dims(task)
    rank_for_model = legacy.model_rank_for_task(task)
    tag = args.tag or legacy.slugify(variant)
    json_path = Path(args.out_dir) / f"{task}_{tag}_seed{seed}.json"
    ckpt_path = Path(args.ckpt_dir) / f"exp88_{task}_{tag}_seed{seed}.pt"
    trace_path = Path(args.trace_dir) / f"{task}_{tag}_seed{seed}_aux_trace.npz"
    existing = [path for path in (json_path, ckpt_path, trace_path) if path.exists()]
    if len(existing) == 3:
        return {"status": "skipped", "path": str(json_path)}
    if existing:
        raise RuntimeError(
            "partial artifact set detected; refusing to overwrite: " + ", ".join(map(str, existing))
        )

    model = legacy.build_model_variant(
        variant=variant,
        input_dim=input_dim,
        output_dim=output_dim,
        rank=rank_for_model,
        d_model=args.d_model,
        rec_dim=args.rec_dim,
        layers=args.layers,
        dropout=args.dropout,
        plru_tau=args.plru_tau,
        plru_c=args.plru_c,
        pan_lambda_min=args.pan_lambda_min,
        pan_lambda_max=args.pan_lambda_max,
        rank_matched_lambda_high=args.rank_matched_lambda_high,
        rank_matched_lambda_low=args.rank_matched_lambda_low,
    ).to(args.device_obj)
    legacy.apply_slow_lambda_init(
        model,
        mode=args.slow_lambda_init_mode,
        lambda_min=args.slow_lambda_min,
        lambda_max=args.slow_lambda_max,
        lambda_fixed=args.slow_lambda_fixed,
        lambda_mean=args.slow_lambda_mean,
        lambda_std=args.slow_lambda_std,
        lambda_low_mean=args.slow_lambda_low_mean,
        lambda_high_mean=args.slow_lambda_high_mean,
        lambda_high_prob=args.slow_lambda_high_prob,
        shuffle=args.slow_lambda_shuffle,
        seed=args.slow_lambda_init_seed,
    )

    pan_thetas = _pan_theta_parameters(model)
    if args.train_pan_theta:
        if not pan_thetas:
            raise RuntimeError("--train-pan-theta was requested, but the model exposes no PAN theta")
        for theta in pan_thetas:
            theta.requires_grad_(True)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    default_start = int(round(int(args.steps) * float(args.pan_warmup_frac))) + 1
    aux_start_step = default_start if int(args.aux_start_step) < 0 else int(args.aux_start_step)

    total_losses: list[float] = []
    task_losses: list[float] = []
    aux_steps: list[int] = []
    aux_losses: list[float] = []
    aux_forward_seconds = 0.0
    t0 = time.time()
    model.train()

    for step in range(1, int(args.steps) + 1):
        if int(args.train_data_seed_base) >= 0:
            train_seed = legacy.deterministic_experiment_seed(
                int(args.train_data_seed_base), task, seed, "train_batch", step
            )
            with legacy.isolated_experiment_rng(train_seed):
                horizon = random.randint(int(args.train_min), int(args.train_max))
                x, y, _, _ = legacy.make_task_batch(
                    task, int(args.batch), horizon, args.device_obj, args
                )
        else:
            horizon = random.randint(int(args.train_min), int(args.train_max))
            x, y, _, _ = legacy.make_task_batch(task, int(args.batch), horizon, args.device_obj, args)
        out = model(x)
        task_loss = F.mse_loss(out, y)
        loss = task_loss

        # Match the RP cadence (e.g. steps 3100, 3200, ...) after warm-up,
        # rather than shifting the cadence by the start-step offset.
        use_aux = step >= aux_start_step and step % int(args.aux_every) == 0
        aux_value = math.nan
        if use_aux:
            _sync_if_cuda(args.device_obj)
            aux_t0 = time.time()
            if int(args.probe_seed_base) >= 0:
                probe_seed = legacy.deterministic_experiment_seed(
                    int(args.probe_seed_base), task, seed, "long_horizon_probe", step
                )
                with legacy.isolated_experiment_rng(probe_seed):
                    probe_x, _, probe_target, _ = legacy.make_task_batch(
                        task,
                        int(args.aux_batch),
                        int(args.aux_sequence_horizon),
                        args.device_obj,
                        args,
                        profile="train",
                    )
            else:
                probe_x, _, probe_target, _ = legacy.make_task_batch(
                    task,
                    int(args.aux_batch),
                    int(args.aux_sequence_horizon),
                    args.device_obj,
                    args,
                    profile="train",
                )
            probe_state = _run_state_with_grad(model, probe_x)
            retained_state = _roll_blank_with_grad(model, probe_state, int(args.aux_blank_horizon))
            aux_pred = model.decode(retained_state)
            aux_loss = F.mse_loss(aux_pred, probe_target)
            loss = loss + float(args.aux_loss_weight) * aux_loss
            aux_value = float(aux_loss.detach().item())
            _sync_if_cuda(args.device_obj)
            aux_forward_seconds += time.time() - aux_t0

        reg = model.regularization_loss() if hasattr(model, "regularization_loss") else None
        if reg is not None:
            loss = loss + reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        opt.step()
        if pan_thetas:
            with torch.no_grad():
                for theta in pan_thetas:
                    theta.clamp_(-float(args.pan_theta_clip), float(args.pan_theta_clip))

        total_losses.append(float(loss.detach().item()))
        task_losses.append(float(task_loss.detach().item()))
        if use_aux:
            aux_steps.append(step)
            aux_losses.append(aux_value)

        if step == 1 or step % max(1, int(args.steps) // 4) == 0:
            lam_msg = ""
            if hasattr(model, "lam_mag"):
                lam = model.lam_mag().detach()
                lam_msg = f" sum_lambda={lam.sum().item():.1f} n>.99={(lam > .99).sum().item()}"
            aux_msg = "" if not use_aux else f" auxH{args.aux_blank_horizon}={aux_value:.5f}"
            print(
                f"[{task} {variant:14s} seed={seed}] {step:5d}/{args.steps} "
                f"task={task_loss.item():.5f}{aux_msg}{lam_msg}",
                flush=True,
            )

    evaluation_seed = ""
    if int(args.eval_seed_base) >= 0:
        evaluation_seed = legacy.deterministic_experiment_seed(
            int(args.eval_seed_base), task, 0, "fixed_evaluation", 0
        )
        with legacy.isolated_experiment_rng(evaluation_seed):
            metrics = legacy.evaluate_model(model, task, args)
    else:
        metrics = legacy.evaluate_model(model, task, args)
    elapsed = time.time() - t0
    result = {
        "task": task,
        "geometry": legacy.task_geometry(task).name,
        "q_dim": legacy.task_geometry(task).q_dim,
        "output_dim": output_dim,
        "input_dim": input_dim,
        "rank_for_model": rank_for_model,
        "model": variant,
        "model_display": legacy.MODEL_DISPLAY_NAMES.get(variant, variant),
        "tag": tag,
        "seed": seed,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "params_total": sum(p.numel() for p in model.parameters()),
        "train_steps": int(args.steps),
        "train_min": int(args.train_min),
        "train_max": int(args.train_max),
        "id_horizon": int(args.id_horizon),
        "temporal_horizons": list(map(int, args.temporal_horizons)),
        "post_holds": list(map(int, args.post_holds)),
        "normal_radii": list(map(float, args.normal_radii)),
        "repeated_kicks": list(map(int, args.repeated_kicks)),
        "velocity_scales": list(map(float, args.velocity_scales)),
        "train_loss_final": float(np.mean(total_losses[-min(50, len(total_losses)) :])),
        "train_task_loss_final": float(np.mean(task_losses[-min(50, len(task_losses)) :])),
        "seconds": elapsed,
        "d_model": int(args.d_model),
        "rec_dim": int(args.rec_dim),
        "layers": int(args.layers),
        "lr": float(args.lr),
        "train_data_seed_base": int(args.train_data_seed_base),
        "probe_seed_base": int(args.probe_seed_base),
        "eval_seed_base": int(args.eval_seed_base),
        "evaluation_seed": evaluation_seed,
        "deterministic_rng_streams": bool(
            int(args.train_data_seed_base) >= 0
            and int(args.probe_seed_base) >= 0
            and int(args.eval_seed_base) >= 0
        ),
        "deterministic_training": bool(args.deterministic_training),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "training_control": "horizon_information_auxiliary",
        "aux_is_compute_matched": False,
        "aux_loss_weight": float(args.aux_loss_weight),
        "aux_every": int(args.aux_every),
        "aux_start_step": int(aux_start_step),
        "aux_batch": int(args.aux_batch),
        "aux_sequence_horizon": int(args.aux_sequence_horizon),
        "aux_blank_horizon": int(args.aux_blank_horizon),
        "aux_updates": len(aux_steps),
        "aux_examples": int(len(aux_steps) * int(args.aux_batch)),
        "aux_recurrent_transitions": int(
            len(aux_steps)
            * int(args.aux_batch)
            * (int(args.aux_sequence_horizon) + 1 + int(args.aux_blank_horizon))
        ),
        "aux_forward_graph_seconds": float(aux_forward_seconds),
        "aux_loss_mean": float(np.mean(aux_losses)) if aux_losses else math.nan,
        "aux_loss_final": float(aux_losses[-1]) if aux_losses else math.nan,
        "train_pan_theta": bool(args.train_pan_theta),
        "pan_theta_clip": float(args.pan_theta_clip) if args.train_pan_theta else "",
        "rp_enabled": False,
        "horizon_information_note": (
            "Matches the RP probe target horizon/cadence, but not RP coordinate-ablation compute."
        ),
    }
    result.update(metrics)

    _atomic_torch_save(ckpt_path, model.state_dict())
    _atomic_trace(
        trace_path,
        train_total_loss=np.asarray(total_losses, dtype=np.float32),
        train_task_loss=np.asarray(task_losses, dtype=np.float32),
        aux_steps=np.asarray(aux_steps, dtype=np.int64),
        aux_loss=np.asarray(aux_losses, dtype=np.float32),
    )
    _atomic_json(json_path, result)
    return {"status": "done", "path": str(json_path)}


def main() -> int:
    args = _parse_args()
    if args.plot_only:
        raise RuntimeError("--plot-only is not supported by the auxiliary trainer")
    _configure_device(args)
    print(train_eval_aux(args), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
