"""Exp74: discrete attractor tasks on the Exp71 full-block scaffold.

Tasks:
  kway_hold: cue one of K classes once, then hold the class under zero input.
  flipflop: N-bit set/reset pulses, then hold and transition between bit states.
"""

import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from exp71_pan_block_pulse_hold import (
    MODEL_DISPLAY_NAMES,
    apply_pan_update,
    apply_slow_lambda_init,
    build_model_variant,
    ensure_dir,
    is_pan_variant,
    normalize_model_variant,
    set_seed,
    slugify,
)


TASKS = ("kway_hold", "flipflop")


def pca_stats(states):
    x = states - states.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    var = s ** 2
    total = float(var.sum()) + 1e-12
    ratio = var / total
    csum = np.cumsum(ratio)
    return {
        "latent_pca_pr": float(total ** 2 / (np.square(var).sum() + 1e-12)),
        "latent_pca_dim90": int(np.searchsorted(csum, 0.90) + 1),
        "latent_pca_dim95": int(np.searchsorted(csum, 0.95) + 1),
        "latent_pca_var1": float(ratio[0]) if ratio.size else 0.0,
        "latent_pca_var2": float(ratio[:2].sum()) if ratio.size >= 2 else float(ratio.sum()),
    }


def task_io_dims(task, k, bits):
    if task == "kway_hold":
        return int(k) + 1, int(k)
    if task == "flipflop":
        return int(bits), int(bits)
    raise ValueError(task)


def model_rank_for_task(task, k, bits):
    if task == "kway_hold":
        return int(k)
    if task == "flipflop":
        return 2 ** int(bits)
    raise ValueError(task)


def make_kway_batch(batch, k, horizon, device):
    labels = torch.randint(0, int(k), (batch,), device=device)
    x = torch.zeros(horizon + 1, batch, int(k) + 1, device=device)
    x[0, torch.arange(batch, device=device), labels] = 1.0
    x[0, :, int(k)] = 1.0
    y = labels.unsqueeze(0).expand(horizon + 1, batch).clone()
    return x, y, labels, {"labels": labels}


def make_flipflop_batch(batch, bits, horizon, device, update_prob):
    bits = int(bits)
    state = torch.randint(0, 2, (batch, bits), device=device, dtype=torch.float32)
    x = torch.zeros(horizon + 1, batch, bits, device=device)
    y = torch.zeros(horizon + 1, batch, bits, device=device)
    x[0] = state * 2.0 - 1.0
    y[0] = state
    updates = torch.zeros(horizon, batch, bits, device=device)
    for t in range(1, horizon + 1):
        mask = torch.rand(batch, bits, device=device) < float(update_prob)
        new_bits = torch.randint(0, 2, (batch, bits), device=device, dtype=torch.float32)
        pulse = new_bits * 2.0 - 1.0
        x[t] = torch.where(mask, pulse, torch.zeros_like(pulse))
        state = torch.where(mask, new_bits, state)
        y[t] = state
        updates[t - 1] = mask.float()
    return x, y, y[-1].clone(), {"updates": updates}


def make_task_batch(task, batch, k, bits, horizon, device, args):
    if task == "kway_hold":
        return make_kway_batch(batch, k, horizon, device)
    if task == "flipflop":
        return make_flipflop_batch(batch, bits, horizon, device, args.flip_update_prob)
    raise ValueError(task)


def discrete_loss(task, logits, target):
    if task == "kway_hold":
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
    if task == "flipflop":
        return F.binary_cross_entropy_with_logits(logits, target)
    raise ValueError(task)


def final_loss(task, logits, target):
    if task == "kway_hold":
        return F.cross_entropy(logits, target)
    if task == "flipflop":
        return F.binary_cross_entropy_with_logits(logits, target)
    raise ValueError(task)


def per_coord_energy(task, pred, target):
    hidden, batch, out_dim = pred.shape
    if task == "kway_hold":
        flat_target = target.unsqueeze(0).expand(hidden, batch).reshape(-1)
        loss = F.cross_entropy(pred.reshape(hidden * batch, out_dim), flat_target, reduction="none")
        return loss.reshape(hidden, batch).mean(dim=1)
    if task == "flipflop":
        tgt = target.unsqueeze(0).expand_as(pred)
        loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
        return loss.mean(dim=(1, 2))
    raise ValueError(task)


def decode_class_from_bits(logits):
    bits = (logits > 0).long()
    weights = (2 ** torch.arange(bits.shape[-1], device=bits.device)).long()
    return (bits * weights).sum(dim=-1)


def bits_from_classes(classes, bits, device):
    weights = (2 ** torch.arange(int(bits), device=device)).long()
    return ((classes.long().unsqueeze(1) & weights.unsqueeze(0)) > 0).float()


def class_from_bits(bits_tensor):
    bits_long = bits_tensor.long()
    weights = (2 ** torch.arange(bits_long.shape[-1], device=bits_long.device)).long()
    return (bits_long * weights).sum(dim=-1)


def add_accuracy_metrics(metrics, prefix, task, logits, target):
    if task == "kway_hold":
        pred = logits.argmax(dim=-1)
        metrics[f"{prefix}_acc"] = float((pred == target).float().mean().item())
        metrics[f"{prefix}_ce"] = float(F.cross_entropy(logits, target).item())
    else:
        pred_bits = (logits > 0).float()
        correct_bits = (pred_bits == target).float()
        metrics[f"{prefix}_bit_acc"] = float(correct_bits.mean().item())
        metrics[f"{prefix}_full_acc"] = float(correct_bits.all(dim=-1).float().mean().item())
        metrics[f"{prefix}_bce"] = float(F.binary_cross_entropy_with_logits(logits, target).item())


def confusion_matrix(task, logits, target, n_classes):
    if task == "kway_hold":
        pred = logits.argmax(dim=-1)
        true = target.long()
    else:
        pred = decode_class_from_bits(logits)
        true = class_from_bits(target.long())
    mat = torch.bincount(true * int(n_classes) + pred, minlength=int(n_classes) * int(n_classes))
    return mat.reshape(int(n_classes), int(n_classes)).cpu().tolist()


@torch.no_grad()
def run_state(model, x_seq):
    state = model.init_state(x_seq.shape[1], x_seq.device)
    for x_t in x_seq:
        state = model.step(x_t, state)
    return state


@torch.no_grad()
def roll_blank(model, state, steps):
    blank = torch.zeros(state.shape[0], model.input_dim, device=state.device)
    cur = state
    for _ in range(int(steps)):
        cur = model.step(blank, cur)
    return cur


def normalized_noise_like(state, radius):
    noise = torch.randn_like(state)
    noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-8)
    return noise * float(radius)


@torch.no_grad()
def compute_pan_scores_for_task(model, task, x_seq, target, h_probe):
    state = run_state(model, x_seq)
    clean_final = roll_blank(model, state, h_probe)
    clean_pred = model.decode(clean_final)
    clean_energy = final_loss(task, clean_pred, target)

    score_payload = []
    batch = state.shape[0]
    total_state = state.shape[1]
    for rec, rec_slice in model.pan_recs_with_slices():
        hidden = rec_slice.stop - rec_slice.start
        ablated = state.unsqueeze(0).expand(hidden, batch, total_state).clone()
        idx = torch.arange(hidden, device=state.device)
        ablated[idx, :, rec_slice.start + idx] = 0.0
        final = roll_blank(model, ablated.reshape(hidden * batch, total_state), h_probe)
        pred = model.decode(final).reshape(hidden, batch, model.output_dim)
        energy = per_coord_energy(task, pred, target)
        score_payload.append((rec, (energy - clean_energy).detach()))
    return score_payload, float(clean_energy.item())


@torch.no_grad()
def state_from_kway(model, labels, k, horizon):
    batch = labels.numel()
    x = torch.zeros(horizon + 1, batch, int(k) + 1, device=labels.device)
    x[0, torch.arange(batch, device=labels.device), labels.long()] = 1.0
    x[0, :, int(k)] = 1.0
    state = run_state(model, x)
    return state, labels.long()


@torch.no_grad()
def state_from_flip_bits(model, bits_tensor, horizon):
    batch, bits = bits_tensor.shape
    x = torch.zeros(horizon + 1, batch, bits, device=bits_tensor.device)
    x[0] = bits_tensor.float() * 2.0 - 1.0
    state = run_state(model, x)
    return state, bits_tensor.float()


@torch.no_grad()
def evaluate_kway_geometry(model, args, device):
    metrics = {}
    k = int(args.k)
    labels = torch.arange(k, device=device)
    proto_state, proto_labels = state_from_kway(model, labels, k, int(args.analysis_horizon))
    proto_next = roll_blank(model, proto_state, 1)
    metrics["prototype_residual_mean"] = float((proto_next - proto_state).norm(dim=-1).mean().item())
    dmat = torch.cdist(proto_state, proto_state)
    off = dmat[~torch.eye(k, dtype=torch.bool, device=device)]
    metrics["between_proto_mean"] = float(off.mean().item())
    metrics["between_proto_min"] = float(off.min().item())

    reps = max(1, int(args.analysis_batch) // k)
    labels_rep = labels.repeat_interleave(reps)
    state, target = state_from_kway(model, labels_rep, k, int(args.analysis_horizon))
    for step in args.recovery_steps:
        perturbed = state + normalized_noise_like(state, args.normal_radius)
        pred = model.decode(roll_blank(model, perturbed, int(step))).argmax(dim=-1)
        metrics[f"recovery_R{step}_acc"] = float((pred == target).float().mean().item())

    for radius in args.basin_radii:
        perturbed = state + normalized_noise_like(state, radius)
        pred = model.decode(roll_blank(model, perturbed, int(args.post_hold))).argmax(dim=-1)
        metrics[f"basin_r{radius:g}_acc"] = float((pred == target).float().mean().item())
        rolled = roll_blank(model, perturbed, int(args.post_hold))
        proto_for_sample = proto_state[target]
        metrics[f"within_r{radius:g}_dist"] = float((rolled - proto_for_sample).norm(dim=-1).mean().item())

    states_np = state.detach().cpu().numpy()
    metrics.update(pca_stats(states_np))
    return metrics


@torch.no_grad()
def evaluate_flip_geometry(model, args, device):
    metrics = {}
    bits = int(args.bits)
    n_classes = 2 ** bits
    classes = torch.arange(n_classes, device=device)
    bit_table = bits_from_classes(classes, bits, device)
    proto_state, proto_bits = state_from_flip_bits(model, bit_table, int(args.analysis_horizon))
    proto_next = roll_blank(model, proto_state, 1)
    metrics["prototype_residual_mean"] = float((proto_next - proto_state).norm(dim=-1).mean().item())
    dmat = torch.cdist(proto_state, proto_state)
    off = dmat[~torch.eye(n_classes, dtype=torch.bool, device=device)]
    metrics["between_proto_mean"] = float(off.mean().item())
    metrics["between_proto_min"] = float(off.min().item())

    reps = max(1, int(args.analysis_batch) // n_classes)
    bits_rep = bit_table.repeat_interleave(reps, dim=0)
    state, target = state_from_flip_bits(model, bits_rep, int(args.analysis_horizon))
    for step in args.recovery_steps:
        perturbed = state + normalized_noise_like(state, args.normal_radius)
        logits = model.decode(roll_blank(model, perturbed, int(step)))
        add_accuracy_metrics(metrics, f"recovery_R{step}", "flipflop", logits, target)

    for radius in args.basin_radii:
        perturbed = state + normalized_noise_like(state, radius)
        rolled = roll_blank(model, perturbed, int(args.post_hold))
        logits = model.decode(rolled)
        add_accuracy_metrics(metrics, f"basin_r{radius:g}", "flipflop", logits, target)
        target_class = class_from_bits(target.long())
        proto_for_sample = proto_state[target_class]
        metrics[f"within_r{radius:g}_dist"] = float((rolled - proto_for_sample).norm(dim=-1).mean().item())

    trans_total = 0
    trans_immediate = 0
    trans_after = 0
    for bit_idx in range(bits):
        next_bits = bit_table.clone()
        next_bits[:, bit_idx] = 1.0 - next_bits[:, bit_idx]
        pulse = torch.zeros(n_classes, bits, device=device)
        pulse[:, bit_idx] = next_bits[:, bit_idx] * 2.0 - 1.0
        moved = model.step(pulse, proto_state)
        pred_now = (model.decode(moved) > 0).float()
        pred_after = (model.decode(roll_blank(model, moved, int(args.transition_hold))) > 0).float()
        trans_total += n_classes
        trans_immediate += int((pred_now == next_bits).all(dim=-1).sum().item())
        trans_after += int((pred_after == next_bits).all(dim=-1).sum().item())
    metrics["transition_immediate_full_acc"] = float(trans_immediate / max(1, trans_total))
    metrics["transition_after_full_acc"] = float(trans_after / max(1, trans_total))

    states_np = proto_state.detach().cpu().numpy()
    metrics.update(pca_stats(states_np))
    return metrics


@torch.no_grad()
def evaluate_model(model, task, args, device):
    metrics = {}
    model.eval()
    n_classes = int(args.k) if task == "kway_hold" else 2 ** int(args.bits)

    for horizon in args.eval_horizons:
        x, y, target, _ = make_task_batch(task, int(args.eval_batch), int(args.k), int(args.bits), int(horizon), device, args)
        out = model(x)
        add_accuracy_metrics(metrics, f"H{horizon}", task, out[-1], target)
        metrics[f"H{horizon}_loss"] = float(discrete_loss(task, out, y).item())
        if int(horizon) == max(args.eval_horizons):
            metrics[f"H{horizon}_confusion"] = confusion_matrix(task, out[-1], target, n_classes)

    x, y, target, _ = make_task_batch(task, int(args.analysis_batch), int(args.k), int(args.bits), int(args.analysis_horizon), device, args)
    state = run_state(model, x)
    post = roll_blank(model, state, int(args.post_hold))
    add_accuracy_metrics(metrics, f"postH{args.post_hold}", task, model.decode(post), target)

    if task == "kway_hold":
        metrics.update(evaluate_kway_geometry(model, args, device))
    else:
        metrics.update(evaluate_flip_geometry(model, args, device))

    if hasattr(model, "lam_mag"):
        lam = model.lam_mag().detach().cpu().numpy()
        metrics["lambda_sum"] = float(lam.sum())
        metrics["lambda_gt_0p9"] = int((lam > 0.9).sum())
        metrics["lambda_gt_0p95"] = int((lam > 0.95).sum())
        metrics["lambda_gt_0p99"] = int((lam > 0.99).sum())
        metrics["lambda_mean"] = float(lam.mean())
        metrics["lambda_max"] = float(lam.max())
        metrics["lambda_min"] = float(lam.min())

    if hasattr(model, "pan_recs_with_slices") and model.pan_recs_with_slices():
        x, _, target, _ = make_task_batch(
            task,
            int(args.pan_probe_batch),
            int(args.k),
            int(args.bits),
            int(args.pan_probe_horizon),
            device,
            args,
        )
        score_payload, pan_probe_loss = compute_pan_scores_for_task(model, task, x, target, int(args.pan_h_probe))
        scores = torch.cat([score.detach().flatten() for _, score in score_payload]).cpu().numpy()
        pan_lams = torch.cat([rec.lam_mag().detach().flatten() for rec, _ in score_payload]).cpu().numpy()
        corr = 0.0
        if scores.size and np.std(scores) > 1e-12 and np.std(pan_lams) > 1e-12:
            corr = float(np.corrcoef(scores, pan_lams)[0, 1])
        metrics.update({
            "pan_eval_probe_loss": pan_probe_loss,
            "pan_score_mean": float(scores.mean()) if scores.size else 0.0,
            "pan_score_max": float(scores.max()) if scores.size else 0.0,
            "pan_score_pos_sum": float(np.clip(scores, 0.0, None).sum()),
            "pan_score_lambda_corr": corr,
        })
    return metrics


def train_eval_one(args):
    task = str(args.task)
    seed = int(args.seed)
    variant = normalize_model_variant(args.model)
    rank_for_model = model_rank_for_task(task, int(args.k), int(args.bits))

    set_seed(seed)
    ensure_dir(args.out_dir)
    ensure_dir(args.ckpt_dir)
    ensure_dir(args.trace_dir)
    input_dim, output_dim = task_io_dims(task, int(args.k), int(args.bits))

    tag = args.tag or slugify(variant)
    size_name = f"k{args.k}" if task == "kway_hold" else f"n{args.bits}"
    json_path = os.path.join(args.out_dir, f"{task}_{size_name}_{tag}_seed{seed}.json")
    ckpt_path = os.path.join(args.ckpt_dir, f"exp74_{task}_{size_name}_{tag}_seed{seed}.pt")
    if os.path.exists(json_path) and os.path.exists(ckpt_path) and not args.force:
        return {"status": "skipped", "path": json_path}

    model = build_model_variant(
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

    apply_slow_lambda_init(
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

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-5)
    warmup_steps = int(round(args.steps * args.pan_warmup_frac)) if is_pan_variant(variant) else 0
    plru_warmup = int(round(args.steps * args.plru_warmup_frac)) if variant in ("P-LRU unit", "P-LRU-Block") else 0

    lambda_trace = None
    if args.log_lambda_trajectory and hasattr(model, "lam_mag"):
        lam0 = model.lam_mag().detach().cpu().numpy().astype(np.float32)
        lambda_trace = {
            "steps": [0],
            "lambdas": [lam0],
            "lambda_gt_0p9": [int((lam0 > 0.9).sum())],
            "lambda_gt_0p95": [int((lam0 > 0.95).sum())],
            "lambda_gt_0p99": [int((lam0 > 0.99).sum())],
            "pan_score_mean": [float("nan")],
            "pan_score_max": [float("nan")],
            "pan_probe_loss": [float("nan")],
        }

    losses, task_losses, reg_losses = [], [], []
    pan_probe_loss = float("nan")
    pan_last_score_mean = float("nan")
    pan_last_score_max = float("nan")
    t0 = time.time()
    model.train()

    for step in range(1, int(args.steps) + 1):
        horizon = random.randint(int(args.train_min), int(args.train_max))
        x, y, target, _ = make_task_batch(task, int(args.batch), int(args.k), int(args.bits), horizon, args.device_obj, args)
        out = model(x)
        task_loss = discrete_loss(task, out, y)
        reg = model.regularization_loss() if hasattr(model, "regularization_loss") else None
        reg_value = 0.0
        loss = task_loss
        if reg is not None:
            reg_weight = 1.0
            if variant in ("P-LRU unit", "P-LRU-Block"):
                reg_weight = 1.0 if plru_warmup == 0 else min(1.0, step / plru_warmup)
            loss = loss + reg_weight * reg
            reg_value = float(reg.item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if is_pan_variant(variant) and step % int(args.pan_probe_every) == 0:
            model.eval()
            probe_x, _, probe_target, _ = make_task_batch(
                task,
                int(args.pan_probe_batch),
                int(args.k),
                int(args.bits),
                int(args.pan_probe_horizon),
                args.device_obj,
                args,
            )
            score_payload, pan_probe_loss = compute_pan_scores_for_task(
                model,
                task,
                probe_x,
                probe_target,
                int(args.pan_h_probe),
            )
            all_scores = torch.cat([scores.detach().flatten() for _, scores in score_payload])
            pan_last_score_mean = float(all_scores.mean().item())
            pan_last_score_max = float(all_scores.max().item())
            if step > warmup_steps:
                apply_pan_update(score_payload, float(args.pan_eta_lambda), float(args.pan_score_eps))
            model.train()

        losses.append(float(loss.item()))
        task_losses.append(float(task_loss.item()))
        reg_losses.append(reg_value)

        if step == 1 or step % max(1, int(args.steps) // 4) == 0:
            lam_msg = ""
            if hasattr(model, "lam_mag"):
                lam = model.lam_mag().detach()
                lam_msg = f" sum_lambda={lam.sum().item():.1f} n>.99={(lam > .99).sum().item()}"
            print(
                f"[{task} {variant:22s} seed={seed}] "
                f"{step:5d}/{args.steps} task={task_loss.item():.5f} reg={reg_value:.4f}{lam_msg}",
                flush=True,
            )

        if lambda_trace is not None and (step == 1 or step == args.steps or step % int(args.lambda_log_every) == 0):
            lam = model.lam_mag().detach().cpu().numpy().astype(np.float32)
            lambda_trace["steps"].append(step)
            lambda_trace["lambdas"].append(lam)
            lambda_trace["lambda_gt_0p9"].append(int((lam > 0.9).sum()))
            lambda_trace["lambda_gt_0p95"].append(int((lam > 0.95).sum()))
            lambda_trace["lambda_gt_0p99"].append(int((lam > 0.99).sum()))
            lambda_trace["pan_score_mean"].append(float(pan_last_score_mean))
            lambda_trace["pan_score_max"].append(float(pan_last_score_max))
            lambda_trace["pan_probe_loss"].append(float(pan_probe_loss))

    metrics = evaluate_model(model, task, args, args.device_obj)
    result = {
        "task": task,
        "k": int(args.k) if task == "kway_hold" else "",
        "bits": int(args.bits) if task == "flipflop" else "",
        "rank_for_model": rank_for_model,
        "model": variant,
        "model_display": MODEL_DISPLAY_NAMES.get(variant, variant),
        "tag": tag,
        "seed": seed,
        "device": str(args.device_obj),
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_steps": int(args.steps),
        "train_min": int(args.train_min),
        "train_max": int(args.train_max),
        "train_loss_final": float(np.mean(losses[-min(50, len(losses)):])),
        "train_task_loss_final": float(np.mean(task_losses[-min(50, len(task_losses)):])),
        "train_reg_loss_final": float(np.mean(reg_losses[-min(50, len(reg_losses)):])),
        "seconds": time.time() - t0,
        "d_model": int(args.d_model),
        "rec_dim": int(args.rec_dim),
        "layers": int(args.layers),
        "slow_lambda_init_mode": args.slow_lambda_init_mode,
        "slow_lambda_min": args.slow_lambda_min,
        "slow_lambda_max": args.slow_lambda_max,
        "plru_tau": args.plru_tau if variant in ("P-LRU unit", "P-LRU-Block") else "",
        "plru_c": args.plru_c if variant in ("P-LRU unit", "P-LRU-Block") else "",
        "plru_warmup_frac": args.plru_warmup_frac if variant in ("P-LRU unit", "P-LRU-Block") else "",
        "pan_lambda_min": args.pan_lambda_min if is_pan_variant(variant) else "",
        "pan_lambda_max": args.pan_lambda_max if is_pan_variant(variant) else "",
        "pan_eta_lambda": args.pan_eta_lambda if is_pan_variant(variant) else "",
        "pan_score_eps": args.pan_score_eps if is_pan_variant(variant) else "",
        "pan_h_probe": args.pan_h_probe if is_pan_variant(variant) else "",
    }
    result.update(metrics)

    torch.save(model.state_dict(), ckpt_path)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    if lambda_trace is not None:
        trace_path = os.path.join(args.trace_dir, f"{task}_{size_name}_{tag}_seed{seed}_lambda_trace.npz")
        np.savez_compressed(
            trace_path,
            steps=np.asarray(lambda_trace["steps"], dtype=np.int64),
            lambdas=np.asarray(lambda_trace["lambdas"], dtype=np.float32),
            lambda_gt_0p9=np.asarray(lambda_trace["lambda_gt_0p9"], dtype=np.int64),
            lambda_gt_0p95=np.asarray(lambda_trace["lambda_gt_0p95"], dtype=np.int64),
            lambda_gt_0p99=np.asarray(lambda_trace["lambda_gt_0p99"], dtype=np.int64),
            pan_score_mean=np.asarray(lambda_trace["pan_score_mean"], dtype=np.float32),
            pan_score_max=np.asarray(lambda_trace["pan_score_max"], dtype=np.float32),
            pan_probe_loss=np.asarray(lambda_trace["pan_probe_loss"], dtype=np.float32),
        )
    return {"status": "done", "path": json_path}


def aggregate(out_dir, csv_path):
    rows = []
    for name in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        if name.endswith(".json"):
            with open(os.path.join(out_dir, name)) as f:
                rows.append(json.load(f))
    if not rows:
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=512)
    p.add_argument("--analysis-batch", type=int, default=256)
    p.add_argument("--train-min", type=int, default=10)
    p.add_argument("--train-max", type=int, default=50)
    p.add_argument("--eval-horizons", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 2000])
    p.add_argument("--analysis-horizon", type=int, default=50)
    p.add_argument("--post-hold", type=int, default=500)
    p.add_argument("--recovery-steps", type=int, nargs="+", default=[0, 20, 100, 500])
    p.add_argument("--normal-radius", type=float, default=0.25)
    p.add_argument("--basin-radii", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0])
    p.add_argument("--flip-update-prob", type=float, default=0.08)
    p.add_argument("--transition-hold", type=int, default=20)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--rec-dim", type=int, default=96)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--plru-tau", type=float, default=0.001)
    p.add_argument("--plru-c", type=float, default=50.0)
    p.add_argument("--plru-warmup-frac", type=float, default=0.3)
    p.add_argument("--rank-matched-lambda-high", type=float, default=0.999)
    p.add_argument("--rank-matched-lambda-low", type=float, default=0.0)
    p.add_argument("--slow-lambda-init-mode", choices=["default", "linspace", "random", "gaussian", "bimodal", "fixed"], default="linspace")
    p.add_argument("--slow-lambda-min", type=float, default=0.90)
    p.add_argument("--slow-lambda-max", type=float, default=0.999)
    p.add_argument("--slow-lambda-fixed", type=float, default=0.999)
    p.add_argument("--slow-lambda-mean", type=float, default=0.5)
    p.add_argument("--slow-lambda-std", type=float, default=0.2)
    p.add_argument("--slow-lambda-low-mean", type=float, default=0.05)
    p.add_argument("--slow-lambda-high-mean", type=float, default=0.95)
    p.add_argument("--slow-lambda-high-prob", type=float, default=0.5)
    p.add_argument("--slow-lambda-shuffle", action="store_true")
    p.add_argument("--slow-lambda-init-seed", type=int, default=-1)
    p.add_argument("--pan-lambda-min", type=float, default=0.90)
    p.add_argument("--pan-lambda-max", type=float, default=0.999)
    p.add_argument("--pan-eta-lambda", type=float, default=3000.0)
    p.add_argument("--pan-score-eps", type=float, default=0.0)
    p.add_argument("--pan-warmup-frac", type=float, default=0.3)
    p.add_argument("--pan-probe-every", type=int, default=100)
    p.add_argument("--pan-probe-batch", type=int, default=96)
    p.add_argument("--pan-probe-horizon", type=int, default=50)
    p.add_argument("--pan-h-probe", type=int, default=500)
    p.add_argument("--log-lambda-trajectory", action="store_true")
    p.add_argument("--lambda-log-every", type=int, default=100)
    p.add_argument("--gpu", type=int, default=-1)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="exp74_discrete_attractor_results")
    p.add_argument("--ckpt-dir", default="checkpoints_exp74_discrete_attractor")
    p.add_argument("--trace-dir", default="traces_exp74_discrete_attractor")
    p.add_argument("--summary-csv", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.plot_only:
        aggregate(args.out_dir, args.summary_csv or os.path.join(args.out_dir, "exp74_discrete_attractor_metrics.csv"))
        return
    if args.gpu >= 0 and args.device == "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.smoke:
        args.steps = min(args.steps, 2)
        args.batch = min(args.batch, 8)
        args.eval_batch = min(args.eval_batch, 8)
        args.analysis_batch = min(args.analysis_batch, 16)
        args.train_min = 2
        args.train_max = 3
        args.eval_horizons = [2]
        args.analysis_horizon = 2
        args.post_hold = 2
        args.recovery_steps = [0, 2]
        args.basin_radii = [0.0, 0.2]
        args.pan_probe_every = 1
        args.pan_probe_batch = 8
        args.pan_probe_horizon = 2
        args.pan_h_probe = 2

    if args.device == "auto":
        args.device_obj = torch.device("cuda:0" if torch.cuda.is_available() and not args.smoke else "cpu")
    else:
        args.device_obj = torch.device(args.device)
    print(train_eval_one(args), flush=True)


if __name__ == "__main__":
    main()
