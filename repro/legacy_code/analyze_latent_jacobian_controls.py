#!/usr/bin/env python3
"""Latent Jacobian controls for hold/integrate attractor dynamics.

This complements the finite normal-kick OOD probe with a local linearization.
For hold tasks we evaluate the one-step zero-input Jacobian.  For integration
tasks we evaluate the one-step Jacobian under the task's tangent input at the
middle of the rollout, comparing input and output tangent bases.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from analyze_normal_kick_ood import (
    AnalysisSpec,
    default_specs,
    load_model,
    make_eval_sequence,
    tangent_basis_at_step,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "analysis_latent_jacobian_controls"


def parse_list(text: str, cast):
    return [cast(x) for x in str(text).split(",") if x.strip()]


def normal_basis_from_tangent(q: torch.Tensor) -> torch.Tensor:
    _, _, vh = torch.linalg.svd(q.T, full_matrices=True)
    return vh[q.shape[1] :].T.contiguous()


@torch.no_grad()
def run_states(model, x):
    state = model.init_state(x.shape[1], x.device)
    states = []
    for x_t in x:
        state = model.step(x_t, state)
        states.append(state)
    return states


def one_step_jacobian(model, state: torch.Tensor, x_next: torch.Tensor):
    x_next = x_next.unsqueeze(0)

    def fn(s_flat):
        return model.step(x_next, s_flat.unsqueeze(0)).squeeze(0)

    return torch.autograd.functional.jacobian(fn, state, vectorize=True)


def per_sample_metrics(model, state, x_next, q_in, q_out, random_normals: int):
    state = state.detach().clone().requires_grad_(True)
    q_in = torch.linalg.qr(q_in.detach(), mode="reduced")[0]
    q_out = torch.linalg.qr(q_out.detach(), mode="reduced")[0]
    n_in = normal_basis_from_tangent(q_in)
    n_out = normal_basis_from_tangent(q_out)
    jac = one_step_jacobian(model, state, x_next)

    with torch.no_grad():
        tangent_block = q_out.T @ jac @ q_in
        tangent_svs = torch.linalg.svdvals(tangent_block)
        tangent_eigs = torch.linalg.eigvals(tangent_block).abs()

        normal_block = n_out.T @ jac @ n_in
        normal_svs = torch.linalg.svdvals(normal_block)
        normal_eigs = torch.linalg.eigvals(normal_block).abs()

        coeff = torch.randn(random_normals, n_in.shape[1], device=jac.device, dtype=jac.dtype)
        normal0 = coeff @ n_in.T
        normal0 = normal0 / normal0.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        normal_after = (jac @ normal0.T).T
        normal_residue = normal_after @ n_out
        normal_residue_gain = normal_residue.norm(dim=-1)

    return {
        "tangent_eig_abs_mean": float(tangent_eigs.mean().item()),
        "tangent_sv_mean": float(tangent_svs.mean().item()),
        "normal_eig_abs_mean": float(normal_eigs.mean().item()),
        "normal_eig_abs_max": float(normal_eigs.max().item()),
        "normal_sv_mean": float(normal_svs.mean().item()),
        "normal_sv_max": float(normal_svs.max().item()),
        "normal_random_residue_mean": float(normal_residue_gain.mean().item()),
        "normal_random_residue_max": float(normal_residue_gain.max().item()),
    }


def mean_sem(values):
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(arr.size))


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["eval_task"], row["label"])
        groups.setdefault(key, []).append(row)
    out = []
    for (task, label), vals in sorted(groups.items()):
        item = {"eval_task": task, "label": label, "n": len(vals)}
        for k in sorted(vals[0]):
            if k in {"eval_task", "label", "seed", "sample", "analysis_step", "checkpoint_task", "tag", "intervention"}:
                continue
            mean, sem = mean_sem([v[k] for v in vals])
            item[f"{k}_mean"] = mean
            item[f"{k}_sem"] = sem
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict]):
    lines = [
        "# Latent Jacobian Controls",
        "",
        "One-step local linearization at the learned latent manifold.",
        "For integration tasks the step uses the task's tangent input, with separate input/output tangent bases.",
        "",
        "| task | model | n | tangent eig | normal random gain | normal eig max |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in sorted(summary, key=lambda x: (x["eval_task"], x["label"])):
        lines.append(
            "| {task} | {label} | {n} | {te:.3f} | {nr:.3f} | {nem:.3f} |".format(
                task=r["eval_task"],
                label=r["label"],
                n=r["n"],
                te=r["tangent_eig_abs_mean_mean"],
                nr=r["normal_random_residue_mean_mean"],
                nem=r["normal_eig_abs_max_mean"],
            )
        )
    lines.extend(
        [
            "",
            "Read:",
            "",
            "```text",
            "tangent_eig measures whether valid manifold directions are preserved locally.",
            "normal_random_gain measures local contraction of random off-manifold perturbations.",
            "normal_eig_max summarizes worst-case normal-subspace linear gain.",
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def labels_match(spec: AnalysisSpec, labels: set[str]):
    if not labels:
        return True
    return spec.label in labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--eval-tasks", default="ring_hold,ring_integrate,line_hold,line_integrate")
    parser.add_argument("--labels", default="AM-LRU,AM-LRU shuffle,AM-LRU all-slow")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--analysis-step", type=int, default=-1)
    parser.add_argument("--random-normals", type=int, default=32)
    parser.add_argument("--tangent-eps", type=float, default=1e-3)
    parser.add_argument("--z-scale", type=float, default=0.50)
    parser.add_argument("--vel-scale", type=float, default=0.08)
    parser.add_argument("--omega-scale", type=float, default=0.18)
    parser.add_argument("--omega-hold-prob", type=float, default=0.25)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    seeds = parse_list(args.seeds, int)
    eval_tasks = set(parse_list(args.eval_tasks, str))
    labels = set(parse_list(args.labels, str)) if args.labels else set()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    analysis_step = int(args.analysis_step)
    if analysis_step < 0:
        analysis_step = int(args.horizon) // 2
    analysis_step = max(0, min(int(args.horizon) - 1, analysis_step))

    rows = []
    for spec in default_specs(eval_tasks):
        if not labels_match(spec, labels):
            continue
        for seed in seeds:
            try:
                model, _ = load_model(spec.checkpoint, seed, device)
            except FileNotFoundError as exc:
                print(f"[skip] {spec.eval_task} {spec.label} seed={seed}: {exc}", flush=True)
                continue
            print(f"[eval] {spec.eval_task} {spec.label} seed={seed}", flush=True)
            torch.manual_seed(71000 + seed)
            x, _, _, aux = make_eval_sequence(
                spec.eval_task,
                spec.checkpoint.dim,
                int(args.batch),
                int(args.horizon),
                device,
                args,
            )
            states = run_states(model, x)
            q_in = tangent_basis_at_step(
                model,
                spec.eval_task,
                spec.checkpoint.dim,
                aux,
                analysis_step,
                float(args.tangent_eps),
            )
            q_out = tangent_basis_at_step(
                model,
                spec.eval_task,
                spec.checkpoint.dim,
                aux,
                analysis_step + 1,
                float(args.tangent_eps),
            )
            sample_count = min(int(args.samples), int(args.batch))
            for i in range(sample_count):
                item = per_sample_metrics(
                    model,
                    states[analysis_step][i],
                    x[analysis_step + 1, i],
                    q_in[i],
                    q_out[i],
                    int(args.random_normals),
                )
                item.update(
                    {
                        "eval_task": spec.eval_task,
                        "checkpoint_task": spec.checkpoint.task,
                        "label": spec.label,
                        "tag": spec.checkpoint.tag,
                        "intervention": spec.checkpoint.intervention,
                        "seed": seed,
                        "sample": i,
                        "analysis_step": analysis_step,
                    }
                )
                rows.append(item)

    summary = summarize(rows)
    write_csv(out_dir / "latent_jacobian_controls_raw.csv", rows)
    write_csv(out_dir / "latent_jacobian_controls_summary.csv", summary)
    write_markdown(out_dir / "LATENT_JACOBIAN_CONTROLS.md", summary)
    print(out_dir / "latent_jacobian_controls_summary.csv")
    print(out_dir / "LATENT_JACOBIAN_CONTROLS.md")


if __name__ == "__main__":
    main()
