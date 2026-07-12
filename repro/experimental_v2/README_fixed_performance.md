# Deterministic fixed-test performance evaluation

`evaluate_fixed_performance.py` rebuilds the performance tables directly from
campaign checkpoints. It does **not** reuse any metric saved by a trainer. A
result JSON is read only for the model variant, dimensions, seed, and other
construction metadata, after which the checkpoint must pass a strict
`load_state_dict`.

## Fixed test definition

One asset set is generated before any model is evaluated. All eight Exp88
tasks receive a fixed `q0.npy`; each applicable ID, temporal-OOD, velocity
scale, sparse, and alternating profile receives a fixed `velocity.npy` and an
integer `schedule.npy`. Every model and every training seed for a task reads
those same arrays. The asset manifest records shapes, dtypes, SHA-256 hashes,
and a hash of the complete test definition.

The primary full-run statistic is
`postH1000_ambient_component_rmse` on the ID profile. The raw table also
contains endpoint component/vector RMSE, post-blank vector RMSE, paired
H0-to-H1000 output drift, sequence RMSE, and the legacy-compatible mean
per-step Euclidean errors split between moving and hold steps. Temporal and
velocity profiles use the same columns. `fixed_performance_summary.csv`
aggregates training seeds; `fixed_performance_macro.csv` reports the available
task coverage explicitly so an anchor-only condition cannot look like an
eight-task result.

## Usage

Run a no-write preflight first:

```bash
python repro/experimental_v2/evaluate_fixed_performance.py \
  --campaign-dir <artifact-root>/<campaign-id> \
  --output-dir <fresh-output-dir> \
  --dry-run
```

Then run the fixed evaluation, preferably on one GPU:

```bash
python repro/experimental_v2/evaluate_fixed_performance.py \
  --campaign-dir <artifact-root>/<campaign-id> \
  --output-dir <fresh-output-dir> \
  --device cuda:0
```

Manifest fields can be filtered without editing the campaign:

```bash
python repro/experimental_v2/evaluate_fixed_performance.py \
  --campaign-dir <artifact-root>/<campaign-id> \
  --output-dir <fresh-output-dir> \
  --conditions rg_lru_main \
  --models RG-LRU-full \
  --tasks ring_hold,ring_integrate \
  --seeds 0,1,2
```

`--smoke` reduces the fixed batch and horizons but keeps the same provenance
and atomic-completion protocol. It is a wiring check, not a paper result.

## Evaluation-only retention permutation

For selected CA-LRU/PAN checkpoints, an optional control evaluates the original
checkpoint and a second copy whose final theta coordinates are permuted:

```bash
python repro/experimental_v2/evaluate_fixed_performance.py \
  --campaign-dir <artifact-root>/<campaign-id> \
  --output-dir <fresh-output-dir> \
  --conditions rp_aligned \
  --retention-permutation \
  --permutation-seed-base 20260713
```

The permutation is derived deterministically from the base seed and checkpoint
identity. Rows record the exact permutation hash, ordered spectra before and
after, the common multiset hash, and an assertion/hash showing that every
non-retention state-dict entry stayed unchanged. The option rejects a selection
containing any non-PAN model.

## Completion and provenance

The output directory must not exist. The evaluator writes `INCOMPLETE` first,
hashes all source and campaign inputs, evaluates, re-hashes the inputs, writes
`run_manifest.json` and `SHA256SUMS` atomically, checks the sources once more,
and writes `COMPLETE` as the final mutation. A crash or changing source/input
leaves `INCOMPLETE` and never produces a valid completion marker.
