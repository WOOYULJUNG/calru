# Manifold benchmark generator v1

This package is separate from the frozen Ságodi source-reproduction task and
the legacy Exp88 data.  It generates the new topology/dimension benchmark with
one explicit indexing contract:

```text
initial_memory     = Phi(q_0)
inputs[t]          = command applied from q_t to q_{t+1}
latent_path        = q_0, ..., q_T
latent_targets[t]  = q_{t+1}
output_targets[t]  = Phi(q_{t+1})
```

The three primary topology tasks are:

- `S1`: scalar angular velocity and a cosine/sine output.
- `T2`: two independent angular velocities and two cosine/sine pairs.
- `S2`: a state-independent 3-D angular velocity `omega_t`; the state is
  updated by Rodrigues rotation `R(delta_t * omega_t) n_t`, while
  `omega_t cross n_t` is stored as `effective_velocity` for analysis.

All topologies use the same parent white noise and one global true-zero dwell
mask.  The GP grid spacing is fixed at `2 / 127`, even when the parent horizon
is longer than the training horizon.  `T^d` uses the first `d` coordinates of
one maximum-dimension parent, so dimensions are exactly nested.

Generate the initial model-independent banks with:

```bash
PYTHONPATH=. python -m repro.manifold_benchmark.build_id_banks \
  --output /path/to/fresh/output
```

The command writes a pickle-free parent NPZ, `T=128` S1/T2/S2 banks, mandatory
SHA-256 sidecars, exact-test results and topology calibration.  It emits
`GENERATOR_ID_BANKS_READY.json` only if these initial checks pass.  This marker
does not claim that the later OOD pairing suite is complete.

## Zero-retuning topology transfer

`topology_transfer_v1.json` freezes the ring-selected settings for RNN, GRU,
LSTM and H-C. Only the input projection, true-`q0` initial-memory map and
output decoder change dimension across S1/T2/S2. The forward contract is
always `initialize(q0) -> step(u0) -> decode -> compare with Phi(q1)`; there is
no cue token, teacher forcing, target noise, state noise or output dropout.

Run one job with:

```bash
PYTHONPATH=. python -m repro.manifold_benchmark.run_topology_transfer \
  --stage fixed_overfit --model hc --topology s2 --seed 9 --device cuda:0 \
  --output /path/to/output
```

The stages are `fixed_overfit` (B=16, T=32, 1000 repeated-batch updates),
`online_smoke` (seed 9, B=64, T=128, 500 fresh-batch updates), and `pilot`
(seeds 10/11/12, 5000 updates). A complete grid can be scheduled with one
restartable queue per GPU:

```bash
PYTHONPATH=. python -m repro.manifold_benchmark.launch_topology_transfer \
  --stage pilot --devices 0,1,2,3,4,5 --output /path/to/output
```

Each completed job contains a manifest, trace, result, checkpoint and
checksum receipt. `online_smoke` only reads the validation bank; `pilot`
evaluates the independently seeded test bank after training. Aggregate a
finished or partially finished root with:

```bash
PYTHONPATH=. python -m repro.manifold_benchmark.summarize_topology_transfer \
  /path/to/output
```

## Topology-specific CA-LRU/H-C search

`topology_hparam_v1.json` defines a separate validation-only capability search.
It adds constant-retention `CA-LRU + RP` beside state-dependent-retention H-C,
keeps noise/dropout disabled, and never opens the test bank during selection.
The registered campaign is six smoke jobs followed by 75 full 5,000-update
runs: 39 broad, 24 refinement, and 12 H-C finalist robustness runs.

```bash
PYTHONPATH=. python -m repro.manifold_benchmark.launch_topology_hparam \
  --phase all --devices 0,1,2,3,4,5 --output /path/to/output
```

Every active run atomically saves `progress.pt` at the report interval. Running
the same command again skips completed receipts and resumes unfinished jobs.
An individual twice-failed cell is recorded in `FAILED.json`; it does not stop
the remaining GPU queues. Broad and refinement selections are frozen to JSON
before the next phase begins.
