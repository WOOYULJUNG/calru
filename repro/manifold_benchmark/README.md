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
