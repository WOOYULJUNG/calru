# Static-gate side pilot

This directory is an isolated exploratory screen. It does **not** replace the
registered CA-LRU implementation or the main topology-tuning campaign.

It compares four width-52 cells on generator-v1 `S1` and `T2`, seed 10:

- static-update GRU, task-gradient retention;
- static-update GRU, gate-intervention RP retention;
- retention-untied nonlinear RNN, task-gradient retention;
- retention-untied nonlinear RNN, gate-intervention RP retention.

All jobs use the true `q0` initializer, predict `q1` after applying `u0`, and
use no state noise, target noise, or dropout. The default launcher performs
300-update smoke runs and records a finite plus loss-ratio diagnostic, then
runs every cell from scratch for 1,500 updates.

The launcher generates one disjoint 4,096-trajectory **train-only** pool and
memory-maps it for every model. Thus the task definition is not regenerated
per update or per model, and paired cells see identical sampled indices. The
already frozen generator-v1 validation and test banks remain evaluation-only.
The default launches eight concurrent jobs across the six available GPUs.

```bash
python -m repro.static_gate_pilot.launch --phase all
python -m repro.static_gate_pilot.aggregate
```

The reported decoder-radial recovery is deliberately labeled exploratory:
the decoder-output radial VJP is not a certified local manifold-normal.
Positive results must be rechecked with the slow-subspace decomposition before
they support an attractor claim.
