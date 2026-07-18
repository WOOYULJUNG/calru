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

## Split-field refinement and CA-LRU comparison

The later split-field campaign expands the screen to `S1`, `T2`, and `S2`.
It keeps autonomous and input-driven fields separate and evaluates task error,
blank memory, local tangent/normal singular gains, finite normal kicks, shape
distortion, and persistent-homology signatures.

The 10,000-update duration diagnostic resumes nine selected 2,000-update
checkpoints with their optimizer state:

```bash
python -m repro.static_gate_pilot.launch_split_gap_followup_v3
python -m repro.static_gate_pilot.launch_split_candidate_dynamics_v2 \
  --candidates \
  /home/biadmin/ca_rnn/experiments/static_gate_split_gap_followup_v3/full_summary.csv \
  --output \
  /home/biadmin/ca_rnn/experiments/static_gate_split_gap_followup_dynamics_v3
```

The CA-LRU-inclusive comparison reanalyzes the validation-selected
topology-tuning-v2 checkpoints (three seeds per topology) with the same
split-field analysis and combines them with RNN, GRU, and LSTM:

```bash
python -m repro.static_gate_pilot.launch_calru_integrated_comparison_v3
```

For topology wrappers, dynamics and persistent homology are computed on the
primary recurrent carrier. Decoder-stream coordinates stored in the reported
state are reconstructed only when decoding; including them in the Jacobian
would measure decoder recomputation rather than recurrent-state dynamics.
