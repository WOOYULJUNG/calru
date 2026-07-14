# CA-LRU-native recipe transfer for the Ságodi ring task

## Status and purpose

This document fixes the provenance and execution boundary for the next
**pilot-only** experiment. It is a recipe transfer, not an exact rerun of
Exp88 and not a continuation of the failed Protocol-A campaign.

The new track keeps the Ságodi angular-integration task and analysis contract,
while transferring the CA-LRU core and the training settings under which the
final state-dependent model previously trained successfully. It must receive
its own executable freeze, fingerprint, artifact root, manifests, and receipts
before launch. The final section documents the reviewed executable interface.

The three evidence families remain separate:

| evidence family | role | may be pooled with the new pilot? |
|---|---|---|
| Exp88 legacy runs | architecture/training provenance and a stability reference | no |
| failed Ságodi Protocol-A freeze | negative engineering evidence | no |
| new CA-LRU-native recipe-transfer pilot | newly trained Ságodi-task evidence | only within its own freeze |

## Canonical legacy reference

The paper-facing final CA-LRU is the legacy variant `PAN-RNW-full`; `PAN-full`
is the linear-update control and must not be used as the canonical model. The
mapping is fixed in
[`docs/LEGACY_NAME_MAP.md`](../../docs/LEGACY_NAME_MAP.md#final-model%EA%B3%BC-update-controls)
and the evidence limitations are recorded in
[`paper/evidence/PROVENANCE.md`](../../paper/evidence/PROVENANCE.md#model-lineage).

### Model core and outer scaffold

The canonical implementation chain is:

1. [`exp71_pan_block_pulse_hold.py`](../legacy_code/exp71_pan_block_pulse_hold.py)
   maps `PAN-RNW-full` to `PAN-RNW-Block` and supplies
   `encoder_bias=False`, `use_norm_in=False`.
2. [`pan_block.py`](../legacy_code/pan_block.py) implements the recurrent writer
   and the shared full-block scaffold.
3. [`exp72_structured_attractor_tasks.py`](../legacy_code/exp72_structured_attractor_tasks.py)
   implements the coordinate-ablation damage score and retention update used
   by Exp88.
4. [`exp88_manifold_attractor_tasks.py`](../legacy_code/exp88_manifold_attractor_tasks.py)
   supplies the training loop and RP calls.
5. [`run_exp88_writer_sweep.py`](../legacy_code/run_exp88_writer_sweep.py) and
   [`run_exp88_manifold_main.py`](../legacy_code/run_exp88_manifold_main.py)
   fix the successful writer-sweep jobs and explicit arguments.

For width 96 and one layer, the exact core is:

\[
h_{t+1}=\Lambda h_t+\sigma(\gamma_{\rm raw})
\left[g([h_t;u_t])-g([h_t;0])\right],
\]

where `g` is `Linear(192,96,bias=True) -> GELU ->
Linear(96,96,bias=True)`. The initial retention values are linearly spaced from
0.999 down to 0.90, their `theta` parameter has `requires_grad=False`, and
`gamma_raw` starts at zero, hence the initial write gain is 0.5. Blank input
cancels the writer exactly, so the carrier map is `F0(h)=Lambda h`.

The surrounding block is also part of the reference, not incidental glue:

- bias-free input encoder;
- no input LayerNorm;
- recurrent output projection with bias;
- GELU, biased GLU projection, residual connection, dropout 0;
- affine output LayerNorm;
- affine head LayerNorm followed by a biased linear readout;
- `decode_mode=stream`, `carry_stream=False`;
- a 96-coordinate causal carrier plus a 96-coordinate overwritten decoder
  stream; only the carrier is the minimum Markov state.

All learned linear layers otherwise use the PyTorch initialization active in
the archived implementation. Exp71 is the **builder lineage**. Its CLI defaults
(`400` updates, width `64`, two layers, RP `eta=300`, blank horizon `200`) are
not the final paper recipe and must not be copied into the new freeze.

### Successful Exp88 recipe

The successful ring-integration reference used:

| item | legacy value |
|---|---:|
| model | `PAN-RNW-full` |
| width / recurrent width / layers | 96 / 96 / 1 |
| task input | six-dimensional cue/velocity-feature sequence |
| initial state | exact zeros; initial angle supplied by a cue token |
| train horizon | uniformly sampled integer 80–260 |
| batch / updates | 256 / 10,000 |
| loss | all-step MSE |
| optimizer | AdamW, LR `1e-3`, weight decay `1e-5` |
| clipping | global gradient norm 1.0 |
| dropout / training state noise | 0 / 0 |
| initial retention | linear 0.999→0.90 |
| RP warm-up | updates 1–3000 |
| RP decisions | 3100, 3200, ..., 10000; 70 calls |
| RP probe | batch 96, task horizon 260, then 500 blank steps, noise off |
| RP score/update | coordinate-ablation damage, fixed epsilon `1e-4`, `eta=3000` |

The six-dimensional ring input is not raw scalar angular velocity. At the cue
token it contains the initial `(cos(theta), sin(theta))` and a cue flag; later
tokens contain `(v, sin(v), cos(v)-1)`. This task contract and the variable
horizon are legacy provenance only.

The canonical successful records are:

- committed JSON:
  `paper/evidence/raw/exp88_writer_sweep_results/`
  `ring_integrate_am_lru_rnw_eps1e-4_seed{0,1,2}.json`;
- aggregate:
  [`paper/evidence/tables/main_manifold_metrics.csv`](../../paper/evidence/tables/main_manifold_metrics.csv),
  run `main.ring_integrate.ca_lru`;
- original-workspace checkpoints, not committed to this repository:
  `/home/biadmin/ca_rnn/checkpoints_exp88_writer_sweep/`
  `exp88_ring_integrate_am_lru_rnw_eps1e-4_seed{0,1,2}.pt`;
- original-workspace training log for the seed-0 reference:
  `/home/biadmin/ca_rnn/logs_exp88_writer_sweep/`
  `0023_ring_integrate_am_lru_rnw_eps1e-4_seed0_gpu4.log`.

The three checkpoint SHA-256 values for seeds 0, 1, and 2 are respectively:

```text
dc20c6dbb8e68898562b66c93033cccde8b9f19876f29d19a4f1f849a014da36
7a7366eb0320e7880b34d69215a996a74bf87fb571aea1de67a199f9e9cd0509
59fedeb16bc1a48aed4b2b0eddc4c5026bd07255e849750ff752382232bd8d59
```

All three load strictly into the canonical builder and contain finite tensors.
Their ID endpoint RMSE values are 0.0007440, 0.0045126, and 0.0012880
(mean 0.0021815), and their counts of carrier coordinates with
`lambda > 0.99` are 6, 7, and 5. These values establish that this recipe once
completed successfully; they are not acceptance thresholds for the transfer.

The repository snapshot does not include these checkpoints and the historical
workspace was not a Git worktree when the runs were made. Therefore this
provenance cannot establish a bit-exact run-time code revision. See
[`paper/evidence/PROVENANCE.md`](../../paper/evidence/PROVENANCE.md#snapshot-%EC%84%B1%EA%B2%A9).

## New recipe-transfer freeze

The new track is deliberately a hybrid at the level of **task versus training
recipe**, but every run inside it uses one fully specified configuration. The
following table is the required boundary.

| component | new pilot decision | source |
|---|---|---|
| latent task | ring angular-velocity integration; GP Cholesky jitter `1e-6` | Ságodi contract plus explicit implementation metadata |
| initialization interface | hidden initialization from `(cos(q0), sin(q0))` | Ságodi contract |
| initial adapter | bias-free `W_otr`; weights initialized with mean 0 and standard deviation `1/sqrt(96)` | Ságodi code-style adapter |
| model input | raw scalar `omega_t` | Ságodi contract |
| sequence | fixed `T=256` velocity tokens | Ságodi contract |
| dynamics target | `q_{t+1} = wrap(q_t + 0.1 * omega_t)` | Ságodi contract |
| velocity process | frozen GP bank/generator | Ságodi contract |
| analysis and fixed banks | Phase-0 state audit and Phase-1 ring gates | Ságodi contract |
| CA-LRU / No-RP core | exact `PAN-RNW-full` scaffold above | Exp71/Exp88 lineage |
| optimizer | AdamW, LR `1e-3`, weight decay `1e-5` | successful Exp88 recipe |
| clipping | global gradient norm 1.0 | successful Exp88 recipe |
| batch / updates | 256 / 10,000 | successful Exp88 recipe |
| dropout / training state noise | 0 / 0 | successful Exp88 recipe |
| RP warm-up and interval | no updates through 3000; every 100 thereafter | successful Exp88 recipe |
| RP decisions | 3100 through 10000 inclusive; 70 calls | successful Exp88 recipe |
| RP probe | batch 96, task horizon **256**, blank horizon 500, noise off | Exp88 numeric recipe with declared horizon, RNG-stream, and pre-warm-up execution adaptations |
| RP score/update | aligned coordinate-ablation damage, fixed epsilon `1e-4`, `eta=3000` | successful ring Exp88 recipe |
| pilot models | CA-LRU, matched final-scaffold No-RP, GRU | existing ring-pilot matrix |
| pilot model seeds | 100, 101, 102, 103, 104 | tuning-only seed bank |

The task probe horizon is 256 because the transferred task is fixed at 256;
it must not be described as the unchanged Exp88 probe of 260. Likewise, raw
scalar velocity, hidden initialization, GP trajectories, and fixed sequence
length are not legacy Exp88 settings.

The randomness and probe execution policy is also an explicit adaptation.
Exp88 used its process-global RNG and computed score-only probes at steps
100--3000 before applying 70 retention updates at steps 3100--10000. The new
pilot uses keyed per-update training batches shared across models, keyed RP
probe banks, and executes the 70 probes that actually make retention-update
decisions. Thus the optimizer/RP numeric values are inherited, but the data
stream and pre-warm-up diagnostic-compute history are not bit-exact Exp88
behavior.

The hidden-initialization adapter is pinned to the
[official Ságodi repository](https://github.com/catniplab/back_to_the_continuous_attractor/tree/cbd7404e9baca4b2dc291560cfc6576bb7b1f078),
commit `cbd7404e9baca4b2dc291560cfc6576bb7b1f078`, file `models.py`, parameter
`RNN.output_to_hidden`: it is a trainable bias-free matrix initialized from a
normal distribution with standard deviation `1/sqrt(hidden_size)`.

No-RP must share the CA-LRU core, initialization, optimizer, batch stream,
initial retention spectrum, and task-gradient policy, with only the RP hook
disabled. Shared training settings must also be applied to GRU unless a new
freeze explicitly records a model-specific exception.

Before execution, the new machine-readable freeze must assert all values in
this table, derive the expected 70 RP step indices, record the initial-adapter
initializer rather than relying on `nn.Linear.reset_parameters`, and fail
closed if any resolved builder property differs.

## What is not transferred

The following legacy properties are explicitly excluded from the new task:

- the six-dimensional cue/velocity-feature input;
- zero-state plus prepended cue-token initialization;
- variable 80–260 training horizon;
- Exp88 task distributions, task seeds, and evaluation samples;
- trained Exp88 parameter values or final retention spectra;
- task-specific output metrics as an acceptance threshold.

The following failed Protocol-A settings are also excluded:

- biased default-PyTorch hidden initializer;
- Adam with LR `1e-2` and weight decay 0;
- batch 64 and 5,000 updates;
- coordinate state-noise standard deviation 0.1 during training;
- RP updates from 1550 every 50 steps, probe batch 256, and epsilon `3e-5`.

Changing these settings creates a new freeze. It does not repair or resume the
old one in place.

## Checkpoint policy

Every native-transfer run starts from a fresh initialization under its own
model seed. Loading, partially loading, remapping, or warm-starting from an
Exp88 checkpoint is prohibited.

The prohibition is substantive: the legacy checkpoint expects a six-feature
cue-driven input and a zero initial state, whereas the new model expects a
one-feature raw-velocity input and a separately learned hidden-state adapter.
The training distributions and sequence schedules also differ. A successful
strict load of the recurrent subset would still leak trained task structure
and would make the result neither a clean Ságodi-task run nor a fresh
recipe-transfer test.

Legacy checkpoints may only be used in a separate, read-only provenance test
that checks builder compatibility, tensor finiteness, and historical metrics.
No tensor or optimizer state from that test may enter a new run artifact.

## Protocol A and Protocol B must not be mixed

Protocol A remains the original-paper reproduction track: hidden init, raw
velocity, batch 64, 5,000 Adam updates, coordinate state noise 0.1, and its own
four-LR/five-pilot selection rule. A Protocol-A result may only use a
Protocol-A freeze and label.

The new experiment is a separately fingerprinted **CA-LRU-native
recipe-transfer pilot on the Ságodi task**. Operationally it belongs on the
Protocol-B/CA-benchmark side because it uses the 10,000-update native budget,
but it is not yet the full confirmatory Protocol B: it transfers one fixed
historical recipe and does not execute the complete equal-budget LR/weight-
decay selection grid.

Consequently:

- do not call the new result a Ságodi original-paper reproduction;
- do not pool seed statistics, success rates, or checkpoints across Protocol A,
  the native-transfer pilot, or legacy Exp88;
- do not compare them in one row as though only the architecture changed;
- do not choose a new setting after inspecting CA-geometry or test-bank
  metrics; any such change requires another pilot freeze;
- retain all-started and task-success-conditional denominators within each
  freeze.

## Sentinel first, then the remaining 14 pilots

The 15-run matrix must not be launched simultaneously. The first and only
sentinel is:

```text
model=ca_lru, model_seed=100, width=96, lr=1e-3
```

It must run the real 10,000-update configuration, not a shortened smoke run.
The remaining jobs are held until the sentinel has:

1. passed the current Phase-0 state/blank-map gate under the new source and
   freeze fingerprint;
2. completed all 10,000 optimizer updates;
3. kept loss, parameters, gradients, optimizer state, retention values, and RP
   diagnostics finite;
4. executed exactly the expected 70 RP decisions;
5. produced a valid checkpoint-bound training-completion receipt and finite
   task metrics.

The mechanical fan-out gate is the verified training-completion receipt, not
whether one tuning seed happens to cross the task-success threshold. The
sentinel is a numerical/full-path validation and must not become a one-seed
performance-selection rule. A non-finite or invalid receipt stops fan-out;
finite but poor task performance is retained and evaluated across the frozen
five-seed pilot rather than silently discarded.

Only then may the remaining 14 jobs start:

- CA-LRU seeds 101–104;
- No-RP seeds 100–104;
- GRU seeds 100–104.

The sentinel remains part of the five-seed CA-LRU pilot; it is not discarded
or rerun merely because its metric is inconvenient. If it fails, preserve the
attempt and stop. A repair changes the freeze identifier and requires a new
sentinel before any fan-out.

## Preservation of the failed freeze

The failed campaign is a forensic snapshot and must remain immutable:

```text
/home/biadmin/ca_rnn/calru_sagodi_phase01_20260713
```

Its campaign label is `sagodi_phase01_ring_pilot_v1-451d95c7078f`, with full
protocol fingerprint
`451d95c7078fa0272890b188576fc0c80b98546608e2e1fa15f7f1a0b96c8d7e`.
The CA-LRU seed-100 attempt failed before RP, at update 1204, when
`optimizer.state0.exp_avg_sq` became non-finite. Five concurrently launched
attempts were terminated and nine jobs remained pending.

At minimum, preserve its `manifest.json`, Phase-0 artifacts and receipt,
evaluation/perturbation banks with hashes, every attempt `config.json`, and all
training logs. The primary failure log is:

```text
/home/biadmin/ca_rnn/calru_sagodi_phase01_20260713/logs/training/
phase1__angle_integrate__ca_lru__w096__seed100__lr0p01/
attempt-1783962760489496079-pid1112912-9e3d647f.log
```

Do not delete, rename, append new runs to, or reuse this artifact root. The new
freeze must use a distinct root such as
`<ARTIFACT_BASE>/<NEW_FREEZE_ID>-<NEW_FINGERPRINT_PREFIX>` and must reference
the failed root only as provenance.

## Reproduction and launch commands

Run these commands only from a clean committed worktree. `--dry-run` is a
setup validation, not a read-only command: it creates the root marker, fixed
banks, and campaign manifest. Use a disposable fresh root for it. Never point
either command at the failed Protocol-A root.

```bash
# 1. Set local paths and validate the new machine-readable freeze.
REPO=/absolute/path/to/calru-paper
PYTHON=/absolute/path/to/python
PROTOCOL="$REPO/repro/sagodi_protocol/calru_native_sagodi_ring_pilot_v1.yaml"
ARTIFACT_ROOT=/absolute/new/path/calru_native_sagodi_ring_pilot_v1
DRY_RUN_ROOT=/tmp/calru_native_sagodi_ring_pilot_v1_dry_run

cd "$REPO"
git status --short
git rev-parse HEAD
PYTHONNOUSERSITE=1 "$PYTHON" -m pytest -q repro/sagodi_protocol/tests
PYTHONNOUSERSITE=1 "$PYTHON" -m repro.sagodi_protocol.config \
  --protocol "$PROTOCOL" \
  --print-fingerprint

# 2. Materialize and audit a disposable 1+14 dry-run plan.
PYTHONNOUSERSITE=1 "$PYTHON" -m repro.sagodi_protocol.orchestrate \
  --protocol "$PROTOCOL" \
  --artifact-root "$DRY_RUN_ROOT" \
  --python "$PYTHON" \
  --gpus 0,1,2,3,4,5 \
  --dry-run

# 3. Launch one orchestrator in tmux.
# It runs Phase 0, then only CA-LRU seed 100. It verifies that sentinel's
# checkpoint-bound completion receipt before automatically releasing the
# remaining 14 jobs; no task-performance threshold is used for fan-out.
tmux new-session -d -s calru_native_sagodi_ring_pilot_v1 \
  "cd '$REPO' && PYTHONNOUSERSITE=1 '$PYTHON' -m repro.sagodi_protocol.orchestrate --protocol '$PROTOCOL' --artifact-root '$ARTIFACT_ROOT' --python '$PYTHON' --gpus 0,1,2,3,4,5"

# 4. Read-only status/receipt verification while running or after completion.
PYTHONNOUSERSITE=1 "$PYTHON" -m repro.sagodi_protocol.status \
  "$ARTIFACT_ROOT" --json

# During the sentinel, the active attempt's atomic file updates every 100 steps.
find "$ARTIFACT_ROOT/attempts/training/phase1__angle_integrate__ca_lru__w096__seed100__lr0p001" \
  -name progress.json -print -exec cat {} \;
```

If setup crashes before the scientific root marker is committed, preserve that
partial root for forensics and choose another fresh root. The orchestrator
fails closed instead of guessing that partial setup state is reusable.
