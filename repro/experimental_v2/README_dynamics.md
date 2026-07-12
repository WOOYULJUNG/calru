# Checkpoint-only dynamics evaluation

`evaluate_checkpoint_dynamics.py` implements the E5--E7 analyses without
training or modifying a checkpoint.  It reuses the Exp88 geometry, sequence,
and model-construction APIs in `../legacy_code`, but treats a full-block state
as two distinct objects:

```text
[ persistent recurrent carrier | transient output stream ]
```

All tangent and normal directions are constructed in, injected into, and
measured in the carrier.  The stream coordinates are copied unchanged at the
intervention time.  For models without a separate stream, the full recurrent
state (for example, both `h` and `c` for an LSTM) is the carrier.

## Analyses

- A deterministic task asset stores intrinsic family points, evaluation
  points, and a paired velocity path as `.npy` files with SHA-256 hashes.
- Blank flow measures raw carrier drift, direction drift, norm change,
  nearest-clean-family distance, intrinsic-coordinate drift, local-neighbor
  separation, and carrier/full-state rank.  A same-coordinate direct-cue state
  is also paired with each driven endpoint to quantify history-conditioned
  carrier and output dispersion.
- Tangent and normal kicks have exactly matched carrier norm.  Every kicked
  rollout is paired with an unperturbed rollout from the same state.  The
  primary perturbation quantity is
  `target_mse_kicked - target_mse_clean`; decoded kicked-clean MSE and
  carrier kicked-clean deviation are also saved.  Nearest-family distance and
  represented-coordinate error include both the clean value and their paired
  kicked-minus-clean difference, so reference-grid/history mismatch is not
  mistaken for a perturbation effect.
- Directional zero-input Jacobian-vector products are propagated for finite
  horizons.  The output is split into the locally transported tangent span
  and its carrier-orthogonal complement; carrier, generated-stream, full-state,
  and decoded-output gains are saved separately.  No dense Jacobian is
  materialized.
- When a model exposes retention magnitudes, the report includes tangent
  alignment with the highest-retention carrier coordinates.

The default spec starts with final `PAN-RNW-full` CA-LRU, LRU, and GRU for
`ring_hold` and `torus_integrate`, seeds 0--2.  Any of the eight Exp88 tasks can
be added with another entry using the same schema.

## Commands

Run from `calru-paper/repro/experimental_v2` or any other directory:

```bash
python evaluate_checkpoint_dynamics.py --dry-run

ARTIFACT_ROOT=/path/to/legacy-artifact-root
OUTPUT_DIR=/path/to/fresh/calru-dynamics-smoke
python evaluate_checkpoint_dynamics.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --device cpu \
  --smoke \
  --output-dir "$OUTPUT_DIR"

OUTPUT_DIR=/path/to/fresh/calru-dynamics-full
python evaluate_checkpoint_dynamics.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --device cuda:0 \
  --tasks ring_hold,torus_integrate \
  --models CA-LRU,LRU,GRU \
  --seeds 0,1,2 \
  --output-dir "$OUTPUT_DIR"
```

When the paper repository is checked out directly inside the legacy artifact
root, `--artifact-root` defaults to the repository's parent directory. In any
other layout, pass it explicitly.

To export a compatible spec from a completed P0 campaign manifest:

```bash
CAMPAIGN_DIR=/path/to/p0-campaign
python export_analysis_specs.py \
  --campaign-manifest "$CAMPAIGN_DIR/manifest.json" \
  --seeds 0,1,2 \
  --require-artifacts \
  --output "$CAMPAIGN_DIR/analysis_specs.json"

python evaluate_checkpoint_dynamics.py \
  --artifact-root "$CAMPAIGN_DIR" \
  --specs "$CAMPAIGN_DIR/analysis_specs.json" \
  --models rp_aligned,rg_lru_main,matched_gru_anchor \
  --seeds 0,1,2 \
  --output-dir /path/to/fresh/p0-dynamics
```

Use `export_analysis_specs.py --legacy-root "$ARTIFACT_ROOT"` to export the
checked-in legacy template instead. One evaluator invocation uses one artifact
root; run legacy and campaign checkpoints separately. Exported specs also
record the SHA-256 of the source campaign manifest or legacy template.

`--smoke` always uses CPU, reduces the assets and horizons, and evaluates only
the first matching checkpoint/seed.  `--dry-run` performs discovery and input
validation without loading PyTorch checkpoints or writing output.

The output directory must not already exist and must not be inside any legacy
result/checkpoint directory.  The evaluator never writes beneath the legacy
artifact directories.  A completed run contains:

```text
assets/*.npy
asset_manifest.json
checkpoint_metrics.csv
blank_flow_metrics.csv
perturbation_metrics.csv
perturbation_summary.csv
jacobian_directional_metrics.csv
jacobian_directional_summary.csv
results.json
SHA256SUMS
COMPLETE
```

`results.json` records the complete analysis configuration plus the SHA-256
of every source JSON/checkpoint. All source JSON, checkpoint, spec, and code
hashes are frozen before evaluation and rechecked before finalization.
`SHA256SUMS` covers every generated output except itself and lifecycle markers;
it is atomically published first and `COMPLETE` is atomically published last.
A failed or interrupted run is not complete and normally retains `INCOMPLETE`.

## Metric conventions

- MSE is ambient component MSE, not vector RMSE.
- `clean_subtracted_target_mse` may be negative; it is a paired difference,
  not clipped at zero.
- For a full-block model decoded from the transient stream, the H=0 decoded
  kick effect is exactly zero by construction: only the carrier was changed.
  H>=1 measures how that carrier perturbation affects subsequent blank flow.
- `nearest_family_distance_normalized` divides carrier distance by the median
  nearest-neighbor spacing of the clean carrier family.
- Angular intrinsic-coordinate error uses wrapped differences.  Surface
  coordinates use ordinary Euclidean differences.
- Standard deviations use the sample convention (`ddof=1`) when at least two
  observations are present.

Run the focused unit tests with:

```bash
python -m unittest -v repro.experimental_v2.tests.test_checkpoint_dynamics
```
