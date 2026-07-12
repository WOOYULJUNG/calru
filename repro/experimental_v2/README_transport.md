# Multi-point ring transport v2

`evaluate_ring_transport_v2.py` reevaluates the three Exp88 writer variants
from stabilized clean ring states. It perturbs no stream coordinates and
compares only the persistent recurrent carrier with the clean reference
family. Raw carrier distances are accompanied by distances normalized by the
local clean-manifold tangent scale.

The default matrix contains 32 base angles, both directions, velocity scales
1 and 2, move lengths 5 and 20, three writers, three training seeds, and five
blank horizons. It produces 2,304 drive conditions and 11,520 CSV rows.

Run a path-only check first:

```bash
OUTPUT_DIR=/path/to/fresh/calru-ring-transport-v2
python repro/experimental_v2/evaluate_ring_transport_v2.py \
  --output-dir "$OUTPUT_DIR" \
  --dry-run
```

Then run the analysis in a new output directory:

```bash
ARTIFACT_ROOT=/path/to/legacy-artifact-root
OUTPUT_DIR=/path/to/fresh/calru-ring-transport-v2
python repro/experimental_v2/evaluate_ring_transport_v2.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0
```

The output manifest records the code commit and SHA-256 of every checkpoint,
source JSON, code source, and generated CSV. Existing Exp88 artifacts are
read-only. The output path must not already exist, even if it is empty. Input
hashes are frozen before evaluation and rechecked before completion;
`manifest.json` and `SHA256SUMS` are atomic, and `COMPLETE` is written last.
