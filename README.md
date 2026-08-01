# RP-LRU

Minimal reproduction code for Retention Plasticity (RP) and RP-LRU.
The repository contains code and the fixed experimental protocol only; it
does not contain checkpoints, logs, figures, or result tables.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Integration with delays

Train the paper configuration (`N=91`, `K=50`, `epsilon=0.20`):

```bash
rplru-train \
  --model rp_lru --dimension 8 --learning-rate 3e-4 --seed 0 \
  --retention-mode rp --initial-lambda-low 0.98 --initial-lambda-high 0.999 \
  --rp-eta-lambda 1 --rp-retention-threshold 0.20 --rp-probe-horizon 50 \
  --output-dir runs/rp_lru_d8_seed0
```

Evaluate the fixed `6 x 6` update-count/hold-length grid:

```bash
rplru-eval \
  --run-dir runs/rp_lru_d8_seed0 \
  --output-dir runs/rp_lru_d8_seed0/ood
```

Run the tangent/normal perturbation diagnostic:

```bash
rplru-geometry \
  --run-dir runs/rp_lru_d8_seed0 \
  --output-dir runs/rp_lru_d8_seed0/geometry
```

The same training command supports `rnn`, `gru`, `lstm`, `lru`,
`chrono_lstm`, `orthogonal_rnn`, and `s4d_legs`. If `--width` is omitted,
the code selects the closest parameter-matched width.

## Spatial localization

```bash
rplru-spatial-train \
  --model rp_lru --bins 3 --seed 0 --learning-rate 1e-3 \
  --rp-threshold 0.09 --rp-eta 1 --rp-interval 50 \
  --output-dir runs/spatial_3x3_seed0

rplru-spatial-eval \
  --run-dir runs/spatial_3x3_seed0 \
  --output-dir runs/spatial_3x3_seed0/eval
```

Use `--bins 5 --rp-threshold 0.12` for the `5 x 5` task. All other frozen
settings, seeds, task ranges, and evaluation conditions are in
[`src/rplru/protocol.json`](src/rplru/protocol.json).

## Test

```bash
pytest
```
