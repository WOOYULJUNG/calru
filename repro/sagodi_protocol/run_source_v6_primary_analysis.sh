#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 BASELINE_ROOT LRU_ROOT STATE_NOISE_ROOT ARTIFACT_ROOT [GPU_IDS]" >&2
  exit 2
fi

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$(git rev-parse --show-toplevel)"
python -m repro.sagodi_protocol.source_v6_primary_analysis_campaign \
  --baseline-root "$1" --lru-root "$2" --state-noise-root "$3" \
  --artifact-root "$4" --gpus "${5:-0,1,2,3,4,5}"
