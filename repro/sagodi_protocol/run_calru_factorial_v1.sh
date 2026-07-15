#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 BASELINE_ROOT LRU_ROOT STATE_NOISE_ROOT ARTIFACT_ROOT [GPU_IDS]" >&2
  exit 2
fi

baseline_root=$1
lru_root=$2
state_noise_root=$3
artifact_root=$4
gpu_ids=${5:-0,1,2,3,4,5}

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$(git rev-parse --show-toplevel)"

echo "pipeline_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "code_commit=$(git rev-parse HEAD)"
python -m repro.sagodi_protocol.calru_factorial_v1 --stage smoke --baseline-root "$baseline_root" --lru-root "$lru_root" --state-noise-root "$state_noise_root" --artifact-root "$artifact_root" --gpus cpu
for stage in rp_sentinel rp_fanout factorial_main; do
  python -m repro.sagodi_protocol.calru_factorial_v1 --stage "$stage" --baseline-root "$baseline_root" --lru-root "$lru_root" --state-noise-root "$state_noise_root" --artifact-root "$artifact_root" --gpus "$gpu_ids"
done
echo "pipeline_complete_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
