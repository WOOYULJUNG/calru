#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 FACTORIAL_ROOT ARTIFACT_ROOT [GPU_IDS]" >&2
  exit 2
fi

factorial_root=$1
artifact_root=$2
gpu_ids=${3:-0,1,2,3,4,5}

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$(git rev-parse --show-toplevel)"

echo "pipeline_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "code_commit=$(git rev-parse HEAD)"
python -m repro.sagodi_protocol.calru_factorial_analysis_v1 \
  --factorial-root "$factorial_root" \
  --artifact-root "$artifact_root" \
  --gpus "$gpu_ids"
echo "pipeline_complete_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
