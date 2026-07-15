#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 BASELINE_PARENT DOWNSTREAM_PARENT ARTIFACT_ROOT [GPU_IDS]" >&2
  exit 2
fi

baseline_parent=$1
downstream_parent=$2
artifact_root=$3
gpu_ids=${4:-0,1,2,3,4,5}

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

echo "pipeline_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "code_commit=$(git rev-parse HEAD)"
echo "baseline_parent=$baseline_parent"
echo "downstream_parent=$downstream_parent"
echo "artifact_root=$artifact_root"
echo "gpu_ids=$gpu_ids"

python -m repro.sagodi_protocol.state_noise_search_v1 \
  --stage smoke --baseline-root "$baseline_parent" \
  --downstream-root "$downstream_parent" --artifact-root "$artifact_root" --gpus cpu
python -m repro.sagodi_protocol.state_noise_search_v1 \
  --stage tuning --baseline-root "$baseline_parent" \
  --downstream-root "$downstream_parent" --artifact-root "$artifact_root" --gpus "$gpu_ids"
python -m repro.sagodi_protocol.state_noise_search_v1 \
  --stage main --baseline-root "$baseline_parent" \
  --downstream-root "$downstream_parent" --artifact-root "$artifact_root" --gpus "$gpu_ids"

echo "pipeline_complete_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
