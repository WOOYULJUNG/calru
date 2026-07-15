#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 BASELINE_PARENT LRU_ARTIFACT_ROOT STATE_NOISE_ARTIFACT_ROOT [GPU_IDS]" >&2
  exit 2
fi

baseline_parent=$1
lru_artifact_root=$2
state_noise_artifact_root=$3
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
echo "lru_artifact_root=$lru_artifact_root"
echo "state_noise_artifact_root=$state_noise_artifact_root"
echo "gpu_ids=$gpu_ids"

# Step 1 completion for the fourth baseline: tune and train LRU only.  The
# historical downstream runner must not continue into No-RP or CA-LRU here.
python -m repro.sagodi_protocol.source_repaired_lru_calru_v6 \
  --stage smoke --baseline-root "$baseline_parent" \
  --artifact-root "$lru_artifact_root" --gpus cpu
for stage in lru_sentinel lru_fanout lru_main; do
  python -m repro.sagodi_protocol.source_repaired_lru_calru_v6 \
    --stage "$stage" --baseline-root "$baseline_parent" \
    --artifact-root "$lru_artifact_root" --gpus "$gpu_ids"
done

# Step 2: freeze all four selected noise-free LRs and tune only state noise.
repro/sagodi_protocol/run_state_noise_search_v1.sh \
  "$baseline_parent" "$lru_artifact_root" "$state_noise_artifact_root" "$gpu_ids"

echo "pipeline_complete_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
