#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 OUTPUT_ROOT [CUDA_DEVICES]" >&2
  exit 2
fi

output_root="$1"
devices="${2:-0,1,2,3,4,5}"
mkdir -p "$output_root"
master_log="$output_root/campaign.log"

attempt=1
maximum_attempts=3
while (( attempt <= maximum_attempts )); do
  echo "=== campaign attempt ${attempt}/${maximum_attempts} $(date --iso-8601=seconds) ===" | tee -a "$master_log"
  PYTHONPATH=. python -m repro.manifold_benchmark.launch_topology_hparam \
    --phase all \
    --devices "$devices" \
    --config repro/manifold_benchmark/topology_hparam_v1.json \
    --output "$output_root" 2>&1 | tee -a "$master_log"
  status=${PIPESTATUS[0]}
  if (( status == 0 )); then
    echo "=== campaign finished successfully $(date --iso-8601=seconds) ===" | tee -a "$master_log"
    exit 0
  fi
  echo "=== campaign exited ${status}; restartable retry follows ===" | tee -a "$master_log"
  attempt=$((attempt + 1))
  if (( attempt <= maximum_attempts )); then
    sleep 10
  fi
done

echo "=== campaign exhausted wrapper retries; inspect CAMPAIGN_STATUS.json ===" | tee -a "$master_log"
exit 1
