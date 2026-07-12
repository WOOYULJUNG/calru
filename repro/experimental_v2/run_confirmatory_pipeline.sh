#!/usr/bin/env bash
set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)

: "${ARTIFACT_ROOT:?Set ARTIFACT_ROOT to a fresh or marked v2 training root}"
: "${ANALYSIS_ROOT:?Set ANALYSIS_ROOT to a fresh analysis output root}"

PYTHON=${PYTHON:-python}
GPUS=${GPUS:-0,1,2,3,4,5}
EXPECTED_COMMIT=${EXPECTED_COMMIT:-}
LOG_ROOT=${LOG_ROOT:-"$ARTIFACT_ROOT/pipeline_logs"}
CONFIG=${CONFIG:-"$HERE/campaign.example.json"}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

mkdir -p "$LOG_ROOT"
STATUS_FILE="$LOG_ROOT/pipeline_status.tsv"

status() {
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" | tee -a "$STATUS_FILE"
}

fail() {
  status pipeline "failed: $*"
  exit 1
}

cd "$REPO_ROOT"
COMMIT=$(git rev-parse HEAD)
if [[ -n "$EXPECTED_COMMIT" && "$COMMIT" != "$EXPECTED_COMMIT" ]]; then
  fail "commit mismatch ($COMMIT != $EXPECTED_COMMIT)"
fi
if [[ -n "$(git status --porcelain)" ]]; then
  fail "repository worktree is not clean"
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
if [[ ${#GPU_ARRAY[@]} -lt 1 ]]; then
  fail "GPUS must contain at least one device"
fi

status preflight started
"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": torch.cuda.device_count()})
PY
nvidia-smi -q > "$LOG_ROOT/nvidia-smi-start.txt"
"$PYTHON" -m pip freeze --all > "$LOG_ROOT/environment.freeze.txt"
sha256sum "$CONFIG" "$HERE/environment.freeze.txt" > "$LOG_ROOT/input_sha256.txt"
printf '%s\n' "$COMMIT" > "$LOG_ROOT/git_commit.txt"
status preflight complete

status training started
set +e
"$PYTHON" "$HERE/launch_p0.py" \
  --config "$CONFIG" \
  --artifact-root "$ARTIFACT_ROOT" \
  --gpus "$GPUS" \
  --max-parallel "${#GPU_ARRAY[@]}" \
  --enable rg_lru_main \
  --enable rg_lru_aux_h500 \
  --enable matched_gru_anchor \
  --enable rg_lru_lr3e4_anchor \
  --enable rg_lru_lr3e3_anchor \
  --disable lru_aux_h500 \
  2>&1 | tee "$LOG_ROOT/training.log"
TRAIN_CODE=${PIPESTATUS[0]}
set -e
if [[ $TRAIN_CODE -ne 0 ]]; then
  fail "training launcher exited $TRAIN_CODE"
fi
status training complete

mapfile -t CAMPAIGNS < <(find "$ARTIFACT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'p0_anchor_v1-*' | sort)
if [[ ${#CAMPAIGNS[@]} -ne 1 ]]; then
  fail "expected exactly one p0_anchor_v1 campaign, found ${#CAMPAIGNS[@]}"
fi
CAMPAIGN_DIR=${CAMPAIGNS[0]}
"$PYTHON" "$HERE/status_p0.py" "$CAMPAIGN_DIR" | tee "$LOG_ROOT/training_status.txt"

if [[ -e "$ANALYSIS_ROOT" ]]; then
  fail "analysis root already exists: $ANALYSIS_ROOT"
fi
mkdir -p "$ANALYSIS_ROOT"
printf '%s\n' "$CAMPAIGN_DIR" > "$ANALYSIS_ROOT/CAMPAIGN_DIR"

status analysis_specs started
"$PYTHON" "$HERE/export_analysis_specs.py" \
  --campaign-manifest "$CAMPAIGN_DIR/manifest.json" \
  --seeds 0,1,2 \
  --require-artifacts \
  --output "$ANALYSIS_ROOT/campaign_dynamics_specs.json" \
  > "$LOG_ROOT/export_analysis_specs.log" 2>&1
status analysis_specs complete

declare -A PIDS
declare -A ANALYSIS_LOGS

run_analysis() {
  local name=$1
  local gpu=$2
  shift 2
  local log="$LOG_ROOT/${name}.log"
  status "$name" "started on gpu${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "$@" > "$log" 2>&1 &
  PIDS["$name"]=$!
  ANALYSIS_LOGS["$name"]=$log
}

run_analysis fixed_all "${GPU_ARRAY[0]}" \
  "$PYTHON" "$HERE/evaluate_fixed_performance.py" \
  --campaign-dir "$CAMPAIGN_DIR" \
  --output-dir "$ANALYSIS_ROOT/fixed_all" \
  --device cuda:0

PERM_GPU=${GPU_ARRAY[1]:-${GPU_ARRAY[0]}}
run_analysis fixed_rp_permutation "$PERM_GPU" \
  "$PYTHON" "$HERE/evaluate_fixed_performance.py" \
  --campaign-dir "$CAMPAIGN_DIR" \
  --output-dir "$ANALYSIS_ROOT/fixed_rp_permutation" \
  --conditions rp_aligned \
  --retention-permutation \
  --device cuda:0

DYNAMICS_GPU=${GPU_ARRAY[2]:-${GPU_ARRAY[0]}}
run_analysis dynamics_campaign "$DYNAMICS_GPU" \
  "$PYTHON" "$HERE/evaluate_checkpoint_dynamics.py" \
  --artifact-root "$CAMPAIGN_DIR" \
  --specs "$ANALYSIS_ROOT/campaign_dynamics_specs.json" \
  --tasks ring_hold,torus_integrate \
  --models rp_aligned,rp_off_frozen_retention,uniform_retention_theta_cap,gradient_lambda_aux,rg_lru_aux_h500,rg_lru_main,matched_gru_anchor \
  --seeds 0,1,2 \
  --jacobian-horizons 1,20,100,500,1000 \
  --device cuda:0 \
  --output-dir "$ANALYSIS_ROOT/dynamics_campaign"

LEGACY_GPU=${GPU_ARRAY[3]:-${GPU_ARRAY[0]}}
run_analysis dynamics_legacy_lru "$LEGACY_GPU" \
  "$PYTHON" "$HERE/evaluate_checkpoint_dynamics.py" \
  --artifact-root "$REPO_ROOT/.." \
  --specs "$HERE/dynamics_specs.json" \
  --tasks ring_hold,torus_integrate \
  --models LRU \
  --seeds 0,1,2 \
  --jacobian-horizons 1,20,100,500,1000 \
  --device cuda:0 \
  --output-dir "$ANALYSIS_ROOT/dynamics_legacy_lru"

TRANSPORT_GPU=${GPU_ARRAY[4]:-${GPU_ARRAY[0]}}
run_analysis ring_transport "$TRANSPORT_GPU" \
  "$PYTHON" "$HERE/evaluate_ring_transport_v2.py" \
  --artifact-root "$REPO_ROOT/.." \
  --output-dir "$ANALYSIS_ROOT/ring_transport" \
  --device cuda:0

ANALYSIS_FAILED=0
for name in "${!PIDS[@]}"; do
  if wait "${PIDS[$name]}"; then
    status "$name" complete
  else
    code=$?
    status "$name" "failed exit=${code} log=${ANALYSIS_LOGS[$name]}"
    ANALYSIS_FAILED=1
  fi
done
if [[ $ANALYSIS_FAILED -ne 0 ]]; then
  fail "one or more checkpoint analyses failed"
fi

for directory in \
  "$ANALYSIS_ROOT/fixed_all" \
  "$ANALYSIS_ROOT/fixed_rp_permutation" \
  "$ANALYSIS_ROOT/dynamics_campaign" \
  "$ANALYSIS_ROOT/dynamics_legacy_lru" \
  "$ANALYSIS_ROOT/ring_transport"; do
  [[ -f "$directory/COMPLETE" ]] || fail "missing COMPLETE: $directory"
  (cd "$directory" && sha256sum -c SHA256SUMS >/dev/null) || fail "checksum failure: $directory"
done

nvidia-smi -q > "$LOG_ROOT/nvidia-smi-end.txt"
status pipeline complete
printf 'campaign=%s\nanalysis=%s\ncommit=%s\n' \
  "$CAMPAIGN_DIR" "$ANALYSIS_ROOT" "$COMMIT" | tee "$LOG_ROOT/PIPELINE_COMPLETE"
