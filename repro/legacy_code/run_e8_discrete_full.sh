#!/usr/bin/env bash
# E8: run the complete discrete battery for seeds 0-2.
# Historical model/tag strings are retained for raw-result compatibility.
# Runs are sequential on one GPU and skip result files that already exist.
set -u
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PY="${PYTHON:-python}"

GPU="${1:-0}"
OUT=exp89_discrete_kway_quick_results
CKPT=checkpoints_exp89_discrete_kway_quick
TRACE=traces_exp89_discrete_kway_quick
LOGS=logs_exp89_discrete_full
mkdir -p "$LOGS"
FAILURES=0

run_job () {
  local task="$1" tag="$2" model="$3" seed="$4"; shift 4
  local json="$OUT/${task}_k16_${tag}_seed${seed}.json"
  if [ "$task" = "flipflop" ]; then
    json="$OUT/${task}_n4_${tag}_seed${seed}.json"
  fi
  if [ -f "$json" ]; then
    echo "[skip] $json exists"
    return 0
  fi
  echo "[run] task=$task tag=$tag model=$model seed=$seed gpu=$GPU $(date -Is)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" exp74_discrete_attractor_tasks.py \
    --task "$task" --model "$model" --tag "$tag" --seed "$seed" \
    --steps 3000 --k 16 --bits 4 \
    --out-dir "$OUT" --ckpt-dir "$CKPT" --trace-dir "$TRACE" \
    "$@" > "$LOGS/${task}_${tag}_seed${seed}.log" 2>&1 \
    || {
      echo "[failed] task=$task tag=$tag seed=$seed"
      FAILURES=$((FAILURES + 1))
    }
}

# flipflop: CAMN + baselines, seeds 0-2
for seed in 0 1 2; do
  run_job flipflop camn_eps3e-5 PAN-RNW-full "$seed" --pan-score-eps 3e-5
  run_job flipflop camn_eps1e-4 PAN-RNW-full "$seed" --pan-score-eps 1e-4
  run_job flipflop rnn RNN "$seed"
  run_job flipflop gru GRU "$seed"
  run_job flipflop lstm LSTM "$seed"
  run_job flipflop lru lru-full "$seed"
done

# kway_hold K=16
for seed in 0 1 2; do
  run_job kway_hold camn_eps3e-5 PAN-RNW-full "$seed" --pan-score-eps 3e-5
  run_job kway_hold camn_eps1e-4 PAN-RNW-full "$seed" --pan-score-eps 1e-4
  run_job kway_hold rnn RNN "$seed"
  run_job kway_hold gru GRU "$seed"
  run_job kway_hold lstm LSTM "$seed"
  run_job kway_hold lru lru-full "$seed"
done

if [ "$FAILURES" -ne 0 ]; then
  echo "[e8 incomplete] $FAILURES job(s) failed" >&2
  exit 1
fi

echo "[e8 done] $(date -Is)"
