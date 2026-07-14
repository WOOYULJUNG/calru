# Baseline-bound LRU → No-RP → CA-LRU v6 동결

이 launcher는 repaired baseline v6의 별도 downstream campaign이다. baseline
main의 계산·receipt·per-model eligibility report가 검증되어야 시작하지만,
RNN/GRU/LSTM 중 zero-eligible 모델이 있어도 LRU 실험은 계속한다.

## 동일 데이터 계약

- baseline의 `tuning.npz`, `main_test.npz`와 `.sha256`을 byte-identical하게
  복사하고 source/destination hash를 parent binding에 기록한다.
- T128 variable-sparsity, source q1 post-update initial memory를 그대로 쓴다.
- online batch key는 새 campaign 이름이 아니라
  `(sagodi_source_repaired_baselines_v6, online_train, seed, update)`다.
- 학습은 Adam/B64/5,000/WD0/constant LR이며 평가 state noise는 0이다.
- 실제 noise grid `[0,.01,.0316228,.1]`은 transition 뒤 coordinate별 iid
  표준편차다. 같은 std가 전체 energy가 같다는 뜻은 아니다. LRU는 104차
  Re/Im causal carrier, No-RP/CA는 52차 real causal carrier에 주입한다.

## 강제 순서

1. LRU: 16-cell sentinel(seed100) → top3 × seeds101--104 → fresh main 0--9.
2. LRU main에서 NMSE < -20 dB seed가 하나 이상일 때만 No-RP를 시작한다.
3. No-RP: 같은 16-cell/fanout/main 절차로 LR/noise를 독립 선택한다.
4. CA-LRU는 No-RP main의 **verified computation** 뒤 시작한다. No-RP가
   zero-eligible여도 RP가 eligibility를 만들 수 있으므로 CA를 막지 않는다.
5. CA-LRU는 No-RP LR/noise를 그대로 상속한다. RP 3 eta × 3 epsilon을
   sentinel/fanout으로 선택하고 fresh main 0--9를 실행한다.

LR/noise 및 RP 선택은 NMSE<-20 eligible count를 먼저 최대화하고,
MSE<.01 descriptive count, MSE/blank-memory metric, frozen grid order 순으로
tie-break한다. 누락 seed가 있는 cell은 선택할 수 없다. 모든 main seed와
eligibility rate를 보고한다.

No-RP와 CA는 같은 seed에서 동일 initialization, online batch, state-noise
generator를 사용한다. RP 전 update는 state dict와 optimizer update가 같아야
한다. CA의 LR/noise를 별도 tuning하지 않는다.

## 실행

downstream smoke도 completed baseline parent와 bank를 요구한다.
smoke는 모델별 2 updates만 실행하므로 CA의 RP schedule은 비어 있고
`rp_trace=[]`, `rp_call_count=0`이어야 한다. 이때 1,024-example bank의
4,096-step blank metric은 의도적으로 생략해 `heldout_blank_memory_mse=null`로
기록한다. RP API 자체는 별도의 reduced-horizon 단위 테스트에서 finite output과
모든 retention-theta 갱신을 확인한다. 이 생략은 full CA stage에는 적용되지 않는다.

```bash
python -m repro.sagodi_protocol.source_repaired_lru_calru_v6 \
  --stage smoke --baseline-root /path/to/baseline_v6 \
  --artifact-root /path/to/downstream_v6 --gpus cpu

# 이후 순서대로
# lru_sentinel, lru_fanout, lru_main,
# no_rp_sentinel, no_rp_fanout, no_rp_main,
# ca_rp_sentinel, ca_rp_fanout, ca_rp_main
```

OOM/SIGKILL/interrupt는 scientific failure로 합성하지 않는다. numerical
non-finite만 denominator-bearing failure다. full stage는 clean commit만 허용하고
worker 시작/종료 때 commit, runtime hashes, config/freeze hashes를 다시 검사한다.
