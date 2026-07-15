# Ságodi 공개 코드 중심 repaired baseline v6 동결 문서

## 주장 범위

이 경로는 **public-code-architecture noise-free controlled training with
paper/code noise provenance and documented repairs, not exact**이다. 논문
Protocol A의 exact reproduction도, 공개
저장소를 bit-exact하게 실행한 것도 아니다. 공개 코드에는 하나의 공통 실행
계약이 없으므로 T128/B64/5k 공통 비교 계약 자체가 controlled adaptation이다.
이전 primary-v4 및 source-resolved-v5 산출물은 덮어쓰지 않는다.

대상은 RNN/GRU/LSTM 세 baseline뿐이다. 이 경로가 끝난 뒤 별도 LRU 탐색을
같은 task와 fixed bank에서 수행하고, LRU에서 NMSE < -20 dB인 seed가 하나
이상 확인된 뒤에만 No-RP/CA-LRU 쌍을 실행한다. 따라서 이 baseline 경로에
LRU나 CA-LRU 결과를 섞어 보고하지 않는다.

## 고정한 공개 코드 근거

- 저장소: `catniplab/back_to_the_continuous_attractor`
- commit: `cbd7404e9baca4b2dc291560cfc6576bb7b1f078`
- 공통 task: `tasks.py:65-108`
- RNN: `models.py:104-319`, `models.py:393-629`,
  `run_training_s.py:233-248`
- GRU: `gru.py:35-80`, `gru.py:119-196`, `gru.py:198-211`
- LSTM: `lstm.py:65-162`, `lstm.py:201-267`, `lstm.py:277-288`

공통 task는 T=128, dt=0.1, trajectory별 sparsity `U(0,2)`, random q0,
post-update q1 target으로 초기 state를 만드는 공개 코드 규약이다. 학습은
online batch 64, Adam, constant LR, early stopping 없음이다. Pilot LR screen은
2,000 updates, 선택된 main만 5,000 updates를 사용한다.

| 모델 | 공개 경로 | v6 controlled adaptation |
|---|---|---|
| RNN | root YAML/runner는 T100, H200, B128, 1,000 updates, StepLR와 early-stop을 소비한다 | T128, H128, B64, 5,000 constant-LR updates, no early-stop, clean q1/loss, state/target/dropout 0에서 LR만 선택 |
| GRU | T128/H128/B64/5k helper, target noise .01, dropout .5, WD 1e-4, clip 100 | public target noise/dropout은 provenance만 유지하고 둘 다 0, state noise 0, clean q1/loss, map 초기화 |
| LSTM | T128/H64/B64/5k helper, target noise .01, WD .01 default, uninitialized maps와 cell-map typo | public target noise는 provenance만 유지하고 0, state noise/dropout 0, clean q1/loss, map/cell typo repair, WD=0 |

따라서 “공개 코드 recipe 그대로 세 모델을 재현했다”라고 쓰지 않는다.

## 명시적 repair

- 공개 코드의 초기화되지 않은 output-to-state map을 명시적으로 초기화한다.
- LSTM cell은 독립 output-to-cell map을 사용한다. 공개 training forward의
  `c_init = tanh(h_init)` 오타를 analysis path와 일치하도록 고친다.
- LSTM recurrent weight decay는 공개 값 0.01에서 0으로 고친다. 동일 조건
  A/B에서 0.01은 collapse했고 WD=0은 학습되었다.
- RNN recurrent bias는 zero로 고친다. 공개 path는 zero로 만든 직후
  `Uniform[-sqrt(H),sqrt(H)]`로 덮어쓴다. 등록된 seed 100, T128/q1,
  LR=0.001, 5,000-update parity에서 zero/noise0은 MSE 0.00576372,
  NMSE -19.3827 dB였고, 공개 literal bias의 최선은 MSE 0.0129422,
  NMSE -15.87 dB였다. 이 수치는 initializer 선택용 audit이며 본 결과가
  아니다.

공개 코드에서 GRU/LSTM의 target noise는 0.01이지만, Ságodi 논문의 §4.1과
S7.3 Eq. 61은 perturbation을 hidden state `x`에 두고 S7.5의 loss는 clean
target MSE를 사용한다. 따라서 공개 값 0.01은 provenance metadata에만 남기고
controlled primary의 **effective target-noise std는 RNN/GRU/LSTM 모두 0**으로
고정한다. 모든 모델은 코드 순서 그대로 clean `[cos θ, sin θ]` target의 첫 시점 `q1`로 초기 state를
구성하고 같은 clean target으로 loss를 계산한다.

controlled primary에서는 output dropout도 세 모델 모두 0이다. 공개 GRU의
dropout 0.5는 provenance에만 남긴다. 다만 GRU recurrent WD 1e-4와 clip 100,
LSTM clip 1 등 deterministic optimizer regularization은 유지한다. RNN WD는 0이며
clip은 없다. 즉 public target noise/dropout 제거는 명시적인 controlled deviation이고
exact public-code reproduction이 아니다.

추가 numerical deviation도 등록한다. upstream global RNG 대신 keyed
`numpy.default_rng` task stream과 seed가 고정된 torch initialization을 쓰며,
GP는 1e-6 jitter를 더한 NumPy Cholesky로 생성한다. 공식 pipeline은 CPU numerical thread를 모두 1로
고정한다. archive/sidecar SHA는 exact하게 묶고, 다른 BLAS의 독립 재생성은
float32 한 ULP의 relative tolerance와 `atol=1e-6`으로 대조한다(mask는 exact). adapter는
batch-first `nn.GRU/LSTM` full forward 대신 time-major cell unroll을 사용한다.
controlled dropout이 0이므로 dropout RNG draw는 없다.

## target/state noise 표기

controlled training과 evaluation에서 **state noise, target noise, output dropout은
세 모델 모두 0**이다. state/target/dropout RNG stream은 disabled, seed null이며
generator를 만들거나 draw하지 않는다. online task batch 자체는 계속 새로 생성되므로
이를 deterministic full-batch training이라고 부르지는 않는다.

noise 관련 원문 차이는 실행값과 분리해 provenance로만 기록한다. Ságodi S7.3의
`ζ ~ N(0, 0.01 I)`는 direct per-coordinate std 0.1을 뜻하지만 이번 controlled
primary에서는 실행하지 않는다. pinned 공개 저장소의 shipped RNN config는 nominal
state noise 0이고 effective noise도 0이다(`params_ang_sparse.yml:13-14`;
transition은 `models.py:333-349`). RNN 코드 경로에 nominal 0.1을 가정하면
`0.1*sqrt(dt=0.1)=0.0316228`이 되지만 이는 **hypothetical formula provenance**일
뿐 실행값이나 shipped-config 값이 아니다.
이와 별개로 이전 내부 `sagodi_source_resolved_v1` recipe는 RNN nominal 0.1,
effective 0.0316228을 담고 있었고 현재 v6는 model build 시 이를 명시적으로
0/0으로 override한다. 이 내부 recipe provenance를 shipped 공개 YAML 또는 논문
literal std와 혼동하지 않는다.
GRU/LSTM 공개 실행 경로에는 대응하는 nominal state-noise 설정/API가 없으므로
해당 provenance는 0이 아니라 `N/A(null)`이며, effective 실행값만 0으로 기록한다.

## 단계와 선택

1. `smoke`: 모델당 2 updates CPU/GPU 실행 검증.
2. `sentinel`: seed 100에서 LR
   `[3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]`, 총 24 runs.
3. `fanout`: 모든 LR을 seed 101로 한 번 더 확인한다. Sentinel과 fanout은
   모두 2,000 updates이며 fanout은 총 24 runs다. 각 LR은 정확히 2 seeds로 선택된다.
4. `main`: 선택된 cell을 5,000 updates, fresh seeds 0--2로 총 9 runs.

Tuning bank와 main bank는 서로 다르며, 모든 모델/cell은 같은 seed와
update에서 동일한 online batch를 공유한다. 실패나 누락을 분모에서
숨기지 않는다. OOM, SIGKILL, user interrupt 같은 infrastructure failure는
scientific seed failure로 합성하지 않고 retryable 상태로 둔다.

LR grid는 공개/논문 rate `1e-4`, `1e-5`, 과거 LSTM lower-bound 징후, LRU의
upper-bound 탐색 `0.03`을 모두 포함한 union이다. 모든 모델에 동일한 8-LR grid와
2-seed/2,000-update pilot 선택 예산을 적용한다.

각 run manifest/checkpoint는 optimizer step 전 named tensor의 canonical
`initial_state_dict_sha256`과 online data, target-noise, state-noise, dropout 및
RP stream identity를 기록한다. 세 noise/dropout stream은 모두 disabled, seed null,
clean-q1/clean-loss로 기록한다.

## 보고 및 gate

모든 pilot main seed 3개를 보고한다. MSE < 0.01은 success count/rate로만
기술한다. 분석 eligibility는 seed별 NMSE < -20 dB이다. 모델마다 eligible
seed가 하나도 없을 때만 scientific gate가 실패한다. 1개뿐이면 pass와
동시에 low-n warning을 남기며, 후속 동역학 분석은 eligible subset에만
적용한다. `COMPUTATION_COMPLETE`는 `SCIENTIFIC_PASS`가 아니다.

## 실행

```bash
python -m repro.sagodi_protocol.source_repaired_baselines_v6 \
  --stage smoke --artifact-root /path/to/v6 --gpus cpu
python -m repro.sagodi_protocol.source_repaired_baselines_v6 \
  --stage sentinel --artifact-root /path/to/v6 --gpus 0,1
python -m repro.sagodi_protocol.source_repaired_baselines_v6 \
  --stage fanout --artifact-root /path/to/v6 --gpus 0,1
python -m repro.sagodi_protocol.source_repaired_baselines_v6 \
  --stage main --artifact-root /path/to/v6 --gpus 0,1
```

full stage는 clean committed worktree에서만 시작한다. child receipt,
checkpoint/result identity, fixed-bank digest와 deterministic content,
parent receipt/selection hash가 모두 맞아야 resume/finalization이 유효하다.
