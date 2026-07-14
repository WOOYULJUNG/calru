# Ságodi 공개 코드 중심 repaired baseline v6 동결 문서

## 주장 범위

이 경로는 **public-code-centered controlled adaptation with documented
repairs, not exact**이다. 논문 Protocol A의 exact reproduction도, 공개
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
online batch 64, Adam, constant LR, early stopping 없음, 5,000 updates이다.

| 모델 | 공개 경로 | v6 controlled adaptation |
|---|---|---|
| RNN | root YAML/runner는 T100, H200, B128, 1,000 updates, StepLR와 early-stop을 소비한다 | T128, H128, B64, 5,000 constant-LR updates, no early-stop, q1 map, LR/noise sweep |
| GRU | T128/H128/B64/5k helper, target noise .01, dropout .5, WD 1e-4, clip 100 | 모델별 recipe는 유지하지만 map을 초기화하고 LR/실제 state-noise를 공통 grid에서 선택 |
| LSTM | T128/H64/B64/5k helper, target noise .01, WD .01 default, uninitialized maps와 cell-map typo | map/cell typo repair, WD=0, LR/실제 state-noise 공통 grid |

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

GRU의 target noise 0.01, dropout 0.5, recurrent WD 1e-4, clip 100과
LSTM의 target noise 0.01, dropout 0, clip 1은 공개 model별 recipe를
유지한다. RNN target noise/dropout/WD는 0이며 clip은 없다.

추가 numerical deviation도 등록한다. upstream global RNG 대신 keyed
`numpy.default_rng`와 분리된 torch generator를 쓰며, GP는 1e-6 jitter를 더한
NumPy Cholesky로 생성한다. archive/sidecar SHA는 exact하게 묶고, 다른 BLAS의
Cholesky 재계산은 `rtol=0, atol=1e-6`으로만 대조한다(mask는 exact). adapter는
batch-first `nn.GRU/LSTM` full forward 대신 time-major cell unroll을 사용한다.
특히 GRU dropout의 난수 draw order는 upstream full-sequence dropout과
bit-exact하다고 주장하지 않는다.

## state noise 표기

탐색축은 실제 transition 이후 state에 더해지는 표준편차다:

`[0, 0.01, 0.0316228, 0.1]`.

RNN 공개 API의 nominal 값은 `actual / sqrt(dt)`로 별도 기록한다. GRU/LSTM
공개 API에는 대응 nominal state-noise 인자가 없으므로 null과
`not_applicable`을 기록한다. 세 모델 모두 실제 주입 위치와 크기는 같다.

## 단계와 선택

1. `smoke`: 모델당 2 updates CPU/GPU 실행 검증.
2. `sentinel`: seed 100에서 LR
   `[1e-2, 3e-3, 1e-3, 3e-4]` × actual noise 4개, 총 48 runs.
3. `fanout`: 모델별 deterministic top 3 cell을 seeds 101--104로 확장,
   총 36 runs. 누락 seed가 있는 cell은 선택할 수 없다.
4. `main`: 선택된 cell을 fresh seeds 0--9로 총 30 runs.

Tuning bank와 main bank는 서로 다르며, 모든 모델/cell은 같은 seed와
update에서 동일한 online batch를 공유한다. 실패나 누락을 분모에서
숨기지 않는다. OOM, SIGKILL, user interrupt 같은 infrastructure failure는
scientific seed failure로 합성하지 않고 retryable 상태로 둔다.

## 보고 및 gate

모든 main seed 10개를 보고한다. MSE < 0.01은 success count/rate로만
기술한다. 분석 eligibility는 seed별 NMSE < -20 dB이다. 모델마다 eligible
seed가 하나도 없을 때만 scientific gate가 실패한다. 1--2개뿐이면 pass와
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
