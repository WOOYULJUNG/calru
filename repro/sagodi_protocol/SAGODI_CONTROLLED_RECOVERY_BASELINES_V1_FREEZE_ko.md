# Ságodi-informed auxiliary controlled benchmark v1 — 동결 문서

이 캠페인의 목적은 RNN/GRU/LSTM baseline이 공통 조건에서 먼저 재현되는지
확인하는 것이다. historical `primary_v4` 및 `source-resolved-v5` artifact의
과학적 identity는 변경하지 않는다. 이 캠페인은 새 ID
`sagodi_paper_baselines_v1`을 사용한다.

이름에 `paper`가 남아 있어도 이 경로는 **exact paper/public-code 재현이 아니다**.
primary인 pure-tanh + noise std `0.0316227766`은 baseline을 동일 조건에서
비교하기 위한 사전 등록된 auxiliary controlled benchmark다. 다음 세 수치 계약을
서로 바꾸어 부르지 않는다.

- benchmark primary: pure tanh + std `0.0316227766`
- recurrence-noise sensitivity: `dt=0.1` leaky tanh + std `0.1`
- recurrence-noise sensitivity: `dt=0.1` leaky tanh + std `0.0316227766`

이 benchmark는 논문과 공개 코드 커밋
`cbd7404e9baca4b2dc291560cfc6576bb7b1f078`에서 모델/task 구조를 가져오되
하나의 출처를 exact 실행했다고 주장하지 않는
`paper_and_source_informed_controlled_benchmark`다. 공개 runner/YAML에는 소비되는 key 29개 누락,
task name 불일치, import module 누락이 있고, RNN의
`Uniform[-sqrt(H),sqrt(H)]` recurrent bias 및 모델별 noise/WD/LR 계약도 서로
충돌한다. 이 문제를 숨긴 채 하나의 exact recipe로 부르지 않는다.

## 1. Primary task

- 1차원 angular integration
- 256 steps, `dt=0.1`
- length scale 1, 표준편차 1인 dense Gaussian-process velocity
- 각 trajectory의 실제 `q0`에서 `[cos(q0), sin(q0)]`를 만들고 학습 가능한
  output-to-state map으로 초기 상태를 구성
- target은 각 velocity update 뒤의 `q_{t+1}`
- 256개 모든 시점의 clean target에 대해 MSE 계산

## 2. 모델

- RNN: hidden 128, pure-tanh primary recurrence, recurrent gain 1.5,
  Xavier-normal input/readout/q0 map, recurrent bias 0
- GRU: PyTorch `GRUCell`, hidden 128, direct linear readout, tanh q0 map.
  input/recurrent/readout weight를 모두 Xavier-normal로 명시 초기화하고 모든
  bias를 0으로 초기화한다.
- LSTM: PyTorch `LSTMCell`, hidden 64, `[h,c]` 전체가 Markov state,
  서로 독립적인 tanh q0 map 두 개, `h` direct readout. input/recurrent/readout
  weight를 모두 Xavier-normal로 명시 초기화하고 모든 bias를 0으로 초기화한다.

RNN/GRU/LSTM의 trainable parameter 수는 각각 17,154 / 50,818 / 17,538이다.
RNN primary는 leaky update가 아니다. `dt=0.1` leaky RNN은 sensitivity로만
등록하고 primary LR을 선택할 수 없다.

## 3. 공통 학습 조건

- Adam, batch 64, 5,000 updates
- weight decay(WD) 0
- target noise 0, dropout 0, gradient clipping 0(비활성)
- 실제 deterministic transition을 계산한 **뒤** recurrent state에 표준편차
  `0.0316227766`의 Gaussian noise를 더함
- RNN/GRU는 `h`, LSTM은 `h`와 `c` 모두에 동일한 실제 표준편차를 적용
- `sqrt(dt)`를 다시 곱하지 않음
- online training batch는 모델 및 model seed와 무관하게 update index로 고정

여기서 WD(weight decay)는 optimizer가 큰 parameter를 벌점으로 줄이는
정규화다. 이 캠페인에서는 baseline별로 다르게 적용하지 않고 모두 0으로
고정한다.

## 4. LR screen과 main 판정

LR grid는 `1e-2, 1e-3, 1e-4, 1e-5`이고 각 LR을 seeds 100–104에서 각각
5,000 updates 전부 학습한다. 100-step 결과나 단일 sentinel seed는 gate로
사용하지 않는다.

각 full run의 trace에는 update 100 online training MSE도 남는다. 이를 이용해
논문의 100-update LR selector를 `paper_100_update_selector_audit.json`으로
재구성하지만 이는 **진단 전용**이다. primary LR은 아래의 사전 등록된
5,000-update/5-seed selector만 결정한다. 두 winner가 달라도 사후 교체하지 않는다.
이 audit은 paper의 selector 시점/집계 규칙을 benchmark trace에 적용한
것이며 paper 학습 결과의 재현이라고 해석하지 않는다.

모델별 LR은 다음 순서로 선택한다.

먼저 각 LR candidate가 **5 seeds 모두 valid completed outcome**을 가져야 한다.
4/5 이하이면 해당 candidate는 선택에서 제외한다. 한 모델에 eligible candidate가
하나도 없으면 selection은 `unavailable`이며 main/sensitivity plan을 만들지 않는다.

1. tuning-bank MSE가 0.01 미만인 seed 수 최대화
2. median MSE 최소화
3. mean MSE 최소화
4. 완전 동률이면 더 작은 LR

main은 선택된 LR로 seeds 0–9를 새로 학습하고, tuning과 완전히 분리된
main-test bank로 최종 checkpoint를 평가한다. 모든 10 seeds의 MSE<0.01
성공 수와 yield를 기술통계로 보고하며, 이것을 8/10 hard gate로 사용하지 않는다.
seed별 held-out `NMSE < -20 dB`일 때만 manifold analysis eligible이다. 모델별
eligible seed가 0이면 scientific hard fail이고, 1–7개이면 분석은 허용하되 강한
low-yield warning과 정확한 분모를 보고한다.

오직 계산이 끝났으나 수치적으로 non-finite가 된 child만 scientific numerical
failure로 분모에 남긴다. 이 경우 `failure.json`, `FAILED`,
`checkpoint_failure.pt`, exact failure receipt를 생성한다. OOM, I/O, process crash,
invalid receipt, SIGINT/KeyboardInterrupt는 과학적 실패가 아니라 retryable
infrastructure failure이며 denominator receipt를 만들지 않고 stage를 중단한다.

## 5. 고정 evaluation banks

- tuning: 1,024 trajectories, task seed 31,001
- main-test: 1,024 trajectories, task seed 31,002

두 bank는 서로 다른 stream key를 사용하며 생성 후 hash를 run manifest와
completion receipt에 결속한다. main은 screen receipt와 LR selection hash에도
결속한다. 로드시 shape만 검사하지 않고 task name/version, task seed, stream key,
256-step dense mask, `dt`, GP parameter, q0/target indexing metadata까지 검증한다.
생성 직후 bank와 SHA sidecar를 별도 registration receipt에 등록한다. BLAS별 GP
Cholesky의 미세한 차이를 새 과학적 bank로 오인하지 않도록 worker에서는 재생성
비교 대신 이 등록 digest, semantic metadata, receipt artifact 집합을 반복 검증한다.
각 child manifest도 bank registration과 registration-receipt hash에 결속한다.

## 6. Sensitivity 경계

다음 세 조건은 `--stage sensitivity`로만 실행한다.

- pure-tanh 전 모델의 post-transition state-noise std를 0.1로 변경
- RNN `dt=0.1` leaky-tanh + std 0.1
- RNN `dt=0.1` leaky-tanh + std `0.0316227766`

세 sensitivity는 모두 recurrence-noise sensitivity일 뿐 paper/public 재현 row가
아니다. screen에서 선택된 primary LR을 그대로 사용한다. sensitivity
결과로 primary LR, primary recurrence, primary noise를 사후 교체할 수 없다.

## 7. 실행

커밋 전에는 smoke만 허용한다. screen/main/sensitivity는 clean commit에서
실행해야 하며 별도의 빈 artifact root를 사용한다.

```bash
python -m repro.sagodi_protocol.sagodi_paper_baselines \
  --stage smoke --artifact-root /path/to/baseline_v1 --gpus 0,1,2

python -m repro.sagodi_protocol.sagodi_paper_baselines \
  --stage screen --artifact-root /path/to/baseline_v1 --gpus 0,1,2,3,4,5

python -m repro.sagodi_protocol.sagodi_paper_baselines \
  --stage main --artifact-root /path/to/baseline_v1 --gpus 0,1,2,3,4,5

python -m repro.sagodi_protocol.sagodi_paper_baselines \
  --stage sensitivity --artifact-root /path/to/baseline_v1 --gpus 0,1,2,3,4,5
```

screen은 60 runs, main은 30 runs, sensitivity는 50 runs다. 각 child는 config,
code, bank hash를 담은 manifest와 final checkpoint, result, trace, completion
receipt를 생성한다. 부분 실행은 보관한 뒤 verified receipt 단위로 재개한다.

각 child process는 BLAS 및 PyTorch intra/inter-op CPU thread를 1로 고정한다.
stage receipt는 `children_binding.json`을 통해 모든 child receipt와 성공/실패
checkpoint hash에 전이적으로 결속한다. child verifier는 outcome별 허용 artifact
집합이 정확히 일치하는지 검사하고 checkpoint 내부 run/result identity도 다시
로드해 검증한다. stage verifier는 현재 child에서 summary/LR selection/audit/gate를
재계산하며 main은 현재 screen receipt와 selection hash에 다시 결속한다.

stage 실행이 모두 끝나면 `COMPUTATION_COMPLETE`를 쓴다. 이는 과학적 통과를
뜻하지 않는다. main은 MSE yield와 seed별 analysis eligibility를
`scientific_gate.json`에 기록하며, 모든 모델이 최소 한 개의 eligible seed를
가질 때만 별도 `SCIENTIFIC_PASS`를 쓴다. downstream
CA-LRU 비교는 `require_scientific_pass(artifact_root)` 검증을 통과해야 하며,
`COMPUTATION_COMPLETE`만으로 시작하면 안 된다. sensitivity는 과학적 gate와
무관한 기술 통계로 실행할 수 있다.

이 파일의 실행 범위는 RNN/GRU/LSTM 복구까지다. 이후 LRU/CA-LRU 비교에서는
동일하게 LR × actual post-transition state-noise grid를 별도 사전 등록해
튜닝해야 하며, 이 baseline campaign이 그 선택을 대신하지 않는다.
