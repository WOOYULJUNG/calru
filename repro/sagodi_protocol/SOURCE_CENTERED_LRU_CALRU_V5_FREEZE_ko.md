# Source-centered LRU/CA-LRU v5 튜닝 동결안

이 캠페인은 Ságodi 공개 코드에서 확인되는 학습률 기준점과 유효 RNN
state-noise 크기를 중심으로 LRU와 CA-LRU의 학습 조건을 다시 선택한다.
Ságodi baseline 자체를 재현하는 캠페인이 아니며, baseline 결과나 기존 v4
artifact를 입력으로 사용하지 않는다.

## 고정 순서

1. LRU와 No-RP 각각의 `learning rate × state noise` 조합을 선택한다.
2. 한 sentinel seed가 ID MSE gate를 통과한 조합 중 상위 3개만 나머지
   네 seed로 확장한다.
3. 다섯 seed가 모두 ID MSE gate를 통과한 조합 중 평균 ID MSE가 가장
   작은 조합을 선택한다.
4. CA-LRU의 선택된 LR/noise를 고정하고 RP 조합을 같은 방식으로
   sentinel-screen한 뒤 확장한다.
5. RP 조합은 다섯 seed 모두 ID gate를 통과해야 하며, 그중 평균 blank
   memory MSE가 가장 작은 조합을 선택한다.

CA-LRU는 No-RP control에서 선택된 LR/noise를 그대로 상속한다. RP 전에는
두 모델의 초기화, online batch, state-noise stream 및 optimizer update가
동일하다. CA-LRU를 별도로 재튜닝하여 RP 효과와 base hyperparameter 효과를
섞는 선택은 허용하지 않는다.

모든 비교 모델은 공개 실행 artifact의 task contract를 공유한다. 한 trial은
`T=12.8, dt=0.1`인 128 step이고, sample마다 `s~Uniform(0,2)`를 뽑는
variable input sparsity를 사용한다. 초기 recurrent state에는 true `q0`가
아니라 공개 코드와 같이 첫 post-update target `q1`의 sin/cos 표현을 넣는다.

## 동결 그리드

- LR: `1e-2, 3e-3, 1e-3, 3e-4`
- 매 step post-update 실제 state-noise std: `0, 0.01, sqrt(0.1)*0.1, 0.1`
- RP eta: `300, 1000, 3000`
- RP damage epsilon: `1e-5, 3e-5, 1e-4`
- sentinel seed: `100`
- 추가 seed: `101, 102, 103, 104`
- 모든 학습: 128 step, 5,000 updates, Adam, batch 64
- ID eligibility: final held-out masked MSE `< 0.01`

Online task와 RP probe RNG에는 `(model_seed, update)`를 포함한다. 같은
seed의 LRU/CA-LRU는 같은 batch를 받아 paired comparison을 이루지만,
서로 다른 seed는 독립적인 online data stream을 갖는다.

`sqrt(0.1)*0.1 = 0.0316227766`은 upstream RNN 구현의 Euler noise가
실제로 한 step에 주입하는 coordinate-wise 표준편차다. `0`은 upstream
GRU/LSTM 실행 경로처럼 recurrent-state noise가 없는 control이고, `0.1`은
기존 v4 조건을 포함하기 위한 상한 control이다.

## 최대 계산량

- Stage 1 sentinel: `LRU/No-RP × 4 LR × 4 noise = 32` runs
- Stage 1 fan-out: 최대 `2 × 3 cells × 4 additional seeds = 24` runs
- Stage 2 sentinel: `3 eta × 3 epsilon = 9` runs
- Stage 2 fan-out: 최대 `3 cells × 4 additional seeds = 12` runs
- 합계: 최대 77 runs, 즉 385,000 optimizer updates

sentinel gate가 실패하면 다음 fan-out은 생성하지 않는다. Stage 1에서 한
모델이라도 eligible cell이 없으면 RP tuning은 시작하지 않는다. Stage 2에서
eligible RP cell이 없으면 최종 선택 artifact를 만들지 않는다.

이 튜닝 bank는 최종 main-test bank와 반드시 분리한다. manifold geometry,
fixed-point 수, OOD 지표는 어떤 hyperparameter 선택에도 사용하지 않는다.

## 실행

먼저 짧은 실행 경로 검사를 하고, 별도의 빈 artifact root에서 tune을 시작한다.

```bash
python -m repro.sagodi_protocol.source_centered_lru_calru_v5 \
  --stage smoke --artifact-root /path/to/v5-smoke --gpus 0

python -m repro.sagodi_protocol.source_centered_lru_calru_v5 \
  --stage tune --artifact-root /path/to/v5-tune --gpus 0,1,2,3
```

최종 소비 파일은 `tune/tuning_summary.json`이다. 여기에는 LRU와 No-RP의
선택 LR/noise, CA-LRU의 상속 관계, 선택 RP 조합이 함께 기록된다.
