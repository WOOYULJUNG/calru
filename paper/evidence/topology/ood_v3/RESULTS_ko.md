# Calibrated topology OOD — exploratory strength selection

이 보고서는 OOD 강도를 고르는 calibration 결과다. ID task gate를 통과한 checkpoint만 모델별 집계에 포함하며, confirmatory seed를 추가하기 전의 탐색 결과로 취급한다.

## ID-qualified seed 수

| Topology | RNN | GRU | LSTM | CA-LRU |
|---|---:|---:|---:|---:|
| S1 | 1 | 3 | 3 | 3 |
| T2 | 0 | 3 | 3 | 3 |
| S2 | 0 | 0 | 3 | 1 |

## 강도 판정

`chance_collapsed`는 ID-qualified seed median이 no-update/chance 기준의 75% 이상인 경우다. `severely_degraded`는 chance보다 낫지만 자기 ID error의 8배를 초과한 경우다. `transition`은 각각 50% 또는 4배를 넘지만 두 failure 기준에는 도달하지 않은 경우다.

| Branch | Headline maximum | Stress-only conditions |
|---|---:|---|
| length | T=1536 | length_h2048 |
| velocity_low | 0.5x | none |
| velocity_high | 2x | velocity_x3, velocity_x4 |
| correlation_low | ell=0.5x | none |
| correlation_high | ell=4x | none |
| dwell | blank block 96 | dwell_block128 |
| blank | H=512 | blank_h768, blank_h1024, blank_h1536, blank_h2048, blank_h4096 |

## 모든 eligible baseline이 무너진 조건

- S1 / T=2048: chance=rnn; >8x ID=gru,lstm
- T2 / T=2048: >8x ID=gru,lstm
- T2 / 3x: >8x ID=gru,lstm
- S2 / 3x: >8x ID=lstm
- S1 / 4x: >8x ID=rnn,gru,lstm
- T2 / 4x: >8x ID=gru,lstm
- S2 / 4x: chance=lstm
- S1 / all blank 128: chance=rnn,gru,lstm
- T2 / all blank 128: chance=gru,lstm
- S2 / all blank 128: chance=lstm
- S2 / H=768: >8x ID=lstm
- T2 / H=1024: >8x ID=gru,lstm
- S2 / H=1024: >8x ID=lstm
- T2 / H=1536: >8x ID=gru,lstm
- S2 / H=1536: >8x ID=lstm
- T2 / H=2048: >8x ID=gru,lstm
- S2 / H=2048: >8x ID=lstm
- S1 / H=4096: >8x ID=rnn,gru,lstm
- T2 / H=4096: chance=gru,lstm
- S2 / H=4096: chance=lstm

## 해석 규칙

- headline figure의 최대 강도는 세 topology 모두에서 적어도 하나의 ID-qualified baseline이 살아 있는 공통 구간으로 제한한다.
- 그보다 강한 조건은 삭제하지 않고 failure-limit stress로 분리한다.
- S2의 RNN·GRU처럼 ID부터 실패한 모델은 OOD failure 분모에서 제외한다.
- trajectory는 paired evaluation sample이며 통계적 replicate는 trained-model seed다.
