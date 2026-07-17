# Topology OOD generalization v1 — 결과

이 분석은 학습된 checkpoint를 다시 최적화하지 않고, 동일한 frozen parent에서 파생한 paired OOD bank로 평가한다. 대표선은 task-success seed 중 validation error가 가장 작은 seed를 사용하며, 성공 seed가 없으면 명시적인 failed-task fallback을 사용한다. 전체 3-seed 결과는 `ood_all_seed_summary.csv`에 별도로 보존한다.

## 대표 seed 선택

| Topology | Model | Seed | Task gate | Selection |
|---|---:|---:|---:|---|
| S1 | RNN | 12 | true | best success |
| S1 | GRU | 12 | true | best success |
| S1 | LSTM | 10 | true | best success |
| S1 | CA-LRU | 10 | true | best success |
| T2 | RNN | 11 | false | best available failed fallback |
| T2 | GRU | 12 | true | best success |
| T2 | LSTM | 12 | true | best success |
| T2 | CA-LRU | 10 | true | best success |
| S2 | RNN | 10 | false | best available failed fallback |
| S2 | GRU | 10 | false | best available failed fallback |
| S2 | LSTM | 12 | true | best success |
| S2 | CA-LRU | 10 | true | best success |

## 핵심 수치

아래 값은 대표 seed의 intrinsic terminal error이며 단위는 radian이다.

| Topology | Model | ID T=128 | T=2048 | T=1024, 2× | ID 후 blank 512 |
|---|---:|---:|---:|---:|---:|
| S1 | RNN | 0.1134 | 1.7305 | 1.3802 | 0.4128 |
| S1 | GRU | 0.0252 | 0.1626 | 0.6831 | 0.0917 |
| S1 | LSTM | 0.0230 | 0.2428 | 0.5384 | 0.1752 |
| S1 | CA-LRU | 0.0122 | 0.4173 | 0.7754 | 0.0123 |
| T2 | RNN | 0.2104 | 1.4348 | 1.5340 | 0.9414 |
| T2 | GRU | 0.0587 | 0.6127 | 0.9526 | 0.2732 |
| T2 | LSTM | 0.0506 | 0.3608 | 0.7290 | 0.3035 |
| T2 | CA-LRU | 0.0208 | 0.3556 | 0.9429 | 0.0209 |
| S2 | RNN | 1.0922 | 1.5314 | 1.6358 | 1.4049 |
| S2 | GRU | 0.1206 | 0.6597 | 1.3258 | 0.4608 |
| S2 | LSTM | 0.0600 | 0.3483 | 1.1812 | 0.3572 |
| S2 | CA-LRU | 0.1019 | 1.3378 | 1.4962 | 0.0973 |

## 3-seed robustness

- S1: ID 뒤 blank 512에서 CA-LRU의 3-seed median은 0.0642 rad이고 가장 좋은 baseline median 0.1752 rad보다 2.73배 낮다.
- T2: ID 뒤 blank 512에서 CA-LRU의 3-seed median은 0.0290 rad이고 가장 좋은 baseline median 0.2732 rad보다 9.42배 낮다.
- S2: ID 뒤 blank 512에서 CA-LRU의 3-seed median은 0.1718 rad이고 가장 좋은 baseline median 0.5023 rad보다 2.92배 낮다.

## 해석

- CA-LRU의 가장 일관된 이점은 입력 종료 뒤 기억 유지다. 이 결과는 세 topology의 3-seed median에서 모두 유지된다.
- 계속 입력을 적분하는 transport 일반화는 topology 의존적이다. CA-LRU는 T2 길이 OOD에서 강하지만, S1의 매우 긴 horizon과 S2의 transport에서는 GRU/LSTM보다 빠르게 악화된다.
- dwell 분포와 control smoothness 변화에는 대체로 안정적이지만, 길이와 속도를 동시에 키운 combined stress에서는 LSTM이 더 강하다.
- 따라서 현재 결과는 'CA-LRU가 모든 OOD에서 우월하다'가 아니라, 'retention plasticity가 blank-memory 안정성을 크게 높이지만 input-driven transport 정확도는 별도의 병목이다'를 지지한다.

## 해석 제한

- baseline은 ring-selected 설정을 topology에 이전했고, CA-LRU는 topology별 validation-selected 설정이다. 이 차이는 표와 manifest에 그대로 기록한다.
- RNN은 T2/S2에서, GRU는 S2에서 task-success seed가 없어 대표선이 failed-task fallback이다. 해당 선을 성공 모델과 동등한 증거로 해석하면 안 된다.
- 대표 seed figure는 구조를 읽기 위한 primary visualization이고, 재현성 판단은 반드시 전체 3-seed CSV와 함께 해야 한다.
