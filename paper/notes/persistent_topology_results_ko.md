# Persistent topology 분석 결과

## 질문

학습된 hidden-state atlas가 단지 task output을 맞추는 것을 넘어 목표
topology와 같은 전역 구조를 형성하며, blank-input dynamics 아래에서도 그
구조를 유지하는가?

## 동결된 방법

- 비교: RNN, GRU, LSTM, CA-LRU, 최종 H-C
- topology: \(S^1,T^2,S^2\)
- seed: 10, 11, 12
- blank horizon: \(0,128,512,2048\)
- primary representation: 256개 transported-endpoint hidden states
- landmark: 이상적 output manifold에서 farthest-point permutation으로 고정한
  128개 anchor. 모든 모델·seed·horizon에 같은 anchor index 적용
- 거리: PCA가 아닌 원래 hidden space의 Euclidean distance
- scale: 각 point cloud의 median fifth-nearest-neighbor distance
- persistent homology: Vietoris-Rips, coefficient field
  \(\mathbb F_2\), \(H_0,H_1,H_2\) 계산
- 기대 strong-bar signature:
  - \(S^1: H_1/H_2=1/0\)
  - \(T^2: H_1/H_2=2/1\)
  - \(S^2: H_1/H_2=0/1\)
- primary threshold: 이상적 topology에서 기대되는 bar 중 가장 약한
  persistence의 0.5배
- sensitivity: 0.5, 0.65, 0.8배

Strong-bar count는 고정 filtration에서의 문자 그대로의 Betti number가 아니라,
목표 Betti signature에 대응하는 유의한 persistent feature의 수다.

## Primary signature match

각 숫자는 기대 \(H_1/H_2\) persistent signature를 만족한 seed 수다.

| Topology | Model | H=0 | H=128 | H=512 | H=2048 |
|---|---|---:|---:|---:|---:|
| \(S^1\) | RNN | 2/3 | 2/3 | 2/3 | 0/3 |
| | GRU | 3/3 | 3/3 | 3/3 | 3/3 |
| | LSTM | 3/3 | 3/3 | 3/3 | 3/3 |
| | CA-LRU | 3/3 | 3/3 | 3/3 | 3/3 |
| | H-C | 3/3 | 3/3 | 3/3 | 3/3 |
| \(T^2\) | RNN | 0/3 | 0/3 | 0/3 | 0/3 |
| | GRU | 2/3 | 2/3 | 0/3 | 0/3 |
| | LSTM | 3/3 | 2/3 | 0/3 | 0/3 |
| | CA-LRU | 3/3 | 3/3 | 3/3 | 3/3 |
| | H-C | 3/3 | 3/3 | 1/3 | 0/3 |
| \(S^2\) | RNN | 0/3 | 0/3 | 0/3 | 0/3 |
| | GRU | 1/3 | 1/3 | 1/3 | 0/3 |
| | LSTM | 3/3 | 3/3 | 0/3 | 0/3 |
| | CA-LRU | 3/3 | 3/3 | 3/3 | 3/3 |
| | H-C | 3/3 | 3/3 | 1/3 | 0/3 |

## 이상적 persistence diagram과의 거리

아래 값은 \(H_1,H_2\) bottleneck distance 평균을 primary strong-bar
threshold로 나눈 3-seed median이다. 낮을수록 이상적 topology와 가깝다.

| Topology | Model | H=0 | H=128 | H=512 | H=2048 |
|---|---|---:|---:|---:|---:|
| \(S^1\) | LSTM | 0.060 | 0.045 | 0.255 | 0.643 |
| | CA-LRU | 0.037 | **0.029** | **0.029** | **0.029** |
| | H-C | **0.022** | 0.025 | 0.074 | 0.626 |
| \(T^2\) | LSTM | 0.221 | 0.488 | 0.969 | 2.654 |
| | CA-LRU | **0.142** | **0.141** | **0.141** | **0.141** |
| | H-C | **0.142** | 0.183 | 0.588 | 0.966 |
| \(S^2\) | LSTM | 0.163 | 0.215 | 0.819 | 1.430 |
| | CA-LRU | **0.152** | **0.152** | **0.152** | **0.152** |
| | H-C | 0.163 | 0.341 | 0.887 | 1.217 |

## 해석

1. 모든 최종 H-C는 blank 이전에 목표 topology를 3/3 seed에서 형성했다.
   따라서 H-C가 새 topology를 전혀 학습하지 못했다는 해석은 틀리다.
2. H-C는 \(S^1\) topology를 2,048 blank step까지 3/3 seed에서 유지한다.
   diagram distance도 LSTM보다 일관되게 작거나 비슷하다.
3. \(T^2\)에서 H-C는 LSTM보다 이상적 diagram에 가깝고 topology 손상이
   늦지만, 512 step에는 1/3, 2,048 step에는 0/3만 signature를 유지한다.
4. \(S^2\)에서 H-C와 LSTM은 128 step까지 topology를 유지하지만 모두
   512 step부터 실패한다. H-C의 명확한 우위는 없다.
5. CA-LRU는 세 topology를 모든 horizon에서 유지한다. 그러나 이것은 앞선
   perturbation 분석에서 normal recovery가 약했던 것과 양립한다. 즉
   CA-LRU는 저장된 topology를 거의 움직이지 않게 유지하지만, manifold 밖
   상태를 적극적으로 복원하는 attractor라는 증거는 아니다.

## 논문에서 가능한 주장

현재 결과는 다음 주장을 지지한다.

> H-C learns hidden-state representations with the correct persistent
> topological signature across ring, torus, and sphere tasks. Relative to
> standard recurrent baselines, it preserves ring topology and delays
> higher-dimensional topological degradation under blank-input dynamics.

반면 다음 강한 주장은 지지하지 않는다.

> H-C robustly preserves arbitrary continuous-attractor topology over very
> long autonomous horizons.

\(T^2,S^2\)의 장기 결과 때문에 primary claim은 ring에 두고, higher-dimensional
topology는 capability와 한계를 함께 보여주는 확장 실험으로 두는 것이 안전하다.

## 민감도와 한계

- 0.5와 0.65 threshold에서 H-C의 핵심 survival pattern은 동일하다.
- 0.8 threshold에서는 \(S^2\), \(H=128\)의 H-C가 1/3로 감소한다. 즉 sphere의
  \(H_2\) feature는 존재하지만 ideal reference보다 persistence margin이 약하다.
- Persistent homology는 topology의 강한 전역 진단이지만 smooth
  homeomorphism이나 normal attraction을 단독으로 증명하지 않는다.
- 모델 간 hidden metric의 전역 비선형 왜곡은 bottleneck distance에 영향을
  준다. 그래서 scalar kNN normalization, signature count, threshold
  sensitivity를 함께 보고한다.
- RNN/GRU/LSTM은 ring-selected zero-retuning이고 CA-LRU/H-C는 topology별
  선택 결과이므로 optimization fairness가 완전히 동일하지 않다.

## 산출물

- 생성 코드:
  `repro/manifold_benchmark/analyze_persistent_topology.py`
- 동결 설정:
  `repro/manifold_benchmark/topology_persistence_v1.json`
- seed-level table:
  `paper/evidence/topology/persistence_v1/seed_metrics.csv`
- summary:
  `paper/evidence/topology/persistence_v1/summary.csv`
- threshold sensitivity:
  `paper/evidence/topology/persistence_v1/threshold_sensitivity_*.csv`
- ideal reference:
  `paper/evidence/topology/persistence_v1/reference.json`
- figure:
  `paper/figures/topology/persistence/*.{pdf,png}`
