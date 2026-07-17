# RNN·GRU·LSTM·CA-LRU·H-C topology 비교

## 한 줄 결론

현재 평가 조건에서 **CA-LRU는 세 topology 모두에서 가장 작은 4,096-step blank-memory error를 보였고, 특히 \(T^2\)에서 task·geometry·장기 기억을 동시에 가장 잘 만족했다.** 다만 단기 task accuracy는 \(S^1\)에서 H-C, \(S^2\)에서 LSTM이 더 좋았으며, \(S^2\)의 CA-LRU는 1/3 seed만 사전 정의된 task gate를 통과했다. 따라서 현 단계의 가장 안전한 결론은 “CA-LRU가 모든 task에서 최고”가 아니라, **“CA-LRU의 near-neutral tangent dynamics가 기존 recurrent baseline보다 훨씬 작은 장기 drift로 이어진다”**이다.

## 비교 조건

- 모델: RNN, GRU, LSTM, CA-LRU+RP, H-C+RP
- topology: \(S^1\), \(T^2\), \(S^2\)
- seed: 10, 11, 12
- 총 비교 checkpoint: \(5\times3\times3=45\)
- 모든 수치는 final 5,000-update checkpoint와 동일한 frozen test/analysis bank에서 측정했다.
- task 실패 seed도 geometry와 dynamics 분석에서 제외하지 않았다.
- task gate는 validation NMSE \(<-20\) dB, hold baseline보다 작은 intrinsic error, finite output을 모두 요구한다.
- task error는 normalized geodesic error에 \(\pi\)를 곱한 radian 값이다.
- blank error와 same-memory error는 topology별 최대 geodesic distance로 정규화한 값이다.

중요한 비교 조건 차이가 있다.

> RNN·GRU·LSTM은 ring에서 선택한 설정을 topology별 재튜닝 없이 이전한 zero-retuning baseline이고, CA-LRU·H-C는 각 topology의 validation 결과로 hyperparameter를 선택했다.

따라서 이 표는 현재 모델들의 실제 결과를 정리한 것이지만, “모든 architecture를 topology별로 완전히 튜닝한 최종 leaderboard”는 아니다. CA-LRU와 classical baseline의 절대 우열을 최종 주장하려면 RNN·GRU·LSTM의 topology-specific tuning control이 추가로 필요하다.

## 1. Task 성능과 장기 기억

아래 값은 3-seed median이다. `통과`는 task gate 통과 seed 수다. 낮을수록 좋은 두 error를 함께 표시했다.

| Topology | Model | 통과 | Test error (rad) ↓ | Blank \(H=4096\) ↓ |
|---|---|---:|---:|---:|
| \(S^1\) | RNN | 1/3 | 0.0840 | 0.2939 |
|  | GRU | 3/3 | 0.0265 | 0.3338 |
|  | LSTM | 3/3 | 0.0197 | 0.2017 |
|  | **CA-LRU** | **3/3** | 0.0503 | **0.0204** |
|  | **H-C** | **3/3** | **0.0168** | 0.1513 |
| \(T^2\) | RNN | 0/3 | 0.2597 | 0.4880 |
|  | GRU | 3/3 | 0.0399 | 0.3760 |
|  | LSTM | 3/3 | 0.0393 | 0.4428 |
|  | **CA-LRU** | **3/3** | **0.0196** | **0.00923** |
|  | **H-C** | **3/3** | 0.0321 | 0.2179 |
| \(S^2\) | RNN | 0/3 | 1.0560 | 0.4723 |
|  | GRU | 0/3 | 1.0305 | 0.4784 |
|  | **LSTM** | **3/3** | **0.0468** | 0.4305 |
|  | **CA-LRU** | 1/3 | 0.0885 | **0.0547** |
|  | **H-C** | 2/3 | 0.0710 | 0.2121 |

전체 task-gate 통과 수는 RNN 1/9, GRU 6/9, LSTM 9/9, CA-LRU 7/9, H-C 8/9이다. 이 기준에서는 LSTM이 가장 안정적으로 task를 학습했고, CA-LRU의 약점은 \(S^2\) seed robustness다.

반면 장기 blank 기억에서는 CA-LRU가 세 topology 모두 가장 좋다.

- \(S^1\): H-C보다 7.4배, LSTM보다 9.9배 작은 error
- \(T^2\): H-C보다 23.6배, GRU/LSTM보다 40.7–48.0배 작은 error
- \(S^2\): H-C보다 3.9배, LSTM보다 7.9배 작은 error

이 차이는 단순한 \(T=128\) task fitting보다 blank-input autonomous dynamics에서 CA-LRU의 장점이 크게 나타난다는 뜻이다.

## 2. Geometry와 attractor dynamics

모든 값은 실패 seed까지 포함한 3-seed median이다.

| Topology | Model | Fiber ↓ | PCA PR | \(g_T(1)\) | \(g_N(1)\) ↓ | \(g_T(128)\) | \(Q(512)\) ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| \(S^1\) | RNN | 0.660 | 2.165 | 1.001 | 0.895 | 1.621 | 0.026 |
|  | GRU | 0.239 | 1.987 | 0.992 | 0.516 | 1.044 | 0.031 |
|  | LSTM | 0.171 | 2.155 | 0.993 | **0.479** | 1.022 | 0.028 |
|  | **CA-LRU** | **0.153** | 2.008 | 0.994 | 0.741 | **0.990** | 0.258 |
|  | **H-C** | 0.369 | 2.018 | 0.989 | 0.746 | 1.189 | 0.085 |
| \(T^2\) | RNN | 0.631 | 4.359 | 0.926 | 0.889 | 1.294 | 0.504 |
|  | GRU | 0.257 | 4.038 | 1.000 | **0.509** | 1.041 | 0.146 |
|  | LSTM | 0.209 | 4.320 | 0.999 | 0.511 | 1.026 | **0.100** |
|  | **CA-LRU** | **0.073** | 3.961 | 0.999 | 0.752 | **0.999** | 0.331 |
|  | **H-C** | 0.333 | 3.992 | 0.999 | 0.902 | 1.178 | 0.491 |
| \(S^2\) | RNN | 0.377 | 1.284 | 0.833 | 0.901 | 0.344 | 0.000 |
|  | GRU | 0.538 | 1.036 | 0.934 | 0.562 | 0.485 | 0.000 |
|  | LSTM | 0.268 | 3.407 | 1.004 | 0.503 | 1.034 | 0.126 |
|  | **CA-LRU** | 0.247 | 2.686 | 0.998 | 0.751 | **0.996** | 0.297 |
|  | **H-C** | **0.195** | 2.997 | 1.002 | **0.459** | 1.099 | 0.126 |

지표 의미:

- Fiber ratio: 같은 memory가 input path에 따라 hidden space에서 얼마나 흩어지는지. 작을수록 task state가 memory 좌표에 더 잘 정렬된다.
- PCA PR: endpoint hidden-state의 유효 선형 차원. 표준 embedding 기준 참고값은 \(S^1\approx2\), \(T^2\approx4\), \(S^2\approx3\)이다. PR만으로 topology를 증명할 수는 없다.
- \(g_T(1)\): 한 step 뒤 tangent perturbation gain. 이상적인 memory direction은 약 1이다.
- \(g_N(1)\): 한 step 뒤 sampled normal perturbation gain. 1보다 작으면 국소 수축이다.
- \(g_T(128)\): 128-step tangent gain. 장기간 1에 가까울수록 memory 좌표를 보존한다.
- \(Q(512)\): 5%-scale finite normal kick 뒤 manifold-normal distance의 비율. 0에 가까울수록 강한 복귀다.

핵심 해석은 다음과 같다.

1. **Normal contraction 하나만으로 continuous-attractor memory를 판정할 수 없다.** GRU와 LSTM은 CA-LRU보다 작은 \(g_N(1)\)과 \(Q(512)\)를 자주 보이지만, blank-memory drift는 훨씬 크다.
2. CA-LRU의 가장 일관된 특징은 강한 수축 자체가 아니라 **\(g_T(128)\approx1\)인 장기 tangent 보존**이다. 세 topology의 median이 0.990, 0.999, 0.996이다.
3. H-C는 \(S^1\)과 \(S^2\)에서 CA-LRU보다 강한 finite normal recovery를 보이지만, \(g_T(128)=1.189,1.099\)로 tangent expansion이 남는다. 이것이 낮은 단기 task error에도 불구하고 blank drift가 커지는 주된 동역학적 설명이다.
4. \(S^2\) RNN·GRU의 \(Q(512)=0\)은 좋은 attractor의 증거가 아니다. 두 모델은 task gate를 0/3 통과했고 PCA PR도 1.28/1.04로 낮다. 즉 perturbation과 clean state가 함께 같은 비기억 상태로 붕괴하면 \(Q\)와 same-memory difference가 인위적으로 작아질 수 있다.

따라서 CA 검증에는 최소한 다음 네 축이 동시에 필요하다.

\[
\text{task-aligned topology}
+\text{tangent preservation}
+\text{normal attraction}
+\text{bounded blank drift}.
\]

## 3. Topology별 판단

### \(S^1\)

H-C가 test error 0.0168 rad로 가장 좋고 LSTM이 0.0197 rad로 뒤따른다. 그러나 4,096-step blank에서는 CA-LRU가 0.0204로 가장 안정적이다. CA-LRU는 fiber ratio 0.153과 tangent gain 0.990도 가장 이상적이다. H-C의 normal recovery는 더 강하지만 tangent expansion 1.189가 장기 drift를 만든다.

판정: **CA-LRU의 ring memory는 지지된다. H-C는 task fitting은 더 좋지만 autonomous retention은 약하다.**

### \(T^2\)

CA-LRU가 task error 0.0196 rad, blank error 0.00923, fiber ratio 0.073, PCA PR 3.961, \(g_T(128)=0.999\)로 가장 균형 있게 좋다. 이 topology에서는 CA-LRU의 장점이 가장 명확하다.

판정: **현재 결과 중 가장 강한 CA-LRU topology-generalization 증거다.**

### \(S^2\)

LSTM이 task gate 3/3과 test error 0.0468 rad로 가장 안정적이다. H-C는 PCA PR 2.997과 fiber ratio 0.195, \(Q(512)=0.126\)으로 geometry와 normal recovery가 좋다. CA-LRU는 blank error 0.0547과 tangent gain 0.996으로 장기 보존은 가장 좋지만 task gate가 1/3이다.

판정: **sphere에서 dynamics의 가능성은 보였지만 CA-LRU의 seed-robust learning은 아직 검증되지 않았다. 현재는 headline 성공이 아니라 제한점과 후속 실험으로 다루는 편이 안전하다.**

## 4. Parameter count

| Topology | RNN | GRU | LSTM | CA-LRU | H-C |
|---|---:|---:|---:|---:|---:|
| \(S^1\) | 17,154 | 50,818 | 17,538 | 17,058 | 22,570 |
| \(T^2\) | 17,796 | 51,716 | 18,180 | 17,320 | 22,832 |
| \(S^2\) | 17,667 | 51,843 | 18,243 | 17,267 | 22,779 |

CA-LRU는 RNN/LSTM과 거의 parameter-matched다. H-C는 CA-LRU보다 약 1.32배, GRU는 약 3.0배 크다. 따라서 CA-LRU의 long-memory 이점은 더 많은 parameter로 설명되지 않는다.

## 5. 논문에 사용할 수 있는 주장과 보류할 주장

현재 바로 사용할 수 있는 주장:

> Across three memory topologies, CA-LRU consistently exhibits near-neutral finite-time tangent dynamics and the lowest long-horizon blank-input drift among the evaluated models.

> Strong normal contraction alone is insufficient for analog memory: gated recurrent baselines may rapidly suppress perturbations while losing task-aligned topology or drifting along the memory manifold.

> The torus result provides the clearest evidence that CA-LRU extends beyond a one-dimensional ring without topology-specific architectural changes.

현재 보류해야 하는 주장:

- CA-LRU가 모든 topology에서 task accuracy도 가장 좋다는 주장
- fully tuned RNN/GRU/LSTM보다 CA-LRU가 절대적으로 우월하다는 주장
- \(S^2\)에서 seed-robust approximate CA가 완전히 확립됐다는 주장
- 작은 \(Q\) 또는 작은 normal gain 하나만으로 CA라고 판정하는 주장

## 산출물

- `fig_A_all_models_task_and_blank`: task와 \(H=4096\) blank-memory seed 비교
- `fig_B_all_models_blank_curves`: horizon별 memory drift
- `fig_C_all_models_geometry`: fiber, local rank, PCA PR
- `fig_D_all_models_dynamics`: one-step/finite-time tangent-normal dynamics
- `fig_E_all_models_endpoint_pca`: seed 10 hidden-state PCA
- `fig_F_all_models_normal_recovery`: finite normal-kick recovery와 same-memory error
- `seed_level_comparison.csv`: 45개 checkpoint의 통합 seed-level 수치
- `model_topology_summary.csv`: 모델×topology 3-seed 요약

그림에서 빈 원은 task gate 실패 seed이며, PCA의 빨간 테두리는 해당 seed 10이 task gate를 실패했다는 뜻이다.
