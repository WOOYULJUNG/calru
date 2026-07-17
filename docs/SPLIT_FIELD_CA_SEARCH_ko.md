# Split-field approximate continuous-attractor search

## 목적

이 실험의 목적은 이전 H-C에서 충분히 확인하지 못한 manifold-specific
normal attraction을 더 일반적인 비선형 recurrence에서 검증하는 것이다.
단순한 장기 slow subspace나 방향 기억은 성공으로 판정하지 않는다.

주 모델은 split-field recurrence 하나로 고정한다.

\[
h_{t+1}
=
\Lambda h_t
+(I-\Lambda)\phi(h_t)
+\Gamma\left[g(h_t,x_t)-g(h_t,0)\right].
\]

동일한 식의 residual 표현은

\[
h_{t+1}
=
h_t
+(I-\Lambda)\left[\phi(h_t)-h_t\right]
+\Gamma\left[g(h_t,x_t)-g(h_t,0)\right]
\]

이다. 두 correction term의 부호는 모두 `+`이다.

현재 구현은

\[
\phi(h)=\tanh(W_\phi h+b_\phi),
\qquad
g(h,x)=\tanh(W_g h+Ux+b_g)
\]

를 사용한다. \(\phi\)와 \(g\)는 서로 다른 recurrent parameters를 가진다.
따라서 blank-input autonomous geometry와 input-conditioned writing을
독립적으로 학습할 수 있다. \(x=0\)일 때 writer 차분은 정확히 0이다.

## 데이터와 topology

모든 과제는 generator-v1과 명시적인 \(q_0\) initialization을 사용한다.

- \(S^1\): scalar angular velocity
- \(T^2\): two independent angular velocities
- \(S^2\): independent 3-D angular velocity와 Rodrigues rotation

훈련에는 state noise, target noise, output dropout을 사용하지 않는다.
Validation/test frozen bank는 훈련에 사용하지 않는다.

## Broad search

Seed 10에서 다음 486개 조합을 1,500 updates까지 탐색한다.

\[
3\ {\rm topologies}
\times 3\ {\rm widths}
\times 2\ {\rm learning\ rates}
\times 3\ \lambda_0
\times 3\ \gamma_0
\times 3\ {\rm recurrent\ gains}
=486.
\]

- width: \(52,80,96\)
- learning rate: \(0.003,0.01\)
- initial retention: \(0.8,0.9,0.95\)
- initial write gain: \(0.3,1,3\)
- recurrent gain: \(0.7,0.9,1.1\)

각 topology에서 task-error 상위 후보와 blank-512 상위 후보의 합집합만
5,000 updates까지 이어서 학습한다. Retention parameters는 이 단계에서
task gradient로 갱신하지 않는다.

## RP calibration

5,000-update 후보마다 task, blank memory, local Jacobian, finite kick 및
manifold evolution을 먼저 분석한다. 서로 다른 기준으로 선택된 최대 세
부모 checkpoint에 대해 같은 optimizer state와 같은 후속 data stream으로
2,000 updates를 추가한다.

- no-RP continuation control
- \(\eta_\lambda\in\{0.1,1\}\)
- signed / positive-only intervention update
- per-intervention \(\theta\) step cap \(0.02\)
- intervention interval 100
- intervention blank horizon 512

RP와 control 모두 총 7,000 task-gradient updates를 갖는다.

## Approximate-CA 판정

Task error와 blank decoded error는 후보 제거용 조건일 뿐이다. 최종 판정은
다음 증거를 함께 사용한다.

1. local tangent singular gain이 1에 가까운가;
2. worst local normal singular gain이 1보다 작은가;
3. tangent-normal gap이 양수인가;
4. finite local-normal kick의 time-evolved manifold distance가 감소하는가;
5. tangent kick이 단순 소멸하지 않고 memory shift로 남는가;
6. blank rollout에서 hidden manifold가 stationary한가, global scaling만
   일어나는가, 또는 anisotropic하게 붕괴하는가;
7. \(S^1,T^2,S^2\) topology와 pairwise geometry가 유지되는가.

Finite-kick distance는 정적인 \(M_0\)가 아니라

\[
M_H=F_0^H(M_0)
\]

를 reference로 사용한다.

## Baseline gate

RNN, GRU, LSTM의 기존 3-seed topology checkpoints에도 같은 local
tangent/normal, finite-kick, blank-manifold 분석을 적용한다. Split-field
후보가 내부 ablation보다 좋아지는 것만으로는 성공이 아니다.

최종 표에는 다음을 명시한다.

- conventional baseline 대비 task error;
- blank-2048 memory error;
- tangent/normal local gains;
- finite normal recovery;
- pairwise shape distortion;
- parameter count.

가장 강한 baseline을 넘지 못한 topology는 성공으로 기록하지 않는다.
이 전체 실험은 exploratory model search이며, 세 seed에서 재현되기 전에는
논문 주 모델 결과로 사용하지 않는다.
