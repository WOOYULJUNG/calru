# CA-LRU Continuous-Attractor Experimental Protocol

> Ságodi et al. (2024), *Back to the Continuous Attractor*를 기준으로 작성한 CA-LRU 실험·분석 사양서  
> 문서 버전: 1.0 (experiment-freeze draft)  
> 작성일: 2026-07-14  
> 목적: CA-LRU를 최소한 **approximate continuous attractor (approximate CA)**로 주장하기 위한 사전 정의, 비교군, 데이터 생성, 분석 및 판정 기준을 고정한다.

| Section | 바로 결정하는 것 |
|---|---|
| 0. CA 조건 | exact와 approximate CA의 필요 증거 |
| 0.5. 용어 | 논문에서 허용·제한할 표현 |
| 1. Baseline | 비교 모델, 예산, noise, seed, ablation |
| 2. Task | Ságodi 재현과 \(d=1,2,4,8\) 데이터 생성 |
| 3. Analysis | C1–C4, radial/ambient contraction, 통계, claim gate |

## 문서의 목표 가설

이 문서에서 사전등록하고 검증할 목표 가설은 다음과 같다. 결과 전에는 확정형 결론으로 사용하지 않는다.

> **We test whether CA-LRU learns approximate continuous-attractor dynamics organized around a task-relevant, approximately invariant slow memory manifold, with bounded tangential drift and recovery from radial and sampled ambient off-manifold perturbations.**

Exact CA는 다음 조건을 추가로 만족할 때만 주장한다.

> Manifold의 모든 점이 autonomous blank dynamics의 fixed point이며, tangent 방향은 중립적이고 모든 normal 방향은 수축한다.

따라서 실험의 논리적 순서는 다음과 같다.

1. 과제를 푸는가?
2. 기억 내용과 같은 topology의 neural manifold가 있는가? — C1
3. Manifold 위의 tangential drift가 느리고 bounded인가? — C2
4. State perturbation에 대해 기억 구조가 non-expansive하고 복원되는가? — C3
5. Dynamics/parameter perturbation 후에도 manifold와 기능이 지속되는가? — C4
6. 추가로 모든 manifold point가 fixed point인가? — exact CA 선택 조건

---

# 0. Continuous-attractor 조건

## 0.1 분석 대상 상태를 먼저 고정한다

모든 CA 분석에 앞서 `state_spec`을 작성한다. 분석 대상 상태 \(s_t\)는 다음 상태를 계산하는 데 필요한 **최소 Markov recurrent state**여야 한다.

```yaml
state_spec:
  primary_state: full_recurrent_markov_state
  components:
    - name: carrier
      dimension: int_from_checkpoint
      feeds_next_step: true
    - name: nonlinear_stream_or_auxiliary_state
      dimension: int_from_checkpoint
      feeds_next_step: verify_by_intervention
  decoded_from: carrier_or_stream_or_full
  overwritten_components: explicit_list_from_step_audit
```

구분 원칙:

- 다음 recurrent state에 영향을 주는 nonlinear state는 primary state에 포함한다.
- 매 step 재계산되고 다음 step에 전혀 feedback되지 않는 diagnostic stream은 primary dynamical state와 별도로 분석한다.
- Radial contraction이 recurrent state에서 발생하는지, decoded output에서만 발생하는지를 분리한다.
- 논문의 주 CA 주장은 primary Markov state에서 평가한다.
- Carrier-only와 diagnostic full-state 결과는 mechanism 분석으로 함께 보고하되 서로 대체하지 않는다.

CA-LRU의 실제 one-step map을 다음처럼 표기한다.

\[
s_{t+1}=F(s_t,u_t),\qquad F_0(s):=F(s,0).
\]

\(F_0\)는 코드의 실제 `model.step(zero_input, state)`를 사용한다. 선형 retention 항뿐 아니라 blank에서 작동하는 모든 비선형 항과 feedback을 포함한다.

**CA-LRU에 대한 현재 가설은 radial contraction이 없다는 것이 아니다.** Full nonlinear dynamics에서 ring의 radial 방향이 수축하는 현상을 출발점으로 삼는다. 다만 논문에서는 그 현상의 귀속을 다음 두 경우로 분리한다.

1. 완전히 blank인 실제 Markov map \(F_0\)에서 수축하면 **autonomous radial attraction**이다.
2. cue 재주입, stream 재계산, 외부 reset 또는 nonzero control이 있어야 수축하면 **input-conditioned recovery**이다.

둘 다 유의미한 결과이지만, exact/approximate CA의 normal attraction에 대한 주 증거는 첫 번째여야 한다. 따라서 perturbation 직후의 input tensor, stream 갱신, overwrite 여부를 함께 로그로 저장한다. Carrier 식의 선형 부분만 보고 full nonlinear radial contraction의 유무를 추론하지 않는다.

Exactness에는 별도의 architecture-level audit가 필요하다. 만약 실제 autonomous carrier map이

\[
F_0^{\mathrm{carrier}}(h)=\Lambda h,
\qquad
0<\lambda_j<1
\]

이고 blank-active feedback이 이를 정확히 보상하지 않는다면, carrier의 nonzero fixed-point continuum은 분석적으로 불가능하고 fixed point는 \(h=0\)뿐이다. 이는 full end-to-end dynamics의 radial contraction을 부정하지 않는다. 오히려 exact CA보다 approximate CA가 자연스러운 목표임을 뜻한다. Actual full-Markov map에서 nonlinear feedback이 \(F_0(s(q))=s(q)\)를 만들거나 \(\lambda=1\) limit이 구현된 경우에만 exact screen을 계속한다.

## 0.2 Exact continuous attractor: 연속시간 정의

연속시간 dynamics

\[
\dot x=f(x),\qquad \mathcal M\subset\mathbb R^n
\]

에서 \(\mathcal M\)이 exact continuous attractor이려면 다음을 만족해야 한다.

### E1. Fixed-point continuum

\[
\forall x\in\mathcal M,\qquad f(x)=0.
\]

Manifold의 일부 점만 fixed point인 것으로는 부족하다.

### E2. Tangential marginal stability

\(\dim\mathcal M=d\)일 때, 모든 \(x\in\mathcal M\)에서 tangent 방향의 \(d\)개 Jacobian eigenvalue가 정확히 0이다. Fixed-point manifold의 tangent vector는 vector-field Jacobian의 kernel에 속한다.

\[
D f(x)T(x)=0,
\qquad
\lambda_i^{T}(x)=0,
\quad i=1,\ldots,d.
\]

### E3. Normal attraction

모든 normal rate는 음수이며, manifold 전역에서 0으로부터 균일하게 떨어져 있어야 한다.

\[
\sup_{x\in\mathcal M}\max_j\operatorname{Re}\lambda_j^{N}(x)<-\alpha,
\qquad \alpha>0.
\]

즉 exact CA는 **fixed-point continuum + tangent neutral stability + normal attraction**이다.

## 0.3 Exact CA의 이산시간 대응

CA-LRU는 이산시간 map이므로 exact 조건을 다음처럼 번역한다.

### D-E1. Fixedness

\[
\forall s\in\mathcal M,\qquad F_0(s)=s.
\]

### D-E2. Tangent preservation

Local tangent basis를 \(T(s)\in\mathbb R^{n\times d}\)라 하면

\[
J_0(s)T(s)=T(s),
\qquad
J_0(s):=\frac{\partial F_0}{\partial s}(s).
\]

Exact한 경우 tangent multiplier는 정확히 1이다. Approximate CA에서는 이 equality의 residual을 측정한다.

### D-E3. Normal contraction

정의상 필요한 것은 임의의 Euclidean orthogonal complement가 한 step마다 단조 수축하는 것이 아니라, invariant stable normal bundle \(E_s^s\)의 uniform exponential stability다. 즉 어떤 \(C>0\), \(0<\rho<1\)가 존재하여

\[
\left\|
D F_0^H(s)\big|_{E_s^s}
\right\|
\le
C\rho^H
\]

가 manifold 전역에서 성립해야 한다. Exact fixed manifold 위에서는 stable-normal multipliers가 모두 unit circle 안에 있다.

Local Euclidean normal basis \(N(s)\)에 대한

\[
\sigma_{\max}
\left(
N(s')^\top J_0(s)N(s)
\right)<1
\]

은 강하고 해석하기 쉬운 **one-step sufficient screen**이지만 일반적인 non-normal dynamics에서 필요조건은 아니다. 따라서 eigenvalue, projected cocycle, finite-kick recovery를 함께 보고한다.

이산 multiplier \(\mu_i\)를 Ságodi의 연속시간 rate와 비교하려면

\[
r_i=\frac{\log|\mu_i|}{\Delta t}
\]

로 변환한다. 이때 \(\mu=1\leftrightarrow r=0\), \(|\mu|<1\leftrightarrow r<0\)이다.

## 0.4 Approximate continuous attractor: 논문의 주 판정 기준

Ságodi et al.의 C1–C4를 CA-LRU에 operationalize한다. Approximate CA 주장을 위해서는 네 조건을 모두 평가한다.

### C1. Memory content와 neural manifold의 smooth approximate correspondence

기억 latent를 \(q\in\mathcal Q\), task-evoked neural state map을 \(s(q)\in\mathcal M\)라 한다.

필수 확인:

1. \(\mathcal M\)의 intrinsic dimension이 \(\dim\mathcal Q=d\)와 일치한다.
2. \(q\mapsto s(q)\mapsto\hat q\) cycle decoding error가 작다.
3. 서로 멀리 떨어진 latent가 동일한 neural memory state로 collapse하지 않는다.
4. Local tangent map \(\partial s/\partial q\)가 rank \(d\)를 유지한다.
5. Ring/torus에서는 periodic topology가 보존된다.

출력 지표:

\[
e_{\mathrm{decode}}
=\mathbb E_q[d_{\mathcal Q}(\hat q(s(q)),q)],
\]

\[
\sigma_{\min}\left(\frac{\partial s}{\partial q}\right),
\qquad
\kappa_T=\frac{\sigma_{\max}(\partial s/\partial q)}
{\sigma_{\min}(\partial s/\partial q)}.
\]

한 latent에 여러 internal states가 대응하는 fiber solution이 발견되면 이를 숨기지 않는다. 동일 \(q\)에 도달하는 여러 경로를 생성하여 within-\(q\) variance와 between-\(q\) separation을 보고한다.

### C2. Bounded slow flow on the manifold

Blank vector field의 이산시간 대응을

\[
v_0(s)=F_0(s)-s
\]

로 정의한다. Local tangent projector \(P_T(s)=T(s)T(s)^\top\)를 사용하여

\[
v_T=P_Tv_0,
\qquad
v_N=(I-P_T)v_0
\]

로 분해한다.

필수 지표:

\[
\|v_T\|_{\infty,\mathcal M}
=\sup_{s\in\mathcal M}\|v_T(s)\|,
\]

\[
\|v_N\|_{\infty,\mathcal M}
=\sup_{s\in\mathcal M}\|v_N(s)\|,
\]

\[
e_{\mathrm{memory}}(H)
=\mathbb E_q[d_{\mathcal Q}(\hat q(F_0^H(s(q))),q)].
\]

`uniform norm`은 실제 finite sample에서는 max와 함께 95th/99th percentile을 보고한다. 한두 개 numerical outlier 때문에 max가 불안정할 수 있기 때문이다.

### C3. S-type robustness: state perturbation

S-type perturbation은 deterministic map \(F_0\)를 바꾸지 않고 state만 변화시킨다.

Ságodi et al.의 최소 해석은 **non-expansive flow**, 즉 관련 Lyapunov exponent가 양수가 아닌 것이다. 본 protocol은 CA-LRU의 radial contraction 주장을 검증하기 위해 이를 더 강하게 나눈다.

- tangent: 기억 차이를 폭발시키지 않는 non-expansion
- normal: manifold로의 attraction
- radial/ambient: normal attraction의 방향별 finite-amplitude 확인

분리할 방향:

1. Tangent perturbation
2. Ring의 radial 또는 task-active in-span normal perturbation
3. Retained/slow subspace 안의 tangent-orthogonal normal perturbation
4. Slow subspace 밖 ambient normal perturbation
5. Isotropic random state noise

Radial contraction 가설은 random normal 평균으로 대체하지 않는다. Ring에서 local radial vector \(n_R(q)\)를 명시적으로 구성하고 다음을 측정한다.

\[
G_R(q,\rho,H)=
\frac{\operatorname{dist}\left(F_0^H(s(q)+\rho n_R(q)),\mathcal M\right)}
{\operatorname{dist}\left(s(q)+\rho n_R(q),\mathcal M\right)}.
\]

\(G_R<1\)이면 radial recovery, \(G_R>1\)이면 radial expansion이다. Local Jacobian뿐 아니라 여러 finite radius에서 측정하여 비선형 basin을 확인한다.

### C4. D-type robustness: dynamics/parameter perturbation

D-type perturbation은 dynamics 자체를 바꾼다.

Ságodi et al.은 noisy stochastic training이 C3/C4를 만족시키는 논거를 제시한다. 본 연구에서는 더 강하고 직접적인 empirical evidence를 위해 학습 사실과 별도로 parameter/vector-field perturbation 후 C1–C3가 지속되는지를 재검사한다.

\[
\vartheta'=\vartheta+\epsilon
\frac{\|\vartheta\|_2}{\|\xi\|_2}\xi,
\qquad \xi\sim\mathcal N(0,I).
\]

다음 parameter group을 분리하여 perturb한다.

- Retention parameters
- Blank-active recurrent parameters and audited radial-source candidates
- Input writer parameters
- Encoder
- Readout
- 전체 recurrent transition

Perturbation 후 다음을 다시 평가한다.

- Task success
- Manifold topology/correspondence
- Tangential drift
- Radial 및 ambient normal recovery
- Tangent–normal timescale gap
- Clean manifold와 perturbed manifold 사이의 aligned distance

State-noise training을 했다는 사실만으로 C4가 입증되었다고 쓰지 않는다.

## Exact와 approximate CA의 주장 단계

| 단계 | 필요한 증거 | 허용되는 표현 |
|---|---|---|
| Task memory | ID task 성능 | recurrent memory model |
| Slow memory | 장기 blank drift가 작음 | slow recurrent memory |
| Approximate CA | C1–C4 모두에 대한 실험 증거 | approximate continuous attractor dynamics |
| Strong sampled approximate CA | C1–C4 + sampled tangent–normal gap + radial/ambient recovery | empirically attracting slow manifold over evaluated anchors/directions |
| Sampled-uniform normal claim | adaptive refinement에서 sampled violation 0 + numerical upper bound | consistent with uniform normal attraction on the tested domain |
| Exact CA | 모든 점 fixed + tangent multiplier 1 + 모든 normal contraction | exact continuous attractor |

CA-LRU의 기본 제출 목표는 **L3 approximate CA**이다. Exact CA와 sampled-uniform normal claim은 각각의 추가 조건이 충분할 때만 사용한다.

## 내부 판정 규칙

수학적 정의와 empirical pass threshold를 구분한다. 이 문서에서 유일한 normative threshold source는 **Section 3.13.2의 claim-gate table**이다. 앞 절의 C1–C4는 정의와 필요한 evidence를 설명하며 별도 숫자 threshold를 만들지 않는다.

Section 3.13.2의 값은 main 결과를 보기 전 pilot seed에서 scale과 numerical error를 확인한 뒤 동결한다. Pilot seed와 main seed를 분리하고 결과 후 task별로 threshold를 바꾸지 않는다.

---

# 0.5. 논문 용어 정리

## 0.5.1 권장 용어

| 영문 용어 | 문서 내 의미 | 사용 조건 |
|---|---|---|
| exact continuous attractor | manifold 전체가 fixed-point continuum이고 normal-attracting | E1–E3 직접 확인 |
| approximate continuous attractor | C1–C4를 만족하는 analog-memory solution | 본 논문의 기본 주장 |
| slow invariant manifold | invariant하며 내부 flow가 느린 manifold | invariance residual과 slow flow 확인 |
| invariant manifold | manifold에서 시작한 trajectory가 dynamics 아래 manifold에 남는 set | 주변 state가 끌려오는 attraction과 구분 |
| task-relevant topology | neural manifold가 memory variable과 같은 topology | C1 evidence |
| fast–slow decomposition | tangent slow flow와 normal fast flow 분리 | local basis와 timescale 분석 |
| tangential drift | manifold 위에서 memory coordinate가 변하는 flow | \(v_T\), geodesic drift |
| radial contraction | ring/surface의 local radial normal perturbation 복원 | explicit radial probe |
| off-manifold recovery | 일반 normal perturbation이 manifold로 복귀 | finite-kick probe |
| radial normal | learned slow/embedding subspace 안에서 tangent에 수직인 normal | full end-to-end radial contraction의 직접 분석 대상 |
| ambient normal | slow/embedding subspace 밖의 high-dimensional normal | random kick가 쉬운 방향에 치우치지 않게 별도 평가 |
| normal hyperbolicity | manifold 전역에서 normal dynamics가 tangent보다 균일하게 빠름 | uniform gap evidence |
| slow point | \(F_0(s)-s\)가 작지만 0은 아닌 점 | fixed point와 구분 |
| fixed point | \(F_0(s)=s\)인 점 | fixed residual/solver |
| S-type robustness | dynamics는 고정하고 state만 perturb | state kick/noise |
| D-type robustness | parameter/vector field를 perturb | weight/dynamics perturb |
| finite-time generalization | 학습 horizon 밖의 유한 시간 기억 | \(2T,4T,8T,16T\) |
| asymptotic dynamics | \(t\to\infty\)의 fixed point/limit cycle/basin 구조 | 주로 \(d=1\) 분석 |
| functional universality | seed/architecture마다 세부 topology는 달라도 공통 slow-manifold motif | 개별 neuron 불변성과 구분 |
| seed-robust organization | 정렬 후 기능, dimension, subspace, spectrum이 안정적 | 최소 10 seed 분석 |
| carrier state | 장기 정보를 운반하도록 설계된 CA-LRU state subset | carrier-only 결과를 full-state 결과로 확대하지 않음 |
| full dynamical state | 미래 transition을 결정하는 모든 causal variable | stream이 feedback되는지 포함해 Markov state를 명시 |
| output-null/fiber dynamics | state는 움직이지만 decoder가 같은 memory를 내는 내부 dynamics | decoded fixedness와 full-state fixedness를 구분 |
| functional seed consistency | 좌표가 달라도 topology, dimension, drift, recovery가 일관됨 | coordinate-level seed invariance와 구분 |
| intrinsic dimension | memory manifold 자체의 dimension \(d\) | ring은 1, \(\mathbb T^d\)는 \(d\) |
| embedding dimension | manifold를 표현하는 subspace dimension | ring의 intrinsic dimension이 1이어도 일반적 선형 embedding은 2차원 |
| ambient dimension | full recurrent state의 dimension | hidden width와 memory dimension을 구분 |

## 0.5.2 제한해서 사용할 용어

### `persistent manifold`

Known normally hyperbolic manifold에 작은 perturbation을 가한 뒤 diffeomorphic manifold가 지속된다는 이론적 맥락에서만 사용한다. Task-trained manifold를 관찰했다는 이유만으로 persistent manifold라고 부르지 않는다.

### `distance to a continuous attractor`

Ságodi의 이론적 가정 아래 manifold 위 tangent vector-field uniform norm을 가리킬 때만 사용한다. Task RMSE나 hidden Euclidean distance를 그대로 CA까지의 거리라고 부르지 않는다.

### `normal hyperbolicity`

일부 점의 평균 eigenvalue gap으로 증명했다고 쓰지 않는다. 샘플링한 manifold 전역의 uniform evidence라고 표현한다.

### `seed invariant`

개별 neuron, coordinate 또는 fixed-point 수가 seed마다 같다는 뜻으로 사용하지 않는다. 다음처럼 쓴다.

> functionally and geometrically consistent across random initializations after alignment

## 0.5.3 피해야 할 표현

- “Existing RNNs cannot form continuous attractors.”
- “All trained RNNs form a CA.”
- “CA-LRU is an exact CA” — all-point fixedness 없이 사용 금지
- “Normal contraction” — random normal target RMSE만으로 사용 금지
- “Seed-invariant neurons/coordinates” — permutation/alignment 고려 없이 사용 금지
- “Persistent manifold” — perturbation/persistence 조건 없이 사용 금지
- “Every metric is best” — 일부 baseline이 우세한 항목이 있으면 사용 금지

## 0.5.4 논문 권장 문장

### 기본 주장

> Conditional on the L3 gates, CA-LRU exhibits approximate continuous-attractor dynamics characterized by a task-relevant slow memory manifold, bounded tangential drift, and recovery from radial and sampled ambient off-manifold perturbations.

### 설계와 emergence의 구분

> Selective retention and timescale separation are explicitly induced by the CA-LRU mechanism, whereas the task-specific attractor geometry emerges from data.

### Seed 결과가 확보된 뒤

> Across random initializations, individual coordinates may differ, while the learned memory dimension, aligned manifold geometry, and tangent–normal dynamics remain consistent.

### Exact CA가 아닌 경우

> CA-LRU is not assumed to realize an exact continuum of fixed points; instead, we test whether it satisfies operational conditions for an approximate continuous attractor over behaviorally relevant timescales.

---
# 1. Baseline 및 비교 설계

## 1.1 비교는 두 protocol로 분리한다

Ságodi 재현과 CA-LRU의 최적 성능 비교를 한 표에 섞지 않는다.

### Protocol A — Ságodi original-paper protocol

목적: Ságodi et al.의 결과와 직접 비교한다.

- 과제: memory-guided saccade, angular-velocity integration, double angular integration
- 모델: vanilla RNN(ReLU/tanh/rectified tanh), GRU, LSTM, CA-LRU
- Width: nominal 64/128/256
- LSTM: vanilla RNN 대비 절반 unit을 Ságodi primary로 사용; nominal same-width는 추가 control
- State noise: 논문 Eq. 61에 따라 recurrent-coordinate standard deviation 0.1
- Batch: 64
- Gradient updates: 5,000
- Optimizer: Adam, betas \((0.9,0.999)\)
- LR 후보: \(\{10^{-2},10^{-3},10^{-4},10^{-5}\}\)
- 데이터: online fresh sampling
- 모델/조건당 main model seeds: 10
- Held-out analysis cutoff: NMSE \(<-20\) dB

Rectified tanh는

\[
\phi(x)=\max(\tanh x,0)
\]

로 고정한다. Ságodi-style vanilla RNN은 Euler step \(\Delta t=0.1\), recurrent initialization gain \(g=1.5\)를 사용하고 실제 update equation과 bias convention을 manifest에 저장한다.

Primary 구현은 공개 코드의 architecture별 예외보다 원 논문 Methods S7의 수식과 서술을 우선한다. 즉 모든 model family에 paper-defined recurrent state noise를 적용하고, vanilla RNN은 논문의 \(\Delta t=0.1\), Xavier initialization, recurrent gain \(g=1.5\)를 따른다. 공개 코드는 tensor shape와 task generator를 확인하는 보조 자료로만 사용하며, 논문 수식과 다르면 차이를 manifest에 기록한다.

### Protocol B — CA-LRU fair benchmark

목적: 현대 baseline과 동일한 데이터·예산 아래 CA-LRU의 성능과 mechanism을 비교한다.

- 과제: 기존 8개 hold/integrate task + Ságodi task + \(d\)-dimensional extensions
- 모델: RNN, GRU, LSTM, real diagonal SSM, complex LRU, CA-LRU
- Training budget: 모든 모델 동일 number of optimizer updates와 동일 online sample 수
- Tuning budget: 모델별 동일한 learning-rate 후보 수와 pilot seed 수
- Evaluation: 동일 fixed evaluation bank와 paired perturbation bank
- 비교 방식: width-matched 결과와 parameter-matched 결과를 별도 표로 보고
- Main model seeds: 10
- Task embedding seeds: main 1개 고정, robustness subset 5개

Protocol A 결과를 Protocol B 결과와 직접 합산하거나, 서로 다른 training budget 결과로 우열을 주장하지 않는다.

모든 baseline에도 CA-LRU와 동일한 L0–L3 gate를 적용한다. RNN의 output이 ring처럼 보이거나 slow point가 존재한다는 사실만으로 CA라고 라벨링하지 않는다. 반대로 exact fixed-point continuum가 없더라도 C1–C4를 통과하면 그 baseline 역시 approximate CA로 분류한다.

## 1.2 필수 baseline 목록

| 그룹 | 모델 | 목적 | Main table |
|---|---|---|---|
| Ságodi reproduction | Vanilla RNN–tanh | 대표 ungated baseline | 필수 |
| Ságodi reproduction | Vanilla RNN–ReLU | activation dependency | appendix/summary |
| Ságodi reproduction | Vanilla RNN–rectified tanh | 원 논문 일치 | appendix/summary |
| Gated | GRU | 강한 표준 baseline | 필수 |
| Gated | LSTM | 장기 기억 baseline | 필수 |
| Structured linear | Real diagonal SSM | 단순 selective-timescale 대조 | 필수 |
| Structured linear | Complex LRU | parameter-near-matched recurrent baseline | 필수 |
| Proposed | CA-LRU | full nonlinear state-dependent model | 필수 |

Generic complex LRU와 CA-LRU No-RP는 구분한다. Generic LRU는 독립 baseline이고, No-RP는 CA-LRU의 encoder/writer/readout과 모든 nonlinear scaffold를 유지한 채 RP mechanism만 제거한 causal control이다.

`SSM` baseline을 S4/Mamba라고 부르지 않는다. 실제 구현이 단순 learned real diagonal SSM이면 그대로 명명한다.

## 1.3 CA-LRU mechanism ablation

최종 CA-LRU와 동일한 outer scaffold를 사용한다.

| Ablation | 바꾸는 요소 | 검증 질문 |
|---|---|---|
| No-RP | retention-plasticity update 비활성화 | RP가 CA 형성/안정성의 원인인가? |
| All-slow | 모든 retention mode를 동일 near-one 값으로 고정 | 선택성이 필요한가? |
| Count-matched slow | CA-LRU가 선택한 수만큼 임의 mode를 slow로 고정 | 단순 slow-mode 개수 효과인가? |
| Shuffled RP | damage score와 coordinate 연결을 shuffle | task-aligned credit가 필요한가? |
| Input-only writer | state dependence 제거 | integration/transport에 recurrent writer가 필요한가? |
| Linear writer | nonlinear writer 제거 | nonlinear geometry/transport가 필요한가? |
| Radial-source ablation | audit에서 확인된 candidate component를 제거·선형화 | radial contraction의 causal source는 무엇인가? |
| State-noise off | hidden-state noise 제거 | S-type robustness가 noise training에 의존하는가? |
| Fixed retention spectrum | 학습된 최종 spectrum은 유지하고 RP update만 제거 | 학습 과정과 최종 spectrum 효과 분리 |

No-RP/all-slow가 과거 linear-update scaffold에서만 실행된 경우 final CA-LRU causal ablation으로 사용하지 않는다. 최종 nonlinear scaffold에서 다시 실행한다.

Ablation의 정확한 동결 규칙:

- **No-RP:** full model과 같은 initialized retention spectrum을 쓰고 RP hook만 끈다. Retention이 task gradient에서 detached되는지 여부도 full model과 동일하게 유지한다.
- **All-slow:** 고정 \(\lambda_{\mathrm{slow}}\)는 pilot에서 정하고 main/test 결과로 선택하지 않는다.
- **Count-matched:** \(k\)는 pilot CA-LRU의 selected-mode median으로 고정하며 main/test CA-LRU에서 가져오지 않는다.
- **Fixed final spectrum:** trained full model에 대한 post-hoc control로 분류한다. 별도 재학습 causal ablation으로 해석하지 않는다.
- **Radial-source:** “restoring network”가 이미 확인되었다고 전제하지 않는다. Blank-map audit와 intervention으로 source candidate를 정한 뒤 명명한다.

## 1.4 공정 비교 규칙

### 동일하게 고정할 것

- Train/validation/evaluation latent trajectories
- Task seed, GP covariance, mixing matrix \(A\), coupling matrix \(B\)
- Sequence length distribution
- Optimizer update 수
- Gradient clipping policy
- State-noise condition
- Evaluation horizon과 perturbation direction/radius
- Checkpoint selection rule
- Success cutoff

### 모델별로 허용할 것

- Learning rate: 동일한 후보 grid에서 선택
- LSTM hidden width: parameter-matched track에서 조정
- 모델 고유 state dimension: 명시하고 state-normalization 사용
- 모델 고유 initialization: 표준 방식 사용하되 initialization family를 기록

### 두 종류의 matching을 분리한다

1. **Width-matched:** nominal hidden width를 동일하게 둔다.
2. **Parameter-matched:** trainable parameter 수를 CA-LRU 대비 \(\pm5\%\) 이내로 맞춘다.

Width-matched 결과를 parameter-matched라고 부르지 않는다. Complex state 하나가 real coordinate 두 개에 해당하면 effective real state dimension도 함께 기록한다.

Parameter-matched baseline width는 task의 실제 input/output dimension을 넣은 trainable-scalar count로 결정한다.

\[
N_m^*
=
\arg\min_N
\left|
P_m(N)-P_{\mathrm{CA\text{-}LRU}}
\right|.
\]

Recurrent/input/output weight와 bias, gate, initial-state encoder, complex real/imaginary parameter, retention parameter를 모두 센다. 각 표에 정확한 count와 상대 차이를 기록하며, 가능하면 \(\pm5\%\), 최대 \(\pm10\%\) 안으로 맞춘다. Parameter matching과 별도로 update당 FLOPs, peak memory, wall-clock time, sequence-token 수도 보고한다.

### 초기 상태의 공정성

두 encoding track을 모든 모델에 동일하게 제공한다.

1. **Hidden-init:** 동일 형태의 learned encoder \(s_0=E_{\mathrm{init}}(y_0)\). Encoder parameter를 count에 포함한다.
2. **Cue-driven:** 모든 모델을 zero 또는 같은 learned constant state에서 시작하고 \(y_0\)를 input token으로만 제공한다.

Cue-driven이 CA-LRU의 primary extension이다. Hidden-init 결과는 Ságodi 직접 재현에 사용하되, 초기화 map이 manifold topology를 직접 심을 가능성을 명시한다.

## 1.5 State-noise 조건

두 noise track을 사용한다.

### Absolute Ságodi noise

\[
\zeta_t\sim\mathcal N(0,0.01I),
\]

즉 coordinate standard deviation 0.1이다. Protocol A original-paper condition의 primary 값이다. CA-LRU에도 같은 원칙으로 다음 time step에 전달되는 primary recurrent state에 이 noise를 적용한다.

### Scale-normalized noise

모델별 state scale 차이를 고려하여

\[
\zeta_t\sim\mathcal N
\left(0,\frac{(\sigma_{\mathrm{rel}}R_{\mathrm{noise}})^2}{n}I\right),
\]

여기서

\[
R_{\mathrm{noise}}^2
=
\mathbb E\|s-\bar s\|_2^2
\]

는 warm-up evaluation state-vector energy이고 \(n\)은 perturbed state dimension이다. 따라서 expected perturbation norm은 \(\sigma_{\mathrm{rel}}R_{\mathrm{noise}}\)다. Protocol B의 공정성 확인용 secondary track이다.

Noise는 training 중 주입하고, deterministic manifold/Jacobian 분석에서는 끈다. Noise 위치를 model별로 기록한다.

- RNN/GRU: recurrent hidden state
- LSTM: hidden \(h\), cell \(c\), 또는 둘 다인지 명시
- CA-LRU: carrier, nonlinear auxiliary state, 또는 full state인지 명시

## 1.6 Seed와 실험 단위

Seed를 다음처럼 분리한다.

```yaml
seeds:
  task_seed: 0          # A, B, GP bank, geometry 고정
  data_stream_seed: 0   # online batch 순서
  model_seed: 0         # parameter initialization
  perturb_seed: 0       # kick/weight perturbation bank
```

Main comparison에서는 task seed와 evaluation bank를 고정하고 model seed만 10개 변화시킨다. Representation rotation에 대한 robustness는 별도 task seeds 5개에서 수행한다.

통계적 독립 단위는 trajectory가 아니라 **trained model seed**이다. 수천 trajectory를 독립 표본처럼 취급하여 유의성을 부풀리지 않는다.

## 1.7 Hyperparameter 선택

1. Pilot model seeds \(\{100,101,102,103,104\}\)만 tuning에 사용한다.
2. Main model seeds \(\{0,1,2,3,4,5,6,7,8,9\}\)는 tuning에 사용하지 않는다.
3. 모델별 동일한 수의 hyperparameter 후보를 허용한다.
4. **Protocol A:** 원 논문처럼 네 LR 각각 5 pilot runs를 실행하고 100 updates 시점의 평균 loss로 선택한다. 원 논문에서 선택된 값은 \(10^{-2}\)다.
5. **Protocol B:** ID validation task score만으로 primary checkpoint와 LR을 선택한다.
6. CA geometry metric을 보고 hyperparameter를 선택하면 해당 metric은 test evidence로 재사용하지 않는다.
7. RP threshold \(\epsilon\)도 held-out pilot에서 선택하고 main run 전에 고정한다.

### Frozen training budget

| 항목 | Protocol A: original paper | Protocol B: CA benchmark |
|---|---:|---:|
| Batch size | 64 | 256 |
| Updates | 5,000 | 10,000 |
| Optimizer | Adam | AdamW |
| Learning rate grid | \(10^{-2},10^{-3},10^{-4},10^{-5}\) | \(10^{-2},3\!\times\!10^{-3},10^{-3},3\!\times\!10^{-4}\) |
| Weight decay grid | 0 | \(0,10^{-5}\) |
| Gradient clipping | original setting 기록 | global norm 1.0 |
| Selection | 5-pilot, 100-update rule | ID validation only |

Primary Protocol B의 checkpoint score는

\[
L_{\mathrm{select}}=L_{\mathrm{ID}}
\]

로 둔다. Long-hold selection이 실제로 필요한 경우 별도 secondary track에서 validation hold \(H=500\)만 사용하고, 이 track의 OOD headline은 \(H\in\{1000,2000,4096\}\)로 제한한다. ID-selected와 hold-selected 결과를 섞지 않는다. Test bank와 CA geometry metric은 checkpoint 선택에 사용하지 않는다.

### CA-LRU RP schedule

Protocol A와 B가 같은 수의 RP decisions를 갖도록 비율을 고정한다.

| 항목 | Protocol A: 5,000 updates | Protocol B: 10,000 updates |
|---|---:|---:|
| Warm-up | 1,500 updates (30%) | 3,000 updates (30%) |
| RP interval | 50 updates | 100 updates |
| RP calls after warm-up | 70 | 70 |
| Probe batch | 256 | 256 |
| Probe horizon | \(\min(T,256)\) | \(\min(T,256)\) |
| Probe noise | off | off |

각 RP call의 probe batch는 call index로 생성한 deterministic bank를 사용하고 coordinate 간 완전히 동일하게 한다. \(\eta_\lambda\), damage normalization, clipping, threshold는 5 pilot seeds에서 동결한다. Calls 수가 같으므로 per-call \(\eta_\lambda\)를 우선 유지하고, 별도 sensitivity에서만 budget-scaled 값을 비교한다.

## 1.8 결과 보고 원칙

각 조건에서 다음을 모두 보고한다.

- `n_trained`
- `n_task_success`
- task success rate
- all-seed metric
- successful-seed conditional metric
- mean, standard deviation, median
- seed-level bootstrap 95% CI
- 실패 seed의 failure mode

Ságodi처럼 success cutoff를 사용할 수 있지만, 실패 모델을 조용히 제외하지 않는다.

---

# 2. Task 및 데이터 생성

## 2.1 데이터 저장 원칙

Training은 Ságodi처럼 online fresh sampling을 사용한다. Evaluation은 재현성과 paired comparison을 위해 fixed bank를 사용한다.

```text
eval_banks/
  sagodi_mgs_id_seed000.npz
  sagodi_angle_id_seed000.npz
  torus_d02_id_seed000.npz
  torus_d04_mixed_seed000.npz
  torus_d08_coupled_seed000.npz
  perturb_ring_grid_seed000.npz
  parameter_perturb_seed000.npz
```

각 bank에는 최소 다음을 저장한다.

```yaml
metadata:
  task_name: string
  task_version: git_commit_or_protocol_version
  task_seed: integer
  delta_t: float
  horizon: integer
  latent_dimension: integer
  input_dimension: integer
  output_dimension: integer
  gp_length_scale: float_or_null
  velocity_scale: float_or_null
  mixing_matrix_A: array_or_null
  coupling_matrix_B: array_or_null
arrays:
  q0: B_by_d
  inputs: B_by_T_by_din
  latent_targets: B_by_T_by_d
  output_targets: B_by_T_by_dout
  mask: B_by_T_by_dout
```

## 2.2 Task A: Memory-guided saccade — Ságodi reproduction

### Latent

\[
\theta\sim U(0,2\pi),
\qquad y(\theta)=(\cos\theta,\sin\theta).
\]

Frozen shapes는 input \(512\times3\), output \(512\times2\), mask \(512\times2\)다.

### Trial

| 구간 | 기본 길이 | 입력 | Target | Mask |
|---|---:|---|---|---:|
| Initial blank | 5 | 0 | 0 | 1 |
| Target cue | 5 | \((\cos\theta,\sin\theta,0)\) | 0 | 1 |
| Delay | \(D\sim\operatorname{DiscreteUniform}\{50,\ldots,399\}\) | 0 | 0 | 1 |
| Go cue | 5 | \((0,0,1)\) | 0 | 1 |
| Response transition | 5 | 0 | 0 | 0 |
| Response | 남은 길이 | 0 | \((\cos\theta,\sin\theta)\) | 1 |

총 길이는 512 steps로 고정한다. 논문의 \(U(50,400)\) 표기와 공개 generator의 half-open integer sampling을 일치시키기 위해 실제 delay support를 50–399로 동결한다. 위 5-step cue 구조와 go 직후 5-step mask도 generator version과 함께 저장한다.

### Evaluation

- ID bank: 4,096 trials
- Delay OOD: \(D\in\{500,1000,2000\}\), trial length를 delay에 맞춰 확장
- Angle grid: 1,024 uniformly spaced angles
- Cue corruption: cue amplitude \(\{0.5,0.75,1.0,1.25,1.5\}\)
- Go-cue timing OOD: early/late query

## 2.3 Task B: Angular-velocity integration — Ságodi reproduction

### GP velocity

Normalized time grid

\[
r_t=\operatorname{linspace}(-1,1,T)_t
\]

에서

\[
\omega_{1:T}\sim\mathcal N(0,K),
\]

\[
K_{tt'}
=
\sigma_v^2
\exp\left[-\frac{(r_t-r_{t'})^2}{2\ell^2}\right],
\qquad
\ell=1,\quad \sigma_v=1.
\]

원 논문 primary condition은 \(\ell=1\)만 사용한다. \(r_t=\operatorname{linspace}(-1,1,T)\)는 재현 가능한 discretization으로 동결하며 GP Cholesky factor는 task seed별로 한 번 계산한다. OOD \(\ell\)-sweep은 원 논문 재현이 아니라 본 연구의 확장이고, 같은 \(r_t\)를 유지한 채 kernel denominator의 \(\ell\)만 바꾸는 통상적 convention을 사용한다. 따라서 큰 \(\ell\)은 더 smooth한 trajectory를 뜻한다.

### Dynamics와 target

\[
\theta_0\sim U(-\pi,\pi),
\]

\[
\theta_{t+1}=\operatorname{wrap}
(\theta_t+\Delta t\,\omega_t),
\]

\[
y_t=(\cos\theta_t,\sin\theta_t).
\]

Indexing은 다음으로 동결한다. Initial encoder/cue는 velocity 적용 전 \(q_0\)를 나타내고, velocity token \(\omega_t\)를 처리한 step의 target은

\[
y_{t+1}
=
(\cos\theta_{t+1},\sin\theta_{t+1})
\]

이다. 즉 pre-update initial memory와 post-update velocity target을 구분한다. 모든 baseline, hidden-init, cue-driven에 같은 convention을 적용한다.

### 기본 설정

| 항목 | 값 |
|---|---:|
| Steps | 256 |
| \(\Delta t\) | 0.1 |
| Input dimension | 1 |
| Output dimension | 2 |
| GP length scale | 1 |
| Loss mask | 모든 time step 1 |
| Training sampling | online |

### 초기 상태 두 track

#### B1. Ságodi-matched hidden initialization

\[
h_0=E_{\mathrm{init}}(\cos\theta_0,\sin\theta_0).
\]

모든 baseline에 동일한 learned output-to-hidden initialization mapping을 제공한다.

#### B2. Cue-driven initialization

모든 recurrent state를 0으로 시작하고 initial angle을 별도의 한 token으로 prepend한다.

\[
x_0=[\cos\theta_0,\sin\theta_0,0,1],
\qquad
x_{t+1}=[0,0,\omega_t,0].
\]

총 sequence는 257 tokens이며 첫 cue token은 loss에서 제외한다. 이후 256 velocity step에 target을 계산한다. Cue token 하나가 추가되는 효과를 피하기 위해 training budget은 loss-bearing token 수로 맞춘다. B2가 task geometry의 data-driven emergence를 주장하는 primary extension이다. B1과 B2 결과를 섞지 않는다.

CA-LRU가 velocity feature \(\nu(\omega)=[\omega,\sin\omega,\cos\omega-1]\)를 사용한다면 모든 baseline에 동일 feature를 제공하고 auxiliary ablation으로 둔다. Ságodi 직접 비교의 primary input은 raw \(\omega\)다.

### Evaluation

- ID: 4,096 GP trajectories
- Long horizon: \(\{2T,4T,8T,16T\}\)
- Velocity scale: \(\{0.5,1.0,1.5,2.0\}\times\)
- GP length scale: \(\ell\in\{0.5,1,2,4\}\)
- Zero-velocity hold: 100/500/1,000/2,000 steps
- Constant velocity: clockwise/counter-clockwise transport

## 2.4 Task C: Double angular integration

두 개의 독립 각도를 적분한다.

\[
q_t=(\theta_t^{(1)},\theta_t^{(2)})\in\mathbb T^2.
\]

\[
\omega^{(1)},\omega^{(2)}\overset{\mathrm{ind}}\sim
\mathcal{GP}(0,K_\ell).
\]

\[
y_t=(
\cos\theta_t^{(1)},\sin\theta_t^{(1)},
\cos\theta_t^{(2)},\sin\theta_t^{(2)}).
\]

논문에 정확한 double-task 길이가 명시되지 않았으므로 본 protocol에서는 256 steps로 고정하고 **CA-LRU-defined reproduction setting**이라고 명시한다.

Hidden-init의 input/output shape는 \(256\times2\)와 \(256\times4\)다. Cue-driven은 \(\Phi_2(q_0)\) cue token을 prepend하고 이후 두 velocity를 제공한다. Single-angle task와 마찬가지로 첫 cue token을 loss에서 제외한다.

### Analysis grid

- (64\times64=4,096) initial-angle grid
- 두 각도 각각 64 bins
- Zero-input autonomous rollout: (16T)
- Geodesic error: 두 wrapped angular error의 Euclidean norm과 합을 모두 저장
- Ságodi reproduction의 double MAE: 두 ring의 개별 angular error 합
- Ságodi reproduction의 uniform norm: 두 projected vector-field component uniform norm의 합
- Fixed-point iteration convergence threshold: \(10^{-4}\)
- Distinct fixed-point separation threshold: \(10^{-2}\)

## 2.5 Task D: \(d\)-dimensional torus integration

### Latent와 canonical output

\[
q_t=(\theta_{t,1},\ldots,\theta_{t,d})\in\mathbb T^d,
\qquad
d\in\{1,2,4,8\},
\]

\[
q_0\sim U([-\pi,\pi)^d),
\]

\[
\Phi_d(q)
=
[\cos\theta_1,\sin\theta_1,\ldots,
\cos\theta_d,\sin\theta_d]^\top
\in\mathbb R^{2d}.
\]

각 coordinate의 canonical velocity는 독립 GP에서 생성한다.

\[
v_{:,j}\sim\mathcal N(0,K),
\qquad
j=1,\ldots,d.
\]

두 scaling track을 모두 실행한다.

- **Per-coordinate:** 모든 \(d\)에서 \(\sigma_v=1\)
- **Energy-matched:** \(\sigma_v(d)=1/\sqrt d\)

첫 번째는 병렬 memory load를, 두 번째는 총 input energy를 통제한 dimension scaling을 측정한다.

### Hidden-init와 cue-driven

Hidden-init는 \(s_0=E_{\mathrm{init}}(\Phi_d(q_0))\)를 사용한다. Cue-driven token은

\[
x_0=[\Phi_d(q_0);0_d;1],
\qquad
x_{t+1}=[0_{2d};a_t;0].
\]

Raw observed velocity가 \(d\)차원이므로 cue-driven input dimension은 \(3d+1\)이다.

## 2.6 고차원 난이도 단계

### D1. Factorized

\[
q_{t+1}
=
\operatorname{wrap}
\left(q_t+\Delta t\,v_t\right),
\qquad
a_t=v_t,
\qquad
y_t=\Phi_d(q_t).
\]

목적은 intrinsic memory dimension 자체의 scaling을 측정하는 것이다.

### D2. Observation-mixed

\[
a_t=Qv_t,\qquad Q\in O(d),
\]

\[
y_t=M\Phi_d(q_t),\qquad M\in O(2d).
\]

Latent dynamics는 factorized이지만 모델이 보는 input/output coordinate는 섞인다. Generator는 \(v_t=Q^\top a_t\)로 latent를 update한다. \(Q\)와 \(M\)은 Gaussian matrix의 QR decomposition 후 diagonal-sign correction으로 Haar orthogonal matrix를 만들고 task-map seed마다 고정한다. 이 condition은 더 어려운 dynamics라기보다 basis reparameterization consistency를 검사한다.

### D3. State-dependent coupled

단순한 constant dense mixing은 coordinate change로 다시 분리될 수 있으므로 stronger condition에서는 state-dependent coupling을 사용한다.

\[
q_{t+1}
=
\operatorname{wrap}
\left[
q_t+\Delta t\,B(q_t)v_t
\right],
\]

\[
B(q)
=
I+\alpha C\operatorname{diag}(\cos q),
\qquad
\alpha=0.25.
\]

\(C\)는 diagonal이 0인 random matrix로 생성하고 spectral norm을 1로 정규화한다. Observation에는 D2의 \(Q,M\)도 적용한다. \(v_t=0\)이면 \(q_{t+1}=q_t\)이므로 blank 상태에서는 torus 전체가 fixed일 수 있지만, 이동 중에는 coordinate가 비선형적으로 결합된다. \(d=1\)에는 이 condition을 적용하지 않는다.

| 이름 | Latent dynamics | Observation | 검증 목적 |
|---|---|---|---|
| Factorized | independent | identity | memory-dimension scaling |
| Observation-mixed | independent | random \(Q,M\) | task-coordinate invariance |
| State-dependent coupled | nonlinear coupled \(B(q)\) | random \(Q,M\) | parallel independent-ring shortcut 억제 |

동일 task-map seed에서는 모든 모델에 같은 \(Q,M,C\)를 제공한다.

## 2.7 Segmented move–hold extension

Ságodi GP integration과 CA-LRU의 hold 특성을 연결하기 위해 별도 extension을 둔다.

```text
initial cue → GP move → zero-input hold → GP move → long zero-input hold
```

기본 segment distribution:

| Segment | 길이 |
|---|---|
| Initial cue | 5 |
| Move 1 | \(U\{32,\ldots,64\}\) |
| Hold 1 | \(U\{32,\ldots,64\}\) |
| Move 2 | \(U\{32,\ldots,64\}\) |
| Final hold | \(256-(5+L_{\mathrm{move1}}+L_{\mathrm{hold1}}+L_{\mathrm{move2}})\) |

따라서 모든 trial은 정확히 256 steps이고 final hold는 59–155 steps다.

이 task는 다음을 같은 trial에서 분리한다.

- input-driven tangent transport
- blank-input tangential drift
- unperturbed hold의 tangential drift
- analysis-only bank에서 hold 시작 직후 kick을 넣은 radial/normal recovery
- hold 뒤 integration 재개

## 2.8 Static (\mathbb T^d) memory-guided task

Memory-guided saccade를 \(d\)차원으로 확장한다.

1. \(q\sim U(\mathbb T^d)\)
2. 5-step cue \(M\Phi_d(q)\)
3. \(D\sim U\{50,\ldots,399\}\) blank delay
4. 5-step go cue
5. 5-step transition mask
6. \(M\Phi_d(q)\) 출력

이 task는 integration 없이 pure memory manifold를 분석할 수 있게 한다.

## 2.9 Train/validation/evaluation 분리

### Training

- Online fresh batches
- Main horizon: 256
- \(q_0\), GP trajectory, segment length를 매 update 새로 샘플링
- Generator key를 \((\text{task},d,\text{geometry},\text{init mode},\text{task seed},\text{update index})\)로 고정하여 같은 condition의 모든 모델이 동일 batch를 받게 한다.

### Validation

- 조건별 fixed 2,048 trials
- Checkpoint/LR/RP-threshold 선택에만 사용
- Main test 및 CA metric에 재사용하지 않음

### ID test

- Fixed 4,096 trials
- 모든 모델·seed에 동일 bank

### OOD banks

- 각 horizon/velocity/GP-length condition당 4,096 trials
- state-noise bank는 ID trial과 noise tensor를 pair로 저장
- MGS long-delay bank는 \(T_{\mathrm{trial}}=D+20+T_{\mathrm{response}}\), \(T_{\mathrm{response}}=128\)로 생성
- 한 bank에서는 한 distribution axis만 변경
- 모든 \(q_{0:T}\), canonical \(v_{0:T}\), observed input, target, mask, \(Q,M,C\), seed를 함께 저장
- Test/OOD bank는 LR, checkpoint, RP threshold 선택에 사용하지 않음

### Manifold bank

| \(d\) | Sampling | 개수 |
|---:|---|---:|
| 1 | uniform grid | 1,024 |
| 2 | \(64\times64\) grid | 4,096 |
| 4 | scrambled Sobol | 8,192 |
| 8 | scrambled Sobol | 16,384 |

### Endpoint path bank

같은 \(q\)에 서로 다른 integration path로 도달하도록 endpoint당 8 trajectories를 생성한다. Path dependence와 decoded-null fiber를 분석한다.

## 2.10 OOD matrix

| 축 | ID | OOD |
|---|---|---|
| Horizon | \(T\) | \(2T,4T,8T,16T\) |
| Blank hold | training distribution | 500,1,000,2,000,4,096 |
| Velocity | \(1.0\times\) | \(0.5,1.5,2.0\times\) |
| GP length | 1 | 0.5,2,4 |
| State perturbation | none | tangent/radial/ambient |
| Dynamics perturbation | none | parameter \(\epsilon\)-sweep |

다음은 OOD test가 아니라 **별도 재학습 축**이다.

| 축 | 별도 training conditions |
|---|---|
| Intrinsic dimension | \(d=1,2,4,8\) |
| Task basis | identity와 task-map seeds의 random \(Q,M\) |
| Dynamics family | factorized와 state-dependent coupled \(B(q)\) |

Input/output shape가 달라지는 dimension을 zero-shot OOD라고 부르지 않는다. 새로운 \(Q,M\) 또는 coupled dynamics도 해당 task family에서 다시 학습한 뒤 seed robustness/scaling으로 비교한다.

---

# 3. Analysis protocol

## 3.0 분석의 논리와 primary endpoint

분석은 “좋은 task score → CA”로 점프하지 않고 다음 순서로 수행한다.

| 순서 | 질문 | Primary endpoint | CA 조건 |
|---:|---|---|---|
| A | 과제를 실제로 푸는가? | ID/OOD geodesic error, NMSE, success rate | 전제 |
| B | 기억 latent와 대응하는 manifold가 있는가? | held-out cycle error, local rank, topology | C1 |
| C | blank일 때 manifold 위 flow가 충분히 느린가? | worst-case drift와 tangent timescale | C2 |
| D | state를 밀어도 돌아오는가? | radial/ambient finite-kick recovery | C3 |
| E | dynamics를 바꿔도 구조가 남는가? | perturbed-model task·geometry·gap | C4 |
| F | manifold의 모든 점이 fixed인가? | full-state fixedness residual 전역분포 | exact CA만 |

논문의 primary CA endpoint는 다음 세 수치의 seed-level joint success로 둔다.

1. 행동 horizon 밖의 **tangential drift bound**
2. 여러 반경에서의 **radial 및 ambient recovery**
3. manifold 전역의 **tangent–normal timescale separation**

평균 task score나 2차원 그림 하나는 primary CA endpoint가 아니다.

## 3.1 Preflight: 분석 상태와 blank map audit

모델별로 분석 전에 다음 unit test를 통과시킨다.

### 3.1.1 State transition audit

1. 미래 transition에 영향을 주는 tensor를 모두 나열한다.
2. 각 tensor에 대해 “다음 step으로 전달 / 매 step overwrite / decoder 전용”을 기록한다.
3. primary state를 최소 Markov state로 pack/unpack하는 함수를 만든다.
4. batch dimension을 제외한 state dimension을 저장한다.
5. zero input의 dtype, normalization, bias channel, time/go flag를 모두 기록한다.

다음 세 map을 분리 저장한다.

\[
F_0^{\mathrm{primary}},\qquad
F_0^{\mathrm{carrier}},\qquad
F_0^{\mathrm{reported/full}}.
\]

주 CA 판정은 \(F_0^{\mathrm{primary}}\)를 사용한다. 나머지는 mechanism 해석용이다.

### 3.1.2 Autonomous-attraction audit

Radial perturbation 직후 20 step 동안 다음을 매 step 저장한다.

- 실제 input tensor와 그 norm
- carrier와 stream의 pre/post-update state
- reset/overwrite mask
- decoder input과 output
- \(F_0(s)-s\)
- nearest-manifold distance

모든 external input이 정확히 zero이고 외부 reset이 없을 때의 복원을 autonomous recovery로 분류한다. 그렇지 않으면 input-conditioned recovery로 따로 보고한다.

### 3.1.3 Determinism and numerical checks

- 분석 중 dropout, training noise, data augmentation을 끈다.
- 같은 state와 input을 두 번 넣었을 때 bitwise 또는 tolerance 내 동일한지 확인한다.
- float64 재분석 subset으로 float32 결과의 민감도를 확인한다.
- autograd Jacobian을 16개 무작위 방향의 finite difference와 비교한다.
- step index나 hidden cache가 함수 밖에 남아 있지 않은지 확인한다.

이 audit를 통과하지 못한 모델에는 “autonomous CA” 분석을 적용하지 않는다.

## 3.2 Task performance와 run inclusion

### 3.2.1 공통 task metric

원 논문 loss mask \(m_{bt}\)가 1인 위치만 사용한 target power-normalized MSE와 dB 값을 primary task metric으로 둔다.

\[
\operatorname{NMSE}
=
\frac{\sum_{b,t}m_{bt}\|y_{bt}-y^*_{bt}\|_2^2}
{\sum_{b,t}m_{bt}\|y^*_{bt}\|_2^2+\varepsilon},
\qquad
\operatorname{NMSE}_{\mathrm{dB}}=10\log_{10}\operatorname{NMSE}.
\]

Full-sequence unmasked NMSE는 descriptive metric으로만 보고한다. Task별 success gate는 이 masked NMSE 하나로 동결한다.

원형 latent의 wrapped error는

\[
\delta(\hat\theta,\theta)
=
\operatorname{wrap}_{[-\pi,\pi)}(\hat\theta-\theta)
\]

로 계산한다. 이후 모든 claim gate에서 사용하는 \(\mathbb T^d\)의 geodesic distance는

\[
d_{\mathbb T^d}(q,q')
=
\frac{1}{\pi}
\left(\frac{1}{d}\sum_{j=1}^{d}\delta_j^2\right)^{1/2}
\in[0,1]
\]

로 정규화한다. 모든 \(0.05/0.10\) geodesic threshold는 이 \(\pi\)-normalized 단위를 뜻한다. 이해를 위해 radian과 degree 단위도 함께 보고한다.

모든 task에서 다음을 저장한다.

- full-sequence NMSE와 masked NMSE
- final-step 및 hold-only geodesic error
- dimension별 angular error
- trial 50/90/95/99 percentile
- validation에서 동결한 success threshold 통과 여부

### 3.2.2 Run inclusion rule

- 학습을 시작한 모든 seed를 success-rate 분모에 포함한다.
- 학습 실패, NaN, cutoff 미달을 숨기지 않고 failure mode로 분류한다.
- Manifold fitting은 task-success seed에만 수행할 수 있지만, 표에는 all-seed 결과와 success-conditional 결과를 둘 다 둔다.
- Ságodi의 \(-20\) dB cutoff는 분석 가능한 network를 선별하는 기준이지 baseline 성능표에서 seed를 삭제하는 기준이 아니다.

## 3.3 Manifold state bank 구축

### 3.3.1 Ring: Ságodi-style slow-manifold reconstruction

Ságodi et al.의 Methods를 따르는 ring 분석은 다음 paper-faithful frozen interpretation을 사용한다. 원문에 없는 trajectory-initialization 세부는 manifest에 별도로 명시한다.

1. 학습 noise를 끈다.
2. task-success network에서 1,024개 task trajectory를 생성한다.
3. Task-defined initialization에서 input-free autonomous dynamics를 task length의 \(16T\)까지 전개한다. “마지막 task state”를 시작점으로 쓰는지, 균일 memory initialization을 쓰는지는 task별로 동결하여 기록한다.
4. 각 trajectory에서 speed가 그 trajectory의 최대 speed의 \(10^{-3}\)보다 작은 state를 slow-state candidate로 모은다.
5. 1,024개 균일한 target angle마다 decoded output이 가장 가까운 candidate를 고른다.
6. 선택된 state를 angle 순서로 정렬하고 periodic cubic spline을 fitting한다.
7. spline에서 1,024개 equally spaced anchor를 다시 샘플링한다.

여기서 \(10^{-3}\)은 Ságodi-style reconstruction heuristic이다. fixed point 또는 CA의 수학적 판정 threshold로 사용하지 않는다.

이 방식이 실패하면 실패 자체를 보고하고, 아래의 task-conditioned atlas를 보조 분석으로 사용한다.

### 3.3.2 Task-conditioned atlas: 모든 \(d\)

Known latent \(q\)를 사용해 anchor map \(s(q)\)를 직접 만든다.

| \(d\) | Anchor | Primary Jacobian subset |
|---:|---|---:|
| 1 | 1,024 uniform angles | 256 stratified points |
| 2 | \(64\times64=4,096\) grid | 256 stratified points |
| 4 | 8,192 scrambled Sobol | 512 Sobol points |
| 8 | 16,384 scrambled Sobol | 512 Sobol points |

각 anchor \(q\)에 대해 다음 state를 별도로 저장한다.

1. cue가 끝난 직후의 canonical state
2. 정해진 integration path로 \(q\)에 도달한 endpoint state
3. 같은 \(q\)에 8개 서로 다른 path로 도달한 endpoint states
4. endpoint에서 \(H_{\mathrm{settle}}\in\{0,5,20,100\}\) step blank rollout한 state

Task endpoint sheet를 곧바로 invariant/slow manifold라고 부르지 않는다. Pilot에서 각 task/model family에 대해 다음을 만족하는 가장 작은 \(H_{\mathrm{settle}}\)을 동결한다.

1. 같은-\(q\) path variance의 다음 5-step relative decrease가 1% 미만
2. settling 구간의 normalized geodesic drift가 0.01 미만
3. transverse residual이 더 이상 체계적으로 감소하지 않음

100 step 안에 이 기준을 만족하지 않으면 “transient task sheet”로 표시하고 L2/L3 manifold analysis를 통과시키지 않는다. Primary atlas는 동결된 settled state를 사용하고, \(H=0,5,20,100\) 결과를 sensitivity로 함께 보고한다.

### 3.3.3 Atlas normalization

Hidden-state distance는 모델마다 scale이 다르므로

\[
R_s
=
\operatorname{median}_{q}
\|s(q)-\bar s\|_2
\]

를 state scale로 사용한다. Distance, kick radius, fixedness residual은 \(R_s\)로 정규화한다. \(R_s\)가 0에 가까운 collapse model은 C1 실패로 처리한다.

\(R_s\)-normalized hidden distance도 arbitrary coordinate rescaling에 완전히 불변하지는 않다. 따라서 architecture 간 headline 비교는 latent geodesic error, dimensionless recovery ratio, pass rate를 사용하고 raw hidden norm 자체의 크기로 우열을 주장하지 않는다. LSTM의 \(h/c\), CA-LRU의 carrier/stream처럼 성격이 다른 block은 block별 scale과 full-state 결과를 함께 보고한다.

Nearest-manifold projection \(\Pi_{\mathcal M}\)은 \(d\le2\)에서 다음 순서로 구현한다.

1. atlas exact nearest neighbor 검색
2. 해당 anchor와 이웃을 이용한 local tangent-plane projection
3. projected point의 latent coordinate를 periodic interpolation

Ring에서는 dense spline projection을 함께 계산해 atlas discretization error를 확인한다.

\(d\ge4\)에서 sparse Sobol nearest neighbor를 정밀 projection으로 사용하지 않는다. Atlas neighbor와 diagnostic decoder로 latent \(q\)를 초기화한 뒤

\[
q^*
=
\arg\min_{q\in\mathbb T^d}
\|s-G(q)\|_2^2
\]

를 8개 multistart periodic optimizer로 푼다. Known-\(q\) held-out states와 synthetic off-manifold states에서 projection error를 측정하며, 95th-percentile projection error가 invariance gate의 20%보다 작아야 해당 gate를 판정한다. 그렇지 않으면 projection-based metric은 inconclusive로 처리한다. Anchor에서 시작한 local kick에는 known local chart를 우선 사용한다.

## 3.4 C1: correspondence, dimension, topology

### 3.4.1 Decoder와 cycle consistency

학습 readout과 별도로 training-atlas만 사용한 diagnostic latent decoder \(D_{\mathrm{lat}}\)를 fitting한다. Evaluation anchor는 held-out으로 둔다.

\[
q
\xrightarrow{s}
s(q)
\xrightarrow{D_{\mathrm{lat}}}
\hat q,
\qquad
e_{\mathrm{cycle}}
=
\mathbb E_q\,d_{\mathbb T^d}(\hat q,q).
\]

Linear decoder와 2-layer diagnostic decoder를 모두 보고한다. 비선형 decoder만 성공하면 manifold는 존재할 수 있지만 선형 task alignment는 약하다고 해석한다.

### 3.4.2 Local tangent rank

Differentiable cue/path generator에서는 autodiff/JVP로 latent Jacobian을 계산한다. Float64 periodic central difference는 numerical check로 사용한다.

\[
S_j(q)
=
\frac{s(q+\epsilon_q e_j)-s(q-\epsilon_q e_j)}
{2\epsilon_q},
\qquad
S(q)=[S_1,\ldots,S_d].
\]

Finite-difference check는
\(\epsilon_q\in\{10^{-3},5\times10^{-3},10^{-2}\}\,\mathrm{rad}\)에서 convergence를 확인한다. QR decomposition

\[
S(q)=T(q)R(q)
\]

으로 orthonormal tangent basis \(T(q)\)를 얻는다.

다음 분포를 manifold 전역에서 보고한다.

- \(\operatorname{rank}S(q)\)
- \(\sigma_{\min}(S(q))\)
- \(\sigma_{\max}(S(q))/\sigma_{\min}(S(q))\)
- 이웃 anchor 사이 tangent principal angle

Local rank threshold는 singular value를 state scale과 \(\epsilon_q\)로 정규화한 뒤 pilot에서 동결한다.

### 3.4.3 Path independence와 fiber structure

같은 \(q\)에 도달한 path \(p=1,\ldots,8\)에 대해

\[
V_{\mathrm{within}}(q)
=
\frac{1}{8}\sum_p
\frac{\|s_p(q)-\bar s(q)\|_2^2}{R_s^2}
\]

와 nearest-neighbor between-\(q\) separation을 비교한다.

- \(V_{\mathrm{within}}\ll V_{\mathrm{between}}\): near-single-sheet manifold
- hidden state는 다르지만 decoder와 future behavior가 같음: output-null fiber
- path에 따라 future output이 달라짐: C1 correspondence failure

Near-single-sheet gate는

\[
\chi_{\mathrm{fiber}}
=
\frac{\operatorname{median}_q V_{\mathrm{within}}(q)}
{\operatorname{median}_q V_{\mathrm{between}}(q)+\varepsilon}
\le0.1
\]

로 사전등록한다.

Fiber가 있으면 primary state가 \(\mathbb T^d\) 자체와 일대일이라고 주장하지 않고, memory quotient 또는 fibered memory manifold라고 기술한다. 이때 \(T=\partial s/\partial q\)만으로 만든 \(I-TT^\top\)는 fiber tangent를 잘못 normal로 분류하므로 다음 분기를 강제한다.

1. \(\chi_{\mathrm{fiber}}\le0.1\): near-single-sheet pipeline을 사용한다.
2. \(\chi_{\mathrm{fiber}}>0.1\): 같은-\(q\) path states의 local PCA로 fiber tangent \(T_F\)를 추정하고, memory-horizontal tangent \(T_M\)를 \(T_F\)에 직교화한 뒤
   \[
   T_{\mathrm{full}}=\operatorname{qr}[T_M,T_F]
   \]
   를 full manifold tangent로 사용한다.
3. Fiber dimension과 future-behavior equivalence를 안정적으로 추정하지 못하면 strict C1 approximate bijection과 L2/L3 gate는 실패로 처리한다.

Fibered solution은 Ságodi의 strict C1 bijection과 다른 generalized solution이므로 main result와 별도 표에 둔다.

### 3.4.4 Topology

Primary evidence는 known-latent mapping, local rank, no-collapse, neighborhood 보존이다. 다음을 함께 보고한다.

- latent k-nearest-neighbor recall in hidden space
- hidden-to-latent trustworthiness/continuity
- pairwise latent geodesic distance와 hidden distance의 Spearman correlation
- periodic seam 전후의 continuity

Persistent homology는 \(d\le2\)에서 보조 증거로 사용한다.

| Manifold | 기대 Betti number |
|---|---|
| \(S^1\) | \(\beta_0=1,\;\beta_1=1\) |
| \(\mathbb T^2\) | \(\beta_0=1,\;\beta_1=2,\;\beta_2=1\) |

\(\mathbb T^2\)의 4,096 points에는 full dense Vietoris–Rips complex를 직접 만들지 않고 1,024-point stratified subsample, sparse Rips 또는 witness complex를 사용한다. \(d>2\)에서는 sample complexity 때문에 persistent homology를 pass/fail gate로 사용하지 않는다. Known latent atlas의 rank와 correspondence가 primary evidence다.

---

## 3.5 C2와 exactness audit: fixedness, invariance, slow flow

### 3.5.1 서로 다른 residual을 섞지 않는다

각 anchor \(q\)에서 다음 네 값을 계산한다. Moving-frame latent coordinate는 diagnostic decoder가 아니라 nearest-manifold projection으로 정한다.

\[
q_{\mathcal M}^{+}(q)
=
\operatorname{coord}
\left[
\Pi_{\mathcal M}(F_0(s(q)))
\right].
\]

\[
r_{\mathrm{fp}}(q)
=
\frac{\|F_0(s(q))-s(q)\|_2}{R_s+\varepsilon},
\]

\[
r_{\mathrm{inv}}(q)
=
\frac{
\operatorname{dist}(F_0(s(q)),\mathcal M)
}{R_s+\varepsilon},
\]

\[
r_{\mathrm{dec}}(q)
=
d_{\mathbb T^d}
\left(
\hat q(F_0(s(q))),q
\right),
\]

\[
v(q)
=
\frac{
\operatorname{wrap}(q_{\mathcal M}^{+}(q)-q)
}{\pi\Delta t}.
\]

의미는 각각 다르다.

| 값 | 작다는 의미 | 작아도 보장하지 않는 것 |
|---|---|---|
| \(r_{\mathrm{fp}}\) | full state가 one-step fixed에 가까움 | normal attraction |
| \(r_{\mathrm{inv}}\) | next state가 manifold에 가까움 | 기억 coordinate 보존 |
| \(r_{\mathrm{dec}}\) | decoded memory가 한 step 보존됨 | full-state fixedness |
| \(v(q)\) | on-manifold latent flow가 느림 | exact fixed point |

따라서 exact CA 판정에는 \(r_{\mathrm{fp}}\)가 필요하고, approximate CA 판정에는 \(r_{\mathrm{inv}}\), \(v(q)\), long-horizon drift가 핵심이다.

### 3.5.2 Uniform slow-flow analysis

Ságodi의 핵심 분석에 맞추어

\[
\eta_{\infty}
=
\max_{q\in\mathrm{atlas}}\|v(q)\|_{\mathbb T^d}
\]

와 95/99/99.5 percentile을 보고한다. Atlas density를 2배로 한 subset에서 max가 안정적인지도 확인한다.

Blank horizon은

\[
H\in\{1,5,20,100,500,1000,2000,4096\}
\]

로 고정하고 다음을 그린다.

\[
e_{\mathrm{drift}}(H,q)
=
d_{\mathbb T^d}
\left(
\hat q(F_0^H(s(q))),q
\right).
\]

Ságodi의 bound에 대응하여 물리시간 \(t=H\Delta t\)에 대한

\[
H\Delta t\,\eta_\infty
\]

와 empirical worst/mean drift를 같은 축에 표시한다. Bound용 \(\eta_\infty\)는 초기 atlas뿐 아니라 clean rollout이 방문한 모든 projected state에서 계산한다. Invariance가 깨지거나 sampling coverage가 불충분하면 theorem-level bound가 아니라 **empirical triangle-inequality proxy**라고 표기한다.

### 3.5.3 Tangent와 normal flow 분해

Local tangent projector \(P_T=T T^\top\)를 이용해

\[
\Delta s(q)=F_0(s(q))-s(q),
\qquad
\Delta s_T=P_T\Delta s,
\qquad
\Delta s_N=(I-P_T)\Delta s
\]

를 계산한다. \(\Delta s_T\)는 drift, \(\Delta s_N\)는 local first-order transverse diagnostic이다. Curved manifold에서는 finite tangent chord도 \(O(\|\Delta s\|^2)\) normal component를 가지므로 invariance의 primary 지표는 \(r_{\mathrm{inv}}\)로 둔다. State-scale normalized norm의 median, 95/99 percentile, max를 모두 저장한다.

### 3.5.4 Fixed-point search와 all-point fixedness

Ring에서는 다음 절차를 사용한다.

1. spline 위 decoded angular flow의 sign reversal을 찾는다.
2. periodic root refinement로 \(v(\theta)=0\)을 푼다.
3. full-state \(r_{\mathrm{fp}}\)를 확인한다.
4. 양쪽 flow가 root로 향하면 stable, 멀어지면 saddle로 분류한다.
5. stable basin의 angular size를 계산한다.

\(d>1\)에서는 1,024 Sobol initializations에서 intrinsic \(v(q)=0\) root search를 수행하되, fixed-point count를 완전한 enumeration이라고 부르지 않는다. 각 root에서 full-state \(r_{\mathrm{fp}}\)를 다시 검사하여 intrinsic-flow zero와 실제 fixed point를 구분한다.

**중요:** finite atlas에서 residual이 작다는 사실은 “모든 manifold point가 fixed”임을 증명하지 않는다. Exact CA는 구조적으로 \(F_0(s(q))=s(q)\)가 성립한다는 해석적 결과나, adaptive refinement에서 numerical precision에 가까운 uniform residual과 별도 구조적 논증이 있을 때만 주장한다.

## 3.6 Local 및 finite-horizon tangent–normal dynamics

### 3.6.1 Full nonlinear Jacobian

실제 blank map의 Jacobian은

\[
J_0(q)
=
\left.
\frac{\partial F_0(s)}{\partial s}
\right|_{s=s(q)}
\]

이다. Moving frame은 decoder extrapolation이 아니라

\[
q^+=q_{\mathcal M}^{+}(q)
\]

에서의 projected manifold basis를 사용한다. Near-single-sheet가 아니면 \(T\) 대신 Section 3.4.3의 \(T_{\mathrm{full}}\)을 쓴다.

\[
A_{TT}(q)=T(q^+)^\top J_0(q)T(q),
\]

\[
A_{NT}(q)=N(q^+)^\top J_0(q)T(q),
\]

\[
A_{TN}(q)=T(q^+)^\top J_0(q)N(q),
\]

\[
A_{NN}(q)=N(q^+)^\top J_0(q)N(q)
\]

를 계산한다. \(N\) 전체를 명시적으로 만들기 비싸면 projector 기반 randomized SVD와 Jacobian-vector product를 사용한다.

Primary 지표:

- tangent multiplier: singular values of \(A_{TT}\)
- tangent-to-normal leakage: \(\|A_{NT}\|_2\)
- normal-to-tangent leakage: \(\|A_{TN}\|_2\)
- worst-case normal gain: \(\sigma_{\max}(A_{NN})\)
- eigenvalue spectrum: secondary local stability description
- continuous-time equivalent rate: \(r=\log|\mu|/\Delta t\)

비정규 dynamics에서는 \(|\lambda|<1\)이어도 transient amplification이 가능하므로 normal eigenvalue만으로 contraction을 주장하지 않는다.

### 3.6.2 Radial, retained-normal, ambient-normal block

Radial, retained, ambient direction은 자동으로 서로 직교하지 않는다. 다음 hierarchy로 QR/Gram–Schmidt orthogonalization한다.

\[
\mathcal N_q
=
\mathcal N_q^{\mathrm{radial}}
\oplus
\left(
\mathcal N_q^{\mathrm{retained}}
\ominus
\mathcal N_q^{\mathrm{radial}}
\right)
\oplus
\mathcal N_q^{\mathrm{ambient}}.
\]

- **Radial:** ring curvature 또는 centered radial vector가 만드는 task-active normal
- **Retained normal:** learned slow/retained subspace 안에서 tangent에 직교
- **Ambient normal:** retained subspace 밖의 directions

Retained subspace는 CA-LRU의 selected high-retention modes와 trajectory covariance slow subspace를 각각 사용한다. 두 정의의 principal angle도 보고하고, orthogonalization 뒤 남은 각 probe family의 rank를 저장한다. Radial vector가 ill-defined인 고차원 torus에서는 factor별 curvature normal

\[
n_{\kappa,j}(q)
\propto
(I-P_T)
\frac{\partial^2s(q)}{\partial q_j^2}
\]

을 사용한다.

### 3.6.3 Finite-horizon Jacobian product

One-step contraction만으로 장기 normal attraction을 판단하지 않는다.

\[
s_{h+1}=F_0(s_h),
\qquad
J_h=
\left.
\frac{\partial F_0}{\partial s}
\right|_{s=s_h},
\qquad
J_{0:H}=J_{H-1}\cdots J_0.
\]

Jacobian은 projected atlas anchor가 아니라 actual clean rollout state \(s_h\)에서 평가한다. 각 \(s_h\)의 frame coordinate \(q_h\)만 \(\Pi_{\mathcal M}(s_h)\)로 얻는다.

Strict transverse cocycle proxy는 매 step normal projection을 넣는다.

\[
C_N^{(H)}
=
P_N(q_H)J_{H-1}P_N(q_{H-1})
\cdots
P_N(q_1)J_0P_N(q_0).
\]

Endpoint에만 projection한 \(P_N(q_H)J_{0:H}P_N(q_0)\)도 finite-kick proxy로 저장하지만 strict bundle-contraction evidence와 구분한다.

\(H\in\{1,10,50\}\)에서 tangent exponent의 양 끝을 계산한다.

\[
\ell_T^{\min}(H)
=
\frac{1}{H}
\log\sigma_{\min}
\left(
T(q_H)^\top J_{0:H}T(q_0)
\right),
\]

\[
\ell_T^{\max}(H)
=
\frac{1}{H}
\log\sigma_{\max}
\left(
T(q_H)^\top J_{0:H}T(q_0)
\right),
\]

\[
\ell_N^{\max}(H)
=
\frac{1}{H}
\log
\|C_N^{(H)}\|_2
\]

를 계산한다. Empirical gap은

\[
\gamma_H(q)
=
\ell_T^{\min}(H)-\ell_N^{\max}(H)
\]

로 정의한다.

해석:

- \(\ell_N^{\max}<0\): worst-case normal contraction
- \(\gamma_H>0\): normal dynamics가 slowest tangent memory dynamics보다 빠름
- \(\ell_T^{\min}\approx0\) 및 \(\ell_T^{\max}\le0\): tangent memory가 neutral/non-expansive

Ságodi C3의 non-expansion을 직접 반영하여 \(H=50\)에서

\[
\exp(H\ell_T^{\max})\le1.05
\]

를 sampled tangent gate로 사용한다. 즉 50-step tangent perturbation amplification이 5%를 넘지 않아야 한다.

Atlas sample의 95%에서 \(\ell_N^{\max}<0\), \(\gamma_H>0\), tangent non-expansion이 유지되면 **95%-coverage empirical transverse-contraction evidence**라고 부른다. 이를 uniform/global normal hyperbolicity라고 부르지 않는다. Strong sampled-uniform claim에는 모든 sampled point에서 violation 0, adaptive midpoint refinement, worst-direction optimization, numerical/projection error upper bound를 추가로 요구한다.

### 3.6.4 CA-LRU state block audit

CA-LRU는 동일 anchor에서 다음을 병렬 보고한다.

1. primary full-Markov-state Jacobian
2. carrier-to-carrier block
3. carrier perturbation이 stream/output으로 전달되는 cross block
4. stream perturbation이 다음 carrier로 feedback되는 cross block

Full-state gain이 작은 이유가 stream overwrite뿐이고 carrier의 task-active radial direction은 줄지 않는다면, 이를 autonomous radial attraction의 핵심 증거로 쓰지 않는다. 반대로 actual \(F_0\)에서 carrier radial error와 decoded radial error가 모두 감소하면 **full end-to-end blank dynamics의 radial contraction**으로 해석한다. Radius/state에 따라 local gain이 유의하게 달라지는 것까지 확인한 뒤에만 contraction law 자체를 nonlinear이라고 부른다.

## 3.7 Nonlinear finite-perturbation experiment

### 3.7.1 Perturbation directions

각 seed/task의 256개 stratified anchor에서 다음 방향을 검사한다.

1. 각 latent axis의 tangent
2. ring centered radial
3. factor별 curvature radial
4. retained-subspace normal
5. ambient random normal 8개
6. one-step worst normal singular vector
7. \(H=50\) finite-horizon worst normal singular vector
8. isotropic full-state random direction

모든 vector는 primary state의 metric에서 unit norm으로 정규화하고, tangent/normal 직교 오차가 \(10^{-6}\) 이하인지 검산한다.

### 3.7.2 Perturbation scale과 horizon

\[
\rho/(R_s/\sqrt d)
\in
\{0.05,0.10,0.25,0.50,1.00\},
\qquad
H\in\{0,1,5,20,100,500,1000\}.
\]

Dimension-scaling primary radius는 \(\rho=0.10R_s/\sqrt d\), primary recovery horizon은 500 step으로 사전 고정한다. 이 값은 factor당 perturbation energy를 맞춘다. 별도로 \(\rho/R_s\)를 고정한 global-energy track도 보고한다. 큰 radius는 basin boundary를 찾는 stress test이며 local CA gate에는 사용하지 않는다.

### 3.7.3 Clean-paired recovery

Clean drift를 recovery로 오인하지 않도록 항상 clean trajectory와 paired한다.

\[
\delta s_H
=
F_0^H(s(q)+\rho n)
-
F_0^H(s(q)).
\]

다음을 저장한다.

\[
R_N(H)
=
\frac{
\|(I-P_T(q_H))\delta s_H\|_2
}{
\|(I-P_T(q_0))\delta s_0\|_2+\varepsilon
},
\]

\[
L_T(H)
=
\frac{\|P_T(q_H)\delta s_H\|_2}{\rho},
\]

\[
E_{\mathrm{excess}}(H)
=
d_{\mathbb T^d}
\left(
\hat q(s_H^{\mathrm{pert}}),
\hat q(s_H^{\mathrm{clean}})
\right).
\]

또한 manifold 자체로의 recovery와 같은-memory trajectory로의 recovery를 분리한다.

\[
R_{\mathcal M}(H)
=
\frac{
\operatorname{dist}(s_H^{\mathrm{pert}},\mathcal M)
}{
\operatorname{dist}(s_0^{\mathrm{pert}},\mathcal M)+\varepsilon
},
\]

\[
R_{\mathrm{same}}(H)
=
\frac{
\|s_H^{\mathrm{pert}}-s_H^{\mathrm{clean}}\|_2
}{
\|s_0^{\mathrm{pert}}-s_0^{\mathrm{clean}}\|_2+\varepsilon
}.
\]

\(R_{\mathcal M}<1\)인데 \(E_{\mathrm{excess}}\)가 크면 다른 memory point로 미끄러진 것이다. 이는 normal recovery와 memory preservation을 동시에 만족한 결과가 아니다.

### 3.7.4 Tangent equivariance

Tangent kick는 원래 \(q\)로 복원되어서는 안 되고 인접 memory로 이동해야 한다. Additive linearization test는 작은

\[
\delta q_j\in\{10^{-3},5\times10^{-3},10^{-2}\}\ \mathrm{rad}
\]

만 사용한다. 이에 대해

\[
E_T(H)
=
d_{\mathbb T^d}
\left[
\hat q
\left(
F_0^H(s(q)+S(q)\delta q)
\right),
\hat q
\left(
F_0^H(s(q+\delta q))
\right)
\right]
\]

를 측정한다. 작은 \(E_T\)는 tangent direction이 collapse되지 않고 memory transport를 수행한다는 증거다.

더 큰 shift \(\delta q_j/\pi\in\{0.01,0.05,0.10\}\)에서는 additive approximation \(s+S\delta q\)를 사용하지 않고, 실제 두 on-manifold anchor \(s(q)\), \(s(q+\delta q)\)의 clean transport와 pair separation을 비교한다.

## 3.8 C3와 C4 robustness

### 3.8.1 C3: S-type state perturbation

C3는 deterministic \(F_0\)를 고정한 채 다음 세 protocol로 평가한다.

1. 앞 절의 single finite kick
2. 매 step Gaussian state noise
3. 반복 kick sequence

Ságodi-matched noise는 coordinate standard deviation 0.1을 그대로 사용한다. Architecture-fair 비교는

\[
\xi_t
\sim
\mathcal N
\left(
0,
\frac{\sigma_s^2R_s^2}{n}I
\right),
\qquad
\sigma_s\in\{0.05,0.10,0.20\}
\]

를 사용한다. 각 level에서 256 initial memories, memory당 8 noise realizations을 고정 bank로 만든다.

반복 kick은 \(t\in\{0,50,100,150\}\)에 \(0.1R_s/\sqrt d\) normal kick을 주고, 각 kick 뒤 50-step recovery curve를 측정한다. 마지막 kick 뒤의 recovery만 떼어내는 것이 아니라 각 event를 clean trajectory에 맞춰 정렬한다.

Training 때 state noise를 사용했다는 사실은 C3의 배경일 뿐 직접 증거가 아니다. Evaluation perturbation에서 recovery를 측정해야 한다.

### 3.8.2 C4: D-type dynamics perturbation

Parameter group \(g\)에 대해

\[
\tilde\theta_g
=
\theta_g
+
\alpha
\frac{\|\theta_g\|_2}{\|\xi_g\|_2+\varepsilon}
\xi_g,
\qquad
\xi_g\sim\mathcal N(0,I)
\]

를 적용한다.

\[
\alpha
\in
\{10^{-4},3\times10^{-4},10^{-3},3\times10^{-3},10^{-2}\}.
\]

Core D-type parameter groups:

- retention logits/spectrum
- recurrent transition 전체
- blank map audit에서 확인된 radial-contraction source candidate

Encoder, readout, zero-input에서 비활성인 input writer는 autonomous \(F_0\)를 바꾸지 않으므로 C4 gate에서 제외하고 representation/input/observation robustness control로 별도 보고한다.

Relative parameter perturbation \(\alpha\)는 parameterization마다 실제 vector-field 변화가 다르다. 각 draw에서 realized perturbation을

\[
\delta_F^{95}
=
\operatorname{q95}_{s\in\mathcal M}
\frac{\|\tilde F_0(s)-F_0(s)\|_2}{R_s},
\]

\[
\delta_J^{95}
=
\operatorname{q95}_{s\in\mathcal M}
\|\tilde J_0(s)-J_0(s)\|_2
\]

로 측정한다. \(\alpha\)는 perturbation generator의 x축이고, architecture 간 C4 비교의 primary x축은 matched \(\delta_F^{95}\) bin이다.

각 perturbed model에서는 clean atlas를 그대로 평가하는 데 그치지 않고 동일 cue/path bank로 \(\tilde{\mathcal M}\)을 다시 구성한다. Clean/perturbed model은 이미 같은 state coordinate system이므로 same-\(q\) raw matched distance와 raw Hausdorff distance를 primary로 사용한다. Orthogonal Procrustes distance는 shape-only secondary metric이다.

- task success와 long-horizon drift
- C1 local rank/cycle error
- \(\eta_\infty\)와 \(r_{\mathrm{inv}}\)
- radial/ambient recovery
- \(\gamma_{50}\)
- matched-point RMS distance
- symmetric Chamfer와 95th-percentile Hausdorff distance

계산은 두 단계로 수행한다.

1. **Screening:** 모든 \(\alpha\)와 core group에서 5 draws, 256 anchors
2. **Primary:** \(\alpha=10^{-3}\)과 matched \(\delta_F^{95}\) bin에서 core group별 20 draws, full primary atlas

전체 5-level full sweep은 headline model/task/dimension에만 수행한다. Primary draw의 최소 90%가 Task, C1, C2, invariance, C3, tangent non-expansion, empirical normal-gap gate를 모두 유지해야 C4를 통과한다. \(\alpha=10^{-2}\)는 stronger-stress 결과로 분리한다.

State-noise training은 parameter/vector-field perturbation이 아니므로 C4의 직접 증거가 아니다.

---

## 3.9 Finite-time과 asymptotic dynamics

### 3.9.1 Finite-time generalization

모든 model/seed를 동일한 paired bank에서

\[
H\in\{T,2T,4T,8T,16T\}
\]

또는 absolute hold \(\{500,1000,2000,4096\}\)로 평가한다. Horizon을 바꿀 때 velocity, GP length scale, noise를 동시에 바꾸지 않는다.

보고 지표:

- component RMSE와 NMSE
- latent geodesic error
- drift slope와 saturation 여부
- task-success survival curve
- \(H\Delta t\,\eta_\infty\) bound/proxy 대비 empirical error

### 3.9.2 Ring asymptotics

Ságodi original-paper track에서는 fitted ring spline의 1,024 points와 local flow direction으로 stable/saddle fixed point 및 basin volume을 계산한다. Primary asymptotic memory metric은 basin probability \(p_y=\operatorname{vol}(\operatorname{Basin}(y))\)에서

\[
-H(X\mid Y)
=
\sum_y p_y\log p_y
\]

로 계산한다. Generic mutual information estimate와 혼합하지 않고 별도 열에 둔다.

Actual-network long rollout은 flow-based basin 판정의 validation으로 headline \(d=1\) model의 successful seeds에서 512 uniform angles을 최대 \(10^5\) blank steps까지 batched/early-stop 방식으로 실행한다.

| 판정 | Operational rule |
|---|---|
| Fixed point | normalized residual \(<10^{-6}\)가 100 step 연속 |
| Periodic orbit | \(p\le4096\)에서 recurrence error \(<10^{-4}\) |
| Not converged | 위 둘을 budget 안에 만족하지 않음 |

Stable fixed-point basin size, neighboring stable/saddle spacing, asymptotic angular error를 계산한다. Ságodi basin-volume metric과 별도로 decoder-based mutual information extension을 supplementary에 보고할 수 있다.

Ring이 장기적으로 몇 개 stable fixed point로 양자화되더라도 finite horizon에 approximate CA로 작동할 수 있다. 이 경우 “finite-time approximate CA with discrete asymptotic attractors”라고 명시한다.

\(d>1\)에서는 fixed-point enumeration을 완전하다고 주장하지 않는다. \(10^4\), \(10^5\) step의 empirical information retention과 root-search 결과를 보조적으로 보고한다.

## 3.10 Seed consistency와 해석 가능성

### 3.10.1 Seed 구조

다섯 seed를 분리 저장한다.

1. model initialization seed
2. online training-stream seed
3. task-map seed \(A,B,Q,M,C\)
4. evaluation-bank seed
5. perturbation/noise seed

주 비교는 동일 task/data/evaluation seed 아래 model seed 10개를 사용한다. Mixed/coupled geometry robustness는 별도 task-map seed 5개에서 반복한다. 계산 여유가 있으면 headline CA-LRU와 strongest baseline을 20 model seeds로 확장한다.

### 3.10.2 좌표 불변성과 기능적 일관성을 구분한다

개별 neuron 또는 raw coordinate의 일치를 기대하지 않는다. 동일 latent \(q\)의 atlas state matrix를 center/whiten하고, training anchor에서 orthogonal Procrustes 또는 CCA map을 fitting한 뒤 held-out anchor에 적용한다. Tangent principal angle과 coordinate-sensitive metric은 이 alignment 뒤에만 계산한다.

- orthogonal Procrustes residual
- linear CKA와 SVCCA
- alignment 후 matched \(q\)의 tangent-subspace principal angle
- pullback vector field \(v(q)\)의 correlation과 sup norm
- normal finite-time Lyapunov spectrum
- sorted retention spectrum과 effective retained dimension
- task tangent subspace와 retained subspace의 principal angle
- radial/ambient recovery curve의 seed variance

주 논문 표현은 다음처럼 제한한다.

> Across random initializations, coordinate realizations differ, while memory dimension, aligned geometry, tangential drift, and off-manifold recovery remain functionally consistent.

“Seed invariant representation”은 사전에 정한 강한 alignment threshold를 실제로 통과할 때만 쓴다. 기본 주장은 **seed-robust functional organization**이다.

### 3.10.3 Implicit geometry claim

CA-LRU가 explicit하게 설계한 것은 selective retention과 timescale bias다. Ring/torus의 좌표와 tangent–normal orientation 자체를 weight에 직접 심었다고 주장하지 않는다.

다음 ablation이 있어야 “geometry emerges implicitly from data”라는 주장이 강해진다.

1. task observation basis를 random rotation해도 결과 유지
2. mixed/coupled task map을 seed별로 바꿔도 topology와 CA metric 유지
3. carrier coordinate permutation 후 재학습에서 결과 유지
4. arbitrary orthogonal carrier rotation에서 mechanism이 유지되는지 별도 검사

마지막 rotation test를 통과하지 못하면 “basis-invariant mechanism”이라고 쓰지 않는다. 대신 “geometry-agnostic with respect to task coordinates”처럼 범위를 한정한다.

## 3.11 통계 분석

### 3.11.1 반복 단위

통계적 독립 반복 단위는 trajectory나 atlas point가 아니라 **학습된 model seed**다. 각 seed 안에서 trial/anchor/perturbation을 먼저 요약한 뒤 seed-level 통계를 계산한다.

### 3.11.2 필수 보고

- seed-level mean과 median
- standard deviation과 IQR
- seed bootstrap 95% CI
- task/CA gate success rate와 Wilson 95% CI
- all-seed 결과와 task-success conditional 결과
- paired seed difference와 effect size

Atlas point와 perturbation draw의 uncertainty가 필요하면 seed를 먼저 resample하고 그 안에서 point/draw를 resample하는 hierarchical bootstrap을 사용한다.

### 3.11.3 모델 비교

같은 training/evaluation stream으로 pairing된 seed에는 paired permutation test 또는 exact Wilcoxon signed-rank test를 사용한다. Effect size는 paired median difference 또는 Hodges–Lehmann estimate를 보고한다. Core baseline × task × dimension의 다중 비교에는 Holm correction을 적용한다.

Trajectory 수천 개를 독립 표본처럼 pool하여 \(p\)-value를 부풀리지 않는다.

## 3.12 Mechanism ablation analysis

모든 핵심 ablation은 최종 state-dependent nonlinear scaffold에서 최소 10 seed로 다시 실행한다.

| 비교 | Mechanistic inference |
|---|---|
| CA-LRU vs No-RP | RP 학습 과정의 기여 |
| CA-LRU vs fixed final spectrum | 학습 경로와 최종 spectrum 분리 |
| CA-LRU vs all-slow | selective retention의 필요 |
| CA-LRU vs count-matched random slow | slow mode 개수보다 task alignment가 중요한가 |
| CA-LRU vs shuffled damage assignment | damage signal과 coordinate 대응의 기여 |
| Full vs input-only writer | recurrent state-dependent transport의 기여 |
| Full vs linear writer | nonlinear task transport의 기여; radial effect는 별도 측정 |
| Full vs audited radial-source ablation | audit에서 찾은 candidate의 radial-contraction 기여 |
| High-retention ablation vs low-retention control | retained mode의 task causal relevance |
| Task-basis rotation | task-coordinate invariance |

RP가 approximate CA를 “유발한다”는 인과 표현은 No-RP와 count-matched control보다 C1–C4 pass rate, \(\gamma_{50}\), long-horizon error, radial recovery가 paired seed 수준에서 개선될 때만 사용한다.

## 3.13 Approximate-CA claim gate

### 3.13.1 단계별 주장

| Level | 필수 증거 | 허용되는 표현 |
|---|---|---|
| L0 | Task threshold | recurrent memory model |
| L1 | L0 + C1 + C2 | slow task-aligned memory manifold |
| L2 | L1 + invariance + C3 + sampled positive normal gap | empirically attracting over evaluated anchors/directions |
| L3 | L2 + direct C4 | approximate continuous attractor in the sense operationalized from Ságodi et al. |
| Exact axis | all-point fixedness + exact tangent neutrality + stable normal bundle | exact continuous attractor, independent of C4 |

CA-LRU의 기본 제출 목표는 **L3 approximate CA**다. Exactness와 D-type robustness는 독립 축이다. Exact CA는 structurally fragile하여 C4를 실패할 수도 있고, L3 approximate CA는 fixed-point continuum 없이 성립할 수 있다.

### 3.13.2 사전등록용 기본 threshold

아래 값은 Ságodi 원문의 보편 정리가 아니라 본 연구의 operational proposal이다. Pilot seed에서 scale을 확인한 뒤 main 결과를 보기 전에 동결한다.

| Gate | Proposed primary threshold |
|---|---|
| Task | 원 논문 loss mask 위 NMSE \(<-20\) dB; geodesic error는 secondary |
| C1 decoding | mean normalized geodesic \(\le0.05\), 95th \(\le0.10\) |
| C1 rank | atlas의 99%에서 normalized \(\sigma_d/\sigma_1\ge10^{-3}\) |
| C1 neighborhood | trustworthiness와 continuity 각각 \(\ge0.95\) |
| C1 sheet/fiber | \(\chi_{\mathrm{fiber}}\le0.1\), 또는 validated full fiber-tangent analysis |
| Invariance | 95th percentile \(r_{\mathrm{inv}}\le0.01\) |
| C2 drift | \(H=4T\) 후 normalized mean \(\le0.05\), 95th \(\le0.10\) |
| C3 normal recovery | \(\rho=0.1R_s/\sqrt d\), \(H=500\)에서 median \(R_N\le0.5\), 95th \(R_N<1\) |
| C3 same-memory | \(H=500\)에서 mean \(E_{\mathrm{excess}}\le0.05\), 95th \(\le0.10\) |
| C3 paper-noise | coordinate std 0.1 evaluation에서 masked NMSE \(<-20\) dB |
| Tangent equivariance | primary shift에서 mean \(E_T\le0.05\), 95th \(\le0.10\) |
| Tangent non-expansion | \(H=50\)에서 \(\exp(H\ell_T^{\max})\le1.05\) |
| Sampled normal gap | \(H=50\)에서 atlas의 95%가 \(\ell_N^{\max}<0\), \(\gamma_{50}>0\) |
| C4 | primary recurrent-map perturb draw의 90% 이상이 모든 L2 gate 유지; matched \(\delta_F^{95}\)로 비교 |
| Model seeds | 10개 중 최소 8개가 L3; pass rate와 conditional metric 모두 보고 |

Threshold를 task 결과를 본 뒤 바꾸지 않는다. \(0.5\times\), \(2\times\) threshold sensitivity를 supplementary에 제시한다.

### 3.13.3 Exact CA gate

Exact CA는 empirical sample만으로 완전히 증명하기 어렵다. 먼저 Section 0.1의 carrier/full-map architecture audit를 수행한다. Carrier-only \(\Lambda h\), \(\lambda_j<1\) 구조가 blank feedback 없이 유지되면 nonzero exact continuum screen을 중단한다.

수학적 exactness에는 다음이 필요하다.

\[
\sup_{q\in\mathcal Q} r_{\mathrm{fp}}(q)\approx0,
\qquad
\sigma(A_{TT}(q))=1,
\qquad
\left\|
D F_0^H\big|_{E^s}
\right\|
\le C\rho^H,\quad \rho<1.
\]

논문용 numerical screen의 예는

\[
\max_q r_{\mathrm{fp}}\le10^{-6},
\qquad
\max_q\|\sigma(A_{TT})-1\|_\infty\le10^{-4}
\]

이지만, 이 threshold를 통과했다는 사실만으로 continuum 전체에 대한 수학적 증명이 되는 것은 아니다. Exact wording에는 architecture-level argument 또는 adaptive dense verification의 한계를 함께 명시한다.

Euclidean one-step \(\sup_q\sigma_{\max}(A_{NN})<1\)은 optional strong monotone-contraction screen이며 exact normal stability의 필요조건으로 사용하지 않는다. C4는 exactness gate에 포함하지 않고 별도 robustness 축으로 보고한다.

### 3.13.4 Claim decision matrix

| 결과 | 최종 해석 |
|---|---|
| Fixedness 실패, C1–C4 통과 | approximate CA — 목표 주장 가능 |
| C1/C2 통과, normal recovery 실패 | slow memory manifold, CA 주장 보류 |
| Radial만 수축, ambient worst-case 팽창 | direction-selective radial recovery; normally attracting CA는 보류 |
| C3 통과, C4 실패 | state-robust attractor-like manifold; Ságodi식 approximate CA는 제한 |
| Full-state만 수축, carrier error는 유지 | stream-overwrite 가능성; autonomous radial-attraction 주장 보류 |
| 모든 exact 조건과 구조적 논증 통과 | exact CA 가능 |
| Exact 조건 통과, C4 실패 | exact but D-type structurally fragile CA; L3 robustness claim과 분리 |

## 3.14 최종 산출물

### 3.14.1 Main figures

1. **Task and manifold:** ID/OOD task 성능, latent-colored atlas, decoded topology
2. **Flow and stability:** fixedness residual, tangent drift, radial/ambient recovery, tangent–normal gap
3. **Dimension scaling:** \(d=1,2,4,8\)의 task error, rank, drift, recovery, retained dimension
4. **S/D robustness:** state noise/kick와 parameter perturbation response curve
5. **Seed consistency:** aligned atlas, tangent principal angle, recovery/gap distribution

### 3.14.2 Main tables

| Table | 내용 |
|---|---|
| 1 | Parameter-matched ID/OOD task performance와 success rate |
| 2 | 모델별 C1–C4 및 L0–L3 pass matrix |
| 3 | CA-LRU mechanism ablation |
| 4 | \(d=1,2,4,8\) scaling과 seed robustness |

### 3.14.3 Artifact layout

~~~text
analysis_protocol.yaml
manifest.json
eval_banks/
  task=<task>/dim=<d>/atlas.npz
runs/
  model=<model>/seed=<seed>/
    config.yaml
    task_metrics.parquet
    atlas_states.zarr
    tangent_basis.zarr
    flow_and_residuals.parquet
    jacobian_summary.parquet
    perturbation_s_type.parquet
    perturbation_d_type.parquet
    fixed_points.parquet
    asymptotic_rollouts.parquet
    claim_gate.json
summary/
  all_seed_metrics.parquet
  success_rates.parquet
  statistical_tests.parquet
  claim_gate_matrix.csv
  figures/
~~~

Manifest에는 code commit, checkpoint hash, complete state definition, parameter count, 모든 seed, evaluation-bank hash, threshold version을 기록한다. Claim gate 파일에는 C1–C4의 metric, threshold, pass/fail, source artifact path를 저장한다.

## 3.15 구현 순서와 중단 기준

### Phase 0 — State audit

- primary Markov state pack/unpack
- actual \(F_0\) unit test
- noise/reset/stream audit
- autograd–finite-difference Jacobian 검산

통과 전에는 CA 결과를 생성하지 않는다.

### Phase 1 — Ring proof-of-concept

- Ságodi Track A reconstruction
- task-conditioned ring atlas
- fixedness/flow/Jacobian
- explicit radial 및 ambient finite kick
- CA-LRU vs strongest baseline vs No-RP

Phase 1에서 radial contraction의 귀속과 clean-paired recovery가 확인되어야 고차원 full sweep으로 간다.

### Phase 2 — Torus

- \(d=2\) factorized와 mixed
- local rank와 topology
- factor별 curvature normal
- S/D robustness

### Phase 3 — Dimension scaling

- \(d=4,8\) Sobol atlas
- Jacobian-vector-product 기반 finite-horizon gap
- memory-dimension 대비 retained-dimension scaling

### Phase 4 — Seed와 mechanism

- 10–20 model seeds
- task-map seed robustness
- final-scaffold ablations
- statistical tests와 claim matrix 동결

### Stop/rename rule

Phase 1–2에서 C1 또는 C2가 반복적으로 실패하면 “approximate CA” 대신 “slow recurrent memory”로 논문 framing을 낮춘다. Radial recovery만 확인되고 ambient normal attraction이 없으면 “full-dynamics radial contraction”을 mechanism claim으로 남기되 “normally attracting manifold”는 주장하지 않는다. Gain이 radius/state에 따라 달라질 때만 nonlinear이라는 수식어를 추가한다.

## 3.16 필수 QA checklist

- [ ] Zero input이 normalization/bias/go channel까지 포함해 실제로 zero인가?
- [ ] Primary state가 미래 transition을 결정하는 모든 variable을 포함하는가?
- [ ] Stream overwrite와 carrier contraction을 분리했는가?
- [ ] Training/evaluation noise가 분석 때 의도대로 on/off인가?
- [ ] 같은 evaluation/perturbation bank를 모든 모델에 사용했는가?
- [ ] Tangent finite difference가 step size에 안정적인가?
- [ ] Tangent와 normal vector의 orthogonality가 검산되었는가?
- [ ] Jacobian-vector product가 finite difference와 일치하는가?
- [ ] Clean drift를 perturbation recovery에서 뺐는가?
- [ ] Manifold projection discretization error를 측정했는가?
- [ ] 평균뿐 아니라 95/99 percentile과 max를 보고했는가?
- [ ] 실패 seed를 success-rate 분모에 포함했는가?
- [ ] C4를 실제 parameter perturbation으로 검사했는가?
- [ ] Exact CA와 approximate CA wording gate를 분리했는가?

---

# 권장 논문 문장

## Main claim

> CA-LRU exhibits approximate continuous-attractor dynamics characterized by a task-aligned slow memory manifold, bounded tangential drift, and autonomous recovery from radial and sampled ambient off-manifold perturbations.

“Autonomous”는 실제 blank \(F_0\) audit를 통과했을 때만 남긴다. “Nonlinear”은 radius/state-dependent gain까지 확인했을 때만 추가한다. 그렇지 않으면 다음처럼 쓴다.

> CA-LRU learns approximate continuous-attractor dynamics with nonlinear off-manifold recovery under the evaluated blank-input protocol.

## Exactness disclaimer

> We do not assume that every point of the learned manifold is an exact fixed point. Instead, we quantify full-state fixedness residuals and evaluate the operational C1–C4 conditions for an approximate continuous attractor over behaviorally relevant and out-of-distribution timescales.

## Mechanism claim

> CA-LRU explicitly induces selective retention and a fast–slow timescale bias, while the task-specific manifold geometry and its tangent–normal organization emerge from data.

## Seed claim

> Across random initializations, the coordinate realization changes, whereas the aligned memory geometry, intrinsic dimension, tangential drift, and radial/ambient recovery remain functionally consistent.

---

# 참고 자료

1. Ságodi, L., Martín-Sánchez, G., Geiger, F., Duong, L., &amp; Orlandi, J. G. (2024). [Back to the Continuous Attractor](https://proceedings.neurips.cc/paper_files/paper/2024/file/7b78a2a7360d5a9ad750834dc5a33bfb-Paper-Conference.pdf). NeurIPS 2024.
2. Ságodi et al., [official analysis and task code](https://github.com/catniplab/back_to_the_continuous_attractor).
3. CA-LRU, [repository](https://github.com/WOOYULJUNG/calru) and [Korean methods note](https://github.com/WOOYULJUNG/calru/blob/main/paper/notes/methods_equations_data_ko.md).
