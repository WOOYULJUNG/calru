# Ságodi-primary v3.1 실험 freeze

- revision: `3.1`
- revised at: `2026-07-14`
- revision reason: carrier ambient-normal finite-perturbation recovery를 primary
  CA evidence에 추가하고, 분석 시 state-noise-disabled/deterministic 조건을
  명확히 한다.
- history: `v3.0`은 ambient/radial kick을 supplementary로만 분류했다. 이미 시작된
  v3.0 실행은 이 변경 뒤 primary v3.1 결과와 혼합하지 않고 superseded artifact로
  보존한다.

## 1. 목적과 경계

이 freeze의 목적은 CA-LRU가 연속 어트랙터와 일치하는 내부 동역학을 형성하는지를
Ságodi et al. (NeurIPS 2024)의 분석으로 평가하는 것이다. Ságodi 분석은 CA-LRU의
제안 방법이 아니라 평가 도구다.

다음은 primary 결과에 포함한다.

1. slow-manifold reconstruction
2. flow reversal 기반 fixed-point topology
3. full local Jacobian eigenspectrum과 top-two timescale separation
4. output-projected vector field의 uniform norm
5. finite-time angular memory error
6. stable basin, asymptotic error, Shannon entropy capacity
7. minimum causal carrier state의 project-defined finite-perturbation
   ambient-normal recovery

다음은 primary 결과와 수치 pass/fail 결정에서 제외한다.

- settling horizon, 8-path atlas, path-variance plateau
- legacy radial/ambient kick의 `D_clean`, `Q_recovery`와 수치 C3 gate
- projected-JVP cocycle, worst-normal contraction
- numerical C1–C4 gate, L0–L3 gate, 8/10 seed gate
- test-time Gaussian state-noise NMSE

위 항목은 삭제하지 않고 `CA-LRU-specific supplementary` 분석으로만 보존한다.
단, §8.5에 새로 고정한 carrier ambient-normal recovery는 legacy kick 분석과
별개의 primary extension이다. 이 extension은 Ságodi 원문의 bit-exact 재현이
아니라 normal attraction을 직접 검사하기 위한 project-defined finite-perturbation
evidence다. 어떤 수치 pass/fail threshold도 두지 않는다.

## 2. 정보원과 우선순위

- 규범 기준: Ságodi et al. 논문 §4.2, §5, §6, S7.6.3
- 보조 기준: 공식 저장소 commit
  `cbd7404e9baca4b2dc291560cfc6576bb7b1f078`
- 논문과 코드가 다르면 논문을 따른다.
- 공식 코드의 미초기화 행렬, swapped Jacobian, hard-coded input dimension 등을
  bit-exact하게 복제하지 않는다.

## 3. Task freeze

- task: one-dimensional angular-velocity integration on `S1`
- sequence length: `T = 256`
- input: raw angular velocity, one scalar per step
- target: post-update `(cos(theta_t), sin(theta_t))`
- initial angle: uniform on `[-pi, pi)`
- velocity: normalized grid `linspace(-1, 1, T)`에서 샘플한 GP
- GP length scale `1`, marginal std `1`, Cholesky jitter `1e-6`
- time step `dt = 0.1`
- loss: 256 step 전체의 output MSE
- online training batch: update마다 새로 생성

## 4. 모델 비교 freeze

Primary table은 parameter-near-matched 단일 track을 사용한다.

| ID | width | total parameters | 역할 |
|---|---:|---:|---|
| `rnn_param206` | 206 | 56,844 | project RNN baseline |
| `gru_sagodi_param135` | 135 | 56,432 | official-style GRU baseline |
| `lstm_param109` | 109 | 56,438 | project LSTM baseline |
| `lru_param96` | 96 | 56,834 | existing `LRU full` baseline |
| `no_rp` | 96 | 56,834 | CA-LRU scaffold, RP disabled |
| `ca_lru` | 96 | 56,834 | proposed model, RP enabled in main training |

RNN/LSTM/LRU는 Ságodi 공식 모델을 bit-exact하게 복제한 것으로 표기하지
않는다. 기존 CA-LRU 실험의 recurrent core·readout scaffold에서 parameter count를
맞춘 project baseline이다. GRU만 공식 코드의 one-layer GRU·direct linear readout
형식을 따른다.

초기 memory encoder는 전 모델에서 bias-free `W_otr`을 checkpoint에 포함하고,
`Normal(0, 1/sqrt(primary_state_dimension))`로 명시적 초기화한다. GRU와
LSTM은 `tanh(W_otr y0)`를 사용하되 LSTM은 전체 `[h,c]` mapping에
tanh를 적용한다. RNN과 LRU, CA-LRU/No-RP는 identity mapping을 사용한다.

상태 잡음 `std = 0.1`은 모든 모델의 minimum full Markov recurrent state에
좌표별로 적용한다. 따라서 LSTM은 `[h,c]` 모두, complex LRU는
real/imag carrier 모두가 대상이다.

## 5. Learning-rate selection freeze

- model: 위 6개
- selection model seeds: `1100, 1101, 1102, 1103, 1104`
- grid: `1e-2, 1e-3, 1e-4, 1e-5`
- runs: `6 * 5 * 4 = 120`
- updates: 100
- optimizer: Adam, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0`
- batch size: 64
- state-noise coordinate std: 0.1
- gradient clipping: none
- CA-LRU Retention Plasticity: selector 동안 0회
- selector metric: 각 run의 online masked MSE at update 100
- aggregation: 모델·LR별 5 seeds arithmetic mean
- winner: 가장 작은 mean loss; 정확한 tie이면 작은 numeric LR
- validation/analysis metric은 LR 선택에 영향을 주지 않는다.
- 120개 전체의 verified completion receipt가 없으면 winner를 생성하지
  않는다.

기존 v2 80-run selector는 새 primary result에 pooling하지 않고 provenance/audit
자료로만 보존한다.

## 6. Main training freeze

- main model seeds: `0,1,2,3,4,5,6,7,8,9`
- runs: `6 * 10 = 60`
- model-specific LR: verified v3 selector receipt의 deterministic winner
- updates: 5,000
- optimizer: Adam, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0`
- batch size: 64
- state-noise coordinate std: 0.1
- gradient clipping: none
- checkpoint: update 5,000
- validation eligibility: NMSE `< -20 dB`
- NMSE: `10 log10(MSE / target_power)`

모든 10 seeds의 training 완료와 validation 결과를 보고한다. `-20 dB`를
통과한 seed만 structural Ságodi summary에 포함하되, 통과하지 못한 seed를
숨기거나 재학습하여 교체하지 않는다.

CA-LRU의 main RP schedule은 기존 실험 레시피에서 전이한 method-specific
freeze이며 Ságodi 논문의 hyperparameter가 아니다.

- warmup: 1,500 updates
- interval: 50 updates
- calls: 70 (`1550, 1600, ..., 5000`)
- probe batch: 96
- probe task horizon: 256
- blank ablation horizon `H_RP`: 500
- eta: 3,000
- damage epsilon: `3e-5`
- No-RP와 모든 baseline: RP 0회

`warmup=1,500`은 5,000-update optimization budget의 30%를 먼저 표준 gradient
학습에 사용하는 경계다. 이후 `1,550..5,000`에서 50 update마다 수행하는 70회의
RP decision과 혼동하지 않는다. `H_RP=500`은 RP damage probe의 autonomous blank
horizon이며 main task horizon `T=256`이나 총 optimization budget 5,000과 다른
수량이다.

각 RP call은 state noise를 끈 probe에서 task의 마지막 post-update target `z`와
probe 종료 state `s`를 얻는다. `H_RP=500` blank steps 뒤 batch-mean squared L2
memory error를

`E_mem(s;z) = B^{-1} sum_b ||D(F_0^H_RP(s_b)) - z_b||_2^2`

로 계산한다. coordinate `j`를 0으로 만드는 `A_j`에 대해
`Delta_j = E_mem(A_j s;z) - E_mem(s;z)`이고,

`theta_j <- clip_[-18,18](theta_j + eta * (Delta_j - epsilon_RP))`,
`lambda_j = sqrt(sigmoid(theta_j))`

를 사용한다. `Delta_j`는 batch mean이며 threshold는 `epsilon_RP=3e-5`다. 이
계산과 `theta` update는 `no_grad`이고, `theta`는 Adam parameter set 밖에 있어
gradient optimizer가 수정하지 않는다.

## 7. Primary manifold reconstruction

논문의 task/autonomous staging에 모호성이 있으므로 이 프로젝트는 다음을
명시적으로 고정한다. 이는 reasonable resolution이지, 논문의 문장을
bit-exact하게 재현한다는 뜻이 아니다.

1. 정확히 1,024개의 held-out GP task trajectory를 새로 생성한다. 여기서
   `noise-free`라는 표현을 사용하지 않는다. trajectory sampling은 stochastic이며,
   additive recurrent-state training noise만 평가에서 비활성화한다.
2. 각 trajectory를 `T=256` 동안 task input으로 진화시켜 endpoint를 얻는다.
3. endpoint에서 외부 입력을 정확히 0으로 하고 `16T=4096` step을
   autonomous rollout한다.
4. 각 trajectory `i`의 primary state speed를
   `s_i(t)=||h_i(t+1)-h_i(t)||_2`로 계산한다.
5. `s_i(t) <= 1e-3 * max_tau s_i(tau)`를 만족하는 상태를 slow
   candidate로 삼는다. max speed가 0인 trajectory는 동일 상태 1개만
   대표로 남긴다.
6. 1,024개 uniform target output angle 각각에 대해 decoded output ring에서
   가장 가까운 candidate를 선택한다.
7. duplicate/near-duplicate angular knots를 deterministic하게 병합한 뒤 hidden
   coordinates에 periodic cubic spline `m(theta)`를 적합한다.

4,096개 bank에서 1,024개를 사전 층화 선택하지 않는다. coverage, seam,
duplicate-knot 검사는 numerical QA로 기록하지만 CA pass/fail gate로 사용하지
않는다.

## 8. Primary dynamics analysis

### 8.1 Discrete vector field

모든 모델을 common discrete-time convention으로 비교한다.

`v_0(s) = F_0(s) - s`

보고용 Jacobian은 둘 다 저장한다.

- map Jacobian `J_F`
- vector-field Jacobian `J_v = J_F - I`

본문의 zero-eigenvalue convention과 일치하는 primary spectrum/gap은 `J_v`를
사용한다. discrete stability audit에는 `J_F`의 spectral radius도 함께
저장한다. complex eigenvalue를 실수로 버리지 않는다.

### 8.2 Projected flow과 uniform norm

모델의 실제 decoder `D`를 사용하여 common finite-step projected flow를

`q(theta) = D(F_0(m(theta))) - D(m(theta))`

로 정의한다. 이는 nonlinear decoder를 포함하는 CA-LRU/LRU에 논문의
linear `W_out v` 정의를 확장한 명시적 adaptation이다.

`||f||_infinity = max_theta ||q(theta)||_2`

은 2D vector의 Euclidean norm이며 scalar angular speed의 최댓값이 아니다.

### 8.3 Fixed points과 topology

1,024 spline samples에서 projected flow의 signed angular component를 사용한다.

- positive to negative: stable tangential fixed point
- negative to positive: tangentially repelling point; normal directions을 따로 검사해
  saddle로 해석
- reversal이 없고 동일 방향 flow: limit-cycle topology; 분석 실패가 아님

zero tolerance, linear root interpolation, cyclic seam merge tolerance은 numerical policy로
artifact에 저장하며 주장 gate로 사용하지 않는다.

### 8.4 Full eigenspectrum

각 spline point의 minimum full Markov state에서 dense local Jacobian을 계산한다.
LSTM은 `[h,c]`, LRU는 real/imag carrier 전체가 대상이다. `J_v`의 eigenvalue를
실수부 내림차순으로 정렬하고 전체 complex spectrum과 다음을 저장한다.

- `lambda1_real`, `lambda2_real`
- `gap = lambda1_real - lambda2_real`
- `tau = -1/lambda_real` when the real part is negative

spline tangent과 eigenvector cosine, projected JVP, scalar pass threshold는 primary에 포함하지
않는다.

### 8.5 Carrier ambient-normal finite-perturbation recovery (v3.1 primary extension)

이 분석은 Ságodi 원문에 명시된 실험을 그대로 재현한 것이 아니다. Ságodi의
normally attractive slow-manifold 해석에 따라, reconstructed manifold에 대한
finite perturbation recovery를 직접 확인하는 project-defined primary extension이다.

분석 state는 output이나 readout-only diagnostic stream이 아니라 다음 recurrent
update를 결정하는 **minimum causal primary carrier state**다. CA-LRU/LRU는 complex
carrier의 real/imag 전체, LSTM은 `[h,c]` 전체를 사용한다. 다른 모델도 동일한
full Markov-state adapter가 반환하는 state를 사용한다. Decoder output만으로
recovery를 정의하지 않는다.

Spline anchor `m(theta)`에서 단위 tangent를

`t(theta) = m'(theta) / ||m'(theta)||_2`

로 표기한다. 실제 수치 구현은 1,024개 sampled spline state에서 periodic central
finite difference
`t_i=(m_{i+1}-m_{i-1})/||m_{i+1}-m_{i-1}||_2`를 사용하며 analytic cubic
derivative를 사용하지 않는다. seed가 고정된 Gaussian direction `r`에서 tangent
성분을 제거하여

`n = (r - t(t^T r)) / ||r - t(t^T r)||_2`

를 만들고 `m(theta) + rho n`에서 zero-input blank map을 rollout한다. 별도
`in_plane_radial` family는 모든 spline carrier state를 centroid 기준으로
center한 행렬의 SVD에서 얻은 global top-two PC plane을 사용한다. anchor tangent를
그 2-PC plane에 투영한 뒤 plane 안에서 90도 회전한 방향을 만들고, signed `+/-`
두 방향을 사용한다. PC plane의 rank가 부족하거나 projected tangent/rotation을
정규화할 수 없는 경우에는 값을 꾸며내지 않고 structural `not estimable`로
기록한다. `ambient_normal`은 sampled finite-amplitude normal-recovery evidence이며
`in_plane_radial`은 geometry를 해석하기 위한 descriptive family다. 여기서 sampled
tangent complement는 manifold 전체의 invariant stable normal bundle과 동일하지
않으며, 이 finite bank는 전역 normal attraction의 수학적 증명이 아니다.

Manifold scale `R_M`은 1,024개 spline carrier state와 그 carrier centroid 사이
Euclidean distance의 RMS다. perturbation radius는 `rho/R_M`으로 고정한다.

- deterministic perturbation seed: `314159`
- anchors: spline index `floor(a * 1024 / 32)`, `a=0,...,31`의 32개
- ambient-normal directions per anchor: `4`
- in-plane radial directions per anchor: signed `+/-`의 `2`
- radii `rho/R_M`: `[0.01, 0.05, 0.1]`
- horizons: `[0, 1, 4, 16, 64, 256, 1024, 4096]`
- registered base perturbations: ambient `32*4*3=384`, radial `32*2*3=192`
- registered horizon-level values: `(384+192)*8=4,608`; canonical NPZ는 이를
  4,608개의 long-form row로 복제하지 않고 base-trial metadata `[576]`, horizon
  vector `[8]`, metric/index matrix `[576,8]`로 저장한다.

각 horizon `k`에서 nearest-spline Euclidean distance

`d_perp(k) = min_phi ||F_0^k(m(theta)+rho n) - m(phi)||_2`

와 ratio `d_perp(k)/d_perp(0)`를 저장한다. 동시에 원 anchor memory에 대한
circular decoded error

`e_mem(k) = d_S1(angle(D(F_0^k(m(theta)+rho n))), theta)`

를 저장한다. artifact에는 family, anchor/direction index와 실제 direction,
anchor angle, normalized/absolute radius, horizon, initial/final nearest-manifold
distance, distance ratio, decoded angle, same-memory error, nearest-manifold index를
base-trial × horizon matrix 단위로 남긴다.

각 perturbed rollout에는 동일 anchor에서 시작하는 matched clean rollout을
동일 horizon으로 함께 계산한다. `clean_manifold_distance`,
`clean_same_memory_error_radians`, perturbation-minus-clean
`excess_same_memory_error_radians`, `manifold_distance_minus_clean`, 그리고
`distance_to_matched_clean_state`와 그 initial-distance ratio도 trial 단위와
동일한 denominator로 보고한다. matched-clean 결과는 absolute normal distance
ratio나 absolute same-memory error를 대체하지 않고 finite blank drift를 분리해
해석하는 보조량이다.

모든 registered trial/horizon을 포함해야 하며 missing 또는 NaN/Inf는 성공으로
집계하지 않는다. 각 family/radius/horizon별로 count, mean, population standard
deviation, median, q05, q95, min, max, finite count, missing/nonfinite count를
보고한다. `d_perp(k)/d_perp(0) < 1` 같은 binary gate나 사후 threshold는 적용하지
않고 결과를 descriptive numeric evidence로만 해석한다.

Direction-construction QA는 scientific recovery gate와 분리한다. 저장된 direction은
`abs(||n||_2-1) <= 1e-4` 및 `abs(t^T n) <= 1e-4`를 만족해야 하며, artifact의
저장 QA 값은 direction/tangent 배열에서 재계산한 값과 일치해야 한다. 이 조건은
예상 방향의 recovery를 요구하는 threshold가 아니라 perturbation 자체가 등록한
tangent-complement construction인지 확인하는 입력 유효성 검사다.

Smoke는 실행 경로 검증용으로 anchor 수를 `min(32, spline_count)`로 줄이고,
frozen horizon 중 blank horizon 이하인 값 및 blank horizon 자체만 사용한다.
Smoke 결과는 full primary 결과로 해석하지 않는다.

## 9. Finite-time과 asymptotic memory

1,024 spline states에서 blank rollout하고 circular angular error를 계산한다.

- instantaneous mean error
- instantaneous maximum error
- cumulative time-averaged mean error
- horizons `T, 3T, 5T, 7T, 9T`
- full `0..16T` curve

Stable fixed points가 있으면 1,024 starting angles을 local flow/basin boundary에
따라 stable fixed points에 할당하고 다음을 보고한다.

- stable/saddle count and angles
- ordered spacing
- basin proportions `p_j`
- Shannon entropy `H = -sum_j p_j log(p_j)` with natural logarithm
- asymptotic mean/max circular error

Limit-cycle topology에서 discrete fixed-point basin entropy는 `N/A`로 보고한다.

CA-LRU의 finite blank map은 retention가 정확히 1이 아니면 수학적으로
원점으로 수축할 수 있다. 따라서 CA-LRU가 asymptotic capacity에서 우세하다는
결론을 freeze에 포함하지 않는다. 이 결과는 descriptive outcome으로 보고하고,
실용 주장은 finite-time memory와 OOD 성능을 중심으로 검증한다.

## 10. 후속 공학 실험

Ságodi-primary 분석 후에 다음을 별도의 engineering-benefit track에서
실행한다.

- training horizon보다 긴 sequence/OOD blank horizon
- input/state perturbation robustness
- unseen trajectory·initial-condition generalization
- seed variability
- seed-level dynamics–utility association

상관관계는 causal evidence로 표현하지 않고 descriptive association으로
보고한다. 예정 연결은 timescale gap–long-horizon error, uniform flow
norm–memory drift, capacity–OOD retention이다.

## 11. 실행·보고 원칙

- held-out GP task input은 stochastic process의 표본이며 `noise-free task`라고
  부르지 않는다. 표본 trajectory가 정해진 뒤의 dynamics analysis에서는 training
  additive recurrent-state noise와 dropout을 끈다.
- 정확한 분석 용어는 `state-noise-disabled evaluation`, `deterministic dynamical
  analysis`, `autonomous zero-input rollout with training state noise disabled`다.
- isotropic Gaussian state perturbation은 engineering/supplementary robustness로만
  유지하며, tangent와 normal 성분을 섞으므로 primary normal-attraction evidence로
  사용하지 않는다.

- 학습 실패와 분석 부적합을 과학적 성공으로 바꾸지 않는다.
- 예상한 방향의 결과를 얻기 위해 seed, LR, horizon, tolerance를 변경하지
  않는다.
- numerical QA failure는 artifact에 남기고 해당 분석을 `not estimable`로 표기한다.
- 모든 full run은 code commit, protocol hash, parent selector receipt, environment,
  seed, architecture metadata, checkpoint hash를 receipt에 결속한다.
- primary table은 mean 외에 individual seed value와 eligibility rate를 공개한다.
