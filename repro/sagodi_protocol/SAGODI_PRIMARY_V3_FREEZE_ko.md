# Ságodi-primary v3 실험 freeze

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

다음은 primary 결과와 수치 pass/fail 결정에서 제외한다.

- settling horizon, 8-path atlas, path-variance plateau
- radial/ambient kick, same-memory recovery, `D_clean`, `Q_recovery`
- projected-JVP cocycle, worst-normal contraction
- numerical C1–C4 gate, L0–L3 gate, 8/10 seed gate
- test-time Gaussian state-noise NMSE

위 항목은 삭제하지 않고 `CA-LRU-specific supplementary` 분석으로만 보존한다.

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
- probe batch: 256
- probe task horizon: 256
- blank ablation horizon: 256
- eta: 3,000
- damage epsilon: `3e-5`
- No-RP와 모든 baseline: RP 0회

## 7. Primary manifold reconstruction

논문의 task/autonomous staging에 모호성이 있으므로 이 프로젝트는 다음을
명시적으로 고정한다. 이는 reasonable resolution이지, 논문의 문장을
bit-exact하게 재현한다는 뜻이 아니다.

1. 정확히 1,024개의 noise-free task trajectory를 새로 생성한다.
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

- 학습 실패와 분석 부적합을 과학적 성공으로 바꾸지 않는다.
- 예상한 방향의 결과를 얻기 위해 seed, LR, horizon, tolerance를 변경하지
  않는다.
- numerical QA failure는 artifact에 남기고 해당 분석을 `not estimable`로 표기한다.
- 모든 full run은 code commit, protocol hash, parent selector receipt, environment,
  seed, architecture metadata, checkpoint hash를 receipt에 결속한다.
- primary table은 mean 외에 individual seed value와 eligibility rate를 공개한다.
