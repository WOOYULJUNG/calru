# Ságodi-exact baseline vs parameter-matched CA-LRU — Primary v4 freeze

상태: 구현·실행 전 과학 사양 고정

날짜: 2026-07-14
대체 대상: `SAGODI_PRIMARY_V3_FREEZE_ko.md`

## 1. v3를 폐기하는 이유

v3는 width-96 CA-LRU의 56,834 parameters에 맞추기 위해 RNN, GRU,
LSTM의 width와 일부 scaffold를 바꿨다. 이 방향은 Ságodi baseline을 보존한
비교가 아니다. v3 partial artifacts는 debugging provenance로만 보존하며 어떤
primary/confirmatory 결과에도 합치지 않는다.

v4의 원칙은 반대다.

1. Ságodi baseline의 recurrence, nominal width, initialization interface와
   linear readout을 고정한다.
2. CA-LRU/No-RP/complex LRU의 width를 Ságodi RNN/LSTM parameter budget에
   맞춘다.
3. task, online batches, optimizer budget, recurrent-state noise와 evaluation
   banks를 공유한다.

## 2. 구현 기준

공개 repository에는 논문과 맞지 않는 설정과 실행상 결함이 있다. 예를 들어
GRU/LSTM의 `W_otr`가 초기화되지 않고 LSTM의 cell initializer가 hidden
initializer로 덮이며, 공개 YAML도 논문의 5,000-update 조건이 아니다. 따라서
v4는 **bit-exact upstream**이 아니라 다음 우선순위를 쓰는
**paper-matched Ságodi reproduction**이다.

1. 논문 Methods S7.1, S7.3--S7.5의 수식과 명시적 설정
2. 표준 PyTorch GRU/LSTM recurrence
3. 공개 코드는 tensor layout과 one-step recurrence parity 확인에만 사용

모든 deviation과 repair는 manifest에 기록한다.

## 3. Primary parameter budget

Primary nominal size는 `N=128`이다. Ságodi 규칙대로 vanilla RNN과 GRU는
128 units, LSTM은 64 units를 사용한다. Input dimension 1, output dimension 2,
bias-free learned `W_otr`와 linear readout을 포함한 count는 다음과 같다.

| model | recurrent width | trainable parameters | 역할 |
|---|---:|---:|---|
| Ságodi tanh RNN | 128 | 17,154 | exact paper-form baseline |
| Ságodi GRU | 128 | 50,818 | exact paper-form gated baseline; paper itself is not parameter-matched to RNN |
| Ságodi LSTM | 64 (`N/2`) | 17,538 | exact paper-form baseline |
| complex LRU | 52 | 17,058 | RNN/LSTM-budget-matched structured baseline |
| CA-LRU No-RP | 52 | 17,058 | causal mechanism control |
| CA-LRU | 52 | 17,058 | proposed model |

Count formulas are

\[
P_{\rm RNN}(N)=N^2+6N+2,
\]

\[
P_{\rm GRU}(N)=3N^2+13N+2,
\]

\[
P_{\rm LSTM}(H)=4H^2+18H+2,\qquad H=N/2,
\]

and, for the frozen CA-LRU/full-LRU scaffold,

\[
P_{\rm CA}(w)=6w^2+16w+2.
\]

Thus `w=52` differs from the RNN by -0.56% and from the LSTM by -2.74%.
The much larger GRU row is retained because changing its paper width would change the
baseline. A GRU-budget-matched `CA-LRU w=91` (51,144 parameters, +0.64%) is a
separately labelled supplementary control and is not pooled with primary `w=52`.

Scaling rows are frozen as follows.

| nominal N | RNN params | LSTM params (`H=N/2`) | CA width | CA params |
|---:|---:|---:|---:|---:|
| 64 | 4,482 | 4,674 | 26 | 4,474 |
| 128 | 17,154 | 17,538 | 52 | 17,058 |
| 256 | 67,074 | 67,842 | 105 | 67,832 |

## 4. Model contracts

### Ságodi tanh RNN

\[
h_{t+1}=0.9h_t+0.1\tanh(W_{\rm rec}h_t+W_{\rm in}u_t+b),
\qquad y_t=W_{\rm out}h_t+b_{\rm out}.
\]

- `W_rec ~ Normal(0, (1.5/sqrt(N))^2)`;
- other weight matrices use Xavier normal;
- biases are zero;
- learned hidden initialization from the true pre-update
  `(cos(q0), sin(q0))`;
- no MLP readout, normalization, dropout, scheduler or early stopping.

### Ságodi GRU

- one standard PyTorch GRU recurrence with 128 hidden units;
- direct biased linear readout;
- learned `tanh(W_otr y0)` initializer;
- no dropout or target/output noise.

### Ságodi LSTM

- one standard PyTorch LSTM recurrence with 64 units;
- the Markov state is full concatenated `[h,c]`;
- independent learned maps from `y0` into `h0` and `c0`, followed by tanh;
- direct biased linear readout from `h`;
- recurrent-state noise is applied to both `h` and `c`.

### CA-LRU, No-RP and complex LRU

The existing full scaffold and one-step equations are unchanged except for width 52.
CA-LRU and No-RP have identical initialization, data stream, optimizer and model
parameters. They must be bit-identical until the first scheduled RP update. No-RP
is never tuned independently.

## 5. Shared task and training contract

- angular-velocity integration, 256 raw-velocity tokens;
- `dt=0.1`, GP length scale 1, marginal standard deviation 1;
- true pre-update `q0` is passed only to the learned initial-state encoder;
- targets are `q[t+1]=q[t]+dt*u[t]`, embedded as cosine/sine;
- all 256 output steps are included in MSE;
- online fresh batches, paired across models by update and seed;
- Adam, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay 0;
- batch size 64, exactly 5,000 optimizer updates;
- no clipping, scheduler, early stopping, target noise or dropout;
- coordinate recurrent-state noise standard deviation 0.1 during training;
- state noise disabled for validation and dynamical analysis;
- final-update checkpoint is retained, while best validation checkpoint is logged only
  as an optimization diagnostic and never silently substituted.

The paper-selected Ságodi baseline LR is `1e-2`. A reproduced 4-LR selector is an
audit; it may not replace the paper row with a chance-level winner.

## 6. CA-LRU tuning freeze

Changing width, state-noise scale, task interface and training budget invalidates the
legacy CA-LRU hyperparameters. Tuning uses model seeds 100--104 only. Main seeds 0--9
remain unseen.

### Stage T0: learning-rate selection

- LR grid: `{1e-2, 1e-3, 1e-4, 1e-5}`;
- RP disabled because the 100-update paper selector precedes every RP schedule;
- record the paper 100-update statistic as an audit only;
- keep the Ságodi baselines at the paper LR `1e-2` and continue those learning sentinels
  to 5,000 updates;
- continue **every** LRU and CA-LRU LR candidate to 5,000 updates, then select only among
  cells for which all five tuning seeds reach the preregistered task-success threshold.
  This prevents selection among MSE approximately 0.5 curves.

CA-LRU and No-RP receive the same selected LR.

### Stage T1: RP selection

- fixed warm-up: 1,500 updates;
- fixed interval: 50 updates, giving 70 decisions through update 5,000;
- `eta_lambda` grid: `{300, 1000, 3000}`;
- `epsilon_RP` grid: `{1e-5, 3e-5, 1e-4}`;
- five tuning seeds for every cell;
- eligibility: finite run and held-out ID MSE below `1e-2`;
- selection among eligible cells: minimum held-out tuning-bank blank-memory error;
- geometry/manifold test-bank metrics are forbidden for tuning.

The 1,024-trajectory tuning bank and 1,024-trajectory main test bank use distinct
registered seeds and stream keys. The main test bank is never read during T0 or T1.

The RP rule remains the paper-model rule already stated in the manuscript: batch-mean
baseline-versus-coordinate-ablation delayed error difference, thresholded in retention
logit space and clipped to `[-18,18]`.

## 7. Fail-closed launch gates

Before fan-out to ten main seeds:

1. one-step RNN equation parity passes;
2. GRUCell versus `nn.GRU` and LSTMCell versus `nn.LSTM` fixed-weight parity passes;
3. initializer uses true `q0` and not the first post-update target;
4. state-noise location and empirical standard deviation pass;
5. parameter counts match this document exactly;
6. one non-main 5,000-update sentinel per baseline reaches MSE below `1e-2`;
7. one CA-LRU tuning configuration reaches the same task-success regime;
8. CA-LRU and No-RP are identical through update 1,500 under paired randomness.

Failure blocks the main campaign. It does not trigger an unrecorded hyperparameter change.
