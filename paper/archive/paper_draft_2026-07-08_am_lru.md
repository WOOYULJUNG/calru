# Learning Attractor Manifolds in Recurrent Networks

*Working draft, updated 2026-07-06.*

## Abstract

Long-horizon sequence memory does not by itself imply continuous-attractor
dynamics. A recurrent system can preserve enough information to reduce a
prediction loss while still drifting under zero input, failing to recover from
off-manifold perturbations, or collapsing nearby continuous memories into a
single state. We evaluate recurrent memory as a long-horizon dynamical object
using three diagnostics: zero-input persistence, normal recovery, and tangent
consistency.

We study RNN, GRU, LSTM, SSM, and LRU baselines together with AM-LRU, an
Attractor-Manifold Linear Recurrent Unit. AM-LRU is an LRU-style recurrent
model whose coordinate-wise retention values are shaped by coordinate ablation:
if ablating a coordinate increases long-horizon memory error, that coordinate
becomes more persistent; if its ablation effect is below threshold, it is drained.
Across line/plane integration and ring-hold tasks, AM-LRU improves the full
continuous-attractor diagnostic bundle. The resulting behavior is not explained
by making all recurrent coordinates slow: the strongest evidence is compact
persistent support concentrated on coordinates with large ablation effects,
plus recovery from normal perturbations without collapse along tangent
directions.

## 1. Introduction

Modern recurrent architectures can remember, but memory is not the same as
continuous-attractor geometry. For analog memory, the target is not a single
stable fixed point. It is a continuous family of valid states: nearby points on
the memory manifold should remain distinct, while perturbations away from that
manifold should contract back.

This paper sits between four related lines of work. Classical continuous
attractor models, including learned line-attractor RNNs and ring-attractor
models of orientation memory, describe ideal line, plane, or ring manifolds for
analog memory. More recent work such as *Back to the Continuous Attractor*
emphasizes that the exact mathematical object is fragile and that useful memory
can be understood through sufficiently long-lived approximate attractor
behavior. Our goal is not to rederive a continuous attractor construction, but
to ask whether trained recurrent sequence models actually exhibit the
behavioral properties expected of such a memory.

Trained RNN dynamics work, from fixed-point and slow-point analysis to
reverse-engineered line-attractor dynamics, has shown how to interpret hidden
computation geometrically. We use the same dynamical viewpoint, but make the
comparison architectural and diagnostic: for the same memory tasks, do RNN,
GRU, LSTM, SSM, and LRU models show persistence under blank input, recovery
from off-manifold perturbations, and preservation along tangent directions?

Modern LRU and SSM models are designed for long-sequence memory, but long
memory in a prediction loss is not identical to attractor geometry. A diagonal
state can decay slowly, or a gated model can retain information, without
necessarily forming a manifold that selectively contracts normal perturbations
while preserving nearby valid memories. This is also distinct from memory-cell
or gating mechanisms in GRU/LSTM: gates can help store a value, but they do not
by themselves specify which state directions should become an attracting
continuous memory manifold.

This distinction matters because ordinary sequence metrics can hide dynamical
failure. A network may predict the right value at a long horizon while its
hidden state is not a stable memory state under zero input. It may also solve
short training horizons while failing long blank rollouts, or preserve a value
only by using a fragile non-attracting trajectory.

We therefore evaluate recurrent memory using long-horizon attractor diagnostics
rather than prediction error alone:

```text
zero-input persistence:
  decoded memory remains stable after input is removed

normal recovery:
  off-manifold hidden perturbations contract back toward the memory manifold

tangent consistency:
  nearby valid memories remain distinct rather than collapsing together
```

The central claim is:

```text
We do not claim that standard recurrent architectures cannot implement
continuous attractors. Rather, under our training setup and diagnostics,
standard sequence training does not reliably induce the long-horizon attractor
properties required for analog memory. AM-LRU improves this gap by preserving
coordinates that are functionally necessary for long-horizon memory.
```

Contributions:

1. We formulate long-horizon diagnostics for continuous-attractor memory in
   recurrent state spaces.
2. Under a matched training and evaluation setup, we show that strong sequence
   memory does not reliably imply persistence, normal recovery, and tangent
   consistency.
3. We introduce AM-LRU, an Attractor-Manifold Linear Recurrent Unit whose
   retention values are shaped by coordinate-ablation effects.
4. We show that the improvement is explained by compact persistent support on
   high-ablation-effect coordinates rather than dense slowness.

## 2. Continuous-Attractor Diagnostics

We do not test for a mathematically exact invariant manifold that holds for
all time. Following Ságodi et al., *Back to the Continuous Attractor*, we treat
the exact object as an idealization and ask whether a trained network
exhibits the corresponding long-lived dynamical behavior. The ideal is a
continuous set of recurrent states that remains stable under blank dynamics,
is neutral along the represented variable, and contracts directions normal to
the memory manifold. In experiments, this means that decay must remain small
over sufficiently long evaluation horizons, rather than literally zero for all
time.

We therefore evaluate the operational behavioral counterpart: persistence
under blank input, recovery after off-manifold perturbation, and preservation
of nearby valid memories along the tangent direction.

For decoded memory `z`, recurrent state `h`, and blank-input rollout `F_0^H`,
we measure:

| diagnostic | meaning |
|---|---|
| `T2000` | decoded error after a length-2000 rollout |
| `post-H500` | decoded error after an additional 500 blank steps |
| `normal500` | decoded error 500 steps after an off-manifold hidden perturbation |
| `tangent500` | decoded error 500 steps after an in-manifold perturbation |

The key distinction is selective recovery. A point attractor contracts
everything. A continuous attractor should contract normal perturbations while
preserving nearby valid memories along tangent directions.

We also use an empirical local-linearization check. At a learned memory state
`h`, define the blank-input Jacobian

```text
J_0(h) = d F_0(h) / d h.
```

Let `Q_T(h)` be an orthonormal basis for the task tangent space, estimated by
small numerical perturbations in the represented memory variable. We report:

```text
tangent gain:
  singular values / eigenvalues of Q_T^T J_0 Q_T should be near 1

normal gain:
  random-vector gains and normal-spectrum summaries are contraction checks
```

The full hidden-state normal spectrum may contain decoder-null neutral
directions. We therefore interpret local linearization together with decoded
normal recovery, not as a replacement for the behavioral diagnostics.

## 3. Attractor Manifold Shaping

AM-LRU is an attractor-manifold variant of a linear recurrent unit. Each
coordinate has a retention value

```text
lambda_j = sqrt(sigmoid(theta_j)).
```

Raw task input `x_t` is first mapped to a network drive

```text
u_t = E(x_t).
```

Here `u_t` is the learned input embedding entering the recurrent carrier, not
the task velocity itself. In the AM-LRU and LRU full-block rows we use the
zero-drive wrapper: the encoder has no bias and the recurrent input
normalization is removed, so blank task input gives zero recurrent drive.

The recurrent carrier is real, diagonal, and coordinate-wise:

```text
h_{j,t+1} = lambda_j h_{j,t} + gamma_j b_j^T u_t.
```

`b_j` is row `j` of the learned input matrix `B`. `gamma_j` is a learned write
gain, implemented as `sigmoid(gamma_raw_j)` and initialized at `0.5`; unlike
canonical complex LRU input normalization, it is not tied to `lambda_j`.

For the one-layer full-block model used in the main experiments, the recurrent
state is mapped back to a stream by a learned output projection, followed by
GELU-GLU, residual addition, output layer normalization, and a linear decoding
head:

```text
r_t       = W_o h_t
s_t       = LN(E(x_t) + GLU(GELU(r_t)))
D(h_t)    = W_D LN(s_t).
```

AM-LRU estimates the functional contribution of coordinate `j` by ablating it
and measuring the increase in long-horizon memory error after blank rollout:

```text
E_mem(m; z) = mean_batch ||D(F_0^H(m)) - z||^2
A_j m       = m with coordinate j set to zero
Delta_j     = E_mem(A_j m; z) - E_mem(m; z).
```

The retention parameter is updated outside the task-gradient path:

```text
theta_j <- theta_j + eta_lambda (Delta_j - epsilon).
```

This rule does not reward slowness directly. It increases persistence when
removing a coordinate increases long-horizon memory error, and decreases
persistence when the ablation effect falls below threshold. Task loss trains the encoder, recurrent
input weights, write gains, recurrent readout, GLU path, and decoder. The
retention parameter `theta` has `requires_grad=False`; it is modified only by
the ablation update after the optimizer step. The implementation clamps
`theta` to `[-18, 18]` only to avoid numerical saturation.

The rule is coordinate-dependent because `Delta_j` is measured by coordinate
ablation. We treat this as an implementation choice rather than a claim of
basis-invariant optimality. Computationally, an ablation probe requires one clean
blank rollout plus coordinate-ablation rollouts for the recurrent carrier; in
our experiments this probe is run periodically rather than at every optimizer
step.

For the diagonal carrier, the update has a direct long-horizon interpretation.
Under blank input,

```text
h_j(H) = lambda_j^H h_j(0)
d h_j(H) / d lambda_j = H lambda_j^{H-1} h_j(0)
d lambda_j / d theta_j > 0.
```

Thus increasing `theta_j` reduces long-horizon decay of coordinate `j`.
The update increases retention only when coordinate ablation increases memory
error beyond `epsilon`; it drains coordinates whose ablation has little effect.
This does not prove global attractor learning, but it explains why the update
targets long-horizon persistence in the diagonal carrier.

## 4. Experimental Setup

We compare RNN, GRU, LSTM, SSM, LRU, and AM-LRU under a matched training and
evaluation protocol. The recurrent width is matched, while parameter counts
are reported explicitly. LRU and AM-LRU share the same full-block wrapper;
RNN/GRU/LSTM/SSM are direct recurrent baselines with MLP readouts.

Main models:

| model | role |
|---|---|
| RNN | vanilla recurrent baseline |
| GRU | gated recurrent baseline |
| LSTM | gated recurrent baseline with memory cell |
| SSM | diagonal state-space baseline |
| LRU | linear recurrent baseline |
| AM-LRU | LRU-style recurrent model with attractor manifold shaping |

Shared settings:

```text
training steps:        10,000
training horizon:      10-50
evaluation horizons:   50, 100, 200, 500, 1000, 2000
recurrent width:       96
layers:                1
optimizer:             AdamW, lr = 1e-3
LRU/AM-LRU lambda init: deterministic linspace 0.999 -> 0.90
AM-LRU eta_lambda:     3000 unless stated otherwise
```

Main tasks:

```text
line/plane integration:
  z_t = z_0 + sum_{k<=t} v_k in R^d

ring hold:
  cue theta once, then hold (cos theta, sin theta) under zero input
```

Main AM-LRU ablation settings:

```text
eta_lambda = 0:
  disables the AM-LRU ablation update while keeping the scaffold fixed

epsilon grid:
  eps in {0, 3e-5, 1e-4}; all grid values are shown in the appendix
```

Baseline fairness:

```text
same width, optimizer, training horizon, and training steps for all main rows
AM-LRU uses an additional periodic ablation probe; final tables should report
relative compute and wall-clock cost
```

Compute accounting to report in the final version:

| model | train steps | extra rollouts | relative compute |
|---|---:|---:|---:|
| RNN/GRU/LSTM/SSM/LRU | `10,000` | `0` | `1.0x` |
| AM-LRU | `10,000` | periodic clean + coordinate-ablation rollouts | to measure |
| AM-LRU `eta_lambda=0` | `10,000` | ablation update disabled | to measure |

Unless noted, tables report means over completed raw run files, aggregated by
`(task, dimension, model tag)`. The main line/ring tables have been refreshed
after the seed extension; `n` is shown per row because several older baseline
runs were only available for seeds `0, 3, 4`, whereas GRU/LSTM/LRU and AM-LRU
rows are available for seeds `0, 1, 2, 3, 4`.

## 5. Results

### 5.1 Baselines Often Miss The Diagnostic Bundle

Under the training setup used here, standard recurrent baselines can store
information, but they do not reliably produce the full continuous-attractor
diagnostic bundle. Their failures differ by task and architecture: some drift
during blank hold, some fail normal recovery, and some fail tangent consistency.

| condition | baseline | normal500 | tangent500 | read |
|---|---|---:|---:|---|
| line d=2 | GRU | `0.1531` | `0.1565` | remembers, but recovery is weak |
| line d=16 | LRU | `0.2895` | `0.2894` | high-rank memory remains unstable |
| ring hold | GRU | `3.713` | `3.667` | closed manifold is not stable |
| ring hold | LRU | `90.31` | `90.17` | recurrence alone is insufficient |

The same pattern holds in the full tables below: the standard architectures can
reduce training loss, but their blank-rollout and perturbation behavior does not
match the continuous-attractor diagnostic bundle.

### 5.2 AM-LRU Improves Line/Plane Memory And Ring Hold

Line/plane integration is the strongest rank-sweep result. AM-LRU
improves the whole attractor bundle: final error, post-hold stability, normal
recovery, and tangent consistency.

The main comparison table reports every primary model rather than selecting a
different comparator for each condition. Lower is better for all diagnostics.

| condition | model | n | primary | post | normal | tangent | params |
|---|---|---:|---:|---:|---:|---:|---:|
| line d=2 | RNN | `3` | `1.419` | `1.499` | `0.4138` | `0.4184` | `16.2k` |
| line d=2 | GRU | `5` | `0.8857` | `0.9858` | `0.1531` | `0.1565` | `36.0k` |
| line d=2 | LSTM | `5` | `1.297` | `1.387` | `0.3864` | `0.3963` | `45.9k` |
| line d=2 | SSM | `3` | `1.422` | `1.506` | `0.3992` | `0.3977` | `7.0k` |
| line d=2 | LRU | `5` | `1.454` | `1.491` | `0.3753` | `0.3774` | `56.8k` |
| line d=2 | AM-LRU, eps=1e-4 | `5` | `0.1683` | `0.1671` | `0.0033` | `0.0025` | `38.3k` |
| line d=8 | RNN | `3` | `0.7952` | `0.8343` | `0.3900` | `0.3899` | `17.8k` |
| line d=8 | GRU | `5` | `0.5523` | `0.6695` | `0.1344` | `0.1344` | `39.8k` |
| line d=8 | LSTM | `5` | `1.044` | `1.070` | `0.4861` | `0.4863` | `50.9k` |
| line d=8 | SSM | `3` | `0.7654` | `0.7898` | `0.3184` | `0.3186` | `8.6k` |
| line d=8 | LRU | `5` | `0.7613` | `0.7869` | `0.3058` | `0.3061` | `58.6k` |
| line d=8 | AM-LRU, eps=3e-5 | `5` | `0.0401` | `0.0369` | `0.0064` | `0.0037` | `40.0k` |
| line d=16 | RNN | `3` | `0.6618` | `0.6867` | `0.4330` | `0.4330` | `19.8k` |
| line d=16 | GRU | `5` | `0.5658` | `0.6686` | `0.1635` | `0.1634` | `45.0k` |
| line d=16 | LSTM | `5` | `0.6980` | `0.7447` | `0.3223` | `0.3222` | `57.6k` |
| line d=16 | SSM | `3` | `0.5855` | `0.6024` | `0.3190` | `0.3189` | `10.6k` |
| line d=16 | LRU | `5` | `0.5708` | `0.5916` | `0.2895` | `0.2894` | `60.9k` |
| line d=16 | AM-LRU, eps=0 | `5` | `0.0412` | `0.0402` | `0.0122` | `0.0098` | `42.4k` |
| ring hold | RNN | `3` | `59.24` | `58.71` | `21.12` | `20.97` | `16.3k` |
| ring hold | GRU | `5` | `13.53` | `16.76` | `3.713` | `3.667` | `36.3k` |
| ring hold | LSTM | `5` | `55.84` | `57.29` | `35.43` | `36.04` | `46.3k` |
| ring hold | SSM | `3` | `88.67` | `88.67` | `94.82` | `95.00` | `7.1k` |
| ring hold | LRU | `5` | `90.43` | `90.54` | `90.31` | `90.17` | `56.9k` |
| ring hold | AM-LRU, eps=3e-5 | `5` | `0.0442` | `0.0442` | `0.3992` | `0.0468` | `38.4k` |

Ring hold gives the cleanest closed-manifold result. The system receives one
cue on `S^1`, then must preserve the represented angle under zero input.

Local Jacobian diagnostics give a complementary view of the same behavior. We
linearize the blank-input map around decoded memory states and measure tangent
gain and normal gain. Tangent eigenvalues near one are not
enough by themselves: several baselines also show locally neutral tangent
directions. The useful signal is the bundle of near-neutral tangent dynamics,
low normal random gain, and long-horizon recovery.

| condition | model | tangent eig | normal eig max | normal random gain |
|---|---|---:|---:|---:|
| line d=2 | GRU | `1.000` | `0.921` | `0.756` |
| line d=2 | LRU | `0.986` | `0.999` | `1.142` |
| line d=2 | LSTM | `1.001` | `0.652` | `0.442` |
| line d=2 | AM-LRU | `1.003` | `1.000` | `0.238` |
| line d=8 | GRU | `1.000` | `0.946` | `0.884` |
| line d=8 | LRU | `1.003` | `0.999` | `1.523` |
| line d=8 | LSTM | `0.998` | `0.674` | `0.475` |
| line d=8 | AM-LRU | `1.006` | `1.000` | `0.296` |
| ring hold | GRU | `1.000` | `0.999` | `0.860` |
| ring hold | LRU | `0.974` | `0.999` | `1.867` |
| ring hold | LSTM | `0.999` | `0.863` | `0.443` |
| ring hold | AM-LRU | `1.000` | `1.000` | `0.467` |

The full hidden normal eigenvalue can remain near one even when decoded normal
recovery is strong, because hidden states may contain decoder-null or redundant
neutral directions. We therefore use Jacobians as a local diagnostic
alongside decoded perturbation recovery, not as a replacement for it.

### 5.3 Compact Retention And Latent Normal Recovery

AM-LRU begins from the same deterministic slow-reservoir
initialization used by the LRU baseline and ablations. Its final
support is compact: roughly `6-19` coordinates with `lambda > .99` across
line/plane ranks and about `10` on ring hold.

With positive `epsilon`, this compact support is produced by large lambda
redistribution. In seed-0 structured traces, thresholded ablation updates
typically reduce the mean lambda from about `0.95` to `0.10-0.27`, while
preserving high-ablation-effect coordinates near `lambda ~= 1`. By contrast,
`epsilon = 0` leaves the mean lambda closer to its initialization and mostly
adds a few persistent coordinates.

The direct control is `eta_lambda = 0`, which disables the AM-LRU ablation update.
The rows below aggregate all completed seeds for the same task and scaffold.

| condition | AM-LRU normal | eta=0 normal | AM-LRU tangent | eta=0 tangent | read |
|---|---:|---:|---:|---:|---|
| line d=2 | `0.0033` | `0.2620` | `0.0025` | `0.2659` | ablation updates are needed |
| line d=8 | `0.0064` | `0.2624` | `0.0037` | `0.2620` | fixed initial support is insufficient |
| line d=16 | `0.0122` | `0.2760` | `0.0098` | `0.2761` | update still matters at high rank |
| ring hold | `0.3992` | `16.64` | `0.0468` | `16.26` | no closed attractor without updates |

Dense-retention controls remain an important ablation target, but the main
claim below does not rely on them. The current clean mechanism evidence is the
gap between learned ablation updates and the `eta_lambda = 0` control under the
same recurrent scaffold.

We also evaluate a repeated normal-kick OOD probe on existing checkpoints. At
each kick time, a hidden-state perturbation is projected away from the local
memory tangent before injection. The reported value is the remaining hidden
normal component, normalized by injected noise budget.

| condition | AM-LRU | shuffled update | all-slow | read |
|---|---:|---:|---:|---|
| ring hold | `0.430` | `0.841` | `0.818` | lower latent normal residue |
| ring integrate | `0.334` | `0.747` | `0.663` | same pattern during tangent-driven input |
| line hold | `0.238` | `0.768` | `0.607` | compact retention contracts off-manifold state |
| line integrate | `0.242` | `0.776` | `0.685` | latent recovery persists under integration |

The same pattern appears in a one-step latent Jacobian analysis. We linearize
the recurrent transition at states on the learned manifold and report tangent
gain and normal random gain. For integration tasks, the Jacobian uses the
task's tangent input at the middle of the rollout.

| condition | tangent gain | AM-LRU normal | shuffled normal | all-slow normal |
|---|---:|---:|---:|---:|
| ring hold | `1.000` | `0.450` | `0.833` | `0.967` |
| ring integrate | `0.998` | `0.356` | `0.757` | `0.774` |
| line hold | `1.000` | `0.238` | `0.767` | `0.768` |
| line integrate | `1.001` | `0.243` | `0.773` | `0.784` |

The defensible mechanism claim is:

```text
AM-LRU learns compact persistence on coordinates with large ablation effects,
not merely a large slow reservoir, and this compact support gives stronger
latent contraction of normal perturbations.
```

### 5.4 Limitation In Controlled Tangent Flow

Ring integration separates stable attractor memory from controllable tangent
dynamics. AM-LRU stabilizes the ring and gives strong
normal/tangent recovery, but active angular flow remains difficult. In
long-hold ring integration, GRU/LSTM baselines can sometimes learn the driven
angular update better, while AM-LRU still has much stronger
perturbation recovery.

This limitation is useful rather than incidental: it shows that forming a
stable continuous memory manifold and learning input-controlled motion along
that manifold are separable problems. The main claim of this paper is about
long-horizon attractor memory; controlled tangent flow is a next target.

## 6. Discussion

Long-horizon memory does not imply continuous-attractor geometry. The relevant
behavior is selective: stable zero-input persistence, recovery from normal
perturbations, and preservation of nearby valid memories along the tangent
direction.

AM-LRU improves these diagnostics by assigning retention according to
functional coordinate-ablation effects. This makes the result different from simply
slowing the whole recurrent reservoir. The method is still basis-dependent
because it uses coordinate ablation, and the results should be read as
long-horizon attractor-like behavior rather than exact continuous attractors.

The broader conclusion is that stable attractor memory and controllable
tangent dynamics should be studied separately. AM-LRU provides a simple
mechanism for the former; the latter remains an open direction for ring
integration and other controlled analog-memory tasks.

## References To Cite

Core continuous-attractor references:

```text
Zhang, 1996.
Classical ring-attractor theory for head-direction memory.

Seung, 1998.
Classical study of continuous-attractor learning in recurrent networks.

Ságodi, Martín-Sánchez, Sokół, Park, 2024.
Modern view of approximate, long-lived continuous attractors.
```

Trained recurrent dynamics:

```text
Sussillo and Barak, 2013.
Low-dimensional dynamical analysis of trained recurrent networks.

Maheswaranathan, Williams, Golub, Ganguli, Sussillo, 2019.
Line-attractor dynamics in trained recurrent sentiment classifiers.
```

Modern recurrent and state-space memory:

```text
Hochreiter and Schmidhuber, 1997.
Gated recurrent memory cells for long time-lag dependencies.

Cho et al., 2014.
Learning Phrase Representations using recurrent encoder-decoder architectures.

Chung et al., 2014.
Empirical evaluation of GRU and LSTM sequence models.

Orvieto et al., 2023.
Modern linear recurrent sequence models for long sequences.

Dao and Gu, 2024.
Structured state-space duality and efficient sequence models.
```

## Appendix A. Experimental Protocol Details

### A.1 Shared Training And Evaluation Protocol

Current main experiments use synthetic data generated online, so each training
and evaluation call samples fresh sequences from the task distribution. There
is not yet a fixed held-out validation file; final model selection should either
fix AM-LRU `epsilon` before testing or use a separate validation seed set.

| item | value |
|---|---:|
| train steps | `10,000` |
| train batch size | `256` |
| train horizon | integer sampled uniformly from `10..50` |
| optimizer | AdamW |
| learning rate | `1e-3` |
| weight decay | `1e-5` |
| gradient clipping | global norm `1.0` |
| eval batch per horizon | `512` |
| evaluation horizons | `50, 100, 200, 500, 1000, 2000` |
| post-hold rollout | `500` blank steps |
| perturbation analysis batch | `96` |
| local Jacobian samples | `8` states in the current Jacobian script |
| recurrent width | `96` |
| layers | `1` |
| dropout | `0.0` |
| validation set size | not fixed in the current draft; final epsilon/model selection should use separate validation seeds |
| current seed set | main rows refreshed after the extension; `n` is shown per row because some baselines are available for `0, 3, 4` while GRU/LSTM/LRU and AM-LRU are available for `0..4` |

The loss is mean squared error over the full output sequence:

```text
L_task = mean_{t,b} || y_hat_{t,b} - y_{t,b} ||^2.
```

For ring tasks, training still uses vector MSE on `(cos theta, sin theta)`.
Reported ring errors use wrapped angle error in degrees.

### A.2 Model Definitions And Parameter Matching

Models are width-matched rather than parameter-count matched. Parameter counts
are reported because the wrappers differ.

| model | implementation used in main tables |
|---|---|
| RNN | `nn.RNNCell` with tanh state and an MLP readout |
| GRU | `nn.GRUCell` with an MLP readout |
| LSTM | `nn.LSTMCell`; state is concatenated `(h, c)`, decoder reads `h` |
| SSM | diagonal state-space baseline, `h_{t+1}=a*h_t+B x_t+b`, `a=sigmoid(a_raw)` |
| LRU | complex diagonal LRU full block with zero-drive wrapper |
| AM-LRU | real diagonal LRU full block with zero-drive wrapper and ablation update |

The direct recurrent baselines use the following state updates:

```text
RNN:
  h_{t+1} = tanh(W_x x_t + W_h h_t + b)

GRU:
  standard GRUCell update

LSTM:
  standard LSTMCell update for (h_t, c_t)

SSM:
  h_{t+1} = a * h_t + B x_t + b
  a       = sigmoid(a_raw), initialized from a_raw = 2.2
```

The LRU and AM-LRU rows share the full-block wrapper:

```text
encoder -> recurrent carrier -> recurrent output projection
        -> GELU/GLU update -> residual -> output LayerNorm -> linear head
```

Inside this wrapper, the LRU carrier is the complex diagonal recurrence

```text
c_{j,t+1} = rho_j exp(i phi_j) c_{j,t} + gamma_j B_j^T u_t
rho_j     = exp(-exp(nu_j))
gamma_j   = sqrt(1 - rho_j^2)
```

where the complex state is stored as real and imaginary components. The AM-LRU
carrier replaces this with the real diagonal recurrence in Section 3:

```text
h_{j,t+1} = lambda_j h_{j,t} + gamma_j b_j^T u_t
lambda_j  = sqrt(sigmoid(theta_j)).
```

For both LRU and AM-LRU, the recurrent input `u_t` is the model stream produced
by the encoder and wrapper, not the task velocity variable.

For LRU and AM-LRU, blank input is a true zero drive because the encoder bias
and recurrent input LayerNorm are disabled. RNN/GRU/LSTM/SSM are direct
step-wise baselines with MLP readouts under the same training and evaluation
protocol.

LRU and AM-LRU retention values are initialized by a deterministic linspace
from `0.999` down to `0.90`. SSM uses the `a_raw=2.2` initialization above, and
RNN/GRU/LSTM use the default PyTorch recurrent-cell initialization.

Representative trainable parameter counts:

| condition | RNN | GRU | LSTM | SSM | LRU | AM-LRU |
|---|---:|---:|---:|---:|---:|---:|
| line d=2 | `16.2k` | `36.0k` | `45.9k` | `7.0k` | `56.8k` | `38.3k` |
| line d=8 | `17.8k` | `39.8k` | `50.9k` | `8.6k` | `58.6k` | `40.0k` |
| line d=16 | `19.8k` | `45.0k` | `57.6k` | `10.6k` | `60.9k` | `42.4k` |
| ring hold | `16.3k` | `36.3k` | `46.3k` | `7.1k` | `56.9k` | `38.4k` |

For LSTM perturbation diagnostics, the perturbation is applied to the full
concatenated recurrent state `(h, c)`. The decoder then reads the perturbed
hidden component `h` after rollout.

### A.3 AM-LRU Ablation Update Details

AM-LRU uses the same AdamW optimizer as the baselines for all trainable
parameters except `theta`. The retention parameter `theta` is not in the
gradient path and is updated periodically:

| item | value |
|---|---:|
| ablation probe frequency | every `100` optimizer steps |
| warmup before retention updates | first `30%` of training steps |
| probe batch | `96` |
| probe trajectory horizon | `50` |
| blank probe rollout `H` | `500` |
| epsilon mode | fixed absolute threshold |
| epsilon grid | `0`, `3e-5`, `1e-4` |
| eta_lambda | `3000` |
| theta clamp | `[-18, 18]` numerical guard |

`Delta_j` is a batch-averaged difference in squared decoded memory error. The
large numerical value of `eta_lambda` should be read relative to this scale:
typical `Delta_j - epsilon` values are small, and the update is applied only
once every `100` optimizer steps after warmup.

Current main rows show one representative AM-LRU epsilon for each displayed
condition, while Appendix D reports the epsilon grid. The final protocol should
select epsilon on a validation split before reporting held-out test numbers.

### A.4 Hardware And Compute Accounting

Experiments are run on six local NVIDIA TITAN RTX GPUs with 24GB memory each.
Each JSON result stores wall-clock seconds and parameter counts. The current
draft reports parameter counts and per-row seed counts; normalized total
training compute is still reported qualitatively because AM-LRU adds periodic
ablation-probe rollouts.

### A.5 Line / Plane Integration

Task:

```text
z_0 ~ Uniform([-z_scale, z_scale]^d)
v_t ~ Uniform([-vel_scale / sqrt(d), vel_scale / sqrt(d)]^d)
z_t = z_0 + sum_{k<=t} v_k
```

Input and output:

```text
input_dim  = 2d + 1
output_dim = d

t = 0:
  x_0[:d]  = z_0
  cue bit  = 1

t > 0:
  x_t[d:2d] = v_t
```

Default values:

| setting | value |
|---|---:|
| `z_scale` | `0.50` |
| `vel_scale` | `0.08` |
| train horizon | random `10..50` |
| eval horizons | `50, 100, 200, 500, 1000, 2000` |
| post-hold | `500` blank steps |
| analysis horizon | `50` |
| normal perturb radius | `0.25 * state_rms` |
| tangent delta | `0.10 / sqrt(d)` |

### A.6 Ring Hold

Task:

```text
theta_0 ~ Uniform([0, 2pi])
target = (cos theta_0, sin theta_0)
```

Input:

```text
input_dim = 6

t = 0:
  x_0[0:2] = (cos theta_0, sin theta_0)
  x_0[5]   = 1

t > 0:
  x_t = 0
```

The target is constant for all timesteps. This is the cleanest test of a closed
continuous attractor under zero input.

### A.7 Ring Integration

Task:

```text
theta_t = theta_0 + sum_{k<=t} omega_k
target_t = (cos theta_t, sin theta_t)
```

Current input encoding:

```text
input_dim = 6

t = 0:
  x_0 = (cos theta_0, sin theta_0, 0, 0, 0, cue=1)

t > 0:
  x_t = (0, 0, omega_t, sin omega_t, cos omega_t - 1, cue=0)
```

The `cos omega_t - 1` term keeps blank updates exactly zero when
`omega_t = 0`.

## Appendix B. Diagnostic Definitions

### B.1 Long-Horizon Error

For vector tasks:

```text
RMSE_T = sqrt(mean(||D(h_T) - z_T||^2 per coordinate)).
```

For ring tasks:

```text
angle_error = mean absolute wrapped angular error in degrees.
radius_rmse = sqrt(mean((||D(h)||_2 - 1)^2)).
```

### B.2 Post-Hold

After an active trajectory reaches its final state, the system is rolled
forward for `500` blank steps:

```text
h_post = F_0^500(h_T)
```

The decoded error from `h_post` to the same final target measures whether the
state is an attracting or drifting memory after input is removed.

### B.3 Normal Recovery

A local tangent basis is estimated by small numerical perturbations around the memory
manifold. A random hidden perturbation is projected into the orthogonal
complement:

```text
n = random hidden noise
tangent = projection of n onto local tangent basis
normal = normalize(n - tangent)
h_pert = h + radius * normal
```

The system then rolls blank for `R` steps. The main reported value is decoded
memory error after `R = 500`.

### B.4 Tangent Consistency

A tangent perturbation creates a nearby valid memory:

```text
line:
  z_0 -> z_0 + delta

ring:
  theta -> theta + delta
```

The correct behavior is to preserve the shifted memory after blank rollout.
Returning to the original memory would indicate point-attractor collapse.

### B.5 Local Jacobian

The Jacobian diagnostic reports:

```text
tangent_eig_abs_mean
tangent_sv_mean
normal_eig_abs_max
normal_random_gain
```

The local Jacobian is a support diagnostic. The main attractor evidence remains
the behavioral bundle: persistence, normal recovery, and tangent consistency.

### B.6 Retention Counts

Reported retention diagnostics:

```text
lambda_sum
lambda_max
lambda_gt_0p9
lambda_gt_0p95
lambda_gt_0p99
```

High retention count alone is insufficient. Dense-persistence controls can
keep many coordinates near one without matching AM-LRU's attractor recovery.

## Appendix C. Controls And Ablations

### C.1 Update And Threshold Controls

| model | setting | paper use | current read |
|---|---|---|---|
| AM-LRU | `eta_lambda = 0` | main text | disables ablation update; much worse |
| GRU/LSTM/LRU | blank-rollout auxiliary loss | appendix | extra blank-hold supervision does not reproduce the result |
| GRU/LSTM/LRU | hidden perturbation augmentation | future control | perturbations are evaluated, but not yet used as a training control |

Representative controls:

| control | condition | model | normal | tangent | lambda > .99 | read |
|---|---|---|---:|---:|---:|---|
| blank-rollout auxiliary | line d=2 | GRU | `0.317` | `0.314` | -- | worse than AM-LRU |
| blank-rollout auxiliary | line d=8 | LRU | `0.313` | `0.313` | `13.0` | worse than AM-LRU |
| blank-rollout auxiliary | ring hold | GRU | `11.12` | `11.36` | -- | does not produce ring attractor |

The important interpretation is that extra blank-rollout supervision alone does
not match AM-LRU's normal recovery and tangent consistency. A clean dense-slow
LRU control should be reported only after it is rerun under the same naming and
aggregation convention.

### C.2 Initialization

The current main setting initializes slow coordinates with:

```text
lambda_j = linspace(0.999, 0.90)
```

This was adopted because AM-LRU needs a reservoir of candidate slow
coordinates. Current summary:

```text
The rule can start from several retention distributions, provided that enough
slow candidates exist early in training.
```

Initialization conditions explored:

| init family | description | current read |
|---|---|---|
| `linspace 0.999 -> 0.90` | deterministic slow reservoir | main setting |
| `random/uniform broad` | broad retention support | can work if enough slow candidates exist |
| `gaussian near 0.9` | slow-biased Gaussian | can work |
| `bimodal` | mixture of fast and slow coordinates | can work |
| `gaussian near 0.5` | mostly non-slow reservoir | weak/failing condition |
| `fixed 0.99` | all coordinates slow | useful diagnostic, less selective |

Current interpretation:

```text
epsilon > 0:
  most low-ablation-effect coordinates are actively drained.
  examples from rank-2 threshold sweep:
    eps=1e-4: mean lambda 0.949 -> 0.073, lambda>.99 9 -> 7
    eps=3e-5: mean lambda 0.949 -> 0.206, lambda>.99 9 -> 7
    eps=1e-3: mean lambda 0.949 -> 0.042, lambda>.99 9 -> 4

epsilon = 0:
  lambda movement is much smaller.
  traces keep mean lambda close to initialization and mainly add a few
  coordinates above .99.
```

The strongest mechanism figure should compare:

```text
initial lambda distribution
final lambda distribution
Delta_j distribution
lambda vs Delta_j scatter
```

### C.3 Repeated Normal-Kick OOD Probe

The repeated normal-kick probe is evaluation-only. Existing checkpoints are
rolled out under either zero-input hold or task integration input. At selected
times, a hidden perturbation is projected orthogonal to the local memory tangent
and injected into the recurrent state. The table below shows the hardest setting
used in the first pass: `100` kicks, radius `1.0 * state_rms`, horizon `500`.

| condition | model | n | clean | perturbed | excess | hidden normal gain |
|---|---|---:|---:|---:|---:|---:|
| ring hold | AM-LRU | `3` | `0.056` | `15.68` | `15.62` | `0.430` |
| ring hold | shuffled update | `3` | `0.147` | `10.32` | `10.17` | `0.841` |
| ring hold | all-slow | `3` | `1.188` | `13.64` | `12.45` | `0.818` |
| ring integrate | AM-LRU | `3` | `15.26` | `23.36` | `8.10` | `0.334` |
| ring integrate | shuffled update | `3` | `15.04` | `23.03` | `7.99` | `0.747` |
| ring integrate | all-slow | `3` | `29.78` | `31.75` | `1.97` | `0.663` |
| line hold | AM-LRU | `3` | `0.0028` | `0.0655` | `0.0627` | `0.238` |
| line hold | shuffled update | `3` | `0.0020` | `0.0502` | `0.0483` | `0.768` |
| line hold | all-slow | `3` | `0.1483` | `0.1505` | `0.0023` | `0.607` |
| line integrate | AM-LRU | `3` | `0.0138` | `0.1535` | `0.1396` | `0.242` |
| line integrate | shuffled update | `3` | `0.0237` | `0.1407` | `0.1170` | `0.776` |
| line integrate | all-slow | `3` | `0.2898` | `0.3005` | `0.0106` | `0.685` |

Read:

```text
AM-LRU consistently leaves a smaller latent normal component than shuffled
or all-slow controls.
```

### C.4 Latent Jacobian Controls

We also compute a local one-step Jacobian at latent states on the learned
manifold. For hold tasks, the step uses zero input. For integration tasks, the
step uses the task's tangent input at the middle of the rollout.

| condition | model | n | tangent eig | normal random gain | normal eig max |
|---|---|---:|---:|---:|---:|
| ring hold | AM-LRU | `24` | `1.000` | `0.450` | `1.000` |
| ring hold | shuffled update | `24` | `1.000` | `0.833` | `1.000` |
| ring hold | all-slow | `24` | `1.000` | `0.967` | `0.999` |
| ring integrate | AM-LRU | `24` | `0.998` | `0.356` | `1.003` |
| ring integrate | shuffled update | `24` | `0.998` | `0.757` | `1.001` |
| ring integrate | all-slow | `24` | `0.998` | `0.774` | `1.001` |
| line hold | AM-LRU | `24` | `1.000` | `0.238` | `1.000` |
| line hold | shuffled update | `24` | `1.000` | `0.767` | `1.000` |
| line hold | all-slow | `24` | `0.999` | `0.768` | `0.999` |
| line integrate | AM-LRU | `24` | `1.001` | `0.243` | `1.000` |
| line integrate | shuffled update | `24` | `1.000` | `0.773` | `1.000` |
| line integrate | all-slow | `24` | `1.000` | `0.784` | `1.000` |

Read:

```text
AM-LRU has tangent eigenvalues near one like the controls, but much smaller
normal random gain. This supports the interpretation that the learned latent
manifold is locally attracting in normal directions.
```

## Appendix D. Secondary Result Tables

These tables use the same raw aggregation convention as the main text. Values
are means over completed seeds, with `n` shown per row.

### D.1 Line Integration

| dim | model | n | T2000 | postH500 | normal500 | tangent500 | params |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | RNN | `3` | `2.009` | `2.154` | `0.7234` | `0.7294` | `16.0k` |
| 1 | GRU | `5` | `1.415` | `1.463` | `0.2029` | `0.2064` | `35.4k` |
| 1 | LSTM | `5` | `1.581` | `1.667` | `0.3342` | `0.3326` | `45.1k` |
| 1 | SSM | `3` | `1.942` | `2.101` | `0.4643` | `0.4742` | `6.8k` |
| 1 | LRU | `5` | `2.024` | `2.076` | `0.4427` | `0.4539` | `56.5k` |
| 1 | AM-LRU eps=0 | `5` | `0.5555` | `0.5556` | `0.0041` | `0.0039` | `38.0k` |
| 1 | AM-LRU eps=3e-5 | `5` | `0.5087` | `0.5082` | `0.0039` | `0.0032` | `38.0k` |
| 1 | AM-LRU eps=1e-4 | `5` | `0.4979` | `0.4975` | `0.0027` | `0.0020` | `38.0k` |
| 2 | RNN | `3` | `1.419` | `1.499` | `0.4138` | `0.4184` | `16.2k` |
| 2 | GRU | `5` | `0.8857` | `0.9858` | `0.1531` | `0.1565` | `36.0k` |
| 2 | LSTM | `5` | `1.297` | `1.387` | `0.3864` | `0.3963` | `45.9k` |
| 2 | SSM | `3` | `1.422` | `1.506` | `0.3992` | `0.3977` | `7.0k` |
| 2 | LRU | `5` | `1.454` | `1.491` | `0.3753` | `0.3774` | `56.8k` |
| 2 | AM-LRU eps=0 | `5` | `0.2225` | `0.2230` | `0.0063` | `0.0059` | `38.3k` |
| 2 | AM-LRU eps=3e-5 | `5` | `0.1792` | `0.1781` | `0.0042` | `0.0035` | `38.3k` |
| 2 | AM-LRU eps=1e-4 | `5` | `0.1683` | `0.1671` | `0.0033` | `0.0025` | `38.3k` |
| 4 | RNN | `3` | `1.021` | `1.094` | `0.4185` | `0.4197` | `16.7k` |
| 4 | GRU | `5` | `0.7445` | `0.8858` | `0.1044` | `0.1052` | `37.3k` |
| 4 | LSTM | `5` | `1.171` | `1.218` | `0.4495` | `0.4523` | `47.6k` |
| 4 | SSM | `3` | `1.027` | `1.071` | `0.3404` | `0.3414` | `7.5k` |
| 4 | LRU | `5` | `1.045` | `1.074` | `0.3381` | `0.3397` | `57.4k` |
| 4 | AM-LRU eps=0 | `5` | `0.1117` | `0.1094` | `0.0080` | `0.0071` | `38.9k` |
| 4 | AM-LRU eps=3e-5 | `5` | `0.0731` | `0.0700` | `0.0060` | `0.0047` | `38.9k` |
| 4 | AM-LRU eps=1e-4 | `5` | `0.0694` | `0.0663` | `0.0069` | `0.0058` | `38.9k` |
| 8 | RNN | `3` | `0.7952` | `0.8343` | `0.3900` | `0.3899` | `17.8k` |
| 8 | GRU | `5` | `0.5523` | `0.6695` | `0.1344` | `0.1344` | `39.8k` |
| 8 | LSTM | `5` | `1.044` | `1.070` | `0.4861` | `0.4863` | `50.9k` |
| 8 | SSM | `3` | `0.7654` | `0.7898` | `0.3184` | `0.3186` | `8.6k` |
| 8 | LRU | `5` | `0.7613` | `0.7869` | `0.3058` | `0.3061` | `58.6k` |
| 8 | AM-LRU eps=0 | `5` | `0.0559` | `0.0538` | `0.0076` | `0.0055` | `40.0k` |
| 8 | AM-LRU eps=3e-5 | `5` | `0.0401` | `0.0369` | `0.0064` | `0.0037` | `40.0k` |
| 8 | AM-LRU eps=1e-4 | `5` | `0.0420` | `0.0387` | `0.0070` | `0.0047` | `40.0k` |
| 16 | RNN | `3` | `0.6618` | `0.6867` | `0.4330` | `0.4330` | `19.8k` |
| 16 | GRU | `5` | `0.5658` | `0.6686` | `0.1635` | `0.1634` | `45.0k` |
| 16 | LSTM | `5` | `0.6980` | `0.7447` | `0.3223` | `0.3222` | `57.6k` |
| 16 | SSM | `3` | `0.5855` | `0.6024` | `0.3190` | `0.3189` | `10.6k` |
| 16 | LRU | `5` | `0.5708` | `0.5916` | `0.2895` | `0.2894` | `60.9k` |
| 16 | AM-LRU eps=0 | `5` | `0.0412` | `0.0402` | `0.0122` | `0.0098` | `42.4k` |
| 16 | AM-LRU eps=3e-5 | `5` | `0.1070` | `0.1054` | `0.0537` | `0.0513` | `42.4k` |
| 16 | AM-LRU eps=1e-4 | `5` | `0.2438` | `0.2433` | `0.1231` | `0.1227` | `42.4k` |

Read:

```text
AM-LRU improves the continuous vector attractor diagnostics across tested
ranks. Positive threshold helps for d=1..8, while d=16 currently prefers zero
threshold.
```

### D.2 Ring Hold

| model | n | angle2000 | post | normal | tangent | params |
|---|---:|---:|---:|---:|---:|---:|
| RNN | `3` | `59.24` | `58.71` | `21.12` | `20.97` | `16.3k` |
| GRU | `5` | `13.53` | `16.76` | `3.713` | `3.667` | `36.3k` |
| LSTM | `5` | `55.84` | `57.29` | `35.43` | `36.04` | `46.3k` |
| SSM | `3` | `88.67` | `88.67` | `94.82` | `95.00` | `7.1k` |
| LRU | `5` | `90.43` | `90.54` | `90.31` | `90.17` | `56.9k` |
| AM-LRU eps=0 | `3` | `0.7677` | `0.7677` | `0.9409` | `0.8482` | `38.4k` |
| AM-LRU eps=3e-5 | `5` | `0.0442` | `0.0442` | `0.3992` | `0.0468` | `38.4k` |
| AM-LRU eps=1e-4 | `5` | `0.0948` | `0.0948` | `0.4148` | `0.1045` | `38.4k` |

Read:

```text
The thresholded ablation update gives near-zero drift and strong radial
recovery. The `eps=0` ring row is available for three completed seeds, while
the thresholded AM-LRU rows are available for five seeds.
```

### D.3 Ring Integration

Ring integration is a limitation and should not be a central claim.

```text
Active angular integration:
  GRU/LSTM baselines can learn the driven angular update better.
  AM-LRU gives much stronger perturbation recovery.

Long-hold setting:
  longer zero-input suffix improves recovery analysis.
  stable ring memory and input-controlled tangent flow remain separable.
```

## Appendix E. Figure Inventory

Current paper-relevant figures:

| purpose | file |
|---|---|
| main normal recovery comparison | `figures_attractor_recovery/attractor_normal_recovery_main.png` |
| ring hold radial recovery | `figures_attractor_recovery/ring_hold_radial_recovery_over_time.png` |
| line hidden-normal recovery | `figures_attractor_recovery/line_hold_hidden_normal_recovery_over_time.png` |
| line in-plane/tangent persistence | `figures_attractor_recovery/line_hold_in_plane_shift_persistence_over_time.png` |

Interpretation notes:

```text
Ring radial recovery:
  shows movement in the decoded ring plane.
  AM-LRU returns toward the ring after radial perturbation.

Line hidden-normal recovery:
  shows contraction from an off-manifold hidden perturbation.

Line in-plane shift:
  this is tangent, so correct behavior is to preserve the shifted memory.
  returning to the original point would indicate point-attractor collapse.
```

## Appendix F. Current Analysis Artifacts

Primary analysis report:

```text
analysis_attractor_properties/attractor_property_report.md
```

CSV summaries:

```text
analysis_attractor_properties/line_integrate_attractor_summary.csv
analysis_attractor_properties/ring_hold_attractor_summary.csv
analysis_attractor_properties/ring_integrate_original_attractor_summary.csv
analysis_attractor_properties/ring_integrate_longhold_attractor_summary.csv
analysis_jacobian_diagnostics/jacobian_diagnostics_3seed.csv
analysis_jacobian_diagnostics/JACOBIAN_DIAGNOSTICS_3seed.md
analysis_normal_kick_ood_all/normal_kick_ood_summary_all.csv
analysis_normal_kick_ood_all/NORMAL_KICK_OOD_SUMMARY_ALL.md
analysis_latent_jacobian_controls/latent_jacobian_controls_summary.csv
analysis_latent_jacobian_controls/LATENT_JACOBIAN_CONTROLS.md
```

Analysis scripts:

```text
analyze_attractor_properties.py
analyze_jacobian_diagnostics.py
analyze_normal_kick_ood.py
analyze_latent_jacobian_controls.py
plot_attractor_recovery_manifold_figure.py
```

## Appendix G. Immediate Next Experiments

Completed seed extension:

```text
main rows refreshed after adding seeds 3 and 4
row-wise n is reported explicitly because a few older RNN/SSM seeds are absent
completed rows:
  line/plane integration rank sweep
  ring hold
  eta_lambda = 0 control
  all primary baselines
```

Jacobian diagnostic:

```text
current status:
  completed first-pass 3-seed analysis on existing checkpoints
  refresh after collecting the missing baseline seeds if Jacobian statistics
  are promoted from appendix diagnostic to main-text evidence

primary conditions:
  line_integrate d=2
  line_integrate d=8
  ring_hold d=1

reported diagnostics:
  tangent_eig_abs_mean and tangent_sv_mean near 1
  normal_eig_abs_max and normal_random_gain as contraction checks
```

Remaining mechanism figures:

```text
lambda trajectory over training
final lambda distribution
Delta_j distribution
lambda vs Delta_j scatter
normal recovery over time
tangent shift preservation over time
```

Additional controls and task coverage:

```text
hidden perturbation augmentation:
  train GRU/LSTM/LRU with explicit hidden perturbation recovery loss.
  use this to check whether AM-LRU's recovery advantage is only due to the
  evaluation objective being absent from baseline training.
```
