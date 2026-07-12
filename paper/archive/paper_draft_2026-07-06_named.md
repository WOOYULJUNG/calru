# Learning Recurrent Dynamics as a Continuous Attractor

*Working draft, updated 2026-07-06.*

## Abstract

Continuous attractors are a classical model of analog memory, but long-horizon
prediction alone does not show that a recurrent model has learned continuous
attractor geometry. A continuous memory should persist under zero input,
recover from perturbations normal to the memory manifold, and preserve nearby
valid memories along tangent directions. We evaluate modern recurrent
architectures as dynamical systems under these diagnostics rather than only as
finite-horizon predictors.

We compare RNN, GRU, LSTM, SSM, LRU, and PAN under a shared recurrent scaffold.
PAN is a damage-aligned persistence rule: a recurrent coordinate becomes more
persistent when ablating it increases long-horizon memory error. Across
line/plane integration and ring hold tasks, standard recurrent architectures
can remember but do not reliably satisfy the full continuous-attractor
diagnostic bundle. PAN gives stronger zero-input persistence, normal recovery,
and tangent consistency. Its advantage is explained by compact persistent
support aligned with coordinate damage, not by making the whole reservoir slow.

## 1. Introduction

Modern recurrent architectures can remember, but memory is not the same as
continuous attractor dynamics. A model may keep enough information to reduce a
sequence loss while its state still drifts under zero input, fails to recover
from hidden perturbations, or collapses nearby continuous memories into a
single point. This distinction matters for analog memory: the desired object is
not one stable state, but a continuous family of valid states.

Classical work on continuous attractors and trained recurrent dynamics provides
the right language for this distinction. Seung (1998) studied continuous
attractor learning in recurrent networks, Zhang (1996) described ring
attractors for head-direction memory, and Sussillo and Barak (2013) showed how
trained RNNs can be analyzed through low-dimensional dynamical structure.
Recent work by Ságodi et al. (2024) sharpens the target: exact continuous
attractors are fragile, but finite-time approximate attractor behavior remains
a meaningful model of analog memory.

We ask whether modern recurrent networks learn this finite-time attractor
geometry on explicit continuous memory tasks. We evaluate RNN, GRU, LSTM, SSM,
LRU, and PAN using three diagnostics:

```text
zero-input persistence:
  decoded memory remains stable after input is removed

normal recovery:
  off-manifold hidden perturbations contract back toward the memory manifold

tangent consistency:
  nearby valid memories remain distinct rather than collapsing together
```

Our central claim is:

```text
Modern recurrent architectures can remember, but they do not reliably learn
continuous attractor geometry. PAN improves this gap through damage-aligned
persistence.
```

Contributions:

1. We formulate finite-time continuous-attractor diagnostics for recurrent
   memory.
2. We show that standard recurrent baselines do not reliably satisfy the full
   diagnostic bundle.
3. We introduce PAN, a damage-aligned persistence rule for recurrent state
   coordinates.
4. We show that PAN's attractor behavior is explained by compact persistent
   support aligned with functional damage, not dense slowness.

## 2. Finite-Time Continuous-Attractor Diagnostics

We do not test for an exact invariant manifold over infinite time. Following
the finite-time view of continuous attractors, we ask whether a learned
recurrent state behaves like an attractor over long but finite evaluation
horizons. This gives a practical diagnostic bundle.

For decoded memory `z`, recurrent state `h`, and blank-input rollout `F_0^H`,
we measure:

| diagnostic | meaning |
|---|---|
| `T2000` | decoded error after a length-2000 rollout |
| `post-H500` | decoded error after an additional 500 blank steps |
| `normal500` | decoded error 500 steps after an off-manifold hidden perturbation |
| `tangent500` | decoded error 500 steps after an in-manifold/tangent perturbation |

The key distinction is selective recovery. A point attractor contracts
everything. A continuous attractor should contract normal perturbations while
preserving nearby valid memories along tangent directions.

We also use an empirical local-linearization check. At a learned memory state
`h`, define the blank-input Jacobian

```text
J_0(h) = d F_0(h) / d h.
```

Let `Q_T(h)` be an orthonormal basis for the task tangent space, estimated by
finite differences in the represented memory variable. We report tangent gain,
normal gain, and tangent leakage:

```text
tangent gain:
  singular values / eigenvalues of Q_T^T J_0 Q_T should be near 1

normal gain:
  eigenvalues, singular values, and random-vector gains in the normal
  complement are reported as contraction checks

tangent leakage:
  ||(I - Q_T Q_T^T) J_0 Q_T|| / ||J_0 Q_T|| should be small
```

The full hidden-state normal spectrum may include decoder-null neutral
directions. We therefore interpret local linearization together with decoded
normal recovery, not as a replacement for the behavioral diagnostics.

## 3. PAN

PAN learns which recurrent coordinates should be persistent. It uses a
coordinate-wise retention parameter

```text
lambda_j = sqrt(sigmoid(theta_j)).
```

In the implementation used here, the persistent carrier is the recurrent state:

```text
h_{j,t+1} = lambda_j h_{j,t} + gamma_j b_j^T u_t.
```

PAN scores a coordinate by ablating it and measuring the increase in
long-horizon memory error after blank rollout:

```text
E_mem(m; z) = ||D(F_0^H(m)) - z||^2
A_j m       = m with coordinate j set to zero
s_j         = E_mem(A_j m; z) - E_mem(m; z).
```

The persistence parameter is updated outside autograd:

```text
theta_j <- theta_j + eta_lambda (s_j - epsilon).
```

Thus PAN does not reward slowness directly. It increases persistence when
removing a coordinate damages long-horizon memory, and decreases persistence
when the damage falls below threshold. Task loss trains the encoder, recurrent
input weights, readout, and decoder; the PAN update modifies only `theta`.

The rule is basis-dependent because damage is measured after coordinate
ablation. We treat this as an implementation choice rather than a claim of
basis-invariant optimality. Computationally, a PAN probe requires one clean
blank rollout plus coordinate-ablation rollouts for the recurrent carrier; in
our experiments this probe is run periodically rather than at every optimizer
step.

## 4. Experimental Setup

We compare RNN, GRU, LSTM, SSM, LRU, and PAN under a matched recurrent
scaffold. LRU-specific variants, shuffle controls, auxiliary losses, and
initialization sweeps are kept in the appendix as ablations.

Shared settings:

```text
training steps:        10,000
training horizon:      10-50
evaluation horizons:   50, 100, 200, 500, 1000, 2000
recurrent width:       96
layers:                1
optimizer:             AdamW, lr = 1e-3
lambda initialization: deterministic linspace 0.999 -> 0.90
PAN eta_lambda:        3000 unless stated otherwise
```

Main tasks:

```text
line/plane integration:
  z_t = z_0 + sum_{k<=t} u_k in R^d

ring hold:
  cue theta once, then hold (cos theta, sin theta) under zero input
```

Fairness controls:

```text
eta_lambda = 0:
  disables the PAN damage update while keeping the scaffold fixed

blank-rollout auxiliary loss:
  tests whether extra blank-hold supervision explains the result

shuffled/random damage:
  tests whether coordinate-specific damage alignment matters

dense/all-slow recurrence:
  tests whether making all coordinates slow is sufficient
```

Current tables report completed seeds `0, 1, 2`; the final main tables will be
refreshed with seeds `0, 1, 2, 3, 4`.

## 5. Results

### 5.1 RQ1: Baselines Remember But Fail The Diagnostic Bundle

Standard recurrent baselines can store information, but they do not reliably
produce continuous-attractor geometry. Their failures differ by task and model:
some drift during blank hold, some fail normal recovery, and some fail tangent
consistency.

| condition | baseline | score | normal500 | tangent500 | read |
|---|---|---:|---:|---:|---|
| line d=2 | GRU | `2.313` | `0.189` | `0.193` | remembers, but recovery is weak |
| line d=16 | LRU | `1.745` | `0.290` | `0.290` | high-rank memory remains unstable |
| ring hold | GRU | `48.61` | `5.10` | `5.08` | closed manifold is not stable |
| ring hold | LRU | `377.68` | `99.04` | `98.58` | recurrence alone is insufficient |

Across the completed continuous task grid, the best PAN row beats the best
standard baseline on post-hold stability, normal recovery, and tangent
consistency in every evaluated condition:

| metric | geometric mean PAN / baseline | wins |
|---|---:|---:|
| post | `0.137` | `8 / 8` |
| normal | `0.056` | `8 / 8` |
| tangent | `0.033` | `8 / 8` |

### 5.2 RQ2: PAN Improves Line/Plane Memory And Ring Hold

Line/plane integration is the strongest rank-sweep result. PAN improves the
whole attractor bundle: final error, post-hold stability, normal recovery, and
tangent consistency.

| dim | best PAN | PAN score | best baseline | baseline score | gain | PAN normal | PAN tangent | lambda > .99 |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | eps=1e-4 | `1.061` | GRU | `3.037` | `2.9x` | `0.0031` | `0.0023` | `6.0` |
| 2 | eps=1e-4 | `0.343` | GRU | `2.313` | `6.7x` | `0.0038` | `0.0030` | `9.0` |
| 4 | eps=1e-4 | `0.141` | GRU | `1.961` | `13.9x` | `0.0061` | `0.0047` | `8.7` |
| 8 | eps=3e-5 | `0.093` | GRU | `1.603` | `17.2x` | `0.0063` | `0.0036` | `12.3` |
| 16 | eps=0 | `0.101` | LRU | `1.745` | `17.2x` | `0.0123` | `0.0099` | `18.7` |

Ring hold gives the cleanest closed-manifold result. The model receives one cue
on `S^1`, then must preserve the represented angle under zero input.

| model | n | score | primary | post | normal | tangent | lambda > .99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PAN eps=3e-5 | `3` | `0.543` | `0.061` | `0.061` | `0.356` | `0.065` | `9.7` |
| PAN eps=1e-4 | `3` | `0.838` | `0.143` | `0.143` | `0.392` | `0.160` | `9.7` |
| RNN | `1` | `36.01` | `11.33` | `11.48` | `6.85` | `6.35` | -- |
| GRU | `3` | `48.61` | `17.27` | `21.16` | `5.10` | `5.08` | -- |
| LRU | `3` | `377.68` | `89.91` | `90.16` | `99.04` | `98.58` | `12.0` |

Local Jacobian diagnostics give a complementary view of the same behavior. We
linearize the blank-input map around decoded memory states and measure tangent
gain, normal gain, and tangent leakage. Tangent eigenvalues near one are not
enough by themselves: several baselines also show locally neutral tangent
directions. The useful signal is the bundle of near-neutral tangent dynamics,
low normal random gain, low leakage, and long-horizon recovery.

| condition | model | tangent eig | normal eig max | normal random gain | leak |
|---|---|---:|---:|---:|---:|
| line d=2 | GRU | `1.000` | `0.921` | `0.756` | `0.011` |
| line d=2 | LRU | `0.986` | `0.999` | `1.142` | `0.031` |
| line d=2 | LSTM | `1.001` | `0.652` | `0.442` | `0.008` |
| line d=2 | PAN | `1.003` | `1.000` | `0.238` | `0.008` |
| line d=8 | GRU | `1.000` | `0.946` | `0.884` | `0.003` |
| line d=8 | LRU | `1.003` | `0.999` | `1.523` | `0.017` |
| line d=8 | LSTM | `0.998` | `0.674` | `0.475` | `0.003` |
| line d=8 | PAN | `1.006` | `1.000` | `0.296` | `0.007` |
| ring hold | GRU | `1.000` | `0.999` | `0.860` | `0.001` |
| ring hold | LRU | `0.974` | `0.999` | `1.867` | `0.020` |
| ring hold | LSTM | `0.999` | `0.863` | `0.443` | `0.001` |
| ring hold | PAN | `1.000` | `1.000` | `0.467` | `0.000` |

The full hidden normal eigenvalue can remain near one even when decoded normal
recovery is strong, because hidden states may contain decoder-null or redundant
neutral directions. We therefore use Jacobians as a finite-time diagnostic
alongside decoded perturbation recovery, not as a replacement for it.

### 5.3 RQ3: Damage Alignment, Not Dense Slowness

PAN begins from the same deterministic slow-reservoir initialization used by
the LRU baseline and ablations. Its final support is compact: roughly `6-19`
coordinates with `lambda > .99` across line/plane ranks and about `10` on ring
hold.

With positive `epsilon`, this compact support is produced by large lambda
redistribution. In seed-0 structured traces, thresholded PAN typically reduces
the mean lambda from about `0.95` to `0.10-0.27`, while preserving high-damage
coordinates near `lambda ~= 1`. By contrast, `epsilon = 0` leaves the mean
lambda closer to its initialization and mostly adds a few persistent
coordinates.

The direct control is `eta_lambda = 0`, which disables the damage update:

| condition | PAN score | eta=0 score | eta=0 normal | eta=0 tangent | read |
|---|---:|---:|---:|---:|---|
| line d=2 | `0.343` | `3.230` | `0.279` | `0.285` | damage updates are needed |
| line d=8 | `0.093` | `1.970` | `0.260` | `0.260` | fixed initial support is insufficient |
| line d=16 | `0.101` | `1.677` | `0.274` | `0.274` | update still matters at high rank |
| ring hold | `0.543` | `189.59` | `15.75` | `15.65` | no closed attractor without updates |

Shuffled damage controls are an important caveat. They can sometimes approach
PAN's decoded error, but usually by making nearly the entire reservoir
persistent and with near-zero score-lambda alignment. The defensible mechanism
claim is therefore:

```text
PAN learns compact, damage-aligned persistence, not merely a large slow
reservoir.
```

### 5.4 RQ4: Limitation In Controlled Tangent Flow

Ring integration separates stable attractor memory from controllable tangent
dynamics. PAN stabilizes the ring and gives strong normal/tangent recovery, but
active angular flow remains difficult. In long-hold ring integration, gated
baselines can sometimes learn the driven angular update better, while PAN still
has much stronger perturbation recovery.

This limitation is useful rather than incidental: it shows that forming a
stable continuous memory manifold and learning input-controlled motion along
that manifold are separable problems. The main claim of this paper is about
finite-time attractor memory; controlled tangent flow is a next target.

## 6. Discussion

Long-horizon memory does not imply continuous attractor geometry. The relevant
behavior is selective: stable zero-input persistence, recovery from normal
perturbations, and preservation of nearby valid memories along the tangent
direction.

PAN improves these diagnostics by assigning persistence according to
functional damage. This makes the result different from simply slowing the
whole recurrent reservoir. The method is still basis-dependent because it uses
coordinate ablation, and the results should be read as finite-time
attractor-like behavior rather than exact continuous attractors.

The broader conclusion is that stable attractor memory and controllable
tangent dynamics should be studied separately. PAN provides a simple mechanism
for the former; the latter remains an open direction for ring integration and
other controlled analog-memory tasks.

## References To Cite

Core continuous-attractor references:

```text
Zhang, 1996.
Representation of Spatial Orientation by the Intrinsic Dynamics of the
Head-Direction Cell Ensemble: A Theory.
Journal of Neuroscience.
https://www.jneurosci.org/content/16/6/2112

Seung, 1998.
Learning Continuous Attractors in Recurrent Networks.
NeurIPS.
https://papers.neurips.cc/paper/1369-learning-continuous-attractors-in-recurrent-networks

Ságodi, Martín-Sánchez, Sokół, Park, 2024.
Back to the Continuous Attractor.
NeurIPS.
https://arxiv.org/abs/2408.00109
```

Trained RNN dynamics:

```text
Sussillo and Barak, 2013.
Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional
Recurrent Neural Networks.
Neural Computation.
https://pubmed.ncbi.nlm.nih.gov/23272922/

Maheswaranathan, Williams, Golub, Ganguli, Sussillo, 2019.
Reverse Engineering Recurrent Networks for Sentiment Classification Reveals
Line Attractor Dynamics.
NeurIPS.
https://arxiv.org/abs/1906.10720
```

Modern recurrent and state-space architectures:

```text
Hochreiter and Schmidhuber, 1997.
Long Short-Term Memory.
Neural Computation.
https://pubmed.ncbi.nlm.nih.gov/9377276/

Cho et al., 2014.
Learning Phrase Representations using RNN Encoder-Decoder for Statistical
Machine Translation.
EMNLP.
https://aclanthology.org/D14-1179/

Chung et al., 2014.
Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.
arXiv.
https://arxiv.org/abs/1412.3555

Orvieto et al., 2023.
Resurrecting Recurrent Neural Networks for Long Sequences.
ICML.
https://proceedings.mlr.press/v202/orvieto23a.html

Beck et al., 2024.
xLSTM: Extended Long Short-Term Memory.
arXiv.
https://arxiv.org/abs/2405.04517

Dao and Gu, 2024.
Transformers are SSMs: Generalized Models and Efficient Algorithms Through
Structured State Space Duality.
ICML.
https://proceedings.mlr.press/v235/dao24a.html
```

## Appendix A. Main Text Boundary

Main text focus:

```text
Introduction:
  memory is not attractor geometry

Finite-time diagnostics:
  persistence, normal recovery, tangent consistency, local Jacobian

PAN:
  coordinate damage score, persistence update, cost, basis dependence

Experimental setup:
  tasks, matched models, horizons, fairness controls

Results:
  RQ1 baselines remember but fail the diagnostic bundle
  RQ2 PAN improves line/plane and ring hold
  RQ3 damage alignment, not dense slowness
  RQ4 limitation in controlled tangent flow

Discussion:
  finite-time attractor-like behavior, basis dependence, stable memory vs flow
```

Appendix material:

```text
exact task generators
exact hyperparameters
model definitions
all secondary metrics
all per-task result tables
PCA caveats
lambda initialization sweeps
implementation details such as warmup and numerical theta bounding
negative or mixed results
file paths for figures and analysis artifacts
```

The appendix stores reproducibility details and secondary analyses.

## Appendix B. Model Definitions

Model variants used in the current experiments:

| model | recurrence | setting |
|---|---|---|
| `RNN` | vanilla recurrent update | direct sequence baseline |
| `GRU` | gated recurrent update | direct sequence baseline |
| `LSTM` | gated recurrent update with cell state | direct sequence baseline |
| `SSM` | diagonal state-space recurrence | direct sequence baseline |
| `LRU` | complex-diagonal LRU | direct sequence baseline |
| `PAN` | PAN real-diagonal recurrence | main proposed model |
| `PAN-unit` | PAN recurrence | recurrent unit diagnostic |

LRU ablations used only for mechanism checks:

| ablation | purpose |
|---|---|
| legacy LRU wrapper | checks whether the wrapper correction explains the result |
| rank-matched LRU support | checks whether exactly `d` slow coordinates are enough |
| dense-persistence LRU | checks whether making nearly all coordinates slow is enough |

## Appendix C. Full-Block Scaffold

The shared full-block sequence model has the form:

```text
x_{1:T} -> encoder -> recurrent block(s) -> output head.
```

The block wrapper is:

```text
u_t = input normalization / projection of x_t
h_t = Rec(h_{t-1}, u_t)
r_t = output projection of h_t
g_t = GLU(GELU(r_t))
x_t^{next block} = x_t + g_t
```

The corrected zero-drive wrapper sets:

```text
encoder_bias = False
use_norm_in  = False
```

for `LRU`, the rank-matched LRU ablation, and `PAN`. The reason is simple:
during blank hold, an all-zero task input should remain an all-zero recurrent
drive. Without this correction, blank hold still injects a learned encoder bias
or a layer-normalized constant, which changes the dynamical question.

Current shared architectural settings:

| setting | value |
|---|---:|
| `d_model` | `96` |
| `rec_dim` | `96` |
| layers | `1` |
| dropout | `0.0` |
| optimizer | AdamW |
| learning rate | `1e-3` |
| weight decay | `1e-5` |
| gradient clip | `1.0` |
| train steps | `10,000` |
| batch size | `256` |
| eval batch | `512` structured, `512` discrete |

## Appendix D. PAN Implementation Details

The conceptual PAN equation is defined by a persistent carrier `m_t`, a maximum
retention `lambda`, and a damage update:

```text
lambda_j  = sqrt(sigmoid(theta_j))

E_mem(m; z) = ||D(F_0^H(m)) - z||^2
s_j         = E_mem(A_j m; z) - E_mem(m; z)
theta_j    <- theta_j + eta_lambda (s_j - epsilon)
```

Carrier choices:

| model | carrier | retention coefficient |
|---|---|---|
| `PAN` | recurrent state `h_t` | `alpha_t = lambda` |

Current implementation details:

| detail | value / behavior |
|---|---|
| `theta.requires_grad` | `False` for PAN |
| task gradient on `theta` | off |
| PAN score update | outside autograd |
| `pan_eta_lambda` | main value `3000`; appendix sweeps use nearby values |
| `pan_score_eps` | current grid: `0`, `1e-5`, `3e-5`, `1e-4`, `3e-4` |
| `pan_probe_every` | every `100` optimizer steps |
| `pan_probe_batch` | `96` |
| `pan_probe_horizon` | `50` |
| `pan_h_probe` | `500` |
| `pan_warmup_frac` | `0.3` in current scripts |
| lambda init | linspace `0.999 -> 0.90` unless an init sweep changes it |

Implementation details:

1. The final paper formula excludes EMA, score clipping, and adaptive
   loss-coefficient schedules. Those were discussion-stage ideas.
2. The code currently bounds `theta` numerically to `[-18, 18]` after a PAN
   update. This is a numerical parameter bound rather than a damage-score
   clipping mechanism.

The PAN score uses a coordinate ablation:

```text
A_j m = m with coordinate j set to zero.
```

For each coordinate, the model compares decoded memory error after `H` blank
steps from the ablated state against the clean state. A positive score means the
coordinate is functionally useful for long-horizon memory. Thresholded PAN asks
whether that usefulness exceeds `epsilon`.

The current PAN ablations use line/plane integration, ring hold, and ring
integration with a post-integration hold. The active grid keeps the
architecture fixed and sweeps the PAN update strength and threshold:

```text
PAN:
  eta {1000, 3000, 10000}, eps {0, 1e-5, 3e-5, 1e-4, 3e-4}

controls:
  eta = 0
  shuffled damage
  random damage
  all-slow lambda
  auxiliary blank-rollout loss
```

Control status:

| control | status | paper use | current read |
|---|---|---|---|
| `eta_lambda = 0` | completed | main text | disables damage update; much worse than PAN |
| shuffled damage | completed | appendix, mechanism caveat | often dense/all-slow; weak score-lambda alignment |
| random damage | completed | appendix | clearly worse than PAN |
| dense/all-slow LRU | completed | appendix | tests whether large lambda alone explains the result |
| blank-rollout auxiliary loss | completed | appendix | extra blank-hold supervision does not reproduce PAN |
| hidden perturbation augmentation | not found as a completed training control | optional future control | perturbations are evaluated, but not yet used as a training augmentation baseline |

Representative appendix-control results:

| control | condition | score | normal | tangent | lambda > .99 | read |
|---|---|---:|---:|---:|---:|---|
| blank-rollout auxiliary | line d=2 | `2.954` | `0.222` | `0.224` | `96.0` | worse than PAN `0.343` |
| blank-rollout auxiliary | line d=8 | `1.633` | `0.178` | `0.178` | `96.0` | worse than PAN `0.093` |
| blank-rollout auxiliary | ring hold | `133.96` | `11.12` | `11.36` | -- | does not produce ring attractor |
| shuffled damage | line d=2 | `0.525` | `0.0028` | `0.0022` | `95.3` | close error, dense support |
| shuffled damage | ring hold | `0.787` | `0.354` | `0.147` | `96.0` | close error, dense support |
| random damage | ring hold | `1.399` | `0.441` | `0.326` | `89.3` | worse and still dense |

The important interpretation is not that every shuffle row fails
catastrophically. Some shuffle controls are competitive in decoded error, but
they lose the mechanism: PAN achieves comparable or better attractor behavior
with compact persistent support aligned with functional damage, while shuffled
or all-slow controls usually use nearly the whole recurrent reservoir.

## Appendix E. Lambda Initialization

The current main setting initializes slow coordinates with:

```text
lambda_j = linspace(0.999, 0.90)
```

This was adopted because PAN needs a reservoir of candidate slow coordinates.
Current summary:

```text
PAN can start from several lambda distributions,
provided that enough slow candidates exist early in training.
```

Initialization conditions explored:

| init family | description | current read |
|---|---|---|
| `linspace 0.999 -> 0.90` | deterministic slow reservoir | main setting |
| `random/uniform broad` | broad lambda support | can work if enough slow candidates exist |
| `gaussian near 0.9` | slow-biased Gaussian | can work |
| `bimodal` | mixture of fast and slow coordinates | can work |
| `gaussian near 0.5` | mostly non-slow reservoir | weak/failing condition |
| `fixed 0.99` | all coordinates slow | useful diagnostic, less selective |

Relevant files:

```text
exp71_rank2_init_sweep_lambda_dynamics_summary.csv
exp71_rank2_init_sweep_lambda_dynamics_summary_distribution_over_steps.csv
exp71_rank2_init_sweep_lambda_dynamics_summary_coordinate_initial_final.csv
exp71_rank2_init_sweep_lambda_dynamics_summary_coordinate_snapshots.csv
exp71_rank2_pan_eps_sweep_lambda_dynamics_summary.csv
figures_exp71_rank2_init_dist_sweep_10k/rank2_init_distribution_sweep_summary.png
figures_exp71_rank2_lambda_diagnostics/rank2_lambda_init_vs_final_distributions.png
figures_exp71_rank2_lambda_diagnostics/rank2_lambda_init_vs_final_sorted_spectra.png
```

Current interpretation:

```text
lambda movement depends strongly on epsilon.

epsilon > 0:
  most low-damage coordinates are actively drained.
  examples from rank-2 epsilon sweep:
    eps=1e-4, lin090: mean lambda 0.949 -> 0.073, lambda>.99 9 -> 7
    eps=3e-5, lin090: mean lambda 0.949 -> 0.206, lambda>.99 9 -> 7
    eps=1e-3, lin090: mean lambda 0.949 -> 0.042, lambda>.99 9 -> 4

epsilon = 0:
  lambda movement is much smaller.
  structured traces keep mean lambda close to initialization and mainly add
  a few coordinates above .99.

The strongest evidence is therefore selective lambda redistribution:
thresholded PAN drains most coordinates while preserving coordinates whose
ablation damages long-horizon memory.
```

This motivates a final appendix figure comparing:

```text
initial lambda distribution
final lambda distribution
damage score distribution
lambda-damage scatter
```

## Appendix F. Exact Task Definitions

### F.1 Line / Plane Integration

Task:

```text
z_0 ~ Uniform([-z_scale, z_scale]^d)
u_t ~ Uniform([-vel_scale / sqrt(d), vel_scale / sqrt(d)]^d)
z_t = z_0 + sum_{k<=t} u_k
```

Input and output:

```text
input_dim  = 2d + 1
output_dim = d

t = 0:
  x_0[:d]  = z_0
  cue bit  = 1

t > 0:
  x_t[d:2d] = u_t
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

### F.2 Ring Hold

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

### F.3 Ring Integration

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

Default values:

| setting | value |
|---|---:|
| `omega_scale` | `0.18` |
| `omega_hold_prob` | `0.25` |
| train horizon | random `10..50` |
| eval horizons | `50, 100, 200, 500, 1000, 2000` |
| post-hold | `500` blank steps |

The `cos omega_t - 1` term keeps blank updates exactly zero when `omega_t = 0`.
This replaced the earlier scalar-only ring integration encoding.

### F.4 K-Way Point Attractor Hold

Task:

```text
K = 16 in the current main run
label y ~ Uniform({0, ..., K-1})
```

Input and output:

```text
input_dim  = K + 1
output_dim = K

t = 0:
  one-hot class cue
  cue bit = 1

t > 0:
  x_t = 0

target:
  class label y at all timesteps
```

Loss:

```text
cross entropy over K classes
```

Attractor diagnostics:

```text
prototype residual:
  ||F_0(h_y) - h_y||

between prototype separation:
  pairwise distances among class prototype states

recovery:
  decode after hidden perturbation and blank rollout

basin radius:
  accuracy after perturbing prototypes at several radii and rolling blank
```

### F.5 N-Bit Flip-Flop

Task:

```text
N = 4 in the current main run
state bits start randomly
at each step each bit updates with probability 0.08
update pulse is +1 or -1 for the new bit value
zero pulse means keep previous bit
```

Input and output:

```text
input_dim  = N
output_dim = N
```

Loss:

```text
binary cross entropy with logits
```

Attractor diagnostics:

```text
hold accuracy:
  full bit-vector accuracy over long horizon

recovery:
  full bit-vector accuracy after hidden perturbation and blank rollout

transition accuracy:
  apply one bit flip pulse from a prototype, then decode immediately and after blank hold
```

This task tests controlled switching between discrete attractor states. It is
therefore stricter than K-way hold.

## Appendix G. Metric Definitions

### G.1 Long-Horizon Error

For vector tasks:

```text
RMSE_T = sqrt(mean(||D(h_T) - z_T||^2 per coordinate)).
```

For ring tasks:

```text
angle_error = mean absolute wrapped angular error in degrees.
radius_rmse = sqrt(mean((||D(h)||_2 - 1)^2)).
```

For discrete tasks:

```text
K-way: class accuracy.
Flip-flop: bit accuracy and full bit-vector accuracy.
```

### G.2 Post-Hold

After an active trajectory reaches its final state, the model is rolled forward
for `500` blank steps:

```text
h_post = F_0^500(h_T)
```

The decoded error from `h_post` to the same final target measures whether the
state is an attracting or drifting memory after input is removed.

### G.3 Normal Recovery

A local tangent basis is estimated by finite differences around the memory
manifold. A random hidden perturbation is projected into the orthogonal
complement:

```text
n = random hidden noise
tangent = projection of n onto local tangent basis
normal = normalize(n - tangent)
h_pert = h + radius * normal
```

The model then rolls blank for `R` steps. The main reported value is decoded
memory error after `R = 500`.

### G.4 Tangent Consistency

A tangent perturbation creates a nearby valid memory:

```text
line:
  z_0 -> z_0 + delta

ring:
  theta -> theta + delta
```

The model should preserve the new nearby memory after blank rollout. This
separates continuous attractors from point attractors: the correct behavior is
to keep nearby memories distinct while avoiding collapse into the original
memory.

### G.5 Latent Participation Ratio

The latent participation ratio estimates effective dimensionality:

```text
PR = (sum_i sigma_i^2)^2 / sum_i sigma_i^4
```

where `sigma_i` are singular values or PCA spectrum terms of the hidden states.
For ring hold, a good latent PR near `2` is expected because the circle occupies
a two-dimensional embedding while having one intrinsic angular coordinate.

### G.6 Lambda Counts

Reported lambda diagnostics:

```text
lambda_sum
lambda_max
lambda_gt_0p9
lambda_gt_0p95
lambda_gt_0p99
```

Interpretation:

```text
lambda count is a reservoir/support diagnostic. The main attractor metrics are
persistence, recovery, tangent consistency, and basin behavior.
High lambda count alone is insufficient; dense-persistence LRU ablations can
keep many lambdas near 1 without matching PAN's attractor recovery.
```

## Appendix H. Full Result Tables

These tables preserve detailed and older one-off result reads. The main text
uses the seed-averaged Priority 1-4 summaries where available; Appendix H is
for reproducibility, controls, and historical comparison rather than the main
claim.

### H.1 Line Integration

| dim | best_pan | pan_T2000 | pan_postH500 | pan_normal_R500 | pan_tangent_R500 | pan_latent_pr | pan_lambda99 | best_non_pan | non_pan_T2000 | eta0_T2000 |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | PAN eps=1e-4 | `0.6212` | `0.6208` | `0.0014` | `0.0002` | `1.512` | `7` | LRU rank-support ablation | `1.264` | `1.479` |
| 2 | PAN eps=1e-4 | `0.1837` | `0.1823` | `0.0064` | `0.0062` | `2.642` | `10` | gru | `0.8857` | `1.257` |
| 4 | PAN eps=1e-4 | `0.0698` | `0.0665` | `0.0078` | `0.0063` | `5.944` | `9` | LRU rank-support ablation | `0.6625` | `0.8699` |
| 8 | PAN eps=3e-5 | `0.0456` | `0.0424` | `0.0061` | `0.0031` | `10.72` | `12` | gru | `0.4839` | `0.6961` |
| 16 | PAN eps=0 | `0.0373` | `0.0361` | `0.0107` | `0.0079` | `16.41` | `19` | LRU rank-support ablation | `0.4056` | `0.5526` |

Read:

```text
PAN wins the continuous vector attractor bundle across tested ranks.
eta=0 is consistently worse.
epsilon helps for d=1..8, while d=16 currently prefers epsilon=0.
```

### H.2 Ring Hold

| tag | angle2000 | post_angle | normal500_angle | tangent500_angle | latent_pr | lambda_gt_0p99 | lambda_sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| PAN eps=3e-5 | `0.0693` | `0.0693` | `0.3366` | `0.0695` | `1.996` | `10` | `21.50` |
| PAN eps=1e-4 | `0.0789` | `0.0789` | `0.3021` | `0.0794` | `1.994` | `10` | `10.01` |
| PAN eps=0 | `1.121` | `1.121` | `1.223` | `1.220` | `2.002` | `11` | `90.23` |
| LRU rank-support ablation | `5.811` | `10.06` | `0.8488` | `0.8651` | `1.997` | `2` | `2.007` |
| gru | `9.144` | `11.93` | `1.601` | `1.605` | `1.994` | -- | -- |
| rnn | `11.33` | `11.48` | `6.847` | `6.348` | `2.318` | -- | -- |
| dense-persistence LRU ablation | `28.80` | `46.11` | `7.446` | `7.715` | `1.956` | `96` | `95.98` |
| lstm | `30.50` | `36.30` | `10.32` | `9.849` | `2.024` | -- | -- |
| LRU | `88.76` | `89.54` | `126.55` | `126.79` | `1.988` | `11` | `91.22` |

Read:

```text
Thresholded PAN gives near-zero drift and strong radial recovery.
Dense-persistence LRU has almost all lambdas high but much weaker hold.
Rank-support LRU has compact support but worse angular persistence.
```

### H.3 Ring Integration, Active Setting

| tag | angle2000 | post_angle | normal500_angle | tangent500_angle | latent_pr | lambda_gt_0p99 |
|---|---:|---:|---:|---:|---:|---:|
| gru | `31.14` | `37.68` | `15.33` | `15.30` | `3.517` | -- |
| PAN eps=1e-4 | `49.09` | `49.10` | `0.4488` | `0.3414` | `3.103` | `10` |
| PAN eps=3e-5 | `49.50` | `49.51` | `0.3748` | `0.2607` | `3.111` | `9` |
| PAN eps=0 | `51.84` | `51.76` | `0.8923` | `0.8850` | `3.168` | `14` |
| lstm | `52.14` | `63.68` | `19.90` | `19.94` | `2.433` | -- |
| LRU rank-support ablation | `86.38` | `86.35` | `46.51` | `45.37` | `2.863` | `2` |
| ssm | `89.38` | `88.45` | `82.67` | `83.90` | `1.539` | `2` |
| LRU | `90.51` | `88.30` | `129.4` | `129.2` | `3.416` | `10` |
| dense-persistence LRU ablation | `94.23` | `93.61` | `13.96` | `13.98` | `2.770` | `96` |
| PAN eta=0 | `91.34` | `90.20` | `47.92` | `47.91` | `2.885` | `9` |

Read:

```text
GRU has the best active angular integration.
PAN has the strongest perturbation recovery.
eta=0 strongly degrades recovery, so the PAN update matters.
```

### H.4 Ring Integration, Long-Hold Training Setting

| tag | angle2000 | post_angle | normal500_angle | tangent500_angle | latent_pr | lambda_gt_0p99 |
|---|---:|---:|---:|---:|---:|---:|
| lstm | `32.53` | `37.39` | `12.43` | `12.52` | `2.647` | -- |
| PAN eps=3e-5 | `44.83` | `44.83` | `0.3803` | `0.1899` | `2.657` | `8` |
| PAN eps=1e-4 | `44.98` | `44.94` | `0.4049` | `0.1312` | `2.619` | `7` |
| PAN eps=0 | `46.11` | `45.88` | `0.3634` | `0.1706` | `2.698` | `10` |
| gru | `64.86` | `65.70` | `4.890` | `4.868` | `3.131` | -- |
| PAN eta=0 | `82.89` | `86.35` | `16.39` | `16.66` | `2.885` | `9` |
| dense-persistence LRU ablation | `83.94` | `85.08` | `16.75` | `16.75` | `2.111` | `96` |
| LRU rank-support ablation | `87.38` | `87.19` | `22.04` | `24.41` | `3.109` | `2` |

Read:

```text
Long-hold training improves PAN's angular error while LSTM remains better on
active angular integration.
PAN still dominates recovery.
```

### H.5 K-Way Point Attractor Hold

| tag | H2000_acc | postH500_acc | recovery_R500_acc | basin_r0p5_acc | between_proto_min | prototype_residual | latent_pr | lambda_gt_0p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rnn | `1.000` | `1.000` | `1.000` | `1.000` | `6.380` | `0.0000` | `3.747` | -- |
| lstm | `1.000` | `1.000` | `1.000` | `1.000` | `119.3` | `3.458` | `4.400` | -- |
| PAN eps=3e-5 | `1.000` | `1.000` | `1.000` | `1.000` | `11.74` | `0.0000` | `12.79` | `14` |
| PAN eps=1e-4 | `1.000` | `1.000` | `1.000` | `1.000` | `10.89` | `0.0135` | `11.46` | `13` |
| PAN eps=0 | `1.000` | `1.000` | `1.000` | `1.000` | `12.64` | `0.0851` | `11.82` | `15` |
| dense-persistence LRU ablation | `0.1113` | `0.9180` | `0.9375` | `0.9375` | `17.55` | `0.9878` | `12.05` | `96` |
| gru | `0.5977` | `0.7188` | `0.6719` | `0.6562` | `5.907` | `0.0153` | `4.843` | -- |
| PAN eta=0 | `0.1270` | `0.5508` | `0.5156` | `0.4922` | `12.57` | `0.0854` | `12.34` | `9` |
| LRU | `0.0488` | `0.0859` | `0.0625` | `0.0586` | `5.396` | `1.126` | `5.832` | `9` |
| ssm | `0.0684` | `0.0781` | `0.0625` | `0.0625` | `0.7108` | `0.0253` | `11.38` | `0` |
| legacy LRU wrapper | `0.0742` | `0.0547` | `0.0625` | `0.0625` | `12.17` | `5.244` | `8.233` | `6` |

Read:

```text
PAN, RNN, and LSTM all show robust point-attractor behavior.
This is a good attractor diagnostic and shows that point attractors are shared
across several recurrent architectures.
The dense-persistence LRU ablation has strong basin recovery despite poor H2000,
which needs careful analysis.
```

### H.6 N-Bit Flip-Flop

| tag | H2000_full_acc | postH500_full_acc | recovery_R500_full_acc | basin_r0p5_full_acc | transition_after_full_acc | between_proto_min | prototype_residual | latent_pr | lambda_gt_0p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rnn | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `3.148` | `0.0000` | `3.969` | -- |
| gru | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `3.204` | `0.0000` | `3.991` | -- |
| lstm | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `3.949` | `0.0000` | `3.953` | -- |
| PAN eta=0 | `0.4688` | `0.1406` | `0.6875` | `0.6875` | `0.2656` | `7.088` | `0.1825` | `4.347` | `9` |
| ssm | `0.8730` | `0.0508` | `0.0625` | `0.0625` | `0.1875` | `0.0623` | `0.0053` | `3.484` | `0` |
| dense-persistence LRU ablation | `0.3770` | `0.0859` | `0.0625` | `0.0625` | `0.3750` | `8.473` | `1.066` | `3.932` | `91` |
| legacy LRU wrapper | `0.8574` | `0.0352` | `0.0625` | `0.0625` | `0.1250` | `2.487` | `2.772` | `3.903` | `6` |
| PAN eps=3e-5 | `0.2324` | `0.0781` | `0.0625` | `0.0625` | `0.0625` | `0.0000` | `0.0000` | `0.0000` | `0` |
| PAN eps=1e-4 | `0.1016` | `0.0781` | `0.0625` | `0.0625` | `0.0625` | `0.0000` | `0.0000` | `0.0000` | `0` |
| PAN eps=0 | `0.7695` | `0.0703` | `0.0625` | `0.0625` | `0.3281` | `2.460` | `0.1832` | `3.669` | `0` |
| LRU | `0.8281` | `0.0391` | `0.0000` | `0.0000` | `0.1719` | `2.304` | `3.860` | `2.618` | `7` |

Read:

```text
RNN, GRU, and LSTM solve controlled discrete switching.
Current PAN variants do not.
This is the current controlled-switching limitation.
```

## Appendix I. Figure Inventory

Current paper-relevant figures:

| purpose | file |
|---|---|
| main normal recovery comparison | `figures_attractor_recovery/attractor_normal_recovery_main.png` |
| LRU-family normal recovery comparison | `figures_attractor_recovery/attractor_normal_recovery_lru_family.png` |
| ring hold radial recovery | `figures_attractor_recovery/ring_hold_radial_recovery_over_time.png` |
| line hidden-normal recovery | `figures_attractor_recovery/line_hold_hidden_normal_recovery_over_time.png` |
| line in-plane/tangent persistence | `figures_attractor_recovery/line_hold_in_plane_shift_persistence_over_time.png` |
| K-way shared displayed R0 recovery | `figures_attractor_recovery/discrete_kway_shared_random_anchor_same_r0_recovery.png` |

Interpretation notes:

```text
Ring radial recovery:
  shows movement in the decoded ring plane.
  PAN returns toward the ring after radial perturbation.

Line hidden-normal recovery:
  shows contraction from an off-manifold hidden perturbation.
  PAN recovers cleanly.

Line in-plane shift:
  this is tangent, so correct behavior is to preserve the shifted memory.
  returning to the original point would indicate point-attractor collapse.

K-way shared displayed R0:
  aligned visualization with shared random anchors.
  useful for comparing recovery behavior as an aligned visualization.
```

## Appendix J. PCA Caveats

PCA is useful for intuition but fragile for integration tasks.

Observed issues:

```text
1. If unordered batch samples are connected as a trajectory, PCA plots can look wrong.
2. If each sample has a different velocity path, the resulting cloud mixes initial state and input history.
3. For line/ring integrate, the right probe is a controlled grid or shared velocity rollout.
4. For discrete tasks, model-specific latent PCA coordinates require alignment before cross-model comparison.
```

Corrections already made:

```text
line integrate:
  use shared-velocity decoded-grid probes.

ring integrate:
  inspect decoded angle/radius and velocity-to-angle behavior rather than PCA alone.

K-way:
  use shared random anchors or same displayed R0 for cross-model visualization.
```

## Appendix K. Current Analysis Artifacts

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
analysis_attractor_properties/kway_point_attractor_summary.csv
analysis_attractor_properties/flipflop_attractor_summary.csv
analysis_jacobian_diagnostics/jacobian_diagnostics_3seed.csv
analysis_jacobian_diagnostics/JACOBIAN_DIAGNOSTICS_3seed.md
```

Analysis scripts:

```text
analyze_attractor_properties.py
analyze_jacobian_diagnostics.py
plot_discrete_shared_perturbation.py
plot_attractor_recovery_manifold_figure.py
plot_exp74_discrete_perturbation.py
plot_exp74_discrete_recovery_pca.py
plot_exp74_discrete_pca.py
```

Structured task scripts:

```text
exp72_structured_attractor_tasks.py
exp74_discrete_attractor_tasks.py
pan_block.py
exp71_pan_block_pulse_hold.py
```

## Appendix L. What The Current Results Say

Strongest evidence:

```text
1. PAN improves continuous vector attractors across d = 1, 2, 4, 8, 16.
2. PAN gives near-perfect ring hold and radial recovery.
3. PAN's recovery advantage survives comparison to LRU ablations.
4. eta_lambda = 0 is much weaker, so the damage update is functionally active.
```

Careful interpretations:

```text
1. High lambda count alone is insufficient.
2. Rank-matched slow support alone is insufficient.
3. Lambda dynamics depend on epsilon: epsilon > 0 produces large redistribution
   and drains low-damage coordinates, while epsilon = 0 moves much less.
4. The strongest evidence is selective lambda redistribution aligned with
   functional damage, not dense slow recurrence.
5. Ring integration shows that stable attractors and input-driven tangent flow are separable.
6. Discrete point attractors and controlled switching are appendix boundary
   checks, not main evidence for the continuous-attractor claim.
```

Current paper-level statement:

```text
PAN promotes persistent attractor geometry in recurrent state spaces.
The strongest support is continuous memory, closed-manifold hold, and perturbation recovery.
The next open question is controllable tangent flow.
```

## Appendix M. Immediate Next Experiments

Required seed extension:

```text
current main result uses seeds 0, 1, 2
final main result should use seeds 0, 1, 2, 3, 4
run only the main paper rows first:
  line/plane integration rank sweep
  ring hold
  eta_lambda = 0 control
  strongest standard baselines
ring integration and appendix controls can remain 3-seed unless needed
```

Jacobian diagnostic:

```text
script:
  analyze_jacobian_diagnostics.py

current status:
  completed first-pass 3-seed analysis on existing checkpoints
  rerun after seeds 3 and 4 finish to refresh the final 5-seed table

primary conditions:
  line_integrate d=2
  line_integrate d=8
  ring_hold d=1

models:
  RNN, GRU, LSTM, SSM, LRU, PAN

reported metrics:
  tangent_eig_abs_mean and tangent_sv_mean near 1
  normal_eig_abs_max and normal_random_gain as contraction checks
  tangent_leak_ratio small
```

PAN-specific ablations:

```text
epsilon scaling with dimension or score magnitude
eta_lambda sweep:
  PAN {1000, 3000, 10000}
PAN update only during hold phase vs all phases
PAN update delayed until after active input training
shuffle damage scores across coordinates before theta update
```

Ring integration diagnostics:

```text
predicted angular displacement vs true cumulative omega
local tangent vector field: omega -> decoded angular step
radius stability during active integration
zero-omega hold inside active trajectories
longer train horizons
train-hold suffix length sweep
```

Discrete attractor follow-up:

```text
K = 16, 32 point attractors with shared perturbation plots
prototype fixed-point finder for K-way point attractors
basin-radius curves across multiple radii
```

Later boundary check:

```text
controlled switching with separated write and hold phases
PAN update only during hold segments
```

Figure work:

```text
one main multi-panel figure for attractor diagnostics
one lambda/damage alignment figure
one ring hold radial recovery figure
one ring integration stability-vs-flow figure
appendix PCA caveat figures
```
