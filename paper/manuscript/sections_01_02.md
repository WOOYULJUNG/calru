# Learning Approximate Attractor Dynamics for Recurrent Memory

## Abstract

*To be written after Sections 3–6 are finalized.*

## 1. Introduction

Many sequential tasks require recurrent models to retain and update continuous-valued information during intervals in which the relevant input is absent. Continuous attractors provide a canonical dynamical account of this capability: an analog value is represented by a continuum of recurrent states, with contraction toward a low-dimensional memory manifold and neutral dynamics along it [Seung 1996; Khona & Fiete 2022]. This organization supports persistent storage, error correction, and input-driven integration [Khona & Fiete 2022]. Population dynamics consistent with this organization have been reported in head-direction and grid-cell systems [Kim et al. 2017; Chaudhuri et al. 2019; Gardner et al. 2022].

Translating this account into a trained artificial recurrent model remains difficult. Classical continuous-attractor networks prescribe the recurrent connectivity required to realize a chosen manifold [Zhang 1996; Burak & Fiete 2009]. Task-trained RNNs can instead develop low-dimensional representations, while fixed- and slow-point analyses can reveal approximate attractor dynamics after training [Cueva & Wei 2018; Sussillo & Barak 2013; Maheswaranathan et al. 2019]. However, neither representation geometry nor low task error guarantees stable analog memory. Exact continuous attractors are also structurally fragile, motivating approximate continuous attractors whose residual motion along an attracting slow manifold remains small over the relevant time horizon [Ságodi et al. 2024]. The open problem is therefore to make such dynamics arise reliably from task training and to determine whether a trained model has realized them.

Task accuracy alone cannot provide this determination. Over a finite evaluation horizon, a model may retain enough information for a correct readout even though its hidden state drifts after the input is removed, fails to return after a perturbation, or moves away from the memory manifold while integrating new inputs [Brownell 2026]. These behaviors can produce similar end-of-sequence errors while corresponding to fundamentally different recurrent dynamics. A trained model should therefore be evaluated directly for retention without input, recovery from perturbations, and input-driven movement along the learned manifold.

We address this problem with CA-LRU, an LRU-based recurrent architecture [Orvieto et al. 2023], and Retention Plasticity, which adapts retention according to the functional contribution of individual state coordinates. Our main contributions are:

- We formulate five direct tests of whether a trained recurrent model realizes an approximate continuous attractor, covering retention without input, recovery after perturbation, and movement along the learned manifold.
- We introduce CA-LRU and Retention Plasticity, which selectively allocate long retention to state coordinates that are necessary for memory while leaving the target manifold to be learned from data.
- Across ring, torus, closed-curve, and bounded-surface tasks, CA-LRU is the only evaluated model to satisfy all five tests. Relative to the strongest recurrent baseline, it reduces drift during blank input by one to two orders of magnitude, generalizes to longer delays and stronger perturbations than those used in training, and assigns long retention to only 3–9 of 96 state coordinates.

## 2. Related Work

### 2.1 Continuous and Approximate Attractors

Continuous-attractor networks represent a variable on a continuum of recurrent states. Classical constructions use structured connectivity to realize line or ring attractors for analog storage [Seung 1996; Zhang 1996] and toroidal dynamics for grid-cell path integration [Burak & Fiete 2009]. Corresponding low-dimensional population structure has been reported in head-direction and grid-cell systems [Kim et al. 2017; Chaudhuri et al. 2019; Gardner et al. 2022; Khona & Fiete 2022]. Theoretical extensions consider memories that move along a manifold and recurrent circuits that support multiple manifolds [Spalla et al. 2021; Cueva et al. 2021].

Exact continuous attractors require finely balanced dynamics. Ságodi et al. [2024] show that perturbations may instead leave an attracting slow manifold whose finite-time memory remains close to that of the ideal system. We adopt this approximate dynamical perspective, but ask how such behavior can be learned in an artificial recurrent architecture rather than created by prescribing the recurrent connectivity.

### 2.2 Recurrent Dynamics Learned from Tasks

RNNs trained for spatial localization, integration, and continuous working memory can develop grid-like or other low-dimensional representations [Cueva & Wei 2018; Banino et al. 2018; Cueva et al. 2021]. Fixed- and slow-point analyses provide a complementary way to characterize recurrent computations after training [Sussillo & Barak 2013; Maheswaranathan et al. 2019]. Learned representations of continuous variables also need not exhibit the symmetry of classical ring models [Darshan & Rivkind 2022].

However, a structured representation does not establish the autonomous dynamics required for stable memory. Park et al. [2023] show that long-lived working-memory computations may arise without an exact continuous attractor, while Brownell [2026] shows that short-horizon performance can coexist with recurrent states that do not persist in longer free runs. Our work therefore evaluates retention, attraction, and input-driven motion directly rather than inferring attractor dynamics from representation geometry or readout accuracy.

### 2.3 Recurrent Architectures for Long-Range Memory

Long-range sequence modeling has been addressed with gated RNNs [Hochreiter & Schmidhuber 1997; Cho et al. 2014] and linear state-space or recurrent layers [Gu, Goel, & Ré 2022; Smith et al. 2023; Orvieto et al. 2023]. LRU uses a diagonal linear recurrence to combine parallel training with efficient recurrent inference [Orvieto et al. 2023], while RG-LRU adds input-dependent gating within a related recurrent structure [De et al. 2024]. Selective state-space models similarly modulate state updates according to the input [Gu & Dao 2023; Dao & Gu 2024]. These architectures target sequence performance but do not explicitly ensure or test attracting manifolds for analog memory.

A separate line encodes attractor structure in the architecture. Hopfield networks use fixed-point energy minima [Hopfield 1982; Ramsauer et al. 2020], while low-rank recurrent models impose low-dimensional dynamics through structured connectivity [Mastrogiuseppe & Ostojic 2018; Dubreuil et al. 2022]. CA-LRU lies between these approaches: it retains the efficient LRU scaffold while learning which state coordinates require long retention, without predefining the geometry of the memory manifold.

## 3. Evaluating Approximate Attractor Dynamics

## 4. CA-LRU and Retention Plasticity

## 5. Experimental Setup

## 6. Results

### 6.1 Task Accuracy Does Not Imply Attractor Dynamics

### 6.2 CA-LRU Satisfies All Five Tests

### 6.3 Input-Driven Motion Remains on the Learned Manifold

### 6.4 Ablation Studies

### 6.5 Generalization Beyond the Training Regime

### 6.6 Retention Scales with Functional Need

## 7. Scope and Limitations

## 8. Conclusion

## Appendix