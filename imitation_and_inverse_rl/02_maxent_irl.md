# Maximum Entropy Inverse RL

## The Ill-Posedness, First

Inverse RL: given demonstrations, recover the reward the demonstrator
optimizes. Two degeneracies make the naive question unanswerable:

```text
1. r = 0 rationalizes everything (every policy is optimal for a
   constant reward) — and so does any reward constant on the reachable
   set;
2. entire equivalence classes of rewards induce the SAME optimal
   policies (potential-based shaping r' = r + gamma Phi(s') - Phi(s)
   preserves optimal policies exactly — Ng–Harada–Russell), so even
   perfect demonstrations cannot separate them.
```

Compare `rlhf_mathematics/01`: preferences identified rewards up to
per-prompt shifts; demonstrations identify strictly less (only through
the argmax, not through calibrated differences). Any IRL method is
therefore a **choice of selection principle** over the feasible set —
the honest framing of everything below.

## Feature Matching And The Selection Principle

Assume `r = w^T phi(s,a)` (the SF decomposition,
`successor_representations/02`). A policy's behavior is summarized, for
this reward class, by its **feature expectations**
`mu_pi = E_pi[sum_t gamma^t phi_t]` — and matching the expert's is
necessary and sufficient for matching performance under every `w`
(Abbeel–Ng 2004):

```math
\mu_\pi=\mu_{E}
\quad\Longrightarrow\quad
w^\top\mu_\pi=w^\top\mu_E\ \ \forall w .
```

But many distributions over trajectories match `mu_E`. **MaxEnt IRL
(Ziebart et al. 2008)** selects the one with maximum entropy — the
least-committed distribution consistent with the data:

```math
\max_{p\in\Delta(\text{paths})}\ \ \mathcal{H}(p)
\qquad\text{s.t.}\quad
\mathbb{E}_{p}\big[\phi(\tau)\big]=\mu_E\ \ (+\text{dynamics consistency}) .
```

## The Solution Is An Exponential Tilt

Lagrangian duality (multiplier `w` on the matching constraint) gives the
maximizer in closed form:

```math
p_w(\tau)\ \propto\ \exp\big(w^\top\phi(\tau)\big)\ \times\ \prod_t P(s_{t+1}\mid s_t,a_t)
\qquad\text{(deterministic dynamics: }p_w(\tau)\propto e^{w^\top\phi(\tau)}\text{)},
```

— trajectories weighted exponentially by their reward: **the sixth
appearance of the exponential-tilt/Legendre transform** in this
repository (`concentration_toolkit/05`,
`regularized_mdps_and_duality/01`, `rlhf_mathematics/02`,
`lqr_and_continuous_control/04`...). The dual objective is the
log-likelihood, and fitting `w` is MLE in this exponential family:

```math
\nabla_w\ \log L
=
\mu_E\ -\ \mathbb{E}_{p_w}\big[\phi(\tau)\big]
\qquad\text{— observed minus model feature expectations,}
```

zero exactly at matching: the moment-matching condition and the MLE
condition coincide, as always in exponential families. Computing the
model expectation requires the partition function over paths — i.e.
**soft value iteration**: the backward recursion for `log Z` is exactly
the soft Bellman operator of `regularized_mdps_and_duality/01` (and for
stochastic dynamics, the honest version must use the
variational/dynamics-respecting form or inherit the risk-seeking
pathology of `regularized_mdps_and_duality/02` — Ziebart's causal-
entropy refinement is precisely that correction).

## What MaxEnt IRL Actually Delivers

```text
1. a UNIQUE, convex answer: the dual is concave in w — the selection
   principle turned ill-posed inversion into convex MLE;
2. a generative model of suboptimal demonstrations: experts are modeled
   as noisily rational (Boltzmann in return) — matching real data
   better than strict optimality, and making the likelihood
   well-defined for imperfect demos;
3. the bridge to RLHF: Bradley–Terry over trajectory pairs
   (rlhf_mathematics/01) is the two-sample comparison restriction of
   exactly this exponential family — preference learning and MaxEnt IRL
   are the same statistical model queried differently;
4. the bridge to GAIL: as the feature class grows to all functions, the
   matching constraint becomes occupancy matching — notebook 03's
   starting point.
```

## What Remains Open

The reward-vs-dynamics confound (a demonstrator with a *different model
of P* is indistinguishable from one with a different `r` — untreated in
the standard theory); scaling the partition-function computation beyond
soft-VI-tractable state spaces (deep MaxEnt IRL approximates it with
importance sampling of unknown bias); and the selection principle
itself — maximum entropy is principled, not unique; risk-sensitive or
resource-rational demonstrator models select differently and are barely
explored.
