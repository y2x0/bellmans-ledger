# The Linear MDP And Its Closure Properties

## The Model (Jin–Yang–Wang–Jordan 2020)

An MDP is **linear in feature map** `phi : S x A -> R^d` if there exist `d`
(signed) measures `mu = (mu_1..mu_d)` on `S` and a vector `theta in R^d`
with

```math
P(s'\mid s,a)=\big\langle\phi(s,a),\ \mu(s')\big\rangle,
\qquad
r(s,a)=\big\langle\phi(s,a),\ \theta\big\rangle .
```

**Read the definition correctly:** it is a restriction on the
*environment*, not on the value class — the transition kernel, viewed as an
`|SA| x |S|` matrix, factors through rank `d`. The features are known; the
`d` measures `mu` and `theta` are the unknowns, but they never need to be
estimated directly.

## Closure Property 1: Every Policy's Q Is Linear

**Proposition.** For *every* policy `pi` (arbitrary, even
non-stationary):

```math
q_\pi(s,a)=\big\langle\phi(s,a),\ w_\pi\big\rangle
\qquad\text{for some }w_\pi\in\mathbb{R}^d .
```

*Proof.* One line, from the Bellman expectation equation:

```math
q_\pi(s,a)
=r(s,a)+\gamma\int v_\pi(s')\,P(ds'\mid s,a)
=\Big\langle\phi(s,a),\ \underbrace{\theta+\gamma\int v_\pi(s')\,\mu(ds')}_{=:\,w_\pi}\Big\rangle. \qquad\blacksquare
```

The integral `∫ v dmu` is a `d`-vector regardless of how complicated `v_pi`
is. Note what was **not** assumed: nothing about `v_pi` being simple. The
kernel's low rank launders arbitrary downstream complexity into `d`
numbers.

## Closure Property 2: Backups Of Anything Are Linear

**Proposition.** For *any* bounded function `v : S -> R` (not just values
of policies — including the clipped, bonus-augmented, data-dependent
monsters an algorithm builds):

```math
(\mathcal{T}v)(s,a):=r(s,a)+\gamma\,\mathbb{E}_{s'}[v(s')]
=\big\langle\phi(s,a),\ \theta+\gamma\textstyle\int v\,d\mu\big\rangle
\ \in\ \mathrm{span}(\phi).
```

Same one line. **This is Bellman completeness** (`T F ⊆ F` for
`F = span(phi)`) — and notice it came from the *environment's* structure,
not from choosing `F` cleverly. Notebook 03 shows this implication does not
reverse: completeness of a class is a knife-edge property, and
environmental low rank is one of the few natural ways to get it.

## What The Closures Buy, Operationally

```text
1. value iteration stays in span(phi): each iterate is d parameters — the
   curse of dimensionality is gone IF the model holds;
2. the regression target of fitted VI is realizable at every round with
   ZERO approximation error (no error propagation through
   value_based_deep_rl/02's (1-gamma)^{-2} amplifier);
3. estimating "the relevant part of" P reduces to d-dimensional ridge
   regression: for any target function v, the vector int v dmu is
   estimated by regressing v(s') on phi(s,a) — notebook 02's engine.
```

## The Rank Restriction Is Severe

**Counterexample (rank vs reality).** A deterministic 2-state MDP with
`phi(s, a) = e_{(s,a)}` (one-hot, `d = |S||A|`) is trivially linear — the
model always holds at `d = SA`, recovering the tabular case and its rates.
The content is entirely at `d << SA`, and that is restrictive: the kernel
matrix `P` must have rank `<= d`. A generic stochastic matrix has full
rank; e.g. any MDP whose next-state distribution depends on `(s,a)` through
more than `d` "effective factors" is out. Structured examples that *are*
low-rank:

```text
- block MDPs: s' drawn per latent block; rank = #blocks;
- feature-mixture dynamics: P = sum_i phi_i(s,a) nu_i — d mixing components;
- tabular with small SA (degenerate but instructive).
```

**Nonexample.** Even the 4x4 gridworld with a wind that depends on the
exact cell is full-rank: `d = |S|`, no compression. Linear *rewards* with
nonlinear dynamics also break it — the model requires the *kernel* to
factor, which is the strong half.

## Position And Foreshadowing

```math
(\mathcal{T},d,\mathcal{F})
=
\big(\text{optimality backup},\ \text{ellipsoid widths in }\bar V_t^{-1}\text{-norm},\ \mathrm{span}(\phi)\big),
```

with the second closure property guaranteeing the operator never leaves
`F`. Notebook 02 runs optimism in this class. Notebook 03 asks the sharp
question: which of the two closure properties was actually doing the work —
"Q* is in the class" (weak, from closure 1 applied to `pi*`) or "backups
stay in the class" (strong, closure 2)? The answer determines whether
sample-efficient RL exists, and it is the strong one.

## What Remains Open

Learning the *features themselves* (representation learning for low-rank
MDPs — FLAMBE-style results exist with poly rates under reachability
assumptions, but practical feature learning with guarantees is open); and
testing the linear-MDP hypothesis from data.
