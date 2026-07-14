# The Distributional Bellman Operator And Its Contraction

## The Random Return

Define the return as a random variable, not its expectation:

```math
Z^\pi(s,a)
=
\sum_{t=0}^\infty \gamma^t R_t
\quad\Big|\quad S_0=s,\ A_0=a,\ \text{actions}\sim\pi,
\qquad
Q^\pi(s,a)=\mathbb{E}\big[Z^\pi(s,a)\big].
```

`Z^pi(s,a)` aggregates three independent sources of randomness:

```text
1. reward noise            R(s,a) may be stochastic
2. transition noise        S' ~ P(.|s,a)
3. continuation randomness Z(S', A') is itself a random return
```

## The Operator

The distributional Bellman equation is an equality **in distribution**:

```math
Z^\pi(s,a)\ \overset{D}{=}\ R(s,a)+\gamma\,Z^\pi(S',A'),
\qquad
S'\sim P(\cdot\mid s,a),\ A'\sim\pi(\cdot\mid S') .
```

The corresponding operator on `Z = {S x A -> P(R)}`: `T^pi Z (s,a)` is the
law of `R(s,a) + gamma Z(S', A')`. Written on laws, it is a **mixture of
affine pushforwards**:

```math
(\mathcal{T}^\pi Z)(s,a)
=
\sum_{s',a'} P(s'\mid s,a)\,\pi(a'\mid s')\ \mathrm{Law}\Big(R(s,a)+\gamma\,Z(s',a')\Big),
```

each component being the law of `Z(s',a')` pushed through
`z -> r + gamma z`, mixed over `(r, s', a')`. Taking expectations of both
sides recovers the classical `T^pi` exactly: the classical theory is the
first-moment shadow of this one.

## The Contraction Theorem

**(Bellemare–Dabney–Munos 2017, Lemma 3.)** For every `p in [1, inf]`:

```math
\boxed{\ \bar d_p\big(\mathcal{T}^\pi Z_1,\ \mathcal{T}^\pi Z_2\big)\ \le\ \gamma\ \bar d_p\big(Z_1,\ Z_2\big)\ }
```

*Proof, by exhibiting a coupling.* Fix `(s,a)`. For each `(s', a')`, let
`(X_{s'a'}, Y_{s'a'})` be the **optimal coupling** of `Z_1(s',a')` and
`Z_2(s',a')` (cost `d_p(Z_1(s',a'), Z_2(s',a'))^p`). Build a coupling of the
two backed-up laws by using the *same* reward sample, the *same* next
state-action sample, and the coupled continuations:

```math
\big(R+\gamma X_{S'A'},\quad R+\gamma Y_{S'A'}\big).
```

Its transport cost bounds the Wasserstein distance:

```math
d_p\big((\mathcal{T}^\pi Z_1)(s,a),(\mathcal{T}^\pi Z_2)(s,a)\big)^p
\ \le\
\mathbb{E}_{R,S',A'}\ \mathbb{E}\big[\,|\gamma X_{S'A'}-\gamma Y_{S'A'}|^p\,\big]
```

```math
=\ \gamma^p\ \mathbb{E}_{S',A'}\big[d_p(Z_1(S',A'),Z_2(S',A'))^p\big]
\ \le\ \gamma^p\ \bar d_p(Z_1,Z_2)^p .
```

Take sup over `(s,a)` and `p`-th roots. ∎

Read the three inequalities against notebook 01's property list: the shared
`R` cancels the shift (property 2), the shared `gamma` scales by `gamma`
(property 2), the shared `(S',A')` makes the mixture average component
distances (property 3). The proof is nothing but those three properties
composed — the coupling method making each one-line.

**Consequence (Banach, `mdp_foundations/04`):** `T^pi` has a unique fixed
point in `(Z, d_p-bar)` — necessarily `Z^pi` — and iterates converge
geometrically **as distributions**: every moment, every quantile, every
risk functional of the estimate converges, not just the mean.

## Where The Metric Choice Is Load-Bearing

The paper notes (and it is instructive to verify) that `T^pi` is **not** a
contraction in total variation, in KL, or in Kolmogorov (sup-CDF) distance.
The one-line intuition: `z -> r + gamma z` *shrinks distances on the return
axis* — which only a metric that measures distance **along the return axis**
can register as a contraction. Vertical metrics (TV, KL, Kolmogorov compare
densities/CDFs pointwise) see the shift `+r` as arbitrary disruption.
Optimal transport is not decoration here; it is the unique working geometry
among the standard candidates.

## What Breaks Next

Everything above is *evaluation* — fixed `pi`. Control replaces `pi` by a
greedy selection inside the operator, and greediness is a statement about
**means**, while the operator acts on **whole distributions**: an
`eps`-change in distribution can flip an argmax and swap in an entirely
different continuation law. The max-lemma that saved the classical proof
(`mdp_foundations/03`) has no distributional analogue — notebook 03.
