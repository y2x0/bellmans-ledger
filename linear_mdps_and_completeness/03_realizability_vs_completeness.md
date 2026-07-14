# Realizability vs Bellman Completeness

## The Two Conditions

For a function class `F ⊆ {S x A -> R}` and the optimality backup `T`:

```math
\text{realizability:}\qquad q^*\in\mathcal{F}
```

```math
\text{Bellman completeness:}\qquad \mathcal{T}f\in\mathcal{F}\quad\forall f\in\mathcal{F}.
```

Completeness implies realizability under mild conditions (iterate `T` from
any `f in F`; the limit `q*` is in the closure of `F`). The converse fails,
and the gap between the two conditions is where sample-efficient RL lives
or dies:

```text
                       online regret          offline w/ coverage
completeness (+opt.)   poly(d) (notebook 02)  poly(d) (offline_rl/04)
realizability only     exp lower bounds       exp lower bounds (notebook 05)
                       (Weisz et al.)
```

## Why Realizability Is Not Enough: The Mechanism

Every algorithm in this repository learns by **regression on backed-up
targets**: fit `f_{k+1} ~ T f_k`. Realizability guarantees the *final
destination* `q*` is expressible. It says nothing about the
*intermediate targets*: `T f_k` for `f_k in F` may lie far outside `F`,
and the projection back into `F` introduces an error that (i) the
algorithm cannot measure (it never sees `T f_k`, only samples of it) and
(ii) compounds through iterations (`value_based_deep_rl/02`'s error
propagation, now with a floor that never shrinks with data).

Sharper: with realizability only, the **Bellman error is not an estimable
quantity.** The population regression loss at `f`,

```math
\mathbb{E}\big[(f(s,a)-r-\gamma\max_{a'}f(s',a'))^2\big]
=
\underbrace{\|f-\mathcal{T}f\|^2_{2,\mu}}_{\text{what we want}}
+\underbrace{\gamma^2\,\mathbb{E}\,\mathrm{Var}_{s'}\big(\max_{a'}f\big)}_{\text{noise, }f\text{-dependent}},
```

carries an `f`-dependent variance term (the double-sampling problem,
`stochastic_approximation/06`), so minimizing empirical Bellman residuals
does not minimize Bellman error. Completeness dissolves the problem:
regression *onto the class* of the target `T f` is well-posed because the
target is IN the class — the projection error is zero and the variance
term becomes an additive constant offset within each regression.

## Completeness Is Not Monotone In The Class

The property that makes completeness treacherous:

**Enlarging `F` can destroy it.** Worked example. State space `{1, 2}`,
one action, deterministic cycle `1 -> 2 -> 1`, `gamma = 1/2`, rewards 0.
The backup is `(Tf)(1) = f(2)/2, (Tf)(2) = f(1)/2`.

```text
F1 = span{(1,1)}:        T(a,a) = (a/2, a/2) in F1.   COMPLETE (and q*=0 in it).
F2 = span{(1,1),(1,0)}:  T(0,... take f=(1,0): Tf = (0, 1/2), and
                         (0,1/2) = b(1,1)+c(1,0) forces c=-1/2, b=1/2 — fine,
                         still in F2 (dim 2 of a 2-dim space: complete trivially).
shrink instead:
F3 = span{(1,0)}:        f=(1,0): Tf=(0,1/2) ∉ F3.    NOT complete, though
                         q* = (0,0) ∈ F3: realizable, incomplete.
```

And in general position, adding one feature to a complete class breaks
closure of the *backup image* — completeness demands `T`-invariance, a
knife-edge alignment between the class and the environment. Practical
consequence, stated plainly: **"use a bigger network" is not a safe
response to RL approximation error** in the way it is in supervised
learning; enlarging the class enlarges the set of representable *wrong*
targets faster than it helps, unless the enlargement preserves
`T`-invariance. (The linear MDP of notebook 01 is precious precisely
because the *environment* hands you an invariant class.)

## The Hardness Theorem For Realizability Alone

**(Weisz–Amortila–Szepesvári 2021, shape.)** There is a family of MDPs
with `A = exp(...)` actions and a `d`-dimensional feature map such that
`q*` is *exactly linear* (`realizable, zero error`), yet **any** planner
with generative-model access needs `min(exp(Omega(d)), exp(Omega(H)))`
queries to output a near-optimal policy.

*Construction idea.* Hide the optimal action direction inside an
exponentially large action set whose features form a near-orthogonal
packing of the sphere (`exp(d)` directions with pairwise inner products
`<= 1/2`). Values are arranged so that every *wrong* action looks
identical (same `q`), while the single good one is distinguishable only by
querying (near-)its own direction — a needle in an `exp(d)` haystack
(`concentration_toolkit/05`'s Fano, with the packing supplying `log M =
Omega(d)`). Realizability holds by construction; completeness fails
because backups of linear functions pick up the max over the packing —
a non-linear kink the class cannot express, which is exactly the
information the learner would have needed for free. ∎ (idea)

## The Design Cell Summary

```text
condition on (F, T)          estimable loss?      poly RL?
completeness                 yes (regression      yes (02; offline_rl/04)
                             targets in-class)
realizability + gap/         sometimes            special cases only
low variance
realizability alone          no (double           NO — exp lower bounds
                             sampling)
```

## What Remains Open

Conditions strictly between the two (e.g. completeness w.r.t. the
*optimal* policy's backup only; low "inherent Bellman error") — several
admit poly-sample algorithms; the exact boundary of learnability is not
mapped, and is arguably the central open problem of RL theory
(the decision-estimation coefficient program is the current best attempt
at drawing it).
