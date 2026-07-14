# Trajectory Importance Sampling: The Exponential Variance, Proved

## The Estimator

From `stochastic_approximation/02`: for trajectories `~ pi_b`,

```math
\hat J^{\mathrm{IS}}
=
\frac{1}{n}\sum_{i=1}^n \rho^{(i)}\,G^{(i)},
\qquad
\rho=\prod_{h=1}^{H}\frac{\pi(a_h\mid s_h)}{\pi_b(a_h\mid s_h)},
```

unbiased whenever `pi << pi_b`. This file pins down the price with an
explicit instance, making the "curse of horizon" a theorem rather than a
slogan.

## The Instance

One state, two actions, horizon `H`:

```math
\pi_b(a)=\tfrac12\ \text{each},
\qquad
\pi(a_1)=1\ \text{(deterministic)},
\qquad
r=\mathbb{1}\{a=a_1\}\ \text{at every step}.
```

True value `J(pi) = H`. The weight of a trajectory that agrees with `pi`
at all `H` steps is `2^H`; any disagreement zeroes the indicator-relevant
contribution (and for clean bookkeeping take `G = H` iff all actions were
`a_1`, else 0 — realized by the reward above).

**The exact second moment.**

```math
\mathbb{E}_b\big[(\rho G)^2\big]
=
\Pr_b\{\text{all }a_1\}\cdot\big(2^H\cdot H\big)^2
=
2^{-H}\,4^H H^2
=
2^{H}H^2 ,
```

```math
\Longrightarrow\quad
\mathrm{Var}_b\big[\rho\,G\big]
=
2^H H^2-H^2
\ =\
H^2\big(2^H-1\big).
```

The estimator is a scaled Bernoulli: with probability `2^{-H}` a
trajectory contributes `2^H H`, else `0`. Unbiased — and **almost every
sample is zero**, with the entire estimate carried by
`~ n 2^{-H}` "hit" trajectories. For a relative-error estimate one needs
hits, hence

```math
n\ =\ \Omega\big(2^{H}\big)
```

samples. No estimator trick applied to *these weights* escapes this:
by the Le Cam/divergence machinery (`concentration_toolkit/05`), the data
distribution under `pi_b` simply carries `2^{-H}`-scale information about
events that `pi` makes certain. Weighted IS (self-normalizing) caps the
variance but concentrates on `0` until a hit occurs — it trades the
variance for a bias of the same exponential character. Per-decision IS
helps polynomially, not exponentially (the last step's weight is still a
product over the prefix).

**Load-bearing audit:**

```text
the pathology needs pi FAR from pi_b at MANY steps: rho is a product;
log rho is a random walk with drift sum_h KL(pi_b||pi)-ish terms. The
variance is exp(sum of per-step divergences). Trajectory IS is fine for
pi ~ pi_b (RLHF's one-step-ish ratios, policy_gradient/06) and hopeless
for genuinely different policies over long horizons.
```

## Why This Is A Statement About Trajectory Space, Not About IS

The deeper reading (setting up notebook 03): the estimator reweights in
**trajectory space**, where the two policies' distributions have
exponentially small overlap. But `J(pi)` does not live in trajectory
space — it is an integral against the **state-action occupancy**
(`mdp_foundations/06`):

```math
J(\pi)=\frac{1}{1-\gamma}\ \mathbb{E}_{(s,a)\sim d^\pi}\big[r(s,a)\big],
```

and the occupancies `d^pi, d^{pi_b}` of the instance above overlap
*substantially* (same single state!). The exponential price was paid for
reweighting a finer object than the estimand requires. Marginalized IS
(03) reweights occupancies directly and collapses the variance to
polynomial — the Markov structure, unused by trajectory IS, is exactly
what it exploits.

```text
                 reweights           overlap needed        variance
trajectory IS    path measures       exp(H)-small          exp(H)
marginalized IS  occupancy measures  poly — d^pi/d^b       poly(H)
                                     bounded
```

## The Semiparametric Preview

Even in the benign-overlap regime there is a floor: the asymptotic
variance of ANY unbiased OPE estimator is bounded below by the
**efficiency bound** (notebook 02), which features
`Var[rho (G - q)]`-type terms — IS's variance is the efficiency bound's
worst case (no model), and doubly robust estimation is the move that
attains the bound rather than the worst case.

## What Remains Open

Nothing about the lower bound; the open ground is adaptive truncation/
clipping schedules for the weights (bias-variance frontiers with
finite-sample guarantees) and OPE under *unknown* `pi_b` (propensity
estimation adds its own semiparametric layer).
