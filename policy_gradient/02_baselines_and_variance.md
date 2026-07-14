# Baselines And Variance

## What May Be Subtracted

For any function `b(s)` of state only:

```math
\mathbb{E}_{a\sim\pi(\cdot\mid s)}\big[\nabla\log\pi(a\mid s)\,b(s)\big]
=
b(s)\,\nabla\!\!\sum_a\pi(a\mid s)
=
b(s)\,\nabla 1
=
0 .
```

So the gradient is unchanged by replacing `q(s,a)` with `q(s,a) - b(s)`:

```math
\nabla J\ \propto\
\mathbb{E}\big[\nabla\log\pi_\theta(a\mid s)\,\big(q_\pi(s,a)-b(s)\big)\big].
```

The score function integrates to zero against any state-measurable factor —
a **control variate** with zero cost in bias. The same computation shows why
`b` must not depend on `a`: an action-dependent baseline leaves a residual
`E_a[grad log pi(a|s) b(s,a)] != 0` (action-dependent baselines exist but
must add back an analytic correction term).

## The Variance-Optimal Baseline

Per state, minimize
`Var[g] = E[||grad log pi||^2 (q - b)^2] - ||grad J||^2` over `b`. Setting
the derivative to zero:

```math
b^*(s)
=
\frac{\mathbb{E}_a\big[\|\nabla\log\pi(a\mid s)\|^2\,q_\pi(s,a)\big]}
     {\mathbb{E}_a\big[\|\nabla\log\pi(a\mid s)\|^2\big]}
```

— a score-weighted average of `q`. Dropping the weights gives
`b(s) = E_a[q_pi(s,a)] = v_pi(s)`: **the value function is the (near-)optimal
baseline**, and the resulting integrand is exactly the advantage

```math
A_\pi(s,a)=q_\pi(s,a)-v_\pi(s).
```

This is the principled origin of actor-critic architecture: the critic
`V_w ~ v_pi` exists to be a baseline (variance reduction) and a bootstrap
(target construction, notebook 05) — not because the policy needs values to
act.

## The Size Of The Problem

Order-of-magnitude anatomy of REINFORCE variance for horizon `H`:

```text
G_t sums O(H) correlated reward terms          -> Var[G_t] = O(H) to O(H^2)
the trajectory gradient sums O(H) score terms  -> another factor up to O(H)
sparse terminal reward: every action in a      -> credit spread uniformly;
trajectory receives the SAME scalar signal        per-action signal-to-noise ~ 1/H
```

Baselines remove the *state-value* component of the signal (often the bulk of
`q`'s magnitude — "was this a good state" vs "was this a good action");
advantages are typically orders of magnitude smaller than returns, and the
variance falls accordingly. What baselines cannot remove: the environment's
own stochasticity between action and consequence. That requires shortening
the horizon of the target — bootstrapping — which reintroduces bias:
notebook 05's dial.

## Entropy Regularization

The standard exploration-preserving term (PPO's `c2 S[pi](s)`):

```math
\mathcal{S}[\pi](s)=-\sum_a\pi(a\mid s)\log\pi(a\mid s),
\qquad
\nabla_\theta\,\mathbb{E}\,\mathcal{S}
\ \text{pushes probability mass toward uniform.}
```

Mechanically it opposes the collapse of `pi` onto the current argmax —
premature determinism is fatal to a method whose gradient estimator's
support is the policy's own support (an action never taken produces no
gradient signal, ever). Maximum-entropy RL (SAC) promotes the bonus from
regularizer to objective, changing the fixed point itself (Boltzmann rather
than greedy policies, soft Bellman operators — same contraction proofs with
`max` replaced by `softmax`/log-sum-exp, which is likewise nonexpansive).
