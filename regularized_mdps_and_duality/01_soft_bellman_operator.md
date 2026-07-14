# The Soft Bellman Operator

## The Regularized Objective

Add an entropy bonus with temperature `tau > 0` to the per-step reward:

```math
J_\tau(\pi)
=
\mathbb{E}_\pi\Bigg[\sum_t\gamma^t\Big(r(s_t,a_t)+\tau\,\mathcal{H}\big(\pi(\cdot\mid s_t)\big)\Big)\Bigg],
\qquad
\mathcal{H}(\pi)=-\sum_a\pi(a)\log\pi(a).
```

## The Core Computation: Legendre–Fenchel

The regularized greedy step per state is a concave program over the
simplex:

```math
\max_{\pi\in\Delta(\mathcal{A})}\ \Big\{\ \langle\pi,\,q\rangle+\tau\,\mathcal{H}(\pi)\ \Big\}.
```

**Proposition.** The value is `tau log sum_a exp(q(a)/tau)` and the unique
maximizer is `pi*(a) ∝ exp(q(a)/tau)`.

*Proof.* Lagrangian with multiplier `lambda` for `sum pi = 1`:
`d/dpi(a): q(a) - tau(1 + log pi(a)) - lambda = 0`, so
`pi(a) = exp(q(a)/tau) e^{-1-lambda/tau}`; normalize to get the softmax;
substitute back:

```math
\langle\pi^*,q\rangle+\tau\mathcal{H}(\pi^*)
=
\sum_a\pi^*(a)\Big(q(a)-q(a)+\tau\log\textstyle\sum_b e^{q(b)/\tau}\Big)
=
\tau\log\sum_a e^{q(a)/\tau}. \qquad\blacksquare
```

In convex-analysis language: **log-sum-exp is the convex conjugate of
negative entropy** restricted to the simplex; the maximizing argument is
the gradient of the conjugate — `grad logsumexp = softmax`. This is the
identity the whole family orbits (and the same one behind Chernoff/KL
duality, `concentration_toolkit/05`, and Donsker–Varadhan,
`rlhf_mathematics/02` — three fields, one transform).

## The Soft Operators

```math
(\mathcal{T}_\tau^\pi v)(s)
=
\sum_a\pi(a\mid s)\Big[r+\tau\log\tfrac{1}{\pi(a\mid s)}+\gamma\,\mathbb{E}_{s'}v(s')\Big],
```

```math
(\mathcal{T}_\tau v)(s)
=
\tau\log\sum_a\exp\Big(\frac{r(s,a)+\gamma\,\mathbb{E}_{s'}v(s')}{\tau}\Big)
\qquad\text{(the soft/optimality version)},
```

with soft values `v_tau^*, q_tau^*` and
`v = tau log sum exp(q/tau)` replacing `v = max_a q`.

## Contraction

**Proposition.** `T_tau` is a `gamma`-contraction in `||.||_inf`.

*Proof.* It suffices (as in `mdp_foundations/03`) that logsumexp is
nonexpansive:

```math
\Big|\tau\log\textstyle\sum e^{f(a)/\tau}-\tau\log\textstyle\sum e^{g(a)/\tau}\Big|
\ \le\
\max_a|f(a)-g(a)|,
```

which follows from the mean value theorem: the gradient of
`f -> tau logsumexp(f/tau)` is a softmax distribution — nonnegative,
summing to 1 — so the directional derivative along `f - g` is a convex
combination of its components, bounded by `||f - g||_inf`. Compose with
the `gamma`-contractive expectation exactly as before. ∎

So Banach applies verbatim (`mdp_foundations/04`): unique `v_tau^*`,
geometric convergence, error bounds — the entire exact theory survives
regularization *with the max-lemma upgraded to a softmax-lemma*. As
`tau -> 0`, logsumexp → max uniformly and everything degenerates to the
classical case; as `tau -> infinity`, the operator forgets `q` and the
optimal policy → uniform.

## The Bias, Priced Exactly

**Proposition.** `0 <= v_tau^*(s) - v^*(s)` compares as

```math
\|v_\tau^*-v^*\|_\infty\ \le\ \frac{\tau\,\log|\mathcal{A}|}{1-\gamma}.
```

*Proof.* Pointwise, for any `q`:
`max_a q <= tau logsumexp(q/tau) <= max_a q + tau log|A|` (upper: each
exponent at most the max; lower: keep only the argmax term). So
`T v <= T_tau v <= T v + tau log|A|` for every `v`; iterate both
inequalities through Banach fixed points and the geometric series turns
the per-step `tau log|A|` into `tau log|A|/(1-gamma)`. ∎

The regularization is a *known, bounded, tunable* bias — contrast the
uncontrolled biases catalogued elsewhere in the repo. Annealing `tau`
trades it against the optimization/exploration benefits below.

## What The Temperature Buys (the ledger)

```text
1. differentiability: softmax is smooth where argmax jumps — gradients
   exist and are Lipschitz; the argmax-discontinuity pathologies
   (distributional_rl/03's control instability; value_based_deep_rl/03's
   greedy chattering) are smoothed with modulus 1/tau;
2. strong convexity: the per-state program is tau-strongly concave in pi,
   giving uniqueness of the policy and, downstream, LINEAR convergence of
   NPG (global_convergence_of_pg/02) — the single most consequential
   purchase;
3. exploration floor: pi*(a) >= exp(-(q_max - q(a))/tau)/|A| — no action's
   probability vanishes, keeping gradient signal alive
   (policy_gradient/02's collapse failure);
4. robustness reading: tau logsumexp is the dual of an adversarial reward
   perturbation bounded in KL — the regularized optimum is minimax against
   tau-scaled reward misspecification (one more face of the same
   transform).
```

## What Remains Open

Nothing at this level — the file is closed convex analysis. The open
questions live where the temperature meets statistics: optimal annealing
schedules with finite-sample guarantees, and the interaction of entropy
bias with pessimism in offline settings.
