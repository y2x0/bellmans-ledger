# The Option-Critic Gradient Theorems

```text
Bacon, Harb & Precup, AAAI 2017. papers/option-critic-bacon-2016.pdf
```

## The Question

Notebooks 01–02 assumed the options given. Option-critic learns them by
gradient ascent: parameterize the intra-option policies
`pi_{omega,theta}(a|s)` and terminations `beta_{omega,nu}(s)`, and
differentiate the expected return through the **augmented chain** whose
state is the pair `(s, omega)` — state plus currently executing option.

The augmented process is Markov with kernel

```math
P\big((s',\omega')\mid(s,\omega)\big)
=
\sum_a\pi_{\omega,\theta}(a\mid s)\,P(s'\mid s,a)\Big[
(1-\beta_{\omega,\nu}(s'))\,\mathbb{1}_{\omega'=\omega}
+\beta_{\omega,\nu}(s')\,\mu(\omega'\mid s')\Big],
```

— continue with the same option, or terminate and let the top-level
policy `mu` re-choose. Both theorems below are the policy gradient
theorem (`policy_gradient/01`) applied to this chain, with the two
parameter groups touching it in different places.

## Theorem 1: Intra-Option Policy Gradient

```math
\frac{\partial J}{\partial\theta}
=
\mathbb{E}_{(s,\omega)\sim d_\mu^{\text{aug}}}\ \Big[\ \sum_a\frac{\partial\pi_{\omega,\theta}(a\mid s)}{\partial\theta}\ q_U(s,\omega,a)\ \Big],
```

where `q_U(s, omega, a) = r(s,a) + gamma E_{s'}[U_omega(s')]` is the
state-option-action value through the continuation function `U` of
notebook 02.

*Derivation.* Run Proof 1 of `policy_gradient/01` on the augmented
chain: differentiate the augmented Bellman equation; `theta` appears
only in the `pi_omega` factor of the kernel and the reward mix; the
recursion for `grad v` solves through the augmented resolvent, yielding
the augmented discounted visitation `d^aug` as the measure. The only
new bookkeeping is that the "next value" seen by the derivative is `U`
(continuation-or-reselect), not `v` — exactly the substitution notebook
02 introduced. ∎

Same shape as always: **score times a value, under the process's own
visitation** — the theorem's content is *which* value (`q_U`) and
*which* visitation (state–option pairs).

## Theorem 2: Termination Gradient

```math
\frac{\partial J}{\partial\nu}
=
-\ \mathbb{E}_{(s',\omega)\sim d_\mu^{\text{aug}}}\ \Big[\ \frac{\partial\beta_{\omega,\nu}(s')}{\partial\nu}\ A_\mu(s',\omega)\ \Big],
\qquad
A_\mu(s',\omega)=q_\mu(s',\omega)-v_\mu(s') .
```

*Derivation.* `nu` enters the kernel only through the
`beta`-interpolation between "keep `omega`" (value `q(s', omega)`) and
"re-choose via `mu`" (value `v(s')`); differentiating that convex
combination pulls out exactly `-(q - v) = -A` as the coefficient of
`d-beta/d-nu`, and the resolvent argument again supplies the visitation
measure. ∎

**Read the sign — it is the theorem's insight.** Raise `beta(s')`
(terminate more) exactly where the *advantage of the current option is
negative*: options should die where they are no longer the best thing
running. Termination learning is advantage-driven exit. Corollary of the
sign: if all options come to look equally good (`A -> 0` everywhere),
the gradient pushes `beta` nowhere in particular and — with the usual
entropy-like drift of function approximation — terminations inflate:

## The Degeneracy The Theorems Predict

Two collapse modes, both observed from day one and both visible in the
gradients:

```text
1. beta -> 1 everywhere: options shrink to primitive actions (the
   augmented chain degenerates to the base MDP; hierarchy evaporates).
   Driver: A_mu(s, omega) is small once mu is decent (all options near
   v), so the termination gradient is dominated by noise + any
   regularization pressure. Standard patch: a "deliberation cost"
   (subtract eta from the reward at each termination) — an explicit
   Omega-regularizer on switching, regularized_mdps_and_duality/04
   style, pricing decision frequency.
2. one option wins everything: mu's selection concentrates; the other
   options stop being visited (d^aug support collapse), stop learning,
   die — the exploration-starvation failure
   (regret_and_exploration/04) at the option level.
```

Both are instances of this repository's conservation principle
(`global_convergence_of_pg/05`): the gradients faithfully optimize
return, and return does not *per se* reward maintaining a diverse,
temporally-extended library — that objective must be added (costs,
entropy over `mu`, diversity bonuses), or the hierarchy is optimized
away.

## What Remains Open

What options *should* be — the objective for option discovery — remains
genuinely unsettled (candidates: bottleneck states, eigenoptions from
the SR's spectrum — `successor_representations/01`, empowerment,
compression). Option-critic settled *how to differentiate* a hierarchy;
the field has not settled *what to ask of one* — notebook 04 states the
candidate answers and their counterexamples.
