# Overestimation Bias And Double Q-Learning

## The Source: max Of Estimates vs Estimate Of max

Let `Q-hat(s', a') = q(s', a') + eps_{a'}` with zero-mean noise `eps`. Then

```math
\mathbb{E}\Big[\max_{a'}\hat Q(s',a')\Big]
\ \ge\
\max_{a'}\mathbb{E}\big[\hat Q(s',a')\big]
=
\max_{a'}q(s',a'),
```

by Jensen (`max` is convex). The target of Q-learning uses the left side; the
Bellman operator wants the right side. The gap is strict whenever noise can
change the argmax.

**Sharpening.** If the true values are equal across `m` actions and the
errors are independent `Uniform[-eps, eps]`, the expected overestimate is

```math
\mathbb{E}\Big[\max_{a}\epsilon_a\Big]=\varepsilon\,\frac{m-1}{m+1}
\ \xrightarrow{m\to\infty}\ \varepsilon,
```

(order statistics of the uniform); for i.i.d. Gaussians it grows like
`sigma sqrt(2 log m)`. The bias grows with the number of actions and with
per-action estimation noise — and function approximation *guarantees*
nonzero, correlated `eps` everywhere.

## Why It Compounds

The overestimate is not a one-shot error: it sits inside the bootstrap and is
propagated backward,

```math
q\ \to\ q+\underbrace{b_1}_{\text{max bias}}\ \to\
r+\gamma(q+b_1)\ \to\ q+b_2+\gamma b_1\ \to\ \cdots
```

so persistent bias `b` inflates values by up to `b/(1-gamma)`. Uniform
inflation would be harmless (greedy is shift-invariant); the harm is that
the bias is **state- and action-dependent** — largest where estimates are
noisiest (rarely visited, high-variance regions) — so it systematically
redirects the greedy policy toward poorly known actions. Distinguish this
from optimism-in-the-face-of-uncertainty: exploration bonuses are designed,
controlled, and vanish with data; max-bias is emergent and follows noise,
not information value.

## Double Q-Learning: Decouple Selection From Evaluation

(van Hasselt 2010; deep version van Hasselt et al. 2016.) Write the
Q-learning target's max as select-then-evaluate with the same function:

```math
y^{\mathrm{QL}}
=
r+\gamma\, Q\big(s',\ \arg\max_{a'}Q(s',a';\theta);\ \theta\big).
```

Use two independent estimators, one to select, one to evaluate:

```math
y^{\mathrm{DQL}}
=
r+\gamma\, Q\big(s',\ \arg\max_{a'}Q(s',a';\theta^{(1)});\ \theta^{(2)}\big).
```

If the noises of the two estimators are independent, then conditioned on the
selected action `a* = argmax Q^{(1)}`, the evaluator satisfies
`E[Q^{(2)}(s', a*)] = q(s', a*) <= max_a q(s', a)`: the estimate is
**unbiased for the chosen action's true value**, hence a *lower* bound in
expectation for the true max — the sign of the residual bias flips from
systematic-up to (mild) down. Tabular double Q-learning retains convergence
to `q_*` under Watkins-type conditions.

**Double DQN** instantiates `theta^{(1)} = theta` (online net selects) and
`theta^{(2)} = theta^-` (target net evaluates) — zero extra parameters, one
changed line, and the empirical value estimates on Atari drop from wildly
above the true discounted return (measured by rollouts) to nearly calibrated,
with large score gains on the games where DQN's inflation was worst.

## The Broader Pattern

```text
max-bias is one instance of: optimizing against a learned estimate exploits
the estimate's errors.

same structure appears in:
  - actor exploiting critic errors in DDPG (fix: TD3's clipped double critic,
    taking min of two critics — deliberate underestimation);
  - policy optimization against a learned reward model (reward hacking);
  - offline RL, where out-of-distribution actions' Q-values are pure
    extrapolation and the max seeks them out (fix: conservative penalties, CQL).

the operator lesson: T's max is safe applied to exact values, dangerous
applied to estimates; every fix inserts pessimism exactly at the argmax.
```
