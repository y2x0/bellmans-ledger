# Successor Features And Generalized Policy Improvement

```text
Barreto et al., NeurIPS 2017. papers/successor-features-barreto-2017.pdf
```

## Successor Features

Assume rewards decompose over known (or learned) features:

```math
r(s,a,s')=\phi(s,a,s')^\top w
\qquad\text{— tasks are points }w\in\mathbb{R}^d .
```

Define the **successor features** of a policy — the discounted expected
feature accumulation (the SR of notebook 01 with `e_s` generalized to
`phi`):

```math
\psi^\pi(s,a)
=
\mathbb{E}_\pi\Big[\sum_{t=0}^\infty\gamma^t\,\phi_{t}\ \Big|\ S_0=s,A_0=a\Big]
\qquad\Longrightarrow\qquad
q_w^\pi(s,a)=\psi^\pi(s,a)^\top w
```

for **every** task `w` simultaneously — one learned object (`psi^pi`,
TD-learnable exactly as in notebook 01), all tasks' evaluations by inner
product.

## Generalized Policy Improvement

Given a library of policies `pi_1..pi_n` with their successor features
(learned on past tasks `w_1..w_n`), and a **new** task `w`: act greedily
with respect to the *max over the library's* evaluations:

```math
\pi(s)\ \in\ \arg\max_a\ \max_{i}\ \psi^{\pi_i}(s,a)^\top w .
```

**Theorem (GPI, Barreto et al. Thm 1).** Let
`q^max(s,a) = max_i q_w^{pi_i}(s,a)` and suppose each is known up to
error `eps` (`|q-tilde_i - q_w^{pi_i}| <= eps`). Then the GPI policy
`pi` satisfies

```math
q_w^{\pi}(s,a)\ \ge\ \max_i\ q_w^{\pi_i}(s,a)\ -\ \frac{2\varepsilon}{1-\gamma} .
```

*Proof.* This is the policy improvement theorem
(`mdp_foundations/05`) with the single policy replaced by the upper
envelope. Exact case (`eps = 0`): `pi` is greedy w.r.t.
`q^max`, and for the Bellman operator of `pi`,

```math
\mathcal{T}^{\pi}q^{\max}(s,a)
=
r+\gamma\,\mathbb{E}\big[q^{\max}(s',\pi(s'))\big]
=
r+\gamma\,\mathbb{E}\big[\max_b\max_i q^{\pi_i}(s',b)\big]
\ \ge\
r+\gamma\,\mathbb{E}\big[q^{\pi_i}(s',\pi_i(s'))\big]
=
q^{\pi_i}(s,a)
```

for every `i` — so `T^pi q^max >= q^max`, and the monotone telescope
(`mdp_foundations/05`) gives `q^pi >= q^max`. The `eps`-version tracks
the approximation through the telescope, paying the standard geometric
series `2 eps/(1-gamma)` (`mdp_foundations/04`'s greedy-error bound,
with the envelope in place of the estimate). ∎

**One improvement step over an entire library, before any learning on
the new task** — the library is at least as good as its best member
everywhere, state by state (not merely the best single member overall:
the max is inside the state).

## The Transfer Bound

**Theorem 2 (Barreto et al.).** For the new task `w`, against its true
optimum `q_w^*`, the GPI policy from library tasks `{w_i}` satisfies

```math
\|q_w^*-q_w^{\pi}\|_\infty
\ \le\
\frac{2}{1-\gamma}\ \Big(\ \|\phi\|_\infty\ \min_i\|w-w_i\|\ +\ \varepsilon\text{-terms}\Big)
```

— suboptimality on the new task is controlled by the **distance from
`w` to the nearest task already mastered**. Task space geometry becomes
performance geometry: a library covering the reward simplex `eps`-densely
guarantees `eps`-transfer everywhere, with the covering-number economics
of `concentration_toolkit/04` deciding library size.

*Proof mechanism:* `pi_i` is optimal for `w_i`; its value under `w`
differs from its value under `w_i` by at most
`||phi|| ||w - w_i|| / (1-gamma)` (rewards differ by that much per
step); chain with the GPI guarantee. ∎

## Position

```math
(\mathcal{T},d,\mathcal{F})
=
\Big(\text{TD on }\psi\ \text{per policy; envelope-greedy improvement},\ \|\cdot\|_\infty,\ \text{linear in }\phi\ \text{per task}\Big),
```

and the design cell it fills: **transfer across rewards with fixed
dynamics** — precisely the regime where model-based transfer is overkill
(no need to re-plan dynamics) and model-free is wasteful (no need to
re-learn propagation). The reward-identifiability themes of
`rlhf_mathematics/01` meet this family head-on: preference-learned
rewards delivered as `w`-vectors over fixed features would make RLHF
tasks GPI-transferable — largely unexplored.

## What Remains Open

Where do the features come from (learned `phi` must make BOTH the reward
linear AND `psi` learnable — a joint condition with no clean
characterization); GPI with function-approximated `psi` inherits
`stochastic_approximation/04–06` per library member; and the library
curation problem (which policies to keep, cf.
`options_and_hierarchy/04`'s discovery question) is open in the same way.
