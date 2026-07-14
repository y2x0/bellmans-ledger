# Blackwell Optimality

## The Hierarchy

Average reward alone under-determines "the best policy": gain ignores
transients entirely (notebook 02). The `n`-discount hierarchy refines it.
A policy `pi*` is **`n`-discount optimal** if

```math
\lim_{\gamma\uparrow1}\ \frac{1}{(1-\gamma)^{n}}\ \big(v_\gamma^{\pi^*}-v_\gamma^{\pi}\big)\ \ge\ 0
\qquad\forall\pi,
```

```text
n = -1:  gain optimality        (rho maximal)
n =  0:  bias optimality        (among gain-optimal: h maximal)
n =  1:  ...and so on up the Laurent coefficients
n = inf: BLACKWELL optimality — optimal for every gamma near 1
```

## The Theorem

**Theorem (Blackwell 1962).** In a finite MDP there exists a policy
`pi_B` and a threshold `gamma_bar < 1` such that `pi_B` is optimal in
the *discounted* sense **simultaneously for every**
`gamma in (gamma_bar, 1)`. Every Blackwell-optimal policy is `n`-discount
optimal for all `n` (in particular gain- and bias-optimal).

*Proof.* Finitely many stationary deterministic policies (sufficient by
`mdp_foundations/01`, for each fixed `gamma`). For each pair
`(pi_1, pi_2)`, the difference

```math
\Delta_{12}(\gamma)=v^{\pi_1}_\gamma(s)-v^{\pi_2}_\gamma(s)
=
\frac{\text{polynomial in }\gamma}{\det(I-\gamma P_1)\,\det(I-\gamma P_2)}
```

is a **rational function of `gamma`** (Cramer's rule on the two linear
systems). A rational function is either identically zero or has finitely
many zeros; hence each `Delta_{12}(gamma, s)` has constant sign on some
interval `(gamma_{12}, 1)`. Take `gamma_bar` = the max of the finitely
many thresholds over pairs and states. On `(gamma_bar, 1)` the (finite)
set of policies is totally preordered by dominance *independently of
gamma*; a maximal element is Blackwell optimal. The Laurent expansion
(notebook 03) shows discounted dominance for all `gamma` near 1 forces
lexicographic dominance of ALL Laurent coefficients — i.e. `n`-discount
optimality for every `n`. ∎

The proof is soft — rational functions and finiteness — which is why it
is easy to under-appreciate: it guarantees that **"the" right policy for
a continuing task is well-defined without choosing a discount**, and that
sufficiently patient discounting finds it.

## The Separating Example (gain ties, bias decides)

```text
two states; from state 1, two actions:
  action a: reward 10 today, then to state 2
  action b: reward 0 today,  then to state 2
state 2: absorbing-ish recurrent loop, reward 1 per step, no choices.
```

Both policies have gain `rho = 1` (long-run behavior identical). Biases:
`h_a(1) - h_b(1) = 10` — action `a` banks a transient windfall the gain
never sees. Gain optimality is indifferent; bias optimality (and any
`gamma < 1`, and common sense) picks `a`. Every real system with
identical steady states but different startup costs is this example;
"average-reward optimal" without bias refinement is an incomplete
specification, and Blackwell optimality is the criterion that closes it.

## Why RL Mostly Ignores This (and when it cannot)

```text
1. discounted RL with gamma near 1 IS approximate Blackwell optimality
   (the theorem's interval) — provided gamma > gamma_bar. But gamma_bar
   is instance-dependent and can be arbitrarily close to 1 (chains with
   slow mixing push the rational functions' last sign changes toward 1):
   the folk gamma = 0.99 silently assumes gamma_bar < 0.99.
2. average-reward RL (06) targets n = -1 only; bias-optimality-aware
   algorithms exist (Puterman ch. 10's policy iteration over the nested
   equations) but have essentially no sample-based descendants — an open
   design space.
3. the hierarchy matters operationally whenever gains tie by symmetry
   (queueing systems with identical throughput but different backlog
   transients; LLM serving policies with equal steady throughput but
   different warmup) — precisely the cases where naive average-reward
   methods return an arbitrary tie-break.
```

## What Remains Open

Sample-efficient learning of bias-optimal (let alone Blackwell-optimal)
policies; estimating `gamma_bar` from data (equivalently: certifying that
a chosen discount is "patient enough" — no practical test exists); and
the function-approximation version of any of it.
