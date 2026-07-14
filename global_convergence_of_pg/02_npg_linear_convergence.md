# Natural Policy Gradient: Dimension-Free And Linear Rates

## The Update, Three Ways

For softmax parameterization, the natural gradient step
(`policy_gradient/04`: precondition by the Fisher matrix) has a closed
form — three equivalent descriptions:

```math
\text{(parameter)}\quad
\theta_{s,a}\ \leftarrow\ \theta_{s,a}+\frac{\eta}{1-\gamma}\,A^{\pi_k}(s,a)
```

```math
\text{(policy)}\quad
\pi_{k+1}(a\mid s)\ \propto\ \pi_k(a\mid s)\,\exp\Big(\frac{\eta\,A^{\pi_k}(s,a)}{1-\gamma}\Big)
```

```math
\text{(mirror descent)}\quad
\pi_{k+1}=\arg\max_\pi\ \Big\{\eta\,\langle\pi,A^{\pi_k}\rangle-(1-\gamma)\,\mathrm{KL}(\pi\|\pi_k)\Big\}\ \text{per state}
```

(the first-to-second step is a computation with the softmax Fisher
pseudo-inverse — Kakade 2001; second-to-third is
`regularized_mdps_and_duality/05`'s multiplicative-weights closed form).
**NPG is per-state multiplicative weights on the advantage.** Note what
vanished versus notebook 01: no `d^{pi}(s)` factor, no `pi(a|s)` factor
in the update — the preconditioner cancelled exactly the two
plateau-makers.

## The Dimension-Free O(1/t) Theorem

**Theorem (Agarwal–Kakade–Lee–Mahajan 2021).** Exact NPG with step `eta`:

```math
J(\pi^*)-J(\pi_K)
\ \le\
\frac{\log|\mathcal{A}|}{\eta\,K}+\frac{1}{(1-\gamma)^2\,K}.
```

**No `|S|`, no `|A|` beyond the log, no distribution-mismatch coefficient,
no dependence on initialization beyond `log|A|`.**

*Proof (it fits here).* Run the three-point/MD analysis
(`regularized_mdps_and_duality/05`) per state with comparator `pi*`,
weighted by `d^{pi*}`:

```math
\mathrm{KL}_{s}\big(\pi^*\|\pi_k\big)-\mathrm{KL}_s\big(\pi^*\|\pi_{k+1}\big)
\ \ge\
\frac{\eta}{1-\gamma}\,\big\langle \pi^*(\cdot\mid s),\,A^{\pi_k}(s,\cdot)\big\rangle
-\log Z_k(s)\ \text{-type terms},
```

and two structural facts finish it: (i) the performance difference lemma
turns `E_{d^{pi*}}<pi*, A^{pi_k}>` into `(1-gamma)(J(pi*) - J(pi_k))` —
the *comparator's* visitation is exactly what the telescoped Bregman
carries, so the mismatch ratio never appears; (ii) the log-partition terms
are controlled by per-step improvement (NPG is also a soft improvement
step — `regularized_mdps_and_duality/03`'s theorem gives
`J(pi_{k+1}) >= J(pi_k)`, absorbing `log Z >= 0`). Telescope over `k`;
the initial `E_{d^{pi*}} KL(pi* || uniform) <= log|A|`. ∎

Compare notebook 01: same objective, same MDP, but the preconditioned
geometry replaced `exp(H)`-capable constants with `log|A|`. **The plateau
was a property of the metric, not the problem** — vanilla PG measures
steps in parameter space where unvisited states barely exist; NPG measures
in per-state KL where all states are first-class. (What the plateau
*encoded* — exploration — has not vanished; it re-enters the moment
gradients are estimated from samples, since `A^{pi_k}` at unvisited states
is exactly what cannot be estimated: notebook 04.)

## Linear Convergence With Entropy

**Theorem (Cen–Cheng–Chen–Wei–Chi 2021).** For the entropy-regularized
objective (`tau > 0`), exact NPG with `eta <= (1-gamma)/tau` converges
**linearly**:

```math
\big\|q_\tau^*-q_\tau^{\pi_K}\big\|_\infty
\ \le\
C\,\big(1-\eta\tau\big)^{K}
\qquad\big(\eta=\tfrac{1-\gamma}{\tau}:\ \text{contraction factor }\gamma\big).
```

*Why linear, structurally.* With entropy, the MD step's fixed point is the
softmax of `q_tau/tau`, and the update becomes — after the same
multiplicative-weights algebra — an **averaging toward the soft-greedy
policy in log-space**:

```math
\log\pi_{k+1}
=
(1-\eta\tau)\,\log\pi_k+\eta\tau\cdot\frac{q_\tau^{\pi_k}}{\tau}\ (-\log Z),
```

i.e. a damped soft policy iteration; the contraction of the soft Bellman
operator (`regularized_mdps_and_duality/01`) converts log-policy averaging
into geometric convergence of values. At the aggressive step
`eta = (1-gamma)/tau` the recursion **is** soft PI and the rate is
`gamma^K`. Regularization → strong convexity per state → linear rates:
the purchase announced in `regularized_mdps_and_duality/01` (§"what the
temperature buys," item 2), now redeemed.

## The Practical Genealogy

```text
NPG (this file)  =  TRPO's idealized limit (policy_gradient/04: the
                    constrained step, solved exactly, small radius)
                =  one MD step (regularized_mdps_and_duality/05)
                =  soft PI, damped (with entropy)
MPO / GRPO-with-KL / RLHF-PPO inherit the multiplicative-weights form:
pi_ref * exp(advantage-like / beta) — the SAME update, with the reference
policy playing pi_k (proximal) or pi_ref (fixed): the two placements of
regularized_mdps_and_duality/05, once more.
```

## What Remains Open

These are exact-gradient theorems; with sampled advantages the rates
degrade by estimation error terms whose optimal coupling to the step size
is not settled (notebook 04 gives the framework). Linear convergence
*without* regularization holds for some step-size schedules
(geometrically increasing `eta`) — its robustness to approximation is
similarly open.
