# Fitted-Q Error Propagation: The Munos–Szepesvári Theorem

## The Algorithm Being Priced

Fitted Q-iteration (FQI) — the idealized skeleton of DQN's outer loop
(`value_based_deep_rl/02`):

```math
Q_{k+1}
=
\arg\min_{f\in\mathcal{F}}\ \frac{1}{n}\sum_{i}\Big(f(s_i,a_i)-\big[r_i+\gamma\max_{a'}Q_k(s'_i,a')\big]\Big)^2,
\qquad
(s_i,a_i)\sim\mu\ \text{(data distribution)} .
```

Each round commits a regression error
`eps_k = Q_{k+1} - T Q_k` (in `L2(mu)`); the theorem tracks how these
errors surface in the **value of the final greedy policy**.

## Concentrability Coefficients

The single new object. For a distribution `rho` (where performance is
measured) and the data distribution `mu`, define for each `m >= 1`

```math
c(m)
=
\sup_{\pi_1,\ldots,\pi_m}\ \Big\|\frac{d\big(\rho\,P_{\pi_1}P_{\pi_2}\cdots P_{\pi_m}\big)}{d\mu}\Big\|_\infty
```

— the worst density ratio of *any* `m`-step future of `rho` (under any
policy sequence) against the data. The **discounted concentrability**:

```math
C_{\rho,\mu}
=
(1-\gamma)^2\sum_{m\ge 1}m\,\gamma^{m-1}\,c(m)
\qquad(\text{finite iff the } c(m) \text{ grow subexponentially}) .
```

## The Theorem

**(Munos 2003/2007; Munos–Szepesvári 2008, shape.)** After `K` rounds of
FQI with per-round errors `||eps_k||_{2,mu} <= eps`, the greedy policy
`pi_K` w.r.t. `Q_K` satisfies

```math
\big\|\,q^*-q^{\pi_K}\big\|_{1,\rho}
\ \le\
\frac{2\gamma}{(1-\gamma)^2}\ \sqrt{C_{\rho,\mu}}\ \ \varepsilon
\ \ +\ \
\frac{4\,\gamma^{K+1}}{(1-\gamma)^2}\,R_{\max}\ \text{(vanishing)} .
```

## The Proof, In Its Three Moves

**Move 1: the pointwise error recursion.** With `e_k = q* - Q_k`:

```math
e_{k+1}
=
q^*-\mathcal{T}Q_k+\underbrace{(\mathcal{T}Q_k-Q_{k+1})}_{-\varepsilon_k}
=
\gamma\,P_{\pi^*}e_k\ \text{-type terms}-\varepsilon_k,
```

more precisely, using `T`'s max structure on both sides (greedy policies
of `q*` and of `Q_k` sandwich the difference — the same two-policy
sandwich as `mdp_foundations/04`'s greedy bound):

```math
\gamma P_{\pi^*}e_k-\varepsilon_k
\ \le\ e_{k+1}\ \le\
\gamma P_{\pi_k}e_k-\varepsilon_k .
```

**Move 2: unroll.** Errors accumulate as
`e_K ~ sum_k gamma^{K-k} (products of stochastic matrices) eps_k` — each
`eps_k` is transported `K - k` steps by policy-dependent kernels and
discounted. The final-policy loss `q* - q^{pi_K}` adds one more resolvent
`(I - gamma P_{pi_K})^{-1}`, contributing one `1/(1-gamma)`; the
accumulation contributes the other. (This is
`value_based_deep_rl/02`'s `(1-gamma)^{-2}`, now with its exact origin.)

**Move 3: change of measure.** Every transported error term is an
integral of `|eps_k|` against a distribution of the form
`rho P_{pi_1} ... P_{pi_m}` — future visitations. The analysis only knows
`||eps_k||` under `mu`. Hölder/Cauchy–Schwarz per term:

```math
\int|\varepsilon_k|\ d\big(\rho P_{\pi_1}\!\cdots P_{\pi_m}\big)
\ \le\
\sqrt{c(m)}\ \|\varepsilon_k\|_{2,\mu},
```

and the discounted, `m`-weighted resummation of the `sqrt(c(m))` is what
defines `C_{rho,mu}`. ∎

**Load-bearing audit:**

```text
sup over POLICY SEQUENCES in c(m)  ->  Move 2's kernels are pi_k- and
                                       pi*-dependent and the pi_k are
                                       data-dependent: the sup is what makes
                                       the bound algorithm-free — and what
                                       makes it enormous (all-policy
                                       concentrability; contrast 04)
L2 regression error                ->  Move 3's Cauchy–Schwarz; L_inf error
                                       would skip concentrability entirely
                                       but is unattainable by regression
gamma < 1                          ->  summability of the c(m) series
```

## The Companion Counterexample

The bound is not pessimism — the blowup is real. Exhibit (Munos-style,
2-action chain): rewards only at the end of an `H`-ish corridor; `mu`
concentrated near the corridor's start (as a behavior policy that rarely
progresses would produce); `F` misspecified by `eps` in a direction whose
backup image points *along* the corridor. Then each iteration transports
the `eps` one step deeper while `mu` gives it vanishing weight:
`c(m) ~ (1/p_progress)^m` grows geometrically, `C = infinity`, and the
greedy policy's value error is `Theta(eps / (1-gamma)^2)` with the
constant realized — while the same data supports Monte-Carlo regression of
`q^{pi_b}` (no transport, no `c(m)`) to accuracy `eps`. **Bootstrapping is
the ingredient that converts distribution mismatch into compounding
error**; MC pays variance instead (`stochastic_approximation/02`) — the
deadly triad's ledger, with exact prices.

## Position

This theorem is the general-`F`, all-policy-coverage pole of the family;
notebook 04's single-policy pessimism and C1/05's exponential lower bound
are the two modern refinements that bracket it — one showing how to need
less coverage (change the algorithm), the other showing coverage alone was
never enough (realizability is not completeness).

## What Remains Open

Tight versions with the sup over policies replaced by the *realized*
policy sequence (data-dependent concentrability — partially available);
first-order (variance-adaptive) versions; and any meaningful average-case
analogue of `c(m)` for natural task distributions.
