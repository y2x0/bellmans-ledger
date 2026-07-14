# Bernstein Bonuses And The Minimax Rate

## The Target

Replace notebook 02's `H^{3/2} sqrt(SAT)` by

```math
\mathrm{Reg}(K)=\tilde O\big(\sqrt{H\,S\,A\,T}\big)
=\tilde O\big(H\sqrt{S\,A\,K}\big),
```

which notebook 04 shows is minimax-optimal (up to logs and lower-order
terms). One `sqrt(H)` must go; it goes exactly where the audit said: the
bonus scale.

## The Bernstein Bonus

Replace the Hoeffding bonus by an **empirical-Bernstein** bonus
(`concentration_toolkit/01–02`; variance estimated from the same data, with
its own lower-order correction):

```math
b^k(s,a)
=
\sqrt{\frac{2\,\widehat{\mathrm{Var}}_{\hat P^k}\big(\bar V^k_{h+1}\big)(s,a)\ \log\frac{\cdot}{\delta}}{N^k(s,a)}}
\ +\ \frac{c\,H\,\log\frac{\cdot}{\delta}}{N^k(s,a)} .
```

Validity of optimism (Step 2 of 02) goes through with Freedman in place of
Azuma. The range term `H/N` survives but now sits at order `1/N` —
lower-order after pigeonholing (`sum 1/N ~ SA log K`, not `sqrt`).

## The Total-Variance Argument

The main regret term is now

```math
\sum_{k,h}\sqrt{\frac{\mathrm{Var}_{P}\big(V_{h+1}^{\pi_k}\big)(s_h^k,a_h^k)}{N^k}}
\ \le\
\sqrt{\underbrace{\sum_{k,h}\frac{1}{N^k}}_{\tilde O(SA)}}\ \cdot\
\sqrt{\underbrace{\sum_{k,h}\mathrm{Var}_{P}\big(V^{\pi_k}_{h+1}\big)(s_h^k,a_h^k)}_{(\dagger)}}
```

by Cauchy–Schwarz. Everything reduces to bounding `(dagger)` — the sum of
on-trajectory one-step value variances — and this is the episodic **law of
total variance** (the same lemma as `finite_time_td_q/04`, in finite-horizon
clothing):

**Lemma.** For any policy `pi`, one episode satisfies

```math
\mathbb{E}_\pi\Bigg[\sum_{h=1}^{H}\mathrm{Var}_{P}\big(V_{h+1}^{\pi}\big)(s_h,a_h)\Bigg]
=
\mathrm{Var}_\pi\Bigg[\sum_{h=1}^H r_h\Bigg]
\ \le\ H^2 .
```

*Proof.* Let `M_h = V_h^pi(s_h) + sum_{u<h} r_u` — the "predicted total
return" process. `M_h` is a martingale (Bellman equation for `V^pi` =
tower property). Its increments are
`r_h + V_{h+1}(s_{h+1}) - V_h(s_h)`, whose conditional variance is exactly
`Var_P(V_{h+1})(s_h, a_h)` (+ reward noise). The variance of a martingale's
terminal value is the *sum* of its increments' conditional variances
(orthogonality of MDS increments); the terminal value is the realized
return, whose variance is at most `H^2/4`. ∎

So `(dagger) <= H^2 K` in expectation (+ Freedman fluctuations), and:

```math
\mathrm{Reg}(K)
\ \le\
\tilde O\Big(\sqrt{SA}\cdot\sqrt{H^2K}\Big)
=
\tilde O\big(H\sqrt{SAK}\big)
=
\tilde O\big(\sqrt{H\,SA\,T}\big). \qquad\blacksquare
```

**The exchange that just happened, in words:** Hoeffding charged every step
the worst case `H`; the martingale identity says the H steps of an episode
share a *single* `H^2` variance budget — an average of `H` per step, i.e.
`sqrt(H)` per step after the square root. One factor of `sqrt(H)`
recovered, and it is the *same* factor, by the *same* lemma, as the
`(1-gamma)^{-4} -> (1-gamma)^{-3}` repair in `finite_time_td_q/04`.
Episodic `H` and discounted `1/(1-gamma)` are one phenomenon.

## Remaining Bookkeeping (named, not belabored)

```text
1. the correction term (*) of 02 — (P-hat - P)(bar-V - V*) — must be kept
   at lower order without paying sqrt(S) in the lead: done by bounding
   bar-V - V* recursively by the regret itself (a bootstrap/self-bounding
   argument) or by Bernstein on the per-state deviations;
2. empirical vs true variance: Var-hat vs Var costs another lower-order
   Freedman term;
3. time-inhomogeneous P (P_h per step) multiplies S by H in the counting
   but is otherwise identical.
```

These are exactly the terms that separate published variants (UCBVI-BF,
EULER, ORLC...); the leading term is the lemma above, full stop.

## What Remains Open

Instance-dependent episodic regret (beyond worst-case: gaps, variance
profiles) is still an active frontier; and the `min(sqrt(HSAT), ...)`
burn-in / low-`K` regime has residual gaps between best upper and lower
bounds.
