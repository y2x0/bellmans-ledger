# Q-Learning Upper Bounds: The Hoeffding Analysis, Audited

## The Cleanest Setting

Synchronous Q-learning with a **generative model**: at each round `t`, for
*every* `(s,a)` simultaneously, draw one sample `s'_{s,a} ~ P(.|s,a)` and
update

```math
Q_{t+1}(s,a)
=
(1-\alpha_t)\,Q_t(s,a)+\alpha_t\Big[r(s,a)+\gamma\max_{a'}Q_t\big(s'_{s,a},a'\big)\Big].
```

This removes exploration (every cell is sampled) and asynchrony, isolating
the pure statistical question: *how many samples per cell until*
`||Q_T - q*||_inf <= eps`? The answer to be proved:

```math
T\ =\ \tilde O\Big(\frac{1}{(1-\gamma)^4\,\varepsilon^2}\Big)
\quad\text{rounds}
\qquad\Big(=\ \tilde O\big(\tfrac{|S||A|}{(1-\gamma)^4\varepsilon^2}\big)\ \text{total samples}\Big),
```

and the point of this file is to watch **each of the four `(1-gamma)`
factors enter**, so notebook 04's removal of one of them is legible.

## The Error Recursion

Write `Delta_t = Q_t - q*`. Subtract the fixed-point identity
`q* = (1-alpha_t) q* + alpha_t [r + gamma P max q*]`... more precisely, per
cell:

```math
\Delta_{t+1}
=
(1-\alpha_t)\Delta_t
+\alpha_t\gamma\big(\hat P_t\,V_t-P\,V^*\big),
\qquad
V_t=\max_a Q_t,\ V^*=\max_a q^*,
```

where `hat-P_t` is the one-sample empirical operator. Split the driving
term:

```math
\hat P_tV_t-PV^*
=
\underbrace{\hat P_t V_t-\hat P_t V^*}_{\text{(i) contraction part}}
+\underbrace{\hat P_t V^*-P V^*}_{\text{(ii) noise part, centered}} .
```

Part (i) is bounded by `||Delta_t||_inf` (the max-lemma of
`mdp_foundations/03`, applied under the sample). Part (ii) is a
**martingale difference with a fixed function `V*`** — no covering needed
(cf. `concentration_toolkit/04`): its range is `[0, Vmax]` with
`Vmax = Rmax/(1-gamma)`. ← *first* `1/(1-gamma)`: the noise scale.

## Unrolling With Rescaled-Linear Steps

Take `alpha_t = (1 + c(1-gamma)t)^{-1}`-type steps (constants matter here;
pure `1/t` is the trap of notebook 01 — its effective drift is
`(1-gamma)/t`, too weak). Unrolling the recursion:

```math
\|\Delta_T\|_\infty
\ \le\
\underbrace{\prod_t\big(1-(1-\gamma)\alpha_t\big)\,\|\Delta_0\|_\infty}_{\text{bias: geometric in }(1-\gamma)T}
+\gamma\,\Big\|\sum_t w_t^{(T)}\,\big(\hat P_t-P\big)V^*\Big\|_\infty,
```

with weights `w_t <= O(alpha)` summing to `O(1)`. The bias term needs
`T = Omega(1/((1-gamma) alpha))` to die. ← *second* `1/(1-gamma)`: the
contraction is weak, information propagates one effective-horizon per
`1/(1-gamma)` updates.

Azuma–Hoeffding (`concentration_toolkit/02`) on the weighted noise sum, per
cell, union over `|S||A|` cells:

```math
\Big|\sum_t w_t(\hat P_t-P)V^*\Big|
\ \le\
V_{\max}\sqrt{2\textstyle\sum_t (w_t)^2\,\log\tfrac{2|S||A|}{\delta}}
\ \approx\
\frac{R_{\max}}{1-\gamma}\sqrt{\alpha\,\log\tfrac{|S||A|}{\delta}} .
```

Setting this `<= (1-gamma) eps` — it must be, because the `gamma/(1-gamma)`
**geometric-series amplification** of persistent per-step error through the
recursion costs the third factor ← *third* `1/(1-gamma)` — forces

```math
\alpha\ \lesssim\ \frac{(1-\gamma)^4\varepsilon^2}{R_{\max}^2\log(\cdot)}
\qquad\Longrightarrow\qquad
T\ \gtrsim\ \frac{1}{(1-\gamma)\alpha}
\ =\ \tilde\Omega\Big(\frac{1}{(1-\gamma)^5\varepsilon^2}\Big)?
```

Careful bookkeeping (two of the factors share; see Even-Dar–Mansour 2003 /
the modern treatment in AJKS ch. 2) lands the classical answer at

```math
T=\tilde O\Big(\frac{1}{(1-\gamma)^4\,\varepsilon^2}\Big)
```

— and indeed Q-learning with polynomial step sizes is provably **not**
better than `(1-gamma)^{-4}` in the worst case (Li et al. 2021: the naive
algorithm, not the naive analysis, is the bottleneck).

## The Audit Table

```text
factor            source                                       removable?
--------------    ------------------------------------------   -----------------
(1-gamma)^{-1}    noise scale Vmax = Rmax/(1-gamma)            no — intrinsic
(1-gamma)^{-1}    weak drift: T ~ 1/((1-gamma) alpha)          no — intrinsic
(1-gamma)^{-1}    error amplification through the resolvent    no — intrinsic
(1-gamma)^{-1}    Hoeffding pays range Vmax where the actual   YES — Bernstein +
                  per-step variance is much smaller             total variance,
                                                                notebook 04
eps^{-2}          coin-flip statistics                          no (lower bound)
log|S||A|         union bound over cells                        no (needed)
```

Three factors are the problem's; one is the analysis-and-algorithm's. The
next notebook removes exactly that one — and notebook 05 proves the
remaining three cannot go.

## What Remains Open

For *asynchronous* Q-learning (single trajectory, no generative model) the
tight dependence on the exploration policy's minimum visitation probability
was resolved only recently (Li et al. 2021+); step-size schedules that adapt
per-cell (as in practice) still lack matching theory.
