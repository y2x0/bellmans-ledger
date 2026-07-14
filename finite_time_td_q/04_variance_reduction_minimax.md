# The Bernstein Repair And The Minimax Rate

## Where Hoeffding Was Wasteful

Notebook 03's noise term paid the *range* `Vmax = Rmax/(1-gamma)` for the
random variable `V*(s')`. Its actual variance,

```math
\sigma_{V^*}^2(s,a)=\mathrm{Var}_{s'\sim P(\cdot\mid s,a)}\big[V^*(s')\big],
```

is usually far smaller — and, crucially, it cannot be large *everywhere
along a trajectory*. That is the content of:

## The Law-of-Total-Variance Lemma

**Lemma.** For any policy `pi` and any `(s,a)`:

```math
\Big\|\ (I-\gamma^2 P_\pi)^{-1}\,\sigma^2_{V_\pi}\ \Big\|_\infty
\ \le\
\frac{\mathrm{Var}\big[\text{return}\big]_{\max}}{1}
\ \le\
\frac{V_{\max}^2}{1},
\qquad\text{hence}\qquad
\Big\|\sum_{t}\gamma^{2t}P_\pi^t\,\sigma^2_{V_\pi}\Big\|_\infty\le V_{\max}^2 .
```

*Proof.* Apply the law of total variance to the return `G` from `(s,a)`
conditionally on the first transition, recursively:

```math
\mathrm{Var}[G\mid s,a]
=
\underbrace{\mathbb{E}\big[\mathrm{Var}[G\mid s']\big]\gamma^2}_{\text{future}}
+\underbrace{\mathrm{Var}_{s'}\big[\mathbb{E}[G\mid s']\big]}_{=\ \gamma^2\sigma_{V}^2(s,a)}\ (+\ \text{reward noise}),
```

unroll: the *discounted sum of per-step value variances along the
trajectory* equals the variance of the return, which is bounded by
`Vmax^2` — **once**, not once per step. ∎

```text
the moral: sum_t gamma^{2t} sigma^2_t <= Vmax^2,  NOT  H * Vmax^2.
per-step variances are a budget shared along the trajectory; charging each
step the worst case (Hoeffding) overcounts by a full horizon factor.
```

## The Repair, Assembled

Replace Azuma by **Freedman/Bernstein** (`concentration_toolkit/02`) in
notebook 03's noise sum: the per-cell deviation becomes

```math
\tilde O\Big(\sqrt{\frac{\sigma^2_{V^*}(s,a)}{N}}+\frac{V_{\max}}{N}\Big)
\quad\text{instead of}\quad
\tilde O\Big(\sqrt{\frac{V_{\max}^2}{N}}\Big),
```

and when this cellwise error is propagated through the resolvent
`(I - gamma P_{pi*})^{-1}`, Cauchy–Schwarz against the total-variance lemma
converts `resolvent * sqrt(variance)` into
`sqrt(resolvent-of-variance) * sqrt(horizon)`:

```math
\Big\|(I-\gamma P)^{-1}\sqrt{\sigma^2_{V^*}}\Big\|_\infty
\le
\sqrt{\frac{1}{1-\gamma}}\,\Big\|\sqrt{(I-\gamma^2 P)^{-1}\sigma^2_{V^*}}\Big\|_\infty
\le
\frac{V_{\max}}{\sqrt{1-\gamma}}\cdot\sqrt{\tfrac{1}{1-\gamma}} ,
```

— one factor of `1/(1-gamma)` where notebook 03 paid two. Net sample
complexity per cell:

```math
N\ =\ \tilde O\Big(\frac{1}{(1-\gamma)^3\,\varepsilon^2}\Big)
\qquad\Longrightarrow\qquad
\tilde O\Big(\frac{|S||A|}{(1-\gamma)^3\,\varepsilon^2}\Big)\ \text{total} .
```

## Who Achieves It

**Model-based (Azar–Munos–Kappen 2013; Agarwal–Kakade–Yang 2020).** Build
the empirical MDP `hat-P` from `N` samples per cell; solve it *exactly*
(`mdp_foundations/`). The analysis is the Bernstein-plus-total-variance
argument above applied to `hat-P` vs `P`, with one extra subtlety: `V*` of
the empirical MDP is not independent of the samples — handled either by an
`s`-absorbing leave-one-out construction (AKY) or by covering. Result:
**plain certainty equivalence is minimax optimal** for
`eps <= 1/sqrt(1-gamma)` — sophistication buys constants only
(`model_based/02` takes this up).

**Model-free (Wainwright 2019).** Q-learning itself is stuck at
`(1-gamma)^{-4}` (notebook 03's closing remark), but **variance-reduced
Q-learning** — SVRG-style: anchor at a reference `Q-bar`, re-estimate
`T Q-bar` with a large batch, run cheap updates on the *difference*
(whose noise scales with `||Q - Q-bar||`, shrinking every epoch) — matches
`(1-gamma)^{-3}`. The variance-reduction epochs implement, in SA form, the
same "charge the variance once" accounting that the model-based proof does
analytically.

```text
                 naive Q-learning    VR Q-learning     model-based CE
samples/cell     (1-g)^{-4} eps^-2   (1-g)^{-3}eps^-2  (1-g)^{-3} eps^-2
matching LB?     no (alg. is loose)  yes               yes
```

## What This Family Keeps Re-Learning

The Hoeffding→Bernstein swap is *never* free: it requires (i) a variance
that is genuinely smaller than range² — supplied by the total-variance
lemma, a fact about MDP structure, not probability; and (ii) handling the
dependence between the variance being estimated and the data — supplied by
leave-one-out or epoch-anchoring devices. Every tight bound in
`regret_and_exploration/03` and `model_based/02` is this same pair of moves
wearing different clothes.

## What Remains Open

The full `eps`-range: minimax optimality of model-based CE breaks for very
coarse `eps ~ Vmax` (known gaps at the "burn-in" scale); non-asymptotic
theory for the *asynchronous, single-trajectory* variance-reduced variants
is incomplete.
