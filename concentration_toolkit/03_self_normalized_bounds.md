# Self-Normalized Bounds And Adaptive Regression

## The Problem Only RL And Bandits Have

Ridge regression theory assumes covariates fixed or i.i.d. In bandits/RL the
covariate at time `t` — the feature `phi_t` of the chosen action — is chosen
**by the algorithm based on past noise**. The estimation error must be
controlled in a data-dependent norm, uniformly over time, under adaptive
design. The right tool bounds the noise process *normalized by its own
accumulated design matrix* — hence "self-normalized."

## Setup

`(F_t)` a filtration; `phi_t` is `F_{t-1}`-measurable (the action is chosen
before the noise); noise `eta_t` is `F_t`-measurable, conditionally centered
and `sigma`-sub-Gaussian:

```math
\mathbb{E}\big[e^{\lambda\eta_t}\mid\mathcal{F}_{t-1}\big]\le e^{\lambda^2\sigma^2/2}
\quad\forall\lambda .
```

Define the noise-weighted feature sum and the regularized Gram matrix:

```math
S_t=\sum_{u=1}^t \eta_u\,\phi_u,
\qquad
\bar V_t=\lambda_{\mathrm{reg}} I+\sum_{u=1}^t\phi_u\phi_u^\top .
```

## The Theorem (Abbasi-Yadkori–Pál–Szepesvári 2011)

With probability at least `1 - delta`, **simultaneously for all** `t >= 0`:

```math
\|S_t\|_{\bar V_t^{-1}}^2
\ \le\
2\sigma^2\,\log\Big(\frac{\det(\bar V_t)^{1/2}\,\det(\lambda_{\mathrm{reg}}I)^{-1/2}}{\delta}\Big).
```

## Proof: The Method Of Mixtures

**Step 1 — a supermartingale per direction.** For any fixed `x in R^d`,

```math
M_t^x=\exp\Big(\frac{x^\top S_t}{\sigma}-\frac{1}{2}\,x^\top\Big(\sum_{u\le t}\phi_u\phi_u^\top\Big)x\Big)
```

is a supermartingale with `E[M_t^x] <= 1`: peel the last term as in
notebook 02; conditionally, `x^T phi_t eta_t / sigma` is sub-Gaussian with
variance proxy `(x^T phi_t)^2`, and the subtracted quadratic exactly pays
for it. (Same engine as Freedman: exponential of the sum minus its variance
process.)

**Step 2 — mix over directions.** Integrate `x` against a Gaussian
`N(0, (lambda_reg)^{-1} I)`:
`M_t = E_{x}[M_t^x]` is still a supermartingale with `E[M_t] <= 1` (Fubini +
Step 1). The Gaussian integral is explicit — completing the square:

```math
M_t
=
\frac{\det(\lambda_{\mathrm{reg}}I)^{1/2}}{\det(\bar V_t)^{1/2}}\,
\exp\Big(\frac{1}{2\sigma^2}\,\|S_t\|^2_{\bar V_t^{-1}}\Big).
```

**Step 3 — Markov + optional stopping.** `Pr{sup_t M_t >= 1/delta} <= delta`
(Ville's inequality for supermartingales), and rearranging the display gives
the theorem, anytime. ∎

The mixture replaces the union bound over directions and over time — no
epsilon-net, no peeling — which is why the constants are clean and the bound
is *anytime for free*. This is the third appearance of the
exponential-supermartingale device; after this file it should feel like one
tool, not three.

## Corollary: Confidence Ellipsoids For Adaptive Ridge Regression

Observe `y_t = phi_t^T theta* + eta_t`; the ridge estimator
`theta_t = V_t^{-1} sum phi_u y_u` (with `V_t = bar-V_t`). Decompose:

```math
\hat\theta_t-\theta^*=\bar V_t^{-1}S_t-\lambda_{\mathrm{reg}}\bar V_t^{-1}\theta^*,
```

whence, with probability `1 - delta`, for all `t`:

```math
\|\hat\theta_t-\theta^*\|_{\bar V_t}
\ \le\
\sigma\sqrt{2\log\frac{\det(\bar V_t)^{1/2}}{\delta\,\lambda_{\mathrm{reg}}^{d/2}}}
\ +\ \sqrt{\lambda_{\mathrm{reg}}}\,\|\theta^*\|
\ \ =:\ \beta_t(\delta),
```

and `det(bar-V_t) <= (lambda_reg + t L^2/d)^d` for `||phi|| <= L` gives
`beta_t = O(sigma sqrt(d log(t/delta)))`. Prediction error at any `phi`:

```math
|\phi^\top(\hat\theta_t-\theta^*)|
\ \le\
\beta_t(\delta)\,\|\phi\|_{\bar V_t^{-1}} .
```

`||phi||_{V^{-1}}` — "how novel is this direction given the data seen" — is
the exploration bonus of LinUCB and of LSVI-UCB
(`linear_mdps_and_completeness/02`). It is the function-approximation
generalization of `1/sqrt(N(s,a))`: for one-hot features the two coincide
exactly.

## The Elliptical Potential Lemma (the companion)

Regret proofs also need: the bonuses cannot stay large forever.

```math
\sum_{t=1}^{n}\min\big(1,\ \|\phi_t\|^2_{\bar V_{t-1}^{-1}}\big)
\ \le\
2\log\frac{\det(\bar V_n)}{\det(\lambda_{\mathrm{reg}}I)}
\ \le\ 2d\,\log\Big(1+\frac{nL^2}{d\lambda_{\mathrm{reg}}}\Big).
```

*Proof.* `det(bar-V_t) = det(bar-V_{t-1}) (1 + ||phi_t||^2_{V_{t-1}^{-1}})`
(matrix determinant lemma); telescope the logs and use
`min(1, u) <= 2 log(1 + u)`. ∎

Confidence widths shrink because determinants grow — the exact
generalization of "each pull shrinks the interval" from
`search_and_planning/01`, with `d log n` replacing `K` as the effective
number of things to learn.

## What Remains Open

Sharp self-normalized bounds beyond sub-Gaussian noise (heavy tails), and
the correct analogue for nonlinear function classes — the latter is exactly
the eluder-dimension frontier of `regret_and_exploration/07`.
