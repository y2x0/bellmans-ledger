# Chernoff, Hoeffding, Bernstein

## The Chernoff Method

For any random variable `X` and any `lambda > 0`, Markov's inequality applied
to the exponential moment:

```math
\Pr\{X\ge t\}
=
\Pr\{e^{\lambda X}\ge e^{\lambda t}\}
\ \le\
e^{-\lambda t}\,\mathbb{E}\big[e^{\lambda X}\big],
```

then optimize over `lambda`. Everything in this file is one computation: a
bound on the moment generating function (mgf), fed through this display. For
sums of independent terms the mgf factorizes,

```math
\mathbb{E}\Big[e^{\lambda\sum_i X_i}\Big]=\prod_i\mathbb{E}\big[e^{\lambda X_i}\big],
```

which is why independence (or, in notebook 02, conditional centering) is the
hypothesis that matters.

## Hoeffding's Lemma

**Lemma.** If `E[X] = 0` and `X in [a, b]` a.s., then

```math
\mathbb{E}\big[e^{\lambda X}\big]\ \le\ \exp\Big(\frac{\lambda^2(b-a)^2}{8}\Big).
```

*Proof.* Let `psi(lambda) = log E[e^{lambda X}]`. Then `psi(0) = 0`,
`psi'(0) = E[X] = 0`, and

```math
\psi''(\lambda)
=
\mathbb{E}_{\lambda}[X^2]-\big(\mathbb{E}_{\lambda}[X]\big)^2
=
\mathrm{Var}_{\lambda}(X),
```

the variance of `X` under the **tilted measure**
`dP_lambda ∝ e^{lambda x} dP`. A random variable supported on `[a, b]` has
variance at most `(b-a)^2/4` under *any* probability measure (its distance
from the midpoint `(a+b)/2` is at most `(b-a)/2` pointwise, so
`Var <= E[(X - (a+b)/2)^2] <= (b-a)^2/4`). Taylor with remainder:

```math
\psi(\lambda)=\psi(0)+\lambda\psi'(0)+\tfrac{\lambda^2}{2}\psi''(\xi)
\ \le\ \frac{\lambda^2(b-a)^2}{8}. \qquad\blacksquare
```

The tilting argument is worth internalizing: mgf bounds are *variance bounds
uniform over exponential tilts*. It reappears verbatim in notebook 05, where
lower bounds tilt measures deliberately.

## Hoeffding's Inequality

**Theorem.** `X_1..X_n` independent, `X_i in [a_i, b_i]`,
`S = sum (X_i - E X_i)`. For all `t >= 0`:

```math
\Pr\{S\ge t\}\ \le\ \exp\Big(\frac{-2t^2}{\sum_i(b_i-a_i)^2}\Big).
```

*Proof.* Chernoff + factorization + the lemma:
`log E[e^{lambda S}] <= lambda^2 sum_i (b_i - a_i)^2 / 8`; minimize
`-lambda t + lambda^2 V/2` (with `V = sum (b_i-a_i)^2/4`) at
`lambda = t/V`, giving `exp(-t^2/2V)`. ∎

Inverted, for i.i.d. terms in `[0, 1]`: with probability `1 - delta`,

```math
\Big|\frac{1}{n}\sum_i X_i-\mu\Big|\ \le\ \sqrt{\frac{\log(2/\delta)}{2n}} .
```

This inversion — from tail bound to confidence radius — is the form RL
consumes: it **is** the UCB1 bonus (`search_and_planning/01`), the UCBVI
bonus (`regret_and_exploration/02`), and the naive Q-learning analysis
(`finite_time_td_q/03`).

## Bernstein's Inequality

Hoeffding charges the full range even when the variance is tiny — exactly
wrong for RL, where near-deterministic transitions are common. Bernstein
repairs it.

**Theorem.** `X_1..X_n` independent, centered, `|X_i| <= M`,
`v = sum_i E[X_i^2]`. Then

```math
\Pr\{S\ge t\}\ \le\ \exp\Big(\frac{-t^2}{2(v+Mt/3)}\Big),
```

equivalently, with probability `1 - delta`,

```math
S\ \le\ \sqrt{2v\log(1/\delta)}\ +\ \frac{M}{3}\log(1/\delta).
```

*Proof of the mgf bound.* For `|x| <= M` and `lambda < 3/M`, expand:

```math
e^{\lambda x}\le 1+\lambda x+\frac{\lambda^2x^2}{2}\sum_{k\ge 0}\Big(\frac{\lambda M}{3}\Big)^k
=1+\lambda x+\frac{\lambda^2 x^2}{2(1-\lambda M/3)},
```

using `x^k <= x^2 M^{k-2}` and `k! >= 2 * 3^{k-2}`. Take expectations
(`E X = 0`), use `1 + u <= e^u`:

```math
\mathbb{E}\big[e^{\lambda X_i}\big]\le\exp\Big(\frac{\lambda^2\,\mathbb{E}[X_i^2]}{2(1-\lambda M/3)}\Big),
```

multiply over `i`, run Chernoff, and choose `lambda = t/(v + Mt/3)`. ∎

**Reading the two-term form:** a `sqrt(variance * log)` leading term plus a
`range * log` lower-order term. When `v << M^2 n`, Bernstein is dramatically
tighter than Hoeffding; when the variance is worst-case (`v ~ M^2 n/4`) they
coincide up to constants.

## Where The Swap Pays In RL (the audit, in advance)

```text
quantity being summed              its variance is small because...
---------------------------------  -------------------------------------
per-step value noise               Var[v*(S')] << Vmax^2 for most (s,a):
  (finite_time_td_q/04)            the law of total variance bounds the
                                   SUM of per-step variances along a
                                   trajectory by Vmax^2, not H * Vmax^2
per-episode regret contribution    same mechanism, episodic form:
  (regret_and_exploration/03)      sum_h Var_h <= H^2 total, not H^3
empirical transition estimates     near-deterministic rows have tiny
  (model_based/02)                 categorical variance
```

Each row is a `(1-gamma)` or `H` factor recovered — the difference between
the naive and the minimax-optimal rates. The mechanism is *never* a smarter
algorithm; it is Bernstein applied where Hoeffding was, plus a variance
recursion.

## What Remains Open Here

Nothing — this file is closed classical material. The open ends live in how
the tools compose with adaptivity (notebook 03's regime) and with function
approximation (notebook 04's regime).
