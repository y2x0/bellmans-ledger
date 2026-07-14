# Uniform Convergence And Covering Numbers

## Why RL Needs Uniformity Twice Over

A pointwise bound controls one pre-specified function. RL needs bounds that
hold for functions **chosen after seeing the data**, in two distinct ways:

```text
1. the algorithm optimizes over a class (the fitted Q, the optimistic value)
   — the selected function is data-dependent;
2. the target of a Bellman backup is ITSELF a data-dependent function
   (the previous iterate), so even evaluating "the" backup error requires
   a bound uniform over plausible previous iterates.
```

The second is the subtle one, and it is exactly the step in the UCBVI and
LSVI-UCB proofs where covering numbers enter
(`regret_and_exploration/02`, `linear_mdps_and_completeness/02`).

## Finite Classes: The Union Bound Baseline

For a finite class `F` of functions into `[0, 1]` and i.i.d. data,
Hoeffding + union bound: with probability `1 - delta`, for all `f in F`,

```math
\Big|\frac1n\sum_i f(X_i)-\mathbb{E}f\Big|
\ \le\
\sqrt{\frac{\log(2|F|/\delta)}{2n}} .
```

The template: *pointwise bound + log-cardinality*. Everything below is a
scheme for giving an infinite class a finite effective cardinality.

## Covering Numbers And Epsilon-Nets

An `eps`-net of `(F, ||.||)` is a finite `F_eps` with every `f in F` within
`eps` of some element; the covering number `N(F, eps, ||.||)` is the
smallest such cardinality.

**Uniform bound via a net (the two-step decomposition).** For `f` with net
representative `f_eps`, in sup-norm:

```math
\Big|\frac1n\sum f(X_i)-\mathbb{E}f\Big|
\le
\underbrace{\Big|\frac1n\sum f_\varepsilon(X_i)-\mathbb{E}f_\varepsilon\Big|}_{\text{union bound over }N(\varepsilon)}
+\ 2\varepsilon .
```

With probability `1 - delta`, uniformly over `F`:

```math
\sup_{f\in F}\Big|\frac1n\sum f(X_i)-\mathbb{E}f\Big|
\ \le\
\sqrt{\frac{\log\big(2N(F,\varepsilon)/\delta\big)}{2n}}+2\varepsilon,
```

then optimize `eps` (typically `eps ~ 1/n`, making the net cost logarithmic).

## The Two Covering Computations RL Actually Uses

**1. Balls in `R^d` (parametric classes).** A Euclidean ball of radius `R`
has

```math
N(\varepsilon)\ \le\ \Big(1+\frac{2R}{\varepsilon}\Big)^{d}
\qquad\Longrightarrow\qquad
\log N=O\Big(d\,\log\frac{R}{\varepsilon}\Big),
```

*proved by the volumetric argument*: a maximal `eps`-separated set has
disjoint `eps/2`-balls inside the `(R + eps/2)`-ball; compare volumes. If a
function class is `L`-Lipschitz in a `d`-dimensional parameter, its sup-norm
covering number inherits `log N = O(d log(LR/eps))`. This is how the class

```math
\mathcal{V}=\Big\{\ s\mapsto\min\big(\max_a\ \phi(s,a)^\top w+\beta\,\|\phi(s,a)\|_{V^{-1}},\ H\big)\ :\ \|w\|\le W,\ \beta\le B,\ V\succeq\lambda I\ \Big\}
```

— the optimistic value functions of LSVI-UCB, parameterized by
`(w, beta, V)`, hence dimension `O(d^2)` — gets its
`log N = O(d^2 log(.))`, which is the origin of the extra `d` in that
algorithm's `sqrt(d^3 ...)` regret. The covering step is not bookkeeping; it
visibly costs a dimension factor.

**2. Monotone/bounded scalar families.** Classes like
`{clip(x - c, 0, H) : c in R}` have `N(eps) = O(H/eps)`: one-dimensional
grids suffice. Used when covering over scalar bonuses or thresholds.

## Uniform Bellman-Error Bound (the composed statement)

The form actually invoked downstream — for transition estimation: with
probability `1 - delta`, for **all** `v in V` (a class with covering number
`N`) and all `(s, a)` with visit count `n(s,a)`:

```math
\big|(\hat P-P)(\cdot\mid s,a)^\top v\big|
\ \le\
\sqrt{\frac{2\,\mathrm{Var}_{P}(v)\,\log\frac{2N\cdot SA\cdot n_{\max}}{\delta}}{n(s,a)}}
+\frac{H\,\log\frac{\cdot}{\delta}}{3\,n(s,a)}\ +\ 2\varepsilon\ \text{terms},
```

assembled from: Bernstein (01) per `(s,a,v)`-triple, the union bound over
the net and over `(s,a)` and counts, and the net approximation error. Every
hypothesis is now visible: boundedness feeds Bernstein's `M`; adaptive
counts are legitimized by notebook 02's optional-stopping remark; the class
size enters only through `log N`.

**The alternative that avoids covering:** if `v` were *independent* of the
data used in `hat-P` — e.g. by data splitting across value-iteration rounds
— pointwise Bernstein suffices. Splitting wastes samples;
covering wastes a `log N`. Modern proofs choose per-context; UCBVI-style
proofs cover, Azar-style generative-model proofs exploit the *fixed* `v*`
(only the optimal value needs the bound — the "leave-one-out" trick of
`finite_time_td_q/04`).

## What Remains Open

For deep networks, sup-norm covering numbers are astronomically large and
these bounds are vacuous; the honest statement is that uniform convergence
as presented here explains linear/kernel RL and does **not** explain deep
RL's generalization — same gap as in supervised learning theory, inherited
wholesale (`linear_mdps_and_completeness/06`).
