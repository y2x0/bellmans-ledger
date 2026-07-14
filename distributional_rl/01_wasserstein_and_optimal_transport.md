# Wasserstein Metrics And Optimal Transport Background

## The Metric

For probability measures `F, G` on `R` (identified with their CDFs) and
`p >= 1`:

```math
d_p(F,G)
=
\left(\int_0^1\big|F^{-1}(u)-G^{-1}(u)\big|^p\,du\right)^{1/p},
\qquad
d_\infty(F,G)=\sup_{u\in(0,1)}\big|F^{-1}(u)-G^{-1}(u)\big| ,
```

with `F^{-1}(u) = inf{x : F(x) >= u}` the quantile function. This
inverse-CDF formula is special to one dimension; the general definition is
the optimal transport problem:

```math
W_p(F,G)^p
=
\inf_{\kappa\in\Gamma(F,G)}\ \mathbb{E}_{(X,Y)\sim\kappa}\big[|X-Y|^p\big],
```

the infimum over all **couplings** — joint laws `kappa` with marginals `F`
and `G`. On `R` the optimal coupling is comonotone (sort both samples:
`X = F^{-1}(U)`, `Y = G^{-1}(U)` for a single uniform `U`), which is exactly
why the one-dimensional formula holds.

`W_1` has the Kantorovich–Rubinstein dual form

```math
W_1(F,G)=\sup_{\|f\|_{\mathrm{Lip}}\le 1}\ \Big|\int f\,dF-\int f\,dG\Big| ,
```

and on `R` equals the area between the CDFs: `W_1 = ∫|F(x) - G(x)| dx`.

## Why This Metric And Not KL Or Total Variation

The properties that make `W_p` the right geometry for returns:

**1. It metrizes "how far is the mass," not "how different is the density."**
Two Diracs at nearby points:

```math
W_p(\delta_x,\delta_y)=|x-y|,
\qquad
\mathrm{TV}(\delta_x,\delta_y)=\mathbb{1}\{x\neq y\},
\qquad
\mathrm{KL}(\delta_x\,\|\,\delta_y)=\infty .
```

Return distributions are supported on shifting, scaled point sets (rewards
plus discounted continuations); KL and TV see disjoint supports as maximally
far regardless of numerical closeness. Wasserstein sees the return *values*.

**2. Exact behavior under the two operations the Bellman operator performs.**
Shift by a constant `r` and scale by `gamma`, applied to a random variable:

```math
W_p(r+\gamma X,\ r+\gamma Y)
=
\gamma\,W_p(X,Y)
```

(immediate from the coupling definition: any coupling of `(X, Y)` induces one
of the transformed pair with cost scaled by `gamma`; the shift cancels).
**This single identity is where the `gamma` of the distributional contraction
will come from** — the metric is homogeneous under exactly the affine map
`z -> r + gamma z` that the Bellman operator applies.

**3. Convexity under mixtures (sub-additivity).** For component-wise mixtures
with common weights:

```math
W_p\Big(\sum_i w_i F_i,\ \sum_i w_i G_i\Big)
\ \le\
\sum_i w_i\,W_p(F_i,G_i)
```

(for `p = 1`; for general `p` the analogous inequality holds with `W_p^p`).
The Bellman operator averages over transitions — mixtures — so nonexpansion
under mixing is the second required ingredient.

**4. It controls moments.** Convergence in `W_p` implies convergence of all
moments up to order `p` (plus weak convergence). Contraction in Wasserstein
is therefore strictly more informative than contraction of expectations: the
classical theory (`mdp_foundations/`) is the projection of the
distributional theory onto the first moment.

## The Maximal (Supremal) Form

Distributional value functions assign a return law to every `(s,a)`:
`Z in Z = {S x A -> P(R)}`. The metric is lifted by the supremum:

```math
\bar d_p(Z_1,Z_2)
=
\sup_{s,a}\ d_p\big(Z_1(s,a),\,Z_2(s,a)\big),
```

the distributional analogue of the sup-norm — chosen for the same reason the
classical theory uses `||.||_inf`: state-wise uniform control composes with
the max/argmax operations of control. `(Z-restricted-to-bounded-moments,
d_p-bar)` is a complete metric space, so Banach's theorem
(`mdp_foundations/04`) is available the moment a contraction is exhibited —
notebook 02's job.

## The Estimation Obstruction (foreshadowing notebook 04)

One more property, negative and consequential: the Wasserstein distance is
**not an expectation of a per-sample loss** — for `hat-F_m` an `m`-sample
empirical measure,

```math
\mathbb{E}\big[W_p(\hat F_m,\,G)\big]\ \neq\ W_p(F,G)
\quad\text{in general (biased, and the bias vanishes slowly)},
```

so SGD on sampled transitions cannot minimize Wasserstein loss unbiasedly.
The theory of notebook 02 is stated in `W_p`; the *algorithm* of notebook 04
trains in KL against a projected target; the gap is genuine, and notebook 05
is about the two ways it was later closed (quantile parameterizations, and
the Cramér distance).
