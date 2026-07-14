# Quantile Regression And The Cramér Geometry

Two responses to C51's theory–practice gap (train in KL, prove in
Wasserstein): change the parameterization so Wasserstein *can* be minimized
by SGD (QR-DQN), or change the metric to one whose sampled loss is unbiased
(Cramér). Both are one design-cell away from C51 in the same taxonomy.

## Transposing The Parameterization

```text
C51:     fixed locations z_i,     learned probabilities p_i(s,a)
QR-DQN:  fixed probabilities 1/N, learned locations theta_i(s,a)
```

(Dabney et al. 2017.) Represent `Z(s,a) = (1/N) sum_i delta_{theta_i(s,a)}`
— `N` equally weighted, movable atoms. The best such approximation to a
target `F` in `W_1` places atom `i` at the quantile midpoint:

```math
\theta_i^*=F^{-1}(\hat\tau_i),
\qquad
\hat\tau_i=\frac{2i+1}{2N}\ \ (i=0,\ldots,N-1),
```

(each atom carries mass `1/N` over the quantile interval
`[i/N, (i+1)/N]`; the `W_1`-optimal representative of that interval of the
quantile function is its midpoint value). So learning the distribution
reduces to learning `N` **quantiles**.

## The Loss That Makes It Work

Quantile regression: the `tau`-quantile of `F` is the minimizer of the
asymmetric absolute loss

```math
\rho_\tau(u)=u\,\big(\tau-\mathbb{1}\{u<0\}\big),
\qquad
F^{-1}(\tau)=\arg\min_q\ \mathbb{E}_{X\sim F}\big[\rho_\tau(X-q)\big].
```

**This expectation is over single samples of `X`** — an unbiased per-sample
loss for a quantile target. Summing over the `N` quantile levels with sampled
Bellman targets `X = r + gamma theta_j(s', a*)`:

```math
L(\theta)
=
\sum_{i=1}^{N}\ \mathbb{E}_{j}\Big[\rho_{\hat\tau_i}\big(r+\gamma\,\theta_j(s',a^*)-\theta_i(s,a)\big)\Big],
```

(in practice the Huberized `rho` for smooth gradients). The estimation
obstruction of notebook 01 is dissolved **by the parameterization**: for
fixed-probability atoms, minimizing `W_1` decomposes into `N` independent
quantile regressions, each SGD-compatible. QR-DQN's projected evaluation
operator is a contraction in `d_inf-bar`, the fixed point is the `W_1`-best
`N`-quantile approximation, and — no `[V_min, V_max]` hyperparameters, since
locations float. Empirically it improved on C51; IQN (sample `tau ~ U[0,1]`
and learn the full quantile *function* `theta(s, a, tau)`) improved again and
makes risk-sensitive control available by reweighting `tau` (any distortion
risk measure is an integral of quantiles).

## The Cramér Alternative

The other repair keeps C51's parameterization and fixes the metric. The
Cramér distance ( = squared-`L2` between CDFs):

```math
\ell_2^2(F,G)=\int_{\mathbb{R}}\big(F(x)-G(x)\big)^2\,dx .
```

Properties, against notebook 01's checklist:

```text
1. horizontal metric: sees mass location, like W_1 (which is the L1-between-
   CDFs analogue);
2. contraction: the distributional T^pi is a gamma^{1/2}-contraction in the
   maximal Cramér metric — weaker modulus, still Banach-compatible;
3. UNBIASED sample gradients: l_2^2 is an expectation of a per-sample
   kernel (it is an energy distance / MMD with kernel -|x-y|), so SGD
   minimizes the true population distance — the property W_p lacks.
```

Rowland et al. 2018 then closed C51's own circle: C51's heuristic
projection-plus-KL update converges, and its natural analysis lives in
Cramér geometry — the algorithm was implicitly a Cramér-space method all
along, discovered before its metric.

## The Cell Structure (unifying view of this family)

```text
                 locations        probabilities     trained in        proved in
C51              fixed grid       learned softmax   KL o projection   Cramér (post hoc)
QR-DQN           learned          fixed 1/N         quantile loss     W1 / d_inf
IQN              learned fn of    sampled tau       quantile loss     W1 (per-sample)
                 tau
```

Every cell answers the family README's question the same way: evaluation
contracts in a horizontal metric; control inherits only its mean guarantees;
the parameterization decides which metric admits unbiased stochastic
gradients — and that, not the contraction theory, is what selects the
algorithm.
