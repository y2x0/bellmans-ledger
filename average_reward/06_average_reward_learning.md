# Average-Reward Learning

## The Two-Unknown Problem

Discounted TD learns one object (`v` or `q`). The Poisson equation
(notebook 02) has **two** unknowns, `(rho, h)`, coupled: the
**differential TD error**

```math
\delta_t
=
R_{t+1}-\hat\rho_t+\hat h(S_{t+1})-\hat h(S_t)
```

has mean zero (given `S_t`, at the solution) only if `rho-hat` is
correct; and `rho` can only be estimated from data whose distribution the
current policy determines. Every algorithm below is a scheme for
estimating both without letting their errors contaminate each other.

## Differential TD(0)

```math
\hat h(S_t)\ \leftarrow\ \hat h(S_t)+\alpha_t\,\delta_t,
\qquad
\hat\rho\ \leftarrow\ \hat\rho+\beta_t\,\delta_t
```

(the second update: the TD error's stationary mean is
`mu^T(r - rho 1) = rho_true - rho-hat` — the TD error itself is an
unbiased signal for the gain error; using it, rather than a raw reward
average, makes the `rho`-estimator variance-reduced by the learned `h`,
a control-variate effect).

**Convergence (tabular, unichain).** Cast as two-timescale stochastic
approximation (`stochastic_approximation/01`) with `beta_t / alpha_t ->
0` (gain slow) or, in Wan–Naik–Sutton's formulation, single-timescale
with the ODE argument done jointly:

```text
fast ODE (h, at frozen rho):   h-dot = T^pi h - h - rho 1 —
                               a span contraction flow (notebook 05's
                               modulus under mixing): h converges to the
                               Poisson solution for the frozen rho, up to
                               the constant drift;
slow ODE (rho):                rho-dot = (stationary mean of delta) =
                               rho_true - rho: exponentially stable
                               scalar flow.
```

The span quotient handles the constant-direction non-uniqueness — the
Lyapunov function is `||h - h*||_sp + |rho - rho*|`; Borkar's
two-timescale theorem then gives a.s. convergence under the usual step
and visitation conditions. **Load-bearing:** unichain (else `rho` is a
vector and the scalar estimator is chasing an average of incompatible
gains); mixing enters the rate through the span modulus, exactly as
notebook 05 promised.

## Differential Q-Learning / R-Learning

Control version:

```math
\hat q(S_t,A_t)\ \leftarrow\ \hat q(S_t,A_t)+\alpha_t\Big[R_{t+1}-\hat\rho+\max_{a'}\hat q(S_{t+1},a')-\hat q(S_t,A_t)\Big],
```

with the same coupled `rho`-update (R-learning, Schwartz 1993; the
convergent modern form Wan–Naik–Sutton 2021, which fixed the
normalization drift that left earlier variants only empirically
supported). Watkins-style convergence (`stochastic_approximation/05`)
transfers with the span machinery replacing the sup-norm contraction —
under (all of): unichain, sufficient exploration, standard steps.

## What Practice Does Instead, And The Cost

```text
practice:  run discounted algorithms with gamma = 0.99–0.999 on
           continuing tasks.
cost:      by the Laurent series (notebook 03), the objective optimized
           is rho + (1-gamma) h + O((1-gamma)^2): a gain objective
           perturbed by (1-gamma) * span(h). Fine when the mixing scale
           << 1/(1-gamma); silently wrong otherwise — and, worse, the
           VARIANCE of discounted estimators blows up as
           1/(1-gamma) while the differential methods' constants carry
           the mixing time instead. On slowly-mixing problems both hurt;
           on fast-mixing problems with long episodes, differential
           methods are strictly the better-conditioned parameterization
           (subtracting rho recenters every target at O(span(h)) instead
           of O(rho/(1-gamma))).
```

This recentering is why average-reward ideas keep re-entering deep RL
through the back door: reward centering / PopArt-style normalization
(`value_based_deep_rl/04` failure 4's fix) is a moving estimate of `rho`
subtracted from targets — differential TD's first half, adopted without
its theory.

## Average-Reward Regret, One Paragraph

The online version (UCRL2-style, `model_based/03`): optimism over models
with confidence sets, achieving `tilde-O(D S sqrt(A T))` regret with `D`
the diameter — the average-reward analogue of
`regret_and_exploration/02–03`, with `D` in the role the horizon/mixing
constants play throughout this family, and a matching
`Omega(sqrt(DSAT))` lower bound via the usual gadgets
(`concentration_toolkit/05`). The open gap between `D` and the tighter
span-of-bias constants (`span(h*) <= D * r_max`, sometimes much smaller)
is the frontier: bounds in `span(h*)` exist but with worse `S`-dependence.

## What Remains Open

Function approximation for `(rho, h)` jointly (the projected Poisson
equation's theory is embryonic — the average-reward TvR analogue is
partial); bias-optimal (notebook 04) sample-based control: essentially
untouched; and closing the `D`-vs-`span(h*)` regret gap.
