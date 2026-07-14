# Generalized Advantage Estimation

```text
Schulman et al. 2016 (arXiv:1506.02438); used verbatim in PPO (paper eq. 10–12).
```

## The Estimation Problem

The gradient wants `A_pi(s_t, a_t)`; we have a trajectory and an approximate
critic `V ~ v_pi`. Candidate advantage estimators, indexed by lookahead `n`:

```math
\hat A_t^{(n)}
=
\underbrace{r_t+\gamma r_{t+1}+\cdots+\gamma^{n-1}r_{t+n-1}+\gamma^n V(s_{t+n})}_{n\text{-step return}}
\ -\ V(s_t).
```

```text
n = 1:    bias from V at horizon 1 (large if V is wrong), variance minimal
n = inf:  unbiased (pure MC), variance maximal
```

— the same dial as TD(`lambda`) (`stochastic_approximation/03`), now applied
to the *integrand of the policy gradient* rather than to the critic's own
target.

## TD Errors As Atoms

Define `delta_t = r_t + gamma V(s_{t+1}) - V(s_t)`. Telescoping (same
mechanism as the performance difference lemma proof):

```math
\hat A_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k\,\delta_{t+k}.
```

`n`-step advantage estimates are partial sums of discounted TD errors.

## The Estimator

Exponentially average the `n`-step estimators with weights
`(1-lambda) lambda^{n-1}` (the GAE analogue of the lambda-return):

```math
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
(1-\lambda)\sum_{n=1}^\infty\lambda^{n-1}\hat A_t^{(n)}
=
\sum_{k=0}^{\infty}(\gamma\lambda)^k\,\delta_{t+k},
```

the second equality by exchanging the order of summation (each `delta_{t+k}`
appears in all `A^(n)` with `n > k`, collecting weight
`(1-lambda) sum_{n>k} lambda^{n-1} gamma^k = (gamma lambda)^k`). PPO uses the
truncated version on a length-`T` segment (paper eq. 11):

```math
\hat A_t=\delta_t+(\gamma\lambda)\delta_{t+1}+\cdots+(\gamma\lambda)^{T-t+1}\delta_{T-1},
```

computed in `O(T)` by the backward recursion
`A_t = delta_t + gamma lambda A_{t+1}`.

## Exactness And Bias, Precisely

**If the critic is exact** (`V = v_pi`), every `delta` has conditional mean
zero given `(s_t, a_t)`... except the first — more precisely

```math
\mathbb{E}\big[\hat A_t^{\mathrm{GAE}}\mid s_t,a_t\big]\Big|_{V=v_\pi}
=
A_\pi(s_t,a_t)
\qquad\text{for every }\lambda,
```

because each `E[delta_{t+k} | s_t, a_t] = E[A_pi\text{-terms telescoping to }0]`
for `k >= 1` while `E[delta_t | s_t, a_t] = A_pi(s_t, a_t)`. So with an exact
critic, `lambda` only tunes variance. **With an inexact critic,** each
`delta` carries error `(gamma P - I)(V - v_pi)` and `lambda` tunes how much
of that error is admitted: `lambda = 0` trusts `V` fully at one step;
`lambda = 1` cancels `V` entirely (the telescoped sum is
`G_t - V(s_t)`, MC with baseline — unbiased regardless of the critic).

```text
gamma:   part of the problem statement (which return is optimized) — though
         in practice also used as variance reduction (gamma < 1 even for
         undiscounted tasks: a bias accepted deliberately).
lambda:  pure estimator parameter; changes no fixed point with exact V,
         trades critic-bias against reward-variance with inexact V.
empirical sweet spot: lambda in [0.9, 0.99]; PPO uses 0.95 everywhere.
```

## The Full PPO Data Path (assembled)

```math
\text{critic loss: } (V_\theta(s_t)-V_t^{\text{targ}})^2,
\qquad
V_t^{\text{targ}}=\hat A_t+V_{\theta_{\text{old}}}(s_t)
```

(the "lambda-return" target — the same estimator recycled as the critic's
regression target), and `A-hat` is standardized per batch (mean 0, std 1) in
most implementations: another zero-gradient-in-expectation manipulation
(state-independent affine maps of the advantage are absorbed by the baseline
argument of notebook 02, up to a global scale on the learning rate).
