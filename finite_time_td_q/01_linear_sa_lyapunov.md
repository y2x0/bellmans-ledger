# Finite-Time Linear Stochastic Approximation

## The Master Iteration

```math
\theta_{t+1}=\theta_t+\alpha_t\big(b-A\theta_t+M_{t+1}\big),
\qquad
\mathbb{E}[M_{t+1}\mid\mathcal{F}_t]=0,
\quad
\mathbb{E}\big[\|M_{t+1}\|^2\mid\mathcal{F}_t\big]\le\sigma^2+\kappa\|\theta_t-\theta^*\|^2,
```

with `theta* = A^{-1} b`. Standing assumption (the "Hurwitz" condition):
`A` is positive definite in the sense

```math
x^\top A x\ \ge\ \mu\,\|x\|^2\quad\forall x
\qquad(\mu>0),
```

**without symmetry** — this is the fixed-point, not gradient-descent,
setting (`stochastic_approximation/04` showed TD's `A` is exactly of this
kind, with `mu = (1-gamma) lambda_min(Sigma)` on-policy). Let
`L = ||A||` be the operator norm.

## Constant Step Size: Geometric Decay To A Noise Ball

**Theorem.** For `alpha <= mu / (2L^2 + 2kappa)` (constant), the error
`e_t = theta_t - theta*` satisfies

```math
\mathbb{E}\|e_t\|^2
\ \le\
(1-\alpha\mu)^{t}\,\|e_0\|^2
+\frac{\alpha\,\sigma^2}{\mu}.
```

*Proof.* Expand the squared error, take conditional expectation (the
martingale cross-term dies):

```math
\mathbb{E}\big[\|e_{t+1}\|^2\mid\mathcal{F}_t\big]
=
\|e_t\|^2-2\alpha\,e_t^\top A e_t+\alpha^2\big(\|Ae_t\|^2+\mathbb{E}\|M_{t+1}\|^2\big)
```

```math
\le\ \|e_t\|^2\big(1-2\alpha\mu+\alpha^2L^2+\alpha^2\kappa\big)+\alpha^2\sigma^2
\ \le\ (1-\alpha\mu)\,\|e_t\|^2+\alpha^2\sigma^2,
```

the last step by the step-size restriction. Unroll the linear recursion and
sum the geometric series `sum_k (1-alpha mu)^k alpha^2 sigma^2 <=
alpha sigma^2 / mu`. ∎

**Load-bearing audit.**

```text
positive definiteness of A   -> the -2 alpha mu ||e||^2 drift term: without
                                it (off-policy TD) the recursion has no
                                negative drift and the argument collapses —
                                Baird, quantitatively
martingale noise             -> kills the cross term; if the noise were
                                biased the ball would inflate by bias/mu
step restriction             -> the alpha^2 curvature/variance terms must
                                not overwhelm the drift
```

This is `stochastic_approximation/01`'s "hover in an O(sqrt(alpha)) ball"
made exact: radius `sqrt(alpha sigma^2 / mu)` — halve the step, gain
`sqrt(2)` accuracy, halve the speed.

## Decaying Steps: The O(1/t) Rate

**Theorem.** With `alpha_t = 2/(mu (t + t_0))`, `t_0 = 4L^2/mu^2 + 4kappa/mu^2`:

```math
\mathbb{E}\|e_t\|^2\ \le\ \frac{C_1\,\|e_0\|^2\,t_0^2}{(t+t_0)^2}+\frac{C_2\,\sigma^2}{\mu^2\,(t+t_0)} .
```

*Proof sketch.* The same one-step inequality now reads
`u_{t+1} <= (1 - 2/(t+t_0)) u_t + 4 sigma^2 / (mu^2 (t+t_0)^2)` with
`u_t = E||e_t||^2`; induct on the claim `u_t <= c/(t+t_0)` — the
`(1 - 2/(t+t_0))` factor beats the `1/(t+t_0) -> 1/(t+1+t_0)` shift
precisely because the decay exponent `2 > 1` (this is why the step must be
`~ 2/mu t` and not `~ 1/(2 mu t)`: steps too small relative to `1/mu`
give `O(t^{-2 alpha mu})`, a *slower-than-1/t* rate — the classic
misspecified-step trap). ∎

```text
the trap, explicitly: alpha_t = c/t with c < 1/(2 mu) gives
E||e_t||^2 = Theta(t^{-2 c mu}) — arbitrarily slow. Robbins–Monro
conditions (stochastic_approximation/01) are indifferent to c; RATES are
not. This is the first place the asymptotic and finite-time theories
genuinely part ways.
```

## Markov Noise: What Changes

If the noise is not an MDS but comes from a mixing Markov chain (the actual
TD setting — consecutive samples share the chain's autocorrelation), the
cross-term `E[e_t^T M_{t+1}]` no longer vanishes; it is controlled by
coupling the chain to its stationary distribution:

```math
\big|\mathbb{E}[e_t^\top M_{t+1}]\big|
\ \le\
C\,\alpha_{t-\tau}\ \tau_{\mathrm{mix}}\,(\ldots),
```

by conditioning `tau_mix = tau_mix(alpha)` steps back (the chain forgets;
the parameters moved only `O(alpha tau_mix)` meanwhile). The net effect
(Bhandari–Russo–Singal; Srikant–Ying): **all bounds above survive with
`sigma^2` and constants inflated by a factor `tau_mix log(1/alpha)`**.
Mixing time is a genuine statistical price, not an artifact: slowly mixing
chains deliver fewer effective samples per step.

## What Remains Open

High-probability (not just in-expectation) bounds with optimal constants
under Markov noise are still being sharpened; and none of this file
survives nonlinear function approximation — there `A` becomes a
state-dependent Jacobian and only local/NTK-regime analogues exist.
