# TD With Linear Function Approximation: The Tsitsiklis–Van Roy Theorem

## Setup

Approximate `v_pi ~ V_w = Phi w` with feature matrix
`Phi in R^{|S| x d}`, full column rank, `d << |S|`. TD(0) becomes

```math
w_{t+1}=w_t+\alpha_t\,\delta_t\,\phi(S_t),
\qquad
\delta_t=R_{t+1}+\gamma\,\phi(S_{t+1})^\top w_t-\phi(S_t)^\top w_t .
```

## The Geometry: Projected Bellman Equation

The Bellman image `T^pi Phi w` generally leaves the feature subspace, so no
`w` solves `Phi w = T^pi Phi w`. Define the `mu`-weighted projection onto the
subspace (`D = diag(mu)`):

```math
\Pi_\mu=\Phi(\Phi^\top D\Phi)^{-1}\Phi^\top D ,
```

and ask instead for the fixed point of the **projected operator**:

```math
\Phi w^\* = \Pi_\mu\,\mathcal{T}^\pi\,\Phi w^\* .
```

## The Key Lemma: On-Policy L2 Nonexpansion

Let `mu` be the **stationary distribution** of `P_pi` (`mu^T P_pi = mu^T`).
Then `P_pi` is nonexpansive in `L2(mu)`:

```math
\|P_\pi v\|_\mu^2
=\sum_s\mu(s)\Big(\sum_{s'}P_\pi(s'\mid s)v(s')\Big)^2
\le\sum_s\mu(s)\sum_{s'}P_\pi(s'\mid s)v(s')^2
=\sum_{s'}\Big(\underbrace{\textstyle\sum_s\mu(s)P_\pi(s'\mid s)}_{=\,\mu(s')}\Big)v(s')^2
=\|v\|_\mu^2,
```

by Jensen and then **stationarity** — the starred step is where
`mu = mu P_pi` is used and is unavailable for any other `mu`. Hence `T^pi` is
a `gamma`-contraction in `||.||_mu`, and since orthogonal projections are
nonexpansive, so is the composition:

```math
\|\Pi_\mu\mathcal{T}^\pi u-\Pi_\mu\mathcal{T}^\pi v\|_\mu\le\gamma\|u-v\|_\mu .
```

`Pi T^pi` has a unique fixed point `Phi w*`. **Everything hinges on sampling
states from the stationary distribution of the policy being evaluated.**

## The Theorem

**(Tsitsiklis & Van Roy 1997.)** For an ergodic chain, features of full rank,
Robbins–Monro steps, and states sampled from the on-policy trajectory:
`w_t -> w*` a.s., where `Phi w*` is the fixed point of `Pi_mu T^pi`, and

```math
\|\Phi w^\*-v_\pi\|_\mu
\ \le\
\frac{1}{\sqrt{1-\gamma^2}}\,\|\Pi_\mu v_\pi-v_\pi\|_\mu .
```

TD converges to within a constant factor of the *best* the feature class can
do; the factor blows up as `gamma -> 1`. (The error-bound proof: Pythagoras
in `L2(mu)` with `Phi w* - Pi v_pi` orthogonal to the approximation residual,
plus contraction.)

## The Mean Dynamics: Why It Is Not Gradient Descent

The expected update is affine, `E[Delta w] = alpha (b - A w)`:

```math
A=\Phi^\top D(I-\gamma P_\pi)\Phi,
\qquad
b=\Phi^\top D\,r_\pi .
```

Convergence of the ODE `w-dot = b - Aw` requires `A` to be **positive
definite** (not symmetric — this is a fixed-point iteration, not a descent
method). On-policy `mu` makes `A` positive definite via the key lemma:

```math
x^\top A x=\|\bar x\|_\mu^2-\gamma\,\langle \bar x, P_\pi \bar x\rangle_\mu
\ \ge\ (1-\gamma)\|\bar x\|_\mu^2>0,
\qquad \bar x=\Phi x .
```

Off-policy `mu` can make `A` indefinite — eigenvalues with negative real
part — and then the *mean* dynamics diverge: no amount of step-size tuning or
extra samples helps. That is Baird's counterexample, notebook 06.

## LSTD: Solving Instead Of Iterating

The fixed point satisfies the linear system `A w* = b`; estimate both from
data and solve:

```math
\hat A_T=\frac{1}{T}\sum_t \phi(S_t)\big(\phi(S_t)-\gamma\phi(S_{t+1})\big)^\top,
\qquad
\hat b_T=\frac{1}{T}\sum_t \phi(S_t)R_{t+1},
\qquad
\hat w=\hat A_T^{-1}\hat b_T .
```

Least-Squares TD is sample-efficient (`O(d^2)` per step, no step size, one
pass) and is exact-in-the-limit for the same fixed point. LSPI wraps it in
policy iteration. The tradeoff against incremental TD is `d^2` memory and the
loss of the anytime/online character — and neither survives to nonlinear
function classes, which is why deep RL iterates.

## Reading For This Notebook

Szepesvári §3.2 (prediction with function approximation) gives this material
with full bookkeeping; Sutton & Barto ch. 9 gives the geometric picture. The
theorem statement worth memorizing is the boxed inequality above — it is the
*only* unconditional convergence + quality guarantee in all of value-function
approximation, and every one of its hypotheses (linear, on-policy, evaluation
not control) is violated by DQN.
