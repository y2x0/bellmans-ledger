# LQR: The Riccati Equation From The Bellman Equation

## The Problem

Linear dynamics, quadratic costs (cost convention — minimization — is
standard here):

```math
x_{t+1}=A\,x_t+B\,u_t+w_t,
\qquad
w_t\sim\mathcal{N}(0,\Sigma_w)\ \text{i.i.d.},
```

```math
J=\mathbb{E}\Bigg[\sum_{t=0}^{H-1}\big(x_t^\top Q\,x_t+u_t^\top R\,u_t\big)+x_H^\top Q_f\,x_H\Bigg],
\qquad Q,Q_f\succeq0,\ R\succ0 .
```

## Quadratic Values, By Induction

**Claim.** The finite-horizon optimal cost-to-go is exactly quadratic
plus a constant: `V_t(x) = x^T P_t x + c_t`.

*Proof (one Bellman backup, which is the whole file).* Base:
`V_H = x^T Q_f x` (`P_H = Q_f`). Step: assume the claim at `t+1`; the
Bellman equation:

```math
V_t(x)=\min_u\ \Big\{x^\top Qx+u^\top Ru+\mathbb{E}\big[(Ax+Bu+w)^\top P_{t+1}(Ax+Bu+w)\big]\Big\}+c\text{-terms}.
```

Expand; the noise cross-terms vanish (`E[w] = 0`), leaving
`tr(P_{t+1} Sigma_w)` in the constant. The bracket is a **convex
quadratic in `u`** (`R + B^T P_{t+1} B > 0`); minimize by completing the
square:

```math
u_t^*=-\underbrace{\big(R+B^\top P_{t+1}B\big)^{-1}B^\top P_{t+1}A}_{=:K_t}\ x
\qquad\text{— LINEAR in the state,}
```

and substituting back:

```math
P_t=Q+A^\top P_{t+1}A-A^\top P_{t+1}B\big(R+B^\top P_{t+1}B\big)^{-1}B^\top P_{t+1}A
```

— the **Riccati recursion**: the Bellman backup, restricted to the
quadratic cone, written in matrix coordinates. Quadratic-in, quadratic-
out closes the induction. ∎

The miracle to name precisely: the quadratics are a
**Bellman-complete class** for linear dynamics
(`linear_mdps_and_completeness/03`'s coveted property, holding here
*exactly and verifiably*) — which is why LQR admits a finite-dimensional
solution and generic MDPs do not.

## Infinite Horizon: The DARE

Under `(A, B)` stabilizable and `(A, Q^{1/2})` detectable, the recursion
converges to the unique stabilizing solution `P` of the **discrete
algebraic Riccati equation** (the display above with `P_t = P_{t+1} =
P`), with constant optimal gain `K` and closed loop `A - BK` stable
(`rho < 1`). Stability here is the SSP properness of
`total_reward_ssp/01` in continuous clothing: `rho(A - BK) < 1` is what
replaces both the discount and termination as the source of
convergence — one more occupant of the repo's "contraction slot."

## Certainty Equivalence — The Exception That Defines The Rule

**Proposition.** The optimal policy `u = -K_t x` is **independent of the
noise covariance** `Sigma_w`: noise enters `V` only through the additive
constants `c_t = c_{t+1} + tr(P_{t+1} Sigma_w)`.

*Proof:* visible in the backup above — `w` contributed only
`tr(P Sigma_w)`, a `u`-free constant; the minimization never saw it. ∎

So for LQR one may plan in the *deterministic* system and apply the
result to the stochastic one, losing nothing. **Mark this as the
exception.** It required: quadratic value (so the noise's effect is its
second moment only) and additive noise (so the second moment is
action-independent). Break either — multiplicative noise, non-quadratic
cost, constraints (even simple input saturation), risk-sensitivity — and
certainty equivalence fails; the whole distributional family
(`distributional_rl/`) exists because generic MDP noise does *not*
integrate out of decisions, and the LQG special case is why classical
control lived happily without it.

(Estimation-side certainty equivalence — plug in estimated `(A-hat,
B-hat)` and control as if true — is a *different*, approximate statement,
and its surprising near-optimality is `model_based/02`'s topic with this
file's exact version as the base case.)

## What Remains Open

Nothing in the classical theorem. The open edges: constrained LQR (MPC
theory — no closed form, the quadratic-cone completeness broken by
inequality boundaries), LQR with multiplicative/parametric uncertainty
(robust control, `mu`-synthesis — hard in general), and the
learning-theoretic questions the rest of this family takes up.
