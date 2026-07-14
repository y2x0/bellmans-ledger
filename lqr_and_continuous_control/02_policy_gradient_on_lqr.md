# Policy Gradient On LQR

## The Cleanest Nonconvex Landscape In RL

Restrict to static linear policies `u = -Kx` (optimal class, by
notebook 01) and view the infinite-horizon cost as a function of the
gain matrix:

```math
J(K)
=
\mathbb{E}_{x_0\sim\mathcal{D}}\Big[\sum_{t}\big(x_t^\top Qx_t+u_t^\top Ru_t\big)\Big]
=
\mathrm{tr}\big(P_K\,\Sigma_{x_0}\big),
```

defined on the (open) set of stabilizing `K` (`rho(A - BK) < 1`), where
`P_K` solves the policy-evaluation Lyapunov equation

```math
P_K=Q+K^\top RK+(A-BK)^\top P_K\,(A-BK)
\qquad\text{(the Bellman expectation equation, quadratic coordinates).}
```

**`J(K)` is nonconvex.** Two exhibits: the domain itself (stabilizing
`K`s) can be a *disconnected* set for multi-input systems (Fazel et al.'s
3-state example — two stabilizing islands separated by unstable gains),
and a nonconvex domain already forbids convexity of the problem; on each
island `J` is still nonconvex in general. So the black-box optimization
picture is bad — and yet:

## Gradient Domination Via The Performance Difference Lemma

The exact gradient (differentiate the Lyapunov equation; matrix calculus):

```math
\nabla J(K)=2\,E_K\,\Sigma_K,
\qquad
E_K:=\big(R+B^\top P_KB\big)K-B^\top P_KA,
```

with `Sigma_K` the closed-loop state covariance. `E_K` is the
**advantage coefficient**: the LQR `Q`-function is quadratic,
`A_K(x, u) = ` (quadratic in `u - (-K x)`) with curvature
`R + B^T P_K B` and linear term `2 E_K x` — one checks directly that
`E_K = 0` iff `K` is the Riccati gain.

**Lemma (gradient domination; Fazel–Ge–Kakade–Mesbahi 2018).**

```math
J(K)-J(K^*)
\ \le\
\frac{\|\Sigma_{K^*}\|}{\ \sigma_{\min}(R)\ \sigma_{\min}(\Sigma_{x_0})^2}\ \big\|\nabla J(K)\big\|_F^2
\quad\text{(up to the paper's exact constants)}.
```

*Proof mechanism — and the reason this file exists.* The performance
difference lemma (`policy_gradient/03`) in quadratic clothing: for any
two stabilizing `K, K'`,

```math
J(K')-J(K)
=
\mathbb{E}_{x\sim\text{traj}(K')}\Big[A_K\big(x,-K'x\big)\Big]
=
\sum_t\mathbb{E}\Big[2x^\top E_K^\top(K-K')x+ x^\top(K'-K)^\top(R+B^\top P_KB)(K'-K)x\Big],
```

— the *old* policy's advantage integrated along the *new* policy's
trajectory, exactly as in the general lemma, but now the advantage is an
explicit quadratic. Choose `K'` = the one-step Gauss–Newton/natural
update `K - (R + B^T P_K B)^{-1} E_K`-direction, complete the square:
the achievable one-step improvement is `>= c ||E_K||^2`-scale, while
`grad J = 2 E_K Sigma_K` ties `||E_K||` to `||grad J||` through
`sigma_min(Sigma_K) >= sigma_min(Sigma_{x_0})`. Rearranged, that is
gradient domination. ∎

**Consequences (same paper).** Exact gradient descent (also natural PG
and Gauss–Newton variants, with progressively better conditioning)
converges **globally at a linear rate** to `K*` from any stabilizing
initialization — nonconvexity notwithstanding; sample-based versions
(zeroth-order/evolutionary gradients from rollouts) inherit polynomial
sample complexity.

## The Audit — Same Ledger As The Tabular Story

The two hypotheses doing tacit work, and their exact tabular twins
(`global_convergence_of_pg/01`):

```text
sigma_min(Sigma_{x0}) > 0    the initial-state distribution excites every
                             direction — LQR's "mu > 0 / d^pi covers S"
                             (coverage). Degenerate x0 => plateaus return.
stabilizing initialization   the domain island containing K* must be
                             reachable — LQR's "interior initialization";
                             finding ANY stabilizing K is itself nontrivial
                             for unstable A (the exploration question in
                             disguise).
```

Same conservation law, quadratic edition: the geometry is benign
*given coverage and a feasible start*; the hard part was moved into the
hypotheses.

## Why This Result Mattered Beyond LQR

It is the sharpest available instance of "policy gradient provably works
on a nonconvex landscape," with every constant explicit — the
proof-of-concept that `global_convergence_of_pg/` generalizes
(gradient domination there, Łojasiewicz here; PDL as the shared engine),
and simultaneously the cautionary calibration: even in the friendliest
continuous problem, the guarantees are conditional on excitation and
stabilization, and both conditions are *statistical* problems the
optimization theorem simply assumes away.

## What Remains Open

Model-free LQR rates matching model-based ones (gap still open in
noise dependence); output-feedback LQR (`u = -K y`, partial
observation): the landscape acquires genuinely bad stationary points —
the LQG counterpart of `pomdps_and_information_state/`'s hardness; and
constrained/robust variants where the domination constant degenerates.
