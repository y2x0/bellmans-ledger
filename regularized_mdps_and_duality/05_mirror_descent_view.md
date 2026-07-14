# The Mirror Descent View Of Policy Optimization

## Mirror Descent In One Page

To maximize a concave `f` over a convex set `X` with geometry given by a
strongly convex potential `psi` (Bregman divergence
`D_psi(x, y) = psi(x) - psi(y) - <grad psi(y), x - y>`):

```math
x_{k+1}
=
\arg\max_{x\in X}\ \Big\{\ \eta\,\langle\nabla f(x_k),\,x\rangle-D_\psi(x,x_k)\ \Big\}.
```

**The three-point lemma.** For any comparator `x* in X`:

```math
\eta\,\langle\nabla f(x_k),\,x^*-x_{k+1}\rangle
\ \le\
D_\psi(x^*,x_k)-D_\psi(x^*,x_{k+1})-D_\psi(x_{k+1},x_k),
```

*(proof: first-order optimality of `x_{k+1}` plus the Bregman
three-point identity — pure algebra).* Summed over `k`, the right side
**telescopes**: total progress toward any comparator is bounded by the
initial divergence `D_psi(x*, x_0)` — a regret bound requiring **no
contraction anywhere**, only convexity along the analysis path.

For `X = Delta(A)` and `psi = ` negative entropy:
`D_psi = KL(x || y)`, and the MD update has the closed form
`x_{k+1}(a) ∝ x_k(a) exp(eta g(a))` — multiplicative weights.

## The Identification

Per state, consider the update

```math
\pi_{k+1}(\cdot\mid s)
=
\arg\max_{\pi}\ \Big\{\eta\,\big\langle \pi,\ q^{\pi_k}(s,\cdot)\big\rangle-\mathrm{KL}\big(\pi\,\|\,\pi_k(\cdot\mid s)\big)\Big\}
=
\text{(closed form) }\ \pi_k\,e^{\eta q^{\pi_k}}/Z .
```

Compare the three families already built:

```text
this update                        = notebook 04's G_Omega with the
                                     PROXIMAL regularizer Omega = KL(.||pi_k)
TRPO (policy_gradient/04)          = the same program with the KL as a
                                     CONSTRAINT and the objective linearized
                                     (surrogate <pi, A^{pi_k}>): an inexact
                                     MD step, trust radius = step size
PPO clip (policy_gradient/06)      = a zeroth-order clamp standing in for
                                     the same proximal term
NPG (global_convergence_of_pg/02)  = EXACTLY this update for softmax
                                     parameterization (the Fisher
                                     preconditioner realizes the KL
                                     geometry in parameter space)
```

**Policy optimization is mirror descent on the simplex bundle** (one
simplex per state, weighted by visitation), with the advantage function as
the gradient (`policy_gradient/01`'s theorem supplying
`grad J ∝ E_{d_pi}[A]`-pairings).

## What The MD Reading Proves

Run the three-point lemma per state, weight by the comparator's visitation
`d^{pi*}`, and sum — using the performance difference lemma
(`policy_gradient/03`) to convert gradient pairings into value gaps:

```math
\min_{k\le K}\ \Big(J(\pi^*)-J(\pi_k)\Big)
\ \le\
\frac{1}{(1-\gamma)\,K}\ \Big(\frac{\mathbb{E}_{s\sim d^{\pi^*}}\mathrm{KL}\big(\pi^*\|\pi_0\big)}{\eta}+\eta\,\text{(advantage-scale)}^2\,K^\prime\Big)
```

— an `O(1/K)`-type **global** convergence bound (details and the sharp
`O(1/K)` statement in `global_convergence_of_pg/02`), whose two
inputs are exactly:

```text
1. the telescoping Bregman sum (this file's lemma) — no Łojasiewicz, no
   contraction, no convexity of J itself;
2. the PDL to relate per-state linear progress to global value progress —
   paying the distribution mismatch d^{pi*}/d^{pi_k} where it must
   (the transfer problem of global_convergence_of_pg/04).
```

The reason MD applies to a *nonconvex* `J`: the analysis never uses
convexity of `J` in `theta` — it works on the simplex where the objective
is **linear in `pi`** per state (the occupancy-LP linearity,
`mdp_foundations/06`), and pays for the linearization through the PDL
mismatch term. The nonconvexity was an artifact of the parameterization
all along; MD does its accounting in the right coordinates.

## The Proximal-vs-Fixed Distinction (the family's hinge)

```text
fixed Omega = tau KL(.||pi_ref):  changes the OBJECTIVE and its optimum —
                                  bias tau-scaled (04, fact 2); this is
                                  RLHF's regularizer (rlhf_mathematics/02),
                                  where the bias is the point;
proximal Omega = KL(.||pi_k):     vanishes at any fixed point — shapes the
                                  PATH only; step size eta trades progress
                                  against the linearization's validity
                                  radius — which is precisely TRPO's
                                  delta and PPO's epsilon.
```

One convex function, two placements, two different theories — and both
placements simultaneously is exactly modern RLHF-PPO (clip + KL-to-ref).

## What Remains Open

Sharp rates for *inexact* MD with the errors RL actually commits
(sampled advantages, stale visitation weights — partial results track both);
and whether PPO's clip can be given a genuine MD-style telescoping bound
(currently: no — the clamp is not a Bregman prox, and the gap between
PPO's practice and MD's theory remains the family's honest loose end).
