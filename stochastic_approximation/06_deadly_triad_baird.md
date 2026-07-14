# The Deadly Triad And Baird's Counterexample

## The Triad

```text
1. function approximation   (generalization across states)
2. bootstrapping            (targets built from current estimates)
3. off-policy data          (update distribution != target policy's stationary distribution)
```

Any two are safe:

```text
FA + bootstrap, on-policy       -> TvR theorem (notebook 04): converges
FA + off-policy, no bootstrap   -> MC regression on IS-corrected targets: converges
bootstrap + off-policy, tabular -> Watkins' theorem (notebook 05): converges
all three                       -> can diverge for ALL step sizes
```

## The Mechanism In One Matrix

From notebook 04, expected linear-TD dynamics are `w-dot = b - Aw` with

```math
A=\Phi^\top D\,(I-\gamma P_\pi)\,\Phi,
\qquad
D=\mathrm{diag}(\mu).
```

On-policy (`mu` stationary for `P_pi`), `A` is positive definite. Off-policy,
`D` and `P_pi` are mismatched and `A` can have an eigenvalue with negative
real part; then `||w_t|| -> infinity` **in expectation** — this is not noise,
not step size, not exploration. The mean field itself is unstable.

Equivalently: `Pi_mu T^pi` is a composition of a `gamma`-contraction *in
`||.||_mu-stationary`* with a projection nonexpansive *in `||.||_mu-data`*.
Two different norms; the composition can expand. Bootstrapping is what makes
the operator's norm matter at all (MC targets don't pass through `P_pi`);
function approximation is what makes the projection nontrivial; off-policy
data is what breaks the norm match.

## Baird's Counterexample (1995), Concretely

Seven states; six "upper" states and one "lower" state 7. Two actions:
`dashed` moves to one of states 1–6 uniformly; `solid` moves to state 7
(then self-loops via solid). Rewards are all **zero**, `gamma = 0.99`.
Features are overparameterized linear:

```math
V(i)=2w_i+w_0\ \ (i=1,\ldots,6),
\qquad
V(7)=w_7+2w_0 .
```

True value function: `v_pi = 0`, exactly representable (`w = 0`). Behavior
policy: `dashed` with probability `6/7`, `solid` with `1/7` (uniform-ish over
states). Target policy: always `solid` (so all bootstrap targets flow through
state 7).

Run off-policy TD(0) (or Q-learning/dynamic programming with this `D`): the
weights diverge to infinity — monotonically, deterministically in the
expected update, for every positive step size. **Even though the true
solution lies in the function class and the data covers every state.**

What goes wrong mechanically: updates are distributed per the behavior
(mostly upper states), but targets bootstrap through state 7's value, whose
feature `w_0` is shared with every state. Each upper-state update inflates
`w_0`; the correction at state 7 is visited too rarely (under `D`) to
compensate; the mismatch feeds back through `gamma P` and compounds.

## Escapes, Each Paying A Different Price

```text
1. Gradient TD (GTD2 / TDC, Sutton et al. 2009):
   do true gradient descent on the projected Bellman error
   MSPBE(w) = || Pi T V_w - V_w ||_mu^2, using a second set of weights to
   estimate the double-sampling term. Two-timescale RM -> provable
   off-policy convergence. Price: slower, an extra learner, rarely used at scale.

2. Emphatic TD (Sutton, Mahmood, White 2016):
   reweight updates by followon traces so the *effective* D matches the
   target policy's discounted occupancy; restores positive definiteness of A.

3. Target networks (DQN): freeze the bootstrap parameter, turning each phase
   into supervised regression toward a fixed target — not a convergence
   proof, but it removes the fast feedback loop that the instability needs
   (value_based_deep_rl/02).

4. Give up bootstrapping: MC / lambda -> 1 targets. Pays the variance bill
   (notebook 02).

5. Give up off-policy: on-policy actor-critic (PPO). Pays with sample reuse
   restrictions — and PPO's clipping is precisely a device to stay
   "near-on-policy" while reusing a batch (policy_gradient/06).
```

## Residual Gradient, And Why It Is Not The Answer

Minimizing the raw Bellman residual `||T V_w - V_w||^2` by true gradient
descent is convergent but targets the wrong object: with stochastic
transitions, the sampled squared residual's expectation includes the variance
of the target (`E[X]^2 != E[X^2]`), so unbiased gradients need **two
independent next-state samples** from the same `(s,a)` — unavailable from
trajectories. Fixable only with a model or a conditional expectation
estimator; hence the field's drift toward MSPBE (GTD) or empirical stability
devices (DQN) instead.

The triad is the single most important negative result to carry into deep
RL: DQN sits squarely on all three legs, and `value_based_deep_rl/` is the
study of why it (mostly) works anyway.

## The Modern Descendant (retrofit)

Baird's example shows one algorithm's mean dynamics diverge. The
quantitative, algorithm-free upgrade is the Wang–Foster–Kakade offline
lower bound (`linear_mdps_and_completeness/05`): with *realizable* linear
values and well-conditioned data coverage, **no** algorithm evaluates a
policy without `exp(Omega(d))` samples — the norm mismatch of this file,
built adversarially and priced. The positive complements: on-policy TvR
(notebook 04), Bellman completeness (`linear_mdps_and_completeness/03`),
and pessimism under single-policy concentrability
(`offline_rl_and_ope/04`).
