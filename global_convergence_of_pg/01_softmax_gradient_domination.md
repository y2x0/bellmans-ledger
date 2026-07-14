# Softmax Gradient Domination

## Setting

Tabular softmax policy — the cleanest honest case:

```math
\pi_\theta(a\mid s)=\frac{e^{\theta_{s,a}}}{\sum_{a'}e^{\theta_{s,a'}}},
\qquad
J(\theta)=\mathbb{E}_{s_0\sim\mu}\big[v_{\pi_\theta}(s_0)\big],
\qquad \mu>0\ \text{componentwise}.
```

The gradient has the explicit per-coordinate form (from
`policy_gradient/01`, softmax Jacobian worked out):

```math
\frac{\partial J}{\partial\theta_{s,a}}
=
\frac{1}{1-\gamma}\ d_\mu^{\pi_\theta}(s)\ \pi_\theta(a\mid s)\ A^{\pi_\theta}(s,a).
```

Read it: the gradient coordinate dies if the state is unvisited
(`d ~ 0`), the action is unplayed (`pi ~ 0`), or there is nothing to
correct (`A ~ 0`). The first two are the villains below.

## Nonconvexity Is Real But Shallow

`J(theta)` is nonconvex even for one-state MDPs (compose the linear-in-pi
value with the non-affine softmax; sublevel sets disconnect under
symmetry). But its spurious stationary points are all "at infinity"
(deterministic policies), and away from them the gradient's *size* lower-
bounds the *suboptimality*:

## The Non-Uniform Łojasiewicz Inequality

**Theorem (Mei–Xiao–Szepesvári–Schuurmans 2020).** For softmax PG:

```math
\big\|\nabla_\theta J(\theta)\big\|_2
\ \ge\
\frac{\min_s\ \pi_\theta(a^*(s)\mid s)}{\sqrt{|S|}\ }\,
\Big\|\frac{d^{\pi^*}_\mu}{d^{\pi_\theta}_\mu}\Big\|_\infty^{-1}\,
(1-\gamma)\,
\big(J(\pi^*)-J(\theta)\big).
```

*Proof skeleton.* The performance difference lemma
(`policy_gradient/03`) writes the suboptimality as
`(1-gamma)^{-1} E_{d^{pi*}}[<pi* - pi, q^{pi}>]`; per state the inner
product is at most the gap at `a*(s)`, which appears in the gradient
coordinate `(s, a*(s))` multiplied by `d^{pi}(s) pi(a*|s)`; convert the
`d^{pi*}`-average to a `d^{pi}`-average by paying the density ratio;
Cauchy–Schwarz across states pays the `sqrt(|S|)`. Every constant in the
display is one of these steps. ∎

**Consequence (global rate).** Gradient-dominated + smooth (`J` is
`O(1/(1-gamma)^3)`-smooth in `theta`) gives, for exact gradient ascent,
the standard descent-lemma recursion `delta_{t+1} <= delta_t - c
delta_t^2`, hence

```math
J(\pi^*)-J(\theta_t)\ =\ O\big(1/t\big)
\qquad\text{— GLOBAL, from any interior initialization.}
```

Asymptotically the iterates converge to `pi*` itself (not merely in
value).

## The Constants Are The Content

The inequality is *non-uniform*: its constant degrades along the path
through

```math
c_t=\min_s\ \pi_{\theta_t}(a^*(s)\mid s)
\qquad\text{and}\qquad
\Big\|\frac{d^{\pi^*}}{d^{\pi_{\theta_t}}}\Big\|_\infty .
```

**The plateau construction (why exp(H) is inside the O(1/t)).** Chain MDP
of length `H` (the combination lock of `regret_and_exploration/04`, in PG
clothing): reward only at the chain's end; initialize uniformly. Then
`d^{pi_0}` puts mass `~ A^{-h}` on depth-`h` states, the gradient
coordinates that matter are exponentially small, and gradient ascent
spends `Omega(A^{H})`-type time traversing a **plateau** — while the
Łojasiewicz bound remains true, its constant `c_t ||d*/d||^{-1}` is
exponentially small there. The theorem and the hardness coexist: global
convergence *eventually*, with the exploration difficulty stored in the
constants, exactly where `policy_gradient/07 §6` gestured.

```text
audit: what softmax-PG's "global convergence" does and does not say
  does:    no spurious LOCAL optima trap exact ascent forever
  doesn't: polynomial time; the mismatch ||d*/d|| is the exploration
           problem wearing an optimization costume — no policy-side
           theorem removes it (04 makes this precise and permanent)
```

## Entropy Regularization Sharpens The Geometry

Adding `tau H(pi)` (`regularized_mdps_and_duality/01`) upgrades the
inequality to a genuine (still non-uniform) **PL inequality with
squared-gradient**, and exact ascent then converges **linearly** to the
`tau`-optimum (Mei et al., companion result): strong-concavity-like
curvature appears because the entropy term penalizes the boundary, keeping
`min pi(a*|s)` bounded below along the path — the plateau's mechanism
(vanishing action probabilities) is structurally forbidden. Combined with
the bias bound `tau log|A|/(1-gamma)`, annealing `tau` gives the standard
two-phase schedule: fast regularized phase, then decay.

## What Remains Open

Tight characterization of softmax-PG's *path-dependent* complexity
(current results: exponential lower bounds for exact PG on chain
instances with uniform start — matching the plateau intuition); stochastic
gradients (all of the above is exact-gradient); and anything beyond
tabular — which is notebook 03–04's territory.
