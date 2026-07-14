# Trust Regions: The TRPO Bound

## The Monotonic Improvement Theorem

(Schulman et al. 2015, sharpening Kakade–Langford.) With
`L_pi(pi')` the first-order surrogate of notebook 03 and

```math
\varepsilon=\max_{s,a}|A_\pi(s,a)|,
\qquad
D_{\mathrm{KL}}^{\max}(\pi,\pi')=\max_s\ \mathrm{KL}\big(\pi(\cdot\mid s)\,\|\,\pi'(\cdot\mid s)\big),
```

```math
\boxed{\
J(\pi')\ \ge\ L_\pi(\pi')-C\,D_{\mathrm{KL}}^{\max}(\pi,\pi'),
\qquad
C=\frac{4\varepsilon\gamma}{(1-\gamma)^2}\ } .
```

*Proof structure.* (i) The lemma of notebook 03 makes the error
`J(pi') - L_pi(pi')` an integral of `A_pi` against the visitation difference
`d^{pi'} - d^{pi}`. (ii) Couple the two chains: run them on a common
probability space agreeing until the first state where the sampled actions
differ; per-step disagreement probability is bounded by
`max_s TV(pi, pi')`, so at time `t` the chains have decoupled with
probability `<= 1 - (1-alpha)^t`, `alpha = max TV`. (iii) Each decoupled step
contributes at most `2 eps` of advantage error, discounted; summing the
geometric-ish series produces the `(1-gamma)^{-2}` (one horizon factor for
the error's persistence, one for accumulation of disagreement probability).
(iv) Pinsker, `TV^2 <= KL/2`, converts TV to KL. ∎

## The Guaranteed-Improvement Recipe

Define `M(pi') = L_pi(pi') - C D_KL^max(pi, pi')`. Then

```math
J(\pi')\ \ge\ M(\pi'),
\qquad
J(\pi)=M(\pi),
```

so **maximizing `M` is a minorize–maximize (MM) algorithm**: any `pi'` with
`M(pi') > M(pi)` strictly improves `J`. Policy optimization with a KL
penalty at coefficient `C` can never overshoot — the theoretical answer to
"how large a step is safe."

## Why TRPO Doesn't Use The Theory Directly

Two substitutions, both acknowledged in the papers:

```text
1. C from the theorem is enormous ((1-gamma)^{-2} and max|A|) -> steps would
   be microscopic. Replace the PENALTY with a hard CONSTRAINT at a tunable
   radius delta.

2. max_s KL is not estimable from samples -> replace with the mean
   E_{s~d^pi} KL. (The theorem needs the max; the mean is the practical
   surrogate, and the gap is a known theoretical hole.)
```

giving TRPO's problem (PPO paper eq. 3–4):

```math
\max_\theta\ \hat{\mathbb{E}}_t\Big[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}\hat A_t\Big]
\quad
\text{s.t.}\quad
\hat{\mathbb{E}}_t\big[\mathrm{KL}[\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t),\pi_\theta(\cdot\mid s_t)]\big]\le\delta .
```

## Solving It: Natural Gradient

Second-order expansion of the constraint at `theta_old`: KL's Hessian is the
**Fisher information matrix**

```math
F(\theta)=\mathbb{E}_{s,a}\big[\nabla\log\pi_\theta(a\mid s)\,\nabla\log\pi_\theta(a\mid s)^\top\big],
\qquad
\mathrm{KL}\approx\tfrac12(\theta-\theta_{\text{old}})^\top F(\theta-\theta_{\text{old}}),
```

and with a linearized objective the constrained maximizer is the **natural
gradient** step (Amari; Kakade 2001):

```math
\theta=\theta_{\text{old}}+\sqrt{\frac{2\delta}{g^\top F^{-1}g}}\;F^{-1}g,
\qquad g=\nabla_\theta L .
```

`F^{-1} g` is the steepest ascent direction when distance between policies is
measured in KL (i.e. on the statistical manifold) rather than in parameter
Euclidean distance — invariant to reparameterization of `theta`, which raw
gradients are not. TRPO computes `F^{-1}g` matrix-free by conjugate gradient
on Fisher-vector products, then backtracking line-search to enforce the exact
(sampled) KL constraint and surrogate improvement.

## The Costs, Which PPO Inherits As Motivations

```text
1. CG + line search: ~10 extra passes per update; complicated to implement
   correctly at scale.
2. Incompatible with parameter noise (dropout) and awkward with shared
   policy/value parameters — the Fisher metric is defined on policy
   parameters only. (Both named explicitly in the PPO paper's introduction.)
3. Hard constraint discards useful curvature information when far from the
   boundary; penalty form would be simpler, but no single beta works across
   problems or even across training time (PPO paper §4's observation).
```

PPO's clipping (notebook 06) is a first-order device engineered to keep the
*effect* of the constraint — bounded per-state policy change, hence a valid
surrogate — while deleting `F`, CG, and the line search.
