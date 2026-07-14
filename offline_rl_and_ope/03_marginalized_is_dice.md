# Marginalized Importance Sampling And The DICE Fixed Point

## The Change Of Object

Notebook 01 diagnosed trajectory IS: it reweights path measures, whose
overlap decays exponentially, when the estimand only needs occupancy
measures (`mdp_foundations/06`), whose overlap is what it is. Define the
**state(-action) density ratio**

```math
w_{\pi/b}(s,a)=\frac{d^\pi(s,a)}{d^{\pi_b}(s,a)}
\qquad\text{(discounted occupancies, normalized)} .
```

**The MIS identity.**

```math
J(\pi)
=
\frac{1}{1-\gamma}\,
\mathbb{E}_{(s,a)\sim d^{\pi_b}}\Big[\ w_{\pi/b}(s,a)\ r(s,a)\ \Big].
```

*Proof.* Immediate from `J = (1-gamma)^{-1} E_{d^pi}[r]` and the
definition of the ratio (change of measure on `S x A`, not on paths). ∎

**The variance collapse.** The estimator
`(1/n) sum w(s_i,a_i) r_i` over *transitions* (not trajectories) has
variance governed by `||w||_inf`-scale quantities:

```math
\mathrm{Var}\ \lesssim\ \frac{\|w\|_\infty\ \cdot\ \text{(reward scale)}^2}{n},
```

polynomial in `H` whenever the occupancy ratio is bounded — including in
notebook 01's hard instance, where trajectory IS pays `2^H` but
`w` is bounded by a constant (the occupancies overlap fine). The
exponential was an artifact of the wrong reweighting space; the Markov
property is what licenses the finer-to-coarser move (path measures push
forward to occupancies losslessly *for the purpose of evaluating
additive-in-time functionals*).

## But `w` Is Unknown: The Fixed-Point Characterization

`w` is defined via `d^pi`, which is what we don't have. It is, however,
pinned down by a **balance (Poisson-like) equation** on the data
distribution. The occupancy flow identity
(`mdp_foundations/06`'s dual constraint) for `d^pi`:

```math
d^\pi(s',a')
=
(1-\gamma)\,\mu_0(s')\pi(a'\mid s')
+\gamma\!\!\sum_{s,a}d^\pi(s,a)\,P(s'\mid s,a)\,\pi(a'\mid s') .
```

Divide by `d^{pi_b}(s',a')` and rewrite every occurrence of `d^pi` as
`w d^{pi_b}`: `w` satisfies, in `L2(d^{pi_b})`,

```math
w(s',a')
=
\frac{(1-\gamma)\mu_0(s')\pi(a'\mid s')}{d^{\pi_b}(s',a')}
+\gamma\,\frac{\mathbb{E}_{(s,a)\sim d^{\pi_b}}\big[w(s,a)\,P(s'\mid s,a)\big]\pi(a'\mid s')}{d^{\pi_b}(s',a')},
```

— a linear fixed-point equation **in quantities estimable from
transitions `(s, a, s') ~ data`**. This is the "backward Bellman
equation": the same `(I - gamma P)`-structure as everything in this
repository, transposed — it propagates *occupancy* forward instead of
*value* backward. Value functions and density ratios are the two adjoint
coordinates of the same linear algebra.

## DICE: The Saddle-Point Form

Solving the balance equation by minimizing its residual hits the
double-sampling obstruction (squared conditional expectations —
`stochastic_approximation/06`), so the DICE family (DualDICE, GenDICE,
…) instead writes the residual **variationally** with a dual test
function `f`:

```math
\min_{w}\ \max_{f}\
\mathbb{E}_{d^{\pi_b}}\Big[\big(\gamma\,\mathbb{E}_{s'\!,a'\sim\pi}f(s',a')-f(s,a)\big)\,w(s,a)\Big]
+(1-\gamma)\,\mathbb{E}_{\mu_0,\pi}\big[f\big]
\ -\ \text{regularizer}(w),
```

which is exactly the **Lagrangian of the occupancy LP**
(`mdp_foundations/06`) with `f` as the value-like multiplier — Fenchel–
Rockafellar duality made algorithmic; the full derivation is
`regularized_mdps_and_duality/06`. Each expectation is a plain average
over data: unbiased SGD applies, no double sampling. The dual variable
`f` converges (at the saddle) to a value-difference function — the
value/occupancy adjointness again, now inside one optimization.

## The Design Table

```text
estimator        reweights      needs               variance     bias
trajectory IS    paths          pi_b known          exp(H)       0
per-decision IS  path prefixes  pi_b known          exp(H)/poly  0
DR (02)          paths          pi_b or model       efficient*   O(err product)
MIS/DICE         occupancies    w-class realizable, poly(H)      class misspec.
                                minimax solvable
```

*at the trajectory-space efficiency bound, which itself carries the
exponential in hard instances.

## What Remains Open

The saddle-point problems are solved with neural `w, f` — convergence of
the *optimization* is uncertified (nonconvex-nonconcave); behavior-
agnostic OPE (unknown `pi_b`) works in the DICE frame but with weaker
guarantees; and sharp finite-sample theory under partial coverage
(`||w||_inf` unbounded on a small set) connects to the pessimism of
notebook 04 and is actively moving.
