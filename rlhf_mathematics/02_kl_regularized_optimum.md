# The KL-Regularized Optimum

## The Problem

Per prompt `x` (suppressed below), over distributions on completions:

```math
\max_{\pi}\quad
\mathbb{E}_{y\sim\pi}\big[r(y)\big]\ -\ \beta\,\mathrm{KL}\big(\pi\,\|\,\pi_{\mathrm{ref}}\big).
```

`pi_ref`: the SFT/pretrained reference. `beta`: the price of moving
probability mass away from it.

## The Closed Form

**Theorem.**

```math
\pi^*(y)
=
\frac{1}{Z}\ \pi_{\mathrm{ref}}(y)\,\exp\big(r(y)/\beta\big),
\qquad
Z=\sum_y\pi_{\mathrm{ref}}(y)\,e^{r(y)/\beta},
```

with optimal value

```math
J^*=\beta\,\log\ \mathbb{E}_{y\sim\pi_{\mathrm{ref}}}\Big[e^{r(y)/\beta}\Big]
\qquad\text{(Donsker–Varadhan).}
```

*Proof.* Write the objective as a single negative KL:

```math
\mathbb{E}_\pi[r]-\beta\,\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})
=
-\beta\ \mathbb{E}_\pi\Big[\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)\,e^{r(y)/\beta}}\Big]
=
-\beta\,\mathrm{KL}\big(\pi\ \|\ \pi^*\big)+\beta\log Z .
```

KL is nonnegative and zero iff its arguments coincide: maximum at
`pi = pi*`, value `beta log Z`. ∎

This is `regularized_mdps_and_duality/01`'s Legendre pair
(`Omega = beta KL(.||pi_ref)` → conjugate `beta log E_ref exp(./beta)`,
maximizer = tilted reference), realized over completions rather than
actions — and it is also `concentration_toolkit/05`'s tilting: **the
optimal RLHF policy is an exponential tilt of the reference in the reward
direction**, the same object that proves Chernoff bounds and constructs
lower bounds. One transform, fourth appearance.

## Reading The Temperature

```math
\beta\to\infty:\ \pi^*\to\pi_{\mathrm{ref}}
\qquad
\beta\to0:\ \pi^*\to\text{argmax}_{\,\mathrm{supp}(\pi_{\mathrm{ref}})}\,r
```

with, along the way, two exact identities worth keeping:

```math
\mathrm{KL}\big(\pi^*\|\pi_{\mathrm{ref}}\big)
=
\frac{\mathbb{E}_{\pi^*}[r]-J^*}{\beta},
\qquad
\frac{\partial}{\partial(1/\beta)}\,\mathbb{E}_{\pi^*}[r]
=
\mathrm{Var}_{\pi^*}\big[r\big]\ \ge 0
```

(the second from differentiating the tilted family — the cumulant-
generating-function calculus: reward gained per unit of inverse
temperature equals the reward variance under the current policy). The
**reward–KL frontier** traced by varying `beta` is concave, with slope
`beta` at the point the policy sits — the exchange rate between reward
and divergence is the temperature itself. Every empirical
"reward vs KL" plot in the RLHF literature is a sampled picture of this
frontier; 04 and 05 are about competitors on it and departures from it.

## Support Is Destiny

`pi*(y) = 0 wherever pi_ref(y) = 0`: **KL-regularized RLHF can only
reweight the reference's support, never leave it.** Consequences:

```text
1. capability ceiling: no amount of reward optimization elicits
   completions the reference assigns zero (or astronomically small)
   probability — RLHF sharpens, it does not teach;
2. safety floor read the same way: behaviors the reference cannot produce
   remain unproducible (at matched support);
3. the practical failure is the near-violation: mass concentrating on
   exp(r/beta)-favored TAILS of pi_ref — regions real but rare in
   reference data, where the reward model extrapolates (01's point 3);
   05 quantifies the damage.
```

## Why PPO At All, If The Solution Is Closed-Form?

`pi*` is explicit but **unnormalizable in practice**: `Z` sums over all
completions, and sampling from `pi_ref e^{r/beta}` exactly is intractable.
The three families of response, which are the rest of this family:

```text
1. PPO-RLHF: ascend the objective directly with policy gradients
   (policy_gradient/06) — the KL term implemented as a per-token reward
   penalty beta * log(pi/pi_ref); converges (in the exact limit) to pi*
   by this file's theorem + global_convergence_of_pg/02 (the update IS
   multiplicative weights toward the tilt);
2. DPO (03): avoid Z by a reparameterization in which it cancels;
3. best-of-n (04): approximate the tilt by sampling and selecting —
   no training at all.
```

## What Remains Open

Nothing about the theorem. The open ground is all in its inputs: `r` off
the comparison distribution (01, 05), sequence-level vs token-level
implementations of the KL (06), and multi-turn objectives where the
per-prompt decomposition itself fails (the completion is no longer the
terminal object — genuine MDP structure returns).
