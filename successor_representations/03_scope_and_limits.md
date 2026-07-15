# Scope And Limits Of Successor Methods

## The Dependence Structure, Stated Exactly

```math
\psi^\pi\ \text{depends on:}\quad P\ \text{(dynamics)},\ \ \pi\ \text{(policy)},\ \ \gamma
\qquad
\text{and NOT on:}\quad w\ \text{(task)} .
```

Every strength and every limit of the family is a corollary of this
line.

## What Transfers: Rewards Only

Changing `w` costs nothing (notebook 02). Changing anything else:

```text
dynamics change (P -> P'):   psi is wrong wholesale — the cached object
                             IS the propagation through P. No SF method
                             survives dynamics shift; that regime belongs
                             to model_based/ (re-plan) or meta-learning
                             (adapt). The factorization chose its side.

policy dependence:           psi^pi evaluates pi's futures. GPI covers
                             the library's span, but the NEW task's true
                             optimum may route through states no library
                             policy visits — the transfer bound's
                             min_i ||w - w_i|| is honest about this: a
                             genuinely novel task direction gets a
                             genuinely weak guarantee. Coverage in task
                             space is the SF analogue of coverage in
                             state space (offline_rl_and_ope/04), with
                             the same character: guarantees degrade with
                             distance from what was practiced.

gamma / horizon change:      bakes into the discounting of psi;
                             re-learnable cheaply via the Laurent-style
                             relations (average_reward/03) but not free.
```

## The Feature Realizability Question

`r = phi^T w` is an assumption with the same epistemics as
`linear_mdps_and_completeness/03`'s realizability: exactly true only if
the feature designer knew the task family in advance. Misspecification
`r = phi^T w + delta` enters the transfer bound additively as
`||delta||_inf/(1-gamma)` — graceful, but it means SF transfer is only
as good as the feature basis is complete over the *future* task
distribution. Learned-`phi` variants must satisfy two masters (linear
rewards AND stable `psi`-TD), the joint condition notebook 02 flagged
open.

## The Contrast Class: UVFAs And Goal Conditioning

The other route to multi-task value is the universal value function
approximator: `q(s, a, g)` with the task/goal `g` as a network input —
generalization across tasks by *function approximation* rather than by
*linear structure*:

```text
                    SF + GPI                    UVFA / goal-conditioned
task representation w in R^d, linear            arbitrary g, learned
transfer guarantee  Thm (notebook 02): bound    none (whatever the net
                    in ||w - w_i||               generalizes)
new-task cost       inner products + argmax     zero-shot forward pass,
                                                 uncontrolled quality
composition         max over library (GPI);     arithmetic in g-space,
                    provable                     heuristic
where each wins     task family truly linear    rich goal spaces (images,
                    in known features            language) where phi is
                                                 the hard part
```

Hindsight relabeling (HER) — re-labeling trajectories with achieved
goals — is the UVFA world's sample-reuse trick, and it is the same
counterfactual-reuse pattern as intra-option learning
(`options_and_hierarchy/02`) and SF's shared-stream TD: one experience
stream, many predictions. The pattern is now visible as one of this
repository's recurring instruments in the applied families.

## Option Keyboards And Composition

GPI's max composes *policies*; a sharper composition uses SFs to
evaluate **behavior mixtures** ("option keyboard"): tasks expressible as
`w = sum_j c_j w_j` get policies assembled from the library with
guarantees inherited from notebook 02's bounds applied to the composite.
The formal limit is the same one distributional control met
(`distributional_rl/03`): max/composition operators over *evaluations*
are sound; over *behaviors* they need the envelope argument, which GPI
supplies only for the max — richer compositions (chaining, sequencing)
push into `options_and_hierarchy/` territory where the guarantees
thin out.

## What Remains Open

SF theory under learned, drifting features (the collapse risks of
notebook 01); GPI's library-selection economics (which `pi_i` to retain
under a memory budget — a covering problem in task space, unformalized);
dynamics-robust extensions (SFs over model ensembles); and the
RLHF interface flagged in notebook 02 — preference-derived `w`-spaces —
which no one has built.
