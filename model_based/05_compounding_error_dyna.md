# Compounding Errors, Branched Rollouts, And Dyna

## The Phenomenon

Roll a learned model forward `k` steps and the state distribution
drifts from reality at a compounding rate: per-step model error `eps_m`
(in TV, on the visited distribution) gives, by iterating the coupling /
triangle inequality one step at a time,

```math
\mathrm{TV}\big(\text{real }k\text{-step dist},\ \text{model }k\text{-step dist}\big)
\ \le\ k\,\varepsilon_m,
```

and the value estimated from length-`k` model rollouts inherits an
error growing (in the worst case) **linearly in `k` with the horizon's
weight behind it** — the trajectory-level face of the simulation
lemma's `(1-gamma)^{-2}` (01), localized to rollout length.

## The Branched-Rollout Bound (Janner et al. 2019, MBPO)

The design question: given a model with error `eps_m` (and a policy
shift `eps_pi` between the data-collecting and current policies), how
long should model rollouts be? MBPO's *branched* scheme: start rollouts
from **real off-policy states** (the replay buffer's distribution, not
the initial-state distribution), roll the model `k` steps, train the
policy on the mixture. The bound (shape):

```math
J(\pi)\ \ge\ J^{\text{branch}}(\pi)
-\ 2R_{\max}\Big[
\frac{\gamma^{k+1}\,\varepsilon_\pi}{(1-\gamma)^2}
+\frac{\gamma^{k}\,\varepsilon_\pi}{1-\gamma}
+\frac{k}{1-\gamma}\,(\varepsilon_m+2\varepsilon_\pi)
\Big].
```

Read the three terms: tail distribution shift beyond the branch
(shrinks with `k`); shift at the branch point; and **model compounding
along the branch (grows linearly in `k`)**. Optimizing the trade gives
a finite `k* > 0` whenever `eps_m` is small relative to `eps_pi` —
"short model rollouts from real states": the theoretical warrant for
what MBPO and successors run (`k` of 1–15, never model-only episodes).

**The bound's honest defect, admitted in the paper:** with `eps_m`
measured as a *global* worst-case error, `k* = 0` often — the bound
cannot justify using the model at all. The working refinement measures
`eps_m` **locally** (model generalization on the shifted distribution),
which is real but no longer a priori: the theory prices the schedule
only after an empirical model-error curve is plugged in.

## The Adversarial Twist: The Planner Seeks The Errors

The compounding analysis treats model error as exogenous noise. It is
not: the policy is **optimized against the model**, so it migrates
toward regions where the model's value is inflated — the max-bias
mechanism (`value_based_deep_rl/03`, `rlhf_mathematics/05`; third
family, same theorem) with the model as the exploited estimator:

```math
\mathbb{E}\Big[\max_\pi \hat J_{\text{model}}(\pi)\Big]\ \ge\ \max_\pi J(\pi)
\quad\text{with the gap concentrated exactly where }\hat P\text{ errs high.}
```

Consequences and the standard controls, each an old friend:

```text
model exploitation      = reward hacking of a dynamics model; the
                          symptom is model-return >> real-return
                          divergence during training (monitor it —
                          policy_gradient/07's dashboard logic);
ensembles + pessimism   = penalize disagreement/uncertainty of a model
                          ensemble (offline: MOPO/MOReL — exactly
                          offline_rl_and_ope/04's pessimism with
                          ensemble-width bonuses; online: soften by
                          sampling models per rollout, PETS-style);
short branches (above)  = bound the leverage any single model error has;
Dyna's real-data anchor = keep training the value on real transitions
                          so model fictions compete with data.
```

## Dyna, As The General Template

Sutton's Dyna (1990): interleave (i) real-experience updates, (ii)
model learning, (iii) `n` "imagined" updates per real step from the
model. In operator terms, the value update applies a **mixture
operator**

```math
\mathcal{F}
=
(1-\lambda_{\text{mix}})\,\hat{\mathcal{T}}_{\text{data}}
+\lambda_{\text{mix}}\,\tilde{\mathcal{T}}_{\text{model}},
```

and the fixed-point analysis is a perturbation statement: if
`T-tilde` is `eps_m`-close to `T` (simulation-lemma pairing, 01) and
both contract, the mixture contracts with a fixed point within
`lambda_mix * eps_m/(1-gamma)` of `v*` — imagination is admissible
exactly in proportion to model quality, and `n` (the imagination ratio)
is `lambda_mix` in schedule form. Prioritized sweeping is Dyna with the
imagined backups ordered by Bellman-residual size — the model used to
*route* credit assignment backward faster than the data stream would
(`value_based_deep_rl/01`'s replay, upgraded from decorrelation device
to planner).

```text
the design summary this family closes on:

  use the model where its errors are (a) small, (b) measurable, and
  (c) paired against low-leverage functionals: short branches, real
  anchors, uncertainty penalties, value-aware losses (04). Every rule
  is the simulation lemma's pairing plus the max-bias caveat, budgeted.
```

## What Remains Open

A priori (not error-curve-plugged) rollout-length theory for deep
models; compounding in *latent* value-equivalent models (04) where TV
has no ambient meaning; and a quantitative theory of model
exploitation — the adversarial-selection gap above has the order-
statistics shape of `rlhf_mathematics/05`'s Goodhart scaling and, as of
this writing, no analogous empirical law.
