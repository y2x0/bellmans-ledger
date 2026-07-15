# Conditioning Is Not Control

Two theorems separate return-conditioning from dynamic programming. Both
are already in this repository in other clothing; this file instantiates
them for sequence models.

## Theorem 1: Conditioning Buys Luck (stochasticity)

The conditional `P(a | s, G = g)` tilts toward actions that *co-occur*
with high returns, crediting the environment's randomness to the action.
This is **exactly** the exact-inference pathology of
`regularized_mdps_and_duality/02` — conditioning on optimality tilts
transition probabilities — now with the sequence model performing the
illegitimate conditioning by construction.

**The worked instance (the gamble, revisited).** One state, two actions:

```text
safe:  reward 0.5 surely.
risky: reward +1 w.p. 1/2, reward -10 w.p. 1/2.       (both terminal)
```

Behavior data: uniform over actions. Optimal: `safe`
(`0.5 > -4.5`). Now condition on the maximal observed return `G = 1`:

```math
\Pr(a=\text{risky}\mid G=1)
=
\frac{\Pr(G{=}1\mid\text{risky})\,\tfrac12}{\Pr(G{=}1)}
=
\frac{\tfrac12\cdot\tfrac12}{\tfrac12\cdot\tfrac12}
=1
```

— the return-conditioned policy at its own most-preferred command plays
the gamble **with certainty**, expected value `-4.5` against the
requested `+1`. No dataset size fixes this: the conditional is estimated
*correctly*; the flaw is that the conditional is the wrong object.
Control requires `argmax_a E[G | s, a]` — an *intervention* on the
action with an *expectation* over the environment; conditioning computes
`P(a | G)` — an *observation* of the action given a *selection* over the
environment. Do-vs-see, in one 2x2 table. ∎

(Deterministic environments collapse the two — no luck to select on —
which is why DT's strongest results are on near-deterministic suites,
and why `rlhf_mathematics/06`'s bandit reading of RLHF is unaffected:
verifier rewards on deterministic decoding have no luck channel. The
same collapse is why BoN — `rlhf_mathematics/04`, selection! — is sound
there: selection over YOUR OWN sampling randomness is legitimate;
selection over the WORLD's randomness is not. That one distinction is
this whole file.)

## Theorem 2: Conditioning Cannot Stitch (compositionality)

Dynamic programming's defining power is composing value across
trajectories that never co-occurred: if the data contains `A -> B`
(good) and `B -> C` (good) as *separate* episodes, the Bellman backup
composes them into `A -> B -> C` (`mdp_foundations/03`'s operator acting
on the shared state `B`; realized statistically by offline TD —
`offline_rl_and_ope/04`).

Return-conditioned supervised learning **cannot**: the training pairs
`(R-hat, s, a)` for state `A` carry only the returns of trajectories
*through* `A` in the data — all of which, by assumption, continued
suboptimally. The conditional at `A` for the stitched return
`g* = r(A->B) + r(B->C)` has **zero training support**: `P(a | A, g*)`
is undefined/extrapolated, not estimated. Formally: the conditional
model is an estimator of the behavior policy's return distribution
factors, and the stitched return is an off-support query of that
distribution — DP's composition happens in *value space* (shared
states), conditioning's generalization happens in *input space*
(`R-hat` as a feature), and no consistency of the estimator closes the
gap. The empirical signature matches: on D4RL's "medium-replay"-style
datasets (whose value is exactly stitchable fragments), TD-based
offline methods beat DT; on "expert"/"medium-expert" (best-slice
imitation suffices), DT is competitive. ∎ (mechanism)

## The Repairs, Each Conceding The Point

```text
Q-Learning DT / value-conditioned DT:  replace raw return with a LEARNED
                                       value/expectile as the conditioner
                                       — re-imports TD to fix stitching;
trajectory transformer + beam search:  re-imports planning (model-based
                                       composition) — model_based/05's
                                       costs return;
environment-stochasticity-aware DT     condition on return QUANTILES or
(ESPER etc.):                          on luck-adjusted statistics —
                                       re-imports exactly the
                                       distributional machinery
                                       (distributional_rl/) that raw
                                       conditioning discarded.
```

Every repair reinstates a component of the classical stack; the honest
summary is that return-conditioning is a *policy parameterization*, not
a replacement for the operator theory — notebook 03 scores where the
parameterization alone is nonetheless the right engineering.

## What Remains Open

A clean characterization of dataset/environment pairs where the
conditional equals the optimal policy (beyond "deterministic + expert
coverage"); finite-sample theory for conditioned sequence models as
policies (nothing like `offline_rl_and_ope/04`'s guarantees exists); and
the right luck-adjustment (Theorem 1's fix) that preserves the
supervised-learning simplicity that made the proposal attractive.
