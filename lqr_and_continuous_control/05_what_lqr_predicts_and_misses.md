# What LQR Predicts And What It Misses

## The Testbed Argument

Recht's "A Tour of RL" thesis: before trusting an RL method's story on
Atari, run it on LQR — the one problem where the optimal answer, the
model-based baseline, and every gap are computable exactly. This file
calibrates the instrument: which deep-RL phenomena LQR reproduces
faithfully, and which it structurally cannot.

## The Worked Comparison: Scalar LQR

Take `x_{t+1} = a x_t + b u_t + w_t`, unknown `(a, b)`, scalar. Three
strategies to reach a near-optimal controller:

**Model-based (identify, then certainty-equivalent control).** Estimate
`(a, b)` by least squares from `n` transitions (excitation supplied by
input noise); plug into the Riccati formula (notebook 01). Errors:
parameter error `~ 1/sqrt(n)` (self-normalized regression bounds,
`concentration_toolkit/03`), and the Riccati solution is
locally Lipschitz in `(a, b)` ⇒ controller suboptimality
`J(K-hat) - J* ~ C/n` (squared, since `J` is smooth with vanishing
gradient at `K*`).

**Model-free policy gradient (zeroth-order on `J(K)`).** Estimate
`grad J(K)` from perturbed rollouts (two-point evaluations of a scalar
function of a matrix): per-step gradient variance scales with rollout
return variance and the perturbation dimension; combined with
notebook 02's linear convergence, sample complexity is polynomial but
with a dimension-and-variance overhead that model-based methods simply
do not pay: **orders of magnitude worse in samples on the same
problem** — and this ordering (model-based ≫ policy search in sample
efficiency, when a low-dimensional model class is correct) is a robust
LQR prediction that transfers to practice.

**Model-free value-based.** Fit the quadratic `Q_K` by LSTD
(`stochastic_approximation/04` with quadratic features — exact class,
Bellman-complete per notebook 01); policy-iterate. Sits between the two:
the value class is `O(dim^2)` parameters vs the model's `O(dim^2)` too —
on LQR the separation from model-based is modest, which is *also*
informative (see "what it misses," item 3).

## What LQR Predicts Correctly

```text
1. the sample-efficiency ordering above, including its driver:
   variance of return-based estimates vs variance of one-step
   regression residuals (the same comparison as MC-vs-TD,
   stochastic_approximation/02–03, resolved the same way);
2. PG's conditional global convergence and its hypotheses — coverage/
   excitation and feasible initialization (notebook 02's audit) —
   including plateau behavior when excitation degenerates;
3. the value of baselines/variance reduction: zeroth-order LQR gradient
   estimators improve exactly per the control-variate calculus of
   policy_gradient/02;
4. fragility to horizon: LQR with rho(A-BK) -> 1 (slow closed loops)
   inflates every constant — the 1/(1-gamma) phenomenology
   (total_reward_ssp/01's "contraction slot") in exact form;
5. robustness/optimization tension: aggressive optimal K trades
   stability margin for performance — visible as the gap between
   J-optimal and robustly-stabilizing controllers; the overoptimization
   shape of rlhf_mathematics/05 (push the objective, lose the margin
   you weren't measuring) in its oldest form.
```

## What LQR Structurally Cannot Show

```text
1. EXPLORATION: LQR with input noise is fully excited — there is no
   needle to find. Nothing about regret_and_exploration/'s combination
   locks, optimism, or eps-greedy's exponential failure has an LQR
   analogue (quadratic landscapes have no deceptive local structure);
2. REPRESENTATION LEARNING: the correct features (quadratics) are known
   and complete. The entire content of linear_mdps_and_completeness/ —
   realizability vs completeness, learned features — is assumed away;
   any method looks feature-robust on LQR;
3. BELLMAN-COMPLETENESS AS A GIFT: value-based methods look
   artificially good (item 3 above) because the class is exactly
   T-closed — LQR cannot exhibit the deadly triad's divergences in
   their virulent form;
4. DISCRETE/COMBINATORIAL structure: no argmax discontinuities, no
   distributional control pathologies (distributional_rl/03), no
   switching-surface kinks (those need constraints or minimum-time
   costs — notebook 03);
5. REWARD MISSPECIFICATION: quadratic costs are dense in nothing;
   Goodhart dynamics (rlhf_mathematics/05) need a reward MODEL, absent
   here.
```

## The Meta-Lesson

LQR isolates the **optimization and estimation** content of RL from its
**exploration and representation** content. A method's LQR performance
tests the former only; the honest benchmark battery pairs LQR with a
combination lock (`regret_and_exploration/04`) — the two smallest
problems that together span the four difficulties. A method that fails
LQR is broken; a method that only shines on LQR has demonstrated
nothing about the half of RL that is hard.

## What Remains Open

Sharp instance-optimal rates for adaptive LQR (regret for unknown
systems: `sqrt(T)` achieved, constants and burn-in still moving);
partial observability (LQG with unknown model — the frontier where
LQR-style analysis meets pomdps_and_information_state/); and a
principled "LQR + lock" composite benchmark theory — currently folklore,
worth formalizing.
