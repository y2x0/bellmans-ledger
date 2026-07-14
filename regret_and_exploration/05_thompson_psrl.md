# Thompson Sampling And Posterior Sampling For RL

## The Algorithm

Optimism (01–04) engineers deliberate overestimation. Posterior sampling
replaces engineering with probability:

```text
PSRL: maintain a posterior over MDPs; at each episode k,
      SAMPLE one MDP  M_k ~ posterior,
      solve it exactly (mdp_foundations/), act with its optimal policy.
```

No bonuses, no confidence sets, no union bounds in the algorithm — all
structure moves into the prior and the analysis.

## The One Identity That Powers The Analysis

Let `M*` be the true MDP (random, under the Bayesian view), `M_k` the
sampled one, `F_k` the pre-episode-`k` history. **Conditioned on `F_k`,
`M_k` and `M*` are identically distributed** (that is what sampling from
the posterior means). Hence for any `F_k`-measurable function `f`:

```math
\mathbb{E}\big[f(M_k)\mid\mathcal{F}_k\big]
=
\mathbb{E}\big[f(M^*)\mid\mathcal{F}_k\big]
\qquad\text{("posterior matching").}
```

**Bayesian regret decomposition.** With `pi_k` optimal for `M_k`, and
writing `V_M^pi` for value in MDP `M`:

```math
\mathbb{E}\big[V^{\pi^*}_{M^*}-V^{\pi_k}_{M^*}\big]
=
\underbrace{\mathbb{E}\big[V^{\pi^*}_{M^*}-V^{\pi_k}_{M_k}\big]}_{=\ 0\ \text{by matching}}
+\ \mathbb{E}\big[V^{\pi_k}_{M_k}-V^{\pi_k}_{M^*}\big].
```

The first bracket vanishes because `V^{pi*}_{M*} = f(M*)` and
`V^{pi_k}_{M_k} = f(M_k)` for the same functional
`f(M) = (value of M's own optimal policy)`. What remains is a **model
estimation error along the played policy** — exactly the quantity the
confidence-width machinery of 02–03 already bounds, *without needing the
widths to be valid upper bounds*:

```math
\mathrm{BayesReg}(K)
\ \le\
\mathbb{E}\sum_k\Big|\big(\hat P^k-P\big)\text{-terms along }\pi_k\Big|
\ =\
\tilde O\big(H\sqrt{SAK}\big),
```

by the same telescope + pigeonhole + (Bernstein) total-variance argument,
run in expectation. (Osband–Russo–Van Roy 2013; the Bernstein refinement
carries over.)

**The conceptual exchange:** optimism buys `V-bar >= V*` *pointwise on a
high-probability event* and pays for it with explicit, inflated,
union-bounded bonuses. PSRL gets optimism **in expectation, exactly, for
free** — the sampled model overshoots and undershoots symmetrically, and
the analysis only ever needed the overshoot *in expectation*.

## Why PSRL Wins Empirically

```text
1. no union-bound slack: bonuses in 02 are inflated by log(SAHK/delta)
   over every cell; the posterior's implicit widths are calibrated per-cell
   to the actual data;
2. coherent joint uncertainty: confidence-set methods treat cells
   independently; a posterior can carry correlations (e.g., "all states in
   this region behave alike"), concentrating exploration;
3. the sampled MDP induces DEEP exploration natively: committing to one
   coherent optimistic model for a whole episode drives the combination
   lock of 04 as effectively as UCB bonuses — per-step independent noise
   (epsilon-greedy, or per-step posterior resampling!) does not. Sampling
   frequency is load-bearing: once per episode, not once per step.
```

Point 3 is the design principle behind deep exploration heuristics:
bootstrapped DQN (an ensemble member per episode ~ a posterior sample),
noisy nets (parameter-space noise held fixed within an episode).

## The Frequentist Gap, Stated Honestly

Everything above bounds **Bayesian** regret — expectation under the prior.
For worst-case (frequentist) regret:

```text
- bandits: Thompson sampling IS frequentist-optimal (Agrawal–Goyal,
  Kaufmann et al.) — the posterior's tails self-correct;
- episodic MDPs: vanilla PSRL's worst-case optimality is NOT established;
  known frequentist results either modify the algorithm (optimistic
  sampling: take max over several posterior draws) or pay extra factors.
  The obstruction: a badly matched prior can under-explore for a long
  time, and the matching identity gives no pointwise guarantee.
```

## Load-Bearing Audit

```text
posterior matching   -> needs EXACT posterior sampling; approximate
                        posteriors (deep ensembles, variational) void the
                        identity, and no general theory prices the
                        approximation
f(M) same functional -> the cancellation requires pi_k optimal FOR M_k;
                        solving the sampled model approximately leaks the
                        approximation error into regret linearly
prior correctness    -> Bayesian regret is an average over the prior; a
                        wrong prior shifts the average — the frequentist gap
```

## What Remains Open

Frequentist optimality of unmodified PSRL in MDPs; principled regret
accounting for *approximate* posteriors (the only kind deep RL can
compute); prior design as a formal handle on structured exploration.
