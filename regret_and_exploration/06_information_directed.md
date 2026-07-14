# Information-Directed Sampling

## The Question Optimism Cannot Ask

Optimism and PSRL both explore only where *reward might be*. But
information about the optimum can live at actions that are **known to be
suboptimal** — a cheap "informative sensor" arm. Optimism will never pull
it (its upper bound is low); a rule that explicitly prices information
will.

## The Information Ratio

Bayesian setting: unknown environment, optimal action `A*` (a random
variable under the posterior). At time `t`, for a candidate action
distribution... define for the chosen rule the one-step quantities

```math
\Delta_t(a)=\mathbb{E}_t\big[r(A^*)-r(a)\big]
\quad\text{(expected instant regret)},
```

```math
g_t(a)=I_t\big(A^*;\,Y_{t,a}\big)
\quad\text{(information the observation from }a\text{ carries about }A^*\text{)},
```

`I_t` = mutual information under the posterior at `t`. IDS chooses the
(possibly randomized) action minimizing the **information ratio**

```math
\Gamma_t(\mu)=\frac{\big(\mathbb{E}_{a\sim\mu}\Delta_t(a)\big)^2}{\mathbb{E}_{a\sim\mu}\,g_t(a)},
\qquad
\mu_t=\arg\min_\mu\ \Gamma_t(\mu).
```

Squared regret per bit: spend regret only at the best exchange rate.

## The Generic Regret Bound

**Theorem (Russo–Van Roy).** If `Gamma_t <= bar-Gamma` a.s. for all `t`,
then

```math
\mathrm{BayesReg}(T)
\ \le\
\sqrt{\bar\Gamma\ H(A^*)\ T}
\ \le\
\sqrt{\bar\Gamma\ T\,\log|\mathcal{A}|},
```

`H(A*)` the entropy of the optimum under the prior.

*Proof.* By Cauchy–Schwarz on the sum of per-step regrets:

```math
\mathrm{Reg}(T)=\sum_t\mathbb{E}\,\Delta_t
\le\sqrt{T\sum_t\big(\mathbb{E}\Delta_t\big)^2}
=\sqrt{T\sum_t\Gamma_t\ \mathbb{E}\,g_t}
\le\sqrt{T\,\bar\Gamma\,\sum_t\mathbb{E}\,g_t},
```

and the total information is budgeted by the chain rule:
`sum_t E[g_t] = I(A*; whole history) <= H(A*)` — **you cannot learn more
about the optimum than its entropy.** ∎

The proof is two lines and contains a genuinely different accounting from
01–04: no confidence widths, no pigeonhole — an *entropy budget* replaces
the *visit-count budget*. Bounding `bar-Gamma` is where the per-setting
work lives:

```text
K-armed bandit:            Gamma <= K/2      ->  Reg <= sqrt(KT log K / 2)
linear bandit, dim d:      Gamma <= d/2      ->  Reg <= sqrt(dT log|A|/2)
full-information:          Gamma <= 1/2
(each proved by decomposing Delta and g against the posterior over A* —
the "one unknown arm" structure makes both sides quadratic forms in the
same object)
```

## Where Optimism Is Provably Beaten

The canonical separation (Russo–Van Roy's sparse linear bandit): actions =
standard basis vectors plus one "revealing" action whose reward is tiny but
whose observation identifies the hidden parameter exactly.

```text
- any optimistic algorithm: the revealing action's upper confidence bound
  is dominated (its reward is known to be small), so it is never pulled;
  regret Omega(number of candidate arms).
- IDS: pulls the revealing action once (its g is huge, so Gamma minimized),
  then exploits; regret O(1).
```

Optimism's blind spot is structural, not a constant: it conflates "worth
visiting" with "possibly rewarding," and the counterexample splits those.
PSRL shares the blind spot (it also only plays candidate-optimal policies).

## RL Instantiation, And The Honest Costs

For episodic MDPs, IDS-style algorithms treat a whole policy as the action
and a trajectory as the observation; bounding the information ratio yields
regret comparable to 03's with structure-adaptive improvements. The costs:

```text
1. computing mutual information under a posterior over MDPs is intractable
   beyond small/conjugate cases — practical versions substitute variance
   proxies for g_t (variance-IDS), losing part of the guarantee;
2. the theory is Bayesian through and through (entropy budget of A*);
   frequentist IDS analogues exist but the elegance — and some results —
   do not transfer;
3. one-step ratio myopia: Gamma_t prices immediate information; in MDPs,
   information may require multi-step plans to reach (the lock of 04) —
   handled by policy-level actions, at exponential action-space cost.
```

## Position In The Family

```text
01–03  optimism: visit where you might be wrong AND rewarded
05     PSRL: sample beliefs, act coherently on them
06     IDS: price information explicitly; the only member that buys
       information at known-bad actions — and the proof that sometimes
       one must
```

## What Remains Open

Tractable information proxies with end-to-end guarantees in deep RL
(current exploration bonuses — RND, pseudo-counts — are unpriced heuristics
for `g_t`); the correct multi-step information ratio for MDPs.
