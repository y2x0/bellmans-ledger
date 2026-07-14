# The Optimism Principle

## The Generic Lemma

An algorithm maintains, at each episode `k`, value estimates
`bar-V^k = (bar-V_1^k, ..., bar-V_H^k)` and acts greedily with respect to
them. Call the estimates **valid optimism** on an event `G` if

```math
\bar V_h^k(s)\ \ge\ V_h^*(s)
\qquad\forall h,s,k\quad\text{on }G .
```

**Lemma (regret <= sum of on-trajectory widths).** On `G`:

```math
\mathrm{Reg}(K)
\ \le\
\sum_{k=1}^K\Big(\bar V_1^k(s_1^k)-V_1^{\pi_k}(s_1^k)\Big),
```

and if the optimistic values are built by backups
`bar-V_h = r + bonus + hat-P bar-V_{h+1}` while the true `V^{pi_k}` obeys
the true backup, the difference telescopes along the visited trajectory:

```math
\bar V_1^k(s_1^k)-V_1^{\pi_k}(s_1^k)
\ \le\
\mathbb{E}_{\pi_k}\Bigg[\sum_{h=1}^{H}\Big(
2\,b_h^k(s_h,a_h)
+\big(\hat P^k-P\big)(\cdot\mid s_h,a_h)^\top\big(\bar V_{h+1}^k-V_{h+1}^{\pi_k}\big)
\Big)\Bigg],
```

so that, once the estimation terms are absorbed into the bonuses (02's
job), **regret is bounded by the sum of the bonuses the algorithm itself
computed, along the states it actually visited.**

*Proof of the first display.* Greediness + optimism:
`V_1^* <= bar-V_1^k` (optimism), so
`V_1^* - V_1^{pi_k} <= bar-V_1^k - V_1^{pi_k}`. ∎

*Proof of the telescope.* Per step, subtract the two backup equations for
the action `a_h = pi_k(s_h)` actually taken (same action on both sides —
this is where greediness w.r.t. `bar-V` is used a second time); the
difference at `h` equals bonus + estimation error + `P`-expectation of the
difference at `h+1`; iterate to `H`. ∎

## Why This Structure Is The Whole Field

Three readings of the lemma:

**1. Exploration without exploration.** Nothing explores "on purpose."
The algorithm always exploits its optimistic model; the *bonus* makes
under-visited cells look valuable, so exploitation visits them; visiting
shrinks their bonus (pigeonhole/potential arguments, 02). Exploration is an
emergent property of systematic overestimation — this is UCB1's
self-extinguishing exploration (`search_and_planning/01`) lifted to
sequential states.

**2. The regret-information exchange.** The right side sums confidence
widths **at visited cells**. The divergence decomposition
(`concentration_toolkit/05`) says information about the environment is
*also* counted at visited cells. Optimism forces the algorithm to spend its
visits where its uncertainty (potential regret *and* potential information)
is largest — the mechanism by which upper bounds can meet lower bounds.

**3. The two failure modes it forbids.** Pessimism (widths subtracted)
never visits uncertain cells: no information, possibly permanent
suboptimality. Zero-width greed (pure exploitation of point estimates)
commits to early noise. Note the exact mirror-image in offline RL
(`offline_rl_and_ope/04`): with **no** ability to visit, pessimism becomes
the correct principle and optimism the broken one. The principle is not
"optimism is good"; it is *"bias the estimate against the thing you cannot
change"* — the data in online RL, the policy's reach in offline RL.

## The Contrast Case: Bandits

At `H = 1` the telescope is trivial and the lemma collapses to UCB1's
analysis: regret <= sum of pulled arms' widths <= (pigeonhole)
`sum_i Delta_i * (log/Delta_i^2)`. Everything that is genuinely *RL* about
02–04 is in the two places `H > 1` bites:

```text
1. the estimation error (P-hat - P)V involves a RANDOM FUNCTION V built
   from the same data -> uniform convergence / fixed-function tricks
   (concentration_toolkit/04);
2. errors at step h are amplified by the remaining horizon -> where the
   H factors in the regret come from, and where Bernstein + total
   variance (03) saves one of them.
```

## Valid Optimism Is Not Free

The lemma's hypothesis — `bar-V >= V*` **with high probability,
simultaneously for all `k, h, s`** — is exactly a uniform confidence
statement, and constructing bonuses that certify it is where all the
concentration machinery is spent (02). Overshooting the necessary bonus
keeps validity but inflates the width sum: sharp regret = the *smallest*
valid bonus. The Hoeffding/Bernstein gap of 02 vs 03 is precisely
"valid but lazy" vs "valid and tight."

## What Remains Open

The principle itself is understood; what remains contested is its
*scope* — 06 exhibits settings where any optimistic algorithm is
provably suboptimal (information can be worth buying at cells optimism
refuses to visit), and 07 shows that beyond linear-ish classes we cannot
even compute valid optimism efficiently.
