# Positive And Negative Dynamic Programming

## The Setting

Total reward, `gamma = 1`, **no properness anywhere** — infinite
undiscounted horizons with sign-restricted rewards
(Blackwell 1967, Strauch 1966):

```text
NEGATIVE DP:  r <= 0 everywhere  (all cost; values in [-inf, 0])
POSITIVE DP:  r >= 0 everywhere  (all gain; values in [0, +inf])
```

Contraction is gone entirely; the surviving tool is **monotonicity plus
order-continuity** (`mdp_foundations/03`'s other property, now working
alone), and the two sign cases are strikingly *not* symmetric.

## Negative DP: Value Iteration Works

**Theorem.** With `r <= 0`, start VI at `v_0 = 0` (the largest possible
value). Then `v_{k+1} = T v_k` is monotone decreasing, its limit
`v_inf = lim v_k` satisfies `v_inf = T v_inf`, and `v_inf = v*` (the
optimal value over all policies). Moreover a greedy policy w.r.t. `v*`
(when the max is attained) IS optimal.

*Proof.* `v_1 = T v_0 <= v_0` since rewards are `<= 0`; monotonicity of
`T` propagates: `v_{k+1} <= v_k`. Decreasing and bounded below by `-inf`
(in the extended reals): pointwise limit exists. Exchange of limit and
`T` holds by monotone convergence (the expectation inside `T` of a
decreasing sequence — Beppo Levi downward, legitimate for `<= 0`
integrands). So `v_inf` is a fixed point. `v_inf >= v*`: each `v_k`
upper-bounds every policy's `k`-step truncated value, and truncation only
adds (nonpositive omitted terms ⇒ truncated `>=` full);
`v_inf <= v*`-side: greedy-at-the-limit achieves `v_inf` by a
verification argument that works *because* deviations are penalized —
overestimating continuation can only be corrected downward, and the
inequality chain closes. ∎

Greedy optimality survives in negative DP for the same reason: a policy
that is one-step consistent with `v*` cannot be *hurt* by the tail (the
tail is `<= 0` and already accounted pessimistically).

## Positive DP: The Asymmetry, And The Counterexample

In positive DP, VI from `v_0 = 0` increases monotonically to the least
fixed point, which does equal `v*` (Strauch) — but **greedy with respect
to `v*` can fail to be optimal**:

**Counterexample.** States `{1, 2, T}`; from state 1:

```text
action a:  terminate with reward 1                    (value 1)
action b:  move to state 2, reward 0
state 2:   single action: move to state 2' ... construct a chain
           2 -> 3 -> 4 -> ... where at state n one may terminate with
           reward 1 - 1/n, or continue.
```

Values: from state 2, continuing forever collects nothing, but
`v*(n) = sup_k (1 - 1/k) = 1` — the sup is approached, never attained.
Then at state 1: `q*(1, a) = 1`, `q*(1, b) = v*(2) = 1` — greedy is
indifferent, and may pick `b`; but *every* policy that picks `b` and
then keeps deferring attains `< 1` (or 0 if it defers forever), while
`a` attains exactly 1. The greedy tie hides an unattained supremum
downstream. **In positive DP, optimism about the tail is unsecured
credit**: `v*` reports the sup; no policy need cash it. (In negative DP
the same structure is harmless — suprema of `<= 0` tails are attained at
"stop.") ∎

The asymmetry in one line:

```text
negative DP: v* is a PROMISE OF COSTS — pessimistic bookkeeping, so
             one-step consistency certifies policies (verification easy,
             VI needs care from above);
positive DP: v* is a PROMISE OF GAINS — optimistic bookkeeping: VI from
             below is safe, but promises may be unattainable; optimal
             policies may fail to exist and greedy is not enough.
```

## Where This Touches Modern RL

```text
1. proofs by monotone convergence from a safe side reappear wherever
   contraction is unavailable: pessimistic offline values are a
   negative-DP-style construction (under-promise, then certify —
   offline_rl_and_ope/04); optimistic online values are positive-DP-style
   and correspondingly need the width machinery to make promises
   attainable (regret_and_exploration/01);
2. the unattained-sup pathology is the deterministic skeleton of
   "value overestimation with no policy to back it"
   (value_based_deep_rl/03) — there driven by noise, here by order
   theory: two independent roots of the same practical symptom;
3. goal-reaching RL with reward = 1 at the goal IS positive DP whenever
   reaching is not guaranteed: the theory says compute from below and
   distrust greedy ties — advice sparse-reward practitioners rediscover
   empirically.
```

## What Remains Open

Classical theory closed (Blackwell/Strauch/Bertsekas). The open ground
is the same as the previous notebooks': learning versions (sample-based
positive/negative DP with the correct one-sided convergence) are barely
studied — the one-sided monotone structure seems tailor-made for
one-sided concentration and has not been systematically exploited.
