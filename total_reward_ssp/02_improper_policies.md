# Improper Policies: The Mixed Case

## The Realistic Assumption Set

Notebook 01's "all policies proper" is rarely true: most tasks contain
do-nothing loops (an Atari agent standing still forever). The
Bertsekas–Tsitsiklis mixed assumptions (stated for costs; we keep the
reward convention and flag signs):

```text
(A1) at least one proper policy exists;
(A2) every improper policy has value -infinity from some state
     (in cost terms: improper policies incur +inf cost — e.g. every
     nonterminal step costs something strictly negative in reward).
```

## The Theorem

**Theorem (BT 1991).** Under (A1)+(A2): the Bellman equation `v = Tv` has
a **unique** solution `v*` among real-valued functions; `v*` is the
optimal value; a greedy policy w.r.t. `v*` is proper and optimal; and VI
converges to `v*` from any initialization `v_0 >= v*` (and from any
`v_0` under mild extra conditions).

*Proof architecture (where each assumption works).* Without a uniform
contraction (improper `pi` have `rho(Q_pi) = 1`), the argument is a
hybrid:

```text
1. (A1) gives one proper pi0: T^{pi0} is a weighted contraction
   (notebook 01) => a well-defined v_{pi0} anchors the values from below;
2. (A2) excludes the improper policies from optimality: any policy that
   loops forever pays -inf, so the sup over policies is attained among
   proper ones;
3. monotonicity + the anchor: iterate from v_0 >= v*; T's monotone
   decreasing sequence is bounded below by v_{pi0}; its limit solves
   v = Tv; uniqueness by a squeeze between T^k of an upper and lower
   anchor (each T^k application routes through SOME policy; improper
   routes are killed by (A2) in the limit, proper routes contract).
```

The proof is genuinely *not* Banach — it is Banach on a sub-family plus
monotone convergence plus an exclusion argument, and each ingredient is
visible in the failure below when (A2) is dropped. ∎ (architecture)

## The Counterexample Without (A2)

One nonterminal state `s`, two actions:

```text
action stay: remain at s, reward 0        (improper, value 0 — FINITE)
action go:   terminate,   reward -1       (proper)
```

The Bellman equation: `v(s) = max( v(s), -1 + 0 )` — i.e.
`v(s) = max(v(s), -1)`: **every** `v(s) >= -1` is a fixed point. The
uniqueness of notebook 01 is gone; worse, VI from `v_0 = 0` returns
`v = 0` and the greedy policy `stay` — improper, collecting nothing
forever, and whether that is "optimal" depends on a convention the
total-reward criterion cannot express (is an unending 0 better than a
terminal -1? total reward says yes; any real deadline says no). (A2) is
exactly the assumption that forbids the criterion from being indifferent
to non-termination: make `stay` cost anything (`reward -eps < 0` per
step) and its value drops to `-inf`, the fixed point snaps to unique
`v(s) = -1`, and greedy = `go`.

```text
practical echo: reward shaping with per-step penalties ("-0.01 per step")
is (A2) manufactured by hand — not a heuristic for "encouraging speed"
but the exact condition that makes the episodic Bellman equation
well-posed against loitering policies. Its absence is the standing bug
behind agents that learn to stall (and behind timeout mishandling,
notebook 04).
```

## SSP Q-Learning Under The Mixed Assumptions

The asynchronous SA proof (`stochastic_approximation/05`) survives with
the hybrid argument replacing the contraction: a.s. convergence of
Q-learning for SSP under (A1)+(A2), boundedness of iterates now a lemma
(via the proper anchor) rather than a gift of the sup-norm contraction
(Tsitsiklis 1994; Yu–Bertsekas for refinements). The practical reading:
episodic Q-learning is sound *because* environments/timeouts enforce
(A1) and reward design enforces (A2) — both, silently.

## What Remains Open

Weakening (A2) (zero-reward loops that are merely *suboptimal* rather
than infinitely bad — partial results via perturbation
`r -> r - eps`, `eps -> 0`); SSP with adversarial or heavy-tailed
termination; and the function-approximation mixed case, where even
stating the right uniqueness class is unresolved.
