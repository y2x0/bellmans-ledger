# The Hardness Of POMDPs

## The Two Theorems

```text
1. (Papadimitriou–Tsitsiklis 1987) Finite-horizon POMDP optimality
   (does a policy achieve value >= V?) is PSPACE-complete — even with
   deterministic transitions and horizon <= |S|.
2. (Madani–Hanks–Condon 1999) Infinite-horizon undiscounted POMDP
   optimality (and eps-optimality for several criteria) is UNDECIDABLE.
```

For calibration: MDPs are P-complete (LP-solvable,
`mdp_foundations/06`); PSPACE contains NP; undecidable is not a
complexity class but a cliff. Partial observability is not "MDPs but
harder" — it changes the computational species of the problem.

## What The PSPACE Reduction Actually Encodes

Membership in PSPACE: enumerate policy trees depth-first — a
horizon-`H` policy needs only the current branch in memory
(`O(H)` space), and values along a branch are computable on the fly.
The instructive half is hardness, by reduction from **QBF**
(truth of `∃x_1 ∀x_2 ∃x_3 ... phi`):

```text
- the agent CHOOSES assignments for the existential variables:
  action = set x_i true/false;
- the HIDDEN STATE carries the universal variables: nature's uniform
  choice, unobserved — the agent must succeed for ALL settings, which
  under optimal value = 1 requirements is exactly the ∀;
- the clause-checking is a deterministic gadget walk; the observation
  channel is designed so the agent cannot see which universal branch it
  is on — forcing its (single) policy to hedge across all of them
  simultaneously.
```

The ∃∀-alternation is the whole point: **an unobserved adversarial-ish
branch under a single policy is a ∀ quantifier**, and stacking H of
them climbs the polynomial hierarchy to PSPACE. In MDPs (full
observation) the agent re-plans per branch — quantifiers collapse to a
single ∃ over policies evaluated by expectation: back to P.

## What Undecidability Actually Encodes

Infinite-horizon: the reduction is from the **emptiness problem for
probabilistic finite automata** (does some word push acceptance
probability above a threshold?) — itself undecidable (Paz 1971;
via encoding, ultimately, Post-correspondence-flavored problems). A
POMDP policy's action sequence plays the automaton's input word; the
belief plays the automaton's state distribution; "value > V" becomes
"acceptance > threshold." The undecidability lives in the interaction
of an *infinite* horizon with *exact* thresholds on a continuous belief
space — the belief trajectory can encode unbounded computation.

```text
scope notes (what is NOT undecidable):
- discounted eps-optimal planning IS computable (the horizon
  effectively truncates: gamma^H tail bounds, mdp_foundations/02 —
  complexity exponential, but decidable);
- finite-memory (finite-state-controller) policy optimization is
  decidable but NP-hard even for small controllers;
- the theorems concern WORST-CASE model-based planning with known
  (P, Z): learning does not rescue you from them, it adds statistics
  on top.
```

## The Load-Bearing Audit For Practice

What the hardness does and does not license concluding:

```text
1. no architecture escapes worst-case: any agent class (RNN,
   transformer-memory, belief-approximator) that could efficiently
   solve all POMDPs would collapse PSPACE — memory design is heuristic
   BY NECESSITY, not by immaturity of the field;
2. every tractability claim is a DISTRIBUTIONAL claim: "our method
   handles partial observability" must implicitly mean "on instances
   whose information structure is benign" — e.g. short-order
   dependence (frame stacking suffices: value_based_deep_rl/01),
   rapidly-identifying observations (beliefs concentrate; the
   effective problem re-Markovizes), or decodable block structure
   (block MDPs, linear_mdps_and_completeness/04 — where poly
   guarantees ARE possible because the hidden state is a function of
   the observation);
3. the QBF gadget names the enemy precisely: LONG-RANGE, NEVER-
   RESOLVED ambiguity that the policy must hedge against — problems
   where what you will never observe still matters. Benchmarks without
   that property test memory capacity, not POMDP-hardness.
```

## The Complexity Table

```text
problem                                        complexity
MDP, finite/discounted                          P-complete
POMDP, finite horizon                           PSPACE-complete
POMDP, infinite horizon, exact/eps (undisc.)    undecidable
POMDP, discounted eps-optimal                   decidable, EXP-hard
finite-state controller, fixed size             NP-hard
block MDP / decodable POMDP (learning)          poly (with function
                                                approximation caveats)
```

## What Remains Open

The mapped territory is worst-case; the unmapped is *parameterized*
hardness — a satisfying parameter that measures "how much unresolved
ambiguity matters" (candidates: belief-covering dimension, information-
decay rates, revealing-observation margins) and cleanly separates the
tractable regimes practice lives in from the QBF-shaped ones. Notebook
05's AIS losses are the current best vocabulary for this and are not
yet a complexity theory.
