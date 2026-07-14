# Exploration Lower Bounds

Two theorems: the minimax `Omega(sqrt(HSAT))` that certifies notebook 03,
and the exponential separation that convicts epsilon-greedy.

## Theorem 1: Omega(sqrt(H S A T)) For Any Algorithm

**The construction (JAO-style, episodic form).** A tree-shaped MDP:

```text
- a uniform "hallway" over ~S/2 states reachable in the first steps,
  regardless of actions (so every algorithm spreads over them);
- at each hallway state s, each of the A actions leads to one of two
  absorbing arms: a fair coin (reward Bernoulli(1/2) each remaining step)
  — except that ONE hidden (s*, a*) pair's arm pays Bernoulli(1/2 + eps).
```

Nature hides `(s*, a*)` uniformly among the `~SA/2` cells. The optimal
policy reaches `s*` and plays `a*`, gaining `~ eps * H` per episode over
any policy that doesn't.

**The information calculation** (`concentration_toolkit/05`). The only
statistical difference between "cell `(s,a)` is special" and "nothing is
special" is a Bernoulli bias `eps` observed on the `~H` steps of episodes
that commit to that cell. By the divergence decomposition, after `K`
episodes:

```math
\mathrm{KL}\big(\mathbb{P}_0\ \|\ \mathbb{P}_{(s,a)}\big)
\ \asymp\
\mathbb{E}_0\big[N(s,a)\big]\cdot H\cdot\varepsilon^2 ,
```

`N(s,a)` = episodes committing to the cell. Under the null, the budget
splits: `sum_{(s,a)} E_0[N(s,a)] = K`, so the **average** cell has
`E_0[N] ~ K/(SA)`, hence average KL `~ KHeps^2/(SA)`. By
Bretagnolle–Huber / Fano over the `SA/2` alternatives: unless

```math
\frac{K\,H\,\varepsilon^2}{SA}\ \gtrsim\ 1,
```

the algorithm cannot locate the special cell better than chance, and pays
`~ eps H` regret on a constant fraction of episodes:
`Reg >= c * K * eps * H` for the worst placement. Optimizing the
adversary's `eps ~ sqrt(SA/(KH))`:

```math
\mathrm{Reg}(K)\ \ge\ c\,K\,H\,\sqrt{\frac{SA}{KH}}
\ =\
c\,\sqrt{H\,SA\cdot KH}
\ =\
\Omega\big(\sqrt{H\,S\,A\,T}\big). \qquad\blacksquare
```

**Audit.** The `H` inside the square root arrives because each committed
episode yields `H` observations of the coin *and* `H` reward — information
and value scale together; constructions that decouple them (shorter
observation windows) give the time-inhomogeneous `Omega(sqrt(H^2 SAT))`
variant. The matching with notebook 03 is exact in `S, A, T` and matches in
`H` for the stationary setting proved here.

## Theorem 2: Epsilon-Greedy Needs Omega(A^H)

**The construction (the combination lock).** A deterministic chain:

```text
states 1 -> 2 -> ... -> H;  at each state, exactly ONE of the A actions
advances; every other action resets to state 1. Reward 1 only upon
reaching state H; all else 0.
```

The optimal policy collects reward every episode. Consider epsilon-greedy
(any epsilon schedule) around a greedy policy that has not yet found the
reward: **all Q-estimates are identically zero**, so "greedy" is arbitrary
and the behavior is effectively (at best) uniformly random among untried
paths. The probability that a single episode executes the one advancing
action `H` times in a row is

```math
\Pr\{\text{reach state }H\}\ \le\ \Big(\frac{1}{A}\Big)^{\!H}
\quad\text{(uniform play; any fixed greedy tie-break does no better in the worst case)},
```

so the expected number of episodes to see the *first* reward — before which
no learning signal exists at all, for ANY value-based update — is
`Omega(A^H)`. ∎

**What exactly is being convicted.** Not the Q-update — the exploration
*measure*. Epsilon-greedy perturbs actions **independently per step,
undirected by uncertainty**; its trajectory distribution puts exponentially
small mass on any specific deep path. Optimism (01–03) instead assigns the
untried cells *value*, so the greedy trajectory distribution concentrates
on unexplored paths — the same lock is opened in `O(poly(H, A))` episodes
by UCBVI: each episode either collects reward or permanently eliminates at
least one (state, action) bonus from being maximal (pigeonhole on
`sum 1/sqrt(N)`).

```text
This pair of facts is the formal content of value_based_deep_rl/04
failure 5 ("Montezuma's Revenge ~ 0%"): DQN's exploration measure is
epsilon-greedy's; the lock is the game's key-door structure. No loss
function fixes a trajectory measure.
```

## The Diagram The Two Theorems Draw

```text
                     per-episode information about the hidden structure
epsilon-greedy       exponentially small (undirected trajectory measure)
optimism             matched to width: visits = information = potential regret
any algorithm        bounded by the divergence decomposition
```

Lower bound 1 says even the best coupling of visits-to-information pays
`sqrt(HSAT)`; lower bound 2 says decoupling them pays exponentially. The
window between is the entire subject of exploration.

## What Remains Open

Lower bounds *within* structured classes (how much does a good prior or a
low-rank structure genuinely reduce the exponent — partial answers in
`linear_mdps_and_completeness/`); and reward-free / task-agnostic
exploration lower bounds, where the `SA` counting changes character.
