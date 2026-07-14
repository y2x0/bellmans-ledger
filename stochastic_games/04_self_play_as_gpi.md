# Self-Play As Generalized Policy Iteration

## The Design Question

AlphaGo/AlphaZero (`search_and_planning/03`) train by self-play and it
works; notebook 02 proved naive mutual best response cycles. What
separates the working recipes from the cycling ones? Three mechanisms,
each a theorem.

## Mechanism 1: Fictitious Self-Play (best-respond to the average)

**Setup.** Maintain a pool of past strategies; each iteration, compute
(approximately) a best response to the **uniform mixture over the
pool**, add it to the pool.

**Theorem (zero-sum).** The pool's average strategy converges to the
Nash/minimax optimum.

*Why.* This is fictitious play (Robinson 1951 proves convergence of FP
in zero-sum games — by an elementary but intricate induction on the
support), which is itself a no-regret-flavored dynamic: best-responding
to the empirical average is "follow the leader," whose zero-sum
averages converge (though FTL alone is not no-regret in general —
zero-sum structure rescues it; regularized versions, i.e. *smooth* FSP,
get the clean `O(1/sqrt(T))` rates from notebook 03's machinery). ∎

The mechanism that kills notebook 02's cycle: the *target* moves at rate
`1/t` (one new member dilutes a size-`t` pool) — the two-timescale
principle (`stochastic_approximation/01`) with the opponent on the slow
clock. AlphaGo's uniform sampling from past selves
(`search_and_planning/03`'s "pool of previous iterations") is exactly
this, implemented statistically.

## Mechanism 2: Double Oracle / PSRO (best-respond to the equilibrium of the pool)

**Setup (McMahan et al. 2003).** Maintain pools `X, Y` of pure
strategies for both players. Iterate:

```text
1. solve the RESTRICTED matrix game over X x Y exactly (an LP — its
   size is the pool, not the game);
2. compute each player's best response to the opponent's restricted-
   game equilibrium mixture, over the FULL strategy space;
3. add the best responses to the pools; stop when neither improves.
```

**Theorem (finite convergence).** In a finite game, double oracle
terminates, and at termination the restricted equilibrium is a Nash
equilibrium of the full game.

*Proof.* Termination: each non-terminal iteration adds a strategy not
already in a pool (it strictly beats the current restricted equilibrium
value, which pool members by definition cannot); pools are subsets of a
finite set — the loop halts. At termination, no full-space best response
improves on the restricted equilibrium: the equilibrium's guarantee
extends to the full game (the defining inequalities of Nash hold against
all strategies, since the best ones were checked). ∎

Worst case the pools grow to the whole game; in practice equilibria have
small support and the pools stay tiny — double oracle is a *column
generation* method (LP decomposition transplanted to games). **PSRO**
(Lanctot et al. 2017) is double oracle with best responses computed by
RL (an inner `policy_gradient/`-family run) and the restricted game
estimated by simulation — plus generalized meta-solvers (replace Nash of
the pool by other mixtures) that interpolate toward Mechanism 1.

## Mechanism 3: AlphaZero — Search As The Best-Response Oracle

The AlphaZero loop, in this family's terms:

```text
opponent    = the current network ITSELF (symmetric self-play)
improvement = MCTS at every move (search_and_planning/02: the visit
              distribution is a policy improvement operator against
              the CURRENT net's play)
evaluation  = value head regression on self-play outcomes
```

Why it does not cycle like notebook 02: in **two-player zero-sum
symmetric** games, improving against yourself is improving against
exactly the opponent you will face — the game's symmetry makes the
self-play "environment" a consistent evaluation target (formally: the
exploitability of the current strategy is a potential-like quantity
that search-improvement decreases in the exact-improvement limit);
and the improvement operator is a *strict* strengthener (lookahead
dominates the prior it expands, `search_and_planning/02`'s
search-as-improvement reading). Zero-sum + symmetry + a genuine
improvement operator is the working triple; drop any one — general-sum
payoffs, asymmetric roles without pools, or improvement by best-response
argmax instead of dampened search — and the cycling risk of notebook 02
returns. League training (AlphaStar) is the acknowledgment: main agents
+ frozen exploiters + past snapshots = Mechanisms 1 and 2 bolted on
because pure Mechanism 3 was observed to cycle in the messier game.

## The Audit Table

```text
mechanism        opponent target          convergence guarantee
naive (02)       current iterate          NONE — cycles (Shapley)
FSP (1)          uniform past mixture     zero-sum: averages -> Nash
double oracle(2) restricted equilibrium   finite games: exact Nash,
                                          finite steps (support-sized)
AlphaZero (3)    self, via search         zero-sum symmetric + exact
                                          improvement: exploitability
                                          decreases; formal guarantees
                                          partial, empirics decisive
```

## What Remains Open

PSRO with approximate inner RL responses: error propagation through the
meta-game is only partially priced; general-sum PSRO's meta-solver
choice (Nash is neither unique nor the right target) is unresolved in
principle; and a full convergence account of the AlphaZero loop (search
depth + network error + self-play distribution shift jointly) does not
exist — the strongest system has the weakest theorem, the recurring
shape of this repository.
