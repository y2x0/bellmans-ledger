# Search And Planning

This family asks:

```text
When a model (or simulator) is available, how does online search trade
compute at decision time against learned value and policy priors?
```

The chain runs from bandit regret (UCB1) through tree search (UCT) to
AlphaGo, where four learned networks are grafted onto MCTS. The unifying
reading: **MCTS is value iteration restricted to the reachable subtree, with
a bandit rule allocating the backups** — and AlphaGo's networks are learned
initializations and proposal distributions for that computation.

Primary source: Silver et al., "Mastering the game of Go with deep neural
networks and tree search," Nature 529:484–489, 2016
(`papers/alphago-silver-2016.pdf`).

## Folder Map

```text
01_bandits_ucb.md        exploration as confidence bounds; the UCB1 regret theorem
02_mcts_uct.md           the four phases; UCT as recursive UCB; convergence
03_alphago.md            all training equations, the search equations, results
```
