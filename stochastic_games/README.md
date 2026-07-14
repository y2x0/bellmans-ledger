# Stochastic Games

This family asks:

```text
What survives of the Bellman machinery when the environment contains
another optimizer — and what provably breaks in self-play?
```

Historical inversion worth savoring: Shapley proved the minimax Bellman
operator contracts in **1953** — four years before Bellman's MDP paper
(`mdp_foundations/07`). Game value theory is not an extension of RL; RL
is the one-player special case.

The arc: the Shapley operator and existence of the value (01); why
"everyone best-responds" is not an algorithm — cycles (02); the
no-regret repair — regret matching, CFR, and what average iterates mean
(03); self-play as generalized policy iteration, done right (04); and
the explicit dynamics of independent learners failing (05).

Prerequisites: `mdp_foundations/03–05`, `policy_gradient/`,
`search_and_planning/03`; `regularized_mdps_and_duality/05` for the
no-regret connections.

## Folder Map

```text
01_shapley_operator.md            minimax backup contracts; the value exists
02_failure_of_greedy_dynamics.md  Shapley's fictitious-play cycle; naive PI
                                  fails in games
03_regret_matching_cfr.md         Blackwell approachability -> regret matching;
                                  CFR's decomposition theorem
04_self_play_as_gpi.md            fictitious self-play, double oracle/PSRO;
                                  AlphaZero's self-play formalized
05_independent_learning.md        two Q-learners in matching pennies:
                                  replicator cycles, worked
```
