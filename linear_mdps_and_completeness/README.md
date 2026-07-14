# Linear MDPs And Bellman Completeness

This family asks:

```text
The deadly triad says function approximation can destroy RL
(stochastic_approximation/06). What structural condition on the MDP-and-
class pair provably restores it — and is "my class can represent Q*"
that condition?
```

The answer to the second question is **no**, and internalizing why —
realizability is not completeness — is the single deepest conceptual point
in modern RL theory. The family's arc: the linear MDP model and its closure
properties (01); the algorithm it licenses, LSVI-UCB (02); the
realizability/completeness split with counterexamples (03); dimensions that
generalize beyond linear (04); the offline exponential lower bound that
shows coverage + realizability still fails (05); and what any of this says
about deep networks (06).

Prerequisites: `concentration_toolkit/03–04`, `regret_and_exploration/01–02, 07`,
`stochastic_approximation/04, 06`.

## Folder Map

```text
01_linear_mdp_closure.md            the model; the two closure properties; rank
02_lsvi_ucb.md                      ridge regression + ellipsoid bonus; regret
03_realizability_vs_completeness.md the split; non-monotonicity; hardness of
                                    realizability alone
04_bellman_rank_bilinear.md         average Bellman errors reveal themselves;
                                    poly-sample RL beyond linear
05_offline_linear_lower_bound.md    Wang–Foster–Kakade: coverage + realizable
                                    Q* still exponential
06_what_deep_networks_change.md     the honest synthesis
```
