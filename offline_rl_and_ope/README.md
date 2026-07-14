# Offline RL And Off-Policy Evaluation

This family asks:

```text
Given a fixed dataset collected by someone else's policy, what can be
certified about a new policy — and what must be refused?
```

This is the regime of replayed logs, medical/industrial data, and RLHF
reward models. Two problems:

```text
OPE  (evaluation):  estimate J(pi) from data ~ pi_b        (01–03)
OPO  (optimization): output a good pi from data ~ pi_b     (04–06)
```

The family's arc mirrors `regret_and_exploration/` in a mirror: there the
agent could visit what it was unsure about, and *optimism* was the correct
bias; here it cannot, and **pessimism** is — bias every estimate against
what the data cannot support. The exponential-in-horizon variance of naive
IS (01) is repaired by structure (02–03); naive fitted methods need
coverage of *all* policies while pessimism needs coverage of *one* (04);
and the fitted-Q error propagation theorem (06) prices what remains.

Prerequisites: `concentration_toolkit/`, `stochastic_approximation/02`,
`linear_mdps_and_completeness/03, 05`.

## Folder Map

```text
01_is_variance_lower_bound.md      trajectory IS: exp(H) variance, proved
02_doubly_robust.md                DR: bias if BOTH wrong; variance analysis;
                                   efficiency bound
03_marginalized_is_dice.md         state-density ratios: exp(H) -> poly(H)
04_pessimism_lcb.md                LCB-VI under single-policy concentrability
05_cql_operator.md                 CQL's implicit pessimistic operator, derived
06_fitted_q_error_propagation.md   the Munos–Szepesvári theorem, proved
```
