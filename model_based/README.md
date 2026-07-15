# Model-Based RL

This family asks:

```text
search_and_planning/ assumed a perfect simulator. What is the calculus
of planning with a LEARNED, wrong model?
```

One algebraic identity — the resolvent difference — prices everything:
the simulation lemma (01) converts model error to value error;
certainty equivalence with that lemma's Bernstein refinement turns out
minimax-optimal (02); optimism executed through models gives the
classical PAC/regret guarantees (03); the value-equivalence principle
says the model need only be right about what the planner asks (04); and
rollout error compounding — with the planner adversarially seeking the
model's errors — is the failure mode that shapes practice (05).

Prerequisites: `mdp_foundations/`, `concentration_toolkit/`,
`finite_time_td_q/04`, `regret_and_exploration/01–03`.

## Folder Map

```text
01_simulation_lemma.md         the resolvent-difference identity; both norms
02_certainty_equivalence.md    plug-in planning is minimax-optimal
03_optimism_through_models.md  R-MAX's explore-or-exploit; UCRL2
04_value_equivalence_muzero.md the model only owes the planner; MuZero
05_compounding_error_dyna.md   branched rollouts; the planner as adversary
06_muzero_paper.md             the paper itself: three functions, the K-step
                               loss as sampled VE constraints, Reanalyze
```
