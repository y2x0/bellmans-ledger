# When Sequence Modeling Wins

## The Scoreboard, By Regime

Notebook 02's two theorems draw the map; this file fills in the
territories.

```text
regime                          winner            governing result
deterministic-ish dynamics,     DT / RvS          Thm 1 vacuous (no luck
expert-quality data                               channel); best-slice BC
                                                  is near-optimal
stitchable fragments            offline TD (CQL/  Thm 2: composition needs
(medium-replay)                 IQL/LCB)          value space
genuinely stochastic            offline TD w/     Thm 1 bites raw
environments                    pessimism; luck-  conditioning at exactly
                                adjusted DT       its preferred command
long horizon, sparse reward,    DT               no bootstrap => no error
no stitching needed                               propagation (offline/06's
                                                  (1-g)^{-2} absent);
                                                  attention assigns credit
                                                  across the whole context
tiny data                       TD methods        sequence models are
                                                  data-hungry estimators of
                                                  a richer object (a full
                                                  conditional family)
huge heterogeneous data,        sequence models   the estimator's regime:
many tasks                                        conditioning variables
                                                  (task, goal, return) are
                                                  just tokens; transfer
                                                  rides the architecture
```

## The Structural Advantages (real, and not about RL at all)

```text
1. optimization: supervised cross-entropy on stationary data — no
   deadly triad, no moving targets, no rotational game dynamics; every
   stability pathology catalogued in this repository is absent BY
   CONSTRUCTION (they were all properties of bootstrapped/adversarial
   objectives, not of decision-making per se);
2. scaling behavior: the loss inherits language-model engineering
   (schedules, scaling laws, infrastructure) — the practical reason the
   approach exists;
3. memory: the context window is an information state
   (pomdps_and_information_state/05) acquired for free — TD methods
   must bolt on recurrence and inherit its instabilities;
4. multi-task unification: goals, returns, language instructions,
   demonstrations are all conditioning tokens — the UVFA/SF transfer
   question (successor_representations/03) answered by architecture
   rather than by algebra, with correspondingly weaker guarantees but
   broader reach.
```

## The Synthesis Position

The field's working resolution — visible across Gato-style generalists,
robot foundation models, and LLM agents — is a division of labor the
theory endorses:

```math
\text{sequence model}
=
\text{the POLICY CLASS and the PRIOR}
\qquad
\text{RL}
=
\text{the IMPROVEMENT OPERATOR on top}
```

— pretrain by imitation/conditioning (cheap, stable, scalable;
`imitation_and_inverse_rl/01`'s regime), then improve by an operator
with actual guarantees: PPO/GRPO against verifiers
(`rlhf_mathematics/06–07`), search at decision time
(`search_and_planning/`), or pessimistic offline refinement
(`offline_rl_and_ope/04`). In the repository's coordinates: the sequence
model supplies `F` (and an initialization deep inside it); the classical
theory still supplies `T` and `d`. Return-conditioning's lasting
contribution is a spectacular `F` — the operator theory it proposed to
retire is the part that came back within two years, exactly at the two
joints (luck, stitching) where the theorems said it must.

## What Remains Open

Whether *any* purely-supervised objective can be made control-complete
(current luck-adjustments are partial); principled uncertainty for
sequence-model policies (the pessimism story has no conditioning
analogue yet); and the empirical frontier — internet-scale pretraining
plus thin RL — is moving faster than its theory, with this family's two
theorems as the only fixed points in sight.
