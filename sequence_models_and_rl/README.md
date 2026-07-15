# Sequence Models And RL

This family asks:

```text
Is reinforcement learning necessary at all, or can a sequence model
conditioned on desired outcomes replace it — and where exactly does
conditioning fail as control?
```

Decision Transformer (`papers/decision-transformer-chen-2021.pdf`)
proposes RL as supervised sequence modeling: predict actions conditioned
on the return you want (01). The proposal has a precise flaw already
proved in this repository — conditioning on outcomes is the exact
risk-seeking pathology of `regularized_mdps_and_duality/02` — plus a
stitching deficit relative to dynamic programming (02); and a calibrated
account of when it wins anyway closes the family (03).

Prerequisites: `regularized_mdps_and_duality/02`,
`offline_rl_and_ope/04`, `imitation_and_inverse_rl/01`,
`rlhf_mathematics/04`.

## Folder Map

```text
01_decision_transformer.md      return-conditioned autoregression; upside-
                                down RL; what the model actually estimates
02_conditioning_is_not_control.md  P(a | high G) vs argmax: the two
                                theorems against conditioning
03_when_sequence_modeling_wins.md  the honest scoreboard vs offline RL
```
