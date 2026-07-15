# Imitation And Inverse RL

This family asks:

```text
Given demonstrations instead of rewards, what can be learned, at what
compounding cost — and is recovering the reward even a well-posed
problem?
```

The arc: behavior cloning pays a quadratic-in-horizon price for
distribution shift, and DAgger's interaction reduces it to linear — both
proved (01); inverse RL is ill-posed until an entropy prior selects a
unique answer, and MaxEnt IRL is another face of this repository's
Legendre/tilting transform (02); GAIL collapses the IRL-then-RL loop
into direct occupancy matching — the Fenchel machinery of the DICE
family, with a GAN as the metric (03,
`papers/gail-ho-ermon-2016.pdf`); failure modes close it (04).

Prerequisites: `mdp_foundations/06`, `regularized_mdps_and_duality/01,
06`, `offline_rl_and_ope/03`, `rlhf_mathematics/01`.

## Folder Map

```text
01_bc_compounding_dagger.md    the eps H^2 theorem; DAgger's eps H via
                               online learning
02_maxent_irl.md               ill-posedness; feature matching + max
                               entropy = exponential family over paths
03_gail_occupancy_matching.md  IRL's dual is occupancy matching; the GAIL
                               objective derived
04_failure_modes.md            reward ambiguity, shift, and the
                               demonstrator ceiling
```
