# LQR And Continuous Control

This family asks:

```text
What does the one exactly-solvable continuous-state control problem
teach — and mislead — about everything else in this repository?
```

The linear-quadratic regulator is to RL what the harmonic oscillator is
to physics: value functions computable in closed form (Riccati, 01), a
nonconvex policy landscape that nonetheless yields to gradient methods
with a provable mechanism (02), the continuous-time limit where Bellman
becomes a PDE (03), a change of variables under which control is exactly
linear algebra (04), and a calibrated list of which deep-RL phenomena it
does and does not reproduce (05).

Prerequisites: `mdp_foundations/`, `policy_gradient/03`,
`global_convergence_of_pg/01`; `regularized_mdps_and_duality/01–02`
for 04.

## Folder Map

```text
01_lqr_riccati.md                 quadratic values by induction; the DARE;
                                  certainty equivalence — the exception
02_policy_gradient_on_lqr.md      J(K) nonconvex yet gradient-dominated;
                                  the PDL in quadratic clothing
03_hjb_viscosity.md               continuous time; where classical solutions
                                  die; viscosity, statement and instinct
04_linearly_solvable_control.md   KL control costs make Bellman LINEAR;
                                  desirability; path-integral policies
05_what_lqr_predicts_and_misses.md the testbed argument, calibrated
```
