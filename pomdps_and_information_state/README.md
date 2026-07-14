# POMDPs And Information State

This family asks:

```text
The entire repository rests on one assumption — the state is observed
(00_problem_setup). What is the exact price of losing it?
```

The answers, in order: the belief state restores Markovness at the cost
of a continuous state space (01); the value over beliefs has exploitable
structure — piecewise linear and convex — with exact algorithms that
explode combinatorially (02); the explosion is fundamental, not
algorithmic: PSPACE-complete finite-horizon, undecidable infinite-
horizon (03); beliefs are not the only sufficient statistic — predictive
representations can be strictly smaller (04); and deep RL's recurrent/
frame-stacked agents are approximate information states with a
quantifiable value loss (05).

Prerequisites: `mdp_foundations/01–04`; `value_based_deep_rl/04`
(failure 3) is the practical scar this family explains.

## Folder Map

```text
01_belief_mdp.md                 Bayes filter; sufficiency proved;
                                 the continuous-state price
02_pwlc_alpha_vectors.md         piecewise-linear convexity by induction;
                                 exact backups; the tiger problem worked
03_hardness.md                   PSPACE-completeness; undecidability;
                                 what the reductions actually encode
04_predictive_state_reps.md      the system-dynamics matrix; PSRs can beat
                                 beliefs; spectral learning
05_approximate_information_states.md  AIS: value-loss bounds for learned
                                 memories; frame stacks and RNNs, priced
```
