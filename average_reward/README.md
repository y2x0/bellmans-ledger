# Average Reward

This family asks:

```text
Everything in this repository leans on gamma < 1 for contraction.
What is the theory when there is no discount?
```

The average-reward criterion `rho(pi) = lim (1/T) E[sum r]` is the honest
objective for continuing tasks (queueing, control loops, servers), and it
forces the machinery to be rebuilt: value functions split into **gain**
and **bias** (02), the discounted theory is recovered as a Laurent
expansion around `gamma = 1` (03), optimality stratifies into a hierarchy
with Blackwell optimality at the top (04), contraction survives only in
the **span seminorm** and only under mixing conditions (05), and learning
requires two timescales (06).

Prerequisites: `mdp_foundations/`; `stochastic_approximation/01` for 06.
Sources: Puterman ch. 8–10; Bertsekas DP-II ch. 4; Wan–Naik–Sutton 2021.

## Folder Map

```text
01_chain_structure_cesaro.md    P* = Cesàro limit exists; unichain vs
                                multichain; where gain is state-dependent
02_gain_bias_poisson.md         the Poisson equation; existence, uniqueness
                                up to constants; interpretation
03_laurent_series.md            v_gamma = rho/(1-gamma) + h + O(1-gamma);
                                the deviation matrix
04_blackwell_optimality.md      the optimality hierarchy; one policy optimal
                                for ALL gamma near 1
05_span_seminorm_rvi.md         span contraction under ergodicity;
                                relative VI; the periodicity failure
06_average_reward_learning.md   differential TD/Q; two-timescale gain
                                estimation; convergence
```
