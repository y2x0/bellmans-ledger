# MDP Foundations

This family asks:

```text
When the model is known exactly, why does a greedy one-step operator
solve an infinite-horizon problem?
```

The answer is a chain of five results, each proved in its own notebook:

```math
\text{Markov property}
\ \Rightarrow\
\text{stationary policies suffice}
\ \Rightarrow\
\mathcal{T}\text{ is a }\gamma\text{-contraction}
\ \Rightarrow\
v_* \text{ exists, unique}
\ \Rightarrow\
\text{greedy}(v_*)\text{ is optimal.}
```

Sources: Sutton & Barto ch. 3–4; Szepesvári Appendix A; Puterman ch. 5–7 for
the measure-theoretic completions; Bellman 1957 for the origin.

## Folder Map

```text
01_mdp_formalism_and_policies.md
    policy classes HR ⊃ MR ⊃ SR ⊃ SD, and why the smallest class loses nothing

02_return_criteria_and_value_functions.md
    discounted / total / average reward; what discounting buys mathematically

03_bellman_operators.md
    T^pi and T*: linearity, monotonicity, contraction — with proofs

04_banach_and_error_bounds.md
    the fixed point theorem, geometric convergence, stopping rules

05_policy_improvement_and_policy_iteration.md
    the improvement theorem, finite convergence, PI as Newton's method

06_linear_programming_and_occupancy_measures.md
    the primal LP over values, the dual over occupancy measures

07_bellman_1957.md
    the original paper: principle of optimality as a functional equation
```
