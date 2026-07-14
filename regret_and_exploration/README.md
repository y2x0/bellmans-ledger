# Regret And Exploration

This family asks:

```text
What is the provable cost of not knowing the MDP while acting in it —
and which exploration principles pay only that cost?
```

Setting: episodic MDPs, horizon `H`, `K` episodes, `T = KH` steps. The
performance measure is **regret**:

```math
\mathrm{Reg}(K)=\sum_{k=1}^{K}\Big(V_1^*(s_1^k)-V_1^{\pi_k}(s_1^k)\Big).
```

The family's arc: one principle (optimism) yields the generic upper-bound
skeleton (01); instantiating it with Hoeffding bonuses gives a complete,
end-to-end regret proof (02); Bernstein bonuses reach the minimax rate (03);
matching lower bounds — including the exponential separation from
epsilon-greedy — close the tabular chapter (04); Bayesian and
information-theoretic alternatives to optimism (05, 06); and what little
survives function approximation (07).

Prerequisites: `concentration_toolkit/` everywhere;
`search_and_planning/01` (UCB1) is the `H = 1` special case of everything
here.

## Folder Map

```text
01_optimism_principle.md            valid optimism => regret <= sum of widths
02_ucbvi_hoeffding.md               the canonical full proof: O(H^2 sqrt(SAT))
03_bernstein_minimax.md             total variance => O(sqrt(H^3 SAT)) ~ minimax
04_lower_bounds.md                  Omega(sqrt(H^3 SAT)); eps-greedy needs Omega(A^H)
05_thompson_psrl.md                 posterior sampling; optimism in expectation
06_information_directed.md          the information ratio; where optimism is beaten
07_exploration_with_fa.md           eluder dimension; the honest deep-RL gap
```
