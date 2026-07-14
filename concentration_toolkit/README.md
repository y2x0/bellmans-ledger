# Concentration Toolkit

This family asks:

```text
Which five probabilistic instruments carry every finite-time result in
reinforcement learning?
```

Nothing in this family mentions an MDP. It exists because the families that
do (`finite_time_td_q/`, `regret_and_exploration/`,
`linear_mdps_and_completeness/`, `offline_rl_and_ope/`) each reduce, at their
technical core, to one of five moves:

```text
1. bound a sum of independent bounded/small-variance terms      (01)
2. bound a sum whose terms are only conditionally centered      (02)
3. bound a regression estimate built from ADAPTIVELY chosen     (03)
   covariates
4. make a bound hold for every function in a class at once      (04)
5. prove no algorithm can do better                             (05)
```

The recurring axis through 01–03 is **range vs variance**: Hoeffding-type
bounds pay the worst-case range, Bernstein-type bounds pay the actual
variance plus a lower-order range term. Nearly every "improved rate" in the
RL literature — `(1-gamma)^{-4} -> (1-gamma)^{-3}`, `H^2 sqrt(SAT) ->
sqrt(H^3 SAT)` — is a Hoeffding-to-Bernstein swap executed inside a value
recursion, plus the law of total variance. Meeting that swap cleanly here,
without MDP clutter, is the point of the family.

## Folder Map

```text
01_chernoff_hoeffding_bernstein.md   the Chernoff method; Hoeffding's lemma
                                     proved from convexity; Bernstein via the
                                     Bennett mgf bound
02_martingale_concentration.md       Azuma–Hoeffding; Freedman; why RL data
                                     is a filtration, not a sample
03_self_normalized_bounds.md         method of mixtures; confidence
                                     ellipsoids for ridge regression under
                                     adaptive design
04_uniform_convergence_covering.md   epsilon-nets, covering numbers, uniform
                                     bounds over function classes
05_lower_bound_technique.md          Le Cam two-point method;
                                     Bretagnolle–Huber; the divergence
                                     decomposition for adaptive experiments
```

Sources: Boucheron–Lugosi–Massart, *Concentration Inequalities*;
Lattimore–Szepesvári, *Bandit Algorithms* ch. 5, 20; Wainwright,
*High-Dimensional Statistics* ch. 2; Abbasi-Yadkori–Pál–Szepesvári 2011.
