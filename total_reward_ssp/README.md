# Total Reward And Stochastic Shortest Paths

This family asks:

```text
Episodic tasks run at gamma = 1. Contraction came from the discount.
What exactly holds it up now?
```

The answer: **properness** — the guarantee of eventual termination —
manufactures a contraction in a *weighted* sup-norm (01); when improper
policies exist the theory survives with surgical care (02); when even
properness fails, monotonicity alone carries a weaker theory with real
pathologies (03); and the episodic RL that practice runs sits on this
family's fine print (04).

Every Atari result, every episodic benchmark in this repository is
implicitly a stochastic shortest path (SSP) problem; this family is the
license it never showed.

Prerequisites: `mdp_foundations/03–05`; `average_reward/` for contrast.
Sources: Bertsekas–Tsitsiklis 1991; Bertsekas DP-II ch. 3; Blackwell/
Strauch for positive/negative DP.

## Folder Map

```text
01_proper_policies_weighted_norms.md   properness => weighted-sup contraction
02_improper_policies.md                mixed case: uniqueness with care;
                                       a counterexample without it
03_positive_negative_dp.md             no properness: monotone theory and
                                       its genuine failures
04_episodic_rl_as_ssp.md               timeouts, truncation, and gamma<1 on
                                       episodic tasks — the fine print
```
