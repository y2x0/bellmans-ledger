# The Mathematics Of RLHF

This family asks:

```text
When a language model is tuned from preference data, what is the exact
optimization problem, what are its closed-form solutions, and what do
those closed forms say about reward hacking?
```

`policy_gradient/06` noted PPO "escaped the field" into LLM alignment.
This family is that escape made rigorous. Its spine is one computation —
the KL-regularized reward maximization has the closed-form solution
`pi* ∝ pi_ref exp(r/beta)` (02, a corollary of
`regularized_mdps_and_duality/01`'s transform) — from which everything
else follows: DPO is its inversion (03), best-of-n is its cheap competitor
(04), overoptimization is its noise analysis (05), and the token-vs-
sequence question is its MDP-theoretic fine print (06).

Prerequisites: `regularized_mdps_and_duality/01–02, 05`;
`policy_gradient/02, 05–06`; `value_based_deep_rl/03` (max-bias) for 05.

## Folder Map

```text
01_bradley_terry_identifiability.md  preference MLE; reward identified only
                                     up to per-prompt shift; what preferences
                                     cannot reveal
02_kl_regularized_optimum.md         pi* ∝ pi_ref e^{r/beta}, value =
                                     beta log E e^{r/beta}; Donsker–Varadhan
03_dpo_derivation.md                 invert 02, partition cancels in pairs;
                                     the implicit reward; gradient analysis
04_best_of_n.md                      KL(BoN||ref) <= log n - (n-1)/n; near-
                                     optimality on the reward-KL frontier
05_overoptimization_goodhart.md      proxy-vs-gold scaling; max-bias = Goodhart;
                                     KL budgets
06_token_mdp_or_bandit.md            deterministic transitions + terminal
                                     reward; GAE(1) collapse; GRPO as
                                     leave-one-out baseline
07_rlvr_and_r1.md                    verifiable rewards delete the Goodhart
                                     noise term; R1-Zero/GRPO dynamics;
                                     the pipeline as this family's taxonomy
```
