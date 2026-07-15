# Options And Hierarchy

This family asks:

```text
What happens to the Bellman machinery when actions are temporally
extended — and what, mathematically, does hierarchy buy?
```

The options framework (Sutton–Precup–Singh 1999) is the rare hierarchy
proposal with a complete operator theory: options induce a semi-Markov
decision process whose Bellman equations, contraction, and learning
algorithms all go through (01, 02); the option-critic theorems
(Bacon–Harb–Precup 2017, `papers/option-critic-bacon-2016.pdf`) extend
the policy gradient theorem to learn the hierarchy itself (03); and the
honest accounting of what hierarchy buys — and how learned options
degenerate — closes it (04).

Prerequisites: `mdp_foundations/03–05`, `policy_gradient/01`,
`stochastic_approximation/05`.

## Folder Map

```text
01_smdp_and_options.md         options; the multi-time model; SMDP Bellman
                               equations contract
02_intra_option_learning.md    intra-option Bellman equations; off-policy
                               learning across options
03_option_critic_gradients.md  the intra-option policy gradient and
                               termination gradient theorems, derived
04_what_hierarchy_buys.md      exploration vs value; the collapse
                               degeneracies; the honest ledger
```
