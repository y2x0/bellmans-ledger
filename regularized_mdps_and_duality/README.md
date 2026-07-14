# Regularized MDPs And Convex Duality

This family asks:

```text
What exactly happens to the Bellman machinery when an entropy or KL term
is added to the objective — and why does every modern method add one?
```

The answer is a single convex-analytic object viewed from six angles: the
**Legendre–Fenchel transform of the negative entropy is log-sum-exp**, and
therefore

```text
01  the regularized greedy step has a closed form (softmax) and the
    regularized Bellman operator (logsumexp) still contracts
02  the same objective drops out of a variational-inference derivation —
    RL as posterior inference (control as inference)
03  soft policy iteration converges, and SAC is its deep instance
04  the general theory: any convex regularizer, one contraction proof,
    one error-propagation theorem (Geist–Scherrer–Pietquin)
05  KL-regularized policy iteration IS mirror descent; TRPO/PPO are its
    inexact steps — regret bounds without contraction
06  dualizing the occupancy LP (mdp_foundations/06) with a regularizer
    yields the DICE estimators — closing the loop with offline_rl/03
```

Prerequisites: `mdp_foundations/03–06`, `policy_gradient/03–04`;
`offline_rl_and_ope/03` for 06's payoff. This family is also the
mathematical basis of `rlhf_mathematics/` (the KL-regularized closed form
is 01's computation over trajectories).

## Folder Map

```text
01_soft_bellman_operator.md      Legendre–Fenchel; softmax policy; contraction;
                                 bias bound tau log|A|/(1-gamma)
02_control_as_inference.md       the ELBO derivation; where naive inference
                                 goes risk-seeking
03_soft_policy_iteration_sac.md  soft improvement theorem; SAC; temperature
                                 as dual ascent
04_geist_scherrer_pietquin.md    Omega-regularized operators in general
05_mirror_descent_view.md        three-point lemma; TRPO/PPO as inexact MD
06_fenchel_rockafellar_dice.md   the regularized LP dual; DICE derived
```
