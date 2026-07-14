# Policy Gradient Methods

This family asks:

```text
If the policy is parameterized directly, what is the exact gradient of the
return, what may replace its integrand, and how large a step is safe?
```

The three questions map onto the notebooks: the policy gradient theorem
answers "what is the gradient"; baselines/GAE answer "what may replace the
integrand" (anything whose conditional expectation preserves it); the
performance difference lemma and trust regions answer "how large a step" —
and PPO is the engineering fixed point of all three.

Primary source: Schulman et al., "Proximal Policy Optimization Algorithms,"
arXiv:1707.06347 (`papers/ppo-schulman-2017.pdf`), read with Kakade–Langford
2002 and Schulman et al. 2015 (TRPO) as its mathematical spine.

## The Coordinates

```math
\text{objective:}\quad
J(\theta)=\mathbb{E}_{s_0\sim\mu}\big[v_{\pi_\theta}(s_0)\big],
\qquad
\text{ascent on }\theta\text{ instead of fixed-point iteration on }v .
```

No Bellman operator is iterated on the policy side; contraction arguments are
replaced by *monotonic improvement* arguments. The critic side still carries
the whole `stochastic_approximation/` inheritance.

## Folder Map

```text
01_policy_gradient_theorem.md      the theorem, two proofs, log-derivative trick
02_baselines_and_variance.md       control variates; optimal baseline; why v_pi
03_performance_difference_lemma.md the exact identity behind all improvement bounds
04_trust_regions_trpo.md           the TRPO lower bound, proof structure, natural gradient
05_gae.md                          generalized advantage estimation, derived
06_ppo_clipped_surrogate.md        the clipped objective, its analysis, the algorithm
07_failure_modes.md                collapse, ratio pathologies, KL drift
```
