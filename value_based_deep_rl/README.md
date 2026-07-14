# Value-Based Deep RL

This family asks:

```text
What happens to the Bellman optimality backup when the function class is a
deep network and the update distribution is a replay buffer?
```

Primary source: Mnih et al., "Human-level control through deep reinforcement
learning," Nature 518:529–533, 2015 (`papers/dqn-mnih-2015.pdf`). Read
against the theory of `stochastic_approximation/04–06`: DQN operates with
all three legs of the deadly triad and substitutes engineering
(replay, target freezing, clipping) for the missing theorems.

## The Coordinates

```math
\mathcal{T}=\text{optimality backup on } q,
\qquad
d=\text{none certified (sup-norm lost to FA, } L^2(\mu)\text{ lost to off-policy)},
\qquad
\mathcal{F}=\text{convnet } Q(s,\cdot\,;\theta).
```

## Folder Map

```text
01_dqn_loss_and_algorithm.md
    the loss, the algorithm, exactly which instability each piece answers

02_replay_and_target_networks.md
    replay as distribution smoothing; target nets as fixed-point splitting

03_overestimation_bias_double_q.md
    E[max] >= max E: quantifying the bias, Double Q-learning's fix

04_failure_modes.md
    what still breaks: triad exposure, aliasing, reward clipping distortions
```
