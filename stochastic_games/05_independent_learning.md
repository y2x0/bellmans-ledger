# Independent Learners: The Dynamics Of Failure

## The Setup Everyone Actually Runs

Drop all game-awareness: each agent runs standard single-agent RL
(Q-learning, PPO) treating the other agents as part of the environment.
This is "independent learning" — the default in multi-agent practice —
and it violates the *first* assumption of this repository
(`00_problem_setup`): the other agents' adaptation makes each agent's
environment **non-Markov and non-stationary**. This file works out
exactly what the violation does in the smallest possible case.

## Two Q-Learners In Matching Pennies

Zero-sum 2x2, payoffs for player 1: `+1` if actions match, `-1` if not;
unique Nash: both uniform `(1/2, 1/2)`, value 0. Each player runs
softmax Q-learning on its two actions with temperature `tau` and
learning rate `alpha`:

```math
Q^i_{t+1}(a)=Q^i_t(a)+\alpha\,\mathbb{1}\{a_t^i=a\}\big(u^i_t-Q^i_t(a)\big),
\qquad
x^i_t=\mathrm{softmax}\big(Q^i_t/\tau\big).
```

## The ODE Limit Is The Replicator (Plus Mutation)

Taking `alpha -> 0` (the ODE method, `stochastic_approximation/01`) and
tracking the policies `x = x^1, y = x^2`, Boltzmann Q-learning's mean
dynamics are (Tuyls et al. 2003; Sato–Crutchfield):

```math
\dot x_a
=
\frac{x_a}{\tau}\Big[(Ay)_a-x^\top Ay\Big]
\ +\ x_a\Big[\mathcal{H}\text{-term: }-\log x_a+\textstyle\sum_b x_b\log x_b\Big],
```

— the **replicator equation** (grow actions whose payoff beats the
current average) plus an entropy-driven mutation term of strength set by
`tau`; symmetrically for `y` with `-A^T`.

**The zero-sum phase portrait, worked.** For `tau -> 0` (pure
replicator) in matching pennies, pass to
`u = log(x_1/x_2), v = log(y_1/y_2)`:

```math
\dot u=\frac{2}{\tau'}\,\tanh\text{-free form: }\ \dot u=c\,\bar y,\qquad
\dot v=-c\,\bar x
\qquad(\bar x=x_1-x_2,\ \bar y=y_1-y_2),
```

a **conservative rotation**: the quantity

```math
E(x,y)=\mathrm{KL}\big(x^*\|x\big)+\mathrm{KL}\big(y^*\|y\big)
\qquad(x^*=y^*=\text{uniform})
```

is a constant of motion — differentiate: `dE/dt = -(x* - x)^T A (y* -
y)`-type terms cancel by the zero-sum antisymmetry (`A = -B^T`). **The
dynamics orbit the equilibrium forever on level sets of `E`; they never
approach it.** With small `tau > 0` the mutation term adds weak inward
drift (orbits spiral slowly toward a smoothed equilibrium); with
discretization (finite `alpha`), the Euler step adds *outward* drift of
order `alpha` (discretizing a conservative rotation spirals out —
standard numerical-ODE fact); the observed behavior of real independent
learners is the competition of the two: quasi-cycles, amplitude set by
`alpha/tau`. ∎

This is notebook 02's cycling, now with the full dynamical-systems
diagnosis: **zero-sum couplings make learning dynamics rotational, not
gradient-like** — the Jacobian at equilibrium has imaginary eigenvalues;
there is no potential to descend (contrast: potential games, where
independent learning IS gradient flow on the potential and converges).

## The Same Disease In Deep RL

```text
- independent PPO in competitive environments: the rotational component
  manifests as strategy chasing (agent A counters B, B counters the
  counter, ...) — the policy-space orbit of the phase portrait above;
- GAN training is the same mathematics (zero-sum, simultaneous
  gradients): its notorious cycling/mode-chasing and the fixes —
  optimism/extragradient (approximating the implicit midpoint method:
  kills the discretization's outward drift), timescale separation
  (two-timescale again), averaging (notebook 03's lesson: iterates
  cycle, averages mean something) — map one-to-one onto multi-agent RL
  stabilizers;
- the replay buffer makes it WORSE here: off-policy data from an
  opponent that no longer exists poisons the Q-target with a stale game
  (nonstationarity + bootstrapping — the deadly triad,
  stochastic_approximation/06, with time-variation as the norm
  mismatch).
```

## The Honest Design Rules Extracted

```text
1. know your game class: potential-like (cooperative, shared reward) ->
   independent learning is principled (gradient flow); zero-sum-like ->
   it is a rotation: stabilize (optimism, averaging, pools) or
   restructure (03, 04);
2. report AVERAGES or exploitability, never last-iterate reward curves
   (which oscillate meaninglessly);
3. slow the opponent (pools, snapshots, leagues) — every working
   large-scale system rediscovers the two-timescale fix.
```

## What Remains Open

Last-iterate convergence for realistic (deep, stochastic, discretized)
independent learners in zero-sum settings — optimistic-gradient theory
covers the convex-concave core, not the neural composition; general-sum
dynamics (heteroclinic cycles, chaos — documented even in 2x2x2 games)
have description but no control theory; and there is no accepted
multi-agent analogue of the single-agent regret framework
(`regret_and_exploration/`) that is simultaneously tractable and
meaningful for `n > 2` adaptive agents.
