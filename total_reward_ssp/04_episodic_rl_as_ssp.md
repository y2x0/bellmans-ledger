# Episodic RL As SSP: The Fine Print Practice Signs

## The Three Termination Types, Formalized

Real episodic environments end in three distinct ways that the SSP
formalism distinguishes and practice frequently conflates:

```text
1. TRUE termination:   an absorbing terminal state of the MDP
                       (death, goal). Part of the transition kernel.
2. TIMEOUT truncation: an exogenous step-limit cuts the episode.
                       NOT part of the MDP — the state did not become
                       terminal; the experimenter looked away.
3. RESET:              the experimenter teleports to an initial state
                       (often bundled with 1 or 2).
```

## The Truncation Bug, With Its Exact Bias

The Q-learning/DQN target treats a flagged "done" as true termination:

```math
y=\begin{cases}
r & \text{done}\\
r+\gamma\max_{a'}Q(s',a') & \text{else}
\end{cases}
```

If "done" includes timeouts, the target at a timeout step deletes the
continuation value of a **non-terminal** state: the learned fixed point
solves a *different* MDP — one where a lottery of sudden death (with the
timeout's hazard) is appended to the dynamics. The bias at a state
`h` steps before a length-`H` timeout is `~ gamma^{H-h} E[V(s_H)]`-scale;
for time-homogeneous policies it deflates values everywhere the horizon
binds, and (worse) *state-dependently* — states typically visited late
in episodes are punished hardest, tilting the greedy policy.

**The fix costs one flag** (bootstrap-on-truncation):

```math
y=r+\gamma\max_{a'}Q(s',a')
\qquad\text{at TIMEOUT steps (bootstrap as if continuing)},
```

reserving the no-bootstrap target for true terminals. Equivalently: the
timeout is part of the *data collection*, not the *objective*; SSP
semantics demand the target represent the objective's kernel. (The
gymnasium `terminated` / `truncated` split is this file as an API.)

**When the timeout SHOULD be modeled:** if the deployed task genuinely
ends at `H` (a game clock), then time-to-go is part of the true state;
the honest formulation appends `t` (finite-horizon MDP — values
`V_h(s)`, one function per stage, exactly as in
`regret_and_exploration/02`) rather than pretending stationarity.
Symptom of getting *this* wrong: an agent that plays a clocked task
"timelessly," hoarding value it will never realize.

## Discounting Episodic Tasks: The Deliberate Distortion

Practice runs `gamma = 0.99` on tasks whose stated objective is
undiscounted episodic return. The SSP lens prices the substitution.
With absorption after (random) time `tau`:

```math
\mathbb{E}\Big[\sum_{t<\tau}\gamma^{t}r_t\Big]
=
\mathbb{E}\Big[\sum_{t<\tau}r_t\Big]
-\ (1-\gamma)\,\mathbb{E}\Big[\sum_{t<\tau}t\,r_t\Big]+O\big((1-\gamma)^2\big)
```

(expand `gamma^t = 1 - (1-gamma)t + ...`): the discounted criterion
equals the true one minus `(1-gamma)` times a **time-weighted reward
moment** — late rewards are shaved. This is the episodic sibling of the
Laurent expansion (`average_reward/03`), and the same verdict: harmless
iff `1/(1-gamma) >> ` episode length; a policy-tilting distortion
otherwise (it manufactures impatience the objective never asked for —
sometimes exploited *deliberately* as shaping toward faster completion:
(A2)-manufacturing per notebook 02, bought with bias).

The variance side of the same trade (`policy_gradient/05`'s dial, GAE's
`gamma` as variance control): `gamma < 1` shrinks the effective horizon
of every return estimate, cutting gradient/target variance by the same
mechanism that biases the objective. Both effects are `O(1-gamma)`; the
practitioner's `gamma` is the knob on this single trade, and the SSP
formalism is what says there is *no* setting that is simply "correct" —
only priced.

## The Assumptions Audit For A Standard Benchmark Run

The theory this family requires vs what a typical DQN/PPO Atari run
supplies:

```text
requirement                       supplied by
(A1) a proper policy exists       game over is reachable; timeouts
                                  backstop it (5-min cap)
(A2) improper policies -inf       NOT supplied by the raw game (idling
                                  is reward-0, finite) — supplied
                                  accidentally by timeouts-as-terminals
                                  (the bug!) or deliberately by per-step
                                  penalties; runs with neither can and
                                  do learn to stall
uniform properness (nb 01)        false; mixed case (nb 02) is the
                                  honest regime
episode length as the modulus     visible empirically: harder games =
                                  longer w_max = slower propagation
                                  (one more reading of why bootstrap
                                  depth/n-step returns help)
```

## What Remains Open

A clean theory of *learning with truncation-aware objectives* (current
fixes make targets consistent; finite-sample effects of truncation on
exploration and on replay composition are unquantified); and
principled per-task `gamma` selection from measurable episode statistics
— the trade above is priced asymptotically, not operationally.
