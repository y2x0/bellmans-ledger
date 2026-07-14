# The Failure Of Greedy Dynamics In Games

## The Question

In MDPs, "act greedily against your current estimate, re-evaluate,
repeat" converges (policy iteration, `mdp_foundations/05`). The
game-theoretic analogue — each player best-responds to the other's
current strategy — is **fictitious play** (best-respond to the
opponent's *empirical average*) or naive alternating best response. Do
they converge to equilibrium? Generically: no.

## Shapley's Cycle (1964)

The counterexample is a single-state game — no dynamics needed — a
3x3 bimatrix (here the zero-sum-like variant; Shapley's original is
general-sum):

```math
A=\begin{pmatrix}0&1&2\\2&0&1\\1&2&0\end{pmatrix}
\qquad\text{(row player's payoffs; a biased rock-paper-scissors)}
```

Fictitious play from, say, `(row 1, col 2)`: each player best-responds
to the opponent's running empirical mixture. Tracking the play: the
best-response targets rotate `1 -> 2 -> 3 -> 1 ...` for both players,
and the **time spent in each phase grows geometrically** (each new
"conviction" must outweigh the accumulated history before the argmax
flips). The empirical mixtures do not converge to the equilibrium
(uniform 1/3 each in the RPS-symmetric case); they orbit it on a cycle
whose period expands forever. ∎ (Shapley's paper computes the phase
lengths explicitly; the geometric growth ratio is checkable by hand
from the payoff gaps.)

**What exactly failed.** In the one-player world, the improvement
theorem rests on the *environment holding still* while the policy
improves against it. In a game, the "environment" (the opponent) is
re-optimizing against you; improvement against a moving target is not
improvement. Formally: best response is a discontinuous map
(argmax — the same discontinuity that broke distributional control,
`distributional_rl/03`), and the composed two-player best-response map
has no monotone potential in general. (In zero-sum games fictitious play
*does* converge in empirical averages — Robinson 1951 — but slowly, and
Shapley's example shows general-sum kills even that.)

## Naive Policy Iteration Fails In Stochastic Games

The MDP intuition "PI is Newton, fast and safe"
(`mdp_foundations/05`) also breaks. For zero-sum stochastic games,
the natural PI — evaluate `v^{(pi, mu)}`, then update BOTH players'
strategies greedily at every state — can **cycle**: van der Wal's
counterexample exhibits a two-state game where simultaneous greedy
updates oscillate between two strategy pairs forever, because a
state-local improvement for player 1 changes continuation values in a
way that reverses player 2's local comparison, and vice versa —
the two players' improvement telescopes point through *each other's*
value functions and cannot both be monotone.

The repairs, each instructive:

```text
Hoffman–Karp (1966):    asymmetric PI — fix player 2's strategy, solve
                        player 1's ENTIRE MDP best response exactly
                        (an inner mdp_foundations/05 run), then update
                        player 2 once; repeat. Monotone in player 2's
                        guarantee: converges. Cost: an MDP solve per
                        outer step.
Shapley iteration:      abandon PI for VI on the Shapley operator (01):
                        the contraction never needed improvement
                        arguments. Slower per-iteration progress,
                        unconditional convergence.
regularized dynamics:   smooth the best response (softmax at tau —
                        regularized_mdps_and_duality/01): continuous
                        response maps admit fixed-point/variational-
                        inequality analyses and damped convergence in
                        classes of games — the game-side payoff of the
                        entire regularization family.
```

## The General Lesson, Stated As An Audit

```text
mdp_foundations/05's improvement theorem used, silently:
  (i)  a FIXED evaluation operator T^pi while improving  -> false in games
  (ii) monotonicity in ONE value function                -> two coupled
                                                            value functions,
                                                            opposite signs
self-play systems inherit the risk whenever both sides adapt
concurrently at comparable timescales — the two-timescale principle
(stochastic_approximation/01) reappears as the DESIGN FIX: freeze one
side (opponent pools, past-self snapshots — search_and_planning/03's
AlphaGo opponent sampling; league training), which is Hoffman–Karp
asymmetry implemented statistically. Notebook 04 develops exactly this.
```

## What Remains Open

Characterizing which game classes admit *symmetric* convergent greedy
dynamics (zero-sum: no in general; potential games: yes — the potential
is the missing monotone quantity); and last-iterate (not just average)
convergence for smoothed dynamics, which is an active optimization-
theory frontier (extragradient/optimistic methods) feeding directly
into notebook 05's failures.
