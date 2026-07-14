# Optimism Through Models: R-MAX And UCRL2

## The Online Problem

No generative model: samples arrive only along the trajectory the agent
drives (`regret_and_exploration/`'s setting, now attacked via models).
The two classical algorithms are the two ways to be optimistic *about a
model*.

## R-MAX: Optimism By Fiat (Brafman–Tennenholtz 2002)

```text
maintain counts; call (s,a) KNOWN when N(s,a) >= m;
build the model:  known cells -> empirical estimates;
                  unknown cells -> a FICTIONAL transition to a paradise
                  state absorbing at reward Rmax;
plan exactly in this model; act; repeat on new knowledge.
```

**Theorem (PAC shape).** With `m = tilde-O(S/( (1-gamma)^4 eps^2 ))`-ish
per cell (Bernstein versions improve the exponent), R-MAX is
`eps`-optimal on all but `tilde-O(S A m / ((1-gamma) eps))`-type steps,
with high probability — polynomial in everything ("PAC-MDP").

*Proof: the explore-or-exploit dichotomy, which is the whole idea.* Fix
an episode-ish window. Either

```text
(a) the agent's trajectory stays in KNOWN cells w.h.p.: there the
    fictional model agrees with the empirical one, which (simulation
    lemma, 01 + per-cell concentration at level m) is eps-accurate —
    the plan is near-optimal in the TRUE MDP too: EXPLOITING; or
(b) the trajectory reaches an UNKNOWN cell with non-negligible
    probability: paradise-reward optimism made unknown cells
    irresistible, so the plan actively drives at them — each such
    window scores a new sample toward some cell's m-threshold:
    EXPLORING, with progress.
```

Case (b) can occur at most `S A m` times (pigeonhole over
known-threshold crossings); every other window is case (a). ∎
The dichotomy is `regret_and_exploration/01`'s optimism lemma in
combinatorial clothing — with the width function degenerated to the
binary known/unknown and the "bonus" to `Rmax` itself. Crude, hence the
loose polynomial; but the proof pattern (either your model is right
where you go, or you learn something bounded-many times) is the
skeleton every model-based online proof reuses.

## UCRL2: Optimism By Confidence Polytope (Jaksch–Ortner–Auer 2010)

Average-reward setting (`average_reward/`), diameter `D`. Maintain
per-cell confidence sets from `concentration_toolkit/`:

```math
\mathcal{M}_t
=
\Big\{\tilde P:\ \big\|\tilde P(\cdot\mid s,a)-\hat P(\cdot\mid s,a)\big\|_1\le\sqrt{\tfrac{14\,S\log(\cdot)}{N(s,a)}}\ \ \forall s,a\Big\}
```

(plus reward intervals), and plan **jointly over policy and model**:

```math
(\pi_k,\tilde M_k)=\arg\max_{\pi,\ \tilde M\in\mathcal{M}_{t_k}}\ \rho^{\pi}_{\tilde M}
```

— optimistic value iteration where each backup also picks the best
transition vector in the L1-ball (a linear program per state with a
closed-form greedy solution: put as much mass as allowed on the
highest-value successor). Episodes are doubled when any count doubles
(the lazy-update schedule that keeps planning cost logarithmic).

**Theorem.** `Reg(T) = tilde-O( D S sqrt(A T) )`, with
`Omega(sqrt(D S A T))` the matching-in-`T` lower bound (the usual
gadget family, `concentration_toolkit/05`, with travel-time `D` pricing
each probe — `average_reward/06` previewed this).

*Proof skeleton against the standard machine:* optimism (the true model
is in `M_t` w.h.p., so the optimistic gain upper-bounds `rho*`);
per-step regret telescopes into confidence widths along the trajectory
(the `regret_and_exploration/01` telescope, with the span of the bias
`h` — bounded by `D Rmax`, `average_reward/05` — where the episodic
proof had `H`); pigeonhole the widths (`sqrt(S/N)` summed =
`S sqrt(A T)`). Every constant lands where the average-reward family
said it would: `D` in the role of the horizon, span in the role of
`Vmax`. ∎

## The Comparison Table

```text
                R-MAX                     UCRL2
optimism        binary (paradise)         continuous (polytope)
guarantee       PAC (few bad steps)       regret (cumulative)
criterion       discounted/episodic       average reward
bonus analogue  Rmax at unknowns          L1-width * span(h)
looseness       very (thresholds)         S-factor from the L1 ball
                                          (Bernstein per-successor
                                          versions shave it)
```

Both predate and prefigure `regret_and_exploration/02–03`: the
model-free UCBVI line is what remains when the confidence polytope is
collapsed onto value backups directly — cheaper, and (with Bernstein)
tighter in `S`. The models' residual advantages online: reusability
(02's point 1) and the ability to plan for *information* (the
optimistic model chooses where its own uncertainty is, which
value-bonus methods only mimic).

## What Remains Open

Closing the `D`-vs-`span(h*)` gap (`average_reward/06`); optimism over
*structured* model classes (parametric dynamics, factored MDPs — the
polytope becomes an ellipsoid/graph object and the planning-over-models
step becomes the bottleneck); and computationally efficient optimistic
planning for anything beyond tabular — the same efficiency wall as
`linear_mdps_and_completeness/04`.
