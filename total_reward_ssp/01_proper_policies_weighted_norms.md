# Proper Policies And Weighted-Norm Contraction

## The SSP Setup

States `S`, plus an absorbing, zero-reward terminal state `T`; no
discount (`gamma = 1`); objective: expected total reward (conventionally
cost, minimized — we keep reward, maximized) until absorption. A
stationary policy `pi` is **proper** if

```math
\Pr_\pi\{\text{reach }T\}\ =\ 1\quad\text{from every state}
\qquad\Longleftrightarrow\qquad
\rho(Q_\pi)<1,
```

where `Q_pi` is `P_pi` restricted to non-terminal states (a
*substochastic* matrix: rows may sum to `< 1`, the deficit being the
per-step absorption probability). Properness = the substochastic
spectral radius drops below 1 = expected hitting times finite.

## The Problem And The Fix

`Q_pi` substochastic gives `rho(Q_pi) < 1` but **not**
`||Q_pi||_inf < 1`: rows with no immediate absorption have sup-norm row
sum exactly 1, so `T^pi v = r_pi + Q_pi v` is not a plain sup-norm
contraction (multi-step absorption is invisible to one-step row sums).
The fix is to measure states by *how long they take to terminate*:

**Theorem (weighted-norm contraction, Bertsekas–Tsitsiklis 1991).** If
all stationary policies are proper, there exist weights `w(s) >= 1` and
`beta < 1` such that every `T^pi`, and hence `T = max_pi T^pi`, is a
`beta`-contraction in

```math
\|v\|_{w}=\max_s\ \frac{|v(s)|}{w(s)} .
```

*Proof, with the explicit weight.* Let `w(s) = ` the maximal (over
policies) expected number of steps to termination from `s` — finite by
properness + finiteness (a max of finitely many finite hitting-time
vectors; itself the fixed point of a "longest expected route" equation).
By its one-step (Bellman-type) recursion,

```math
w(s)\ \ge\ 1+\big(Q_\pi w\big)(s)\qquad\forall\pi,s
\quad\Longrightarrow\quad
\big(Q_\pi w\big)(s)\ \le\ w(s)-1\ \le\ \Big(1-\tfrac{1}{w_{\max}}\Big)\,w(s),
```

using `w >= 1`. So `Q_pi` shrinks the weight vector by the uniform factor
`beta = 1 - 1/w_max`. Then for any `u, v`:

```math
\big|\mathcal{T}^\pi u-\mathcal{T}^\pi v\big|(s)
=\big|Q_\pi(u-v)\big|(s)
\le\|u-v\|_w\,\big(Q_\pi w\big)(s)
\le\beta\,\|u-v\|_w\,w(s),
```

i.e. `||T^pi u - T^pi v||_w <= beta ||u - v||_w`; the max-lemma
(`mdp_foundations/03`) lifts it to `T`. ∎

**Read the modulus:** `beta = 1 - 1/w_max` — contraction strength =
inverse of the worst expected episode length. The role of
`1/(1-gamma)` (discounted) and of the mixing time / diameter
(average reward, `average_reward/05–06`) is here played by **expected
time-to-termination**: three families, one slot, three occupants —

```text
discounted:      1/(1-gamma)     (chosen by the designer)
average reward:  mixing time     (property of the chain)
SSP/episodic:    E[episode len]  (property of chain AND termination)
```

## Everything Downstream, For Free

With the weighted contraction in hand, the entire `mdp_foundations/`
apparatus replays verbatim on `(B(S), ||.||_w)`:

```text
- unique fixed points v_pi, v*; VI converges geometrically at rate beta;
- a posteriori bounds with (1-beta)^{-1} = w_max replacing (1-gamma)^{-1};
- policy iteration converges finitely (improvement theorem: the monotone
  telescope never used the discount, only monotonicity + convergence of
  (T^pi)^k — both restored);
- Q-learning for SSP converges a.s. under properness + the Watkins
  conditions (the asynchronous SA proof of stochastic_approximation/05
  runs in the weighted norm).
```

This is the repository's cleanest illustration that **the discount was
never the point — a contraction was**, and any structural source of one
(discount, mixing, termination) buys the same theory.

## Load-Bearing Audit

```text
ALL policies proper     -> the uniform beta over pi; notebook 02 is about
                           relaxing exactly this (some policies improper)
finite S                -> finiteness/attainment of w; infinite-state SSP
                           needs w as an explicit Lyapunov function —
                           the analyst supplies what finiteness supplied
r bounded               -> fixed points finite in ||.||_w
```

## What Remains Open

Nothing classical here; the live edges are statistical — SSP regret
(online episodic RL with unknown, policy-dependent episode lengths:
minimax rates involve `w_max`-type constants and were settled only
recently) and SSP with function approximation, which inherits all of
`linear_mdps_and_completeness/`'s questions plus the weight estimation.
