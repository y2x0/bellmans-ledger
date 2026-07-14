# Chain Structure And The Cesàro Limit

## The Object That Replaces The Resolvent

Discounted theory ran on `(I - gamma P)^{-1}`. Average-reward theory runs
on the **Cesàro limit** of the transition powers:

```math
P^{*}
=
\lim_{T\to\infty}\ \frac{1}{T}\sum_{t=0}^{T-1}P^{t} .
```

**Theorem (existence, finite chains).** `P*` exists for every finite
stochastic matrix, and satisfies

```math
P P^*=P^*P=P^*P^*=P^* .
```

*Proof.* Decompose the state space into recurrent classes
`R_1..R_m` and transient states (finite-chain structure theory). On an
aperiodic recurrent class, `P^t -> 1 mu_j^T` (its stationary distribution)
by Perron–Frobenius, and ordinary limits are Cesàro limits. On a periodic
class with period `d`, `P^t` cycles among `d` limit matrices; the Cesàro
average of a convergent cycle converges to the cycle mean — this is
exactly why the *Cesàro* limit is the right notion: **periodicity kills
`lim P^t` but not `lim (1/T) sum P^t`.** For transient states, `P^t`'s
mass drains onto the recurrent classes geometrically, contributing
absorption probabilities. Assembling: `P*(s, .)` = (probability of
absorption into class `j` from `s`) × (stationary distribution of class
`j`). The algebraic identities follow from `P* = lim (1/T) sum P^t` and
dominated passing of `P` through the limit. ∎

`P*` is the "infinite-horizon averaging operator": row `s` is the
long-run empirical state distribution started from `s`.

## The Chain Taxonomy

For a *policy* `pi` in an MDP, classify the induced chain `P_pi`; for the
*MDP*, classify over all policies:

```text
ergodic:     one recurrent class, aperiodic, no transient states —
             P* = 1 mu^T (rank one), everything is cleanest
unichain:    one recurrent class + possibly transient states —
             P* still rank one: the GAIN is a scalar
multichain:  >= 2 recurrent classes under some policy —
             P* has rank = #classes; gain is STATE-DEPENDENT
weakly       every state reachable from every state under SOME policy
communicating: (the MDP-level condition regret theory uses)
```

## The Multichain Example (gain is a vector)

Three states, actions only in state 1:

```text
state 1: action L -> state 2;  action R -> state 3     (reward 0 either way)
state 2: absorbing, reward 1 per step
state 3: absorbing, reward 0 per step
```

Under any policy, states 2 and 3 are separate recurrent classes:

```math
\rho(2)=1,\qquad \rho(3)=0,\qquad \rho(1)=\begin{cases}1&\text{policy plays }L\\ 0&\text{plays }R\end{cases}
```

— the optimal *gain* is the vector `(1, 1, 0)`, not a scalar, and
"maximize average reward" requires optimizing where you get absorbed
before optimizing within the class. Multichain optimality equations are
correspondingly a *pair* of nested equations (gain optimality, then bias
optimality within gain-optimal actions — notebook 04's hierarchy in
embryo). Most RL literature assumes unichain/communicating precisely to
collapse `rho` to a scalar; this example is what that assumption buys.

## The Averaging Identities Downstream Files Consume

```math
\rho_\pi=P_\pi^{*}\,r_\pi
\qquad\text{(gain = long-run average of rewards along the chain)},
```

and the two projections that organize all of average-reward linear
algebra: `P*` (onto the "steady-state" eigenspace of eigenvalue 1) and
`I - P*` (onto the "transient" complement). Every vector splits as

```math
v=\underbrace{P^{*}v}_{\text{steady part}}+\underbrace{(I-P^{*})v}_{\text{transient part}},
```

and the next two notebooks are exactly: the Poisson equation solves for
the transient part (02); the Laurent series is this splitting applied to
the discounted resolvent (03).

## What Remains Open

Nothing here (classical); the live issues start when structure must be
*learned* — identifying unichain-ness or mixing rates from data is itself
statistically hard, and average-reward RL theory (06 and the regret
literature) pays for it in mixing-time/diameter constants.
