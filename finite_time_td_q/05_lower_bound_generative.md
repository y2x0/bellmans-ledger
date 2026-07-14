# The Generative-Model Lower Bound

## The Claim

```math
\text{Any algorithm that outputs }\hat Q\text{ with }
\Pr\big\{\|\hat Q-q^*\|_\infty\le\varepsilon\big\}\ge 3/4
\text{ on every MDP requires}
```

```math
N\ =\ \Omega\Big(\frac{|S||A|}{(1-\gamma)^3\,\varepsilon^2}\Big)
\ \text{total generative-model samples.}
```

Matching notebook 04's upper bound: the problem is closed. This file builds
the hard family and runs the reduction, using only
`concentration_toolkit/05`.

## The Hard Instance: One Cell

The atom is a single state whose value estimation requires many samples
**because the discount recycles small biases for a full horizon**. Fix
`gamma` and consider the two-state gadget, controlled by a bias parameter
`p`:

```text
state x, action a:  reward 1, then
                    with prob p      -> stay at x
                    with prob 1-p    -> to absorbing sink (value 0)
```

The value of `(x, a)`:

```math
q_p(x,a)=\frac{1}{1-\gamma p},
\qquad
\frac{\partial q_p}{\partial p}=\frac{\gamma}{(1-\gamma p)^2}.
```

Take the base `p_0 = gamma`-ish (so `1 - gamma p_0 ~ (1-gamma)`): then

```math
q_{p_0+\Delta}-q_{p_0}\ \approx\ \frac{\gamma\,\Delta}{(1-\gamma)^2}.
```

**The lever:** a perturbation of size `Delta` in the transition probability
moves the value by `Delta/(1-gamma)^2` — the resolvent's derivative is the
squared horizon. To create a value gap of `2 eps`, it suffices to perturb

```math
\Delta\ \asymp\ \varepsilon\,(1-\gamma)^2 .
```

## The Information Calculation

Distinguishing `Bernoulli(p_0)` from `Bernoulli(p_0 + Delta)` — which is all
the samples from this cell reveal — has per-sample divergence

```math
\mathrm{KL}\big(p_0\ \|\ p_0+\Delta\big)
\ \asymp\
\frac{\Delta^2}{p_0(1-p_0)}
\ \asymp\
\frac{\varepsilon^2(1-\gamma)^4}{1-\gamma}
\ =\
\varepsilon^2(1-\gamma)^3,
```

using `1 - p_0 ~ (1-gamma)` — **the variance of the coin is itself small**,
`p_0(1-p_0) ~ (1-gamma)`, which buys back one horizon factor and is exactly
the trace, on the lower-bound side, of the total-variance phenomenon of
notebook 04. (A construction with `p_0 = 1/2` would only prove
`(1-gamma)^{-2}`; placing the coin near its deterministic edge is the
essential trick.)

By the two-point method + Bretagnolle–Huber (`concentration_toolkit/05`),
deciding the cell's version reliably needs

```math
N_{\text{cell}}
\ \ge\
\frac{c}{\mathrm{KL}}
\ =\
\Omega\Big(\frac{1}{(1-\gamma)^3\,\varepsilon^2}\Big)
```

samples of that cell. And an `eps`-accurate `hat-q` **is** a reliable
decider: output "perturbed" iff `hat-q(x,a)` is closer to the perturbed
value — correctness of `hat-q` within `eps` implies correct decision, since
the versions differ by `2 eps`. ∎ (one cell)

## Scaling Out To |S||A|

Assemble `|S||A|/2` independent copies of the gadget (disjoint state pairs;
actions index parallel gadgets). Nature draws each cell's version
independently at random. Two routes to the conclusion, both standard:

```text
route 1 (per-cell to total, averaging): a uniformly eps-accurate answer
   decides ALL cells; by independence, the algorithm's total budget N
   splits across cells, and Fano / direct averaging over the ensemble
   forces N >= (#cells) * N_cell.

route 2 (Fano, concentration_toolkit/05): M = 2^{#cells} well-separated
   environments; log M = Omega(|S||A|); per-pair KL controlled as above.
```

Either way:

```math
N\ =\ \Omega\Big(\frac{|S||A|}{(1-\gamma)^3\,\varepsilon^2}\Big). \qquad\blacksquare
```

**Load-bearing audit.**

```text
generative model     -> the bound counts SAMPLES, not exploration; with a
                        single trajectory the same gadgets plus reachability
                        costs give the harder online bounds
                        (regret_and_exploration/04)
eps < ~1/sqrt(1-g)   -> for very coarse eps the calculation changes (the
                        coin can't be placed); this is the "burn-in" regime
                        where upper/lower still have gaps (notebook 04)
independence of      -> needed for the budget-splitting/Fano step; the
gadget cells            gadgets must not leak information about each other
```

## Why This File Matters Beyond Its Theorem

The construction is a *template*, reused with modifications everywhere a
lower bound is claimed in this repository:

```text
regret_and_exploration/04   the same coin, hidden behind a hard-to-reach
                            chain: reachability multiplies the price
linear_mdps/05              the same lever (resolvent derivative), but the
                            perturbation must live in a d-dimensional
                            feature-null direction — geometry replaces
                            counting
offline_rl_and_ope/01       the same coin, with the visitation fixed by the
                            behavior policy: coverage replaces exploration
```

One gadget, three fields.

## What Remains Open

Instance-optimal (as opposed to minimax) lower bounds — matching the
*variance profile* of a given MDP rather than the worst case — are
established for evaluation but only partially for control.
