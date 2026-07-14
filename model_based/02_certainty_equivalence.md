# Certainty Equivalence Is Minimax-Optimal

## The Simplest Imaginable Algorithm

Generative model, `N` samples per `(s,a)` (`finite_time_td_q/`'s
setting). Build the empirical MDP and solve it exactly:

```math
\hat P(s'\mid s,a)=\frac{\#\{s'\}}{N},
\qquad
\hat\pi^*=\text{exact optimal policy of }(\hat P,r)
\quad(\text{mdp\_foundations, any solver}).
```

No bonuses, no pessimism, no algorithmic ideas — "plug in and plan."
The theorem this file proves in outline: **this achieves the minimax
sample complexity**

```math
N_{\text{total}}
=
\tilde O\Big(\frac{|S||A|}{(1-\gamma)^3\,\varepsilon^2}\Big),
```

matching `finite_time_td_q/05`'s lower bound — the model-based side of
that closed chapter, and a genuinely surprising result: the naive
algorithm is optimal; only the naive *analysis* was loose.

## Why The Naive Analysis Loses A Horizon Factor

Chain the simulation lemma (01, sup-form) with per-cell TV
concentration: `TV(P, hat-P) ~ sqrt(S/N)` (an L1/Weissman bound,
`concentration_toolkit/01+04`) gives

```math
\varepsilon\ \sim\ \frac{1}{(1-\gamma)^2}\sqrt{\frac{S}{N}}
\quad\Longrightarrow\quad
N\sim\frac{S}{(1-\gamma)^4\varepsilon^2}
```

— an extra `S` **and** an extra `1/(1-gamma)`. Both are analysis
artifacts:

## The Two Repairs (Azar et al. 2013; Agarwal–Kakade–Yang 2020)

**Repair 1: pair with the value, Bernstein, total variance.** Don't
bound `TV`; bound the scalar `(P - hat-P)(.|s,a)^T v*` directly —
Bernstein (`concentration_toolkit/01`) gives
`~ sqrt(Var_P(v*)/N)` per cell (no `S`: it is one scalar per cell, not
an `S`-vector), and propagating through the resolvent with
Cauchy–Schwarz + the total-variance lemma (`finite_time_td_q/04`,
verbatim) spends only `(1-gamma)^{-3/2}` where the naive chain spent
`(1-gamma)^{-2}`:

```math
\varepsilon\ \sim\ \frac{1}{(1-\gamma)^{3/2}}\ \sqrt{\frac{1}{N}}
\quad\Longrightarrow\quad
N\ \sim\ \frac{1}{(1-\gamma)^{3}\,\varepsilon^{2}}\ \text{per cell.}
```

**Repair 2: the dependence problem, and leave-one-out.** The pairing
above needs concentration of `(hat-P - P)^T v` for `v = v*` (fixed —
fine) but the control analysis also needs it for
`v = hat-v*` — **built from the same samples**: exactly the
data-dependence that `concentration_toolkit/04` says costs a covering
number (and coverings here would re-import the lost factors). The AKY
device: for each state `s`, construct the auxiliary family of MDPs in
which `s` is absorbing with a grid of fixed values `u`; the optimal
values of these MDPs are independent of the samples at `s`, form a
one-dimensional (in `u`) family whose covering is cheap, and sandwich
`hat-v*`'s dependence on cell `s` tightly. Union over the grid instead
of over a function class — the dependence is broken **state-locally**
at logarithmic cost. (Compare the fixed-`V*` trick of
`regret_and_exploration/02` — same enemy, different weapon.)

Together: `hat-pi*` is `eps`-optimal with `N = tilde-O((1-gamma)^{-3}
eps^{-2})` per cell, for the full range `eps in (0, 1/sqrt(1-gamma)]`
(AKY closed the range Azar's original left partial). ∎ (outline)

## What The Result Means

```text
1. the model is a SUFFICIENT STATISTIC for everything: with the same
   sample budget, the empirical model supports eps-optimal POLICIES,
   values, AND — reused — any later reward function on the same
   dynamics (transfer for free; purely model-free methods have no
   analogous artifact);
2. "model-based vs model-free" at the generative-model level is a
   non-contest statistically (both hit the same minimax rate:
   finite_time_td_q/04's VR-Q-learning vs this) — the real differences
   live in computation (planning cost vs SA cost) and in the ONLINE
   setting (exploration through models: notebook 03, where model
   confidence sets are the natural optimism carrier);
3. sophistication buys constants, not rates: the entire algorithmic
   ingenuity budget should therefore be spent where the assumptions
   break — unknown reachability (03), function approximation (04),
   and planner-model interaction (05).
```

## What Remains Open

The burn-in regime (very coarse `eps ~ Vmax`: upper and lower bounds
still mismatched in lower-order terms); certainty equivalence with
*misspecified* model classes (plug-in optimality is a well-specified
statement; the agnostic version connects to 04's value equivalence);
and the online analogue of the leave-one-out device beyond tabular.
