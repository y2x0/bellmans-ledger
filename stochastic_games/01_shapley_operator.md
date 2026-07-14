# The Shapley Operator

## Zero-Sum Stochastic Games

Two players; state `s`; simultaneous actions `a in A, b in B`; player 1
receives `r(s, a, b)`, player 2 receives `-r(s, a, b)`; transitions
`P(s'|s, a, b)`; discount `gamma`. Player 1 maximizes, player 2
minimizes, both with full knowledge.

## The Operator

Replace `mdp_foundations/03`'s `max_a` by the **value of a matrix game**:

```math
(\mathcal{T}v)(s)
=
\mathrm{val}\Big[\ \underbrace{r(s,a,b)+\gamma\textstyle\sum_{s'}P(s'\mid s,a,b)\,v(s')}_{=:\ M_v(s)\in\mathbb{R}^{A\times B}}\ \Big],
```

where `val[M] = max_{x in Delta(A)} min_{y in Delta(B)} x^T M y`
(= the min-max, by von Neumann's minimax theorem — the game has a value
in mixed strategies).

## The Contraction

**Lemma (val is nonexpansive).** For matrices `M, N`:

```math
\big|\mathrm{val}[M]-\mathrm{val}[N]\big|\ \le\ \max_{a,b}\big|M_{ab}-N_{ab}\big| .
```

*Proof.* Let `(x*, y*)` be optimal for `M` and `y'` optimal for `N`.
Then

```math
\mathrm{val}[M]-\mathrm{val}[N]
\ \le\
x^{*\top}My'-x^{*\top}Ny'
=
x^{*\top}(M-N)\,y'
\ \le\
\|M-N\|_{\max},
```

(first inequality: `val[M] <= x*^T M y'` since `y'` may be suboptimal
against `x*` for `M`... careful — with `val[M] = max min`,
`val[M] = min_y x*^T M y <= x*^T M y'`; and
`val[N] = max_x min_y x^T N y >= min_y x*^T N y`, so
`-val[N] <= -x*^T N y''` for the minimizing `y''`; run the display with
`y''` in both slots). Symmetrize. ∎ — the scalar max-lemma
(`mdp_foundations/03`) applied **twice**, once per player: bilinear
forms over product simplices are maxima of minima of affine functions,
and each layer is nonexpansive.

**Theorem (Shapley 1953).** `T` is a `gamma`-contraction in
`||.||_inf`; its unique fixed point `v*` is the **value of the
stochastic game** (each player has a stationary mixed strategy
guaranteeing `v*` against any opponent play); optimal stationary
strategies are the per-state matrix-game optima of `M_{v*}(s)`.

*Proof.* Contraction: the lemma + the `gamma`-contractive expectation,
exactly as in the MDP case. That the fixed point is the game value:
verification both ways — if player 1 plays the maximin mixed strategy of
`M_{v*}(s)` at every state, then against ANY opponent behavior the
one-stage payoff-plus-continuation is `>= v*(s)` in expectation
(supermartingale argument, optional stopping — the "policy improvement
telescope" of `mdp_foundations/05`, run with an adversary); symmetrically
for player 2 with `<=`. The two guarantees pinch. ∎

## What Transfers From The MDP Theory, And What Does Not

```text
transfers verbatim:
  - Banach apparatus: VI ("Shapley iteration") converges geometrically;
    a posteriori bounds; greedy(v*) — here maximin(M_{v*}) — is optimal
  - Q-learning analogue (minimax-Q, Littman 1994): replace max by val
    in the tabular update; Watkins-style convergence goes through
    (stochastic_approximation/05's proof only used contraction +
    monotone-ish structure; val is where max was)

does NOT transfer:
  - deterministic optimal strategies: val needs MIXED strategies
    (matching pennies); the SD-sufficiency of mdp_foundations/01 is a
    one-player privilege — vertices of the occupancy polytope lose
    their monopoly the moment payoffs couple two occupancies
  - naive policy iteration: the improvement step's monotonicity breaks
    against a re-optimizing opponent — notebook 02's subject
  - general-sum anything: with non-zero-sum payoffs "val" has no
    single-valued, nonexpansive replacement (Nash values are non-unique
    and discontinuous in payoffs); the operator theory genuinely ends
    at zero-sum (and its potential-game/cooperative cousins)
```

## The Per-Step Cost

Each backup at each state now solves a matrix game — an LP of size
`|A| x |B|` (the minimax LP duality pair), vs the MDP's `O(|A|)` scan.
Everything downstream that "just calls val" (minimax-Q, game VI) carries
this LP inside its inner loop; large-scale practice (04) exists largely
to amortize it.

## What Remains Open

At this level: nothing for finite zero-sum discounted games (closed since
1953). The open territory is undiscounted/average-reward stochastic games
(existence of the value: the Mertens–Neyman theorem, vastly harder),
sample complexity of learning zero-sum games (near-minimax rates are
recent), and everything general-sum — where the next notebooks pick up.
