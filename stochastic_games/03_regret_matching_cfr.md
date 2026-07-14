# Regret Matching And Counterfactual Regret Minimization

## The Repair For Cycling: Stop Best-Responding, Start Regret-Minimizing

Notebook 02's dynamics failed because best response is discontinuous
and history-less. No-regret dynamics smooth both: play mixtures
proportional to accumulated regrets, and let the guarantees attach to
**time averages**.

## External Regret And Its Meaning

Player `i` plays mixed strategies `x_t`; the environment (other players)
produces payoff vectors `u_t`. External regret after `T` rounds:

```math
R_T=\max_{a}\ \sum_{t=1}^T\Big(u_t(a)-\big\langle x_t,u_t\big\rangle\Big)
\qquad\text{(vs the best FIXED action in hindsight).}
```

**Theorem (no-regret → equilibrium).** If in a game all players run
algorithms with `R_T = o(T)`, the **empirical joint distribution** of
play converges to the set of **coarse correlated equilibria** (CCE);
in two-player zero-sum games, the marginal average strategies converge
to the minimax optimum with duality gap `<= (R_T^1 + R_T^2)/T`.

*Proof (zero-sum case, three lines).* Let `bar-x, bar-y` be average
strategies. For any comparator `x`:
`(1/T) sum <x, A y_t> - <x_t, A y_t> <= R_T^1/T`, i.e.
`<x, A bar-y> <= (1/T) sum <x_t, A y_t> + R_T^1/T`; symmetrically for
player 2 with `>= ... - R_T^2/T`. Chain the two: every `x` does at most
`(R^1+R^2)/T` better against `bar-y` than the realized average payoff,
and every `y` at most that much better against `bar-x`: `(bar-x, bar-y)`
is an `((R^1+R^2)/T)`-saddle point. ∎

Note the exact deliverable: **averages, not iterates** — the actual play
may cycle forever (notebook 02, 05); the *time-averaged* play is what
acquires meaning. This is the precise sense in which no-regret dynamics
"solve" games.

## Regret Matching (Hart–Mas-Colell 2000)

The simplest no-regret rule, and the one poker AI runs on:

```math
x_{t+1}(a)\ \propto\ \big[R_t(a)\big]^+
\qquad
\Big(R_t(a)=\textstyle\sum_{\tau\le t}u_\tau(a)-\langle x_\tau,u_\tau\rangle\Big),
```

uniform if all regrets are nonpositive. **Regret bound:**
`R_T <= Delta sqrt(|A| T)` (payoff range `Delta`).

*Proof via Blackwell approachability / the potential
`Phi_t = sum_a ([R_t(a)]^+)^2`.* One step:

```math
\Phi_{t+1}\le\Phi_t+2\sum_a[R_t(a)]^+\,\big(u_{t+1}(a)-\langle x_{t+1},u_{t+1}\rangle\big)+\Delta^2|A| ,
```

and the middle term is **zero by the choice of `x_{t+1}`**:

```math
\sum_a[R_t(a)]^+u_{t+1}(a)
=
\Big(\sum_b [R_t(b)]^+\Big)\,\big\langle x_{t+1},u_{t+1}\big\rangle .
```

So `Phi_T <= Delta^2 |A| T`, and
`max_a [R_T(a)]^+ <= sqrt(Phi_T) <= Delta sqrt(|A| T)`. ∎
(The rule is engineered to make the drift orthogonal to the regret
vector — Blackwell's approachability condition for the nonpositive
orthant, in its scalar-potential form.)

## CFR: The Decomposition That Conquered Poker

Extensive-form games (sequential, imperfect information) have strategy
spaces exponential in the game tree — running regret matching on whole
strategies is hopeless. CFR (Zinkevich et al. 2007) localizes:

At each **information set** `I` (a decision point up to what the player
can distinguish), define counterfactual values
`v_I(a)` = expected payoff given `I` is reached and `a` taken, weighting
by opponents'/chance's reach probabilities only — and run an independent
regret matcher per infoset on these values.

**Theorem (regret decomposition).** Total external regret is bounded by
the sum of per-infoset counterfactual regrets:

```math
R_T^{\text{full game}}
\ \le\
\sum_{I}\ R_T^{\text{cf}}(I)
\ \le\
|\mathcal{I}|\ \Delta\sqrt{|A|\,T}.
```

*Proof mechanism.* Induction from the leaves up the player's decision
tree: the regret of the best deviation at the root decomposes into
"deviate now" plus "deviate later," and the counterfactual weighting is
exactly what makes the later terms *independent of the current infoset's
mixing* — the deviation gain telescopes along the tree
(the same telescoping-through-decisions skeleton as the performance
difference lemma, `policy_gradient/03`, on the game tree). ∎

Combined with the zero-sum averaging theorem: **average CFR strategies
converge to Nash at rate `O(|I| Delta sqrt(|A|/T))`** — the algorithm
behind superhuman poker (with sampling variants: MCCFR; with function
approximation: DeepCFR; with the `+`-variant's empirical speedups).

## Interfaces

```text
to regularized_mdps_and_duality/05:  regret matching and multiplicative-
    weights/MD are siblings (different potentials over the simplex);
    MD's three-point telescope and RM's quadratic potential are the two
    standard no-regret engines;
to rlhf/self-play LLM training:      preference optimization against a
    co-adapting opponent (self-play fine-tuning, debate) inherits
    exactly this file's guarantee structure — averages converge to
    equilibria of the modeled game, iterates need not.
```

## What Remains Open

Last-iterate convergence for RM-style dynamics (recent progress via
optimism); tight CFR rates in `|I|` (the linear factor is loose in
practice); and sound function-approximation CFR (DeepCFR works;
its regret decomposition with shared networks is not fully priced).
