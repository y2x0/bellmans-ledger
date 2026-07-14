# Bellman 1957: A Markovian Decision Process

```text
R. Bellman, "A Markovian Decision Process," Journal of Mathematics and
Mechanics 6(4):679–684, 1957.  (Also RAND paper P-1066.)
Companion essay: "The Theory of Dynamic Programming," Bull. AMS 60:503–515, 1954.
PDF: papers/bellman-1957.pdf  (scan of the journal pages; no text layer)
```

This ~6-page paper is where the phrase *Markovian decision process* enters
the literature, and where the functional-equation method is applied to
stochastic sequential decision problems. Reading it after notebooks 01–05 is
an exercise in recognition: everything is there, in 1957 notation.

## The Model, As Bellman States It

A system occupies one of `N` states. In state `i`, choosing decision `k`
yields expected return `r_i^k` and moves the system to state `j` with
probability `p_{ij}^k`. The problem: choose decisions to maximize expected
return over an infinite horizon.

## The Principle Of Optimality

> An optimal policy has the property that, whatever the initial state and
> initial decision are, the remaining decisions must constitute an optimal
> policy with regard to the state resulting from the first decision.

The principle is not an axiom; it is a consequence of the additive structure
of the return and the Markov property, and its content is precisely the
tower-property computation in `mdp_foundations/03` that produced
`v_pi = T^pi v_pi`. What the principle does rhetorically is license writing
a **functional equation** for the optimal return `f(i)`:

```math
f(i)=\max_k\Big[r_i^k+\sum_{j}p_{ij}^k\,f(j)\Big]
```

(undiscounted in parts of the paper — Bellman works with total and average
return variants; the discounted version inserts the `gamma` we have carried
throughout). This is the Bellman optimality equation of notebook 02, born.

## What The Paper Proves

**Existence and uniqueness** of the solution, by the method of **successive
approximations**: pick `f_0`, iterate

```math
f_{n+1}(i)=\max_k\Big[r_i^k+\sum_j p_{ij}^k f_n(j)\Big],
```

and show the sequence converges monotonically to the unique solution. In
modern language this is exactly value iteration plus the contraction /
monotone convergence arguments of notebook 04 — Banach's theorem (1922)
predates the paper, and Bellman's argument is the constructive half of it,
adapted to the max-operator.

**Approximation in policy space.** Bellman contrasts iterating on the value
function with iterating on the policy — computing the return of a fixed
policy, then improving it — anticipating the policy iteration algorithm that
Howard's 1960 monograph (*Dynamic Programming and Markov Processes*) develops
fully, and hence the whole GPI frame of `mdp_foundations/05`.

## Historical Load-Bearing Points

```text
1. The reduction from strategy search to a fixed-point problem on |S|
   unknowns is THE conceptual move of the field; every notebook in this
   repository iterates, samples, or projects Bellman's functional equation.

2. "Curse of dimensionality" is Bellman's own coinage (1957 book, Dynamic
   Programming): the state enumeration implicit in f(i) is exponential in
   state dimensionality. Sections value_based_deep_rl/ onward exist because
   of this sentence.

3. The 1954 Bulletin essay is the readable manifesto: principle of
   optimality, deterministic and stochastic examples, calculus-of-variations
   connections. The 1957 paper is the tight mathematical note. Read the
   essay for the worldview, the paper for the theorem.
```

## What Is *Not* There

No learning: `p_{ij}^k` and `r_i^k` are known throughout. The question of
solving the functional equation from samples of the transition process —
without ever forming `p` — waits for Robbins–Monro (1951) to be married to
DP, which is Watkins 1989 / Q-learning, i.e. `stochastic_approximation/05`.
The gap between this paper and that one is exactly the gap between the
`mdp_foundations/` and `stochastic_approximation/` families.
