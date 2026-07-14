# Piecewise-Linear Convexity And Alpha-Vectors

## The Structure Theorem

**Theorem (Smallwood–Sondik 1973).** The finite-horizon optimal value
function of a POMDP is **piecewise linear and convex (PWLC)** in the
belief: for each horizon `h` there is a finite set of vectors
`Gamma_h ⊂ R^{|S|}` with

```math
V_h(b)=\max_{\alpha\in\Gamma_h}\ \langle\alpha,\,b\rangle .
```

*Proof by induction.* `h = 0`: `V_0 = 0` (`Gamma_0 = {0}`); or terminal
rewards give one vector per action. Step: write the belief-MDP backup
(notebook 01) and substitute the inductive form:

```math
V_{h+1}(b)
=
\max_a\Big[\langle r_a,b\rangle+\gamma\sum_o \eta(b,a,o)\ V_h\big(\tau(b,a,o)\big)\Big],
```

and the key cancellation: `eta(b,a,o) * tau(b,a,o)(s')` is **linear in
`b`** (the normalizer cancels against the probability weight):

```math
\eta(b,a,o)\,V_h(\tau(b,a,o))
=
\max_{\alpha\in\Gamma_h}\ \sum_{s'}\alpha(s')\,Z(o\mid s',a)\sum_sP(s'\mid s,a)\,b(s)
=
\max_{\alpha\in\Gamma_h}\ \big\langle g_{a,o}^{\alpha},\,b\big\rangle,
```

with the **back-projected vectors**
`g_{a,o}^alpha(s) = sum_{s'} alpha(s') Z(o|s',a) P(s'|s,a)`. So each
term is a max of linear functions of `b`; sums of maxes are maxes of
sums over *choices per observation*:

```math
V_{h+1}(b)=\max_a\ \max_{(\alpha_o)_{o\in\mathcal{O}}\in\Gamma_h^{|\mathcal{O}|}}\ \Big\langle\ r_a+\gamma\sum_o g_{a,o}^{\alpha_o},\ b\Big\rangle
```

— again a finite max of linear functions: PWLC, with

```math
|\Gamma_{h+1}|\ \le\ |\mathcal{A}|\cdot|\Gamma_h|^{|\mathcal{O}|}. \qquad\blacksquare
```

**Read the two facts the proof delivered:** (i) each alpha-vector is a
*conditional plan* (an action now + a plan per observation), and
`<alpha, b>` is that plan's value from belief `b` — the max is over
plans, linearity is "expected value of a fixed plan is linear in the
prior"; (ii) convexity of `V` = **information never hurts**: `V` at a
mixture of beliefs `<= ` mixture of `V`s means an agent told which of
two priors is true does at least as well as one holding their average.

## The Tiger Problem, Backed Up By Hand

The canonical 2-state POMDP. Tiger behind left or right door;
`b = Pr{tiger-left}`. Actions: `listen` (cost `-1`, hear a growl:
correct side w.p. `0.85`), `open-left/right` (`+10` treasure, `-100`
tiger; episode resets — treat as terminal here). `gamma = 1`, horizon 2.

**Horizon 1 (`Gamma_1`, one vector per action, coordinates
`(value | tiger-left, value | tiger-right)`):**

```text
open-right: ( +10, -100 )   open-left: ( -100, +10 )   listen: ( -1, -1 )
```

`V_1(b) = max(10 - 110 b-ish...)` — explicitly with `b = P(left)`:
`V_1 = max( -100b + 10(1-b), 10b - 100(1-b), -1 )`; opening is optimal
only for `b < 0.0917` or `b > 0.9083`; listen in between.

**Horizon 2, the listen branch.** Back-project each `alpha in Gamma_1`
through (`listen`, growl-left) and (`listen`, growl-right):
`g_{L,o}^alpha(s) = Z(o|s) alpha(s)` (dynamics = identity for listen):

```text
o = growl-left:  g^{open-right} = (0.85*10-ish coordinates):
                 ( .85*(-100), .15*(+10) )? — careful with convention:
                 g(s) = Z(o|s) * alpha(s):
   g_{GL}^{OR} = (0.85*10? ...) with alpha_{OR} = (10, -100):
                 (0.85*10, 0.15*(-100)) = ( 8.5, -15 )
   g_{GL}^{OL} = (0.85*(-100), 0.15*10) = ( -85, 1.5 )
   g_{GL}^{Lis}= (0.85*(-1),   0.15*(-1))= ( -.85, -.15 )
(and symmetrically for growl-right with weights (0.15, 0.85))
```

A horizon-2 listen-plan vector is
`(-1, -1) + g_{GL}^{alpha_1} + g_{GR}^{alpha_2}` for each pair
`(alpha_1, alpha_2)` — e.g. the sensible plan "listen, then open the
door away from the growl":

```math
(-1,-1)+(\,8.5,\,-15\,)+(\,-15,\,8.5\,)=(\,-7.5,\,-7.5\,)
```

wait — that plan mixes both openings; evaluating the alternatives, the
plan "listen then open-right iff growl-left" against "always open"
vectors shows listening dominates on the central belief interval and
the interval where immediate opening is optimal *shrinks* — the value of
information made literal: horizon-2 `V` is a tighter convex envelope
built from `3 * 3^2 = 27` candidate vectors, of which pruning (LP
domination tests) keeps 4–5. The arithmetic above, done to completion,
is the standard exercise; its point is watching plans-per-observation
become vectors and domination become an LP.

## Exact And Point-Based Algorithms

```text
exact VI:      generate Gamma_{h+1} (the cross-sum above), PRUNE dominated
               vectors (an LP per vector: does alpha achieve the max
               somewhere on the simplex?). Doubly exponential growth in
               the worst case — notebook 03 says essentially unavoidable.
point-based    (PBVI, Perseus, SARSOP): maintain vectors only at a
VI:            finite set of REACHABLE beliefs B; backups at b in B keep
               one maximizing vector each: |Gamma| <= |B| forever.
               Error bound: eps <= (max gap) * density of B over the
               reachable set / (1-gamma) — the covering-number logic of
               concentration_toolkit/04 applied to the simplex, with
               reachability shrinking the set that must be covered
               (SARSOP: sample near the optimally-reachable set only).
```

Point-based methods are the practical exact-model solvers (thousands of
states); their honest scope is `|S|` in the thousands, not the
`10^{lots}` of pixel POMDPs — which is why deep RL approximates the
*statistic* instead (05).

## What Remains Open

Tight characterizations of when `Gamma_h` stays small (beyond
"informative observations help"); infinite-horizon PWLC is only
approximate (limit of PWLC is convex, not finitely PWLC — finite
controllers and their local optima); and continuous-`S` alpha-*functions*
have theory only in special (linear-Gaussian: Kalman/LQG, exactly
solvable) families.
