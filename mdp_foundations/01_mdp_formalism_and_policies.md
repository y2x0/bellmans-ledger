# MDP Formalism And Policy Classes

## The Object

```math
M=(\mathcal{S},\mathcal{A},P,r,\gamma),
\qquad
P:\mathcal{S}\times\mathcal{A}\to\Delta(\mathcal{S}),
\qquad
r:\mathcal{S}\times\mathcal{A}\to[-R_{\max},R_{\max}].
```

Throughout the foundations family `S` and `A` are finite; the results extend
to Borel state spaces with measurable-selection care (Puterman ch. 2), and the
places where that care actually matters are flagged.

## The Lattice of Policy Classes

A policy is a rule for choosing actions. Four nested classes:

```math
\Pi^{HR}\ \supset\ \Pi^{MR}\ \supset\ \Pi^{SR}\ \supset\ \Pi^{SD}
```

```text
HR  history-dependent randomized:   pi_t(a | s_0, a_0, ..., s_t)
MR  Markov randomized:              pi_t(a | s_t)          (time-varying)
SR  stationary randomized:          pi(a | s_t)            (time-invariant)
SD  stationary deterministic:       a = pi(s_t)
```

The a priori search space is `HR` — uncountably infinite-dimensional. The
entire computational tractability of MDPs rests on collapsing it.

## Theorem (Sufficiency of Stationary Deterministic Policies)

For a finite discounted MDP:

```math
\sup_{\pi\in\Pi^{HR}} v_\pi(s)
=
\max_{\pi\in\Pi^{SD}} v_\pi(s)
\qquad\text{for every }s\text{ simultaneously.}
```

### Proof structure (three collapses)

**HR → MR.** Fix `pi in HR` and initial state `s`. Define the marginal
state-action occupancy at each time:

```math
\rho_t(s',a')=\mathbb{P}_s^{\pi}\{S_t=s',A_t=a'\}.
```

Construct the Markov policy
```math
\tilde\pi_t(a'\mid s')=\frac{\rho_t(s',a')}{\sum_{a''}\rho_t(s',a'')}.
```
By induction on `t`, `tilde-pi` reproduces every marginal `rho_t` exactly.
Since the expected return is a function of the marginals only,

```math
v_{\tilde\pi}(s)
=
\sum_{t=0}^\infty \gamma^t \sum_{s',a'}\rho_t(s',a')\,r(s',a')
=
v_\pi(s).
```

History dependence buys nothing because the objective is linear in the
occupancy marginals. (This is also the germ of the LP formulation in
notebook 06.)

**MR → SR.** Handled by the fixed-point machinery of notebooks 03–04:
`v_*` exists as the unique fixed point of the optimality operator, and its
value is attained by a stationary policy, so time-varying policies cannot
exceed it.

**SR → SD.** Given the optimal `v_*`, choose in each state

```math
\pi(s)\in\arg\max_a\Big[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)v_*(s')\Big].
```

The argmax is attained because `A` is finite. Notebook 05 proves this greedy
policy achieves `v_*`. Randomization buys nothing because a linear functional
over a simplex is maximized at a vertex. ∎

### Where each collapse fails

```text
HR -> MR   fails under partial observability (POMDPs): the marginal
           construction needs the state to be observed. Memory becomes
           genuinely necessary; the belief state restores Markovness at the
           cost of a continuous state space.

SR -> SD   fails in constrained MDPs and in games/multi-agent settings:
           with multiple objectives the optimum lies inside the occupancy
           polytope, not at a vertex, so optimal policies are genuinely
           stochastic. (Also why PPO-style methods keep stochastic policies
           during learning: exploration and gradient flow, not optimality.)

finite A   in continuous action spaces the argmax may not be attained;
           one needs compactness of A and upper semicontinuity of the
           bracket (measurable selection theorems).
```

## Induced Markov Chain

A stationary `pi` reduces the MDP to a Markov chain with

```math
P_\pi(s'\mid s)=\sum_a \pi(a\mid s)P(s'\mid s,a),
\qquad
r_\pi(s)=\sum_a \pi(a\mid s)r(s,a),
```

in matrix form a row-stochastic `P_pi in R^{|S| x |S|}` and reward vector
`r_pi in R^{|S|}`. All spectral statements later (e.g. `||gamma P_pi|| < 1`,
the resolvent `(I - gamma P_pi)^{-1}`) are about this matrix. Since `P_pi` is
row-stochastic its spectral radius is 1, hence

```math
\rho(\gamma P_\pi)=\gamma<1,
```

and `I - gamma P_pi` is invertible with the Neumann series

```math
(I-\gamma P_\pi)^{-1}=\sum_{t=0}^\infty \gamma^t P_\pi^t \ \ge 0
\quad\text{(entrywise)}.
```

The entrywise nonnegativity of the resolvent is used silently in every
monotonicity argument that follows; it is worth noticing once, here.
