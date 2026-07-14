# The Linear Programming Formulation And Occupancy Measures

Dynamic programming is not the only exact solution method. The Bellman
optimality equation is equivalent to a linear program, and its dual
introduces the object — the occupancy measure — that policy gradient methods
implicitly optimize over.

## The Primal: Values

Key fact (from monotonicity, notebook 03): if `v >= T v` then `v >= v_*`
(apply `T` repeatedly: `v >= T v >= T^2 v >= ... -> v_*`). So `v_*` is the
**smallest superharmonic** function, i.e. the solution of

```math
\begin{aligned}
\min_{v\in\mathbb{R}^{|S|}}\quad & \textstyle\sum_s \mu(s)\,v(s)\\
\text{s.t.}\quad & v(s)\ \ge\ r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,v(s')
\qquad\forall (s,a),
\end{aligned}
```

for any strictly positive weights `mu`. The constraint linearizes the `max`:
"`v >= T v`" is one linear inequality per action. At the optimum the
constraint is tight in each state for at least one action — the optimal
actions.

## The Dual: Occupancy Measures

Assigning a multiplier `lambda(s,a) >= 0` per constraint and dualizing:

```math
\begin{aligned}
\max_{\lambda\ge 0}\quad & \sum_{s,a}\lambda(s,a)\,r(s,a)\\
\text{s.t.}\quad
& \sum_a \lambda(s',a)
=
\mu(s')+\gamma\sum_{s,a}P(s'\mid s,a)\,\lambda(s,a)
\qquad\forall s' .
\end{aligned}
```

The constraint is a **flow conservation** law: mass into state `s'` (initial
mass `mu`, plus discounted transported mass) equals mass out. Its solutions
are exactly the discounted state-action occupancy measures

```math
\lambda_\pi(s,a)
=
\sum_{t=0}^\infty \gamma^t\,\mathbb{P}_\mu^\pi\{S_t=s,\,A_t=a\},
```

and the objective is exactly the expected return:

```math
\sum_{s,a}\lambda_\pi(s,a)r(s,a)=\mathbb{E}_\mu^\pi\Big[\sum_t\gamma^t r(S_t,A_t)\Big]
=
(1-\gamma)^{-1}\,\mathbb{E}_{s\sim d_\mu^\pi,\,a\sim\pi}[r(s,a)]\cdot(1-\gamma)\ .
```

Policy recovery is by conditioning:

```math
\pi(a\mid s)=\frac{\lambda(s,a)}{\sum_{a'}\lambda(s,a')}.
```

## What The Dual View Buys

**1. The feasible set is a polytope.** The map `pi -> lambda_pi` is a
bijection between stationary policies and the occupancy polytope, and the
return is **linear** in `lambda` (while highly nonconvex in policy
parameters `theta`). Every difficulty of policy-gradient optimization is an
artifact of the parameterization, not the underlying problem.

**2. Vertices are deterministic policies.** A linear objective over a
polytope is maximized at a vertex; vertices have one active action per state.
This is the cleanest proof that SD policies suffice — and adding linear
constraints (e.g. cost bounds, `E[cost] <= c`) moves the optimum off the
vertices, proving optimal constrained policies are genuinely stochastic
(the `SR -> SD` failure flagged in notebook 01).

**3. Strong duality = Bellman equation.** Complementary slackness says:
`lambda(s,a) > 0` only where the primal constraint is tight, i.e. occupancy
flows only through Bellman-optimal actions. The DP and LP theories are the
same theorem written in two coordinate systems.

**4. The discounted state distribution.** Normalizing occupancy over states,

```math
d_\mu^\pi(s)=(1-\gamma)\sum_{t=0}^\infty\gamma^t\,\mathbb{P}_\mu^\pi\{S_t=s\},
```

is the measure that appears — always, and non-negotiably — in the policy
gradient theorem (`policy_gradient/01`) and the performance difference lemma
(`policy_gradient/03`). It is worth having met it here first as a dual
variable: gradients of the return with respect to the policy are inner
products against occupancy, because occupancy is the Lagrange dual of value.
