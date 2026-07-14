# Return Criteria And Value Functions

## Three Criteria

```math
\text{discounted:}\quad
G_t=\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}
```

```math
\text{total (episodic):}\quad
G_t=\sum_{k=0}^{T-t-1} R_{t+k+1},\qquad T=\text{absorption time}
```

```math
\text{average reward:}\quad
\rho(\pi)=\lim_{T\to\infty}\frac{1}{T}\,\mathbb{E}_\pi\Big[\sum_{t=1}^{T}R_t\Big]
```

## What Discounting Buys

Discounting is usually motivated economically. Its real role is analytic.
Three purchases:

**1. Boundedness.** `|G_t| <= R_max / (1 - gamma)` pointwise, so `v_pi` is a
well-defined bounded function for every policy with no assumptions on the
chain structure.

**2. Contraction modulus.** `gamma` **is** the contraction constant of the
Bellman operators (notebook 03). Undiscounted total reward gives a modulus of
1 — no contraction, and existence/uniqueness require structural assumptions
instead (proper policies: every policy reaches an absorbing state a.s.;
stochastic shortest path theory, Bertsekas–Tsitsiklis).

**3. An effective horizon.** Rewards beyond `H_eff` contribute at most `eps`:

```math
\sum_{k\ge H}\gamma^k R_{\max}
=
\frac{\gamma^H R_{\max}}{1-\gamma}\le\varepsilon
\quad\Longleftrightarrow\quad
H\ \ge\ \frac{1}{1-\gamma}\log\frac{R_{\max}}{\varepsilon(1-\gamma)}.
```

So `1/(1-gamma)` behaves as a horizon, and every error bound in this
repository carries factors of `1/(1-gamma)` — often squared. Where a
`(1-gamma)^{-2}` appears (e.g. the TRPO penalty coefficient), one factor is
"errors persist for a horizon" and the second is "the visitation distribution
can also shift for a horizon."

## Value Functions

```math
v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],
\qquad
q_\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
```

The conditioning is well-defined by stationarity of `pi`: the law of `G_t`
given `S_t = s` does not depend on `t` (time-homogeneous chain). For
non-stationary policies one must index `v` by `t`; this is precisely what
sufficiency of stationary policies (notebook 01) lets us avoid.

Interconversions used constantly:

```math
v_\pi(s)=\sum_a \pi(a\mid s)\,q_\pi(s,a),
\qquad
q_\pi(s,a)=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,v_\pi(s').
```

The **advantage** measures the value of deviating for one step:

```math
A_\pi(s,a)=q_\pi(s,a)-v_\pi(s),
\qquad
\sum_a\pi(a\mid s)A_\pi(s,a)=0.
```

Its expectation under `pi` vanishing in every state is the invariant that
makes it the correct integrand for policy gradients (`policy_gradient/01`)
and the correct target for improvement bounds (`policy_gradient/03`).

## Closed Form For Evaluation

The Bellman expectation identity (derived in notebook 03) in matrix form:

```math
v_\pi=r_\pi+\gamma P_\pi v_\pi
\quad\Longrightarrow\quad
v_\pi=(I-\gamma P_\pi)^{-1} r_\pi
=\sum_{t=0}^\infty \gamma^t P_\pi^t\, r_\pi .
```

Reading the Neumann series right-to-left: the value function is the
discounted superposition of expected rewards along the chain. Exact
evaluation is an `O(|S|^3)` linear solve; everything in
`stochastic_approximation/` is an iterative, sampled substitute for this
solve, and LSTD (`stochastic_approximation/04`) is this solve projected onto
a feature subspace.

## Optimal Value Functions

```math
v_*(s)=\max_\pi v_\pi(s),
\qquad
q_*(s,a)=\max_\pi q_\pi(s,a),
```

```math
v_*(s)=\max_a q_*(s,a),
\qquad
q_*(s,a)=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}q_*(s',a').
```

The `max` over policies is over the full class `Pi^HR`; notebook 01 justified
reading it as a max over `Pi^SD`. The recursive characterizations are the
Bellman optimality equations, and their nonlinearity (the interior `max`) is
the single reason the field exists: linear systems get solved, nonlinear
fixed-point equations get iterated.
