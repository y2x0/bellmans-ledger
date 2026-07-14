# Bellman Operators: Monotonicity And Contraction

## The Two Operators

On the space `B(S)` of bounded functions `v : S -> R` with the sup-norm
`||v||_inf = max_s |v(s)|`:

```math
(\mathcal{T}^\pi v)(s)
=
\sum_a\pi(a\mid s)\Big[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,v(s')\Big]
\qquad\text{(evaluation, affine)}
```

```math
(\mathcal{T} v)(s)
=
\max_a\Big[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,v(s')\Big]
\qquad\text{(optimality, convex)}
```

In vector form `T^pi v = r_pi + gamma P_pi v`. Note `T` is the upper envelope
of the affine family `{T^pi}`:

```math
\mathcal{T}v=\max_{\pi\in\Pi^{SD}}\mathcal{T}^\pi v
\quad\text{(pointwise max)},
```

so `T` is convex and piecewise-affine in `v` — the geometric fact behind the
Newton's-method reading of policy iteration (notebook 05).

## Property 1: Monotonicity

```math
u\le v \ \text{(componentwise)}
\quad\Longrightarrow\quad
\mathcal{T}^\pi u\le \mathcal{T}^\pi v
\ \ \text{and}\ \
\mathcal{T} u\le \mathcal{T} v .
```

*Proof.* `T^pi u - T^pi v = gamma P_pi (u - v)` and `P_pi` has nonnegative
entries. For `T`: let `pi_v` be greedy for `v`; then
`T u >= T^{pi_v} u >= T^{pi_v} v = T v` reversed appropriately — concretely,
for each `s`, the maximizing action for `u` gives a lower bound through `v`'s
bracket. ∎

Monotonicity is logically independent of contraction and is doing separate
work: it converts one-step inequalities into all-horizons inequalities
(the engine of the policy improvement theorem, notebook 05).

## Property 2: Constant Shift

```math
\mathcal{T}(v+c\mathbf{1})=\mathcal{T}v+\gamma c\,\mathbf{1},
\qquad c\in\mathbb{R}.
```

Immediate from `P_pi 1 = 1`. Monotonicity + this shift property together
*imply* contraction, which is the cleanest route to the proof:

## Property 3: gamma-Contraction In The Sup-Norm

```math
\|\mathcal{T}u-\mathcal{T}v\|_\infty\le\gamma\|u-v\|_\infty,
\qquad
\|\mathcal{T}^\pi u-\mathcal{T}^\pi v\|_\infty\le\gamma\|u-v\|_\infty .
```

*Proof via shift.* Let `c = ||u - v||_inf`, so `v - c1 <= u <= v + c1`.
Apply `T` and use monotonicity plus the shift property:

```math
\mathcal{T}v-\gamma c\mathbf{1}
\ \le\
\mathcal{T}u
\ \le\
\mathcal{T}v+\gamma c\mathbf{1}
\quad\Longrightarrow\quad
\|\mathcal{T}u-\mathcal{T}v\|_\infty\le\gamma c.\qquad\blacksquare
```

*Direct proof for `T` (worth knowing because it isolates the max-lemma).*
For any two families `{f(a)}, {g(a)}`:

```math
\big|\max_a f(a)-\max_a g(a)\big|\ \le\ \max_a|f(a)-g(a)|.
```

(Let `a*` attain `max f`; then
`max f - max g <= f(a*) - g(a*) <= max_a |f - g|`; symmetrize.)
Apply with `f, g` the brackets of `u, v`:

```math
|(\mathcal{T}u)(s)-(\mathcal{T}v)(s)|
\le
\max_a\ \gamma\sum_{s'}P(s'\mid s,a)\,|u(s')-v(s')|
\le
\gamma\|u-v\|_\infty. \qquad\blacksquare
```

The max-lemma is the entire reason optimality "costs nothing" in sup-norm:
the nonexpansive max composes with the `gamma`-contractive expectation. Keep
it in view — it is exactly the step that **fails** for distributions
(`distributional_rl/03`) and for projections in `L2` (`stochastic_approximation/06`).

## Fixed Points

```math
\mathcal{T}^\pi v_\pi=v_\pi,
\qquad
\mathcal{T} v_*=v_* .
```

The first is the Bellman expectation equation — derived by conditioning the
return on the first transition:

```math
v_\pi(s)
=\mathbb{E}_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s]
=\sum_a\pi(a\mid s)\Big[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)v_\pi(s')\Big],
```

using the Markov property and the tower rule to replace
`E[G_{t+1} | S_{t+1}=s']` by `v_pi(s')`. The second is the Bellman optimality
equation; that its unique fixed point actually equals `sup_pi v_pi` is proved
in notebook 04, and is a genuine theorem — a priori, "fixed point of `T`" and
"value of the best policy" are different definitions.

## Action-Value Versions

```math
(\mathcal{T}^\pi q)(s,a)=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\sum_{a'}\pi(a'\mid s')q(s',a'),
```

```math
(\mathcal{T} q)(s,a)=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}q(s',a').
```

Identical properties, identical proofs, on `B(S x A)`. The `q`-form of `T` is
the operator that Q-learning stochastically approximates and that DQN turns
into a regression target; its `max` sitting *inside* an expectation to be
estimated from samples is the origin of overestimation bias
(`value_based_deep_rl/03`).
