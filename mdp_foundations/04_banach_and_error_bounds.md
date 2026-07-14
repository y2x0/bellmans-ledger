# Banach Fixed Point Theorem And Error Bounds

## The Theorem

Let `(X, d)` be a complete metric space and `F : X -> X` a `gamma`-contraction,
`d(Fx, Fy) <= gamma d(x, y)` with `gamma in [0,1)`. Then:

```text
1. F has exactly one fixed point x*.
2. From any x_0, the iterates x_{k+1} = F x_k converge to x*.
3. The convergence is geometric with explicit constants.
```

*Proof.* **Cauchy.** `d(x_{k+1}, x_k) <= gamma^k d(x_1, x_0)`, so for `m > k`
the triangle inequality and the geometric series give

```math
d(x_m,x_k)\le\sum_{j=k}^{m-1}\gamma^{j}\,d(x_1,x_0)\le\frac{\gamma^{k}}{1-\gamma}\,d(x_1,x_0)\ \xrightarrow{k\to\infty}\ 0 .
```

Completeness gives a limit `x*`. **Fixed.** Contractions are continuous, so
`F x* = F(lim x_k) = lim x_{k+1} = x*`. **Unique.** If `Fx = x`, `Fy = y`:
`d(x,y) = d(Fx, Fy) <= gamma d(x,y)` forces `d(x,y) = 0`. ∎

`(B(S), ||.||_inf)` is complete (uniform limits of bounded functions are
bounded), so the theorem applies verbatim to `T^pi` and `T` from notebook 03.

## The Two Error Bounds That Matter

**A priori (rate):**

```math
\|v_k-v_*\|_\infty\ \le\ \gamma^k\,\|v_0-v_*\|_\infty .
```

To reach accuracy `eps` from `||v_0 - v*|| <= R_max/(1-gamma)`:

```math
k\ \ge\ \frac{\log\frac{R_{\max}}{\varepsilon(1-\gamma)}}{\log(1/\gamma)}
\ \approx\
\frac{1}{1-\gamma}\log\frac{R_{\max}}{\varepsilon(1-\gamma)} .
```

Value iteration is cheap per sweep (`O(|S|^2 |A|)`) but pays the effective
horizon in sweep count. As `gamma -> 1`, both the target and the algorithm
degrade together.

**A posteriori (stopping rule).** From the triangle inequality
`||v_k - v*|| <= ||v_k - Tv_k|| + ||Tv_k - v*||` and contraction on the second
term (`Tv_k` vs `Tv_* = v_*`):

```math
\|v_k-v_*\|_\infty\ \le\ \frac{\|\mathcal{T}v_k-v_k\|_\infty}{1-\gamma}.
```

The **Bellman residual** `||Tv - v||` is computable without knowing `v*`;
dividing by `1 - gamma` converts it into a true error bound. This quantity is
the exact ancestor of every TD error: `delta_t` is a one-sample estimate of a
Bellman residual component.

## From Value Error To Policy Error

Converging in value is not yet acting well. Let `pi` be greedy with respect
to an approximate `v`. Then:

```math
\|v_\pi - v_*\|_\infty\ \le\ \frac{2\gamma}{1-\gamma}\,\|v-v_*\|_\infty .
```

*Proof.* Since `pi` is greedy for `v`, `T^pi v = T v`. Then

```math
\|v_\pi-v_*\|
\le
\|\mathcal{T}^\pi v_\pi-\mathcal{T}^\pi v\|
+\|\mathcal{T}^\pi v-\mathcal{T}v_*\|
\le
\gamma\|v_\pi-v_*\| + \gamma\|v_\pi - v\|\ \text{...}
```

more carefully:
`||v_pi - v*|| <= ||T^pi v_pi - T^pi v|| + ||T v - T v_*||
<= gamma ||v_pi - v|| + gamma ||v - v_*||
<= gamma ||v_pi - v_*|| + 2 gamma ||v - v_*||`,
and rearranging gives the claim. ∎

Two consequences worth internalizing:

```text
1. The 1/(1-gamma) amplification: small value error can mean large policy
   suboptimality near gamma = 1. Function-approximation value errors are
   *paid* through this lens.

2. Finite MDPs have finitely many SD policies, so once
   ||v_k - v*|| < (min positive action-gap)/2 the greedy policy is exactly
   optimal — value iteration identifies pi* in finitely many sweeps even
   though v_k -> v* only in the limit.
```

## Why The Fixed Point Of T Is The Optimal Value

Banach gives `T` a unique fixed point `v`. Two inequalities close the loop
with `v* = sup_pi v_pi`:

**(i) `v >= v_pi` for every `pi`.** `T v = v >= T^pi v` (max over actions
dominates any average). Monotonicity of `T^pi` then gives
`v >= (T^pi)^k v -> v_pi`.

**(ii) `v` is attained.** Let `pi_g` be greedy for `v`. Then
`T^{pi_g} v = T v = v`, so `v` is the fixed point of `T^{pi_g}`, i.e.
`v = v_{pi_g}`.

Together: `v = v_* = max_pi v_pi`, and greedy(`v_*`) is an optimal stationary
deterministic policy. This completes the chain promised in the family README
and discharges the `MR -> SR -> SD` collapses of notebook 01. ∎

## What To Watch Later

Everything above lives in the sup-norm. Sampled algorithms average, and
averages live in `L2(mu)` for a state distribution `mu`. `P_pi` is
nonexpansive in `L2(mu)` **only** when `mu` is its stationary distribution —
the single fact from which both the Tsitsiklis–Van Roy convergence theorem
and Baird's divergence example fall out (`stochastic_approximation/04`, `06`).
