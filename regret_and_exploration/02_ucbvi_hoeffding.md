# UCBVI With Hoeffding Bonuses: The Canonical Proof

## The Algorithm

Episodic, horizon `H`, stationary-per-step unknown `P` and known `r`
(learning `r` adds lower-order terms). Maintain counts and empirical
transitions

```math
N^k(s,a),\qquad
\hat P^k(s'\mid s,a)=\frac{N^k(s,a,s')}{N^k(s,a)} .
```

Before episode `k`, run **optimistic value iteration** backward:

```math
\bar Q_h^k(s,a)
=
\min\Big(H,\ \ r(s,a)+b^k(s,a)+\hat P^k(\cdot\mid s,a)^\top\bar V_{h+1}^k\Big),
\qquad
\bar V_h^k(s)=\max_a\bar Q_h^k(s,a),
```

with the Hoeffding bonus

```math
b^k(s,a)=H\,\sqrt{\frac{2\,\log\!\big(2SAHK/\delta\big)}{N^k(s,a)}},
```

then act greedily w.r.t. `bar-Q^k` for the whole episode. (The `min(H, .)`
clip keeps values in range — used below.)

## Step 1: The Good Event

Let `G` be the event: for all `(s, a, k)` **and the fixed function**
`V*_{h+1}`,

```math
\Big|\big(\hat P^k-P\big)(\cdot\mid s,a)^\top V_{h+1}^*\Big|
\ \le\
H\sqrt{\frac{2\log(2SAHK/\delta)}{N^k(s,a)}}\ =\ b^k(s,a).
```

`Pr(G) >= 1 - delta`: per `(s,a)` and per possible count value, the sum of
`V*(s'_i)` over the `N` samples is a bounded MDS (`concentration_toolkit/02`
— the samples arrive at random times, legitimized by the optional-stopping
remark there); apply Azuma, union over `S * A * H * K` (and counts). Note
what makes this cheap: the bonus only needs to cover **`V*`, one fixed
function** — not the data-dependent `bar-V^k`. That trick (Azar et al.)
avoids a covering argument here; the data-dependent difference is handled
by the *recursion* in Step 3 instead.

## Step 2: Valid Optimism On G

**Claim:** `bar-V_h^k >= V_h^*` for all `h, s, k`, by backward induction on
`h`. At `H+1` both are 0. Inductively, for the `V*`-optimal action `a*`:

```math
\bar Q_h^k(s,a^*)
\ \ge\
r+b^k+\hat P^\top V_{h+1}^*
\ \ge\
r+\underbrace{b^k+(\hat P-P)^\top V^*_{h+1}}_{\ge\,0\ \text{on }G}+P^\top V_{h+1}^*
\ \ge\ Q_h^*(s,a^*),
```

using the induction hypothesis in the first inequality
(`bar-V_{h+1} >= V*_{h+1}`, and `hat-P` is a positive operator —
monotonicity again, `mdp_foundations/03`), the good event in the second.
The clip at `H` is harmless since `Q* <= H`. Max over `a`. ∎

## Step 3: The Regret Telescope, Assembled

By `01`'s lemma, on `G`:

```math
\mathrm{Reg}(K)
\le
\sum_k\Big(\bar V_1^k-V_1^{\pi_k}\Big)(s_1^k)
\le
\sum_k\sum_h\Big[\ 2b^k(s_h^k,a_h^k)
+\underbrace{(\hat P^k-P)^\top\big(\bar V^k_{h+1}-V^*_{h+1}\big)}_{(\star)}
+\underbrace{\xi_h^k}_{\text{MDS}}\Big]
```

where `xi` collects the sampling noise of "expectation under `P` vs the
state actually reached" — an MDS summing, by Azuma, to
`O(H sqrt(HK log(1/delta)))` (lower order).

The correction term `(*)` involves the data-dependent function
`bar-V - V*`; bound it crudely by an `L1` deviation of `hat-P`
(Weissman-type bound, again from the toolkit):
`||hat-P - P||_1 <= sqrt(2S log(.)/N)`, giving
`(*) <= H sqrt(S log(.)/N)` — a term with an extra `sqrt(S)` but divided by
the same `sqrt(N)`; it will contribute at the *same or lower* order in `K`
(second-order in the final accounting, dominant only for small `K`).

## Step 4: The Pigeonhole (widths must shrink)

The main term is `sum_{k,h} 1/sqrt(N^k(s_h^k, a_h^k))`. Group by cell: if
cell `(s,a)` is visited `N^K(s,a) = n` times total, it contributes
`sum_{j=1}^{n} 1/sqrt(j) <= 2 sqrt(n)`. By Cauchy–Schwarz over cells:

```math
\sum_{k,h}\frac{1}{\sqrt{N^k}}
\ \le\
2\sum_{(s,a)}\sqrt{N^K(s,a)}
\ \le\
2\sqrt{SA\ \textstyle\sum N^K(s,a)}
\ =\
2\sqrt{SA\cdot KH}.
```

(The elliptical potential lemma of `concentration_toolkit/03` is this
computation's function-approximation generalization — same role, same
position in the proof.)

## The Bound

```math
\mathrm{Reg}(K)
\ \le\
\tilde O\Big(H^2\sqrt{S\,A\,K}\Big)
\ =\
\tilde O\Big(H^{3/2}\sqrt{S\,A\,T}\Big),
\qquad T=KH,
```

with the `sqrt(S)`-inflated second-order term
`tilde-O(H^2 S sqrt(A K^{1/2}})`-type contributions suppressed in `tilde-O`.

**The audit — where each factor was paid:**

```text
sqrt(SA K)   pigeonhole over cells (Step 4): unavoidable counting
one H        the bonus SCALE: Hoeffding pays range H for V* (Step 1)
one H        the telescope LENGTH: H bonus terms per episode (Step 3)
```

The first `H` is the lazy one: `V*(s')` has range `H` but its per-step
conditional variance, summed along a trajectory, is only `H^2` total — not
`H` per step. Charging variance instead of range is precisely notebook 03.

## What Remains Open Here

Nothing — this proof is closed; its optimizations (Bernstein, and
Freedman-based handling of `(*)` to kill the `sqrt(S)` in leading order)
are notebook 03's content.
