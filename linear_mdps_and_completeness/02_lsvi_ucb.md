# LSVI-UCB

## The Algorithm

Episodic linear MDP, horizon `H`. Before episode `k`, compute backward for
`h = H..1` a ridge regression against the *optimistic* next-step value:

```math
w_h^k=\big(\Lambda_h^k\big)^{-1}\sum_{\tau<k}\phi_h^\tau\,\Big[r_h^\tau+\bar V_{h+1}^k(s_{h+1}^\tau)\Big],
\qquad
\Lambda_h^k=\lambda I+\sum_{\tau<k}\phi_h^\tau(\phi_h^\tau)^\top,
```

```math
\bar Q_h^k(s,a)=\min\Big(H,\ \ \phi(s,a)^\top w_h^k+\beta\,\big\|\phi(s,a)\big\|_{(\Lambda_h^k)^{-1}}\Big),
\qquad
\bar V_h^k(s)=\max_a\bar Q_h^k(s,a),
```

then act greedily. Structure-for-structure identical to UCBVI
(`regret_and_exploration/02`) with:

```text
counts N(s,a)             ->  Gram matrix Lambda
1/sqrt(N) bonus           ->  ||phi||_{Lambda^{-1}} bonus
                              (identical for one-hot phi)
empirical model P-hat     ->  ridge regression against the target
                              r + V-bar(s')
```

## Why Regression Against V-bar Is Legitimate

The regression's implicit claim: `T V-bar_{h+1}` is linear in `phi`. That
is **closure property 2** of notebook 01, applied to the data-dependent,
clipped, bonus-bearing `V-bar` — the exact place where "linear MDP" (an
environment property) is consumed, and where merely assuming "Q* is linear"
would already fail (the backup of the optimistic value is not `Q*`).

## The Concentration Step

Error decomposition at `(h, k)`:

```math
\phi^\top w_h^k-\big(\mathcal{T}\bar V^k_{h+1}\big)(s,a)
=
\underbrace{\phi^\top(\Lambda^{-1})\textstyle\sum_\tau\phi^\tau\,\eta^\tau}_{\text{noise}}
\ -\ \underbrace{\lambda\,\phi^\top\Lambda^{-1}w^{\mathcal{T}\bar V}}_{\text{ridge bias}},
```

`eta^tau = [r + V-bar(s')] - E[.|s,a]` conditionally centered. Two tools
finish it:

**1. Self-normalized bound** (`concentration_toolkit/03`): the noise term is
`<= ||phi||_{Lambda^{-1}} * ||sum phi eta||_{Lambda^{-1}}`, and the method
of mixtures bounds the second factor by
`~ H sqrt(d log(K/delta) + log N_eps)`.

**2. Covering** (`concentration_toolkit/04`): the mixture bound needs `eta`'s
conditional centering **given the regression target function** — but
`V-bar_{h+1}^k` is built from the same data. Fix: uniform bound over the
class of possible optimistic values

```math
\mathcal{V}=\Big\{s\mapsto\min\big(H,\max_a\ \phi^\top w+\beta\|\phi\|_{\Lambda^{-1}}\big):
\|w\|\le W,\ \Lambda\succeq\lambda I\Big\},
```

parameterized by `(w, Lambda)` — dimension `O(d^2)`, so
`log N = O(d^2 log(.))`. Total:

```math
\beta=\tilde O\big(d\,H\big)
\qquad\text{certifies the good event.}
```

(The `d` in `beta` — one `sqrt(d)` from the mixture determinant, one
`sqrt(d^2) = d`-ish from the cover, combining to `tilde-O(dH)` — is where
this proof genuinely pays for function approximation; UCBVI's `beta` was
`H` with no dimension.)

## Optimism And The Telescope

Backward induction gives `bar-V >= V*` on the good event exactly as in
`regret_and_exploration/02` Step 2 (monotonicity + the bonus covering the
regression error — closure property 2 again ensures the inductive backup
comparison is between two functions the analysis controls). The regret
telescope (ibid., Step 3) leaves

```math
\mathrm{Reg}(K)\ \lesssim\ \beta\ \sum_{k,h}\big\|\phi_h^k\big\|_{(\Lambda_h^k)^{-1}}\ +\ \text{MDS terms}.
```

## The Potential Step

Cauchy–Schwarz + the **elliptical potential lemma**
(`concentration_toolkit/03`, proved there):

```math
\sum_{k}\big\|\phi_h^k\big\|_{(\Lambda_h^k)^{-1}}
\le
\sqrt{K\ \textstyle\sum_k\|\phi_h^k\|^2_{(\Lambda^k_h)^{-1}}}
\le
\sqrt{K\cdot 2d\log\big(1+\tfrac{K}{\lambda d}\big)} .
```

## The Bound

```math
\boxed{\ \mathrm{Reg}(K)\ =\ \tilde O\Big(\sqrt{d^3\,H^4\,K}\Big)\ }
\qquad
\big(=\tilde O(\sqrt{d^3H^3T})\big),
```

**no `S`, no `A`, anywhere** — the entire point. Audit of the exponents:

```text
d^{1/2}   elliptical potential (the "number of directions to learn")
d^1       beta: mixture + covering of the optimistic class
H^2       one H of bonus scale (Hoeffding-style — a Bernstein variant
          shaves it, as in regret_and_exploration/03), one H of telescope
```

Known improvements: `sqrt(d^2 H^3 T)`-type rates via Bernstein-weighted
ridge regression; the minimax lower bound is
`Omega(sqrt(d^2 H^2 T))`-ish — small gaps remain, unlike the closed tabular
chapter.

## What Remains Open

Closing the `d`-gap; computational efficiency of the `max_a` over
continuous actions; and — the family's real question — everything here
dies without closure property 2: notebook 03.
