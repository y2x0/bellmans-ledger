# Gain, Bias, And The Poisson Equation

## The Two Numbers That Replace One Value

Fix a policy; drop subscripts. The **gain** and **bias**:

```math
\rho=P^{*}r
\qquad\text{(per-step average, possibly state-dependent)},
```

```math
h
=
\lim_{T\to\infty}\ \mathbb{E}\Bigg[\sum_{t=0}^{T-1}\big(r(S_t)-\rho(S_t)\big)\Bigg]
\qquad\text{(total transient EXCESS reward)} .
```

`rho` is the cruising altitude; `h(s)` is the accumulated advantage of
*starting* at `s` before the chain forgets its start — all long-run
information that the gain erases. Two states with equal gain can have
very different biases (one passes through lucrative transients), and
average-reward *learning* is really bias learning: `rho` is one number
(unichain), `h` is the function.

## The Poisson Equation

**Theorem.** `(rho, h)` satisfy

```math
\rho+h\ =\ r+P\,h
\qquad\text{with}\qquad
P^{*}h=0,
```

and under unichain these determine `rho` uniquely and `h` uniquely (the
normalization `P* h = 0` fixing the additive constant; any solution pair
differs from `(rho, h)` by `h -> h + c 1`).

*Proof of existence.* Define the **deviation matrix**

```math
H=\big(I-P+P^{*}\big)^{-1}\big(I-P^{*}\big).
```

`(I - P + P*)` is invertible: on the steady eigenspace it acts as
identity (`P` and `P*` cancel... precisely: for `v = P* v`, `(I - P +
P*)v = v - Pv + v = v` using `P P* = P*`); on the transient complement,
`P` restricted there has spectral radius `< 1` (its eigenvalue-1
eigenspace was removed), so `I - P` is invertible there. Set `h = H r`.
Verify: multiplying out and using the `P*`-identities of notebook 01,

```math
(I-P)h=(I-P)Hr=(I-P^{*})r=r-\rho,
```

which is the Poisson equation rearranged; and `P* H = 0` gives the
normalization. ∎

*Proof of uniqueness (unichain).* If `(rho', h')` also solves it,
subtract: `(rho - rho') = (I - P)(h' - h)`; multiply by `P*`:
`P*(rho - rho') = 0`, and in the unichain case `rho, rho'` are constant
vectors, so `rho = rho'`. Then `(I - P)(h - h') = 0` means `h - h'` is
harmonic — constant on the recurrent class, hence (by the transient part's
determination from the recurrent part) a global constant, killed by the
normalization. ∎

## Why "Poisson"

The equation `(I - P)h = r - rho` is the discrete-Markov-chain analogue
of `Delta h = -(f - f-bar)` — the potential equation for a source with its
mean removed; `h` is the *potential* of the reward field over the chain's
geometry. The mean-removal is forced: `(I - P)` annihilates constants, so
`r - rho` must be orthogonal to the stationary distribution
(`mu^T(r - rho) = 0` — which is the definition of `rho`) for solvability:
Fredholm alternative, chain edition.

## The Average-Reward Bellman Equations

Control version (unichain), the pair that replaces `v = Tv`:

```math
\rho^{*}+h^{*}(s)
=
\max_a\ \Big[r(s,a)+\sum_{s'}P(s'\mid s,a)\,h^{*}(s')\Big],
```

whose solutions give: `rho*` = the optimal gain, and greedy w.r.t. `h*`
is gain-optimal. Note the structure: **`h` plays the role of `v`, and
`rho` enters as an unknown normalizing level** — one extra scalar
unknown, matching the one lost degree of freedom (`h` defined up to a
constant). Everything in `mdp_foundations/05` (improvement, policy
iteration) has an analogue with `(rho, h)` in place of `v`, with
multichain versions requiring the nested pair mentioned in notebook 01.

## Interpretations That Recur Downstream

```text
h as relative value:    h(s) - h(s') = long-run preference for starting
                        at s over s' — all that action selection needs
                        (differences kill the constant ambiguity);
h as the TD fixed pt:   the differential TD error
                        delta = r - rho-hat + h(S') - h(S) has mean zero
                        at the solution: notebook 06's algorithm;
h as Laurent term:      h is EXACTLY the O(1) coefficient of the
                        discounted value's expansion at gamma -> 1:
                        notebook 03 — the bridge back to the rest of the
                        repository.
```

## What Remains Open

Closed classical material in the tabular case; with function
approximation the pair `(rho, h)` needs a compatible normalization in the
feature space and the clean uniqueness story above degrades — part of why
average-reward FA theory lags discounted (06).
