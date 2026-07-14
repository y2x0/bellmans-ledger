# The Lower Bound Technique

## The Logic Of All Lower Bounds

Every sample-complexity or regret lower bound in this repository has the
same skeleton:

```text
1. construct TWO (or a family of) environments that require DIFFERENT
   behavior to succeed;
2. make them STATISTICALLY CLOSE: small KL divergence between the data
   distributions they induce;
3. conclude: any algorithm behaves similarly on both (it cannot tell them
   apart), so it fails on at least one.
```

Steps 1 and 3 are combinatorics; step 2 is this file.

## Le Cam's Two-Point Method

**Lemma.** For any event `A` and probability measures `P, Q`:

```math
P(A)+Q(A^c)\ \ge\ 1-\mathrm{TV}(P,Q)\ \ge\ \tfrac12\,e^{-\mathrm{KL}(P\|Q)} .
```

The first inequality is immediate from the definition of total variation
(`TV = sup_A |P(A) - Q(A)|`). The second is Bretagnolle–Huber:

**Bretagnolle–Huber.** `1 - TV(P, Q) >= (1/2) exp(-KL(P || Q))`.

*Proof.* `1 - TV = ∫ min(dP, dQ)`. Using
`min(a,b) >= a b / (a + b) >= (1/2) min(a,b)`... the clean route: by
Cauchy–Schwarz,

```math
\Big(\int\min(dP,dQ)\Big)\Big(\int\max(dP,dQ)\Big)\ \ge\ \Big(\int\sqrt{dP\,dQ}\Big)^2,
```

and `∫max = 2 - ∫min <= 2`, so
`∫min >= (1/2)(∫sqrt(dP dQ))^2`. The Bhattacharyya term is controlled by
KL via Jensen:

```math
\int\sqrt{dP\,dQ}=\mathbb{E}_P\Big[e^{\frac12\log\frac{dQ}{dP}}\Big]
\ \ge\ e^{\frac12\mathbb{E}_P\log\frac{dQ}{dP}}=e^{-\frac12\mathrm{KL}(P\|Q)} .
\qquad\blacksquare
```

**Usage pattern.** Let `A` = "the algorithm outputs the answer correct for
`P`". If correctness for `P` and for `Q` are mutually exclusive, then
`max(P(error), Q(error)) >= (1/4) e^{-KL}`: to be reliably correct
(`error <= delta`) the algorithm's data must carry
`KL >= log(1/4delta)` of divergence — and KL is what samples buy, linearly:

```math
\mathrm{KL}\big(P^{\otimes n}\,\|\,Q^{\otimes n}\big)=n\,\mathrm{KL}(P\|Q)
\quad\Longrightarrow\quad
n\ \ge\ \frac{\log(1/4\delta)}{\mathrm{KL}(P\|Q)} .
```

For two coins with biases `1/2` and `1/2 + eps`,
`KL = O(eps^2)`, giving the universal `n = Omega(eps^{-2} log(1/delta))` —
the atom from which `finite_time_td_q/05` and
`regret_and_exploration/04` build their `Omega((1-gamma)^{-3})` and
`Omega(sqrt(H^3 SAT))` by placing such coins inside MDPs so that each coin
flip costs `~ 1/(1-gamma)` or `~ H` of value.

## The Divergence Decomposition (adaptive data)

RL data is collected adaptively, so `P` and `Q` above are measures over
whole interaction histories. The chain rule for KL over the history
factorization gives — because the *algorithm's* conditional action choices
are identical under both environments (same algorithm!) and cancel:

```math
\mathrm{KL}\big(P^{\text{hist}}\,\|\,Q^{\text{hist}}\big)
=
\sum_{(s,a)}\mathbb{E}_P\big[N(s,a)\big]\ \mathrm{KL}\Big(P(\cdot\mid s,a)\,\big\|\,Q(\cdot\mid s,a)\Big).
```

*Proof sketch.* Write the history likelihood ratio; policy terms
`pi(a_t | h_t)` appear in both numerator and denominator and cancel; what
remains is a sum over time of per-transition log-ratios; take expectations
and group by `(s,a)`. ∎

**This identity is the whole game.** It says: the information an adaptive
algorithm gathers about the difference between two environments is
*the expected visit count of the cells where they differ, times the per-visit
KL*. Lower bound constructions therefore hide the difference in cells the
algorithm cannot afford to visit often — and the bound
`E_P[N(s,a)]` is simultaneously the quantity regret analysis controls from
above. Upper and lower bounds meet at this display, which is why the tight
results (`finite_time_td_q/04–05`, `regret_and_exploration/03–04`) are
possible at all.

## Fano, For Many Alternatives

When `M` alternatives are needed (e.g. one per state-action cell):

```math
\Pr\{\text{identify the true environment}\}
\ \le\
\frac{\max_{i\neq j}\mathrm{KL}(P_i\|P_j)\cdot(\text{samples})+\log 2}{\log M},
```

turning `log M` into an extra dimension factor. This is how `S` and `A`
enter multiplicatively in the MDP lower bounds: `M ~ SA` well-separated
environments, each pair close in KL.

## The Tilting Connection

Notebook 01 proved upper bounds by *bounding* mgfs — variance control
uniform over exponential tilts. Lower bounds *choose* a tilt: the
alternative `Q` is a small exponential tilting of `P` in the payoff-relevant
direction. Chernoff exponents and KL divergences are Legendre duals of each
other; concentration and information lower bounds are the two faces of the
same convex-analytic object. (The same Legendre duality resurfaces,
unreasonably, in `regularized_mdps_and_duality/01` — log-sum-exp and
entropy — and in `rlhf_mathematics/02` — Donsker–Varadhan.)

## What Remains Open

Nothing at this level of generality; the open problems are always in step 1
— constructing hard instances that respect additional structure (e.g. lower
bounds against *realizable linear* function classes,
`linear_mdps_and_completeness/05`, where the geometry of the construction is
the entire difficulty).
