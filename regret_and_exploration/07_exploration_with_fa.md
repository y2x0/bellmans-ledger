# Exploration With Function Approximation

## The Question

Notebooks 01–06 count states. With function approximation the state space
is unbounded; what replaces `S A` in the width-sum accounting? The answer,
where one exists, is always a **dimension of the function class** — and the
sharpest general-purpose such dimension for optimism is the eluder
dimension.

## Eluder Dimension

Fix a class `F` of functions `X -> R` and a scale `eps`. Say `x` is
**`eps`-independent** of `x_1..x_n` w.r.t. `F` if two functions in `F` can
agree on all `x_i` (within `eps`, in the aggregate sense
`sqrt(sum (f-g)(x_i)^2) <= eps`) while differing by more than `eps` at `x`.

```math
\dim_E(\mathcal{F},\varepsilon)
=
\text{length of the longest sequence in which EVERY point is }
\varepsilon\text{-independent of its predecessors.}
```

It measures how long the class can keep "surprising" you at new points
after being pinned down at old ones — a sequential, adversarially-ordered
strengthening of VC/metric dimensions.

```text
linear:        dim_E = O(d log(1/eps))
generalized    dim_E = O(d r^2 log(1/eps)),  r = ratio of link-function
linear (GLM):  slopes
quadratic:     O(d^2 log(1/eps))
1-Lipschitz    exponential in the domain dimension — the class where
functions:     everything fails
```

## The Eluder Regret Lemma

The width-sum machine of 02/`concentration_toolkit/03`, generalized:

**Lemma.** Let width
`w_t(x) = sup_{f,g in F_t} (f - g)(x)` where `F_t` is the version space
(functions consistent with data up to the confidence radius). Then the
number of times the width at the queried point can exceed `eps` is at most

```math
\#\{t\le T:\ w_t(x_t)>\varepsilon\}
\ \le\
\Big(\frac{4\beta_T}{\varepsilon^2}+1\Big)\,\dim_E(\mathcal{F},\varepsilon),
```

(`beta_T` the confidence radius), and integrating over scales:

```math
\sum_{t\le T}w_t(x_t)
\ =\
\tilde O\Big(\sqrt{\dim_E(\mathcal{F})\ \beta_T\ T}\Big).
```

*Proof idea.* If `w_t(x_t) > eps`, two version-space functions disagree by
`> eps` at `x_t`; because both fit past data within `beta`, `x_t` can have
been "pinned" (`eps`-dependent on disjoint prior witness sets) at most
`~ beta/eps^2` times; each pinning consumes one step of an eluder
sequence, of which there are at most `dim_E` per chain — pigeonhole over
chains. ∎ The elliptical potential lemma is exactly this argument
specialized to linear `F` (each display in `concentration_toolkit/03` has
its counterpart here).

**Consequence.** Optimistic least-squares algorithms (bandits: Russo–Van
Roy; RL: "LSVI with general F" under completeness assumptions) achieve

```math
\mathrm{Reg}(T)=\tilde O\Big(H\sqrt{\dim_E(\mathcal{F})\,\log N(\mathcal{F})\ T}\Big)
```

— `dim_E` replacing `SA`, covering numbers (`concentration_toolkit/04`)
replacing the union bound. For linear classes this recovers
LSVI-UCB (`linear_mdps_and_completeness/02`).

## The Honest State Of Deep-RL Exploration

```text
what the theory licenses          what practice runs
-------------------------------   ----------------------------------------
optimism with widths from a       epsilon-greedy (convicted exponential,
version space over F               04) or entropy bonuses (undirected)
dim_E small: linear/GLM-ish F     deep nets: dim_E effectively vacuous
posterior sampling with exact     bootstrapped ensembles / noisy nets
posteriors (05)                    (uncontrolled posterior approximations)
information pricing (06)          RND / pseudo-counts / curiosity —
                                   unpriced g_t proxies
```

The heuristics are not random: each is a computable shadow of a principled
quantity —

```math
\text{RND error, pseudo-count bonus}\ \sim\ \text{width }w_t(x);
\qquad
\text{ensemble disagreement}\ \sim\ \text{posterior spread};
```

what is missing is the *validity* half (the certified `V-bar >= V*` of 01)
and the *budget* half (the pigeonhole/eluder counting). Hard-exploration
benchmarks (Montezuma, the locks of 04) are precisely where the missing
halves bind.

## Load-Bearing Audit For The Whole Family

```text
tabular counting  ->  eluder dimension       (this file)
Azuma per cell    ->  self-normalized bound  (concentration_toolkit/03)
union over cells  ->  covering number        (concentration_toolkit/04)
optimism validity ->  completeness-type assumptions on F
                      (linear_mdps_and_completeness/03 — and this is the
                      hinge: without completeness, even VALID optimism can
                      have version spaces that never shrink)
```

## What Remains Open

Almost everything that matters: a dimension that is simultaneously (i)
small for neural classes on natural tasks, (ii) sufficient for optimism,
(iii) computable. Candidate frameworks (bilinear classes, decision-estimation
coefficient) are the current frontier — the DEC in particular reframes
01–06's whole dichotomy as one minimax exchange rate between regret and
estimation, and is the closest thing to a final answer this family can
currently cite.
