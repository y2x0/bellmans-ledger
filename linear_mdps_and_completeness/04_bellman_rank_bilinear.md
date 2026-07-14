# Bellman Rank And Bilinear Classes

## The Question

Notebooks 01–02 need the environment low-rank; notebook 03 shows
realizability alone is hopeless. Is there a *general structural parameter*
of the (MDP, function class) **pair** whose smallness makes RL
polynomial — covering linear MDPs, block MDPs, low-rank POMDP-ish models,
and more, in one theorem? Bellman rank (Jiang et al. 2017) was the first
such parameter; bilinear classes (Du et al. 2021) the consolidation.

## Average Bellman Error

For hypotheses `f, g` in a class `F` (each `f` inducing greedy policy
`pi_f` and value `V_f`), define the **average Bellman error of `g` under
`f`'s roll-in**:

```math
\mathcal{E}_h(f,g)
=
\mathbb{E}_{s_h\sim\pi_f}\Big[\ g_h(s_h,\pi_g(s_h))-r_h-g_{h+1}(s_{h+1},\pi_g(s_{h+1}))\ \Big]
```

— roll in with one hypothesis, test the *self-consistency* of another.
Two elementary but decisive facts:

```text
1. E_h(f, f*) = 0 for the true optimal f* (it satisfies Bellman
   identically under ANY roll-in);
2. sum_h E_h(f, f) = V_f(s_1) - V^{pi_f}(s_1): a hypothesis's total
   self-inconsistency under its OWN roll-in equals exactly its optimism
   gap — the amount by which it lies about its own policy's value.
   (Proof: the same telescoping-through-TD-errors identity as the
   performance difference lemma, policy_gradient/03.)
```

Fact 2 means: an over-optimistic hypothesis **must** expose a nonzero
average Bellman error along its own trajectories. Errors cannot hide —
*if* the matrix of errors has structure:

## Bellman Rank

`F` has Bellman rank `d` if, for each `h`, the matrix

```math
\big[\mathcal{E}_h(f,g)\big]_{f,g\in\mathcal{F}}
\quad\text{has rank}\ \le d:
\qquad
\mathcal{E}_h(f,g)=\big\langle\ \nu_h(f),\ \xi_h(g)\ \big\rangle,
\quad \nu,\xi\in\mathbb{R}^d .
```

Roll-in effects and hypothesis-error effects factor through `d`
dimensions. Verified instances:

```text
linear MDP (01):     nu = state distribution's feature expectation,
                     xi = the hypothesis's Bellman-residual vector; rank d
block MDP:           rank = number of latent blocks
tabular:             rank <= |S|
reactive PSR-ish:    rank = PSR core dimension
```

## The Elimination Theorem

**Theorem (OLIVE, shape).** With Bellman rank `d`, the optimistic
elimination algorithm —

```text
repeat: choose the most optimistic surviving f; roll out pi_f;
        if its measured self-error (fact 2) is small: near-optimal, STOP;
        else: estimate E_h(f, .) for all survivors from the same roll-in
              data, and ELIMINATE every g with |E_h(f,g)| large
```

— terminates after `O(dH log(...))` elimination rounds, with total samples
`poly(d, H, log|F|, 1/eps)`.

*The volumetric core.* Each elimination round produces a roll-in vector
`nu_h(f_k)` such that the surviving set's error vectors `xi` must satisfy
`<nu_k, xi> ~ 0` for all future survivors, while `f_k`'s own error had
`<nu_k, xi(f_k)>` large. So `nu_k` is (quantitatively) linearly independent
of `nu_1..nu_{k-1}` — you can pick at most `~ d` such directions before
the ellipsoid of consistent `xi`s collapses; the elliptical-potential/
eluder counting of `concentration_toolkit/03` /
`regret_and_exploration/07` again, now in Bellman-error space. ∎ (shape)

**What this achieves conceptually:** polynomial learning with NEITHER
completeness NOR environmental low rank as stated — the low-rank
requirement moved to the *error geometry* of the pair `(MDP, F)`. Fact 2
supplies the exploration principle (optimism is self-incriminating), the
rank supplies the budget.

**The honest cost:** OLIVE is computationally intractable (elimination
over an implicit class) — it is an information-theoretic possibility
proof, not an algorithm one runs.

## Bilinear Classes (the consolidation)

Du et al. 2021 abstract the pattern: a **bilinear class** requires only
that some *discrepancy functional* `l_f(g)` (Bellman error, model error,
...) factor as `<W_h(g) - W_h(f*), X_h(f)>` with estimable surrogates.
This absorbs Bellman rank, linear MDPs, linear mixture models, kernelized
variants, and some POMDPs into one `poly(d, H)` theorem of the same
elimination shape. The decision-estimation coefficient (Foster et al.)
then reframes the whole enterprise minimax-style — cited as the frontier
in `regret_and_exploration/07`.

## Position In The Family

```text
01: environment low-rank        -> completeness for free, efficient algs
03: realizability alone         -> exponential; loss not even estimable
04: low-rank ERROR geometry     -> poly SAMPLES, intractable computation
```

The gap between 04's statistical possibility and 01–02's computational
tractability is the field's standing trade: no known parameter is
simultaneously general, sample-sufficient, and algorithmically usable.

## What Remains Open

Exactly that trade; plus lower bounds showing OLIVE's `d H` rounds are
necessary; plus any *practical* algorithm certified for a bilinear class
beyond the linear-MDP cell.
