# CQL As A Pessimistic Operator

## The Problem CQL Solves

Notebook 04's LCB needs `N(s,a)` — meaningless for neural classes. CQL
(Conservative Q-Learning, Kumar et al. 2020) builds the pessimism into the
**loss**, with no counts: penalize the Q-values of actions the *policy
would like* and reward those the *data actually contains*.

## The Objective

Standard fitted-Q loss plus a conservative penalty with weight `alpha`:

```math
\min_{Q}\quad
\alpha\ \Big(\mathbb{E}_{s\sim\mathcal{D},\,a\sim\nu(\cdot\mid s)}\big[Q(s,a)\big]
-\mathbb{E}_{(s,a)\sim\mathcal{D}}\big[Q(s,a)\big]\Big)
+\tfrac12\,\mathbb{E}_{\mathcal{D}}\Big[\big(Q-\hat{\mathcal{T}}Q_{\text{targ}}\big)^2\Big],
```

where `nu` is the action-proposal distribution — in the strongest variant,
the **adversarial** choice `nu = argmax` of the first term (with an entropy
regularizer, giving a closed form below).

## The Implicit Operator, Derived

Fix the target and ask what `Q` the penalized regression returns,
pointwise. Setting the functional derivative w.r.t. `Q(s,a)` to zero at a
data state `s`:

```math
0=\alpha\,\big(\nu(a\mid s)-\hat\pi_b(a\mid s)\big)+\hat\mu(s)\big(Q(s,a)-(\hat{\mathcal{T}}Q)(s,a)\big)\cdot\ldots
```

cleaning up (per-state, weights normalized):

```math
\boxed{\
Q(s,a)
=
(\hat{\mathcal{T}}Q_{\text{targ}})(s,a)
-\alpha\,\frac{\nu(a\mid s)-\hat\pi_b(a\mid s)}{\hat\pi_b(a\mid s)}\ }
```

— fitted-Q with a **per-action pessimism penalty proportional to how much
more the proposal distribution wants the action than the data supports
it.** Three readings:

```text
1. in-support actions (nu ~ pi_b):        penalty ~ 0  — data speaks;
2. OOD actions the policy covets          penalty ~ alpha * nu/pi_b — large
   (nu large, pi_b ~ 0):                  exactly where extrapolation error
                                          plus max-bias would live;
3. actions the data has but the policy    penalty NEGATIVE: mild optimism
   ignores:                               on well-supported alternatives.
```

The `1/N`-style count of notebook 04 has been replaced by the **density
ratio `nu/pi_b`** — pessimism weighted by distribution mismatch instead of
by visitation, which is exactly the right generalization when states never
repeat (compare: `C^pi` of notebook 04 is also a density ratio; CQL
localizes that global constant into the loss).

With the entropy-regularized adversarial `nu`, the first penalty term
evaluates in closed form (Legendre–Fenchel,
`regularized_mdps_and_duality/01`):

```math
\mathbb{E}_{a\sim\nu^*}[Q]\ \longrightarrow\ \log\sum_a e^{Q(s,a)},
\qquad\text{penalty}
=
\mathbb{E}_{\mathcal{D}}\Big[\log\textstyle\sum_a e^{Q(s,a)}-\mathbb{E}_{a\sim\hat\pi_b}Q(s,a)\Big],
```

— logsumexp-minus-data-mean, the form implemented in practice: a softmax
pessimism that needs no explicit behavior model.

## The Lower-Bound Guarantee

**Proposition (Kumar et al., shape).** For `alpha` large enough relative
to the regression error scale, the CQL iterate's implied value
under-estimates: for the learned policy's evaluation,

```math
\hat V(s)\ \le\ V^{\hat\pi}(s)\quad\forall s\ \text{w.h.p.}
```

— i.e. CQL restores **Move 1** of notebook 04's proof (self-claims are
under-estimates) without counts. What is *not* restored at the same
strength: Move 2's comparator-specific bound with `C^pi` — the neural-class
analogue holds only under additional realizability/completeness-type
conditions (`linear_mdps_and_completeness/03, 05` explain why some such
condition is unavoidable, not a proof artifact).

## Position Among The Fixes

```text
                    pessimism carrier            needs
LCB-VI (04)         count bonuses 1/sqrt(N)      tabular/linear counts
CQL (this file)     density-ratio penalty in     alpha tuning; behavior
                    the loss                     samples only
IQL / one-step      avoid OOD argmax entirely    weaker: no explicit
                    (expectile regression)       penalty, no OOD queries
BCQ/TD3+BC          constrain pi toward pi_b     behavior cloning accuracy
```

All are implementations of one sentence — *never let the argmax consult
values the data cannot falsify* — which is also
`value_based_deep_rl/03`'s closing lesson ("insert pessimism exactly at
the argmax"), now as the organizing principle of an entire field.

## What Remains Open

Principled `alpha` (current practice: sweep it); CQL's interaction with
distributional heads and ensembles (empirically strong, theoretically
unpriced); and the general neural-class comparator bound — the honest
statement is that CQL is notebook 04's theorem *minus* the parts that
required counting, plus assumptions research has not finished pricing.
