# Doubly Robust Off-Policy Evaluation

## The Estimator

Combine a (possibly wrong) model `hat-q` with (possibly wrong) importance
weights `hat-rho`, so that each corrects the other. Bandit case first
(`H = 1`, action `a ~ pi_b`, target `pi`):

```math
\hat J^{\mathrm{DR}}
=
\underbrace{\mathbb{E}_{a\sim\pi}\big[\hat q(a)\big]}_{\text{model baseline}}
+\underbrace{\hat\rho(a)\,\big(r-\hat q(a)\big)}_{\text{IS-corrected residual}},
\qquad
\hat\rho(a)=\frac{\pi(a)}{\hat\pi_b(a)} .
```

## Double Robustness, Proved

**Proposition.** `E[J-hat-DR] = J(pi)` if **either** `hat-q = q` (model
correct, weights arbitrary) **or** `hat-pi_b = pi_b` (weights correct,
model arbitrary).

*Proof.* Write the estimator's expectation under the truth
(`a ~ pi_b`, `E[r|a] = q(a)`):

```math
\mathbb{E}[\hat J]
=
\mathbb{E}_{\pi}[\hat q]
+\mathbb{E}_{a\sim\pi_b}\Big[\hat\rho(a)\big(q(a)-\hat q(a)\big)\Big].
```

If the weights are correct, `E_{pi_b}[rho (q - q-hat)] = E_pi[q - q-hat]`,
and the display telescopes to `E_pi[q] = J`. If instead the model is
correct, the second term is identically zero and the first is `J`. ∎

More precisely the bias is a **product of the two errors**:

```math
\mathrm{Bias}
=
\mathbb{E}_{a\sim\pi_b}\Big[\big(\hat\rho(a)-\rho(a)\big)\big(q(a)-\hat q(a)\big)\Big]
```

— second-order. This product-of-errors structure is the estimator's entire
identity, and the reason it tolerates machine-learned nuisances: each
nuisance need only be `o(n^{-1/4})` for the product to be `o(n^{-1/2})`,
below the CLT noise floor (the Neyman-orthogonality / double-ML principle,
in its RL instance).

## The Sequential (Per-Step) DR Estimator

For horizon `H` (Jiang–Li 2016; Thomas–Brunskill 2016), recurse backward —
DR at every step, with the model supplying baselines `hat-q_h, hat-v_h`:

```math
\hat J^{\mathrm{DR}}_{H+1}=0,
\qquad
\hat J^{\mathrm{DR}}_{h}
=
\hat v_h(s_h)
+\rho_h\Big(r_h+\gamma\,\hat J^{\mathrm{DR}}_{h+1}-\hat q_h(s_h,a_h)\Big),
```

with per-step ratios `rho_h = pi(a_h|s_h)/pi_b(a_h|s_h)`. Unbiasedness
under correct weights: telescoping + tower property, exactly the
`G - v = sum gamma^k delta_k` identity (`policy_gradient/05`) with IS
ratios attached — the third appearance of the telescope.

## Variance, And The Baseline Reading

For the bandit case with correct weights:

```math
\mathrm{Var}\big[\hat J^{\mathrm{DR}}\big]
=
\mathrm{Var}_{a\sim\pi_b}\Big[\rho(a)\,\big(q(a)-\hat q(a)\big)\Big]
+\mathbb{E}\big[\rho^2\,\mathrm{Var}(r\mid a)\big]
+\ \ldots
```

— the model enters only through the residual `q - hat-q`: **DR is IS with
a control variate** (`policy_gradient/02`'s zero-mean subtraction, in
estimation clothing), and a good model shrinks the dominant variance term
without touching bias. Worked 2-step example of the sequential form:
per-step residual variances are weighted by *partial* products
`rho_1..rho_h` — each step's noise pays only the mismatch accumulated so
far, the per-decision principle (`stochastic_approximation/02`) inherited
automatically.

## The Efficiency Bound

**(Jiang–Li 2016, semiparametric).** For discrete-time OPE, the asymptotic
variance of any regular unbiased estimator is at least

```math
\mathrm{V}^*
=
\sum_{h=1}^{H}\ \mathbb{E}\Big[\ \rho_{1:h}^2\ \mathrm{Var}\big(r_h+\gamma\,\hat J_{h+1}\ \big|\ s_h,a_h\big)\Big]
\quad\text{(evaluated at the true nuisances)},
```

and sequential DR **attains** it when both nuisances are consistent. So
the estimator hierarchy closes:

```text
IS          unbiased, worst-case variance (no model information used)
WIS         biased, bounded variance
DR          unbiased if either nuisance correct; EFFICIENT if both
MIS/DICE    changes the reweighted OBJECT (occupancies) — notebook 03 —
            attacking the exp(H) inside rho_{1:h} itself
```

DR is optimal *given trajectory-space weighting*; it does not repair
notebook 01's exponential overlap problem (the `rho_{1:h}^2` in `V*` says
the bound itself explodes with mismatch). The two fixes are orthogonal and
composable — DR versions of marginalized estimators exist and inherit both
properties.

## What Remains Open

Finite-sample (non-asymptotic) efficiency theory; DR with *estimated
behavior policy* in adaptive-logging regimes (weights become dependent);
and robust variants when neither nuisance is consistent but both are
"close" — the product-bias bound is the tool, the sharp constants are not
settled.
