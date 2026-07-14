# Bradley–Terry And What Preferences Identify

## The Model

Preference data: prompt `x`, two completions `(y_1, y_2)`, a human label
for which is better. The Bradley–Terry model posits a latent reward
`r(x, y)` with

```math
\Pr\{y_1\succ y_2\mid x\}
=
\sigma\big(r(x,y_1)-r(x,y_2)\big)
=
\frac{e^{r(x,y_1)}}{e^{r(x,y_1)}+e^{r(x,y_2)}},
\qquad
\sigma(z)=\frac{1}{1+e^{-z}} .
```

The reward model is fit by MLE on labeled pairs:

```math
\mathcal{L}(r)
=
-\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\big(r(x,y_w)-r(x,y_l)\big)\Big].
```

This is logistic regression on reward *differences* — convex in `r` when
`r` ranges over a linear class; with a neural `r`, convex in the output
layer given features.

## Identifiability: The Exact Statement

**Proposition.** The Bradley–Terry likelihood is invariant under
per-prompt shifts: for any function `c(x)`,

```math
r(x,y)\ \longmapsto\ r(x,y)+c(x)
```

changes no preference probability (the shift cancels in every
difference). Conversely, if two reward functions induce identical
preference distributions over all pairs (and the comparison graph is
connected per prompt), they differ by exactly such a shift.

*Proof.* Invariance is immediate. For the converse: equal probabilities
give equal differences `r(x,y) - r(x,y')` for all compared pairs;
connectedness propagates equality of differences to the whole completion
set; fixing any anchor completion per prompt pins the residual to a pure
`c(x)`. ∎

**Consistency.** With the true model in class, i.i.d. pairs, and a
sampling distribution giving each pair-class positive mass, MLE recovers
`r` up to the shift class at the standard `O(1/sqrt(n))` parametric rate
— nothing exotic; the content is what the equivalence class *means*:

## Why The Shift Ambiguity Is Harmless — And What Is Not

The downstream consumer (02) solves
`max_pi E[r] - beta KL(pi || pi_ref)` **per prompt**. Adding `c(x)` shifts
the objective by a constant per prompt and leaves the argmax unchanged:

```math
\pi^*_{r+c}=\pi^*_{r}
\qquad\text{— the ambiguity is exactly the part the policy cannot see.}
```

So Bradley–Terry identifies precisely enough reward for
KL-regularized policy optimization. What it does **not** identify:

```text
1. any monotone re-scaling beyond the model: preferences pin down r only
   THROUGH the assumed link sigma. If humans actually generate labels by
   some other link (heavier tails, label noise rho), the fitted r is a
   nonlinearly distorted version of the "true" utility — differences of
   size delta are compressed/expanded by the mis-link. The policy DOES
   see this distortion (it changes exp(r/beta) ratios).
2. cross-prompt scale: nothing calibrates r(x, .) ranges across prompts;
   a prompt with confident labels gets larger fitted gaps than one with
   noisy labels — the effective per-prompt temperature varies (a real,
   observed pathology: beta is global, the noise is not).
3. anything off the comparison distribution: r is estimated where pairs
   were sampled (typically: pairs from pi_ref-ish models). Its values on
   the policy's LATER outputs are extrapolation — the seed of 05.
4. intensity: "slightly better" and "vastly better" produce the same
   label; BT infers intensity only through win-RATE aggregation, i.e.
   from noise, not from signal.
```

## Label Noise And The Effective Temperature

A useful exact reduction: if true preferences follow BT with reward `r*`
but labels are flipped with probability `rho`, the observed process is

```math
\Pr\{\text{label says }y_1\}
=
(1-\rho)\,\sigma(\Delta r)+\rho\,\sigma(-\Delta r)
=
\rho+(1-2\rho)\,\sigma(\Delta r),
```

which is *not* a BT model in any reward — the MLE fit inside the BT class
converges to a **compressed** reward (smaller differences, since the
best-achievable probabilities are capped at `1-rho`). Downstream, by 02's
closed form, compression of `r` is equivalent to raising `beta`: label
noise silently strengthens the effective KL regularization. This is one
of the few places where a data pathology maps to a *benign* systematic
effect — contrast 05, where the pathologies are malign.

## Position

```math
(\mathcal{T},d,\mathcal{F})
=
\big(\text{none — pure estimation},\ \text{logistic log-loss},\ r_\phi\big),
```

with the estimand fixed only up to the fiber `{r + c(x)}` — a quotient the
next file's optimization respects exactly.

## What Remains Open

Preference models beyond BT (ties, non-transitivity — real human panels
violate transitivity at measurable rates, and BT projects that away);
per-prompt temperature/noise estimation; and active selection of pairs
(which comparisons most reduce policy-relevant uncertainty — an
information-directed question, `regret_and_exploration/06`, essentially
unexplored in this setting).
