# The Simulation Lemma

## The One Identity

For a fixed policy `pi`, values under two models `P` and `hat-P` (same
rewards; reward error adds an obvious term):

```math
v_\pi-\hat v_\pi
=
(I-\gamma P_\pi)^{-1}r_\pi-(I-\gamma\hat P_\pi)^{-1}r_\pi .
```

**The resolvent-difference identity.** For any invertible `X, Y`:
`X^{-1} - Y^{-1} = X^{-1}(Y - X)Y^{-1}`. With `X = I - gamma P_pi`,
`Y = I - gamma hat-P_pi`:

```math
\boxed{\
v_\pi-\hat v_\pi
=
\gamma\,(I-\gamma P_\pi)^{-1}\,\big(P_\pi-\hat P_\pi\big)\,\hat v_\pi\ }
```

— the value gap is the model error `(P - hat-P)`, **measured against the
approximate value function** `hat-v`, then propagated through the true
resolvent. Every model-based bound in this family is this display plus
a norm choice.

## The Sup-Norm (Worst-Case) Corollary

Take `||.||_inf`; the resolvent has norm `<= 1/(1-gamma)`
(`mdp_foundations/01`), and
`|(P - hat-P)(.|s,a)^T hat-v| <= ||P - hat-P||_1 * ||hat-v||_inf / 2`-ish
(Hölder; TV pairing with the centered value is sharper by a factor 2):

```math
\|v_\pi-\hat v_\pi\|_\infty
\ \le\
\frac{\gamma\ R_{\max}}{(1-\gamma)^2}\ \max_{s,a}\ \mathrm{TV}\big(P(\cdot\mid s,a),\,\hat P(\cdot\mid s,a)\big),
```

using `||hat-v||_inf <= Rmax/(1-gamma)`. **Audit of the two horizon
factors:** one from the resolvent (errors persist), one from the value's
scale (errors are measured against a horizon-sized function) — the same
pair as everywhere (`offline_rl_and_ope/06`, `value_based_deep_rl/02`),
here in their cleanest form. Add reward error `eps_r`:
`+ eps_r/(1-gamma)`.

For *control* (plan optimally in `hat-P`, deploy in `P`): apply the
policy-error bound of `mdp_foundations/04` on top, or directly the
two-sided version — plan-in-model optimality plus two simulation-lemma
applications gives

```math
v^{*}-v^{\hat\pi^*}
\ \le\
\frac{2\gamma\,R_{\max}}{(1-\gamma)^2}\ \max_{s,a}\mathrm{TV}
\qquad\text{(the classical Kearns–Singh form).}
```

## The Value-Aware Refinement (the identity read honestly)

The boxed identity does **not** pair the model error with a worst-case
function — it pairs it with `hat-v` specifically:

```math
\big|(P-\hat P)^\top\hat v\big|
\quad\text{vs}\quad
\mathrm{TV}(P,\hat P)\cdot\|\hat v\|_\infty :
```

the left side can be arbitrarily smaller — model errors **orthogonal to
the value function are free**. Two consequences already in the
repository's orbit:

```text
1. Bernstein/variance versions: pairing with hat-v through Cauchy-Schwarz
   against Var_P(hat-v) instead of Hölder against TV gives the
   variance-weighted error that finite_time_td_q/04 exploited — one
   horizon factor recovered, minimax achieved (notebook 02);
2. value-aware model learning: train hat-P to minimize
   |(P - hat-P)^T v| over the values the planner will use, not
   likelihood/TV — the seed of the value-equivalence principle
   (notebook 04). Likelihood-trained models spend capacity on
   value-irrelevant detail (pixels of the sky); the lemma says exactly
   which errors the planner will actually pay for.
```

## Both Norms Stated (for downstream reuse)

```math
\text{sup-form:}\quad
\|v_\pi-\hat v_\pi\|_\infty\le\frac{\gamma}{1-\gamma}\,\max_{s,a}\big|(P-\hat P)^\top\hat v_\pi\big|
```

```math
\text{distribution-form:}\quad
\big|J(\pi)-\hat J(\pi)\big|
\le
\frac{\gamma}{1-\gamma}\ \mathbb{E}_{(s,a)\sim d^\pi}\Big[\big|(P-\hat P)(\cdot\mid s,a)^\top\hat v_\pi\big|\Big]
```

— the second (take `mu`-weighted inner products of the boxed identity
with the occupancy, `mdp_foundations/06`) is the one learned-model
practice needs: model error matters **where the policy goes**, which is
both the blessing (errors off-trajectory are free) and the curse (the
improved policy goes where the model was never trained — notebook 05's
distribution shift).

## What Remains Open

Nothing in the identity; everything in choosing the pairing — the open
program is value-aware/decision-aware model losses with end-to-end
guarantees (IterVAML, VaGraM, and the MuZero line of notebook 04 are
the current instances, none with a complete theory under function
approximation).
