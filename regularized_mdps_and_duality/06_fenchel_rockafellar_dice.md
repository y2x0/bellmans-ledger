# Fenchel–Rockafellar Duality And The DICE Derivation

## The Promise To Keep

`mdp_foundations/06` exposed the occupancy LP; `offline_rl_and_ope/03`
used its balance constraint and asserted a saddle-point estimator. This
file derives that estimator, exhibiting the DICE family as **regularized
LP duality executed on samples** — the family's convex-analytic capstone.

## The Regularized Occupancy Program

Estimate `J(pi)` for a target `pi` from data `~ d^{pi_b} =: d^D`. Consider
the program over occupancy-like variables `d` (write
`w(s,a) = d(s,a)/d^D(s,a)`):

```math
\max_{d\ \ge 0}\quad
\sum_{s,a}d(s,a)\,r(s,a)\ -\ \alpha\,D_f\big(d\,\|\,d^{D}\big)
\qquad
\text{s.t.}\quad
\underbrace{\sum_a d(s',a')=(1-\gamma)\mu_0\pi+\gamma\,\mathcal{P}^\pi_* d}_{\text{Bellman flow (mdp_foundations/06), FOR }\pi},
```

`D_f` an `f`-divergence penalizing departure from the data occupancy,
`P^pi_*` the `pi`-directed transposed transition operator. At `alpha = 0`
the constraint forces `d = d^pi` and the value is `J(pi)` (up to the
`(1-gamma)` convention); `alpha > 0` regularizes the (ill-posed, sampled)
problem.

## Dualize

Lagrange multiplier `nu(s,a)` (one per flow constraint — a **value-shaped
object**, as the LP dual always produces, `mdp_foundations/06`).
Lagrangian, after the standard regrouping of the flow term against `d`:

```math
L(d,\nu)
=
(1-\gamma)\,\mathbb{E}_{\mu_0,\pi}[\nu]
+\sum_{s,a}d(s,a)\,\Big[\underbrace{r+\gamma\,\mathcal{P}^\pi\nu-\nu}_{=:\ \delta_\nu\ \text{(a Bellman residual!)}}\Big](s,a)
-\alpha\,D_f\big(d\|d^D\big).
```

Now eliminate `d` in closed form — this is the Fenchel–Rockafellar step:
writing `d = w d^D` and maximizing pointwise over `w >= 0`,

```math
\max_{w\ge0}\ \Big\{w\,\delta_\nu-\alpha f(w)\Big\}
=
\alpha f^*\!\Big(\frac{\delta_\nu}{\alpha}\Big)
\qquad\text{(the convex conjugate — 01's move, on the ray)},
```

with maximizer `w* = (f')^{-1}(delta_nu / alpha)`. The dual problem:

```math
\boxed{\
\min_{\nu}\quad
(1-\gamma)\,\mathbb{E}_{s\sim\mu_0,a\sim\pi}\big[\nu(s,a)\big]
\ +\ \alpha\ \mathbb{E}_{(s,a)\sim d^{D}}\Big[f^*\Big(\tfrac{r+\gamma\mathcal{P}^\pi\nu-\nu}{\alpha}\Big)(s,a)\Big]\ }
```

**Every term is a sampleable expectation**: the first from initial states
(and `pi`, which we can query); the second from data transitions — the
operator `P^pi nu (s,a) = E_{s'|s,a} E_{a'~pi} nu(s',a')` is estimated by
the observed `s'` and a sampled `a'`. Unbiased SGD on `nu` applies. No
importance weights on trajectories anywhere; no double sampling (the
conjugate is of the *linear* residual, not its square — this is precisely
how the `stochastic_approximation/06` obstruction is dodged: convex
duality replaced the square).

## Recovering Everything

```text
the ratio:      w*(s,a) = (f')^{-1}(delta_nu*(s,a)/alpha) — the
                marginalized IS weight of offline_rl_and_ope/03, now as a
                dual optimality condition;
the estimate:   J-hat = E_{d^D}[w* r]  (or read J off the dual value);
DualDICE:       f(w) = (w-1)^2/2  ->  f*(x)=x^2/2+x: least-squares-shaped
                dual in nu; alpha -> 0 recovers exact evaluation;
GenDICE etc.:   other f, plus normalization constraints (an extra scalar
                multiplier) for the average-reward variant;
the adjoint     nu* differs from a q-value of pi by a constant-ish
reading:        correction: values are Lagrange multipliers of occupancy
                flow — the third time this repo has met the fact
                (LP duality; the backward/forward operators of
                offline_rl_and_ope/03; here as the dual variable).
```

## Why This Framing Matters Beyond OPE

The same Lagrangian, with `pi` also free, is the **primal-dual form of the
whole RL problem**: policy (via `d`), value (via `nu`), and ratio (via
`w`) as three coordinates of one saddle point. Offline RL, imitation
(match `d` to expert occupancy — the divergence term with roles swapped),
and constrained RL (extra multipliers) are all this one program with
different terms frozen. The regularizer `alpha D_f` is what makes the
sampled saddle well-posed — the family's thesis (regularization as the
enabling device of modern RL) in its most literal form.

## What Remains Open

With neural `nu, w` the saddle is nonconvex-nonconcave: practical DICE
training can cycle or collapse, and no satisfactory optimization theory
exists; the statistical theory under partial coverage (where `w` is
unbounded) merges into `offline_rl_and_ope/04`'s pessimism program —
penalized/pessimistic DICE variants are current research.
