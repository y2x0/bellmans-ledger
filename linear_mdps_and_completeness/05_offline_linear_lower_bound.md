# The Offline Linear Lower Bound (Wang–Foster–Kakade)

## The Claim

Offline policy evaluation/optimization with

```text
1. REALIZABLE linear values:  q_pi (resp. q*) exactly in span(phi), and
2. GOOD COVERAGE: the data distribution's feature covariance is
   well-conditioned (lambda_min bounded below — the best one can ask)
```

still requires `min( exp(Omega(d)), exp(Omega(H)) )` samples in the worst
case. Coverage + realizability — the two conditions practitioners check —
are **jointly insufficient** offline. This is Baird
(`stochastic_approximation/06`) upgraded from "an algorithm diverges" to
"no algorithm can succeed."

## The Geometry Of The Construction

The engine is error amplification through backups in a norm the data
cannot see. Sketch of the `H`-step construction:

**The feature geometry.** Take `~ exp(d)`-many states per level arranged so
their features `phi(s)` form a near-orthogonal packing of the unit sphere
in `R^d` (pairwise `|<phi_i, phi_j>| <= 1/2`; exists by
Johnson–Lindenstrauss-type volume arguments — the same packing as notebook
03's hardness result; it is *the* recurring object of linear hardness).

**The dynamics.** From each level-`h` state, transitions mix over level-
`h+1` states such that the *conditional mean feature* is a **contraction
in the data-visible directions but not in one hidden direction** `u_h`:
the value at level `h` equals a fixed multiple `(1/rho)` of the projection
of the level-`h+1` value onto `u_h`, with `rho < 1` chosen so that

```math
\text{per-level amplification}=\frac{1}{\rho}
\quad\text{while}\quad
\text{per-level data-visible signal}=O\Big(\frac{1}{\sqrt{\exp(d)}}\Big):
```

each backup multiplies the *value* consequence of an estimation error by
`1/rho`, but any *sample* from the data distribution constrains the hidden
direction only through inner products with near-orthogonal features —
i.e., with exponentially small leverage. Realizability is maintained
exactly (values are linear by construction at every level); coverage is
maintained (the data distribution is near-isotropic over the packing).

**The accounting.** After `H` levels, a per-level estimation error `eps_0`
that the data cannot resolve below `eps_0 ~ 1/sqrt(n / exp(d))` becomes a
value error `eps_0 / rho^H`. To output a correct evaluation, the learner
needs `eps_0 rho^{-H} <= eps`, forcing

```math
n\ \ge\ \exp(\Omega(d))
\quad\text{or}\quad
\text{(if }d\text{ large) } \rho^{-H}\text{ forces } \exp(\Omega(H)) .
```

Formalized via the Fano/packing machinery of `concentration_toolkit/05`:
`exp(Omega(d))` alternatives (which hidden direction carries the value),
pairwise KL per sample `exp(-Omega(d))`. ∎ (shape)

## What Exactly Fails, In The Language Of This Repo

```text
the norm mismatch, quantified:  the backup is contractive in sup-norm over
REACHED states under pi, but the data norm is L2(mu_data); the construction
maximizes the ratio ||.||_backup-relevant / ||.||_data over the class —
precisely the ratio the deadly triad file said was unbounded, now built
adversarially and priced.

concentrability was the wrong condition: lambda_min(Sigma_data) bounded
below (feature-space coverage) does NOT control the distribution mismatch
along BACKUP PATHS. The sufficient condition that does — Bellman
completeness (03) or single-policy concentrability WITH pessimism
(offline_rl_and_ope/04) — each repairs a different half.
```

## The Table This Theorem Completes

```text
offline, linear class:
                          coverage        no coverage
completeness              poly (FQI-type) poly w/ pessimism, target-dependent
realizability only        EXP (this file) EXP (a fortiori)
```

Read together with notebook 03's online hardness: realizability fails in
every data regime; completeness (or environmental low rank) rescues every
data regime. The conceptual boundary of the family is exactly the
completeness line, from both sides now.

## Practical Import

This theorem is the precise reason offline deep RL (`offline_rl_and_ope/`)
is built on *pessimism about actions the data does not support* rather
than on "our network can represent Q* and the buffer is diverse":
the latter pair is this file's hypothesis set, and it is insufficient in
principle, not merely in practice.

## What Remains Open

The matching upper bound territory: the exact sample complexity under
completeness with partial coverage (constants and exponents still moving);
and whether *natural* (non-adversarial) MDP families exhibit the
amplification geometry — empirical evidence says mildly yes (value errors
do compound off-support), but no formal average-case theory exists.
