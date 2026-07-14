# The Geometry Of Policy Optimization, Assembled

## The Three Spaces

Policy optimization happens in three coordinate systems at once; every
result in this family is a statement about one of the maps between them:

```math
\underbrace{\theta\in\mathbb{R}^{p}}_{\text{parameters}}
\ \xrightarrow{\ \pi_\cdot\ }\
\underbrace{\pi\in\prod_s\Delta(\mathcal{A})}_{\text{policy bundle}}
\ \xrightarrow{\ \lambda_\cdot\ }\
\underbrace{\lambda^\pi\in\Lambda}_{\text{occupancy polytope}},
\qquad
J(\theta)=\langle\lambda^{\pi_\theta},\,r\rangle .
```

```text
in Lambda:   J is LINEAR; the feasible set is a polytope; optima at
             vertices = deterministic policies (mdp_foundations/06).
             All difficulty gone — and all tractability too: Lambda is
             defined by the unknown P.
in pi-space: J is nonconvex but gradient-dominated (01); per-state
             linear with the OTHER states' occupancies as weights.
in theta:    everything bad lives in the two Jacobians: softmax
             saturation (d pi / d theta ~ 0 at near-deterministic pi)
             and visitation vanishing (d lambda / d pi kills unvisited
             states).
```

The plateau of notebook 01 = both Jacobians small simultaneously; NPG
(02) = re-metrize `theta`-space so the first Jacobian is the identity in
the KL geometry; the transfer term (04) = the second Jacobian's
statistical shadow. One picture, four notebooks.

## Entropy As Interior-Point Method

The regularized objective in occupancy coordinates:

```math
\max_{\lambda\in\Lambda}\ \ \langle\lambda,r\rangle+\tau\,\widetilde{\mathcal{H}}(\lambda),
\qquad
\widetilde{\mathcal{H}}(\lambda)=-\sum_{s,a}\lambda(s,a)\log\frac{\lambda(s,a)}{\sum_{a'}\lambda(s,a')}
```

(the conditional entropy of `a` given `s`, expressed in `lambda` — a
**concave** function of `lambda`; check: it is a sum of per-state
`-x log(x/X)`-type perspective functions).

**Proposition (the central-path reading).** The `tau`-regularized problem
is a barrier-smoothed LP: strictly concave on the interior fibers, its
unique optimum `lambda_tau^*` traces a **central path** that converges to
the (face of) optimal vertices as `tau -> 0`; along the path,

```math
\text{curvature}\ \succeq\ \frac{\tau}{\text{(occupancy scale)}}
\quad\text{on action-fibers}
```

— strong concavity proportional to `tau`, degenerating exactly as the
path approaches the boundary (deterministic policies). *Proof sketch:*
the Hessian of `-x log(x/X)` on a fiber is `diag(1/x)`-like, bounded below
by `tau` times the inverse occupancy on the simplex's interior;
positivity of interior occupancies under `mu > 0` closes it. ∎

This is the geometric restatement of every regularization purchase in the
repo:

```text
linear NPG convergence with entropy (02)   = interior-point iterates on a
                                             barrier-smoothed LP converge
                                             linearly at fixed tau
tau-annealing schedules                    = path-following
the exploration floor (reg_mdps/01)        = the barrier keeps lambda off
                                             the boundary where gradients
                                             (and data!) die
the bias tau log|A|/(1-gamma)              = distance from central path
                                             to vertex at parameter tau
```

## The Conservation Law (the family's thesis, stated once)

Collecting 01 → 04:

```math
\text{difficulty}(\text{policy optimization})
=
\text{optimization geometry}
\ +\
\text{statistical coverage},
\qquad
\text{and reparameterization only moves mass between the two terms.}
```

Vanilla PG stores exploration hardness as plateau constants; NPG moves it
into the critic's transfer coefficient; exploration bonuses
(`regret_and_exploration/`) are the only term that *reduces* the sum, by
changing the data distribution itself. Any proposed policy-optimization
improvement should be audited against this ledger: which term did it
shrink, and where did the mass go?

## Interfaces Out

```text
to rlhf_mathematics/:  RLHF operates at large tau (strong KL to pi_ref)
                       on a short-horizon problem: deliberately deep in
                       the interior, where this family's geometry is
                       benign — one reason RLHF-PPO is stabler than
                       game-playing PPO.
to unifying_view/01:   the (T, d, F) table gains its policy-side column:
                       operator = MD/NPG step, metric = per-state KL
                       weighted by visitation, class = policy
                       parameterization; failure modes = the two
                       Jacobians.
```

## What Remains Open

A quantitative central-path theory for *deep* policy classes (the
`theta -> pi` map is no longer surjective onto the bundle; the barrier
lives in a space the network may not cover); and the interaction of the
conservation law with representation learning — whether learned features
can genuinely cheapen coverage rather than relocate it.
