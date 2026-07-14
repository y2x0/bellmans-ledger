# Hamilton–Jacobi–Bellman And Viscosity Solutions

## The Continuous-Time Limit

Dynamics `x-dot = f(x, u)`, running reward `r(x, u)`, discount rate
`rho > 0` (`gamma = e^{-rho dt}`):

```math
v(x)=\sup_{u(\cdot)}\ \int_0^\infty e^{-\rho t}\,r\big(x(t),u(t)\big)\,dt .
```

**Deriving HJB from the DPP.** The dynamic programming principle over a
short window `dt` (Bellman's principle of optimality,
`mdp_foundations/07`, in continuous time):

```math
v(x)=\sup_u\ \Big\{r(x,u)\,dt+e^{-\rho\,dt}\,v\big(x+f(x,u)\,dt\big)\Big\}+o(dt).
```

Expand (`e^{-rho dt} ~ 1 - rho dt`; `v(x + f dt) ~ v + <grad v, f> dt`),
subtract `v(x)`, divide by `dt`:

```math
\boxed{\ \rho\,v(x)=\max_u\ \Big[r(x,u)+\big\langle\nabla v(x),\,f(x,u)\big\rangle\Big]\ }
```

— the HJB equation: a **first-order nonlinear PDE** whose right side is
the Hamiltonian `H(x, grad v)`. The Bellman *operator* has become a
differential operator; "backup" has become "characteristic flow." (With
diffusion noise `dx = f dt + sigma dW`, Itô adds
`(1/2) tr(sigma sigma^T grad^2 v)` — second-order HJB; LQR's Riccati
equation is exactly this PDE solved on the quadratic ansatz.)

## Where Classical Solutions Die

HJB generically has **no C¹ solution**, and the failure is the rule, not
a corner case. The canonical 1D exhibit: `x-dot = u`, `|u| <= 1`, reward
= reach the origin quickly (minimum-time / equivalently
`r = -1` until arrival). The value is

```math
v(x)=-|x|
\qquad\text{(time to origin at full speed)},
```

with a **kink at `x = 0`** — precisely at the switching surface where
the optimal control jumps (`u = -sign(x)`). Any problem whose optimal
policy switches discretely (i.e., almost any interesting one) plants
kinks in `v` along the switching sets; on the other side, *too many*
Lipschitz functions satisfy the PDE almost everywhere (for the eikonal-
type equation `|v'| = 1`, every sawtooth qualifies): a.e.-solutions are
non-unique. Classical PDE theory offers a dilemma — no smooth solutions,
too many non-smooth ones.

## Viscosity Solutions: The Right Weak Sense

**Definition (Crandall–Lions).** A continuous `v` is a *viscosity
subsolution* if for every smooth test function `phi` touching `v` from
above at `x0` (local max of `v - phi`), the PDE inequality
`rho phi <= H(x0, grad phi)`... (with the sign conventions matched to
the sup); a *supersolution* with touching from below and the reverse
inequality; a **viscosity solution** if both.

The derivatives are borrowed from smooth test functions at touching
points — defined even at kinks, and *one-sidedly*, which encodes the
direction from which information flows (the "vanishing viscosity" limit
`-eps Laplacian -> 0` that names the notion selects the same solution).

**Theorem (statement).** Under standard conditions (Lipschitz `f, r`,
`rho > 0`): the value function is the **unique bounded uniformly
continuous viscosity solution** of HJB. Uniqueness is the hard half
(the comparison principle: sub `<=` super), and it is exactly what
restores the Banach-flavored situation — one equation, one solution,
and it is the value.

*The instinct to carry:* viscosity = the PDE-side implementation of the
DPP's one-sided structure — the same role monotonicity played in
`total_reward_ssp/03` (order, not smoothness, is what survives), and the
kink-selection mirrors "max of smooth things": `v` is a sup of smooth
flows, hence semiconvex-ish with kinks pointing one way only.

## What This Buys RL

```text
1. legitimacy of discretization: monotone finite-difference/semi-
   Lagrangian schemes converge TO THE VISCOSITY solution
   (Barles–Souganidis: consistency + monotonicity + stability suffice) —
   and a semi-Lagrangian scheme on a grid IS a discounted MDP
   (mdp_foundations/), with gamma = e^{-rho dt}. The repo's tabular
   theory is a convergent numerical method for HJB; this file is where
   the two literatures shake hands.
2. continuous-time RL: advantage rates, cf. the (1-gamma)->0 scalings —
   TD in the dt->0 limit estimates rho v - <grad v, f> - r; naive
   Q-learning degenerates (Q collapses to V at first order in dt — the
   advantage needs 1/dt rescaling): the known dt-fragility of
   discretized deep RL, diagnosed at the PDE level.
3. the switching-surface picture: policy kinks = value kinks = where
   function approximation of v is hardest — a geometric prediction for
   where value-based continuous control concentrates its errors
   (empirically borne out near contact/decision boundaries).
```

## What Remains Open

Numerics beat the curse of dimensionality only via structure (the grid
schemes above are exponential in `dim x`); neural HJB solvers (PINNs,
deep BSDE) lack viscosity-respecting guarantees — enforcing the
comparison principle in learned solutions is an open interface between
this file and the deep-RL families.
