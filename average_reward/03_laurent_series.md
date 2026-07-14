# The Laurent Series Of The Discounted Value

## The Bridge Theorem

**Theorem (Laurent expansion, Puterman 8.2.3).** For a fixed policy, as
`gamma -> 1`:

```math
v_\gamma
=
\frac{\rho}{1-\gamma}\ +\ h\ +\ O(1-\gamma),
```

where `(rho, h)` are the gain and (normalized) bias of notebook 02. More
precisely, with `beta = (1-gamma)/gamma` and the deviation matrix `H`:

```math
v_\gamma=(I-\gamma P)^{-1}r
=
\frac{1}{1-\gamma}\,\rho+H r+\sum_{n=1}^{\infty}\beta^{n}\,(-H)^{n} H r ,
```

a convergent Laurent series in `beta` for `gamma` close enough to 1.

## The Proof

Split the resolvent along notebook 01's two projections. Write
`r = P* r + (I - P*) r = rho + (I - P*)r`.

**Steady part.** `(I - gamma P)^{-1} rho`: since `P rho = P P* r = P* r =
rho` (the gain vector is `P`-invariant),

```math
(I-\gamma P)^{-1}\rho=\sum_t\gamma^t P^t\rho=\sum_t\gamma^t\rho=\frac{\rho}{1-\gamma}.
```

The pole, isolated: **discounted value diverges at `gamma -> 1` exactly
along the gain direction, at rate `1/(1-gamma)`.**

**Transient part.** On the complement, `P` has spectral radius `< 1`
(notebook 02's invertibility argument), so `(I - gamma P)^{-1}(I - P*)r`
is analytic in `gamma` at 1; evaluate at `gamma = 1`:
`(I - P)^{-1}(I - P*) r = H r = h` (the deviation-matrix identities of
notebook 02), and expanding the analytic part in powers of `beta` around
0 produces the stated higher terms (each differentiation of the resolvent
brings down another `-H`). ∎

## What The Expansion Explains, Term By Term

**1. Every `1/(1-gamma)` in this repository, at once.** The repo's bounds
carry `(1-gamma)^{-1}` and `(1-gamma)^{-2}` factors
(`mdp_foundations/02, 04`, `finite_time_td_q/`, everywhere). The Laurent
series says: the leading behavior of value *is* `rho/(1-gamma)` — the
divergence is not an analysis artifact but the objective's actual shape;
discounted methods near `gamma = 1` are numerically differentiating a
pole.

**2. Why discounted methods rank policies correctly near `gamma = 1`.**
For two policies,

```math
v_\gamma^{\pi_1}-v_\gamma^{\pi_2}
=
\frac{\rho_1-\rho_2}{1-\gamma}+(h_1-h_2)+O(1-\gamma):
```

if the gains differ, the sign is dominated by the gain difference for
`gamma` near 1 (this is the engine of Blackwell optimality, notebook 04);
if gains tie, the *bias* decides — the discounted criterion at high
`gamma` is a lexicographic (gain, then bias, then higher terms) ranking,
which is exactly the `n`-discount hierarchy.

**3. The practical gamma dial, made precise.** Practitioners set
`gamma = 0.99` "for stability" on continuing tasks whose true objective
is average reward. The expansion prices this: optimizing `v_gamma`
optimizes `rho + (1-gamma) h + O((1-gamma)^2)` after rescaling — a
bias-perturbed gain objective with perturbation `(1-gamma) * span(h)`.
The error is negligible iff `1/(1-gamma)` well exceeds the chain's
mixing/transient scale (`span(h)` ~ mixing time × reward scale) — the
folk rule "gamma horizon >> mixing time," derived rather than assumed.

**4. The deviation matrix as the chain's Green's function.**
`H = sum_t (P^t - P*)` (when the sum converges — aperiodic case): the
accumulated departure of the `t`-step law from steady state. Its norm is
a mixing time in disguise; `span(h) <= 2 ||H|| ||r||`-type bounds tie
every constant in this family to how fast the chain forgets — the
quantity that replaces `1/(1-gamma)` as "the" difficulty parameter of
undiscounted RL (and appears in `finite_time_td_q/02`'s Markov-noise
inflation and the diameter of average-reward regret bounds).

## What Remains Open

Nothing in the expansion itself. The open frontier is algorithmic: methods
that *use* the Laurent structure (solve for `rho` and `h` jointly rather
than pushing `gamma -> 1`) exist in the tabular case (notebook 06,
Puterman's multichain algorithms) but their function-approximation and
finite-sample theory is thin — arguably the cleanest under-exploited idea
this family offers modern RL.
