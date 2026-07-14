# Value Equivalence And MuZero

## The Reorientation

Notebook 01's refinement said: model errors orthogonal to the value are
free. The **value-equivalence principle** (Grimm–Barreto–Singh–Silver
2020) promotes this from a bound to a definition: a model owes the
planner nothing except the answers to the queries the planner will
make.

## The Definition And The Basic Propositions

Fix a set of policies `Pi` and functions `V`. A model `tilde-P` is
**value-equivalent** to the true `P` w.r.t. `(Pi, V)` if

```math
\tilde{\mathcal{T}}^\pi v=\mathcal{T}^\pi v
\qquad\forall\pi\in\Pi,\ v\in\mathcal{V}
```

(the Bellman backups agree on the specified queries — not the kernels).

**Proposition 1 (structure).** The set of value-equivalent models is an
affine subspace of the space of transition kernels: the constraints

```math
\sum_{s'}\big(\tilde P-P\big)(s'\mid s,a)\,v(s')=0
\quad\forall v\in\mathcal{V}\ (\text{per }s,a\text{, mixed over }\pi)
```

are linear in `tilde-P`. *Proof:* read the display. ∎ Its codimension is
`~ dim span(V)` per cell: **small query sets leave enormous freedom** —
many wrong-as-physics models are exactly right as decision supports.

**Proposition 2 (planning losses coincide).** If `tilde-P` is
value-equivalent w.r.t. `(Pi, V)` with `V` closed under the relevant
backups (e.g. `V ⊇` the value iterates the planner generates, or the
fixed points `{v_pi}`), then planning in `tilde-P` (over `Pi`, with
those backups) returns the same values and the same greedy choices as
planning in `P`. *Proof:* induction over the planner's own backup
sequence — every quantity the planner ever computes is built from
queries on which the models agree. ∎

The fine print IS the theorem: closure of `V` under the planner's
operations. Equivalence w.r.t. too small a `V` licenses nothing once
the planner's iterates leave it — the completeness lesson of
`linear_mdps_and_completeness/03`, transposed to models.

## MuZero As Learned Value Equivalence

MuZero (Schrittwieser et al. 2019) closes the `search_and_planning/`
arc: AlphaZero's MCTS without a known simulator. The architecture —

```math
z_0=h_\theta(\text{observations}),
\qquad
(z_{k+1},\,\hat r_k)=g_\theta(z_k,a_k),
\qquad
(\hat p_k,\hat v_k)=f_\theta(z_k)
```

— an encoder, a latent dynamics, and prediction heads; MCTS
(`search_and_planning/02–03`, PUCT unchanged) runs entirely in latent
space on `g`. **The training losses are the value-equivalence
constraints, sampled:** unroll `g` for `K` steps along real
trajectories and regress

```text
hat-r_k  -> observed rewards          (reward-backup agreement)
hat-v_k  -> search/bootstrapped value targets   (value agreement)
hat-p_k  -> the MCTS visit distribution         (policy agreement)
```

Nothing anywhere asks `z` or `g` to reconstruct observations: the model
is trained *only* on planner-queried functionals — value equivalence
w.r.t. `V = {values, rewards, search policies along visited
sequences}`, learned. (The AIS conditions of
`pomdps_and_information_state/05` are the same contract — (P1) reward
and value prediction, (P2) self-consistent latent dynamics — which is
the deep reason MuZero's latent state works as a state.)

**What it gave up, per Proposition 1's freedom:** the model transfers
only to queries near its training set —

```text
- new REWARD functions void the equivalence class (V changed):
  vanilla MuZero does not transfer across tasks the way a kernel-
  accurate model would (02's point 1 forfeited);
- off-search-distribution rollouts (long unrolls, counterfactuals)
  query where no constraint was imposed: latent drift — notebook 05's
  compounding, in latent space, with no likelihood anchor;
- diagnostics are harder: a value-equivalent model cannot be validated
  against held-out TRANSITIONS, only against held-out RETURNS.
```

## The Design Axis This File Fixes

```text
model trained on          transfers across      pays capacity for
likelihood (P-hat)        rewards, planners     value-irrelevant detail
value equivalence         its (Pi, V) only      nothing wasted
```

Between the poles: multi-task value equivalence (train against a
*family* of reward functions — successor-feature-flavored), and
likelihood-regularized latent models (Dreamer-family, which buy
diagnosability back with a reconstruction term). The principle that
organizes the axis is notebook 01's pairing: **choose the model loss by
the functionals the downstream computation will pair the model error
against.**

## What Remains Open

Sample-complexity separations between value-equivalent and likelihood
model learning (intuition says VE needs less; clean theorems are
partial); principled `V`-closure in practice (how large must the query
set be for Proposition 2's license, with a learned planner?); and the
latent-drift problem under long-horizon planning — the open interface
with notebook 05.
