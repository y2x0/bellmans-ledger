# MuZero: The Paper

```text
Schrittwieser et al., Nature 2020 (arXiv 2019).
papers/muzero-schrittwieser-2019.pdf
Read against: model_based/04 (value equivalence — the license this
paper operationalizes) and search_and_planning/02–03 (the search it
inherits).
```

## The Three Functions

MuZero plans with a model that never predicts observations. Three
learned maps:

```math
\text{representation:}\quad s^0=h_\theta(o_{1:t})
\qquad\text{(observations }\to\text{ latent root state)}
```

```math
\text{dynamics:}\quad (s^{k},\ r^{k})=g_\theta(s^{k-1},\,a^{k})
\qquad\text{(latent step: next latent + predicted reward)}
```

```math
\text{prediction:}\quad (p^{k},\ v^{k})=f_\theta(s^{k})
\qquad\text{(policy prior + value at a latent state)}
```

The latent `s^k` has **no reconstruction target, no environment
semantics** — it is constrained only through what planning consumes:
rewards, values, policies. This is the value-equivalence principle
(`model_based/04`) as an architecture: the model owes the planner
correct evaluations of the functionals it will apply, and nothing else.

## The Search

AlphaZero's PUCT-MCTS (`search_and_planning/02–03`) run inside the
latent space: selection by `Q + c(s) P/(1+N)`-type bonus, expansion by
one `g_theta` step, leaf evaluation by `f_theta`, backup of `n`-step
discounted values (Atari; game outcomes for board games), with
min-max normalization of `Q` inside the tree (needed because Atari
rewards are unbounded, unlike win/loss). Search output: visit
distribution `pi_MCTS` and root value — the improvement operator of
`search_and_planning/02`, unchanged; only the simulator underneath it
is now learned and latent.

## The Loss (where value equivalence becomes sampled constraints)

Unroll the learned dynamics `K` steps (K = 5) along **actual observed
action sequences** and tie every unrolled prediction to a target from
the real trajectory:

```math
L_t(\theta)
=
\sum_{k=0}^{K}\Big[\
\ell^{r}\big(u_{t+k},\ r_t^{k}\big)
+\ell^{v}\big(z_{t+k},\ v_t^{k}\big)
+\ell^{p}\big(\pi^{\mathrm{MCTS}}_{t+k},\ p_t^{k}\big)
\Big]+c\|\theta\|^2,
```

```text
u:        observed rewards
z:        n-step bootstrapped returns (Atari) or final outcomes (games)
pi_MCTS:  the search's own visit distribution — the policy target
```

Each term is a sampled **Bellman-agreement constraint** in the sense of
`model_based/04`: the latent model is trained exactly and only to make
`(reward, value, policy)` along experienced action sequences match the
environment-plus-search's versions. The affine-subspace picture there
says many models satisfy these constraints; the loss picks one — with
gradients flowing through the recurrent `g_theta` (scaled by `1/2` per
step, latents scale-bounded) rather than through any observation model.

## Results, And The Two Findings With Theoretical Content

```text
board games:  matches AlphaZero (which HAS the true rules) in Go, chess,
              shogi — the value-equivalent latent simulator is
              PLANNING-COMPLETE for these domains: nothing the true
              rules provide, beyond what (r, v, p)-agreement encodes,
              was needed by the search;
Atari:        state of the art (mean/median human-normalized above prior
              model-free) — the first time a planning agent won the
              flagship model-free benchmark.
```

**Finding 1 — planning tolerance.** Performance grows with search budget
at evaluation (more simulations, better play) but saturates and even
degrades if the *training-time* search exceeds what the value/policy
heads can absorb — the compounding-error economics of `model_based/05`
showing up as an optimal search depth for a given model quality.

**Finding 2 — Reanalyze.** Old trajectories are periodically re-searched
with the current network to produce fresh `pi_MCTS, z` targets: replay
(`value_based_deep_rl/02`) where the *targets*, not just the data, are
revalued — legitimate precisely because the targets are search outputs,
not environment outcomes. This is the sample-efficiency mechanism of the
follow-ups (MuZero Reanalyze / EfficientZero), and it is DAgger's
"relabel with the current expert" (`imitation_and_inverse_rl/01`) with
search as the expert.

## The Audit

```text
what is proved:      the VE proposition (model_based/04): exact
                     (r, v, p)-agreement on all reachable action
                     sequences => planning outputs identical to true-
                     model planning;
what is trained:     agreement on K-step windows of VISITED sequences,
                     under the CURRENT policy's visitation — a sampled,
                     truncated, on-distribution shadow of the premise;
the gap:             off-visitation action sequences constrain nothing:
                     the latent model may be arbitrarily wrong exactly
                     where the search proposes novel lines — the
                     concentrability/OOD gap (offline_rl/06,
                     linear_mdps/05) relocated into latent dynamics;
                     search regularization (the prior p, limited depth)
                     is what keeps the planner inside the model's
                     competence — an implicit pessimism, unpriced.
```

## What Remains Open

Formal guarantees for the closed loop (learned VE model + PUCT + its own
visitation shaping the constraints) do not exist — the strongest
planning system inherits `stochastic_games/04`'s epitaph: weakest
theorem, strongest empirics. Stochastic environments need distributional
latents (Stochastic MuZero's afterstates/codes — `distributional_rl/`'s
concerns re-entering); and continuous action MCTS in latent space
(Sampled MuZero) trades the discrete theory of
`search_and_planning/01–02` for sampling arguments still being made
rigorous.
