# GAIL: Imitation As Occupancy Matching

```text
Ho & Ermon, NeurIPS 2016. papers/gail-ho-ermon-2016.pdf
```

## The Composition To Collapse

The classical pipeline is IRL-then-RL: recover `r-hat` (notebook 02),
then solve the MDP for it. Ho–Ermon ask what the *composition*
`RL ∘ IRL` computes, and answer it in closed form using the machinery
this repository built in `mdp_foundations/06` and
`regularized_mdps_and_duality/06`.

## The Derivation

Write regularized MaxEnt-style IRL with a convex reward regularizer
`psi` (cost convention `c = -r` in the paper; rewards kept here):

```math
\mathrm{IRL}_\psi(\pi_E)
=
\arg\max_{r}\ \Big[\ \min_\pi\ \big(-\mathcal{H}(\pi)-\mathbb{E}_\pi[r]\big)\ +\ \mathbb{E}_{\pi_E}[r]\ -\ \psi(r)\Big],
```

(find a reward under which the expert beats every entropy-regularized
policy, regularized by `psi`). Every expectation is an inner product
with an **occupancy measure** (`mdp_foundations/06`):
`E_pi[r] = <rho_pi, r>`. Then `RL ∘ IRL_psi` is a max–min in
`(pi, r)`; the objective is concave in `r`, convex-ish in `rho_pi` over
the occupancy polytope; swap the order (minimax duality on the polytope)
and **eliminate `r` by conjugacy** — the same Fenchel step as
`regularized_mdps_and_duality/06`:

```math
\boxed{\
\mathrm{RL}\circ\mathrm{IRL}_\psi(\pi_E)
=
\arg\min_{\pi}\ \ \psi^*\big(\rho_\pi-\rho_{\pi_E}\big)\ -\ \mathcal{H}(\pi)\ }
```

**(Ho–Ermon, Prop. 3.2.)** The reward has vanished; what the pipeline
computes is a policy whose occupancy measure is close to the expert's,
with the divergence chosen by the regularizer's conjugate `psi*`, plus
an entropy bonus. **Imitation is occupancy matching; the reward was a
dual variable all along** — the third time this repository has caught a
"value-like" object being a Lagrange multiplier of occupancy flow
(`mdp_foundations/06`, `regularized_mdps_and_duality/06`, here).

Instances of the correspondence:

```text
psi = 0 (exact IRL):        psi* = indicator  ->  exact occupancy matching
                            (feasible but statistically brittle)
psi = feature-class         psi* = feature-expectation matching  ->
indicator:                  Abbeel–Ng / MaxEnt IRL (notebook 02) recovered
psi = GAIL's logistic       psi* = Jensen–Shannon-type divergence:
regularizer:                min_pi  D_JS-ish(rho_pi, rho_E) - lambda H(pi)
```

## The Algorithm

The JS-type divergence is realized variationally by its GAN form — a
discriminator `D_w(s, a)` classifying expert vs policy state-actions:

```math
\max_{w}\ \
\mathbb{E}_{\pi_\theta}\big[\log D_w(s,a)\big]
+\mathbb{E}_{\pi_E}\big[\log(1-D_w(s,a))\big],
```

alternated with a policy step (TRPO/PPO — `policy_gradient/04–06`) on
the **surrogate reward** `r-tilde(s,a) = -log D_w(s,a)`: the policy is
rewarded for visiting state-actions the discriminator cannot
distinguish from the expert's. Notes with theoretical content:

```text
1. the discriminator is a LEARNED LOCAL METRIC on occupancy space — the
   same role the Wasserstein/Cramér choice played in distributional_rl/:
   the divergence is a design axis (f-GAN variants, Wasserstein-GAIL,
   primal DICE-style ValueDICE all occupy cells of it);
2. it inherits GAN dynamics wholesale — the zero-sum rotational
   failure modes of stochastic_games/05 (cycling, discriminator
   overpowering) ARE GAIL's instabilities, same mathematics;
3. versus BC (notebook 01): matching occupancies rather than
   conditionals means the learner is trained toward states the EXPERT
   visits by a signal defined on states the LEARNER visits — the
   on-policy correction of DAgger without expert queries, paid for with
   environment interaction and adversarial optimization;
4. sample economics: the paper's headline is expert-sample efficiency
   (few demonstrations, much environment interaction) — the exact
   mirror of BC (many demos, zero interaction): the two methods bracket
   the demos-vs-interaction budget line, with DAgger's expert-query
   axis the third dimension.
```

## What Remains Open

Convergence of the coupled discriminator–policy system (nonconvex-
nonconcave; last-iterate theory absent — `stochastic_games/05`'s open
end, verbatim); the right divergence for imitation (support-coverage vs
mode-seeking trade, task-dependent, untheorized); and off-policy /
offline GAIL (ValueDICE and successors move the matching into the DICE
frame of `offline_rl_and_ope/03` — active, incomplete).
