# Soft Policy Iteration And SAC

## Soft Policy Iteration

The regularized analogue of `mdp_foundations/05`:

```text
(E) soft evaluation:   compute q_tau^pi  (fixed point of T_tau^pi — Banach, 01)
(I) soft improvement:  pi'(.|s) = softmax(q_tau^pi(s,.)/tau)
                       = argmax_pi' { <pi', q_tau^pi> + tau H(pi') }
```

## The Soft Improvement Theorem, Proved

**Theorem.** `q_tau^{pi'} >= q_tau^{pi}` pointwise, with equality iff
`pi` is already the softmax of its own `q`.

*Proof.* By the argmax property of `pi'` (per state):

```math
\langle\pi',q_\tau^\pi\rangle+\tau\mathcal{H}(\pi')
\ \ge\
\langle\pi,q_\tau^\pi\rangle+\tau\mathcal{H}(\pi)
\ =\ v_\tau^\pi(s).
```

The left side is exactly `(T_tau^{pi'} v_tau^pi)(s)`'s bracket... i.e. the
display says `T_tau^{pi'} v_tau^pi >= v_tau^pi`. Now run the monotone
telescope of the classical improvement proof (`mdp_foundations/05`) with
the soft operator — `T_tau^{pi'}` is monotone (positive combination of
monotone pieces; the entropy term is `pi'`-fixed) — so

```math
v_\tau^\pi\le\mathcal{T}_{\tau}^{\pi'}v_\tau^\pi\le(\mathcal{T}_\tau^{\pi'})^2v_\tau^\pi\le\cdots\to v_\tau^{\pi'}. \qquad\blacksquare
```

Uniqueness of the softmax argmax (strong concavity, 01) upgrades
"equality iff fixed point," and since the per-state programs have the
unique common solution `pi_tau^*`, **exact soft PI converges to the soft
optimum** (finitely many... — in the tabular case, monotone convergence to
the unique fixed point of `T_tau`; with the linear-convergence rate proved
via 01's contraction).

The classical proof survived regularization with two substitutions: argmax
→ regularized argmax, max-inequality → the argmax-program inequality. This
is notebook 04's general pattern in its most important special case.

## SAC = Approximate Soft PI

Soft Actor-Critic (Haarnoja et al. 2018) implements the loop with function
approximation and off-policy data:

**Critic (soft evaluation, one SGD step in place of the fixed point):**

```math
L_Q=\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\Big[\Big(Q_\theta(s,a)-r-\gamma\,\big(Q_{\bar\theta}(s',a')-\tau\log\pi_\phi(a'\mid s')\big)\Big)^2\Big],
\quad a'\sim\pi_\phi(\cdot\mid s'),
```

the `-tau log pi` term being the sampled entropy bonus (target network
`bar-theta`, min-of-two critics against max-bias —
`value_based_deep_rl/03`).

**Actor (soft improvement, projected):** minimize per-state KL to the
softmax target — which, dropping the `phi`-free partition function, is

```math
L_\pi=\mathbb{E}_{s\sim\mathcal{D},\,a\sim\pi_\phi}\Big[\tau\log\pi_\phi(a\mid s)-Q_\theta(s,a)\Big],
```

optimized by the reparameterization trick (`a = f_phi(s, xi)`, pathwise
gradients through `Q` — lower variance than the score-function estimator
of `policy_gradient/01`, available because `Q_theta` is differentiable in
`a`). The improvement theorem above is exactly what makes "KL-project onto
the softmax of the current critic" a sound improvement step even when the
projection is inexact (the theorem's inequality degrades gracefully with
the projection error — notebook 04 prices it).

**Temperature as dual ascent.** SAC v2 replaces fixed `tau` with a
constraint `E[H(pi(.|s))] >= H-bar`. The Lagrangian in `tau >= 0`:

```math
\max_\pi\min_{\tau\ge0}\ \ J(\pi)+\tau\big(\mathbb{E}[\mathcal{H}]-\bar{\mathcal{H}}\big)
\quad\Longrightarrow\quad
L(\tau)=\tau\,\mathbb{E}_{s,a\sim\pi}\big[-\log\pi(a\mid s)-\bar{\mathcal{H}}\big],
```

with SGD on `tau` (in practice on `log tau`): entropy above target →
`tau` decreases (exploit more), below → increases. The temperature is a
Lagrange multiplier and its "auto-tuning" is dual ascent — not a trick, a
duality.

## Why Off-Policy Is (More) Legitimate Here

SAC trains from a replay buffer without importance weights. Two structural
mitigations relative to raw off-policy TD
(`stochastic_approximation/06`): the *evaluation* target is the current
policy's soft value with the action `a'` **resampled from the current
`pi_phi`** (only the state marginal is off-policy — the action mismatch,
which drives Baird-type counterexamples' worst geometry, is corrected by
construction); and the entropy floor keeps `pi` from drifting so far that
the buffer's state distribution is irrelevant. Neither is a proof — the
triad exposure remains — but the norm mismatch is measurably smaller, which
matches SAC's empirical stability advantage over hard-max critics.

## What Remains Open

Convergence of the coupled SAC system (two critics, actor, temperature —
four timescales, none separated) is unproved beyond the exact-PI skeleton;
the right joint theory of entropy regularization + pessimism for the
offline variant (CQL is SAC-shaped — `offline_rl_and_ope/05`) is active
ground.
