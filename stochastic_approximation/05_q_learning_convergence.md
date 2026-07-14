# Q-Learning Convergence: Watkins' Theorem

## The Algorithm

```math
Q_{t+1}(S_t,A_t)
=
Q_t(S_t,A_t)+\alpha_t(S_t,A_t)\Big[R_{t+1}+\gamma\max_{a'}Q_t(S_{t+1},a')-Q_t(S_t,A_t)\Big],
```

all other entries unchanged. This is an **asynchronous** stochastic
approximation of the optimality operator on `B(S x A)`:

```math
(\mathcal{T}q)(s,a)=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}q(s',a') .
```

## The Theorem

**(Watkins & Dayan 1992; Tsitsiklis 1994; Jaakkola–Jordan–Singh 1994.)**
Suppose:

```text
1. bounded rewards;
2. per-pair step sizes with   sum_t alpha_t(s,a) = inf,  sum_t alpha_t(s,a)^2 < inf;
3. every pair (s,a) is updated infinitely often.
```

Then `Q_t -> q_*` almost surely, and greedy(`Q_t`) is eventually optimal.

Two remarkable features: the behavior policy is *unconstrained* beyond
condition 3 (fully off-policy, no importance weights), and there is no model
estimation anywhere — the operator is sampled directly.

## Proof Skeleton

Write the update as RM iteration with

```math
h(Q)(s,a)=(\mathcal{T}Q)(s,a)-Q(s,a),
\qquad
M_{t+1}=\Big[R_{t+1}+\gamma\max_{a'}Q_t(S_{t+1},a')\Big]-(\mathcal{T}Q_t)(S_t,A_t).
```

**Step 1 — noise is martingale.** Given `F_t`, the sampled target's
expectation over `R, S'` is exactly `(T Q_t)(S_t, A_t)`; note carefully that
the `max` is applied to the *same* `Q_t` inside both terms, so no bias enters
the *convergence* argument. (Bias enters the *finite-time quality* of the
estimate — `value_based_deep_rl/03` — but not the fixed point: the fixed
point equation `q = Tq` is exact.)

**Step 2 — contraction.** `T` is a sup-norm `gamma`-contraction
(`mdp_foundations/03`), fixed point `q_*`.

**Step 3 — asynchronous scaffolding (the JJS argument).** Consider the error
`Delta_t = Q_t - q_*` and the auxiliary sequence
`beta_{k+1} = gamma beta_k` starting from `beta_0 >= ||Delta_0||`. Show by
induction: for each `k`, there is a (random, a.s. finite) time after which
`||Delta_t|| <= beta_k`. The inductive step splits the error into a
contraction part (deterministically `<= gamma beta_k` once all pairs have
been updated enough — condition 3) and a noise part (an RM iteration with
summable variance, converging to 0 a.s. by martingale convergence —
condition 2). Since `beta_k -> 0`, `Q_t -> q_*`. ∎

The proof is exactly "contraction + Robbins–Monro," per-component, with
condition 3 ensuring the contraction actually gets applied everywhere. This
is where **exploration becomes a hypothesis of a theorem**: epsilon-greedy
with persistent epsilon satisfies it in ergodic MDPs; a purely greedy policy
does not, and the theorem genuinely fails without it.

## What Does *Not* Follow

```text
1. Nothing about rates. Finite-time analyses came much later
   (Even-Dar & Mansour 2003: polynomial in 1/(1-gamma) with careful steps;
   speedy/zap variants improve constants).

2. Nothing under function approximation. Step 3 is a sup-norm argument tied
   to tabular (componentwise) updates. Replace the table by Phi w and the
   contraction has to survive a projection that is only nonexpansive in
   L2(mu) — with the max making the effective target policy differ from the
   behavior, mu is never stationary for the target. Divergence follows:
   notebook 06.

3. Nothing about the quality of Q_t as an estimator at finite t: the max of
   noisy estimates is upward-biased (Jensen), and the bias compounds through
   bootstrapping — the DQN-era overestimation literature
   (value_based_deep_rl/03).
```

## SARSA's Convergence, For Contrast

SARSA converges to `q_*` only if the behavior policy is simultaneously (i)
greedy in the limit and (ii) infinitely exploring ("GLIE" — e.g.
`epsilon_t -> 0` at the right rate): the operator being sampled is `T^{pi_t}`
for the *changing* behavior policy, so the target must itself converge to
greedy. Q-learning decouples target from behavior at the operator level —
the source of both its power (off-policy learning, replay compatibility) and
its instability under approximation.
