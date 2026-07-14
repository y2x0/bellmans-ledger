# The Policy Gradient Theorem

## Statement

For `J(theta) = E_{s0 ~ mu}[v_{pi_theta}(s0)]` with differentiable
`pi_theta`:

```math
\nabla_\theta J(\theta)
=
\frac{1}{1-\gamma}\;
\mathbb{E}_{s\sim d_\mu^{\pi_\theta},\ a\sim\pi_\theta(\cdot\mid s)}
\Big[\nabla_\theta\log\pi_\theta(a\mid s)\ q_{\pi_\theta}(s,a)\Big],
```

where `d_mu^pi` is the discounted state-visitation distribution
(`mdp_foundations/06`):

```math
d_\mu^\pi(s)=(1-\gamma)\sum_{t=0}^\infty\gamma^t\,\mathbb{P}_\mu^\pi\{S_t=s\}.
```

**The content of the theorem is what is absent:** no
`grad_theta d_mu^pi` term. The gradient of the stationary/visitation
distribution — the genuinely hard object, since the environment's response to
policy change is unknown — cancels. Differentiating the policy requires no
model.

## Proof 1: Unrolling The Value Recursion

Differentiate `v_pi(s) = sum_a pi(a|s) q_pi(s,a)` by the product rule:

```math
\nabla v_\pi(s)
=
\sum_a\Big[\nabla\pi(a\mid s)\,q_\pi(s,a)
+\pi(a\mid s)\,\nabla q_\pi(s,a)\Big],
```

and `q_pi(s,a) = r(s,a) + gamma sum_{s'} P(s'|s,a) v_pi(s')` gives
`grad q_pi(s,a) = gamma sum_{s'} P(s'|s,a) grad v_pi(s')` (the model terms
carry no `theta`). Substituting yields a **linear recursion for**
`grad v_pi` — itself a Bellman-type equation:

```math
\nabla v_\pi(s)
=
g(s)+\gamma\sum_{s'}P_\pi(s'\mid s)\,\nabla v_\pi(s'),
\qquad
g(s):=\sum_a\nabla\pi(a\mid s)\,q_\pi(s,a).
```

Solve by the Neumann series (`mdp_foundations/01`):

```math
\nabla v_\pi=(I-\gamma P_\pi)^{-1}g
=\sum_{t=0}^\infty\gamma^t P_\pi^t\,g
\quad\Longrightarrow\quad
\nabla J=\sum_s\Big[\sum_t\gamma^t\,\mathbb{P}_\mu\{S_t=s\}\Big]\,g(s),
```

which is the claimed expectation once `sum_a grad pi q` is rewritten with the
**log-derivative trick** `grad pi = pi grad log pi`. ∎

The proof exhibits `grad v_pi` as the fixed point of the *same* evaluation
operator that defines `v_pi` — gradients of values are values of a different
"reward" `g`. This is why critics and gradient estimators share convergence
theory.

## Proof 2: Trajectory Likelihood (REINFORCE)

`J(theta) = integral p_theta(tau) G(tau) dtau` over trajectories. Then

```math
\nabla J
=\int p_\theta(\tau)\,\nabla\log p_\theta(\tau)\,G(\tau)\,d\tau,
\qquad
\nabla\log p_\theta(\tau)=\sum_t\nabla\log\pi_\theta(a_t\mid s_t),
```

because `log p_theta(tau) = log mu(s_0) + sum_t [log pi_theta(a_t|s_t) +
log P(s_{t+1}|s_t,a_t)]` and every non-policy term is `theta`-free — the
model drops out *by differentiation*, the second appearance of the theorem's
main miracle. Causality (rewards before `t` are independent of `a_t`; check
via `E[grad log pi_t · R_k] = 0` for `k <= t` by the tower rule) reduces
`G(tau)` to the reward-to-go, giving the practical estimator

```math
\hat g=\sum_t\nabla\log\pi_\theta(a_t\mid s_t)\ G_t .
```

Replacing `G_t` (MC) by `q(s_t, a_t)`, `A(s_t, a_t)`, or a bootstrapped
estimate produces the actor-critic spectrum; validity of each replacement is
notebook 02's subject.

## The PPO Paper's Form

PPO (paper eq. 1–2) writes the estimator as

```math
\hat g=\hat{\mathbb{E}}_t\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t\big],
\qquad
L^{PG}(\theta)=\hat{\mathbb{E}}_t\big[\log\pi_\theta(a_t\mid s_t)\,\hat A_t\big],
```

the "loss whose gradient is the policy gradient" for autodiff. The paper's
opening observation — that running multiple optimization epochs on `L^{PG}`
with the same batch is "not well-justified" and "empirically ... leads to
destructively large policy updates" — is precise: after the first step,
`theta != theta_old`, the states in the batch are no longer distributed as
`d^{pi_theta}`, and `L^{PG}`'s gradient is no longer `grad J`. Every later
notebook in this family is an answer to that one sentence.

## Two Fine Points

```text
1. The discounted d_mu^pi is the correct measure, but implementations
   average over states as collected (effectively undiscounted visitation).
   This is a known, deliberate bias (Thomas 2014); it vanishes as gamma -> 1
   and is universally accepted in practice.

2. Deterministic policies admit their own theorem
   (grad J = E[grad_theta mu_theta(s) grad_a q(s,a)|_{a=mu(s)}], Silver 2014)
   — the DPG/DDPG line; it trades the log-derivative's high variance for
   reliance on critic action-gradients.
```
