# The Performance Difference Lemma

## Statement (Kakade & Langford 2002)

For any two policies `pi'` and `pi`:

```math
J(\pi')-J(\pi)
=
\frac{1}{1-\gamma}\,
\mathbb{E}_{s\sim d_\mu^{\pi'}}\ \mathbb{E}_{a\sim\pi'(\cdot\mid s)}
\big[A_\pi(s,a)\big].
```

An **exact identity**, not a bound: the improvement of `pi'` over `pi`
equals the advantage of `pi` — the *old* policy's advantage function —
averaged over the *new* policy's visitation.

## Proof

Write the value difference as a telescoping sum along `pi'`-trajectories.
For a trajectory `tau ~ pi'` from `s0 ~ mu`:

```math
J(\pi')-J(\pi)
=
\mathbb{E}_{\tau\sim\pi'}\Big[\sum_{t=0}^\infty\gamma^t r(s_t,a_t)\Big]-\mathbb{E}_{s_0}\big[v_\pi(s_0)\big].
```

Insert the telescope
`v_pi(s_0) = sum_t gamma^t [v_pi(s_t) - gamma v_pi(s_{t+1})]` (valid path by
path; the tail vanishes by discounting):

```math
J(\pi')-J(\pi)
=
\mathbb{E}_{\tau\sim\pi'}\Big[\sum_t\gamma^t\big(r(s_t,a_t)+\gamma v_\pi(s_{t+1})-v_\pi(s_t)\big)\Big].
```

Given `(s_t, a_t)`, the conditional expectation of the bracket over
`s_{t+1} ~ P` is `q_pi(s_t, a_t) - v_pi(s_t) = A_pi(s_t, a_t)`. Summing the
discounted visitation into `d_mu^{pi'}` gives the lemma. ∎

The proof is the telescoping trick that also powers GAE (notebook 05) — the
bracket is precisely a TD error with respect to `v_pi`.

## Why This Identity Runs The Whole Family

**1. It re-derives the policy improvement theorem.** If
`E_{a~pi'}[A_pi(s,a)] >= 0` for all `s`, the right side is nonnegative:
`J(pi') >= J(pi)` — `mdp_foundations/05` recovered in one line, now in a
form that quantifies *how much* improvement.

**2. It isolates the exact difficulty of policy optimization.** We can
estimate `A_pi` from data collected by `pi`. But the lemma weights it by
`d^{pi'}` — the visitation of the policy we have *not yet deployed*. The
computable surrogate replaces `d^{pi'}` by `d^{pi}`:

```math
L_\pi(\pi')
=
J(\pi)+\frac{1}{1-\gamma}\,
\mathbb{E}_{s\sim d_\mu^{\pi}}\ \mathbb{E}_{a\sim\pi'}
\big[A_\pi(s,a)\big],
```

and with one-step importance sampling on the action
(`E_{a~pi'}[A] = E_{a~pi}[ (pi'/pi) A ]`) this is exactly TRPO/PPO's
surrogate `E[r_t(theta) A_t]` (PPO paper eq. 3/6, the "CPI" objective).

**3. It measures the surrogate's error.** Two properties of `L`:

```math
L_\pi(\pi)=J(\pi),
\qquad
\nabla_{\theta'}L_{\pi_\theta}(\pi_{\theta'})\big|_{\theta'=\theta}=\nabla_\theta J(\theta)
```

— the surrogate matches the true objective to **first order** at the current
policy. All error is second-order in the policy change, entering exclusively
through the mismatch `d^{pi'} vs d^{pi}`, and total variation between the
visitation distributions is controlled by per-state divergence of the
policies (each step's TV compounds at most linearly over the effective
horizon):

```math
\|d^{\pi'}-d^{\pi}\|_1
\ \le\
\frac{2\gamma}{1-\gamma}\,
\max_s\ \mathrm{TV}\big(\pi'(\cdot\mid s),\pi(\cdot\mid s)\big).
```

Small policy change ⇒ small distribution shift ⇒ trustworthy surrogate.
Quantifying "small" is the trust region bound, notebook 04.

## Conservative Policy Iteration (the ancestor)

Kakade–Langford's own algorithm: mix rather than replace,

```math
\pi_{\text{new}}=(1-\alpha)\,\pi+\alpha\,\pi'^{\text{greedy}},
```

and choose `alpha` small enough that the surrogate's guaranteed improvement
exceeds the distribution-shift penalty — yielding the first monotonic
improvement guarantee with function approximation. TRPO's contribution
(notebook 04) is extending the guarantee from mixtures to arbitrary policy
pairs, with KL replacing the mixture coefficient as the notion of "small."
