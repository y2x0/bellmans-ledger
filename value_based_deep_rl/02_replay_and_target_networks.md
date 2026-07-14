# Replay And Target Networks As Operator Surgery

## Target Networks: Splitting The Fixed-Point Iteration

Online Q-learning with function approximation moves the target the instant
the parameters move: the map being iterated is

```math
\theta\ \mapsto\ \theta-\alpha\nabla_\theta\big(y(\theta)-Q_\theta\big)^2,
```

a fully coupled system in which an update that raises `Q(s_t, a_t)` typically
raises `Q(s_{t+1}, a)` too (shared features), which raises the target `y`,
which demands raising `Q(s_t, a_t)` again — a positive feedback loop through
the bootstrap. This loop is the function-approximation face of the
`gamma P` feedback in Baird's example (`stochastic_approximation/06`).

The target network **splits** the iteration into an outer fixed-point step
and an inner regression:

```math
\text{outer:}\quad
\theta^-_{k+1}=\theta_k^\dagger,
\qquad
\text{inner:}\quad
\theta_k^\dagger\approx\arg\min_\theta\ \mathbb{E}_{U(D)}\big(r+\gamma\max_{a'}Q(s',a';\theta_k^-)-Q(s,a;\theta)\big)^2 .
```

If the inner minimization were exact and the class unrestricted, the outer
iteration would be exactly **fitted value iteration**:

```math
Q_{k+1}=\Pi_{\mathcal{F},\hat\mu}\,\mathcal{T}\,Q_k ,
```

value iteration interleaved with a supervised projection. Error propagation
for fitted VI is understood (Munos–Szepesvári): if each regression incurs
error `eps_k` in `L^p(mu)`, then

```math
\limsup_k\|Q_k-q_*\|
\ \lesssim\
\frac{C_{\mu}}{(1-\gamma)^2}\,\max_k\varepsilon_k ,
```

with `C_mu` a concentrability coefficient measuring the mismatch between the
sampling distribution and the distributions the operator drags it through.
The `(1-gamma)^{-2}` and the concentrability constant are the honest price
tags: errors are amplified by the horizon and by distribution mismatch.
(The full theorem — the exact definition of `C_mu` as a discounted sum of
per-step density ratios, the three-move proof, and the companion
counterexample where `C = infinity` and fitted-Q diverges while MC
regression succeeds on the same data — is now proved in
`offline_rl_and_ope/06`.)
DQN's `C ~ 10^4` steps between syncs is an inexact, SGD-budgeted version of
the inner solve.

**The tradeoff dial:** small `C` → tighter coupling, faster tracking, more
instability; large `C` → stale targets, slower propagation of information
backward through time (each sync propagates values roughly one more backup
deep), more stability. Soft updates
`theta^- <- tau theta + (1-tau) theta^-` (DDPG-style) are the continuous
version of the same dial.

## Replay: Reshaping The Update Distribution

The `A`-matrix analysis (`stochastic_approximation/04`) says stability is a
property of the **update distribution** `D = diag(mu)`. Online RL hands you
`mu_t` = the current policy's transient distribution — nonstationary,
autocorrelated, and policy-coupled. Replay substitutes

```math
\hat\mu_{\text{replay}}
=
\frac{1}{|D|}\sum_{\tau=t-|D|}^{t}\mu_{\pi_\tau},
```

a sliding-window **mixture over recent policies**. Effects, in decreasing
order of mathematical respectability:

```text
1. decorrelation: uniform draws break the Markov autocorrelation of
   consecutive samples, restoring the near-i.i.d. sampling that SGD
   variance bounds assume;

2. distribution smoothing: mu changes at rate O(1/|D|) per step instead of
   at the policy's rate — the feedback loop (b) of notebook 01 is slowed by
   a factor of the buffer size;

3. sample reuse: each transition contributes to ~ |D|/batch updates,
   the sample-efficiency argument for off-policy methods.
```

What replay does **not** do: make `hat-mu` stationary for the *target*
(greedy) policy — the norm mismatch of the deadly triad remains, uncorrected
by any importance ratio. DQN works with one-step targets, where the missing
correction is confined to the state distribution (the action is corrected by
the `max` itself); with `n`-step targets the missing per-step ratios grow,
which is why uncorrected `n`-step + replay (as in Rainbow) is a bias the
practitioner accepts knowingly.

**Prioritized replay** (Schaul et al. 2016) samples transitions with
probability `p_i ∝ |delta_i|^omega`, then must importance-correct the
now-biased *loss* estimate with weights `(N p_i)^{-beta}` — a reminder that
any deliberate reshaping of `mu` re-enters the analysis as bias unless paid
for.

## The Two Devices Interact

Replay makes the data distribution lag the policy; target networks make the
regression target lag the parameters. Both are **low-pass filters** on the
two coupled processes of generalized policy iteration
(`mdp_foundations/05`): evaluation chasing improvement chasing evaluation.
DQN's stability is a statement that filtering both loops below their
resonant frequency suffices in practice, even where no contraction can be
certified — the engineering shadow of two-timescale stochastic approximation
(`stochastic_approximation/01`).
