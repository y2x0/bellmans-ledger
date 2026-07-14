# Monte Carlo Estimation And Importance Sampling

## Monte Carlo Value Estimation

Estimate `v_pi(s) = E_pi[G_t | S_t = s]` by averaging observed returns from
visits to `s`. No bootstrapping: the target `G_t` is the *actual* return.

```text
first-visit MC:  average returns following the first visit to s per episode
                 -> i.i.d. samples of G | S=s; unbiased; CLT applies cleanly.

every-visit MC:  average over all visits
                 -> samples within an episode are dependent; biased for finite n,
                    consistent; usually lower MSE in practice.
```

Incremental form is Robbins–Monro with `h(V) = v_pi - V`:

```math
V(s)\leftarrow V(s)+\alpha\big[G_t-V(s)\big].
```

## The Bias–Variance Position

```math
\mathrm{target}=G_t:
\qquad
\text{bias}=0,
\qquad
\mathrm{Var}[G_t]\ \text{compounds over the horizon} .
```

Return variance accumulates the randomness of every reward, transition, and
action to episode end — `O(H)` terms with correlations. TD replaces all but
one step of that randomness by the current estimate (`notebook 03`): bias in,
variance out. GAE (`policy_gradient/05`) interpolates the whole spectrum.

MC's compensating virtues: no reliance on the Markov property (targets are
model-free in the strongest sense), no divergence risk under function
approximation (it is plain regression on a fixed target), and no bootstrap
bias propagation.

## Off-Policy Evaluation: Importance Sampling

Data from behavior policy `b`, target `pi`, with support
`pi(a|s) > 0 => b(a|s) > 0`. Correct the measure trajectory-wise:

```math
\rho_{t:T-1}
=
\prod_{k=t}^{T-1}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)},
\qquad
\mathbb{E}_b\big[\rho_{t:T-1}\,G_t\mid S_t=s\big]=v_\pi(s).
```

*Proof of unbiasedness:* the trajectory density under `b` times `rho` equals
the density under `pi`; transition factors cancel because they are policy-
independent — importance weights never require knowledge of `P`.

## Ordinary vs Weighted IS

Given returns `G_i` with weights `rho_i` from `n` (first-visit) trajectories:

```math
\text{ordinary:}\quad
\hat v^{\mathrm{OIS}}=\frac{1}{n}\sum_{i=1}^n\rho_i G_i,
\qquad
\text{weighted:}\quad
\hat v^{\mathrm{WIS}}=\frac{\sum_i \rho_i G_i}{\sum_i \rho_i}.
```

```text
             bias                variance
OIS          unbiased            can be unbounded (even infinite)
WIS          biased (O(1/n),     bounded by max |G|; self-normalizing
             consistent)
```

The variance pathology of OIS is structural, not a corner case: `rho` is a
product of ratios, so `log rho` is a random walk and `rho` is approximately
log-normal with variance **exponential in horizon**. A single trajectory with
`rho ~ gamma^{-H}`-scale weight dominates the average. WIS trades a vanishing
bias for sane variance and is almost always preferred.

## Per-Decision Importance Sampling

Rewards at time `k` depend only on actions up to `k`, so future ratios are
unnecessary per-reward:

```math
\mathbb{E}_b\big[\rho_{t:T-1}G_t\big]
=
\mathbb{E}_b\Big[\sum_{k=t}^{T-1}\gamma^{k-t}\,\rho_{t:k}\,R_{k+1}\Big],
```

using `E[rho_{k+1:T-1} | F_{k+1}] = 1` (each factor has conditional mean 1).
Strictly smaller weights per term, same expectation — the standard variance
reduction, and the seed of the tree-backup / Retrace(`lambda`) family used in
deep off-policy learning.

## Where This Family Reappears

```text
1. One-step IS ratio pi/b with b = pi_old is exactly PPO's r_t(theta)
   (policy_gradient/06): PPO is per-decision importance sampling truncated
   to one step, with clipping as a hard variance control.

2. Q-learning avoids IS entirely by bootstrapping through the max
   (notebook 05) — the off-policy correction is absorbed by the operator,
   at the price of the deadly-triad exposure (notebook 06).

3. Replay buffers in DQN are off-policy data used *without* IS correction;
   the target network + one-step targets are what keep the missing ratios
   from mattering (value_based_deep_rl/02).
```
