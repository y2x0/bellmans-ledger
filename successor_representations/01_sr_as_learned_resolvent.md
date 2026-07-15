# The Successor Representation As A Learned Resolvent

## The Object

For a fixed policy `pi` with state chain `P_pi`, the **successor
representation** is the matrix of discounted expected future visitations:

```math
M_\pi(s,s')
=
\mathbb{E}_\pi\Big[\sum_{t=0}^\infty\gamma^t\,\mathbb{1}\{S_t=s'\}\ \Big|\ S_0=s\Big]
\qquad\Longleftrightarrow\qquad
M_\pi=\sum_t\gamma^tP_\pi^t=(I-\gamma P_\pi)^{-1} .
```

Row `s` answers: *starting here, where will I spend my discounted
future?* This is the Neumann series of `mdp_foundations/01` — the object
that solved policy evaluation (`v = M r`), priced every error bound (the
resolvent amplification), defined occupancy measures
(`mdp_foundations/06`: `d^pi ∝ mu^T M`), and appeared as the deviation
matrix's cousin in `average_reward/02` — now promoted from proof device
to *learned representation*.

## The Factorization That Motivates Learning It

```math
v_\pi=M_\pi\,r
\qquad\text{— dynamics-under-}\pi\ \text{(M) and task (r), factored.}
```

Change the reward, keep the world: re-multiply, no re-learning. The SR
caches the part of evaluation that is expensive (propagation through
time) and exposes the part that changes across tasks (the reward) as a
linear read-out. Contrast the model-based/model-free poles this
repository has maintained: the SR is strictly between them —

```text
model-free (cached v):    one number per state; re-learn per task
SR (cached M):            propagation cached, task linear; no P needed
model-based (cached P):   compose at plan time; any task, any policy —
                          at planning cost and compounding-error risk
                          (model_based/05)
```

## TD-Learnability

`M` satisfies a Bellman equation **per successor column** — with the
indicator as the "reward":

```math
M_\pi(s,\cdot)=\mathbb{e}_s^\top+\gamma\,\mathbb{E}_{s'\sim P_\pi}\big[M_\pi(s',\cdot)\big],
```

so the TD(0) machinery applies verbatim, vector-valued:

```math
M(S_t,\cdot)\ \leftarrow\ M(S_t,\cdot)+\alpha\Big[\mathbb{e}_{S_t}^\top+\gamma\,M(S_{t+1},\cdot)-M(S_t,\cdot)\Big],
```

with the entire convergence inheritance: tabular a.s. convergence
(`stochastic_approximation/03` per column), linear-FA fixed point and
TvR bound (`stochastic_approximation/04` — the SR with features is
learned by `|features|` parallel TD learners sharing samples, an
instance of the many-predictions-one-stream pattern of
`options_and_hierarchy/02`). The "rewards" (indicators/features) are
known and noiseless; all statistical difficulty is the transition noise
— SR learning is TD with the reward-estimation confound removed.

## Spectral Structure

The SR's eigenvectors are those of `P_pi` (shared, with eigenvalues
mapped `lambda -> 1/(1 - gamma lambda)`): its top spectrum encodes the
chain's slow mixing directions — the same objects as
`average_reward/03`'s deviation matrix and the mixing times of
`finite_time_td_q/02`. Two exploitable consequences:

```text
1. eigenoptions: options that ascend/descend the SR's slow eigenvectors
   traverse the state space's large-scale geometry (bottleneck-crossing
   behaviors from pure dynamics, no reward) — the candidate discovery
   objective of options_and_hierarchy/04;
2. representation: SR rows embed states by their FUTURES — states are
   close iff they lead to the same places. This is the geometry deep
   auxiliary losses reach for, and the observed correspondence with
   hippocampal place-cell predictive maps (Stachenfeld et al. 2017) is
   the field's most striking bio-computational match since TD/dopamine
   (value_based_deep_rl/01's opening).
```

## What Remains Open

Deep SR (learned `phi` plus learned `psi` over it) faces a circularity —
the features being predicted are themselves moving (the representation-
collapse risks of self-predictive learning); principled objectives that
pin `phi` (reward prediction, observation reconstruction, orthogonality)
each trade coverage vs relevance with no settled theory. And the
control-side gap is structural: `M` is `pi`-dependent, which is exactly
where notebook 02's GPI enters.
