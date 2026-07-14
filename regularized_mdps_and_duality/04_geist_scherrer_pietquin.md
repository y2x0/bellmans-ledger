# The General Theory Of Regularized MDPs

```text
Geist, Scherrer & Pietquin, "A Theory of Regularized Markov Decision
Processes," ICML 2019 — the umbrella over notebooks 01–03 and 05.
```

## The Setup

Replace negative entropy by **any** strongly convex regularizer
`Omega : Delta(A) -> R` (per state). Define the regularized greedy value —
the `Omega`-conjugate:

```math
\Omega^*(q)
=
\max_{\pi\in\Delta(\mathcal{A})}\ \big\{\langle\pi,q\rangle-\Omega(\pi)\big\},
\qquad
\mathcal{G}_\Omega(q)=\arg\max\ (\text{unique, by strong convexity}),
```

and the operators

```math
(\mathcal{T}_\Omega^\pi v)(s)=\langle\pi,q_v(s,\cdot)\rangle-\Omega(\pi(\cdot\mid s)),
\qquad
(\mathcal{T}_\Omega v)(s)=\Omega^*\big(q_v(s,\cdot)\big),
```

with `q_v(s,a) = r(s,a) + gamma E[v(s')]`. Instances:

```text
Omega = 0                        -> classical max            (mdp_foundations)
Omega = -tau H                   -> tau*logsumexp, softmax   (01–03)
Omega = tau KL(. || pi_ref)      -> tau log E_ref exp(q/tau),
                                    pi ∝ pi_ref e^{q/tau}    (05; rlhf_math/02)
Omega = Tsallis entropy          -> sparsemax (sparse support)
```

## The Three Structural Facts, Proved Once

**1. `Omega*` is nonexpansive.** For any `q_1, q_2`:
`Omega*` is a max of affine functions of `q`, hence convex, with gradient
`grad Omega*(q) = G_Omega(q) in Delta(A)` (Danskin). Mean value theorem
along `q_1 - q_2`: the derivative is a convex combination of coordinates
of `q_1 - q_2`, so

```math
|\Omega^*(q_1)-\Omega^*(q_2)|\ \le\ \|q_1-q_2\|_\infty .
```

**Consequence: `T_Omega` is a `gamma`-contraction** in sup-norm — the
softmax-lemma of 01 was one instance of "conjugates of convex regularizers
have simplex-valued gradients." Banach: unique `v_Omega^*`, geometric
convergence, error bounds — the entire classical apparatus, for every
`Omega` at once.

**2. The bias is the regularizer's range.** If
`L_Omega <= Omega <= U_Omega` on the simplex, then (same sandwich proof as
01):

```math
\|v_\Omega^*-v^*\|_\infty\ \le\ \frac{U_\Omega-L_\Omega}{1-\gamma}.
```

**3. Regularized improvement.** The argmax-program inequality of 03's
proof used nothing about entropy: for `pi' = G_Omega(q^pi)`,
`T_Omega^{pi'} v^pi >= v^pi`, and the monotone telescope gives
`v_Omega^{pi'} >= v_Omega^{pi}`. **Soft PI converges for every strongly
convex `Omega`.**

## Regularized MPI And Error Propagation

The paper's main theorem covers the practical loop — regularized Modified
Policy Iteration: `m` applications of `T_Omega^{pi_k}` for evaluation
(`m=1`: VI; `m=inf`: PI), greedy step `G_Omega`, with per-iteration errors
`eps_k` (evaluation) and `eps'_k` (greedy step):

```math
\limsup_{k}\ \|v_\Omega^*-v_\Omega^{\pi_k}\|
\ \le\
\frac{2\gamma}{(1-\gamma)^2}\ \limsup_k\ \big(\|\varepsilon_k\|+\|\varepsilon'_k\|\big)
\qquad\text{(norms/concentrability as in offline_rl/06)},
```

**identical shape to the unregularized theory** — regularization costs
nothing in error propagation and buys:

```text
- uniqueness/smoothness of the greedy step (the eps'_k of a soft greedy
  step is controllable; a hard argmax's is not — chattering);
- a quantified stability: G_Omega is (1/strong-convexity)-Lipschitz as a
  map q -> pi, so critic error moves the policy by a bounded amount.
  Compare distributional_rl/03's control pathology: the discontinuous
  selector was the villain; Omega buys the continuity that fixes it.
```

## The Unifying Observation About Modern Algorithms

The paper's taxonomy, extended — nearly every deep method is regularized
MPI with a particular `(Omega, m, error process)`:

```text
algorithm     Omega                    greedy step realized as
soft VI/SQL   -tau H                   exact logsumexp backup
SAC           -tau H                   KL-projection onto softmax (03)
TRPO/PPO      tau KL(.||pi_k)          one MD step (05) — Omega CHANGES
                                       each iteration (proximal, not fixed)
MPO           KL(.||pi_k), E-step      sample-based softmax + M-step proj.
DQN           0                        hard argmax (all the pathologies)
```

The TRPO/PPO row is the pivot to notebook 05: a *fixed* `Omega` biases
the final objective (fact 2); a *proximal* `Omega` — KL to the previous
iterate — vanishes at convergence and shapes only the path. Mirror
descent is the theory of the proximal case.

## What Remains Open

The regularized theory with function approximation inherits all of
`linear_mdps_and_completeness/`'s open ground; beyond that, the choice of
`Omega` as a *design variable* (sparsity via Tsallis, robustness via other
divergences, per-state adaptive temperatures) has taxonomy but little
optimality theory — which `Omega` is right for which uncertainty structure
is essentially unanswered.
