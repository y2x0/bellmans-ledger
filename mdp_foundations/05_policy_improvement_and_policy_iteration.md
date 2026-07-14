# Policy Improvement And Policy Iteration

## The Policy Improvement Theorem

Let `pi` be stationary and `pi'` satisfy, in every state,

```math
\sum_a \pi'(a\mid s)\,q_\pi(s,a)\ \ge\ v_\pi(s)
\qquad\Longleftrightarrow\qquad
\mathcal{T}^{\pi'}v_\pi\ \ge\ v_\pi .
```

Then `v_{pi'} >= v_pi`, with strict inequality in some state if the premise
is strict in some state.

*Proof.* Monotonicity of `T^{pi'}` turns the one-step inequality into a
telescope:

```math
v_\pi
\ \le\ \mathcal{T}^{\pi'}v_\pi
\ \le\ (\mathcal{T}^{\pi'})^2 v_\pi
\ \le\ \cdots
\ \le\ \lim_k (\mathcal{T}^{\pi'})^k v_\pi
\ =\ v_{\pi'},
```

the limit by Banach (notebook 04). ∎

The premise is exactly "the one-step advantage of `pi'` over `pi` is
nonnegative everywhere":

```math
\sum_a\pi'(a\mid s)\,A_\pi(s,a)\ \ge\ 0\quad\forall s .
```

**The caveat that generates modern policy optimization:** the theorem needs
the inequality in *every* state. With function approximation you can only
certify it in expectation over *visited* states — under whose distribution?
The performance difference lemma (`policy_gradient/03`) answers exactly, and
TRPO/PPO are machinery for controlling the gap between "improvement under
`d_pi`" and "improvement under `d_{pi'}`".

## Policy Iteration

```text
repeat:
    (E)  policy evaluation:   solve v_{pi_k} = (I - gamma P_{pi_k})^{-1} r_{pi_k}
    (I)  policy improvement:  pi_{k+1}(s) = argmax_a [ r(s,a) + gamma sum_{s'} P(s'|s,a) v_{pi_k}(s') ]
until pi_{k+1} = pi_k
```

**Finite convergence.** Greedy `pi_{k+1}` satisfies
`T^{pi_{k+1}} v_{pi_k} = T v_{pi_k} >= T^{pi_k} v_{pi_k} = v_{pi_k}`, so by
improvement `v_{pi_{k+1}} >= v_{pi_k}`: the sequence is monotone over a
finite policy set (`<= |A|^{|S|}`), hence terminates. At termination
`T v_{pi_k} = v_{pi_k}`, so `v_{pi_k} = v_*` by uniqueness. ∎

**Rate.** PI converges at least as fast as VI per iteration
(`v_{pi_{k+1}} >= T v_{pi_k}`, both converging up to `v_*`), typically far
faster; recent results give strongly polynomial bounds for fixed `gamma`
(Ye 2011: `O(|S|^2 |A| / (1-gamma) * log(1/(1-gamma)))` iterations).

## Policy Iteration As Newton's Method

Define the residual map `B(v) = T v - v`; we seek its root. `T` is
piecewise-affine (upper envelope of the affine `T^pi`, notebook 03). On the
region where `pi_v` is greedy for `v`, `B(v) = r_{pi_v} + (gamma P_{pi_v} - I)v`
with "Jacobian" `gamma P_{pi_v} - I`. The Newton step:

```math
v_{k+1}
=
v_k-\big(\gamma P_{\pi_k}-I\big)^{-1}\big(r_{\pi_k}+(\gamma P_{\pi_k}-I)v_k\big)
=
(I-\gamma P_{\pi_k})^{-1}r_{\pi_k}
=
v_{\pi_k}.
```

Newton's method on the Bellman residual **is** policy iteration: the
linearization at `v_k` is choosing the greedy policy, and solving the linear
system is exact evaluation. This explains the empirical pattern
(quadratic-like terminal convergence, few iterations) and situates
**modified/optimistic PI** — replace exact evaluation by `m` applications of
`T^{pi_k}` — as damped Newton. `m = 1` is value iteration; `m = infinity` is
PI; actor-critic algorithms are stochastic optimistic PI with small `m`.

## Generalized Policy Iteration (Sutton & Barto's frame)

```text
Almost every algorithm in this repository is two interacting processes:

    evaluation:   pull the value estimate toward v_{pi}   (critic, TD, replay regression)
    improvement:  pull the policy toward greedy(v)        (actor, argmax, clipped surrogate,
                                                           MCTS visit counts)

The processes compete locally and cooperate globally; convergence arguments
are about keeping the improvement step from outrunning the evaluation step.
Two-timescale stochastic approximation (stochastic_approximation/01) is the
formal version of that sentence.
```
