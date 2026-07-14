# TD(0), TD(lambda), And The Forward–Backward Equivalence

## TD(0)

```math
V(S_t)\leftarrow V(S_t)+\alpha\,\delta_t,
\qquad
\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t).
```

The TD error `delta_t` is the one-sample Bellman residual. Its conditional
expectation under the true value function vanishes:

```math
\mathbb{E}_\pi[\delta_t\mid S_t=s]\Big|_{V=v_\pi}=(\mathcal{T}^\pi v_\pi)(s)-v_\pi(s)=0,
```

so `v_pi` is the equilibrium of the update — the fixed-point characterization
doing stochastic work.

**TD is biased while learning** (`V(S_{t+1})` is wrong during training) but
its target has one step of environment randomness instead of a full horizon.
The MC-vs-TD tradeoff is the bias–variance tradeoff of `notebook 02`, now
with the bias induced by *bootstrapping* rather than sampling.

**TD is not gradient descent.** There is no function `J` with
`grad J = -E[delta_t grad V(S_t)]` in general (the expected update is not a
symmetric linear map — see notebook 04's matrix `A`). TD is a fixed-point
iteration, and its convergence theory is contraction-based, not
descent-based. This distinction is why divergence (notebook 06) is possible
at all.

## The lambda-Return: Forward View

Define the `n`-step return

```math
G_t^{(n)}=R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^n V(S_{t+n}),
```

interpolating MC (`n = infinity`) and TD(0) (`n = 1`). The **lambda-return**
takes the exponentially weighted mixture:

```math
G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)},
```

(weights sum to 1; for episodic tasks the tail collapses onto the full MC
return with weight `lambda^{T-t-1}`). `lambda = 0` recovers TD(0);
`lambda = 1` recovers MC. In operator language, the forward view applies

```math
\mathcal{T}_\lambda^\pi
=
(1-\lambda)\sum_{n=1}^\infty \lambda^{n-1}(\mathcal{T}^\pi)^n ,
```

a `gamma(1-lambda)/(1-gamma lambda)`-contraction — strictly stronger
contraction as `lambda` grows, which is the operator-theoretic face of "more
lambda = less bootstrap bias."

## Eligibility Traces: Backward View

The forward view is acausal (needs future rewards). The backward view is the
causal implementation: maintain a decaying memory of visited states,

```math
e_t(s)=\gamma\lambda\, e_{t-1}(s)+\mathbb{1}\{S_t=s\},
\qquad
V(s)\leftarrow V(s)+\alpha\,\delta_t\,e_t(s)\quad\forall s .
```

Each new TD error is broadcast backward to recently visited states with
exponentially decaying credit — a credit-assignment mechanism with `O(|S|)`
work per step and no lookahead.

## The Equivalence

**Theorem (offline equivalence).** For a fixed value function within an
episode, the total update of backward TD(`lambda`) equals the total update of
the forward lambda-return algorithm:

```math
\sum_{t=0}^{T-1}\alpha\,\delta_t\,e_t(s)
=
\sum_{t=0}^{T-1}\alpha\big[G_t^\lambda-V(S_t)\big]\mathbb{1}\{S_t=s\} .
```

*Proof mechanism:* telescope the lambda-return through TD errors —

```math
G_t^\lambda-V(S_t)=\sum_{k=t}^{T-1}(\gamma\lambda)^{k-t}\,\delta_k ,
```

(check: expand `G_t^lambda` recursively as
`G_t^lambda = R_{t+1} + gamma[(1-lambda)V(S_{t+1}) + lambda G_{t+1}^lambda]`
and subtract `V(S_t)`), then exchange the double sum over `(t, k)`: crediting
state `S_t` with future `delta_k`s (forward) is identical to crediting
`delta_k` to past states via traces (backward). ∎

Online updates break exactness (later `delta`s use updated `V`); the "true
online TD(lambda)" algorithms restore it with dutch traces.

## Control Versions

```math
\text{SARSA:}\quad
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\big]
```

```math
\text{Q-learning:}\quad
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')-Q(S_t,A_t)\big]
```

SARSA samples `T^pi` for the *current, exploring* policy (on-policy);
Q-learning samples the optimality operator `T` regardless of the behavior
(off-policy). Sutton & Barto's cliff-walk example is the canonical behavioral
separation: SARSA learns the safe path (it prices in its own epsilon-greedy
stumbles), Q-learning learns the optimal-but-cliff-adjacent path while
*performing* worse under exploration. The convergence of Q-learning is
notebook 05; what happens to each under function approximation is notebooks
04 and 06.
