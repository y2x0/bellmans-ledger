# Problem Setup

## The Interaction Protocol

An agent and an environment alternate. At time `t` the agent observes a state,
emits an action, and receives a reward and a successor state:

```math
S_0,\ A_0,\ R_1,\ S_1,\ A_1,\ R_2,\ S_2,\ \ldots
```

The environment is a controlled Markov chain. The controlling object is the
**transition kernel**:

```math
P(s',r\mid s,a)
=
\Pr\{S_{t+1}=s',\,R_{t+1}=r \mid S_t=s,\,A_t=a\},
```

and the Markov assumption is that this conditional law is invariant to the
entire history given `(s, a)`:

```math
\Pr\{S_{t+1},R_{t+1}\mid S_t,A_t\}
=
\Pr\{S_{t+1},R_{t+1}\mid S_0,A_0,R_1,\ldots,S_t,A_t\}.
```

Everything in this repository is downstream of this one assumption. It is what
lets an infinite-horizon control problem collapse to a functional equation on
`|S|` unknowns rather than a search over history-dependent strategies.

## The Probability Space

Fix a policy `pi` and an initial distribution `mu`. The interaction defines a
unique probability measure on trajectory space (Ionescu-Tulcea):

```math
\mathbb{P}_\mu^\pi
\quad\text{on}\quad
\Omega=(\mathcal{S}\times\mathcal{A}\times\mathbb{R})^{\infty},
```

```math
\mathbb{P}_\mu^\pi(S_0=s_0,A_0=a_0,\ldots)
=
\mu(s_0)\,\pi(a_0\mid s_0)\,P(s_1,r_1\mid s_0,a_0)\,\pi(a_1\mid s_1)\cdots
```

All expectations `E_pi[...]` in later notebooks are expectations under this
measure. When two algorithms disagree about "the" expected return, they are
usually integrating against different measures on this same space — different
`pi` in the sampling distribution vs. the target of learning is exactly the
on-policy / off-policy distinction.

## The Objective

The **return** from time `t` is the discounted sum of future rewards:

```math
G_t=\sum_{k=0}^{\infty}\gamma^k R_{t+k+1},
\qquad
|G_t|\le \frac{R_{\max}}{1-\gamma}.
```

The bound holds pointwise because rewards are bounded and `gamma < 1`; this is
what makes every value function below finite and every interchange of limit
and expectation below legal (dominated convergence).

The control problem:

```math
\text{find }\pi^\*
\quad\text{maximizing}\quad
J(\pi)=\mathbb{E}_\mu^\pi[G_0]
\quad\text{simultaneously for all }\mu.
```

That a single `pi*` can be optimal for **all** starting states — and can be
chosen stationary, Markov, and deterministic — is a theorem, not a
definition. It is proved in `mdp_foundations/01`.

## The Three Regimes

```text
planning:
    P and r known, S small        -> exact operators, mdp_foundations/
learning:
    P and r unknown, S small      -> sampled operators, stochastic_approximation/
approximation:
    S enormous or continuous      -> projected/parameterized operators,
                                     value_based_deep_rl/, policy_gradient/,
                                     search_and_planning/, distributional_rl/
```

The mathematical content of deep RL is what survives the passage from row one
to row three, and the failure modes are what does not.
