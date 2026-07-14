# Mathematics of Reinforcement Learning

This notebook family asks:

```text
What fixed point is being computed, in what metric does its operator contract,
and what happens when the fixed point can only be sampled or approximated?
```

Every algorithm in this repository is a way of finding, approximating, or
sampling the fixed point of some Bellman operator. Methods differ in exactly
three coordinates:

```math
\big(\ \mathcal{T},\ d,\ \mathcal{F}\ \big)
```

```text
T:
    which operator (evaluation T^pi, optimality T*, distributional T^pi,
    projected Pi T, empirical/sampled T-hat)

d:
    which metric the operator contracts in (sup-norm, weighted L2,
    maximal Wasserstein) -- and whether it contracts at all

F:
    which function class approximates the fixed point (table, linear span,
    deep network, categorical distribution, search tree statistics)
```

Each notebook should answer:

```text
1. What object is the fixed point (value, action-value, policy, measure)?
2. What operator generates it, and is the operator a contraction? In what metric?
3. What is sampled vs. computed exactly?
4. What approximation error does the function class introduce, and where does it go?
5. What convergence guarantee survives, under what conditions?
6. What failure mode follows when a condition is violated?
```

## Source Texts

| Reading | Where it lives here |
|---|---|
| Sutton & Barto, *RL: An Introduction* 2e, ch. 1–6 | `mdp_foundations/`, `stochastic_approximation/` |
| Bellman 1957, *A Markovian Decision Process* | `mdp_foundations/07` |
| Szepesvári 2010, *Algorithms for RL* | proofs threaded throughout; reading map in `unifying_view/` |
| Mnih et al. 2015 (DQN, Nature) | `value_based_deep_rl/` |
| Schulman et al. 2017 (PPO) | `policy_gradient/` |
| Silver et al. 2016 (AlphaGo, Nature) | `search_and_planning/` |
| Bellemare, Dabney, Munos 2017 (C51) | `distributional_rl/` |

PDFs are in `papers/`.

## Folder Map

```text
00_problem_setup.md            the agent-environment interface as a probability model

mdp_foundations/               exact dynamic programming: the model is known
    01  MDP formalism, policy classes, sufficiency of stationary Markov policies
    02  return criteria, existence and measurability of value functions
    03  Bellman operators: monotonicity, contraction, both proved
    04  Banach fixed point theorem, a priori / a posteriori error bounds
    05  policy improvement theorem, policy iteration, Newton's method view
    06  linear programming formulation, occupancy measures, dual LP
    07  Bellman 1957: the original functional equation

stochastic_approximation/      the model is unknown: learn from samples
    01  Robbins–Monro, the ODE method, two-timescale iterates
    02  Monte Carlo estimation, ordinary vs. weighted importance sampling
    03  TD(0) and TD(lambda): forward/backward views, eligibility traces
    04  convergence with linear function approximation: Tsitsiklis–Van Roy
    05  Q-learning convergence: Watkins' theorem via asynchronous SA
    06  the deadly triad: Baird's counterexample, divergence mechanics

value_based_deep_rl/           the Bellman backup as regression (DQN)
    01  DQN: the loss, and which instability each component addresses
    02  replay and target networks as distribution and correlation control
    03  overestimation bias: the max as a biased estimator, Double Q
    04  failure modes

policy_gradient/               direct ascent on J(theta) (PPO)
    01  the policy gradient theorem, both proofs
    02  baselines, variance, and why the value function is the right baseline
    03  the performance difference lemma (Kakade–Langford)
    04  trust regions: the TRPO monotonic improvement bound
    05  generalized advantage estimation: the bias-variance dial
    06  PPO: the clipped surrogate as a pessimistic lower bound
    07  failure modes

search_and_planning/           amortizing the Bellman backup into a tree (AlphaGo)
    01  bandits and UCB1: the regret bound behind tree search
    02  MCTS and UCT: value backup as recursive bandit
    03  AlphaGo: four networks + search, every training equation

distributional_rl/             the return as a random variable (C51)
    01  Wasserstein metrics and optimal transport background
    02  the distributional Bellman operator: gamma-contraction in d_p-bar
    03  control is not a contraction: the counterexamples
    04  C51: categorical parametrization, the projection, the KL surrogate
    05  quantile regression: minimizing W1 directly, Cramér geometry

unifying_view/
    01  every method as (operator, metric, function class)
    02  reading map: Sutton–Barto vs. Szepesvári, theorem index

--- expansion (see PLAN.md for the per-file contract) ---

concentration_toolkit/         the five probabilistic instruments, proved
finite_time_td_q/              rates for TD/Q; the closed minimax chapter
regret_and_exploration/        optimism, UCBVI proved, lower bounds, PSRL, IDS
linear_mdps_and_completeness/  when function approximation provably works
offline_rl_and_ope/            evaluation and pessimism from fixed data
regularized_mdps_and_duality/  entropy/KL: soft operators, MD, DICE duality
global_convergence_of_pg/      Łojasiewicz, NPG rates, the conservation law
rlhf_mathematics/              Bradley–Terry -> closed form -> DPO/BoN/Goodhart
average_reward/                gain/bias, Laurent series, Blackwell, span norm
total_reward_ssp/              gamma = 1: properness, weighted norms, timeouts
lqr_and_continuous_control/    Riccati, PG on LQR, HJB/viscosity, path integrals
stochastic_games/              Shapley 1953, cycling, CFR, self-play theory
pomdps_and_information_state/  beliefs, alpha-vectors, PSPACE, PSRs, AIS
model_based/                   simulation lemma, certainty equivalence, MuZero
```

## Notation

```math
M=(\mathcal{S},\mathcal{A},P,r,\gamma),\qquad
P(\cdot\mid s,a)\in\Delta(\mathcal{S}),\qquad
r:\mathcal{S}\times\mathcal{A}\to[-R_{\max},R_{\max}],\qquad
\gamma\in[0,1).
```

Value functions are `v` (state) and `q` (state-action); operators are
calligraphic; `pi` is a policy; `d` is a metric; hats denote sampled or
empirical quantities; `Pi` is a projection.
