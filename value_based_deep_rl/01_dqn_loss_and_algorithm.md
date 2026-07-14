# DQN: The Loss And The Algorithm

```text
Mnih et al. 2015, Nature. papers/dqn-mnih-2015.pdf
```

## The Object Being Learned

A convnet `Q(s, . ; theta)` mapping a state (four stacked, downsampled
`84 x 84` luminance frames) to one Q-value **per action** — so a single
forward pass evaluates the `max_a'` in the target; the architecture makes
the optimality backup `O(1)` forward passes rather than `O(|A|)`.

The state construction matters mathematically: a single Atari frame is not
Markov (velocities invisible, sprite flicker); the paper stacks 4 frames and
max-pools consecutive frames to *manufacture* approximate Markovness, i.e.
to earn the right to use the MDP machinery at all. Formally the paper treats
the full action-observation sequence as the state and notes the stack is a
fixed-length surrogate.

## The Loss

```math
L_i(\theta_i)
=
\mathbb{E}_{(s,a,r,s')\sim U(D)}
\Big[\big(\underbrace{r+\gamma\max_{a'}Q(s',a';\theta_i^-)}_{y\text{, target}}-\ Q(s,a;\theta_i)\big)^2\Big]
```

```text
D          replay buffer: last 1M transitions e_t = (s_t, a_t, r_t, s_{t+1})
U(D)       uniform minibatch sampling (batch 32)
theta_i^-  target network parameters, cloned from theta every C ~ 10^4 steps,
           held fixed in between
```

Gradient (semi-gradient — the target is treated as a constant even though it
depends on parameters through `theta^-`):

```math
\nabla_{\theta_i}L
=
\mathbb{E}\Big[\big(r+\gamma\max_{a'}Q(s',a';\theta_i^-)-Q(s,a;\theta_i)\big)\,\nabla_{\theta_i}Q(s,a;\theta_i)\Big].
```

Setting `theta^- = theta_{i-1}` and one sample per update recovers exactly
tabular Q-learning when `Q` is a table — the algorithm is Watkins' iteration
(`stochastic_approximation/05`) with the table replaced by `F = convnet` and
the update distribution replaced by `U(D)`.

## The Three Instabilities, And The Paper's Answer To Each

The paper names its enemies precisely (citing Tsitsiklis–Van Roy):

```text
instability                                   answer
-------------------------------------------   ------------------------------
(a) sequential correlation of observations    replay: uniform draws from 1M
                                              past transitions decorrelate
                                              the regression batch

(b) policy shift: small Q changes move the    replay again: the effective
    greedy policy, hence the data             update distribution is an
    distribution -> feedback loops            average over many past
                                              policies, smoothing the
                                              distribution shift

(c) target correlation: y moves whenever      target network: y is built from
    theta moves (bootstrapping through        frozen theta^-, so between
    the same network)                         syncs the problem is plain
                                              supervised regression toward a
                                              FIXED target
```

Notebook 02 gives each of these its operator-theoretic reading.

## Secondary Controls

```math
\text{reward clipping: } r\mapsto \mathrm{clip}(r,-1,1)
\qquad
\text{TD-error clipping: } \delta\mapsto\mathrm{clip}(\delta,-1,1)
```

Reward clipping equalizes gradient scale across 49 games (one learning rate
for all) at the cost of distorting the objective — the agent maximizes reward
*frequency*, not magnitude. TD-error clipping is equivalent to switching from
squared to absolute loss outside `[-1, 1]` — i.e. a Huber loss — bounding the
gradient contribution of any single transition. Exploration is
epsilon-greedy, annealed `1.0 -> 0.1` over 1M frames; optimizer RMSProp.

## The Algorithm

```text
initialize D (capacity N), Q with random theta, target Q-hat with theta^- = theta
for each episode:
    for t = 1..T:
        a_t = epsilon-greedy(Q(phi(s_t), . ; theta))
        execute a_t; observe r_t, x_{t+1}; store (phi_t, a_t, r_t, phi_{t+1}) in D
        sample uniform minibatch (phi_j, a_j, r_j, phi_{j+1}) from D
        y_j = r_j                                    if terminal
            = r_j + gamma max_a' Q-hat(phi_{j+1},a'; theta^-)   otherwise
        SGD step on (y_j - Q(phi_j, a_j; theta))^2
        every C steps: theta^- <- theta
```

## Results, And The Two Claims That Matter

Same architecture + hyperparameters across 49 Atari games, pixels and score
as only inputs: above the best prior method on 43 games; `>= 75%` of a
professional human tester's normalized score on 29. Normalization:

```math
\text{score}_{\text{norm}}
=100\times\frac{\text{agent}-\text{random}}{\text{human}-\text{random}} .
```

The paper's ablation (Extended Data Table 3) is the scientifically important
result: with replay and target network removed the agent collapses (e.g.
Breakout `316.8 -> 3.2`). **Neither component is an optimization nicety; each
is load-bearing against a named instability.** Where DQN still fails —
Montezuma's Revenge at ~0% — the binding constraint is exploration and
long-horizon credit, which nothing in the loss addresses.

## Reading The Paper Against The Theory

```text
theorem it would like to invoke     which hypothesis DQN violates
--------------------------------    -------------------------------------
Watkins (tabular Q-learning)        tabular updates (uses a convnet)
Tsitsiklis–Van Roy (linear TD)      linearity, on-policy mu, evaluation-only
Banach on Pi T                      no norm in which Pi_replay T contracts
```

There is still no general convergence proof for DQN. Its empirical stability
is best understood as: target freezing approximately restores a fixed-point
splitting (notebook 02), replay approximately restores a stationary update
distribution, and clipping bounds the damage of the remaining mismatch.
