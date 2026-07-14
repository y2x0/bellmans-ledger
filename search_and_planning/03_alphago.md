# AlphaGo: Four Networks And A Tree

```text
Silver et al. 2016, Nature. papers/alphago-silver-2016.pdf
```

## The Problem Size

Perfect-information games have an optimal value function `v*(s)` computable
in principle over `~ b^d` sequences; Go has `b ~ 250`, `d ~ 150`. The paper's
two structural moves: cut **depth** by a learned `v(s) ~ v*(s)` at truncated
leaves, cut **breadth** by sampling from a learned prior `p(a|s)`. Both were
believed intractable for Go before deep networks.

## The Training Pipeline (four functions, four losses)

**1. SL policy network** `p_sigma(a|s)` — 13-layer convnet on a
`19 x 19 x 48` feature stack, trained on 30M KGS expert positions by
maximum likelihood:

```math
\Delta\sigma\ \propto\ \frac{\partial\log p_\sigma(a\mid s)}{\partial\sigma} .
```

57.0% held-out move-prediction accuracy (prior SOTA 44.4%); 3 ms per
evaluation. Small accuracy gains translated to large playing-strength gains
(paper Fig. 2a).

**2. Fast rollout policy** `p_pi(a|s)` — linear softmax over local pattern
features: 24.2% accuracy at **2 microseconds** — a 1500x speed ratio that is
the entire reason rollouts are feasible in the search.

**3. RL policy network** `p_rho` — initialized `rho = sigma`, then REINFORCE
(`policy_gradient/01`) in self-play against a **randomly sampled previous
iteration** (opponent pool prevents overfitting to the current self):

```math
\Delta\rho\ \propto\ \frac{\partial\log p_\rho(a_t\mid s_t)}{\partial\rho}\,z_t,
\qquad
z_t=\pm 1\ \text{terminal win/loss from the player's perspective},
```

i.e. policy gradient with the raw terminal return and no critic (baseline 0;
the paper's follow-up work adds `v(s)` as baseline). Result: 80% wins vs the
SL network; **85% wins vs Pachi (100k-rollout MCTS) with no search at all** —
prior convnet-only baselines won ~11%.

**4. Value network** `v_theta(s) ~ v^{p_rho}(s)` — same trunk, scalar tanh
head, regression on outcomes:

```math
\Delta\theta\ \propto\ \frac{\partial v_\theta(s)}{\partial\theta}\,\big(z-v_\theta(s)\big).
```

**The overfitting lesson (worth memorizing).** Trained on positions from
complete KGS games: train/test MSE `0.19 / 0.37` — memorized games, because
successive positions are near-duplicates sharing one label (the correlated-
sample failure of naive regression). Fix: a fresh dataset of **30M positions,
each from a distinct self-play game** — train/test `0.226 / 0.234`.
Decorrelating the regression set, not changing the model, closed the gap.
A single `v_theta` forward pass matched the accuracy of full `p_rho`
rollouts at **15,000x less compute** (paper Fig. 2b).

## The Search (asynchronous PUCT-MCTS)

Edge statistics `{P(s,a), N(s,a), Q(s,a)}`. Selection:

```math
a_t=\arg\max_a\ \big(Q(s_t,a)+u(s_t,a)\big),
\qquad
u(s,a)\ \propto\ \frac{P(s,a)}{1+N(s,a)},
```

prior-scaled, count-decaying exploration (the PUCT rule of notebook 01). At
leaf expansion the SL network writes the priors:
`P(s,a) = p_sigma(a|s)`. Leaf evaluation mixes two estimators:

```math
V(s_L)=(1-\lambda)\,v_\theta(s_L)+\lambda\,z_L,
```

`z_L` the outcome of a `p_pi` rollout to terminal. Backup, over the `n`
simulations traversing each edge:

```math
N(s,a)=\sum_{i=1}^n \mathbb{1}(s,a,i),
\qquad
Q(s,a)=\frac{1}{N(s,a)}\sum_{i=1}^n\mathbb{1}(s,a,i)\,V(s_L^i).
```

Final move: **most-visited** root action (visit counts are lower-variance
than Q at low counts). Compute: 40 threads, 48 CPUs, 8 GPUs (distributed:
1202 CPUs, 176 GPUs), simulations on CPU with network evaluations batched
asynchronously on GPU.

## Two Findings With Theoretical Content

```text
1. The WEAKER SL policy beat the stronger RL policy as the search prior,
   while the RL-DERIVED value function beat the SL-derived one as evaluator.
   Interpretation: a prior must cover the diverse beam of moves search needs
   to consider (humans provide breadth); an evaluator must be calibrated
   about the strongest available play (RL self-play provides that). Priors
   and evaluators are different statistical objects with different optimal
   training distributions.

2. lambda = 0.5 beat lambda = 0 (value net only) and lambda = 1 (rollouts
   only), each variant already beating all prior programs. The two
   evaluators err differently (learned bias vs rollout-policy bias +
   variance); mixing them is variance/bias hedging, cf. GAE's lambda.
```

## Results

```text
tournament: 494/495 (99.8%) vs other Go programs; 77–99% giving 4 handicap
            stones; distributed AlphaGo: 100% vs programs, 77% vs
            single-machine AlphaGo
match:      def. Fan Hui (2p, European champion) 5–0, October 2015 --
            first professional defeat in even 19x19 Go, previously
            forecast a decade away
```

## The Aftermath, In This Repository's Language

AlphaGo Zero / AlphaZero (2017) close the GPI loop of notebook 02: delete
human data (`p_sigma`), rollouts (`p_pi`), and the separate RL stage; train
one network `(p, v)` toward the search's own outputs
(`pi_MCTS ~ N^{1/tau}`, `z`), making **MCTS itself the policy improvement
operator** inside generalized policy iteration. MuZero (2019) removes the
known simulator, learning the model implicitly. The 2016 paper is the
transitional fossil: every later simplification is visible in it as a moving
part it still carried.

Both threads now have their own theory in this repository: *why self-play
does not cycle* (zero-sum + symmetry + a genuine improvement operator, vs
the Shapley cycling of naive mutual best response) is
`stochastic_games/04`; *what a learned latent model owes the search*
(value equivalence — MuZero's training losses as sampled
Bellman-agreement constraints) is `model_based/04`.
