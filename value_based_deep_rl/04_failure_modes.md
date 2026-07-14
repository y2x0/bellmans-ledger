# Failure Modes Of Value-Based Deep RL

Each failure mode below is a violated hypothesis from the foundations
families, with its empirical signature.

## 1. Triad Divergence (violated: norm-matched update distribution)

```text
hypothesis broken:  mu stationary for the bootstrapped operator
                    (stochastic_approximation/04)
signature:          Q-values grow without bound; loss explodes late in
                    training ("soft divergence" usually self-corrects when
                    values saturate the reward-clipped scale; hard divergence
                    does not)
levers:             larger target-sync gap, smaller learning rate, Double Q,
                    shorter bootstrap (n=1), on-policy-ish buffers
```

## 2. Overestimation Spiral (violated: unbiased operator estimate)

```text
hypothesis broken:  targets estimate T Q unbiasedly (notebook 03)
signature:          predicted Q rises while actual return stalls; the gap
                    IS measurable — always plot E[Q] vs achieved discounted
                    return on held-out states
levers:             Double DQN; ensembles/min-of-two; distributional heads
                    (distributional_rl/04 — C51's projection bounds targets)
```

## 3. State Aliasing (violated: Markov property)

```text
hypothesis broken:  the input is a sufficient state (00_problem_setup)
signature:          value oscillation between perceptually identical but
                    situationally different states; policies that "flail"
                    where memory is needed
levers:             frame stacking (DQN's fix), recurrence (DRQN), belief
                    states. No value-function trick repairs a non-Markov
                    state: the fixed-point equation itself is ill-posed.
theory (retrofit):  the missing theory now exists in this repo — the
                    belief-MDP reduction and its price
                    (pomdps_and_information_state/01, 03), and the
                    approximate-information-state bound that prices frame
                    stacks and RNN states as (eps, delta)-summaries with a
                    value-loss guarantee (pomdps_and_information_state/05).
```

## 4. Reward Clipping Objective Distortion (deliberate hypothesis violation)

```text
what is optimized:  frequency of positive rewards, not score
signature:          indifference between 1-point and 100-point events
                    (e.g. Ms. Pac-Man pellets vs ghosts)
levers:             PopArt normalization (learn unclipped, normalize targets
                    adaptively); per-game reward scaling
```

## 5. Exploration Starvation (violated: infinite visitation, Watkins cond. 3)

```text
hypothesis broken:  every (s,a) updated infinitely often
                    (stochastic_approximation/05)
signature:          Montezuma's Revenge ~ 0% human: epsilon-greedy's
                    expected time to execute a specific k-step sequence is
                    (1/epsilon)^k — exponential; no loss modification helps
levers:             count/pseudo-count bonuses, intrinsic motivation, RND,
                    Go-Explore; mathematically: change mu, not T
theory (retrofit):  the exponential claim is now a proved theorem — the
                    combination-lock lower bound Omega(A^H) for
                    epsilon-greedy, and the optimism machinery that opens
                    the same lock in poly time, are in
                    regret_and_exploration/04 (with the full UCBVI proof
                    in 02–03 and the eluder/deep-RL gap in 07).
```

## 6. Bootstrap Error Propagation From Terminal Mishandling

```text
a small one with outsized effect: treating time-limit truncation as true
termination sets the target to r instead of r + gamma max Q(s',.) at states
that are NOT absorbing — a biased operator at exactly the horizon boundary.
correct handling (bootstrap-on-truncation) is a one-line fix and a
recurring real-world bug.
```

## The Meta-Failure

DQN's stack (replay + frozen targets + clipping + epsilon-greedy) is tuned
for the regime "dense-ish rewards, short-ish credit horizons, discrete
actions, cheap simulation." Each departure from that regime removes the
slack that lets the missing theorems go unnoticed. The successor families —
policy gradients (`policy_gradient/`), search hybrids
(`search_and_planning/`), distributional heads (`distributional_rl/`) — can
each be read as restoring one missing guarantee: on-policy contraction-free
ascent, model-based lookahead replacing bootstrap depth, and a better-behaved
regression target, respectively.
