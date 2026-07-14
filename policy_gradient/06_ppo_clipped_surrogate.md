# PPO: The Clipped Surrogate Objective

```text
Schulman, Wolski, Dhariwal, Radford, Klimov 2017. papers/ppo-schulman-2017.pdf
```

## The Objective

With the probability ratio

```math
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)},
\qquad r_t(\theta_{\text{old}})=1,
```

the surrogate of notebook 03 is `L^{CPI} = E-hat[r_t A-hat_t]` — unconstrained
maximization of which produces "excessively large" updates. PPO's objective
(paper eq. 7):

```math
\boxed{\
L^{\mathrm{CLIP}}(\theta)
=
\hat{\mathbb{E}}_t\Big[\min\big(\,r_t(\theta)\hat A_t,\ \ \mathrm{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat A_t\big)\Big]\ },
\qquad \epsilon=0.2 .
```

## Reading The min–clip Composition Exactly

Per timestep, four regimes:

```text
A_t > 0, r_t <= 1+eps :  objective = r_t A_t        (normal ascent incentive)
A_t > 0, r_t >  1+eps :  objective = (1+eps) A_t    (constant: gradient = 0;
                                                     no further incentive to
                                                     raise this action's prob)
A_t < 0, r_t >= 1-eps :  objective = r_t A_t        (normal descent incentive)
A_t < 0, r_t <  1-eps :  objective = (1-eps) A_t    (constant: gradient = 0;
                                                     no incentive to keep
                                                     crushing the prob)
```

Because of the outer `min`, clipping engages **only in the direction of
improvement**: if the ratio has moved so that the objective got *worse*
(e.g. `r_t` fell while `A_t > 0`), the unclipped term is smaller and is kept
— the update still sees the full corrective gradient. Hence:

```math
L^{\mathrm{CLIP}}(\theta)\ \le\ L^{\mathrm{CPI}}(\theta),
\qquad
L^{\mathrm{CLIP}}=L^{\mathrm{CPI}}\ \text{to first order at }\theta_{\text{old}},
```

a **pessimistic lower bound** on the surrogate that removes only the
incentive to leave the ratio interval `[1-eps, 1+eps]` — a pointwise,
first-order imitation of the TRPO minorant `L - C KL^max` (notebook 04),
with the per-state trust region enforced on the sampled action's ratio
instead of on a KL. Note carefully what is *not* guaranteed: clipping bounds
the incentive, not the ratio itself (a large step can still jump past the
interval — the gradient just vanishes rather than pulls back), and there is
no monotonic improvement theorem for PPO. The theory is inherited
heuristically from notebook 04, and empirically it inherits well.

## The Full Loss And Algorithm

Shared policy/value network, entropy bonus (paper eq. 9):

```math
L_t(\theta)
=
\hat{\mathbb{E}}_t\Big[L_t^{\mathrm{CLIP}}(\theta)
-c_1\big(V_\theta(s_t)-V_t^{\mathrm{targ}}\big)^2
+c_2\,\mathcal{S}[\pi_\theta](s_t)\Big],
```

advantages by truncated GAE (notebook 05).

```text
Algorithm (actor-critic style, paper Alg. 1):
    for iteration = 1, 2, ...:
        for actor = 1..N:  run pi_old for T steps; compute A-hat_1..A-hat_T
        optimize L wrt theta: K epochs of minibatch Adam over the NT samples
        theta_old <- theta
```

The point of the whole construction is the line "K epochs": the clipped
objective is what makes **reusing a batch for multiple epochs** safe-ish,
which is where PPO's sample-efficiency-per-wallclock comes from. Signature
hyperparameters (paper Tables 3 & 5):

```text
MuJoCo:  T=2048, Adam 3e-4, 10 epochs, minibatch 64, gamma .99, lambda .95
Atari:   T=128, N=8 actors, 3 epochs, minibatch 32x8, eps = 0.1 * alpha-anneal,
         c1 = 1, c2 = 0.01, learning rate 2.5e-4 * alpha-anneal
```

## The Adaptive-KL Variant (paper §4)

`L^{KLPEN} = E-hat[r_t A_t - beta KL[pi_old, pi]]` with

```text
d = E-hat[KL];   d < d_targ/1.5  ->  beta <- beta/2
                 d > d_targ*1.5  ->  beta <- beta*2
```

— included as a baseline; the paper reports it underperforms clipping. This
is the empirical resolution of TRPO's penalty-vs-constraint dilemma: neither
a fixed nor an adapted `beta` matched a hard per-sample ratio interval.

## The Evidence (paper §6)

```text
ablation (7 MuJoCo tasks, normalized score):
    no clipping/penalty  -0.39     <- worse than random on HalfCheetah
    clip eps=0.2          0.82     <- best
    adaptive KL           0.68–0.74
    fixed KL              0.62–0.72

continuous control: beats TRPO, CEM, vanilla PG, A2C(+TR) on almost all tasks
Roboschool humanoid: learns run/steer/get-up at 50–100M timesteps
Atari (49 games):   beats A2C decisively (30/49 on all-training metric),
                    ~ ACER on final performance, far simpler
```

## Position In The Repository's Coordinates

```math
\mathcal{T}: \text{none on the policy — MM-style ascent on a surrogate};
\qquad
d: \text{per-state ratio interval, a poor man's KL ball};
\qquad
\mathcal{F}: \text{any differentiable } \pi_\theta .
```

PPO is also the algorithm that escaped the field: RLHF for language models is
PPO with `pi_old` the reference/previous policy, a learned reward model as
`r`, and an explicit KL-to-reference penalty restored to the reward — i.e.
notebook 04's penalty form and notebook 06's clipping running simultaneously.
The mathematics in this family is exactly the mathematics of aligning LLMs.
