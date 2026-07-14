# Robbins–Monro And The ODE Method

## The Root-Finding Problem Under Noise

Find `theta*` with `h(theta*) = 0` when only noisy evaluations
`h(theta) + M` are available. Robbins–Monro (1951):

```math
\theta_{t+1}=\theta_t+\alpha_t\big[h(\theta_t)+M_{t+1}\big],
\qquad
\mathbb{E}[M_{t+1}\mid\mathcal{F}_t]=0 .
```

## The Step-Size Conditions

```math
\sum_{t=0}^\infty \alpha_t=\infty,
\qquad
\sum_{t=0}^\infty \alpha_t^2<\infty .
```

```text
sum alpha = inf:      the iterate can travel arbitrarily far; without it the
                      total motion is bounded and theta can stall short of theta*.

sum alpha^2 < inf:    the injected noise variance is summable; the cumulative
                      martingale sum_t alpha_t M_{t+1} converges a.s.
                      (martingale convergence theorem), so noise is
                      asymptotically annihilated.
```

Canonical choice `alpha_t = c/t` satisfies both; constant step sizes satisfy
neither and give convergence *in distribution to a ball* around `theta*` of
radius `O(sqrt(alpha))` — which is why deep RL, which uses constant/Adam
steps, should be expected to hover, not converge.

## Every Tabular RL Update Is An RM Iteration

TD(0), per state, with `F` the Bellman operator:

```math
V(S_t)\leftarrow V(S_t)+\alpha_t\big[\underbrace{R_{t+1}+\gamma V(S_{t+1})}_{\text{sample of }(\mathcal{T}^\pi V)(S_t)}-V(S_t)\big] ,
```

```math
h(V)=\mathcal{T}^\pi V-V,
\qquad
M_{t+1}=\big(R_{t+1}+\gamma V(S_{t+1})\big)-(\mathcal{T}^\pi V)(S_t).
```

The noise is a genuine martingale difference **because the target uses the
actual sampled next state** — the expectation of the sampled backup given
`S_t` is exactly the operator applied at `S_t`. Verifying this unbiasedness
is question 2 of the family README; it fails, e.g., for naive
double-sampling-free residual-gradient methods.

## The ODE Method

The modern convergence tool (Ljung; Kushner–Clark; Borkar). Interpolate the
iterates in "algorithmic time" `sum alpha_t`; as step sizes vanish, the noise
averages out and trajectories track the **mean ODE**:

```math
\dot\theta(t)=h(\theta(t)).
```

**Theorem (informal, Borkar–Meyn).** If (i) `h` is Lipschitz, (ii) RM step
sizes, (iii) martingale noise with conditionally bounded second moments,
(iv) iterates remain bounded a.s., then `theta_t` converges a.s. to a
(possibly sample-path-dependent) internally chain-transitive invariant set of
the ODE. If the ODE has a unique globally asymptotically stable equilibrium
`theta*`, then `theta_t -> theta*` a.s.

For contraction-driven RL updates, `h(theta) = F theta - theta` with `F` a
`gamma`-contraction (in a suitable norm): the ODE is globally exponentially
stable at the unique fixed point — one can take `V(theta) = ||theta - theta*||`
as a Lyapunov function since

```math
\frac{d}{dt}\|\theta-\theta^*\|
\le
\|\mathcal{F}\theta-\mathcal{F}\theta^*\|-\|\theta-\theta^*\|
\le
-(1-\gamma)\|\theta-\theta^*\| .
```

So: **contraction (analysis) + Robbins–Monro (probability) = convergence of
the sampled algorithm.** This one sentence is the skeleton of the TD,
Q-learning, and TvR theorems in notebooks 03–05.

## Asynchronous Updates

RL updates one state (or one `(s,a)`) per step, not the whole vector. The
asynchronous extension (Tsitsiklis 1994) requires:

```text
1. every component is updated infinitely often, and
2. relative update frequencies are bounded,
```

with per-component step-size counters. For sup-norm contractions the
asynchronous iteration still converges — sup-norm contraction gives
componentwise progress that no update ordering can undo. This is precisely
the form of Watkins' Q-learning theorem (notebook 05), and requirement 1 is
where "sufficient exploration" enters as *mathematics* rather than folklore.

## Two-Timescale Iterates

Actor-critic runs two coupled RM iterations:

```math
\text{critic:}\quad w_{t+1}=w_t+\beta_t\,f(w_t,\theta_t),
\qquad
\text{actor:}\quad \theta_{t+1}=\theta_t+\alpha_t\,g(w_t,\theta_t),
```

with `alpha_t / beta_t -> 0` (actor slower). The analysis (Borkar): on the
fast timescale `theta` is quasi-static, so `w_t` tracks the equilibrium
`w*(theta_t)` of the critic ODE; on the slow timescale the actor sees a
converged critic and follows

```math
\dot\theta=g\big(w^*(\theta),\theta\big).
```

This is the formal content of "the improvement step must not outrun the
evaluation step" from `mdp_foundations/05` — and the reason single-timescale
deep actor-critics (where the critic is *not* re-solved) rely on trust
regions and small updates (`policy_gradient/04`, `06`) as a substitute for
timescale separation.

## Rates (retrofit)

Everything above is asymptotic. The finite-time counterparts now exist in
this repository: `finite_time_td_q/01` proves the `O(1/t)` mean-square rate
for contractive linear SA (and the constant-step `O(alpha)` noise ball made
exact), including the `alpha_t = c/t` small-`c` trap that the Robbins–Monro
conditions cannot see; `finite_time_td_q/06` gives Polyak–Ruppert averaging
as the statistically optimal way to run these iterations, and the honest
account of why deep RL runs EMA + Adam instead.
