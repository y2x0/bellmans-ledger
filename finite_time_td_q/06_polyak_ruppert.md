# Polyak–Ruppert Averaging And What Practice Actually Runs

## The Idea

Run SA with *larger* (slower-decaying) steps — which track fast but hover
(notebook 01's noise ball) — and report the **average of the iterates**:

```math
\theta_{t+1}=\theta_t+\alpha_t\big(b-A\theta_t+M_{t+1}\big),
\qquad
\alpha_t=\alpha_0\,t^{-\beta},\ \beta\in(1/2,1),
\qquad
\bar\theta_T=\frac{1}{T}\sum_{t=1}^{T}\theta_t .
```

The individual iterate `theta_t` oscillates in a ball of radius
`~ sqrt(alpha_t)`; the average kills the oscillation.

## The Theorem

**(Polyak–Juditsky 1992; linear-SA form.)** Under the notebook 01
assumptions with martingale noise of conditional covariance
`Sigma = E[M M^T]` at `theta*`:

```math
\sqrt{T}\,\big(\bar\theta_T-\theta^*\big)
\ \xrightarrow{d}\
\mathcal{N}\Big(0,\ A^{-1}\,\Sigma\,A^{-\top}\Big),
```

and `A^{-1} Sigma A^{-T}` is the **optimal** asymptotic covariance: no
step-size sequence, no preconditioner, no second-order scheme beats it
(it equals the asymptotic covariance of the intractable "solve the empirical
system exactly" estimator).

*Proof mechanism (the two-line version).* Sum the update over `t` and
divide by `T alpha`... more instructively: write the error recursion as
`e_{t+1} = (I - alpha_t A) e_t + alpha_t M_{t+1}`, unroll, and average. The
average of the noise contributions telescopes into

```math
\bar e_T\ \approx\ \frac{1}{T}\,A^{-1}\sum_{t\le T}M_{t+1}\ +\ \text{(vanishing bias terms)},
```

— averaging converts "noise smoothed through the dynamics with gain
`alpha`" into "noise premultiplied by the *exact* inverse `A^{-1}`", i.e.
the average behaves like the least-squares solution. The CLT for martingales
finishes. The condition `beta < 1` is what makes the bias terms
(`prod (1 - alpha A)` transients) vanish faster than `1/sqrt(T)`; `beta >
1/2` keeps the per-step noise summable in the required sense. ∎

**Consequences worth having verbatim:**

```text
1. optimal variance WITHOUT knowing A: the algorithm never inverts or even
   estimates A, yet matches the estimator that does. Averaging is implicit
   preconditioning.
2. step size becomes a free lunch parameter: any beta in (1/2, 1) gives the
   same limit — robustness that raw SA (notebook 01's trap) conspicuously
   lacks.
3. for TD: averaged TD(0) attains the local asymptotic minimax variance for
   policy evaluation — the strongest optimality statement available for any
   TD variant.
```

## Why Deep RL Runs None Of This

The theory above prescribes: slowly decaying SGD + uniform iterate
averaging. Practice (DQN, PPO, everything in `value_based_deep_rl/`,
`policy_gradient/`) runs: **Adam with near-constant steps, no averaging**
(or at most a *target network*, which is a lagged copy, not an average, and
exists for stability, not variance — `value_based_deep_rl/02`). Reasons,
stated honestly:

```text
1. nonstationarity: the target of learning (the policy's own value, the
   greedy policy) MOVES; averaging toward an old fixed point is averaging
   toward the wrong thing. PR theory is for a fixed root b - A theta = 0.
2. the transient regime is the whole run: deep RL rarely reaches the
   asymptotic regime where PR's advantages bind; constant steps track
   drifting targets (notebook 01: O(alpha) ball around a MOVING theta*(t)
   is often exactly what is wanted).
3. Adam's per-coordinate rescaling is a crude stand-in for the A^{-1}
   preconditioning PR gets by averaging — with no comparable guarantee, but
   effective in the transient.
```

The exceptions prove the rule: **EMA of weights** (Polyak averaging with
exponential forgetting) is standard in modern deep RL / deep learning
(target networks are its 0/1-weight ancestor), precisely because
exponential forgetting is the version of averaging that tolerates a moving
target. The forgetting rate plays `beta`'s role, trading tracking against
variance — the same dial, renamed.

## The Family's Closing Ledger

```text
question                              answer          where
rate of contractive linear SA         O(1/t), tight   01
TD(0) with mixing data                + tau_mix       02
Q-learning, naive                     (1-g)^{-4}      03
Q-learning problem, actual difficulty (1-g)^{-3}      04 = 05 (matched)
statistically optimal SA              PR averaging    06
what practice runs instead            EMA + Adam      06, honestly unpriced
```

## What Remains Open

Nonasymptotic PR theory with Markov noise and optimal constants is recent
and still moving; a satisfying theory of EMA-under-drift (the thing deep RL
actually does) — rate as a function of target drift speed — does not yet
exist. This family's tools are all in place for whoever writes it.
