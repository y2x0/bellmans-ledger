# TD(0) With Linear Features: Finite-Time Bounds

## Instantiating The Master Theorem

TD(0) with features `phi(s) in R^d`, `||phi|| <= 1`, is linear SA
(`stochastic_approximation/04`) with

```math
A=\Phi^\top D(I-\gamma P_\pi)\Phi,
\qquad
b=\Phi^\top D r_\pi,
\qquad
M_{t+1}=\big(\delta_t-\mathbb{E}[\delta_t\mid\mathcal{F}_t]\big)\phi(S_t).
```

The drift constant was computed there:

```math
x^\top Ax\ \ge\ (1-\gamma)\,\lambda_{\min}\!\big(\Phi^\top D\Phi\big)\,\|x\|^2
\ =:\ \mu\,\|x\|^2
\qquad\text{(}D=\mathrm{diag}(\mu_{\text{stat}})\text{, on-policy)} .
```

Noise scale: `|delta_t| <= Rmax + 2 ||theta|| =>` near `theta*`,
`sigma^2 = O((Rmax + ||theta*||)^2)`.

**Corollary (i.i.d. sampling).** Sampling `S_t ~ mu_stat` independently each
step (the "i.i.d. model"), TD(0) with `alpha_t = 2/(mu(t + t_0))` obeys

```math
\mathbb{E}\big\|\theta_t-\theta^*\big\|^2
\ =\
O\Big(\frac{\sigma^2}{(1-\gamma)^2\,\lambda_{\min}^2\ t}\Big),
```

and with constant `alpha`: geometric decay to a ball of radius
`sqrt(alpha) sigma / sqrt(mu)`. Both are notebook 01 verbatim; the RL
content is entirely in the values of `mu` and `sigma`.

**Reading the constants.** Two deficiencies of the *problem*, not the proof:

```text
(1-gamma)^{-2}:  the drift is weak when gamma ~ 1 — the same horizon factor
                 as everywhere (mdp_foundations/02);
lambda_min^{-2}: poorly conditioned features slow learning even with exact
                 linear realizability. Feature design IS conditioning.
```

## Markov Sampling: The Honest Setting

Real TD walks the chain: `S_{t+1} ~ P_pi(. | S_t)`. Consecutive noise terms
correlate, and `E[M_{t+1} | F_t] != 0` relative to the *stationary* mean the
analysis wants. The repair (notebook 01's coupling argument) in detail:

**The mixing lemma.** For a uniformly ergodic chain with mixing time

```math
\tau_{\mathrm{mix}}(\epsilon)=\min\{t:\ \sup_s\ \mathrm{TV}\big(P^t(\cdot\mid s),\ \mu_{\text{stat}}\big)\le\epsilon\},
```

condition the offending cross-term `tau = tau_mix(alpha)` steps in the past:

```math
\mathbb{E}\big[e_t^\top g(\theta_{t},\,S_t)\big]
=
\underbrace{\mathbb{E}\big[e_{t-\tau}^\top g(\theta_{t-\tau},S_t)\big]}_{\le\ \|e\|\cdot O(\alpha\tau)\ \text{by TV-closeness}}
+\underbrace{\text{drift of }(e,\theta)\text{ over }\tau\text{ steps}}_{O(\alpha\tau)\cdot(\ldots)}
```

— given `S_{t-tau}`, the law of `S_t` is within `alpha` of stationary
(choice of `tau`), so the conditional expectation nearly vanishes; and the
parameters moved only `O(alpha tau)` in the interim (bounded updates). Both
error terms carry `alpha tau_mix`. ∎

**Theorem (Bhandari–Russo–Singal 2018, shape).** Under Markov sampling,
projected TD(0) with constant step `alpha`:

```math
\mathbb{E}\big\|\theta_t-\theta^*\big\|^2
\ =\
O\Big((1-\alpha\mu)^t + \frac{\alpha\,\sigma^2\ \tau_{\mathrm{mix}}\log(1/\alpha)}{\mu}\Big),
```

i.e. the i.i.d. bound with the noise ball inflated by
`tau_mix log(1/alpha)`; with decaying steps,
`O(tau_mix log t / ((1-gamma)^2 t))`.

**Load-bearing audit.**

```text
uniform ergodicity     -> the TV-coupling step; periodic or nearly-reducible
                          chains (tau_mix huge) genuinely learn slower —
                          this is a fact about the data, not the proof
projection to a ball   -> keeps ||theta|| bounded so the drift-over-tau
                          term is controllable; removable with more work
on-policy D            -> mu > 0. Off-policy: mu can be <= 0 and NO rate
                          exists because the limit doesn't (stochastic_
                          approximation/06)
```

## TD(lambda), In One Remark

The same template applies to TD(`lambda`) with eligibility traces; the
operator's contraction modulus improves to
`gamma(1-lambda)/(1-gamma lambda)` (`stochastic_approximation/03`), so `mu`
improves, but `sigma^2` grows with trace length — the finite-time bound
exhibits the bias–variance tradeoff as an explicit
`lambda`-dependent constant, optimized in the interior. The asymptotic
theory cannot even *see* this tradeoff (all `lambda` converge); rates can.

## What Remains Open

Tight *instance-dependent* (not worst-case-over-features) TD rates; the
correct nonasymptotic theory for TD with target networks (the outer-loop
splitting of `value_based_deep_rl/02` — partial results only); anything
nonlinear.
