# NPG With Function Approximation: The Transfer-Error Theorem

## The Question

Notebook 02's `log|A|/(eta K)` was exact-gradient, tabular. Real NPG fits
a critic in a restricted class and estimates from samples under the
*current* policy's distribution. What survives?

## The Setup (Q-NPG shape)

At each iteration, fit a compatible-style critic by least squares under
the current visitation `nu_k` (state-actions from `d^{pi_k}` with
`pi_k`'s actions):

```math
w_k\ \approx\ \arg\min_w\ \mathbb{E}_{\nu_k}\Big[\big(w^\top\psi_k(s,a)-A^{\pi_k}(s,a)\big)^2\Big],
\qquad
\theta_{k+1}=\theta_k+\eta\,w_k .
```

Two error sources, named:

```math
\varepsilon_{\mathrm{stat}}:\ \text{finite-sample excess risk of the regression under }\nu_k;
```

```math
\varepsilon_{\mathrm{approx}}
=
\min_w\ \mathbb{E}_{\nu_k}\big[(w^\top\psi_k-A^{\pi_k})^2\big]
\quad\text{(the class simply cannot express }A^{\pi_k}\text{).}
```

## The Theorem (Agarwal–Kakade–Lee–Mahajan 2021, shape)

```math
\min_{k\le K}\ J(\pi^*)-J(\pi_k)
\ \le\
\underbrace{\frac{\log|\mathcal{A}|}{\eta K}+\frac{\eta\,G^2}{(1-\gamma)^2}}_{\text{notebook 02's MD terms}}
\ +\
\underbrace{\sqrt{\frac{|\mathcal{A}|\ \kappa}{(1-\gamma)^{3}}\ \big(\varepsilon_{\mathrm{stat}}+\varepsilon_{\mathrm{approx}}\big)}}_{\text{the transfer term}},
```

where the **distribution-mismatch (transfer) coefficient**

```math
\kappa
\ =\
\sup_k\ \Big\|\frac{d^{\pi^*}\otimes\pi^*}{\nu_k}\Big\|_\infty\text{-type ratio}
\qquad\big(\text{or its weaker }L_2(\nu_k)\text{ form}\big)
```

measures how well the *data the critic was fit on* covers the
*comparator's* state-actions.

## Where The Mismatch Re-Enters — The Proof's One New Step

Notebook 02's MD telescope needs the per-state pairing
`<pi*(.|s) - pi_k(.|s), A^{pi_k}(s,.)>` averaged under `d^{pi*}`. With an
approximate critic, the pairing's error is

```math
\mathbb{E}_{d^{\pi^*}\!,\pi^*\ \text{and}\ \pi_k}\big[\,|A^{\pi_k}-\hat A_k|\,\big]
\ \le\
\sqrt{\kappa}\ \cdot\
\underbrace{\big\|A^{\pi_k}-\hat A_k\big\|_{L_2(\nu_k)}}_{=\sqrt{\varepsilon_{\mathrm{stat}}+\varepsilon_{\mathrm{approx}}}},
```

by Cauchy–Schwarz / change of measure — **the critic's error is certified
where the data was, and the theorem needs it where `pi*` goes.** The
mismatch coefficient is the exchange rate, and it is irreducible in this
proof architecture for the same reason it was irreducible in offline RL
(`offline_rl_and_ope/04`): errors can only be controlled on distributions
one can sample.

```text
the audit, and the punchline of the whole family:

  notebook 01:  vanilla PG — mismatch appears in the OPTIMIZATION
                (Łojasiewicz constant): plateaus.
  notebook 02:  exact NPG — mismatch VANISHES: geometry fixed it.
  this file:    sampled NPG — mismatch reappears in the STATISTICS
                (critic transfer): kappa.

exploration is conserved. Fixing the metric moved the exploration
difficulty from the optimizer's path into the estimator's coverage — it
did not, and cannot, delete it. The only actual remedies change nu_k:
exploratory restarts/resets (assumed by the best bounds), optimism
(regret_and_exploration/), or a genuinely exploratory behavior mixture.
```

## What The Theorem Licenses In Practice

```text
1. approximation error enters as sqrt(eps_approx) — NOT amplified by
   (1-gamma)^{-2} error-propagation as in fitted-Q (offline_rl/06):
   policy-side methods degrade more gracefully with misspecified value
   classes; this is a genuine, provable advantage of actor-critic over
   pure value iteration.
2. eps_stat ~ O(1/sqrt(n)) per iteration with n on-policy samples: total
   sample complexity poly(1/eps, 1/(1-gamma), kappa, log|A|) — no |S|
   for linear-compatible classes (dimension enters via the regression).
3. the constants justify the folk practice of policy_gradient/: collect
   fresh on-policy batches per update (keeps nu_k current), and prefer
   advantage regression to Q regression (smaller targets — 02's baseline
   logic feeding eps_stat).
```

## What Remains Open

Whether `kappa` can be replaced by an *algorithm-dependent, path-averaged*
quantity for practically-initialized problems (worst-case `kappa` is
vacuous in hard-exploration MDPs — correctly, per the conservation
principle); optimal step/batch coupling under `eps_stat`; and the deep-
critic version, where `eps_approx` is unknowable and only the qualitative
lesson (graceful `sqrt` degradation + coverage sensitivity) transfers.
