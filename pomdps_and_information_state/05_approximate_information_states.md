# Approximate Information States

## The Question Practice Forces

Deep agents compress history with frame stacks, RNN hidden states, or
transformer contexts — none of which is an exact belief (01) or PSR
(04). What makes a learned summary *good enough*, and what does its
imperfection cost in value? The AIS framework
(Subramanian–Sinha–Seraj–Mahajan 2022) answers with a two-condition
definition and a bound.

## The Definition

A history compression `z_t = sigma(h_t)` (updated recursively:
`z_{t+1} = phi(z_t, a_t, o_{t+1})`) is an **(eps, delta)-approximate
information state** if it approximately supports the two uses a state
has — predicting reward and predicting its own evolution:

```math
\text{(P1: reward)}\qquad
\big|\ \mathbb{E}[R_{t+1}\mid h_t,a_t]-\hat r(z_t,a_t)\ \big|\ \le\ \varepsilon,
```

```math
\text{(P2: self-prediction)}\qquad
d\Big(\ \Pr\{z_{t+1}\in\cdot\mid h_t,a_t\},\ \ \hat P(\cdot\mid z_t,a_t)\ \Big)\ \le\ \delta,
```

for a metric `d` on distributions (the bound below needs a metric that
controls integrals of the value class — Wasserstein for Lipschitz
values, total variation for bounded: the `distributional_rl/01` metric-
selection logic, reappearing on the *state* rather than the return).

`eps = delta = 0` recovers exact information states — beliefs and PSRs
both qualify (01's sufficiency theorem verifies (P1)/(P2) for `b_t`;
04's rank theorem for PSR predictions).

## The Value-Loss Theorem

**Theorem (shape).** Let `hat-pi` be optimal for the *approximate* model
`(hat-r, hat-P)` on `z`-space, applied to the true POMDP through
`sigma`. Then

```math
\big|\,V^{*}(h)-V^{\hat\pi}(h)\,\big|
\ \le\
\frac{2\varepsilon}{1-\gamma}
\ +\
\frac{2\gamma\,\delta\ L_{\hat V}}{(1-\gamma)}
\qquad\text{for all histories }h,
```

with `L_{hat-V}` the (metric-`d`-dual) norm of the approximate value
function (`<= Rmax/(1-gamma)` for TV; a Lipschitz constant for `W_1` —
whence one more horizon factor in the worst case).

*Proof mechanism.* The standard two-model value-difference telescope —
**the simulation lemma** (`model_based/01`) applied to the pair (true
process on histories, approximate process on `z`): per step, the reward
mismatch contributes `eps`, the transition mismatch contributes
`delta * (how much value can differ across the discrepancy) = delta *
L_V`; both persist for a `1/(1-gamma)` horizon (the resolvent
geometric series, as always); a second telescope converts "value of
`hat-pi` under the approximate model" to "under the true model." ∎

The theorem's real content is the **choice of (P1)+(P2) as the
definition**: they are precisely the two error channels the simulation
lemma can price. Summaries that predict *observations* (generative
world-models), or that are merely discriminative for the current
action, satisfy neither condition per se — prediction of *reward and of
your own next summary* is the minimal self-consistent contract.

## Reading The Standard Architectures Through The Bound

```text
frame stacking (DQN):   z = last 4 frames. (P1)/(P2) hold with small
                        eps, delta iff the process is ~4-Markov in
                        observations (flicker: yes; occluded object
                        permanence: no — delta is large exactly when
                        something off-screen matters: the QBF-shaped
                        ambiguity of 03);
RNN state (DRQN, R2D2): z trained by TD — which optimizes a PROXY of
                        (P1) (value prediction) and nothing like (P2):
                        recurrent value agents can satisfy the letter of
                        value-fit while carrying a self-inconsistent
                        state (documented burn-in pathologies: R2D2's
                        stored-state staleness IS a delta-violation);
world models            explicitly train (P1) + (P2)-style losses
(Dreamer-family):       (latent reward and latent dynamics prediction):
                        the AIS bound is, structurally, their
                        justification — with delta measured in the
                        latent metric the value net actually respects
                        (the honest gap: that metric is not controlled);
transformer contexts:   z = attention over a WINDOW: (P2) violated by
                        truncation at exactly the horizons beyond the
                        window — the bound degrades additively with the
                        tail weight of dependence past the context.
```

## The Design Rules The Bound Extracts

```text
1. train the summary on BOTH losses: reward relevance (P1) alone allows
   self-inconsistent states; self-prediction (P2) alone allows reward-
   irrelevant compressions (predicting noise perfectly);
2. the metric in (P2) must match the value class (Lipschitz value =>
   Wasserstein suffices and is weaker/easier than TV — cheaper world
   models are licensed when values are smooth);
3. eps and delta are MEASURABLE on data (regression residuals of the
   two predictors): the bound is a monitorable certificate — rare in
   deep RL — even if the constants are loose.
```

## What Remains Open

The bound is worst-case over histories and loose in the horizon; a
theory of *learned* `sigma` (does gradient training find small
(eps, delta) summaries when they exist? — the representation-learning
question, `linear_mdps_and_completeness/06`'s three-legged open problem
in POMDP form); and the right parameterized hardness (03's closing
question) for which AIS's `(eps, delta)` is the leading candidate
vocabulary.
