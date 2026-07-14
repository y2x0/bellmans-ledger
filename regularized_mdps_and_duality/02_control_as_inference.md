# Control As Inference

## The Construction

Attach to each timestep a binary "optimality" variable `O_t` with
likelihood

```math
p(O_t=1\mid s_t,a_t)\ \propto\ \exp\big(r(s_t,a_t)/\tau\big),
```

and consider the graphical model: the known dynamics `p(s'|s,a)`, a
reference action prior `p(a|s)` (take uniform; a general prior yields KL
instead of entropy), and the `O_t` hanging off each `(s_t, a_t)`. The
inference question: **what is the posterior over trajectories given that
every step was "optimal,"** `p(tau | O_{1:T} = 1)`?

## The Variational Derivation

Approximate the posterior with a distribution `q` over trajectories that
**must respect the true dynamics** (this constraint is the crux — see the
pathology below), i.e.
`q(tau) = p(s_1) prod_t p(s_{t+1}|s_t,a_t) pi(a_t|s_t)` with free `pi`.
The evidence lower bound:

```math
\log p(O_{1:T}=1)
\ \ge\
\mathbb{E}_{q}\Big[\log\frac{p(\tau,O_{1:T}=1)}{q(\tau)}\Big]
=
\mathbb{E}_{\pi}\Bigg[\sum_t \frac{r(s_t,a_t)}{\tau}-\log\frac{\pi(a_t\mid s_t)}{p(a_t\mid s_t)}\Bigg]+\text{const},
```

*because the dynamics terms cancel between `p` and `q`* (both contain the
same `p(s'|s,a)` factors — the same cancellation as in the policy gradient
likelihood proof, `policy_gradient/01`). Multiplying by `tau`:

```math
\text{ELBO}\ \propto\
\mathbb{E}_\pi\Big[\sum_t r(s_t,a_t)\Big]-\tau\sum_t\mathrm{KL}\big(\pi(\cdot\mid s_t)\,\|\,p(\cdot\mid s_t)\big)
```

— **exactly the KL/entropy-regularized objective of notebook 01** (entropy
when the prior is uniform). Maximum-entropy RL *is* variational inference
in this model; the temperature is the reward-to-evidence conversion rate.

## Exact Inference = Soft Bellman

Run backward message passing for the untruncated posterior. Define the
backward message
`beta_t(s,a) = p(O_{t:T}=1 | s_t=s, a_t=a)`. The recursion:

```math
\beta_t(s,a)
=
e^{r(s,a)/\tau}\ \mathbb{E}_{s'}\Big[\ \textstyle\sum_{a'}p(a'\mid s')\,\beta_{t+1}(s',a')\ \Big],
```

and substituting `beta = exp(q/tau)`:

```math
q_t(s,a)=r(s,a)+\tau\log\ \mathbb{E}_{s'}\Big[\exp\big(v_{t+1}(s')/\tau\big)\Big],
\qquad
v(s)=\tau\log\sum_a p(a\mid s)e^{q(s,a)/\tau}
```

— the soft Bellman recursion of notebook 01, **except for one detail**,
which is the file's second point:

## The Risk-Seeking Pathology

Exact inference produced
`tau log E_{s'} exp(v/tau)` — a **soft max over the environment's
randomness**, not the expectation `E_{s'}[v]`. The exact posterior is
optimistic about dynamics: conditioning on optimality tilts *transition*
probabilities toward lucky outcomes, as if the agent could choose its
luck.

**Worked 2-state example.** One decision: action `safe` gives reward
`0.5` surely; action `risky` moves to a coin-flip state: heads reward
`1`, tails reward `-10`, each `1/2` under the true dynamics.

```math
\text{true values: } q(\text{safe})=0.5,
\qquad
q(\text{risky})=\tfrac12(1)+\tfrac12(-10)=-4.5 .
```

Exact-inference "value" of risky at small `tau`:
`tau log( (e^{1/tau} + e^{-10/tau})/2 ) -> 1 - tau log 2 ~ 1 > 0.5` — the
posterior *prefers the gamble*, because trajectories that received heads
are upweighted by their own optimality likelihood. Conditioning on success
is not a plan.

**The repair is the variational family constraint above:** by forcing `q`
to carry the *true* dynamics factors, the ELBO derivation replaces the
softmax-over-`s'` with the honest `E_{s'}` (the dynamics cancel instead of
being tilted). Maximum-entropy RL as practiced (notebook 01's operator,
SAC) is the *variational*, dynamics-respecting solution; the exact
posterior is the seductive wrong one. In deterministic environments the
two coincide — which is why the distinction is invisible in many demos and
bites in stochastic ones.

## What The Inference View Buys

```text
1. this derivation — the objective is not an ad hoc bonus but the unique
   ELBO of a natural model (and the prior p(a|s) slot is exactly where
   pi_ref enters RLHF: rlhf_mathematics/02);
2. structured approximations: hierarchical policies, latent-variable
   policies, and message-passing planners become inference algorithms in
   the same model (planning-as-inference);
3. the composability of log-probabilities: skills/goals compose by adding
   soft Q-functions, with exact semantics inherited from the model;
4. a precise warning label (this file's pathology) for every "just
   condition on high reward" scheme — including naive reward-conditioned
   sequence models, which inherit the same optimism about stochasticity.
```

## What Remains Open

The variational family for *partially observed* control-as-inference
(memory + inference interact); and finite-sample statistical theory for
inference-derived planners — the frame is exact but its estimators are
mostly unpriced.
