# Intra-Option Learning

## The Waste In SMDP Learning

Naive SMDP Q-learning treats each option execution as one atomic
transition: run `omega` for `tau` steps, then do a single update

```math
Q(s,\omega)\ \leftarrow\ Q(s,\omega)+\alpha\Big[\textstyle\sum_k\gamma^k r_{t+k}+\gamma^\tau\max_{\omega'}Q(s',\omega')-Q(s,\omega)\Big].
```

Convergent (it is Watkins' algorithm on the SMDP,
`stochastic_approximation/05` + notebook 01's contraction), but it
learns **one** thing from `tau` steps of experience, only about the
**executed** option, only from its **start state**. Intra-option methods
repair all three at once.

## The Intra-Option Bellman Equations

Decompose the option value stepwise. Define the value of *continuing*
`omega` from `(s, a)`:

```math
U_\omega(s')
=
\big(1-\beta_\omega(s')\big)\,q(s',\omega)
+\beta_\omega(s')\,\max_{\omega'}q(s',\omega')
\qquad\text{(continue, or terminate and re-choose)},
```

```math
q(s,\omega)
=
\sum_a\pi_\omega(a\mid s)\Big[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,U_\omega(s')\Big].
```

*Derivation:* condition the option's discounted return on its first
primitive step and on whether it terminates at `s'` — the tower property
plus the definition of `beta`; the two-case structure of `U` is the only
new ingredient over `mdp_foundations/03`. ∎ At the one-step-option
degeneracy (`beta = 1`) `U` collapses to `max q` and the equation is the
primitive Q-optimality equation — the consistency check again.

## Intra-Option Q-Learning

The equations are **per-step** identities, so every primitive transition
`(s_t, a_t, r_t, s_{t+1})` yields a valid sampled backup for **every
option consistent with the action taken**:

```math
Q(s_t,\omega)\ \leftarrow\ Q(s_t,\omega)+\alpha\Big[r_t+\gamma\,\hat U_\omega(s_{t+1})-Q(s_t,\omega)\Big]
\qquad
\text{for ALL }\omega:\ \pi_\omega(a_t\mid s_t)>0 ,
```

with `U-hat` built from the current `Q` and `beta_omega(s_{t+1})`. One
step of experience updates the whole eligible option library — off-policy
learning **across options**, with no importance ratios when the
intra-option policies are deterministic at the taken action (the general
stochastic case adds a one-step ratio `pi_omega(a_t|s_t)/behavior`,
inheriting `stochastic_approximation/02`'s machinery).

**Convergence (SPS 1999, deterministic intra-option policies).** Under
Watkins-type conditions, intra-option Q-learning converges a.s. to the
optimal option values `q_*` — the proof is again asynchronous SA
(`stochastic_approximation/05`) once the intra-option equations are shown
to define a contraction: they do, jointly in `(q, U)`, with the same
`gamma` (the `U`-substitution is a monotone nonexpansive map composed
into the backup). ∎

```text
what was gained over SMDP learning, itemized:
  1. tau updates per execution instead of 1;
  2. every consistent option learns from every step (counterfactual
     reuse — the option library as a shared dataset);
  3. r_omega and p_omega (notebook 01's model) can likewise be learned
     per-step ("intra-option model learning"), giving the planning
     representation without atomic executions.
```

## Position In The Repository's Coordinates

```math
(\mathcal{T},d,\mathcal{F})
=
\Big(\text{intra-option coupled backup},\ \|\cdot\|_\infty+\text{RM},\ \text{table over }S\times\Omega\Big),
```

with the distinctive feature that the *same data stream* feeds `|Omega|`
coupled operators — the cleanest classical instance of what deep RL now
calls auxiliary tasks / universal value functions: many predictions, one
experience stream, shared convergence theory. (GVFs/Horde and successor
features — `successor_representations/` — are this idea with the option
index generalized to arbitrary cumulants.)

## What Remains Open

Function-approximation intra-option learning inherits all of
`stochastic_approximation/04–06`'s conditions *per option* plus their
coupling — no clean TvR analogue exists; and the interaction of
per-step counterfactual updates with replay buffers (stale `beta`,
`pi_omega`) is unpriced — practice (e.g. option-critic implementations)
simply runs it.
