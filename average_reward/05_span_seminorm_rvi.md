# The Span Seminorm And Relative Value Iteration

## The Right (Semi)Norm

The average-reward Bellman operator

```math
(\mathcal{T}h)(s)=\max_a\Big[r(s,a)+\sum_{s'}P(s'\mid s,a)\,h(s')\Big]
```

(no discount, no `rho` — value iteration just runs `h_{k+1} = T h_k`) is
**not** a sup-norm contraction: `T(h + c1) = Th + c1` (modulus exactly 1
along constants). But constants are precisely the direction that doesn't
matter (`h` is defined up to one, notebook 02). Quotient them out — the
**span seminorm**:

```math
\|v\|_{\mathrm{sp}}
=
\max_s v(s)-\min_s v(s)
\qquad\big(=0\ \text{iff}\ v\ \text{constant}\big).
```

## The Contraction Theorem

**Theorem.** Suppose the MDP satisfies a one-step mixing (Doeblin/
ergodicity) condition: there is `alpha > 0` with

```math
\sum_{s'}\min\big(P(s'\mid s_1,a_1),\ P(s'\mid s_2,a_2)\big)\ \ge\ \alpha
\qquad\forall (s_1,a_1),(s_2,a_2).
```

Then `T` is a span contraction with modulus `1 - alpha`:

```math
\|\mathcal{T}u-\mathcal{T}v\|_{\mathrm{sp}}\ \le\ (1-\alpha)\,\|u-v\|_{\mathrm{sp}} .
```

*Proof.* Let `w = u - v`. As in the classical max-lemma
(`mdp_foundations/03`), for the states attaining the span of `Tu - Tv`
there are actions with

```math
(\mathcal{T}u-\mathcal{T}v)(s_1)-(\mathcal{T}u-\mathcal{T}v)(s_2)
\ \le\
p_1^\top w-p_2^\top w,
\qquad p_i=P(\cdot\mid s_i,a_i).
```

Now the coupling step: let `q = min(p_1, p_2)` componentwise,
`|q| >= alpha`. Then

```math
p_1^\top w-p_2^\top w
=(p_1-q)^\top w-(p_2-q)^\top w
\le
(1-\alpha)\max w-(1-\alpha)\min w
=(1-\alpha)\|w\|_{\mathrm{sp}},
```

since `p_i - q` are nonnegative vectors of equal mass `<= 1 - alpha`. ∎

**The modulus is a mixing coefficient, not a discount.** The role
`gamma` played everywhere else is here played by "how much any two
state-actions' futures overlap in one step" — the analytic confirmation
of notebook 03's claim that mixing time replaces `1/(1-gamma)` as the
difficulty parameter. Under the theorem, Banach on the quotient space
gives: `h*` unique up to constants, `rho*` recovered as the (constant)
per-iteration drift, geometric convergence in span.

## Relative Value Iteration

Raw VI's iterates `h_k` drift by `~ rho*` per step (unbounded); RVI pins
the constant by re-anchoring at a reference state `s0`:

```math
h_{k+1}=\mathcal{T}h_k-\big(\mathcal{T}h_k\big)(s_0)\,\mathbf{1},
\qquad
\rho_k=(\mathcal{T}h_k)(s_0)\ \to\ \rho^* .
```

Subtracting a constant is invisible in span, so RVI inherits the span
contraction verbatim; the anchor value converges to the optimal gain.
(Any normalization works — anchor state, mean-zero, min-zero.)

## The Periodicity Failure And Its Repair

Without mixing, span contraction genuinely fails. **The 2-cycle:** states
`{1, 2}`, deterministic swap, one action, `r = (1, 0)`.

```math
\mathcal{T}h=\big(1+h(2),\ h(1)\big),
\qquad
\|\mathcal{T}u-\mathcal{T}v\|_{\mathrm{sp}}=\|u-v\|_{\mathrm{sp}}
\quad\text{(the swap permutes, preserving spans)} .
```

VI oscillates forever (`h_k` alternates between two profiles; `rho_k`
between 1 and 0 — the true gain is `1/2`, the Cesàro compromise of
notebook 01). *The repair — the aperiodicity transformation:* replace
`P <- (1-tau) I + tau P` (lazy chain, `tau in (0,1)`). This preserves
gain-optimal policies and biases up to scale (the Poisson equation
transforms linearly: `(I - P_lazy) = tau (I - P)`), and the identity
component gives every pair of rows overlap `>= 1 - tau`... more to the
point it makes the chain aperiodic, restoring convergence — the standard
preprocessing before running RVI, and the average-reward ancestor of
every "add laziness/damping to kill oscillation" trick in the repo
(target-network damping, `value_based_deep_rl/02`; EMA,
`finite_time_td_q/06`).

## Weakly Communicating MDPs (the honest general case)

One-step Doeblin is strong. Under weak communication, span contraction
holds only for the `m`-step operator (`m` ~ diameter-ish) or not
uniformly at all; RVI still converges under aperiodic unichain
assumptions but without a clean modulus, and the average-reward regret
literature (UCRL2's `D sqrt(SAT)` — `model_based/03`) pays the
**diameter** `D = max expected travel time between states` exactly where
this file pays `1/alpha`. Same phenomenon, worst-case-ified.

## What Remains Open

Span-seminorm theory with function approximation (what is the projected
span fixed point? — largely undeveloped, the average-reward twin of
`stochastic_approximation/04`); and optimal aperiodicity/anchoring
choices for RVI's finite-time behavior, which the existing asymptotic
theory does not rank.
