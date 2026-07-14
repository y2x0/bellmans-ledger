# Overoptimization: Goodhart's Law, Quantified

## The Setup

Two rewards: the **gold** `r*` (what humans actually value, measurable
only by fresh human eval) and the **proxy** `hat-r` (the trained reward
model, 01). The policy optimizes `hat-r`; we care about `r*`. Write the
proxy's error as

```math
\hat r(y)=r^*(y)+\epsilon(y),
\qquad
\epsilon:\ \text{small ON the preference distribution, unknown OFF it.}
```

## The Mechanism Is Max-Bias

Optimizing against `hat-r` — by exponential tilt (02), by BoN (04), or by
PPO — concentrates the policy on the **upper tail of `hat-r`**, and the
upper tail of `r* + eps` systematically over-represents positive `eps`:
the same order-statistics computation as `value_based_deep_rl/03`. For
BoN this is exact: with `eps` independent noise of scale `sigma`,

```math
\mathbb{E}\big[r^*(Y_{\mathrm{BoN}})\big]
=
\mathbb{E}\big[\hat r(Y_{\mathrm{BoN}})\big]-\mathbb{E}\big[\epsilon(Y_{\mathrm{BoN}})\big]
\ \approx\
\mathbb{E}[\hat r_{\mathrm{BoN}}]-\underbrace{\sigma\sqrt{2\log n}\cdot\rho\text{-ish}}_{\text{selected noise}},
```

the selected-noise term growing with optimization pressure `log n =`
(essentially) the KL budget of 04. **Goodhart's law here is not a
metaphor; it is `E[max] > max E` applied to a noisy objective**, third
appearance of the identical mechanism (Q-learning targets, offline
argmax, now reward models).

## The Empirical Scaling Laws (Gao–Schulman–Hilton 2023)

With `d = sqrt(KL(pi || pi_ref))` as the optimization-distance variable,
gold reward follows, remarkably cleanly:

```math
\text{BoN:}\quad
r^*(d)=d\,(\alpha_{\mathrm{bon}}-\beta_{\mathrm{bon}}\,d),
\qquad
\text{PPO:}\quad
r^*(d)=d\,(\alpha_{\mathrm{ppo}}-\beta_{\mathrm{ppo}}\,\log d),
```

while the **proxy** reward keeps rising monotonically in `d` — the two
curves separate, the gold curve peaks and *declines*: overoptimization
has an interior optimum in optimization pressure. The coefficients scale
smoothly: `beta`-coefficients shrink with reward-model size and data
(better proxies tolerate more pressure); `alpha` grows with policy size.
The hump is universal across their settings; only its location moves.

**The quadratic's origin, in one heuristic:** gold gain is linear in `d`
at rate `alpha` (moving along the true frontier, 02) while selected noise
grows like `d * (noise picked up per unit distance)` — with `eps` scale
effectively growing as the policy leaves the proxy's training
distribution (01, point 3), the loss term is superlinear; a
linear-minus-quadratic ansatz is its lowest-order form. The theory
matching these curves exactly (beyond special noise models) is open.

## KL Budgets As The Control Variable

The operational conclusion practice has converged on, stated as this
family understands it:

```text
1. d = sqrt(KL) is the right x-axis: it is policy-side, measurable
   online, comparable across methods (04 makes BoN's exactly log n -
   (n-1)/n), and the scaling laws are clean in it — optimization
   pressure, not step count, is the quantity to ration;
2. stop (or size beta) near the gold hump: with a validation gold signal
   (periodic human eval / a held-out stronger RM), the interior optimum
   is estimable; without one, the beta-coefficients' scaling with RM
   size/data gives the budget a priori;
3. under a quadratic model r*(d) = alpha d - beta_2 d^2, the optimum is
   d* = alpha/(2 beta_2) with achievable gold alpha^2/(4 beta_2):
   BETTER PROXIES (smaller beta_2) raise the ceiling quadratically —
   reward-model quality dominates policy-optimization cleverness, which
   matches practice;
4. ensembling/uncertainty-penalizing the RM (pessimism at the argmax —
   offline_rl_and_ope/05's principle, fourth appearance) flattens the
   noise-selection term directly.
```

## What This File Does *Not* Claim

The analysis covers **statistical** Goodhart: selection of independent-ish
proxy noise. It does not cover *causal/adversarial* failure — the policy
discovering inputs that exploit systematic RM blindspots (sycophancy,
format hacking, length bias), where `eps` is neither small on-distribution
nor noise-like, and where the quadratic phenomenology can fail abruptly
rather than smoothly. Length bias alone reliably produces the observed
"proxy up, gold down" signature through a single interpretable direction
of `eps`. The decomposition of observed overoptimization into
statistical-vs-systematic components is measurement work every deployment
must redo.

## What Remains Open

A predictive theory of the scaling-law coefficients; overoptimization
under DPO (no on-policy reward queries — different, empirically milder-
but-present dynamics; 03's ledger); and principled online gold-signal
allocation (when to spend human evaluations — an optimal-stopping problem
this family has all the tools to pose and none of the literature to
cite closed).
