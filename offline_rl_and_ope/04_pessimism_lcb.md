# Pessimism And Single-Policy Concentrability

## The Wrong Default, First

Naive offline optimization — fitted Q-iteration on the dataset, output the
greedy policy — inherits notebook 06's requirement: **all-policy
concentrability** (`sup_pi ||d^pi / mu_data||`), because the argmax may
route the learned policy through any state-action region, and the analysis
must control estimation error wherever the *learned* policy goes. Worked
failure shape: one out-of-support action with an erroneously high fitted
`Q` (noise, extrapolation — the max-bias of `value_based_deep_rl/03` with
no data to correct it) attracts the greedy policy; the certified-error
region and the executed region diverge; performance is arbitrarily bad
while the training loss is excellent. The data cannot veto what it never
saw — unless the algorithm charges for absence:

## LCB Value Iteration

Penalize instead of bonusing — optimism's mirror
(`regret_and_exploration/01`, run backward):

```math
\hat Q_h(s,a)
=
\max\Big(0,\ \ \hat r+\hat P\hat V_{h+1}(s,a)\ -\ b(s,a)\Big),
\qquad
b(s,a)\asymp H\sqrt{\frac{\log(SAH/\delta)}{N(s,a)\vee 1}},
```

(`b = H`-scale at unvisited cells: absent data = maximal penalty), then
`pi-hat` greedy w.r.t. the pessimistic `hat-Q`.

## The Guarantee

**Theorem (Rashidinejad et al. 2021, shape).** On the good event
(bonuses valid, `concentration_toolkit/02` exactly as in
`regret_and_exploration/02`), for **any comparator policy** `pi`:

```math
J(\pi)-J(\hat\pi)
\ \le\
2\sum_{h}\ \mathbb{E}_{(s_h,a_h)\sim d^{\pi}}\big[\,b(s_h,a_h)\,\big]
\ \lesssim\
H^2\,\sqrt{\frac{C^{\pi}\,S\,\log(\cdot)}{n}},
```

where

```math
C^{\pi}
=
\max_{h,s,a}\ \frac{d^{\pi}_h(s,a)}{\mu_{\mathrm{data},h}(s,a)}
\qquad\text{(SINGLE-policy concentrability)} .
```

**Read the quantifiers carefully — they are the theorem.** The bound
holds simultaneously for every comparator `pi`, and each comparator pays
only *its own* coverage constant `C^pi`. The dataset need not cover the
whole MDP; it need only cover **some good policy**, and the algorithm —
without knowing which policy that is — competes with it.

## The Proof's Two Moves

*Move 1: pessimism inverts the optimism lemma.* On the good event,
`hat-Q <= Q^{hat-pi}`-side inequalities give (backward induction,
monotonicity — the mirror image of `regret_and_exploration/02` Step 2):

```math
J(\hat\pi)\ \ge\ \hat V_1(s_1)
\qquad\text{(the algorithm's claims about itself are UNDER-estimates)} .
```

*Move 2: the comparator's loss localizes to the comparator's support.*
For any `pi`, the same telescope as always
(`regret_and_exploration/01`), run under `d^pi`:

```math
J(\pi)-\hat V_1(s_1)
\ \le\
\sum_h\mathbb{E}_{d^\pi}\big[2\,b(s_h,a_h)\big],
```

because along `pi`'s own trajectories, the pessimistic backup differs
from the true one by at most twice the bonus. Combine.
`E_{d^pi}[1/sqrt(N)]` then converts to `sqrt(C^pi / n)` by
Cauchy–Schwarz against the data distribution. ∎

```text
where each hypothesis is spent:
valid bonuses      -> move 1 AND move 2 (the good event, both signs)
pessimism' sign    -> move 1: without it, J(pi-hat) >= V-hat fails and the
                      executed policy can exploit fictitious value —
                      exactly the naive-FQI failure above
single-policy C    -> move 2's Cauchy–Schwarz: only d^pi / mu appears;
                      the learned policy's OWN visitation never needs
                      coverage because move 1 already vetoed uncovered
                      routes
```

## The Symmetry Worth Memorizing

```math
\text{online:}\quad
\text{bias TOWARD uncertainty}\ \Rightarrow\ \text{visit \& learn}
\qquad
\text{offline:}\quad
\text{bias AGAINST uncertainty}\ \Rightarrow\ \text{avoid \& certify}
```

One principle — *slant estimates against what you cannot change* — with
the sign set by whether uncertainty is reducible (you can visit) or not.
Both proofs are the same telescope with opposite inequality directions;
having built `regret_and_exploration/02`, this theorem is ~two pages.

Bernstein-ized bonuses give the same `H`-savings as
`regret_and_exploration/03`; linear-MDP versions exist with
`Lambda^{-1}`-ellipsoid penalties (and the C1/05 lower bound explains why
completeness-type structure is still required there).

## What Remains Open

The optimal interpolation when coverage is partial *and* interaction is
occasionally allowed (hybrid RL); tight instance-dependent constants
(current `C^pi` is a max — path-dependent refinements exist); and
principled pessimism scales for neural function classes, where `N(s,a)`
has no meaning — which is exactly the gap CQL fills heuristically:
notebook 05.
