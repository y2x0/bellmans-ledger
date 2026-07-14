# Failure Modes Of Policy Gradient Methods

## 1. Entropy Collapse (violated: persistent exploration)

```text
mechanism:   the gradient estimator only sees actions the policy takes;
             as pi sharpens, gradient support shrinks; a premature peak is
             self-reinforcing (positive A on the modal action -> sharper)
signature:   policy entropy falls monotonically to ~0 early; learning stops
             at a mediocre plateau; different seeds plateau differently
levers:      entropy bonus c2 (blunt), larger batch (less noise-driven
             sharpening), max-ent objectives (SAC), forced stochasticity
```

## 2. Ratio Blow-Up And The Limits Of Clipping

```text
mechanism:   clipping zeroes the gradient outside [1-eps, 1+eps] but does
             NOT project ratios back; with K epochs and momentum (Adam),
             parameters keep drifting through zero-gradient regions;
             measured KL(pi_old, pi) can exceed any target ("KL drift")
signature:   occasional catastrophic updates; performance cliffs that
             coincide with spikes in mean/max ratio or KL
levers:      early-stop epochs on a KL threshold (the standard fix,
             spinningup-style), smaller K, KL penalty in addition to clip,
             value/advantage recomputation between epochs
```

## 3. Critic Bias Poisoning The Gradient (violated: exact-critic premise of GAE)

```text
mechanism:   with lambda < 1, GAE admits critic error into E[A-hat]; a
             systematically wrong V (e.g. after a distribution shift the
             critic hasn't tracked) biases every advantage in the batch
signature:   advantage histograms shifted off zero per state-region;
             value-loss spikes preceding policy degradation
levers:      raise lambda toward 1 (pay variance), stronger critic training,
             separate critic network (avoid c1 competing with policy loss)
```

## 4. Surrogate/Visitation Mismatch At Scale (violated: small-step premise)

```text
mechanism:   everything in notebooks 03-04 assumed d^{pi'} ~ d^{pi}; long
             optimization on stale d^{pi_old} optimizes the wrong measure —
             the exact failure the PPO paper documents for multi-epoch
             vanilla PG ("destructively large policy updates")
signature:   train-batch surrogate rises while true return falls
             (the definitive diagnostic: log both)
```

## 5. Reward Hacking Of Learned Rewards (violated: r is the objective)

```text
when r is a model (RLHF), the policy optimizes the model's errors with the
full force of this family's machinery; KL-to-reference penalties are the
trust-region idea repurposed: stay close to a distribution where the reward
model is calibrated. Same math, new failure surface.
```

## 6. Nonconvexity Artifacts (revised — now quantified)

```text
J(theta) is nonconvex even for softmax-tabular policies, but the original
"inherent" framing was too coarse. The precise picture, proved in
global_convergence_of_pg/:

- softmax PG satisfies a non-uniform Łojasiewicz inequality: GLOBAL
  convergence at O(1/t), with the plateau hardness stored in the constant
  min_s pi(a*|s) * ||d^pi*/d^pi||^{-1} — exponentially small on
  hard-exploration chains (01: the plateau is real AND global convergence
  is true; both, simultaneously);
- natural PG deletes those constants entirely: dimension-free
  log|A|/(eta K), and LINEAR convergence with entropy regularization (02);
- what NPG cannot delete: the mismatch reappears as the critic's transfer
  coefficient under sampling (04) — the conservation law (05): exploration
  difficulty moves between the optimization and the statistics; only
  changing the data distribution reduces it.

The LP view (mdp_foundations/06) stands: linear in occupancy space, all
hardness in the theta -> lambda map — now with the map's geometry priced.
```

## The Diagnostic Dashboard

The quantities this family's theory says to monitor, each tied to a theorem:

```text
mean & max KL(pi_old, pi)      trust region actually respected? (nb 04)
fraction of clipped ratios     how binding is the constraint? (nb 06)
policy entropy                 collapse? (nb 02)
explained variance of critic   GAE bias regime (nb 05)
surrogate vs true return       visitation mismatch (nb 03)
```
