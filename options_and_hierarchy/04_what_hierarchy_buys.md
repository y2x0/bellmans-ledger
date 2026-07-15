# What Hierarchy Buys: The Honest Ledger

## The Three Claimed Purchases, Audited

**1. Planning speed — PROVED.** Notebook 01's modulus: options of
expected length `L` contract with `E[gamma^tau] ~ gamma^L`, so value
iteration needs `~ 1/L` as many backups. Unconditional, quantified,
real — but note it is a claim about *computation with a known option
model*, not about learning.

**2. Exploration reach — TRUE WITH CONDITIONS.** A random walk over
options travels farther than a random walk over primitives: `k`
option-steps of length `~L` cover `~ sqrt(k) * L`-scale distances where
`kL` primitive random steps cover `~ sqrt(kL)`. Against the combination
lock (`regret_and_exploration/04`): options that happen to encode
lock-opening subsequences convert the `Omega(A^H)` barrier into
`Omega(|Omega|^{H/L})` — exponential improvement **iff the option set
already contains the right subsequences.** Hierarchy relocates the
exploration problem from "find the action sequence" to "have the right
options," which is only progress if options are acquired from something
other than the task's own sparse reward (transfer, unsupervised
discovery, demonstrations). The lock is conserved, not dissolved —
the conservation principle again (`global_convergence_of_pg/05`).

**3. Credit assignment — TRUE, AS VARIANCE REDUCTION.** With
commitments of length `L`, the top-level policy makes `H/L` decisions
per episode: the score-function variance argument
(`policy_gradient/02`, variance growing with decision count) improves by
`~L`, and TD error propagation (`finite_time_td_q/`) shortens
horizon-wise as in purchase 1. The price is bias: committing forgoes
`L-1` reconsiderations — bounded by the interruption theorem
(notebook 01): interruptible hierarchies pay ~nothing, rigid ones pay
`sum of foregone advantages` along executions.

## The Degeneracy Table (from notebook 03, completed)

```text
collapse             mechanism                       standard counter
beta -> 1            termination gradient ~ -A ~ 0   deliberation cost eta
                     once policy is decent            (switching price)
one-option-rules     d^aug support collapse           entropy over mu;
                                                      option diversity bonus
options = duplicated all omega converge to the same   decorrelation /
primitives           behavior (nothing enforces       diversity objectives;
                     specialization)                  fixed random subgoals
```

Each counter is an explicit regularizer — hierarchy persists only if
*paid for* (`regularized_mdps_and_duality/04`'s frame: the objective
`J + Omega(structure)` with `Omega` pricing switching/diversity), which
is the family's cleanest conclusion: **temporal abstraction is not
emergent from return maximization; it is a constraint or a prior.**

## Option Discovery: The Candidate Objectives

What should `Omega`-the-option-set optimize? The live proposals, each
with its counterexample:

```text
bottleneck states        options that funnel through graph cuts
(betweenness, cuts):     (doorways). Counterexample: reward structure
                         orthogonal to the graph's geometry — bottleneck
                         options can be perfectly useless for the task.
eigenoptions             options ascending eigenvectors of the SR /
(SR spectrum):           graph Laplacian (successor_representations/01):
                         principled, reward-free, geometry-adapted; same
                         counterexample as above, plus policy-dependence
                         of the SR.
empowerment /            maximize mutual information between option
diversity (DIAYN):       identity and reached states: yields distinguish-
                         able skills; nothing ties them to reward, and
                         the MI objective saturates on trivially
                         distinguishable but useless behaviors.
compression / MDL:       options = reusable subsequences that shorten
                         description of good trajectories: needs good
                         trajectories first (chicken-and-egg with
                         exploration).
```

No proposal dominates; the honest current answer is that option
discovery is a *transfer* question (amortize structure across a task
distribution — the successor-features/GPI machinery of
`successor_representations/02` is the cleanest formal version) rather
than a single-task one.

## Interfaces Out

```text
to rlhf_mathematics/06:   multi-step tool-use/reasoning "skills" in LLM
                          agents are options in all but name — subgoal
                          proposals, termination-on-verification; the
                          collapse modes and the deliberation-cost fix
                          transfer verbatim;
to search_and_planning/:  MCTS over macro-actions inherits purchase 1's
                          modulus argument inside the tree;
to model_based/:          option models (r_omega, p_omega) are the
                          natural coarse-grained learned models — value-
                          equivalence (model_based/04) at the option
                          timescale is largely unexplored territory.
```

## What Remains Open

A theorem of the form "for task distribution D, the optimal option set
is X" exists only in toy cases; regret/sample-complexity bounds *with
learned options in the loop* (rather than given ones) are essentially
absent; and the empirical center of gravity has shifted to implicit
hierarchy (recurrent/transformer policies discovering temporal
structure without explicit `beta`) — whether the explicit framework's
guarantees can be recovered for implicit hierarchies is open.
