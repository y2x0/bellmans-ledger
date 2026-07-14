# Monte Carlo Tree Search And UCT

## The Four Phases

MCTS builds an asymmetric tree of visited states, one simulation at a time:

```text
1. SELECTION    from the root, descend by a tree policy (a bandit rule per
                node) until reaching a leaf of the stored tree
2. EXPANSION    add one (or more) children of the leaf to the tree
3. EVALUATION   estimate the leaf's value: random/heuristic rollout to
                terminal, or (AlphaGo) a value network
4. BACKUP       propagate the evaluation up the visited path, updating
                per-edge statistics (N, Q)
```

Per edge, the stored statistics are a running count and mean:

```math
N(s,a)\ \mathrel{+}=1,
\qquad
Q(s,a)\ \leftarrow\ Q(s,a)+\frac{V_{\text{leaf}}-Q(s,a)}{N(s,a)} ,
```

— a Monte Carlo value estimate whose "policy" is the improving tree policy
itself.

## UCT: UCB1 At Every Node

(Kocsis & Szepesvári 2006.) The tree policy treats each node as a bandit
over its children:

```math
a=\arg\max_{a'}\ Q(s,a')+c\,\sqrt{\frac{\ln N(s)}{N(s,a')}} .
```

The subtlety UCT's analysis must face: unlike a true bandit, a child's
reward distribution is **nonstationary** — it improves as the subtree below
it deepens and its own tree policy sharpens. The Kocsis–Szepesvári argument
handles this with a drift condition (the empirical means converge and
confidence bounds still apply after inflation), giving:

```text
Theorem (informal). At every node the probability of selecting a suboptimal
action at the root -> 0, and root values converge to the minimax/optimal
value; the bias after n simulations at the root is O(log n / n).
```

Failure probability decays polynomially; worst-case instances require
exponentially many simulations before the bound bites (Coquelin–Munos), so
the guarantee is asymptotic reassurance, not a finite-time certificate —
matching MCTS practice, where strength is empirical.

## MCTS As Focused Value Iteration

Compare the backup with `mdp_foundations/04`:

```text
value iteration:   synchronous sweeps of T over ALL states
MCTS:              asynchronous, sampled backups over REACHABLE states,
                   allocated by a regret-minimizing rule that concentrates
                   compute where (a) values are promising and (b)
                   uncertainty is high
```

MCTS is the answer to the curse of dimensionality at *decision time*: it
never represents the full state space, only the tree its own simulations
grow. The anytime property (each simulation strictly refines the root
estimate) is Banach-iteration monotonicity translated to the sampled,
tree-local setting.

## Design Axes (where AlphaGo will intervene)

```math
\text{selection prior:}\quad
\text{uniform (UCT)}\ \to\ P(s,a)\ \text{from a policy network (PUCT)}
```

```math
\text{leaf evaluation:}\quad
\text{random rollout}\ \to\ (1-\lambda)v_\theta(s_L)+\lambda z_L
\ \text{(value net + fast rollout mix)}
```

```math
\text{expansion:}\quad
\text{all children}\ \to\ \text{lazily, priors stored at expansion}
```

Rollout evaluation is unbiased for the rollout policy's value but that policy
is weak (high bias w.r.t. optimal play) and high-variance; a value network is
biased but cheap and consistent. AlphaGo's `lambda = 0.5` mixture
(notebook 03) is a bias–variance compromise directly analogous to GAE's
`lambda` (`policy_gradient/05`) — the same dial, in the planning column.

## Search As Policy Improvement

The deepest reading, made explicit by the AlphaGo-Zero line: the visit
distribution at the root after search,

```math
\pi_{\text{MCTS}}(a\mid s)\ \propto\ N(s,a)^{1/\tau},
```

is a **strictly improved policy** relative to the prior `P` that guided the
search (search concentrates visits on actions that survived lookahead
scrutiny). Training the policy network toward `pi_MCTS` and the value
network toward search outcomes is generalized policy iteration
(`mdp_foundations/05`) with MCTS as the improvement operator — improvement no
longer by argmax over a learned `q`, but by *computation* against a model.
