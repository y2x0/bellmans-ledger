# What Deep Networks Change

## The Honest Frame

Nothing in notebooks 01–05 applies verbatim to deep RL: networks are not
linear in their features, their covering numbers make the uniform bounds
vacuous (`concentration_toolkit/04`), and no one verifies completeness of
a ResNet class. This file states what *does* transfer, what is known in
limited regimes, and which practical phenomena the linear theory
retro-dicts correctly.

## What Transfers As-Is

```text
1. the double-sampling obstruction (03): a statement about the LOSS, not
   the class — Bellman residual minimization is biased for any F. This is
   why deep RL does fitted targets (DQN) rather than residual gradients.

2. error propagation and concentrability (05; value_based_deep_rl/02):
   the amplification geometry is class-agnostic; deep value errors off
   the data distribution compound through backups exactly as the linear
   construction predicts.

3. the norm-mismatch reading of the deadly triad: nothing about
   nonlinearity repairs it; deep off-policy divergence is observed and
   its mechanism is the linear one.
```

## The NTK Regime: Linear Theory As A Limit Theorem

In the infinite-width/lazy-training limit, a network's training dynamics
linearize around initialization:

```math
f_\theta(x)\ \approx\ f_{\theta_0}(x)+\nabla_\theta f_{\theta_0}(x)^\top(\theta-\theta_0),
```

i.e. **a linear class with features
`phi(x) = grad_theta f_{theta_0}(x)`** — the neural tangent kernel. Under
NTK conditions, LSVI-UCB-style results port: optimistic NN-value-iteration
enjoys `tilde-O(sqrt(d_eff^3 H^4 K))`-type regret with `d_eff` the
effective dimension of the NTK Gram matrix (Yang et al., Zhou et al.).
Honest assessment:

```text
- it is a real theorem about real networks, in a regime (lazy training)
  that deliberately forbids feature learning;
- feature learning is empirically the entire point of deep RL (the DQN
  t-SNE figures; representation transfer);
- so NTK results certify deep RL exactly where it is least deep.
```

## Where The Linear Theory Retro-dicts Practice

The most useful role of 01–05 is explanatory:

```text
observed practice                     the theory's account
-----------------------------------   --------------------------------------
target networks stabilize training    a frozen target makes each phase a
                                      supervised regression whose target is
                                      IN (or near) the representable class —
                                      a poor man's completeness-per-phase
                                      (03; value_based_deep_rl/02)

wider nets don't fix RL the way       enlarging F does not preserve
they fix supervised learning          T-invariance — completeness is
                                      non-monotone (03's knife-edge)

offline RL needs explicit pessimism   coverage + expressivity is
                                      insufficient IN PRINCIPLE (05)

auxiliary/self-supervised losses      they shape phi toward linear-MDP-like
help control performance              structure: making P approximately
                                      factor through learned features is
                                      exactly manufacturing notebook 01's
                                      hypothesis (explicit in FLAMBE/
                                      latent-model methods)
```

The last row is the constructive reading of the whole family: **since the
guarantees attach to low-rank/complete representations, representation
learning in deep RL is the project of making the theory's hypotheses
true**, rather than assuming them.

## The Open Problem, Stated As This Family Sees It

Find a parameter `kappa(F, MDP)` such that:

```text
(i)   kappa is small for practically trained networks on natural tasks;
(ii)  small kappa + a practical algorithm  =>  poly-sample guarantees;
(iii) kappa is estimable from data (so the guarantee is checkable).
```

Bellman rank/bilinear (04) achieve (ii) without (i, iii)-computability;
NTK dimensions achieve (i, ii) in the lazy regime only; the
decision-estimation coefficient sharpens (ii)'s boundary but not (i, iii).
All three legs together: open, and in this family's view the central
theoretical problem of deep reinforcement learning.
