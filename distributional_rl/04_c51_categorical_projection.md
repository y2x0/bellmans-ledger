# C51: Categorical Parametrization And The Projected Update

```text
Bellemare, Dabney & Munos 2017, §4. papers/c51-bellemare-2017.pdf
```

## The Function Class

Fix `N` atoms on a common support (Atari: `N = 51`,
`[V_min, V_max] = [-10, 10]` — hence "C51"):

```math
z_i=V_{\min}+i\,\Delta z,
\qquad
\Delta z=\frac{V_{\max}-V_{\min}}{N-1},
\qquad
i=0,\ldots,N-1,
```

and parameterize each state-action's return law as a categorical
distribution over these **fixed** locations, probabilities from a softmax
head:

```math
Z_\theta(s,a)=\sum_{i} p_i(s,a)\,\delta_{z_i},
\qquad
p_i(s,a)=\frac{e^{\theta_i(s,a)}}{\sum_j e^{\theta_j(s,a)}} .
```

The network is DQN's convnet with the scalar-per-action head replaced by
`N` logits per action; `Q(s,a) = sum_i z_i p_i(s,a)` for action selection.

## The Support Problem And The Projection

Apply the sampled Bellman map to each atom: `T-hat z_j = r + gamma z_j`.
These land **off the grid** (and outside `[V_min, V_max]` if unclipped), so
the backed-up categorical is not in the function class. C51 projects it back
with `Phi`, distributing each shifted atom's mass **linearly to its two
neighboring grid points**: with
`b_j = (clip(T-hat z_j, V_min, V_max) - V_min)/Delta z`, `l = floor(b_j)`,
`u = ceil(b_j)`:

```math
(\Phi\hat{\mathcal{T}}Z)_l\ \mathrel{+}=\ p_j\,(u-b_j),
\qquad
(\Phi\hat{\mathcal{T}}Z)_u\ \mathrel{+}=\ p_j\,(b_j-l).
```

This linear interpolation is exactly the **W1-optimal** (also
Cramér-optimal) projection of a point mass onto two neighboring support
points — the projection respects the horizontal geometry of notebook 01
even though the training loss (below) will not.

## The Loss

Sample transition `(s, a, r, s')`; greedy-by-mean action
`a* = argmax_{a'} sum_i z_i p_i(s', a')` **under the target network**
`theta-tilde`; minimize the cross-entropy between the projected target and
the prediction:

```math
L(\theta)
=
D_{\mathrm{KL}}\Big(\Phi\,\hat{\mathcal{T}}Z_{\tilde\theta}(s,a)\ \Big\|\ Z_\theta(s,a)\Big)
=
-\sum_i m_i \log p_i(s,a;\theta) + \text{const},
```

`m_i` the projected target masses. Since the support is shared and fixed,
this is an **N-way soft-label classification problem** — distributional RL
reduced to the best-conditioned loss deep learning has. Replay, target
network, and epsilon-greedy are inherited from DQN unchanged
(`value_based_deep_rl/01`).

## The Theory–Practice Gap, Stated Honestly

```text
theory (notebook 02):   contraction is in Wasserstein d_p-bar
algorithm (this one):   trains in KL against a projected target
```

Two reasons for the swap, both in the paper: (i) the sampled Wasserstein
loss is biased (`E[W(empirical, target)] != W(true, target)` — notebook 01's
obstruction), so its SGD is not minimizing the population distance; (ii) KL
between shared-support categoricals gives clean, bounded gradients
(`m_i / p_i` weights). What is provable about the composite operator: the
projected evaluation operator `Phi T^pi` **is** a `gamma`-contraction in the
**Cramér distance** `l_2` (Rowland et al. 2018), with fixed point within
`O(Delta z)` of the true `Z^pi` — the honest metric for C51 turned out to be
Cramér, not Wasserstein; notebook 05.

KL against the projection also silently requires the support to cover the
true returns: mass at true returns outside `[V_min, V_max]` is clipped to
the boundary atoms — a bias the practitioner sets with two hyperparameters.

## Results And What They Argue

```text
Atari-57, human-normalized:  mean 701% vs 592% for the best prior comparable
                             agent; new SOTA at publication
largest gains:               sparse/hard games (e.g. big jumps on
                             Seaquest, Q*bert, up from near-DQN elsewhere);
                             the paper highlights faster propagation of
                             rare-event information
qualitative:                 learned distributions are visibly multimodal --
                             e.g. "clear the screen or die" states show two
                             separated modes whose weights shift with play
```

Why would predicting 51 numbers whose *mean* is all the policy uses beat
predicting the mean directly?

```text
1. richer target: matching a full law is many auxiliary tasks sharing the
   trunk -- better representations (the auxiliary-task reading);
2. better-behaved regression: cross-entropy on bounded categoricals vs
   L2 on a moving scalar with outliers -- gradient conditioning;
3. bias control: bounded support caps target magnitude (a soft version of
   DQN's clipping), and averaging over modes damps the argmax chattering
   of notebook 03.
```

All three are statements about **optimization and statistics**, not about
control-theoretic optimality — consistent with notebook 03's conclusion that
the distributional fixed point is not what the performance rests on.
