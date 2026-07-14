# Distributional RL

This family asks:

```text
What survives of the contraction theory when the object of learning is the
LAW of the return rather than its mean?
```

Primary source: Bellemare, Dabney & Munos, "A Distributional Perspective on
Reinforcement Learning," ICML 2017 (`papers/c51-bellemare-2017.pdf`).

The answers, per notebook: evaluation contracts — but only in Wasserstein
geometry (optimal transport enters RL through the *metric*, notebook 01–02);
control does **not** contract and may have no fixed point (notebook 03); yet
the practical algorithm (C51, notebook 04) set the Atari state of the art,
and closing its theory–practice gap produced quantile methods (notebook 05).

## The Coordinates

```math
\mathcal{T}: \text{distributional Bellman operator on } \mathcal{Z}=\{Z:\mathcal{S}\times\mathcal{A}\to\mathscr{P}(\mathbb{R})\},
\qquad
d: \bar d_p\ \text{(maximal Wasserstein)},
\qquad
\mathcal{F}: \text{categorical distributions on fixed atoms.}
```

## Folder Map

```text
01_wasserstein_and_optimal_transport.md   the metric: definitions, duality, properties
02_distributional_bellman_contraction.md  the operator and the gamma-contraction proof
03_control_is_not_a_contraction.md        counterexamples; what still converges
04_c51_categorical_projection.md          the algorithm: atoms, projection, KL loss
05_quantile_regression_cramer.md          minimizing W1 directly; the Cramér alternative
```
