# Global Convergence Of Policy Gradients

This family asks:

```text
policy_gradient/07 called J(theta)'s nonconvexity "inherent." Yet softmax
policy gradient provably reaches GLOBAL optima. What is the actual
geometry, and what does it cost?
```

The resolution: `J(theta)` is nonconvex but not adversarially so — for
natural parameterizations it satisfies **gradient-domination
(Łojasiewicz-type) inequalities** whose constants encode the real
difficulty (exploration), and whose preconditioned (natural-gradient/
mirror-descent) versions delete the bad constants entirely, at the price
of a distribution-mismatch term that no policy method escapes.

Prerequisites: `policy_gradient/01–04`,
`regularized_mdps_and_duality/01, 05`; `mdp_foundations/06` for the
geometry file.

## Folder Map

```text
01_softmax_gradient_domination.md   the non-uniform Łojasiewicz inequality;
                                    O(1/t) global rate; the exp(H) plateau
02_npg_linear_convergence.md        NPG dimension-free O(1/t); entropy-
                                    regularized NPG converges LINEARLY
03_compatible_function_approx.md    Sutton et al. Part 2 proved; compatible
                                    critic = natural gradient
04_npg_with_approximation.md        the transfer-error theorem; the mismatch
                                    coefficient as the irreducible price
05_geometry_synthesis.md            the occupancy polytope picture assembled
```
