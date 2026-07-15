# Successor Representations

This family asks:

```text
The resolvent (I - gamma P_pi)^{-1} appears in every proof in this
repository. What happens if you LEARN it as an object?
```

The successor representation (Dayan 1993) is exactly that: the
discounted expected future occupancy matrix, TD-learnable, whose inner
product with any reward vector is a value function (01). Successor
features (Barreto et al. 2017,
`papers/successor-features-barreto-2017.pdf`) generalize it to features
and prove the **generalized policy improvement** theorem — zero-shot
transfer across rewards with a quantified guarantee (02). The limits —
policy dependence, dynamics changes — close the family (03).

Prerequisites: `mdp_foundations/01–02, 05`,
`stochastic_approximation/03–04`.

## Folder Map

```text
01_sr_as_learned_resolvent.md    M = (I-gamma P)^{-1}; TD on occupancies;
                                 spectral structure and eigenoptions
02_successor_features_gpi.md     psi, r = phi.w; the GPI theorem with its
                                 bound, proved
03_scope_and_limits.md           what transfers (rewards) and what cannot
                                 (dynamics, policies); UVFA contrast
```
