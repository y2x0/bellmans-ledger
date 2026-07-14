# Control Is Not A Contraction

## The Optimality Operator

The distributional analogue of `T` applies a greedy selection **by means**:

```math
(\mathcal{T}Z)(s,a)
=
\mathrm{Law}\Big(R(s,a)+\gamma\,Z\big(S',\ a^*(S')\big)\Big),
\qquad
a^*(s')=\arg\max_{a'}\ \mathbb{E}\big[Z(s',a')\big].
```

The mean of the output still obeys the classical optimality backup — so the
first-moment theory is untouched. The question is what happens to the rest
of the distribution.

## Pathology 1: Not A Contraction (paper, Prop. 1)

There exist `Z_1, Z_2` with

```math
\bar d_1(\mathcal{T}Z_1,\ \mathcal{T}Z_2)\ >\ \bar d_1(Z_1,\ Z_2).
```

*Mechanism of the counterexample.* Construct a state with two actions whose
**means are within `eps`** but whose **distributions are far apart** (say,
one deterministic, one a wide two-point law). Let `Z_1` and `Z_2` differ by
an `eps`-perturbation that flips which action has the larger mean. Then
`T Z_1` and `T Z_2` bootstrap **different actions' entire distributions**
into the backup: input distance `O(eps)`, output distance `O(1)`. The
argmax is a discontinuous selector, and the operator inherits the
discontinuity — amplification, not contraction. ∎

Note the exact contrast with the classical max-lemma
(`mdp_foundations/03`): `|max f - max g| <= max|f - g|` holds for *scalars*
because the max's *value* is continuous in its arguments even when the
argmax jumps. Distributionally, the operator returns the *argument* (the
selected distribution), not the value — and the argument jumps.

## Pathology 2: Fixed Points May Not Exist

The paper exhibits MDPs (with mean-ties between actions) where **no**
`Z` satisfies `T Z = Z`: any candidate's tie-breaking is unstable under the
operator, which cycles among the distributions of the tied optimal policies.
When ties are broken consistently and the optimal policy is unique, `Z^{pi*}`
is a fixed point — but uniqueness of `pi*` is an assumption about the MDP,
not something the operator provides.

## Pathology 3: What Convergence Survives (paper, Thm. 1)

Iterating `Z_{k+1} = T Z_k`:

```math
\mathbb{E}\big[Z_k\big]\ \longrightarrow\ Q^*
\quad\text{uniformly (the classical theory, embedded),}
```

and, if there is a unique optimal policy, `Z_k -> Z^{pi*}` in `d_p-bar`.
In general one gets only convergence to the set of **nonstationary optimal
value distributions** — laws realizable by time-varying sequences of optimal
policies: the iterates keep switching among mean-optimal continuations, and
the distributional tail of `Z_k` reflects the switching history rather than
any single policy.

```text
summary table
              evaluation T^pi                control T
fixed point   exists, unique (= Z^pi)        may not exist
contraction   gamma, in d_p-bar, any p       provably not (in any d_p-bar)
convergence   geometric, all moments         means only; distributions to a
                                             nonstationary set (unique pi* =>
                                             full convergence)
```

## Why Practice Survives The Theory

Three reasons the pathologies do not sink the algorithms built in
notebook 04:

```text
1. The means still converge — control quality rests on the first moment,
   which retains the full classical guarantee.

2. The pathologies are driven by exact mean-ties and argmax flips; under
   function approximation with SGD, updates average over minibatches and
   the effective policy changes slowly — the "chattering" is damped
   (the paper argues distributional learning reduces chattering in
   practice, by giving the network a richer, more stable target).

3. The distributional part is doing its real work as an AUXILIARY
   representation-learning signal and a better-conditioned regression
   target; its own fixed-point exactness is not what the control
   performance is resting on.
```

The honest statement: distributional **control** is used as if the
evaluation theory licensed it, the license is known to be invalid in
general, and the empirical returns (notebook 04) justified the trespass.
Later theory (e.g. on categorical/quantile dynamics with sufficiently rich
parameterizations) recovers guarantees in restricted settings, but the
gap identified in this 2017 paper is still the honest frame.
