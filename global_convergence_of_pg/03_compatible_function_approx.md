# Compatible Function Approximation

## The Theorem Nobody Reads (Sutton–McAllester–Singh–Mansour 1999, Part 2)

The policy gradient theorem's integrand is the true `q_pi`. Practice
substitutes a learned critic `Q_w`. The 1999 paper's second theorem gives
the exact condition under which the substitution is **not** an
approximation:

**Theorem.** Suppose the critic class is the policy's own score space,

```math
Q_w(s,a)=w^\top\underbrace{\nabla_\theta\log\pi_\theta(a\mid s)}_{=:\ \psi_\theta(s,a)}
\qquad\text{("compatible features"),}
```

and `w*` minimizes the on-policy weighted squared error

```math
w^*=\arg\min_w\ \mathbb{E}_{s\sim d^{\pi},\,a\sim\pi}\Big[\big(Q_w(s,a)-q_\pi(s,a)\big)^2\Big].
```

Then the policy gradient computed with the *approximate* critic is exact:

```math
\mathbb{E}_{d^\pi,\pi}\big[\psi_\theta(s,a)\,Q_{w^*}(s,a)\big]
=
\mathbb{E}_{d^\pi,\pi}\big[\psi_\theta(s,a)\,q_\pi(s,a)\big]
=
(1-\gamma)\,\nabla_\theta J(\theta).
```

*Proof.* First-order optimality of `w*`:

```math
\mathbb{E}\big[\psi\,\big(Q_{w^*}-q_\pi\big)\big]=0
\qquad\text{(the residual is orthogonal to the features)},
```

and since the gradient estimator pairs the critic *against exactly those
features* `psi`, the residual's contribution vanishes identically:
substitute `q_pi = Q_{w*} + (q_pi - Q_{w*})` into the pairing. ∎

The critic can be arbitrarily wrong **as a value function** — all that
matters is its projection onto the score space, and least squares gets
that projection right by construction. The gradient only ever consults
`d`-many directions of the infinite-dimensional `q_pi`; compatibility
means fitting precisely those.

Two structural corollaries:

```text
1. compatible features have zero mean under pi(.|s)
   (E_a[grad log pi] = 0), so Q_w is automatically an ADVANTAGE estimate —
   the baseline of policy_gradient/02 is built in;
2. for softmax, psi_{s,a} = e_{s,a} - pi(.|s): compatible critics are
   per-state centered tables — advantage functions exactly.
```

## Compatible Critic Weights ARE The Natural Gradient

**Proposition (Kakade 2001).** With compatible `Q_{w*}`:

```math
w^*\ =\ F(\theta)^{-1}\,\nabla_\theta J(\theta)\cdot(1-\gamma)
\qquad\text{— the natural gradient direction itself.}
```

*Proof.* Two displays already on the table:

```math
F(\theta)=\mathbb{E}\big[\psi\psi^\top\big]
\quad\text{(Fisher, policy_gradient/04)},
\qquad
\mathbb{E}\big[\psi\,\psi^\top\big]w^*=\mathbb{E}\big[\psi\,q_\pi\big]
\quad\text{(normal equations)},
```

and the right side of the normal equations is `(1-gamma) grad J` by the
policy gradient theorem. So `F w* = (1-gamma) grad J`. ∎

**Reading:** the natural gradient is not "gradient, then precondition" —
it is *what a correctly-specified critic's weights already are.* An
actor-critic with a compatible critic and step `theta += eta w` is doing
NPG (notebook 02) without ever forming `F`. This identity is the
load-bearing link between the critic-fitting view of actor-critic and the
information-geometric view of trust regions.

## Why Practice Ignores The Theorem (and what it pays)

```text
1. compatible features move with theta: the critic class must be re-fit
   from scratch every policy step (no warm-started deep critic persists);
2. the class is tiny (dim = dim theta, linear): high critic VARIANCE per
   batch — the theorem trades all bias for variance, and TD-style critics
   (biased, low-variance: policy_gradient/05's dial) usually win in
   sample-constrained practice;
3. the least-squares must be under d^pi with pi's OWN actions: replayed
   data breaks the orthogonality that carried the proof.
```

The honest summary: modern actor-critics accept a *biased* gradient (their
critic's residual is not orthogonal to the scores) in exchange for critic
sample-efficiency, and the size of that bias is exactly the
"approximation error transferred through the score pairing" that
notebook 04's theorem prices as `sqrt(transfer error)`.

## What Remains Open

Compatibility for deep policies (the score space is the NTK-ish tangent
space — partially explored); and quantifying the incompatibility bias of
standard TD critics as a function of the feature mismatch, for which only
coarse bounds exist.
