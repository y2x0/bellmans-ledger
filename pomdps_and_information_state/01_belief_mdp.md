# The Belief MDP

## The Model

A POMDP adds an observation channel to the MDP:

```math
(\mathcal{S},\mathcal{A},P,r,\gamma)\ +\ (\mathcal{O},\ Z),
\qquad
Z(o\mid s',a)=\Pr\{O_{t+1}=o\mid S_{t+1}=s',A_t=a\},
```

with the agent seeing only `(A_0, O_1, A_1, O_2, ...)` — never `S_t`.
Policies map **histories** to actions; the exponential blowup of
`Pi^{HR}` that `mdp_foundations/01` collapsed is back, and this time the
collapse to stationary-Markov-in-`S` is *false* (memory genuinely
helps — the marginal-matching proof there required observing `s`).

## The Bayes Filter

The **belief** is the posterior over the hidden state:

```math
b_t(s)=\Pr\{S_t=s\mid \text{history}_t\} .
```

**Update (one step of Bayes).** After action `a` and observation `o`:

```math
b'(s')
=
\frac{Z(o\mid s',a)\ \sum_{s}P(s'\mid s,a)\,b(s)}
     {\underbrace{\sum_{s''}Z(o\mid s'',a)\sum_sP(s''\mid s,a)\,b(s)}_{=\ \Pr\{o\mid b,a\}\ =:\ \eta(b,a,o)}}
\ \ =:\ \tau(b,a,o)(s').
```

Predict through the dynamics, weight by the observation likelihood,
normalize — a recursively computable statistic of the history
(each update needs only `(b, a, o)`, not the past).

## The Sufficiency Theorem

**Theorem.** The belief process is Markov, and the belief is sufficient
for optimal control: define the **belief MDP** with state space
`Delta(S)`, transitions

```math
\Pr\{b_{t+1}=\tau(b,a,o)\mid b,a\}=\eta(b,a,o),
\qquad
\tilde r(b,a)=\sum_s b(s)\,r(s,a);
```

then (i) the belief MDP's optimal value at `b(history)` equals the
POMDP's optimal value at that history, and (ii) any optimal policy of
the belief MDP, applied as `a_t = pi(b_t)`, is optimal for the POMDP.

*Proof.* Two invocations of conditional expectation:

**(Markovness/consistency)** Given the history, the joint law of
`(R_{t+1}, O_{t+1})` under action `a` depends on the history only
through `b_t`:

```math
\Pr\{o,\,s'\mid \text{hist},a\}
=
\sum_s b_t(s)\,P(s'\mid s,a)\,Z(o\mid s',a),
```

by the tower property over `S_t | hist ~ b_t` — the history enters only
through the posterior it induces, which is what "posterior" means. Hence
rewards and belief transitions are functions of `(b, a)` alone: the
belief process is a controlled Markov chain with the stated kernel.

**(Value equality)** With the reward identity
`E[r(S_t, a) | hist] = tilde-r(b_t, a)`, every history-policy's expected
return is reproducible by a belief-policy (condition every term of the
return on the filtration and collapse via the first part, inductively
backward from horizon), and conversely belief-policies are
history-policies. Suprema over the two coincide. ∎

**What was actually used — the audit:**

```text
known model (P, Z)    -> the filter is COMPUTED, not learned; unknown-
                         model POMDPs must learn the statistic too (04, 05)
exact posterior       -> approximate filters (particles, learned RNN
                         states) void (i) exactly by their approximation
                         error — priced in 05
reward = f(s, a)      -> reward observable through b; reward depending on
                         the HISTORY itself would break the collapse
```

## The Price: A Continuous Simplex State

Everything from `mdp_foundations/` now applies **on `Delta(S)`** — the
Bellman operator on functions of `b` is a `gamma`-contraction in
`||.||_inf` (same proof; the max-lemma and the expectation over `o`
compose as always), value iteration converges, greedy-w.r.t.-`v*` is
optimal. What is lost is *finiteness*: `Delta(S)` is a
`(|S|-1)`-dimensional continuum, so the tabular algorithms are
inapplicable as stated. The two escapes:

```text
structure:  v* on the simplex is not arbitrary — it is piecewise linear
            and convex (finite horizon), enabling exact finite
            representations: notebook 02;
hardness:   but the representations grow doubly exponentially, and
            notebook 03 shows no algorithm does fundamentally better in
            the worst case.
```

## Two Working Intuitions

```text
1. beliefs mix planning and estimation: the belief transition depends on
   the agent's own action — actions now have INFORMATION VALUE (move to
   see), and optimal POMDP behavior includes deliberate sensing; the
   exploration/exploitation tension of regret_and_exploration/ appears
   INSIDE a single known model (and indeed bandits-with-learning ARE
   POMDPs with the unknown parameter as hidden state — PSRL's posterior,
   regret_and_exploration/05, is a belief state);
2. the belief is a compression certificate: b_t is THE minimal
   sufficient statistic exactly when the filter is minimal (04 shows
   smaller PREDICTIVE statistics can exist when the reward/dynamics
   don't need the full posterior).
```

## What Remains Open

Nothing in the classical reduction; the open ground is entirely in
computing/learning with it — notebooks 02–05.
