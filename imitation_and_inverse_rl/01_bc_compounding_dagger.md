# Behavior Cloning, Compounding Error, And DAgger

## Behavior Cloning

Reduce imitation to supervised learning: fit
`pi-hat = argmin E_{s ~ d_expert}[loss(pi(s), pi_expert(s))]`. Suppose
training succeeds:

```math
\mathbb{E}_{s\sim d_{\pi^*}}\big[\mathbb{1}\{\hat\pi(s)\neq\pi^*(s)\}\big]\ \le\ \varepsilon
\qquad\text{(}\varepsilon\text{-accurate ON THE EXPERT'S distribution).}
```

## The Compounding Theorem

**Theorem (Ross–Bagnell 2010).** For an episodic task of horizon `H`
with per-step cost in `[0,1]`:

```math
J(\hat\pi)\ \le\ J(\pi^*)\ +\ H^2\,\varepsilon
\qquad\text{— and the }H^2\text{ is TIGHT.}
```

*Proof of the upper bound.* Couple the two policies' trajectories: they
agree until `pi-hat`'s first mistake. At each step `t`, the probability
of a first mistake is at most `eps` (the state is still on-distribution
until then — this is where the training guarantee applies); once off
the expert's states, nothing is guaranteed, and the remaining `H - t`
steps may each cost 1. Summing:

```math
J(\hat\pi)-J(\pi^*)\ \le\ \sum_{t=1}^{H}\ \varepsilon\,(H-t+1)\ \le\ H^2\varepsilon\ (\text{up to }\tfrac12). \qquad\blacksquare
```

*Tightness.* The tightrope MDP: a chain where the expert walks the edge;
any deviation falls into an absorbing zero-reward abyss for the rest of
the episode. Per-step error `eps` on-distribution converts to
`~ eps H` probability of falling, times `~H` lost steps: `Theta(eps H^2)`
realized. ∎

**The mechanism is distribution shift, not learning failure**: the
learner was `eps`-good exactly where it was trained (`d_{pi*}`) and is
*evaluated* on its own induced distribution `d_{pi-hat}` — the same
train/test measure mismatch as `offline_rl_and_ope/06` and
`global_convergence_of_pg/04` (one more entry for the conservation
ledger), here with a quadratic price because each error compounds
through the remaining horizon.

## DAgger: Interaction Buys Back One H

**The fix:** query the expert *on the learner's own states*. DAgger
(Dataset Aggregation, Ross–Gordon–Bagnell 2011) iterates:

```text
for i = 1..N:
    run pi_i; collect states s ~ d_{pi_i}; label them with expert actions;
    aggregate into D; pi_{i+1} = supervised fit on ALL of D
return best pi_i
```

**Theorem.** If the supervised learner is no-regret over the sequence of
aggregated datasets (e.g. FTL on a convex loss), then some returned
policy satisfies

```math
J(\hat\pi)\ \le\ J(\pi^*)\ +\ u\,H\,\big(\varepsilon_N+o(1)\big),
```

with `eps_N` the best achievable training error *on the learner's own
distribution* and `u` a per-mistake cost bound: **linear in `H`.**

*Proof shape.* DAgger is **online learning against the sequence of
distributions `d_{pi_i}`** — each round's loss is the classification
error under the current policy's own visitation. The no-regret property
(`stochastic_games/03`'s machinery, used verbatim: average regret
`-> 0`) guarantees the *average* policy errs at most `eps_N + regret/N`
under its own distribution; and an error bound that holds under the
policy's OWN distribution converts to value loss with one `H` (each
step's mistake costs at most `u` in expectation forward — the
performance difference lemma, `policy_gradient/03`, with the roles
arranged so the mismatch term never appears because the measure is
already the learner's). ∎

```text
the exchange, in the family's terms:
BC:      certify on d_expert, execute on d_learner  -> pay the mismatch: H^2
DAgger:  certify on d_learner (by querying there)   -> no mismatch: H
price:   an interactive expert — often the expensive resource; the whole
         design space of imitation (DAgger variants, disagreement-based
         querying, safe interventions) is negotiating this price.
```

## Where This Sits Today

Behavior cloning at scale (large expert corpora, strong function
classes) is the *pretraining* of LLMs and robot foundation models —
`eps` driven small enough that `eps H^2` is tolerable for moderate `H`.
The theorem still governs the margins: long-horizon agentic rollouts
revive the quadratic (small per-token error, compounding drift — the
observed degradation of long agent trajectories), and the modern
correctives are DAgger's descendants in disguise: on-policy correction
(RL fine-tuning — `rlhf_mathematics/06`), self-generated-data filtering
(certify on your own distribution), and verifier-gated execution.

## What Remains Open

Optimal query budgets (when is each expert label worth most — an
information-directed question, `regret_and_exploration/06`); compounding
bounds under partial observability (the coupling argument needs the
state); and sharp constants for stochastic experts, where "mistake"
itself needs the divergence language of notebook 02.
