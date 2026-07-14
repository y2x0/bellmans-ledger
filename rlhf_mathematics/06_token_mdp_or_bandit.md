# Token MDP Or Contextual Bandit?

## The Two Framings

Generation as a decision process, per prompt `x`:

```text
token MDP:   state = (x, y_{<t}); action = next token; transition =
             DETERMINISTIC append; reward = 0 until EOS, then r(x, y).
             Horizon = generation length. gamma = 1.
sequence     one state (the prompt); action = the whole completion y;
bandit:      reward r(x,y). Horizon 1.
```

Both are formally correct MDPs. The question with content: does the token
MDP's *structure* buy anything — i.e. does RLHF need `H > 1` machinery?

## The Collapse Proposition

**Proposition.** In the token MDP (deterministic transitions, terminal
reward), GAE with `lambda = 1` (`policy_gradient/05`) reduces PPO's
advantage, at every token `t`, to

```math
\hat A_t\ =\ r(x,y)\ -\ V(x,y_{<t_0}\ldots)\Big|_{\text{baseline at }t}
\qquad\text{i.e. (sequence reward)}-\text{(state baseline)} .
```

*Proof.* `A-hat_t = sum_{k>=t} (gamma lambda)^{k-t} delta_k` with
`delta_k = r_k + V(s_{k+1}) - V(s_k)`. With `gamma = lambda = 1` the sum
telescopes: intermediate `V`s cancel pairwise, `r_k = 0` except the last,
leaving `A-hat_t = r(x,y) - V(s_t)` (with `V(terminal) = 0`). ∎

So sequence-level PPO ("the bandit view") and token-level PPO with
`lambda = 1` are the **same algorithm** up to the baseline's granularity:
every token of a completion receives the same scalar signal, recentred by
a per-prefix value. What `lambda < 1` would add — bootstrapping through a
learned per-token `V` — inserts critic bias with *no variance-reduction
payoff from the environment side*, because the environment contributes no
per-step noise (transitions are deterministic; the only randomness is the
policy's own sampling and the terminal reward). This is
`policy_gradient/05`'s bias–variance dial evaluated at the corner
"all randomness is at the end": the dial's variance argument for
bootstrapping evaporates.

**Where token structure DOES pay:** the per-token KL penalty
`beta log(pi/pi_ref)(a_t | s_t)` is a genuine per-step reward — dense,
known, differentiable — and per-token baselines/values reduce variance of
*credit within the sequence* (which tokens moved the reward). The honest
summary: RLHF-as-practiced is a contextual bandit with (i) an
exponentially structured action space factorized by the policy, and (ii)
a dense KL shaping term; it is not using — and does not need — deep
RL's machinery for stochastic multi-step credit assignment.

## GRPO, Read In This Frame

GRPO (Shao et al. 2024) drops the critic entirely: sample a **group** of
`G` completions per prompt, use the group's statistics as the baseline:

```math
\hat A^{(i)}
=
\frac{r_i-\mathrm{mean}(r_1..r_G)}{\mathrm{std}(r_1..r_G)},
\qquad\text{(applied to every token of completion }i\text{)},
```

with PPO's clipped ratio and a KL term. This family recognizes each
piece:

```text
group mean      = a Monte Carlo estimate of V(x) — the per-prompt value —
                  from siblings: a leave-one-out-flavored control variate
                  (policy_gradient/02's optimal-baseline logic, estimated
                  from the batch rather than a network; RLOO is the exact
                  leave-one-out version, mean over the other G-1);
group std       = per-prompt reward scale normalization — the per-prompt
                  effective-temperature correction that 01 (point 2) said
                  a global beta lacks; it also introduces a small bias
                  (a random, correlated denominator) accepted for
                  robustness;
no critic       = the Collapse Proposition says the critic was only a
                  baseline anyway; for G samples/prompt, the empirical
                  baseline is cheaper and unbiased-ish, at the price of
                  G× generation compute per update.
```

GRPO is thus *not* a new RL principle: it is the bandit view taken
seriously, with the variance reduction moved from a learned value network
into sampling. Its empirical success in reasoning RL (where `r` is a
verifier: exact, ungameable within its domain — removing 05's noise term)
is consistent with this reading: when the reward is clean and the problem
is a bandit, the bandit algorithm wins on simplicity.

## Where Genuine Multi-Step Structure Returns

```text
1. multi-turn dialogue/agentic loops: the environment (user, tools,
   world) injects stochastic transitions BETWEEN the policy's outputs —
   real states, real credit assignment, gamma < 1 meaningful again;
2. intermediate verification (process reward models): nonzero rewards at
   internal steps — the terminal-reward collapse no longer applies, and
   lambda < 1 / per-step values regain their role;
3. tree search over reasoning steps (search_and_planning/ applied to
   token prefixes): value functions over prefixes as leaf evaluators —
   AlphaGo's decomposition, with the LLM as both prior and environment.
```

Each is an active frontier precisely because it re-imports a family of
this repository (stochastic_approximation, policy_gradient's full dial,
search_and_planning) that vanilla RLHF's bandit collapse had excused.

## What Remains Open

Principled process-reward credit (the PRM-vs-ORM question is
`policy_gradient/05`'s dial plus 01's identifiability, unresolved
jointly); variance theory for group baselines under small `G` and
correlated sampling; and the right discounting/KL placement for
multi-turn RLHF — the token/turn/episode hierarchy has no settled
operator theory yet.
