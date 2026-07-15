# RLVR And DeepSeek-R1: Verifiable Rewards Change The Theory

```text
DeepSeek-AI, 2025. papers/deepseek-r1-2025.pdf
Read against: 05 (whose central pathology this setting deletes) and 06
(whose bandit frame this setting completes).
```

## The Regime Shift

RLHF's mathematics (01–05) was organized around one enemy: the reward is
a **model**, fit from finite preferences, exploitable off-distribution.
Reinforcement learning from verifiable rewards (RLVR) replaces it:

```math
r(x,y)
=
\mathbb{1}\{\texttt{verify}(x,y)=\text{pass}\}
\qquad
\text{(answer checkers, unit tests, format rules)},
```

deterministic, exact on its domain, and — decisively — **not consumed by
optimization**: the Goodhart analysis of 05 had gold-vs-proxy error
`eps(y)` selected by the policy's tilt; here `eps = 0` on the verifier's
domain. The `beta`-KL leash, whose entire job was bounding proxy
exploitation, loses its main client (R1-Zero runs without a reward-side
KL penalty in the headline recipe). What remains of 05 is only its
*systematic* half, relocated: hack the verifier's blind spots
(format tricks, test-case overfitting in code) — a specification
problem, not a statistical one.

## R1-Zero: The Minimal Recipe

Base model + GRPO (06's group-baseline policy gradient) + rule rewards
(accuracy + format) + a template requesting reasoning-then-answer. No
SFT stage, no learned reward model, no process supervision. In this
repository's coordinates, per prompt:

```math
\text{a contextual bandit (06's collapse)}\quad+\quad
\text{exact }r\quad+\quad
\text{multiplicative-weights-style ascent (global_convergence_of_pg/02)},
```

with the group baseline as the per-prompt control variate
(`policy_gradient/02`) and advantage
`A_i = (r_i - mean)/std` broadcast over tokens.

**The observed dynamics, and their reading here:**

```text
1. response length grows spontaneously over training — the policy buys
   more serial computation with more tokens; nothing in the reward asks
   for length: it is instrumental, the policy discovering that deeper
   search (in its own token space) raises pass probability;
2. emergent reflection ("wait — let me re-check...") and the paper's
   "aha moment": self-verification behaviors arise from outcome reward
   alone. The mechanics are BoN internalized (04): sampling a
   candidate, checking it, and resampling INSIDE one trajectory is a
   sequentialized selection procedure — the policy learns to run its
   own rejection sampling because the verifier pays only the final
   answer;
3. pass@k considerations govern everything: GRPO's signal exists only
   when a prompt's group contains BOTH successes and failures
   (all-pass or all-fail groups have zero advantage — std collapses).
   Training lives on the moving frontier of problems the model solves
   SOMETIMES — an automatic curriculum, and the practical reason
   difficulty curation matters more than algorithm choice.
```

Point 3 is the exploration problem (`regret_and_exploration/04`) in its
LLM form: the verifier gives no gradient toward a first success; RLVR
sharpens `pass@k` into `pass@1` but struggles to create successes from
nothing — which is why base-model quality (the prior containing rare
correct traces) is load-bearing, exactly as `rlhf_mathematics/02`'s
support constraint predicted (`pi* ∝ pi_ref e^{r/beta}` can only
reweight the support).

## The Full R1 Pipeline (and why each stage exists)

```text
cold-start SFT (small, curated CoT)     fixes R1-Zero's readability/
                                        language mixing — a support/
                                        format prior, not capability;
reasoning RL (GRPO + rule rewards       the engine, as in R1-Zero, plus
+ language-consistency reward)          one soft reward re-admitting a
                                        modeled term (and its 05-risks);
rejection-sampling SFT (~800k)          BoN distillation (04): sample,
                                        filter by verifier, imitate —
                                        selection converted to weights;
all-scenario RLHF (helpfulness/         the 01–05 machinery returns for
harmlessness preference RMs)            the unverifiable objectives.
```

The pipeline is this family's taxonomy executed in sequence: imitation
for support, verifier-RL for capability, BoN-distillation for
consolidation, preference-RL for the residual objectives. **Distillation
coda:** the paper's small models trained by pure SFT on R1's outputs
beat the same models trained by direct RL — imitation of a stronger
reasoner's filtered traces transfers better than exploring from a weak
prior; `imitation_and_inverse_rl/01`'s ceiling theorem, used as a
feature (the ceiling is above the student).

## The Open Ledger

```text
process vs outcome reward:   PRMs re-introduce learned-reward Goodhart
                             (05) to buy credit assignment (06's
                             multi-step structure) — the trade is live;
verifier scope:              RLVR ends where verification ends; the
                             generalization of verifier-trained
                             reasoning to unverifiable domains is
                             empirically real, theoretically unexplained;
entropy collapse:            heavy RLVR sharpens the policy (pass@1 up,
                             pass@k at large k sometimes DOWN vs base) —
                             the support-narrowing cost of tilting,
                             02's frontier traded along a new axis;
length as compute:           no theory prices the token-budget/accuracy
                             frontier the length growth is climbing —
                             the newest open problem this repository
                             can state precisely and cannot yet solve.
```
