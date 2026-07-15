# Failure Modes Of Imitation And Inverse RL

## 1. Compounding Drift (violated: train-test measure match)

```text
mechanism:   notebook 01's theorem — eps on d_expert becomes eps H^2 in
             value; the quadratic is REALIZED whenever small errors move
             the learner somewhere the demos never went
signature:   long-horizon rollouts degrade smoothly then catastrophically;
             short-horizon metrics look fine
levers:      DAgger-style interaction; on-policy fine-tuning (RLHF's
             actual historical role for LLMs); noise-injected experts
             (DART) — widen d_expert to cover the drift corridor
```

## 2. Reward Ambiguity (violated: identifiability)

```text
mechanism:   notebook 02's ill-posedness — shaping-equivalent rewards
             fit demonstrations equally; the selected reward is an
             artifact of the selection principle (entropy, regularizer,
             feature class)
signature:   recovered r-hat transfers badly to changed dynamics or
             horizons EVEN when imitation in-domain is perfect (the
             equivalence class disperses exactly when P or gamma change
             — shaping-invariance is dynamics-relative)
levers:      demos from MULTIPLE environments (intersect the feasible
             sets); preferences on top of demos (rlhf_mathematics/01
             adds difference information demos lack); report the
             equivalence class, not a point estimate — rarely done
```

## 3. The Demonstrator Ceiling (violated: nothing — a scope fact)

```text
mechanism:   every method in this family targets the expert's occupancy
             or conditionals: J(learner) <= J(expert) + optimization
             noise, BY DESIGN
signature:   plateaus exactly at demonstrator skill
levers:      leaving the family — RL on a task signal (rlhf_mathematics/
             07's RLVR is the clean modern instance: imitation-pretrain,
             then verifier-reward RL exceeds the demonstrators);
             offline-RL stitching (offline_rl_and_ope/04) can exceed
             demos where they contain complementary segments
```

## 4. Causal Confusion (violated: the state is what you think it is)

```text
mechanism:   the policy latches onto features CORRELATED with expert
             actions rather than their causes (the brake-light indicator
             predicting braking); occupancy matching cannot distinguish
             cause from correlate inside the demo distribution
signature:   in-distribution imitation excellent, catastrophic under the
             lightest intervention — the imitation-family's version of
             Goodhart (rlhf_mathematics/05): optimizing a proxy
             (feature correlation) that decouples from the target under
             the policy's own influence
levers:      interventional data (DAgger again, doing double duty);
             causal feature selection; graph priors — none settled
```

## 5. Adversarial Optimization Pathologies (GAIL-specific)

```text
mechanism:   stochastic_games/05's rotational dynamics: discriminator-
             policy cycling, discriminator saturation (gradients vanish
             when it wins), mode collapse onto easy occupancy regions
signature:   oscillating imitation quality; sensitivity to update ratios
levers:      gradient penalties, spectral norm, divergence swaps
             (Wasserstein), primal formulations (ValueDICE) that delete
             the inner max entirely
```

## The Family's One-Sentence Summary

Imitation methods differ in **which distribution they certify errors
under** (expert's: BC; learner's: DAgger, GAIL) and **which object they
match** (conditionals: BC/DAgger; occupancies: GAIL/DICE; selected
rewards: IRL) — and every failure above is a mismatch between the
certified thing and the deployed thing, the repository's oldest theme
wearing its fourth costume.
