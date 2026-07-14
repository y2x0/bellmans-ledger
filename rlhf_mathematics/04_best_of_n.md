# Best-of-n: The Zeroth-Order Competitor

## The Policy

Sample `n` completions i.i.d. from `pi_ref`, output the one with the
highest reward:

```math
\pi_{\mathrm{BoN}}(y)\ =\ \Pr\big\{\text{argmax}_{i\le n}\,r(Y_i)=y\big\},
\qquad Y_i\overset{iid}{\sim}\pi_{\mathrm{ref}} .
```

No training, no gradient, one hyperparameter. For continuous rewards (no
ties), the density is explicit: `y` wins iff the other `n-1` draws score
below it, so

```math
\pi_{\mathrm{BoN}}(y)\ =\ n\,\pi_{\mathrm{ref}}(y)\,F\big(r(y)\big)^{\,n-1},
\qquad
F(t)=\Pr_{\pi_{\mathrm{ref}}}\{r(Y)\le t\}
```

— a reweighting of the reference by a *rank-based* tilt `n F^{n-1}`,
where 02's optimum used the *exponential* tilt `e^{r/beta}/Z`. Both are
monotone in `r`; the comparison of the two tilts is this file.

## The KL Formula

**Proposition.**

```math
\mathrm{KL}\big(\pi_{\mathrm{BoN}}\ \|\ \pi_{\mathrm{ref}}\big)
\ = \
\log n-\frac{n-1}{n}
\qquad\text{(exactly, for atomless }r(Y)\text{)}.
```

*Proof.* From the density above,

```math
\mathrm{KL}
=
\mathbb{E}_{\mathrm{BoN}}\Big[\log\big(n\,F(r(y))^{n-1}\big)\Big]
=
\log n+(n-1)\,\mathbb{E}\big[\log U_{(n)}\big],
```

where `U_(n) = F(r(Y_win))` is the maximum of `n` i.i.d. `Uniform[0,1]`
(probability integral transform). Its density is `n u^{n-1}`, so

```math
\mathbb{E}[\log U_{(n)}]=\int_0^1 n\,u^{n-1}\log u\,du=-\frac1n,
```

giving `log n - (n-1)/n`. ∎ (With ties/atoms, the formula is an upper
bound.)

**Read it:** the divergence cost of selection grows only
**logarithmically in `n`** — doubling the compute costs `~log 2` extra
nats. Selection is an extraordinarily KL-cheap way to move mass toward
high reward.

## Near-Optimality On The Frontier

How does the rank tilt compare with the exponential tilt at *matched* KL?

```text
- among ALL policies at KL budget k, 02's exponential tilt maximizes
  expected TRUE reward (that is exactly 02's theorem, read as: the
  frontier is achieved by the tilted family);
- BoN at n ~ e^{k+ (n-1)/n} sits at budget k; results of Yang et al. /
  Beirami et al. (2024) show its expected reward is within o(1)/second-
  order terms of the frontier value for well-behaved reward
  distributions: the rank tilt n F^{n-1} and the calibrated exponential
  tilt concentrate on the same upper tail. BoN is near-frontier-optimal,
  not merely cheap.
```

Heuristic for the matching: at large `n`, BoN conditions on the top
`1/n` quantile of `pi_ref`'s reward distribution; the exponential tilt at
`beta` matched to give the same mean reward conditions similarly on the
tail — both are tail-conditionings, differing at second order in tail
shape.

## Why This Is The Mandatory Baseline

```text
1. it consumes the same two artifacts as PPO-RLHF (pi_ref, r) with no
   training instability, no policy_gradient/07 failure modes;
2. its KL is a KNOWN CONSTANT (log n - (n-1)/n) rather than an outcome
   of tuning — the frontier point is chosen, not discovered post hoc;
3. empirically it matches or beats trained RLHF at matched KL in many
   setups; a trained method that cannot beat BoN at its own KL point has
   negative value-added (the honest evaluation demanded by 05's
   overoptimization analysis, since both suffer it identically through
   the shared r);
4. its costs are at INFERENCE (n forward passes per query) vs PPO's at
   training — the comparison is a compute-placement decision, and
   distillation of BoN back into a policy (BoN-SFT / iterative BoN)
   converts one to the other, blurring into an RLHF algorithm in its own
   right.
```

## The Selection–Optimization Duality

BoN is `value_based_deep_rl/03`'s `E[max]` structure used
*constructively*: selection bias, which was the villain of Q-learning
(and will be the precise mechanism of 05's Goodhart analysis), is here
the entire engine — bias toward high `r` is the point, and the
`log n - (n-1)/n` formula prices exactly how much distributional
distortion buys how much selection. The same order statistics, signed
opposite.

## What Remains Open

Sharp non-asymptotic frontier gaps for realistic (heavy-tailed,
miscalibrated) reward distributions; optimal *adaptive* n per prompt
(spend samples where reward variance is high — an information-allocation
question); and the theory of iterated BoN-distillation as a fixed-point
scheme (each round re-tilts; the composition is an exponential-tilt
ladder whose stability under a LEARNED r is 05's problem).
