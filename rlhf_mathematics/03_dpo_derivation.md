# DPO: The Derivation In Full

## The Inversion

Solve 02's closed form for the reward:

```math
\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\,e^{r(x,y)/\beta}
\quad\Longleftrightarrow\quad
r(x,y)=\beta\,\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x).
```

**Every reward is expressible through the policy it induces** — up to the
`beta log Z(x)` term, which is a pure per-prompt shift, i.e. exactly the
direction Bradley–Terry cannot see (01's identifiability class). This
alignment of the two nuisances is the entire trick:

## Substitute Into Bradley–Terry

For a preference pair `(y_w, y_l)` at prompt `x`, the BT probability
depends on the reward *difference*, where `beta log Z(x)` cancels:

```math
r(x,y_w)-r(x,y_l)
=
\beta\log\frac{\pi(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\beta\log\frac{\pi(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)} .
```

The preference MLE (01) becomes a loss **directly on the policy**:

```math
\boxed{\
\mathcal{L}_{\mathrm{DPO}}(\pi)
=
-\,\mathbb{E}_{(x,y_w,y_l)}\bigg[\log\sigma\Big(
\beta\log\tfrac{\pi(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\beta\log\tfrac{\pi(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\Big)\bigg]\ }
```

**Theorem (exactness).** Minimizing `L_DPO` over unrestricted policies is
equivalent to: fit `r` by BT-MLE (01), then solve the KL-regularized
problem (02) exactly. *Proof:* the map `pi <-> r` above is a bijection
between (policies with `pi_ref`'s support) and (rewards modulo per-prompt
shift); the loss is the BT likelihood transported through it; the
minimizer over `r`-space pushes forward to `pi* = pi*(r_MLE)`. ∎

No reward model is trained, no samples are generated, no `Z` is
estimated: supervised learning on preference pairs, with the RL problem
solved *by the parameterization*.

## The Implicit Reward And The Gradient

DPO's fitted policy defines `hat-r(x,y) = beta log(pi/pi_ref)` — the
**implicit reward**, usable diagnostically like any reward model.
Differentiating the loss:

```math
\nabla\mathcal{L}_{\mathrm{DPO}}
=
-\beta\,\mathbb{E}\Big[
\underbrace{\sigma\big(\hat r(x,y_l)-\hat r(x,y_w)\big)}_{\text{weight: how WRONG the pair is}}
\Big(\nabla\log\pi(y_w\mid x)-\nabla\log\pi(y_l\mid x)\Big)\Big].
```

Read it: raise the winner's likelihood, lower the loser's, with per-pair
weight = the model's current probability of *mis-ranking* — already-
correct pairs contribute ~0. It is weighted contrastive log-likelihood;
the weight is what distinguishes it from naive "SFT on winners, unlearn
losers," and removing it demonstrably degrades to that.

## What Is Lost vs PPO-RLHF (the honest ledger)

```text
1. off-distribution generalization of the reward: an explicit reward
   model is queried ON THE POLICY'S OWN SAMPLES during PPO — it
   interpolates/extrapolates to fresh completions. DPO's implicit reward
   is fit only where the preference data sits; the policy is never
   evaluated on its own samples during training (no on-policy term
   anywhere). Consequence: DPO is anchored to the preference
   distribution; PPO explores the reward landscape (for better — 
   capability elicitation — and worse — 05's overoptimization).

2. the unrestricted-class assumption: exactness needs the bijection to be
   realizable; with a finite-capacity pi_theta, DPO minimizes a
   DIFFERENT projection of the BT likelihood than (reward fit +
   regularized RL) would — the two pipelines genuinely diverge under
   misspecification, and which projection is better is unresolved.

3. the KL is implicit, not enforced: nothing in L_DPO measures
   KL(pi || pi_ref) on actual samples; empirical DPO runs drift further
   in sequence-KL than beta "promises" (the loss can be lowered by
   pushing pi(y_l) -> 0 aggressively — mass leaves BOTH pair completions
   toward unseen y, an observed failure mode, motivating variants: IPO's
   bounded loss, conservative/CPO clamps).
```

## The Family It Spawned

One-line taxonomy through this file's lens — all are "choose a link, a
weight, a regularizer for the implicit-reward difference":

```text
IPO:   replace log sigma by a squared hinge on the difference (bounded,
       fixes the pi(y_l)->0 pathology in theory);
KTO:   unpaired data — per-completion utility against a reference point
       (prospect-theoretic link);
SimPO: drop pi_ref, normalize by length — implicit reward = average
       log-prob (changes the identified object, not just the estimator);
GRPO:  not in this family — it is on-policy PG (06), kept distinct.
```

## What Remains Open

The misspecification question (which projection wins, when); principled
control of the actual sequence-KL under DPO-style training; and the
statistics of the implicit reward as an OOD detector — all active,
none settled.
