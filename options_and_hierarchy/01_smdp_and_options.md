# Options And The Semi-MDP

## The Object

An **option** is a temporally extended behavior:

```math
\omega=(\ \mathcal{I}_\omega,\ \pi_\omega,\ \beta_\omega\ ),
\qquad
\mathcal{I}_\omega\subseteq\mathcal{S},\quad
\pi_\omega(a\mid s),\quad
\beta_\omega(s)\in[0,1],
```

— where it may start (initiation set), how it acts while running
(intra-option policy), and the per-state probability it stops
(termination function). A primitive action is the degenerate option
(`beta = 1` everywhere): options strictly generalize actions, so any
theory over options must contain the base theory as the one-step case —
a consistency check used repeatedly below.

## The Multi-Time Model

Executing `omega` from `s` until termination compresses a random-length
trajectory into two aggregate quantities:

```math
r_\omega(s)=\mathbb{E}\Big[\sum_{k=0}^{\tau-1}\gamma^{k}\,r_{t+k}\ \Big|\ s,\ \omega\Big],
\qquad
p_\omega(s'\mid s)=\mathbb{E}\Big[\gamma^{\tau}\,\mathbb{1}\{S_{t+\tau}=s'\}\ \Big|\ s,\ \omega\Big],
```

with `tau` the (random) duration. **Read `p_omega` carefully:** it is not
a probability distribution — it is a *discounted* sub-probability
(`sum_{s'} p_omega(s'|s) = E[gamma^tau] < 1`). The discount for the
elapsed time is folded into the "transition," which is exactly what makes
the algebra below identical to the primitive case.

## The SMDP Bellman Equations

For a policy-over-options `mu(omega | s)`, values over the option set
`Omega`:

```math
v_\mu(s)=\sum_{\omega}\mu(\omega\mid s)\Big[r_\omega(s)+\sum_{s'}p_\omega(s'\mid s)\,v_\mu(s')\Big],
```

```math
q_*(s,\omega)=r_\omega(s)+\sum_{s'}p_\omega(s'\mid s)\ \max_{\omega'}q_*(s',\omega') .
```

**Proposition (contraction).** The SMDP optimality operator is a
contraction in `||.||_inf` with modulus

```math
\bar\gamma
=
\max_{s,\omega}\ \sum_{s'}p_\omega(s'\mid s)
=
\max_{s,\omega}\ \mathbb{E}\big[\gamma^{\tau(s,\omega)}\big]
\ \le\ \gamma\ <\ 1 ,
```

(each `tau >= 1`), by the identical proof as `mdp_foundations/03` — the
max-lemma composes with a sub-stochastic expectation. ∎

**The modulus is the theorem.** Longer options mean smaller
`E[gamma^tau]`, hence a *stronger* contraction: value information
propagates one **option** per backup instead of one step. An option of
expected length `L` contracts like an effective discount `gamma^L` —
planning with good options shortens the effective horizon from
`1/(1-gamma)` to roughly `1/(1-gamma^L) ~ 1/(L(1-gamma))` backups. This
is the entire *planning* case for hierarchy, stated as one inequality.
(Compare: this is a *designed* instance of the weighted-norm mechanism of
`total_reward_ssp/01` — termination manufactures contraction there,
option completion manufactures extra contraction here.)

## The Consistency Checks

```text
1. Omega = primitive actions  =>  E[gamma^tau] = gamma, both equations
   reduce exactly to mdp_foundations/02's — the theory is a strict
   extension;
2. planning over Omega is planning in a legitimate (sub-stochastic
   kernel) MDP: ALL of mdp_foundations/ applies verbatim — VI, PI, LP,
   error bounds — with (r_omega, p_omega) as the model;
3. the price of the reduction: optimality is now over policies THAT
   COMMIT — mu chooses omega and must run it to termination. The optimal
   option-policy is generally WORSE than the optimal primitive policy
   unless the option set is lucky or termination is interruptible:
```

**Interruption.** Allowing early termination whenever continuing the
current option is dominated —
`q_mu(s, omega-continue) < v_mu(s)` — never hurts:

```math
v_{\mu'}\ \ge\ v_\mu
\qquad\text{(interruption improvement, SPS 1999 Thm 2)},
```

proved by the policy-improvement telescope of `mdp_foundations/05`
applied at interruption states. Interruption is `max(commit, reconsider)`
— the option framework's version of the improvement theorem, and the
formal reason "hierarchies with escape hatches" dominate rigid ones.

## What Remains Open

The framework is closed; *which options* is not — everything downstream
(03, 04) is the attempt to learn `(pi_omega, beta_omega)` rather than
hand them over, and the degeneracies of that attempt are the family's
honest frontier.
