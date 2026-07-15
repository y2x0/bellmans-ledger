# Every Method As (Operator, Metric, Function Class)

## The Claim

Fix the three coordinates from the repository README:

```math
\mathcal{T}\ \text{(which Bellman-type operator)},
\qquad
d\ \text{(which metric certifies contraction)},
\qquad
\mathcal{F}\ \text{(which class holds the fixed point's approximation)} .
```

Every method studied here is a cell; every convergence theorem is a
statement that the three coordinates are compatible; every failure mode is a
mismatch between two of them.

## The Table

```text
method            operator T                metric d              class F              guarantee
----------------  ------------------------  --------------------  -------------------  ------------------------
value iteration   T (optimality)            sup-norm              table                v* exactly, geometric
policy iteration  T^pi solve + greedy       sup-norm              table                pi* in finite steps
LP formulation    constraints v >= T v      (polytope geometry)   table                v* exactly (duality)
MC prediction     none (no bootstrap)       L2 (regression)       any                  v_pi, unbiased, high var
TD(0), tabular    T^pi sampled              sup-norm + RM         table                v_pi a.s.
TD(lambda)        T_lambda^pi sampled       sup-norm + RM         table                v_pi a.s., better modulus
linear TD         Pi_mu T^pi sampled        L2(mu), mu ON-POLICY  span(Phi)            TvR fixed point + bound
Q-learning        T sampled, async          sup-norm + RM         table                q* a.s. (Watkins)
DQN               Pi_replay T sampled       NONE CERTIFIED        convnet              none; empirical stability
Double DQN        T with decoupled argmax   none certified        convnet              bias sign flipped
REINFORCE         none: ascent on J         (nonconvex ascent)    any pi_theta         local, unbiased gradient
actor-critic      T^pi (critic) + ascent    two-timescale RM      V_w + pi_theta       local (timescale sep.)
TRPO              MM on L - C*KL            KL trust region       any pi_theta         monotonic improvement*
PPO               clipped surrogate         ratio interval        any pi_theta         none; inherits TRPO
                                                                                       heuristically
UCB1              none (stateless)          confidence radii      table of K arms      O(log n) regret
UCT / MCTS        T sampled on subtree      per-node UCB          visited tree         asymptotic optimality
AlphaGo           T on subtree + nets       PUCT bonus            tree + 4 convnets    none formal; 5-0
C51 (evaluation)  Phi T^pi                  Cramér (post hoc)     categorical atoms    contraction, O(dz) bias
C51 (control)     Phi T                     provably NOT W_p      categorical atoms    means only
QR-DQN            Pi_W1 T^pi                d_inf-bar / W1        quantile atoms       contraction (evaluation)
--- expansion families ---
UCBVI             T + Hoeffding bonus       sup-norm good event   table (episodic)     O(H^2 sqrt(SAT)) regret
UCBVI-Bernstein   T + variance bonus        Freedman + total var  table                sqrt(H^3 SAT) ~ minimax
LSVI-UCB          T + ellipsoid bonus       ||.||_{Lambda^{-1}}   span(phi), linear    sqrt(d^3 H^4 K), no S,A
PSRL              T of a SAMPLED model      posterior matching    table + prior        Bayes regret ~ minimax
IDS               none: info-ratio argmin   entropy budget        per-setting          sqrt(Gamma H(A*) T)
LCB-VI (offline)  T - bonus (pessimism)     same good event       table                single-policy C^pi
CQL               T - density-ratio pen.    (implicit)            deep                 value lower bound only
fitted-Q / FQI    Pi_F T                    L2(mu)+concentrability any F               (1-g)^{-2} sqrt(C) eps
DICE              occupancy-flow dual       Fenchel–Rockafellar   w, nu classes        saddle = J(pi), poly(H)
soft VI / SAC     T_tau (logsumexp)         sup-norm              any / deep           contraction; bias tau log|A|
NPG / mirror desc MD step on simplexes      per-state KL Bregman  softmax / any pi     O(1/K) global; linear w/ tau
DPO               none: MLE via bijection   logistic loss         pi = implicit r      exact under realizability
BoN               rank tilt n F^{n-1}       KL = log n - (n-1)/n  sampling only        near reward-KL frontier
diff. TD/Q        Poisson eq. sampled       span seminorm + RM    table                a.s. under unichain
SSP VI/Q          T (total reward)          weighted sup ||.||_w  table                properness => contraction
Riccati / LQR     T on quadratic cone       closed-loop stability quadratics (complete) exact; PG: grad-dominated
Shapley VI        val-backup (minimax)      sup-norm              table (zero-sum)     gamma-contraction (1953)
CFR               per-infoset regret match  quadratic potential   tree infosets        avg -> Nash, sqrt(T)
R-MAX / UCRL2     T of optimistic model     L1 model polytope     tabular model        PAC / D S sqrt(AT)
MuZero            T of value-equiv. model   Bellman-agreement     latent + MCTS        none formal; VE license
belief-MDP VI     T on Delta(S)             sup-norm (PWLC)       alpha-vectors        exact; PSPACE-complete
--- paper-driven additions ---
SMDP / options VI T over (r_w, p_w)         sup-norm, modulus     table over S x Omega E[gamma^tau]-contraction
                                            E[gamma^tau]
intra-option QL   coupled (q, U) backup     sup-norm + RM         table                a.s. (SPS 1999)
option-critic     PG thm on augmented chain (nonconvex ascent)    pi_w, beta_w         local; collapse modes
SR / SF TD        T^pi per feature column   L2(mu) / sup + RM     M or psi tables/lin  TvR per column
SF + GPI          envelope-greedy improve   sup-norm              linear in phi per w  q^pi >= max_i q^{pi_i}
                                                                                       - 2eps/(1-g)
BC                none: regression          L2(d_expert)          any                  J* + eps H^2 (tight)
DAgger            no-regret online fit      L2(d_learner)         any                  J* + u eps H
MaxEnt IRL        soft-VI inside MLE        exp-family logloss    linear r = w.phi     unique convex selection
GAIL              occupancy-matching saddle psi* divergence (JS)  rho_pi vs rho_E      = RL o IRL_psi (Prop 3.2)
Decision Transf.  none: conditional MLE     sequence logloss      transformer          behavior conditional only;
                                                                                       fails luck + stitching
MuZero            T of latent VE model      K-step (r,v,p)-agree  latent + PUCT        none formal; planning-
                                                                                       complete empirically
RLVR / GRPO (R1)  MW ascent, exact r        group-normalized adv  LLM policy           deletes proxy Goodhart;
                                                                                       frontier-of-pass@k signal

* under max-KL; practice substitutes mean-KL
```

## Failure Modes As Coordinate Mismatches

```text
deadly triad          d-mismatch: operator contracts in sup-norm /
                      L2(mu_stationary); updates are weighted by a different
                      mu. (stochastic_approximation/06)

overestimation        T-mismatch: the algorithm applies E-then-max where the
                      operator specifies max-then-E. (value_based_deep_rl/03)

PPO collapse / drift  d-mismatch: the certified region is a KL ball; the
                      enforced region is a per-sample ratio interval with
                      zero gradient outside, not a projection.
                      (policy_gradient/07)

distributional        T-mismatch at the argmax: greedy selection is
control               mean-continuous but distribution-discontinuous.
                      (distributional_rl/03)

C51's KL loss         d-mismatch, benign: trained in KL, contracts in
                      Cramér; the projection quietly reconciles them.
                      (distributional_rl/04-05)

exploration           neither T nor d: a violated *visitation* hypothesis
starvation            (Watkins cond. 3; UCB is the repaired version).
```

## Three Recurring Instruments

Across all families, the same three mathematical instruments do almost all
of the work:

```math
\textbf{1. Contraction + Banach:}\quad
\|\mathcal{T}u-\mathcal{T}v\|\le\gamma\|u-v\|
\ \Rightarrow\ \text{existence, uniqueness, geometric rate, a posteriori bounds.}
```

```math
\textbf{2. The telescope:}\quad
G-v(s_0)=\sum_t\gamma^t\delta_t
\quad\text{(lambda-returns, GAE, performance difference lemma — one identity, three families).}
```

```math
\textbf{3. The score integral:}\quad
\mathbb{E}_{a\sim\pi}[\nabla\log\pi(a\mid s)\,b(s)]=0
\quad\text{(policy gradients, baselines, likelihood-ratio IS — measure-change as differentiation).}
```

A reader who can reproduce these three from memory, plus the coupling
argument of `distributional_rl/02`, can re-derive the load-bearing
mathematics of every paper in `papers/`.

## Two More Instruments (from the expansion)

The expansion families added exactly two devices of the same rank:

```math
\textbf{4. The exponential supermartingale:}\quad
Z_t=\exp\Big(\lambda S_t-\tfrac{\lambda^2}{2}V_t\Big)\ \text{is a supermartingale}
```

(Freedman, the self-normalized bound, every confidence set and every
regret proof — `concentration_toolkit/02–03`), and

```math
\textbf{5. The Legendre transform of entropy:}\quad
\log\sum e^{q/\tau}\ \leftrightarrow\ \tau\,\mathrm{KL}
```

(soft Bellman, control-as-inference, mirror descent, the RLHF closed
form, linearly solvable control, Chernoff/lower-bound tilting — five
families, one identity: `regularized_mdps_and_duality/01`,
`rlhf_mathematics/02`, `concentration_toolkit/05`,
`lqr_and_continuous_control/04`).

And one conservation law, stated once (`global_convergence_of_pg/05`):
reparameterization moves difficulty between optimization geometry and
statistical coverage; only changing the data distribution reduces their
sum. Five instruments and a conservation law — that is the compressed
content of ~120 files.
