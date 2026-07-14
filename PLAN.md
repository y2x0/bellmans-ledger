# PLAN: The Deep Expansion

This document governs all future notebooks in this repository. It is a
contract, not a wishlist: a family enters this plan only with a per-file
specification of the results to be proved, and a file ships only if it meets
the depth contract below.

## The Depth Contract

Every file must contain:

```text
1. at least one theorem proved in full, OR one counterexample worked with
   explicit numbers (states, transition matrices, features) — never gestured;
2. an explicit statement of where each hypothesis is used in the proof
   (the "load-bearing" audit);
3. its position in the (operator T, metric d, function class F) coordinate
   system of unifying_view/01;
4. a "what remains open" section naming the honest frontier.
```

Banned: survey files, algorithm zoos, files whose math could be transcribed
from a blog post, history-only files, files that state results without either
proof or counterexample.

## Dependency Graph

```text
existing repo ──┬── A. classical completions (average reward, SSP, LQR)
                │
                ├── B. statistical theory ── B1 concentration toolkit
                │      (how many samples)    B2 finite-time TD/Q
                │                            B3 regret & exploration   ── C1
                │
                ├── C. approximation theory ── C1 linear MDPs / completeness
                │      (when FA provably works) C2 offline RL & pessimism
                │
                ├── D. the policy as object ── D1 regularized MDPs & duality
                │                              D2 global convergence of PG
                │                              D3 mathematics of RLHF
                │
                ├── E. other agents & hidden state ── E1 stochastic games
                │                                     E2 POMDPs
                │
                └── F. model-based theory
```

B1 is a prerequisite for B2, B3, C1, C2. D1 is a prerequisite for D2, D3.
A, E, F are independent of B–D and of each other.

Recommended execution order: **B1 → B2 → B3 → C1 → C2 → D1 → D2 → D3 → A → E → F**
(statistics first: it retrofits rigor onto everything already written; the
existing repo repeatedly says "rates came later" — B2/B3 are those rates).

---

# Phase A — Classical Completions

## A1. `average_reward/` — 6 files

Central question: *what replaces the discount factor's contraction when
there is no discount?* (The existing repo leans on `gamma < 1` everywhere;
this family is RL with that crutch removed.)

```text
01_chain_structure_and_cesaro_limits.md
   PROVE: for finite Markov chains the Cesàro limit P* = lim (1/T) sum P^t
   always exists (via periodicity decomposition); P*P = PP* = P*P* = P*.
   Unichain vs multichain taxonomy with a 3-state multichain example where
   the "optimal gain" is state-dependent.

02_gain_bias_and_the_poisson_equation.md
   Define gain rho(pi) = P*_pi r_pi, bias h. PROVE existence/uniqueness
   (up to null-space of I - P*) of solutions to the Poisson equation
   rho + h = r_pi + P_pi h. Interpretation: h = transient excess reward.

03_laurent_series_of_the_resolvent.md
   PROVE the Laurent expansion (Puterman 8.2):
   v_gamma = rho/(1-gamma) + h + O(1-gamma)  via the Drazin/deviation
   matrix H = (I - P + P*)^{-1}(I - P*). This is the exact bridge between
   the discounted repo and this family: discounted values blow up along
   rho with h as the finite shadow.

04_blackwell_optimality.md
   PROVE: exists gamma_bar < 1 such that one policy is optimal for ALL
   gamma in (gamma_bar, 1) (finitely many policies + Laurent ordering
   argument). The n-discount optimality hierarchy (gain-optimal < bias-
   optimal < ... < Blackwell). Worked example where gain-optimal policies
   differ in bias.

05_span_seminorm_and_relative_vi.md
   The span seminorm ||v||_sp = max v - min v. PROVE: T is a span
   contraction with modulus = one minus a Doeblin/ergodicity coefficient
   (NOT gamma) under aperiodic unichain; relative value iteration
   converges. Show span-contraction FAILS for periodic chains (2-cycle
   example) and the aperiodicity transformation that repairs it.

06_average_reward_learning.md
   Differential TD / R-learning as SA on the Poisson equation; why the
   gain estimate must run on a slower timescale (two-timescale, cites
   stochastic_approximation/01). PROVE convergence of differential TD(0)
   tabular under unichain via the ODE method with the span Lyapunov
   function. Open: average-reward function approximation is genuinely
   less settled than discounted.
```

Sources: Puterman ch. 8–10; Bertsekas DP II ch. 4; Wan–Naik–Sutton 2021.

## A2. `total_reward_ssp/` — 4 files

Central question: *what replaces contraction when `gamma = 1` but episodes
end?* (Every Atari/episodic result in the repo implicitly lives here.)

```text
01_proper_policies_and_weighted_norms.md
   Stochastic shortest path setup. PROVE: if all policies are proper
   (reach the terminal state w.p. 1), T is a contraction in a WEIGHTED
   sup-norm ||v||_w = max |v(s)|/w(s) for a suitable w built from hitting
   times (Bertsekas–Tsitsiklis 1991). Explicit w for a 3-state chain.

02_improper_policies_and_semicontraction.md
   The mixed case: proper optimal policy exists but improper policies
   present. PROVE Bellman equation still has unique solution among
   functions where improper policies have -inf value; VI converges from
   above. Counterexample where an improper policy has finite value and
   uniqueness fails.

03_positive_negative_dp.md
   Blackwell's positive/negative dynamic programming: monotone (not
   contractive) convergence. PROVE: in negative DP (all rewards <= 0),
   VI from v_0 = 0 converges to v*; in positive DP greedy w.r.t. v* can
   FAIL to be optimal — construct the classic counterexample.

04_episodic_rl_as_ssp.md
   Time-limit truncation vs true termination formalized (retro-fits
   value_based_deep_rl/04 §6 with the actual theory); horizon-dependent
   value functions; why gamma < 1 on episodic tasks is variance reduction
   with a quantified bias (cite policy_gradient/05).
```

Sources: Bertsekas–Tsitsiklis "An Analysis of Stochastic Shortest Path
Problems" 1991; Bertsekas DP II ch. 3.

## A3. `lqr_and_continuous_control/` — 5 files

Central question: *what does the one exactly-solvable continuous MDP teach
about everything else?*

```text
01_lqr_riccati_from_bellman.md
   x' = Ax + Bu + w, cost x'Qx + u'Ru. PROVE by induction that finite-
   horizon value functions are exactly quadratic, Bellman backup = Riccati
   recursion; infinite-horizon: existence of stabilizing solution under
   (A,B) stabilizable, K* = -(R + B'PB)^{-1}B'PA. Certainty equivalence:
   noise w shifts value by a constant, policy unchanged — PROVE and mark
   as the exceptional case (false in general MDPs).

02_policy_gradient_on_lqr.md
   J(K) over stabilizing K is NONCONVEX (exhibit the disconnected-
   sublevel-set example) yet gradient dominated: PROVE the key lemma of
   Fazel et al. 2018 (J(K) - J(K*) <= (norm constants) * ||grad J(K)||^2)
   via the value-difference/advantage identity — which is EXACTLY the
   performance difference lemma of policy_gradient/03 in quadratic
   clothing. Consequence: linear convergence of exact PG. The cleanest
   known instance of "PG works despite nonconvexity."

03_hjb_and_viscosity.md
   Continuous time: derive HJB rho v = max_u [r + grad v . f] from the
   dynamic programming principle over dt. Why classical solutions fail
   (kinks at switching surfaces — 1D example worked) and the viscosity
   solution definition; statement (not proof) of uniqueness. Discretization
   back to the discounted Bellman equation with gamma = e^{-rho dt}.

04_linearly_solvable_control.md
   Todorov's KL-cost MDPs: with control cost KL(pi || p_passive), the
   exponentiated Bellman equation is LINEAR: z = exp(r) P z, desirability
   z = e^{v}. PROVE. Optimal policy proportional to p * z. This is the
   ancestor of the soft/entropy-regularized operators in D1 and the
   path-integral policy family.

05_what_lqr_predicts_and_misses.md
   The LQR-as-testbed argument: which deep-RL phenomena LQR reproduces
   (PG variance, model-based sample-efficiency gap — cite Recht's
   "Tour of RL") and which it structurally cannot (exploration hardness,
   nonstationary visitation). Explicit worked comparison of model-based
   least-squares vs policy gradient sample complexity on scalar LQR.
```

Sources: Anderson–Moore; Fazel–Ge–Kakade–Mesbahi 2018; Recht 2019;
Todorov 2006; Fleming–Soner (viscosity, reference only).

---

# Phase B — Statistical Theory

## B1. `concentration_toolkit/` — 5 files

Central question: *which five inequalities carry all finite-time RL?*
(Everything in B2–C2 cites this family; nothing here mentions MDPs.)

```text
01_chernoff_hoeffding_bernstein.md
   PROVE the Chernoff method; Hoeffding's lemma (with the full convexity
   argument); Hoeffding and Bernstein inequalities. The variance-vs-range
   distinction as the recurring theme: every "improved" RL bound in B2/B3
   is a Hoeffding->Bernstein swap somewhere.

02_martingale_concentration.md
   PROVE Azuma–Hoeffding; Freedman's inequality (stated with proof of the
   supermartingale core). Why RL needs these and not i.i.d. bounds: data
   generated by adaptive policies is a filtration, not a sample.

03_self_normalized_bounds.md
   PROVE the method-of-mixtures self-normalized bound for vector-valued
   martingales (Abbasi-Yadkori–Pál–Szepesvári Thm 1) — the engine of
   every linear bandit/linear MDP result in B3/C1. Corollary: confidence
   ellipsoids for ridge regression under adaptive design.

04_uniform_convergence_and_covering.md
   Covering numbers, the union-bound-over-an-epsilon-net technique;
   PROVE uniform Hoeffding over a finite class and over an L_inf ball of
   linear functions. Where RL needs uniformity: the value function being
   bounded is itself data-dependent (optimism iterates over a class).

05_lower_bound_technique.md
   PROVE Le Cam's two-point method and the Bretagnolle–Huber inequality;
   KL divergence decomposition for adaptive experiments (the "divergence
   decomposition" of bandit lower bounds). This file is the hammer for
   every Omega(.) in B2/B3/C2.
```

Sources: Boucheron–Lugosi–Massart; Lattimore–Szepesvári ch. 5, 20;
Wainwright HD Statistics ch. 2.

## B2. `finite_time_td_q/` — 6 files

Central question: *the existing repo proves asymptotic convergence
(Watkins, TvR); what are the actual rates, and are they tight?*

```text
01_linear_sa_lyapunov.md
   Finite-time analysis of contractive linear SA (Bhandari–Russo–Singal /
   Srikant–Ying): PROVE E||theta_t - theta*||^2 = O(1/t) with decaying
   steps and O(alpha) bias with constant steps, via the drift inequality
   with the Hurwitz Lyapunov matrix. This retrofits
   stochastic_approximation/01 with rates.

02_td0_finite_time.md
   Apply 01 to TD(0) with linear FA under i.i.d. and Markov sampling
   (the mixing-time correction: PROVE the coupling/uniform-mixing lemma
   that converts Markov noise to inflated variance). Explicit constants
   in terms of the feature covariance's minimum eigenvalue and (1-gamma).

03_q_learning_upper_bounds.md
   Synchronous Q-learning with a generative model: PROVE the
   O(|S||A|/( (1-gamma)^4 eps^2 )) bound via Hoeffding + union bound +
   error recursion through the contraction; identify exactly where each
   (1-gamma) factor enters (this is the "audit" the depth contract wants).

04_variance_reduction_and_minimax.md
   Why 03 is loose: the law-of-total-variance argument; Bernstein
   (B1/01) + variance-of-the-value recursion gives (1-gamma)^{-3};
   statement + proof sketch of Azar et al. model-based minimax optimality
   and Wainwright's variance-reduced Q-learning matching it model-free.

05_lower_bound_generative_model.md
   PROVE the Omega(|S||A|/((1-gamma)^3 eps^2)) lower bound: explicit
   hard MDP family (two-armed states with gamma-dependent gaps), reduction
   to testing biased coins, apply B1/05. The matching of 04 and 05 is the
   one completely-settled chapter of RL sample complexity — worth having
   end-to-end in one family.

06_polyak_ruppert_and_practice.md
   PROVE asymptotic normality / optimal covariance of averaged SA iterates
   (statement + the variance calculation for linear SA); why deep RL's
   Adam-plus-constant-steps sits outside this theory (revisits
   stochastic_approximation/01's "hover, don't converge" honestly, with
   the O(sqrt(alpha)) stationary-distribution radius made precise).
```

Sources: Bhandari–Russo–Singal 2018; Srikant–Ying 2019; Even-Dar–Mansour
2003; Azar–Munos–Kappen 2013; Wainwright 2019 (two papers); AJKS ch. 2.

## B3. `regret_and_exploration/` — 7 files

Central question: *the repo calls epsilon-greedy "primitive"
(value_based_deep_rl/04) and proves UCB1 for bandits; what is the actual
theory of exploration in MDPs?*

```text
01_optimism_principle.md
   The generic optimism lemma: if V_k >= V* pointwise (valid optimism)
   then regret <= sum of per-episode widths — PROVE, exhibiting regret
   analysis as a telescoping of confidence widths along visited
   trajectories. The bandit UCB1 proof (search_and_planning/01) as the
   degenerate H=1 case.

02_ucbvi_hoeffding.md
   Full regret proof of UCBVI with Hoeffding bonuses: O(H^2 sqrt(SAT))
   — the error-propagation recursion, the "good event" construction
   (B1/02), the visit-count pigeonhole. This is the canonical complete
   proof of the family; do it end to end.

03_bernstein_and_minimax.md
   The Bernstein refinement: bonus from empirical variance + law of total
   variance across a trajectory (sum of per-step variances <= H^2, not
   H^3) -> O(sqrt(H^3 SAT)) ~ minimax. PROVE the total-variance lemma;
   sketch the rest against 02's skeleton.

04_lower_bound_construction.md
   PROVE the Omega(sqrt(H^3 SAT)) lower bound: the JAO/"needle in a
   haystack" MDP — S states arranged so that only one (s,a) is
   epsilon-better, reduction to SA independent bandit instances, B1/05.
   Also: exponential lower bound for undirected exploration (why
   epsilon-greedy needs Omega(A^H) on chain MDPs — worked, this is the
   theorem behind value_based_deep_rl/04 failure 5).

05_thompson_and_psrl.md
   Posterior sampling for RL: PROVE the Bayesian-regret bound via the
   posterior-matching trick (optimistic-in-expectation without explicit
   bonuses); the frequentist gap (statement of known partial results).
   Why PSRL beats OFU empirically: bonus-free, no union bound looseness.

06_information_directed.md
   Information-directed sampling: the information ratio, PROVE the
   generic IDS regret bound (Russo–Van Roy) for bandits; the RL extension
   (statement); when optimism is provably suboptimal (the
   "sparse-linear-bandit" example where IDS wins).

07_exploration_with_function_approximation.md
   Eluder dimension defined + PROVE the eluder regret lemma for
   generalized linear bandits; statement of LSVI-UCB (bridge to C1);
   the honest state: no practical deep-RL algorithm inherits these
   guarantees; RND/pseudocounts as heuristic surrogates for the widths
   of 01.
```

Sources: Jaksch–Ortner–Auer 2010; Azar et al. 2017; Osband–Van Roy;
Russo–Van Roy 2014/2018; Lattimore–Szepesvári ch. 38; AJKS ch. 7–8.

---

# Phase C — Function Approximation Theory

## C1. `linear_mdps_and_completeness/` — 6 files

Central question: *the deadly triad (stochastic_approximation/06) says FA
can fail; what structural assumptions make it provably succeed — and are
they believable?*

```text
01_linear_mdp_definition.md
   Linear MDP: P(s'|s,a) = <phi(s,a), mu(s')>, r = <phi, theta>. PROVE
   the two closure properties that make everything work: Q_pi is linear
   in phi for EVERY pi, and Bellman backups of linear functions are
   linear. Why this is a strong assumption (the transition kernel has
   rank d — exhibit a 2-state nonlinear-MDP counterexample).

02_lsvi_ucb.md
   LSVI-UCB: ridge regression per step + self-normalized ellipsoid bonus
   (B1/03). PROVE the single-step concentration and the elliptical
   potential lemma; assemble the O(sqrt(d^3 H^4 T)) regret sketch. The
   point: optimism + completeness replaces the table, no |S| anywhere.

03_realizability_vs_completeness.md
   The central conceptual file. Definitions: realizability (Q* in F) vs
   Bellman completeness (T F ⊆ F). PROVE completeness is NOT monotone in
   F (enlarging the class can break it — worked example); exhibit the
   known counterexample pattern where realizability alone + generative
   model still needs exp(d) samples (Weisz–Amortila–Szepesvári statement,
   construction sketched). Moral: "my network can represent Q*" is not
   the right condition; "my class is closed under backup" is.

04_bellman_rank_bilinear.md
   Bellman rank and bilinear classes: PROVE the key lemma (average
   Bellman error of a hypothesis reveals itself in d rounds of
   eliminations — the volumetric/elliptic argument), giving
   polynomial-sample RL for low-Bellman-rank problems. Position block
   MDPs / reactive POMDPs in the hierarchy.

05_offline_linear_lower_bound.md
   Wang–Foster–Kakade: offline RL with REALIZABLE linear Q* and good
   feature coverage still requires exponential samples — construction
   worked in detail (the geometry: errors amplify through H steps of
   backup because coverage is in the wrong norm). This is Baird's
   counterexample's modern, quantitative descendant; cross-reference
   stochastic_approximation/06.

06_what_deep_networks_change.md
   Honest synthesis: NTK-regime results (statement); why the linear
   theory's d is not a network's parameter count; completeness as the
   hidden reason target networks help (a frozen target makes the
   regression problem well-posed within the class — ties to
   value_based_deep_rl/02); open problems list.
```

Sources: Jin–Yang–Wang–Jordan 2020; Jiang et al. 2017; Du et al. 2021;
Wang–Foster–Kakade 2020; Weisz et al. 2021; AJKS ch. 8–9.

## C2. `offline_rl_and_ope/` — 6 files

Central question: *what can be certified from a fixed dataset, with no
interaction?* (This is the regime of RLHF reward models and most applied
RL; the repo currently touches it only via replay buffers.)

```text
01_ope_importance_sampling_revisited.md
   Trajectory IS from stochastic_approximation/02, now with finite-sample
   analysis: PROVE the exponential-in-horizon variance lower bound for
   trajectory IS (explicit two-action chain).

02_doubly_robust.md
   The DR estimator: PROVE double robustness (unbiased if EITHER the
   model or the weights are correct) and derive its variance; statement
   of the semiparametric efficiency bound (Jiang–Li) — DR achieves it.
   The recursive DR-through-time estimator worked on a 2-step example.

03_marginalized_is_curse_of_horizon.md
   Replace trajectory ratios with state-density ratios d_pi/d_b: PROVE
   the identity E_b[(d_pi/d_b)(s) (pi/b)(a|s) r] = J(pi) and the variance
   collapse from exp(H) to poly(H). The DICE fixed-point equation for
   the ratio (bridge to D1's Fenchel duality).

04_pessimism_lcb.md
   The mirror of optimism: PROVE the LCB-VI suboptimality bound under
   SINGLE-POLICY concentrability (only the target policy needs coverage,
   not all policies) — the key modern result (Rashidinejad et al.).
   Why naive fitted-Q needs ALL-policy concentrability: worked failure.

05_cql_as_operator.md
   CQL's penalized objective: derive the closed form of its implicit
   pessimistic operator (the logsumexp-vs-data-mean penalty as an
   underestimation of OOD actions); connect to
   value_based_deep_rl/03's "insert pessimism at the argmax."

06_fitted_q_error_propagation.md
   The full Munos–Szepesvári concentrability analysis promised in
   value_based_deep_rl/02: PROVE the error-propagation theorem with the
   discounted-future concentrability coefficients, and exhibit an MDP
   where the coefficient is infinite and fitted-Q diverges while the
   same data supports MC regression fine.
```

Sources: Jiang–Li 2016; Thomas–Brunskill 2016; Liu et al. 2018;
Nachum et al. (DICE); Rashidinejad et al. 2021; Kumar et al. 2020;
Munos–Szepesvári 2008.

---

# Phase D — The Policy As The Object

## D1. `regularized_mdps_and_duality/` — 6 files

Central question: *what is the exact mathematics of adding an entropy or
KL term to the objective — and why does every modern method do it?*

```text
01_soft_bellman_operator.md
   Entropy-regularized control: PROVE the Legendre–Fenchel computation
   max_pi { <pi, q> + tau H(pi) } = tau log sum_a exp(q/tau), maximizer
   pi ∝ exp(q/tau); the soft Bellman operator is a gamma-contraction
   (logsumexp is nonexpansive — prove via its gradient being a
   distribution). Bias bound: ||v*_tau - v*|| <= tau log|A| / (1-gamma).

02_control_as_inference.md
   The variational derivation: optimality variables O_t with
   p(O|tau) ∝ exp(sum r), PROVE that maximizing the ELBO over policies
   yields exactly the entropy-regularized objective, and that the exact
   posterior backward messages are the soft Bellman recursion. Where the
   naive derivation gets risk-seeking behavior wrong (the max-ent
   "optimism under stochastic dynamics" pathology) — worked 2-state
   example.

03_sac_derivation.md
   Soft policy iteration: PROVE soft policy improvement (the KL-projection
   step improves the soft objective — the regularized analogue of
   mdp_foundations/05) and convergence of exact soft PI. SAC as its
   function-approximation instance; temperature auto-tuning as dual
   ascent on the entropy constraint (derive the dual).

04_geist_scherrer_pietquin.md
   The general theory: regularizer Omega(pi) convex, PROVE the
   regularized Bellman operator's contraction and the propagation-of-
   errors theorem for regularized MPI; entropy and KL as special cases.
   This file is the umbrella over 01–03 and D2.

05_mirror_descent_view.md
   PROVE: KL-regularized policy iteration IS mirror descent on the
   simplex with negative-entropy potential, per state; TRPO/PPO as
   inexact MD steps (policy_gradient/04's constraint = the proximal
   term). The three-point lemma of MD and what it buys: telescoping
   regret without contraction.

06_fenchel_rockafellar_dice.md
   The LP of mdp_foundations/06 revisited through Fenchel–Rockafellar
   duality: PROVE the derivation of the off-policy stationary-ratio
   estimators (DualDICE-style) as the dual of a regularized LP; the
   occupancy polytope (mdp_foundations/06) as the primal feasible set.
   Closes the loop: the repo's LP file was the seed; this is the tree.
```

Sources: Ziebart 2010; Levine 2018 (tutorial); Haarnoja et al. 2018;
Geist–Scherrer–Pietquin 2019; Neu–Jonsson–Gómez 2017; Nachum–Dai 2020.

## D2. `global_convergence_of_pg/` — 5 files

Central question: *policy_gradient/07 lists nonconvexity as "inherent" —
but softmax PG provably reaches global optima. What is the actual
geometry?*

```text
01_softmax_gradient_domination.md
   Tabular softmax policies: PROVE the non-uniform Łojasiewicz inequality
   (Mei et al.): ||grad J||_2 >= (min_s pi(a*(s)|s)/ (sqrt(S) * dist
   factors)) * (J* - J). Consequence: exact PG converges globally at
   O(1/t) — but the constant contains min pi(a*), which can be
   exponentially small: derive the exp(-H)-plateau example, reconciling
   with policy_gradient/07 §6.

02_natural_pg_linear_convergence.md
   PROVE: entropy-regularized NPG converges LINEARLY (Cen et al.) — the
   proof is a soft policy iteration contraction in disguise (uses D1/01,
   D1/05); unregularized NPG gets O(1/t) dimension-free (Agarwal et al.
   Thm: the MD regret telescoping, no Łojasiewicz needed).

03_compatible_function_approximation.md
   PROVE Sutton et al. 1999 Part 2: if the critic is the L2(d_pi)
   projection of q_pi onto span{grad log pi}, the resulting PG is EXACT
   despite the approximation. Why practice ignores it (the compatible
   features are policy-dependent) and what NPG preserves of it
   (compatible critic weights = natural gradient — prove the identity).

04_npg_with_approximation.md
   The Agarwal–Kakade–Jiang–Sun transfer-error theorem: PROVE the bound
   J* - J(pi_T) <= (MD regret term) + (distribution-mismatch coefficient)
   * sqrt(transfer error). The mismatch coefficient ||d*/d_pi||_inf as
   THE quantity policy optimization cannot escape — exploration returns
   through the back door (ties B3 to this family).

05_geometry_synthesis.md
   The occupancy-polytope picture assembled: J linear in lambda
   (mdp_foundations/06), theta -> lambda smooth but non-surjective-
   gradient; plateaus = near-deterministic policies far from optimum;
   entropy regularization as interior-point smoothing (D1) — PROVE the
   strong-convexity-along-the-central-path lemma for the regularized LP.
```

Sources: Agarwal–Kakade–Lee–Mahajan 2021; Mei et al. 2020; Cen et al.
2021; Sutton–McAllester–Singh–Mansour 1999; Kakade 2001.

## D3. `rlhf_mathematics/` — 6 files

Central question: *what exactly is being optimized when a language model is
tuned from preferences — and what do the closed forms say about reward
hacking?* (The repo notes PPO "escaped the field" at policy_gradient/06;
this family is that escape, made rigorous.)

```text
01_bradley_terry_and_identifiability.md
   Preference model P(y1 > y2) = sigma(r(y1) - r(y2)). PROVE: MLE
   consistency conditions; r identifiable only up to per-prompt additive
   shift; the induced equivalence class and why it suffices for policy
   optimization (shift-invariance of the KL-regularized argmax). What
   preferences CANNOT identify: any monotone transform beyond
   Bradley–Terry's assumption — state the honest epistemics.

02_kl_regularized_optimum.md
   PROVE the closed form: max_pi E_pi[r] - beta KL(pi || pi_ref) is
   solved by pi*(y|x) ∝ pi_ref(y|x) exp(r(x,y)/beta), optimal value
   = beta log E_ref[exp(r/beta)] (Donsker–Varadhan, cf. D1/02). Beta as
   inverse temperature interpolating pi_ref (beta -> inf) and reward
   argmax (beta -> 0). Everything else in this family is a corollary
   of this file.

03_dpo_derivation.md
   Invert 02: r(x,y) = beta log (pi*(y|x)/pi_ref(y|x)) + beta log Z(x);
   substitute into Bradley–Terry — the partition function CANCELS in
   pairwise differences: PROVE the DPO loss equals MLE of 01 under the
   reparameterization. What is lost vs PPO-RLHF: off-preference-
   distribution generalization of the implicit reward; the gradient
   analysis (DPO's weighting term) derived explicitly.

04_best_of_n.md
   PROVE: KL(BoN || pi_ref) <= log n - (n-1)/n (exact for continuous
   rewards); BoN's reward-vs-KL frontier is near-optimal among all
   policies at matched KL (statement + proof sketch of the
   Yang et al. / Beirami et al. results). BoN as the zeroth-order
   competitor every RLHF run should be compared against.

05_overoptimization_goodhart.md
   The proxy-vs-gold reward gap: Gao–Schulman–Hilton scaling laws
   (statement, and the regression-to-the-mean derivation of WHY
   optimizing a noisy proxy overshoots: order-statistics argument
   from value_based_deep_rl/03 transplanted — max-bias IS Goodhart).
   KL budget as the control variable; derive the optimal-stopping-in-KL
   heuristic under a quadratic proxy-error model.

06_token_mdp_or_bandit.md
   Is RLHF RL at all? The token-level MDP (deterministic transitions,
   terminal reward) vs contextual-bandit-with-long-actions framing:
   PROVE that with deterministic transitions and terminal reward, GAE
   with lambda=1 reduces PPO's advantage to (reward - baseline) — the
   sequence-level view; what GRPO's group-normalized estimator is
   estimating (per-prompt baseline = leave-one-out control variate,
   cf. policy_gradient/02). Where genuine multi-step structure returns:
   multi-turn tool use, reasoning with intermediate verification.
```

Sources: Christiano et al. 2017; Rafailov et al. 2023; Gao et al. 2023;
Beirami et al. 2024; Ahmadian et al. 2024 (RLOO); Shao et al. 2024 (GRPO).

---

# Phase E — Other Agents And Hidden State

## E1. `stochastic_games/` — 5 files

Central question: *the minimax Bellman operator predates Bellman's MDP
paper (Shapley 1953); what survives of the contraction theory when the
environment contains an adversary — and what provably breaks in self-play?*

```text
01_shapley_operator.md
   Zero-sum stochastic games: PROVE the Shapley operator
   (Tv)(s) = val[ matrix game of r + gamma P v ] is a gamma-contraction
   (val is nonexpansive — prove via the minimax theorem + the scalar
   max-lemma of mdp_foundations/03 applied twice); existence of the value
   and of stationary optimal strategies. Historical note: this is 1953 —
   the contraction argument existed before the MDP.

02_failure_of_greedy_dynamics.md
   Why "each player best-responds" is not an algorithm: Shapley's
   fictitious-play cycle (the 3x3 game, worked); policy iteration for
   games needs care (Hoffman–Karp vs naive PI counterexample — van der
   Wal). The general lesson: improvement theorems (mdp_foundations/05)
   are single-agent property, not a game property.

03_regret_matching_cfr.md
   PROVE Blackwell approachability -> regret matching's regret bound;
   external regret -> coarse correlated equilibrium (proof); CFR:
   counterfactual regret decomposition theorem proved (regret bounded by
   sum of per-infoset counterfactual regrets); why average (not last)
   iterates converge.

04_self_play_as_gpi.md
   AlphaGo/AlphaZero self-play (search_and_planning/03) formalized:
   PROVE that in zero-sum games, best-response dynamics against a
   uniform mixture over past selves (fictitious self-play) converges to
   Nash; league training / PSRO as double oracle — PROVE double oracle's
   finite convergence.

05_independent_learning_pathologies.md
   Two independent Q-learners in a 2x2 matrix game: work the
   nonconvergence/cycling dynamics explicitly (the replicator-dynamics
   reduction of infinitesimal-step Q-learning, its Hopf-type orbits in
   matching pennies). The Markov assumption audit: each agent makes the
   other's environment nonstationary — 00_problem_setup's one assumption,
   violated structurally.
```

Sources: Shapley 1953; Cesa-Bianchi–Lugosi; Zinkevich et al. 2007;
Lanctot et al. 2017; Mertikopoulos et al.

## E2. `pomdps_and_information_state/` — 5 files

Central question: *the whole repo rests on observing the Markov state
(00_problem_setup); what is the exact price of losing it?*

```text
01_belief_mdp.md
   PROVE the belief update (Bayes filter) and the sufficiency theorem:
   the belief b_t is a sufficient statistic — the belief process is
   Markov and value-equivalent to the POMDP. The price: a simplex-valued
   continuous state space.

02_pwlc_alpha_vectors.md
   PROVE by induction: finite-horizon optimal value over beliefs is
   piecewise-linear convex, V(b) = max_alpha <alpha, b>; the exact DP
   backup on alpha-vector sets, with the worked 2-state tiger problem
   (numbers, all backups for horizon 2). Why exact solving explodes:
   vector count growth |Gamma_{t+1}| = |A| |Gamma_t|^{|O|}; point-based
   methods as the practical projection.

03_hardness.md
   Statements with proof IDEAS made concrete: finite-horizon POMDP
   optimality is PSPACE-complete (Papadimitriou–Tsitsiklis — the QBF
   encoding sketched); infinite-horizon undecidable (Madani et al.).
   The lesson for deep RL: no architecture escapes this in the worst
   case; tractability claims are always distributional claims.

04_predictive_state_representations.md
   PSRs: PROVE that the system-dynamics matrix's rank bounds the
   dimension of a linear sufficient statistic of history, and that PSRs
   can be strictly more compact than belief states (the worked float/gate
   example); spectral learning of PSRs (statement + the SVD identity).

05_deep_rl_approximate_information_states.md
   Frame stacking (DQN), recurrence, and transformers-as-history-
   compressors as approximate information states: the AIS framework
   (Subramanian et al.) — PROVE the value-loss bound in terms of the
   AIS's prediction errors. Retro-fits value_based_deep_rl/04 failure 3
   with its missing theory.
```

Sources: Åström 1965; Smallwood–Sondik 1973; Papadimitriou–Tsitsiklis
1987; Littman–Sutton–Singh 2002; Subramanian et al. 2022.

---

# Phase F — Model-Based Theory

## F1. `model_based/` — 5 files

Central question: *search_and_planning assumed a perfect simulator; what
is the calculus of planning with a LEARNED, wrong model?*

```text
01_simulation_lemma.md
   PROVE the simulation lemma in both norms: value difference under two
   models <= gamma/(1-gamma)^2 * sup TV(P, P-hat) * Rmax (+ reward-error
   term), via the resolvent-difference identity
   (I-gPhat)^{-1} - (I-gP)^{-1} = gamma (I-gPhat)^{-1}(Phat-P)(I-gP)^{-1}
   — one algebraic identity carrying the whole family.

02_certainty_equivalence.md
   Generative model, empirical transition matrix, plan exactly in it:
   PROVE the (1-gamma)^{-3} sample complexity via the variance-Bernstein
   argument (this is where B2/04's minimax rate is ACHIEVED — the
   model-based side of that equivalence). The surprise, stated exactly:
   naive plug-in is minimax optimal; sophistication buys constants only.

03_optimism_through_models.md
   R-MAX and UCRL2: PROVE R-MAX's polynomial PAC bound via the
   explore-or-exploit dichotomy (either the known-state MDP is accurate
   — simulation lemma — or unknown states are reached fast); UCRL2's
   confidence-polytope over models and its regret (sketch against
   B3/02's skeleton).

04_value_equivalence_muzero.md
   The value-equivalence principle: models need only agree with the
   true MDP on the functionals the planner applies. PROVE the basic VE
   proposition (Grimm et al.): the set of value-equivalent models is an
   affine subspace, planning losses of VE models coincide. MuZero as
   learned-VE + MCTS (closes the search_and_planning/03 arc); what VE
   gives up: transfer to new rewards.

05_compounding_error_dyna.md
   k-step model rollouts: PROVE the branched-rollout return bound
   (Janner et al.) and its pessimism (the bound is loose exactly when
   the model generalizes); Dyna as mixing real/model backups —
   convergence condition as a contraction average. The honest empirical
   summary: model error is heteroscedastic and adversarially selected by
   the planner (max-bias again — value_based_deep_rl/03's pattern, third
   appearance).
```

Sources: Kearns–Singh 2002; Brafman–Tennenholtz 2002; Jaksch et al. 2010;
Agarwal–Kakade–Yang 2020; Grimm et al. 2020; Janner et al. 2019.

---

# Scope Summary

```text
phase   families  files   the one-line payoff
A       3         15      RL without the discount crutch; the solvable case
B       3         18      every asymptotic claim in the repo gets its rate,
                          and exploration gets its actual theory
C       2         12      exactly when function approximation is sound
D       3         17      regularization/duality; PG global convergence; RLHF
E       2         10      adversaries and hidden state — the two broken
                          assumptions, priced
F       1         5       planning with wrong models
total   14        77      (existing: 43 files -> target: ~120)
```

## Retrofit List (edits to existing files as phases land)

```text
stochastic_approximation/01  gains rates            <- B2/01
stochastic_approximation/06  gains WFK lower bound  <- C1/05
value_based_deep_rl/02       gains full Munos–Szepesvári proof <- C2/06
value_based_deep_rl/04       failure 3 gains AIS theory <- E2/05,
                             failure 5 gains the exp lower bound <- B3/04
policy_gradient/07           §6 revised: plateaus quantified <- D2/01
search_and_planning/03       aftermath section links E1/04, F1/04
unifying_view/01             table grows a row per new method
unifying_view/02             theorem index grows ~40 entries
```
