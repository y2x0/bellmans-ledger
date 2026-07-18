# Theorem Index

One line per notebook: the result it proves or the counterexample it
works. Statements cited from the literature without proof are marked
inside each file, not here. Companion index:
[attention-ledger/THEOREMS.md](https://github.com/y2x0/attention-ledger/blob/main/THEOREMS.md).

## mdp_foundations/

| File | What it proves |
|---|---|
| 01 | MDP formalism; sufficiency of stationary Markov policies |
| 02 | Return criteria; existence and measurability of value functions |
| 03 | Bellman operators: monotonicity and gamma-contraction, both proved |
| 04 | Banach fixed point theorem; a priori / a posteriori error bounds |
| 05 | Policy improvement theorem; PI as Newton's method |
| 06 | The LP formulation; occupancy measures; the dual LP |
| 07 | Bellman 1957: the original functional equation, annotated |

## stochastic_approximation/

| File | What it proves |
|---|---|
| 01 | Robbins–Monro; the ODE method (Borkar–Meyn); two-timescale iterates |
| 02 | Monte Carlo estimation; ordinary vs weighted importance sampling |
| 03 | TD(lambda): the offline forward/backward equivalence theorem |
| 04 | Tsitsiklis–Van Roy: TD(0) with linear FA converges (projected fixed point) |
| 05 | Watkins' theorem: Q-learning convergence via asynchronous SA |
| 06 | The deadly triad: Baird's counterexample, divergence mechanics worked |

## value_based_deep_rl/

| File | What it proves |
|---|---|
| 01 | The DQN loss; which instability each component addresses |
| 02 | Replay and target networks as distribution and correlation control |
| 03 | The max is a biased estimator (order statistics); Double Q's fix |
| 04 | Failure modes, each traced to a violated hypothesis |

## policy_gradient/

| File | What it proves |
|---|---|
| 01 | The policy gradient theorem, both proofs |
| 02 | Baselines and variance; why the value function is the right baseline |
| 03 | The performance difference lemma (Kakade–Langford) |
| 04 | The TRPO monotonic improvement bound |
| 05 | GAE: the bias–variance dial, derived |
| 06 | PPO's clipped surrogate as a pessimistic lower bound |
| 07 | Failure modes: plateaus, entropy collapse, mis-specified baselines |

## search_and_planning/

| File | What it proves |
|---|---|
| 01 | UCB1's regret bound: the theorem behind tree search |
| 02 | MCTS/UCT: value backup as recursive bandit |
| 03 | AlphaGo: four networks + search, every training equation derived |

## distributional_rl/

| File | What it proves |
|---|---|
| 01 | Wasserstein metrics and optimal transport background |
| 02 | The distributional Bellman operator: gamma-contraction in max-W_p |
| 03 | Control is NOT a contraction: the counterexamples, worked |
| 04 | C51: the categorical projection and the KL surrogate |
| 05 | Quantile regression: minimizing W1 directly; Cramer geometry |

## average_reward/

| File | What it proves |
|---|---|
| 01 | Cesaro limit P* exists for finite chains; P*P = PP* = P*; multichain example |
| 02 | Gain and bias; existence/uniqueness for the Poisson equation |
| 03 | The Laurent expansion v_gamma = rho/(1-gamma) + h + O(1-gamma) |
| 04 | Blackwell optimality: one policy optimal for all gamma near 1 |
| 05 | Span-seminorm contraction (Doeblin modulus); the periodic-chain failure |
| 06 | Differential TD(0) convergence via the ODE method, tabular unichain |

## total_reward_ssp/

| File | What it proves |
|---|---|
| 01 | Proper policies: contraction in a weighted sup-norm (explicit w) |
| 02 | Improper policies: uniqueness and its failure, counterexample worked |
| 03 | Positive/negative DP: monotone convergence; the greedy-fails example |
| 04 | Episodic RL as SSP: timeout truncation formalized |

## lqr_and_continuous_control/

| File | What it proves |
|---|---|
| 01 | Bellman backup = Riccati recursion; certainty equivalence proved |
| 02 | J(K) nonconvex yet gradient-dominated (Fazel et al. key lemma) |
| 03 | HJB from the DP principle; where classical solutions fail (kink worked) |
| 04 | Linearly solvable control: the exponentiated Bellman equation is linear |
| 05 | What LQR predicts and structurally misses; sample-complexity comparison |

## concentration_toolkit/

| File | What it proves |
|---|---|
| 01 | The Chernoff method; Hoeffding's lemma; Hoeffding and Bernstein |
| 02 | Azuma–Hoeffding; Freedman's inequality (supermartingale core) |
| 03 | The method-of-mixtures self-normalized bound; confidence ellipsoids |
| 04 | Covering numbers; uniform Hoeffding over nets and linear balls |
| 05 | Le Cam two-point; Bretagnolle–Huber; the divergence decomposition |

## finite_time_td_q/

| File | What it proves |
|---|---|
| 01 | Finite-time contractive linear SA: O(1/t) decay, O(alpha) bias |
| 02 | TD(0) rates under i.i.d. and Markov sampling (mixing correction proved) |
| 03 | Synchronous Q-learning: the (1-gamma)^-4 bound, every factor audited |
| 04 | The Hoeffding-to-Bernstein swap: law of total variance, (1-gamma)^-3 |
| 05 | The Omega(SA/((1-gamma)^3 eps^2)) lower bound: hard MDP family worked |
| 06 | Polyak–Ruppert averaging; the O(sqrt(alpha)) hover radius made precise |

## regret_and_exploration/

| File | What it proves |
|---|---|
| 01 | The generic optimism lemma: regret telescopes confidence widths |
| 02 | UCBVI with Hoeffding bonuses: the complete regret proof |
| 03 | The Bernstein refinement: total-variance lemma, near-minimax rate |
| 04 | The Omega(sqrt(H^3 SAT)) construction; epsilon-greedy's exp lower bound |
| 05 | PSRL: Bayesian regret via posterior matching |
| 06 | IDS: the information-ratio regret bound (Russo–Van Roy) |
| 07 | Eluder dimension and its regret lemma; the honest deep-RL gap |

## linear_mdps_and_completeness/

| File | What it proves |
|---|---|
| 01 | Linear MDP closure: Q_pi linear for every pi; the rank-d fine print |
| 02 | LSVI-UCB: self-normalized concentration + elliptical potential lemma |
| 03 | Realizability vs completeness; completeness non-monotone (worked) |
| 04 | Bellman rank: the elimination lemma (volumetric argument) |
| 05 | Wang–Foster–Kakade: realizable offline linear RL needs exp samples |
| 06 | What deep networks change: completeness as why target nets help |

## offline_rl_and_ope/

| File | What it proves |
|---|---|
| 01 | Trajectory IS: the exponential-in-horizon variance lower bound |
| 02 | Doubly robust estimation: double robustness proved, variance derived |
| 03 | Marginalized IS: the density-ratio identity; variance collapse to poly(H) |
| 04 | Pessimism: LCB-VI under single-policy concentrability |
| 05 | CQL's implicit pessimistic operator, closed form derived |
| 06 | Munos–Szepesvari error propagation; infinite-coefficient MDP worked |

## regularized_mdps_and_duality/

| File | What it proves |
|---|---|
| 01 | The soft Bellman operator: Legendre pair, contraction, bias bound |
| 02 | Control as inference: ELBO = entropy-regularized RL; the risk-seeking bug |
| 03 | Soft policy improvement and convergence of soft PI; SAC's temperature dual |
| 04 | Geist–Scherrer–Pietquin: regularized MPI and error propagation |
| 05 | KL-regularized PI IS mirror descent; the three-point lemma's payoff |
| 06 | Fenchel–Rockafellar duality: DICE estimators from the regularized LP |

## global_convergence_of_pg/

| File | What it proves |
|---|---|
| 01 | Softmax PG: the non-uniform Lojasiewicz inequality; the exp(-H) plateau |
| 02 | Entropy-regularized NPG converges linearly; unregularized O(1/t) |
| 03 | Compatible function approximation (Sutton et al. part 2), proved |
| 04 | The NPG transfer-error theorem; the mismatch coefficient's return |
| 05 | The occupancy-polytope geometry assembled; central-path convexity |

## rlhf_mathematics/

| File | What it proves |
|---|---|
| 01 | Bradley–Terry identifiability: shift equivalence class, what preferences can't see |
| 02 | The KL-regularized optimum: closed form, Donsker–Varadhan, support is destiny |
| 03 | DPO derived: the partition function cancels; the gradient analysis |
| 04 | Best-of-n: KL(BoN||ref) <= log n - (n-1)/n, exact for continuous rewards |
| 05 | Overoptimization: max-bias IS Goodhart; the KL-budget stopping heuristic |
| 06 | Token MDP or bandit: GAE(lambda=1) reduces PPO to sequence-level REINFORCE |
| 07 | RLVR and R1: verifiable rewards, GRPO's estimator, what changes at the frontier |

## stochastic_games/

| File | What it proves |
|---|---|
| 01 | The Shapley operator is a gamma-contraction (val is nonexpansive) |
| 02 | Shapley's fictitious-play cycle; naive PI for games fails (van der Wal) |
| 03 | Blackwell approachability -> regret matching; the CFR decomposition theorem |
| 04 | Fictitious self-play converges in zero-sum; double oracle terminates |
| 05 | Two independent Q-learners cycle: the replicator reduction worked |

## pomdps_and_information_state/

| File | What it proves |
|---|---|
| 01 | The belief update; sufficiency: the belief MDP is value-equivalent |
| 02 | Piecewise-linear convexity by induction; the tiger problem, all backups |
| 03 | PSPACE-completeness (QBF sketch); undecidability at infinite horizon |
| 04 | PSR rank bounds; strictly more compact than beliefs (float/gate example) |
| 05 | Approximate information states: the value-loss bound (Subramanian) |

## model_based/

| File | What it proves |
|---|---|
| 01 | The simulation lemma, both norms, via the resolvent-difference identity |
| 02 | Certainty equivalence achieves the minimax (1-gamma)^-3 rate |
| 03 | R-MAX's explore-or-exploit dichotomy; UCRL2's confidence polytope |
| 04 | Value equivalence: the VE model set is affine; planning losses coincide |
| 05 | Branched rollouts: the Janner bound; Dyna as a contraction average |
| 06 | MuZero, the paper: learned-VE + MCTS, every training equation |

## options_and_hierarchy/

| File | What it proves |
|---|---|
| 01 | Options induce an SMDP; the SMDP optimality operator contracts |
| 02 | Intra-option Bellman equations; off-policy learning across options |
| 03 | The intra-option policy gradient and termination gradient theorems |
| 04 | What hierarchy buys (exploration, not value); the collapse degeneracies |

## successor_representations/

| File | What it proves |
|---|---|
| 01 | The SR is the learned resolvent (I - gamma P)^-1; TD on occupancies |
| 02 | Successor features + GPI: the transfer theorem with its bound, proved |
| 03 | Scope: rewards transfer, dynamics and policies do not; UVFA contrast |

## imitation_and_inverse_rl/

| File | What it proves |
|---|---|
| 01 | BC pays eps H^2 (Ross–Bagnell); DAgger's eps H via no-regret |
| 02 | IRL is ill-posed; MaxEnt IRL = the exponential-family/tilting transform |
| 03 | IRL's dual is occupancy matching; the GAIL objective derived |
| 04 | Failure modes: reward ambiguity, shift, the demonstrator ceiling |

## sequence_models_and_rl/

| File | What it proves |
|---|---|
| 01 | Decision Transformer: what return-conditioned autoregression estimates |
| 02 | Conditioning is not control: the two theorems (risk-seeking; no stitching) |
| 03 | When sequence modeling wins anyway: the calibrated scoreboard |

## unifying_view/

| File | What it holds |
|---|---|
| 01 | Every method as one row of (operator T, metric d, function class F) |
| 02 | The reading map: Sutton–Barto vs Szepesvari; cross-references |

## The Recurring Instruments

Five arguments carry a disproportionate share of this ledger:

```text
the contraction argument        mdp_foundations/03-04 -> every family:
                                Shapley (1953!) before Bellman, soft
                                operators, SMDPs, distributional T^pi
the Legendre/tilting transform  softmax = tilted reference: Chernoff
                                (concentration/01), soft Bellman
                                (regularized/01), linearly solvable
                                control (lqr/04), MaxEnt IRL
                                (imitation/02), RLHF's closed form
                                (rlhf/02) — one transform, five fields
the performance difference      policy_gradient/03 -> TRPO's bound,
lemma                           LQR's gradient domination, NPG transfer
max-bias / order statistics     Double Q (value_based/03), Goodhart
                                (rlhf/05), model exploitation
                                (model_based/05) — the same estimator
                                pathology, three costumes
the resolvent                   (I - gamma P)^-1: Laurent expansion
                                (average_reward/03), simulation lemma
                                (model_based/01), the SR learned as an
                                object (successor_representations/01)
```

The transformer-side continuation of this program lives at
[attention-ledger](https://github.com/y2x0/attention-ledger).
