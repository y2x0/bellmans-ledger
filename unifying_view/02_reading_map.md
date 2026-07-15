# Reading Map

## Traversal Order

```text
pass 1 — the exact theory (model known)
    Sutton & Barto ch. 1–4          alongside mdp_foundations/01–05
    Szepesvári Appendix A           the proofs S&B narrates; do them by hand
    Bellman 1957 (papers/)          one afternoon; mdp_foundations/07 as guide

pass 2 — the sampled theory (model unknown)
    Sutton & Barto ch. 5–6          alongside stochastic_approximation/02–03
    Szepesvári §2–3                 the stochastic-approximation formalization
    Sutton & Barto ch. 9, 11        alongside stochastic_approximation/04, 06
    Szepesvári §4.3                 alongside stochastic_approximation/05

pass 3 — the papers (approximation regime)
    Mnih 2015 (DQN)                 value_based_deep_rl/01–04
    Schulman 2017 (PPO)             policy_gradient/01–07 in order;
                                    the paper itself is short — the depth is
                                    in Kakade–Langford 2002 and TRPO 2015
    Silver 2016 (AlphaGo)           search_and_planning/01–03
    Bellemare 2017 (C51)            distributional_rl/01–05

pass 4 — close the loop
    unifying_view/01                place everything; re-derive the three
                                    instruments from memory
```

## Sutton & Barto vs Szepesvári, Honestly

```text
                       Sutton & Barto 2e            Szepesvári 2010
role                   intuition, breadth,          compressed rigor,
                       algorithm zoo                theorem statements
MDP theory             ch. 3–4, proofs sketched     Appendix A, proofs given
TD                     ch. 6, 12 (traces)           §3 as stochastic approx.
function approx.       ch. 9–11 (incl. triad)       §3.2–3.3 (LSTD, conditions)
control                ch. 5–6, 10                  §4 (with convergence refs)
what each omits        rates, precise conditions    deep RL entirely (2010)
```

Use S&B to know *what the algorithms are*; use Szepesvári to know *what is
actually true about them*. Disagreements in emphasis (e.g. S&B's semi-gradient
framing vs Szepesvári's fixed-point framing of TD) are resolved in
`stochastic_approximation/03–04`.

## Theorem Index (result -> notebook -> source)

```text
stationary deterministic policies suffice      mdp_foundations/01    Puterman 6.2.4
Bellman operators contract (sup-norm)          mdp_foundations/03    Szepesvári A.2
Banach fixed point; error/stopping bounds      mdp_foundations/04    Szepesvári A.1
greedy(v*) optimal; VI finite identification   mdp_foundations/04-05 Puterman ch.6
policy improvement; PI finite convergence      mdp_foundations/05    S&B 4.2–4.3
PI = Newton on Bellman residual                mdp_foundations/05    Puterman 6.4
LP duality; occupancy polytope                 mdp_foundations/06    Puterman 6.9
functional equation, successive approx.        mdp_foundations/07    Bellman 1957
Robbins–Monro; ODE method; two timescales      stoch_approx/01       Borkar 2008
IS unbiasedness; OIS/WIS tradeoff              stoch_approx/02       S&B 5.5
forward = backward view of TD(lambda)          stoch_approx/03       S&B 12.1–12.2
TvR: linear on-policy TD converges + bound     stoch_approx/04       Tsitsiklis–Van Roy 1997
Watkins: tabular Q-learning converges a.s.     stoch_approx/05       Watkins–Dayan 1992; JJS 1994
Baird divergence; MSPBE/GTD escape             stoch_approx/06       Baird 1995; Sutton 2009
fitted-VI error propagation                    value_deep/02         Munos–Szepesvári 2008
E[max] bias; double estimator fix              value_deep/03         van Hasselt 2010, 2016
policy gradient theorem (two proofs)           policy_gradient/01    Sutton et al. 1999
zero-bias baselines; optimal baseline          policy_gradient/02    Greensmith et al. 2004
performance difference lemma                   policy_gradient/03    Kakade–Langford 2002
TRPO minorization bound; natural gradient      policy_gradient/04    Schulman 2015; Kakade 2001
GAE = discounted sum of TD errors              policy_gradient/05    Schulman 2016
L^CLIP pessimistic bound; PPO ablations        policy_gradient/06    Schulman 2017
UCB1 log-regret                                search_planning/01    Auer et al. 2002
UCT asymptotic optimality                      search_planning/02    Kocsis–Szepesvári 2006
AlphaGo training + search equations            search_planning/03    Silver 2016
W_p coupling form; affine homogeneity          distributional/01     Villani ch.6
distributional T^pi is gamma-contraction       distributional/02     BDM 2017 Lemma 3
control non-contraction, no fixed point        distributional/03     BDM 2017 Prop.1–Thm.1
C51 projection + KL; Cramér reconciliation     distributional/04–05  BDM 2017; Rowland 2018
quantile projection; QR loss unbiasedness      distributional/05     Dabney 2017

--- expansion families ---
Hoeffding lemma via tilting; Bernstein mgf     concentration/01      Boucheron–Lugosi–Massart
Azuma; Freedman supermartingale                concentration/02      Freedman 1975
mixtures; ellipsoids; elliptical potential     concentration/03      Abbasi-Yadkori et al. 2011
covering numbers; uniform Bellman bounds       concentration/04      Wainwright ch.5
Le Cam; Bretagnolle–Huber; divergence decomp.  concentration/05      Lattimore–Szepesvári ch.15
linear SA O(1/t); constant-step noise ball     finite_time/01        Bhandari et al. 2018
TD(0) rates; Markov-noise mixing inflation     finite_time/02        Srikant–Ying 2019
Q-learning (1-g)^{-4}; the horizon audit       finite_time/03        Even-Dar–Mansour 2003
total-variance lemma; minimax (1-g)^{-3}       finite_time/04        Azar 2013; Wainwright 2019
generative lower bound Omega((1-g)^{-3})       finite_time/05        Azar et al. 2013
Polyak–Ruppert optimality; EMA reading         finite_time/06        Polyak–Juditsky 1992
optimism lemma: regret <= width sum            regret/01             AJKS ch.7
UCBVI full proof O(H^2 sqrt(SAT))              regret/02             Azar et al. 2017
episodic total-variance; sqrt(H^3 SAT)         regret/03             Azar et al. 2017
Omega(sqrt(HSAT)); eps-greedy Omega(A^H)       regret/04             Jaksch et al.; the lock
PSRL posterior matching; Bayes regret          regret/05             Osband et al. 2013
information ratio bound; IDS beats optimism    regret/06             Russo–Van Roy 2014/18
eluder dimension; width counting for F         regret/07             Russo–Van Roy 2013
linear MDP closure properties                  linear_mdps/01        Jin et al. 2020
LSVI-UCB sqrt(d^3 H^4 K)                       linear_mdps/02        Jin et al. 2020
realizability != completeness; exp hardness    linear_mdps/03        Weisz et al. 2021
Bellman rank; OLIVE elimination                linear_mdps/04        Jiang 2017; Du 2021
offline realizable+coverage exp lower bound    linear_mdps/05        Wang–Foster–Kakade 2020
trajectory IS exp(H) variance (exact)          offline/01            worked instance
doubly robust: product bias; efficiency        offline/02            Jiang–Li 2016
marginalized IS; backward flow equation        offline/03            Liu et al. 2018
pessimism + single-policy concentrability      offline/04            Rashidinejad et al. 2021
CQL implicit pessimistic operator              offline/05            Kumar et al. 2020
fitted-Q concentrability thm + counterexample  offline/06            Munos–Szepesvári 2008
logsumexp = entropy conjugate; soft contraction reg_mdps/01          classical
control-as-inference; risk-seeking pathology   reg_mdps/02           Levine 2018
soft improvement theorem; SAC; dual temp       reg_mdps/03           Haarnoja et al. 2018
Omega-regularized operators; error propagation reg_mdps/04           Geist et al. 2019
three-point lemma; NPG/TRPO/PPO as MD          reg_mdps/05           Neu et al. 2017
Fenchel–Rockafellar dual; DICE derived         reg_mdps/06           Nachum–Dai 2020
non-uniform Łojasiewicz; O(1/t) global PG      global_pg/01          Mei et al. 2020
NPG log|A|/(eta K); linear rate w/ entropy     global_pg/02          Agarwal 2021; Cen 2021
compatible FA theorem; w* = natural gradient   global_pg/03          Sutton 1999; Kakade 2001
NPG transfer-error theorem; kappa              global_pg/04          Agarwal et al. 2021
central path; the conservation law             global_pg/05          synthesis
Bradley–Terry identifiability (mod shifts)     rlhf/01               classical
pi* = pi_ref e^{r/beta}/Z; Donsker–Varadhan    rlhf/02               classical
DPO exactness via partition cancellation       rlhf/03               Rafailov et al. 2023
KL(BoN||ref) = log n - (n-1)/n                 rlhf/04               Beirami et al. 2024
Goodhart = selected max-bias; scaling laws     rlhf/05               Gao et al. 2023
GAE(1) collapse; GRPO = group baseline         rlhf/06               Shao et al. 2024
Cesàro limit P*; unichain taxonomy             average_reward/01     Puterman ch.8
Poisson equation; deviation matrix             average_reward/02     Puterman 8.2
Laurent series of the resolvent                average_reward/03     Puterman 8.2.3
Blackwell optimality (rational functions)      average_reward/04     Blackwell 1962
span contraction; RVI; 2-cycle failure         average_reward/05     Puterman 8.5
differential TD/Q convergence                  average_reward/06     Wan–Naik–Sutton 2021
properness => weighted-norm contraction        ssp/01                Bertsekas–Tsitsiklis 1991
mixed case uniqueness; loitering example       ssp/02                BT 1991
positive/negative DP asymmetry                 ssp/03                Blackwell 1967; Strauch 1966
truncation-vs-termination bias; gamma trade    ssp/04                practice, formalized
Riccati from Bellman; certainty equivalence    lqr/01                classical
J(K) gradient domination; global linear rate   lqr/02                Fazel et al. 2018
HJB from DPP; viscosity uniqueness             lqr/03                Crandall–Lions
exponentiated Bellman is linear; MPPI          lqr/04                Todorov 2006; Kappen
LQR testbed calibration                        lqr/05                Recht 2019
Shapley operator contraction; game value       games/01              Shapley 1953
fictitious-play cycle; naive PI fails          games/02              Shapley 1964; van der Wal
regret matching; CFR decomposition             games/03              Hart–Mas-Colell; Zinkevich
FSP averages -> Nash; double oracle finite     games/04              Robinson 1951; McMahan 2003
replicator ODE; conserved KL orbits            games/05              Tuyls 2003; Sato et al.
belief sufficiency theorem                     pomdps/01             Åström 1965
PWLC by induction; tiger backups               pomdps/02             Smallwood–Sondik 1973
PSPACE-completeness; undecidability            pomdps/03             Papadimitriou–Tsitsiklis
system-dynamics rank; spectral PSRs            pomdps/04             Littman et al. 2002
AIS value-loss bound                           pomdps/05             Subramanian et al. 2022
resolvent-difference simulation lemma          model_based/01        Kearns–Singh 2002
certainty equivalence minimax; leave-one-out   model_based/02        Azar 2013; AKY 2020
R-MAX dichotomy; UCRL2 D S sqrt(AT)            model_based/03        Brafman–Tennenholtz; Jaksch
value equivalence propositions; MuZero         model_based/04        Grimm et al. 2020
branched rollout bound; model exploitation     model_based/05        Janner et al. 2019

--- paper-driven additions (G-phase) ---
SMDP contraction, modulus E[gamma^tau]         options/01            Sutton–Precup–Singh 1999
interruption improvement theorem               options/01            SPS 1999 Thm 2
intra-option equations + convergence           options/02            SPS 1999
intra-option policy + termination gradients    options/03            Bacon–Harb–Precup 2017
SR = resolvent; TD per column                  successor_reps/01     Dayan 1993
GPI theorem + transfer bound                   successor_reps/02     Barreto et al. 2017
BC eps H^2 (tight); DAgger eps H               imitation/01          Ross–Bagnell 2010/11
MaxEnt IRL = exponential-family MLE            imitation/02          Ziebart et al. 2008
RL o IRL_psi = psi*-occupancy matching         imitation/03          Ho–Ermon 2016 Prop 3.2
DT estimates the behavior conditional          seq_models/01         Chen et al. 2021
conditioning buys luck; cannot stitch          seq_models/02         (worked; ESPER et al.)
MuZero K-step loss as sampled VE constraints   model_based/06        Schrittwieser et al. 2019
RLVR deletes proxy-Goodhart; GRPO frontier     rlhf/07               DeepSeek-AI 2025
```

## Threads Closed By The Expansion

```text
was open in v1                          now lives at
average-reward theory; Blackwell        average_reward/01–06
finite-time TD/Q; variance reduction    finite_time_td_q/01–06
offline RL: CQL, pessimism              offline_rl_and_ope/04–05
mirror-descent view of PPO              regularized_mdps_and_duality/05
global convergence of softmax PG        global_convergence_of_pg/01–02
RLHF / DPO                              rlhf_mathematics/01–06
MuZero / value equivalence              model_based/04
self-play theory (AlphaZero)            stochastic_games/04
```

## Open Threads (the frontier after ~120 files)

```text
concentration/finite-time   EMA-under-drift theory (what deep RL actually
                            runs); instance-optimal control lower bounds
regret/linear_mdps          a dimension that is small for neural classes,
                            sufficient for optimism, and computable — the
                            DEC program (regret/07, linear_mdps/06)
offline                     pessimism scales for neural classes with
                            comparator guarantees (CQL's missing half)
regularized/global_pg       PPO's clip as a genuine Bregman prox (no
                            telescoping bound exists); learned features in
                            the conservation law
rlhf                        process rewards vs identifiability; predictive
                            theory of overoptimization coefficients;
                            multi-turn operator theory
average_reward/ssp          function approximation for (rho, h); the
                            D-vs-span(h*) regret gap
games                       last-iterate convergence for deep zero-sum;
                            n>2 multi-agent regret frameworks
pomdps                      parameterized hardness between benign and
                            QBF-shaped ambiguity; transformers as
                            predictive statistics — the missing theorem
model_based                 value-aware losses end-to-end; a scaling law
                            for model exploitation (Goodhart for dynamics)
distributional_rl           IQN/FQF; risk-sensitive control; the
                            distributional RL book (BDR 2023)
```
