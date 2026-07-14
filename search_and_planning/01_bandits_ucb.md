# Bandits And UCB1

## The Stateless Problem

`K` arms; arm `i` yields i.i.d. rewards in `[0,1]` with unknown mean `mu_i`.
Let `mu* = max_i mu_i`, gaps `Delta_i = mu* - mu_i`. Performance is
**regret** after `n` pulls:

```math
R_n=n\mu^*-\mathbb{E}\Big[\sum_{t=1}^n X_{I_t,t}\Big]
=\sum_{i}\Delta_i\,\mathbb{E}[T_i(n)],
```

`T_i(n)` the pulls of arm `i`. The bandit is the MDP with one state; what
remains when you delete the transition structure is exactly the
exploration–exploitation problem, isolated.

## UCB1 (Auer, Cesa-Bianchi, Fischer 2002)

Pull the arm maximizing an optimistic index:

```math
I_t=\arg\max_i\ \underbrace{\hat\mu_{i}}_{\text{empirical mean}}
+\underbrace{\sqrt{\frac{2\ln t}{T_i(t)}}}_{\text{confidence radius}} .
```

The radius comes from Hoeffding's inequality: for `T_i` samples,

```math
\Pr\big\{|\hat\mu_i-\mu_i|>\varepsilon\big\}\le 2e^{-2T_i\varepsilon^2},
\qquad
\varepsilon=\sqrt{\tfrac{2\ln t}{T_i}}
\ \Rightarrow\
\Pr\le 2t^{-4}.
```

"Optimism in the face of uncertainty": act as if every arm is as good as its
confidence interval allows. Under-explored arms have wide intervals and get
pulled; each pull shrinks the interval — exploration is *self-extinguishing*
exactly where it is unneeded.

## The Regret Theorem

```math
\mathbb{E}[T_i(n)]
\ \le\
\frac{8\ln n}{\Delta_i^2}+1+\frac{\pi^2}{3}
\qquad\Longrightarrow\qquad
R_n\ \le\ \sum_{i:\Delta_i>0}\Big(\frac{8\ln n}{\Delta_i}\Big)+O(1).
```

*Proof sketch.* Arm `i` is pulled at time `t` only if its index exceeds the
best arm's. That needs one of: (a) `mu*`'s interval failed (prob `<= t^{-4}`),
(b) arm `i`'s interval failed (`<= t^{-4}`), or (c) both intervals hold but
overlap, which forces `T_i <= 8 ln t / Delta_i^2`. Sum the failure
probabilities (convergent series — the `pi^2/3`) and the overlap bound. ∎

Logarithmic regret is optimal in order (Lai–Robbins lower bound:
`R_n >= (sum_i Delta_i / KL(mu_i, mu*)) ln n` asymptotically); UCB1 achieves
it without knowing the gaps.

## Variants That Matter Downstream

```text
UCB with tunable exploration:   index = mu-hat + c sqrt(ln t / T_i);
                                c trades constant factors -- this is the c
                                appearing in UCT and (transformed) in AlphaGo's
                                c_puct.

PUCT (predictor-UCB):           index = Q(i) + c * P(i) * sqrt(N) / (1 + N(i));
                                a PRIOR P over arms scales exploration --
                                arms believed good get explored first.
                                AlphaGo's selection rule (notebook 03) is this,
                                with P = the SL policy network. Note the
                                bonus decays like 1/N(i), not sqrt(ln N / N(i)):
                                with a good prior one accepts weaker
                                asymptotic exploration for better early focus.

epsilon-greedy for contrast:    explores forever at rate epsilon, uniformly:
                                LINEAR regret unless annealed; and even
                                annealed, it explores without direction.
                                UCB's log-regret is the formal argument that
                                DQN-style exploration is primitive
                                (value_based_deep_rl/04 failure 5).
```

## Why This Notebook Exists In An RL Repository

Tree search will face, at *every internal node*, a fresh bandit: which child
deserves the next simulation? UCT (notebook 02) is literally UCB1 applied
recursively, and its convergence proof leans on the regret bound to argue
that value estimates concentrate along optimal paths. The bandit is the
quantum of decision-time exploration.
