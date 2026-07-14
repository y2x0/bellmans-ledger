# Stochastic Approximation

This family asks:

```text
What survives when the Bellman operator can only be sampled,
one transition at a time?
```

The template for every algorithm here is a noisy fixed-point iteration:

```math
\theta_{t+1}=\theta_t+\alpha_t\big[\ (\mathcal{F}\theta_t-\theta_t)\ +\ M_{t+1}\ \big],
```

where `F` is some Bellman-type operator and `M_{t+1}` is a martingale
difference (mean-zero given the past). The questions each notebook must
answer:

```text
1. What is F, and is it a contraction? In which norm?
2. Is the noise really martingale (unbiased given the filtration)?
3. Do the step sizes satisfy Robbins–Monro?
4. Does the sampling distribution match the norm in which F contracts?
```

Divergence phenomena (notebook 06) are always a "no" to question 4.

Sources: Sutton & Barto ch. 5–6, 11; Szepesvári §3–4 and the stochastic
approximation references therein; Tsitsiklis & Van Roy 1997; Watkins & Dayan
1992; Baird 1995.

## Folder Map

```text
01_robbins_monro_and_ode_method.md
    step-size conditions, the ODE method, two-timescale iterates

02_monte_carlo_and_importance_sampling.md
    first/every-visit MC, ordinary vs weighted IS, variance pathology

03_td_lambda_forward_backward.md
    TD(0), the lambda-return, eligibility traces, forward = backward

04_tsitsiklis_van_roy_linear_fa.md
    projected Bellman equation, on-policy L2 contraction, the TvR theorem

05_q_learning_convergence.md
    Watkins' theorem as asynchronous stochastic approximation

06_deadly_triad_baird.md
    off-policy + bootstrapping + function approximation: exact divergence
```
