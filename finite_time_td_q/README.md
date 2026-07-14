# Finite-Time Analysis Of TD And Q-Learning

This family asks:

```text
The repo proves TD and Q-learning converge (Watkins, Tsitsiklis–Van Roy).
At what RATE — and is the rate optimal?
```

The asymptotic theory (`stochastic_approximation/`) says *whether*; this
family says *how fast, with what constants, in which factors of*
`1/(1-gamma)` — and closes the loop with matching lower bounds. The
storyline:

```text
01  contractive linear SA has O(1/t) mean-square error — the master theorem
02  TD(0) instantiates it; Markov (non-i.i.d.) sampling costs a mixing time
03  naive Q-learning analysis: (1-gamma)^{-4} via Hoeffding — audit of
    every horizon factor
04  the Bernstein + total-variance repair: (1-gamma)^{-3}, minimax-optimal
05  the matching lower bound: hard MDPs + coin testing (concentration_toolkit/05)
06  Polyak–Ruppert averaging: the statistically optimal way to run SA —
    and why deep RL doesn't run it
```

Prerequisite: `concentration_toolkit/` throughout;
`stochastic_approximation/01, 04, 05` for the objects being rated.

## The Coordinates

Same operators and classes as `stochastic_approximation/`; the new axis is
`n` (samples), and the results are two-sided:

```math
\underbrace{\tilde O\Big(\frac{|S||A|}{(1-\gamma)^3\varepsilon^2}\Big)}_{\text{upper, 04}}
\qquad\text{vs}\qquad
\underbrace{\Omega\Big(\frac{|S||A|}{(1-\gamma)^3\varepsilon^2}\Big)}_{\text{lower, 05}}
```

— the one chapter of RL sample complexity that is completely settled.
