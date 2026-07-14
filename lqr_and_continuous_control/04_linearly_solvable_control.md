# Linearly Solvable Control

## Todorov's Reformulation

Give up the discrete action set. The controller directly chooses the
next-state distribution, paying a KL price for deviating from a
**passive dynamics** `p(.|s)`:

```math
\text{choose }u(\cdot\mid s)\in\Delta(\mathcal{S}),
\qquad
\text{reward}=r(s)-\tau\,\mathrm{KL}\big(u(\cdot\mid s)\,\|\,p(\cdot\mid s)\big).
```

(Compare `regularized_mdps_and_duality/01`: there the KL price was on
the *action* distribution; here it is on the *transition* itself —
a stronger idealization: fully controllable dynamics with a soft leash.)

## The Bellman Equation Linearizes

The optimality equation (undiscounted first-exit form, `v` the value,
terminal values fixed on a boundary):

```math
v(s)=r(s)+\max_{u\in\Delta}\ \Big\{-\tau\,\mathrm{KL}\big(u\,\|\,p\big)+\mathbb{E}_{s'\sim u}\big[v(s')\big]\Big\}.
```

The inner max is exactly the Donsker–Varadhan/Legendre computation of
`regularized_mdps_and_duality/01` /`rlhf_mathematics/02` (fifth
appearance of the transform), with closed form:

```math
\max_u\{\cdot\}
=
\tau\,\log\ \mathbb{E}_{s'\sim p}\Big[e^{v(s')/\tau}\Big],
\qquad
u^*(s'\mid s)=\frac{p(s'\mid s)\,e^{v(s')/\tau}}{\mathbb{E}_p[e^{v/\tau}]}.
```

Substitute and **exponentiate**: define the desirability
`z(s) = e^{v(s)/tau}`. Then

```math
\boxed{\ z(s)=e^{\,r(s)/\tau}\ \sum_{s'}p(s'\mid s)\,z(s')\ }
\qquad\Longleftrightarrow\qquad
z=G\,P\,z,\quad G=\mathrm{diag}\big(e^{r/\tau}\big)
```

— the nonlinear Bellman equation has become a **linear** eigenvector/
boundary-value problem in `z`. The max was absorbed by the exponential
because softmax-of-KL-control *is* log-sum-exp, and log-sum-exp is log of
a linear operation.

## What Linearity Buys

```text
1. superposition of tasks: if z_1, z_2 solve the problem for terminal
   reward sets 1 and 2, then a z_1/z_2-mixture solves the mixed task —
   EXACT compositionality of optimal controllers (the formal basis for
   "compose soft Q-functions" heuristics in deep max-ent RL);
2. eigenproblem solvers replace DP: power iteration on GP computes
   control; spectral structure (the dominant eigenvalue) gives the
   average-reward analogue (tying to average_reward/ via the
   Perron root — log of the eigenvalue = optimal gain);
3. duality with estimation: z propagates like an unnormalized likelihood
   in a hidden Markov model — optimal control = Bayesian smoothing in a
   model where reward is log-likelihood: control-as-inference
   (regularized_mdps_and_duality/02) holding EXACTLY, not variationally,
   because the fully-controllable-transitions idealization removes the
   dynamics/tilting conflict that produced the risk-seeking pathology
   there. (The pathology's fingerprint survives: softmax over s' means
   the controller "buys luck" — here legitimately, since it truly can
   steer s'.)
```

## Path-Integral Control

The continuous-time twin (Kappen): for diffusions
`dx = f dt + sigma(u dt + dW)` with cost `(1/2)|u|^2` (the KL price in
disguise — Girsanov), the exponentiated HJB (notebook 03) linearizes the
same way, and the Feynman–Kac formula renders the solution as a **path
integral**:

```math
z(x)=\mathbb{E}_{\text{passive paths from }x}\Big[e^{\frac{1}{\tau}\int r\,dt}\Big]
\qquad\Longrightarrow\qquad
u^*=\ \text{reweighted average of sampled passive paths.}
```

Optimal controls computed by **importance sampling over rollouts of the
uncontrolled system** — no value iteration, no gradients. This is the
theory behind MPPI (model-predictive path integral control), a
practically dominant sampling MPC method in robotics: sample trajectories,
exponentially tilt by return, average the first action —
`rlhf_mathematics/04`'s BoN/tilt logic, operating inside a control loop.

## The Price Of The Magic

```text
the idealization:   control authority = full next-state distributions,
                    cost = KL. Real actuators offer a CONSTRAINED family
                    of achievable transition kernels; projecting onto
                    the LS class is a modeling error with no general
                    bound (and R != c*I control costs, state-dependent
                    noise, hard constraints all break the linearization);
the noise coupling: in the diffusion version, control cost and noise
                    covariance must be linked (sigma appears in both) —
                    "you can only push where the noise lives";
tau's role:         as everywhere in the regularized families, tau -> 0
                    recovers hard control but destroys the conditioning
                    (the linear operator GP becomes a max — numerically,
                    the eigenvector concentrates and sampling degenerates:
                    the SAME low-temperature failure as importance
                    sampling at small beta in rlhf_mathematics/04-05).
```

## What Remains Open

Characterizing the projection error onto the linearly-solvable class for
realistic actuation; sample complexity of path-integral estimators as
`tau -> 0` (the variance blows up exponentially — a
`concentration_toolkit/05`-style analysis exists in pieces); and
compositionality (item 1) under function approximation, where the exact
superposition becomes a heuristic with unquantified interference.
