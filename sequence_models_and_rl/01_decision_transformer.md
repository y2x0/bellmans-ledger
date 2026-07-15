# Decision Transformer

```text
Chen et al., NeurIPS 2021. papers/decision-transformer-chen-2021.pdf
```

## The Proposal

Abandon TD, abandon policy gradients: model offline trajectories as
sequences and let conditioning do the control. Trajectories are
re-serialized with the **return-to-go** prepended to each step:

```math
\tau=\Big(\hat R_1,\,s_1,\,a_1,\ \hat R_2,\,s_2,\,a_2,\ \ldots\Big),
\qquad
\hat R_t=\sum_{t'=t}^{T}r_{t'} ,
```

and a causally-masked transformer (GPT architecture, context window `K`
steps) is trained by plain supervised learning to predict actions:

```math
\min_\theta\ \ \mathbb{E}_{\tau\sim\mathcal{D}}\Big[\sum_t\ \ell\Big(a_t,\ \pi_\theta\big(\cdot\mid \hat R_{t-K:t},\,s_{t-K:t},\,a_{t-K:t-1}\big)\Big)\Big].
```

At deployment: set `R-hat_1` to the return you *want* (e.g. maximal in
the dataset, or higher), act, decrement by observed rewards
(`R-hat_{t+1} = R-hat_t - r_t`), repeat. No value function, no Bellman
backup, no importance weight — the entire `stochastic_approximation/`
inheritance is discarded by construction, along with (the paper's
motivation) the deadly triad and error propagation
(`offline_rl_and_ope/06`) that plague offline TD methods.

## What The Model Actually Estimates

The training objective's population optimum is the **conditional
behavior distribution**:

```math
\pi_{\mathrm{DT}}(a\mid s,\hat R)
=
\Pr_{\mathcal{D}}\big(a_t=a\ \big|\ s_t=s,\ \text{return-to-go}=\hat R\big)
```

— *the statistics of what the behavior policy did, on the occasions it
went on to achieve `R-hat`*. Three immediate consequences, each
developed in notebook 02:

```text
1. it is imitation of a return-selected slice of the data — BC
   (imitation_and_inverse_rl/01) on the sub-dataset {trajectories
   achieving R-hat}, with all of BC's compounding exposure plus a
   selection effect;
2. the conditioning event mixes the agent's skill with the
   environment's luck: achieving R-hat may have been CAUSED by actions
   or merely ACCOMPANIED by fortunate transitions — the model cannot
   tell (notebook 02's theorem);
3. asking for R-hat beyond the data support queries pure extrapolation
   of the sequence model — the analogue of the OOD-argmax problem of
   offline_rl_and_ope/04, without the pessimism machinery that fixed it.
```

## Results, Read Fairly

On the standard offline suites (D4RL Gym, Atari, Key-to-Door), DT
matches or exceeds the contemporary model-free offline baselines (CQL
et al.) — with notable strengths the theory would predict:

```text
- long horizons / sparse rewards: no bootstrap means no error
  propagation over the horizon (the (1-gamma)^{-2} of offline_rl/06
  simply absent); Key-to-Door-style credit assignment via attention
  over the full context outperforms TD's step-by-step propagation;
- dense, high-quality data: conditioning on high R-hat ~ imitating the
  best behavior present — exactly BC-on-the-best-slice, which is strong
  when the best slice is good;
- return targeting: achieved return tracks requested return remarkably
  linearly within the data's support (the model is a usable
  return-conditioned family of policies, not one policy).
```

The regime where it *loses* — stochastic environments and datasets
requiring stitching — is notebook 02's subject, because both losses are
theorems, not empirical accidents.

## The Lineage

Upside-down RL (Schmidhuber 2019) proposed command-conditioned
supervised learning; RvS (Emmons et al. 2021) showed two-layer MLPs
conditioned on outcomes match DT (the transformer is not the point; the
*conditioning* is); trajectory transformer (Janner et al. 2021) models
the full sequence including states and plans by beam search — a
model-based cousin that re-imports planning (`model_based/`) and
therefore escapes some of notebook 02's critique at the price of
compounding model error (`model_based/05`).

## What Remains Open

The correct conditioning variable (expectile-of-return, value estimates,
or "success" flags instead of raw return — each changes the notebook-02
analysis); context length as memory (DT as an approximate information
state — `pomdps_and_information_state/05` — is underexplored); and
scaling laws: whether return-conditioned pretraining inherits language-
model scaling is empirically contested territory.
