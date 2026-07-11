---
id: arxiv:2204.02515
type: paper
title: Inferring Rewards from Language in Context
url: https://arxiv.org/abs/2204.02515
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-math-and-code
---

# Inferring Rewards from Language in Context

### Core Problem
Traditional instruction-following models map natural language commands to specific actions within a single immediate context. However, these models fail to generalize to new environments because they do not infer the underlying "why"—the reward function—that motivates a user's preferences. The authors address the problem of inferring a general reward function $\theta$ from natural language utterances across repeated interactions, allowing an agent to autonomously execute high-reward trajectories in previously unseen contexts.

### Method
The authors propose a Bayesian framework where a rational listener, $L_2$, maintains and updates a posterior distribution over reward parameters $\theta$ based on observed utterances $u$ and contexts $\mathcal{M}$.

**Step-by-Step Recipe:**
1.  **Reward Parameterization:** The user's preference is modeled as a linear reward function $r_\theta(\xi) = \theta^\top \phi(\xi)$, where $\phi(\xi)$ is a feature vector of the trajectory (or flight) $\xi$.
2.  **Bayesian Update:** The agent updates the posterior over $\theta$ across a sequence of interactions $i$:

$$
p(\theta \mid u_{1:i}, \mathcal{M}_{1:i}) \propto p(\theta \mid u_i, \mathcal{M}_i) \times p(\theta \mid u_{1:i-1}, \mathcal{M}_{1:i-1})
$$

3.  **Speaker Modeling ($S_1$):** The listener $L_2$ reasons about a speaker $S_1$ who produces utterances based on a mixture of two goals: eliciting a specific action in the current context and describing the general reward. This is governed by a "nearsightedness" parameter $\alpha$:

$$
p_{S_1}(u \mid \theta, \mathcal{M}) = \alpha p_{\text{action}}(u \mid \theta, \mathcal{M}) + (1 - \alpha) p_{\text{reward}}(u \mid \theta)
$$

4.  **Action Component ($p_{\text{action}}$):** This models the probability that a speaker refers to a trajectory $\xi$ to elicit a correct action:

$$
p_{\text{action}}(u \mid \theta, \mathcal{M}) = \sum_{\xi} p_{\text{refer}}(u \mid \xi, \mathcal{M}) p_{\text{opt}}(\xi \mid \theta, \mathcal{M})
$$

    *   **Optimality Model ($p_{\text{opt}}$):** A Boltzmann distribution where $p_{\text{opt}}(\xi \mid \theta, \mathcal{M}) \propto \exp(\beta r_\theta(\xi; \mathcal{M}))$.
    *   **Reference Model ($p_{\text{refer}}$):** Modeled as $p_{\text{refer}}(u \mid \xi, \mathcal{M}) \propto p_{L_{\text{base}}}(\xi \mid u, \mathcal{M})$, where $L_{\text{base}}$ is a base instruction-following listener.
5.  **Reward Component ($p_{\text{reward}}$):** Modeled via a base speaker $S_{\text{base}}$ that maps rewards directly to reward-descriptive utterances: $p_{\text{reward}}(u \mid \theta) = p_{S_{\text{base}}}(u \mid \theta)$.
6.  **Implementation:** $L_{\text{base}}$ and $S_{\text{base}}$ use BERT-base encoders for utterances and MLP encoders for actions/rewards, trained on the FLIGHTPREF dataset.

### FLIGHTPREF Dataset
To evaluate the model, the authors created **FLIGHTPREF**, a dataset of natural language interactions in a flight booking game.
*   **Setup:** Users are given a hidden reward vector $\theta \in \{-1, -0.5, 0, 0.5, 1\}^8$.
*   **Data:** 2,568 utterances across 813 games (6 rounds per game).
*   **Task:** The assistant must learn $\theta$ to select the optimal flight in unseen option sets.

### Key Quantitative Results
The models were evaluated on **held-out accuracy** (the percentage of 1,000 randomly generated flight sets where the inferred reward $\hat{\theta}$ selected the same optimal flight as the true reward $\theta^*$).

| Method | Held-out Accuracy (%) |
| :--- | :--- |
| Action-only ($\alpha=1.0$) | $52.8 \pm 0.97$ |
| Reward-only ($\alpha=0.0$) | $57.8 \pm 0.95$ |
| **Action + Reward (Ours, $\alpha=0.5$)** | **$59.1 \pm 0.96$** |

**Comparisons:**
*   The full model significantly outperformed action-only and reward-only baselines ($p < .05$).
*   **Oracle Baselines:** An oracle that perfectly infers $k$ random features achieved $43.0\%$ for $k=1$, $51.5\%$ for $k=2$, $60.2\%$ for $k=3$, and $64.7\%$ for $k=4$. The proposed model's performance is comparable to an oracle with nearly 3 perfectly known features.
*   **Multi-turn Learning:** The full model outperformed baselines across all numbers of observed utterances, with statistically significant benefits over reward-only models at 5+ utterances.

### Limitations
*   **Base Model Error:** Errors in the base listener $L_{\text{base}}$ propagate to reward inference. Skipping updates where $L_{\text{base}}$ predicted the incorrect option improved held-out accuracy by $6\%$ over the reward-only model.
*   **Integration Ambiguity:** The authors note the difficulty in optimally integrating evidence; for example, determining if "I like the cheap JetBlue flight" indicates a general preference for JetBlue or is merely a unique identifier for a specific flight in the current set.
*   **Reward Complexity:** The current study uses linear reward functions; the authors suggest that future work should explore deep reward representations.
