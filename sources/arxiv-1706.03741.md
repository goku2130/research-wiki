---
id: arxiv:1706.03741
type: paper
title: Deep reinforcement learning from human preferences (Christiano et al. 2017)
url: https://arxiv.org/abs/1706.03741
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlhf-ppo-pipeline
---

# Deep Reinforcement Learning from Human Preferences

### Core Problem
In complex reinforcement learning (RL) environments, defining a precise reward function is often difficult because goals may be poorly defined, complex, or hard to specify mathematically. While inverse reinforcement learning can extract rewards from demonstrations, this is impractical for tasks that are difficult for humans to demonstrate (e.g., controlling robots with non-human morphology). The authors seek a method to communicate complex goals to RL agents using non-expert human feedback that is economical in terms of human time and scales to deep RL systems.

### Method
The authors propose learning a reward function $\hat{r}$ from human preferences over pairs of trajectory segments and then optimizing a policy $\pi$ to maximize this predicted reward. The system operates via three asynchronous processes:

1.  **Policy Interaction:** The policy $\pi$ interacts with the environment to produce trajectories. The policy is updated using traditional RL algorithms—**Advantage Actor-Critic (A2C)** for Atari games and **Trust Region Policy Optimization (TRPO)** for MuJoCo robotics tasks—to maximize the sum of predicted rewards $r_t = \hat{r}(o_t, a_t)$.
2.  **Preference Elicitation:** The system selects pairs of trajectory segments $(\sigma^1, \sigma^2)$, typically 1–2 seconds long, and presents them to a human. The human indicates which segment is preferred, if they are equal, or if they are incomparable.
3.  **Reward Function Fitting:** A reward predictor $\hat{r}$ (a deep neural network) is trained via supervised learning to fit the collected preferences.

To improve stability and performance, the authors employ an **ensemble of predictors**, use $\ell_2$ regularization with a validation set, and assume a 10% probability that the human responds uniformly at random to account for human error. Queries are selected by sampling pairs and choosing those with the highest variance in predictions across the ensemble.

### Key Formulas
The reward function $\hat{r}$ is treated as a latent factor. The probability that a human prefers segment $\sigma^1$ over $\sigma^2$ is modeled using the Bradley-Terry model:

$$
\hat{P}[\sigma^1 \succ \sigma^2] = \frac{\exp \sum \hat{r}(o_t^1, a_t^1)}{\exp \sum \hat{r}(o_t^1, a_t^1) + \exp \sum \hat{r}(o_t^2, a_t^2)}
$$

The parameters of $\hat{r}$ are optimized by minimizing the cross-entropy loss between these predictions and the actual human labels $\mu$:

$$
\text{loss}(\hat{r}) = - \sum_{(\sigma^1, \sigma^2, \mu) \in \mathcal{D}} \mu(1) \log \hat{P}[\sigma^1 \succ \sigma^2] + \mu(2) \log \hat{P}[\sigma^2 \succ \sigma^1]
$$

### Key Quantitative Results
The approach requires feedback on less than 1% of the agent's interactions.

*   **Simulated Robotics (MuJoCo):** With 700 human queries, the agent nearly matched the performance of RL trained on the true reward function across eight tasks. With 1,400 synthetic labels, the algorithm occasionally outperformed RL using the true reward, suggesting the learned reward function may be better shaped.
*   **Atari Games:** Using 5,500 human queries, the agent showed substantial learning across seven games. In *BeamRider* and *Pong*, synthetic labels matched or approached RL performance with only 3,300 labels.
*   **Novel Behaviors:** The system learned complex behaviors with approximately one hour of human feedback:
    *   **Hopper:** Sequence of backflips (900 queries).
    *   **Half-Cheetah:** Moving forward on one leg (800 queries).
    *   **Enduro:** Keeping alongside other cars ($\sim 1,300$ queries).

### Stated Limitations
*   **Online Requirement:** Ablation studies showed that "offline" training (collecting queries only at the start) performed poorly. Due to the nonstationarity of the occupancy distribution, the agent often exploited the reward predictor, leading to "bizarre behavior" (e.g., in *Pong*, avoiding losing points without attempting to score).
*   **Query Granularity:** Using single frames instead of trajectory segments significantly impaired performance, as humans require context to evaluate behavior.
*   **Human Inconsistency:** Real human feedback was occasionally less efficient than synthetic oracle feedback due to labeling errors or inconsistent rates of feedback.
*   **Query Selection:** In some tasks, the variance-based query selection actually impaired performance compared to random sampling.
