---
id: arxiv:1706.03741
type: paper
title: Deep Reinforcement Learning from Human Preferences (RLHF Foundation)
url: https://arxiv.org/abs/1706.03741
retrieved: '2026-07-11'
maturity: comprehensive
topic: policy-gradient-methods
---

# Deep Reinforcement Learning from Human Preferences

### Core Problem
Traditional reinforcement learning (RL) relies on well-specified reward functions. However, many real-world goals are too complex or poorly defined to be hand-engineered, and designing approximate rewards often leads to "reward hacking" where the agent optimizes the reward without satisfying the actual intent. While imitation learning can extract rewards from demonstrations, it is inapplicable to tasks that humans cannot easily demonstrate (e.g., controlling high-degree-of-freedom robots). Direct human feedback as a reward signal is prohibitively expensive for deep RL, which requires millions of interactions. The authors seek a method to communicate complex goals to RL agents using minimal human feedback.

### Method
The proposed approach learns a reward function $\hat{r}$ from human preferences over trajectory segments and then optimizes a policy $\pi$ to maximize this predicted reward. The system operates via three asynchronous processes:

1.  **Policy Optimization:** The policy $\pi$ interacts with the environment to produce trajectories. It is updated using traditional RL algorithms—specifically **Advantage Actor-Critic (A2C)** for Atari games and **Trust Region Policy Optimization (TRPO)** for MuJoCo robotics tasks—to maximize the sum of rewards predicted by $\hat{r}$.
2.  **Preference Elicitation:** The system selects pairs of trajectory segments ($\sigma^1, \sigma^2$), typically 1–2 seconds long, and presents them as video clips to a human. The human indicates which segment is preferred, if they are equal, or if they are incomparable.
3.  **Reward Function Fitting:** The reward predictor $\hat{r}$ (a deep neural network) is trained via supervised learning to fit the collected comparisons.

To improve stability and performance, the authors implement several modifications:
*   **Ensembling:** An ensemble of predictors is trained on bootstrapped samples of the preference database; the final $\hat{r}$ is the average of these normalized predictors.
*   **Regularization:** $\ell_2$ regularization is used, with coefficients adjusted to keep validation loss between 1.1 and 1.5 times the training loss.
*   **Noise Modeling:** The model assumes a 10% probability that the human responds uniformly at random to account for human error.
*   **Active Querying:** Pairs are selected for human labeling based on the highest variance in predictions across the ensemble members.

### Key Formulas
The reward function $\hat{r}$ is treated as a preference-predictor using a Bradley-Terry model. The probability that a human prefers segment $\sigma^1$ over $\sigma^2$ is modeled as:

$$
\hat{P}[\sigma^1 \succ \sigma^2] = \frac{\exp \sum \hat{r}(o_t^1, a_t^1)}{\exp \sum \hat{r}(o_t^1, a_t^1) + \exp \sum \hat{r}(o_t^2, a_t^2)}
$$

The parameters of $\hat{r}$ are optimized by minimizing the cross-entropy loss:

$$
\text{loss}(\hat{r}) = - \sum_{(\sigma^1, \sigma^2, \mu) \in \mathcal{D}} \mu(1) \log \hat{P}[\sigma^1 \succ \sigma^2] + \mu(2) \log \hat{P}[\sigma^2 \succ \sigma^1]
$$

where $\mu$ is the distribution over the human's choice.

### Quantitative Results
The method was evaluated on MuJoCo robotics and Atari games, requiring feedback on less than 1% of environment interactions.

*   **MuJoCo:** With 700 human queries, the agent nearly matched the performance of RL trained on the true reward. With 1,400 synthetic labels, the algorithm occasionally outperformed RL with the true reward, suggesting the learned reward function provided better shaping.
*   **Atari:** Using 5,500 human queries, the agent showed substantial learning. Synthetic labels matched or approached RL performance in *BeamRider* and *Pong* with approximately 3,300 labels.
*   **Novel Behaviors:** The system learned complex behaviors with roughly one hour of human feedback:
    *   **Hopper:** Sequence of backflips (900 queries).
    *   **Half-Cheetah:** Moving forward on one leg (800 queries).
    *   **Enduro:** Staying even with other cars (1,300 queries).

### Limitations
*   **Offline Training:** Training the reward predictor on a static dataset (no online queries) performed poorly. Due to the nonstationarity of the occupancy distribution, the agent often developed "bizarre" behaviors, such as avoiding losing points in *Pong* without attempting to score.
*   **Feedback Granularity:** Comparing single frames was significantly less helpful than comparing short clips; longer clips provided necessary context for human evaluation.
*   **Human Consistency:** Real human feedback was occasionally less efficient than synthetic oracle feedback due to labeling errors or inconsistent rates of labeling.
*   **Task Complexity:** Some environments (e.g., *Qbert*) proved difficult to learn from short clips because the clips were confusing to evaluate.
