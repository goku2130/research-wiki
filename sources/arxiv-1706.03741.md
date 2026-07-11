---
id: arxiv:1706.03741
type: paper
title: Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017)
url: https://arxiv.org/abs/1706.03741
retrieved: '2026-07-11'
maturity: comprehensive
topic: mdp-formulation
---

# Deep Reinforcement Learning from Human Preferences

### Core Problem
Traditional reinforcement learning (RL) relies on well-specified reward functions. However, many real-world goals are complex or poorly defined, making them difficult to hand-engineer without causing "reward hacking," where the agent optimizes the reward function without satisfying the actual human intent. While imitation learning can address this, it is inapplicable to tasks that are difficult for humans to demonstrate (e.g., controlling robots with non-human morphology). The authors seek a method to communicate complex goals to RL systems using minimal human feedback, enabling the learning of behaviors that can be recognized but not necessarily demonstrated.

### Method
The authors propose learning a reward function $\hat{r}$ from human preferences over trajectory segments and then optimizing a policy $\pi$ to maximize that predicted reward. Both $\pi$ and $\hat{r}$ are parameterized by deep neural networks. The system operates via three asynchronous processes:

1.  **Policy Optimization:** The policy $\pi$ interacts with the environment to produce trajectories. It is updated using traditional RL algorithms—Advantage Actor-Critic (A2C) for Atari games and Trust Region Policy Optimization (TRPO) for MuJoCo robotics—to maximize the sum of predicted rewards $r_t = \hat{r}(o_t, a_t)$.
2.  **Preference Elicitation:** The system selects pairs of trajectory segments $(\sigma^1, \sigma^2)$, typically 1–2 seconds long. A human overseer indicates which segment is preferred, if they are equal, or if they are incomparable.
3.  **Reward Function Fitting:** The reward predictor $\hat{r}$ is optimized via supervised learning to fit the collected comparisons.

**Key Formulas**
The reward function is treated as a preference predictor using a Bradley-Terry model. The probability that a human prefers segment $\sigma^1$ over $\sigma^2$ is modeled as:

$$
\hat{P}[\sigma^1 \succ \sigma^2] = \frac{\exp \sum \hat{r}(o_t^1, a_t^1)}{\exp \sum \hat{r}(o_t^1, a_t^1) + \exp \sum \hat{r}(o_t^2, a_t^2)}
$$

The parameters of $\hat{r}$ are optimized by minimizing the cross-entropy loss:

$$
\text{loss}(\hat{r}) = - \sum_{(\sigma^1, \sigma^2, \mu) \in \mathcal{D}} \mu(1) \log \hat{P}[\sigma^1 \succ \sigma^2] + \mu(2) \log \hat{P}[\sigma^2 \succ \sigma^1]
$$

where $\mu$ is the distribution over the human's choice.

**Implementation Details**
*   **Ensembling:** To handle uncertainty, the authors fit an ensemble of predictors and average their normalized results.
*   **Query Selection:** Pairs are selected for human labeling based on the highest variance in predictions across the ensemble.
*   **Regularization:** $\ell_2$ regularization is used, with the coefficient adjusted to keep validation loss between 1.1 and 1.5 times the training loss.
*   **Noise Modeling:** The model assumes a 10% chance that the human responds uniformly at random.

### Key Quantitative Results
The approach was tested on MuJoCo robotics and Atari games, requiring feedback on less than 1% of agent interactions.

*   **MuJoCo Robotics:** With 700 human queries, the agent nearly matched the performance of RL trained on the true reward function. With 1,400 synthetic (oracle) labels, the algorithm slightly outperformed RL using the true reward, suggesting the learned reward function may be better shaped.
*   **Atari Games:** Using 5,500 human queries, the agent showed substantial learning. Synthetic labels matched or approached RL performance on *BeamRider* and *Pong* with only 3,300 labels.
*   **Novel Behaviors:** The system learned complex behaviors from approximately one hour of human feedback:
    *   **Hopper:** Sequence of backflips (900 queries).
    *   **Half-Cheetah:** Moving forward on one leg (800 queries).
    *   **Enduro:** Keeping alongside other cars ($\sim 1,300$ queries).

### Limitations and Ablations
*   **Online vs. Offline:** Training the reward predictor only on initial data (offline) led to poor performance and "bizarre behavior," such as an agent in *Pong* avoiding losing points without attempting to score. Human feedback must be intertwined with RL learning.
*   **Segments vs. Frames:** Using single frames instead of short clips significantly impaired performance; clips provide necessary context and are more helpful per query.
*   **Human Noise:** Real human feedback was generally slightly less effective than synthetic oracle feedback due to labeling errors, inconsistency, or uneven labeling rates.
*   **Task Complexity:** Some environments (e.g., *Qbert*) were difficult to learn because short clips were confusing for human raters to evaluate.
