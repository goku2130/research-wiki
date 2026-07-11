---
id: arxiv:1706.03741
type: paper
title: Deep Reinforcement Learning from Human Preferences
url: https://arxiv.org/abs/1706.03741
retrieved: '2026-07-11'
maturity: comprehensive
topic: ppo-for-llms
---

## Deep Reinforcement Learning from Human Preferences

### Core Problem
The core problem addressed is the difficulty of specifying complex goals for sophisticated Reinforcement Learning (RL) systems in real-world environments, particularly when a well-defined reward function is unavailable or hard to engineer. Traditional RL relies on explicit reward functions, but many tasks have goals that are complex, poorly defined, or difficult to quantify. Inverse Reinforcement Learning (IRL) and imitation learning require demonstrations, which are not always feasible for complex behaviors or non-human morphologies. Directly using human feedback as a reward signal is prohibitively expensive for deep RL systems requiring vast amounts of experience. The authors aim to enable RL agents to solve tasks where only desired behavior can be recognized (not demonstrated), allow non-expert users to teach agents, scale to large problems, and be economical with user feedback.

### Method/Recipe Step by Step
The method involves learning a reward function from human preferences between pairs of trajectory segments and then optimizing a policy based on this learned reward function. This process runs asynchronously with three main components:

1.  **Policy Optimization:**
    *   A policy $\pi$ (parameterized by a deep neural network) interacts with the environment, generating trajectories.
    *   A traditional RL algorithm (e.g., A2C for Atari, TRPO for robotics) updates the policy parameters to maximize the sum of predicted rewards $r_t = \hat{r}(o_t, a_t)$.
    *   Rewards produced by $\hat{r}$ are normalized to have zero mean and constant standard deviation.
    *   An entropy bonus is added to encourage exploration, especially given the non-stationary nature of the learned reward function.

2.  **Preference Elicitation:**
    *   Pairs of trajectory segments $(\sigma^1, \sigma^2)$ are selected from the generated trajectories.
    *   These segments are presented to a human overseer as short video clips (1-2 seconds long).
    *   The human indicates a preference for $\sigma^1$, $\sigma^2$, or marks them as equally good or incomparable.
    *   These judgments are stored in a database $\mathcal{D}$ as triples $(\sigma^1, \sigma^2, \mu)$, where $\mu$ is a distribution over choices.

3.  **Reward Function Fitting:**
    *   A reward function estimate $\hat{r}$ (parameterized by a deep neural network) is trained via supervised learning to fit the human comparisons in $\mathcal{D}$.
    *   The training minimizes the cross-entropy loss between predicted preferences and human labels.
    *   **Modifications:**
        *   An ensemble of reward predictors is used, each trained on bootstrapped samples from $\mathcal{D}$. The final $\hat{r}$ is the average of independently normalized ensemble members.
        *   A fraction of data ($1/e$) is held out for validation, and $\ell_2$ regularization (with adaptive coefficient) and dropout are applied to prevent overfitting.
        *   A 10% chance of random human response is assumed in the preference prediction model to account for human error.

**Query Selection:**
*   Queries are selected by approximating uncertainty in the reward function estimator.
*   A large number of candidate segment pairs are sampled.
*   Each ensemble member predicts preferences for these pairs.
*   Pairs with the highest variance in predictions across ensemble members are selected for human comparison.

**Asynchronous Training:** The three processes run asynchronously: trajectories flow from policy optimization to preference elicitation, human comparisons flow to reward function fitting, and updated $\hat{r}$ parameters flow back to policy optimization.

### Key Formulas in LaTeX

1.  **Predicted Probability of Preference:**

$$
\hat{P}[\sigma^1 \succ \sigma^2] = \frac{\exp \sum \hat{r}(o_t^1, a_t^1)}{\exp \sum \hat{r}(o_t^1, a_t^1) + \exp \sum \hat{r}(o_t^2, a_t^2)}
$$

    This formula models the human's probability of preferring segment $\sigma^1$ over $\sigma^2$ based on the sum of predicted rewards over each segment, following the Bradley-Terry model.

2.  **Cross-Entropy Loss for Reward Function Fitting:**

$$
\text{loss}(\hat{r}) = - \sum_{(\sigma^1, \sigma^2, \mu) \in \mathcal{D}} \mu(1) \log \hat{P}[\sigma^1 \succ \sigma^2] + \mu(2) \log \hat{P}[\sigma^2 \succ \sigma^1]
$$

    This loss function is minimized to train the reward function $\hat{r}$, where $\mu(1)$ and $\mu(2)$ represent the human's preference for $\sigma^1$ and $\sigma^2$ respectively.

### Key Quantitative Results and Numbers

*   **Human Feedback Efficiency:** Achieved effective solutions with less than 1% of the agent's interactions with the environment.
*   **Human Time Investment:**
    *   For standard RL tasks (Atari, MuJoCo robotics): 15 minutes to 5 hours of human time (hundreds to thousands of comparisons). Contractors responded to queries in 3-5 seconds on average.
    *   For novel behaviors (e.g., backflip, standing on one leg): about an hour of human time (900 queries for backflip, 800 for one-legged standing, 1300 for Enduro "even mode").
*   **Performance vs. True Reward RL:**
    *   **MuJoCo Robotics:** With 700 human labels, the method nearly matched or slightly outperformed RL with true reward on most tasks (e.g., Ant, Hopper, Walker, Half-Cheetah, Reacher, Swimmer, Pendulum, Double-Pendulum). On Ant, human feedback significantly outperformed synthetic feedback, attributed to better reward shaping.
    *   **Atari Games:** With 5,500 human labels, the method showed substantial learning, matching or exceeding RL on some games (BeamRider, Pong). On Seaquest and Qbert, synthetic feedback eventually reached RL levels but learned slower. On SpaceInvaders and Breakout, it improved substantially but did not match RL. On Enduro, human feedback outperformed A3C by shaping the reward.
*   **Ablation Studies:**
    *   **Offline Reward Training:** Performed poorly, leading to "bizarre behavior" (e.g., Pong agent avoiding losing points but not scoring). This highlights the need for online feedback.
    *   **Single Frames vs. Clips:** Using single frames required significantly more comparisons to achieve the same results. Longer clips (1-2 seconds) were found to be more helpful per clip, as they provided more context and reduced initial evaluation time for humans.
*   **Computational Cost:** For Atari experiments, compute cost was ~$25 (16 CPUs, 1 Nvidia K80 GPU for ~1 day). 5,000 labels cost ~$36 at US minimum wage, suggesting compute cost is comparable to non-expert feedback cost.

### Stated Limitations

*   **Non-stationary Reward Function:** The learned reward function $\hat{r}$ can be non-stationary, requiring RL algorithms robust to such changes (e.g., policy gradient methods).
*   **Human Error and Inconsistency:** Real human feedback can be less effective than synthetic feedback due to human error, inconsistency between contractors, or uneven labeling rates.
*   **Query Selection Suboptimality:** The current query selection method (maximizing variance across ensemble members) is a "crude approximation" and in some tasks "impairs performance." The ideal approach would be based on expected value of information.
*   **Reward Function Complexity:** While the method generalizes to complex reward functions, the specific reward predictor architecture used for MuJoCo tasks (two-layer neural network) is simpler than what might be needed for more complex tasks.
*   **Discounting:** The preference prediction formula (Equation 1) does not use discounting, which implies indifference about when events happen in a segment. Explicit discounting or inferring human discount functions could be considered.
*   **Partial Observability:** In general partially observable environments, reward functions might depend on the entire sequence of observations, requiring recurrent neural networks, which was not explicitly explored for all environments.
*   **Comparison vs. Absolute Scores:** While comparisons were easier for humans, the paper notes that for continuous control tasks, predicting comparisons worked much better than predicting scores due to varying reward scales. In Atari, clipping rewards and predicting only the sign avoided these difficulties, but this is not suitable for continuous control.
