---
id: arxiv:1706.03741
type: paper
title: Deep Reinforcement Learning from Human Preferences
url: https://arxiv.org/abs/1706.03741
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlhf-ppo-pipeline
---

**Core Problem**
Deep reinforcement learning (RL) traditionally requires well-specified reward functions, which are difficult or impossible to hand-craft for complex, poorly-defined, or non-demonstrable real-world tasks. Directly using human feedback as a reward signal is computationally prohibitive, as modern RL systems demand hundreds or thousands of hours of environmental interaction. The core challenge is to economically scale human feedback to deep RL by learning a reward function from pairwise preferences, drastically reducing the human oversight burden while maintaining performance.

**Method/Recipe Step by Step**
The algorithm maintains a policy network $\pi$ and a reward predictor network $\hat{r}$, updated asynchronously through three interleaved processes:
1. **Policy Rollout:** $\pi$ interacts with the environment to generate trajectory segments $\sigma$.
2. **Preference Elicitation:** Pairs of short video clips (1–2 seconds) are presented to a human overseer. The human selects the preferred segment, indicates a tie, or marks the pair as incomparable. Judgments are stored as triples $(\sigma^1, \sigma^2, \mu)$ in a database $\mathcal{D}$, where $\mu$ is a distribution over $\{1, 2\}$.
3. **Reward Model Fitting:** $\hat{r}$ is updated via supervised learning to minimize prediction error against the collected preferences.
4. **Policy Optimization:** $\pi$ is updated via RL to maximize the cumulative predicted reward from $\hat{r}$.

Implementation details include using an ensemble of reward predictors trained with $\ell_2$ regularization and dropout, assuming a 10% uniform response noise rate to model human error. Query selection approximates reward uncertainty by sampling candidate segment pairs and selecting those with the highest prediction variance across ensemble members. Policy optimization employs Advantage Actor-Critic (A2C) for Atari environments and Trust Region Policy Optimization (TRPO) for simulated robotics, with rewards normalized to zero mean and unit standard deviation.

**Key Formulas**
The framework models human preferences using the Bradley-Terry model. The predicted probability that segment $\sigma^1$ is preferred over $\sigma^2$ is:

$$
\hat{P}[\sigma^1 \succ \sigma^2] = \frac{\exp \sum \hat{r}(o_t^1, a_t^1)}{\exp \sum \hat{r}(o_t^1, a_t^1) + \exp \sum \hat{r}(o_t^2, a_t^2)}.
$$

The reward predictor is trained by minimizing the cross-entropy loss against human labels:

$$
\text{loss}(\hat{r}) = - \sum_{(\sigma^1, \sigma^2, \mu) \in \mathcal{D}} \mu(1) \log \hat{P}[\sigma^1 \succ \sigma^2] + \mu(2) \log \hat{P}[\sigma^2 \succ \sigma^1].
$$

**Key Quantitative Results and Numbers**
The approach reduces human oversight requirements to less than 1% of the agent's interactions, cutting oversight costs by approximately three orders of magnitude. On eight MuJoCo simulated robotics tasks, 700 human labels nearly match full RL performance with true rewards, while 1,400 labels occasionally exceed them due to superior reward shaping. In seven Atari games, 5,500 human labels match or surpass standard RL baselines on BeamRider, Pong, and Enduro. The method successfully trains novel, complex behaviors requiring only about one hour of human time: a Hopper robot performing continuous backflips (900 queries), a Half-Cheetah walking on one leg (800 queries), and an Enduro agent maintaining pace with traffic (~1,300 queries). Total experimental costs are minimal, with compute averaging ~$25 per day and human labor ~$36 for 5,000 labels.

**Stated Limitations**
Offline training of the reward predictor fails due to the non-stationarity of the agent's occupancy distribution, causing the model to learn only partial reward structures that lead to degenerate behaviors (e.g., avoiding losses without scoring in Pong). Human feedback introduces noise and inconsistency, particularly when contractors provide uneven labeling rates or struggle with ambiguous short clips. The active query selection strategy relies on a crude ensemble variance approximation rather than the expected value of information, which ablation studies show can sometimes impair performance. Finally, the authors note diminishing returns on further sample-complexity reductions, as the computational cost of training now rivals the economic cost of non-expert human feedback.
