---
id: arxiv:1706.03741
type: paper
title: Deep Reinforcement Learning from Human Preferences
url: https://arxiv.org/abs/1706.03741
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlaif
---

**Core Problem**
Deep reinforcement learning (RL) has scaled successfully in domains with well-specified reward functions, but many real-world objectives are complex, poorly-defined, or impossible to hand-craft programmatically. Direct human feedback as a reward signal is prohibitively expensive, as modern RL agents require hundreds or thousands of hours of environmental interaction. The authors address the reward specification bottleneck by developing a framework that learns complex behaviors from non-expert human preferences over trajectory segments, reducing feedback requirements by several orders of magnitude while maintaining practical applicability to state-of-the-art RL systems.

**Method/Recipe Step-by-Step**
The algorithm maintains a policy $\pi$ and a reward predictor $\hat{r}$, both parameterized by deep neural networks, and operates through an asynchronous three-process loop:
1. **Trajectory Generation:** The policy $\pi$ interacts with the environment to produce trajectory segments, visualized as 1–2 second video clips.
2. **Preference Elicitation:** A human overseer compares pairs of clips, outputting a preference distribution $\mu$ over the two segments (including ties or "incomparable" responses).
3. **Reward Fitting:** The parameters of $\hat{r}$ are optimized via supervised learning to fit the accumulated preference database $\mathcal{D}$.
4. **Policy Optimization:** $\pi$ is updated using RL (A2C for Atari, TRPO for robotics) to maximize the sum of predicted rewards $\hat{r}(o_t, a_t)$. Predicted rewards are normalized to zero mean and constant standard deviation.
5. **Query Selection:** Candidate clip pairs are sampled, and the system evaluates ensemble disagreement. Pairs with the highest variance across predictors are selected for human comparison.
6. **Robustness Modifications:** The reward predictor uses an ensemble of networks trained on bootstrapped subsets, $\ell_2$ regularization, dropout, and a 10% uniform random response rate to account for human error. Processes run asynchronously, with labels fed at a decaying rate to balance initial predictor quality with distributional adaptation.

**Key Formulas**
The preference modeling follows a Bradley-Terry formulation, where the probability of preferring segment $\sigma^1$ over $\sigma^2$ is:
$$\hat{P}[\sigma^1 \succ \sigma^2] = \frac{\exp \sum \hat{r}(o_t^1, a_t^1)}{\exp \sum \hat{r}(o_t^1, a_t^1) + \exp \sum \hat{r}(o_t^2, a_t^2)}.$$
The reward predictor minimizes the corresponding cross-entropy loss:
$$\text{loss}(\hat{r}) = - \sum_{(\sigma^1, \sigma^2, \mu) \in \mathcal{D}} \mu(1) \log \hat{P}[\sigma^1 \succ \sigma^2] + \mu(2) \log \hat{P}[\sigma^2 \succ \sigma^1].$$

**Key Quantitative Results**
The approach requires less than 1% of the agent's total environmental interactions. In MuJoCo robotics tasks, 700 human queries suffice to nearly match standard RL performance, while 1,400 synthetic queries slightly exceed true-reward RL on several tasks. For Atari games, 5,500 queries (roughly 5 hours of human labor at minimum wage) enable performance matching or exceeding RL on BeamRider and Pong, with substantial improvement on others. Real human feedback generally performs comparably to synthetic feedback, though labeling inconsistency and uneven state-space coverage can introduce noise. Novel, complex behaviors—such as a Hopper robot performing backflips (900 queries, <1 hour) and a Half-Cheetah walking on one leg (800 queries, <1 hour)—were successfully learned without hand-engineered rewards. Compute costs for training are approximately $25 per day on standard hardware, making human labor the dominant expense.

**Stated Limitations**
Offline reward training fails catastrophically due to non-stationarity; without interleaved human feedback, the predictor captures only partial reward structure, leading to pathological behaviors like avoiding negative outcomes without pursuing positive ones. Query selection via ensemble variance is acknowledged as a crude approximation, with expected information gain being theoretically superior but unexplored. Human feedback introduces noise, ties, and "can't tell" responses, particularly in visually complex or context-dependent games like Qbert. Finally, the authors note that compute costs for training these systems are already comparable to non-expert human labor, suggesting diminishing returns on further sample-complexity optimizations.
