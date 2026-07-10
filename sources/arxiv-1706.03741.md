---
id: arxiv:1706.03741
type: paper
title: Deep reinforcement learning from human preferences
url: https://arxiv.org/abs/1706.03741
retrieved: '2026-07-10'
maturity: comprehensive
topic: ppo-for-llms
---

**Core Problem**
Deep reinforcement learning (RL) has scaled successfully in domains with well-specified reward functions, yet many real-world objectives are complex, poorly defined, or impossible to hand-engineer (e.g., robot locomotion or table cleaning). Directly using human feedback as a reward signal is prohibitively expensive because modern RL requires hundreds or thousands of hours of environmental interaction. The core challenge is to enable non-expert humans to communicate complex goals to RL agents economically, without requiring task demonstrations or explicit reward functions, while drastically reducing the volume of required feedback to make human oversight practically viable.

**Method/Recipe**
The algorithm learns a reward function from pairwise human preferences and simultaneously optimizes a policy to maximize the predicted reward. Training operates asynchronously through three sequential processes: (1) a policy network interacts with the environment to generate trajectory segments; (2) the policy parameters are updated via a standard RL algorithm (Advantage Actor-Critic for Atari, Trust Region Policy Optimization for robotics) to maximize the sum of predicted rewards from the reward model; (3) pairs of trajectory segments (1–2 second video clips) are presented to a human overseer for comparison. Human judgments are recorded as preference triples, and the reward network parameters are updated via supervised learning to fit the accumulated comparisons. To optimize feedback efficiency, the system selects query pairs by sampling many segments, using an ensemble of reward predictors to identify pairs with the highest prediction variance, approximating uncertainty. The reward model employs an ensemble of independently trained predictors, applies zero-mean normalization, uses a held-out validation set for adaptive regularization, and incorporates a 10% uniform response rate to account for constant human error.

**Key Formulas**
The preference elicitation follows the Bradley-Terry model, where the probability that a human prefers trajectory segment $\tau_1$ over $\tau_2$ is modeled as:
$$P(\tau_1 \succ \tau_2) = \frac{e^{R(\tau_1)}}{e^{R(\tau_1)} + e^{R(\tau_2)}}$$
The reward network is trained by minimizing the cross-entropy loss between these predicted probabilities and the actual human labels. This formulation equates reward differences with a preference ranking scale, analogous to the Elo system.

**Quantitative Results**
Experiments span eight simulated robotics tasks in MuJoCo and seven Atari games in the Arcade Learning Environment. The system requires feedback on less than 1% of the agent’s environmental interactions. Human contractors provided feedback in 3–5 seconds per query, requiring between 15 minutes and 5 hours of total human time. In robotics, 700 human queries allowed the agent to nearly match direct RL performance, while 1,400 synthetic queries slightly exceeded ground-truth RL, likely due to better reward shaping. On Atari tasks, 5,500 human queries were used. With synthetic feedback, 3,300 labels matched or approached RL performance on BeamRider and Pong, while Seaquest and Qbert approached RL levels but learned more slowly. SpaceInvaders and Breakout showed substantial improvement but never matched direct RL. Notably, human feedback on the Ant task significantly outperformed synthetic feedback due to explicit preference for upright posture.

**Stated Limitations**
The authors note several constraints. The uncertainty-based query selection is described as a "crude approximation," with ablation studies indicating it can occasionally impair performance. Learning from a learned reward function exhibits higher variance and lower stability compared to direct RL. The method does not assume the ability to reset the environment to arbitrary states, meaning comparison segments start from different states, which complicates preference interpretation. Furthermore, the algorithm's performance on certain Atari environments remains inferior to standard RL, and the 10% random response assumption is a simplified model of human error that may not capture all rating inconsistencies.
