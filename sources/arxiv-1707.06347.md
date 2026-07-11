---
id: arxiv:1707.06347
type: paper
title: Proximal Policy Optimization Algorithms
url: https://arxiv.org/abs/1707.06347
retrieved: '2026-07-11'
maturity: comprehensive
topic: async-and-off-policy-rl
---

The core problem addressed is the instability of standard policy gradient methods when performing multiple optimization steps on the same sampled data, which frequently causes destructive policy updates. While Trust Region Policy Optimization (TRPO) mitigates this via constrained optimization, it requires complex second-order solvers and is incompatible with architectures employing dropout or parameter sharing. Proximal Policy Optimization (PPO) resolves these issues by introducing a first-order optimization framework that enables safe, multi-epoch minibatch updates through a clipped surrogate objective.

The PPO algorithm follows a strict iterative recipe: (1) $N$ parallel actors execute the current policy $\pi_{\theta_{\text{old}}}$ for $T$ timesteps to collect trajectories; (2) advantage estimates $\hat{A}_t$ are computed using Generalized Advantage Estimation (GAE) or truncated variants; (3) the surrogate loss is constructed by clipping the probability ratio $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ to restrict policy updates; (4) stochastic gradient ascent (typically Adam) maximizes this loss over $K$ epochs using minibatches of size $M \leq NT$; (5) $\theta_{\text{old}}$ is updated to $\theta$ for the subsequent iteration. The primary clipped objective is defined as:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right) \right].
$$

This formulation creates a pessimistic lower bound on the unclipped surrogate objective, removing incentives to push $r_t(\theta)$ outside $[1-\epsilon, 1+\epsilon]$. As an alternative, PPO supports an adaptive KL-penalty objective:

$$
L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t - \beta \text{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)] \right],
$$

where $\beta$ is dynamically adjusted to maintain a target KL divergence $d_{\text{targ}}$. When policies and value functions share parameters, the combined objective incorporates a value function loss $L^{VF}$ and entropy bonus $S$:

$$
L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right].
$$

Quantitative evaluations demonstrate PPO’s efficacy across continuous and discrete domains. On seven MuJoCo continuous control tasks trained for one million timesteps, the clipped objective with $\epsilon=0.2$ achieved the highest average normalized score of $0.82$, significantly outperforming the unclipped baseline ($-0.39$) and adaptive KL variants ($0.68$–$0.74$). PPO consistently surpassed TRPO, A2C, and cross-entropy methods across these environments. In the Arcade Learning Environment across 49 Atari games, PPO won $30$ of $49$ games based on average episode reward over the training period and $19$ of $49$ based on the final $100$ episodes, demonstrating superior sample complexity compared to A2C and performance comparable to ACER with substantially simpler implementation. Standard hyperparameters included $\gamma=0.99$, $\lambda=0.95$, $T=2048$ (continuous) or $T=128$ (Atari), and $K=10$ or $K=3$ epochs, respectively.

The authors explicitly note several limitations. The KL-penalty variant consistently underperformed the clipped surrogate objective in empirical tests. Tuning the penalty coefficient $\beta$ or target KL $d_{\text{targ}}$ remains necessary, and the adaptive scheme occasionally yields policy updates that deviate significantly from $d_{\text{targ}}$, though such occurrences are rare. Additionally, clipping in log space was evaluated but yielded no performance gains. While PPO improves robustness and data efficiency over vanilla policy gradients, it still requires careful selection of $\epsilon$ and other hyperparameters, and its theoretical guarantees rely on the clipped objective serving as a valid lower bound under the assumed Markovian setting.
