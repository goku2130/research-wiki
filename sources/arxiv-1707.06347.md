---
id: arxiv:1707.06347
type: paper
title: Proximal Policy Optimization Algorithms
url: https://arxiv.org/abs/1707.06347
retrieved: '2026-07-11'
maturity: comprehensive
topic: policy-gradient-methods
---

**Core Problem**
Standard policy gradient methods suffer from poor data efficiency and robustness because performing multiple optimization steps on the same sampled data often triggers destructively large policy updates. Trust Region Policy Optimization (TRPO) stabilizes training via constrained optimization but requires complex second-order solvers and is incompatible with architectures using dropout or parameter sharing. Proximal Policy Optimization (PPO) addresses these limitations by introducing a first-order optimization framework that enables multiple epochs of minibatch updates while guaranteeing stable, monotonic policy improvement without second-order computations.

**Method and Recipe**
PPO alternates between environment interaction and surrogate objective optimization. The algorithm proceeds as follows:
1. **Data Collection:** $N$ parallel actors run the current policy $\pi_{\theta_{\text{old}}}$ in the environment for $T$ timesteps, collecting trajectories.
2. **Advantage Estimation:** Advantage values $\hat{A}_t$ are computed using a truncated Generalized Advantage Estimation (GAE) scheme: $\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \dots + (\gamma\lambda)^{T-t+1}\delta_{T-1}$, where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.
3. **Surrogate Construction:** The $NT$ collected samples are used to build a surrogate loss function.
4. **Optimization:** The surrogate loss is optimized via stochastic gradient ascent (typically Adam) for $K$ epochs using minibatches of size $M \leq NT$.
5. **Parameter Update:** The policy parameters are updated ($\theta_{\text{old}} \leftarrow \theta$), and the cycle repeats.

**Key Formulas**
The central innovation is the clipped surrogate objective, which constrains the probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ to prevent excessive policy shifts:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

This creates a pessimistic lower bound on the unclipped conservative policy iteration objective, removing incentives to move $r_t$ outside $[1-\epsilon, 1+\epsilon]$. When parameter sharing between policy and value networks is employed, the objective combines the clipped surrogate, a value function squared-error loss $L^{VF}$, and an entropy bonus $S$:

$$
L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]
$$

Alternatively, PPO supports an adaptive KL-penalized objective:

$$
L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t - \beta \operatorname{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)] \right]
$$

where $\beta$ is dynamically scaled to maintain a target KL divergence $d_{\text{targ}}$.

**Quantitative Results**
On seven continuous-control MuJoCo tasks trained for one million timesteps, the clipped variant with $\epsilon=0.2$ achieved an average normalized score of 0.82, outperforming unclipped baselines (-0.39), other $\epsilon$ values, and both fixed and adaptive KL penalties. PPO surpassed TRPO, Cross-Entropy Method, vanilla policy gradient, A2C, and A2C-TRPO across nearly all environments. In the Atari benchmark across 49 games, PPO demonstrated significantly better sample complexity than A2C, winning 30/49 games on average training reward and 19/49 on final performance, compared to A2C’s 1 win in each metric. PPO also successfully learned complex 3D humanoid locomotion, steering, and recovery tasks.

**Stated Limitations**
The authors explicitly note that the KL-penalty variant consistently underperformed the clipped objective across all tested configurations. Additionally, while the adaptive KL mechanism reduces sensitivity to hyperparameter choices, the algorithm still requires tuning of $\epsilon$, $d_{\text{targ}}$, and learning rates. The heuristic constants used to adjust $\beta$ (1.5 and 2) are noted as non-critical but remain necessary for the adaptive mechanism to function. PPO thus trades minor hyperparameter sensitivity for substantially improved simplicity, architectural compatibility, and sample efficiency relative to TRPO.
