---
id: arxiv:1707.06347
type: paper
title: Proximal Policy Optimization Algorithms
url: https://arxiv.org/abs/1707.06347
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

The core problem addressed by Proximal Policy Optimization (PPO) is the instability and poor data efficiency of standard policy gradient methods when performing multiple optimization epochs on the same trajectory, which often yields destructively large policy updates. While Trust Region Policy Optimization (TRPO) ensures monotonic improvement through second-order optimization and hard constraints, it is computationally complex, incompatible with architectures featuring dropout or parameter sharing, and difficult to scale. PPO seeks to replicate TRPO’s stability using only first-order optimization, enabling multiple minibatch epochs while remaining simple to implement and broadly applicable.

The PPO algorithm operates by alternating between data collection and surrogate objective optimization. In each iteration, $N$ parallel actors execute the current policy $\pi_{\theta_{\text{old}}}$ for $T$ timesteps to collect trajectories. Advantage estimates $\hat{A}_t$ are computed using Generalized Advantage Estimation (GAE). The algorithm then constructs a combined surrogate loss and optimizes it via stochastic gradient ascent (typically Adam) for $K$ epochs using minibatches of size $M \leq NT$. After optimization, $\theta_{\text{old}}$ is updated to the new parameters. The surrogate objective employs a clipped probability ratio to penalize excessive policy changes. Alternatively, a KL-penalized objective with an adaptive coefficient can be used, though the clipped variant is preferred.

The foundational policy gradient estimator is $\hat{g} = \hat{\mathbb{E}}_t [\nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t]$, derived from $L^{PG}(\theta) = \hat{\mathbb{E}}_t [\log \pi_\theta(a_t \mid s_t) \hat{A}_t]$. PPO modifies the conservative policy iteration objective $L^{CPI}(\theta) = \hat{\mathbb{E}}_t [r_t(\theta) \hat{A}_t]$, where $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$. The primary clipped surrogate objective is:
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta)\hat{A}_t, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
where $\epsilon$ is a clipping hyperparameter. This formulation creates a pessimistic lower bound that removes the incentive to push $r_t(\theta)$ outside $[1-\epsilon, 1+\epsilon]$. The full training objective combines the policy surrogate with a value function loss $L_t^{VF}(\theta) = (V_\theta(s_t) - V_t^{\text{targ}})^2$ and an entropy bonus $S[\pi_\theta](s_t)$:
$$L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]$$
Advantages are estimated via truncated GAE: $\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \dots + (\gamma\lambda)^{T-t+1}\delta_{T-1}$, with $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

Quantitative evaluations demonstrate PPO’s efficacy. On seven continuous control MuJoCo tasks trained for one million timesteps, the clipped objective with $\epsilon=0.2$ achieved the highest average normalized score of 0.82 across 21 runs, outperforming TRPO, cross-entropy methods, and A2C variants. On the Arcade Learning Environment across 49 games, PPO demonstrated superior sample complexity, winning 30 out of 49 games on average training reward and 19 on final performance metrics, significantly surpassing A2C and matching ACER with simpler implementation. Standard hyperparameters include $T=2048$, $K=10$, $M=64$, $\gamma=0.99$, and $\lambda=0.95$.

The authors note several limitations and practical considerations. The KL-penalty variant, which adapts $\beta$ to maintain a target KL divergence $d_{\text{targ}}$, consistently underperformed the clipped objective. Selecting a fixed penalty coefficient $\beta$ is notoriously difficult across different problems or as learning dynamics shift. Furthermore, while PPO is robust, it still requires careful hyperparameter tuning (e.g., $\epsilon$ or $d_{\text{targ}}$), and the method's stability relies on the clipping mechanism to prevent destructive updates when multiple epochs are applied.
