---
id: arxiv:1707.06347
type: paper
title: Proximal Policy Optimization Algorithms
url: https://arxiv.org/abs/1707.06347
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlhf-ppo-pipeline
---

**Core Problem**
Existing reinforcement learning paradigms exhibit critical trade-offs: deep Q-learning fails on continuous control tasks, vanilla policy gradient methods suffer from poor data efficiency and robustness, and trust region policy optimization (TRPO) guarantees stability but requires complex second-order optimization and is incompatible with architectures employing dropout or parameter sharing. Proximal Policy Optimization (PPO) is introduced to deliver TRPO’s monotonic improvement guarantees using only first-order stochastic gradient ascent, enabling multiple epochs of minibatch updates while remaining computationally simple, broadly applicable, and empirically more sample-efficient.

**Methodology & Algorithmic Recipe**
PPO alternates between environment interaction and surrogate objective optimization. In each iteration, $N$ parallel actors execute the current policy $\pi_{\theta_{\text{old}}}$ for $T$ timesteps to collect trajectory segments. Advantage estimates $\hat{A}_t$ are computed using Generalized Advantage Estimation (GAE). The aggregated $NT$ samples are then used to optimize a surrogate objective for $K$ epochs via minibatch stochastic gradient descent (typically Adam). After optimization, $\theta_{\text{old}}$ is updated to the new $\theta$. The training loop combines the policy surrogate with a value function error term and an entropy bonus to stabilize learning and encourage exploration.

**Key Formulations**
Let $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ denote the probability ratio. The conservative policy iteration objective $L^{CPI}(\theta) = \hat{\mathbb{E}}_t[r_t(\theta)\hat{A}_t]$ is modified by clipping the ratio within $[1-\epsilon, 1+\epsilon]$ to prevent destructive updates. The primary clipped surrogate objective is:
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta)\hat{A}_t, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
This formulation acts as a pessimistic lower bound on the unclipped objective, removing the incentive to shift $r_t$ beyond the clip interval when it would artificially improve the objective. An alternative KL-penalized variant optimizes:
$$L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t \left[ r_t(\theta)\hat{A}_t - \beta \text{KL}[\pi_{\theta_{\text{old}}}, \pi_\theta] \right]$$
with an adaptively adjusted penalty coefficient $\beta$ targeting a specific KL divergence $d_{\text{targ}}$. The full combined objective is:
$$L^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]$$
where $L_t^{VF}$ is a squared-error value loss and $S$ is an entropy bonus. Advantages are estimated via $\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \dots + (\gamma\lambda)^{T-t+1}\delta_{T-1}$ with $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

**Quantitative Results**
On seven MuJoCo continuous control tasks over one million timesteps, PPO-Clip with $\epsilon=0.2$ achieved an average normalized score of 0.82, outperforming unclipped baselines (-0.39), fixed/adaptive KL penalties, TRPO, cross-entropy methods, and A2C variants. In the Arcade Learning Environment across 49 Atari games, PPO achieved superior sample complexity compared to A2C and matched ACER’s performance while requiring significantly less architectural complexity. PPO also successfully learned high-dimensional humanoid locomotion, including running, steering, and obstacle recovery. Standard hyperparameters include $\epsilon=0.2$, horizon $T=2048$, $K=10$ epochs, Adam stepsize $3\times10^{-4}$, $\gamma=0.99$, and $\lambda=0.95$.

**Stated Limitations**
The authors explicitly note that the KL-penalized variant consistently underperformed the clipped objective in their experiments. Tuning the penalty coefficient $\beta$ or target KL $d_{\text{targ}}$ remains necessary, though the adaptive scheme mitigates sensitivity. The heuristic factors (1.5 and 2) used to adjust $\beta$ occasionally yield updates diverging from $d_{targ}$, albeit rarely. Furthermore, without clipping or penalties, multiple optimization epochs on identical data provably induce destructively large policy updates, underscoring the necessity of the proposed constraints. The method still requires careful hyperparameter selection across different problem domains, though it demonstrates greater robustness than vanilla policy gradients.
