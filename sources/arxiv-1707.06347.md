---
id: arxiv:1707.06347
type: paper
title: Proximal Policy Optimization Algorithms
url: https://arxiv.org/abs/1707.06347
retrieved: '2026-07-11'
maturity: comprehensive
topic: entropy-and-exploration
---

**Core Problem**
Standard policy gradient methods suffer from poor data efficiency and robustness, particularly when performing multiple optimization steps on the same trajectory data, which empirically leads to destructively large policy updates. While Trust Region Policy Optimization (TRPO) guarantees monotonic improvement, it requires complex second-order optimization, is computationally intensive, and is incompatible with neural architectures employing dropout or parameter sharing between policy and value functions. The field requires a scalable, first-order algorithm that retains TRPO’s stability and data efficiency while remaining simple to implement and broadly applicable.

**Method & Algorithmic Recipe**
Proximal Policy Optimization (PPO) addresses this by alternating between data collection and optimization using a novel surrogate objective that permits multiple epochs of minibatch updates. The algorithmic recipe proceeds as follows: (1) $N$ parallel actors execute the current policy $\pi_{\theta_{\text{old}}}$ in the environment to collect $T$ timesteps of data; (2) advantage estimates are computed using truncated Generalized Advantage Estimation (GAE); (3) the surrogate loss is optimized via stochastic gradient ascent (typically Adam) for $K$ epochs using a minibatch size $M \leq NT$; (4) the optimized parameters replace $\theta_{\text{old}}$ for the next iteration. PPO implements two primary surrogate objectives: a clipped probability ratio variant (PPO-Clip) and a KL-divergence penalty variant (PPO-Penalty).

**Key Formulations**
The policy gradient estimator is defined as $\hat{g} = \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]$, derived from the standard objective $L^{PG}(\theta) = \hat{\mathbb{E}}_t \left[ \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]$. PPO replaces this with a clipped surrogate objective:
\[
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
\]
where $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ and $\epsilon$ constrains the ratio. The penalty variant optimizes:
\[
L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t - \beta \operatorname{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)] \right]
\]
with an adaptive coefficient $\beta$ that scales by factors of $1.5$ and $2$ to track a target KL divergence $d_{\text{targ}}$. When policy and value networks share parameters, the combined objective incorporates a value function error $L_t^{VF}$ and entropy bonus $S$:
\[
L^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]
\]
Advantage estimates utilize truncated GAE: $\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \dots + (\gamma\lambda)^{T-t+1}\delta_{T-1}$, where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

**Quantitative Results**
On seven continuous control MuJoCo environments over one million timesteps, PPO-Clip with $\epsilon=0.2$ achieved the highest average normalized score of $0.82$, outperforming ablated objectives and established methods like TRPO, CEM, and A2C. Standard hyperparameters included $T=2048$, $K=10$, $M=64$, $\gamma=0.99$, $\lambda=0.95$, and an Adam stepsize of $3 \times 10^{-4}$. In the Atari benchmark across 49 games, PPO won 30 out of 49 games on average training reward and 19 on final performance, significantly surpassing A2C (1 win each) and matching ACER’s sample complexity while requiring substantially simpler implementation.

**Stated Limitations**
The authors explicitly note that the KL-penalty variant consistently underperformed the clipped surrogate objective across experiments. Tuning the penalty coefficient $\beta$ or target KL $d_{\text{targ}}$ remains necessary, and the adaptive adjustment relies on heuristic scaling factors that, while robust, require manual specification. Fixed $\beta$ values are difficult to generalize across tasks or even within a single task as learning dynamics shift. Additionally, the authors report that clipping probability ratios in log space yielded no performance improvement over linear clipping. Despite these constraints, PPO strikes a favorable balance between sample complexity, implementation simplicity, and wall-time efficiency compared to prior online policy gradient methods.
