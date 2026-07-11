---
id: arxiv:1707.06347
type: paper
title: Proximal Policy Optimization Algorithms (Schulman et al. 2017)
url: https://arxiv.org/abs/1707.06347
retrieved: '2026-07-11'
maturity: comprehensive
topic: distributed-rl-training
---

# Proximal Policy Optimization (PPO)

### Core Problem
Standard policy gradient methods often suffer from poor data efficiency and instability; specifically, performing multiple steps of optimization on a single batch of data typically leads to destructively large policy updates. While Trust Region Policy Optimization (TRPO) addresses this by constraining the policy update size, it is computationally complex, relies on second-order optimization, and is incompatible with architectures involving parameter sharing or noise (e.g., dropout). The goal of PPO is to achieve the stability and reliability of TRPO using only first-order optimization, making it simpler to implement and more scalable.

### Method
PPO alternates between sampling data through environment interaction and optimizing a "surrogate" objective function using stochastic gradient ascent. The authors propose two main variants: a clipped surrogate objective and an adaptive KL penalty.

#### 1. Clipped Surrogate Objective
The primary contribution is the $L^{CLIP}$ objective, which penalizes changes that move the policy too far from the previous iteration. Let the probability ratio be defined as:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

The clipped objective is formulated as:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min (r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t) \right]
$$

where $\epsilon$ is a hyperparameter (e.g., $0.2$) and $\hat{A}_t$ is the advantage estimator. This objective creates a pessimistic lower bound on the performance; it removes the incentive to move $r_t$ outside the interval $[1-\epsilon, 1+\epsilon]$ when the change would improve the objective, but still accounts for changes that make the objective worse.

#### 2. Adaptive KL Penalty
As an alternative, PPO can use a KL divergence penalty with a coefficient $\beta$ that is adjusted dynamically to maintain a target KL divergence $d_{\text{targ}}$:

$$
L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t - \beta \text{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)] \right]
$$

If the measured KL divergence $d < d_{\text{targ}}/1.5$, $\beta$ is halved; if $d > d_{\text{targ}} \times 1.5$, $\beta$ is doubled.

#### 3. Combined Objective for Actor-Critic
For architectures sharing parameters between the policy and value function, the objective is augmented with a squared-error value function loss $L_t^{VF}$ and an entropy bonus $S$ to encourage exploration:

$$
L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]
$$

### Implementation Recipe
1. **Data Collection:** $N$ parallel actors run the current policy $\pi_{\theta_{\text{old}}}$ for $T$ timesteps.
2. **Advantage Estimation:** Compute advantage estimates $\hat{A}_t$ (e.g., using Generalized Advantage Estimation).
3. **Optimization:** Optimize the surrogate loss $L$ with respect to $\theta$ using minibatch SGD or Adam for $K$ epochs.
4. **Update:** Set $\theta_{\text{old}} \leftarrow \theta$ and repeat.

### Key Quantitative Results
*   **Continuous Control (MuJoCo):** In a benchmark of 7 simulated robotics tasks, the clipped objective with $\epsilon=0.2$ achieved the highest average normalized score of **0.82**. In contrast, the "no clipping or penalty" setting resulted in a negative score (**-0.39**), and adaptive KL achieved between **0.68 and 0.74**.
*   **Atari Games:** PPO was compared against A2C and ACER across 49 games.
    *   **Average reward over all training:** PPO "won" **30** games, ACER won 18, and A2C won 1.
    *   **Average reward over last 100 episodes:** ACER won **28** games, PPO won 19, and A2C won 1.
*   **General Performance:** PPO outperformed TRPO, CEM, and A2C on nearly all continuous control environments tested.

### Limitations
The authors note that the adaptive KL penalty variant performed worse than the clipped surrogate objective in their experiments. Additionally, they highlight that choosing a fixed penalty coefficient $\beta$ for the KL objective is difficult because the optimal value can change across different problems or even during the course of learning within a single problem.
