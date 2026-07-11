---
id: arxiv:1707.06347
type: paper
title: Proximal Policy Optimization Algorithms
url: https://arxiv.org/abs/1707.06347
retrieved: '2026-07-11'
maturity: comprehensive
topic: ppo-for-llms
---

The paper introduces Proximal Policy Optimization (PPO), a new family of policy gradient methods designed to be simpler to implement, more general, and have better sample complexity than existing methods like Trust Region Policy Optimization (TRPO).

**Core Problem:**
The core problem PPO addresses is the challenge of developing a scalable, data-efficient, and robust reinforcement learning algorithm using neural network function approximators. Existing methods have limitations: deep Q-learning struggles with continuous control and is poorly understood, vanilla policy gradient methods suffer from poor data efficiency and robustness, and TRPO is complex, incompatible with noise-inducing architectures (e.g., dropout), and cannot share parameters between policy and value functions or auxiliary tasks. A key issue with standard policy gradient methods is that performing multiple optimization steps on the same data can lead to destructively large policy updates.

**Method/Recipe Step by Step:**
PPO alternates between sampling data from the environment and optimizing a "surrogate" objective function using stochastic gradient ascent. Unlike standard policy gradient methods that perform one gradient update per data sample, PPO enables multiple epochs of minibatch updates.

The general PPO algorithm (Actor-Critic Style) proceeds as follows:
1.  **Initialization:** Initialize policy parameters $\theta$.
2.  **Loop for iterations:**
    a.  **Data Collection:** For each of N parallel actors, run the current policy $\pi_{\theta_{\text{old}}}$ in the environment for T timesteps.
    b.  **Advantage Estimation:** Compute advantage estimates $\hat{A}_1, \ldots, \hat{A}_T$ for the collected data. This often uses Generalized Advantage Estimation (GAE) or similar methods.
    c.  **Optimization:** Optimize the surrogate loss $L$ with respect to $\theta$ using K epochs of minibatch SGD (or Adam) with a minibatch size $M \leq NT$.
    d.  **Policy Update:** Set $\theta_{\text{old}} \leftarrow \theta$.

**Key Formulas:**

1.  **Policy Gradient Estimator (Standard):**
    $\hat{g} = \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]$

2.  **Policy Gradient Objective (Standard):**
    $L^{PG}(\theta) = \hat{\mathbb{E}}_t \left[ \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]$

3.  **TRPO Surrogate Objective with Constraint:**
    $\underset{\theta}{\text{maximize}} \quad \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right]$
    $\text{subject to} \quad \hat{\mathbb{E}}_t [\text{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)]] \leq \delta.$

4.  **TRPO Surrogate Objective with Penalty:**
    $\underset{\theta}{\text{maximize}} \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t - \beta \text{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)] \right]$

5.  **PPO Clipped Surrogate Objective ($L^{CLIP}$):**
    Let $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}$.
    $L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min (r_t(\theta) \hat{A}_t, \text{clip} (r_t(\theta), 1 - \epsilon , 1 + \epsilon) \hat{A}_t) \right]$
    where $\epsilon$ is a hyperparameter (e.g., 0.2). This objective forms a pessimistic estimate (lower bound) of the policy's performance.

6.  **PPO KL-Penalized Objective ($L^{KLPEN}$):**
    $L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)} \hat{A}_t - \beta \text{KL} [ \pi_{\theta_{\mathrm{old}}} (\cdot \mid s_t), \pi_\theta (\cdot \mid s_t) ] \right]$
    The coefficient $\beta$ is adaptively adjusted:
    - If $d < d_{\mathrm{targ}} / 1.5$, $\beta \leftarrow \beta / 2$
    - If $d > d_{\mathrm{targ}} \times 1.5$, $\beta \leftarrow \beta \times 2$
    where $d = \hat{\mathbb{E}}_t[\mathrm{KL}[\pi_{\theta_{\mathrm{old}}}(\cdot |s_t),\pi_\theta (\cdot |s_t)]]$ and $d_{\mathrm{targ}}$ is a target KL divergence.

7.  **Full PPO Objective (with Value Function and Entropy Bonus):**
    $L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S [ \pi_\theta ] (s_t) \right]$
    where $c_1, c_2$ are coefficients, $S$ is an entropy bonus, and $L_t^{VF}$ is a squared-error loss $(V_{\theta}(s_t) - V_t^{\mathrm{targ}})^2$.

8.  **Generalized Advantage Estimation (GAE) for $\hat{A}_t$:**
    $\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + \dots + (\gamma \lambda)^{T-t+1} \delta_{T-1}$
    where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

**Key Quantitative Results and Numbers:**

*   **Continuous Control (MuJoCo) - Surrogate Objective Comparison (Table 1):**
    *   No clipping or penalty: -0.39 (avg. normalized score)
    *   Clipping, $\epsilon = 0.1$: 0.76
    *   Clipping, $\epsilon = 0.2$: **0.82** (best performing clipping variant)
    *   Clipping, $\epsilon = 0.3$: 0.70
    *   Adaptive KL $d_{\text{targ}} = 0.003$: 0.68
    *   Adaptive KL $d_{\text{targ}} = 0.01$: 0.74
    *   Adaptive KL $d_{\text{targ}} = 0.03$: 0.71
    *   Fixed KL, $\beta = 0.3$: 0.62
    *   Fixed KL, $\beta = 1$: 0.71
    *   Fixed KL, $\beta = 3$: 0.72
    *   Fixed KL, $\beta = 10$: 0.69
    *   The clipped surrogate objective with $\epsilon=0.2$ performed best on this benchmark.

*   **Continuous Control (MuJoCo) - Algorithm Comparison (Figure 3):** PPO generally outperformed TRPO, CEM, vanilla policy gradient with adaptive stepsize, A2C, and A2C with trust region across 7 simulated robotics tasks (HalfCheetah, Hopper, InvertedDoublePendulum, InvertedPendulum, Reacher, Swimmer, Walker2d).

*   **Atari Games - Algorithm Comparison (Table 2):**
    *   **Avg. episode reward over all training:** PPO won 30 games, ACER won 18, A2C won 1.
    *   **Avg. episode reward over last 100 episodes:** ACER won 28 games, PPO won 19, A2C won 1, with 1 tie.
    *   PPO showed significantly better sample complexity than A2C and similar performance to ACER, despite being simpler.

**Stated Limitations:**
*   The paper notes that while the theory justifying TRPO suggests using a penalty instead of a constraint, it is hard to choose a single value of $\beta$ that performs well across different problems or even within a single problem as characteristics change. Simply using a fixed $\beta$ with SGD is insufficient.
*   The adaptive KL penalty method performed worse than the clipped surrogate objective in their experiments.
*   The parameters for adapting $\beta$ (1.5 and 2) are chosen heuristically, though the algorithm is not very sensitive to them. The initial value of $\beta$ is also a hyperparameter but not critical in practice due to rapid adjustment.
*   The experiments for comparing surrogate objectives did not share parameters between the policy and value function and did not use an entropy bonus, which might affect the generalizability of these specific comparisons.
