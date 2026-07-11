---
id: arxiv:1707.06347
type: paper
title: Proximal Policy Optimization Algorithms
url: https://arxiv.org/abs/1707.06347
retrieved: '2026-07-11'
maturity: comprehensive
topic: ppo-for-llms
---

# Proximal Policy Optimization (PPO) Algorithms

### Core Problem
Standard policy gradient methods often suffer from poor data efficiency and instability; specifically, performing multiple epochs of optimization on a single batch of data typically leads to destructively large policy updates. While Trust Region Policy Optimization (TRPO) addresses this by constraining the policy update size, it is computationally complex, relies on second-order optimization, and is incompatible with architectures utilizing parameter sharing or noise (e.g., dropout). PPO seeks to achieve the stability and reliability of TRPO using only first-order optimization.

### Method and Recipe
PPO introduces a "surrogate" objective function that penalizes changes to the policy that move the probability ratio too far from 1. 

**1. Probability Ratio**
The algorithm defines the probability ratio $r_t(\theta)$ as:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

**2. Clipped Surrogate Objective**
To prevent excessively large updates, PPO uses a clipped objective $L^{CLIP}(\theta)$:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min (r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t) \right]
$$

where $\epsilon$ is a hyperparameter (e.g., $0.2$) and $\hat{A}_t$ is the advantage estimator. This objective forms a pessimistic lower bound on the unclipped objective, removing the incentive to move $r_t$ outside the interval $[1-\epsilon, 1+\epsilon]$.

**3. Combined Actor-Critic Objective**
For architectures sharing parameters between the policy and value function, PPO maximizes a combined objective:

$$
L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]
$$

where $L_t^{VF}$ is the squared-error loss $(V_\theta(s_t) - V_t^{\text{targ}})^2$, $S$ is an entropy bonus for exploration, and $c_1, c_2$ are coefficients.

**4. Step-by-Step Recipe**
The PPO actor-critic algorithm proceeds as follows:
1. **Data Collection:** $N$ parallel actors run the current policy $\pi_{\theta_{\text{old}}}$ for $T$ timesteps.
2. **Advantage Estimation:** Compute advantage estimates $\hat{A}_1, \dots, \hat{A}_T$ (typically using truncated Generalized Advantage Estimation).
3. **Optimization:** Optimize the surrogate objective $L$ with respect to $\theta$ using minibatch SGD or Adam for $K$ epochs, with a minibatch size $M \leq NT$.
4. **Update:** Set $\theta_{\text{old}} \leftarrow \theta$ and repeat.

### Key Quantitative Results
**Continuous Control (MuJoCo):**
In tests across 7 simulated robotics tasks over one million timesteps, the clipped objective outperformed other variants. The average normalized score (where 0 is random and 1 is best) was:
*   **Clipping ($\epsilon = 0.2$):** 0.82
*   **Clipping ($\epsilon = 0.1$):** 0.76
*   **Adaptive KL ($d_{\text{targ}} = 0.01$):** 0.74
*   **No clipping or penalty:** -0.39 (indicating performance worse than a random policy in some environments).

**Atari Domain:**
PPO was compared against A2C and ACER across 49 games:
*   **Average reward over all training:** PPO "won" 30 games, ACER won 18, and A2C won 1.
*   **Average reward over last 100 episodes:** ACER won 28 games, PPO won 19, and A2C won 1.

### Stated Limitations
The authors note that while an adaptive KL penalty coefficient $\beta$ can be used to achieve a target KL divergence $d_{\text{targ}}$, this approach performed worse than the clipped surrogate objective in their experiments. Additionally, they observed that simply using a fixed penalty coefficient $\beta$ with SGD was insufficient to emulate the monotonic improvement of TRPO.
