---
id: arxiv:1707.06347
type: paper
title: Proximal Policy Optimization Algorithms (Schulman et al., 2017)
url: https://arxiv.org/abs/1707.06347
retrieved: '2026-07-11'
maturity: comprehensive
topic: mdp-formulation
---

# Proximal Policy Optimization (PPO)

### Core Problem
Standard policy gradient methods often suffer from poor data efficiency and instability; specifically, performing multiple epochs of optimization on a single batch of data typically leads to destructively large policy updates. While Trust Region Policy Optimization (TRPO) addresses this by constraining the policy update size, it is computationally complex, relies on second-order optimization, and is incompatible with architectures utilizing parameter sharing or noise (e.g., dropout). The goal of PPO is to achieve the stability and reliability of TRPO using only first-order optimization.

### Method
PPO introduces a "clipped" surrogate objective to penalize changes to the policy that move the probability ratio too far from 1. The algorithm alternates between sampling data and performing multiple epochs of minibatch stochastic gradient ascent.

**Step-by-Step Recipe:**
1. **Data Collection:** $N$ parallel actors run the current policy $\pi_{\theta_{\text{old}}}$ for $T$ timesteps.
2. **Advantage Estimation:** Compute advantage estimates $\hat{A}_t$ for each timestep. The authors utilize a truncated version of Generalized Advantage Estimation (GAE).
3. **Optimization:** Optimize the surrogate objective $L^{CLIP}$ using minibatch SGD or Adam for $K$ epochs.
4. **Parameter Update:** Update the old policy parameters: $\theta_{\text{old}} \leftarrow \theta$.
5. **Combined Loss (Optional):** When sharing parameters between the policy and value function, the objective is augmented with a squared-error value function loss $L_t^{VF}$ and an entropy bonus $S$ to encourage exploration.

### Key Formulas
The probability ratio is defined as:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

The **Clipped Surrogate Objective** is:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min (r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t) \right]
$$

where $\epsilon$ is a hyperparameter (e.g., $0.2$).

The **Combined Objective** for actor-critic architectures is:

$$
L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]
$$

where $L_t^{VF} = (V_\theta(s_t) - V_t^{\text{targ}})^2$.

The **Truncated GAE** estimator is:

$$
\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}, \quad \text{where } \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

### Key Quantitative Results
**Continuous Control (MuJoCo):**
In a benchmark of 7 simulated robotics tasks over one million timesteps, the clipped objective outperformed other variants. The average normalized scores were:
* **Clipping ($\epsilon=0.2$):** 0.82 (Best)
* **Clipping ($\epsilon=0.1$):** 0.76
* **Adaptive KL ($d_{\text{targ}}=0.01$):** 0.74
* **No clipping or penalty:** -0.39

PPO outperformed TRPO, CEM, A2C, and vanilla policy gradient on nearly all continuous control environments tested.

**Atari Domain:**
PPO was compared against A2C and ACER across 49 games:
* **Average reward over all training:** PPO "won" 30 games, ACER won 18, and A2C won 1.
* **Average reward over last 100 episodes:** ACER "won" 28 games, PPO won 19, and A2C won 1.

### Stated Limitations
The authors note that while an adaptive KL penalty coefficient can be used as an alternative to clipping, it performed worse in their experiments. Additionally, the "no clipping or penalty" setting was found to be unstable, leading to very negative scores in certain environments (e.g., HalfCheetah).
