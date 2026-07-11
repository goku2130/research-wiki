---
id: arxiv:1810.12558
type: paper
title: Relative Importance Sampling for off-Policy Actor-Critic in Deep Reinforcement
  Learning
url: https://arxiv.org/abs/1810.12558
retrieved: '2026-07-11'
maturity: comprehensive
topic: async-and-off-policy-rl
---

# Relative Importance Sampling for off-Policy Actor-Critic

## Core Problem
Off-policy reinforcement learning (RL) is inherently less stable than on-policy learning due to the distributional mismatch between the target policy ($\pi$) and the behavior policy ($b$). This discrepancy introduces significant variance, particularly in continuous action spaces and long-horizon tasks. While Importance Sampling (IS) is traditionally used to correct this mismatch, it often produces extreme values (high variance), which can lead to unstable learning and poor convergence.

## Method: RIS-off-PAC
The authors propose **Relative Importance Sampling (RIS)**, a "smooth" version of IS designed to bound variance and stabilize the learning process. They implement this within two actor-critic frameworks: **RIS-off-PAC** (standard gradient) and **RIS-off-PNAC** (natural gradient).

### Step-by-Step Recipe
1.  **Initialization**: Initialize policy parameters $\theta$, critic parameters $\phi$, discount factor $\gamma$, and a smoothness parameter $\beta \in [0, 1]$.
2.  **Sampling**: Select an action $a_t$ based on the behavior policy $b(\cdot|s_t)$.
3.  **Weight Calculation**: Compute the relative importance weight $\mu_{t,\beta}$ using the target and behavior policies.
4.  **Critic Update**: 
    *   Calculate the temporal difference (TD) residual $\delta_t^{V_\phi^\pi}$ using the reward from the behavior policy action.
    *   Update the value function $V_\phi^\pi$ by minimizing the squared TD residual: $J_V(\phi) = \mathbb{E}[\frac{1}{2}(\delta_t^{V_\phi^\pi})^2]$.
5.  **Actor Update**:
    *   **For RIS-off-PAC**: Update $\theta$ using the RIS-weighted policy gradient: $\nabla_\theta J_{\mu_\beta}(\theta) \approx \mu_{t,\beta} \nabla_\theta \log \pi_\theta(a_t|s_t) \delta_t^{V_\phi^\pi}$.
    *   **For RIS-off-PNAC**: Replace the standard gradient with a natural gradient estimate $G^{-1}(\theta) \nabla_\theta \log \pi_\theta(a_t|s_t) \delta_t^{V_\phi^\pi}$.
6.  **Iteration**: Repeat until the terminal state is reached.

## Key Formulas
The **Relative Importance Sampling (RIS) ratio** is defined as:

$$
\mu_{\beta}=\frac{e^{\pi(a|s)}}{\beta e^{\pi(a|s)}+(1-\beta)e^{b(a|s)}}
$$

The **TD residual** used for the critic is:

$$
\delta_{t}^{V^{\pi}}=r(s_{t},a_{t}\sim b(.|s_{t}))+\gamma V^{\pi}(s_{t+1})-V^{\pi}(s_{t})
$$

The **variance of the RIS estimator** is given by:

$$
V_{\beta}(\hat{\mu}_{\beta})=\frac{2\gamma(1-\gamma)(1-\beta)}{[\beta\pi(A|S)+(1-\beta)b(A|S)]^{2}}
$$

## Key Quantitative Results
The algorithms were tested on OpenAI Gym environments and compared against A3C, PPO, PG, and SAC. Statistical significance was confirmed via Kruskal-Wallis tests ($p < 0.05$ for all pairs).

**Average Rewards (Last 100 Episodes):**

| Environment | RIS-off-PAC | RIS-off-PNAC | A3C | PPO | SAC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CartPole-v0** | $1176.27 \pm 2.18$ | $\mathbf{1386.66 \pm 0.16}$ | $1147.20 \pm 18.22$ | $158.59 \pm 1.96$ | N/A |
| **Humanoid-v2** | $141.36 \pm 2.14$ | $\mathbf{144.50 \pm 0.40}$ | $105.66 \pm 1.11$ | $108.74 \pm 1.38$ | $105.33 \pm 1.11$ |
| **MountainCar-v0**| $\mathbf{-124.66 \pm 1.51}$| $-146.80 \pm 1.83$ | $-1089.51 \pm 8.89$| $-6448.20 \pm 82.62$| N/A |
| **Pendulum-v0** | $-6.18 \pm 0.02$ | $\mathbf{-3.78 \pm 0.00}$ | $-11.43 \pm 0.00$ | $-13.99 \pm 0.01$ | N/A |

**Key Observations:**
*   **$\beta$ Impact**: As $\beta$ increases from 0 to 1, the variance of the RIS estimator decreases toward zero. Higher $\beta$ values generally led to higher average rewards and increased stability.
*   **Performance**: RIS-off-PNAC generally outperformed all other methods in CartPole, Pendulum, and Humanoid-v2, while RIS-off-PAC was superior in MountainCar-v0.

## Stated Limitations
The authors note that the effectiveness of the method relies on the selection of the smoothness parameter $\beta$. While they demonstrate that increasing $\beta$ generally reduces instability and maximizes reward, the optimal value may vary by environment. Additionally, the current implementation is focused on specific control tasks, leaving extensions to other IS methods (like Per-decision or Weighted IS) for future work.
