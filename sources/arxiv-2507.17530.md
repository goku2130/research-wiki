---
id: arxiv:2507.17530
type: paper
title: Generalized Advantage Estimation for Distributional Policy Gradients
url: https://arxiv.org/abs/2507.17530v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: policy-gradient-methods
---

## Generalized Advantage Estimation for Distributional Policy Gradients

### Core Problem

Traditional Generalized Advantage Estimation (GAE) effectively reduces variance in policy gradient estimates by using an exponentially weighted advantage function. However, GAE is designed for scalar value functions and cannot directly handle value distributions, which are crucial in distributional reinforcement learning (RL) for capturing inherent stochasticity and improving robustness to noise. Distributional RL methods, while using metrics like the Wasserstein distance to compare return distributions, lack a mechanism to quantify the "superiority" or "direction" between distributions, which is essential for defining an advantage function that guides policy updates.

### Method/Recipe Step by Step

The proposed method, Distributional GAE (DGAE), extends GAE to distributional RL by introducing a Wasserstein-like directional metric.

1.  **Define Wasserstein-like Directional Metric:**
    *   Unlike the traditional Wasserstein metric which uses a convex cost function $h(U-V) = \|U-V\|_p$, DGAE proposes a linear cost function $h(x,y) \triangleq L(x-y)$, where $L$ is a linear function. This allows for measuring both distance and "superiority" (direction of mass transfer) between distributions.
    *   **Definition 2:** For random variables $U, V$ with CDFs $F_U, G_V$ and compact supports, the Wasserstein-like directional metric $d$ is defined as:

$$
d(F_U, G_V) \triangleq \text{inf}_{U,V}(U-V) = \int_{0}^{1}L(F_U^{-1}(q)-G_V^{-1}(q)) dq
$$

    *   Superiority is defined based on the net mass transfer direction: negative direction implies $F_U$ is superior to $G_V$, and positive implies $G_V$ is superior to $F_U$.

2.  **Define Distributional Temporal Difference (TD) Error:**
    *   Using the Wasserstein-like directional metric, the distributional TD error $\delta^G(s_t, a_t)$ for a given state-action pair $(s_t, a_t)$ and value distribution $G(\cdot)$ is defined as:

$$
\delta^G(s_t, a_t) \triangleq d(r(s_t, a_t) + \gamma G(S_{t+1}), G(s_t)) = r(s_t, a_t) + d(\gamma G(S_{t+1}), G(s_t))
$$

3.  **Derive DGAE:**
    *   **Lemma 1:** For a random variable $U$ with inverse CDF $F_U^{-1}$ and a constant scalar $\eta \in \mathbb{R}$, the inverse CDF of $\eta U$ is $\eta F_U^{-1}$. Mathematically, $F_{\eta U}^{-1} = \eta F_U^{-1}$.
    *   **Theorem 1:** The DGAE $\hat{\mathcal{A}}_{\mathrm{D G A E}}^{\gamma,\lambda}(s_t, a_t)$ is defined as the exponentially weighted average of $n$-step estimators:

$$
\hat{\mathcal{A}}_{\mathrm{D G A E}}^{\gamma,\lambda}(s_t, a_t) = \sum_{k=0}^{\infty}(\gamma\lambda)^{k}\delta^{\hat{G}}(s_{t+k}, a_{t+k})
$$

        where $\delta^{\hat{G}}(s_{t+k}, a_{t+k})$ is the estimated distributional TD error (from step 2), $\gamma \in [0,1]$ is the discounting factor, and $\lambda \in (0,1)$ is a control parameter. Both $\gamma$ and $\lambda$ manage the bias-variance trade-off.

4.  **Integrate DGAE into Policy Gradient Algorithms (Algorithm 1):**
    *   Initialize policy network $\pi_{\theta_0}$ and value distribution network $G_{\phi_0}$.
    *   **Loop:**
        *   Sample $n$ time steps $\{s_t, a_t, r_t, s_{t+1}\}$ using the current policy $\pi_{\theta_i}$.
        *   Compute $\delta^{G_{\phi_i}}(s_t, a_t)$ for all time steps.
        *   Compute advantage estimates $\hat{\mathcal{A}}_{\mathrm{D G A E}}^{\gamma,\lambda}(s_t, a_t)$ using the truncated sum:

$$
\hat{\mathcal{A}}_{\mathrm{D G A E}}^{\gamma,\lambda}(s_t, a_t) = \sum_{k=0}^{n-1}(\gamma\lambda)^{k}\delta^{G_{\phi_i}}(s_{t+k}, a_{t+k})
$$

        *   Update policy parameters $\theta_i$ using the policy gradient approach with $\hat{\mathcal{A}}_{\mathrm{D G A E}}^{\gamma,\lambda}$ as the signal $\psi_k$:

$$
\theta_{i+1} \leftarrow \theta_i + \alpha\nabla_{\theta_i}J(\pi_{\theta_i})
$$

        *   Compute value loss $\mathcal{L}_{G_{\phi_i}}$ using the quantile-Huber loss (Eq. 17) between the value distribution $F_G^{-1}$ and its Bellman update $F_{\mathcal{T}G}^{-1}$:

$$
\mathcal{L}_{G_{\phi_i}} = \sum_{q_i}\mathbb{E}_{q_j}[\rho_{q_i}^{\kappa}(\mathcal{T}G_{\phi_i}(s_t, q_j) - G_{\phi_i}(s_t, q_i))]
$$

            where $\rho_q^{\kappa}(u) = |q-\delta_{\{u<0\}}|\mathcal{L}_{\kappa}(u)$ and $\mathcal{L}_{\kappa}(u)$ is the Huber loss.
        *   Update value network parameters $\phi$ using gradient descent:

$$
\phi_{i+1} \leftarrow \phi_i - \alpha\nabla_{\phi}\mathcal{L}_{G_{\phi_i}}
$$

### Key Formulas in LaTeX

*   **Wasserstein-like directional metric:**

$$
d(F_U, G_V) \triangleq \text{inf}_{U,V}(U-V) = \int_{0}^{1}L(F_U^{-1}(q)-G_V^{-1}(q)) dq
$$

*   **Distributional TD error:**

$$
\delta^G(s_t, a_t) \triangleq d(r(s_t, a_t) + \gamma G(S_{t+1}), G(s_t)) = r(s_t, a_t) + d(\gamma G(S_{t+1}), G(s_t))
$$

*   **DGAE formula:**

$$
\hat{\mathcal{A}}_{\mathrm{D G A E}}^{\gamma,\lambda}(s_t, a_t) = \sum_{k=0}^{\infty}(\gamma\lambda)^{k}\delta^{\hat{G}}(s_{t+k}, a_{t+k})
$$

*   **Quantile-Huber loss:**

$$
\mathcal{L}_{G_{\phi_i}} = \sum_{q_i}\mathbb{E}_{q_j}[\rho_{q_i}^{\kappa}(\mathcal{T}G_{\phi_i}(s_t, q_j) - G_{\phi_i}(s_t, q_i))]
$$

    where $\rho_q^{\kappa}(u) = |q-\delta_{\{u<0\}}|\mathcal{L}_{\kappa}(u)$ and

$$
\mathcal{L}_{\kappa}(u)=\begin{cases}{\frac{1}{2}u^{2},}&{\mathbf{i f}\ |u|\leq\kappa,}\\ {\kappa\left(|u|-\frac{1}{2}\kappa\right),}&{\mathbf{o t h e r w i s e}.}\\ \end{cases}
$$

### Key Quantitative Results and Numbers

*   Experiments were conducted on four OpenAI Gym MuJoCo environments: Ant, Hopper, Swimmer, and Walker2d.
*   Algorithms were trained for $10^7$ timesteps on Ant and Hopper, and $5 \times 10^6$ timesteps on Swimmer and Walker2d.
*   Each configuration used 5 random seeds.
*   **Ant environment:** DPPO significantly outperforms baseline PPO, and DTRPO shows considerable improvement over baseline TRPO. A2C and DA2C showed consistently poor performance.
*   **Hopper environment:** Both distributional variants (DPPO and DTRPO) surpass their baseline counterparts. DA2C shows a slight improvement over A2C, but overall performance of A2C/DA2C remains weak.
*   **Swimmer environment:** DTRPO and TRPO display very similar performance and outperform other algorithms. DPPO performs better than PPO, and DA2C outperforms A2C.
*   **Walker environment:** DPPO achieves the best results by a significant margin, followed by DTRPO. DA2C outperforms A2C, but PPO fails to deliver competitive results.
*   In the Hopper environment, scenarios where distributions with similar means yielded low advantage values (regardless of variance) occurred in about 0.093% of instances.
*   The sampling efficiency of DGAE-integrated algorithms is similar to traditional GAE baselines.

### Stated Limitations

*   The Wasserstein-like directional metric does not measure shape similarity between distributions, meaning two distributions with similar means but different variances can produce a low directional metric value. However, such scenarios rarely arise in practice (e.g., 0.093% in Hopper).
*   The paper limits DGAE integration to on-policy policy gradient methods (PPO, TRPO, A2C) and does not explore off-policy methods like DDPG, TD3, or SAC, which bypass advantage computation.
*   A2C and DA2C showed consistently poor performance across all environments.
