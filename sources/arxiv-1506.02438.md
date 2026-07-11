---
id: arxiv:1506.02438
type: paper
title: High-Dimensional Continuous Control Using Generalized Advantage Estimation
  and Trust Region Policy Optimization
url: https://arxiv.org/pdf/1506.02438
retrieved: '2026-07-11'
maturity: comprehensive
topic: policy-gradient-methods
---

This paper introduces Generalized Advantage Estimation (GAE) to address the challenges of high variance and instability in policy gradient methods for reinforcement learning, particularly in high-dimensional continuous control tasks.

**Core Problem:**
Policy gradient methods suffer from high variance in gradient estimates, which scales unfavorably with the time horizon, and instability due to the nonstationarity of incoming data. While value functions can reduce variance, they often introduce bias. The goal is to develop a policy gradient estimator that significantly reduces variance while maintaining a tolerable level of bias, and to achieve stable policy improvement.

**Method/Recipe Step-by-Step:**

1.  **Policy Gradient Formulation:** The policy gradient is generally expressed as:

$$
g = \mathbb{E} \left[ \sum_{t=0}^{\infty} \Psi_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$

    where $\Psi_t$ is an estimate of the advantage function $A^\pi(s_t, a_t) := Q^\pi(s_t, a_t) - V^\pi(s_t)$. The paper uses a discounted version of the advantage function, $A^{\pi,\gamma}(s_t, a_t)$, where $\gamma \in [0, 1]$ is a parameter to reduce variance by downweighting delayed effects.

$$
A^{\pi,\gamma}(s_t, a_t) := Q^{\pi,\gamma}(s_t, a_t) - V^{\pi,\gamma}(s_t)
$$

    The discounted policy gradient is:

$$
g^\gamma := \mathbb{E}_{s_{0:\infty}, a_{0:\infty}} \left[ \sum_{t=0}^{\infty} A^{\pi,\gamma}(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$

2.  **Generalized Advantage Estimation (GAE):**
    *   Define the TD residual of an approximate value function $V$ with discount $\gamma$:

$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

    *   Define $k$-step advantage estimators:

$$
\hat{A}_t^{(k)} := \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = -V(s_t) + r_t + \gamma r_{t+1} + \dots + \gamma^{k-1} r_{t+k-1} + \gamma^k V(s_{t+k})
$$

    *   The GAE($\gamma, \lambda$) is an exponentially-weighted average of these $k$-step estimators:

$$
\hat{A}_{t}^{\mathrm{GAE}(\gamma,\lambda)} := \sum_{l=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l}^{V}
$$

    *   Special cases:
        *   GAE($\gamma,0$): $\hat{A}_{t} := \delta_{t} = r_{t} + \gamma V(s_{t+1}) - V(s_{t})$ (low variance, high bias if $V \neq V^{\pi,\gamma}$)
        *   GAE($\gamma,1$): $\hat{A}_{t} := \sum_{l=0}^{\infty}\gamma^{l}\delta_{t+l} = \sum_{l=0}^{\infty}\gamma^{l}r_{t+l} - V(s_{t})$ (low bias, high variance)
    *   The GAE provides a biased estimator of $g^\gamma$:

$$
g^{\gamma} \approx \mathbb{E}\left[\sum_{t=0}^{\infty}\nabla_{\theta}\log\pi_{\theta}(a_{t} \mid s_{t})\hat{A}_{t}^{\mathrm{GAE}(\gamma,\lambda)}\right] = \mathbb{E}\left[\sum_{t=0}^{\infty}\nabla_{\theta}\log\pi_{\theta}(a_{t} \mid s_{t})\sum_{l=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l}^{V}\right]
$$

3.  **Value Function Estimation:**
    *   A neural network is used to approximate the value function $V_\phi(s)$.
    *   The value function is optimized by minimizing the squared error between $V_\phi(s_n)$ and the discounted sum of rewards $\hat{V}_{n} = \sum_{l=0}^{\infty} \gamma^{l} r_{t+l}$ (Monte Carlo or TD(1) approach).
    *   A trust region optimization method is used to train the value function, preventing overfitting:

$$
\underset {\phi} {\text { minimize }} \sum_ {n = 1} ^ {N} \| V _ {\phi} (s _ {n}) - \hat {V} _ {n} \| ^ {2}
$$

$$
\text { subject to } \quad \frac {1}{N} \sum_ {n = 1} ^ {N} \frac {\left\| V _ {\phi} (s _ {n}) - V _ {\phi_ {\mathrm{old}}} (s _ {n}) \right\| ^ {2}}{2 \sigma^ {2}} \leq \epsilon
$$

        where $\sigma^{2} = \frac{1}{N}\sum_{n=1}^{N}\|V_{\phi_{\mathrm{old}}}(s_{n}) - \hat{V}_{n}\|^{2}$. This is approximately solved using the conjugate gradient algorithm.

4.  **Policy Optimization Algorithm (TRPO):**
    *   The policy is updated using Trust Region Policy Optimization (TRPO) (Schulman et al., 2015).
    *   Each iteration, TRPO approximately solves:

$$
\underset{\theta}{\text{minimize}} L_{\theta_{old}}(\theta)
$$

$$
\text{subject to } \overline{D}_{\text{KL}}^{\theta_{old}}(\pi_{\theta_{old}}, \pi_{\theta}) \leq \epsilon
$$

$$
\text{where } L_{\theta_{old}}(\theta) = \frac{1}{N} \sum_{n=1}^{N} \frac{\pi_{\theta}(a_n \mid s_n)}{\pi_{\theta_{old}}(a_n \mid s_n)} \hat{A}_n
$$

$$
\overline{D}_{\text{KL}}^{\theta_{old}}(\pi_{\theta_{old}}, \pi_{\theta}) = \frac{1}{N} \sum_{n=1}^{N} D_{KL}(\pi_{\theta_{old}}(\cdot \mid s_n) \parallel \pi_{\theta}(\cdot \mid s_n))
$$

    *   This is solved by linearizing the objective and quadraticizing the constraint, yielding a step direction $\theta - \theta_{old} \propto -F^{-1}g$, where $F$ is the average Fisher information matrix and $g$ is the policy gradient estimate.

5.  **Overall Algorithm Loop:**
    *   Initialize policy parameter $\theta_0$ and value function parameter $\phi_0$.
    *   For $i = 0, 1, 2, \dots$:
        *   Simulate current policy $\pi_{\theta_i}$ until $N$ timesteps are obtained.
        *   Compute $\delta_t^V$ at all timesteps using $V = V_{\phi_i}$.
        *   Compute $\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$ at all timesteps.
        *   Compute $\theta_{i+1}$ with TRPO update.
        *   Compute $\phi_{i+1}$ with the trust region value function optimization.

**Key Quantitative Results and Numbers:**

*   **Cart-Pole Task:** Best performance achieved with $\gamma \in [0.96, 0.99]$ and $\lambda \in [0.92, 0.99]$.
*   **3D Bipedal Locomotion:**
    *   Neural network architecture: three hidden layers (100, 50, 25 tanh units) for both policy and value function.
    *   State dimensions: 33; Actuated degrees of freedom: 10.
    *   Batch size: 50,000 timesteps.
    *   Learning time: 1000 batches $\times$ 50,000 timesteps/batch $\times$ 0.01 sec/timestep $\approx$ 5.8 days of simulated experience.
    *   Best performance with $\gamma \in [0.99, 0.995]$ and $\lambda \in [0.96, 0.99]$.
*   **Quadrupedal Locomotion:**
    *   State dimensions: 29; Actuated degrees of freedom: 8.
    *   Batch size: 200,000 timesteps.
    *   Best results with $\lambda = 0.96$ (fixed $\gamma = 0.995$).
*   **Biped Getting Up:**
    *   Batch size: 200,000 timesteps.
    *   Value function always helped; results roughly same for $\lambda = 0.96$ and $\lambda = 1$ (fixed $\gamma = 0.995$).
*   **Computational Resources:** 3D robot tasks took 2-4 hours per trial on 16-32 core machines.
*   **Neural Network Parameters:** Over $10^4$ parameters for both policy and value function in 3D tasks.

**Stated Limitations:**

*   The choice of $\gamma < 1$ introduces bias into the policy gradient estimate regardless of value function accuracy.
*   The choice of $\lambda < 1$ introduces bias only when the value function is inaccurate.
*   The paper notes that the best value of $\lambda$ is typically much lower than the best value of $\gamma$, suggesting $\lambda$ introduces less bias for a reasonably accurate value function.
*   The algorithm requires a large amount of simulated experience (e.g., 5.8 days of real-time equivalent for bipedal locomotion), which might be challenging for real-world robotic applications without state reset capabilities or self-damage prevention.
*   The relationship between value function estimation error and policy gradient estimation error is not fully understood, making it difficult to choose an optimal error metric for value function fitting.
*   The paper does not explore using a shared function approximation architecture for policy and value function, which could lead to faster learning.
*   Comparison with concurrent work using continuous-valued action differentiation methods (e.g., Lillicrap et al., 2015; Heess et al., 2015) is limited, as those papers consider lower-dimensional state and action spaces.
