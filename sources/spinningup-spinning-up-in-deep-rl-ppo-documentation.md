---
id: spinningup:spinning-up-in-deep-rl-ppo-documentation
type: web
title: 'Spinning Up in Deep RL: PPO Documentation'
url: https://spinningup.openai.com/en/latest/algorithms/ppo.html
retrieved: '2026-07-12'
maturity: comprehensive
topic: ppo-for-llms
---

# Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) is a family of first-order reinforcement learning methods designed to address a fundamental challenge in policy gradient optimization: determining the largest possible improvement step that can be taken using current data without causing a "performance collapse" by stepping too far from the previous policy. While Trust Region Policy Optimization (TRPO) addresses this via complex second-order methods, PPO achieves similar empirical performance using simpler first-order methods and specialized objective functions to keep new policies close to old ones.

### The PPO-Clip Method
The primary variant used by OpenAI is **PPO-Clip**, which avoids hard KL-divergence constraints. Instead, it uses a clipped surrogate objective to remove incentives for the policy to move drastically.

The policy is updated by maximizing the following objective:

$$
\theta_{k+1} = \arg \max_{\theta} \mathbb{E}_{s,a \sim \pi_{\theta_k}}[L(s,a,\theta_k, \theta)]
$$

The objective function $L$ is defined as:

$$
L(s,a,\theta_k,\theta) = \min\left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a), \text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1+\epsilon \right) A^{\pi_{\theta_k}}(s,a) \right)
$$

Where $\epsilon$ is a small hyperparameter representing the allowed distance between the new and old policy. A simplified version of this objective is often used in implementation:

$$
L(s,a,\theta_k,\theta) = \min\left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a), g(\epsilon, A^{\pi_{\theta_k}}(s,a)) \right)
$$

with $g(\epsilon, A)$ defined as:

$$
g(\epsilon, A) = \begin{cases} (1 + \epsilon) A & A \geq 0 \\ (1 - \epsilon) A & A < 0 \end{cases}
$$

This mechanism ensures that if the advantage $A$ is positive, the objective hits a ceiling once $\pi_{\theta}(a|s) > (1+\epsilon) \pi_{\theta_k}(a|s)$. Conversely, if the advantage is negative, it hits a ceiling once $\pi_{\theta}(a|s) < (1-\epsilon) \pi_{\theta_k}(a|s)$.

### Algorithmic Recipe
PPO is an on-policy algorithm that can be applied to both discrete and continuous action spaces. The training loop follows these steps:

1. **Trajectory Collection**: Collect a set of trajectories $\mathcal{D}_k = \{\tau_i\}$ by running the current policy $\pi_k = \pi(\theta_k)$ in the environment.
2. **Reward Calculation**: Compute the rewards-to-go $\hat{R}_t$.
3. **Advantage Estimation**: Compute advantage estimates $\hat{A}_t$ using the current value function $V_{\phi_k}$ (the Spinning Up implementation specifically utilizes Generalized Advantage Estimation, or GAE).
4. **Policy Update**: Maximize the PPO-Clip objective using stochastic gradient ascent (typically via the Adam optimizer).
5. **Value Function Update**: Fit the value function by minimizing the mean-squared error (MSE) via regression:

$$
\phi_{k+1} = \arg \min_{\phi} \frac{1}{|\mathcal{D}_k| T} \sum_{\tau \in \mathcal{D}_k} \sum_{t=0}^T\left( V_{\phi} (s_t) - \hat{R}_t \right)^2
$$

### Key Quantitative Parameters
Based on the Spinning Up implementation, typical hyperparameters include:
* **Clip Ratio ($\epsilon$):** $0.1$ to $0.3$ (default $0.2$).
* **Target KL:** $0.01$ or $0.05$ (used for early stopping).
* **Discount Factor ($\gamma$):** $0.99$.
* **GAE Lambda ($\lambda$):** $0.97$.
* **Learning Rates:** Policy optimizer ($\pi_{lr}$) = $0.0003$; Value function optimizer ($vf_{lr}$) = $0.001$.
* **Epochs/Iterations:** $50$ epochs of interaction; maximum of $80$ gradient descent steps for both policy and value function per epoch.

### Limitations
* **Local Optima**: Because PPO trains a stochastic policy in an on-policy manner, it explores by sampling from its latest policy. As the policy becomes progressively less random to exploit rewards, it may become trapped in local optima.
* **Policy Drift**: Clipping alone may not fully prevent the policy from moving too far from the old policy. To mitigate this, the implementation employs **early stopping**, which halts gradient steps if the mean KL-divergence exceeds the `target_kl` threshold.
