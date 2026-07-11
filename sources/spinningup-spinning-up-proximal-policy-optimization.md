---
id: spinningup:spinning-up-proximal-policy-optimization
type: web
title: 'Spinning Up: Proximal Policy Optimization (OpenAI)'
url: https://spinningup.openai.com/en/latest/algorithms/ppo.html
retrieved: '2026-07-11'
maturity: comprehensive
topic: ppo-for-llms
---

# Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) is a family of first-order reinforcement learning methods designed to address a core challenge in policy gradient optimization: taking the largest possible improvement step using available data without stepping so far that it triggers a performance collapse. While Trust Region Policy Optimization (TRPO) addresses this via complex second-order methods, PPO achieves similar empirical performance using simpler first-order methods and specialized objective functions to keep new policies close to old ones.

### Method: PPO-Clip
The primary variant used by OpenAI is **PPO-Clip**, which avoids hard KL-divergence constraints in favor of a clipped objective function. This clipping removes the incentive for the new policy $\pi_\theta$ to deviate significantly from the old policy $\pi_{\theta_k}$.

#### Key Formulas
The policy is updated by maximizing the following objective:

$$
\theta_{k+1} = \arg \max_{\theta} \underset{s,a \sim \pi_{\theta_k}}{{\mathrm E}}\left[ L(s,a,\theta_k, \theta)\right]
$$

The standard PPO-Clip objective $L$ is defined as:

$$
L(s,a,\theta_k,\theta) = \min\left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\; \text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1+\epsilon \right) A^{\pi_{\theta_k}}(s,a) \right)
$$

Spinning Up implements a simplified version of this objective:

$$
L(s,a,\theta_k,\theta) = \min\left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\; g(\epsilon, A^{\pi_{\theta_k}}(s,a)) \right)
$$

where:

$$
g(\epsilon, A) = \left\{ \begin{array}{ll} (1 + \epsilon) A & A \geq 0 \\ (1 - \epsilon) A & A < 0 \end{array} \right.
$$

In this formulation, $\epsilon$ is a small hyperparameter controlling the allowed distance between the new and old policies. If the advantage $A$ is positive, the objective is capped once $\pi_\theta(a|s) > (1+\epsilon)\pi_{\theta_k}(a|s)$. If the advantage is negative, the objective is capped once $\pi_\theta(a|s) < (1-\epsilon)\pi_{\theta_k}(a|s)$.

### Implementation Recipe
PPO is an on-policy algorithm compatible with both discrete and continuous action spaces. The training loop follows these steps:

1. **Trajectory Collection**: Collect a set of trajectories $\mathcal{D}_k$ by running the current policy $\pi_k$ in the environment.
2. **Reward Calculation**: Compute the rewards-to-go $\hat{R}_t$.
3. **Advantage Estimation**: Compute advantage estimates $\hat{A}_t$ using the current value function $V_{\phi_k}$ (the Spinning Up implementation specifically utilizes Generalized Advantage Estimation, or GAE).
4. **Policy Update**: Maximize the PPO-Clip objective using stochastic gradient ascent (typically with the Adam optimizer).
5. **Value Function Update**: Fit the value function $\phi$ by minimizing the mean-squared error (MSE) via regression:

$$
\phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{\phi} (s_t) - \hat{R}_t \right)^2
$$

### Quantitative Parameters
Based on the provided documentation, typical hyperparameters include:
* **Clip Ratio ($\epsilon$)**: $0.1$ to $0.3$.
* **Target KL**: $0.01$ or $0.05$ (used for early stopping).
* **Discount Factor ($\gamma$)**: $0.99$.
* **GAE Lambda ($\lambda$)**: Close to $1$ (e.g., $0.97$).
* **Learning Rates**: $\pi_{lr} = 0.0003$ and $vf_{lr} = 0.001$.

### Limitations
* **Exploration**: Because PPO trains a stochastic policy on-policy, it explores by sampling from its latest policy. As the policy exploits found rewards and becomes progressively less random, it may become trapped in local optima.
* **Stability**: While clipping regularizes updates, it does not strictly guarantee that the new policy remains close to the old one. To mitigate this, the Spinning Up implementation employs **early stopping**, which halts gradient steps if the mean KL-divergence exceeds the `target_kl` threshold.
