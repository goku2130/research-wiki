---
id: arxiv:1506.02438
type: paper
title: High-Dimensional Continuous Control Using Generalized Advantage Estimation
url: https://arxiv.org/abs/1506.02438
retrieved: '2026-07-11'
maturity: comprehensive
topic: policy-gradient-methods
---

Policy gradient methods directly optimize cumulative reward but suffer from high sample complexity due to gradient variance that scales with the time horizon. While value functions reduce variance, they introduce bias that can prevent convergence to optimal policies. The core challenge is balancing this bias-variance tradeoff in high-dimensional, continuous control tasks with long credit assignment horizons. The authors propose Generalized Advantage Estimation (GAE), a parameterized estimator that interpolates between low-bias/high-variance and high-bias/low-variance advantage estimates. GAE is combined with Trust Region Policy Optimization (TRPO) for stable policy updates and a trust-region-based regression method for value function training. The algorithm iterates as follows: simulate trajectories using the current policy until a batch of $N$ timesteps is collected; compute TD residuals using the current value function; compute GAE advantages via an exponentially weighted sum of future residuals; update policy parameters using TRPO to maximize a surrogate objective under a KL-divergence constraint; update value function parameters by solving a constrained least-squares regression problem that limits the average squared deviation from the previous value function. The policy update uses the value function from the current iteration, not the updated one, to avoid compounding bias.

The policy gradient takes the form $g = \mathbb{E} \left[ \sum_{t=0}^{\infty} \Psi_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$, where $\Psi_t$ is replaced by the GAE estimator. The core advantage estimator is defined as:
$$\hat{A}_{t}^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l}^{V}$$
where $\gamma$ acts as a variance-reduction discount and $\lambda$ controls the interpolation between one-step and infinite-step returns. The TRPO policy update solves:
$$\underset{\theta}{\text{minimize}} \frac{1}{N} \sum_{n=1}^{N} \frac{\pi_{\theta}(a_n \mid s_n)}{\pi_{\theta_{\text{old}}}(a_n \mid s_n)} \hat{A}_n \quad \text{subject to } \overline{D}_{\text{KL}}^{\theta_{\text{old}}}(\pi_{\theta_{\text{old}}}, \pi_{\theta}) \leq \epsilon$$
The value function is optimized via:
$$\underset{\phi}{\text{minimize}} \sum_{n=1}^{N} \| V_{\phi}(s_n) - \hat{V}_n \|^2 \quad \text{subject to } \frac{1}{N} \sum_{n=1}^{N} \frac{\| V_{\phi}(s_n) - V_{\phi_{\text{old}}}(s_n) \|^2}{2\sigma^2} \leq \epsilon$$
This constrained regression is solved approximately using conjugate gradients with a Gauss-Newton Hessian approximation.

Experiments were conducted on cart-pole balancing and 3D simulated locomotion tasks using the MuJoCo physics engine. Neural network policies and value functions used three hidden layers (100, 50, 25 tanh units), exceeding $10^4$ parameters each. For the 3D biped (33 state dimensions, 10 actuators), learning required 50,000 timesteps per batch over 1,000 batches, consuming approximately 5.8 days of real-time simulation. The quadruped task (29 states, 8 actuators) used 200,000 timesteps per batch. Optimal performance consistently occurred at intermediate parameter ranges: $\gamma \in [0.96, 0.995]$ and $\lambda \in [0.92, 0.99]$. Setting $\lambda = 0$ introduced excessive bias and degraded performance, while $\lambda = 1$ yielded high variance. The trained policies achieved fast, stable gaits and successfully learned to stand from a prone position.

The approach requires careful manual tuning of $\gamma$ and $\lambda$, with no automatic adaptation mechanism provided. The theoretical relationship between value function approximation error and policy gradient estimation error remains uncharacterized, limiting principled value function training objectives. The method is computationally intensive, relying on parallelized simulation and conjugate gradient optimization. Furthermore, the experiments are entirely model-free and simulator-based; real-world deployment would require robust state-resetting mechanisms to prevent hardware damage during exploration.
