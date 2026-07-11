---
id: huggingface:navigating-the-rlhf-landscape-from-polic
type: web
title: 'Navigating the RLHF Landscape: From Policy Gradients to PPO, GAE, and DPO
  for LLM Alignment'
url: https://huggingface.co/blog/NormalUhr/rlhf-pipeline
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlhf-ppo-pipeline
---

The provided article, "Navigating the RLHF Landscape: From Policy Gradients to PPO, GAE, and DPO for LLM Alignment," by Yihua Zhang, explores various Reinforcement Learning from Human Feedback (RLHF) methods for aligning Large Language Models (LLMs).

**Core Problem:**
The central problem addressed is the optimization of a policy (e.g., an LLM's strategy for generating text) to maximize expected returns (rewards) in a reinforcement learning setting. This involves refining policy parameters ($\theta$) to improve performance, often in scenarios where exact gradient computation is infeasible due to long trajectories or large models, leading to high variance in gradient estimates.

**Method/Recipe Step-by-Step (Policy Gradient Optimization and REINFORCE):**

1.  **On-Policy vs. Off-Policy Distinction:**
    *   **On-Policy (e.g., PPO):** Model actively generates its own data samples during training. Computationally demanding due to real-time generation, but offers higher theoretical performance.
    *   **Off-Policy (e.g., DPO):** Training relies on pre-collected data, eliminating real-time generation. Faster but sensitive to data alignment with the model's current capabilities.

2.  **Key Components in an On-Policy Framework (for LLMs):**
    *   **Actor:** The model that generates sentences.
    *   **Critic:** Provides immediate feedback for generated output, updated alongside the Actor.
    *   **Reward Model:** Assigns final scores or preference evaluations; typically fixed during training.
    *   **Reference Model:** Unique to PPO for large models, prevents the Actor from deviating too far from the original pre-trained distribution.

3.  **Policy Gradient Optimization (General Steps):**
    *   **Objective:** Maximize the expected return $J(\pi)$ of a policy $\pi_{\theta}$.
    *   **Parameter Update:** Use stochastic gradient ascent: $\theta_{k+1} = \theta_k + \alpha \nabla_{\theta} J(\pi_{\theta}) \big|_{\theta_k}$.
    *   **Gradient Computation (Derivation):**
        *   Expand expectation: $\nabla_{\theta}J(\pi_{\theta}) = \nabla_{\theta} \text{E}_{\tau \sim \pi_{\theta}} [R(\tau)] = \nabla_{\theta} \int_{\tau} P(\tau|\theta) R(\tau) \, d\tau$.
        *   Interchange gradient and integral: $= \int_{\tau} \nabla_{\theta} P(\tau|\theta) R(\tau) \, d\tau$.
        *   Apply log-derivative trick: $= \int_{\tau} P(\tau|\theta) \nabla_{\theta} \log P(\tau|\theta) \cdot R(\tau) \, d\tau$.
        *   Return to expectation form: $= \text{E}_{\tau \sim \pi_{\theta}} \left[ \nabla_{\theta} \log P(\tau|\theta) \cdot R(\tau) \right]$.
        *   Decompose $\nabla_{\theta} \log P(\tau|\theta)$: Since $P(\tau|\theta) = \rho_0(s_0) \prod_{i=0}^{T-1} P(s_{i+1}|s_i, a_i) \pi_{\theta}(a_i|s_i)$, then $\nabla_{\theta} \log P(\tau|\theta) = \sum_{i=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_i|s_i)$.

4.  **REINFORCE Algorithm (Specific Implementation Steps):**
    *   **Construct Policy Network:** A neural network taking state $s_t$ as input and outputting a probability distribution $P(a_t | s_t)$ over actions.
    *   **Trajectory Sampling:** Play games using the current policy to sample trajectories $\tau$ and record rewards.
    *   **Gradient Computation:** Estimate the gradient from the collected dataset $\mathcal{D}$ of games.
    *   **Parameter Update:** Update policy parameters using stochastic gradient ascent.
    *   **Iterative Optimization:** Repeat the cycle until policy converges.

5.  **Variance Reduction Techniques:**
    *   **Focusing on Future Rewards (Rewards-to-Go):** Instead of using the total trajectory reward $R(\tau)$, consider only rewards from the current step $t$ to the end of the game: $\sum_{t'=t}^T r(s_{i,t'}, a_{i,t'})$.
    *   **Introducing a Baseline:** Subtract a baseline $b$ (often the value function $V^{\pi}(s)$) from the future rewards. The term $\sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) - b$ becomes the "advantage."
    *   **Introducing Q and V Functions:** Define the Q-function $Q^{\pi}(s,a)$ as the total future return for taking action $a$ in state $s$. The advantage function is $A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$.
    *   **Estimating Advantage Term (GAE):** One way to estimate the advantage is $A^{\pi}(s_t,a_t) = [r(s_t,a_t) + \gamma V^{\pi}(s_{t+1})] - V^{\pi}(s_t)$.

**Key Formulas (in LaTeX):**

*   **Objective Function:**
    $\pi^* = \arg \max_{\pi} J(\pi)$
    $J(\pi_{\theta}) = \int_{\tau} P(\tau | \pi) R(\tau) = \mathbb{E}_{\tau \sim \pi} [R(\tau)]$
*   **Trajectory Probability:**
    $P(\tau | \pi) = \rho_0(s_0) \prod_{t=0}^{T-1} P(s_{t+1} | s_t, a_t) \pi(a_t | s_t)$
*   **Total Return for a Trajectory:**
    $R(\tau) = \sum_{t=0}^\infty \gamma^t r_t$
*   **Policy Gradient (General Form):**
    $\nabla_{\theta} J(\pi_{\theta}) = \text{E}_{\tau \sim \pi_{\theta}} \left[ \nabla_{\theta} \log P(\tau|\theta) \cdot R(\tau) \right]$
*   **Decomposed Log-Derivative:**
    $\nabla_{\theta} \log P(\tau|\theta) = \sum_{i=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_i|s_i)$
*   **Final Policy Gradient Formula (REINFORCE):**
    $\nabla_{\theta} J(\pi_{\theta}) = \text{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{i=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_i|s_i) \cdot R(\tau) \right]$
*   **Sample-based Policy Gradient (REINFORCE):**
    $\hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^T \nabla_{\theta} \text{log} \pi_\theta (a_t | s_t) R(\tau)$
*   **Policy Parameter Update:**
    $\theta_{k+1} = \theta_k + \alpha \hat{g}$
*   **Policy Gradient with Rewards-to-Go:**
    $\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=0}^T \nabla_{\theta} \log \pi_{\theta}(a_{i,t} | s_{i,t}) \right) \left( \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) \right)$
*   **Policy Gradient with Baseline:**
    $\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=0}^T \nabla_{\theta} \log \pi_{\theta}(a_{i,t} | s_{i,t}) \right) \left( \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) - b \right)$
*   **Advantage Function:**
    $A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$
*   **Policy Gradient with Advantage Function:**
    $\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=0}^T \nabla_{\theta} \log \pi_{\theta}(a_{i,t} | s_{i,t}) \right) A^{\pi}(s, a)$
*   **One-step Advantage Estimate:**
    $A^{\pi}(s_t,a_t) = [r(s_t,a_t)+\gamma V^{\pi}(s_{t+1})]-V^{\pi}(s_t)$

**Key Quantitative Results and Numbers:**
The article does not present specific quantitative results or numerical benchmarks for the algorithms discussed. It focuses on theoretical derivations and conceptual explanations. The discount factor $\gamma$ is defined as being in the range $[0, 1]$.

**Stated Limitations:**
*   **On-Policy Methods (PPO):** Tend to be computationally demanding and time-consuming, with the generation step being the main bottleneck. For generative tasks, outputting tokens one at a time is resource-intensive. Requires enormous computational resources, especially when components (Actor, Critic, Reward Model, Reference Model) are large (e.g., multiple 70B-parameter models).
*   **Off-Policy Methods (DPO):** Generally faster during training but highly sensitive to how well pre-collected data aligns with the model’s current capabilities. Learning can suffer if there's a significant mismatch (data too challenging or too trivial).
*   **Policy Gradient Optimization (General):** Computing the exact gradient requires summing or integrating over all possible trajectories, which is computationally infeasible for complex problems. Long trajectory lengths make auto-differentiation memory-intensive.
*   **REINFORCE Algorithm:** Necessitates a large number of game samples to mitigate the high variance inherent in the gradient estimates. The gradient estimator, while theoretically unbiased, can have extremely high variance, making evaluation unstable. Using the total trajectory reward $R(\tau)$ for every step $t$ is not entirely reasonable as decisions should only consider future outcomes.
