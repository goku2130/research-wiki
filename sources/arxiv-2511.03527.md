---
id: arxiv:2511.03527
type: paper
title: Learning Without Critics? Revisiting GRPO in Classical Control and LLMs
url: https://arxiv.org/abs/2511.03527
retrieved: '2026-07-12'
maturity: comprehensive
topic: grpo
---

# Summary: Learning Without Critics? Revisiting GRPO in Classical Control and LLMs

### Core Problem
Group Relative Policy Optimization (GRPO) was developed for Large Language Models (LLMs) to reduce computational and memory overhead by eliminating the learned critic (value function) and instead estimating advantages through group-relative comparisons of trajectories. This study investigates whether this critic-free approach is a viable general-purpose alternative to Proximal Policy Optimization (PPO) in classical reinforcement learning (RL) environments, specifically examining the impact of baseline estimation, discount factors, and group sampling strategies across discrete and continuous control tasks.

### Method and Recipe
The authors evaluated PPO and GRPO across five environments: CartPole-v1, Acrobot-v1, MountainCarContinuous-v0, HalfCheetah-v4, and Humanoid-v4. All agents were trained with a budget of 1M environment steps, using a learning rate of $2.5 \times 10^{-4}$, a clipping coefficient $\epsilon = 0.2$, and 4 epochs per iteration.

**1. Implementation Details:**
*   **PPO:** Utilizes a learned value function $V_\phi(s)$ with Generalized Advantage Estimation (GAE, $\lambda=0.95, \gamma=0.99$) and a rollout length of $N_{steps}=128$.
*   **GRPO:** Eliminates the critic. It uses Monte Carlo returns $R(\tau_i)$ for entire episodes ($\tau_i$) and a rollout length $N_{steps}=H$ (full episode horizon).
*   **Grouping Strategy:** In the absence of LLM-style prompts, groups are formed from $G$ trajectories collected concurrently from parallel environments within a single batch.

**2. Ablation Studies:**
*   **Baseline Comparison:** Compared PPO's learned critic against critic-free alternatives: no baseline, Random Gaussian, Exponential Moving Average (EMA), Batch Mean, and GRPO-Batch (batch mean with standard deviation scaling).
*   **Discount Factor Sweep:** Evaluated GRPO across $\gamma \in \{0, 0.1, 0.5, 0.95, 0.99, 1\}$.
*   **Group Size Sweep:** Evaluated GRPO across $G \in \{8, 16, 32, 64, 128\}$.

### Key Formulas
The objective is to maximize the expected discounted return:

$$
J(\theta)=\mathbb{E}_{\pi_{\theta}}\big[\sum_{t=0}^{\infty}\gamma^{t}r_t\big]
$$

GRPO replaces the learned critic with a group-relative advantage $\hat{A}^{\mathsf{GBPO}}$ for a group of $G$ trajectories:

$$
\hat{A}^{\mathsf{GBPO}}(\tau_{i})=\frac{R(\tau_{i})-\mu_{G}}{\sigma_{G}+\epsilon}
$$

Where the group mean $\mu_G$ and variance $\sigma_G^2$ are:

$$
\mu_{G}=\frac{1}{G}\sum_{j=1}^{G}R(\tau_{j}), \quad \sigma_{G}^{2}=\frac{1}{G}\sum_{j=1}^{G}(R(\tau_{j})-\mu_{G})^{2}
$$

### Key Quantitative Results
*   **Critic Necessity:** PPO with a learned value function substantially outperformed all critic-free baselines in long-horizon tasks. The only exception was **CartPole**, where critic-free methods often exceeded PPO, likely because PPO suffered from overfitting or policy collapse in this short-horizon environment.
*   **Discount Factors:** For most environments, $\gamma=0.99$ was optimal. However, **HalfCheetah** (the only environment without early termination) performed worst at $\gamma=1$ and optimal at $\gamma=0.9$.
*   **Group Size:** Contrary to expectations, smaller group sizes ($G=8$) generally outperformed larger ones, even when controlling for update frequency.
*   **Early Termination:** The authors found that early termination is a critical factor; it creates a natural separation between successful and unsuccessful trajectories, allowing critic-free methods to extract learning signals that are otherwise diluted in environments like HalfCheetah.

### Stated Limitations
*   **Grouping Strategy:** The batch-based grouping used may be suboptimal because it mixes potentially unrelated episodes from different parallel environments, unlike prompt-based grouping in LLMs.
*   **Hyperparameter Exploration:** The study only partially explored the hyperparameter space regarding mini-batches, entropy regularization, and learning rates.
*   **Scope:** Evaluation was limited to fully observable standard RL benchmarks; results may differ in sparse reward or partially observable domains.
