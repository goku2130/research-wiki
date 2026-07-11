---
id: aayushgarg:understanding-grpo-ppo-without-the-criti
type: web
title: 'Understanding GRPO: PPO without the Critic'
url: https://aayushgarg.dev/posts/2026-01-01-understanding-grpo.html
retrieved: '2026-07-11'
maturity: comprehensive
topic: grpo
---

# Group Relative Policy Optimization (GRPO)

Group Relative Policy Optimization (GRPO), introduced in the DeepSeekMath paper, is a reinforcement learning algorithm designed for the post-training of Large Language Models (LLMs). It is conceptualized as a version of Proximal Policy Optimization (PPO) that eliminates the need for a critic (value function).

### Core Problem
Standard PPO requires training a value function (critic) alongside the policy to estimate baselines for advantage computation. In the context of LLMs, this presents two primary challenges:
1. **Computational Overhead:** The critic is typically a separate model with the same architecture as the policy. Consequently, PPO requires four models to reside in memory simultaneously: the policy, the critic, the reference model, and the reward model.
2. **Reward Sparsity:** PPO utilizes Generalized Advantage Estimation (GAE), which requires per-token rewards to compute Temporal Difference (TD) residuals. However, LLM fine-tuning often relies on outcome rewards—a single score assigned only to the final token—making it difficult for a critic to learn accurate per-token values.

### Method and Recipe
GRPO replaces the learned value function with a Monte Carlo baseline derived from group sampling. The step-by-step process is as follows:

1. **Group Sampling:** For each prompt $q$, the model samples a group of $G$ completions $\{o_1, o_2, \ldots, o_G\}$ from the old policy $\pi_{\theta_{\text{old}}}$.
2. **Reward Collection:** Each completion is assigned a reward $\{r_1, r_2, \ldots, r_G\}$ by a reward model.
3. **Advantage Computation:** Instead of using a critic, the advantage for each completion is computed relative to the mean reward of the group and normalized by the group's standard deviation.
4. **Objective Optimization:** The policy is updated using a clipped surrogate objective (similar to PPO) to maintain stability, but the KL divergence penalty is moved from the reward signal directly into the loss function.
5. **KL Estimation:** To reduce variance and ensure non-negativity, a specific unbiased KL estimator is used rather than a simple log-ratio.

### Key Formulas

**Group-Relative Advantage:**

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}
$$

**Full GRPO Objective:**

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D},\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(o|q)}\left[\frac{1}{G}\sum_{i=1}^G \left( \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}\hat{A}_i, \; \text{clip}(\cdot) \cdot \hat{A}_i\right) - \beta \, D_{\text{KL}}\left(\pi_\theta \| \pi_{\text{ref}}\right) \right)\right]
$$

**Unbiased Non-negative KL Estimator:**

$$
D_{\text{KL}}^{(t)} = \frac{\pi_{\text{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - 1
$$

### Extensions and Variations
* **Process Supervision:** For complex reasoning, GRPO can be extended from outcome supervision to process supervision. In this mode, rewards are assigned at the end of each reasoning step. Advantages are computed as the sum of normalized cumulative future rewards:

$$
\hat{A}_{i,t} = \sum_{\text{index}(j) \geq t} \tilde{r}_i^{\text{index}(j)}
$$

* **Comparison to RLOO:** Unlike REINFORCE Leave-One-Out (RLOO), which uses a baseline excluding the current sample and lacks clipping, GRPO incorporates PPO-style clipping and reward normalization.

### Quantitative Results and Limitations
* **Memory Efficiency:** GRPO reduces memory consumption and training complexity by eliminating the critic model, which in PPO would be one of four resident models.
* **Limitations:** The source notes that outcome supervision (single final reward) may be insufficient or inefficient for supervising policies in complex mathematical tasks, necessitating the use of process supervision to accelerate learning.
