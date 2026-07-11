---
id: snorkel:grpo-group-relative-policy-optimization-
type: web
title: GRPO (Group Relative Policy Optimization), explained - Snorkel AI
url: https://snorkel.ai/grpo/
retrieved: '2026-07-11'
maturity: comprehensive
topic: grpo
---

GRPO (Group Relative Policy Optimization) is a reinforcement learning algorithm for training large language models, introduced by DeepSeek in the DeepSeekMath paper (2024) and scaled by DeepSeek-R1 (2025). It is designed to address the challenges of large-scale reinforcement learning for reasoning tasks.

**Core Problem:**
The core problem GRPO addresses is the computational cost and instability associated with the value network (critic) in traditional PPO (Proximal Policy Optimization) algorithms, especially when applied to large language models. This critic requires additional memory and compute for training and inference, and its learning process can introduce instability.

**Method/Recipe Step by Step:**
For each training prompt, GRPO follows a four-step process:
1.  **Sample a group:** The current policy generates multiple responses (typically 8 to 64 samples) for the same prompt.
2.  **Score each response:** A reward signal is applied to each generated response. For verifiable tasks like math or code, this can be a correctness check (RLVR); otherwise, a reward model is used.
3.  **Compute group-relative advantage:** Each response's reward is normalized against the group. This involves subtracting the group mean and dividing by the group standard deviation. This normalized score serves as the advantage, eliminating the need for a separate critic network.
4.  **Update the policy:** The policy is updated using a PPO-style clipped objective and a KL penalty towards a reference model. This update reinforces responses that scored above the group average.

**Key Formulas (Implicit in Description):**
While specific LaTeX formulas are not provided in the text, the description of the group-relative advantage implies the following operations:
Let $R_i$ be the reward for response $i$ in a group of $N$ responses.
Let $\mu_R$ be the mean reward of the group:

$$
\mu_R = \frac{1}{N} \sum_{i=1}^{N} R_i
$$

Let $\sigma_R$ be the standard deviation of the group rewards:

$$
\sigma_R = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (R_i - \mu_R)^2}
$$

The group-relative advantage $A_i$ for response $i$ is then:

$$
A_i = \frac{R_i - \mu_R}{\sigma_R}
$$

The policy update would then use this $A_i$ in a clipped objective function similar to PPO.

**Key Quantitative Results and Numbers:**
*   GRPO was introduced in 2024 and scaled by DeepSeek-R1 in January 2025.
*   It typically samples a group of 8 to 64 responses per prompt.
*   GiGPO (Group-in-Group Policy Optimization), a successor to GRPO, reports gains of more than 12% over GRPO on ALFWorld and more than 9% on WebShop at the same memory cost.

**Stated Limitations:**
*   **Long-horizon credit assignment:** GRPO struggles with tasks that involve a long sequence of actions, such as those encountered by modern AI agents. This is because it typically assigns a single reward at the end of a long task. If an agent fails after many steps, every step receives the same negative signal, making it difficult for the model to identify which specific actions led to failure. This results in noisy gradients and unstable training for long, multi-step tasks, often leading to slow convergence or failure to converge.
*   **Single response, verifiable outcome bias:** GRPO works best when a task involves a single response with one verifiable outcome.
