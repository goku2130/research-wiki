---
id: youtube:deepseek-s-grpo-group-relative-policy-op
type: web
title: DeepSeek's GRPO (Group Relative Policy Optimization) - YouTube
url: https://www.youtube.com/watch?v=xT4jxQUl0X8
retrieved: '2026-07-12'
maturity: comprehensive
topic: grpo
---

The DeepSeek's Group Relative Policy Optimization (GRPO) is a reinforcement learning (RL) algorithm designed to efficiently train large language models (LLMs), particularly for reasoning tasks. It aims to reduce the computational cost typically associated with training such models, as demonstrated by DeepSeek's R1 model. GRPO is presented as an evolution of existing algorithms like Proximal Policy Optimization (PPO) and is specifically applied in the context of "reasoning fine-tuning" where a deterministic correctness signal is available.

### Core Problem

The core problem GRPO addresses is the high computational cost and complexity of training LLMs using traditional RL methods, especially for tasks where explicit reference answers are available (e.g., math, computer science). Traditional methods like Reinforcement Learning with Human Feedback (RLHF) and PPO often involve training additional reward models or complex state-value functions, leading to increased memory requirements, computational resources, and cognitive load for practitioners. GRPO seeks to simplify the RL training process for LLMs by reducing variance and computational overhead while maintaining or improving performance.

### Method/Recipe Step by Step

GRPO operates within the LLM training pipeline after initial pre-training and instruction fine-tuning. For reasoning tasks, it leverages a binary correctness signal as a reward.

1.  **Initial State and Actions**: An LLM (policy $\pi_\theta$) is given an instruction (initial state $S_0$). It then predicts tokens sequentially, with each token prediction being an action $A_t$ that updates the state $S_t$. This continues until an end token is predicted, concluding an episode.
2.  **Reward Assignment**: For reasoning tasks, a single final reward $R$ is assigned at the end of the episode. This reward is typically 1 for a correct answer and 0 for an incorrect answer, determined by comparing the model's final output against a ground truth using heuristics. Intermediate rewards are generally zero.
3.  **Policy Gradient Calculation**: GRPO is a policy gradient method. The goal is to update the model parameters $\theta$ to increase the probability of actions that lead to positive rewards. The fundamental policy gradient update is:

$$
\theta_{new} = \theta_{old} + \alpha \nabla_\theta \log \pi_\theta(A_t | S_t) R
$$

    where $\alpha$ is the learning rate.
4.  **Baseline Calculation (Group Relative)**: Unlike PPO which uses a state-value function (critic) to estimate a per-state baseline, GRPO computes a simpler, group-relative baseline.
    *   For a given instruction, the agent generates a full trajectory and receives a final reward.
    *   GRPO samples an additional group of trajectories (typically 4 to 8) for the same instruction and collects their final rewards.
    *   The baseline $B$ is the average reward of this group.
    *   The advantage $A_t$ (or simply $A$ in GRPO's simplified context) is calculated as the final reward minus the mean reward of the group, normalized by the standard deviation of the group rewards:

$$
A = \frac{R - \text{mean}(R_{group})}{\text{std}(R_{group})}
$$

5.  **PPO-based Loss Function**: GRPO builds upon the PPO loss function to constrain model updates and manage variance. The PPO loss uses a ratio $r_t(\theta)$ between the new and old policy probabilities:

$$
r_t(\theta) = \frac{\pi_\theta(A_t | S_t)}{\pi_{\theta_{old}}(A_t | S_t)}
$$

    The PPO objective function is:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
$$

    where $\hat{A}_t$ is the advantage estimate and $\epsilon$ is a hyperparameter (e.g., 0.2).
6.  **GRPO Regularization Term**: GRPO adds an extra regularization term to the PPO loss. This term is the KL divergence between the current policy $\pi_\theta$ and a "reference policy" $\pi_{ref}$. The reference policy is the state of the model *before* the reasoning fine-tuning stage (e.g., a preference fine-tuned model).

$$
L^{GRPO}(\theta) = L^{CLIP}(\theta) - \beta D_{KL}(\pi_\theta || \pi_{ref})
$$

    where $\beta$ is a coefficient for the regularization term. This term ensures the model stays close to its original state, compensating for the simplifications made in baseline estimation.
7.  **Gradient Ascent**: Both PPO and GRPO perform gradient ascent on their respective loss functions to update the policy parameters $\theta$.

### Key Formulas in LaTeX

*   **Policy Gradient Update (General)**:

$$
\theta_{new} = \theta_{old} + \alpha \nabla_\theta \log \pi_\theta(A_t | S_t) R
$$

*   **GRPO Advantage Calculation**:

$$
A = \frac{R - \text{mean}(R_{group})}{\text{std}(R_{group})}
$$

*   **PPO Clipped Objective Function**:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(\frac{\pi_\theta(A_t | S_t)}{\pi_{\theta_{old}}(A_t | S_t)} \hat{A}_t, \text{clip}\left(\frac{\pi_\theta(A_t | S_t)}{\pi_{\theta_{old}}(A_t | S_t)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t\right) \right]
$$

*   **GRPO Loss Function**:

$$
L^{GRPO}(\theta) = L^{CLIP}(\theta) - \beta D_{KL}(\pi_\theta || \pi_{ref})
$$

### Key Quantitative Results and Numbers

The video mentions that DeepSeek's R1 model, which uses GRPO, was "allegedly trained much more cheaply" and "at a fraction of the cost normally incurred by OpenAI." However, no specific quantitative cost savings or performance metrics for GRPO itself are provided in the transcript.

A comparison between "outcome supervision" (single final reward) and "process supervision" (intermediate rewards) on the GSM8K dataset (a middle school math questions dataset) showed "a little bit of Delta" in favor of process supervision. However, the presenter concludes that "these small gains don't justify the overhead," suggesting that the simpler outcome supervision is preferred for GRPO.

### Stated Limitations

*   **Simplifying Assumptions**: GRPO makes several simplifying assumptions compared to more complex RL algorithms like PPO, such as using a single final reward instead of per-step returns and a group-relative baseline instead of a state-value function. While these simplifications reduce computational overhead, they might introduce more variance.
*   **Lack of Intermediate Rewards**: The effectiveness of GRPO is particularly noted in scenarios where intermediate rewards are not readily available or are inaccurate (e.g., from reward models). In environments with constant, reliable intermediate feedback (like video games), the simplifications of GRPO might be less advantageous.
*   **Cognitive Load (PPO/Actor-Critic)**: While not a limitation of GRPO itself, the video highlights that actor-critic methods (like PPO) impose a significant "cognitive load for practitioners" due to their complexity, which GRPO aims to alleviate.
