---
id: magazine:the-state-of-reinforcement-learning-for-
type: web
title: The State of Reinforcement Learning for LLM Reasoning - Ahead of AI
url: https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training
retrieved: '2026-07-12'
maturity: comprehensive
topic: rl-for-reasoning
---

The provided article, "The State of Reinforcement Learning for LLM Reasoning," by Sebastian Raschka, PhD, published on Ahead of AI on April 19, 2025, discusses the application of reinforcement learning (RL) to enhance the reasoning capabilities of Large Language Models (LLMs).

**Core Problem:**
The article identifies that scaling model size and data alone for LLMs, as seen with models like GPT-4.5 and Llama 4, is reaching its limits for improving reasoning capabilities. Competitors like xAI and Anthropic are integrating explicit reasoning features, suggesting a need for specialized training methods beyond conventional pre-training and supervised fine-tuning to achieve advanced reasoning. OpenAI's o3 model, which used 10x more training compute than o1, demonstrates that strategic investment in reinforcement learning can significantly improve reasoning.

**Method/Recipe Step by Step (RLHF for LLM Alignment):**
The article details the Reinforcement Learning with Human Feedback (RLHF) methodology, which serves as a foundation for reasoning-specific RL modifications. RLHF typically involves three steps:

1.  **RLHF Step 1: Supervised Fine-tuning (SFT)**
    *   **Objective:** To create a base model for subsequent RLHF fine-tuning.
    *   **Process:** Prompts are sampled, and human annotators write high-quality responses. This dataset is then used to fine-tune a pre-trained base LLM in a supervised manner.

2.  **RLHF Step 2: Creating a Reward Model (RM)**
    *   **Objective:** To automate the labor-intensive human ranking process by training a model to predict reward scores.
    *   **Process:**
        *   The SFT model generates multiple responses (e.g., four) for each prompt.
        *   Human annotators rank these responses based on preference.
        *   A reward model is designed and trained on this ranking dataset.
        *   The reward model typically originates from the SFT model, with its output layer (next-token classification) replaced by a regression layer with a single output node to predict a reward score.

3.  **RLHF Step 3: Fine-tuning via Proximal Policy Optimization (PPO)**
    *   **Objective:** To further align the SFT model using the reward model's feedback.
    *   **Process:** The SFT model (referred to as the "policy") is updated using the PPO algorithm, guided by reward scores from the reward model. This step involves several models:
        *   **Policy:** The LLM being trained (initially the SFT model).
        *   **Reward Model:** Predicts the actual reward for generated responses.
        *   **Critic (Value Model):** A trainable model that estimates the expected reward.
        *   **Reference Model (Original Policy):** A copy of the SFT model used to prevent the policy from deviating too much.

**PPO Algorithm Steps:**

1.  **Compute Ratio of Next-Token Probabilities:**
    *   Calculates the ratio of probabilities of actions (next tokens) from the current policy being trained (`new_policy_prob`) and the policy before the current update (`old_policy_prob`).
    *   Formula: $\text{ratio} = \text{new\_policy\_prob} / \text{old\_policy\_prob}$

2.  **Compute Raw Score (Advantage-Weighted Ratio):**
    *   Multiplies the `ratio` by the `advantage` of the action.
    *   `advantage` is calculated as the difference between the `actual_reward` (from the reward model) and the `expected_reward` (from the critic).
    *   Formula: $\text{raw\_score} = \text{ratio} \times \text{advantage}$
    *   Formula: $\text{advantage} = \text{actual\_reward} - \text{expected\_reward}$ (simplified, in reality involves Generalized Advantage Estimation (GAE)).

3.  **Compute Clipped Score:**
    *   Limits the `ratio` within a specified range (e.g., 0.8 to 1.2) to prevent overly large policy updates.
    *   Formula: $\text{clipped\_ratio} = \text{clamp}(\text{ratio}, 0.8, 1.2)$
    *   Formula: $\text{clipped\_score} = \text{clipped\_ratio} \times \text{advantage}$

4.  **Determine Final Score:**
    *   Takes the minimum of the `raw_score` and `clipped_score` to ensure cautious updates.
    *   If `advantage` is positive, it caps the reward to prevent over-trusting good results.
    *   If `advantage` is negative, it limits the penalty to avoid overreacting to single bad results.
    *   Formula: $\text{final\_score} = \min(\text{raw\_score}, \text{clipped\_score})$

5.  **Calculate Loss Function:**
    *   The `final_score` is maximized during training (by minimizing its negative).
    *   A Kullback-Leibler (KL) divergence penalty is added to keep the new policy close to a `reference_policy` (the original SFT model), preventing drastic changes.
    *   Formula: $\text{loss} = -\text{final\_score} + \beta \times \text{KL}(\text{new\_policy} || \text{reference\_policy})$
        *   $\beta$ is a hyperparameter controlling the penalty strength.

**Key Quantitative Results and Numbers:**
*   OpenAI's o3 reasoning model used **10x more training compute** compared to o1, demonstrating significant investment in compute for reasoning via RL methods.
*   The PPO clipping range is exemplified as **0.8 to 1.2** for the ratio of probabilities.

**Stated Limitations:**
*   The article simplifies the `advantage` computation, omitting the details of Generalized Advantage Estimation (GAE) to avoid bloating the explanation.
*   The PPO walkthrough is acknowledged by the author as potentially "overboard," suggesting its complexity might be a barrier for some readers.
