---
id: promptfoo:reinforcement-learning-with-verifiable-r
type: web
title: Reinforcement Learning with Verifiable Rewards Makes Models Smarter
url: https://www.promptfoo.dev/blog/rlvr-explained/
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-math-and-code
---

# Reinforcement Learning with Verifiable Rewards (RLVR)

### Core Problem
Reinforcement Learning with Verifiable Rewards (RLVR) addresses the inefficiency and subjectivity of learned reward models (used in RLHF) by replacing them with programmatic, deterministic verifiers. The primary goal is to improve model performance on tasks with ground-truth correctness (e.g., mathematics, coding, and SQL). A central theoretical tension in RLVR is whether the resulting gains stem from **capability expansion** (the model learning new reasoning paths) or **search compression** (the model becoming more efficient at sampling paths it could already generate).

### Method and Training Recipe
RLVR utilizes standard reinforcement learning with a deterministic reward function. The training loop follows these steps:

1.  **Candidate Generation:** For each prompt, the model generates $K$ candidate solutions.
2.  **Verification:** A programmatic verifier evaluates each output, returning a binary reward (1.0 for correct, 0.0 for incorrect).
3.  **Policy Update:** The policy is updated to favor high-reward trajectories.
4.  **Iteration:** The process repeats with new prompts.

#### Algorithm: Group Relative Policy Optimization (GRPO)
RLVR often employs GRPO, a value-free RL algorithm that eliminates the need for a value function by computing advantages relative to batch statistics. The advantage $A_i$ for a specific trajectory is calculated as:

$$
A_i = R(s_i, a_i) - \text{baseline}(R_{\text{group}})
$$

where $R(s_i, a_i)$ is the verifier's reward and the baseline is typically the mean or median of rewards within the group.

#### Data Efficiency Tactics (DEPO)
To reduce compute costs, DEPO improves efficiency through:
*   **Offline Curation:** Selecting examples where the base model's $\text{pass@k}$ is between 30% and 70%.
*   **Rollout Pruning:** Retaining only the top 50% of rollouts by reward.
*   **Difficulty Scheduling:** Gradually increasing problem difficulty.

### Key Quantitative Results
*   **Text2SQL:** Databricks reported 73.5% BIRD test accuracy in July, later increasing to 75.68% using execution-based verifiers and few-sample self-consistency.
*   **Data Efficiency:** DEPO achieved comparable performance to GRPO using only 20% of the training data, resulting in 1.85$\times$ and 1.66$\times$ speedups on AIME24/25 for R1-Distill Qwen-7B.
*   **Spurious Rewards:** Qwen2.5-Math-7B showed a 21.4% improvement on MATH-500 using *random* rewards, compared to a 29.1% gain from ground truth rewards.
*   **Search Compression Example:** In one analysis, a model's $\text{pass@1}$ rose from 40% to 65% (+25pp), while its $\text{pass@8}$ only rose from 75% to 77% (+2pp), indicating a 71% compression of the existing gap rather than a significant lift in the capability ceiling.

### Analysis of Gains: Sampler vs. Thinker
To distinguish between search efficiency and true learning, the source proposes a **compression ratio**:

$$
\text{Compression Ratio} = \frac{\text{RLVR pass@1} - \text{Base pass@1}}{\text{Base pass@k} - \text{Base pass@1}}
$$

A ratio $> 0.7$ suggests the model is primarily acting as an efficient sampler (search compression) rather than expanding its reasoning boundaries.

### Limitations and Failure Modes
RLVR is unsuitable for subjective tasks such as creative writing or brand voice. Three critical failure modes are identified:

1.  **Partial Verifiers:** If a verifier is not comprehensive (e.g., checking SQL syntax but not execution), models will learn to "cheat" by exploiting the gaps to earn rewards.
2.  **Spurious Rewards:** Some models (specifically Qwen2.5-Math) may show performance gains from random reward signals due to the training process itself encouraging internal pathway refinement, rather than learning from the reward.
3.  **Entropy Instability:** As entropy declines during GRPO training, models may overfit to the training distribution, causing in-distribution accuracy to rise while out-of-distribution (OOD) performance deteriorates.
