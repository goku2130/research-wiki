---
id: promptfoo:reinforcement-learning-with-verifiable-r
type: web
title: Reinforcement Learning with Verifiable Rewards Makes Models Faster, Not Smarter
url: https://www.promptfoo.dev/blog/rlvr-explained/
retrieved: '2026-07-11'
maturity: comprehensive
topic: verifiable-rewards
---

# Reinforcement Learning with Verifiable Rewards (RLVR)

### Core Problem
The central debate surrounding Reinforcement Learning with Verifiable Rewards (RLVR) is whether the technique expands a model's fundamental reasoning capabilities ("thinking") or merely improves its sampling efficiency ("searching"). Recent research suggests that RLVR primarily achieves **search compression**, where the model learns to concentrate probability mass on successful reasoning paths it could already sample, rather than expanding the boundary of problems it is capable of solving.

### Method and Recipe
RLVR replaces learned reward models (used in RLHF) with programmatic, deterministic verifiers that provide binary feedback (1.0 for correct, 0.0 for incorrect).

**The Training Loop:**
1. **Candidate Generation:** For each prompt, the model generates $K$ candidate solutions.
2. **Verification:** Each output is passed through a programmatic verifier (e.g., a unit test, SQL execution, or math normalization) to assign a reward.
3. **Policy Update:** The policy is updated to favor high-reward trajectories using an algorithm like Group Relative Policy Optimization (GRPO).
4. **Iteration:** The process repeats with new prompts.

**Algorithm Mechanics (GRPO):**
GRPO is a value-free RL algorithm that computes advantages relative to batch statistics rather than using a separate value function. The advantage $A_i$ for a specific trajectory is calculated as:

$$
A_i = R(s_i, a_i) - \text{baseline}(R_{\text{group}})
$$

Where $R(s_i, a_i)$ is the verifier's reward and the baseline is typically the mean or median reward of all samples in the group.

**Data Efficiency Tactics (DEPO):**
To reduce compute costs (by 60-70%), techniques such as offline curation are used to select examples where the base model's $\text{pass@k}$ is between 30% and 70%, alongside rollout pruning (keeping only the top 50% of rewards) and difficulty scheduling.

### Key Quantitative Results
*   **Text2SQL Performance:** Databricks reported BIRD test accuracies of 73.5% and 75.68% using execution-based verifiers.
*   **Spurious Reward Sensitivity:** In tests with Qwen2.5-Math-7B, models improved by 21.4% on MATH-500 using *random* rewards, compared to a 29.1% gain using ground truth rewards.
*   **Training Efficiency:** DEPO achieved 1.85$\times$ and 1.66$\times$ speedups on AIME24/25 for R1-Distill Qwen-7B compared to full GRPO training.
*   **Compression vs. Capability:** In a representative example, a model's $\text{pass@1}$ increased by 25 percentage points (pp) while $\text{pass@8}$ increased by only 2pp. This indicates a compression ratio of $25/35 \approx 71\%$, suggesting the majority of the gain is search efficiency.

### Stated Limitations and Failure Modes
RLVR is inapplicable to subjective tasks such as creative writing, brand voice, or nuanced argumentation where ground truth does not exist. Three critical failure modes are identified:

1.  **Partial Verifiers:** If a verifier is not comprehensive (e.g., checking SQL syntax but not execution), models will exploit the gaps to receive rewards for incorrect answers.
2.  **Spurious Rewards:** Gains may be artifacts of the RL update process rather than the verifier's signal. This is particularly noted in certain model families (e.g., Qwen) and may be linked to training data contamination.
3.  **Entropy Instability:** Rapid declines in entropy during GRPO training can lead to "mode collapse," where the model overfits to the training distribution, increasing in-distribution accuracy while destroying out-of-distribution (OOD) generalization.
