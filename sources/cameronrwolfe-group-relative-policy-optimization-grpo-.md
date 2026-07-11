---
id: cameronrwolfe:group-relative-policy-optimization-grpo-
type: web
title: Group Relative Policy Optimization (GRPO) - Deep (Learning) Focus
url: https://cameronrwolfe.substack.com/p/grpo
retrieved: '2026-07-11'
maturity: comprehensive
topic: grpo
---

The author highlights that Group Relative Policy Optimization (GRPO) has become the prevalent Reinforcement Learning (RL) optimizer for enhancing the reasoning capabilities of Large Language Models (LLMs), particularly in the context of Reinforcement Learning with Verifiable Rewards (RLVR). While Proximal Policy Optimization (PPO) was historically used for aligning LLMs to human preferences via Reinforcement Learning from Human Feedback (RLHF), GRPO is now central to training Large Reasoning Models (LRMs) due to its efficiency and simplicity.

**Core Problem:**
The core problem addressed is the need for an efficient and effective RL optimizer to train LLMs for complex reasoning tasks, especially when verifiable rewards are available. Traditional RLHF, often using PPO, relies on reward models which introduce complexity, training overhead, and the risk of reward hacking. RLVR aims to mitigate these issues by using deterministic, verifiable rewards, but still requires a robust RL optimizer to translate these rewards into effective policy updates for LLMs.

**Method/Recipe Step by Step:**
GRPO operates within the general framework of online RL for LLMs, which involves:
1.  **Prompt Sampling and Completion Generation:** A batch of prompts is sampled, and the current LLM policy generates one or more completions for each prompt.
2.  **Reward Computation:** A reward is computed for each completion. In the context of RLVR, these rewards are derived from verifiable sources (e.g., ground truth answers, rule-based verifiers, or LLM judges for math/coding problems). This contrasts with RLHF, where rewards come from a learned reward model.
3.  **Policy Update:** GRPO is then used as the RL optimizer to compute the policy update, adjusting the LLM's weights to maximize the computed rewards. This process iteratively refines the LLM's ability to generate correct and verifiable responses.

The overall RLVR process with GRPO for training LRMs can be summarized as:
*   **Select a Verifiable Domain:** Choose a domain where correctness can be objectively verified (e.g., mathematics, coding).
*   **Create a Verifiable Dataset:** Develop a dataset with prompts and either known ground-truth answers or rule-based verification mechanisms.
*   **Instruct LLM for Parsable Output:** Guide the LLM to format its output such that the final answer can be easily parsed for verification.
*   **Iterative RL Training with GRPO:**
    *   Generate completions using the current LLM.
    *   Verify completions against ground truth or rules to obtain rewards.
    *   Apply GRPO to update the LLM's policy based on these rewards.

**Key Formulas in LaTeX:**
The source mentions the reward model loss function derived from the Bradley-Terry model for RLHF, but does not provide specific formulas for GRPO itself. The reward model loss function is given as:

$$
L(\theta) = -\sum_{(x, y_w, y_l) \in D} \log(\sigma(r_\theta(x, y_w) - r_\theta(x, y_l)))
$$

where:
*   $D$ is the preference dataset.
*   $(x, y_w, y_l)$ represents a prompt $x$, a preferred completion $y_w$, and a rejected completion $y_l$.
*   $r_\theta(x, y)$ is the score assigned by the reward model with parameters $\theta$ to completion $y$ for prompt $x$.
*   $\sigma$ is the sigmoid function.

**Key Quantitative Results and Numbers:**
The source states that GRPO is "more efficient and approachable than its predecessors" and has played a role in creating "more powerful reasoning models." It highlights its use in training "open reasoning models like DeepSeek-R1." However, no specific quantitative results (e.g., performance metrics, speedups, or model sizes) are provided for GRPO itself or models trained with it. The text mentions "over 72,000 subscribers" to the Deep (Learning) Focus newsletter.

**Stated Limitations:**
The source acknowledges that "RL training— *even with GRPO*—is a complex process that presents a seemingly endless frontier of open research questions." While GRPO is described as "refreshingly simple— *and effective*," the broader field of RL for LLMs, including RLVR, still faces challenges. Specifically, the limitations of RLVR mentioned include:
*   **Verifiable Domain Requirement:** RLVR is limited to domains where rewards can be deterministically verified, such as math or coding. This restricts its applicability to tasks without clear, objective correctness criteria.
*   **String Matching Limitations:** Simple string matching for verification is not always sufficient, requiring more robust validation logic or LLM judges to handle variations in output format.
*   **Ongoing Research:** The field is still active, with ongoing research into tweaking GRPO, scaling RLVR training, expanding to non-verifiable domains via rubrics, using curriculum learning, and combining verifiable and non-verifiable rewards.
