Reinforcement Learning with Verifiable Rewards (RLVR) is a paradigm that optimizes Large Language Models (LLMs) by utilizing deterministic reward signals—such as code compilers or mathematical ground-truth keys—rather than learned reward models. This approach is primarily used to elicit complex reasoning behaviors and Chain-of-Thought (CoT) refinement in domains where correctness is binary and verifiable [source:arxiv:2506.14245][source:arxiv:2504.13837].

## Core Mechanisms and the "Logic Prior"

The fundamental objective of RLVR is to maximize the expected verifiable reward: $\max_{\theta} \mathbb{E}_{p \sim \mathcal{D}}[R(M_\theta(p))]$ [source:arxiv:2504.13837]. In practice, this is often implemented via Group Relative Policy Optimization (GRPO), which computes token-level advantages based on the normalized rewards of a group of sampled responses for the same prompt [source:arxiv:2506.14245][source:arxiv:2508.14029].

A central theoretical question in RLVR is how answer-only rewards (the "verifiable dot") can incentivize the correctness of intermediate reasoning steps (the CoT). This is explained by the **Logic Prior assumption**, formulated as:
$$P(C_{ans}=1 \mid C_{cot}=1) > P(C_{ans}=1 \mid C_{cot}=0) \quad \text{(Eq. 4)} [source:arxiv:2506.14245]$$
where $C_{ans}$ and $C_{cot}$ are binary indicators of answer and CoT correctness, respectively. Under this assumption, GRPO implicitly amplifies the advantage gap between correct and incorrect CoTs, as correct reasoning paths are statistically more likely to lead to the verifiable reward [source:arxiv:2506.14245].

## The Capacity vs. Efficiency Debate

There is a significant disagreement in current research regarding whether RLVR expands the actual reasoning capabilities of a model or merely optimizes its sampling efficiency.

| Perspective | Core Argument | Evidence/Metric | Source |
| :--- | :--- | :--- | :--- |
| **Sampling Efficiency** | RLVR does not solve new problems; it just makes the model more likely to sample a path it already "knew" [source:arxiv:2504.13837]. | $\text{Pass@}k$ shows base models surpassing RLVR models as $k$ increases; RLVR-solved sets are subsets of base-solved sets [source:arxiv:2504.13837]. | [source:arxiv:2504.13837] |
| **Capacity Expansion** | RLVR fundamentally improves the quality of reasoning, creating paths that did not exist in the base model [source:arxiv:2506.14245]. | $\text{CoT-Pass@}k$ (requiring both answer and CoT correctness) shows RLVR models consistently outperforming base models even at high $k$ [source:arxiv:2506.14245]. | [source:arxiv:2506.14245] |

This conflict is further complicated by the observation that **1-shot RLVR**—training on a single high-variance example—can trigger massive gains in test accuracy. For instance, applying 1-shot RLVR to Qwen2.5-Math-1.5B elevates MATH500 performance from 36.0% to 73.6%, a result that matches the performance achieved by training on a larger 1.2k DeepScaleR subset [source:arxiv:2504.20571]. This suggests that the "knowledge" may be latent in the base model, and RLVR acts as a trigger for format correction and latent capacity activation [source:arxiv:2504.20571].

## The "RLVR Tax" and Empirical Pitfalls

Recent analysis suggests that reported gains from RLVR are often inflated due to several confounds, termed the "RLVR tax" [source:arxiv:2509.21882]:

1.  **Budget Mismatches:** Many reports compare RLVR models (sampled many times) against base models (sampled fewer times). When sampling budgets $k$ are matched, performance drops are substantial (e.g., Open-RS3-1.5B on AIME-24 dropped from 46.70 to 30.94) [source:arxiv:2509.21882].
2.  **Calibration Drift:** RLVR often reduces the model's tendency to abstain from hard problems, shifting risk from "I don't know" to "assertive errors" [source:arxiv:2509.21882]. For example, while DeepSeek-V3 abstains on 480 items, its RLVR-trained counterpart, DeepSeek-R1, attempts fewer items (only 81), though it improves Expected Calibration Error (ECE) [source:arxiv:2509.21882].
3.  **Entropy Collapse:** Training on fixed datasets leads to "policy entropy collapse," where the model memorizes correct solutions and generates homogeneous trajectories, causing $\text{Pass@}k$ to plateau [source:arxiv:2508.14029].

## Advanced RLVR Extensions

### 1. Variational and Zero-Data Synthesis
To combat entropy collapse and data bottlenecks, two primary methods have emerged:
*   **Self-play with Variational Problem Synthesis (SvS):** The model identifies "challenging" problems (group accuracy 12.5%–50%) and synthesizes variational problems that preserve the original semantics and answer [source:arxiv:2508.14029]. This sustains policy entropy and improves $\text{Pass@}32$ on AIME24 by 18.3% [source:arxiv:2508.14029].
*   **Absolute Zero (AZ):** A self-play paradigm where the model alternates between a **proposer** (generating tasks) and a **solver** (solving them), grounded in a code executor [source:arxiv:2505.03335]. It uses a learnability reward $r_{learn}(t) = 4 \cdot r_{sol}(t) \cdot (1 - r_{sol}(t))$ to ensure tasks are neither too easy nor too hard [source:arxiv:2505.03335].

### 2. Hybrid Guidance and Open-Ended Rewards
*   **FEST (FEw-ShoT RLVR):** To solve the "rare rollout" problem (where the model never hits the correct answer by chance), FEST combines GRPO on an RL dataset with a semi-online DPO loss on a small (e.g., 128-shot) SFT dataset [source:arxiv:2605.15012].
*   **RLVRR (Reference-based Rewards):** To apply RLVR to open-ended generation, RLVRR replaces the "verifiable dot" with a **reward chain**. It uses an LLM to extract verifiable keywords from references and computes a content reward based on the Longest Common Subsequence (LCS) [source:arxiv:2601.18533]:
    $$r_c = \frac{1}{K} \sum_{k=1}^K \max_{r \in \mathcal{R}} \frac{\text{LCS}(K^{r,r}_k, K^y_k)}{\max(|K^{r,r}_k|, |K^y_k|)} \quad \text{(Eq. 4)} [source:arxiv:2601.18533]$$

## Current status and trajectory

RLVR is currently a **rising** technique, particularly as the industry shifts toward "reasoning models" (e.g., DeepSeek-R1). However, the field is moving away from vanilla answer-only rewards due to the documented issues of entropy collapse [source:arxiv:2508.14029] and the "RLVR tax" [source:arxiv:2509.21882]. The trajectory suggests a transition toward **process-level rewards** (via reference chains [source:arxiv:2601.18533]) and **autonomous data synthesis** (via self-play [source:arxiv:2505.03335][source:arxiv:2508.14029]) to break the dependency on human-curated datasets.

## Key takeaways
*   **Mechanism:** RLVR leverages the "Logic Prior," where correct intermediate reasoning is statistically correlated with correct final answers, allowing answer-only rewards to implicitly optimize CoT [source:arxiv:2506.14245].
*   **Capacity Debate:** There is a fundamental disagreement on whether RLVR expands reasoning capacity; evidence depends on whether one uses $\text{Pass@}k$ (suggests efficiency) or $\text{CoT-Pass@}k$ (suggests capacity) [source:arxiv:2504.13837][source:arxiv:2506.14245].
*   **Efficiency:** 1-shot RLVR proves that minimal data can trigger significant latent reasoning capabilities, sometimes matching results of full-dataset training [source:arxiv:2504.20571].
*   **Risks:** "RLVR tax" includes calibration drift, budget-dependent performance inflation, and policy entropy collapse [source:arxiv:2509.21882][source:arxiv:2508.14029].
*   **Innovation:** New methods like AZ and SvS remove the need for external data through self-play and variational synthesis [source:arxiv:2505.03335][source:arxiv:2508.14029].

## Related topics
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)