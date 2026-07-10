---
title: GRPO (Group Relative Policy Optimization)
maturity: developing
updated: '2026-07-10'
sources:
- arxiv:2402.03300
- arxiv:2501.12948
- nature:deepseek-r1-incentivizes-reasoning-in-ll
- arxiv:2503.14476
- arxiv:2605.21125
- arxiv:2603.01162
- huggingface:advanced-understanding-of-group-relative
- cameronrwolfe:group-relative-policy-optimization-grpo-
open_questions:
- Does the removal of the KL penalty in DAPO lead to better long-term stability or
  merely faster initial convergence compared to DeepSeek's constrained approach?
- How can GRPO be adapted for non-verifiable domains where rule-based rewards are
  unavailable without re-introducing the computational cost of a neural critic?
- To what extent does the "Oracle property" hold when using virtual samples (AVSPO)
  that introduce controlled bias into the gradient?
---

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm designed to scale reasoning capabilities in large language models (LLMs) by eliminating the critic model typically required in actor-critic frameworks [source:arxiv:2402.03300]. It optimizes policies by estimating the advantage baseline directly from the relative rewards of a group of sampled outputs for each prompt [source:huggingface:advanced-understanding-of-group-relative].

## Core Mechanism: Critic-Free Advantage Estimation
In traditional Proximal Policy Optimization (PPO), a separate value network (the critic) is trained to estimate the expected return $V(s)$, which is used to compute the advantage $A(s, a) = Q(s, a) - V(s)$ [source:cameronrwolfe:group-relative-policy-optimization-grpo-]. GRPO removes this critic model to reduce memory and compute overhead [source:arxiv:2402.03300][source:huggingface:advanced-understanding-of-group-relative].

### The GRPO Recipe
The algorithm operates through a three-step iterative process [source:huggingface:advanced-understanding-of-group-relative]:
1. **Group Sampling:** For each input prompt $q$, the model samples a group of $G$ distinct completions $\{o_1, o_2, \dots, o_G\}$ from the current policy $\pi_\theta$ [source:huggingface:advanced-understanding-of-group-relative].
2. **Relative Advantage Calculation:** Each output is assigned a reward $r_i$ (e.g., via a rule-based verifier for math correctness) [source:huggingface:advanced-understanding-of-group-relative]. The advantage $\hat{A}_i$ is then computed by standardizing the reward within the group:
   $$\hat{A}_i = \frac{r_i - \mu_{group}}{\sigma_{group} + \epsilon}$$
   where $\mu_{group}$ and $\sigma_{group}$ are the mean and standard deviation of rewards within the group, and $\epsilon$ (e.g., $10^{-8}$) ensures numerical stability [source:huggingface:advanced-understanding-of-group-relative][source:arxiv:2605.21125].
3. **Policy Update:** The policy is updated using a clipped surrogate objective that incorporates a Kullback-Leibler (KL) divergence penalty to prevent the model from diverging too far from a reference policy $\pi_{ref}$ [source:huggingface:advanced-understanding-of-group-relative][source:arxiv:2501.12948].

## Mathematical Framework
The GRPO objective function $\mathcal{J}(\theta)$ maximizes the following:
$$\mathcal{J}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^G \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} \hat{A}_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) - \beta D_{KL}(\pi_\theta \| \pi_{ref}) \right]$$
where $\epsilon$ is the clipping hyperparameter and $\beta$ is the KL penalty coefficient [source:arxiv:2501.12948].

### Theoretical Properties
Recent analysis has mapped the GRPO policy gradient to classical **U-statistics** [source:arxiv:2603.01162]. Using Hoeffding decomposition, research demonstrates that:
* **Oracle Property:** GRPO is asymptotically equivalent to an algorithm utilizing a true critic network [source:arxiv:2603.01162].
* **Convergence:** The gradient estimator is consistent, and the suboptimality gap can be characterized without requiring parameter identifiability, which is critical for overparameterized LLMs [source:arxiv:2603.01162].
* **Scaling Law:** There exists a universal scaling law for the optimal group size $G$ that depends on model architecture and training data, rather than the training budget [source:arxiv:2603.01162].

## Implementation Pipelines: DeepSeek-R1
DeepSeek has utilized GRPO in two distinct configurations to incentivize reasoning [source:arxiv:2501.12948]:

| Feature | DeepSeek-R1-Zero | DeepSeek-R1 |
| :--- | :--- | :--- |
| **Cold Start** | None (Pure RL) | SFT on long CoT examples |
| **Reward Signal** | Rule-based (Accuracy + Format) | Accuracy + Language Consistency |
| **Pipeline** | Single-stage RL | Multistage: SFT $\rightarrow$ RL $\rightarrow$ Rejection Sampling $\rightarrow$ RL |
| **Output Quality** | Poor readability, language mixing | High readability, aligned |
| **AIME 2024 (pass@1)** | 15.6% $\rightarrow$ 71.0% | 79.8% |

**DeepSeek-R1-Zero** demonstrates that pure RL can induce the emergence of "self-correction" and "aha moments" in reasoning, though it suffers from chaotic outputs [source:arxiv:2501.12948]. **DeepSeek-R1** mitigates this via a "cold start" SFT phase and rejection sampling to refine the reasoning trajectories before final alignment [source:arxiv:2501.12948].

## Scaling Challenges and Mitigations
Despite its efficiency, GRPO faces several failure modes during large-scale deployment.

### 1. Advantage Collapse
**Problem:** When all outputs in a group receive the same reward (e.g., all correct or all incorrect), the variance $\sigma_{group}$ vanishes, causing advantages and gradients to collapse to zero [source:arxiv:2605.21125]. This affects 28–45% of training batches in math tasks [source:arxiv:2605.21125].
**Mitigation (AVSPO):** Adaptive Virtual Sample Policy Optimization (AVSPO) monitors the Advantage Collapse Rate (ACR) and injects synthetic virtual reward samples into collapsed groups to restore non-zero variance [source:arxiv:2605.21125].

### 2. Entropy Collapse and Reward Noise
**Problem:** Naive GRPO can suffer from vanishing policy diversity (entropy collapse) and gradient instability when prompts achieve 100% accuracy [source:arxiv:2503.14476].
**Mitigation (DAPO):** Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) introduces:
* **Clip-Higher:** Decouples upper/lower clipping bounds to allow low-probability tokens to explore more freely [source:arxiv:2503.14476].
* **Dynamic Sampling:** Filters out prompts with 0% or 100% accuracy to ensure every batch contains a learning signal [source:arxiv:2503.14476].
* **Token-Level Loss:** Averages gradients per token to prevent excessive sequence length growth [source:arxiv:2503.14476].

## Technical Disagreements
Sources differ on the role of the KL divergence penalty in reasoning tasks:
* **The Stability View:** DeepSeek-R1 and DeepSeekMath utilize the KL penalty ($\beta D_{KL}$) as a critical constraint to maintain training stability and prevent the policy from diverging into nonsensical outputs [source:arxiv:2501.12948][source:huggingface:advanced-understanding-of-group-relative].
* **The Divergence View:** DAPO reports that long-chain-of-thought reasoning inherently diverges from the base model; consequently, they propose **removing the KL penalty entirely** to facilitate the emergence of reasoning capabilities [source:arxiv:2503.14476].

## Current Status and Trajectory
GRPO is currently **rising** and is becoming a foundational algorithm for Reinforcement Learning with Verifiable Rewards (RLVR) [source:arxiv:2603.01162]. Its adoption is driven by the success of the DeepSeek-R1 series and the theoretical validation of its "oracle property" [source:arxiv:2603.01162]. While it is the default for verifiable reasoning (math, code), its application to non-verifiable, subjective tasks (creative writing) is not widely reported and remains a challenge due to the lack of objective reward signals [source:cameronrwolfe:group-relative-policy-optimization-grpo-][source:huggingface:advanced-understanding-of-group-relative].

## Key Takeaways
* **Critic-Free:** Replaces the value network with a group-relative mean reward $\mu_{group}$ to estimate advantages [source:huggingface:advanced-understanding-of-group-relative].
* **Efficiency:** Significantly reduces memory overhead, making large-scale RL feasible for reasoning models [source:arxiv:2402.03300].
* **Theoretic Basis:** Mathematically equivalent to a U-statistic, providing asymptotic guarantees similar to actor-critic models [source:arxiv:2603.01162].
* **Failure Modes:** Susceptible to "Advantage Collapse" (zero variance) and "Entropy Collapse," requiring adaptations like AVSPO and DAPO [source:arxiv:2605.21125][source:arxiv:2503.14476].
* **Requirement:** Primarily effective for tasks with verifiable ground truths (RLVR) [source:cameronrwolfe:group-relative-policy-optimization-grpo-].

## References
- [source:arxiv:2402.03300] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [source:arxiv:2501.12948] [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [source:nature:deepseek-r1-incentivizes-reasoning-in-ll] [DeepSeek-R1 incentivizes reasoning in LLMs through pure reinforcement learning (Nature)](https://www.nature.com/articles/s41586-025-09422-z)
- [source:arxiv:2503.14476] [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [source:arxiv:2605.21125] [Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation](https://arxiv.org/abs/2605.21125)
- [source:arxiv:2603.01162] [Demystifying Group Relative Policy Optimization: Its Policy Gradient is a U-Statistic](https://arxiv.org/abs/2603.01162)
- [source:huggingface:advanced-understanding-of-group-relative] [Advanced Understanding of Group Relative Policy Optimization (GRPO) in DeepSeekMath](https://huggingface.co/learn/llm-course/chapter12/3b)
- [source:cameronrwolfe:group-relative-policy-optimization-grpo-] [Group Relative Policy Optimization (GRPO) - Deep (Learning) Focus](https://cameronrwolfe.substack.com/p/grpo)
