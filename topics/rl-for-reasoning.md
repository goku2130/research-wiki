---
title: RL for reasoning models
maturity: developing
updated: '2026-07-10'
sources:
- arxiv:2501.12948
- arxiv:2402.03300
- arxiv:2503.06639
- arxiv:2503.20783
- arxiv:2506.14245
- arxiv:2601.17223
- icml:demystifying-long-chain-of-thought-reaso
- arxiv:2510.08049
open_questions:
- How can verifiable process rewards (VPRMs) be scaled to open-ended domains that
  lack rigid, rule-based guidelines?
- To what extent does the "aha moment" in long CoT represent actual cognitive correction
  versus a stochastic search artifact of the RL objective?
- Can the reasoning boundary be extended in domains where binary verification is impossible
  (e.g., creative writing or general qualitative analysis)?
---

Reinforcement learning (RL) for reasoning models focuses on incentivizing the generation of long, self-correcting chains-of-thought (CoT) through verifiable reward signals. This paradigm shifts away from traditional human-preference alignment toward objective, rule-based verification to scale test-time compute and reasoning depth.

## Core Optimization: Group Relative Policy Optimization (GRPO)
To avoid the high memory overhead of critic models in PPO, reasoning models frequently employ Group Relative Policy Optimization (GRPO) [source:arxiv:2402.03300]. GRPO optimizes a clipped surrogate objective penalized by a KL divergence term relative to a reference policy $\pi_{ref}$ [source:arxiv:2501.12948]:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^G \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} \hat{A}_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

The advantage $\hat{A}_i$ is computed by normalizing rewards within a group of $G$ sampled outputs for the same prompt $q$, eliminating the need for a value function [source:arxiv:2501.12948]:
$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r}$$
Theoretically, when using verifiable binary rewards, GRPO acts as an adaptive weighted contrastive loss where correct outcomes are weighted inversely to the current probability of success $p_{t-1}$ [source:arxiv:2503.06639].

## Training Recipes for Reasoning (o1/R1-style)
The transition from base models to reasoning models typically follows one of two paths: pure RL or multi-stage pipelines.

### Pure RL (The R1-Zero Approach)
DeepSeek-R1-Zero demonstrates that reasoning capabilities can emerge from applying GRPO directly to a base model without supervised fine-tuning (SFT) [source:arxiv:2501.12948]. This approach utilizes a simple template enforcing `<think>` and `<answer>` tags and relies on rule-based accuracy rewards [source:arxiv:2501.12948]. While this produces "aha moments" (self-correction), it suffers from poor readability, language mixing, and chaotic formatting [source:arxiv:2501.12948].

### Multi-Stage Pipeline (The R1 Approach)
To stabilize the emergence of reasoning, a four-stage pipeline is utilized [source:arxiv:2501.12948]:
1. **Cold-Start SFT**: Fine-tuning on thousands of long CoT examples to establish human-readable priors.
2. **Reasoning RL**: GRPO with accuracy and language consistency rewards to fix language mixing.
3. **Rejection Sampling & SFT**: Generating $\sim 600\text{k}$ correct reasoning trajectories and $\sim 200\text{k}$ non-reasoning samples to retrain the base model.
4. **Final RL**: A general alignment stage for all scenario prompts.

Distillation of these patterns from a large R1 model into smaller dense models (e.g., Qwen/Llama) has been shown to outperform applying RL directly to those smaller models [source:arxiv:2501.12948].

## Verifiable Rewards and Process Supervision
A central tension in reasoning RL is whether to reward only the final outcome or the intermediate steps.

### Outcome-level Verification (RLVR)
Reinforcement Learning with Verifiable Rewards (RLVR) uses binary rewards based on programmatic correctness (e.g., compiler output or math answer) [source:arxiv:2506.14245]. This relies on the **Logic Prior assumption**: the probability of a correct answer given a correct CoT is higher than the probability of a correct answer given an incorrect CoT [source:arxiv:2506.14245]:
$$P(C_{Ans}=1 \mid C_{CoT}=1) > P(C_{Ans}=1 \mid C_{CoT}=0)$$

### Process-level Supervision (PRM/VPRM)
Process Reward Models (PRMs) provide dense, step-level signals [source:arxiv:2510.08049]. While neural PRMs are common, they are prone to reward hacking [source:arxiv:2601.17223]. Verifiable Process Reward Models (VPRMs) replace neural judges with deterministic, rule-based verifiers for specific domains (e.g., medical Cochrane guidelines), rewarding steps that satisfy explicit guidelines [source:arxiv:2601.17223].

## Technical Disagreements and Critical Perspectives

| Topic | Perspective A | Perspective B | Resolution/Nuance |
| :--- | :--- | :--- | :--- |
| **Reasoning Boundary** | RLVR expands the reasoning boundary, enabling the model to solve problems it couldn't previously [source:arxiv:2506.14245]. | RLVR merely improves sampling efficiency; correct paths already exist in pre-trained weights [source:arxiv:2506.14245]. | CoT-Pass@K metrics suggest RLVR maintains a gap over base models even at $K=1024$, supporting boundary expansion [source:arxiv:2506.14245]. |
| **Long CoT Emergence** | Long CoTs are a sign of emergent complex reasoning and self-correction [source:arxiv:2501.12948]. | Long CoTs may be an artifact of GRPO's length bias ($\mathcal{L} \propto 1/|\tau_i|$), inflating length for incorrect answers [source:arxiv:2503.20783]. | Dr. GRPO removes length-division to curb artificial inflation while maintaining performance [source:arxiv:2503.20783]. |
| **SFT Necessity** | SFT is a foundational requirement to provide the model with reasoning templates [source:arxiv:2501.12948]. | SFT is not an absolute prerequisite, though it improves sample efficiency [source:icml:demystifying-long-chain-of-thought-reaso]. | Base models (e.g., Qwen2.5) may already exhibit "SFT-like" behavior due to pretraining biases [source:arxiv:2503.20783]. |

## Quantitative Performance Benchmarks

| Model | AIME 2024 (Pass@1) | MATH-500 | Codeforces (Elo) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **R1-Zero** | 71.0% | - | - | Pure RL, matches o1-0912 [source:arxiv:2501.12948] |
| **R1** | 79.8% | 97.3% | 2029 | Surpasses o1-1217 [source:arxiv:2501.12948] |
| **DeepSeekMath-7B** | - | 51.7% | - | GRPO on math subset [source:arxiv:2402.03300] |
| **R1-Distill-32B** | 72.6% | 94.3% | - | Distilled from R1 [source:arxiv:2501.12948] |

## Current status and trajectory
The use of **GRPO and RLVR** is rapidly becoming the default for open-source reasoning models due to the efficiency of critic-free optimization and the availability of programmatic verifiers in STEM [source:arxiv:2402.03300][source:arxiv:2501.12948]. **Pure RL (Zero-style)** training is likely fading in favor of multi-stage pipelines (Cold-start SFT $\rightarrow$ RL) because the latter resolves critical stability and readability issues [source:arxiv:2501.12948]. **VPRMs** are a rising technique for structured, high-stakes domains (e.g., medicine) where deterministic rule-based verification is possible [source:arxiv:2601.17223], though they are not yet widely reported as a general-purpose replacement for outcome rewards in open-ended tasks.

## Key takeaways
* **GRPO** reduces memory by using group-relative advantages instead of a separate critic model [source:arxiv:2402.03300].
* **Verifiable rewards** (binary outcomes) can implicitly incentivize correct reasoning paths via the Logic Prior assumption [source:arxiv:2506.14245].
* **Multi-stage pipelines** (SFT $\rightarrow$ RL $\rightarrow$ Rejection Sampling $\rightarrow$ RL) are superior to pure RL for producing human-usable reasoning models [source:arxiv:2501.12948].
* **Length bias** in GRPO can create a false impression of emergent reasoning; removing response-length normalization (Dr. GRPO) can mitigate this [source:arxiv:2503.20783].
* **Distillation** is more effective than RL for transferring reasoning capabilities to small dense models [source:arxiv:2501.12948].

## Related topics
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)

## References
- [source:arxiv:2501.12948] [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [source:arxiv:2402.03300] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [source:arxiv:2503.06639] [Reinforcement Learning with Verifiable Rewards: GRPO's Effective Loss, Dynamics, and Success Amplification](https://arxiv.org/abs/2503.06639)
- [source:arxiv:2503.20783] [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
- [source:arxiv:2506.14245] [Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs](https://arxiv.org/abs/2506.14245)
- [source:arxiv:2601.17223] [Beyond Outcome Verification: Verifiable Process Reward Models for Structured Reasoning](https://arxiv.org/abs/2601.17223)
- [source:icml:demystifying-long-chain-of-thought-reaso] [Demystifying Long Chain-of-Thought Reasoning](https://icml.cc/virtual/2025/poster/45449)
- [source:arxiv:2510.08049] [A Survey of Process Reward Models: From Outcome Signals to Process Supervisions for Large Language Models](https://arxiv.org/abs/2510.08049)
