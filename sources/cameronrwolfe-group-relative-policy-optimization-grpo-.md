---
id: cameronrwolfe:group-relative-policy-optimization-grpo-
type: web
title: Group Relative Policy Optimization (GRPO) - Deep (Learning) Focus
url: https://cameronrwolfe.substack.com/p/grpo
retrieved: '2026-07-10'
maturity: comprehensive
topic: grpo
---

**Core Problem**
Reinforcement learning (RL) has become central to aligning large language models (LLMs) to human preferences and enhancing their complex reasoning capabilities. Traditional RLHF pipelines rely on neural reward models and the Proximal Policy Optimization (PPO) optimizer, introducing significant computational overhead, training complexity, and susceptibility to reward hacking. To address these inefficiencies, particularly for training Large Reasoning Models (LRMs) via Reinforcement Learning with Verifiable Rewards (RLVR), researchers have adopted Group Relative Policy Optimization (GRPO). GRPO was developed to simplify the RL optimization process while preserving the stability and effectiveness required for large-scale reasoning training.

**Methodological Framework & Algorithmic Recipe**
The RL training paradigm for LLMs follows a consistent online loop that GRPO operationalizes through a streamlined recipe: (1) sample a diverse batch of prompts; (2) generate one or multiple completions per prompt using the current policy; (3) compute rewards for each completion using deterministic verifiers (e.g., string matching for math or sandbox execution for code) rather than learned reward models; and (4) update the policy weights via GRPO to maximize reward while constraining policy drift. This process is typically preceded by supervised finetuning (SFT) to establish a baseline. During RLVR, the model generates extended chain-of-thought reasoning trajectories wrapped in special tokens, which are evaluated against verifiable ground truths. GRPO builds directly upon PPO’s architecture but removes the need for a separate critic model and simplifies advantage estimation, making it more parameter-efficient and stable for reasoning-focused models.

**Key Formulations**
PPO’s surrogate objective forms the mathematical foundation for GRPO. The algorithm relies on the policy ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$, which is clipped to the interval $[1 - \epsilon, 1 + \epsilon]$ to enforce a trust region and prevent catastrophic policy updates. The surrogate objective is formulated as the minimum of the clipped and unclipped terms, creating a pessimistic lower bound that disincentivizes large probability shifts. Advantage estimation utilizes the standard definition $A(s, a) = Q(s, a) - V(s)$, where the value function $V(s)$ is typically approximated by a shared critic or value head. To maintain alignment with the initial supervised model, KL divergence is incorporated directly into the reward signal as a penalty term, approximated via the difference in log probabilities between the current and reference policies: $\text{KL} \approx \log \pi_{\theta_{old}} - \log \pi_\theta$. The adjusted reward becomes $R_{\text{adjusted}} = R - \beta \cdot \text{KL}$, ensuring the policy optimizes rewards without diverging excessively from its pre-trained distribution.

**Quantitative Results**
The provided source does not report specific quantitative metrics, performance percentages, or compute benchmarks for GRPO. It notes that LRMs trained with RLVR and GRPO exhibit predictable scaling laws relative to RL compute steps, and highlights the emergence of open reasoning models like DeepSeek-R1 and Kimi-K2 that match or exceed closed counterparts. However, no numerical results or empirical measurements are detailed in the text.

**Stated Limitations**
Despite its efficiency, GRPO and the broader LRM training paradigm face several constraints. LRMs optimized for verifiable domains exhibit performance bias, often underperforming in non-verifiable tasks such as creative writing, summarization, or knowledge retrieval. The extended chain-of-thought reasoning required for complex tasks increases inference latency, computational cost, and verbosity, making LRMs inefficient for simpler queries. Additionally, these models may suffer from alignment deficiencies, including poor instruction following or non-standard formatting, and can exhibit overthinking behaviors that introduce errors. The source emphasizes that RL training remains a complex, open research frontier, with ongoing challenges in balancing reward design, scaling laws, and cross-domain generalization.
