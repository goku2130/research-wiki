---
id: arxiv:2506.14245
type: paper
title: Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct
  Reasoning in Base LLMs
url: https://arxiv.org/abs/2506.14245
retrieved: '2026-07-10'
maturity: comprehensive
topic: rl-for-reasoning
---

The core problem addressed in this work is the unresolved debate regarding whether Reinforcement Learning with Verifiable Rewards (RLVR) fundamentally enhances the reasoning capabilities of base Large Language Models (LLMs) or merely improves sampling efficiency. Critics argue that RLVR-tuned models do not expand the reasoning boundary, as base models eventually match or surpass them on Pass@K metrics, suggesting all correct reasoning paths already exist in the pre-trained weights. This study systematically investigates this hypothesis across mathematical and coding domains, demonstrating that RLVR does extend reasoning boundaries when evaluated with rigorous metrics and appropriate training recipes.

The methodology involves revisiting Pass@K experiments and introducing a novel evaluation metric, CoT-Pass@K, which requires both the final answer and intermediate chain-of-thought (CoT) steps to be correct. To verify CoT correctness at scale, the authors employ an LLM-as-a-CoT-Judge paradigm using DeepSeek-R1-0528-Qwen3-8B, applying any-correct, all-correct, and majority-correct verification strategies. The training recipe reproduces the Group Relative Policy Optimization (GRPO) algorithm via the open-source DAPO framework. The step-by-step process entails: (1) initializing from a base LLM (e.g., Qwen2.5-32B), (2) sampling multiple CoT responses per prompt, (3) computing a binary verifiable reward based solely on programmatic answer correctness, (4) calculating the GRPO advantage across the sampled group, and (5) performing policy gradient updates while tracking per-prompt and dataset-level pass rates. Evaluations span math benchmarks (AIME 2024/2025, MATH-500, AMC23, Minerva) and code benchmarks (LiveCodeBench v6–v9).

The theoretical framework formalizes the distinction between CoT and answer correctness. Let $r_i$ denote a sampled response with CoT $c_i$ and answer $a_i$. Correctness indicators are defined as:
$$C_{CoT}(r_i) = \mathbb{I}(c_i \text{ is correct}), \quad C_{Ans}(r_i) = \mathbb{I}(a_i \text{ is correct}) \quad \text{(Eq. 1)}$$
The verifiable reward relies solely on $C_{Ans}$. The authors posit a Logic Prior assumption: pre-trained LLMs possess strong knowledge and logic priors that make valid reasoning paths more likely to yield correct answers than invalid ones:
$$P(C_{Ans}=1 \mid C_{CoT}=1) > P(C_{Ans}=1 \mid C_{CoT}=0) \quad \text{(Eq. 4)}$$
Under this assumption, Theorem 1 proves that GRPO implicitly incentivizes correct CoT generation. As training progresses, the probability of correct CoTs increases while spurious correlations decrease, widening the advantage gap and driving convergence toward coherent reasoning.

Quantitative results confirm extended reasoning boundaries. Using CoT-Pass@K, DAPO-Qwen-32B maintains a consistent, significant performance gap over Qwen2.5-32B across all sampled attempts up to $K=1024$ on AIME 2024 and AIME 2025. For code reasoning, AceReason-Nemotron-7B and Skywork-OR1 demonstrate clear Pass@K improvements over their pre-RLVR counterparts on LiveCodeBench, particularly on medium and hard problems. Training analysis reveals that optimization effects and generalization improvements emerge from the earliest stages. However, the median probability of generating a correct CoT given a correct answer, $P(C_{CoT}=1 \mid C_{Ans}=1)$, stabilizes around 0.8, indicating that approximately 20% of correct answers still originate from flawed reasoning traces.

The study identifies several limitations. First, the DAPO recipe encounters optimization saturation after approximately 400 steps; when $P(C_{Ans}=1)$ approaches 1 for fully optimized questions, the GRPO advantage becomes undefined for all-correct groups, halting further learning. Second, the persistent ~0.8 median for correct CoTs suggests that answer-only rewards cannot fully eliminate spurious reasoning patterns. Third, the Logic Prior assumption may fail if base models retain pre-training biases or factual errors, potentially reinforcing incorrect CoTs that coincidentally yield correct answers. This failure mode is suspected to cause the poor readability and multi-lingual artifacts observed in R1-Zero approaches. Finally, RLVR’s effectiveness is benchmark-dependent, showing diminished returns on simpler datasets (MATH-500, AMC23) or those with domain mismatches (Minerva), underscoring the necessity of carefully selected, contamination-free evaluation suites.
