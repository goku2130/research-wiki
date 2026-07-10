---
id: arxiv:2203.02155
type: paper
title: Training language models to follow instructions with human feedback
url: https://arxiv.org/abs/2203.02155
retrieved: '2026-07-10'
maturity: comprehensive
topic: reward-modeling
---

**Core Problem**
Large language models (LLMs) optimized for next-token prediction on web text exhibit fundamental misalignment with user intent. Despite scaling parameter counts, these models frequently generate untruthful, toxic, biased, or unhelpful outputs and often ignore explicit instructions. The standard language modeling objective diverges from the desired goal of following user instructions helpfully and safely, necessitating targeted alignment techniques.

**Methodology**
The authors implement a three-stage fine-tuning pipeline to align GPT-3 architectures with human preferences:
1. **Supervised Demonstration Fine-Tuning:** Contractors provide human-written demonstrations of desired behavior across a diverse prompt distribution (primarily sourced from OpenAI API usage, filtered for personally identifiable information). A pretrained GPT-3 model is fine-tuned on this dataset via supervised learning.
2. **Reward Model Training:** A separate dataset of pairwise human comparisons is collected, where labelers rank model outputs for identical prompts. A reward model is trained to predict human-preferred outputs from this comparison data.
3. **Reinforcement Learning from Human Feedback (RLHF):** The reward model serves as a scalar reward function. The supervised policy is further optimized using the Proximal Policy Optimization (PPO) algorithm to maximize reward. This pipeline can iterate, continuously collecting comparison data on updated policies to refine the reward model and policy.

**Key Formulas**
The provided excerpt details the algorithmic workflow but does not explicitly present mathematical equations. The optimization framework relies on PPO to maximize the expected reward predicted by the trained reward model. To mitigate performance regressions on standard benchmarks, the authors introduce a hybrid update strategy (PPO-ptx) that combines PPO reward maximization with gradient updates that increase the log-likelihood of the pretraining distribution.

**Quantitative Results**
The study trains 1.3B, 6B, and 175B parameter models. Human evaluations reveal that the 1.3B InstructGPT model is preferred over the 175B GPT-3 baseline despite having over 100× fewer parameters. The 175B InstructGPT model outperforms standard 175B GPT-3 in 85.3% of head-to-head comparisons and beats few-shot 175B GPT-3 in 71.4% of cases. Regarding truthfulness, InstructGPT generates accurate answers approximately twice as often as GPT-3 on the TruthfulQA benchmark. For closed-domain tasks, hallucination rates decrease from 41% in GPT-3 to 21% in InstructGPT. Toxicity is reduced by approximately 25% when models are prompted to be respectful, though no significant improvements are observed on standard bias benchmarks (Winogender, CrowSPairs). The training datasets comprise ~13k prompts for supervised fine-tuning, ~33k for reward modeling, and ~31k for RLHF. Additionally, models exhibit strong generalization to held-out labelers and zero-shot domains like code summarization and non-English instructions, despite minimal exposure during fine-tuning.

**Stated Limitations**
Despite significant improvements, InstructGPT models still commit simple errors, including instruction failure, factual fabrication, excessive hedging, and inability to detect false premises. The alignment process reflects the specific preferences of the contractor labelers and researchers rather than universal human values. Furthermore, fine-tuning on public instruction datasets (FLAN, T0) yields inferior alignment compared to human preference data. The authors acknowledge that reducing toxicity does not automatically resolve bias, and that broader evaluation across diverse user populations and ambiguous preference scenarios remains necessary. Performance regressions on tasks like SQuAD and DROP occur during RLHF but can be partially offset by combining reward maximization with pretraining likelihood objectives.
