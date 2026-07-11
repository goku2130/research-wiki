---
id: arxiv:2401.10020
type: paper
title: Self-Rewarding Language Models (Zhang et al., 2024)
url: https://arxiv.org/abs/2401.10020
retrieved: '2026-07-11'
maturity: comprehensive
topic: rejection-sampling-and-bon
---

**Core Problem**
Current large language model alignment paradigms, including Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO), rely on frozen reward models trained from human preference data. This architecture creates a dual bottleneck: the reward signal is capped by human annotator performance, and the frozen reward model cannot adapt or improve alongside the language model during training. Consequently, these methods limit the trajectory toward superhuman agents that require continuously evolving, superhuman feedback signals.

**Method & Iterative Recipe**
The authors propose Self-Rewarding Language Models, which unify instruction generation and reward modeling into a single iterative training loop. The procedure begins with a base pretrained model $M_0$. First, the model undergoes Supervised Fine-Tuning (SFT) on seed Instruction Fine-Tuning (IFT) data and Evaluation Fine-Tuning (EFT) data, yielding the initial model $M_1$. In each subsequent iteration $t$, the model executes a Self-Instruction Creation phase: it generates novel prompts via few-shot prompting, produces $N$ diverse candidate responses, and evaluates them using an LLM-as-a-Judge prompt. This prompt assigns scores based on an additive five-criteria rubric (relevance, coverage, usefulness, clarity, expertise). The highest and lowest-scoring responses are paired to form preference tuples, discarding any ties. These pairs constitute AI Feedback Training (AIFT) data. The next model $M_{t+1}$ is then trained via DPO on the augmented dataset. This cycle repeats, allowing the model to iteratively refine both its generation and self-evaluation capabilities without external reward models.

**Key Formulas**
The iterative training sequence is formally defined as:
$$M_1 = \text{SFT}(M_0; \text{IFT} + \text{EFT})$$
$$M_{t+1} = \text{DPO}(M_t; \mathrm{AIFT}(M_t)) \quad \text{for } t \geq 1$$
The self-reward mechanism operates on a bounded additive scale where candidate responses receive scores $r_i^n \in [0, 5]$. The DPO optimization employs a learning rate of $10^{-6}$ decaying to $10^{-7}$, batch size 16, dropout 0.1, and a temperature parameter $\beta = 0.1$.

**Quantitative Results**
Evaluated on Llama 2 70B, the iterative approach yields consistent gains across instruction following and reward modeling. On AlpacaEval 2.0 (win rate against GPT-4 Turbo), $M_1$ achieves 9.94%, $M_2$ reaches 15.38%, and $M_3$ attains 20.44%, surpassing proprietary systems like Claude 2 (17.19%) and Gemini Pro (16.85%). Head-to-head evaluations confirm $M_2$ wins 55.5% against $M_1$, while $M_3$ wins 47.7% against $M_2$. Reward modeling alignment with human preferences improves steadily: pairwise accuracy rises from 65.1% (SFT baseline) to 78.7% ($M_1$), 80.4% ($M_2$), and 81.7% ($M_3$). MT-Bench scores increase from 6.85 to 7.25 across iterations. The pipeline generated 3,964 preference pairs for $M_2$ and 6,942 for $M_3$. Fine-grained analysis shows gains are strongest in humanities, writing, and roleplay, while mathematics and logical reasoning exhibit smaller improvements.

**Stated Limitations**
The authors characterize the findings as preliminary, noting only three iterations were tested in a single configuration. The scaling laws for extended iterative training remain unexplored. Generation length increases substantially across iterations (1092 to 2552 tokens), raising questions about length-quality correlations and potential reward-hacking. Safety evaluations are absent, though the authors suggest the self-rewarding framework could eventually enhance safety alignment. Additionally, the method exhibits an alignment tax on certain NLP benchmarks (e.g., ARC-Challenge, HellaSwag), likely due to the Open Assistant seed data lacking domain-specific reasoning tasks. The approach also currently relies on a fixed external model for prompt generation, though self-prompting is noted as a future direction.
