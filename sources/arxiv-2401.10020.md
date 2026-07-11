---
id: arxiv:2401.10020
type: paper
title: 'Self-Rewarding LLMs: A Framework for Aligning Language Models via Self-Generated
  Feedback'
url: https://arxiv.org/abs/2401.10020
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlaif
---

**Core Problem**
Current LLM alignment paradigms, including Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO), rely on frozen reward models trained from static human preference datasets. This architecture creates two critical bottlenecks: performance is capped at the quality of human annotators, and the reward signal cannot adapt or improve alongside the policy model during training. The authors identify the need for a self-improving feedback mechanism that eliminates the fixed reward model bottleneck, enabling continuous alignment through AI-generated preferences.

**Method/Recipe Step by Step**
The proposed Self-Rewarding framework operates via an iterative self-alignment loop:
1. **Initialization:** A base pretrained model $M_0$ is fine-tuned via Supervised Fine-Tuning (SFT) on two seed datasets: Instruction Fine-Tuning (IFT) data for general instruction following, and Evaluation Fine-Tuning (EFT) data to calibrate the model's LLM-as-a-Judge capability. This yields iteration model $M_1$.
2. **Self-Instruction Creation (Iteration $t$):** The current model $M_t$ generates a new prompt $x_i$ using few-shot prompting from the seed IFT data. It then samples $N=4$ diverse candidate responses $\{y_i^1, \ldots, y_i^N\}$ and evaluates them using its own LLM-as-a-Judge capability. Evaluation follows an additive five-point rubric across five criteria (relevance, coverage, usefulness, clarity, expertise), producing scores $r_i^n \in [0, 5]$.
3. **Preference Pair Construction:** The highest-scoring response $y_i^w$ and lowest-scoring response $y_i^l$ are paired with $x_i$ to form preference triplets $(x_i, y_i^w, y_i^l)$. Pairs with identical scores are discarded.
4. **Instruction Following Training:** The preference pairs constitute AI Feedback Training (AIFT) data. The model is updated via Direct Preference Optimization (DPO) on the augmented dataset to produce $M_{t+1}$.
5. **Iteration:** Steps 2–4 repeat, generating a model sequence $M_1 \to M_2 \to M_3$, where each iteration trains on self-generated feedback from the previous model.

**Key Formulas & Definitions**
The iterative training sequence is formally defined as:
$$M_0 \xrightarrow{\text{SFT (IFT+EFT)}} M_1 \xrightarrow{\text{DPO (AIFT}(M_1)\text{)}} M_2 \xrightarrow{\text{DPO (AIFT}(M_2)\text{)}} M_3$$
The reward assignment constraint is bounded by:
$$r_i^n \in [0, 5]$$
where scores are accumulated additively based on the LLM-as-a-Judge prompt criteria.

**Key Quantitative Results**
Iterative self-rewarding yields progressive gains across instruction following and reward modeling:
- **AlpacaEval 2.0 Win Rate (vs. GPT-4 Turbo):** $M_1$: 9.94%, $M_2$: 15.38%, $M_3$: 20.44%. $M_3$ surpasses Claude 2 (17.19%), Gemini Pro (16.85%), and GPT-4 0613 (15.76%).
- **Head-to-Head Win Rates (GPT-4 judged):** $M_2$ wins 55.5% against $M_1$; $M_3$ wins 47.7% against $M_2$.
- **MT-Bench Overall Score:** Increases from 6.85 (SFT baseline) to 7.25 ($M_3$), with largest gains in humanities, writing, and STEM categories.
- **Reward Modeling Accuracy (vs. Human Rankings):** Pairwise accuracy improves from 65.1% (SFT baseline) to 78.7% ($M_1$), 80.4% ($M_2$), and 81.7% ($M_3$).
- **Generation Length:** Average response length grows from 1,092 tokens ($M_1$) to 1,552 ($M_2$) and 2,552 ($M_3$).
- **Dataset Scale:** AIFT($M_1$) contains 3,964 preference pairs; AIFT($M_2$) contains 6,942 pairs.

**Stated Limitations**
The authors explicitly note several constraints: (1) Only three iterations were tested in a single setting, leaving scaling laws for extended training or different model capacities unexplored. (2) The observed increase in response length correlates with quality metrics, requiring deeper analysis to isolate length bias from genuine capability gains. (3) Potential reward-hacking risks exist due to the dual use of LLM-as-a-Judge for training and LLM-based evaluators for benchmarking, necessitating rigorous scrutiny. (4) Safety evaluations were not conducted, representing a critical gap for future safety-aligned self-rewarding frameworks. (5) Performance on standard NLP benchmarks (ARC, GSM8K, MMLU) shows slight regression or stagnation, indicating an alignment tax likely caused by the narrow distribution of the OpenAssistant seed data. The authors acknowledge that while self-improvement is promising, it will likely saturate in practical deployments.
