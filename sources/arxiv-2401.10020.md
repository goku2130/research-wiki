---
id: arxiv:2401.10020
type: paper
title: Self-Rewarding Language Models
url: https://arxiv.org/abs/2401.10020
retrieved: '2026-07-11'
maturity: comprehensive
topic: agentic-and-tool-use-rl
---

**Core Problem**
Current large language model (LLM) alignment paradigms, including Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO), rely on human preference datasets or frozen reward models. These approaches are fundamentally bottlenecked by human performance ceilings and computational constraints. Because the reward model is trained separately and kept static during alignment, it cannot adapt or improve as the policy model evolves, preventing continuous self-improvement and limiting the potential for superhuman agent development.

**Method/Recipe Step by Step**
The authors introduce *Self-Rewarding Language Models*, a unified framework where the LLM simultaneously generates responses and evaluates them via LLM-as-a-Judge prompting. The training proceeds iteratively using AI Feedback (AIF):
1. **Initialization:** A base pretrained LLM ($M_0$) is supervised fine-tuned (SFT) on two seed datasets: Instruction Fine-Tuning (IFT) data for general response generation, and Evaluation Fine-Tuning (EFT) data containing LLM-as-a-Judge examples with chain-of-thought justifications and additive 5-point scores.
2. **Self-Instruction Creation:** For each iteration $t$, the current model $M_t$ generates novel prompts via few-shot prompting from seed data. It then produces $N$ candidate responses and evaluates them using its own LLM-as-a-Judge capability, assigning scores $r_i^n \in [0, 5]$ based on five additive criteria (relevance, coverage, usefulness, clarity, and expertise).
3. **Preference Optimization:** The highest- and lowest-scoring responses form preference pairs $(x_i, y_i^w, y_i^l)$. These pairs, augmented with the original seed data, constitute AI Feedback Training (AIFT) data. The model is updated via Direct Preference Optimization (DPO) to yield $M_{t+1}$. This cycle repeats, allowing the reward modeling capability to improve alongside instruction following.

**Key Formulas**
The iterative training sequence is formally defined as:
- $M_0$: Base pretrained LLM.
- $M_1$: Initialized with $M_0$, fine-tuned on IFT + EFT data via SFT.
- $M_{t+1}$: Initialized with $M_t$, trained on $\mathrm{AIFT}(M_t)$ data via DPO.
The reward mechanism employs an additive scoring framework where points are accumulated across quality criteria, culminating in a final formatted output: `Score: <total points>`. While the explicit DPO loss function is not detailed in the source, the update rule is strictly defined as $M_{t+1} \leftarrow \text{DPO}(M_t, \mathrm{AIFT}(M_t))$.

**Key Quantitative Results**
Evaluated on Llama 2 70B, the iterative approach yields substantial, monotonic improvements:
- **AlpacaEval 2.0** (win rate vs. GPT-4 Turbo): $M_1$ achieves 9.94%, $M_2$ reaches 15.38%, and $M_3$ attains 20.44%, surpassing Claude 2 (17.19%), Gemini Pro (16.85%), and GPT-4 0613 (15.76%).
- **Head-to-Head Win Rates:** $M_2$ wins 55.5% against $M_1$ (11.7%), while $M_3$ wins 47.7% against $M_2$ (12.5%).
- **Reward Modeling Alignment:** Pairwise accuracy against human rankings improves from 65.1% (SFT baseline) to 78.7% ($M_1$), 80.4% ($M_2$), and 81.7% ($M_3$).
- **MT-Bench Scores:** Overall scores rise from 6.85 (SFT) to 7.25 ($M_3$).
- **Dataset Scaling:** Self-instruction creation adds 3,964 preference pairs for training $M_2$ and 6,942 pairs for $M_3$.
- **Generation Length:** Average response lengths expand from 1,092 tokens ($M_1$) to 1,552 ($M_2$) and 2,552 ($M_3$).

**Stated Limitations**
The authors characterize the findings as preliminary, noting that only three training iterations were tested in a single setting, leaving scaling laws unexplored. The observed increase in generation length correlates with quality estimates but requires deeper analysis to rule out reward-hacking or length bias. Safety evaluations were not conducted, representing a critical gap for future alignment work. Furthermore, gains in mathematical and logical reasoning tasks are comparatively modest, attributed to underrepresentation in the Open Assistant seed data. Finally, the method exhibits an "alignment tax" on standard NLP benchmarks (e.g., ARC, HellaSwag, GSM8K), where performance slightly regresses compared to the base model, highlighting the need for broader seed data distributions to preserve general capabilities.
