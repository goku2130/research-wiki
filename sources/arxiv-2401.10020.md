---
id: arxiv:2401.10020
type: paper
title: Self-Rewarding Language Model Generation via Self-Consistency and Rule-Based
  Rewards
url: https://arxiv.org/abs/2401.10020
retrieved: '2026-07-11'
maturity: comprehensive
topic: verifiable-rewards
---

**Core Problem**
Standard Large Language Model (LLM) alignment via Reinforcement Learning from Human Feedback (RLHF) or Direct Preference Optimization (DPO) relies on frozen reward models trained from human preference data. This architecture creates two critical bottlenecks: the reward signal is capped by human performance limits, and the frozen reward model cannot adapt or improve alongside the LLM during training. To overcome this ceiling, the authors propose *Self-Rewarding Language Models*, where the LLM itself generates and evaluates its own training data via LLM-as-a-Judge prompting, enabling a virtuous cycle of iterative self-improvement in both instruction following and reward modeling capabilities.

**Methodology**
The training pipeline operates iteratively, denoted as a model sequence $M_0 \to M_1 \to M_2 \to \ldots \to M_T$. Initialization begins with a base pretrained model $M_0$, which is fine-tuned on Instruction Fine-Tuning (IFT) and Evaluation Fine-Tuning (EFT) seed data via Supervised Fine-Tuning (SFT) to produce $M_1$. Each subsequent iteration $t$ follows a deterministic two-phase recipe:
1. **Self-Instruction Creation:** New prompts $x_i$ are generated via few-shot prompting from seed IFT data. The current model $M_t$ generates $N=4$ candidate responses $\{y_i^1, \ldots, y_i^N\}$ using sampling ($T=0.7, p=0.9$). These candidates are evaluated using an additive 5-point LLM-as-a-Judge prompt assessing relevance, coverage, usefulness, clarity, and expertise, yielding scores $r_i^n \in [0, 5]$. Evaluations are averaged over three decodes to mitigate variance.
2. **Instruction Following Training:** Preference pairs $(x_i, y_i^w, y_i^l)$ are constructed by selecting the highest and lowest scoring candidates, discarding ties. The resulting AI Feedback Training (AIFT) dataset is used to train $M_{t+1}$ via DPO ($\beta=0.1$, learning rate $10^{-6}$, batch size 16). This loop is repeated to yield $M_2$ and $M_3$.

**Quantitative Results**
Evaluated on Llama 2 70B, the iterative self-rewarding approach yields consistent gains across multiple axes. On AlpacaEval 2.0 (win rate against GPT-4 Turbo), performance rises from $9.94\%$ for $M_1$ to $15.38\%$ for $M_2$, and $20.44\%$ for $M_3$, surpassing Claude 2 ($17.19\%$), Gemini Pro ($16.85\%$), and GPT-4 0613 ($15.76\%$). Head-to-head evaluations confirm $M_2$ outperforms $M_1$ ($55.5\%$ vs $11.7\%$ wins), and $M_3$ outperforms $M_2$ ($47.7\%$ vs $12.5\%$ wins). Reward modeling capability against human rankings also improves: pairwise accuracy increases from $65.1\%$ (SFT baseline) to $78.7\%$ ($M_1$), $80.4\%$ ($M_2$), and $81.7\%$ ($M_3$). MT-Bench scores improve from $6.85$ to $7.25$. Notably, average generation length scales across iterations ($1092 \to 1552 \to 2552$ tokens). The AIFT datasets contain $3,964$ and $6,942$ preference pairs for training $M_2$ and $M_3$, respectively.

**Stated Limitations**
The authors explicitly note the findings are preliminary, with only three iterations tested and scaling laws for extended training or different model sizes unexplored. The observed increase in response length correlates with quality metrics, raising unresolved questions about potential reward-hacking or length bias. Comprehensive safety evaluations were not conducted, leaving safety alignment within the self-rewarding loop unverified. Furthermore, the method exhibits an alignment tax on standard NLP benchmarks (e.g., ARC, GSM8K, MMLU), where performance slightly regresses or stagnates, attributed to domain mismatch between the OpenAssistant seed data and these specialized tasks. The approach also demonstrates high sensitivity to prompt design, as alternative LLM-as-a-Judge formulations significantly degrade reward accuracy.
