---
id: arxiv:2406.06592
type: paper
title: Improve Mathematical Reasoning in Language Models by Automated Process Supervision
url: https://arxiv.org/abs/2406.06592
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-reasoning
---

Complex multi-step reasoning in large language models (LLMs) remains hindered by the limitations of outcome-based supervision. While Outcome Reward Models (ORMs) evaluate final answers, they fail to penalize or reward intermediate reasoning steps, which is critical for lengthy, multi-hop problems. Process Reward Models (PRMs) address this by providing granular, step-level feedback, but their development is bottlenecked by the prohibitive cost of human annotation and the computational inefficiency of brute-force per-step Monte Carlo estimation. To resolve this, the authors propose OmegaPRM, a fully automated, divide-and-conquer Monte Carlo Tree Search (MCTS) algorithm designed to efficiently generate high-quality process supervision data without human intervention.

The OmegaPRM method constructs a state-action tree where each node represents a partial Chain-of-Thought (CoT) solution and each edge denotes a reasoning step. The algorithm leverages binary search to rapidly locate the first incorrect step in a solution, reducing the time complexity of error localization from $O(kM)$ to $O(k \log M)$, where $k$ is the number of rollouts and $M$ is the total steps. During tree traversal, the algorithm maintains a pool of previously sampled rollouts and selects candidates using a heuristic that prioritizes "supposed-to-be-correct" wrong-answer trajectories. These rollouts are then expanded via binary search, and the resulting partial solutions are added to the tree. The PRM is subsequently trained on the tree edges using a pointwise classification loss with soft labels derived from Monte Carlo correctness estimates.

The core mathematical formulations underpinning OmegaPRM include the Monte Carlo estimation of a prefix solution's correctness:

$$
c_t = \text{MonteCarlo}(q, x_{1:t}) = \frac{\text{num}(\text{correct rollouts from } t\text{-th step})}{\text{num}(\text{total rollouts from } t\text{-th step})}
$$

The state-rollout value function guides rollout selection by balancing correctness proximity and length penalties:

$$
Q(s, r) = \alpha^{1 - \mathrm{MC}(s)} \cdot \beta^{\frac{\mathrm{len}(r)}{L}}
$$

Selection follows a PUCT-inspired exploration term:

$$
U(s) = c_{\mathrm{puct}} \frac{\sqrt{\sum_i N(s_i)}}{1 + N(s)}
$$

Finally, the PRM is optimized via a binary cross-entropy loss:

$$
\mathcal{L}_{\mathrm{pointwise}} = \sum_{i=1}^{N} \hat{y}_i \log y_i + (1 - \hat{y}_i) \log(1 - y_i)
$$

Quantitatively, OmegaPRM demonstrates substantial efficiency and performance gains. Under an identical computational budget, the algorithm generates 15 million data points compared to 200,000 via brute-force Monte Carlo, yielding a 75-fold efficiency improvement. The automated pipeline successfully collected over 1.5 million process supervision annotations. When integrated with weighted self-consistency decoding, the trained PRM significantly boosts mathematical reasoning. For the instruction-tuned Gemini Pro model, success rates increased from 51% to 69.4% on MATH500 and from 86.4% to 93.6% on GSM8K. Similarly, Gemma2 27B improved from 42.3% to 58.2% on MATH500 and from 74.0% to 92.2% on GSM8K. The PRM trained with pointwise soft labels achieved 70.1% accuracy on a dedicated step-level classification test set.

Despite these advances, the authors acknowledge two primary limitations. First, the automated annotation process introduces noise in the form of false positives and negatives, particularly when questions are too easy or too hard for the model, though empirical results show PRMs trained on this noisy data still outperform human-annotated baselines. Second, the method strictly requires both a question and a golden answer pair to initiate the MCTS and validate rollouts, which restricts its direct applicability to open-ended tasks lacking definitive ground-truth solutions.
