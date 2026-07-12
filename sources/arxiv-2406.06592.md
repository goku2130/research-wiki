---
id: arxiv:2406.06592
type: paper
title: Improve Mathematical Reasoning in Language Models by Automated Process Supervision
  (OmegaPRM)
url: https://arxiv.org/abs/2406.06592
retrieved: '2026-07-12'
maturity: comprehensive
topic: process-vs-outcome-rewards
---

# OmegaPRM: Automated Process Supervision for Mathematical Reasoning

### Core Problem
Large Language Models (LLMs) struggle with complex multi-step reasoning tasks, such as mathematics. While Outcome Reward Models (ORMs) verify final answers, they fail to penalize intermediate reasoning errors in lengthy chains of thought (CoT). Process Reward Models (PRMs) address this by providing granular, step-by-step feedback. However, collecting the necessary process supervision data is typically prohibitively expensive, relying either on labor-intensive human annotation or computationally inefficient per-step Monte Carlo (MC) estimation.

### Method
The authors propose **OmegaPRM**, a divide-and-conquer Monte Carlo Tree Search (MCTS) algorithm designed to automate the collection of high-quality process supervision data without human intervention.

#### 1. Monte Carlo Estimation and Binary Search
To determine the correctness of a partial solution prefix $x_{1:t}$, a "completer" policy samples $k$ rollouts from that step. The correctness level $c_t$ is estimated as:

$$
c_t = \text{MonteCarlo}(q, x_{1:t}) = \frac{\text{num}(\text{correct rollouts from } t\text{-th step})}{\text{num}(\text{total rollouts from } t\text{-th step})}
$$

To avoid the $O(kM)$ cost of brute-force estimation (where $M$ is the number of steps), OmegaPRM uses **binary search** to locate the first error. By splitting the solution at the midpoint and performing rollouts, the algorithm narrows the error location to $O(k \log M)$ complexity.

#### 2. OmegaPRM MCTS Algorithm
OmegaPRM organizes reasoning paths into a state-action tree where nodes are states (question + prefix) and edges are actions (reasoning steps).
*   **Tree Statistics:** Nodes store visit counts $N(s)$, MC estimations $\text{MC}(s)$, and a state-rollout value function $Q(s, r)$:

$$
Q (s, r) = \alpha^{1 - \mathrm{MC} (s)} \cdot \beta^{\frac{\mathrm{len} (r)}{L}}
$$

    This heuristic prioritizes "supposed-to-be-correct" wrong-answer rollouts (where $\text{MC}(s)$ is near 1 but the final answer is wrong) to help the PRM learn to detect subtle mistakes.
*   **Selection:** A rollout $(s, r)$ is selected from a pool where $0 < \text{MC}(s) < 1$ using a PUCT-variant:

$$
U(s) = c_{\mathrm{puct}} \frac{\sqrt{\sum_i N(s_i)}}{1 + N(s)}
$$

*   **Execution:** Binary search is performed on the selected rollout to identify the first error. New states and edges are added to the tree, and statistics are updated.

#### 3. PRM Training and Inference
The resulting state-action tree provides triplets of (question, partial solution, label). The PRM is trained using a pointwise classification loss:

$$
\mathcal{L}_{\mathrm{pointwise}} = \sum_{i=1}^{N} \hat{y}_i \log y_i + (1 - \hat{y}_i) \log(1 - y_i)
$$

The authors found that **pointwise soft labels** ($\hat{y} = \text{MC}(s)$) performed best. For inference, the PRM is combined with a weighted self-consistency decoding algorithm to rerank candidate solutions.

### Key Quantitative Results
OmegaPRM enabled the collection of over **1.5 million** process supervision annotations.

**Performance Gains (Weighted Majority Voting @ 64):**
*   **Gemini Pro:**
    *   MATH500: Increased from 51% to **69.4%**.
    *   GSM8K: Increased from 86.4% to **93.6%**.
*   **Gemma2 27B:**
    *   MATH500: Increased from 42.3% to **58.2%**.
    *   GSM8K: Increased from 74.0% to **92.2%**.

**Efficiency and Accuracy:**
*   **Computational Efficiency:** OmegaPRM demonstrated a **75-times efficiency improvement** over brute-force methods, generating 15 million data points compared to 200K for the same computational budget.
*   **PRM Accuracy:** The pointwise soft label objective achieved the highest per-step correctness classification accuracy at **70.1%**.

### Limitations
*   **Annotation Noise:** Automated process annotation introduces false positives and false negatives. While the PRM still outperformed those trained on human-annotated PRM800K data, the precise impact of this noise is uncertain.
*   **Dependency on Golden Answers:** The method requires a question and a golden answer pair to verify rollouts, limiting its current applicability to open-ended tasks.
