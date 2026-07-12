---
id: arxiv:2312.08935
type: paper
title: 'Math-Shepherd: A Label-free Step-by-Step Verifier for Mathematical Reasoning
  (Wang et al., 2023)'
url: https://arxiv.org/abs/2312.08935
retrieved: '2026-07-12'
maturity: comprehensive
topic: process-vs-outcome-rewards
---

# Math-Shepherd: A Label-free Step-by-Step Verifier for Mathematical Reasoning

**Math-Shepherd** is a process-oriented reward model (PRM) framework designed to improve the mathematical reasoning of Large Language Models (LLMs) by assigning reward scores to individual reasoning steps.

### Core Problem
Existing Process Reward Models (PRMs) typically rely on expensive, labor-intensive human annotations to label the correctness of each intermediate step in a mathematical solution. While Outcome Reward Models (ORMs) can be trained automatically by checking the final answer, they lack the precision to identify exactly where a reasoning chain fails, which limits their effectiveness in reinforcement learning and complex verification.

### Method/Recipe
Math-Shepherd eliminates the need for human labels by defining the quality of an intermediate step as its potential to deduce the correct final answer. The automatic annotation process follows these steps:

1.  **Sampling:** For a given math problem with a golden answer $a^*$ and a step-by-step solution, a specific intermediate step $s_i$ is selected.
2.  **Completion:** A fine-tuned LLM, acting as a "completer," is used to decode $N$ subsequent reasoning paths from that step $s_i$, resulting in a set of finalized solutions and their corresponding answers $A = \{a_j\}_{j=1}^N$.
3.  **Estimation:** The quality of the step $y_{s_i}$ is estimated using one of two methods:
    *   **Hard Estimation (HE):** A step is labeled as "good" (1) if at least one decoded path reaches the correct answer; otherwise, it is "bad" (0).
    *   **Soft Estimation (SE):** The label is the frequency with which the step reaches the correct answer.
4.  **Training:** The PRM is trained as a binary classifier using these automatic labels.
5.  **Deployment:** The trained PRM is applied in two scenarios:
    *   **Verification:** Reranking $N$ candidate solutions (Best-of-N) by selecting the one with the highest reward.
    *   **Reinforcement Learning:** Using step-by-step Proximal Policy Optimization (PPO) to reinforce the generator.

### Key Formulas
The **Outcome Reward Model (ORM)** is trained with a cross-entropy loss:

$$
\mathcal{L}_{ORM} = y_s \log r_s + (1 - y_s) \log(1 - r_s)
$$

where $y_s$ is the golden label (1 if correct, 0 otherwise) and $r_s$ is the sigmoid score.

The **Process Reward Model (PRM)** extends this to each step $i$ across $K$ steps:

$$
\mathcal{L}_{PRM} = \sum_{i=1}^{K} y_{s_i} \log r_{s_i} + (1 - y_{s_i}) \log(1 - r_{s_i})
$$

The **Hard Estimation (HE)** label is defined as:

$$
y_{s_i}^{HE} = \begin{cases} 1 & \exists a_j \in A, a_j = a^* \\ 0 & \text{Otherwise} \end{cases}
$$

The **Soft Estimation (SE)** label is defined as:

$$
y_{s_i}^{SE} = \frac{\sum_{j=1}^{N} \mathbb{I}(a_j = a^*)}{N}
$$

For verification combining self-consistency and reward models, the final prediction $a_{sc+rm}$ is:

$$
a_{sc+rm} = \arg\max_{a} \sum_{i=1}^{N} \mathbb{I}(a_i = a) \cdot RM(p, \mathcal{S}_i)
$$

### Key Quantitative Results
Math-Shepherd demonstrates significant improvements across various open-source LLMs on the GSM8K and MATH benchmarks:

*   **Reinforcement Learning (PPO):** For Mistral-7B, step-by-step PPO improved accuracy from **77.9% $\to$ 84.1%** on GSM8K and **28.6% $\to$ 33.0%** on MATH.
*   **Verification (Best-of-256):**
    *   **Mistral-7B:** Accuracy reached **89.1%** (GSM8K) and **43.5%** (MATH).
    *   **DeepSeek-67B:** Achieved **93.3%** (GSM8K) and **48.1%** (MATH).
*   **Data Efficiency:** PRM outperformed ORM by approximately **4% accuracy** when using a small training dataset of 10k instances.
*   **Generalization:** In an out-of-distribution test on the Hungarian national final exam, the PRM outperformed the ORM by **9 scores**.

### Stated Limitations
1.  **Computational Cost:** The completion process requires decoding $N$ subsequent paths for every step to determine labels, which is resource-intensive.
2.  **Annotation Noise:** Automatic process annotations contain noise. While the model still outperforms some human-annotated datasets (like PRM800K) due to better data distribution and quantity, the full impact of this noise remains undetermined.
