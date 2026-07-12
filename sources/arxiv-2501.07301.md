---
id: arxiv:2501.07301
type: paper
title: The Lessons of Developing Process Reward Models (arXiv)
url: https://arxiv.org/abs/2501.07301
retrieved: '2026-07-12'
maturity: comprehensive
topic: reward-modeling
---

# Summary: The Lessons of Developing Process Reward Models in Mathematical Reasoning

### Core Problem
The development of Process Reward Models (PRMs) for mathematical reasoning is hindered by two primary challenges: unreliable data annotation and biased evaluation. The authors identify that commonly used **Monte Carlo (MC) estimation**—which labels a step as correct if it leads to a correct final answer—introduces significant noise because policy models can generate correct answers from flawed intermediate steps. Additionally, the standard **Best-of-N (BoN)** evaluation metric is biased; it prioritizes the final answer over the reasoning process, causing PRMs to degrade into Outcome Reward Models (ORMs) that ignore intermediate errors if the final result is correct.

### Method/Recipe
To address these issues, the authors propose a **consensus filtering mechanism** and a comprehensive evaluation framework.

**1. Data Synthesis and Expansion**
*   **Generation:** For approximately 500,000 queries, 6–8 diverse responses are generated using Qwen2.5-Math-Instruct (7B and 72B).
*   **Step Splitting:** Responses are split into individual steps using the `\n\n` delimiter.
*   **MC Estimation:** For each step, 8 independent completions are sampled. A **hard label** is assigned: a step is "correct" if at least one completion yields the correct final answer; otherwise, it is "incorrect." All steps following the first incorrect step are removed.

**2. Consensus Filtering**
*   **LLM-as-a-Judge:** Qwen2.5-Instruct-72B is used as a critic to verify the reasoning process step-by-step.
*   **Filtering:** An instance is retained only if the LLM-as-a-judge and the MC estimation reach a consensus on the exact location of the error. This process preserves approximately 40% of the original data.

**3. Model Training**
*   **Architecture:** PRMs are initialized from Qwen2.5-Math-7B/72B-Instruct. The language modeling head is replaced with a scalar-value head consisting of two linear layers.
*   **Objective:** The model is trained as a binary classifier using cross-entropy (CE) loss on the last token of each step.

**4. Evaluation Framework**
*   **Response-level:** Best-of-N (BoN) sampling, where the final response score is the product of individual step scores.
*   **Step-level:** Use of **PROCESSBENCH** to measure the model's ability to localize the first erroneous step.

### Key Quantitative Results
The authors released **Qwen2.5-Math-PRM-7B** and **Qwen2.5-Math-PRM-72B**.

*   **BoN Performance (Best-of-8):** Using the Qwen2.5-Math-7B-Instruct policy model, Qwen2.5-Math-PRM-7B achieved an average accuracy of **67.6%**, surpassing the majority voting baseline (maj@8) of **66.2%**. The 72B version achieved **69.3%**.
*   **Step-wise Error Identification:** On PROCESSBENCH, Qwen2.5-Math-PRM-7B outperformed all open-source PRMs and GPT-4o-0806, although it remained behind o1-mini.
*   **Data Efficiency:** Consensus filtering achieved performance comparable to LLM-as-a-judge while using only 40% of the data.
*   **Evaluation Bias:** The authors found that some existing open-source PRMs had over **40%** of their minimum step scores concentrated on the final answer step, confirming a shift toward outcome-based assessment.

### Key Formulas
The overall solution score $S$ in the BoN evaluation is calculated as the product of the scores $s$ of all $k$ steps in the response:

$$
S = \prod_{i=1}^{k} s_i
$$

For MC estimation, the label $L$ for a step is determined by the empirical probability of success across $M$ completions:

$$
L = \begin{cases} 1, & \text{if } \sum_{j=1}^{M} \mathbb{I}(\text{completion}_j = \text{correct}) \ge 1 \\ 0, & \text{otherwise} \end{cases}
$$

### Stated Limitations
1.  **Performance Gap:** A significant gap remains between the PRM's performance and the BoN upper bound ($\text{pass@8}$).
2.  **RL Integration:** The best practices for utilizing these PRMs within reinforcement learning (RL) frameworks remain unexplored.
3.  **Human Data Utilization:** The method for efficiently integrating high-quality human-annotated data with weakly supervised methods is not yet fully developed.
