---
id: arxiv:2203.14465
type: paper
title: 'STaR: Bootstrapping Reasoning With Reasoning'
url: https://arxiv.org/abs/2203.14465
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

# STaR: Bootstrapping Reasoning With Reasoning

### Core Problem
Inducing "chain-of-thought" rationale generation in large language models (LLMs) typically requires either the construction of massive, expensive human-annotated rationale datasets or the use of few-shot prompting, which often yields lower accuracy than fine-tuning. The authors seek a method to iteratively bootstrap a model's ability to generate high-quality rationales using only a small set of initial examples and a large dataset of questions and answers without accompanying rationales.

### Method
The **Self-Taught Reasoner (STaR)** is an iterative bootstrapping loop designed to improve a model's reasoning capabilities. The process follows these steps:

1.  **Rationale Generation:** The model $M$ is prompted with a small set of few-shot examples containing rationales. It generates a rationale $\hat{r}_i$ and a final answer $\hat{y}_i$ for each problem $x_i$ in the dataset.
2.  **Filtering:** The model only retains rationales that lead to the correct ground-truth answer ($\hat{y}_i = y_i$).
3.  **Rationalization:** To prevent the model from plateauing (as it cannot learn from problems it fails to solve), the authors introduce "rationalization." For problems the model answered incorrectly, the correct answer $y_i$ is provided as a hint. The model then reasons backward to generate a rationale $\hat{r}_i^{\text{rat}}$ that justifies the correct answer. These are also filtered for correctness.
4.  **Fine-tuning:** The original pre-trained model $M$ is fine-tuned on the combined dataset of correct generated rationales and correct rationalized rationales.
5.  **Iteration:** The process repeats, using the newly improved model to generate a higher-quality training set for the next iteration.

The authors frame STaR as an approximation of a reinforcement learning (RL) policy gradient objective. The total expected reward $J$ across the dataset is:

$$
J (M, X, Y) = \sum_{i} \mathbb{E}_{\hat{r}_{i}, \hat{y}_{i} \sim p_{M}(\cdot | x_{i})} \mathbb{1}(\hat{y}_{i} = y_{i})
$$

The gradient is obtained via the log-derivative trick:

$$
\nabla J (M, X, Y) = \sum_{i} \mathbb{E}_{\hat{r}_{i}, \hat{y}_{i} \sim p_{M}(\cdot | x_{i})} \left[ \mathbb{1}(\hat{y}_{i} = y_{i}) \cdot \nabla \log p_{M}(\hat{y}_{i}, \hat{r}_{i} \mid x_{i}) \right]
$$

### Key Quantitative Results
The authors evaluated STaR using GPT-J (6B parameters) across three domains:

**CommonsenseQA (CQA)**
STaR significantly outperformed direct fine-tuning and few-shot baselines:
*   **Few-shot Direct GPT-J:** 20.9%
*   **Few-shot CoT GPT-J:** 36.6%
*   **GPT-J Direct Finetuned:** 60.0%
*   **STaR (without rationalization):** 68.8%
*   **STaR (with rationalization):** 72.5%
Notably, STaR with rationalization performed comparably to a GPT-3 model (30$\times$ larger) fine-tuned to predict answers directly (73.0%).

**Arithmetic (n-digit summation)**
*   **Overall Accuracy:** STaR achieved 89.5% accuracy after 16 iterations, compared to 76.3% for a baseline trained without rationales.
*   **Impact of Rationalization:** Without rationalization, the model improved stagewise (learning $n$-digit sums only after mastering $n-1$). With rationalization, 2-digit addition improved from $<1\%$ (few-shot) to 32% after a single fine-tuning iteration.

**Grade School Math (GSM8K)**
*   **Few-shot Direct GPT-J:** 3.0%
*   **GPT-J Direct Finetuned:** 5.8%
*   **STaR (without rationalization):** 10.1%
*   **STaR (with rationalization):** 10.7%

### Stated Limitations
*   **Initial Capability:** The model must have a baseline few-shot performance above chance to begin bootstrapping; for example, GPT-2 was unable to bootstrap even in the arithmetic domain.
*   **Chance Performance:** In settings with high chance performance (e.g., binary decisions), the model may generate many poor rationales that still lead to the correct answer, confounding the learning process.
*   **Hinting Complexity:** The method for providing "hints" during rationalization is not standardized and may be non-trivial to implement across different contexts.
