---
id: arxiv:2203.14465
type: paper
title: 'STaR: Self-Taught Reasoner: Bootstrapping Reasoning With Reasoning'
url: https://arxiv.org/abs/2203.14465
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

# STaR: Self-Taught Reasoner

**STaR (Self-Taught Reasoner)** is a bootstrapping technique designed to improve the reasoning capabilities of large language models (LLMs) by iteratively generating and learning from their own rationales.

### Core Problem
Generating "chain-of-thought" rationales improves LLM performance on complex reasoning tasks. However, inducing this behavior typically requires either the construction of massive, expensive human-annotated rationale datasets or the use of few-shot prompting, which often underperforms compared to models fine-tuned on larger datasets. STaR addresses this by leveraging a small number of initial rationale examples and a large dataset of questions and answers (without rationales) to bootstrap the model's ability to perform increasingly complex reasoning.

### Method
The STaR process is an iterative loop consisting of rationale generation, rationalization, and fine-tuning.

**Step-by-Step Recipe:**
1. **Rationale Generation:** Given a pretrained LLM $M$ and a dataset $\mathcal{D}$ of problems $x$ and answers $y$, the model is prompted with a small set of few-shot rationale examples $\mathcal{P}$. The model generates a rationale $\hat{r}_i$ and a predicted answer $\hat{y}_i$ for each problem $x_i$.
2. **Filtering:** The model only retains rationales that lead to the correct answer ($\hat{y}_i = y_i$).
3. **Rationalization:** To prevent the model from stagnating on problems it cannot solve, STaR employs "rationalization." For problems where the model failed to generate a correct answer, the ground-truth answer $y_i$ is provided as a hint. The model is then tasked with reasoning backward to generate a rationale $\hat{r}_i^{\text{rat}}$ that justifies the correct answer.
4. **Combined Filtering:** Rationalized rationales are filtered to ensure they actually lead to the correct answer.
5. **Fine-tuning:** The original pretrained model $M$ is fine-tuned on the combined dataset of correct rationales (both self-generated and rationalized).
6. **Iteration:** The process repeats using the newly fine-tuned model to generate the next training set until performance plateaus.

### Theoretical Framework
STaR is presented as an approximation to a reinforcement learning (RL) policy gradient objective. The model is viewed as a discrete latent variable model $p_M(y \mid x) = \sum_r p(r \mid x)p(y \mid x,r)$. The total expected reward across the dataset is:

$$
J (M, X, Y) = \sum_{i} \mathbb{E}_{\hat{r}_{i}, \hat{y}_{i} \sim p_{M}(\cdot | x_{i})} \mathbb{1}(\hat{y}_{i} = y_{i})
$$

The gradient is obtained via the log-derivative trick:

$$
\nabla J (M, X, Y) = \sum_{i} \mathbb{E}_{\hat{r}_{i}, \hat{y}_{i} \sim p_{M}(\cdot | x_{i})} \left[ \mathbb{1}(\hat{y}_{i} = y_{i}) \cdot \nabla \log p_{M}(\hat{y}_{i}, \hat{r}_{i} \mid x_{i}) \right]
$$

STaR approximates this by greedily decoding samples to reduce variance and taking multiple gradient steps on the filtered batch of correct rationales.

### Key Quantitative Results
The authors evaluated STaR using GPT-J (6B parameters) across three domains:

**CommonsenseQA (CQA) Dev Set Accuracy:**
* **Few-shot Direct GPT-J:** 20.9%
* **Few-shot CoT GPT-J:** 36.6%
* **GPT-J Direct Finetuned:** 60.0%
* **STaR (without rationalization):** 68.8%
* **STaR (with rationalization):** 72.5%
* *Comparison:* STaR with rationalization performed comparably to a 30$\times$ larger GPT-3 model fine-tuned for direct prediction (73.0%).

**Arithmetic (n-digit summation):**
* **STaR overall accuracy:** 89.5% (after 16 iterations).
* **Baseline (direct finetuning without rationales):** 76.3%.
* *Observation:* Rationalization allowed the model to learn multiple digit lengths simultaneously, whereas without it, the model improved stagewise (e.g., mastering $n-1$ digits before $n$ digits).

**GSM8K (Grade School Math) Test Accuracy:**
* **Few-shot Direct GPT-J:** 3.0%
* **GPT-J Direct Finetuned:** 5.8%
* **STaR (without rationalization):** 10.1%
* **STaR (with rationalization):** 10.7%

### Stated Limitations
* **Initial Capability:** The model must have a baseline reasoning ability; few-shot performance must be above chance for the first iteration to succeed. For example, GPT-2 was unable to bootstrap even in the arithmetic domain.
* **Chance Performance:** In settings with high chance performance (e.g., binary decisions), the model may generate many poor rationales that happen to lead to the correct answer, confounding the learning process.
* **Hinting Complexity:** The method for adding "hints" during rationalization is not universal and may be non-trivial to implement across different contexts.
