---
id: arxiv:2203.14465
type: paper
title: 'STaR: Self-Taught Reasoner: Bootstrapping Reasoning With Reasoning'
url: https://arxiv.org/abs/2203.14465
retrieved: '2026-07-11'
maturity: comprehensive
topic: self-improvement-and-self-play
---

# STaR: Self-Taught Reasoner

The **Self-Taught Reasoner (STaR)** is a bootstrapping technique designed to improve the reasoning capabilities of large language models (LLMs) by iteratively leveraging a small set of rationale examples and a large dataset of questions and answers without rationales.

### Core Problem
Inducing "chain-of-thought" (CoT) rationale generation typically requires either the construction of massive, expensive human-annotated rationale datasets or the use of few-shot prompting, which often substantially underperforms compared to models fine-tuned on larger datasets. STaR addresses this by allowing a model to improve itself through its own generated reasoning.

### Method
STaR employs an iterative loop to bootstrap reasoning. Given a pretrained LLM $M$, a dataset of problems and answers $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^D$, and a small prompt set $\mathcal{P}$ of examples with rationales, the process is as follows:

1.  **Rationale Generation:** The model is prompted with $\mathcal{P}$ to generate a rationale $\hat{r}_i$ and a final answer $\hat{y}_i$ for each problem $x_i$ in $\mathcal{D}$.
2.  **Filtering:** The model only keeps rationales that lead to the correct ground-truth answer ($\hat{y}_i = y_i$).
3.  **Rationalization:** To prevent the model from stagnating on problems it cannot solve, STaR uses "rationalization." For problems where the model failed to generate the correct answer, the correct answer $y_i$ is provided as a hint. The model then reasons backward to generate a rationale $\hat{r}_i^{\text{rat}}$ that justifies the correct answer. These are also filtered for correctness.
4.  **Fine-tuning:** The original pretrained model $M$ is fine-tuned on the combined dataset of correct rationales (both generated and rationalized).
5.  **Iteration:** This process is repeated, using the newly fine-tuned model to generate the next training set until performance plateaus. To avoid overfitting, the model is retrained from the original $M$ in each outer loop.

### Theoretical Framework
STaR approximates a reinforcement learning (RL) policy gradient objective. The model is viewed as a discrete latent variable model $p_M(y \mid x) = \sum_r p(r \mid x)p(y \mid x,r)$. The total expected reward $J$ across the dataset is:

$$
J (M, X, Y) = \sum_{i} \mathbb{E}_{\hat{r}_{i}, \hat{y}_{i} \sim p_{M}(\cdot | x_{i})} \mathbb{1}(\hat{y}_{i} = y_{i})
$$

The gradient is obtained via the log-derivative trick:

$$
\nabla J (M, X, Y) = \sum_{i} \mathbb{E}_{\hat{r}_{i}, \hat{y}_{i} \sim p_{M}(\cdot | x_{i})} \left[ \mathbb{1}(\hat{y}_{i} = y_{i}) \cdot \nabla \log p_{M}(\hat{y}_{i}, \hat{r}_{i} \mid x_{i}) \right]
$$

The filtering process in STaR corresponds to the indicator function $\mathbb{1}(\hat{y}_i = y_i)$, which discards gradients for incorrect rationales.

### Key Quantitative Results
Experiments were conducted using GPT-J (6B parameters) across three domains:

*   **CommonsenseQA (CQA):** STaR with rationalization achieved **72.5%** accuracy, performing comparably to a 30$\times$ larger GPT-3 model fine-tuned for direct prediction (**73.0%**). It significantly outperformed GPT-J fine-tuned for direct prediction (**60.0%**) and few-shot CoT GPT-J (**36.6%**).
*   **Arithmetic (n-digit summation):** After 16 iterations, STaR achieved **89.5%** overall accuracy, compared to **76.3%** for a baseline trained without rationales. Rationalization accelerated learning; for 2-digit addition, accuracy jumped from **<1%** (few-shot) to **32%** after one iteration.
*   **GSM8K (Grade School Math):** STaR with rationalization reached **10.7%** test accuracy, surpassing both the direct fine-tuned GPT-J (**5.8%**) and few-shot CoT GPT-J (**3.1%**).

### Stated Limitations
*   **Initial Capability:** The model must have a baseline few-shot performance above chance to begin bootstrapping; for example, GPT-2 was unable to bootstrap even in the arithmetic domain.
*   **Chance Performance:** In settings with high chance performance (e.g., binary decisions), the model generates many poor rationales that can confound the learning process.
*   **Reasoning Quality:** The model can still exhibit logical fallacies, such as "begging the question" (implying the answer in the question) or providing "red herrings" (true statements that do not support the claim).
*   **Hint Implementation:** Providing the "hint" for rationalization may be non-trivial depending on the context.
