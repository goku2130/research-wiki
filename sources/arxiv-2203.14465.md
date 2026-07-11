---
id: arxiv:2203.14465
type: paper
title: 'STaR: Bootstrapping Reasoning With Reasoning'
url: https://arxiv.org/abs/2203.14465
retrieved: '2026-07-11'
maturity: comprehensive
topic: self-improvement-and-self-play
---

**Core Problem**
Inducing large language models (LLMs) to generate step-by-step rationales significantly improves performance on complex reasoning tasks, yet existing approaches face severe constraints. Constructing rationale datasets manually is prohibitively expensive and infeasible for novel domains, while template-based methods require pre-existing general solutions. Alternatively, few-shot chain-of-thought prompting improves accuracy but substantially underperforms models fine-tuned to directly predict answers on larger datasets. The core challenge is bootstrapping rationale generation from a small number of examples and a large dataset without ground-truth rationales, without manual annotation, and without sacrificing accuracy.

**Method and Algorithm**
The Self-Taught Reasoner (STaR) addresses this through an iterative bootstrapping loop that combines forward rationale generation with backward rationalization. Given a pretrained LLM $M$, a dataset $\mathcal{D}=\{(x_i, y_i)\}$, and a small few-shot prompt set $\mathcal{P}$ containing initial rationales, STaR executes the following steps per iteration:
1. **Rationale Generation:** The model is prompted with $\mathcal{P}$ and each question $x_i$ to generate a rationale $\hat{r}_i$ and predicted answer $\hat{y}_i$.
2. **Rationalization:** For questions where $\hat{y}_i \neq y_i$, the model is prompted with the question, the ground-truth answer $y_i$ as a hint, and $\mathcal{P}$ to generate a backward-reasoned rationale $\hat{r}_i^{\text{rat}}$.
3. **Filtering:** Only rationales that ultimately yield the correct answer are retained ($\hat{y}_i = y_i$ for generation; $\hat{y}_i^{\text{rat}} = y_i$ for rationalization).
4. **Fine-tuning:** The model is fine-tuned from scratch on the combined filtered dataset. The loop repeats until performance plateaus. This process leverages the model’s pre-existing reasoning capacity to iteratively refine its own training data.

**Key Formulas**
STaR approximates a reinforcement learning policy gradient objective. The model is treated as a discrete latent variable model $p_M(y|x) = \sum_r p(r|x)p(y|x,r)$. The expected reward across the dataset is defined as:
\[
J(M, X, Y) = \sum_i \mathbb{E}_{\hat{r}_i, \hat{y}_i \sim p_M(\cdot|x_i)} \mathbb{1}(\hat{y}_i = y_i),
\]
with the gradient computed via the log-derivative trick:
\[
\nabla J(M, X, Y) = \sum_i \mathbb{E}_{\hat{r}_i, \hat{y}_i \sim p_M(\cdot|x_i)} \left[ \mathbb{1}(\hat{y}_i = y_i) \cdot \nabla \log p_M(\hat{y}_i, \hat{r}_i | x_i) \right].
\]
The indicator function $\mathbb{1}(\hat{y}_i = y_i)$ acts as a reward filter, discarding gradients for incorrect rationales, which corresponds to STaR’s filtering step.

**Quantitative Results**
Evaluated on a 6B-parameter GPT-J model, STaR demonstrates substantial gains across symbolic and natural language reasoning tasks. On arithmetic (n-digit addition), STaR achieves 89.5% accuracy after 16 iterations, outperforming a direct fine-tuning baseline (76.3%). Rationalization accelerates learning, jumping 2-digit addition accuracy from <1% to 32% after a single iteration. On CommonsenseQA, STaR without rationalization reaches 68.8% accuracy (training on 69.7% of data), while STaR with rationalization achieves 72.5% (training on 86.7% of data). This matches the performance of a 30× larger GPT-3 model fine-tuned directly (73.0%) and surpasses few-shot chain-of-thought baselines, including a 137B-parameter LaMDA model (55.6%). On GSM8K, STaR improves accuracy to 10.1% (without rationalization) and 10.7% (with rationalization), compared to a 5.8% direct fine-tuning baseline. Human evaluations indicate STaR-generated rationales are ranked 30% more favorably than few-shot rationales ($p=.039$) and 74% more favorably than human-generated rationales ($p<.001$).

**Stated Limitations**
The authors identify several constraints. STaR requires the initial model to possess non-trivial few-shot reasoning capabilities; smaller models like GPT-2 fail to bootstrap. Tasks with high chance-performance (e.g., binary decisions) generate excessive poor rationales, confounding the filtering mechanism, with no proposed solution for filtering bad reasoning in these settings. Rationalization relies on a hint format that may be non-trivial to construct across domains. Additionally, using higher sampling temperatures to expand training data is counterproductive, as it increases correct answers paired with irrelevant reasoning and is computationally inefficient. The inclusion of few-shot prompts during training also presents a trade-off between maintaining rationale style consistency and computational cost.
