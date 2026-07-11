---
id: aclanthology:beyond-the-surface-measuring-self-prefer
type: web
title: 'Beyond the Surface: Measuring Self-Preference in LLM Judgments'
url: https://aclanthology.org/2025.emnlp-main.86.pdf
retrieved: '2026-07-11'
maturity: comprehensive
topic: judging-bias-and-contamination
---

# Research Summary: Beyond the Surface: Measuring Self-Preference in LLM Judgments

## Core Problem
Large Language Models (LLMs) acting as judges often exhibit **self-preference bias**, the tendency to favor their own generated responses over those of other models. Previous measurement methods typically calculated the difference between the scores a judge assigned to its own responses versus others. However, the authors argue that this approach conflates **response quality** with **bias**: if a judge model is genuinely higher-quality than the comparison model, it will naturally assign itself higher scores, leading to an overestimation of bias.

## Method: The DBG Score
To isolate bias from quality, the authors propose the **DBG score**, which uses "gold judgments" as a proxy for ground-truth quality.

### Step-by-Step Recipe
1.  **Response Generation**: For a given instruction $x$, two models ($A$ and $B$) generate responses $r_A$ and $r_B$.
2.  **Judge Evaluation**: Model $A$ acts as the judge. To mitigate position bias, the order of $r_A$ and $r_B$ is swapped, and the average probability of the output tokens "A" or "B" is collected to determine the win rate $w_A$.
3.  **Gold Judgment Construction**: An unbiased gold judgment $w^*$ is created by aggregating the evaluation results from three strong LLMs (GPT-4o-mini, Gemini-1.5-Flash, and DeepSeek-V3).
4.  **Bias Calculation**: The DBG score is calculated as the difference between the judge model's win rate and the gold judgment win rate: $\hat{w}_A = w_A - w^*$.

### Key Formulas
The authors model the preference probability as:

$$
\mathbb{P}(r_{A}\succ r_{B}\mid x)=\sigma(S_{A}(r_{A})-S_{A}(r_{B}))
$$

Where $S_A(r)$ is the score assigned by judge $A$, approximated as the sum of true quality $Q(r)$ and inherent bias $b_A(r)$:

$$
S_A(r) \approx Q(r) + b_A(r)
$$

Substituting these, the probability becomes:

$$
\mathbb{P}(r_A \succ r_B | x) = \sigma(\delta + b_A)
$$

where $\delta = Q(r_A) - Q(r_B)$ represents the quality gap and $b_A$ represents the self-preference bias. The DBG score $\hat{w}_A$ is formulated as:

$$
\hat{w}_A = \mathbb{E}_x[\sigma(\delta + b_A)] - \mathbb{E}_x[\sigma(\delta)]
$$

## Key Quantitative Results
### Model Scale and Version
*   **Size Correlation**: Larger models exhibit significantly less self-preference bias. For example, Llama-3.1-70B showed a DBG score of $0.4\%$, while Llama-3.1-8B showed $21.6\%$. Similarly, Qwen2.5-0.5B-Instruct had a DBG of $41.7\%$, compared to $2.1\%$ for Qwen2.5-14B-Instruct.
*   **Training Phase**: Both pre-trained and post-trained models exhibit bias, indicating that bias is not solely introduced during post-training.

### Reasoning Models (LRMs)
Large Reasoning Models also display self-preference bias. The DBG score for DS-R1-Distill-Qwen-32B was $4.8\%$, which is higher than that of the non-reasoning Qwen2.5-72B-Instruct ($2.6\%$).

### Mitigation Factors
*   **Response Style**: Rewriting responses into a unified "attractive" or "humorous" style reduces bias. Llama-3.1-8B's DBG score dropped from $18.7\%$ to $7.2\%$ (attractive) and $5.9\%$ (humorous).
*   **Post-training Data**: Training different models on the same dataset reduces bias. Llama-3.1-8B-UltraChat and Qwen2.5-7B-UltraChat exhibited DBG scores of $2.1\%$ and $1.1\%$, respectively.

### Validation
*   **Human Alignment**: Gold judgments agreed with human annotations on $74\%$ of samples.
*   **Judge Agreement**: Pairwise accuracy among the three gold judge models ranged from $80.9\%$ to $86.5\%$.

## Stated Limitations
*   **Gold Judge Capability**: Due to cost, the authors used "mini" or "flash" versions of gold judges rather than the most powerful versions (e.g., GPT-4o or Gemini-1.5-Pro).
*   **Remaining Biases**: While position and length biases were mitigated, other factors like authority or sentiment bias may still influence results.
*   **Task Scope**: The study was limited to instruction-following and translation tasks.
