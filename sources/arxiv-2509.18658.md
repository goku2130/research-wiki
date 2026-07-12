---
id: arxiv:2509.18658
type: paper
title: 'Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal
  Prediction'
url: https://arxiv.org/abs/2509.18658
retrieved: '2026-07-12'
maturity: comprehensive
topic: llm-as-judge
---

# Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction

### Core Problem
While the "LLM-as-a-judge" paradigm is widely used to evaluate natural language generation (NLG), it often suffers from inherent randomness, bias, and overconfidence. Existing uncertainty quantification methods—such as self-reported confidence or token-level probabilities—are often unreliable or computationally expensive. There is a critical need for a reliable, distribution-free framework to quantify the uncertainty of LLM-based scoring, particularly for high-stakes applications like healthcare and finance.

### Method
The authors propose a framework using **split conformal prediction (CP)** to construct prediction intervals for rating-based evaluations. The process follows these steps:

1.  **Feature Extraction**: The framework extracts token logits from the LLM judge. A rule-based strategy locates the rating token, and log probabilities for all potential rating tokens (e.g., 1–5 on a Likert scale) are extracted. Tokens with equivalent meanings (e.g., "two" and "2") are aggregated into a $K$-dimensional feature vector $z$.
2.  **Non-conformity Scoring**: Using a held-out calibration set, a non-conformity score $s(z, y)$ is computed to measure the discrepancy between the predicted score $\hat{y}$ and the ground truth $y$.
3.  **Interval Construction**: The $\lceil (n+1)(1-\alpha) \rceil$-quantile $\hat{q}$ of the calibration scores is calculated for a desired miscoverage rate $\alpha$. For a test point $z_{test}$, the prediction interval is defined as:

$$
\mathcal{C}(z_{test}, \hat{y}_{test}) = [\hat{y}_{test} - \hat{q}, \hat{y}_{test} + \hat{q}]
$$

4.  **Boundary Adjustment**: To align continuous intervals with discrete ordinal ratings, the authors apply a boundary adjustment. This involves redefining the non-conformity score $s'(z, y)$ and shrinking or expanding the interval boundaries:
    *   **Shrinking**: $l' = \lceil l \rceil, u' = \lfloor u \rfloor$ (preserves coverage).
    *   **Expanding**: $l' = \lfloor l \rfloor$ or $u' = \lceil u \rceil$ (increases coverage).
5.  **Midpoint Estimation**: The midpoint of the resulting interval is used as a low-bias alternative to the raw model score or weighted average for final decision-making.

### Key Formulas
The framework relies on the statistical coverage guarantee of conformal prediction:

$$
1-\alpha \leq \mathbb{P}(y_{test} \in \mathcal{C}(z_{test}, \hat{y}_{test})) \leq 1-\alpha + \frac{1}{n+1}
$$

For regression-based CP, the non-conformity score is typically:

$$
s(z, y) = |\hat{y} - y|
$$

### Key Quantitative Results
The authors evaluated nine CP methods across three LLM judges (GPT-4o mini, DeepSeek-R1-Distill-Qwen-32B, and Qwen2.5-72B-Instruct) on summarization (SummEval, DialSumm) and reasoning (ROSCOE) tasks.

*   **Coverage and Efficiency**: Most methods achieved $\sim 90\%$ coverage in summarization tasks. Reasoning tasks showed lower coverage due to smaller calibration sets. Boundary adjustment consistently improved coverage; for example, using LVD on e-SNLI with Qwen2.5-72B-Instruct, coverage increased from $85.96\%$ to $95.53\%$.
*   **Bias Reduction via Midpoints**: Interval midpoints significantly outperformed raw scores and weighted averages in error metrics. For GPT-4o mini evaluating fluency on SummEval, the midpoint from R2CCP reduced the Mean Squared Error (MSE) by $88.7\%$, dropping from $3.907$ to $0.443$. Mean Absolute Error (MAE) for midpoints was consistently below $0.5$.
*   **Recommended Configurations**: The authors suggest **DeepSeek-R1-Distill-Qwen-32B + G-Eval + LVD** for high-stake applications and **Qwen2.5-72B-Instruct + R2CCP + SocREval** for maximum efficiency.

### Limitations
*   **Task Scope**: The analysis was limited to summarization and reasoning; other NLG tasks (e.g., machine translation, multimodal generation) were not explored.
*   **Annotation Dependency**: The framework requires high-quality human annotations for calibration. Biased human labels can lead to biased prediction intervals.
*   **Heuristic Nature**: The use of the interval midpoint is a heuristic choice and should not be interpreted as a formal statistical mean or mode.
