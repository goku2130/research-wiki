---
id: arxiv:2607.08535
type: paper
title: 'When the Judge Changes, So Does the Measurement: Auditing LLM-as-Judge Reliability'
url: https://arxiv.org/abs/2607.08535
retrieved: '2026-07-12'
maturity: comprehensive
topic: llm-as-judge
---

# Summary: When the Judge Changes, So Does the Measurement: Auditing LLM-as-Judge Reliability

### Core Problem
The authors address **evaluator-replacement ambiguity**: the phenomenon where an LLM-as-judge score changes when the evaluator model is replaced, even if the candidate responses remain fixed. This ambiguity makes it unclear whether a score change reflects an actual improvement in the candidate system, a change in the judge's capability, a shift in the judge's biases, or artifacts in the evaluation pipeline (e.g., parsing and aggregation). The paper treats LLM-as-judge reliability as a measurement-validity problem rather than a simple model-selection task.

### Method and Experimental Design
The researchers employed an auditable measurement framework across four datasets: **LLMBar** (adversarial), **PandaLM** (broad-domain), **Chatbot Arena** (human preference), and **Judge’s Verdict** (TechQA-derived). They tested reliability along two primary axes:
1.  **Parameter Scaling:** Qwen3 dense judges (1.7B, 4B, 14B, 32B).
2.  **Released-Model Upgrades:** MiniMax API sequence (M2, M2.1, M2.5, M2.7).

The study was decomposed into four experiments:
*   **Experiment 1 (Validity):** Measured single-judge accuracy, Cohen’s $\kappa$, and Spearman rank correlation.
*   **Experiment 2 (Bias):** Probed for position-flip rates, verbosity bias (using padding strings), and granularity sensitivity.
*   **Experiment 3 (Aggregation):** Evaluated majority-vote juries (homogeneous and heterogeneous) to test error independence.
*   **Experiment 4 (Debate):** A case study on structured debate protocols to test protocol auditability.

Statistical significance for adjacent model comparisons was determined using two-sided McNemar tests with Holm correction.

### Key Formulas
To analyze jury reliability, the authors used a $\rho$-corrected beta-binomial model to predict jury accuracy, where $p$ is the single-judge baseline accuracy and $\rho$ is the intra-class error correlation:

$$
q \sim \text{Beta}(\alpha, \beta)
$$

$$
\alpha = ps, \quad \beta = (1 - p)s, \quad s = \frac{1}{\rho} - 1
$$

The jury accuracy is then calculated as the probability that a majority of $K$ votes are correct under this distribution.

### Key Quantitative Results
*   **Non-Interchangeable Upgrades:** Judge upgrades do not follow a uniform pattern. Qwen3 showed a robust, significant gain from 1.7B to 4B on LLMBar (accuracy $0.463 \to 0.617$) and Arena. Conversely, no adjacent releases in the MiniMax sequence reached $p < 0.05$ significance.
*   **Capability-Fairness Association:** Stronger judges are less biased but not unbiased. On LLMBar, the position-flip rate decreased from $0.320$ (Qwen3-1.7B) to $0.117\text{--}0.147$ (MiniMax releases). A strong Pearson correlation ($r = -0.957$) exists between accuracy and position-flip rates.
*   **Error Correlation in Juries:** Majority voting provided minimal gains due to high error correlation $\rho$. For Qwen3 homogeneous juries, $\rho$ ranged from $0.944$ to $0.972$ on LLMBar; for MiniMax, $\rho$ was $0.664\text{--}0.706$. For Qwen3-1.7B on LLMBar, increasing jury size $K$ from 1 to 3 to 5 only moved accuracy from $0.463 \to 0.475 \to 0.482$.
*   **Debate Shifts:** Structured debate caused substantial shifts in decisions. Pairing Qwen3-1.7B with higher-capability judges (GLM-5.1, MiniMax-M2.7, or mimo-v2-pro) increased final accuracy by $+0.317$, $+0.305$, and $+0.289$, respectively.

### Stated Limitations
*   **Construct Validity:** The reliability proxies used (accuracy, bias probes, $\rho$) may not exhaust all dimensions, such as calibration or rationale quality.
*   **Internal Validity:** The MiniMax axis was an observed release sequence rather than a controlled ablation.
*   **Auditability:** The debate experiment lacked raw outputs and parse-success logs, meaning shifts cannot be definitively attributed to deliberation rather than fallback behaviors.
*   **External Validity:** Results are limited to two model families, four datasets, and a small set of prompts.
