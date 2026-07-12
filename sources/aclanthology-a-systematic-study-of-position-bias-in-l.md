---
id: aclanthology:a-systematic-study-of-position-bias-in-l
type: web
title: A Systematic Study of Position Bias in LLM-as-a-Judge (ACL Anthology)
url: https://aclanthology.org/2025.ijcnlp-long.18.pdf
retrieved: '2026-07-12'
maturity: comprehensive
topic: llm-as-judge
---

# Summary: A Systematic Study of Position Bias in LLM-as-a-Judge

### Core Problem
The "LLM-as-a-Judge" framework is often used to automate the evaluation of other LLMs, but it is compromised by **position bias**: the tendency of a judge to favor a candidate solution based on its position in the prompt rather than its objective content. Prior research has been preliminary, primarily focused on pairwise comparisons, and has lacked a systematic exploration of the factors contributing to this bias or its behavior in complex list-wise settings.

### Method and Recipe
The researchers conducted an exploratory study using 15 LLM judges (from the GPT, Claude, Gemini, and Llama families) across two benchmarks: **MT-Bench** (8 tasks, 10 questions per task) and **DevBench** (14 tasks, 8 questions per task), totaling over 150,000 evaluation instances.

**1. Evaluation Settings**
*   **Pairwise Comparison:** Judges select the better of two solutions. To test for bias, the order of solutions is swapped, and the judge is queried again.
*   **List-wise Comparison:** Judges select the best candidate from a list of three or more. The order is permuted such that each candidate appears in every possible position exactly once.

**2. Metrics**
The study introduces three primary metrics to quantify reliability and bias:
*   **Repetition Stability (RS):** Measures if judgments are consistent across identical queries to ensure bias is not due to random variation.

$$
RC = \frac{1}{N} \sum_{j=1}^{N} \frac{1}{n_j} \max_{k \in S} \left\{ C_k^i \right\}
$$

*   **Position Consistency (PC):** The ratio of instances where the judge prefers the same solution regardless of its position.

$$
PC = \frac{1}{n} \sum_{j=1}^{n} 1_{(C_k^i, \dots, C_p^i) \in V}
$$

*   **Preference Fairness (PF):** A normalized score centering on zero (fair), where positive values indicate recency preference and negative values indicate primacy preference.

$$
PF = \frac{PF_{nw} - S_{\min}}{S_{\max} - S_{\min}} \times 2 - 1, \text{ where } PF_{nw} = (rcn \times irr) - (pcn \times ipr)
$$

**3. Factor Analysis**
To identify drivers of bias, the authors used **bidirectional stepwise regression** based on the Akaike Information Criterion (AIC):

$$
\text{AIC} = 2k - 2\log(L)
$$

They tested Judge-level (familial property), Candidate-level (answer quality gap), and Task-level (category, input/output length) factors. The **answer quality gap** ($\delta_0$) was calculated using the overall win rate ($owr$):

$$
owr = \frac{1}{n} [C_w + \frac{1}{p} (C_l + C_1)], \quad \delta_0 = \frac{|owr - 1|}{p}
$$

### Key Quantitative Results
*   **Stability:** Capable judges (e.g., GPT-4, Claude-3.5-Sonnet, Llama-3.7-30B) exhibited high repetition stability (RS > 0.95), proving position bias is a systematic tendency rather than random noise.
*   **Judge and Task Variance:** Bias is highly volatile. For example, Claude-3.5-Sonnet was nearly fair on MT-Bench ($PF = 0.01$) but showed strong recency bias on DevBench ($PF = 0.22$).
*   **Quality Gap Correlation:** Position Consistency (PC) is positively proportional to the answer quality gap; larger disparities in solution quality make judges more consistent and fair.
*   **Length Dependency:** Prompt and input lengths had negligible impact. Task output length was a statistically significant predictor of PF, but its influence was minimal.
*   **Agreement Analysis:** Over 50% of the dataset was "easy" (disagreement $\le 3$ among 15 judges), while fewer than 2% were "especially hard" (disagreement $\ge 8$).

### Stated Limitations
*   **Scope:** The study only evaluated 12 closed-source and 3 open-source judges, and list-wise evaluations were limited to three-candidate lists.
*   **Data Access:** Due to the proprietary nature of closed-source models, the authors could not analyze specific architectural factors or parameter sizes, relying instead on "familial properties."
*   **Approach:** The analysis was conducted *post hoc* and focused on understanding and quantifying bias rather than developing active mitigation strategies.
