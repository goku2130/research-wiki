---
id: arxiv:2607.08065
type: paper
title: When LLMs Agree, Are They Right? Auditing Self-Consistency and Cross-Model
  Agreement as Confidence Signals
url: https://arxiv.org/abs/2607.08065
retrieved: '2026-07-11'
maturity: comprehensive
topic: llm-as-judge
---

# Summary: When LLMs Agree, Are They Right?

### Core Problem
The study audits the pervasive assumption in AI evaluation pipelines—specifically in "LLM-as-judge" and ensemble systems—that **consistency** (agreement among multiple judges or multiple stochastic samples from a single model) serves as a reliable proxy for **correctness**. The author posits that agreement is not accuracy; instead, high consistency can be driven by shared biases, memorized heuristics, or option-position priors rather than truth.

### Method and Recipe
The author conducted a large-scale cross-runner audit using 53 runners who generated $K=50$ samples for assigned cases across two benchmarks: **GPQA Diamond** (graduate-level four-option multiple choice) and **AIME** (competition mathematics). The study produced 265,000 samples across three experimental axes:
*   **Axis A (Model Tier):** Comparison of `gpt-4.1-nano` vs. `gpt-4.1-mini` (zero-shot).
*   **Axis B (Prompting):** Comparison of `gpt-4.1-mini` zero-shot vs. chain-of-thought (CoT).
*   **Axis C (Scale):** Comparison of `gpt-4.1-mini` vs. `gpt-4.1` (frontier model, zero-shot).

**Analysis Pipeline:**
1.  **Metrics:** The author calculated sample accuracy, majority correctness, and self-consistency.
2.  **Statistical Inference:** Used a hierarchical runner-clustered bootstrap ($B=2000$) to correct for runner-level dependence and a case-clustered bootstrap to ensure results were not artifacts of the clustering choice.
3.  **Calibration:** Measured Expected Calibration Error (ECE) and Brier scores.
4.  **Controls:** An option-shuffle control was used to detect positional bias, and an exploratory cross-family check was performed on three Claude tiers (Haiku, Sonnet, Opus) via agent sessions.

### Key Formulas
The study defines the following primary variables:
*   **Sample Accuracy ($A$):** The proportion of correct samples among $K$ draws:

$$
A = \frac{n_{\text{correct}}}{K}
$$

*   **Majority Correctness ($M$):** A binary indicator of whether the most frequent answer matches the ground truth:

$$
M = \mathbb{1}[\text{majority answer} = \text{ground truth}]
$$

*   **Self-Consistency ($C$):** The agreement rate among the $K$ samples.
*   **Predictive Strength:** Measured via Spearman’s rank correlation $\rho(C, M)$ between agreement and majority correctness.

### Key Quantitative Results
*   **Weak Predictive Power:** Agreement is a positive but weak predictor of correctness, with $\rho(C, M)$ ranging from $0.20$ to $0.59$.
*   **Frontier Over-confidence:** The most consistent model (`gpt-4.1`) exhibited the worst calibration. On GPQA, it had the highest mean agreement ($\bar{C} = 0.89$) but the lowest correlation with correctness ($\rho = 0.20$). 
    *   It reached $C \geq 0.8$ on 77% of GPQA cases, yet **48% of those high-agreement cases were wrong**.
    *   Its ECE on GPQA was $0.41$, the highest of all twelve tested cells.
*   **Chain-of-Thought (CoT):** CoT robustly improved accuracy ($\Delta A = +0.067$ on GPQA; $p = 2.1 \times 10^{-5}$) but only marginally improved the agreement-correctness signal ($\Delta \rho(C, M) = +0.123$ on GPQA).
*   **Positional Bias:** In GPQA, option "D" was consistently under-selected ($\approx 15\text{--}16\%$) regardless of content, indicating that a portion of "confidence" is positional.
*   **Cross-Family Recurrence:** The Claude Opus model mirrored the frontier pattern: it was the most self-consistent ($\bar{C} = 0.94$) but not the most accurate and the worst-calibrated (ECE = $0.35$) compared to Claude Sonnet.
*   **Shared Errors:** GPT-4.1 and Claude tiers often picked the same wrong answers with high confidence (mean $C \approx 0.85$ for GPT, $\approx 0.78$ for Claude), suggesting shared pretraining biases.
*   **Routing:** A confidence-routed cascade was dominated by simply always using the mid-tier model; escalating to the over-confident frontier model actually decreased majority accuracy.

### Stated Limitations
*   **Data Provenance:** The primary audit used secondary data from graduate students; exact model snapshots and timestamps were not logged.
*   **Claude Check:** The cross-family check was agent-mediated, meaning temperature was not controlled and $K$ was lower ($K=10$).
*   **Scope:** The study focused on MCQ and integer math, not open-ended generation or code.
*   **Sample Size:** Baseline and shuffle studies used small case sets ($n=48\text{--}50$), resulting in wide confidence intervals.
*   **Noise:** $K=50$ samples still leave substantial measurement noise.
