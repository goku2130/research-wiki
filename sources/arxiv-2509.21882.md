---
id: arxiv:2509.21882
type: paper
title: The Hidden Costs and Measurement Gaps of Reinforcement Learning with Verifiable
  Rewards
url: https://arxiv.org/abs/2509.21882
retrieved: '2026-07-11'
maturity: comprehensive
topic: verifiable-rewards
---

# Summary: The Hidden Costs and Measurement Gaps of Reinforcement Learning with Verifiable Rewards

### Core Problem
Reinforcement Learning with Verifiable Rewards (RLVR) is widely used to improve Large Language Model (LLM) performance in structured domains like mathematics and coding. However, the authors argue that reported gains are frequently overstated due to three primary confounds:
1. **Budget Mismatch:** Comparing RLVR models (often evaluated with high sampling budgets, e.g., $pass@k$) against baselines with low budgets (e.g., $pass@1$), which conflates policy improvement with increased search.
2. **The "RLVR Tax":** Unintended side effects including "attempt inflation" (converting abstentions into confident but incorrect answers), miscalibration, erosion of instruction fidelity, and increased safety/privacy risks due to longer reasoning traces.
3. **Data Contamination:** The risk that models are memorizing benchmark solutions rather than developing general reasoning capabilities.

### Method and Evaluation Recipe
The authors employ a multi-pronged approach to isolate genuine reasoning gains from artifacts:

1. **Budget-Parity Reproductions:** The authors use a standardized evaluation (SoberScore) that matches decoding budgets, prompt families, and verifiers across models. They report $avg@32$ (the mean of $pass@1$ across 32 independent decodes) to stabilize estimates.
2. **Tax Quantification:** To measure attempt inflation, they track:
    * **Not attempted:** The number of items with no extractable answer.
    * **Shared Accuracy:** Accuracy calculated only on items that both the baseline and RLVR models attempted.
    * **Expected Calibration Error (ECE):** Computed from the model's stated confidence.
3. **Partial-Prompt Contamination Probes:** To detect memorization, the authors reveal only the first $x\%$ of a problem ($x \in \{80, 60, 40\}$) and greedily decode the suffix. High accuracy ($ACC@x$) or exact match ($EM@x$) on the hidden tail indicates contamination.
4. **Tax-Aware Minimum Standard:** The authors propose a reporting protocol requiring:
    * Matched sampling budgets and saturation curves (accuracy vs. $k$).
    * Tracking of refusal rates, shared accuracy, and ECE.
    * Judge-robustness stress tests (perturbing prompt templates) when using LLM judges.
    * Explicit contamination screens and the use of clean, held-out sets.

### Key Formulas
The paper utilizes the following metrics to evaluate performance and calibration:
* **$pass@k$**: The probability that at least one of $k$ generated samples is correct.
* **$avg@k$**: The mean of $pass@1$ across $k$ independent draws.
* **$maj@k$**: The accuracy of the majority vote over $k$ samples.
* **Expected Calibration Error (ECE)**: A measure of the difference between predicted confidence and actual accuracy.

### Key Quantitative Results
* **Attempt Inflation:** In a factual-QA control, moving from Qwen2.5-14B-Instruct to R1-Distill-Qwen-14B (SFT) reduced "Not attempted" items from $1,136$ to $102$, yet shared accuracy actually decreased from $12.5\%$ to $10.5\%$, and ECE worsened from $0.598$ to $0.692$.
* **Evaluation Gaps:** Standardizing evaluations significantly reduced reported gains. For AIME-24:
    * **Open-RS3-1.5B:** Reported $46.70\% \rightarrow$ Standardized $30.94\%$ ($\Delta = +15.76$).
    * **STILL-3-1.5B:** Reported $39.33\% \rightarrow$ Standardized $31.46\%$ ($\Delta = +7.87$).
    * **DAPO-Qwen-32B:** Reported $50.00\% \rightarrow$ Standardized $51.56\%$ ($\Delta = -1.56$).
* **Contamination Evidence:** Qwen3-14B-Base achieved $58.2\%$ accuracy on the $80\%$ prefix probe ($ACC@80$) for the legacy MATH-500 set, but collapsed to $0.0\%$ on the fresh AIME-2025 set, suggesting heavy memorization of older benchmarks.

### Stated Limitations
The authors note that their analysis is limited to verifiable domains (math, code, structured QA) and may not transfer to agentic or multimodal settings. The "attempt" metric depends on specific extraction rules, and the partial-prompt probe, while high-precision, is not exhaustive and may miss semantic overlap. Finally, the study does not include reward-component ablations (e.g., testing specific refusal or grounding rewards) as these would require additional training runs.
