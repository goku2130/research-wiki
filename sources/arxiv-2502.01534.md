---
id: arxiv:2502.01534
type: paper
title: 'Preference Leakage: A Contamination Problem in LLM-as-a-Judge'
url: https://arxiv.org/html/2502.01534v3
retrieved: '2026-07-11'
maturity: comprehensive
topic: judging-bias-and-contamination
---

# Preference Leakage in LLM-as-a-Judge

### Core Problem
**Preference Leakage** is a contamination problem in the "LLM-as-a-Judge" paradigm. It occurs when the LLM used for synthetic data generation ($M_G$) and the LLM used as the evaluator ($M_J$) are closely related. This relatedness causes the judge to exhibit a systematic bias toward student models ($M_S$) trained on $M_G$'s synthetic data. Unlike egocentric bias (where a model favors its own outputs), preference leakage occurs because the student model inherits "spurious features"—such as style, format, and wording—from the generator, which the related judge is predisposed to favor.

### Method and Recipe
The researchers investigated preference leakage through the following step-by-step process:

1.  **Defining Relatedness**: Three types of relatedness between $M_G$ and $M_J$ were established:
    *   **Same Model**: $M_G \equiv M_J$.
    *   **Inheritance**: One model is fine-tuned from the other or trained on the other's outputs.
    *   **Same Model Family**: Models share a common architectural blueprint and overlapping pre-training datasets (e.g., different versions of GPT).
2.  **Data Synthesis and Training**:
    *   30,000 prompts were sampled from the Ultrafeedback dataset.
    *   $M_G$ (e.g., GPT-4o, Gemini-1.5, LLaMA-3.3) generated synthetic responses.
    *   Student models (e.g., Mistral-7B, Qwen-2.5-14B) were trained on this data via Supervised Fine-Tuning (SFT).
3.  **Evaluation**: Student models were paired and compared using various judge models ($M_J$) across benchmarks including **Arena-Hard** and **AlpacaEval 2.0**.
4.  **Quantification**: The researchers developed the **Preference Leakage Score (PLS)** to measure the bias of judges toward their related students.
5.  **Mechanism Analysis**: 
    *   **Recognition Test**: Prompted judges to identify their student models.
    *   **Classification**: Trained a BERT classifier to distinguish student model outputs.
    *   **Ablation**: Used a paraphrasing pipeline to remove style, format, and wording to see if PLS decreased.
6.  **Mitigation Testing**: Evaluated five strategies: prompting, Chain-of-Thought (CoT), paraphrasing, auto-calibration, and contextual calibration.

### Key Formulas
The condition for preference leakage is defined as the inflation of the evaluation score $S_{M_J}$ when $M_G$ is related to $M_J$ ($\sim_{rel}$):

$$
E_{x, y_S \sim P_{M_S}} [S_{M_J}(y_S | x) \mid M_G \sim_{rel} M_J] > E_{x, y_S \sim P_{M_S}} [S_{M_J'}(y_S | x) \mid M_G \not\sim_{rel} M_J']
$$

The **Preference Leakage Score (PLS)** for a model pair $(i, j)$ is:

$$
\mathrm{PLS}(i, j) = \frac {\left(\frac {\mathrm {W R} (i , i) - \mathrm {A V G} (i , j)}{\mathrm {A V G} (i , j)}\right) + \left(\frac {\mathrm {W R} (j , j) - \mathrm {A V G} (j , i)}{\mathrm {A V G} (j , i)}\right)}{2}
$$

Where $\mathrm{WR}(i, j)$ is the win-rate of student $i$ judged by judge $j$, and:

$$
\text{AVG}(i, j) = \frac{\text{WR}(i, i) + \text{WR}(i, j)}{2}
$$

The **Error Bias** for mitigation analysis is:

$$
\mathrm{Error Bias} = \frac {N_{\mathrm{target-prefer-other-win}}}{N_{\mathrm{other-win}}} - \frac {N_{\mathrm{other-prefer-target-win}}}{N_{\mathrm{target-win}}}
$$

### Key Quantitative Results
*   **Prevalence**: In main experiments, Mistral-7B and Qwen-2.5-14B showed average PLS of **23.6%** and **27.9%** respectively when using GPT-4o and Gemini-1.5 as the generator/judge pair.
*   **Relatedness Impact**: The average PLS was highest for the **Same Model (23.6%)**, followed by **Inheritance (19.3%–22.3%)**, and **Same Family/Same Series (8.9%)**.
*   **Learning Methods**: SFT exhibited the highest leakage (**23.6%**), while DPO was significantly lower (**5.2%**) and ICL showed negligible or negative leakage (**-2.7%**).
*   **Model Size**: Smaller student models (e.g., LLaMA-3-1B, Qwen-2.5-3B) exhibited higher PLS than larger counterparts.
*   **Detectability**: Judge LLMs failed to recognize their students (accuracy near random guess), but a BERT classifier achieved **82.4%** accuracy in classifying student responses.
*   **Mitigation**: **Contextual Calibration** was the most effective, reducing Error Bias from **17.8** (base) to **7.3**.

### Stated Limitations
*   **Detectability**: Preference leakage is subtler and harder for judge LLMs to detect than egocentric bias.
*   **Subjectivity**: Leakage is more pronounced in subjective question types (e.g., writing, programming) and subjective judgment dimensions (e.g., fairness) compared to objective ones (e.g., mathematics, completeness).
*   **Real-world Data**: The authors note that few student-teacher pairs are publicly documented in leaderboards, limiting the scale of real-world impact analysis.
