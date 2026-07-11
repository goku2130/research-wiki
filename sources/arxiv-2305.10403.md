---
id: arxiv:2305.10403
type: paper
title: How Far Can Camels Go? Exploring the Limits of Instruction Tuning (AlpacaEval
  original paper, Li et al., 2023)
url: https://arxiv.org/abs/2305.10403
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-and-winrate-evals
---

# PaLM 2 Technical Report

### Core Problem
The primary objective of PaLM 2 was to develop a successor to the PaLM model that improved multilingual capabilities and reasoning while increasing compute efficiency. The researchers sought to move beyond simply scaling model size, instead focusing on meticulous data selection, efficient architectures, and optimized training objectives to reduce inference latency and serving costs.

### Method and Recipe
The development of PaLM 2 followed a multi-pronged approach to optimization:

1.  **Compute-Optimal Scaling**: The team validated scaling laws to determine the optimal ratio between model size ($N$) and training data size ($D$). They found that $N$ and $D$ should be scaled roughly 1:1 to achieve the best performance for a given compute budget.
2.  **Diverse Dataset Mixture**: The pre-training corpus was expanded to include a significantly higher percentage of non-English data, spanning hundreds of languages and domains (including mathematics, programming languages, and parallel multilingual documents). Deduplication was applied to mitigate verbatim memorization.
3.  **Architectural and Objective Improvements**: Based on the Transformer architecture, PaLM 2 utilized a tuned mixture of different pre-training objectives (inspired by UL2) rather than a single causal or masked language modeling objective.
4.  **Context Expansion**: The model was trained to support a significantly increased context length to improve long-range reasoning, summarization, and dialogue.
5.  **Safety and Control Mechanisms**: 
    *   **Control Tokens**: Special tokens marking toxicity levels (derived from the Perspective API) were added to a small fraction of the pre-training data to allow inference-time control over toxicity.
    *   **Canaries**: Special "interleave" and "shuffle" canary sequences were injected to measure memorization rates across different languages.
6.  **Model Variants**: The family includes Small (S), Medium (M), and Large (L) versions. Additionally, a coding-specific variant, **PaLM 2-S***, was created by continuing the training of PaLM 2-S on a code-heavy, multilingual mixture.

### Key Formulas
The study utilized a heuristic for training compute based on the following relationship:

$$
\text{FLOPs} \approx 6ND
$$

Where $N$ represents the number of parameters and $D$ represents the number of training tokens.

### Key Quantitative Results
PaLM 2 demonstrated significant improvements over the original PaLM across multiple benchmarks:

*   **Scaling Laws**: For a compute budget of $1 \times 10^{22}$ FLOPs, the projected optimal parameter size was $10.7\text{B}$.
*   **English QA and Classification**: In a 1-shot setting, PaLM 2-L achieved an average accuracy of $76.9\%$, compared to PaLM's $70.4\%$.
*   **Multilingual QA (TyDi QA)**: In the challenging "no-context" setting, PaLM 2-L achieved an average F1 score of $40.3\%$, significantly higher than PaLM's $31.5\%$.
*   **Reasoning**: On the BIG-Bench Hard (BB Hard) benchmark using Chain-of-Thought (CoT) prompting, PaLM 2 scored $78.1\%$, outperforming PaLM's $65.2\%$.
*   **Mathematics**: On GSM8K (with CoT and self-consistency), PaLM 2 achieved $92.2\%$ (Flan-PaLM 2 variant), surpassing GPT-4's $92.0\%$ and PaLM's $74.4\%$.
*   **Coding**: PaLM 2-S* achieved a $37.6\%$ pass@1 on HumanEval, outperforming the much larger PaLM-Coder-540B ($35.9\%$).
*   **Translation**: On WMT21 Chinese $\to$ English, PaLM 2 achieved an MQM human score of $3.0$ (lower is better), outperforming Google Translate ($3.1$) and PaLM ($3.7$).
*   **Toxicity Control**: Using "Low toxicity" control tokens reduced the probability of toxic continuation from a baseline of $0.075$ to $0.033$.

### Stated Limitations
*   **Metric Proxy Gap**: The researchers noted that training loss is not a perfect proxy for downstream performance; for example, a $9.5\text{B}$ model with the lowest loss slightly underperformed a $16.1\text{B}$ model on specific tasks.
*   **Translation Harms**: While gender agreement improved in some languages, PaLM 2 showed lower gender agreement scores than PaLM when translating into Arabic, Hindi, and Telugu.
*   **Toxicity**: Despite control tokens, toxicity persisted in dialog uses, with rates ranging from $1\%$ to $17.9\%$ depending on the language and identity terms referenced.
*   **Mitigation Efficacy**: General-purpose inference-time control tokens were found to be less effective than specialized downstream mitigation methods (e.g., those used in LaMDA).
