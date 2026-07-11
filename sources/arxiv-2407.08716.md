---
id: arxiv:2407.08716
type: paper
title: A Taxonomy for Data Contamination in Large Language Models
url: https://arxiv.org/abs/2407.08716
retrieved: '2026-07-11'
maturity: comprehensive
topic: judging-bias-and-contamination
---

# Taxonomy for Data Contamination in Large Language Models

### Core Problem
Large Language Models (LLMs) are often pretrained on massive web corpora that may inadvertently include evaluation datasets. This "data contamination" inflates performance metrics and invalidates the reliability of zero-shot or few-shot evaluations. The authors identify a lack of standardized terminology and a gap in understanding how different forms of contamination—specifically non-verbatim or "approximate" contamination—impact downstream task performance.

### Taxonomy and Formalization
The authors define contamination as any leakage of information that provides a signal for the correct label for at least one example in a test set $D$. This is formalized as a function $f(D)$, which may be a composition of multiple contamination functions:

$$
f = f^{(1)} \circ f^{(2)} \circ \dots \circ f^{(n)}
$$

The taxonomy categorizes contamination by two levels of properties:

**1. Dataset-level Properties**
*   **Selection:** A function that selects a subset $D' \subset D$, leaking only some examples.
*   **Distribution:** A function that combines $D$ with non-contaminating documents (e.g., shuffling), increasing the token span of the contaminated region.

**2. Instance-level Properties**
*   **Masking:** A function $h(\langle x_i, \hat{y}_i \rangle)$ that removes parts of the input or output.
*   **Noising:** Modifying the surface form (e.g., paraphrasing or using "silver" labels).
*   **Augmenting:** Adding additional context (relevant or irrelevant) to the example.

The authors distinguish these from **Prior Task Understanding** (learning a task from non-contamination sources) and **Transductive Learning** (pretraining on test inputs only, without labels), neither of which are classified as contamination under this taxonomy.

### Experimental Method
The authors used `nanoGPT` initialized with `gpt2-large` weights to measure the impact of contamination on summarization (XSum, SAMSum, CNN/Daily Mail) and question answering (SQuAD, CBT).

**Step-by-Step Recipe:**
1.  **Data Splitting:** Create equal-sized splits of Train, In-domain, and Test data.
2.  **Control Integration:** Add 10,000 samples of OpenAI’s WebText to continued pretraining to mitigate recency bias.
3.  **Training Configurations:**
    *   **Baseline:** Finetune the initial model on the train split.
    *   **Cheating:** Finetune the initial model directly on the test split.
    *   **Contamination:** Perform continued pretraining using $f(\text{test split})$, then finetune on the train split.
    *   **In-Domain:** Perform continued pretraining on task-relevant but uncontaminated data, then finetune on the train split.
4.  **Execution:** Use a blocksize of 1024 tokens, batchsize of 1, temperature of 0, and a maximum completion length of 200 tokens.

### Key Quantitative Results
Across all tasks, the "Cheating" setting (finetuning on test data) yielded the highest performance.

**Question Answering (QA):**
Contaminated settings (Verbatim, Noised, Distribution) significantly outperformed the Baseline and often outperformed In-domain settings, suggesting that seeing test-set data—even in approximate forms—unfairly boosts performance.
*   **SQuAD (Exact Match):** Baseline (41.76 $\pm$ 1.01) $\rightarrow$ Verbatim (53.38 $\pm$ 0.94) $\rightarrow$ Cheating (55.73 $\pm$ 0.94).
*   **CBT (Exact Match):** Baseline (19.41 $\pm$ 0.99) $\rightarrow$ Verbatim (52.06 $\pm$ 0.88) $\rightarrow$ Cheating (54.27 $\pm$ 0.85).

**Summarization:**
The performance boost from contamination was less pronounced than in QA. Verbatim and In-domain settings often performed on par, suggesting that for summarization, the benefit may stem from general in-domain exposure rather than specific test-set leakage. However, the **Noised** setting (using GPT-3.5 generated summaries) generally inflated scores over the Baseline across most datasets, except for XSum.

### Stated Limitations
*   **Recency Bias:** Contaminated data was introduced at the end of pretraining rather than randomly throughout.
*   **Model Generalizability:** Experiments were limited to a single model (`gpt2-large`); larger models may exhibit stronger memorization and more pronounced contamination effects.
*   **Data Accessibility:** Because GPT-2's original pretraining data is not public, results are approximations.
*   **Architecture:** Different architectures and training procedures may yield different results.
