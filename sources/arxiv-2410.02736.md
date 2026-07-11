---
id: arxiv:2410.02736
type: paper
title: Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge (Wang et al., 2024)
url: https://arxiv.org/html/2410.02736v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: llm-as-judge
---

# Summary: Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge

### Core Problem
The use of Large Language Models (LLMs) as judges—either through pairwise comparison or direct scoring—has become a standard for evaluating model responses and providing supervised rewards. However, the reliability of "LLM-as-a-Judge" is undermined by multifaceted biases. Previous research into these biases has been limited by a narrow scope, high costs associated with human-annotated ground truth, and small test datasets that are susceptible to random interference.

### The CALM Framework
To address these issues, Wang et al. (2024) propose **CALM** (Comprehensive Assessment of Language Model Judge Biases), an automated "attack-and-detect" framework. CALM systematically quantifies biases by applying principle-guided perturbations to the input and measuring the consistency of the judge's output.

#### Step-by-Step Method
1.  **Bias Categorization**: The authors identify 12 distinct bias types:
    *   **Content-related**: Verbosity (favoring length), Fallacy-Oversight (ignoring logical errors), Sentiment (emotional preference), and Authority (favoring citations).
    *   **Context-related**: Position (favoring specific placements), Compassion-Fade (model name vs. anonymity), Bandwagon (majority opinion), and Distraction (irrelevant details).
    *   **Identity/Process-related**: Diversity (demographic bias), Chain-of-Thought (CoT) influence, Self-Enhancement (favoring own outputs), and Refinement-Aware (favoring known refined answers).
2.  **Dataset Construction**: Three datasets are utilized:
    *   **Fact-related**: Based on GSM8K, MATH, and ScienceQA.
    *   **Refinement-aware**: Based on CommonsenseQA, Quora-QuAD, and TruthfulQA.
    *   **Alignment**: Sourced from various DPO datasets (e.g., Truthy-DPO, Orca-DPO).
3.  **Automated Perturbation**: An LLM (GPT-4o) is used as a perturbation function $g(\cdot)$ to modify the system instruction $I$ or the response $R$ to inject a specific bias without altering the factual correctness or original meaning.
4.  **Quantification**: The judge model is run on the original prompt $P$ and the perturbed prompt $\hat{P}$. The results are compared using specific metrics.

### Key Formulas
The framework defines the input prompt as $P = (I, Q, R)$ (for scoring) or $P = (I, Q, R_1, R_2)$ (for pairwise comparison). The output of the judge is $y = \text{LLM}(P)$ and the perturbed output is $\hat{y} = \text{LLM}(\hat{P})$.

**Robustness Rate (RR)** and **Consistency Rate (CR)** are calculated over dataset $D$:

$$
\mathbf{RR} = \frac{1}{|D|} \sum_{i=1}^{|D|} \mathbb{I}(y^i = \hat{y}^i), \quad \mathbf{CR} = \frac{1}{|D|} \sum_{i=1}^{|D|} \mathbb{I}(y^i = y_{\text{rand}}^i)
$$

**CoT Accuracy** is measured by comparing the judge's selection to the ground truth $R$:

$$
\text{Acc}_{\text{ofi}} = \frac{1}{|D|} \sum_{i=1}^{|D|} \mathbb{I}(y^i = R^i), \quad \text{Acc}_{\text{hack}} = \frac{1}{|D|} \sum_{i=1}^{|D|} \mathbb{I}(\hat{y}^i = R^i)
$$

**Error Rates** for Self-Enhancement (SE) and Refinement-Aware (RA) biases:

$$
\text{ErrorRate}_{\text{SE}} = \left| 1 - \frac{y_{\text{self}}}{y_{\text{other}}} \right|, \quad \text{ErrorRate}_{\text{RA}} = \left| 1 - \frac{y_{\text{ref}}}{y_{\text{ref}}'} \right|
$$

### Key Quantitative Results
The authors evaluated six models: ChatGPT, GPT-4-Turbo, GPT-4o, GLM-4, Claude-3.5, and Qwen2.

*   **General Resilience**: Claude-3.5 generally demonstrated the highest resilience to biases.
*   **Position Bias**: This bias scales with the number of candidates; when evaluating three or four options, most models' RR dropped below 0.5.
*   **Self-Enhancement**: Significant bias was observed; models consistently rated their own outputs more favorably than those of other models.
*   **CoT Impact**: Step-by-step reasoning generally improved accuracy. For example, GLM-4 saw a 7% increase in accuracy, while GPT-4-Turbo saw a marginal 0.7% increase.
*   **Sentiment and Authority**: Models generally resisted emotionally charged responses (fear had the most significant impact). Regarding authority, book and quote formats were more influential than URL citations.
*   **Robustness Rates**: In the alignment dataset ($D_{\text{AL}}$), RR varied significantly by bias; for example, ChatGPT's RR for position bias was 0.566, while GPT-4o's was 0.776.

### Stated Limitations
The authors acknowledge that absolute objectivity is unattainable, as all judgments carry some degree of subjectivity. They distinguish between **explicit biases** (where the LLM states its preference, e.g., Authority bias) and **implicit biases** (where the judgment changes without the model acknowledging the influence, e.g., Refinement-aware bias). This disparity suggests that LLM internal processing often differs from its expressed reasoning.
