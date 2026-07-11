---
id: arxiv:2604.00008
type: paper
title: How Trustworthy Are LLM-as-Judge Ratings for Interpretive Responses? Implications
  for Qualitative Research Workflows
url: https://arxiv.org/abs/2604.00008
retrieved: '2026-07-11'
maturity: comprehensive
topic: llm-as-judge
---

# Summary: How Trustworthy Are LLM-as-Judge Ratings for Interpretive Responses?

## Core Problem
Qualitative research increasingly integrates Large Language Models (LLMs) for interpretive tasks such as summarizing interviews and generating codes. However, practitioners often adopt a single model "as is" without systematic evaluation of its interpretive quality. While "LLM-as-judge" frameworks (using one LLM to evaluate another) have shown success in structured domains, it remains unclear if they can reliably capture the nuance, context, and speaker intent required for qualitative interpretive analysis, or if they can inform model-level selection.

## Method
The researchers employed a comparative pipeline to evaluate the alignment between automated LLM judgments and human expert ratings.

### 1. Data Generation
*   **Dataset:** 712 meaning-bearing conversational excerpts from eight semi-structured interviews with K–12 mathematics teachers.
*   **Inference Models:** Five models generated one-sentence interpretive responses for each excerpt: Command R+ (Cohere), Gemini 2.5 Pro (Google), GPT-5.1 (OpenAI), Llama 4 Scout-17B Instruct (Meta), and Qwen 3-32B Dense (Alibaba).
*   **Prompt:** Models were instructed to convey what the speaker was trying to express without repeating exact wording.

### 2. Automated Evaluation (LLM-as-Judge)
Claude 3.5 Sonnet (via AWS Bedrock) served as the judge, scoring 3,560 responses across five metrics:
*   **Faithfulness:** Grounding in context/absence of hallucinations.
*   **Correctness:** Factual and logical accuracy.
*   **Coherence:** Clarity and logical consistency.
*   **Harmfulness & Stereotyping:** Safety-related metrics.

### 3. Human Evaluation
A stratified purposive sample of 60 responses was rated by four trained evaluators (three per item) using a 1–5 scale across three criteria:
*   **Interpretive Accuracy:** Faithful representation of intended meaning.
*   **Nuance Preservation:** Retention of implicit or layered meaning.
*   **Interpretive Coherence:** Analytic clarity and structural appropriateness.

### 4. Analysis
The study used the Intraclass Correlation Coefficient $\text{ICC}(3,k)$ for human reliability, Spearman rank correlations ($\rho$) for model-level alignment, and Mean Absolute Error ($\text{MAE}$) to measure score magnitude differences.

## Key Quantitative Results
*   **Human Reliability:** Average ratings showed acceptable to good reliability, with $\text{ICC}(3,k)$ ranging from $0.744$ to $0.782$.
*   **Model-Level Alignment:** The composite human score and composite automated scores showed moderate rank-order alignment ($\rho = .60, p = .28$) with an $\text{MAE}$ of $0.91$.
*   **Metric-Specific Performance:**
    *   **Coherence:** Provided the strongest alignment for interpretive accuracy ($\rho = .63, \text{MAE} = 1.44$) and nuance preservation ($\rho = .67, \text{MAE} = 1.75$).
    *   **Faithfulness & Correctness:** Showed very strong correspondence with human judgments of *interpretive coherence* ($\rho = .90, p = .04$; $\text{MAE} = 0.14$ and $0.16$, respectively).
    *   **Safety Metrics:** Harmfulness and Stereotyping showed negative or negligible correlations with all human interpretive criteria.

## Limitations and Divergence Patterns
The study identified a systematic "asymmetric interpretive standard" between humans and the LLM judge at the excerpt level:

1.  **Penalization of Pragmatic Inference:** The LLM judge frequently down-scored responses that captured intended meaning or affect (endorsed by humans) because the information was "not explicitly stated" in the literal text.
2.  **Over-rewarding Surface Alignment:** The LLM judge assigned high Faithfulness scores to responses that mirrored the source text's topical content but were "semantically thin" and lacked meaningful interpretation.
3.  **Structural Over-weighting:** High Correctness scores were often given for structural adequacy (compliance with prompt format) rather than interpretive depth.
4.  **Coherence Metric Ceiling:** The automated Coherence metric lacked diagnostic value, as 59 of 60 sampled responses received a perfect score of $1.0$, functioning as a binary indicator rather than a nuanced scale.

## Conclusion
The authors conclude that LLM-as-judge methods are effective for **model elimination** (identifying and removing underperforming models) but unsuitable for **final model selection**, which still requires human judgment to validate pragmatic meaning and interpretive depth.
