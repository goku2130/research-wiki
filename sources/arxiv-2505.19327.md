---
id: arxiv:2505.19327
type: paper
title: Paying Alignment Tax with Contrastive Learning
url: https://arxiv.org/abs/2505.19327
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

# Paying Alignment Tax with Contrastive Learning

### Core Problem
The authors address the "alignment tax" associated with debiasing large language models (LLMs). Existing bias mitigation techniques—such as Counterfactual Data Augmentation (CDA), Sentence Debiasing, and Instructive Debiasing—often create a fundamental trade-off: while they may reduce toxicity, they frequently degrade model capabilities, including factual accuracy, knowledge retention, and general coherence. This degradation is most pronounced in smaller models and often manifests as increased model uncertainty, explicit abstention from answering sensitive questions, or the generation of unintelligible/repetitive outputs.

### Method
The proposed framework utilizes contrastive learning to simultaneously mitigate bias and preserve faithfulness by explicitly modeling the boundary between desired and undesired outputs.

#### 1. Data Augmentation
The framework generates high-quality positive and negative example pairs:
*   **Positive Examples:** Created via multi-path backtranslation using Helsinki-NLP models through three intermediate languages (German, French, and Spanish) to ensure linguistic diversity while preserving semantic meaning.
*   **Negative Examples:** Generated through three strategies:
    *   **Adversarial Toxic Generation:** Using GPT-Neo with few-shot prompting to create intentionally biased versions of neutral text.
    *   **Low Confidence Generation:** Using GPT2 and "inverse beam search" to select sequences with the lowest beam scores, targeting outputs prone to hallucination.
    *   **Entity-Based Manipulation:** Utilizing a SpaCy transformer pipeline for named entity recognition (NER) to perform entity swapping and masked regeneration (via RoBERTa-large and FLAN-T5).

#### 2. Model Architecture
The system adds a specialized contrastive head to a base LM. It maintains token-level granularity using a projection head:

$$
h_{i}=\text{G E L U}(W_{1}x_{i}+b_{1})
$$

$$
z_{i}=\mathrm{N o r m a l i z e}(W_{2}h_{i}+b_{2})
$$

To prioritize faithfulness, the model employs **Named Entity-Focused Contrast**, using a binary mask $m_i$ to pool representations of named entities (NEs):

$$
r={\frac{\sum_{i}m_{i}z_{i}}{\sum_{i}m_{i}}}
$$

#### 3. Training Objective
The total loss $\mathcal{L}$ is a weighted combination of cross-entropy loss ($\mathcal{L}_{ce}$) and contrastive loss ($\mathcal{L}_{cl}$):

$$
\mathcal{L}=\mathcal{L}_{c e}+\alpha\mathcal{L}_{c l}
$$

The contrastive loss incorporates dynamic scaling based on the presence of toxicity in the batch:

$$
\mathcal{L}_{c l}=w_{t o x}\cdot\frac{-\sum_{(i,j)\in\mathcal{P}}\text{l o g}(\text{s i m}(r_{i},r_{j}))}{|\mathcal{P}|}
$$

where $w_{tox}$ is a weight applied when toxic content is detected, and $\mathcal{P}$ is the set of valid positive pairs.

### Key Quantitative Results
The authors demonstrate that their method is the first to consistently improve both toxicity and faithfulness simultaneously across different model scales.

*   **Alignment Tax Analysis:** The study found a negative correlation of $-0.549$ between model size and capability degradation, indicating that smaller models suffer more from traditional debiasing.
*   **Reddit TL;DR Summarization Results:**
    *   **GPT2:** Toxicity reduced by $\downarrow 0.007$; Faithfulness increased by $\uparrow 0.018$.
    *   **Phi2:** Toxicity reduced by $\downarrow 0.014$; Faithfulness increased by $\uparrow 0.195$.
    *   **Llama2-7B:** Toxicity reduced by $\downarrow 0.013$; Faithfulness increased by $\uparrow 0.285$.
*   **Comparison with CDA:** While CDA achieved higher toxicity reductions (e.g., $\downarrow 0.053$ to $\downarrow 0.062$ for Llama2-7B), it consistently degraded faithfulness ($\downarrow 0.004$ to $\downarrow 0.005$).

### Limitations
1.  **Model Scale Dependency:** The effectiveness of the framework still varies with model size, suggesting that underlying representation capacity remains a limiting factor.
2.  **Entity Reliance:** The use of named entity detection for faithfulness may not generalize to domains where critical information is conveyed through complex relationships rather than specific entities.
3.  **Subtle Bias:** The toxicity-based dynamic loss scaling may fail to capture subtle forms of bias that do not manifest as explicit toxicity.
