---
id: arxiv:2010.14534
type: paper
title: Goal Misgeneralization in Deep Reinforcement Learning
url: https://arxiv.org/abs/2010.14534
retrieved: '2026-07-11'
maturity: comprehensive
topic: sycophancy-and-misgeneralization
---

# Summary: Unmasking Contextual Stereotypes: Measuring and Mitigating BERT’s Gender Bias

### Core Problem
The research addresses the presence of gender bias in contextualized word embeddings, specifically focusing on BERT. The authors seek to determine if the biases encoded in BERT reflect real-world workforce statistics or are based on stereotypes. Furthermore, the study investigates whether bias mitigation techniques developed for English are portable to morphologically rich languages with grammatical gender, such as German.

### Method
The authors employ a two-phase approach: measuring association bias and mitigating that bias through fine-tuning.

#### 1. Measuring Association Bias
The authors created the **Bias Evaluation Corpus with Professions (BEC-Pro)** in English and German. This template-based corpus uses five sentence patterns combining gender-denoting person words (targets) and profession terms (attributes). Professions were selected from 2019 U.S. Bureau of Labor Statistics data and divided into three groups:
*   **High female participation:** 88.3%–98.7%
*   **Low female participation:** 0.7%–3.3%
*   **Balanced distribution:** 48.5%–53.3%

To measure bias, the authors query BERT's Masked Language Model (MLM). They calculate the probability of a target word $T$ given an attribute $A$ ($p_T$) and compare it to the prior probability ($p_{prior}$) where the attribute is masked. The association score is defined as:

$$
\log \frac{p_T}{p_{prior}}
$$

A positive value indicates the attribute increases the likelihood of the target word relative to the prior.

#### 2. Bias Mitigation
To mitigate bias in the English BERT-BASE model, the authors applied **Name-based Counterfactual Data Substitution (CDS)** to the GAP corpus, swapping the gender of person-denoting words and first names. The model was then fine-tuned on this gender-swapped data using the following specifications:
*   **Optimizer:** AdamW with a linear scheduler and warm-up.
*   **Learning Rate:** $5 \times 10^{-5}$.
*   **Epochs:** 3.
*   **Batch Size:** 1.

### Key Quantitative Results
#### English BERT
The results confirmed that BERT's pre-associations align with real-world workforce statistics (positive for pro-typical pairings and negative for anti-typical pairings). Fine-tuning on the CDS-augmented GAP corpus generally reduced these biases (Wilcoxon tests significant at $p=2e-16$).

Key mean association shifts (pre $\rightarrow$ post) from Table 4:
*   **Male-typical jobs:** Female person words shifted from $-0.35$ to $0.20$ (diff: $0.55$, $r = -0.47$).
*   **Female-typical jobs:** Female person words shifted from $0.50$ to $0.36$ (diff: $-0.14$, $r = -0.32$); Male person words shifted from $-0.68$ to $-0.14$ (diff: $0.55$).
*   **Balanced jobs:** Female person words shifted from $-0.83$ to $0.13$ (diff: $0.96$, $r = -0.58$).

The authors noted that female person words exhibited more extreme pre-association values and were more susceptible to change during fine-tuning than male person words, which remained relatively stable.

#### German BERT
The measurement method failed to capture social gender bias in German. Across all three profession groups, mean associations were consistently higher for female person words ($\approx 2.1$) than for male person words ($\approx 1.4$). The authors attribute this to grammatical gender agreement and the linguistic markedness of the feminine suffix *-in*, which overrides social stereotypes in the model's probability distributions.

### Stated Limitations
The authors identify two primary limitations:
1.  **Model Scope:** The study focused exclusively on the uncased BERT-BASE model, leaving other contextualized embeddings (e.g., RoBERTa, GPT-2, ELMo) unexplored.
2.  **Methodological Subjectivity:** The template-based approach relies on curated lists of words and specific sentence contexts, which may introduce human bias and influence target likelihoods due to BERT's sensitivity to the entire sentence context.
