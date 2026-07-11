---
id: arxiv:2601.03746
type: paper
title: Whose Facts Win? LLM Source Preferences under Knowledge Conflicts
url: https://arxiv.org/abs/2601.03746
retrieved: '2026-07-11'
maturity: comprehensive
topic: judging-bias-and-contamination
---

# Summary: Whose Facts Win? LLM Source Preferences under Knowledge Conflicts

### Core Problem
The authors investigate how the attributed source of retrieved information influences the resolution of inter-context knowledge conflicts in Large Language Models (LLMs). While previous research has explored conflicts between parametric and contextual knowledge, the role of source credibility—specifically how LLMs weigh information from different types of sources (e.g., government vs. social media)—remains unexamined. The study specifically aims to isolate source preferences from pre-training biases and determine if these preferences can be overridden by repetition or majority biases.

### Method
The researchers developed a controlled framework using synthetic data to isolate source effects:

1.  **Data Construction**: They created 7,440 conflict pairs by perturbing attributes of fictional entities from the NeoQA dataset. This ensures the models cannot rely on parametric knowledge to resolve conflicts.
2.  **Synthetic Sources**: To avoid real-world political or cultural biases, they generated fictional sources across four types: **Government**, **Newspaper**, **Person**, and **Social Media users**.
3.  **Evaluation Setup**: 13 open-weight LLMs (from the Qwen, OLMo, Llama, and Gemma families) were tested using a multiple-choice question answering (MCQA) format. To control for position bias, every prompt was presented in two versions (original and reversed).
4.  **Metric**: Instead of using generative text, the authors measured the deterministic probabilities of answer tokens. The **Source Preference (SP)** for a model $\theta$ is calculated as the difference in probability for answer $A$ when sources are introduced ($C'$) versus when they are absent ($C$):

$$
SP(\theta,T_{A},x,T_{B},y)=P_{\theta}(A|C^{\prime})-P_{\theta}(A|C)
$$

    Where the normalized probability is defined as:

$$
P_{\theta}(A|C)=\frac{P_{\theta}^{\prime}(A|C)}{\sum_{x\in\{A,B\}}P_{\theta}^{\prime}(x|C)}
$$

    The aggregate measure $\widehat{SP}(\theta;X,Y)$ averages these values over the dataset $D$.

### Key Results
*   **Source Hierarchy**: LLMs exhibit a consistent, transitive credibility hierarchy: $\text{Government} > \text{Newspaper} > \text{Person/Social Media}$. Inter-model agreement was high (average Kendall’s $W = 0.74$).
*   **Intra-Type Preferences**: Models prefer sources with higher "reputed credibility," specifically those with higher newspaper circulation or social media follower counts.
*   **Repetition and Majority Bias**: Source preferences are fragile. Repeating information from a low-credibility source (e.g., two social media sources vs. one government source) can flip the model's preference. 
    *   **2-Table Majority**: Flipped preferences with an average $\widehat{SP}$ gap of $33.90$.
    *   **Repetition (no majority)**: Flipped preferences with an average $\widehat{SP}$ gap of $30.04$.
*   **Prompting Limitations**: Instructing models to consider source credibility and ignore repetition was insufficient to maintain the original source hierarchy.

### Mitigation Strategy
The authors propose a teacher-student knowledge distillation method using LoRA to make models invariant to repetition. The student model $f_s$ is trained to mimic the frozen teacher model $f_t$ using a weighted loss ($\lambda=0.2$) of two Kullback–Leibler (KL) divergences:

$$
\mathcal{L}=\lambda D_{K L}(f_{t}(C_{U}^{\prime})\Vert f_{s}(C_{U}^{\prime})) + (1-\lambda)D_{K L}(f_{t}(C_{U}^{\prime})\Vert f_{s}(C_{R}^{\prime}))
$$

where $C_U'$ is the unattributed/non-repeated context and $C_R'$ is the repeated context.

**Quantitative Results of Mitigation (Gemma-3-4B):**
*   **Repetition Bias Reduction**: Reduced by up to $79.2\%$ (e.g., government vs. no source $\widehat{SP}$ gap dropped from $45.9$ to $9.55$).
*   **Preference Retention**: Maintained at least $72.5\%$ of original source preferences.

### Limitations
*   **Synthetic Setting**: The use of fictional entities and sources may not fully capture the complexity of real-world expertise or topic-specific authority.
*   **Evaluation Strategy**: The study relied exclusively on forced-choice MCQA, omitting step-by-step reasoning or generative answers.
*   **Domain and Language**: The research was limited to factual content in English, utilizing U.S.-centric naming and newspaper templates, which may not generalize across cultures.
