---
id: arxiv:2505.12843
type: paper
title: Bias Fitting to Mitigate Length Bias of Reward Model in RLHF
url: https://arxiv.org/html/2505.12843v1
retrieved: '2026-07-12'
maturity: comprehensive
topic: length-and-format-bias
---

# FiMi-RM: Bias Fitting to Mitigate Length Bias of Reward Model in RLHF

### Core Problem
Reinforcement Learning from Human Feedback (RLHF) relies on reward models (RMs) to align large language models (LLMs) with human preferences. However, these models often suffer from **length bias**, a form of reward hacking where the RM assigns higher scores to longer responses regardless of their actual quality. Existing mitigation strategies typically either fail to characterize the specific form of the bias or assume a simplistic linear relationship between response length and reward. This prevents them from capturing complex, non-linear bias patterns, leading to policies that generate unnecessarily verbose outputs.

### Method: FiMi-RM Framework
FiMi-RM (Bias Fitting to Mitigate Length Bias of Reward Model) is a multi-stage framework designed to autonomously learn and correct non-linear length-reward correlations. It utilizes two models: a standard Reward Model ($model_r$) and a lightweight Fitting Model ($model_f$).

#### Step-by-Step Recipe
1.  **Warm-Up Stage**: A standard reward model is trained using the Bradley-Terry model. This stage deliberately preserves the inherent length bias to provide a baseline for the fitting model to characterize.
2.  **Bias-Fitting Stage**: A lightweight model ($model_f$) is trained to map response length to the reward score. 
    *   **Architecture**: The model uses Length Encoding (LE) to embed length information, followed by a ResNet architecture and a final linear projection layer (regression head).
    *   **Objective**: The model minimizes the discrepancy between the predicted reward $\hat{r}$ and the actual reward $r$ by maximizing the Pearson correlation coefficient.
3.  **Debiasing Stage**: The reward model is retrained to decouple its outputs from length dependence while maintaining its ability to model human preferences.
    *   **Training Schedule**: The RM and the fitting model are trained in an alternating fashion (e.g., every 8 steps).
    *   **Objective**: A composite loss is used to ensure the RM's output is uncorrelated with the fitting model's predicted bias while still satisfying the Bradley-Terry preference objective.

### Key Formulas
**Bradley-Terry Loss (Warm-up and Debiasing):**

$$
\mathcal{L}_{\mathsf{B T}}=-\mathbb{E}_{(x,y_{w},y_{l})}\left[\log\sigma\left(r(x,y_{w})-r(x,y_{l})\right)\right]
$$

**Fitting Model Loss:**

$$
\mathcal{L}_{fit}=-\mathcal{L}_{pearson}, \quad \mathcal{L}_{pearson}=\left|\rho\left(\boldsymbol{r}_{sg},\hat{r}\right)\right|
$$

Where $r_{sg}$ is a stop-gradient copy of the reward and $\rho$ is the Pearson correlation coefficient:

$$
\rho(r,\hat{r})=\frac{\sum_{i=1}^{n}(r_{i}-\bar{r})(\hat{r}_{i}-\bar{\hat{r}})}{\sqrt{\sum_{i=1}^{n}(r_{i}-\bar{r})^{2}}\sqrt{\sum_{i=1}^{n}(\hat{r}_{i}-\bar{\hat{r}})^{2}}}
$$

**Debiased Composite Loss:**

$$
\mathcal{L}_{debiased}=\mathcal{L}_{pearson}'+\mathcal{L}_{B T}
$$

**Length Encoding (LE):**

$$
\text{L E}(\mathsf{l e n}(y))=\Bigg[\text{s i n}\left(\frac{\mathsf{l e n}(y)}{10000^{2j/d}}\right),\text{c o s}\left(\frac{\mathsf{l e n}(y)}{10000^{2j/d}}\right)\Bigg]_{j=0}^{\frac{d}{2}-1}
$$

### Key Quantitative Results
Experiments were conducted using Qwen2.5-7B and Gemma2-9B models on the HH dataset.

*   **Preference Accuracy**: FiMi-RM achieved a more balanced accuracy across subsets. For Qwen2.5-7B, the gap between accuracy on "chosen-longer" (C-longer) and "rejected-longer" (R-longer) samples was reduced from **80.72% vs 56.88%** (Vanilla RM) to **73.60% vs 65.70%**.
*   **Length-Controlled Win Rate (LC-WR)**:
    *   **Best-of-N (BoN)**: Qwen2.5-7B LC-WR increased from **68.25%** (Vanilla) to **72.59%**. Gemma2-9B LC-WR increased from **62.91%** to **66.68%**.
    *   **DPO**: Qwen2.5-7B LC-WR increased from **58.56%** to **62.17%**.
*   **Generalization**: On MT-Bench (Qwen2.5-7B), FiMi-RM achieved an average score of **4.82** compared to **4.44** for the Vanilla RM.
*   **Computational Efficiency**: Because the fitting model is very small ($\approx 6.4\text{K}$ parameters), total training time per epoch was reduced by approximately **25%** compared to the Vanilla RM.

### Stated Limitations
The authors note that it remains an open question whether human preferences are entirely independent of length, as humans often favor detailed responses in specific tasks (e.g., open-ended QA). Additionally, they acknowledge that stronger alignment capabilities could potentially be misused to align models with harmful content, necessitating stronger regulation.
