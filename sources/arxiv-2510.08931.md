---
id: arxiv:2510.08931
type: paper
title: 'RADAR: Mechanistic Pathways for Detecting Data Contamination in LLM Evaluation'
url: https://arxiv.org/abs/2510.08931
retrieved: '2026-07-11'
maturity: comprehensive
topic: judging-bias-and-contamination
---

# RADAR: Mechanistic Pathways for Detecting Data Contamination in LLM Evaluation

### Core Problem
Data contamination occurs when Large Language Models (LLMs) memorize training data, allowing them to achieve high benchmark scores through recall rather than genuine reasoning. Traditional detection methods—such as n-gram overlap or training corpus comparison—are limited because they often require access to the model's training data, fail to detect paraphrased contamination, and cannot distinguish whether a specific correct response was generated via memorization or active computation.

### Method
RADAR (Recall vs. Reasoning Detection through Activation Representation) is a framework that uses mechanistic interpretability to distinguish between recall-based and reasoning-based internal processing. The pipeline consists of three primary stages:

1.  **Mechanistic Analyzer**: The framework interfaces with a target LLM (e.g., DialoGPT-medium) to extract internal states, specifically attention weights and hidden states, during a single forward pass.
2.  **Feature Extraction**: RADAR computes 37 distinct features divided into two categories:
    *   **Surface Features (17)**: These track the model's output trajectory across layers, focusing on confidence statistics (mean, max, min), convergence properties (speed, slope), and information-theoretic measures (entropy, information gain).
    *   **Mechanistic Features (21)**: These analyze deep internal dynamics, including attention specialization (entropy of heads), circuit dynamics (activation flow variance), intervention sensitivity (ablation robustness), and working memory (hidden state rank evolution).
3.  **Classification System**: Feature vectors are normalized using a `StandardScaler` and passed to an ensemble of four supervised classifiers: Random Forest, Gradient Boosting, Support Vector Machine (SVM), and Logistic Regression.

### Key Formulas
**Feature Normalization:**

$$
x_{i}^{\prime}=\frac{x_{i}-\mu_{i}}{\sigma_{i}}
$$

**Ensemble Prediction:**
The final label $\hat{y}$ (where $y=1$ is recall and $y=0$ is reasoning) is determined by the majority vote of $M=4$ classifiers:

$$
\hat{y}=1\left[\frac{1}{M}\sum_{j=1}^{M}\hat{y}_{j}>\frac{1}{2}\right]
$$

**Attention Entropy (for specialization):**

$$
H_{l,h}=-\sum_{i,j}A_{l,h}^{(i,j)}\log A_{l,h}^{(i,j)}
$$

Where $A_{l,h}^{(i,j)}$ is the attention weight from position $i$ to $j$ in head $h$ of layer $l$.

**Specialized Head Count:**

$$
N_{\mathrm{spec}}=\sum_{l=1}^{L}\sum_{h=1}^{H}1[H_{l,h}<\tau]
$$

(Typically using a threshold $\tau=1.5$).

**Circuit Complexity:**

$$
C_{\mathrm{circuit}}=\sigma_{\mathrm{var}}^{2}\cdot\gamma_{\mathrm{norm}}
$$

Where $\sigma_{\mathrm{var}}^{2}$ is activation variance growth and $\gamma_{\mathrm{norm}}$ is the norm growth trajectory.

### Quantitative Results
RADAR was evaluated on a test set of 100 examples, achieving an **overall accuracy of 93.0%**.

| Category | Accuracy |
| :--- | :--- |
| **Overall** | **93.0%** |
| Recall Tasks | 97.7% |
| Reasoning Tasks | 89.3% |
| Clear Recall | 100% (20/20) |
| Clear Reasoning | 100% (20/20) |
| Complex Reasoning | 100% (30/30) |
| Challenging/Ambiguous Cases | 76.7% (23/30) |

Analysis showed a clear separation in the **Recall Detection Score (RDS)**, with a mean of **0.933 for recall** tasks compared to **0.375 for reasoning** tasks. Recall was characterized by higher early confidence, faster convergence, and more specialized attention heads.

### Limitations
The authors state that several mechanistic features rely on **proxy measures** rather than direct experimental computation:
*   **Causal effects and activation patching** are approximated via attention entropy rather than actual interventions.
*   **Critical components** are approximated using the count of specialized heads.
*   **Working memory complexity** is approximated using representation rank evolution.
