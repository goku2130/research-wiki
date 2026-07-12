---
id: arxiv:2411.04991
type: paper
title: Bradley-Terry Models in Preference-Based Reward Modeling (arXiv)
url: https://arxiv.org/html/2411.04991v1
retrieved: '2026-07-12'
maturity: comprehensive
topic: reward-modeling
---

# Summary: Rethinking Bradley-Terry Models in Preference-Based Reward Modeling

## Core Problem
The Bradley-Terry (BT) model is the standard for converting pairwise preference annotations into reward values for Large Language Model (LLM) alignment. However, its application in reward modeling differs significantly from its original use in multi-player games (e.g., LLM Arenas). In reward modeling, comparisons are extremely sparse—often far below the theoretical lower bound of $\mathcal{O}(N \log N)$ comparisons required for consistent estimation—and the model must predict rewards for unseen pairs. The authors investigate whether the BT model is theoretically sound in this sparse regression setting and whether alternative objectives, such as order consistency, can provide more robust reward signals.

## Method and Recipe
The authors propose a transition from strict BT probability modeling to a broader framework of **Order Consistency**.

### 1. Theoretical Foundation of BT Regression
The authors treat BT reward modeling as a nonparametric logistic regression problem using Multilayer Perceptrons (MLPs). They establish an asymptotic theory for neural network-based BT regression by analyzing the **Truncated KL risk**:

$$
R_{B}(\mathbf{p}_{0},\hat{p})=\mathbb{E}\left[p_{0}^{\top}\text{m i n}\left(B,\text{l o g}\frac{p_{0}}{\hat{p}}\right)\right]
$$

They prove that under smoothness and regularity assumptions, an MLP can approximate the true preference probability $p_0$, which in turn allows the model to approximate the true reward difference up to an additive constant.

### 2. Order-Consistent Reward Modeling
The authors argue that a reward model does not need to predict probabilities accurately; it only needs to preserve the correct ranking through a monotonic transformation. They define **Order Consistency** as the probability that a reward model's ordering agrees with the annotation:

$$
\text{Loc}(\hat{r}) = \mathbb{E}_{x_1,x_2,y_1,y_2,h} [\mathbb{I}(h = \hat{H})]
$$

Based on this, they propose a **Classification-based Reward Model**. Instead of the anti-symmetric Siamese structure required by BT, they use off-the-shelf binary classifiers (e.g., LightGBM or MLP) to predict the preference of a response. The resulting logit is used as a proxy for the reward.

### 3. Cross-Prompt Annotation Strategy
The authors challenge the convention of restricting comparisons to responses from the same prompt. They theoretically demonstrate that cross-prompt comparisons increase utility diversity, which improves annotation quality. They prove that for unimodal symmetric utility distributions, the expected annotation quality $\mathcal{Q}$ is higher for cross-prompt pairs than for same-prompt pairs.

## Key Quantitative Results
The authors conducted over 12,000 experimental runs across 6 base LLMs (Gemma 2b/7b, LLaMA3-8b and their SFT versions) and 2 datasets (Anthropic Harmless and Helpful).

*   **Objective Performance:** Classification-based models (CLF-MLP and CLF-LGB) generally outperformed BT-MLP models in Best-of-N (BoN) sampling.
*   **Robustness to Noise:** Classification models exhibited greater resilience to annotation error. BT models only outperformed classification models when annotation quality was very high (error rates $< 10\%$).
*   **Annotation Quantity:** Classification models showed more consistent performance gains as the number of training annotations increased (from 5,000 to 40,000).
*   **Cross-Prompt Gains:** Cross-prompt annotations significantly improved reward model performance compared to same-prompt annotations. This gain was most pronounced in "Similar Comparison" settings where responses to a single prompt lacked diversity.

## Stated Limitations
*   **BT Model Constraints:** The BT model cannot capture non-transitive preferences and may lead to problematic overfitting in direct preference optimization (DPO) when sampled preferences are deterministic.
*   **Diversity Dependence:** The effectiveness of same-prompt reward modeling is heavily dependent on the diversity of the generated responses; if responses are too similar, the resulting reward model is often uninformative.
*   **Logit Divergence:** The theoretical connection between probability and reward suggests that comparisons should be between pairs relatively close in reward to avoid the diverging behavior of the logit function.
