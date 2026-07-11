---
id: arxiv:2306.03236
type: paper
title: A Study of Global and Episodic Bonuses for Exploration in Contextual MDPs
url: https://arxiv.org/abs/2306.03236
retrieved: '2026-07-11'
maturity: comprehensive
topic: entropy-and-exploration
---

# Summary: A Study of Global and Episodic Bonuses for Exploration in Contextual MDPs

## Core Problem
The authors investigate the effectiveness of **global novelty bonuses** (computed using an agent's entire training history) versus **episodic novelty bonuses** (computed using only the current episode's experience) within **Contextual Markov Decision Processes (CMDPs)**. In CMDPs, environments differ across episodes, making it unclear whether an agent should treat a state as "novel" based on its total experience or its experience within the current context. The central goal is to determine the conditions under which each bonus type succeeds and how to combine them for robust performance.

## Conceptual Framework
The authors propose a framework to explain the performance of these bonuses based on the variance of the optimal value function in representation space across contexts. Let $\psi: \mathcal{S} \to \mathcal{Z}$ be a feature extractor. They define the value function over $\mathcal{Z}$ for a context $c$ as:

$$
V_{\psi,c}^{\star}(z)=\text{i n f}_{s\in\psi^{-1}(z)}V^{\star}(s)
$$

The framework posits that:
1. **Global bonuses fail** when $V_{\psi,c}^{\star}$ varies significantly across contexts. In such cases, a region of $\mathcal{Z}$ visited in one episode may be high-value in a later episode, but the agent will avoid it because the global bonus has already been exhausted.
2. **Episodic bonuses succeed** when $V_{\psi,c}^{\star}$ varies significantly, provided the agent can cover the $\mathcal{Z}$ space within a single episode.
3. **Global bonuses succeed** when $V_{\psi,c}^{\star}$ is stable across contexts, as visiting a high-value region in $\mathcal{Z}$ once benefits the agent across all contexts.

## Method and Recipe
The researchers evaluated various bonus designs and combination strategies across interpretable gridworlds, pixel-based environments, and the MiniHack suite.

### 1. Bonus Definitions
*   **Random Network Distillation (RND) [Global]:** Uses the mean squared error between a fixed random network $\bar{f}$ and a predictor network $f$:

$$
b_{\mathrm{R N D}}(s_{t})=\|f(s_{t})-\bar{f}(s_{t})\|_{2}^{2}
$$

*   **Elliptical Episodic Bonus (E3B) [Episodic]:** Uses a learned feature extractor $\phi$ and an episodic covariance matrix:

$$
b_{\mathrm{E3B}}(s_{t})=\phi(s_{t})^{\top}\Big[\sum_{i=t_{0}}^{t-1}\phi(s_{i})\phi(s_{i})^{\top}+\lambda I\Big]^{-1}\phi(s_{t})
$$

*   **Combined Bonus:** The authors found that **multiplicative combination** is superior to additive combination. A simple version is:

$$
b_{\text{c o m b i n e d}}(s_{t})=\mathbb{I}[N_{e}(\psi(s_{t}))=1]\cdot\frac{1}{\sqrt{N(\psi(s_{t}))}}
$$

### 2. Experimental Pipeline
1.  **Baseline Testing:** Compare global vs. episodic bonuses in environments with varying numbers of contexts $|C|$ (from 1 to $\infty$).
2.  **Scaling:** Apply function-approximation bonuses (RND and E3B) to high-dimensional pixel settings (Habitat and Montezuma's Revenge).
3.  **Ablation on MiniHack:** Combine E3B's elliptical bonus with global bonuses from RND, NovelD, and AGAC using both addition and multiplication.
4.  **Optimization:** Use IMPALA or DD-PPO as the base policy optimizer.

## Key Quantitative Results
*   **Contextual Sensitivity:** In the `MultiRoom` environment with positional encodings, the global bonus's performance dropped from $0.99 \pm 0.00$ ($|C|=1$) to $0.00 \pm 0.00$ ($|C|=\infty$), while the episodic bonus remained robust at $0.87 \pm 0.10$ ($|C|=\infty$).
*   **Pixel-Based Trade-offs:** 
    *   **Habitat:** Episodic bonuses (E3B) significantly outperformed global bonuses (RND/ICM).
    *   **Montezuma's Revenge:** Global bonuses (RND) significantly outperformed episodic bonuses (E3B).
*   **MiniHack SOTA:** Multiplicatively combining E3B with either RND or NovelD produced a statistically significant improvement in median and IQM performance over E3B alone, setting a new state-of-the-art across 16 tasks.

## Stated Limitations
The authors identify two primary technical limitations:
1.  **Sub-optimality of Combined Bonus:** While the multiplicative bonus is more robust across diverse tasks, it does not always match the performance of the single best-performing bonus for a specific task (e.g., it performed worse than the episodic bonus alone in Habitat).
2.  **Simple Combination Strategy:** The study only explored simple additive and multiplicative combinations; it did not investigate adaptive methods for combining bonuses based on real-time environment interaction.
