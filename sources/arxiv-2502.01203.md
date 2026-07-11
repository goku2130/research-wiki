---
id: arxiv:2502.01203
type: paper
title: 'KL-Regularized RLHF with Multiple Reference Models: Exact Solutions and Sample
  Complexity'
url: https://arxiv.org/abs/2502.01203
retrieved: '2026-07-11'
maturity: comprehensive
topic: kl-regularization
---

This paper, "KL-Regularized RLHF with Multiple Reference Models: Exact Solutions and Sample Complexity," addresses the limitations of single-reference models in Reinforcement Learning from Human Feedback (RLHF) for Large Language Models (LLMs) by introducing a framework for incorporating multiple reference models. The core problem is the lack of exact solutions and theoretical guarantees for RLHF when multiple reference models are used, which restricts diversity, can lead to model overfitting, and underutilizes available pre-trained models.

**Method/Recipe Step by Step:**

1.  **Problem Formulation:** The paper extends the standard RLHF objective to include multiple reference models.
    *   For reverse KL-regularized RLHF, the objective is to maximize the expected reward while penalizing deviation from a weighted combination of multiple reference policies.
    *   For forward KL-regularized RLHF, the objective is similar, but the KL divergence is calculated in the forward direction.
2.  **Exact Solution for Reverse KL-Regularized RLHF:** The paper derives an exact closed-form solution for the policy that maximizes the reverse KL-regularized objective with multiple reference models.
3.  **Exact Solution for Forward KL-Regularized RLHF:** An implicit solution is provided for the policy that maximizes the forward KL-regularized objective with multiple reference models.
4.  **Statistical Analysis and Sample Complexity:** The paper provides theoretical guarantees for both reverse and forward KL-regularized scenarios, including upper bounds on the sub-optimality and optimality gaps, and analyzes their sample complexity.
5.  **Extension to DPO:** The derived solutions for both reverse and forward KL-regularized RLHF are extended to the Direct Preference Optimization (DPO) framework.
6.  **Experimental Validation:** Experiments are conducted using online policy gradient and offline RLHF settings to evaluate the benefits of multiple reference models and compare the exact analytical solution against approximations.

**Key Formulas in LaTeX:**

*   **Reverse KL-regularized RLHF objective with multiple reference models:**

$$
\max _ {\pi} \mathbb {E} _ {Y \sim \pi (\cdot | x)} [ r (x, Y) ] - \frac {1}{\gamma} \left(\sum_ {i = 1} ^ {K} \alpha_ {i} \mathrm{KL} (\pi (\cdot | x) \| \pi_ {\text {ref}, i} (\cdot | x))\right)
$$

    where $\alpha_{i}$ are weighting coefficients such that $\sum_{i=1}^{K}\alpha_{i}=1$.

*   **Exact solution for reverse KL-regularized RLHF with multiple reference models:**

$$
\pi_ {\theta^ {*}} ^ {\gamma} (y | x) = \frac {\widehat {\pi} _ {\boldsymbol {\alpha} , \text {ref}} (y | x)}{\widehat {Z} (x)} \exp (\gamma r _ {\theta^ {*}} (x, y))
$$

    where

$$
\widehat {\pi} _ {\boldsymbol {\alpha}, \text {ref}} (y | x) = \frac {\prod_ {i = 1} ^ {K} \pi_ {\text {ref} , i} ^ {\alpha_ {i}} (y | x)}{F _ {\boldsymbol {\alpha}} (x)}, \quad F _ {\boldsymbol {\alpha}} (x) = \sum_ {y \in \mathcal {Y}} \prod_ {i = 1} ^ {K} \pi_ {\text {ref}, i} ^ {\alpha_ {i}} (y | x)
$$

    and $\widehat{Z}(x) = \sum_{y} \widehat{\pi}_{\boldsymbol{\alpha},\text{ref}}(y|x) \exp (\gamma r(x,y))$.

*   **Forward KL-regularized RLHF objective with multiple reference models:**

$$
\max_{\pi} \mathbb{E}_{Y \sim \pi(\cdot|x)}[r(x, Y)] - \frac{1}{\gamma} \left( \sum_{i=1}^{K} \beta_i \text{KL}(\pi_{\text{ref},i}(\cdot|x) \| \pi(\cdot|x)) \right)
$$

    where $\beta_i$ are weighting coefficients such that $\sum_{i=1}^{K} \beta_i = 1$.

*   **Implicit solution for forward KL-regularized RLHF with multiple reference models:**

$$
\tilde{\pi}_{\theta^*}^{\gamma}(y|x) = \frac{\bar{\pi}_{\boldsymbol{\beta}, \text{ref}}(y|x)}{\gamma(\tilde{Z}(x) - r_{\theta^*}(x, y))}
$$

    where $\bar{\pi}_{\boldsymbol{\beta}, \text{ref}}(y|x) = \sum_{i=1}^{K} \beta_i \pi_{\text{ref},i}(y|x)$, and $\tilde{Z}(x)$ is the solution to $\int_{y \in \mathcal{Y}} \tilde{\pi}_{\theta^*}^{\gamma}(y|x) = 1$.

*   **Upper bound on sub-optimality gap for reverse KL-regularized RLHF (Theorem 5.2):**

$$
\mathcal{J}^{\gamma}(\pi_{\theta^*}^{\gamma}(\cdot|x), \pi_{\hat{\theta}}^{\hat{\gamma}}(\cdot|x)) \leq \gamma C_{\boldsymbol{\alpha}, \varepsilon_{\text{rkl}}} 128 e^{4R_{\max}} R_{\max}^2 \frac{\log(|\mathcal{R}|/\delta)}{n}
$$

*   **Upper bound on sub-optimality gap for forward KL-regularized RLHF (Theorem 6.2):**

$$
\hat {\mathcal {J}} ^ {\gamma} \left(\tilde {\pi} _ {\theta^ {*}} ^ {\gamma} (\cdot | x), \tilde {\pi} _ {\theta} ^ {\gamma} (\cdot | x)\right) \leq 1 6 C _ {\boldsymbol {\beta}, \varepsilon_ {\mathrm{rkl}}} e ^ {2 R _ {\max}} R _ {\max} \sqrt {\frac {\log (| \mathcal {R} | / \delta)}{n}}
$$

**Key Quantitative Results and Numbers:**

*   **Sample Complexity for Reverse KL-regularized RLHF:**
    *   Sub-optimality gap: $O(1/n)$
    *   Optimality gap: $O(1/\sqrt{n})$
*   **Sample Complexity for Forward KL-regularized RLHF:**
    *   Sub-optimality gap: $O(1/\sqrt{n})$
    *   Optimality gap: $O(1/\sqrt{n})$ (when $\gamma = n$)
*   **Experimental Results (Win Rate on UltraFeedback with Qwen 2.5 7B):**
    *   Base model: 8.6%
    *   SFT model: 43.4%
    *   DPO (single reference – SFT model): 56.4%
    *   DPO (single reference – 14B model): 59.8%
    *   **Ours (DPO with both references): 66.1%** (This indicates a substantial gain in performance by combining both references).
*   **Dependency on $R_{\max}$ and Coverage Constant:** All existing bounds for RKL regularization scale as $O(\exp(R_{\max}))$ and $O(C_{\alpha,\varepsilon_{\mathrm{rkl}}})$.

**Stated Limitations:**

*   The main limitation is the **assumption of bounded reward functions**. While the paper notes that this assumption can be relaxed under certain growth conditions or by applying monotonic, bounded transformations to rewards, it remains a key constraint.
*   Future work aims to address this by extending analysis to unbounded or sub-Gaussian reward functions.
*   Other future directions include investigating general $f$-divergences, preference models beyond Bradley–Terry, combining with doubly robust preference-optimization algorithms, and extending inference-time algorithms to the multiple-reference setting.
