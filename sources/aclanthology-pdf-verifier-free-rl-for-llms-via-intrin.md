---
id: aclanthology:pdf-verifier-free-rl-for-llms-via-intrin
type: web
title: '[PDF] Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward'
url: https://aclanthology.org/2026.findings-acl.1606.pdf
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-math-and-code
---

## Summary of "Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward"

This paper introduces Verifier-free Intrinsic Gradient-Norm Reward (VIGOR), a novel intrinsic reward mechanism for Reinforcement Learning (RL) post-training of Large Language Models (LLMs) that operates without requiring gold labels or domain-specific verifiers.

### Core Problem

Reinforcement Learning with Verifiable Rewards (RLVR) has shown promise for LLM post-training but is limited by its dependency on task-specific, programmatic verifiers (e.g., exact-match checkers for math, unit tests for code). When such verifiers are unavailable, obtaining reliable reward signals is challenging, restricting RLVR's applicability and scalability to new tasks and open-ended generation settings. Existing verifier-free alternatives, such as majority voting or intrinsic rewards based on entropy/confidence, often suffer from limitations like restricted applicability to extractable answers or instability and performance degradation as training progresses. The core problem is the lack of a broadly applicable and stable verifier-free reward for free-form LLM outputs that relies solely on the policy model itself.

### Method/Recipe Step by Step

VIGOR integrates into the Group Relative Policy Optimization (GRPO) framework and involves the following steps:

1.  **Sample Completions:** For a given prompt $x$, sample a group of $G$ completions $\{y_i\}_{i=1}^G$ from the current policy $\pi_\theta$.
2.  **Compute Average Token-Level Negative Log-Likelihood (NLL):** For each completion $y = (y_1, \ldots, y_T)$ of length $T$, calculate its average token-level NLL:
    $\ell_{\text{mean}}(x, y) = \frac{1}{T} \sum_{t=1}^{T} \ell_t(x, y)$, where $\ell_t(x,y) = -\log \pi_\theta (y_t\mid x,y_{< t})$.
3.  **Compute Gradient Norm:** Calculate the $\ell_2$ norm of the gradient of the average token-level NLL with respect to the model parameters $\theta$:
    $\mathbf{g}(x,y)=\nabla_{\theta}\ell_{\mathrm{mean}}(x,y)$, and then $\|\mathbf{g}(x,y)\|_{2}$. This gradient norm is detached from the computation graph.
4.  **Apply Length Correction:** To counteract a systematic length bias where $\|\mathbf{g}(x,y)\|_{2}$ tends to shrink as $T$ grows, the raw gradient norm is multiplied by $\sqrt{T}$ and negated to convert minimization into maximization:
    $S_{\mathrm{GN}}(x, y) = - \sqrt{T} \| \mathbf{g}(x, y) \|_{2}$.
5.  **Apply Rank-based Normalization:** The raw signals $\{S_{\mathrm{GN}}(x,y_i)\}_{i=1}^G$ for a group are sorted from worst to best (smaller $S_{\mathrm{GN}}$ indicates larger gradient norms and worse completions). Each completion $y_i$ is assigned an integer rank $\operatorname{rank}_i \in \{0, \ldots, G-1\}$. The normalized intrinsic reward is then computed as:
    $R_{\mathrm{GN}}(x, y_i) = 2 \frac{\operatorname{rank}_i}{G - 1} - 1$. This maps the worst completion to -1 and the best to +1.
6.  **Compute Group-Relative Advantages:** The rank-normalized rewards $\{R_{\mathrm{GN}}(x,y_j)\}_{j=1}^G$ are standardized within each group to obtain group-relative advantages $\hat{A}_i$:
    $\hat{A}_i = \frac{R(x, y_i) - \frac{1}{G} \sum_{j=1}^{G} R(x, y_j)}{\sqrt{\frac{1}{G} \sum_{j=1}^{G} \left(R(x, y_j) - \hat{R}\right)^2}}$.
7.  **Policy Optimization:** The policy $\pi_\theta$ is updated by maximizing a PPO-style objective that encourages completions with high group-relative advantages, regularized by the KL divergence between the current and reference policy:
    $\mathcal{J}(\pi_{\theta}) = \mathbb{E} \left[ \frac{1}{G} \sum_{i = 1}^{G} \frac{1}{| y_i |} \sum_{t = 1}^{| y_i |} \min \left(r_{i, t}(\theta) \hat{A}_i, \bar{r}_{i, t} \hat{A}_i\right) - \beta \mathrm{KL} \left(\pi_{\theta}, \pi_{\text{ref}}\right) \right]$,
    where $r_{i,t}(\theta)=\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}|x,y_{i,<t})}$ is the token-level probability ratio, and $\bar{r}_{i,t}=\operatorname{clip}(r_{i,t}(\theta),1-\epsilon,1+\epsilon)$ is its clipped version.

### Key Formulas in LaTeX

*   **Average Token-Level Negative Log-Likelihood:**
    $\ell_{\text{mean}}(x, y) = \frac{1}{T} \sum_{t=1}^{T} \ell_t(x, y)$
*   **Length-Corrected Gradient-Norm Signal:**
    $S_{\mathrm{GN}}(x, y) = - \sqrt{T} \| \mathbf{g}(x, y) \|_{2}$
*   **Rank-Normalized Intrinsic Reward:**
    $R_{\mathrm{GN}}(x, y_i) = 2 \frac{\operatorname{rank}_i}{G - 1} - 1$
*   **Group-Relative Advantage:**
    $\hat{A}_i = \frac{R(x, y_i) - \frac{1}{G} \sum_{j=1}^{G} R(x, y_j)}{\sqrt{\frac{1}{G} \sum_{j=1}^{G} \left(R(x, y_j) - \hat{R}\right)^2}}$
*   **Policy Optimization Objective:**
    $\mathcal{J}(\pi_{\theta}) = \mathbb{E} \left[ \frac{1}{G} \sum_{i = 1}^{G} \frac{1}{| y_i |} \sum_{t = 1}^{| y_i |} \min \left(r_{i, t}(\theta) \hat{A}_i, \bar{r}_{i, t} \hat{A}_i\right) - \beta \mathrm{KL} \left(\pi_{\theta}, \pi_{\text{ref}}\right) \right]$

### Key Quantitative Results and Numbers

*   **Qwen2.5-7B-Base post-trained on MATH:**
    *   VIGOR improves average math accuracy by **+3.31%** over the RLIF baseline (INTUITOR).
    *   VIGOR improves average code accuracy by **+1.91%** over the RLIF baseline (INTUITOR) when trained only on math data, demonstrating cross-domain transfer.
    *   Specifically, on Qwen2.5-7B-Base, VIGOR achieves **69.77%** average math accuracy (vs. 66.46% for INTUITOR and 42.58% for Base model) and **40.42%** average code accuracy (vs. 38.51% for INTUITOR and 9.69% for Base model).
    *   VIGOR improves GSM8K accuracy by **+45.64%** and AMC accuracy by **+22.74%** on Qwen2.5-7B over the Base model.
*   **Qwen2.5-3B-Base post-trained on MATH:**
    *   VIGOR achieves **59.14%** average math accuracy (vs. 57.10% for INTUITOR and 48.34% for Base model).
    *   VIGOR achieves **27.95%** average code accuracy (vs. 26.79% for INTUITOR and 16.98% for Base model).
*   **Ablation Study (Qwen2.5-3B-Base, MATH):**
    *   Removing $\sqrt{T}$ length correction leads to catastrophic degradation, with average math accuracy collapsing from **59.14%** to **20.71%** and average code accuracy from **27.95%** to **0.00%**.
    *   Removing rank shaping leads to a slight decrease in average math accuracy from **59.14%** to **58.00%** and average code accuracy from **27.95%** to **27.07%**.
*   **Training Cost (Qwen2.5-7B, 66 steps on 8xH800):**
    *   VIGOR: **3h35m00s** wall-clock time, **28.67** GPU-hours, **74.00 GB** average memory.
    *   INTUITOR: **3h57m35s** wall-clock time, **31.68** GPU-hours, **66.32 GB** average memory.
    *   GT-Reward: **2h34m08s** wall-clock time, **20.55** GPU-hours, **66.23 GB** average memory.
    *   VIGOR (LM-head-only): **2h47m16s** wall-clock time, **22.31** GPU-hours, **66.25 GB** average memory.

### Stated Limitations

*   The study primarily targets verifiable reasoning tasks (math and code); its transferability to more open-ended generation settings (e.g., long-form writing, dialogue) is unclear and may require additional constraints to prevent pathological optimization.
*   VIGOR requires per-sample gradient-norm computation, which introduces non-trivial automatic-differentiation overhead compared to forward-only signals. While an LM-head-only variant mitigates this cost, scaling to substantially larger models might still necessitate further approximations.
*   Gradient norm is only a proxy objective and may not consistently track downstream utility, leaving room for potential misalignment or reward exploitation.
