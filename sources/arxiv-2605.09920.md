---
id: arxiv:2605.09920
type: paper
title: Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward - arXiv
url: https://arxiv.org/html/2605.09920v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-math-and-code
---

## Summary of "Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward"

### Core Problem

Reinforcement Learning with Verifiable Rewards (RLVR) for Large Language Models (LLMs) is limited by its dependency on gold labels or domain-specific verifiers, restricting its scalability to new tasks and domains. Existing verifier-free alternatives, such as majority voting or intrinsic rewards based on entropy/self-certainty, can suffer from limitations like applicability to open-ended generation or instability and performance degradation during training. There is a need for verifier-free rewards that are broadly applicable to free-form outputs and stable during RL, without relying on auxiliary components beyond the policy model itself.

### Method: Verifier-free Intrinsic Gradient-Norm Reward (VIGOR)

VIGOR proposes an intrinsic reward signal derived solely from the policy model's internal dynamics, specifically the gradient norm of the teacher-forced negative log-likelihood. The method integrates this reward into a Group Relative Policy Optimization (GRPO) framework.

**Step-by-step Recipe:**

1.  **Sample Completions:** For a given prompt $x$, sample a group of $G$ completions $\{y_i\}_{i=1}^G$ from the current policy $\pi_\theta$.
2.  **Compute Average Token-Level Negative Log-Likelihood:** For each completion $y = (y_1, \ldots, y_T)$ of non-padding length $T = |y|$, calculate the average token-level negative log-likelihood:
    $$ \ell_{\text{mean}}(x, y) = \frac{1}{T} \sum_{t=1}^{T} \ell_t(x, y) $$
    where $\ell_t(x,y) = -\log \pi_\theta (y_t\mid x,y_{< t})$ is the per-token NLL.
3.  **Compute Gradient Norm:** Calculate the $\ell_2$ norm of the gradient of the average token-level negative log-likelihood with respect to the model parameters $\theta$:
    $$ ||\mathbf{g}(x,y)||_2 = ||\nabla_{\theta}\ell_{\mathrm{mean}}(x,y)||_2 $$
    This gradient norm is detached from the computation graph, treating it as a scalar reward signal.
4.  **Apply Length Correction:** To counteract a systematic length bias (where $||\mathbf{g}(x,y)||_2$ tends to shrink as $T$ grows, approximately $O(1/\sqrt{T})$), the raw gradient norm is scaled by $\sqrt{T}$:
    $$ S_{\mathrm{GN}}(x, y) = - \sqrt{T} ||\mathbf{g}(x, y)||_2 $$
    The negative sign converts gradient-norm minimization into a reward maximization objective.
5.  **Rank-based Normalization:** The raw signals $\{S_{\mathrm{GN}}(x,y_i)\}_{i=1}^G$ for a group of completions are transformed into a normalized rank-based intrinsic reward. Completions are sorted from worst to best (smaller $S_{\mathrm{GN}}$ indicates larger gradient norms and thus worse completions), assigned an integer rank $\operatorname{rank}_i \in \{0,\ldots,G-1\}$. The normalized reward is:
    $$ R_{\mathrm{GN}}(x, y_i) = 2 \frac{\operatorname{rank}_i}{G - 1} - 1 $$
    This assigns rewards from -1 (worst) to +1 (best).
6.  **Policy Optimization:** The policy $\pi_\theta$ is updated by maximizing a PPO-style objective, using the group-relative advantage $\hat{A}_i$ computed by mean-std normalizing the $\{R_{\mathrm{GN}}(x,y_j)\}_{j=1}^G$ within each prompt group. The objective is:
    $$
    \begin{array}{l} \mathcal {J} (\pi_ {\theta}) = \mathbb {E} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{| y _ {i} |} \sum_ {t = 1} ^ {| y _ {i} |} \min \left(r _ {i, t} (\theta) \hat {A} _ {i}, \bar {r} _ {i, t} \hat {A} _ {i}\right) \right. \\ \left. - \beta \mathrm{KL} \left(\pi_ {\theta}, \pi_ {\text {ref}}\right) \right], \end{array}
    $$
    where $r_{i,t}(\theta)=\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}|x,y_{i,<t})}$ is the token-level probability ratio, $\bar{r}_{i,t}=\operatorname{clip}(r_{i,t}(\theta),1-\epsilon,1+\epsilon)$ is its clipped version, and $\hat{A}_i$ is the group-relative advantage.

### Key Quantitative Results and Numbers

*   **Performance on MATH (Qwen2.5-7B-Base):**
    *   VIGOR improves average math accuracy by **+3.31%** over the RLIF baseline (INTUITOR), achieving **69.77%** average math accuracy (compared to INTUITOR's 66.46%).
    *   Specifically, VIGOR improves GSM8K by **+1.51%** (88.70% vs 87.19%) and AMC by **+8.43%** (44.42% vs 35.99%) over INTUITOR.
    *   VIGOR also exhibits cross-domain transfer, improving average code accuracy by **+1.91%** over INTUITOR (40.42% vs 38.51%) when trained only on math data.
*   **Performance on MATH (Qwen2.5-3B-Base):**
    *   VIGOR achieves **59.14%** average math accuracy, outperforming INTUITOR (57.10%) by **+2.04%**.
    *   Cross-domain transfer to code: VIGOR achieves **27.95%** average code accuracy, outperforming INTUITOR (26.79%) by **+1.16%**.
*   **Performance on CodeContests (Qwen2.5-3B-Base, lightweight setup):**
    *   VIGOR yields gains on code reasoning, improving average code performance from 16.98% (Base) to **23.64%**.
    *   VIGOR improves average math accuracy from 48.34% (Base) to **56.57%**.
*   **Ablation Study:**
    *   Removing $\sqrt{T}$ length correction for Qwen2.5-3B-Base leads to severe degradation, with GSM8K accuracy dropping from 81.80% to **0.08%** and average code accuracy from 27.95% to **0.00%**.
    *   Removing rank shaping for Qwen2.5-7B-Base causes an 8.90% accuracy regression on MMLU-Pro.
*   **Training Dynamics:** VIGOR shows more stable training dynamics compared to INTUITOR, which exhibits late-stage regression in accuracy. VIGOR maintains higher and more stable top-25% accuracy.
*   **Computational Cost (Qwen2.5-7B, 66 steps):**
    *   VIGOR: 3h35m wall-clock time, 28.67 GPU-hours, 74.00 GB average GPU memory.
    *   LM-head-only VIGOR: 2h47m16s wall-clock time, 22.31 GPU-hours, 66.25 GB average GPU memory.

### Stated Limitations

*   **Applicability to Open-ended Generation:** The study primarily targets verifiable reasoning tasks. It is unclear how well VIGOR transfers to more open-ended generation settings (e.g., long-form writing or dialogue), where additional constraints might be needed to avoid pathological optimization.
*   **Computational Overhead:** VIGOR requires per-sample gradient-norm computation, which introduces non-trivial automatic-differentiation overhead compared to forward-only signals. While the LM-head-only variant mitigates this, scaling to substantially larger models may still require further approximations.
*   **Proxy Objective:** Gradient norm is only a proxy objective and may not consistently track downstream utility, potentially leading to misalignment or reward exploitation.
