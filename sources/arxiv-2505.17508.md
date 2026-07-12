---
id: arxiv:2505.17508
type: paper
title: On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning
url: https://arxiv.org/abs/2505.17508
retrieved: '2026-07-12'
maturity: comprehensive
topic: ppo-for-llms
---

# Summary: On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning

### Core Problem
Policy gradient (PG) algorithms are widely used to enhance LLM reasoning, typically employing Kullback-Leibler (KL) regularization to stabilize updates and prevent catastrophic forgetting. However, the design space—including the choice of KL direction (forward vs. reverse), normalization (normalized vs. unnormalized), and specific estimators (e.g., $k_1, k_2, k_3$)—is fragmented. The authors identify a specific off-policy importance-weighting mismatch in Group Relative Policy Optimization (GRPO), where the KL penalty omits an essential importance weight, meaning the resulting gradient does not correspond to the intended off-policy objective.

### Method: The Regularized Policy Gradient (RPG) Framework
The authors propose the **Regularized Policy Gradient (RPG)** view, a unified derivation that specifies the weighting required for various KL variants to ensure the optimized surrogate yields the exact gradient of the intended objective.

#### 1. KL Formulations
RPG distinguishes between normalized and unnormalized KL divergences. For unnormalized measures $\pi_{\text{old}}$ with total mass $Z_{\text{old}} = \int \pi_{\text{old}}(x) dx$, the framework defines:
*   **Unnormalized Forward KL (UFKL):** 

$$
\text{UKL}(\pi_{\text{old}} \| \pi_\theta) = \int_x \pi_{\text{old}}(x) \log \frac{\pi_{\text{old}}(x)}{\pi_\theta(x)} dx + \int_x (\pi_\theta(x) - \pi_{\text{old}}(x)) dx
$$

*   **Unnormalized Reverse KL (URKL):**

$$
\text{UKL}(\pi_\theta \| \pi_{\text{old}}) = \int_x \pi_\theta(x) \log \frac{\pi_\theta(x)}{\pi_{\text{old}}(x)} dx + \int_x (\pi_{\text{old}}(x) - \pi_\theta(x)) dx
$$

The authors prove that the widely used **$k_3$ estimator**, defined as $k_3(y) = y - 1 - \log y$, is equivalent to the unnormalized KL divergence.

#### 2. Optimization Recipe
The RPG framework follows a four-step iterative process:
1.  **Objective Construction:** Define $J(\theta) = \mathbb{E}_{\pi_\theta}[R(x)] - \beta \cdot \text{Divergence}$.
2.  **Gradient Derivation:** Derive the off-policy gradient $\nabla_\theta J(\theta)$ using importance weights $w(x) = \pi_\theta(x)/\pi_{\text{old}}(x)$.
3.  **Surrogate Formulation:** Create either a **fully differentiable surrogate** or a **REINFORCE-style loss** using the stop-gradient operator ($\text{SG}$). For URKL, the differentiable surrogate is:

$$
\mathcal{L}_{\text{URKL}}(\theta) = Z_{\text{old}} \mathbb{E}_{x \sim \tilde{\pi}_{\text{old}}}[-w(x)R(x) + \beta(w(x) \log w(x) - w(x))]
$$

4.  **Policy Update:** Optimize $\theta$ and periodically update the reference policy $\pi_{\text{old}}$.

#### 3. RPG-Style Clip
To manage the high variance of importance weights $w(x)$ in off-policy settings, the authors introduce **RPG-Style Clip**. This adapts the Dual-Clip technique by decomposing the weight into the importance ratio $w(x)$ and a regularized advantage $\hat{A}_{\text{reg}}(x)$. The REINFORCE-style loss becomes:

$$
\mathcal{L}^{\text{RPG-Clip}}(\theta) = -\mathbb{E}_{x \sim \tilde{\pi}_{\text{old}}}[\mathcal{C}(w(x), \text{SG}(\hat{A}_{\text{reg}}(x))) \log \pi_\theta(x)]
$$

where $\mathcal{C}$ clips $w$ to $[1-\epsilon_1, 1+\epsilon_2]$ for positive advantages and enforces a lower bound $c$ for negative advantages.

### Key Quantitative Results
Experiments were conducted using Qwen3-4B and Qwen2.5-7B-Instruct on mathematical reasoning benchmarks (AIME24, AIME25, AMC23) with 8K context length:
*   **Performance Gain:** RPG-REINFORCE with RPG-Style Clip improved accuracy by up to **+6 absolute percentage points** over the DAPO baseline on AIME24 and AIME25.
*   **State-of-the-Art Comparison:** On AIME25, RPG-REINFORCE with RPG-Style Clip achieved **52% accuracy**, surpassing the official Qwen3-4B-Instruct model (**47%**).
*   **Stability:** RPG methods demonstrated more stable training progressions in reward (critic score) and policy entropy compared to GRPO.

### Stated Limitations
The authors note that **RPG-Style Clip** introduces a controllable bias-variance trade-off governed by the clipping parameters $(\epsilon_1, \epsilon_2)$. They conclude that developing principled schedules for these clipping parameters would be a valuable future direction.
