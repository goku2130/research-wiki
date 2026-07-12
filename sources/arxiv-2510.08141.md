---
id: arxiv:2510.08141
type: paper
title: Entropy Is Controllable in Reinforcement Fine-tuning
url: https://arxiv.org/html/2510.08141v3
retrieved: '2026-07-12'
maturity: comprehensive
topic: entropy-and-exploration
---

# SCOPE-RL: Stable and Quantitative Control of Policy Entropy in RL Post-Training

### Core Problem
Reinforcement learning (RL) post-training for reasoning Large Language Models (LLMs), particularly when using Group Relative Policy Optimization (GRPO), frequently suffers from **entropy collapse**. This phenomenon occurs when policy entropy decreases monotonically, leading to homogeneous outputs and the premature disappearance of exploration. While existing remedies—such as entropy bonuses or asymmetric clipping—attempt to mitigate this, they often introduce unstable training dynamics, causing entropy to oscillate between collapse and explosion and resulting in reward degradation.

### Method: SCOPE-RL
The authors identify a **temperature–sign asymmetry** in entropy dynamics: under high-temperature sampling, positive samples promote entropy growth, whereas negative samples accelerate collapse. Conversely, under low-temperature sampling, negative samples increase entropy but cause rapid reward collapse.

SCOPE-RL leverages this asymmetry by introducing a lightweight regularization term based on temperature-adaptive positive samples. The "recipe" for the framework is as follows:

1.  **Monitor Entropy:** Calculate the policy entropy $\mathcal{H}(\pi_{\theta_{\text{old}}})$ of the previous training step.
2.  **Adaptive Temperature Scaling:** Determine a sampling temperature $T$ based on a target entropy threshold $H_0$. If current entropy is below $H_0$, $T$ increases to encourage exploration; if above $H_0$, $T$ decreases to maintain stability.
3.  **Positive Sample Filtering:** Draw a small number of auxiliary samples using the adaptive temperature $T$, retaining only those with a positive reward ($R=1$).
4.  **Objective Integration:** Augment the standard GRPO objective with a regularization term derived from these temperature-adjusted positive samples.

#### Key Formulas
The temperature $T$ is calculated as:

$$
T = \text{clip}(1 + \mathcal{H}_0 - \mathcal{H}(\pi_{\theta_{\text{old}}}), 0.8, 1.2)
$$

The SCOPE-RL objective is defined as:

$$
\mathcal{J}_{\text{SCOPE}}(\theta) = \mathcal{J}_{\text{GRPO}}(\theta) + \alpha \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^{G'} \sim \pi_{\theta_{\text{old}}}^T(O|q)} \left[ \frac{1}{G'} \sum_{i=1}^{G'} \mathbf{1}[R(q, o_i) = 1] \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min(r_{i,t}(\theta), \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)) \right]
$$

where $r_{i,t}(\theta) = \frac{\pi_{\theta}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$ and $\alpha$ represents the auxiliary sample ratio.

### Key Quantitative Results
Experiments were conducted on Qwen2.5-Math-7B, Qwen2.5-7B, and Qwen3-4B using the DAPO-17K dataset.

*   **Reasoning Performance:** SCOPE-RL consistently outperformed GRPO across all base models in average benchmark scores:
    *   **Qwen2.5-Math-7B:** Average score increased from 51.60 (GRPO) to 54.84 (+3.23).
    *   **Qwen2.5-7B:** Average score increased from 47.71 (GRPO) to 49.81 (+2.10).
    *   **Qwen3-4B:** Average score increased from 67.77 (GRPO) to 69.73 (+1.96).
*   **Many-Sample Performance (Pass@k):** On Qwen2.5-7B, SCOPE-RL significantly improved Pass@1024 over GRPO on AIME24 (86.7 vs 73.3) and AIME25 (76.7 vs 63.3).
*   **Exploration Trade-off:** The relationship between entropy and performance is non-monotonic. The best average performance (54.84) was achieved at a moderate target entropy of $H_0 = 0.50$, outperforming $H_0 = 0.25$ (53.00) and $H_0 = 0.75$ (53.45).
*   **Efficiency:** Effective entropy control was achieved using a very small auxiliary budget ($\alpha = 1/64$), representing less than 2% additional auxiliary samples.

### Stated Limitations
The theoretical analysis (Theorem 1) relies on simplifying assumptions, including a tabular softmax policy, binary rewards, and orthogonal logit gradients. Furthermore, the proof specifically focuses on the regime where entropy is decreasing ($\Delta\mathcal{H} < 0$), which characterizes entropy collapse. The authors note that the theory does not fully cover regimes where entropy is already increasing ($\Delta\mathcal{H} > 0$), making quantitative bounds harder to derive for those specific cases.
