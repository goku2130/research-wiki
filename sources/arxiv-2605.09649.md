---
id: arxiv:2605.09649
type: paper
title: 'Make Each Token Count: Towards Improving Long-Context Performance with KV
  Cache Eviction'
url: https://arxiv.org/abs/2605.09649
retrieved: '2026-07-11'
maturity: comprehensive
topic: rollout-generation-infra
---

# Summary: Make Each Token Count: Towards Improving Long-Context Performance with KV Cache Eviction

### Core Problem
In long-context inference for Large Language Models (LLMs) and Vision-Language Models (VLMs), the key-value (KV) cache grows linearly with sequence length, creating memory and computational bottlenecks. While existing KV eviction methods aim to compress the cache to approximate full-cache inference, they often degrade model performance. The authors identify **attention dilution** as a primary cause of failure in long contexts: as the number of irrelevant tokens (distractors) increases, the softmax normalization in self-attention spreads mass across these distractors, diluting the attention assigned to useful evidence. Consequently, full-cache inference is not always optimal; selective eviction can actually improve reasoning by suppressing distractors and sharpening attention.

### Method: Dynamic Budget TrimKV (DBTrimKV)
The authors propose a learnable, globally calibrated KV eviction policy that treats eviction as a mechanism for improving reasoning rather than mere compression.

**Step-by-Step Recipe:**
1.  **Retention Gate Implementation:** For every token at position $t$, layer $\ell$, and head $h$, a lightweight retention gate $g_{\ell,h}$ predicts a scalar retention coefficient $\beta_{\ell,h,t} \in [0, 1]$ based on the token embedding $x_t$.
2.  **Global Calibration via Weight Tying:** To make scores comparable across different layers and heads, the final scoring projection $(\mathbf{w}_g, b_g)$ of all retention gates is shared (tied). This ensures that a score in one head has the same meaning as a score in another.
3.  **Geometric Retention:** The model employs a geometric decay form for retention weights: $r_{t,i} = \beta_i^{t-i}$. This serves as a query-agnostic proxy for a token's "future utility" or persistence.
4.  **Global Budget Allocation:** Instead of fixed per-layer or per-head budgets, the system maintains a single global KV budget $M_{global}$.
5.  **Global Ranking and Eviction:** At each decoding step $t$, tokens are ranked by their predicted future utility $\tilde{G}$. The top $M_{global}$ tokens across all layers, heads, and modalities are retained, while others are evicted.
6.  **Training:** The gates are trained to match a full-cache teacher model while minimizing a capacity loss $\mathcal{L}_{cap}$ to enforce the global memory constraint.

### Key Formulas
**Attention Dilution ($\delta_t$):** The fraction of attention mass assigned to distractors:

$$
\delta_{t}:=1-\sum_{i\in U_{t}}\alpha_{t,i}
$$

where $U_t$ is the subset of useful tokens.

**Retention-Gated Attention:**

$$
\tilde{\alpha}_{t,i}=\frac{\beta_{i}^{t-i}e^{z_{t,i}}}{\sum_{j\in C_{t}}\beta_{j}^{t-j}e^{z_{t,j}}}
$$

**Global Future Utility Score ($\tilde{G}$):**

$$
\widetilde{G}_{\ell,h,i}(t)=\sum_{s=t+1}^{T}\beta_{\ell,h,i}^{s-i}=\beta_{\ell,h,i}^{t+1-i}\,\frac{1-\beta_{\ell,h,i}^{T-t}}{1-\beta_{\ell,h,i}}
$$

**Capacity Loss:**

$$
\mathcal{L}_{\mathrm{c a p}}=\sum_{t=1}^{T}\mathrm{m a x}\left(0,\,\sum_{\ell,h}\sum_{i=1}^{t}\beta_{\ell,h,i}^{t-i}-M_{\mathrm{g l o b a l}}\right)
$$

### Key Quantitative Results
*   **Performance Gains over Full Cache:** DBTrimKV can exceed vanilla full-cache performance. On long-form generation tasks, it surpassed vanilla inference by up to **3.75%**. On the LongBench-v2 benchmark (using Phi-3-mini-128K), it achieved a **9.20% average accuracy improvement** over the Full KV baseline.
*   **VLM Reasoning:** In image and video reasoning benchmarks, DBTrimKV consistently outperformed other eviction methods (e.g., SnapKV, AdaKV). For example, on MathVisionmini with a budget of 256, DBTrimKV achieved **69.91%** accuracy compared to **65.67%** for TrimKV and **50.89%** for SnapKV.
*   **Multi-turn Dialogue (MMDU):** At a highly constrained budget of 128, DBTrimKV's overall score was **3.43** (vs. Vanilla's **3.57**), but it specifically exceeded vanilla in **Visual Perception (3.54 vs. 3.40)** and **Logical Coherence (4.11 vs. 3.90)**.
*   **Efficiency:** DBTrimKV remains highly efficient using PagedAttention, outperforming vanilla inference at scale as context and generation lengths increase.

### Stated Limitations
The authors note that DBTrimKV introduces a small computational overhead compared to the non-global TrimKV due to the requirements of handling variable-length, head-specific caches. While still significantly faster than vanilla inference, this overhead is a result of the dynamic budgeting mechanism.
