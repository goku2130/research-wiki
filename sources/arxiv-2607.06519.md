---
id: arxiv:2607.06519
type: paper
title: 'FreqDepthKV: Frequency-Guided Depth Sharing for Robust KV Cache Compression
  in Long-Context LLM Inference'
url: https://arxiv.org/abs/2607.06519
retrieved: '2026-07-12'
maturity: comprehensive
topic: rollout-generation-infra
---

# FreqDepthKV: Frequency-Guided Depth Sharing for Robust KV Cache Compression

FreqDepthKV is an inference-time KV cache compression method designed for long-context Large Language Model (LLM) inference. It addresses the trade-off between memory reduction and the preservation of layer-specific evidence necessary for retrieval-heavy and reasoning-heavy tasks.

### Core Problem
Long-context inference is bottlenecked by the memory and bandwidth costs of the KV cache. While existing methods reduce the sequence dimension (token eviction) or numerical precision (quantization), they often fail to preserve "high-frequency" layer-specific information. Uniform depth sharing—where adjacent layers share cache entries—can erase decisive evidence localized to specific token-head-layer interactions, leading to shifts in attention logits and degraded accuracy in needle-retrieval or code-generation tasks.

### Method
FreqDepthKV factorizes the KV cache across the depth dimension into shared low-frequency components and sparse high-frequency residuals. The process follows these steps:

1.  **Depth-Frequency Factorization**: Adjacent layers are grouped into blocks of size $B$ (typically $B=4$ for middle layers and $B=2$ for boundary layers). The KV states for a block $b$ and head $h$, denoted as $X_{b,h}^{K} = [K_{\ell,h}]_{\ell \in b} \in \mathbb{R}^{B \times T \times d_h}$, are transformed using a fixed Discrete Cosine Transform (DCT) basis $F_B$:

$$
Z_{b,h}^{K} = F_B X_{b,h}^{K}, \quad Z_{b,h}^{V} = F_B X_{b,h}^{V}
$$

    The first coefficient group is treated as the shared low-frequency component, while subsequent groups encode high-frequency layer deviations.

2.  **Online Probing and Routing**: During the prefill phase, a lightweight probe samples query positions $\mathcal{P}$ (from recent tokens, boundaries, and high-entropy rows). Each head is assigned to one of three modes—`SHARED`, `RESIDUAL`, or `EXACT`—by minimizing a reconstruction-aware routing loss:

$$
\mathcal{L}_{b,h}(m) = \frac{1}{|\mathcal{P}|} \sum_{t \in \mathcal{P}} \left\| \frac{Q_{\ell,h,t} \widehat{K}_{\ell,h}^{(m)\top}}{\sqrt{d_h}} - \frac{Q_{\ell,h,t} K_{\ell,h}^\top}{\sqrt{d_h}} \right\|_2^2 + \lambda \Omega(m)
$$

    where $\widehat{K}_{\ell,h}^{(m)}$ is the reconstructed key cache under mode $m$, and $\Omega(m)$ is the normalized memory cost. The penalty $\lambda$ is adjusted to meet a target memory budget.

3.  **Sparse Residual Selection**: For heads in `RESIDUAL` mode, the system retains high-frequency coefficients only for the most critical tokens. Tokens are scored by the maximum logit perturbation they induce if their residuals are removed:

$$
s_{b,h,t} = \max_{q \in \mathcal{P}} \left| Q_{\ell,h,q} (K_{\ell,h,t} - \widehat{K}_{\ell,h,t}^{\text{SHARED}})^\top \right|
$$

    The top $r_{b,h}$ tokens are selected to retain their high-frequency key and value residuals.

4.  **Decoding**: During autoregressive generation, the attention kernel reconstructs keys and values on-the-fly by broadcasting shared coefficients and adding sparse residuals for indexed tokens, avoiding the materialization of the full cache.

### Key Quantitative Results
Evaluated with a 32k-token prefill window across long-context QA, summarization, and code generation, FreqDepthKV achieved the following:

*   **Task Accuracy**: Exact Match (EM) of **58.3**, F1 of **63.0**, ROUGE-L of **32.5**, and pass@1 of **48.1**. These results closely match the Full KV cache baseline (EM 58.7, F1 63.4).
*   **Efficiency**:
    *   **Decoding Throughput**: 70.4 tokens/s (compared to 38.2 tokens/s for Full KV).
    *   **Time to First Token (TTFT)**: 2.06 seconds.
    *   **Peak KV Memory**: 6.2 GB.
    *   **Effective Compression Ratio**: 3.9$\times$.
*   **Comparison**: FreqDepthKV outperformed MiniCache (3.6$\times$ compression, 56.6 EM) and other baselines like SnapKV and KIVI across all aggregate metrics.

### Limitations
The primary limitation is that the routing decision (Shared, Residual, or Exact) is determined during the prefill stage and remains fixed throughout the decoding process. The authors note that the policy is not "generation-aware," meaning it cannot adapt as the generated sequence reveals new dependencies.
