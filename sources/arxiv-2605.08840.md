---
id: arxiv:2605.08840
type: paper
title: 'ReST-KV: Robust KV Cache Eviction with Layer-wise Output Reconstruction and
  Spatial-Temporal Smoothing'
url: https://arxiv.org/abs/2605.08840
retrieved: '2026-07-12'
maturity: comprehensive
topic: rollout-generation-infra
---

# ReST-KV: Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing

ReST-KV is a KV cache eviction method designed to reduce the memory and latency overhead of Large Language Models (LLMs) during long-context generative inference.

### Core Problem
Existing KV cache eviction methods typically rely on raw attention weights to identify "important" tokens. However, these approaches overlook two critical factors:
1. **Attention Redistribution:** Removing a token alters the attention landscape, redistributing weights among remaining tokens. Simplistic reliance on initial weights ignores how this redistribution affects the final model output.
2. **Spatial-Temporal Dynamics:** Token importance is not static; it fluctuates over time (temporal variations) and shifts across positions (spatial shifts).

### Method
ReST-KV reformulates KV cache eviction as an optimization problem aimed at preserving the attention output at each layer under a fixed memory budget $B$. The process follows these steps:

1. **Layer-wise Output Reconstruction:** Instead of using attention weights, ReST-KV measures the reconstruction error (using $\ell_2$ distance) caused by removing an individual KV pair. This captures the redistribution effect by modeling how the remaining KV pairs fail to compensate for the excluded value.
2. **Temporal Smoothing:** To handle temporal fluctuations, the method applies an Exponential Moving Average (EMA) to the reconstruction indicators over a recent query window $S_w$, assigning higher weights to recent queries.
3. **Adaptive Window-based Spatial Smoothing (AWS):** To capture spatial shifts, the method splits the observation window into front and rear halves to calculate the shift in the average index of top-B important pairs. This signal adaptively adjusts the size and shift of a sliding window used to smooth the indicators.
4. **Eviction:** The final smoothed indicators are used to select the top-B KV pairs for retention.

### Key Formulas
The importance indicator $\mathbf{I}_{t}[n]$ for the $n$-th KV pair is defined as:

$$
\mathbf{I}_{t}[n]=\frac{\mathbf{A}_{t}[n]}{1-\mathbf{A}_{t}[n]}\left\|\text{MHA}\left(\mathbf{x}_{t},\langle\mathbf{K}_{T},\mathbf{V}_{T}\rangle\right)-\mathbf{v}_{n}\mathbf{W}_{O}\right\|_{2}
$$

Temporal smoothing via EMA is computed as:

$$
\text{EMA}(\mathbf{I}_{t_{1}:t_{2}}[n]) = \begin{cases} \alpha I_{t_2}[n] + (1 - \alpha) \text{EMA}(\mathbf{I}_{t_1:t_2-1}[n]), & \text{if } t_1 < t_2 \\ I_{t_1}[n], & \text{elif } t_1 = t_2 \end{cases}
$$

The adaptive spatial window size $W_s$ is determined by the difference $\Delta D$ between the average indices of the rear ($\mathbf{D}_{\text{rear}}$) and front ($\mathbf{D}_{\text{front}}$) halves of the window:

$$
W_{s}=2\cdot\left\lfloor\frac{|\mathbf{D}_{\text{rear}}-\mathbf{D}_{\text{front}}|}{\beta}\right\rfloor+1
$$

The final smoothed indicator $\mathcal{I}_{t}[n]$ is the average over the adaptive window:

$$
\mathcal{I}_{t}[n]=\frac{\sum_{k=-\lfloor W_{s}/2\rfloor+\gamma_{\text{shift}}}^{\lfloor W_{s}/2\rfloor+\gamma_{\text{shift}}}\hat{\mathbf{I}}_{t}[k]}{W_{s}}
$$

### Key Quantitative Results
*   **Benchmark Performance:** ReST-KV outperformed state-of-the-art baselines by **2.58% on LongBench** and **15.2% on RULER**.
*   **Retrieval Accuracy:** In Needle-in-a-Haystack tests (Mistral-7B, $B_{total}=1024L$), it maintained **98% of the full model's performance**. On Llama-3.1-8B, it achieved **100% accuracy** while storing only 1/32 of original tokens.
*   **Efficiency (128k context):**
    *   **Decoding Latency:** Achieved a **10.61$\times$ reduction** compared to full cache.
    *   **Peak Memory:** Reduced peak memory usage by approximately **36.0%**.
    *   **TTFT:** When integrated with FlexPrefill, it achieved a **2.37$\times$ speedup** in Time-To-First-Token.
*   **InfiniteBench:** Achieved an average accuracy of **38.8%**, compared to **36.8%** for SnapKV.
*   **Overhead:** Introduced only $\approx 2\%$ prefill overhead over SnapKV.

### Stated Limitations
The authors note that certain complex tasks, specifically the **Common Word Extraction (CWE) aggregation task** with uniform word distributions and **MK-NIAH-3** (Multi-Key Needle-in-a-Haystack), remain difficult for all tested eviction methods, including ReST-KV.
