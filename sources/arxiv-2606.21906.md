---
id: arxiv:2606.21906
type: paper
title: 'Deeper is Not Always Better: Mitigating the Alignment Tax via Confident Layer
  Decoding'
url: https://arxiv.org/abs/2606.21906
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

# Confident Layer Decoding: Mitigating the Alignment Tax

**Confident Decoding** is a training-free, drop-in decoding strategy designed to mitigate the "alignment tax"—a phenomenon where the final layers of aligned Large Language Models (LLMs) perturb refined reasoning trajectories toward generic, safe, or high-frequency tokens.

### Core Problem: The Guess–Refine–Perturb Dynamic
The authors challenge the assumption that deeper representations are always more reliable. Through analysis of residual-stream dynamics, they identify a three-phase progression in LLM forward passes:
1.  **Phase I (Guess):** Shallow layers ($l \lesssim 0.15L$) create coarse statistical guesses with high directional volatility.
2.  **Phase II (Refine):** Intermediate layers ($0.15L \lesssim l \lesssim 0.95L$) perform incremental, directionally faithful updates to refine semantic trajectories.
3.  **Phase III (Perturbation):** Final layers ($l \gtrsim 0.95L$) introduce significant representational shifts. In complex reasoning tasks, this "alignment tax" creates a planning–pragmatics tradeoff, where the model's internal reasoning is overridden by alignment-preferred distributions.

### Method: Confident Decoding
Confident Decoding dynamically selects the most reliable near-final layer for each token by identifying an **Entropy Valley**—the local entropy minimum encountered when scanning backward from the final layer.

**Step-by-Step Recipe:**
1.  **Candidate Extraction:** During the standard forward pass, the model collects residual-stream states $\{\mathbf{x}_t^{(\ell)}\}$ for a near-final window of layers $\mathcal{L} = \{L-M+1, \dots, L\}$.
2.  **Logit Projection:** For each candidate layer $\ell$, the state is normalized and projected using the frozen final-layer unembedding matrix $W_U$:

$$
\tilde{\mathbf{h}}_{t}^{(\ell)} = \text{Norm}(\mathbf{x}_{t}^{(\ell)}), \quad \mathbf{z}_{t}^{(\ell)} = W_{U}\tilde{\mathbf{h}}_{t}^{(\ell)}, \quad \mathbf{p}_{t}^{(\ell)} = \text{softmax}(\mathbf{z}_{t}^{(\ell)})
$$

3.  **Entropy Calculation:** The Shannon entropy for each layer is computed:

$$
H_{t}^{(\ell)} = -\sum_{v \in \mathcal{V}} p_{t}^{(\ell)}(v) \log p_{t}^{(\ell)}(v)
$$

4.  **Conservative Backward Search:** The algorithm scans from $\ell = L$ backward. It selects the first layer $\hat{V}$ where moving one layer shallower fails to strictly decrease entropy:

$$
\hat{V} = \max \{l < L \mid \hat{H}(l-1) \geq \hat{H}(l)\}
$$

5.  **Sampling:** The logits $\mathbf{z}_t^{(\hat{V})}$ from the selected layer are forwarded to the sampler.

### Key Quantitative Results
The method was evaluated across dense and Mixture-of-Experts (MoE) architectures (Qwen3.5, gpt-oss, Gemma-4).

*   **Reasoning Gains:** Confident Decoding consistently improved performance on frontier benchmarks. Notable absolute gains include:
    *   **GPQA-Diamond:** +6.5% for Qwen3.5-35B-A3B.
    *   **LiveCodeBench v6:** +9.4% for Qwen3.5-27B.
    *   **Omni-MATH (Level 4/Hardest):** +22.4 points for gpt-oss-20b.
*   **Alignment Tax Validation:** The performance delta was significantly larger for instruction-tuned models than for base models, confirming that Phase III perturbations are a byproduct of post-training alignment.
*   **Stability:** Performance on safety (Air-Bench) and creative writing (WritingBench) remained stable or improved, suggesting that safety guardrails are preserved while "over-conservative" refusals are reduced.
*   **Efficiency:** 
    *   **Latency:** Less than 2% increase per token.
    *   **Memory:** Zero additional KV-cache overhead.
    *   **Surgical Intervention:** Only $\sim 11.5\%$ of tokens trigger a backward scan, and only $\sim 2.47\%$ result in an actual token substitution.

### Limitations
*   **Basis Shift:** The method relies on the final unembedding matrix $W_U$ to probe intermediate layers; while bounded, this introduces projection noise $\epsilon$ and potential vocabulary mismatch in shallower layers.
*   **Symptomatic Treatment:** Confident Decoding mitigates the effects of the alignment tax during inference but does not resolve the root cause during the training phase.
