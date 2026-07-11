---
id: arxiv:2308.16369
type: paper
title: 'xLLM: A Unified Serving System for Large Language Models'
url: https://arxiv.org/abs/2308.16369
retrieved: '2026-07-11'
maturity: comprehensive
topic: rollout-generation-infra
---

**Core Problem**
Large Language Model (LLM) inference is fundamentally inefficient due to the divergent computational characteristics of its two execution phases. The *prefill* phase processes input prompts in parallel, saturating GPU compute even at a batch size of one. Conversely, the *decode* phase generates tokens autoregressively, operating as a memory-bound workload that severely underutilizes GPUs. At small batch sizes, the decode cost per token can reach approximately 200× the prefill cost. When deployed across multiple GPUs using pipeline parallelism, these disparate compute requirements create micro-batch execution variance. This variance manifests as pipeline bubbles—periods of GPU inactivity caused by varying prompt lengths, prefill/decode compute mismatches, and differing KV cache lengths—resulting in substantial throughput degradation.

**Methodology & Implementation**
SARATHI addresses these bottlenecks through a structured, step-by-step scheduling methodology:
1. **Chunked-Prefills:** A single prefill request is partitioned into equal-sized compute chunks. To maintain mathematical equivalence with a full prefill, the system dynamically adjusts the attention mask across iterations, ensuring each query token can only attend to preceding keys and values.
2. **Decode-Maximal Batching:** Hybrid batches are constructed by allocating one slot to a prefill chunk and filling the remaining slots with decode requests. This design ensures uniform compute units across all micro-batches.
3. **Operation Fusion:** To exploit the hybrid batch, SARATHI fuses the linear operations (preproj, postproj, and FFN layers) of the prefill chunk and the piggybacking decodes into a single matrix-matrix multiplication. This reuses fetched model weights, effectively converting the memory-bound decode phase into a compute-bound operation.
4. **Chunk Size Optimization:** The system selects a chunk size that balances prefill efficiency against decode piggybacking capacity. Selection must also account for GPU tile quantization (e.g., 128×128 tiles) to prevent thread-block waste when matrix dimensions are not multiples of the tile size.
5. **Pipeline Scheduling:** By guaranteeing uniform compute requirements across all hybrid batches, SARATHI eliminates micro-batch imbalance, drastically reducing pipeline bubbles in parallel deployments.

**Key Formulas**
The maximum permissible batch size $B$ is constrained by available GPU memory $M_G$, model parameter memory $M_S$, maximum sequence length $L$, and per-token KV cache size $m_{kv}$:

$$
B = \lfloor \left( \frac{M_G - M_S}{L * m_{kv}} \right) \rfloor
$$

Peak throughput is achieved when the prefill-to-decode token ratio ($P:D$) aligns with the chunk size $C$ and batch size $B$:

$$
P:D = \frac{C}{B - 1}
$$

**Quantitative Results**
Empirical evaluations demonstrate substantial performance gains across diverse hardware and model scales. On an A6000 GPU with LLaMA-13B, SARATHI improves decode throughput by up to 10× and end-to-end throughput by up to 1.33×. For LLaMA-33B on an A100 GPU, decode throughput increases by up to 4.25× with a 1.25× end-to-end speedup. In a simulated 64-GPU GPT-3 deployment using 8-way pipeline parallelism, the system reduces pipeline bubble time by 6.29×, yielding a 1.91× improvement in end-to-end throughput. Ablation studies confirm that decode-maximal batching reduces per-token decode time from 12.49 ms to 1.2 ms, while chunked-prefills incur manageable overheads, limiting prefill efficiency loss to under 20% for optimal chunk sizes.

**Stated Limitations**
The authors acknowledge several constraints. SARATHI exclusively targets throughput optimization via scheduling, omitting concurrent improvements to latency, queuing delays, or fairness guarantees. The optimal chunk size depends on known workload characteristics (P:D ratio, hardware FLOPS/memory bandwidth, and model architecture), leaving dynamic chunk selection for future research. Furthermore, experimental evaluations assume uniform sequence lengths within batches, whereas production workloads exhibit significant length variance. Finally, the system was tested only up to 3K sequence lengths and P:D ratios between 1 and 200; supporting ultra-long contexts (tens to hundreds of thousands of tokens) remains an open challenge due to the quadratic scaling of attention costs.
