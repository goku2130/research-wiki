---
id: arxiv:2205.14135
type: paper
title: 'FlashAttention: Fast and Memory-Efficient Exact Attention'
url: https://arxiv.org/abs/2205.14135
retrieved: '2026-07-11'
maturity: comprehensive
topic: rollout-generation-infra
---

**Core Problem**
Transformer self-attention exhibits quadratic time and memory complexity relative to sequence length $N$, primarily because standard implementations materialize the $N \times N$ attention matrix in slow GPU high-bandwidth memory (HBM). Modern GPUs are increasingly memory-bound, meaning attention kernels are bottlenecked by HBM reads/writes rather than arithmetic throughput. While approximate attention methods reduce floating-point operations (FLOPs), they frequently ignore memory access overhead, failing to deliver actual wall-clock speedups.

**Method/Recipe Step by Step**
FLASHATTENTION resolves this by introducing an IO-aware exact attention algorithm that eliminates HBM materialization of the attention matrix. The implementation proceeds through the following steps:
1. **Block Partitioning:** Inputs $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$ are divided into blocks sized to fit within the GPU's on-chip SRAM (size $M$).
2. **Tiled Forward Pass:** The algorithm iterates over blocks of $\mathbf{K}$ and $\mathbf{V}$, loading each block into SRAM. For each $\mathbf{K},\mathbf{V}$ block, it loops over all blocks of $\mathbf{Q}$, computing partial attention scores entirely in SRAM.
3. **Incremental Softmax:** To avoid storing the full attention matrix, softmax normalization is computed incrementally across blocks using algebraic aggregation. The algorithm tracks running row-wise maximums ($m$) and exponentiated sums ($\ell$), updating them after each block to maintain numerical stability.
4. **Kernel Fusion:** All operations (matrix multiplication, masking, dropout, softmax, and final weighted sum) are fused into a single CUDA kernel, loading inputs once from HBM and writing the final output $\mathbf{O}$ back only after computation completes.
5. **Recomputation for Backward Pass:** Instead of materializing $O(N^2)$ intermediate values for gradients, the forward pass saves only $\mathbf{O}$ and the softmax statistics $(m, \ell)$. During backpropagation, the attention matrix and gradients are recomputed on-chip from the loaded blocks, functioning as selective gradient checkpointing that trades FLOPs for drastically reduced HBM transfers.

**Key Formulas & Complexity**
The standard attention computation is defined as $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top$, $\mathbf{P} = \mathrm{softmax}(\mathbf{S})$, and $\mathbf{O} = \mathbf{P}\mathbf{V}$. FLASHATTENTION decomposes the softmax for concatenated vectors $x^{(1)}, x^{(2)}$ as:

$$
m(x) = \max(m(x^{(1)}), m(x^{(2)})), \quad \ell(x) = e^{m(x^{(1)})-m(x)}\ell(x^{(1)}) + e^{m(x^{(2)})-m(x)}\ell(x^{(2)}).
$$

Theoretical analysis proves that standard attention requires $\Theta(Nd + N^2)$ HBM accesses, whereas FLASHATTENTION reduces this to $\Theta(N^2 d^2 M^{-1})$. A proven lower bound demonstrates this IO complexity is asymptotically optimal for exact attention across all SRAM sizes. Extending the method to block-sparse patterns with sparsity ratio $s$ yields an IO complexity of $\Theta(Nd + N^2 d^2 M^{-1} s)$.

**Key Quantitative Results**
Empirical benchmarks demonstrate substantial performance gains. FLASHATTENTION achieves a 15% end-to-end speedup over the MLPerf 1.1 BERT-large training record (sequence length 512), a 3× speedup on GPT-2 (sequence length 1K), and a 2.4× speedup on the Long-Range Arena benchmark (sequence length 1K–4K). The attention computation alone runs up to 7.6× faster than PyTorch baselines. Memory footprint scales linearly with sequence length, reducing HBM usage by up to 20× compared to exact attention baselines. By enabling longer contexts, FLASHATTENTION improves GPT-2 perplexity by 0.7 points at sequence length 4K and increases long-document classification accuracy by 6.4 points. Notably, it enables the first Transformers to achieve better-than-chance performance on the Path-X (16K sequence, 61.4% accuracy) and Path-256 (64K sequence, 63.1% accuracy) challenges. Block-sparse FLASHATTENTION further accelerates training by 2–4× over the dense variant.

**Stated Limitations**
The authors identify three primary limitations. First, implementing IO-aware attention currently requires writing custom CUDA kernels for each architectural variant, demanding significant low-level engineering effort and limiting cross-architecture portability. Second, the current implementation is optimized for single-GPU execution; extending the IO-aware framework to multi-GPU parallelization requires additional analysis of inter-GPU data transfers. Third, the approach highlights a broader need for compilers that can automatically translate high-level attention definitions into optimized, IO-aware CUDA kernels, similar to existing image processing compilers.
