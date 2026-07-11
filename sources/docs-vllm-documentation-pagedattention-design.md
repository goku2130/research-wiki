---
id: docs:vllm-documentation-pagedattention-design
type: web
title: 'vLLM Documentation: PagedAttention Design'
url: https://docs.vllm.ai/en/latest/design/paged_attention/
retrieved: '2026-07-11'
maturity: comprehensive
topic: rollout-generation-infra
---

# PagedAttention Kernel Design

This document describes the design of a multi-head query attention kernel implemented for vLLM, specifically optimized for paged KV caches. The primary objective is to achieve high performance by utilizing a specialized memory layout and access patterns that promote memory coalescing when reading data from global memory to shared memory.

### Core Problem
The kernel must efficiently perform attention calculations where key and value caches are stored in non-contiguous "blocks" (paged memory) rather than a single contiguous tensor. This fragmentation requires a kernel that can map logical sequence tokens to physical block locations while maintaining high GPU throughput.

### Implementation Method
The kernel processes one head for one sequence per thread block. The execution flow follows these steps:

1.  **Query Fetching**: The kernel reads the query token from global memory into shared memory (`q_vecs`). To ensure memory coalescing, each thread in a `THREAD_GROUP_SIZE` fetches a portion of the query token data.
2.  **Key Fetching and QK Calculation**: 
    *   The kernel iterates through the physical blocks of the key cache.
    *   Key data is read from `k_cache` into register memory (`k_vecs`) because it is accessed only once per thread.
    *   A dot product is performed between the shared `q_vecs` and the register-based `k_vecs`. A cross-thread group reduction is used to produce the full dot product result (`qk`) for the entire head size.
3.  **Softmax Normalization**:
    *   **Max Reduction**: The kernel identifies the maximum `qk` value (`qk_max`) across the entire thread block using warp-level shuffle instructions (`VLLM_SHFL_XOR_SYNC`) and shared memory.
    *   **Sum Reduction**: It calculates the sum of exponentials (`exp_sum`) of the shifted logits ($\text{logit} - qk_{max}$) across the thread block.
    *   **Normalization**: The final normalized softmax results are stored in shared memory as `logits`.
4.  **Value Fetching and LV Calculation**:
    *   Value data is retrieved from `v_cache`. Unlike keys, value elements are fetched such that one thread retrieves `V_VEC_SIZE` elements from the same tokens across different rows.
    *   These `v_vecs` are dot-multiplied with the corresponding `logits_vec` and accumulated in local registers (`accs`).
5.  **Final Reduction and Output**:
    *   The accumulated results in `accs` are reduced first within each warp and then across all warps in the thread block using shared memory.
    *   The final result is written to the global memory output pointer `out`.

### Key Formulas
The kernel implements the softmax operation using the following mathematical steps to ensure numerical stability:

$$
m(x) := \max_i x_i
$$

$$
f(x) := [e^{x_1 - m(x)}, \dots, e^{x_B - m(x)}]
$$

$$
\ell(x) := \sum_i f(x)_i
$$

$$
\text{softmax}(x) := \frac{f(x)}{\ell(x)}
$$

### Quantitative Specifications
*   **Memory Access**: `VEC_SIZE` and `V_VEC_SIZE` are configured so that thread groups and individual threads fetch 16 bytes of data per operation.
*   **Example Configuration**: For `scalar_t` as FP16 (2 bytes) and a `THREAD_GROUP_SIZE` of 2:
    *   `VEC_SIZE` = 4
    *   `V_VEC_SIZE` = 8
*   **Hardware Mapping**: 
    *   **Warp**: 32 threads; processes one block of key tokens per iteration.
    *   **Thread Block**: Multiple warps; processes the entire context for one query token and one head.
    *   **Grid**: Shaped as `(num_heads, num_seqs, max_num_partitions)`.

### Limitations
The documentation explicitly states that this is a **historical document** based on the original vLLM paper and no longer describes the current vLLM codebase. Additionally, it omits specific implementation details regarding the exact index calculations for data and the low-level dot multiplication logic.
