---
id: arxiv:2309.06180
type: paper
title: Efficient Memory Management for Large Language Model Serving with PagedAttention
  (vLLM)
url: https://arxiv.org/abs/2309.06180
retrieved: '2026-07-11'
maturity: comprehensive
topic: rollout-generation-infra
---

# Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)

## Core Problem
High-throughput serving of Large Language Models (LLMs) is primarily **memory-bound** due to the Key-Value (KV) cache. In autoregressive generation, the KV cache grows dynamically, but existing serving systems (e.g., FasterTransformer, Orca) require tensors to be stored in contiguous memory. This leads to three types of memory waste:
1. **Internal Fragmentation:** Pre-allocating memory based on the maximum possible sequence length, even if the actual length is much shorter.
2. **Reserved Slots:** Memory reserved for future tokens that remains unused for the duration of the request.
3. **External Fragmentation:** Gaps created by the memory allocator.

Profiling reveals that in existing systems, only **20.4% to 38.2%** of the KV cache memory is used to store actual token states. Additionally, contiguous allocation prevents efficient memory sharing during complex decoding (e.g., beam search), requiring expensive memory copies.

## Method: PagedAttention and vLLM
vLLM introduces **PagedAttention**, an algorithm inspired by virtual memory and paging in operating systems. It allows the KV cache to be stored in non-contiguous physical memory.

### Step-by-Step Recipe
1. **Block Partitioning:** The KV cache for each sequence is divided into fixed-size **KV blocks** (default size $B=16$). Each block contains the key and value vectors for $B$ tokens.
2. **Logical-to-Physical Mapping:** A centralized **KV cache manager** maintains a block table that maps logical blocks (the sequence order) to physical blocks (actual GPU DRAM locations).
3. **On-Demand Allocation:** 
   - During the **prompt phase**, vLLM allocates only the blocks necessary to fit the input prompt.
   - During the **autoregressive generation phase**, new physical blocks are allocated only when the current last logical block is full.
4. **Memory Sharing (Copy-on-Write):** 
   - For parallel sampling or beam search, multiple sequences share the same physical blocks for their common prefix.
   - A **reference count** is maintained for each physical block.
   - When a sequence needs to modify a shared block, vLLM employs a **copy-on-write (CoW)** mechanism: it allocates a new physical block, copies the data, and decrements the reference count of the original block.
5. **Preemption and Recovery:** If GPU memory is exhausted, vLLM uses a first-come-first-serve (FCFS) policy to preempt the latest requests. Evicted blocks are recovered via:
   - **Swapping:** Moving blocks to CPU RAM.
   - **Recomputation:** Regenerating the KV cache from the prompt and generated tokens.

### Key Formulas
The standard self-attention output $o_i$ is computed as:

$$
a_{ij} = \frac{\exp(q_i^T k_j / \sqrt{d})}{\sum_{t=1}^i \exp(q_i^T k_t / \sqrt{d})}, \ o_i = \sum_{j=1}^i a_{ij} v_j
$$

PagedAttention transforms this into a block-wise computation:

$$
A_{ij} = \frac{\exp(q_i^T K_j / \sqrt{d})}{\sum_{t=1}^{\lceil i/B \rceil} \exp(q_i^T K_t 1 / \sqrt{d})}, \quad o_i = \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^T
$$

where $K_j$ and $V_j$ are the key and value blocks, and $A_{ij}$ is the row vector of attention scores for the $j$-th block.

## Key Quantitative Results
*   **Throughput:** vLLM improves throughput by **2-4x** compared to state-of-the-art systems like FasterTransformer and Orca.
*   **Memory Savings:** 
    *   **Parallel Sampling:** 6.1% to 30.5% memory reduction.
    *   **Beam Search:** 37.6% to 66.3% memory reduction.
*   **Shared Prefixes:** vLLM achieved **1.67x** higher throughput for one-shot prefixes and **3.58x** for few-shot prefixes compared to Orca (Oracle).
*   **Request Capacity:** For OPT-13B, vLLM processed **2.2x** more concurrent requests than Orca (Oracle) and **4.3x** more than Orca (Max).

## Stated Limitations
*   **Kernel Overhead:** The indirection of the block table and handling variable sequence lengths result in **20-26% higher attention kernel latency** compared to highly optimized contiguous implementations like FasterTransformer.
*   **Applicability:** The paging approach is specifically beneficial for workloads with dynamic memory allocation and memory-bound performance (like LLM serving). It may degrade performance in compute-bound tasks or DNN training where tensor shapes are static and memory can be optimized a priori.
