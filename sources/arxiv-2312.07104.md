---
id: arxiv:2312.07104
type: paper
title: 'SGLang: Efficient Execution of Structured Language Model Programs'
url: https://arxiv.org/abs/2312.07104
retrieved: '2026-07-12'
maturity: comprehensive
topic: rollout-generation-infra
---

# SGLang: Efficient Execution of Structured Language Model Programs

SGLang is a system designed to optimize the programming and execution of "Language Model (LM) Programs"—complex workflows that require multiple LLM calls, control flow, and structured inputs/outputs.

### Core Problem
The authors identify two primary inefficiencies in current LLM interaction systems:
1.  **Programming Complexity:** Developing multi-call workflows is tedious, requiring extensive string manipulation, brittle output parsing, and manual implementation of parallelism.
2.  **Execution Inefficiency:** Existing inference engines lack workload-specific knowledge, leading to redundant computation and memory usage. Specifically, they fail to reuse the Key-Value (KV) cache across calls sharing common prefixes and perform constrained decoding (e.g., JSON mode) inefficiently by decoding only one token at a time.

### Method and Implementation
SGLang consists of a frontend domain-specific language (DSL) embedded in Python and a co-designed runtime.

#### 1. Frontend Programming Model
SGLang provides high-level primitives to manage prompt states and execution:
*   **Generation Primitives:** `extend`, `gen`, and `select`.
*   **Parallelism Control:** `fork` (creates parallel copies of the prompt) and `join`.
*   **Execution Modes:** An **interpreter** treats prompts as asynchronous streams for non-blocking execution, while a **compiler** represents programs as computational graphs for static optimization.

#### 2. Runtime Optimizations
The SGLang Runtime (SRT) implements three key acceleration techniques:

*   **RadixAttention:** Instead of discarding the KV cache after a request, SGLang stores it in a radix tree. This allows the system to automatically match and reuse prefixes across different requests. It employs an LRU (Least Recently Used) eviction policy.
*   **Cache-Aware Scheduling:** To maximize the cache hit rate, the scheduler prioritizes requests with the longest matched prefixes (longest-shared-prefix-first), which the authors prove is equivalent to a depth-first search (DFS) order on the radix tree.
*   **Compressed Finite State Machine (FSM):** For constrained decoding (via regular expressions), SGLang converts the regex into an FSM. It compresses adjacent "singular-transition edges" (edges with only one successor and one acceptable character) into single edges. This allows the runtime to decode multiple tokens in a single forward pass.
*   **API Speculative Execution:** For black-box APIs (e.g., GPT-4), the system ignores stop conditions to generate extra tokens speculatively. These are then matched and reused for subsequent primitives, reducing total API calls and input token costs.

### Key Formulas
The system evaluates efficiency using the **cache hit rate**, defined as:

$$
\text{Cache Hit Rate} = \frac{\sum_{r\in R}\text{number of cached prefill tokens in } r}{\sum_{r\in R}\text{number of prefill tokens in } r}
$$

### Quantitative Results
SGLang was evaluated against baselines including vLLM, Guidance, and LMQL using Llama-2, Mixtral, and LLaVA models.

*   **General Performance:** SGLang achieves up to $6.4\times$ higher throughput and up to $3.7\times$ lower latency compared to state-of-the-art systems.
*   **Multi-modal Throughput:** 
    *   **LLaVA-v1.5-7B (image):** $1.15 \text{ image/s}$ (vs. $0.18 \text{ image/s}$ in the original implementation).
    *   **LLaVA-NeXT-34B (video):** $0.10 \text{ frame/s}$ (vs. $0.02 \text{ frame/s}$ in the original implementation).
*   **API Efficiency:** API speculative execution reduced input token costs by approximately threefold for a Wikipedia field extraction task.
*   **Overhead:** The management of RadixAttention data structures introduced negligible overhead ($<0.3\%$).
*   **Cache Effectiveness:** Achieved cache hit rates ranged from $50\%$ to $99\%$, reaching $96\%$ of the theoretical optimal hit rate on average.

### Stated Limitations
*   **Scheduling:** The greedy cache-aware scheduling algorithm may lead to request starvation.
*   **Decoding Accuracy:** The compressed FSM can suffer from "distorted probability," where the mapping between strings and tokens may misrepresent the intended probability distribution of choices.
*   **Compiler Reliability:** Using GPT-4 for graph-based code movement in compiler mode can be "too aggressive," occasionally altering the original semantics of the prompts.
*   **Scope:** The authors note the need for future support for additional output modalities, fuzzy semantic matching in RadixAttention, and multi-level memory hierarchy (DRAM/Disk) support.
