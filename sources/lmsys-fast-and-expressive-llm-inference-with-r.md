---
id: lmsys:fast-and-expressive-llm-inference-with-r
type: web
title: Fast and Expressive LLM Inference with RadixAttention and SGLang (LMSYS Blog)
url: https://www.lmsys.org/blog/2024-01-17-sglang/
retrieved: '2026-07-12'
maturity: comprehensive
topic: rollout-generation-infra
---

# SGLang: Fast and Expressive LLM Inference with RadixAttention

SGLang is a structured generation language and runtime system designed to optimize the execution of complex Large Language Model (LLM) applications. It co-designs a frontend domain-specific language (DSL) with a specialized backend runtime to improve the speed, controllability, and programming efficiency of LLM-based workflows.

### Core Problem
Complex LLM applications—such as agents, reasoning chains, and multi-turn chats—often require multiple chained generation calls and advanced prompting techniques. A primary inefficiency in current systems is the lack of automatic **KV (Key-Value) cache reuse**. When different prompts share the same prefix (e.g., few-shot examples, system prompts, or chat history), existing systems often perform redundant computations and memory allocations. While some manual configurations exist, no system automatically accommodates the diverse reuse patterns found in complex LLM workloads.

### Method: Backend and Frontend Co-Design

#### Backend: RadixAttention
To automate KV cache reuse, SGLang introduces **RadixAttention**, which manages the KV cache using a radix tree (a space-efficient version of a prefix tree).

1.  **Data Structure**: The system maintains a radix tree where edges are labeled with token sequences (keys) and nodes map to their corresponding KV cache tensors (values).
2.  **Memory Management**: KV cache tensors are stored on the GPU using a paged layout, where each page corresponds to one token. The tree structure itself is stored on the CPU to minimize overhead.
3.  **Workflow**:
    *   The frontend sends full prompts to the runtime.
    *   The runtime performs automatic prefix matching against the radix tree.
    *   If a prefix is found, the corresponding KV cache is reused; otherwise, new tensors are computed and inserted into the tree.
4.  **Eviction Policy**: Due to limited GPU memory, SGLang implements a **Least Recently Used (LRU)** eviction policy that recursively evicts leaf nodes.
5.  **Compatibility**: RadixAttention is compatible with paged attention and continuous batching and can be extended to handle image tokens for multi-modal models.

#### Frontend: SGLang DSL
SGLang provides a Python-embedded DSL to control the generation process. It supports two execution modes: **interpreter mode** (eager execution) and **compiler mode** (tracing the program as a dataflow graph for optimizations like instruction selection and auto-tuning).

Key primitives include:
*   `gen`: Invokes a non-blocking LLM generation and stores the result in a variable.
*   `fork`: Creates multiple parallel copies of a prompt to enable branching.
*   `[variable_name]`: Retrieves the result of a previous generation.
*   `choices`: Imposes specific constraints on the LLM output.
*   `run`: Executes the SGLang function with provided arguments.

### Quantitative Results
SGLang was evaluated using Llama-7B (on a single NVIDIA A10G GPU) and Mixtral-8x7B (on 8 NVIDIA A10G GPUs with tensor parallelism) in FP16 precision. Benchmarks included MMLU, HellaSwag, ReAct Agent, Tree-of-Thought, JSON decoding, synthetic chat (short and long), DSPy RAG, and LLaVA Bench.

*   **Throughput**: SGLang achieved up to **5 times higher throughput** compared to baseline systems, including vLLM, Guidance, and Hugging Face TGI.
*   **Latency**: The system demonstrated significant improvements in first-token latency due to prefix cache hits.
*   **Overhead**: Ablation studies indicated no noticeable overhead when RadixAttention is enabled even in the absence of cache hits.

### Limitations
The authors note that current storage and eviction strategies are basic, suggesting that future iterations of the system could benefit from the development of more advanced multi-layer storage strategies and more sophisticated eviction policies.
