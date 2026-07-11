---
title: Rollout generation infrastructure
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2312.07104
- arxiv:2309.06180
- lmsys:fast-and-expressive-llm-inference-with-r
- docs:vllm-documentation-pagedattention-design
- blog:sglang-vs-vllm-the-new-throughput-king-g
- runpod:when-to-choose-sglang-over-vllm-multi-tu
- inclusion-ai:the-community-stories-of-vllm-and-sglang
open_questions:
- How does RadixAttention's partial-prefix matching perform on **PPO/GRPO rollouts**
  where system prompts and few-shots are identical but completions diverge — does
  it outperform vLLM's APC which requires exact matches?
- What is the **KV cache memory pressure** when running **thousands of concurrent
  rollout trajectories** with long contexts (32k–128k) on multi-GPU tensor-parallel
  setups — does RadixAttention's LRU eviction cause thrashing vs vLLM's block-level
  paging?
- Can **vLLM's V1 scheduler + chunked prefill** close the multi-turn gap with SGLang,
  or is radix-tree prefix matching fundamentally superior for dynamic conversation
  histories?
- How do **speculative decoding draft models** interact with **RadixAttention/Compressed
  FSM** — does speculative verification benefit from prefix cache hits on the draft
  model's KV cache?
---

Rollout generation infrastructure has become the critical bottleneck for scaling RLHF and inference-time compute, with vLLM and SGLang representing the two dominant open-source serving engines that implement fundamentally different KV cache reuse strategies. Their architectural choices — PagedAttention's block-level paging versus RadixAttention's radix-tree prefix matching — create divergent performance profiles across single-round batch inference, multi-turn conversations, and structured generation workloads.

## Architectural Foundations: PagedAttention vs RadixAttention

### vLLM: PagedAttention and Block-Level Memory Management

vLLM addresses the memory-bound nature of LLM serving by applying OS-style virtual memory concepts to the KV cache [source:arxiv:2309.06180]. The core innovation is **PagedAttention**, which partitions the KV cache of each sequence into fixed-size blocks (default $B=16$ tokens per block) and maintains a logical-to-physical block table managed by a centralized KV cache manager [source:arxiv:2309.06180]. This eliminates the three major sources of memory waste in contiguous allocation: internal fragmentation (pre-allocating for max sequence length), reserved slots for future tokens, and external fragmentation from the allocator [source:arxiv:2309.06180]. Profiling showed only **20.4%–38.2%** of KV cache memory stored actual token states in pre-vLLM systems [source:arxiv:2309.06180].

The attention computation is reformulated as a block-wise operation:

$$
A_{ij} = \frac{\exp(q_i^T K_j / \sqrt{d})}{\sum_{t=1}^{\lceil i/B \rceil} \exp(q_i^T K_t \mathbf{1} / \sqrt{d})}, \quad o_i = \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^T
$$

where $K_j, V_j$ are the $j$-th key/value blocks and $A_{ij}$ is the row vector of attention scores for that block [source:arxiv:2309.06180].

**Memory sharing via copy-on-write (CoW)** enables efficient parallel sampling and beam search: multiple sequences share physical blocks for common prefixes with reference counting; modification triggers allocation of a new block, data copy, and reference decrement [source:arxiv:2309.06180]. This yielded **37.6%–66.3% memory reduction for beam search** and **1.67×–3.58× higher throughput for shared prefixes** vs Orca [source:arxiv:2309.06180].

The kernel implementation processes one head for one sequence per thread block, fetching queries into shared memory, keys into registers (single access), performing QK dot products with cross-thread-group reduction, then softmax via warp-level max/sum reductions, and finally value fetching with per-thread $V\_VEC\_SIZE$ elements accumulated in registers [source:docs:vllm-documentation-pagedattention-design]. The kernel uses 16-byte memory transactions (`VEC_SIZE`/`V_VEC_SIZE` configured for FP16) and maps warps to key blocks, thread blocks to full context per query/head, grid to `(num_heads, num_seqs, max_num_partitions)` [source:docs:vllm-documentation-pagedattention-design]. **Caveat**: this documentation is historical and does not reflect the current vLLM codebase [source:docs:vllm-documentation-pagedattention-design].

### SGLang: RadixAttention and Prefix-Tree KV Reuse

SGLang's **RadixAttention** manages the KV cache as a radix tree (compressed prefix tree) where edges are labeled with token sequences and nodes map to KV cache tensors [source:arxiv:2312.07104][source:lmsys:fast-and-expressive-llm-inference-with-r]. The tree structure resides on CPU to minimize GPU overhead; KV tensors are stored on GPU in a paged layout (one page per token) [source:lmsys:fast-and-expressive-llm-inference-with-r]. On each request, the runtime performs automatic prefix matching against the tree: matching prefixes reuse cached KV tensors; non-matching suffixes are computed and inserted [source:arxiv:2312.07104][source:lmsys:fast-and-expressive-llm-inference-with-r]. An **LRU eviction policy** recursively removes leaf nodes when GPU memory is exhausted [source:arxiv:2312.07104][source:lmsys:fast-and-expressive-llm-inference-with-r].

**Cache-aware scheduling** maximizes hit rates by prioritizing requests with the longest matched prefix — proven equivalent to depth-first search (DFS) order on the radix tree [source:arxiv:2312.07104]. The frontend `fork` primitive sends prompt prefixes as "hints" to ensure correct radix tree insertion [source:blog:sglang-vs-vllm-the-new-throughput-king-g]. RadixAttention extends to multi-modal tokens (images/video) [source:lmsys:fast-and-expressive-llm-inference-with-r].

**Compressed Finite State Machines** accelerate constrained decoding (regex/JSON): adjacent "singular-transition edges" (only one valid next character) are merged into single compressed edges, enabling **jump-forward decoding of multiple tokens per forward pass** instead of token-by-token masking [source:arxiv:2312.07104][source:blog:sglang-vs-vllm-the-new-throughput-king-g]. This yielded **1.6× throughput increase on JSON decoding** [source:blog:sglang-vs-vllm-the-new-throughput-king-g].

**API Speculative Execution** for black-box models ignores stop conditions on the first call, generates extra tokens speculatively, then matches/reuses them for subsequent `gen` primitives — reducing input token costs **~3×** for field extraction [source:arxiv:2312.07104][source:blog:sglang-vs-vllm-the-new-throughput-king-g].

## Quantitative Comparison: Throughput, Latency, and Workload Sensitivity

| Metric | vLLM (PagedAttention) | SGLang (RadixAttention) | Source |
|--------|----------------------|------------------------|--------|
| **Throughput vs baselines** | 2–4× vs FasterTransformer/Orca | Up to 5–6.4× vs vLLM/Guidance/TGI | [source:arxiv:2309.06180][source:arxiv:2312.07104][source:lmsys:fast-and-expressive-llm-inference-with-r][source:blog:sglang-vs-vllm-the-new-throughput-king-g] |
| **Multi-turn cached throughput (7k ctx, DeepSeek-R1-Distill-Llama-70B, 2×H100)** | 32.8 tok/s (+14% vs fresh) | 35.0 tok/s (+20% vs fresh) | [source:runpod:when-to-choose-sglang-over-vllm-multi-tu] |
| **One-shot throughput (same setup)** | 60.0 tok/s | 52.7 tok/s | [source:runpod:when-to-choose-sglang-over-vllm-multi-tu] |
| **Structured decoding (JSON)** | Baseline | 1.6× via Compressed FSM | [source:blog:sglang-vs-vllm-the-new-throughput-king-g] |
| **Memory reduction (parallel sampling)** | 6.1%–30.5% | Not directly reported | [source:arxiv:2309.06180] |
| **Memory reduction (beam search)** | 37.6%–66.3% | Not directly reported | [source:arxiv:2309.06180] |
| **Cache hit rate (RadixAttention)** | N/A | 50%–99%, avg 96% of theoretical optimal | [source:arxiv:2312.07104] |
| **RadixAttention overhead** | N/A | <0.3% management overhead | [source:arxiv:2312.07104] |
| **vLLM kernel overhead** | 20–26% higher attention latency vs contiguous | N/A | [source:arxiv:2309.06180] |

**Critical disagreement on workload suitability**: The RunPod benchmark shows **vLLM 1.1× faster on one-shot** (60.0 vs 52.7 tok/s) while **SGLang leads on multi-turn cached** (35.0 vs 32.8 tok/s) [source:runpod:when-to-choose-sglang-over-vllm-multi-tu]. The GoPenAI blog reports **SGLang 10–20% faster on multi-turn** vs vLLM [source:blog:sglang-vs-vllm-the-new-throughput-king-g]. SGLang's RadixAttention automatically handles **partial prefix overlaps** in dynamic conversations; vLLM's Automatic Prefix Caching (APC) requires **exact token-sequence matches** and often needs manual configuration [source:runpod:when-to-choose-sglang-over-vllm-multi-tu][source:blog:sglang-vs-vllm-the-new-throughput-king-g]. Alibaba Cloud benchmarks on Qwen models "generally favored SGLang in single-GPU and dual-GPU setups" but outcomes vary by hardware/model/config [source:inclusion-ai:the-community-stories-of-vllm-and-sglang].

## Converged Advanced Features (2024–2025)

Both engines have converged on a shared optimization stack [source:inclusion-ai:the-community-stories-of-vllm-and-sglang]:
- **Chunked Prefill**: Splits long prefill into chunks to interleave with decode, reducing tail latency.
- **Speculative Decoding**: Draft model proposes tokens, target model verifies in parallel.
- **Disaggregated Serving**: Separates prefill and decode onto different GPU pools.
- **CUDA Graphs**: Captures decode kernels to eliminate CPU launch overhead.
- **Operator Libraries**: Integration with FlashInfer, FlashAttention-3, DeepGEMM for fused kernels.

vLLM's **V1 refactor (early 2025)** addressed CPU scheduling overhead that exceeded **50% of total inference time** in some scenarios [source:inclusion-ai:the-community-stories-of-vllm-and-sglang]. vLLM v0.6.0 reduced latency **~5×** and improved performance **~2.7×** via CPU-scheduling optimizations [source:inclusion-ai:the-community-stories-of-vllm-and-sglang]. vLLM claims **5× traffic capacity and 30× throughput vs HF Transformers backend** [source:inclusion-ai:the-community-stories-of-vllm-and-sglang].

## Current Status and Trajectory

**vLLM** is the **de facto default for high-throughput batch inference** and production deployments requiring stability, broad hardware support (NVIDIA, AMD, Intel, TPU), and a large ecosystem (10k+ contributors, ~2k PR submitters, 12h–3d issue response) [source:inclusion-ai:the-community-stories-of-vllm-and-sglang]. Its PagedAttention + CoW design is proven at scale; the V1 refactor resolves the major CPU bottleneck. **Rising** — active development, expanding hardware backends, adoption in major LLM serving platforms.

**SGLang** is **rising rapidly for structured generation, agentic workflows, and multi-turn chat** where dynamic prefix reuse and constrained decoding matter. Its RadixAttention + Compressed FSM + DSL co-design targets workloads vLLM's APC handles less naturally. Community is smaller (~half vLLM's contributors, 3–5d issue response) but with **30% contributor overlap** (194 dual contributors) [source:inclusion-ai:the-community-stories-of-vllm-and-sglang]. **Not widely reported** as a general-purpose batch inference replacement; the one-shot deficit (RunPod) and DSL learning curve limit default adoption. Trajectory: **specialized dominance** in reasoning/agent rollouts, potential convergence if vLLM adopts radix-tree prefix caching natively.

**Key unresolved tension**: Whether RadixAttention's automatic partial-prefix matching generalizes better than APC's exact-match simplicity for *RL rollout workloads* — where prompts share system prompts, few-shot examples, and evolving conversation histories but rarely have identical prefixes across trajectories. No public benchmark directly compares them on **PPO/GRPO rollout generation** with repeated system prompts and varying completions.

## Key Takeaways

- **PagedAttention (vLLM)** solves memory fragmentation via fixed-size blocks + CoW, excelling at batch inference with exact prefix sharing (templated prompts, beam search). Kernel overhead 20–26% vs contiguous attention [source:arxiv:2309.06180].
- **RadixAttention (SGLang)** enables automatic partial-prefix reuse via radix tree + LRU + DFS scheduling, excelling at dynamic multi-turn and structured generation. Management overhead <0.3% [source:arxiv:2312.07104].
- **Compressed FSM** (SGLang-only) enables multi-token jump-forward decoding for regex/JSON constraints — **1.6× JSON throughput** [source:blog:sglang-vs-vllm-the-new-throughput-king-g].
- **Workload determines winner**: vLLM for one-shot/batch (60 vs 52.7 tok/s on 70B); SGLang for multi-turn cached (35 vs 32.8 tok/s) and structured output [source:runpod:when-to-choose-sglang-over-vllm-multi-tu][source:blog:sglang-vs-vllm-the-new-throughput-king-g].
- **Both converged** on chunked prefill, speculative decoding, disaggregated serving, CUDA graphs, FlashInfer/FlashAttention/DeepGEMM [source:inclusion-ai:the-community-stories-of-vllm-and-sglang].
- **vLLM V1 refactor** resolved >50% CPU scheduling overhead; v0.6.0: 5× latency reduction, 2.7× perf [source:inclusion-ai:the-community-stories-of-vllm-and-sglang].
- **No direct benchmark** on RL rollout generation (repeated system prompts + varying trajectories) — the critical workload for PPO/GRPO/RLVR.

## Related Topics

- [Distributed RL training for LLMs](distributed-rl-training.md) — rollout generation is the data-collection frontend for distributed policy optimization.
- [Async and off-policy RL](async-and-off-policy-rl.md) — rollout infrastructure must support asynchronous generation and off-policy correction.
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md) — rollout engines implement the inference-time compute scaling (best-of-N, beam search, MCTS) that interacts with RL training.
- [RL for reasoning models](rl-for-reasoning.md) — reasoning rollouts require long-context, multi-turn, and structured generation where SGLang's RadixAttention/FSM shine.
- [RL for math and code](rl-for-math-and-code.md) — verifiable-reward rollouts need high-throughput, constrained decoding (JSON/program syntax).
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md) — agent rollouts are multi-turn, tool-call heavy, and benefit from prefix caching across trajectories.
- [Rejection sampling and Best-of-N](rejection-sampling-and-bon.md) — rollout engines must efficiently generate many candidates per prompt (parallel sampling, prefix reuse).
- [Verifiable rewards (RLVR)](verifiable-rewards.md) — rollout infrastructure for code/math execution environments with deterministic rewards.

## References
- [source:arxiv:2312.07104] [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [source:arxiv:2309.06180] [Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [source:lmsys:fast-and-expressive-llm-inference-with-r] [Fast and Expressive LLM Inference with RadixAttention and SGLang (LMSYS Blog)](https://www.lmsys.org/blog/2024-01-17-sglang/)
- [source:docs:vllm-documentation-pagedattention-design] [vLLM Documentation: PagedAttention Design](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [source:blog:sglang-vs-vllm-the-new-throughput-king-g] [SGLang vs. vLLM: The New Throughput King? (GoPenAI)](https://blog.gopenai.com/sglang-vs-vllm-the-new-throughput-king-7daec596f7fa)
- [source:runpod:when-to-choose-sglang-over-vllm-multi-tu] [When to Choose SGLang Over vLLM: Multi-Turn Conversations (RunPod)](https://www.runpod.io/blog/sglang-vs-vllm-kv-cache)
- [source:inclusion-ai:the-community-stories-of-vllm-and-sglang] [The Community Stories of vLLM and SGLang (Inclusion AI)](https://www.inclusion-ai.org/blog/llm-landscape-vllm-sgl/)
