---
title: Rollout generation infrastructure
maturity: comprehensive
updated: '2026-07-11'
sources:
- lmsys:fast-and-expressive-llm-inference-with-r
- runpod:when-to-choose-sglang-over-vllm-multi-tu
- arxiv:2309.06180
- blog:sglang-vs-vllm-the-new-throughput-king-g
- arxiv:2312.07104
- docs:vllm-documentation-pagedattention-design
- inclusion-ai:the-community-stories-of-vllm-and-sglang
- arxiv:2605.24259
- arxiv:2606.01387
- arxiv:2605.08840
- arxiv:2605.09649
- arxiv:2512.06727
- arxiv:2607.06519
- arxiv:2504.03648
open_questions:
- Will vLLM adopt radix-tree prefix caching natively, or will its APC evolve to handle
  partial overlaps via block-level fuzzy matching?
- Can Resident KV Claim contracts be standardized across runtimes (vLLM, SGLang, TensorRT-LLM,
  Triton) to enable portable rollout generation with guaranteed KV reuse semantics?
- Does DBTrimKV's attention-dilution mitigation generalize to RL rollout workloads
  where distractors are not static but generated adversarially by the policy?
- How do advanced eviction methods (ReST-KV, FreqDepthKV, KV-CAR) interact with speculative
  decoding and chunked prefill in production rollout pipelines?
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

## KV Cache Conformance Contracts and Resident Claims

### The Resident KV Claim Contract

A new line of work formalizes the tension between **resident reusable KV** (cached prefixes for future requests) and **active live KV** (memory required for in-flight requests) as a conformance contract problem [source:arxiv:2605.24259]. Existing mechanisms—priority-based eviction or "write no-admit" policies—lack a formal contract to handle scenarios where both cannot fit in the usable memory pool, resulting in "unreported resident loss" where valuable cached state is destroyed to serve a current request without explicit arbitration or telemetry.

The **Resident KV Claim** contract defines a schema with required fields: `claim_id`, `owner_scope`, `cache_identity` (model/tokenizer/format), `object_id`, `materialization_predicate` (the condition for the cache to be "useful"), `footprint_blocks`, `protection_mode`, and optional `duration_steps`. Protection modes include:
- `soft_priority`: Influences eviction order
- `hard_protected`: Cannot be broken without explicit refusal, demotion, or expiry
- `demotable`/`offloadable`/`expiring`: Specific lifecycle transitions before loss

The lifecycle follows: $\text{Submitted} \rightarrow \text{Accepted} \rightarrow \text{Materialized} \rightarrow (\text{Demoted/Expired/Refused/Harmed})$. When conflicts occur, the **Active/Resident Arbiter** must choose an explicit action: **Resident Victim Exclusion** (protect residents; refuse/defer active), **Write No-Admit** (serve active but prevent its KV from becoming resident), **Relaxation** (demote/expire resident before eviction), or other actions (offloading, bounding active-live KV, routing to another worker).

The feasibility boundary is formalized as:

$$
\text{protected\_resident\_kv} + \text{active\_live\_kv} \le \text{usable\_kv}
$$

If violated, the runtime must trigger explicit arbitration (e.g., `active_refused`) rather than ordinary cache eviction.

A vLLM prototype implementation adds resident claim metadata, hard-protected victim exclusion, and scheduler-visible active refusal. In a "60/70/80" litmus test (60 protected resident blocks, 70 active live blocks, 80 usable blocks):
- **Native/Write No-Admit**: Served active request but evicted 50 resident blocks, leaving only 10
- **Hard Resident Exclusion**: Preserved all 60 resident blocks and converted failure into a **scheduler-visible active refusal**

TTFT measurements on SmolLM2 and Qwen2.5-Coder-7B showed drops from **0.345s** (first request) to **0.042s** and **0.035s** (repeats), demonstrating serving-visible value of preserved residents. A live scheduler pressure test with 40 resident blocks and active request requiring 46 blocks (total 86 > 68 usable) resulted in a `protected_resident_capacity_refused` stop reason with a recorded **19-block shortfall** [source:arxiv:2605.24259].

### Fail-Closed Lowering onto Serving Runtimes

The **Fail-Closed Lowering** framework determines when a concrete runtime primitive or adapter conformantly satisfies the semantic obligations of a ResidentClaim contract [source:arxiv:2606.01387]. ResidentClaim modes (`best_effort`, `soft_priority`, `hard_protected`, `offloadable`, `routed_reuse`) are mapped to "obligation bundles" $O[m]$ including `claim_identity`, `explicit_acceptance`, `materialization_predicate`, and `ordered_lifecycle_events`.

The lowering relation requires every obligation $o \in O[m]$ to be supported by anchored evidence at an allowed adapter depth:

$$
\text{Lower}(d, a, E, m) \iff \forall o \in O[m], \exists e \in (d.\text{native} \cup d.\text{adapters}[\le a] \cup E) \text{ s.t. } \text{supports}(e, o) \land \text{anchored}(e) \land \text{depth\_allowed}(a, o)
$$

The checker assigns labels: `native_sound`, `sound_with_adapter`, `approximate`, `rejected`, `unknown`. **No public runtime currently achieves `native_sound` conformance.**

- **TensorRT-LLM**: `sound_with_adapter` for `soft_priority` under controlled pressure (5/5 runs lower-priority prompts lost first; 13/13 claims joinable before pressure; 16/16 materialization changes reconstructable)
- **SGLang/HiCache**: `sound_with_adapter` for `best_effort` via telemetry joins (warm shared-prefix request: `cached_tokens=35` vs `0` for cold/wrong-prefix; 100 `BlockStored` events)
- **Patched vLLM Witness**: Local `backend_patch` demonstrated `offloadable` semantics implementability (131/131 subprocesses valid event sequences; 30/30 observation passes; 0/41 false-positive passes; 3/3 repetitions failure attributed only to target claim; median resident wall time $0.187\text{s}$, failure-to-outcome latency $151,544\text{ns}$) [source:arxiv:2606.01387]

**Limitation**: Refusal occurs at scheduler-boundary (invalid-KV-load handling) rather than pre-admission scheduling refusal; TTFT unavailable due to vLLM `OffloadingConnector` metrics serialization issue.

## Advanced KV Cache Eviction and Compression Methods

### ReST-KV: Layer-wise Output Reconstruction with Spatial-Temporal Smoothing

ReST-KV reformulates KV cache eviction as an optimization problem preserving attention output at each layer under fixed memory budget $B$ [source:arxiv:2605.08840]. It addresses two overlooked factors: **attention redistribution** (removing a token alters attention landscape) and **spatial-temporal dynamics** (token importance fluctuates over time and shifts across positions).

**Method**:
1. **Layer-wise Output Reconstruction**: Measures reconstruction error ($\ell_2$ distance) caused by removing an individual KV pair, capturing redistribution effects
2. **Temporal Smoothing**: Exponential Moving Average (EMA) over recent query window $S_w$
3. **Adaptive Window-based Spatial Smoothing (AWS)**: Splits observation window into front/rear halves to calculate shift in average index of top-B important pairs, adaptively adjusting sliding window size/shift
4. **Eviction**: Selects top-B KV pairs by final smoothed indicators

**Key formulas**:
- Importance indicator: $\mathbf{I}_{t}[n]=\frac{\mathbf{A}_{t}[n]}{1-\mathbf{A}_{t}[n]}\left\|\text{MHA}\left(\mathbf{x}_{t},\langle\mathbf{K}_{T},\mathbf{V}_{T}\rangle\right)-\mathbf{v}_{n}\mathbf{W}_{O}\right\|_{2}$
- Temporal EMA: $\text{EMA}(\mathbf{I}_{t_{1}:t_{2}}[n]) = \alpha I_{t_2}[n] + (1 - \alpha) \text{EMA}(\mathbf{I}_{t_1:t_2-1}[n])$
- Adaptive spatial window: $W_{s}=2\cdot\left\lfloor\frac{|\mathbf{D}_{\text{rear}}-\mathbf{D}_{\text{front}}|}{\beta}\right\rfloor+1$
- Final smoothed indicator: $\mathcal{I}_{t}[n]=\frac{\sum_{k=-\lfloor W_{s}/2\rfloor+\gamma_{\text{shift}}}^{\lfloor W_{s}/2\rfloor+\gamma_{\text{shift}}}\hat{\mathbf{I}}_{t}[k]}{W_{s}}$

**Results**: **2.58% improvement on LongBench**, **15.2% on RULER**; **98% Needle-in-Haystack performance** (Mistral-7B, $B_{total}=1024L$); **100% accuracy** on Llama-3.1-8B storing only 1/32 tokens; **10.61× decoding latency reduction**, **36% peak memory reduction**, **2.37× TTFT speedup** with FlexPrefill (128k context); **38.8% InfiniteBench** vs 36.8% SnapKV; **~2% prefill overhead** over SnapKV. **Limitation**: Struggles on Common Word Extraction (CWE) and MK-NIAH-3 tasks [source:arxiv:2605.08840].

### DBTrimKV: Dynamic Budget Trim with Global Calibration

DBTrimKV treats eviction as a mechanism for **improving reasoning** rather than mere compression, identifying **attention dilution** as a primary failure cause in long contexts: as distractors increase, softmax spreads mass across them, diluting attention on useful evidence [source:arxiv:2512.06727].

**Method**:
1. **Retention Gate**: Lightweight per-token/layer/head gate $g_{\ell,h}$ predicts retention coefficient $\beta_{\ell,h,t} \in [0, 1]$ from token embedding
2. **Global Calibration via Weight Tying**: Final scoring projection $(\mathbf{w}_g, b_g)$ shared across all gates for cross-layer/head comparability
3. **Geometric Retention**: $r_{t,i} = \beta_i^{t-i}$ as query-agnostic proxy for "future utility"
4. **Global Budget Allocation**: Single global KV budget $M_{global}$ (not per-layer/head)
5. **Global Ranking and Eviction**: At each decode step, tokens ranked by predicted future utility $\tilde{G}$; top $M_{global}$ retained across all layers, heads, modalities
6. **Training**: Gates trained to match full-cache teacher while minimizing capacity loss $\mathcal{L}_{cap}$

**Key formulas**:
- Attention dilution: $\delta_{t}:=1-\sum_{i\in U_{t}}\alpha_{t,i}$
- Retention-gated attention: $\tilde{\alpha}_{t,i}=\frac{\beta_{i}^{t-i}e^{z_{t,i}}}{\sum_{j\in C_{t}}\beta_{j}^{t-j}e^{z_{t,j}}}$
- Global future utility: $\widetilde{G}_{\ell,h,i}(t)=\sum_{s=t+1}^{T}\beta_{\ell,h,i}^{s-i}=\beta_{\ell,h,i}^{t+1-i}\,\frac{1-\beta_{\ell,h,i}^{T-t}}{1-\beta_{\ell,h,i}}$
- Capacity loss: $\mathcal{L}_{\mathrm{cap}}=\sum_{t=1}^{T}\mathrm{max}\left(0,\,\sum_{\ell,h}\sum_{i=1}^{t}\beta_{\ell,h,i}^{t-i}-M_{\mathrm{global}}\right)$

**Results**: **Exceeds vanilla full-cache by up to 3.75%** on long-form generation; **9.20% average accuracy improvement** on LongBench-v2 (Phi-3-mini-128K); **VLM reasoning**: MathVisionmini budget 256 → **69.91%** vs 65.67% TrimKV, 50.89% SnapKV; **MMDU budget 128**: overall 3.43 vs vanilla 3.57, but **Visual Perception 3.54 vs 3.40**, **Logical Coherence 4.11 vs 3.90**; efficient with PagedAttention. **Limitation**: Small computational overhead vs non-global TrimKV due to variable-length, head-specific caches [source:arxiv:2512.06727].

### FreqDepthKV: Frequency-Guided Depth Sharing

FreqDepthKV factorizes KV cache across depth dimension into shared low-frequency components and sparse high-frequency residuals [source:arxiv:2607.06519]. Uniform depth sharing erases decisive evidence localized to specific token-head-layer interactions.

**Method**:
1. **Depth-Frequency Factorization**: Adjacent layers grouped into blocks (typically $B=4$ middle, $B=2$ boundary). KV states transformed via fixed DCT basis $F_B$: $Z_{b,h}^{K} = F_B X_{b,h}^{K}$. First coefficient group = shared low-frequency; subsequent = high-frequency deviations.
2. **Online Probing and Routing**: During prefill, lightweight probe samples query positions $\mathcal{P}$. Each head assigned to `SHARED`, `RESIDUAL`, or `EXACT` mode by minimizing reconstruction-aware routing loss:

$$
\mathcal{L}_{b,h}(m) = \frac{1}{|\mathcal{P}|} \sum_{t \in \mathcal{P}} \left\| \frac{Q_{\ell,h,t} \widehat{K}_{\ell,h}^{(m)\top}}{\sqrt{d_h}} - \frac{Q_{\ell,h,t} K_{\ell,h}^\top}{\sqrt{d_h}} \right\|_2^2 + \lambda \Omega(m)
$$

   Penalty $\lambda$ adjusted to meet memory budget.
3. **Sparse Residual Selection**: For `RESIDUAL` heads, retain high-frequency coefficients only for critical tokens scored by max logit perturbation:

$$
s_{b,h,t} = \max_{q \in \mathcal{P}} \left| Q_{\ell,h,q} (K_{\ell,h,t} - \widehat{K}_{\ell,h,t}^{\text{SHARED}})^\top \right|
$$

   Top $r_{b,h}$ tokens retain residuals.
4. **Decoding**: On-the-fly reconstruction by broadcasting shared coefficients + sparse residuals for indexed tokens.

**Results** (32k prefill window): **EM 58.3, F1 63.0, ROUGE-L 32.5, pass@1 48.1** (near Full KV: EM 58.7, F1 63.4); **70.4 tok/s decoding throughput** (vs 38.2 Full KV); **TTFT 2.06s**; **Peak KV memory 6.2 GB**; **3.9× effective compression**. Outperforms MiniCache (3.6×, EM 56.6), SnapKV, KIVI. **Limitation**: Routing decision fixed at prefill, not generation-aware [source:arxiv:2607.06519].

### KV-CAR: Autoencoder Compression with KV Reuse

KV-CAR employs **layer-wise autoencoder compression** and **similarity-guided KV reuse** [source:arxiv:2512.06727].

**Autoencoder**: Per-layer encoder reduces embedding dimension $D$ to latent $d \ll D$ (two FC layers + BatchNorm + LeakyReLU); decoder mirrors encoder. KV compressed before HBM storage, expanded before attention. Optional Int8 quantization stacked on top.

**KV Reuse**: L1-norm similarity between adjacent layer KV matrices; if exceeding threshold, layer $N$ heads replaced by layer $N-1$.

**Staged training**:
1. **Independent**: Base model frozen, per-layer autoencoder trained (CE + scaled L1 reconstruction loss)
2. **Joint**: All autoencoders integrated, fine-tune encoders only (sum scaled L1 + CE)
3. **Reuse**: Similarity analysis over epoch → head selection → fine-tune (CE + scaled L1 between actual/reused KV)

**Results** (GPT-2 774M, TinyLLaMA 1.1B): **Up to 47.85% KV memory reduction** (GPT-2 WikiText); TinyLLaMA stable perplexity with 11 layers compressed (25% savings), **50% memory reduction** on Winogrande with 22 layers; GPT-2 **41.6% savings** on WikiText/PIQA/Winogrande with 10 layers compressed; **5,248 token sequence length increase** at batch 64 with 75% compression (A40). **Limitation**: Accuracy-memory tradeoff varies by dataset (C4 less tolerant >4-6 layers); excessive head replacement increases perplexity significantly [source:arxiv:2512.06727].

## Cloud-Native Serving Frameworks: AIBrix

AIBrix provides a cloud-native framework co-designing system-level orchestration with inference engines (vLLM, SGLang) [source:arxiv:2504.03648]. It addresses systemic bottlenecks: traditional cloud stacks (Knative, Istio) lack GPU awareness; ML frameworks (KServe, RayServe) lack deep LLM-specific integration (stateful KV coordination, variable I/O lengths, heterogeneous GPU utilization).

**Architecture**: Split **Control Plane** (model metadata, autoscaling, LoRA adapters) and **Data Plane** (request dispatching, execution).

**Components**:
1. **High-Density LoRA Management**: Dynamic registration/loading/unloading via Kubernetes Service/EndpointSlice
2. **LLM-Aware API Gateway** (Envoy extension): Routing policies — `random`, `throughput`, `least-request`, `least-kv-cache`, `least-latency`, `prefix-cache-aware`
3. **Unified AI Runtime**: Vendor-agnostic abstraction + GPU streaming loader (bypass disk I/O)
4. **LLM-Specific Autoscaling**: Sliding window metric aggregation (KPA + APA)
5. **Distributed KV Cache Pool**: DRAM-based, cross-engine, scan-resistant eviction, async metadata updates, shared-memory data exchange for token reuse across nodes
6. **Mixed-Grain Orchestration**: Kubernetes (coarse) + Ray (fine-grained)
7. **SLO-Driven Heterogeneous Serving**: ILP-based GPU optimizer for dynamic GPU config selection
8. **Accelerator Diagnostics**: Automated fault detection/failure mockup (NVIDIA GPU, Ascend 910B NPU)

**Results**: **19.2% mean latency reduction**, **79% P99 reduction** (LLM-aware gateway); **11.5% latency reduction**, **11.4% throughput increase**, **33% oscillation reduction** vs HPA (autoscaling); **Distributed KV Cache + vLLM prefix caching** (4×A10, Bird-SQL): **~50% peak throughput increase**, **~65% avg TTFT reduction**, **~77% P99 TTFT reduction**, **~30% avg ITL reduction**, **~72% P99 ITL reduction**; **10% cost reduction** in mixed A10/L20 vs homogeneous L20 within SLO (20% latency increase bound). **Limitations**: Routing/heterogeneous serving not fully evaluated under non-ideal workloads; profiling overhead impractical for dynamic workloads; future roofline model analysis suggested [source:arxiv:2504.03648].

## Current Status and Trajectory

**vLLM** is the **de facto default for high-throughput batch inference** and production deployments requiring stability, broad hardware support (NVIDIA, AMD, Intel, TPU), and a large ecosystem (10k+ contributors, ~2k PR submitters, 12h–3d issue response) [source:inclusion-ai:the-community-stories-of-vllm-and-sglang]. Its PagedAttention + CoW design is proven at scale; the V1 refactor resolves the major CPU bottleneck. **Rising** — active development, expanding hardware backends, adoption in major LLM serving platforms.

**SGLang** is **rising rapidly for structured generation, agentic workflows, and multi-turn chat** where dynamic prefix reuse and constrained decoding matter. Its RadixAttention + Compressed FSM + DSL co-design targets workloads vLLM's APC handles less naturally. Community is smaller (~half vLLM's contributors, 3–5d issue response) but with **30% contributor overlap** (194 dual contributors) [source:inclusion-ai:the-community-stories-of-vllm-and-sglang]. **Not widely reported** as a general-purpose batch inference replacement; the one-shot deficit (RunPod) and DSL learning curve limit default adoption. Trajectory: **specialized dominance** in reasoning/agent rollouts, potential convergence if vLLM adopts radix-tree prefix caching natively.

**Emerging formalization layer**: The **Resident KV Claim** contract and **Fail-Closed Lowering** checker introduce a conformance framework for KV cache reuse guarantees across runtimes [source:arxiv:2605.24259][source:arxiv:2606.01387]. No runtime achieves `native_sound` conformance yet; TensorRT-LLM and SGLang/HiCache achieve `sound_with_adapter` for specific modes; vLLM requires a backend patch for `offloadable` semantics. This formalization may drive runtime convergence on explicit arbitration primitives (resident victim exclusion, active refusal) rather than silent eviction.

**Advanced eviction/compression convergence**: ReST-KV, DBTrimKV, FreqDepthKV, and KV-CAR represent a new generation of **inference-time KV optimization** that goes beyond simple eviction to include layer-wise reconstruction, global budget allocation, frequency-domain factorization, and autoencoder compression [source:arxiv:2605.08840][source:arxiv:2512.06727][source:arxiv:2607.06519][source:arxiv:2512.06727]. DBTrimKV's demonstration that **selective eviction can exceed full-cache performance** (9.20% LongBench-v2 improvement) by mitigating attention dilution challenges the compression-accuracy tradeoff assumption. FreqDepthKV's **3.9× compression with near-lossless accuracy** and ReST-KV's **15.2% RULER gain** suggest these techniques are approaching production readiness.

**Cloud-native orchestration**: AIBrix demonstrates that **cross-engine distributed KV cache pools** and **LLM-aware routing** can yield **50% throughput gains** and **65-77% TTFT reductions** over single-engine prefix caching [source:arxiv:2504.03648]. Its ILP-based heterogeneous GPU optimizer achieves **10% cost reduction** within SLOs. The framework's integration with both vLLM and SGLang positions it as a potential unification layer.

**Key unresolved tension**: Whether RadixAttention's automatic partial-prefix matching generalizes better than APC's exact-match simplicity for *RL rollout workloads* — where prompts share system prompts, few-shot examples, and evolving conversation histories but rarely have identical prefixes across trajectories. No public benchmark directly compares them on **PPO/GRPO rollout generation** with repeated system prompts and varying completions. The new conformance contracts and advanced eviction methods add a third dimension: **whether runtimes will adopt explicit arbitration primitives** (resident claims, hard protection, active refusal) that could fundamentally change how rollout generators manage KV cache pressure during large-scale generation.

## Key Takeaways

- **PagedAttention (vLLM)** solves memory fragmentation via fixed-size blocks + CoW, excelling at batch inference with exact prefix sharing (templated prompts, beam search). Kernel overhead 20–26% vs contiguous attention [source:arxiv:2309.06180].
- **RadixAttention (SGLang)** enables automatic partial-prefix reuse via radix tree + LRU + DFS scheduling, excelling at dynamic multi-turn and structured generation. Management overhead <0.3% [source:arxiv:2312.07104].
- **Compressed FSM** (SGLang-only) enables multi-token jump-forward decoding for regex/JSON constraints — **1.6× JSON throughput** [source:blog:sglang-vs-vllm-the-new-throughput-king-g].
- **Workload determines winner**: vLLM for one-shot/batch (60 vs 52.7 tok/s on 70B); SGLang for multi-turn cached (35 vs 32.8 tok/s) and structured output [source:runpod:when-to-choose-sglang-over-vllm-multi-tu][source:blog:sglang-vs-vllm-the-new-throughput-king-g].
- **Both converged** on chunked prefill, speculative decoding, disaggregated serving, CUDA graphs, FlashInfer/FlashAttention/DeepGEMM [source:inclusion-ai:the-community-stories-of-vllm-and-sglang].
- **vLLM V1 refactor** resolved >50% CPU scheduling overhead; v0.6.0: 5× latency reduction, 2.7× perf [source:inclusion-ai:the-community-stories-of-vllm-and-sglang].
- **Resident KV Claims** introduce a formal conformance contract for active/resident KV arbitration, converting silent resident loss into explicit scheduler-visible refusal [source:arxiv:2605.24259][source:arxiv:2606.01387].
- **Advanced eviction methods** (ReST-KV, DBTrimKV, FreqDepthKV, KV-CAR) achieve **near-lossless or accuracy-improving compression** via layer-wise reconstruction, global budget allocation, frequency-domain factorization, and autoencoder compression [source:arxiv:2605.08840][source:arxiv:2512.06727][source:arxiv:2607.06519][source:arxiv:2512.06727].
- **DBTrimKV exceeds full-cache performance** by 9.20% on LongBench-v2 by mitigating attention dilution — eviction as reasoning enhancement [source:arxiv:2512.06727].
- **AIBrix** demonstrates cross-engine distributed KV cache pools yielding **50% throughput gains** and **65-77% TTFT reductions** with LLM-aware routing and ILP-based heterogeneous GPU optimization [source:arxiv:2504.03648].
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
- [KL regularization in RLHF](kl-regularization.md) — rollout generation infrastructure must support KL-controlled sampling for policy optimization.
- [Reward modeling for LLMs](reward-modeling.md) — rollout engines often serve reward models for online evaluation during generation.
- [Process vs outcome reward models](process-vs-outcome-rewards.md) — rollout infrastructure for step-wise vs outcome reward evaluation.
- [RLHF/PPO pipeline](rlhf-ppo-pipeline.md) — rollout generation is the first stage of the RLHF/PPO pipeline.

## References
- [source:lmsys:fast-and-expressive-llm-inference-with-r] [Fast and Expressive LLM Inference with RadixAttention and SGLang (LMSYS Blog)](https://www.lmsys.org/blog/2024-01-17-sglang/)
- [source:runpod:when-to-choose-sglang-over-vllm-multi-tu] [When to Choose SGLang Over vLLM: Multi-Turn Conversations (RunPod)](https://www.runpod.io/blog/sglang-vs-vllm-kv-cache)
- [source:arxiv:2309.06180] [Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [source:blog:sglang-vs-vllm-the-new-throughput-king-g] [SGLang vs. vLLM: The New Throughput King? (GoPenAI)](https://blog.gopenai.com/sglang-vs-vllm-the-new-throughput-king-7daec596f7fa)
- [source:arxiv:2312.07104] [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [source:docs:vllm-documentation-pagedattention-design] [vLLM Documentation: PagedAttention Design](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [source:inclusion-ai:the-community-stories-of-vllm-and-sglang] [The Community Stories of vLLM and SGLang (Inclusion AI)](https://www.inclusion-ai.org/blog/llm-landscape-vllm-sgl/)
- [source:arxiv:2605.24259] [Resident KV Claims: A Conformance Contract for Future Reuse under Active KV Pressure](https://arxiv.org/abs/2605.24259)
- [source:arxiv:2606.01387] [Fail-Closed Lowering of Resident KV Claims onto LLM Serving Runtimes](https://arxiv.org/abs/2606.01387)
- [source:arxiv:2605.08840] [ReST-KV: Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing](https://arxiv.org/abs/2605.08840)
- [source:arxiv:2605.09649] [Make Each Token Count: Towards Improving Long-Context Performance with KV Cache Eviction](https://arxiv.org/abs/2605.09649)
- [source:arxiv:2512.06727] [KV-CAR: KV Cache Compression using Autoencoders and KV Reuse in Large Language Models](https://arxiv.org/abs/2512.06727)
- [source:arxiv:2607.06519] [FreqDepthKV: Frequency-Guided Depth Sharing for Robust KV Cache Compression in Long-Context LLM Inference](https://arxiv.org/abs/2607.06519)
- [source:arxiv:2504.03648] [AIBrix: Towards Scalable, Cost-Effective Large Language Model Serving](https://arxiv.org/abs/2504.03648)
