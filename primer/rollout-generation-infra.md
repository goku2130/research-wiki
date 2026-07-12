---
title: Rollout generation infrastructure
kind: primer
reference: ../topics/rollout-generation-infra.md
updated: '2026-07-12'
---

# Rollout Generation Infrastructure: PagedAttention vs RadixAttention

Rollout generation — the repeated, high-throughput sampling of model outputs for RLHF, rejection sampling, and agentic workflows — has made KV cache reuse the central systems bottleneck. Two open-source engines dominate: vLLM with **PagedAttention** (block-level paging + copy-on-write) and SGLang with **RadixAttention** (radix-tree prefix matching + LRU eviction). By the end of this primer you will understand why their architectural divergence creates a clean workload split (vLLM wins one-shot batch throughput; SGLang wins multi-turn and structured generation), how the emerging **Resident KV Claim** contract formalizes the tension between cached prefixes and in-flight requests, and why advanced eviction methods (ReST-KV, DBTrimKV, FreqDepthKV, KV-CAR) are turning compression into a reasoning enhancer rather than a fidelity trade-off.

---

## Core Mechanism 1: PagedAttention — OS-Style Virtual Memory for KV Cache

**Intuition.** Pre-vLLM servers allocated a contiguous KV buffer per sequence up to the maximum context length. At any moment, only the *filled* prefix mattered; the rest sat idle, fragmenting GPU memory. PagedAttention borrows the OS idea of *paging*: split each sequence's KV cache into fixed-size blocks (default 16 tokens), store blocks in non-contiguous physical memory, and maintain a logical-to-physical block table. When a sequence grows, you allocate the next free block — no pre-reservation, no external fragmentation.

**The key equation** reformulates attention as a block-wise reduction. For query $q_i$ at position $i$, let $B$ be the block size. The attention output is computed by iterating over $\lceil i/B \rceil$ key/value blocks:

$$
A_{ij} = \frac{\exp(q_i^T K_j / \sqrt{d})}{\sum_{t=1}^{\lceil i/B \rceil} \exp(q_i^T K_t \mathbf{1} / \sqrt{d})}, \quad
o_i = \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^T
$$

where $K_j, V_j$ are the $j$-th key/value blocks and $A_{ij}$ is the row vector of scores for that block. The kernel maps one thread block per (head, sequence), loads queries into shared memory, streams keys through registers, and reduces partial dot products across warps before softmax and value aggregation.

**Why it works.** The block table enables **copy-on-write (CoW) sharing**: multiple sequences with a common prefix point to the same physical blocks (reference-counted). On first write to a shared block, the engine allocates a fresh block, copies data, and decrements the old reference. This yields 37.6–66.3% memory reduction for beam search and 1.67–3.58× throughput gains on shared prefixes versus Orca. The trade-off: the indirection adds 20–26% attention latency versus a contiguous kernel.

---

## Core Mechanism 2: RadixAttention — Prefix Tree with Automatic Reuse

**Intuition.** Real workloads (multi-turn chat, few-shot prompts, agent trajectories) share *partial* prefixes — system prompts, few-shot examples, evolving conversation histories — but rarely have *identical* full prefixes. A radix tree (compressed prefix tree) stores every unique token sequence as a path from the root; each node holds the KV tensor for its edge's token span. On a new request, the runtime walks the tree: matching prefixes reuse cached KVs; the first divergence triggers fresh computation and tree insertion. An LRU policy evicts leaf nodes when GPU memory fills.

**The key equation** is the scheduling objective. **Cache-aware scheduling** maximizes prefix-hit rate by prioritizing requests with the longest matched prefix — proven equivalent to depth-first search (DFS) order on the radix tree. If $L(r)$ is the matched prefix length for request $r$, the scheduler picks $\arg\max_r L(r)$. This simple rule drives RadixAttention's 50–99% cache hit rates (avg 96% of theoretical optimal) with <0.3% management overhead.

**Why it works.** The tree lives on CPU (minimal GPU overhead); KV tensors sit on GPU in a paged layout (one page per token). The frontend `fork` primitive sends prompt prefixes as "hints" to guarantee correct insertion. Crucially, **Compressed Finite State Machines** merge "singular-transition edges" in constrained decoding (regex/JSON), enabling jump-forward decoding of multiple tokens per forward pass — a 1.6× throughput gain on JSON. **API Speculative Execution** ignores stop conditions on the first call, generates extra tokens speculatively, then reuses them for subsequent `gen` primitives, cutting input token costs ~3× for field extraction.

---

## Runnable Check: PagedAttention Block Allocation

```python
# Simulates PagedAttention block table allocation with copy-on-write sharing
# Block size = 16 tokens; physical blocks are 0-indexed integers

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.free = list(range(num_blocks))          # physical block IDs
        self.refcount = [0] * num_blocks             # reference counts
        self.tables = {}                             # seq_id -> list[physical_block]

    def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        """Allocate blocks for a new sequence; returns physical block IDs."""
        needed = (num_tokens + self.block_size - 1) // self.block_size
        assert needed <= len(self.free), "OOM"
        blocks = [self.free.pop() for _ in range(needed)]
        for b in blocks:
            self.refcount[b] = 1
        self.tables[seq_id] = blocks
        return blocks

    def fork(self, parent_id: int, child_id: int, prefix_tokens: int) -> list[int]:
        """Child shares parent's prefix blocks (CoW); returns child's block table."""
        parent_blocks = self.tables[parent_id]
        prefix_blocks = (prefix_tokens + self.block_size - 1) // self.block_size
        child_blocks = parent_blocks[:prefix_blocks]
        for b in child_blocks:
            self.refcount[b] += 1
        self.tables[child_id] = child_blocks
        return child_blocks

    def cow_write(self, seq_id: int, token_idx: int) -> int:
        """Copy-on-write at token_idx; returns new physical block ID."""
        block_idx = token_idx // self.block_size
        old_block = self.tables[seq_id][block_idx]
        if self.refcount[old_block] == 1:
            return old_block  # exclusive ownership, no copy needed
        # CoW: allocate new block, copy (simulated), update refcounts
        new_block = self.free.pop()
        self.refcount[old_block] -= 1
        self.refcount[new_block] = 1
        self.tables[seq_id][block_idx] = new_block
        return new_block

    def free_seq(self, seq_id: int):
        for b in self.tables.pop(seq_id):
            self.refcount[b] -= 1
            if self.refcount[b] == 0:
                self.free.append(b)

# ---- Assertions ----
mgr = BlockManager(num_blocks=100, block_size=16)

# Parent: 48 tokens -> 3 blocks
p_blocks = mgr.allocate("parent", 48)
assert len(p_blocks) == 3 and all(mgr.refcount[b] == 1 for b in p_blocks)

# Child forks first 32 tokens (2 blocks) -> shares 2 blocks
c_blocks = mgr.fork("parent", "child", 32)
assert len(c_blocks) == 2
assert c_blocks == p_blocks[:2]
assert all(mgr.refcount[b] == 2 for b in c_blocks)  # refcount = 2

# Child writes at token 20 (block 1) -> CoW triggers new block
new_b = mgr.cow_write("child", 20)
assert new_b != c_blocks[1]
assert mgr.refcount[c_blocks[1]] == 1   # parent still holds original
assert mgr.refcount[new_b] == 1         # child owns new block
assert mgr.tables["child"][1] == new_b  # child's table updated

# Free parent -> child's shared block refcount drops to 1, not freed
mgr.free_seq("parent")
assert mgr.refcount[c_blocks[0]] == 1
assert mgr.refcount[c_blocks[1]] == 0   # parent's original block 1 freed
assert c_blocks[0] in mgr.free          # block 0 returned to free list

print("All PagedAttention CoW invariants hold.")
```

---

## Load-Bearing Disagreements

1. **One-shot vs. multi-turn throughput.** RunPod's controlled benchmark (DeepSeek-R1-Distill-Llama-70B, 2×H100, 7k context) shows **vLLM 1.1× faster on one-shot** (60.0 vs 52.7 tok/s) while **SGLang leads on multi-turn cached** (35.0 vs 32.8 tok/s, +20% vs fresh). The GoPenAI blog corroborates a 10–20% SGLang edge on multi-turn. The root cause: vLLM's Automatic Prefix Caching (APC) requires *exact* token-sequence matches and often needs manual configuration; RadixAttention automatically matches *partial* prefixes in dynamic conversations.

2. **Exact-match vs. partial-prefix reuse.** vLLM's APC is a hash-based exact-prefix lookup — ideal for templated prompts, beam search, and parallel sampling where prefixes are identical by construction. RadixAttention's tree structure naturally handles overlapping but non-identical prefixes (system prompt + varying few-shots, evolving chat history). No public benchmark directly compares them on **PPO/GRPO rollout generation** with repeated system prompts and diverging completions — the critical workload for RL training.

---

## Advanced Eviction & Compression: Compression as Reasoning Enhancement

Four recent methods move beyond naive eviction:

- **ReST-KV** reformulates eviction as layer-wise output reconstruction with spatial-temporal smoothing (EMA + adaptive window). It measures the $\ell_2$ error of removing a KV pair *after* attention redistribution, then smooths importance scores over time and position. Result: 15.2% gain on RULER, 98% Needle-in-Haystack at 1/32 token retention.
- **DBTrimKV** identifies **attention dilution** — softmax mass spreading over distractors — as the primary long-context failure. It learns per-token/head retention gates $\beta_{\ell,h,t}$, shares the final projection across all gates for global comparability, and evicts by a single global budget $M_{\text{global}}$. **Exceeds full-cache performance by 9.20% on LongBench-v2** — eviction *improves* reasoning by suppressing dilution.
- **FreqDepthKV** factorizes KV across depth via DCT: shared low-frequency components + sparse high-frequency residuals routed per-head at prefill. Achieves 3.9× compression with near-lossless accuracy (EM 58.3 vs 58.7 full KV).
- **KV-CAR** stacks per-layer autoencoders (dimension $D \to d \ll D$) with similarity-guided head reuse across adjacent layers. Up to 47.85% KV memory reduction on GPT-2.

---

## Resident KV Claims: From Silent Eviction to Explicit Arbitration

The **Resident KV Claim** contract formalizes the tension between *resident reusable KV* (cached prefixes for future requests) and *active live KV* (memory for in-flight requests). A claim carries `protection_mode` (`soft_priority`, `hard_protected`, `demotable`, `offloadable`, `expiring`) and a `materialization_predicate`. The feasibility boundary is:

$$
\text{protected\_resident\_kv} + \text{active\_live\_kv} \le \text{usable\_kv}
$$

When violated, the **Active/Resident Arbiter** must choose an explicit action — **Resident Victim Exclusion** (protect residents; refuse active), **Write No-Admit** (serve active but block its KV from becoming resident), **Relaxation** (demote/expire resident first) — rather than silently evicting. A vLLM prototype demonstrates this: in a "60/70/80" litmus test (60 protected resident, 70 active, 80 usable), native eviction destroys 50 resident blocks; hard exclusion preserves all 60 and converts the conflict into a **scheduler-visible active refusal**. The **Fail-Closed Lowering** checker maps claim modes to obligation bundles; no runtime yet achieves `native_sound` conformance (TensorRT-LLM and SGLang/HiCache reach `sound_with_adapter` for specific modes; vLLM needs a backend patch for `offloadable`).

---

## Current Status
vLLM is the de facto default for high-throughput batch inference; SGLang dominates structured generation and multi-turn chat; Resident KV Claims and advanced eviction methods are converging toward explicit arbitration primitives and compression that can exceed full-cache accuracy.

## Full Reference
See the comprehensive reference article "Rollout generation infrastructure" for all sources, benchmarks, and related-topic links.

---
*Full reference (citations, derivations, variants):* [Rollout generation infrastructure](../topics/rollout-generation-infra.md)
