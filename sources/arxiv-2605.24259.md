---
id: arxiv:2605.24259
type: paper
title: 'Resident KV Claims: A Conformance Contract for Future Reuse under Active KV
  Pressure'
url: https://arxiv.org/abs/2605.24259
retrieved: '2026-07-12'
maturity: comprehensive
topic: rollout-generation-infra
---

# Resident KV Claims: A Conformance Contract for Future Reuse

### Core Problem
In LLM serving, there is a critical tension between **resident reusable KV** (cached prefixes for future requests) and **active live KV** (memory required for in-flight requests). Existing mechanisms—such as priority-based eviction or "write no-admit" policies—lack a formal conformance contract to handle scenarios where both cannot fit in the usable memory pool. Specifically, a "write no-admit" policy prevents new active KV from being cached for the future, but it does not prevent the active request's live allocation from evicting existing resident KV. This results in "unreported resident loss," where valuable cached state is destroyed to serve a current request without explicit arbitration or telemetry.

### Method and Recipe
The author proposes a **Resident KV Claim** conformance contract that shifts the focus from simple cache replacement to an active/resident arbitration framework.

**1. The Resident Claim Schema**
A claim is defined by a set of required fields:
*   **Input:** `claim_id`, `owner_scope`, `cache_identity` (model/tokenizer/format), `object_id`, `materialization_predicate` (the condition for the cache to be "useful"), `footprint_blocks`, `protection_mode`, and optional `duration_steps`.
*   **Protection Modes:** 
    *   `soft_priority`: Influences eviction order.
    *   `hard_protected`: Cannot be broken without explicit refusal, demotion, or expiry.
    *   `demotable`/`offloadable`/`expiring`: Specific lifecycle transitions before loss.

**2. Lifecycle and Arbitration**
The contract follows a lifecycle: $\text{Submitted} \rightarrow \text{Accepted} \rightarrow \text{Materialized} \rightarrow (\text{Demoted/Expired/Refused/Harmed})$. 
When a conflict occurs, the **Active/Resident Arbiter** must choose an explicit action:
*   **Resident Victim Exclusion:** Protects residents; the active request is refused or deferred.
*   **Write No-Admit:** Serves the active request but prevents its KV from becoming resident.
*   **Relaxation:** Demotes or expires the resident claim before eviction.
*   **Other Actions:** Offloading, bounding active-live KV, or routing to another worker.

**3. Implementation and Evaluation**
*   **MicroRuntime:** An executable model used to test contract semantics and materialization predicates.
*   **vLLM Prototype:** A patch-level implementation adding resident claim metadata, hard-protected victim exclusion, and scheduler-visible active refusal.
*   **Litmus Suite:** A series of probes (e.g., the "60/70/80" scenario) to verify if the runtime converts silent resident loss into explicit active-side refusal.

### Key Formulas
The feasibility of coexisting resident and active KV is defined by the boundary:

$$
\text{protected\_resident\_kv} + \text{active\_live\_kv} \le \text{usable\_kv}
$$

If this inequality is violated, the runtime must trigger an explicit arbitration action (e.g., `active_refused`) rather than performing ordinary cache eviction.

### Key Quantitative Results
*   **Materialization vs. Block Count:** In controlled traces, a "naive fair share" policy retained 1,104 cached tokens across three spans but produced a **thresholded value of 0** because the leading-prefix materialization predicate was not met. In contrast, "complete-prefix" policies retained fewer total blocks but achieved a value of 18.
*   **The 60/70/80 Boundary:** In a setup with 60 protected resident blocks, 70 active live blocks, and an 80-block usable pool:
    *   **Native/Write No-Admit:** Served the active request but evicted 50 resident blocks, leaving only 10.
    *   **Hard Resident Exclusion:** Preserved all 60 resident blocks and converted the failure into a **scheduler-visible active refusal**.
*   **TTFT Motivation:** Using SmolLM2 and Qwen2.5-Coder-7B, repeated prompts showed Time To First Token (TTFT) dropping from **0.345s** (first request) to **0.042s** and **0.035s** (repeats), demonstrating the serving-visible value of preserved residents.
*   **Live Scheduler Pressure:** A run with 40 resident blocks and an active request requiring 46 blocks (total 86 > 68 usable) resulted in a `protected_resident_capacity_refused` stop reason with a recorded **19-block shortfall**.

### Stated Limitations
*   **Not a Performance Tool:** The work does not claim production speedups, p95/p99 latency improvements, or end-to-end throughput gains.
*   **Prototype Grade:** The vLLM implementation is a patch-level, env-var driven prototype, not a stabilized upstream API.
*   **Scope:** It does not provide a universal allocator, a learned prediction model for reuse, or a solution for multi-tenant fairness.
*   **Generality:** Empirical evidence is primarily based on vLLM; other runtimes (SGLang, TensorRT-LLM) are used as design comparators.
