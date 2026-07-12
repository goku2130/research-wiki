---
id: arxiv:2606.01387
type: paper
title: Fail-Closed Lowering of Resident KV Claims onto LLM Serving Runtimes
url: https://arxiv.org/abs/2606.01387
retrieved: '2026-07-12'
maturity: comprehensive
topic: rollout-generation-infra
---

# Summary: Fail-Closed Lowering of Resident KV Claims onto LLM Serving Runtimes

### Core Problem
LLM serving runtimes provide various KV-cache primitives (e.g., retention priority, TTL, offloading, and KV-aware routing). However, these primitives are often "weaker" than the formal obligations required for a **ResidentClaim**—a contract ensuring specific future-reuse of KV data. A runtime may expose a "priority" field without accepting responsibility for the resulting reuse claim. The core problem is "lowering": determining when a concrete runtime primitive or adapter conformantly satisfies the semantic obligations of a ResidentClaim contract.

### Method and Lowering Relation
The author proposes a fail-closed lowering relation and a checker to classify runtime support. The process follows these steps:

1.  **Define Mode Obligations**: ResidentClaim modes (e.g., `best_effort`, `soft_priority`, `hard_protected`, `offloadable`, `routed_reuse`) are mapped to "obligation bundles" $O[m]$. Obligations include `claim_identity`, `explicit_acceptance`, `materialization_predicate`, and `ordered_lifecycle_events`.
2.  **Curate Descriptors**: Runtimes are described via descriptors containing native evidence ($d.\text{native}$) and adapter evidence ($d.\text{adapters}$).
3.  **Apply the Lowering Relation**: A backend, adapter, and evidence set $E$ satisfy a mode $m$ if every required obligation $o \in O[m]$ is supported by anchored evidence at an allowed adapter depth.

The formal lowering condition is defined as:

$$
\text{Lower}(d, a, E, m) \iff \forall o \in O[m], \exists e \in (d.\text{native} \cup d.\text{adapters}[\le a] \cup E) \text{ s.t. } \text{supports}(e, o) \land \text{anchored}(e) \land \text{depth\_allowed}(a, o)
$$

The checker assigns one of five labels:
*   **native_sound**: Natively satisfies all obligations.
*   **sound_with_adapter**: Satisfied via an adapter or backend patch.
*   **approximate**: Related primitives exist, but obligations are missing.
*   **rejected**: The mapping is unsound or violates the contract.
*   **unknown**: Evidence is inconclusive.

### Key Quantitative Results
The study found that no public runtime currently achieves `native_sound` conformance.

*   **TensorRT-LLM**: Achieved `sound_with_adapter` for `soft_priority` under controlled pressure. In trials, lower-priority prompts were lost first in 5/5 runs for both original and swapped priority assignments. 13/13 claims were joinable before pressure, and 16/16 materialization changes were reconstructable.
*   **SGLang/HiCache**: Achieved `sound_with_adapter` for `best_effort` via telemetry joins. A warm shared-prefix request reported `cached_tokens=35` compared to `0` for cold or wrong-prefix controls, with 100 `BlockStored` events.
*   **Patched vLLM Witness**: A local `backend_patch` was used to demonstrate the implementability of `offloadable` semantics. 
    *   **Stability**: 131/131 subprocesses produced valid event sequences.
    *   **Accuracy**: 30/30 observation passes and 30/30 failure-outcome passes; 0/41 false-positive passes.
    *   **Attribution**: In 3/3 repetitions, failure/refusal was attributed only to the target claim.
    *   **Diagnostics**: Median resident wall time was $0.187\text{s}$, and failure-to-outcome latency was $151,544\text{ns}$.

### Stated Limitations
*   **Lack of Native Support**: No native conformance was established for TensorRT-LLM, SGLang, Dynamo, or upstream vLLM.
*   **Witness Scope**: The vLLM results are based on a local `backend_patch` and do not represent upstream support or production performance.
*   **Refusal Boundary**: Refusal occurs at the scheduler-boundary (invalid-KV-load handling) rather than as pre-admission scheduling refusal.
*   **Experimental Constraints**: Restoration failure was tested using controlled injection; connector runs were small, in-process, and limited to a single GPU.
*   **Metrics**: TTFT (Time to First Token) was unavailable due to a vLLM `OffloadingConnector` metrics serialization issue.
