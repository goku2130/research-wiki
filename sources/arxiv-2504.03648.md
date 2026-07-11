---
id: arxiv:2504.03648
type: paper
title: 'AIBrix: Towards Scalable, Cost-Effective Large Language Model Serving'
url: https://arxiv.org/abs/2504.03648
retrieved: '2026-07-11'
maturity: comprehensive
topic: rollout-generation-infra
---

# AIBrix: Towards Scalable, Cost-Effective Large Language Model Serving

AIBrix is a cloud-native, open-source framework designed to optimize the deployment of large language models (LLMs) by co-designing system-level orchestration with inference engines (e.g., vLLM).

### Core Problem
While model and engine optimizations (like PagedAttention) improve efficiency, real-world production deployment faces systemic bottlenecks. Traditional cloud-native stacks (e.g., Knative, Istio) are designed for stateless microservices and lack GPU awareness, failing to address token-based rate limiting or KV cache memory pressure. Conversely, traditional ML serving frameworks (e.g., KServe, RayServe) lack deep integration with LLM-specific requirements such as stateful KV cache coordination, variable input/output lengths, and heterogeneous GPU utilization.

### Method and Architecture
AIBrix employs a split architecture consisting of a **Control Plane** (managing model metadata, autoscaling, and LoRA adapters) and a **Data Plane** (handling request dispatching and execution). The framework implements the following technical components:

1.  **High-Density LoRA Management:** Enables dynamic registration, loading, and unloading of LoRA adapters. It utilizes Kubernetes Service and EndpointSlice mechanisms to optimize adapter discovery and placement.
2.  **LLM-Aware API Gateway:** An extension of the Envoy Gateway that replaces blind request distribution with specialized routing policies:
    *   `random`, `throughput`, `least-request`, `least-kv-cache`, `least-latency`, and `prefix-cache-aware` (prioritizing instances with reusable prefix caches).
3.  **Unified AI Runtime:** An abstraction layer that provides vendor-agnostic compatibility between the control plane and various inference engines. It includes a **GPU streaming loader** to bypass disk I/O bottlenecks during model loading.
4.  **LLM-Specific Autoscaling:** Implements sliding window metric aggregation to reduce real-time metric propagation delay, utilizing Knative Pod Autoscaler (KPA) and AIBrix Pod Autoscaler (APA).
5.  **Distributed KV Cache Pool:** A DRAM-based, cross-engine cache that utilizes a scan-resistant eviction policy, asynchronous metadata updates, and shared-memory-based data exchange to enable token reuse across nodes.
6.  **Mixed-Grain Orchestration:** A hybrid approach combining Kubernetes for coarse-grained resource management and Ray for fine-grained application orchestration.
7.  **SLO-Driven Heterogeneous Serving:** A GPU optimizer that uses an Integer Linear Programming (ILP)-based solution to dynamically select GPU configurations based on workload profiling to balance cost and Service Level Objectives (SLOs).
8.  **Accelerator Diagnostics:** Tools for automated fault detection and failure mockup testing for Nvidia GPUs and Ascend 910B NPUs.

### Key Quantitative Results
AIBrix demonstrates significant improvements over standard vLLM and native Kubernetes HPA configurations:

*   **Routing Efficiency:** The LLM-aware gateway reduced mean latency by **19.2%** and P99 latency by **79%**.
*   **Autoscaling:** Compared to native HPA, AIBrix reduced latency by **11.5%**, increased token throughput by **11.4%**, and reduced scaling oscillations by **33%**.
*   **Distributed KV Cache:** When combined with vLLM's prefix caching (evaluated on 4 $\times$ Nvidia A10 GPUs using Bird-SQL), AIBrix achieved:
    *   **Peak Throughput:** $\approx 50\%$ increase.
    *   **Time to First Token (TTFT):** Average reduced by $\approx 65\%$; P99 reduced by $\approx 77\%$.
    *   **Inter-Token Latency (ITL):** Average reduced by $\approx 30\%$; P99 reduced by $\approx 72\%$.
*   **Heterogeneous Serving:** In a mixed A10/L20 GPU setup, AIBrix achieved a **10% cost reduction** compared to a homogeneous L20 deployment, while keeping latency increases within the specified SLO (up to 20% increase).

### Limitations
The authors identify several constraints:
*   **Generalization:** Routing and heterogeneous serving were not fully evaluated under non-ideal workloads.
*   **Profiling Overhead:** The GPU optimizer and autoscaler currently rely on offline model profiling, which is impractical for highly dynamic workloads.
*   **Future Mitigation:** The authors suggest adopting roofline model analysis to provide a more lightweight, structured approach to profiling heterogeneous performance.
