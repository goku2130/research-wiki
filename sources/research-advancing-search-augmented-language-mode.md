---
id: research:advancing-search-augmented-language-mode
type: web
title: Advancing Search-Augmented Language Models
url: https://research.perplexity.ai/articles/advancing-search-augmented-language-models
retrieved: '2026-07-12'
maturity: comprehensive
topic: agentic-and-tool-use-rl
---

# Advancing Search-Augmented Language Models

### Core Problem
Training frontier web search agents requires the joint optimization of factual accuracy, trajectory efficiency, and user preference alignment. Perplexity identifies a tension between these objectives: models optimized solely for accuracy tend to overuse tools, while those optimized for brevity often sacrifice reliability or completeness. The central challenge is to improve search capabilities without degrading deployment-critical guardrails (e.g., abstention and language consistency).

### Method/Recipe
Perplexity employs a two-stage post-training pipeline to disentangle deployment constraints from search optimization.

**Stage 1: Supervised Fine-Tuning (SFT)**
The policy model (initialized from the Qwen3 family) undergoes SFT to establish non-negotiable behaviors. The SFT mixture consists of:
1. **Preference-oriented examples:** Data targeting tone, clarity, formatting, and language consistency.
2. **Production-format tool-use trajectories:** Sampled queries across single-, two-, and multi-turn patterns annotated with a tool-calling harness to prevent degradation of base search capabilities.

**Stage 2: On-Policy Reinforcement Learning (RL)**
The SFT checkpoint is refined using Group Relative Policy Optimization (GRPO). To mitigate training-inference mismatch and prevent training collapse, the authors apply token-level Importance Sampling (IS).

**RL Data Construction**
The RL stage utilizes a heterogeneous dataset mixture (90% verifiable QA, 10% rubric-based chat) to balance optimization variance:
*   **Verifiable Search-Agent QA:** Synthetic multi-hop questions are created via a sequential expansion method:
    1. **Seed selection:** Extract atomic statements from entities with multi-source confirmation.
    2. **Multi-hop chain construction:** Link entities 2–4 times.
    3. **Name-free question synthesis:** Recursively replace entity names with descriptors.
    4. **Verification:** Retain only questions with unique answers verified by independent solvers.
*   **Rubric-based General Chat:** For non-verifiable queries, deployment requirements are converted into atomic, objective, and necessary rubrics. These are calibrated using a pass@4 filter to ensure an informative training signal.

### Key Formulas
The core of the RL stage is a **gated aggregation** reward design, ensuring that correctness is a prerequisite for receiving preference credit to prevent reward hacking:

$$
R(\tau_i) = r_{\text{base}}(\tau_i) \cdot s(\tau_i) - \text{pen}_{\text{eff}}(\tau_i)
$$

Where:
*   $r_{\text{base}}(\tau_i) \in \{0, 1\}$ is the binary baseline reward (QA correctness or rubric satisfaction).
*   $s(\tau_i) \in [0, 1]$ is the preference score derived from a Bradley-Terry reward model.
*   $\text{pen}_{\text{eff}}(\tau_i)$ is the combined efficiency penalty.

**Efficiency Penalties**
The model uses group-relative, anchored penalties. For tool calls, the penalty $p_i$ is based on the excess tool calls $\Delta_i$ relative to a group baseline $b_g$:

$$
\Delta_i = \max(0, \tilde{c}_i - b_g) \implies p_i = 1 - \exp(-\Delta_i)
$$

The total efficiency penalty is a weighted sum:

$$
\text{pen}_{\text{eff}}(\tau_i) = \alpha \cdot \text{pen}_{\text{tool}}(\tau_i) + \beta \cdot \text{pen}_{\text{len}}(\tau_i)
$$

### Key Quantitative Results
The pipeline was evaluated using `Qwen3.5-Large` (397B-A17B) and `Qwen3.5-Medium` (122B-A10B).

*   **Search Accuracy:** `Qwen3.5-Large-SFT-RL` matched or exceeded `gpt-5.4` on the **FRAMES** and **Facts Open** benchmarks.
*   **Internal Preference:** The `pplx-sbs-search` metric improved from **0.602** (base) to **0.742** (SFT-RL).
*   **Training Dynamics:** Gradient norms remained stable, and training-inference KL divergence stayed within the $1e-3$ scale.
*   **Efficiency:** Tool-call frequency and generation length were successfully controlled, with penalties decreasing monotonically during training.

### Stated Limitations
The authors note that jointly optimizing search quality and deployment constraints in a single RL stage is difficult; RL-only training typically underperforms on deployment guardrails, while SFT alone can compromise search performance. Additionally, they observed that smaller reward models fail to capture fine-grained preferences and can reinforce undesirable policy behaviors.
