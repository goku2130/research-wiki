---
id: arxiv:2501.02497
type: paper
title: 'Test-Time Compute: from System-1 Thinking to System-2 Reasoning'
url: https://arxiv.org/html/2501.02497v2
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

# Summary: Test-Time Compute: From System-1 Thinking to System-2 Reasoning

## Core Problem
While scaling training data and parameters has driven LLM progress, further scaling is hindered by data scarcity and resource constraints. Existing models primarily rely on **System-1 thinking** (fast, intuitive, pattern-based), which lacks robustness and fails at complex reasoning tasks. The core challenge is transitioning to **System-2 thinking** (slow, deliberate, reflective) by scaling **test-time compute**—increasing the computational effort during the inference phase to unlock higher reasoning capabilities.

## Methodology
The authors categorize test-time compute into two primary paradigms: Test-Time Adaptation (TTA) for System-1 models and Test-Time Reasoning for System-2 models.

### 1. Test-Time Adaptation (System-1)
TTA aims to improve robustness and generalization against distribution shifts through four methods:
*   **Updating the Model:** Fine-tuning a subset of parameters (e.g., normalization layers, soft prompts, or adapters) using unsupervised signals. This includes **Test-Time Training (TTT)** via auxiliary tasks and **Fully Test-Time Adaptation (FTTA)** using internal feedback like entropy minimization.
*   **Modifying the Input:** Optimizing In-Context Learning (ICL) by retrieving semantically similar demonstrations (using BM25 or SentenceBERT) and optimizing their ordering.
*   **Editing the Representation:** Using steering vectors (e.g., ActAdd) or projecting representations onto specific directions (SEA) to align intermediate layer information with the desired output.
*   **Calibrating the Output:** Fusing the model's probability distribution with retrieved probabilities from a memory pool or $k$-NN classifier.

### 2. Test-Time Reasoning (System-2)
This paradigm enables explicit, logical thinking through a three-part recipe:

**Step A: Feedback Modeling**
*   **Score-based Verifiers:** Outcome-based Reward Models (ORMs) evaluate final results, while Process-based Reward Models (PRMs) evaluate individual reasoning steps.
*   **Generative-based Critics:** LLMs provide natural language critiques, either training-free (LLM-as-a-Judge) or via Supervised Fine-Tuning (SFT) and DPO.

**Step B: Search Strategies**
*   **Repeated Sampling:** Employs **Majority Voting** (selecting the most frequent answer) or **Best-of-N (BoN)** sampling (using a verifier to pick the highest-scoring response).
*   **Self-Correction:** An iterative loop where the model revises outputs based on feedback from humans, external tools (e.g., compilers), other models, or self-critique.
*   **Tree Search:** Combines parallel brainstorming with backtracking. **Uninformed search** (BFS/DFS) is used in Tree-of-Thought (ToT), while **Heuristic search** (A*, MCTS) uses value functions to guide expansion through selection, expansion, simulation, and backpropagation.

**Step C: Improvement Training**
Models are fine-tuned to approximate the BoN distribution (e.g., ReST) or trained on high-quality trajectories collected via MCTS to reduce the search space during future inference.

## Key Quantitative Results
*   **Reasoning Gains:** Self-consistency CoT improves accuracy by **18%** over vanilla CoT in mathematical reasoning tasks.
*   **Data Efficiency:** ThinkPRM requires only **1%** of the process supervision data compared to discriminative PRMs.
*   **Critic Training:** Strong critique abilities can be achieved with as few as **40K samples** for SFT and DPO.
*   **Benchmark Difficulty:** On the FrontierMath benchmark, even the advanced o3 model has not achieved **30% accuracy**.

## Stated Limitations
*   **System-1 TTA:** Parameter updating is often unstable and inefficient for LLMs; output calibration risks knowledge leakage; representation editing requires manual prior knowledge.
*   **System-2 Reasoning:** 
    *   **Repeated Sampling:** High computational cost and inference latency.
    *   **Self-Correction:** Effectiveness is controversial; models often struggle to locate errors even if they possess the ability to correct them.
    *   **Tree Search:** High implementation complexity.
*   **General:** There is currently no universal test-time scaling law, and System-2 models struggle to generalize to non-symbolic reasoning tasks.
