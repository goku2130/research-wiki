---
id: neurips:co-evolving-llm-coder-and-unit-tester-vi
type: web
title: Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning
url: https://neurips.cc/virtual/2025/poster/115329
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-math-and-code
---

The paper "Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning" introduces CURE, a reinforcement learning framework designed to improve the coding and unit test generation capabilities of Large Language Models (LLMs) without relying on ground-truth code solutions for supervision.

**Core Problem:**
Traditional methods for fine-tuning LLMs for unit test generation heavily depend on ground-truth code solutions, which are costly and labor-intensive to collect. This limitation restricts the scalability and diversity of training data. The core problem CURE addresses is how to co-evolve an LLM's ability to generate both correct code and effective unit tests without this external ground-truth supervision, thereby enabling more flexible and scalable training.

**Method/Recipe Step by Step:**
CURE operates as a novel reinforcement learning framework that co-evolves a self-play agent functioning as both a code generator and a unit test generator. The process involves:

1.  **Co-evolutionary Training:** The framework trains a single LLM to perform two distinct but interacting roles: generating code and generating unit tests.
2.  **Interaction-Based Supervision:** Instead of ground-truth code, CURE uses the interaction outcomes between generated codes and generated tests to provide mutual supervision.
3.  **Reward Design:** A dedicated reward mechanism is designed to guide the co-evolution. This reward is based on a pairwise reward matrix derived from the execution results of generated codes against generated tests.
    *   The coder produces various solutions, some correct and some incorrect.
    *   The unit tester learns to identify flaws in the coder's incorrect solutions, thereby improving its ability to generate robust and non-naive tests.
4.  **Response-Length-Guided Reward Transformation (for Long-CoT models):** For models employing long-chain-of-thought (Long-CoT) reasoning, a transformation is applied to the reward. This aims to make the unit test generator more efficient by encouraging shorter, yet effective, test generation, addressing the slow inference issue associated with Long-CoT models.

**Key Formulas in LaTeX:**
The provided text describes the reward design conceptually as a "pairwise reward matrix based on interactions between generated codes and generated tests." However, it does not explicitly provide the mathematical formulas for this reward function or the response-length-guided reward transformation.

**Key Quantitative Results and Numbers:**
*   **Test-time scaling:** CURE models achieved a **6.2% improvement** in accuracy over the base model.
*   **Agentic unit test generation:** CURE models showed a **25.1% improvement** in accuracy.
*   **CURE-4B performance:** The CURE-4B model consistently outperformed Qwen3-4B.
*   **Inference efficiency (unit test generation):** CURE-4B achieved **64.8% inference efficiency** compared to Qwen3-4B in unit test generation.
*   CURE models were derived from base models of varying sizes: **4B, 7B, and 14B**.

**Stated Limitations:**
The source does not explicitly list limitations of the CURE framework itself. However, it implicitly highlights a limitation of *traditional approaches* to unit test generation:
*   "Traditional approaches for fine-tuning LLMs on unit test generation rely heavily on ground-truth code solutions in the training data."
*   "However, training unit test generators in these ways requires supervision from ground-truth code solutions, whose collection is both costly and labor-intensive, thereby limiting the scale and diversity of usable training data."
*   It also mentions a limitation of Long-CoT models: "while long-chain-of-thought (long-CoT) models represent some of the most advanced AI capabilities to date, they suffer from extremely slow inference." CURE addresses this with a specific reward transformation.
