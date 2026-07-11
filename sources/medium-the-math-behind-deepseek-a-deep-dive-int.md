---
id: medium:the-math-behind-deepseek-a-deep-dive-int
type: web
title: 'The Math Behind DeepSeek: A Deep Dive into Group Relative ...'
url: https://medium.com/@sahin.samia/the-math-behind-deepseek-a-deep-dive-into-group-relative-policy-optimization-grpo-8a75007491ba
retrieved: '2026-07-11'
maturity: comprehensive
topic: grpo
---

The provided article introduces Group Relative Policy Optimization (GRPO) as the core reinforcement learning (RL) algorithm powering DeepSeek's reasoning capabilities in Large Language Models (LLMs).

**Core Problem:**
Traditional RL methods, such as Proximal Policy Optimization (PPO), face significant challenges when applied to reasoning tasks in LLMs. These challenges include:
1.  **Dependency on a Critic Model:** PPO necessitates a separate critic model to estimate the value of each response, which doubles memory requirements and complicates training, especially for subjective or nuanced evaluations.
2.  **High Computational Cost:** RL pipelines are computationally intensive for iterative evaluation and optimization of responses, and these costs escalate with large LLMs.
3.  **Scalability Issues:** Absolute reward evaluations struggle to generalize across diverse reasoning tasks.

**Method/Recipe Step by Step:**
GRPO is designed to enhance reasoning in LLMs by optimizing the model through the evaluation of groups of responses relative to one another, rather than relying on external evaluators (critics).

The article outlines how GRPO addresses the aforementioned challenges:
1.  **Critic-Free Optimization:** GRPO eliminates the need for a critic model by comparing responses within a group. This approach significantly reduces computational overhead.
2.  **Relative Evaluation:** Instead of depending on an external evaluator, GRPO utilizes group dynamics to assess the quality of responses.

The article does not provide further steps or a detailed recipe for the GRPO algorithm beyond these high-level descriptions.

**Key Formulas in LaTeX:**
The provided text does not contain any mathematical formulas.

**Key Quantitative Results and Numbers:**
The provided text does not contain any quantitative results or numbers.

**Stated Limitations:**
The article focuses on the advantages of GRPO over traditional RL methods and does not explicitly state any limitations of the GRPO algorithm itself. The limitations discussed are those of traditional RL methods like PPO, which GRPO aims to overcome.
