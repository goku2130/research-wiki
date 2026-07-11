---
id: deepmind:alphageometry-2-new-geometry-reasoning-s
type: web
title: 'AlphaGeometry 2: New geometry reasoning system'
url: https://deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level/
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

# AlphaGeometry 2 and AlphaProof

AlphaGeometry 2 and AlphaProof are advanced AI reasoning systems developed by Google DeepMind to solve complex mathematical problems at the level of the International Mathematical Olympiad (IMO).

### Core Problem
Current AI systems struggle with general mathematical reasoning due to two primary constraints: limited reasoning skills and a scarcity of high-quality training data. While natural language models can access vast amounts of data, they are prone to "hallucinations"—generating plausible but logically incorrect intermediate reasoning steps. Conversely, formal mathematical languages provide verifiable correctness but suffer from a lack of human-written training data.

### Methodology

#### AlphaProof
AlphaProof is a reinforcement-learning (RL) based system designed for formal mathematical reasoning using the **Lean** formal language. Its "recipe" consists of the following steps:
1. **Formalization**: A fine-tuned Gemini model translates natural language problem statements into formal Lean statements.
2. **Candidate Generation**: The system generates potential solution candidates.
3. **Verification**: AlphaProof searches through possible proof steps within Lean to either prove or disprove the candidates.
4. **Reinforcement**: Every verified proof is used to reinforce the underlying language model via the **AlphaZero** algorithm.
5. **Iterative Training**: The system was trained over several weeks by proving or disproving millions of problems. During competitions, this loop is applied to self-generated variations of the target problems to refine the final solution.

#### AlphaGeometry 2
AlphaGeometry 2 is a neuro-symbolic hybrid system specifically optimized for geometry. Its improvements over the original AlphaGeometry include:
1. **Enhanced Language Model**: Based on Gemini and trained from scratch using an order of magnitude more synthetic data.
2. **Accelerated Symbolic Engine**: The symbolic engine is two orders of magnitude faster than its predecessor.
3. **Knowledge-Sharing Mechanism**: A novel mechanism that enables the combination of different search trees to tackle more complex problems.
4. **Expanded Scope**: The model is capable of handling equations of angles, ratios, distances, and the movement of objects.

### Key Quantitative Results
The combined system was tested on the 2024 IMO problems, which were manually translated into formal language.

*   **Overall Performance**: The systems solved 4 out of 6 problems, scoring **28 out of 42 points**. This is equivalent to the top end of the silver-medal category (the gold-medal threshold was 29 points).
*   **Problem Breakdown**:
    *   **AlphaProof**: Solved two algebra problems and one number theory problem. This included the competition's hardest problem, which was solved by only five human contestants.
    *   **AlphaGeometry 2**: Solved the geometry problem (Problem 4) in **19 seconds** after formalization.
*   **Historical Benchmark**: AlphaGeometry 2 solved **83%** of all historical IMO geometry problems from the previous 25 years, compared to **53%** achieved by the first AlphaGeometry.

For the geometry problem solved, the system successfully proved the conclusion:

$$
\angle KIL + \angle XPY = 180^\circ
$$

### Limitations
*   **Domain Gaps**: The systems were unable to solve the two combinatorics problems from the IMO 2024.
*   **Input Requirements**: The problems required manual translation into formal mathematical language before the systems could process them.
*   **Computational Time**: While some problems were solved in seconds, others required up to three days of computation.
