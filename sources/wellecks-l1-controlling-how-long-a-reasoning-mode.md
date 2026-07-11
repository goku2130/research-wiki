---
id: wellecks:l1-controlling-how-long-a-reasoning-mode
type: web
title: 'L1: Controlling how long a reasoning model thinks with reinforcement learning'
url: https://wellecks.com/data/welleck2025scifm_tutorial.pdf
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

# Summary: Test-Time Scaling for Mathematical Reasoning

This tutorial outlines strategies for improving the performance of large language models (LLMs) on complex tasks, such as mathematical reasoning, by increasing compute at generation time (test-time scaling) rather than relying solely on pretraining compute.

### Core Problem
The central challenge is to leverage test-time compute to improve accuracy. While scaling pretraining (larger models/datasets) is a standard approach, test-time scaling focuses on how a model generates outputs—either by generating multiple candidates or generating longer, more detailed reasoning chains—to reach a correct solution.

### Methods and Strategies
The author introduces a unified view called **Meta-Generation**, where a "meta-generator" manages a "generator" (the LLM) through various strategies:

#### 1. Parallel Generation
This approach explores the output space by generating $N$ full sequences and aggregating them:
*   **Best-of-N:** Candidates are sampled, and the one with the highest score from a reward model $V(y)$ is selected: $\arg \max_{y \in \{y^{(1)}, \dots, y^{(N)}\}} V(y)$.
*   **Voting (Self-Consistency):** The most frequent answer among candidates is selected.
*   **Weighted Voting:** Combines reward model scores with voting to improve selection.

#### 2. Tree Search
Unlike parallel generation, tree search leverages intermediate evaluation to backtrack and explore. It requires defined states $s$, transitions $s \to s'$, and scores $v(s)$. An example provided is **Reward Balanced Search (Rebase)**, which uses a budget $P$ to balance exploration and exploitation.

#### 3. Refinement and Self-Correction
This involves iteratively improving a generation $y^{(i)}$ using feedback $F(y^{(i)})$:

$$
y^{(i+1)} = g(x, y^{(i)}, F(y^{(i)}))
$$

*   **Extrinsic Feedback:** Uses external tools (e.g., program verifiers, code interpreters, retrievers). This is generally successful as it provides new information and localizes errors.
*   **Intrinsic Feedback:** The model re-prompts itself. This yields mixed results in mathematical reasoning because the feedback is often too noisy.

#### 4. Long Chain-of-Thought (CoT)
This shifts from parallel to sequential scaling. Models are trained via reinforcement learning (RL) to generate a "thought" $z$ before the final answer $a$.
*   **Training:** RL policies reward the model based on the correctness of the final answer.
*   **Emergent Patterns:** Models develop behaviors such as expressing uncertainty ("Wait..."), backtracking ("Alternatively..."), and self-verification ("Let's check if we made an error").
*   **L1 Control:** To manage the cost of long CoT, the **L1 model** uses RL to adhere to length constraints provided in the prompt (e.g., "use up to 1000 tokens") by applying a length constraint penalty to the reward.

### Key Formulas and Quantitative Results
The tutorial highlights the convergence of voting accuracy as the number of candidates $N \to \infty$:

$$
\frac{1}{M}\sum_{i=1}^{M}{\mathbb{I}}\left[a_{i}^{*}=\arg\max_{a}\sum_{z}v(x,z,a)g(z,a|x)\right]
$$

Where $M$ is the number of test examples, $z$ is the solution path, and $a$ is the answer.

**Quantitative Observations:**
*   **Response Length:** Training for long CoT (e.g., DeepSeek-AI) can lead to response lengths exceeding 10,000 tokens.
*   **Inference Scaling Laws:** Research indicates that using a smaller model and generating more tokens is often more compute-optimal than using a larger model with fewer tokens.

### Stated Limitations
*   **Reward Model Imperfection:** Best-of-N can suffer from "over-optimization" if the reward model is imperfect.
*   **Intrinsic Feedback:** Self-correction without external tools is often ineffective for reasoning tasks due to noisy feedback.
*   **Convergence Ceiling:** Voting accuracy does not improve indefinitely; it eventually converges to a limit determined by the generator $g$ and reward model $v$.
