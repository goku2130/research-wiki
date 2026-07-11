---
id: arxiv:2309.03409
type: paper
title: Large Language Models as Optimizers (RLVR/Reasoning context)
url: https://arxiv.org/abs/2309.03409
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

# Optimization by PROmpting (OPRO)

**Optimization by PROmpting (OPRO)** is a framework that leverages Large Language Models (LLMs) as optimizers to solve optimization problems described in natural language. The core problem OPRO addresses is the challenge of derivative-free optimization in discrete or complex search spaces—such as prompt engineering—where traditional gradient-based algorithms are inapplicable.

### Methodology
OPRO treats the LLM as a black-box optimizer that iteratively generates new solutions based on a "meta-prompt." The process follows these steps:

1.  **Meta-Prompt Construction**: The optimizer LLM is provided with a meta-prompt containing three components:
    *   **Optimization Problem Description**: A natural language description of the objective function, constraints, and (for prompt optimization) task exemplars.
    *   **Optimization Trajectory**: A history of previously generated solutions and their corresponding scores, sorted in ascending order of performance.
    *   **Meta-Instructions**: Instructions guiding the LLM to generate a new solution that outperforms the existing trajectory.
2.  **Solution Generation**: The LLM generates a batch of candidate solutions. To ensure stability and balance the exploration-exploitation trade-off, OPRO generates multiple solutions per step (e.g., 8) and tunes the sampling temperature (default $\text{temp} = 1.0$).
3.  **Evaluation**: A separate "scorer" (either a mathematical function or another LLM) evaluates the candidates.
4.  **Iteration**: The new solutions and their scores are appended to the optimization trajectory, and the process repeats until convergence or a maximum step limit is reached.

### Mathematical Optimization Case Studies
The authors demonstrate OPRO on two classic problems:
*   **Linear Regression**: In a one-dimensional setting ($y = wx + b + \epsilon$), OPRO successfully identified descent directions to find global optima. GPT-4 demonstrated the highest efficiency in proposing reasonable next steps from history.
*   **Traveling Salesman Problem (TSP)**: OPRO was compared against Nearest Neighbor (NN) and Farthest Insertion (FI) heuristics. The FI heuristic's minimal insertion cost is defined as:

$$
c(k) = \min_{(i,j)} d(i,k) + d(k,j) - d(i,j)
$$

    where $i$ and $j$ are adjacent nodes in the current tour. While GPT-4 reached global optima for small-scale problems ($n=10$) significantly faster than other LLMs, OPRO's performance degraded as $n$ increased, with FI eventually outperforming the LLMs.

### Prompt Optimization Results
The primary application of OPRO is finding instructions that maximize task accuracy. The framework utilizes a "scorer LLM" to evaluate prompts on a small training subset (e.g., 3.5% of GSM8K).

*   **GSM8K**: OPRO-optimized prompts outperformed human-designed baselines by up to 8%. For example, using PaLM 2-L-IT as the optimizer and PaLM 2-L as the scorer, OPRO found the instruction *"Take a deep breath and work on this problem step-by-step,"* achieving **80.2%** accuracy, compared to **71.8%** for the baseline *"Let’s think step by step."*
*   **Big-Bench Hard (BBH)**: OPRO-optimized prompts outperformed human-designed prompts by up to **50%**. On 19 of 23 tasks, OPRO improved performance by over 5% compared to the "Let's think step by step" baseline.

### Limitations
The authors identify several technical limitations:
*   **Context Window**: The LLM's context limit restricts the scale of problems (e.g., high-dimensional regression or TSP with many nodes) that can be described in the meta-prompt.
*   **Loss Landscapes**: "Bumpy" landscapes (e.g., the Rosenbrock function) can cause the LLM to get stuck in local optima or narrow valleys.
*   **Reliability**: Optimizer LLMs occasionally hallucinate mathematical values or fail to follow instructions to generate unique solutions.
*   **Overfitting**: Training accuracies are often 5%–20% higher than test accuracies, though the relative ranking of prompts generally remains consistent.
