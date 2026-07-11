---
id: arxiv:2506.03136
type: paper
title: Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning
url: https://arxiv.org/html/2506.03136v2
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-math-and-code
---

# Summary: Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning

### Core Problem
Training Large Language Models (LLMs) to generate unit tests typically requires supervision from ground-truth code solutions. However, collecting such data is costly and labor-intensive, limiting the scalability and diversity of training sets. The authors investigate whether a code generator and a unit test generator can co-evolve through reinforcement learning (RL) without access to ground-truth code solutions, using their mutual interaction outcomes as a reward signal.

### Method: The CURE Framework
The authors propose **CURE**, a co-evolutionary RL framework where a single policy LLM acts as both a coder and a unit tester. The process follows these steps:

1.  **Sample Generation**: For a given task $q$, the model generates $n$ candidate code solutions $s_j$ and $m$ task-derived unit tests $u_k$.
2.  **Execution Matrix**: The $n$ solutions are executed against the $m$ generated tests and $t_q$ ground-truth unit tests, resulting in a binary evaluation matrix $\mathcal{B}^\star \in \{0,1\}^{n \times (m+t_q)}$.
3.  **Solution Reward**: A solution $s_j$ is rewarded based on how many ground-truth unit tests it passes:

$$
\mathcal{R}_{s_{j}}^{\star}=\sum_{l=1}^{t_{q}}\mathcal{B}_{j,m+l}^{\star}
$$

4.  **Unit Test Reward (Reward Precision)**: To avoid "naive" tests that are overly permissive, the framework optimizes for **reward precision** ($\mu$). The reward for a generated unit test $u_k$ is estimated by its ability to distinguish correct solutions (those passing all ground-truth tests) from incorrect ones:

$$
\mathcal{R}_{u_{k}}^{\star}=-\sum_{l=1}^{n}(1-\mathcal{I}_{s_{l}})\mathcal{B}_{l,k}^{\star}+\left(\prod_{l=1}^{n}\mathcal{I}_{s_{l}}\mathcal{B}_{l,k}^{\star}\right)\left(\sum_{l=1}^{n}(1-\mathcal{I}_{s_{l}})\right)
$$

    where $\mathcal{I}_{s_{j}}=\prod_{l=1}^{t_{q}}\mathcal{B}_{j,m+l}^{\star}$ is an indicator of solution correctness.
5.  **Iterative Optimization**: The policy $\pi_\theta$ is optimized using a PPO-style objective $\mathcal{J}(\theta, \{o_i\}_{i=1}^G)$ that incorporates a KL divergence penalty to maintain stability.
6.  **Long-CoT Efficiency**: For long-chain-of-thought (long-CoT) models, a response-length-guided reward transformation is applied to penalize overly long responses, reducing inference latency while preserving reward signals.

### Key Quantitative Results
The authors developed the **ReasonFlux-Coder** series (4B, 7B, and 14B). Key findings include:

*   **Performance Gains**: ReasonFlux-Coder-7B and 14B (optimized from Qwen2.5-Instruct) improved one-shot code generation accuracy by **5.3%** and Best-of-N (BoN) accuracy by **9.0%**, outperforming Qwen-Coder, DeepSeek-Coder, and Seed-Coder.
*   **Long-CoT Efficiency**: ReasonFlux-Coder-4B consistently outperformed Qwen3-4B. The response-length-guided transformation reduced the average response length for unit test generation to **64.8%** of the original length.
*   **Downstream Applications**:
    *   **Agentic Coding**: Achieved an **8.1%** improvement over the base model when integrated into pipelines like AlphaCodium and S*.
    *   **Agentic Unit Test Generation**: Improved unit test accuracy by **25.1%** over the base model.
    *   **API Scaling**: Using ReasonFlux-Coder-4B as a unit tester for GPT-4o improved one-shot performance by **7.0%** while reducing API costs compared to scaling GPT-4o-mini alone.
*   **Label-free RL**: The model can serve as an effective reward model for RL on base models, achieving improvements comparable to those using ground-truth labeled supervision.

### Stated Limitations
The authors note that while standard models benefit significantly from BoN scaling, long-CoT models gain less from such strategies because they have already captured much of the benefit through internal reasoning scaling. Additionally, while the framework eliminates the need for ground-truth *code*, it still utilizes ground-truth *unit tests* to define the correctness indicator $\mathcal{I}_{s_{j}}$.
