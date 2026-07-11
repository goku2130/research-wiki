---
id: arxiv:2510.09988
type: paper
title: Unifying Tree Search Algorithm and Reward Design for LLM Reasoning
url: https://arxiv.org/abs/2510.09988
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

# Summary: Unifying Tree Search Algorithm and Reward Design for LLM Reasoning

### Core Problem
The research landscape for deliberative tree search in Large Language Models (LLMs) is currently fragmented, lacking a common mathematical formalism and consistent notation. A primary conceptual ambiguity exists regarding the role of the reward signal: whether it serves as a transient heuristic for **Test-Time Scaling (TTS)**—allocating auxiliary computation during inference—or as a durable learning target for **Self-Improvement**, where search-generated data is used to update model parameters.

### Proposed Framework and Method
The authors introduce a unified framework that deconstructs search-based reasoning into three core components: the **Search Mechanism**, the **Reward Formulation**, and the **Transition Function**. 

#### 1. The Dual-Optimization Paradigm
The framework distinguishes between two orthogonal scaling paths:
*   **Training-Time Scaling:** Optimizes in the latent parameter space $\Theta \subseteq \mathbb{R}^N$ using gradients $\nabla_{\theta}\mathcal{L}$ to create a static policy $\pi$.
*   **Test-Time Scaling (TTS):** Optimizes in the task-defined objective space $\mathcal{P}(Q)$ to find an optimal reasoning trace $p^*$ for a specific problem $Q$:

$$
p^* = \arg \max_{p \in \mathcal{A}(\pi, Q, \mathcal{C}_{\text{infer}})} V(p)
$$

    where $\mathcal{A}$ is the search algorithm and $\mathcal{C}_{\text{infer}}$ is the compute budget.

#### 2. Reward as a Unified Signal
The authors reconcile Reinforcement Learning (RL) and Search by treating them as two optimizers for one objective:
*   **Internalization (RL):** Rewards are used to shape a durable policy $\theta^*$ to maximize an objective $J_{RL}$ while maintaining alignment via KL divergence:

$$
\theta^* = \arg \max_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}} [G(\tau)] - \lambda \int_{s \in \tau} D_{KL}(\pi_{\theta}(\cdot|s) \| \pi_{\mathcal{P}}(\cdot|s)) ds
$$

*   **Externalization (Search):** Rewards act as ephemeral guides for a specific plan $p^*$:

$$
p^* = \arg \max_{p \in \mathcal{P}_{\text{plan}}} \left[ \sum_{t=0}^{T-1} \gamma^t R_{\text{ext}}(s_t, a_t) + \mathcal{H}_{\theta}(s_T, p) \right]
$$

#### 3. Joint Space Optimization
The authors propose that true test-time scaling requires joint optimization over two spaces:
*   **Prompt Space ($\mathcal{P}$):** Searching for the optimal reasoning structure or "algorithm" (template $p$).
*   **Answer Space ($\mathcal{S}$):** Searching for the correct solution trace $s$ given a template $p$.
The ultimate objective is: $s^* = \arg \max_{p \in \mathcal{P}, s \in \mathcal{S}_p} V(s)$.

### MCTS Implementation Recipe
The survey focuses heavily on Monte Carlo Tree Search (MCTS), which operates through four phases:
1.  **Selection:** Navigating the tree using the Upper Confidence bounds applied to Trees (UCT) policy:

$$
a = \arg \max_{a \in A(s)} \left( Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a)}} \right)
$$

2.  **Expansion:** Adding new nodes sampled by the LLM.
3.  **Simulation:** Performing rollouts to estimate long-term rewards.
4.  **Backpropagation:** Updating the quality values $Q$ and visit counts $N$ of ancestor nodes.

### Key Quantitative Results
While the survey synthesizes numerous models, it focuses on qualitative architectural comparisons rather than a single benchmark table. It notes that:
*   **ReST-MCTS** achieves superior performance on MATH, GPQA, and CEval by using MCTS to generate high-quality traces for iterative training.
*   **RAP** improves efficiency and accuracy on Blocksworld and logical reasoning tasks by using the LLM as both a policy and a world model.
*   **PG-TD** shows significant improvements in code correctness on APPS and CodeContests by using test case execution as the reward signal.

### Stated Limitations
*   **State Space Complexity:** LLM reasoning spaces are combinatorially vast and high-dimensional, making exhaustive exploration computationally infeasible.
*   **Heuristic Trade-offs:** There is a constant tension between the computational cost of calculating a heuristic $h(n)$ and the resulting search efficiency.
*   **Prompt Space Gap:** Most current methods fix a single prompt template (searching only the Answer Space), failing to address the meta-level challenge of discovering the most effective reasoning algorithm for a given task.
