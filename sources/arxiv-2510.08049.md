---
id: arxiv:2510.08049
type: paper
title: 'A Survey of Process Reward Models: From Outcome to Process Supervision'
url: https://arxiv.org/html/2510.08049v3
retrieved: '2026-07-12'
maturity: comprehensive
topic: process-vs-outcome-rewards
---

# Summary: A Survey of Process Reward Models

## Core Problem
Conventional alignment for Large Language Model (LLM) reasoning is dominated by **Outcome Reward Models (ORMs)**, which provide a single, coarse signal based solely on the final answer. This approach struggles with long, complex reasoning chains because it cannot diagnose intermediate errors, provide fine-grained credit assignment, or adaptively allocate computation. **Process Reward Models (PRMs)** address this by evaluating and guiding reasoning at the step or trajectory level, transforming reward modeling from a one-shot verdict into an iterative controller.

## The PRM Loop: Method and Recipe
The authors define a closed-loop pipeline for implementing PRMs: **Generate Process Data $\rightarrow$ Build PRMs $\rightarrow$ Use PRMs $\rightarrow$ Produce Better Data**.

### 1. Generating Process Data
*   **Human Annotation:** Direct expert verification of intermediate steps (e.g., PRM800K).
*   **Automated Supervision:** Using symbolic tools, consistency checks, execution feedback, or Monte Carlo Tree Search (MCTS) to identify errors (e.g., Math-Shepherd, OmegaPRM).
*   **Semi-automated/Hybrid:** Combining small-scale human-curated "anchors" with automated expansion to balance fidelity and scalability.

### 2. Building PRMs
The survey categorizes PRM architectures into four types:
*   **Discriminative PRMs:** Learn a scoring function $f_\theta$ to predict per-step correctness.
*   **Generative PRMs:** A two-stage "think-then-judge" process where the model generates a critique chain $z_t$ before scoring.
*   **Implicit PRMs:** Infer rewards from indirect signals (e.g., outcome feedback) without explicit step labels.
*   **Other Architectures:** Including graph-based, retrieval-augmented, and hierarchical structures.

### 3. Using PRMs
*   **Test-Time Scaling:** Allocating compute during inference via Best-of-N re-ranking, verification-guided decoding, or guided search (e.g., beam annealing, MCTS).
*   **RL for Policy Learning:** Using PRMs as dense, step-level rewards in reinforcement learning (e.g., PPO) to stabilize credit assignment and accelerate policy convergence.

## Key Formulas
**Discriminative Scoring:**
The reward $r_t$ for a partial solution $s_{1:t}$ given input $x$ is:

$$
r_t = \sigma(f_\theta(x, s_{1:t})) \in (0, 1)
$$

**Loss Functions:**
Pointwise Binary Cross-Entropy (BCE):

$$
\mathcal{L}_{\text{point}}^{\text{BCE}} = \mathbb{E} [ - y_t \log r_t - (1 - y_t) \log (1 - r_t) ]
$$

Pointwise Mean Squared Error (MSE):

$$
\mathcal{L}_{\text{point}}^{\text{MSE}} = \mathbb{E} [ (r_t - y_t)^2 ]
$$

Pairwise Preference Loss:

$$
\mathbb{P}_\theta(u \succ v) = \sigma(f_\theta(u) - f_\theta(v)) \implies \mathcal{L}_{\text{pair}} = \mathbb{E} [ - \log \mathbb{P}_\theta(u \succ v) ]
$$

**Generative PRM Logic:**
The model generates a critique $z_t \sim p_\phi(z_t \mid x, s_{1:t})$ and scores it via $r_t = h_\psi(x, s_{1:t}, z_t)$. The joint training objective is:

$$
\mathcal{L}_{\text{gen}} = - \log p_\phi(z_t^\star \mid x, s_{1:t}) + \lambda \text{BCE}(r_t, y_t)
$$

## Key Quantitative Results
*   **Benchmark Performance:** On **Processbench**, `o1-mini` achieved the highest average score (87.9), followed by `ACTPRM-X` (76.0). On **PRMBench**, `o1-mini` (68.8) and `RetrievalPRM-7B` (68.9) showed the highest overall performance.
*   **Annotation Scale:** Benchmarks like Processbench and PRMBench contain an average of 7.1 and 13.4 steps per problem, respectively, illustrating the increased workload compared to ORMs.
*   **Stability Analysis:** Analysis of the PRM800K dataset revealed that the standard deviation ($SD$) of token length at the step level (71.7) is significantly higher than the trajectory-level $SD$ (50.6), indicating that step-level signals are noisier.

## Stated Limitations
*   **Resource Efficiency:** PRMs are significantly more expensive to train than ORMs due to the requirement for step-wise labels.
*   **Anti-Hacking Robustness:** PRMs are more susceptible to "length hacking" or verbosity bias than ORMs.
*   **Generalization:** PRMs have limited cross-domain generalization because step definitions (e.g., math vs. code) vary fundamentally.
*   **Cognitive Scalability:** Human annotators may create a "human ceiling" when models surpass human reasoning capabilities.
*   **Automated Supervision Risks:** Automated pipelines can create "Echo Chambers" where models reinforce shared misconceptions or succumb to Goodhart's Law.
*   **Granularity Tension:** Rigid segmentation (e.g., splitting by newlines) often conflicts with the natural semantic flow of reasoning.
*   **Proxy-Reward Gap:** High accuracy on static benchmarks does not always translate to effective navigation during dynamic inference.
