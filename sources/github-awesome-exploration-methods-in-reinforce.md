---
id: github:awesome-exploration-methods-in-reinforce
type: web
title: Awesome Exploration Methods in Reinforcement Learning (GitHub)
url: https://github.com/opendilab/awesome-exploration-rl
retrieved: '2026-07-11'
maturity: comprehensive
topic: entropy-and-exploration
---

# Summary: Awesome Exploration Methods in Reinforcement Learning

This source is a curated research repository and taxonomy designed to track the frontier of **Exploration methods in Reinforcement Learning (ERL)**. Rather than presenting a single algorithm, it provides a structured framework for categorizing the diverse strategies used to solve the exploration-exploitation trade-off.

### Core Problem
The central challenge addressed is the balance between **exploration** (searching the state-action space for new rewards) and **exploitation** (leveraging known high-reward paths). This is particularly critical in "hard exploration" environments—exemplified by `MiniGrid-ObstructedMaze-Full-v0`—where achieving a goal may require dozens or hundreds of specific sequential actions, necessitating a systematic approach to explore the state-action space to learn the required skills.

### Taxonomy of Exploration RL Methods
The repository organizes ERL methods based on the phase of the reinforcement learning process in which the exploration component is applied: the **collect phase** (interacting with the environment to gather experience) and the **train phase** (updating the policy based on collected experience).

#### 1. Augmented Collecting Strategy
These strategies are applied during the collect phase to influence how the agent interacts with the environment. They are divided into four categories:
*   **Action Selection Perturbation:** Introducing noise or randomness into action choices.
*   **Action Selection Guidance:** Using specific heuristics or signals to guide action choices.
*   **State Selection Guidance:** Directing the agent toward specific, high-value, or novel states.
*   **Parameter Space Perturbation:** Adding noise directly to the policy parameters (e.g., NoisyNet).

#### 2. Augmented Training Strategy
These strategies are applied during the train phase to modify the learning objective or the way experience is processed. They are divided into seven categories:
*   **Count Based:** Using visit counts to encourage exploration of rarely visited states.
*   **Prediction Based:** Utilizing prediction errors (e.g., RND, ICM) as intrinsic rewards.
*   **Information Theory Based:** Leveraging information-theoretic measures (e.g., empowerment).
*   **Entropy Augmented:** Adding entropy terms to the objective to maintain policy diversity (e.g., SAC).
*   **Bayesian Posterior Based:** Using posterior sampling or uncertainty estimates (e.g., PSRL).
*   **Goal Based:** Utilizing goal-conditioned RL or hindsight experience (e.g., HER).
*   **(Expert) Demo Data:** Incorporating expert demonstrations to guide the agent (e.g., DQfD).

### Key Research Trends (2021–2026)
The repository tracks a vast array of recent papers from ICML, ICLR, and NeurIPS. Recent trends highlighted in the 2025–2026 listings include:
*   **LLM-Based Exploration:** Research into "Exploration Hacking" in LLMs, rubric-scaffolded RL for reasoning, and latent exploration decoding for large reasoning models.
*   **Advanced Intrinsic Motivation:** Use of diffusion models for unsupervised RL, spectral Bellman methods, and graph-theoretic intrinsic rewards based on effective resistance.
*   **Multi-Agent and Embodied RL:** Exploration in decentralized multi-agent RL (MARL) and covariance volume maximization for embodied latent exploration.

### Quantitative Results and Formulas
As this source is a curated bibliography and taxonomy, it does not provide specific mathematical formulas or aggregate quantitative results. It serves as a pointer to the primary literature where such data is located.

### Stated Limitations
The authors explicitly state that the provided taxonomy is **non-exhaustive**. Additionally, they note that there is often overlap between the categories, and a single algorithm may belong to several different categories simultaneously.
