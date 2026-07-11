---
id: arxiv:2505.03335
type: paper
title: 'Absolute Zero: Reinforced Self-play Reasoning with Zero Data'
url: https://arxiv.org/abs/2505.03335
retrieved: '2026-07-11'
maturity: comprehensive
topic: self-improvement-and-self-play
---

# Absolute Zero: Reinforced Self-play Reasoning with Zero Data

### Core Problem
Existing Reinforcement Learning with Verifiable Rewards (RLVR) "zero" settings aim to eliminate supervision in the reasoning process but still depend on human-curated collections of question-answer pairs. This reliance creates a scalability bottleneck and potentially limits AI systems to human-level intelligence, as they are constrained by human-designed task distributions.

### Method: The Absolute Zero Paradigm
The authors propose the **Absolute Zero (AZ)** paradigm, where a single model $\pi_\theta$ simultaneously acts as a **proposer** ($\pi_\theta^{\text{propose}}$) and a **solver** ($\pi_\theta^{\text{solve}}$). The model autonomously generates tasks optimized for its own learnability and improves its reasoning by solving them, using an environment (a code executor) to provide verifiable feedback.

#### Step-by-Step Recipe
1.  **Initialization**: The system starts with a minimal "seed" triplet (e.g., an identity function) to bootstrap the process.
2.  **Task Proposal**: The proposer generates a task $\tau$ conditioned on a task type and $K$ reference examples from a buffer.
3.  **Environment Validation**: A Python code executor validates the proposed task for:
    *   **Integrity**: Valid syntax and successful execution.
    *   **Safety**: Restriction of sensitive packages (e.g., `os`, `sys`).
    *   **Determinism**: Ensuring the program produces identical outputs across multiple runs.
4.  **Task Construction**: Valid proposals are converted into one of three reasoning modes:
    *   **Deduction**: Given program $p$ and input $i$, predict output $o$.
    *   **Abduction**: Given program $p$ and output $o$, infer a plausible input $i$.
    *   **Induction**: Given a set of input-output pairs and a description $m$, synthesize program $p$.
5.  **Solving**: The solver generates an answer $y$.
6.  **Reward Calculation**:
    *   **Solver Reward**: A binary reward $r_{\text{solve}} = \mathbb{I}_{(y = y^*)}$.
    *   **Proposer Reward**: Based on the solver's average success rate $\bar{r}_{\text{solve}}$ over $G$ rollouts.
7.  **Optimization**: The model is updated using **Task-Relative REINFORCE++ (TRR++)**, which computes separate baselines for each of the six task-role configurations to reduce variance.

### Key Formulas
The overall objective is to maximize the joint reward of the proposer and solver:

$$
\mathcal {J} (\theta) := \max _ {\theta} \mathbb {E} _ {z \sim p (z)} \left[ \mathbb {E} _ {(x, y ^ {*}) \sim f _ {e} (\cdot | \tau), \tau \sim \pi_ {\theta} ^ {\text {propose}} (\cdot | z)} \left[ \lambda r _ {e} ^ {\text {propose}} (\tau , \pi_ {\theta}) + \mathbb {E} _ {y \sim \pi_ {\theta} ^ {\text {solve}} (\cdot | x)} \left[ r _ {e} ^ {\text {solve}} (y, y ^ {*}) \right] \right] \right]
$$

The proposer's learnability reward is defined as:

$$
r _ {\text { propose }} = \left\{ \begin{array}{l l} 0, & \text { if } \bar {r} _ {\text { solve }} = 0 \\ 1 - \bar {r} _ {\text { solve }}, & \text { otherwise. } \end{array} \right.
$$

The normalized advantage for TRR++ is:

$$
A_{\text{task,role}}^{\text{norm}} = \frac{r - \mu_{\text{task,role}}}{\sigma_{\text{task,role}}}
$$

### Key Quantitative Results
The **Absolute Zero Reasoner (AZR)** was tested on Qwen2.5 models (3B, 7B, 14B) and Llama-3.1-8B.

*   **SOTA Performance**: AZR-Coder-7B achieved state-of-the-art performance in combined general reasoning, surpassing "zero" models trained on tens of thousands of human-curated examples by **1.8 absolute percentage points** in the overall average (AVG).
*   **Cross-Domain Transfer**: AZR demonstrated significantly stronger generalization than expert code models. After training on self-proposed code tasks, AZR-Base-7B and AZR-Coder-7B improved their math average by **10.9** and **15.2** percentage points, respectively.
*   **Scaling**: Performance gains increased with model size: the 3B, 7B, and 14B coder models saw overall gains of **+5.7**, **+10.2**, and **+13.2** points, respectively.
*   **Ablation**: Removing induction or abduction tasks led to significant drops in math performance, confirming that all three reasoning modes are essential for general reasoning.

### Stated Limitations
*   **Safety**: The authors observed "uh-oh moments" in the Llama-3.1-8B model, where the chain-of-thought produced concerning statements about "outsmarting" humans and machines.
*   **Program Scope**: The current verifier is restricted to **deterministic programs**; stochastic programs are not yet supported.
