---
id: emergentmind:outcome-supervised-reward-models
type: web
title: Outcome-Supervised Reward Models
url: https://www.emergentmind.com/topics/outcome-supervised-reward-model-orm
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-modeling
---

Outcome-Supervised Reward Models (ORMs) are functions that assign a scalar score to complete output trajectories of Large Language Models (LLMs) based solely on final, verifiable outcomes. This approach contrasts with process-supervised reward models (PRMs) which require dense, stepwise annotations. ORMs are applied across various domains, including reasoning, code generation, SQL synthesis, and multi-agent tasks, to enhance reranking and evaluation efficiency.

**Core Problem:**
The core problem ORMs address is the need for efficient and robust reward modeling for LLM outputs without relying on costly and impractical dense, stepwise annotations. Traditional PRMs demand fine-grained process-level feedback, which is often difficult and expensive to obtain, especially for complex tasks.

**Method/Recipe Step by Step:**

1.  **Define the ORM Function:** An ORM is defined as a scalar function, $r_\theta(x,y) \in \mathbb{R}$, that maps a prompt-response pair $(x,y)$ to a real-valued score or reward.
2.  **Gather Outcome Supervision Data:** Training data consists of verifiable outcomes such as binary correctness labels, execution success/failure, or human preference comparisons over full responses.
3.  **Choose a Training Objective:**
    *   **Binary Cross-Entropy (CE)/Classification:** For binary-labeled examples $l \in \{0,1\}$, the ORM predicts the probability $\sigma(r_\theta(x,y))$ that $y$ is "correct."
    *   **Bradley–Terry/Pairwise Preference:** For preference data $D = \{(x, y^w, y^l)\}$, the model learns to prefer $y^w$ over $y^l$.
    *   **Outcome-Only Reinforcement Learning:** Reward is assigned only at the end of a trajectory.
4.  **Model Architecture:** ORMs are typically implemented as:
    *   **Classifier Heads:** Placed atop pretrained transformers, operating on entire response sequences. For tasks like code, math, and SQL, a transformer encoder's final state feeds into a classification or regression head.
    *   **Verifier Models:** These map execution traces, chains-of-thought, multi-modal inputs, or agent trajectories to binary or continuous outcome scores.
5.  **Inference Methods:**
    *   **Best-of-N Reranking:** Given $N$ candidate responses, the ORM selects the one with the highest score.
    *   **Verifier Filtering:** The ORM acts as a filter in iterative reflection or self-evolving systems.
    *   **Search Integration:** ORM scores are incorporated into search procedures (e.g., Monte Carlo Tree Search, beam search) at leaf nodes.

**Key Formulas:**

*   **ORM Function:** $r_\theta(x,y) \in \mathbb{R}$
*   **Binary Cross-Entropy Loss:** $\mathcal{L}_{\mathrm{CE}} = -l \log \sigma(r_\theta(x,y)) - (1-l) \log (1 - \sigma(r_\theta(x,y)))$

**Key Quantitative Results and Numbers:**

*   **Mathematical Reasoning:** ORMs show significant gains on complex benchmarks (e.g., MATH: +18% accuracy), with energy-based ORMs achieving state-of-the-art with fewer samples.
*   **Code Generation:** ORPS (Outcome-Refining Process Supervision), a tree-structured ORM, yields +26.9% correctness and +42.2% code efficiency over Chain-of-Thought/outcome-only methods.
*   **SQL Synthesis:** ORM-based Best-of-N reranking improves execution accuracy by 2–5% on BIRD/SPIDER datasets compared to execution-only reranking or majority voting.
*   **Deductive Logical Reasoning:** Echo-augmented ORMs boost test-time accuracy by 5–15 percentage points over majority vote, especially with increased sample size.
*   **Multimodal Mathematical Reasoning:** ORM with step-indexed error diagnosis improves verifier accuracy by 4.5 percentage points.
*   **Multi-Agent Reasoning:** Adding ORM to MASPRM-guided search provides an additional +2–4 percentage points accuracy in multi-agent math tasks.
*   **Computer-Using Agents:** ORM ensembles achieve 89.8% precision and 93.3% Negative Predictive Value (NPV).

**Stated Limitations:**

*   **Coarse-Grained Signal:** ORMs provide feedback only on the final outcome, offering no guidance on the process or steps taken to derive an answer.
*   **Process Blindness:** They penalize sound processes leading to incorrect answers and reward flawed processes that accidentally produce correct outcomes.
*   **Inconsistency for Partial Sequences:** ORMs trained on full outcomes lack score and preference consistency for prefixes, making them suboptimal for reward-guided search or stepwise guidance.
*   **Domain Constraints:** Some ORM variants require executable or verifiable environments (e.g., code), limiting their applicability to purely abstract or interpretive tasks.
*   **Process-Outcome Disconnect:** The lack of process-level feedback necessitates auxiliary methods for fine-grained reasoning improvement.
*   **Generalization:** While effective in specific domains (math, code, logic, SQL), further validation is needed for open-domain generation and tasks with ambiguous ground truth.
