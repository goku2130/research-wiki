---
id: nature:olympiad-level-formal-mathematical-reaso
type: web
title: Olympiad-level formal mathematical reasoning with reinforcement learning (AlphaProof)
url: https://www.nature.com/articles/s41586-025-09833-y
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

# AlphaProof: Olympiad-Level Formal Mathematical Reasoning

AlphaProof is a reinforcement learning (RL) agent designed to solve complex mathematical problems by discovering formal proofs within the Lean theorem prover environment.

### Core Problem
The primary challenge addressed is the lack of formal verification in large language models (LLMs) trained on informal mathematical text. While LLMs can generate human-like mathematical discourse, they cannot guarantee the correctness of their reasoning. AlphaProof aims to bridge this gap by combining the experiential learning of RL with the rigorous, automated verification provided by formal languages like Lean, enabling the AI to tackle elite mathematics competition problems that require deep, multi-step reasoning.

### Method and Recipe
AlphaProof treats formal theorem proving as a sequential decision-making problem.

**1. The Lean RL Environment**
*   **State ($s_t$):** The current Lean tactic state, consisting of established hypotheses and remaining goals.
*   **Action ($a_t$):** A Lean tactic proposed as a text string.
*   **Reward ($r_t$):** To incentivize brevity, the agent receives $r_t = -1$ for every tactic applied.
*   **Return ($G_t$):** The sum of rewards until termination. For proof states that decompose into multiple independent subgoals, the return is defined as the minimum return over those subgoals:

$$
G_t = \min(\text{returns of subgoals})
$$

**2. Prover Agent Architecture**
The agent utilizes a **proof network**, a 3-billion-parameter encoder-decoder transformer. This network produces two outputs:
*   **Policy:** A list of $N$ promising tactics to apply.
*   **Value Function:** An estimate of the expected return $G_t$ for the current state.

These outputs guide a specialized **tree search** using an AND-OR tree structure to handle subgoal decomposition. The search incorporates progressive sampling and action sampling to explore a broader range of strategies.

**3. Training Pipeline**
*   **Pretraining:** The network is pretrained on $\approx 300$ billion tokens of code and mathematical text using next-token prediction.
*   **Supervised Fine-Tuning (SFT):** The model is fine-tuned on $\approx 300,000$ state-tactic pairs extracted from the Mathlib library to learn Lean syntax and expert tactics.
*   **Main RL Loop:** To overcome the scarcity of formal data, a Gemini-based LLM auto-formalizes $\approx 1$ million natural-language problems into a dataset of $\approx 80$ million formal Lean problems. Distributed actors use tree search to prove or disprove these statements; Lean-verified outcomes provide the grounded feedback used to iteratively update the proof network's policy and value function.

**4. Inference and Test-Time RL (TTRL)**
For exceptionally difficult problems, AlphaProof employs **test-time RL**. This involves generating millions of related formal problem variants to create a bespoke curriculum. The agent then undergoes a focused RL process on these variants to adapt its reasoning strategies to the specific structures of the target problem.

### Key Quantitative Results
*   **IMO 2024 Performance:** AlphaProof solved three out of five non-geometry problems, including the competition's most difficult problem.
*   **Overall Standing:** When combined with AlphaGeometry 2, the system achieved a score equivalent to a **silver medallist** at the International Mathematical Olympiad (IMO), the first time an AI has reached medal-level performance.
*   **Data Scale:** The system utilized a 3B parameter model, 300B pretraining tokens, and a formal training set of 80 million problems.

### Limitations
The authors state that the computational time required for AlphaProof to find solutions far exceeds the time allowed for human contestants during the competition.
