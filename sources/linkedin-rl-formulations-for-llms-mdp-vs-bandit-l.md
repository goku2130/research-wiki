---
id: linkedin:rl-formulations-for-llms-mdp-vs-bandit-l
type: web
title: 'RL formulations for LLMs: MDP vs Bandit - LinkedIn'
url: https://www.linkedin.com/posts/cwolferesearch_there-are-two-common-rl-training-formulations-activity-7380972876264497153-J2gt
retrieved: '2026-07-11'
maturity: comprehensive
topic: mdp-formulation
---

The author discusses two common Reinforcement Learning (RL) training formulations for Large Language Models (LLMs): the Markov Decision Process (MDP) formulation and the bandit formulation.

**Core Problem:**
LLMs generate output token by token, which can be framed as a sequential decision-making process. The challenge lies in effectively applying RL to this process, especially considering how rewards are typically assigned to LLM outputs.

**Method/Recipe Step by Step:**

1.  **LLM Output Generation as RL Setup:**
    *   An LLM generates output by predicting tokens sequentially.
    *   Each token prediction can be modeled as an individual action in an RL framework.

2.  **MDP Formulation:**
    *   **Definition:** Each token in the LLM's output is modeled as an individual action within a Markov Decision Process.
    *   **Components:**
        *   **States:** The initial state is the prompt. As tokens are predicted, they are added to the current state, which then informs the prediction of the next token (autoregressive next token prediction).
        *   **Actions:** Selecting a token from the LLM's predicted token distribution. Each token is its own action.
        *   **Policy:** Acts by predicting tokens.
        *   **Transition Probabilities:** Implicit in the LLM's autoregressive generation process.
        *   **Rewards:** In a pure MDP, a reward would be provided at every step (after each token prediction).
    *   **Trajectory:** The process continues until the LLM predicts a stop token (e.g., `<|end_of_text|>` or `<eos>`), yielding a full trajectory.
    *   **Outcome vs. Process Reward:** While MDPs typically imply step-wise rewards, LLMs are usually trained with "outcome supervision," meaning a reward is assigned only *after* the complete response is generated (i.e., after the `<eos>` token).

3.  **Bandit Formulation:**
    *   **Definition:** The entire LLM response (completion) for a given prompt is modeled as a single action that receives an outcome reward. This formulation is inspired by the contextual bandit problem in probability theory.
    *   **Components:**
        *   **Agent's Choice:** The agent chooses a single "action."
        *   **Action for LLMs:** The full completion generated for a prompt.
        *   **Reward:** An outcome reward is received after the single action (full completion) is executed.
        *   **Episode End:** The episode ends after this single action and reward.
    *   **Credit Assignment:** This formulation addresses the credit assignment problem in outcome supervision settings by treating the entire output as the unit of action.

**Key Formulas (Implicit, not explicitly stated in LaTeX):**
The text describes the *structure* of the formulations rather than specific mathematical equations.
*   **MDP:** Involves states, actions, transition probabilities, and rewards. The underlying mathematical framework would typically involve Bellman equations or policy gradient formulations, but these are not provided.
*   **Bandit:** Involves an agent choosing an action and receiving a reward. The underlying mathematical framework would typically involve maximizing expected reward for a single action, but specific equations are not provided.

**Key Quantitative Results and Numbers:**
No specific quantitative results or numbers are provided in the text.

**Stated Limitations:**
*   **MDP Formulation with Outcome Rewards:** The primary challenge for the MDP formulation in LLMs is that rewards are typically only available at the end of a generation (outcome supervision), rather than at each token generation step. This complicates credit assignment for individual token actions.
*   **Simplicity vs. Complexity:** While the bandit formulation is simpler and often fitting for LLMs due to outcome rewards, MDP formulations are also commonly used. The text notes that the bandit formulation "has been argued to perform similarly despite this simplicity," implying that its simplicity might be seen as a potential limitation by some, even if it's often effective.

**Examples of Algorithms and Formulations:**
*   **PPO (Proximal Policy Optimization):** Uses an MDP formulation.
*   **REINFORCE:** Adopts a bandit formulation.
*   **RLOO (Reinforcement Learning with Outcome Optimization):** Adopts a bandit formulation.
