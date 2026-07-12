---
id: huggingface:is-rlhf-a-sequence-level-or-token-level-
type: web
title: Is RLHF a Sequence-level or Token-level Problem? (RL-LLM Wiki)
url: https://huggingface.co/datasets/rl-llm-wiki/knowledge-base/blob/main/topics/foundations/mdp-formulation.md
retrieved: '2026-07-12'
maturity: comprehensive
topic: mdp-formulation
---

# The MDP Formulation of LLM Generation

### Core Problem
The central challenge is determining the most effective way to cast Large Language Model (LLM) text generation as a sequential decision process for Reinforcement Learning (RL). Specifically, the source explores whether RLHF (Reinforcement Learning from Human Feedback) is best modeled as a **token-level Markov Decision Process (MDP)** or a **sequence-level contextual bandit**. The "degenerate" nature of LLM generation—characterized by deterministic transitions and typically terminal rewards—often renders complex RL machinery (like long-horizon credit assignment) idle, making the sequence-level bandit view a sufficient and often more efficient approximation.

### MDP Formulation
The process of autoregressive generation for a fixed prompt $x$ is mapped to an MDP as follows:

*   **State ($s_t$):** The prompt plus all tokens generated up to that point: $s_t = (x, a_0, \dots, a_{t-1})$.
*   **Action ($a_t$):** The next token selected from the vocabulary $\mathcal{V}$.
*   **Policy ($\pi_\theta(a_t \mid s_t)$):** The language model's probability distribution for the next token.
*   **Transition:** Deterministic; the next state is simply the current state concatenated with the chosen action: $s_{t+1} = s_t \oplus a_t$.
*   **Reward ($r_t$):** Typically $0$ for all $t < T$, with a single scalar reward provided by a reward model or verifier upon completion of the sequence $(x, y)$.
*   **Episode:** A complete generated sequence $y = (a_0, \dots, a_{T-1})$, terminating at an EOS token or a length cap.

### The Sequence-Level Bandit Collapse
When transitions are deterministic and the reward is terminal, the MDP collapses into a **contextual bandit**. In this view, the model samples a prompt (context), emits a full response (a single "arm" from a massive action space), and receives one scalar reward. 

The source notes that under a terminal reward with a discount factor $\gamma=1$, the token-level policy gradient (using return-to-go) and the sequence-level bandit gradient are mathematically identical vectors.

### Divergence from the Bandit View
The token-level MDP view remains "load-bearing" in two specific scenarios where the reward signal becomes dense:
1.  **Per-token KL Penalty:** To prevent reward hacking, a dense penalty is added at every step:

$$
-\beta \log \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)}
$$

2.  **Process Rewards:** When a process reward model (PRM) scores intermediate reasoning steps, injecting rewards at step boundaries rather than only at the end.

### Algorithmic Consequences
The degenerate structure of the LLM MDP drives several specific design choices in RLHF algorithms:
*   **Discounting:** Because episodes are short and rewards are terminal, discounting future rewards would only down-weight the primary signal. Consequently, models like InstructGPT use **no discounting** ($\gamma=1$).
*   **GAE (Generalized Advantage Estimation):** The long-horizon machinery of GAE is largely idle because there is little long-range structure to exploit when the reward is terminal.
*   **Critic-Free Methods (e.g., GRPO):** Since estimating a per-token value function is difficult and provides limited benefit for terminal rewards, GRPO removes the value network entirely. Instead, it uses the **mean reward of a group of sampled responses** as a Monte-Carlo baseline.
*   **Normalization:** Different recipes handle accounting differently; for example, DeepSeek-R1 uses a per-output objective, whereas the original GRPO used per-token length normalization.

### Stated Limitations and Open Questions
The source identifies several unresolved theoretical and practical questions:
*   **Value Modeling:** It remains unknown if modeling intermediate state value genuinely provides benefits beyond what is achieved by process rewards.
*   **Credit Assignment:** With a terminal reward and an action space of $10^4$–$10^5$ tokens, the optimal method for per-token credit assignment (uniform broadcast vs. learned value functions vs. process rewards) is unsettled.
*   **Optimization:** It is unclear if there are optimization methods that can specifically exploit the deterministic-transition structure of LLMs, rather than relying on generic stochastic-dynamics RL machinery.
