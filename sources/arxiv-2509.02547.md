---
id: arxiv:2509.02547
type: paper
title: 'The Landscape of Agentic Reinforcement Learning for LLMs: A Survey'
url: https://arxiv.org/abs/2509.02547
retrieved: '2026-07-12'
maturity: comprehensive
topic: agentic-and-tool-use-rl
---

# Summary: The Landscape of Agentic Reinforcement Learning for LLMs

### Core Problem
The authors identify a paradigm shift from conventional **LLM RL** (treating models as passive sequence generators optimized for single-turn alignment) to **Agentic Reinforcement Learning (Agentic RL)**. The core problem is the lack of a unified framework that conceptualizes LLMs as learnable policies embedded in sequential decision-making loops. While previous research focused on isolated capabilities or static benchmarks, Agentic RL seeks to transform LLMs into autonomous agents capable of long-horizon cognitive behaviors in partially observable, dynamic environments.

### Formalization and Method
The survey formalizes the shift from Preference-Based Reinforcement Fine-Tuning (PBRFT) to Agentic RL by contrasting their underlying Markov Decision Process (MDP) structures.

**1. PBRFT (Degenerate MDP):**
PBRFT is modeled as a single-step decision problem with a horizon $T=1$ and discount factor $\gamma=1$:

$$
\langle S_{trad}, A_{trad}, P_{trad}, R_{trad}, T=1, \gamma=1 \rangle
$$

The objective is to maximize the expected reward of a single response: $J_{trad}(\theta) = \mathbb{E}_{a \sim \pi_\theta}[r(a)]$.

**2. Agentic RL (POMDP):**
Agentic RL is modeled as a Partially Observable Markov Decision Process (POMDP):

$$
\langle S_{agent}, A_{agent}, P_{agent}, R_{agent}, \gamma, O \rangle
$$

Where the agent receives observations $o_t = O(s_t)$ and the action space is split into natural language and structured actions: $\mathcal{A}_{agent} = \mathcal{A}_{text} \cup \mathcal{A}_{action}$. The objective is to maximize the discounted cumulative reward:

$$
J_{agent}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \gamma^t R_{agent}(s_t, a_t) \right], \quad 0 < \gamma < 1
$$

**RL Algorithmic Recipe:**
The survey categorizes the optimization mechanisms into four primary families:
*   **REINFORCE:** Foundational policy gradient using a baseline $b(s)$ to reduce variance.
*   **PPO:** Uses a clipped objective to prevent destructively large policy updates.
*   **DPO:** Bypasses explicit reward models by optimizing a likelihood-based objective on preference data.
*   **GRPO:** Eliminates the critic network by using group-relative rewards to estimate advantages.

### Capability Taxonomy
Agentic RL empowers LLMs through six core optimizable modules:
1.  **Planning:** Either as an *external guide* (training a value function to guide search like MCTS) or an *internal driver* (directly refining the LLM's internal policy).
2.  **Tool Use:** Evolution from ReAct-style prompting to Tool-Integrated Reasoning (TIR), where RL optimizes the timing and fidelity of tool invocations.
3.  **Memory:** Transitioning from static RAG stores to RL-controlled subsystems that manage explicit (natural language) or implicit (latent) tokens.
4.  **Self-Improvement:** Moving from verbal self-correction (prompt-based) to internalized self-correction (gradient-based) and iterative self-training (self-play/curriculum generation).
5.  **Reasoning:** Balancing "Fast Reasoning" (System 1: intuitive/pattern-driven) with "Slow Reasoning" (System 2: deliberate/structured traces).
6.  **Perception:** Shifting from passive perception to active visual cognition via grounding-driven and tool-driven active perception.

### Key Quantitative Results
The survey highlights the performance gap between open-source and closed-source research agents. Specifically, on the **BrowseComp** benchmark (which measures the ability to locate hard-to-find information), **OpenAI Deep Research** achieves a **51.5% pass@1** rate, significantly outperforming most open-source baselines.

### Stated Limitations
*   **Temporal Credit Assignment:** In long-horizon tasks, sparse and delayed rewards make it difficult to identify which specific action in a sequence led to success or failure.
*   **Reasoning Trade-offs:** Slow reasoning models are susceptible to "overthinking," resulting in excessive latency or unnecessarily long chains of thought.
*   **Training Stability:** Live web-search training is hindered by high API costs and the instability caused by noisy environmental information.
*   **Static Reflection:** Current reflection mechanisms are largely handcrafted; there is a need for "meta-evolution" where agents learn how to self-correct more effectively over time.
