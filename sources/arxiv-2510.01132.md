---
id: arxiv:2510.01132
type: paper
title: A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning
url: https://arxiv.org/html/2510.01132v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: agentic-and-tool-use-rl
---

# Summary: A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning

### Core Problem
Training Large Language Models (LLMs) as autonomous agents in multi-turn environments is challenging due to extended planning horizons, the need for sequential decision-making, and the prevalence of sparse rewards. Existing implementations are fragmented, often conflating tool-augmented single queries with true multi-turn interaction. There is a lack of systematic analysis regarding which design choices—across environment, reward, and policy—are critical for effective agentic reinforcement learning (RL).

### The Multi-turn RL Recipe
The authors propose a co-design recipe based on three interdependent pillars:

1.  **Environment (Curriculum Learning):** Start training on simpler environments with lower spatial and object complexity. Agents develop reusable skills (e.g., state tracking and spatial exploration) in simple settings that generalize to complex, long-horizon tasks.
2.  **Policy (Priors and Stability):** 
    *   **SFT Initialization:** Use a strong Supervised Fine-Tuning (SFT) prior from minimal demonstrations to accelerate convergence.
    *   **Algorithm Selection:** Employ stabilized, biased policy gradient methods (e.g., PPO, GRPO) rather than unbiased estimators (RLOO) or naive gradients (Reinforce++).
    *   **Budget Allocation:** Balance the SFT-to-RL ratio. For a fixed budget, a hybrid approach (e.g., 60 demonstrations followed by 400 RL episodes) optimizes both in-domain accuracy and generalization.
3.  **Reward (Granularity and Verification):** Prioritize dense, verified execution-based feedback (e.g., unit test pass ratios) over sparse binary rewards or model-based judge approximations.

### Key Formulas
The study utilizes PPO for policy optimization. The clipped surrogate objective is defined as:

$$
L^{CLIP}(\theta) = \mathbb{E}_{t \sim n} \left[ \sum_{i=0}^{T} \sum_{i=1}^{n+1} \min \left( r_i^2(\theta) \dot{A}_i^2, \text{clip}(r_i^2(\theta), 1 - \epsilon, 1 + \epsilon) \dot{A}_i^2 \right) \right]
$$

Where the probability ratio for each token is:

$$
r_i^2(\theta) = \frac{\pi_{\theta}\left(a_i^2\right|h_i, a_i^{2-1})}{\pi_{\theta,i^2}\left(a_i^2\right|h_i, a_i^{2-1}}
$$

### Key Quantitative Results
The authors evaluated the recipe using Qwen models across TextWorld, ALFWorld, and SWE-Gym.

*   **Performance Gains:** On TextWorld (w2-o3-q4), Qwen-1.5B improved from a **17% base success rate to 88%** using PPO.
*   **Complexity Scaling:** Performance declines as complexity increases. PPO's improvement over the base model dropped from **88% (base)** to **51%** in the most complex TextWorld setting (w8-o12-q4). Larger models handled complexity better; Qwen-7B achieved **72% success** on w4-o6-q8.
*   **SFT vs. RL Budget:** A model with 60 SFT demonstrations and 400 RL episodes achieved **85% in-domain success** and **59% generalization** to complex tasks, outperforming pure SFT (55% generalization) or pure RL (11% generalization) under the same budget.
*   **Algorithm Comparison:** On w2-o3-q4 (Qwen-1.5B), PPO (**88%**) significantly outperformed RLOO (**51%**), while Reinforce++ and GRPO showed negligible gains (**18%**).
*   **Reward Density (SWE-Gym):** In software engineering tasks, sparse binary rewards yielded only **4.2% success**, whereas dense, ratio-based verified rewards boosted success to **22%**. Model-based judges (CodeRM-8B and GPT-4.1) underperformed, achieving **7.2% and 9.3%** respectively.
*   **Exploration Horizon:** In TextWorld, performance saturated beyond **8 exploration steps** (for 4-step optimal solutions). In SWE-Gym, increasing tool calls from 10 to 40 increased success from **3% to 22%**.

### Stated Limitations
*   **Cross-Domain Priors:** Initializing agents with SFT priors from a different domain (e.g., ALFWorld $\rightarrow$ TextWorld) causes rapid policy collapse due to conflicting behavioral biases.
*   **Reward Quality:** While dense rewards accelerate training, their utility is highly dependent on the quality of the signal; poorly designed intermediate rewards can introduce noise and misguide the agent.
*   **Model-Based Proxies:** Model-based judges are not yet a viable substitute for execution-based verifiers in complex reasoning tasks.
