---
id: arxiv:2605.19461
type: paper
title: 'Beyond Mode Collapse: Distribution Matching for Diverse Reasoning'
url: https://arxiv.org/html/2605.19461v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

# Beyond Mode Collapse: Distribution Matching for Diverse Reasoning

### Core Problem
Large Reasoning Models (LRMs) trained via on-policy reinforcement learning (RL), such as Group Relative Policy Optimization (GRPO), frequently suffer from **mode collapse**. This phenomenon occurs when the policy concentrates its probability mass on the first high-reward trajectory it discovers, prematurely halting exploration of alternative strategies. The authors identify the root cause as the implicit minimization of **reverse KL divergence** ($D_{\text{KL}}(\pi_\theta \| p^*)$), which is inherently "mode-seeking." While effective for refining a known mode, reverse KL fails to maintain a distribution over multiple diverse high-reward solutions, leading to brittle reasoning and reduced robustness.

### Method: Distribution-Matching Policy Optimization (DMPO)
DMPO mitigates mode collapse by approximating **forward KL minimization** ($D_{\text{KL}}(p^* \| \pi_\theta)$), which is "mode-covering" and encourages the policy to match the full support of the target distribution. Because sampling from the global reward-weighted target distribution is intractable, DMPO operates at the **group level**.

**Step-by-Step Recipe:**
1. **Group Sampling:** For a given query $x$, sample a group of $G$ trajectories $\mathcal{O} = \{o_1, \dots, o_G\}$ from the current policy $\pi_{\theta_{\text{old}}}$.
2. **Target Distribution Construction:** Define a group-conditional Boltzmann distribution $p$ where trajectories with higher rewards receive higher probability:

$$
p(o_{i}\mid\mathcal{O})=\frac{\exp(r(o_{i})/\alpha)}{\sum_{j=1}^{G}\exp(r(o_{j})/\alpha)}
$$

   where $\alpha$ is a temperature parameter.
3. **Policy Distribution Normalization:** To prevent length bias (since probabilities decay exponentially with trajectory length), compute the length-normalized log-likelihood $\phi(o_i)$:

$$
\phi(o_i) = \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \log \pi_\theta(o_{i,t} \mid o_{i,<t}, x)
$$

   Then, define the group-level policy distribution $q_\theta$:

$$
q_{\theta}(o_{i}\mid\mathcal{O})=\frac{\exp(\phi(o_{i}))}{\sum_{j=1}^{G}\exp(\phi(o_{j}))}
$$

4. **Distribution Alignment:** Align $q_\theta$ to $p$ using Mean Squared Error (MSE) instead of KL divergence to ensure numerical stability and bounded gradients:

$$
\mathcal{L}_{\text{DM}}(\theta)=\frac{1}{G}\sum_{i=1}^{G}\left(p(o_{i}\mid\mathcal{O})-q_{\theta}(o_{i}\mid\mathcal{O})\right)^{2}
$$

5. **Unified Optimization:** Combine the distribution-matching loss with the standard GRPO objective:

$$
\mathcal{L}_{\text{DMPO}}(\theta)=\mathcal{L}_{\text{GRPO}}(\theta)+\lambda\mathcal{L}_{\text{DM}}(\theta)
$$

   where $\lambda$ balances reward maximization (exploitation) and diversity preservation (exploration).

### Key Quantitative Results
The authors introduced **MM-NP-Bench**, a multimodal benchmark of 10 NP-hard combinatorial optimization tasks, using **Success Rate (SR)** (feasibility) and **Quality Ratio (QR)** (optimality relative to a heuristic solver) to measure performance.

*   **Combinatorial Optimization:**
    *   **Vision-based MM-NP-Bench:** DMPO achieved a **43.1% QR** compared to GRPO's **38.4%** (a 12% relative improvement) and a higher SR (**61.9%** vs. **55.7%**).
    *   **Text-based NP-Bench:** DMPO achieved a **43.9% QR** compared to GRPO's **40.1%** (a 9% relative improvement).
*   **Generalization:**
    *   **Mathematical Reasoning:** DMPO improved performance by **2.0%** on average across six benchmarks (including AIME and MATH500).
    *   **Out-of-Domain Tasks:** DMPO showed a **2.3%** average improvement on seven visual reasoning benchmarks (e.g., MathVista, LogicVista).
*   **Training Dynamics:** DMPO prevented "length collapse" (where GRPO responses degenerated from $\sim 600$ to $<200$ tokens), maintaining robust reasoning chains of $\sim 400$ tokens.

### Stated Limitations
1. **Reward Dependency:** DMPO currently requires exact, rule-based verifiable rewards. Extending the method to open-ended tasks with subjective or learned rewards remains an open challenge.
2. **Sparse Rewards:** In environments with extremely sparse rewards, if no valid solution is sampled within a group, the local group approximation cannot recover the global optimum independently. The authors suggest that replay buffers or off-policy extensions may be necessary for such regimes.
