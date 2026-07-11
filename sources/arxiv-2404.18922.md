---
id: arxiv:2404.18922
type: paper
title: 'DPO Meets PPO: Reinforced Token Optimization for RLHF'
url: https://arxiv.org/html/2404.18922v2
retrieved: '2026-07-11'
maturity: comprehensive
topic: mdp-formulation
---

# Reinforced Token Optimization (RTO) for RLHF

### Core Problem
Traditional Reinforcement Learning from Human Feedback (RLHF) pipelines typically formulate the problem as a **contextual bandit**, where rewards are assigned at the sentence level. However, Proximal Policy Optimization (PPO), the standard algorithm for alignment, is designed for **Markov Decision Processes (MDPs)** that require token-wise (step-wise) rewards. In current open-source implementations, the learned sentence-level reward is usually assigned only to the final token, while all preceding tokens receive zero learned reward. This sparsity leads to training instability and sample inefficiency.

### Method: Reinforced Token Optimization (RTO)
RTO addresses this mismatch by modeling RLHF as an MDP, enabling the use of dense, token-wise reward signals. The practical implementation integrates Direct Preference Optimization (DPO) and PPO in a two-stage process:

1.  **Token-wise Reward Learning:** Instead of training a separate scalar reward model via Maximum Likelihood Estimation (MLE), RTO leverages the insight that a DPO-aligned policy $\pi_{\text{dpo}}$ implicitly represents a reward function. A token-wise reward is extracted by calculating the log-ratio between the DPO policy and the reference policy $\pi_{\text{ref}}$.
2.  **Policy Optimization via PPO:** The extracted token-wise reward is used as the reward signal for PPO training. To prevent responses from becoming excessively long or short, a sentence-level MLE reward $r_{\text{MLE}}$ is optionally added to the final token.

#### Key Formulas
The preference between two trajectories $\tau^1$ and $\tau^2$ is modeled using the Bradley-Terry (BT) model extended to token-wise rewards:

$$
\mathbb{P}(\tau^1 \succ \tau^2) = \sigma \left( \sum_{h=1}^H r(s_h^1, a_h^1) - \sum_{h=1}^H r(s_h^2, a_h^2) \right)
$$

The practical RTO reward function $r_{\text{rto}}$ for a token $y_h$ given prompt $x$ and history $y_{1:h-1}$ is defined as:

$$
r_{\text{rto}}(x, y_{1:h}) = \begin{cases} \beta_{1} \log \frac{\pi_{\text{dpo}}(y_{h} \mid x, y_{1:h-1})}{\pi_{\text{ref}}(y_{h} \mid x, y_{1:h-1})} - \beta_{2} \log \frac{\pi(y_{h} \mid x, y_{1:h-1})}{\pi_{\text{ref}}(y_{h} \mid x, y_{1:h-1})} & \text{if } h \le H-1, \\ \beta_{1} \log \frac{\pi_{\text{dpo}}(y_{h} \mid x, y_{1:h-1})}{\pi_{\text{ref}}(y_{h} \mid x, y_{1:h-1})} - \beta_{2} \log \frac{\pi(y_{h} \mid x, y_{1:h-1})}{\pi_{\text{ref}}(y_{h} \mid x, y_{1:h-1})} + \beta_{3} \cdot r_{\text{MLE}}(x, y_{1:H}) & \text{if } h = H, \end{cases}
$$

where $\pi$ is the current policy being updated, and $\beta_1, \beta_2, \beta_3$ are tuning parameters.

### Key Quantitative Results
RTO was evaluated using Llama-3-8B on the UltraFeedback dataset across two benchmarks: AlpacaEval 2 (AE) and Arena-Hard (AH).

*   **Performance Gains:** RTO outperformed PPO and direct preference learning baselines (DPO, SimPO, TDPO). Specifically, RTO improved upon PPO by **7.53 points** on the AlpacaEval 2 length-controlled (LC) win rate (27.00 vs 19.47) and by **4.1 points** on the Arena-Hard style-controlled (SC) win rate (20.3 vs 16.2).
*   **Sample Efficiency:** RTO demonstrated significantly better data scaling than PPO. It reached PPO-level performance using only **1/8 of the training data**. While PPO performance saturated early, RTO continued to improve as more data was added.
*   **Theoretical Complexity:** The authors prove that finding an optimal response with sentence-wise rewards requires a sample complexity of $A^H$ (where $A$ is action set size and $H$ is horizon), whereas token-wise rewards reduce this to $A^{\min \{\xi + 1, H\}}$.

### Stated Limitations
*   **Hyperparameter Tuning:** The practical implementation relies on several tuning parameters ($\beta_1, \beta_2, \beta_3$) to balance the DPO reward, KL penalty, and sentence-level reward.
*   **Theoretical Assumptions:** The theoretical sample complexity guarantees for the "Theoretical Version" of RTO assume a linear reward function ($r(s, a) = \phi(s, a)^\top \theta^*$).
*   **Reward Hacking:** The authors acknowledge that reward optimization generally risks "reward hacking," where the model exploits imperfections in the reward function, necessitating the use of KL regularization and ensemble rewards (like $r_{\text{MLE}}$).
