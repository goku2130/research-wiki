---
id: arxiv:2502.18770
type: paper
title: Reward Shaping to Mitigate Reward Hacking in RLHF
url: https://arxiv.org/abs/2502.18770
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-hacking
---

# Reward Shaping to Mitigate Reward Hacking in RLHF

### Core Problem
Reinforcement Learning from Human Feedback (RLHF), particularly when using Proximal Policy Optimization (PPO), is highly susceptible to **reward hacking**. This occurs when a policy model exploits flaws or ambiguities in the proxy reward model to maximize reward signals without achieving genuine alignment, often resulting in degenerate behaviors such as excessive verbosity or repetitive outputs. The authors identify that excessively high rewards can impair the critic's ability to learn the value function and lead to unstable, overly aggressive policy updates.

### Proposed Method: Preference As Reward (PAR)
The authors propose two design principles for effective reward shaping:
1. **Boundedness:** RL rewards should be bounded to stabilize critic training.
2. **Growth Pattern:** RL rewards should exhibit rapid initial growth followed by gradual convergence to encourage optimization in "safer" low-reward regions.

Guided by these principles, the authors introduce **Preference As Reward (PAR)**. PAR transforms the proxy reward into a preference score by applying a sigmoid function to the reward centered around one or more reference rewards.

#### Step-by-Step Recipe
1. **Obtain Proxy Reward:** The reward model $r_\phi$ provides a proxy reward $r$ for the policy's response $y$.
2. **Sample Reference Rewards:** Sample $M$ reference responses $y_{\text{ref}}^m$ from the reference model (SFT model) and calculate their proxy rewards $r_{\text{ref}}^m = r_\phi(x, y_{\text{ref}}^m)$.
3. **Center and Transform:** Subtract the reference reward from the proxy reward and apply the sigmoid function $\sigma$.
4. **Aggregate:** Average the results across all $M$ reference rewards to produce the final RL reward $r_{\text{RL}}$.
5. **PPO Update:** Use $r_{\text{RL}}$ as the terminal reward in the PPO pipeline, combined with a KL divergence penalty for per-token rewards.

### Key Formulas
The RL reward for PAR is defined as:

$$
r_{\mathrm{RL}} = \frac{1}{M} \sum_{m=1}^{M} \sigma(r - r_{\mathrm{ref}}^m)
$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$.

The authors provide two theoretical guarantees:
*   **Return Variance Bound:** If per-step rewards $|r_l| < 1$, the variance of the return $G_t$ is upper bounded:

$$
\text{Var}[G_t] \leq \frac{1}{(1-\gamma)^2}
$$

*   **Minimum-Variance Estimator:** Under logistic preference noise, the sigmoid transformation is the unique minimum-variance unbiased shaping function for the policy gradient.

### Key Quantitative Results
Evaluated on **Gemma2-2B** using the **Ultrafeedback-Binarized** and **HH-RLHF** datasets, PAR demonstrated superior performance and robustness:

*   **Benchmark Performance:** On AlpacaEval 2.0, PAR achieved a win rate at least **5 percentage points higher** than competing approaches. According to Table 1, PAR reached an **LC Winrate of 70.81%** and a **Winrate of 75.37%** on AlpacaEval 2.0, and an **Overall score of 5.063** on MT-Bench.
*   **Data Efficiency:** PAR is highly efficient, requiring only a **single reference reward** ($M=1$) to achieve performance comparable to using $M=10$.
*   **Robustness:** While baselines like Minmax and WARM failed when training was extended to two epochs, PAR remained robust against reward hacking throughout the extended duration.
*   **Comparison:** PAR outperformed other mitigation baselines including ODIN, Reg, Meanstd, Clip, and LSC, which failed to prevent the decline in win rate as proxy rewards increased.

### Stated Limitations
The authors note two primary limitations:
1. **Peak Performance:** PAR effectively mitigates reward hacking and extends the window for early stopping, but it does not improve the peak performance (the win rate of the absolute best checkpoint).
2. **Dynamics:** The specific dynamics of reward adjustment—specifically the optimal initial rate of increase and the pace of convergence—are not yet fully elucidated.
