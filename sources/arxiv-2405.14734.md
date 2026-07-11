---
id: arxiv:2405.14734
type: paper
title: 'SimPO: Simple Preference Optimization with a Reference-Free Reward'
url: https://arxiv.org/abs/2405.14734
retrieved: '2026-07-11'
maturity: comprehensive
topic: dpo-variants
---

# SimPO: Simple Preference Optimization with a Reference-Free Reward

### Core Problem
Direct Preference Optimization (DPO) is a widely used offline preference optimization algorithm, but it suffers from two primary drawbacks. First, it requires a reference model ($\pi_{\text{ref}}$) during training, which increases computational and memory overhead. Second, there is a discrepancy between the implicit reward optimized during training (the log ratio of the policy model to the reference model) and the metric used during inference (the average log-likelihood of the generated sequence). This mismatch can lead to suboptimal performance and a tendency to exploit length bias, where the model artificially inflates the probabilities of longer sequences.

### Method
SimPO addresses these issues by aligning the reward function directly with the generation metric and removing the reference model. The method follows these steps:

1.  **Length-Normalized Reward:** Instead of using a log ratio, SimPO uses the average log probability of all tokens in a response as the implicit reward. This ensures the reward is normalized by the response length $|y|$, preventing the model from favoring longer, lower-quality sequences.
2.  **Reference-Free Formulation:** By utilizing the policy model's own average log-likelihood, the need for a reference model is eliminated, reducing memory and compute requirements.
3.  **Target Reward Margin:** To improve generalization and better separate winning and losing responses, SimPO introduces a target reward margin $\gamma > 0$ into the Bradley-Terry objective. This forces the reward of the winning response to exceed that of the losing response by at least $\gamma$.

#### Key Formulas
The average log-likelihood $p_\theta(y \mid x)$ is defined as:

$$
p_\theta(y \mid x) = \frac{1}{|y|} \log \pi_\theta(y \mid x) = \frac{1}{|y|} \sum_{i=1}^{|y|} \log \pi_\theta(y_i \mid x, y_{<i})
$$

The SimPO reward is formulated as:

$$
r_{\text{SimPO}}(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y \mid x)
$$

The final SimPO objective function is:

$$
\mathcal{L}_{\text{SimPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l|x) - \gamma \right) \right]
$$

where $\beta$ is a scaling constant and $\gamma$ is the target reward margin.

### Key Quantitative Results
SimPO consistently outperformed DPO and other variants across multiple benchmarks (AlpacaEval 2, Arena-Hard, and MT-Bench) using Mistral, Llama 3, and Gemma 2 models.

*   **Performance Gains:** SimPO outperformed DPO by up to **6.4 points** on AlpacaEval 2 and up to **7.5 points** on Arena-Hard.
*   **State-of-the-Art (SOTA):** The top-performing model (Gemma-2-9B-it-SimPO) achieved a **72.4% length-controlled win rate** on AlpacaEval 2 and a **59.1% win rate** on Arena-Hard. It ranked 1st among models with $<10\text{B}$ parameters on the Chatbot Arena leaderboard.
*   **Efficiency:** Compared to vanilla DPO, SimPO reduced overall runtime by approximately **20%** and peak GPU memory usage by about **10%**.
*   **Length Bias:** SimPO demonstrated minimal length exploitation, maintaining a Spearman correlation between likelihood and length similar to the SFT model ($\rho = 0.34$), whereas SimPO without length normalization showed a strong positive correlation ($\rho = 0.82$).

### Stated Limitations
*   **Theoretical Analysis:** The authors note that a more rigorous theoretical analysis is required to fully understand the effectiveness of the approach.
*   **Hyperparameter Tuning:** The target reward margin $\gamma$ requires manual tuning.
*   **Safety and Honesty:** The algorithm does not explicitly incorporate constraints for safety or honesty.
*   **Reasoning Tasks:** SimPO (and preference optimization in general) can lead to performance drops on reasoning-heavy tasks, such as GSM8K.
