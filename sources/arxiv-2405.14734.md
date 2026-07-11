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

SimPO is an offline preference optimization algorithm designed to align large language models (LLMs) with human preferences. It simplifies the training process by eliminating the need for a reference model and aligning the training reward more closely with the metrics used during model generation.

### Core Problem
The authors identify two primary drawbacks in Direct Preference Optimization (DPO):
1. **Reward-Generation Mismatch:** DPO uses an implicit reward based on the log ratio between the current policy and a reference model. However, during inference, models are typically guided by the average log-likelihood of the response. This discrepancy means that a higher DPO reward does not necessarily correlate with a higher generation likelihood.
2. **Resource Overhead:** The requirement for a reference model during training increases computational and memory costs.
3. **Length Bias:** Without explicit normalization, preference optimization can lead to "length exploitation," where the model artificially inflates the probability of longer sequences to increase rewards.

### Method
SimPO addresses these issues by replacing the reference-based reward with a length-normalized, reference-free reward and introducing a target margin.

**Step-by-Step Recipe:**
1. **Define Length-Normalized Reward:** Instead of a log ratio, SimPO uses the average log probability of all tokens in a response as the implicit reward. This ensures the reward is directly aligned with the likelihood metric used in beam search and multiple-choice tasks.
2. **Apply Scaling:** A constant $\beta$ is used to control the scaling of the reward difference.
3. **Introduce Target Reward Margin:** A margin $\gamma > 0$ is added to the Bradley-Terry objective. This forces the reward of the winning response ($y_w$) to exceed that of the losing response ($y_l$) by at least $\gamma$, improving generalization.
4. **Optimize the Objective:** The model is trained by minimizing the SimPO loss function without the need for a reference model or KL regularization.

**Key Formulas:**
The average log-likelihood is defined as:

$$
p_\theta(y \mid x) = \frac{1}{|y|} \log \pi_\theta(y \mid x) = \frac{1}{|y|} \sum_{i=1}^{|y|} \log \pi_\theta(y_i \mid x, y_{<i})
$$

The SimPO reward is formulated as:

$$
r_{\text{SimPO}}(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y \mid x)
$$

The final SimPO objective is:

$$
\mathcal{L}_{\text{SimPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l|x) - \gamma \right) \right]
$$

### Key Quantitative Results
SimPO was evaluated across Mistral, Llama 3, and Gemma 2 models using benchmarks including AlpacaEval 2, Arena-Hard, and MT-Bench.

*   **Performance Gains:** SimPO consistently outperformed DPO, with improvements of up to **6.4 points on AlpacaEval 2** and **7.5 points on Arena-Hard**.
*   **State-of-the-Art (SOTA):** The top-performing model (Gemma-2-9B-it-SimPO) achieved a **72.4% length-controlled win rate** on AlpacaEval 2 and a **59.1% win rate** on Arena-Hard. It ranked 1st among models under 10B parameters on the Chatbot Arena leaderboard (as of September 16, 2024).
*   **Efficiency:** In the Llama-3-Base setting using 8$\times$H100 GPUs, SimPO reduced runtime by approximately **20%** and peak GPU memory usage by about **10%** compared to vanilla DPO.
*   **Length Control:** SimPO demonstrated minimal length exploitation, maintaining a Spearman correlation between likelihood and response length similar to the SFT model ($\rho = 0.34$), whereas SimPO without length normalization showed a strong positive correlation ($\rho = 0.82$).

### Stated Limitations
*   **Theoretical Analysis:** The authors note a need for more rigorous theoretical analysis to fully understand the algorithm's effectiveness.
*   **Hyperparameter Tuning:** The target reward margin $\gamma$ requires manual tuning.
*   **Safety and Honesty:** The method does not explicitly incorporate safety or honesty constraints, though it showed competitive performance on TruthfulQA.
*   **Reasoning Tasks:** SimPO (and preference optimization in general) can lead to performance drops on reasoning-heavy tasks such as GSM8K.
